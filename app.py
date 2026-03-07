import gradio as gr
import urllib.parse
import urllib.request
import re
import os
import html as html_lib
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor

# ── Load Emotion Explorer HTML at startup ───────────────────
import json as _json

_EMOTION_UI_SRCDOC = ""
try:
    _html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_ui.html")
    with open(_html_path, "r", encoding="utf-8") as _f:
        _raw = _f.read()
    # Song library will be injected after song_library is loaded (see below)
    print(f"[Timbre] emotion_ui.html read ({len(_raw):,} chars)")
except Exception as _e:
    print(f"[Timbre] WARNING: Could not load emotion_ui.html: {_e}")
from download_models import ensure_models
ensure_models()  # 確保模型已下載（HF Spaces 首次啟動時）
from recommend_v2 import recommend, song_library, song_features

# ── YouTube ID cache ────────────────────────────────────────
_YT_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_id_cache.json")
_yt_cache: dict = {}
try:
    with open(_YT_CACHE_PATH, "r", encoding="utf-8") as _cf:
        _yt_cache = _json.load(_cf)
    print(f"[Timbre] Loaded YT cache: {len(_yt_cache)} entries")
except Exception:
    print("[Timbre] No YT cache found — will populate in background")

# ── Inject song library into Emotion Explorer HTML ──────────
if _raw:
    _songs_compact = [
        {
            "t":  str(r["title"]),
            "b":  round(float(r["bpm"]),        1),
            "v":  round(float(r["valence"]),    2),
            "a":  round(float(r["arousal"]),    2),
            "h":  round(float(r["mood_happy"]),      3),
            "s":  round(float(r["mood_sad"]),        3),
            "ag": round(float(r["mood_aggressive"]), 3),
            "r":  round(float(r["mood_relaxed"]),    3),
            "p":  round(float(r["mood_party"]),      3),
            "d":  round(float(r["danceability"]),    3),
            "yt": _yt_cache.get(str(r["title"])),   # None if not yet cached
        }
        for _, r in song_features.iterrows()
    ]
    _yt_ok = sum(1 for s in _songs_compact if s["yt"])
    print(f"[Timbre] {_yt_ok}/{len(_songs_compact)} songs have cached YouTube IDs")

    _song_script = (
        "<script>window.__TIMBRE_SONGS__="
        + _json.dumps(_songs_compact, ensure_ascii=True, separators=(",", ":"))
        + ";</script>"
    )
    _raw_injected = _raw.replace("</head>", _song_script + "</head>", 1)
    _EMOTION_UI_SRCDOC = _raw_injected.replace("&", "&amp;").replace('"', "&quot;")
    print(f"[Timbre] Injected {len(_songs_compact)} songs into srcdoc ({len(_EMOTION_UI_SRCDOC):,} chars)")

    # ── Background thread: fill missing YouTube IDs and save cache ──
    def _fill_yt_cache_bg():
        missing = [s["t"] for s in _songs_compact if not s["yt"]]
        if not missing:
            return
        print(f"[Timbre] BG: fetching {len(missing)} missing YouTube IDs …")

        def _fetch(title):
            try:
                return title, get_youtube_video_id(title)
            except Exception:
                return title, None

        updated = 0
        with ThreadPoolExecutor(max_workers=10) as _ex:
            for title, vid in _ex.map(_fetch, missing):
                if vid:
                    _yt_cache[title] = vid
                    updated += 1

        if updated:
            try:
                with open(_YT_CACHE_PATH, "w", encoding="utf-8") as _cf:
                    _json.dump(_yt_cache, _cf, ensure_ascii=False)
                print(f"[Timbre] BG: saved YT cache ({len(_yt_cache)} entries). "
                      "Restart the Space to serve embedded players.")
            except Exception as _e:
                print(f"[Timbre] BG: failed to save YT cache: {_e}")

    threading.Thread(target=_fill_yt_cache_bg, daemon=True).start()


def get_local_audio_path(filename):
    """取得本地音檔路徑（如果存在的話）"""
    possible_paths = [
        os.path.join("songs", filename),
        os.path.join("..", "songs", filename)
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None


# ── i18n 翻譯字典 ───────────────────────────────────────────
I18N = {
    "zh": {
        "title": "🎵 Timbre 聲學規格引擎",
        "subtitle": "輸入情緒描述，AI 幫你找到最匹配的參考音樂，並自動生成聲學規格建議書。",
        "mood_label": "情緒描述",
        "mood_placeholder": "描述你的情緒或場景，例如：深夜開車，有點孤獨...",
        "client_btn": "🎬 我是業主（找參考音樂）",
        "musician_btn": "🎸 我是音樂人（看聲學規格）",
        "empty_input": "請輸入情緒描述",
        "client_header": "🎵 推薦參考音樂",
        "client_footer_1": "✅ 確認後，系統將把你的需求轉換為樂手可執行的聲學規格書",
        "musician_header": "🎸 業主需求分析",
        "similarity": "相似度",
        "spec_title": "📋 聲學規格建議書",
        "spec_subtitle": "根據以上參考曲目自動生成",
        "tempo_label": "🎯 建議速度",
        "energy_label": "⚡ 能量強度",
        "tone_label": "🌈 情緒色彩",
        "tags_label": "🏷️ 風格標籤",
        "production_label": "💡 製作建議",
        "slow": "慢板", "moderate": "中板", "fast": "快板", "very_fast": "極快",
        "energy_low": "低能量、柔和",
        "energy_med": "中等能量",
        "energy_high": "高能量、有張力",
        "energy_max": "極高能量、爆發力",
        "tone_dark": "偏暗、負面",
        "tone_neutral": "中性偏沉",
        "tone_bright": "正面、明亮",
        "tone_very_bright": "非常正面、開朗",
        "tag_happy": "歡樂", "tag_sad": "感傷", "tag_aggressive": "激烈",
        "tag_relaxed": "放鬆", "tag_party": "派對", "tag_groovy": "律動感",
        "tag_neutral": "中性",
        "sug_groove": "強調節奏律動，可加入明顯的鼓組 groove",
        "sug_soft": "以柔和音色為主，可使用 pad、木吉他、鋼琴",
        "sug_distortion": "可加入失真吉他或強力鼓點增加衝擊感",
        "sug_minor": "選擇小調和聲，營造感傷氛圍",
        "sug_major": "選擇大調和聲，保持明亮的旋律線",
        "sug_synth": "可加入合成器、電子鼓點，營造派對氛圍",
        "sug_space": "注意留白與空間感，不要過度編曲",
        "sug_free": "依照業主情緒描述自由發揮",
        "search_yt": "在 YouTube 搜尋",
    },
    "en": {
        "title": "🎵 Timbre Audio-to-Brief Engine",
        "subtitle": "Describe a mood or scene, and AI will find the best-matching reference music and auto-generate an acoustic specification brief.",
        "mood_label": "Mood Description",
        "mood_placeholder": "Describe the mood or scene, e.g.: late night drive, feeling lonely...",
        "client_btn": "🎬 Client — Find Reference Music",
        "musician_btn": "🎸 Musician — Acoustic Spec Brief",
        "empty_input": "Please enter a mood description",
        "client_header": "🎵 Recommended Reference Tracks",
        "client_footer_1": "✅ Once confirmed, the system will generate an executable acoustic spec brief for musicians.",
        "musician_header": "🎸 Client Needs Analysis",
        "similarity": "Similarity",
        "spec_title": "📋 Acoustic Specification Brief",
        "spec_subtitle": "Auto-generated from reference tracks above",
        "tempo_label": "🎯 Suggested Tempo",
        "energy_label": "⚡ Energy Level",
        "tone_label": "🌈 Emotional Tone",
        "tags_label": "🏷️ Style Tags",
        "production_label": "💡 Production Suggestions",
        "slow": "Slow", "moderate": "Moderate", "fast": "Fast", "very_fast": "Very Fast",
        "energy_low": "Low Energy / Soft",
        "energy_med": "Medium Energy",
        "energy_high": "High Energy / Tense",
        "energy_max": "Explosive Energy",
        "tone_dark": "Dark / Negative",
        "tone_neutral": "Neutral-Dark",
        "tone_bright": "Positive / Bright",
        "tone_very_bright": "Very Positive / Cheerful",
        "tag_happy": "Happy", "tag_sad": "Sad", "tag_aggressive": "Aggressive",
        "tag_relaxed": "Relaxed", "tag_party": "Party", "tag_groovy": "Groovy",
        "tag_neutral": "Neutral",
        "sug_groove": "Emphasize rhythmic groove with prominent drum patterns",
        "sug_soft": "Use soft timbres: pads, acoustic guitar, piano",
        "sug_distortion": "Add distorted guitars or powerful drum hits for impact",
        "sug_minor": "Use minor key harmony to create a melancholic atmosphere",
        "sug_major": "Use major key harmony with bright melodic lines",
        "sug_synth": "Add synths and electronic drums for a party vibe",
        "sug_space": "Leave space and air in the arrangement, avoid over-producing",
        "sug_free": "Follow the client's mood description freely",
        "search_yt": "Search on YouTube",
    },
}


def t(key, lang="en"):
    """取得翻譯文字"""
    return I18N.get(lang, I18N["en"]).get(key, key)


# ── YouTube 搜尋與嵌入 ──────────────────────────────────────
from ytmusicapi import YTMusic
ytmusic = YTMusic()

def get_youtube_video_id(query):
    """取得 YouTube Video ID，使用三重 Fallback 機制確保能在 HF Spaces 成功"""
    # 1. ytmusicapi (YouTube Music 原生搜尋，包含官方 MV 與純音軌，極低廣告干擾)
    try:
        results = ytmusic.search(query)
        for r in results:
            if 'videoId' in r and r['videoId']:
                return r['videoId']
    except Exception as e:
        print(f"  [Fallback 1 Failed] ytmusicapi: {e}")

    # 2. DuckDuckGo Search (穩定的網頁抓取)
    try:
        encoded = urllib.parse.quote('site:youtube.com/watch ' + query)
        req = urllib.request.Request(
            f'https://html.duckduckgo.com/html/?q={encoded}', 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        html = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
        match = re.search(r'v=([a-zA-Z0-9_-]{11})', html)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"  [Fallback 2 Failed] DuckDuckGo: {e}")

    # 3. YouTube Search HTML (最易受廣告干擾)
    try:
        encoded = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://www.youtube.com/results?search_query={encoded}", 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        html = urllib.request.urlopen(req, timeout=5).read().decode("utf-8")
        match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"  [Fallback 3 Failed] YouTube HTML: {e}")

    return None


def get_youtube_search_url(title):
    query = urllib.parse.quote(title)
    return f"https://www.youtube.com/results?search_query={query}"


def build_player_html(title, video_id, lang="zh"):
    """生成嵌入式 YouTube 播放器 HTML"""
    if video_id:
        return f'''
        <div style="margin: 8px 0;">
            <iframe width="100%" height="200" 
                src="https://www.youtube.com/embed/{video_id}" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        </div>'''
    else:
        search_url = get_youtube_search_url(title)
        return f'''
        <div style="margin: 8px 0;">
            <a href="{search_url}" target="_blank" 
               style="display:inline-block; padding:8px 16px; background:#ff4444; color:white; 
                      border-radius:6px; text-decoration:none; font-size:14px;">
                🔗 {t("search_yt", lang)}
            </a>
        </div>'''


# ── CSS ─────────────────────────────────────────────────────
CARD_STYLE = """
<style>
    .timbre-results {
        background: #ffffff;
        color: #333;
        padding: 16px;
        border-radius: 12px;
    }
    .timbre-results h2 {
        color: #212529;
        margin: 0 0 12px 0;
    }
    .song-card {
        background: #f8f9fa;
        color: #333;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .song-card h3 {
        color: #212529;
        margin: 0 0 8px 0;
        font-size: 18px;
    }
    .song-card .score {
        color: #555;
        font-size: 13px;
    }
    .spec-section {
        background: #fff8e1;
        color: #333;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        border: 1px solid #ffe082;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .spec-section h2 {
        color: #e65100;
        margin: 0 0 16px 0;
    }
    .spec-line {
        color: #333;
        font-size: 15px;
        line-height: 1.8;
        margin: 4px 0;
    }
    .suggestion-item {
        color: #444;
        font-size: 14px;
        line-height: 1.6;
        margin: 2px 0;
        padding-left: 8px;
    }
    .brief-footer {
        color: #666;
        font-size: 12px;
        margin-top: 16px;
        text-align: center;
    }
    .feat-detail {
        color: #555;
        font-size: 13px;
        margin-top: 8px;
        line-height: 1.6;
    }
</style>
<div class="timbre-results">
"""



def recommend_for_client(mood, lang):
    """業主版：推薦音樂 + 嵌入式 YouTube 播放器 / 本地播放器"""
    if not mood.strip():
        return f"<p>{t('empty_input', lang)}</p>"

    results = recommend(mood, top_k=3, return_results=True)

    html = CARD_STYLE
    html += f"<h2 style='color:#212529;'>{t('client_header', lang)}</h2>"

    for i, (idx, score) in enumerate(results):
        title = song_library.iloc[idx]['title']
        filename = song_library.iloc[idx]['filename']
        local_path = get_local_audio_path(filename)
        
        if local_path:
            player = f'<div style="margin: 8px 0;"><audio controls src="file={local_path}" style="width:100%"></audio></div>'
        else:
            video_id = get_youtube_video_id(title)
            player = build_player_html(title, video_id, lang)

        html += f'''
        <div class="song-card">
            <h3>{i+1}. {title}</h3>
            {player}
        </div>'''

    html += f'<div class="brief-footer">{t("client_footer_1", lang)}</div>'
    html += '</div>'
    return html


# ── 聲學規格建議 ────────────────────────────────────────────
def generate_acoustic_brief_html(avg, lang):
    """根據平均聲學特徵自動產生聲學規格建議（HTML 版）"""

    # 速度
    bpm = avg['bpm']
    if bpm < 80:
        tempo_desc = t("slow", lang)
    elif bpm < 110:
        tempo_desc = t("moderate", lang)
    elif bpm < 140:
        tempo_desc = t("fast", lang)
    else:
        tempo_desc = t("very_fast", lang)

    # 能量
    arousal = avg['arousal']
    if arousal < 3.5:
        energy_desc = t("energy_low", lang)
    elif arousal < 5.5:
        energy_desc = t("energy_med", lang)
    elif arousal < 7:
        energy_desc = t("energy_high", lang)
    else:
        energy_desc = t("energy_max", lang)

    # 情緒色彩
    valence = avg['valence']
    if valence < 3:
        valence_desc = t("tone_dark", lang)
    elif valence < 5:
        valence_desc = t("tone_neutral", lang)
    elif valence < 7:
        valence_desc = t("tone_bright", lang)
    else:
        valence_desc = t("tone_very_bright", lang)

    # 風格標籤
    mood_tags = []
    if avg['mood_happy'] > 0.4:
        mood_tags.append(t("tag_happy", lang))
    if avg['mood_sad'] > 0.4:
        mood_tags.append(t("tag_sad", lang))
    if avg['mood_aggressive'] > 0.3:
        mood_tags.append(t("tag_aggressive", lang))
    if avg['mood_relaxed'] > 0.4:
        mood_tags.append(t("tag_relaxed", lang))
    if avg['mood_party'] > 0.4:
        mood_tags.append(t("tag_party", lang))
    if avg['danceability'] > 0.6:
        mood_tags.append(t("tag_groovy", lang))
    if not mood_tags:
        mood_tags.append(t("tag_neutral", lang))

    # 製作建議
    suggestions = []
    if avg['danceability'] > 0.6:
        suggestions.append(t("sug_groove", lang))
    if avg['mood_relaxed'] > 0.5 and arousal < 4:
        suggestions.append(t("sug_soft", lang))
    if avg['mood_aggressive'] > 0.3 and arousal > 5:
        suggestions.append(t("sug_distortion", lang))
    if avg['mood_sad'] > 0.4 and valence < 4:
        suggestions.append(t("sug_minor", lang))
    if avg['mood_happy'] > 0.5 and valence > 5:
        suggestions.append(t("sug_major", lang))
    if avg['mood_party'] > 0.4 and avg['danceability'] > 0.5:
        suggestions.append(t("sug_synth", lang))
    if arousal < 3:
        suggestions.append(t("sug_space", lang))
    if not suggestions:
        suggestions.append(t("sug_free", lang))

    html = ""
    html += f'<div class="spec-line">{t("tempo_label", lang)}：~{bpm:.0f} BPM（{tempo_desc}）</div>'
    html += f'<div class="spec-line">{t("energy_label", lang)}：{energy_desc}（arousal {arousal:.1f}/9）</div>'
    html += f'<div class="spec-line">{t("tone_label", lang)}：{valence_desc}（valence {valence:.1f}/9）</div>'
    html += f'<div class="spec-line">{t("tags_label", lang)}：{" / ".join(mood_tags)}</div>'

    html += f'<div class="spec-line" style="margin-top:12px;"><strong>{t("production_label", lang)}：</strong></div>'
    for i, s in enumerate(suggestions[:5]):
        html += f'<div class="suggestion-item">• {s}</div>'

    return html


def recommend_for_musician(mood, lang):
    """音樂人版：嵌入式播放器 + 聲學參數 + 規格書"""
    if not mood.strip():
        return f"<p>{t('empty_input', lang)}</p>"

    results = recommend(mood, top_k=3, return_results=True)

    html = CARD_STYLE
    html += f"<h2 style='color:#212529;'>{t('musician_header', lang)}</h2>"
    feature_rows = []

    for i, (idx, score) in enumerate(results):
        title = song_library.iloc[idx]['title']
        filename = song_library.iloc[idx]['filename']
        feature_row = song_features[song_features['title'] == title]
        
        local_path = get_local_audio_path(filename)
        if local_path:
            player = f'<div style="margin: 8px 0;"><audio controls src="file={local_path}" style="width:100%"></audio></div>'
        else:
            video_id = get_youtube_video_id(title)
            player = build_player_html(title, video_id, lang)

        html += f'<div class="song-card">'
        html += f'<h3>{i+1}. {title}</h3>'
        html += f'<div class="score">{t("similarity", lang)}：{score:.3f}</div>'
        html += player

        if not feature_row.empty:
            row = feature_row.iloc[0]
            feature_rows.append(row)
            html += f'''
            <div class="feat-detail">
                BPM：{row['bpm']:.0f} &nbsp;│&nbsp;
                Valence：{row['valence']:.2f} &nbsp;│&nbsp;
                Arousal：{row['arousal']:.2f}<br>
                Mood — 
                Happy {row['mood_happy']:.2f} / 
                Sad {row['mood_sad']:.2f} / 
                Aggressive {row['mood_aggressive']:.2f} / 
                Relaxed {row['mood_relaxed']:.2f} / 
                Party {row['mood_party']:.2f}<br>
                Danceability：{row['danceability']:.2f}
            </div>'''
        html += '</div>'

    # 聲學規格書
    if feature_rows:
        avg = pd.DataFrame(feature_rows)[
            ['bpm', 'valence', 'arousal',
             'mood_happy', 'mood_sad', 'mood_aggressive',
             'mood_relaxed', 'mood_party', 'danceability']
        ].mean()

        html += f'''
        <div class="spec-section">
            <h2>{t("spec_title", lang)}</h2>
            <div style="color:#888; font-size:12px; margin-bottom:12px;">
                {t("spec_subtitle", lang)}
            </div>
            {generate_acoustic_brief_html(avg, lang)}
        </div>'''
    html += '</div>'
    return html


# ── Gradio 介面 ─────────────────────────────────────────────
with gr.Blocks(title="Timbre Audio-to-Brief Engine") as demo:

    with gr.Tabs():

        # ── Tab 1: Emotion Explorer (interactive bubble UI) ──────────
        with gr.Tab("🌌 Emotion Explorer"):
            if _EMOTION_UI_SRCDOC:
                gr.HTML(
                    f'<iframe srcdoc="{_EMOTION_UI_SRCDOC}"'
                    ' sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"'
                    ' style="width:100%; height:90vh; border:none; border-radius:10px; display:block;"'
                    ' title="Emotion Explorer"></iframe>'
                )
            else:
                gr.HTML(
                    '<p style="color:#ccc; text-align:center; padding:40px;">'
                    '⚠️ Emotion Explorer could not be loaded. Please use the 📝 Text Search tab.</p>'
                )

        # ── Tab 2: Text Search (original text-based interface) ───────
        with gr.Tab("📝 Text Search"):
            # 語言狀態
            lang_state = gr.State("en")

            # 頂部：標題 + 語言切換
            with gr.Row():
                with gr.Column(scale=5):
                    title_md = gr.Markdown("# 🎵 Timbre Audio-to-Brief Engine")
                    subtitle_md = gr.Markdown("Describe a mood or scene, and AI will find the best-matching reference music and auto-generate an acoustic specification brief.")
                with gr.Column(scale=1, min_width=120):
                    lang_toggle = gr.Radio(
                        choices=["中文", "English"],
                        value="English",
                        label="Language / 語言",
                        interactive=True,
                    )

            mood_input = gr.Textbox(
                placeholder="Describe the mood or scene, e.g.: late night drive, feeling lonely...",
                label="Mood Description",
                lines=2
            )

            with gr.Row():
                client_btn = gr.Button("🎬 Client — Find Reference Music", variant="primary")
                musician_btn = gr.Button("🎸 Musician — Acoustic Spec Brief", variant="secondary")

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Column(elem_classes="timbre-results"):
                        output_html = gr.HTML()

            # 語言切換邏輯
            def switch_language(lang_choice):
                lang = "en" if lang_choice == "English" else "zh"
                return (
                    lang,
                    f"# {t('title', lang)}",
                    t("subtitle", lang),
                    gr.update(label=t("mood_label", lang), placeholder=t("mood_placeholder", lang)),
                    gr.update(value=t("client_btn", lang)),
                    gr.update(value=t("musician_btn", lang)),
                )

            lang_toggle.change(
                fn=switch_language,
                inputs=[lang_toggle],
                outputs=[lang_state, title_md, subtitle_md, mood_input, client_btn, musician_btn],
            )

            outputs = [output_html]

            # api_name exposes these as /call/recommend_client and /call/recommend_musician
            # so the Emotion Explorer iframe can call them via fetch
            client_btn.click(
                fn=recommend_for_client,
                inputs=[mood_input, lang_state],
                outputs=outputs,
                api_name="recommend_client",
            )
            musician_btn.click(
                fn=recommend_for_musician,
                inputs=[mood_input, lang_state],
                outputs=outputs,
                api_name="recommend_musician",
            )


demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["songs", "../songs"])