import gradio as gr
from download_models import ensure_models
ensure_models()  # 確保模型已下載（HF Spaces 首次啟動時）
from recommend_v2 import recommend, song_library, song_features

def get_youtube_search_url(title):
    query = title.replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}"

def recommend_for_client(mood):
    """業主版：簡單、直覺、可以試聽"""
    if not mood.strip():
        return "請輸入情緒描述"
    
    results = recommend(mood, top_k=3, return_results=True)
    output = "🎵 根據你的描述，推薦以下參考音樂：\n\n"
    
    for i, (idx, score) in enumerate(results):
        title = song_library.iloc[idx]['title']
        youtube_url = get_youtube_search_url(title)
        output += f"{i+1}. {title}\n"
        output += f"   🔗 {youtube_url}\n\n"
    
    output += "\n✅ 確認後，系統將把你的需求轉換為樂手可執行的聲學規格書"
    return output

def generate_acoustic_brief(avg):
    """根據平均聲學特徵自動產生聲學規格建議"""
    brief_lines = []

    # ── 速度建議 ──
    bpm = avg['bpm']
    if bpm < 80:
        tempo_desc = "慢板 (Slow)"
    elif bpm < 110:
        tempo_desc = "中板 (Moderate)"
    elif bpm < 140:
        tempo_desc = "快板 (Fast)"
    else:
        tempo_desc = "極快 (Very Fast)"
    brief_lines.append(f"🎯 建議速度：{bpm:.0f} BPM 左右（{tempo_desc}）")

    # ── 能量 / 情感強度 ──
    arousal = avg['arousal']
    if arousal < 3.5:
        energy_desc = "低能量、柔和"
    elif arousal < 5.5:
        energy_desc = "中等能量"
    elif arousal < 7:
        energy_desc = "高能量、有張力"
    else:
        energy_desc = "極高能量、爆發力"
    brief_lines.append(f"⚡ 能量強度：{energy_desc}（arousal {arousal:.1f}/9）")

    # ── 情緒色彩 ──
    valence = avg['valence']
    if valence < 3:
        valence_desc = "偏暗、負面"
    elif valence < 5:
        valence_desc = "中性偏沉"
    elif valence < 7:
        valence_desc = "正面、明亮"
    else:
        valence_desc = "非常正面、開朗"
    brief_lines.append(f"🌈 情緒色彩：{valence_desc}（valence {valence:.1f}/9）")

    # ── 主要風格 tag ──
    mood_tags = []
    if avg['mood_happy'] > 0.4:
        mood_tags.append("歡樂 Happy")
    if avg['mood_sad'] > 0.4:
        mood_tags.append("感傷 Sad")
    if avg['mood_aggressive'] > 0.3:
        mood_tags.append("激烈 Aggressive")
    if avg['mood_relaxed'] > 0.4:
        mood_tags.append("放鬆 Relaxed")
    if avg['mood_party'] > 0.4:
        mood_tags.append("派對 Party")
    if avg['danceability'] > 0.6:
        mood_tags.append("律動感 Groovy")
    if not mood_tags:
        mood_tags.append("中性 Neutral")
    brief_lines.append(f"🏷️ 風格標籤：{' / '.join(mood_tags)}")

    # ── 製作建議 ──
    suggestions = []
    if avg['danceability'] > 0.6:
        suggestions.append("強調節奏律動，可加入明顯的鼓組 groove")
    if avg['mood_relaxed'] > 0.5 and arousal < 4:
        suggestions.append("以柔和音色為主，可使用 pad、木吉他、鋼琴")
    if avg['mood_aggressive'] > 0.3 and arousal > 5:
        suggestions.append("可加入失真吉他或強力鼓點增加衝擊感")
    if avg['mood_sad'] > 0.4 and valence < 4:
        suggestions.append("選擇小調和聲，營造感傷氛圍")
    if avg['mood_happy'] > 0.5 and valence > 5:
        suggestions.append("選擇大調和聲，保持明亮的旋律線")
    if avg['mood_party'] > 0.4 and avg['danceability'] > 0.5:
        suggestions.append("可加入合成器、電子鼓點，營造派對氛圍")
    if arousal < 3:
        suggestions.append("注意留白與空間感，不要過度編曲")
    if not suggestions:
        suggestions.append("依照業主情緒描述自由發揮")

    brief_lines.append("\n💡 製作建議：")
    for s in suggestions:
        brief_lines.append(f"   • {s}")

    return "\n".join(brief_lines)


def recommend_for_musician(mood):
    """音樂人版：顯示聲學參數 + 自動生成聲學規格建議"""
    if not mood.strip():
        return "請輸入情緒描述"

    results = recommend(mood, top_k=3, return_results=True)

    # ── 每首歌的聲學數據 ──
    output = "🎸 業主需求分析：\n\n"
    feature_rows = []

    for i, (idx, score) in enumerate(results):
        title = song_library.iloc[idx]['title']
        feature_row = song_features[song_features['title'] == title]

        output += f"{i+1}. {title}\n"
        output += f"   相似度：{score:.3f}\n"

        if not feature_row.empty:
            row = feature_row.iloc[0]
            feature_rows.append(row)
            output += f"   BPM：{row['bpm']:.0f}\n"
            output += f"   Valence：{row['valence']:.2f}　Arousal：{row['arousal']:.2f}\n"
            output += f"   Mood — "
            output += f"Happy {row['mood_happy']:.2f} / "
            output += f"Sad {row['mood_sad']:.2f} / "
            output += f"Aggressive {row['mood_aggressive']:.2f} / "
            output += f"Relaxed {row['mood_relaxed']:.2f} / "
            output += f"Party {row['mood_party']:.2f}\n"
            output += f"   Danceability：{row['danceability']:.2f}\n"
        output += "\n"

    # ── 聲學規格書 ──
    if feature_rows:
        import pandas as pd
        avg = pd.DataFrame(feature_rows)[
            ['bpm', 'valence', 'arousal',
             'mood_happy', 'mood_sad', 'mood_aggressive',
             'mood_relaxed', 'mood_party', 'danceability']
        ].mean()

        output += "━" * 40 + "\n"
        output += "📋 聲學規格建議書（根據以上參考曲目自動生成）\n\n"
        output += generate_acoustic_brief(avg)
        output += "\n"

    return output

# 建立雙介面
with gr.Blocks(title="Timbre Audio-to-Brief Engine") as demo:
    gr.Markdown("# 🎵 Timbre Audio-to-Brief Engine")
    gr.Markdown("輸入情緒描述，AI 幫你找到最匹配的參考音樂")
    
    with gr.Row():
        mood_input = gr.Textbox(
            placeholder="描述你的情緒或場景，例如：深夜開車，有點孤獨...",
            label="情緒描述",
            lines=2
        )
    
    with gr.Row():
        client_btn = gr.Button("🎬 我是業主（找參考音樂）", variant="primary")
        musician_btn = gr.Button("🎸 我是音樂人（看聲學規格）", variant="secondary")
    
    output_box = gr.Textbox(label="推薦結果", lines=15)
    
    client_btn.click(fn=recommend_for_client, inputs=mood_input, outputs=output_box)
    musician_btn.click(fn=recommend_for_musician, inputs=mood_input, outputs=output_box)

demo.launch(server_name="0.0.0.0", server_port=7860)