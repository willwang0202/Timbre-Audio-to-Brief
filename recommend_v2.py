"""
Timbre 推薦引擎 v3（Essentia 版）
完全不依賴 CLAP 或 Spotify API

策略：
1. 用戶輸入情緒描述
2. 偵測所有匹配的關鍵字 → 混合多個目標特徵向量
3. 計算每首歌與目標的 cosine similarity
4. 排序推薦
"""
import numpy as np
import pandas as pd
import argostranslate.translate

# 載入特徵數據
song_library = pd.read_csv("song_library.csv")
song_features = pd.read_csv("song_features.csv")

# 用於推薦的特徵欄位（加入 BPM 讓高速/低速歌曲更好區分）
FEATURE_COLS = [
    "valence", "arousal", "bpm",
    "mood_happy", "mood_sad", "mood_aggressive",
    "mood_relaxed", "mood_party", "danceability",
]

# ── 修正 Essentia 模型的偏差 ──────────────────────────────
# 問題：Essentia 把所有管弦樂都判為 relaxed（包括緊張的管弦樂）
# 解法：
# 1. relaxed 分數用 arousal 修正（arousal 高 = 不太可能真的 relaxed）
# 2. sad 分數也用 arousal 修正（真正悲傷的歌通常 arousal 不高）

# arousal 範圍 [1,9]，低於 4 才是真正 calm
arousal_norm = (song_features["arousal"] - song_features["arousal"].min()) / \
               (song_features["arousal"].max() - song_features["arousal"].min() + 1e-8)

# relaxed 修正：arousal 越高，relaxed 越不可信
song_features["mood_relaxed_corrected"] = (
    song_features["mood_relaxed"]
    * (1 - song_features["mood_aggressive"])
    * (1 - arousal_norm * 0.6)  # arousal 高的歌 relaxed 打 6 折
)

# sad 修正：arousal 太高的 sad 不太對（激烈的歌不是 sad）
song_features["mood_sad_corrected"] = (
    song_features["mood_sad"]
    * (1 - arousal_norm * 0.3)  # 微調，不要太激進
)

# 正規化特徵到 [0, 1]
feature_matrix = song_features[FEATURE_COLS].copy()
# 用修正後的值取代原始的
feature_matrix["mood_relaxed"] = song_features["mood_relaxed_corrected"]
feature_matrix["mood_sad"] = song_features["mood_sad_corrected"]
for col in FEATURE_COLS:
    min_val = feature_matrix[col].min()
    max_val = feature_matrix[col].max()
    feature_matrix[col] = (feature_matrix[col] - min_val) / (max_val - min_val + 1e-8)

feature_vectors = feature_matrix.values  # shape: (n_songs, n_features)

# ── 情緒描述 → 目標特徵向量 ──────────────────────────────
# valence, arousal, bpm, mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party, danceability
MOOD_PROFILES = {
    "happy": {
        "valence": 0.9, "arousal": 0.7, "bpm": 0.6,
        "mood_happy": 0.9, "mood_sad": 0.05, "mood_aggressive": 0.05,
        "mood_relaxed": 0.3, "mood_party": 0.5, "danceability": 0.6,
    },
    "sad": {
        "valence": 0.15, "arousal": 0.2, "bpm": 0.25,
        "mood_happy": 0.05, "mood_sad": 0.9, "mood_aggressive": 0.05,
        "mood_relaxed": 0.5, "mood_party": 0.05, "danceability": 0.1,
    },
    "energetic": {
        "valence": 0.6, "arousal": 0.95, "bpm": 0.8,
        "mood_happy": 0.5, "mood_sad": 0.05, "mood_aggressive": 0.4,
        "mood_relaxed": 0.05, "mood_party": 0.6, "danceability": 0.85,
    },
    "calm": {
        "valence": 0.5, "arousal": 0.1, "bpm": 0.15,
        "mood_happy": 0.2, "mood_sad": 0.1, "mood_aggressive": 0.01,
        "mood_relaxed": 0.95, "mood_party": 0.05, "danceability": 0.15,
    },
    "chill": {
        "valence": 0.45, "arousal": 0.15, "bpm": 0.2,
        "mood_happy": 0.15, "mood_sad": 0.2, "mood_aggressive": 0.01,
        "mood_relaxed": 0.9, "mood_party": 0.05, "danceability": 0.2,
    },
    "romantic": {
        "valence": 0.55, "arousal": 0.5, "bpm": 0.45,
        "mood_happy": 0.45, "mood_sad": 0.55, "mood_aggressive": 0.02,
        "mood_relaxed": 0.55, "mood_party": 0.15, "danceability": 0.6,
    },
    "party": {
        "valence": 0.85, "arousal": 0.9, "bpm": 0.75,
        "mood_happy": 0.7, "mood_sad": 0.02, "mood_aggressive": 0.15,
        "mood_relaxed": 0.05, "mood_party": 0.95, "danceability": 0.95,
    },
    "angry": {
        "valence": 0.1, "arousal": 0.95, "bpm": 0.85,
        "mood_happy": 0.02, "mood_sad": 0.15, "mood_aggressive": 0.95,
        "mood_relaxed": 0.02, "mood_party": 0.15, "danceability": 0.35,
    },
    "focused": {
        "valence": 0.4, "arousal": 0.25, "bpm": 0.25,
        "mood_happy": 0.15, "mood_sad": 0.15, "mood_aggressive": 0.05,
        "mood_relaxed": 0.7, "mood_party": 0.05, "danceability": 0.2,
    },
    "epic": {
        "valence": 0.5, "arousal": 0.85, "bpm": 0.7,
        "mood_happy": 0.2, "mood_sad": 0.15, "mood_aggressive": 0.6,
        "mood_relaxed": 0.05, "mood_party": 0.2, "danceability": 0.25,
    },
    "nostalgic": {
        "valence": 0.35, "arousal": 0.25, "bpm": 0.25,
        "mood_happy": 0.2, "mood_sad": 0.6, "mood_aggressive": 0.02,
        "mood_relaxed": 0.7, "mood_party": 0.05, "danceability": 0.15,
    },
    "drive": {
        "valence": 0.55, "arousal": 0.55, "bpm": 0.55,
        "mood_happy": 0.3, "mood_sad": 0.1, "mood_aggressive": 0.1,
        "mood_relaxed": 0.4, "mood_party": 0.3, "danceability": 0.5,
    },
    "melancholy": {
        "valence": 0.2, "arousal": 0.15, "bpm": 0.2,
        "mood_happy": 0.05, "mood_sad": 0.85, "mood_aggressive": 0.02,
        "mood_relaxed": 0.7, "mood_party": 0.02, "danceability": 0.05,
    },
    "upbeat": {
        "valence": 0.8, "arousal": 0.75, "bpm": 0.7,
        "mood_happy": 0.8, "mood_sad": 0.05, "mood_aggressive": 0.1,
        "mood_relaxed": 0.15, "mood_party": 0.7, "danceability": 0.8,
    },
}

# 關鍵字映射（每個關鍵字有權重，越具體的關鍵字權重越高）
KEYWORD_MAP = {
    "happy": (["happy", "joy", "sunny", "cheerful", "sunshine", "refresh", "bright",
               "開心", "快樂", "高興", "愉快", "陽光"], 1.0),
    "sad": (["sad", "cry", "heartbreak", "depressed", "grief", "mourn",
             "傷心", "悲傷", "難過", "哭", "心碎", "痛苦"], 1.0),
    "energetic": (["pump", "energy", "workout", "determined", "fight", "power", "intense",
                   "熱血", "衝刺", "運動", "激動", "拼命", "燃燒"], 1.0),
    "calm": (["calm", "peaceful", "serene", "quiet", "gentle", "soft", "tranquil",
              "平靜", "安靜", "寧靜", "溫柔", "柔和"], 1.5),  # 高權重，因為 calm 意圖通常很明確
    "chill": (["chill", "lofi", "lo-fi", "vibe", "laid back", "mellow", "cozy",
               "放鬆", "慵懶", "舒服", "chill"], 1.5),
    "romantic": (["love", "romantic", "heart", "butterflies", "kiss", "date", "sweet",
                  "愛情", "浪漫", "甜蜜", "約會", "心動", "告白"], 1.0),
    "party": (["party", "dance", "club", "disco", "rave", "celebrate",
               "派對", "跳舞", "慶祝", "夜店", "KTV"], 1.0),
    "angry": (["angry", "rage", "fury", "destroy", "pissed", "mad",
               "生氣", "憤怒", "暴躁", "不爽"], 1.0),
    "focused": (["focus", "study", "concentrate", "productive", "coding", "work",
                 "專注", "讀書", "工作", "專心", "趕報告", "考試"], 1.0),
    "epic": (["epic", "cinematic", "grand", "heroic", "boss", "battle", "war",
              "史詩", "壯觀", "英雄", "戰鬥", "BOSS"], 1.0),
    "nostalgic": (["nostalgic", "memory", "remember", "miss", "past", "childhood",
                   "懷念", "回憶", "想念", "以前", "從前"], 1.0),
    "drive": (["drive", "driving", "road", "highway", "car", "night drive",
               "開車", "兜風", "公路"], 1.2),
    "melancholy": (["melancholy", "lonely", "alone", "solitude", "empty", "hollow",
                    "孤獨", "寂寞", "空虛", "一個人"], 0.8),  # 低權重，很多場景都有 "alone" 但不一定是憂鬱
    "upbeat": (["upbeat", "fun", "playful", "lively", "exciting",
                "好玩", "有趣", "活潑", "興奮"], 1.0),
}


def detect_mood_profiles(text):
    """偵測所有匹配的 mood 並回傳加權混合向量"""
    text_lower = text.lower()
    matched = []

    for mood, (keywords, weight) in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append((mood, weight))
                break  # 每個 mood 只匹配一次

    return matched


def blend_profiles(matched_moods):
    """把多個 mood profile 按權重混合"""
    if not matched_moods:
        return None

    total_weight = sum(w for _, w in matched_moods)
    blended = np.zeros(len(FEATURE_COLS))

    for mood, weight in matched_moods:
        profile = MOOD_PROFILES[mood]
        vec = np.array([profile[col] for col in FEATURE_COLS])
        blended += vec * (weight / total_weight)

    return blended


def translate_to_english(text):
    """中文→英文翻譯，帶有 fallback 機制"""
    if not text or not text.strip():
        return text
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    if ascii_ratio > 0.8:
        return text

    # 先嘗試 argostranslate
    try:
        translated = argostranslate.translate.translate(text, "zh", "en")
        if translated and translated.strip():
            print(f"  [翻譯] {text} → {translated}")
            return translated
    except Exception as e:
        print(f"  [翻譯失敗] argostranslate error: {e}")

    # Fallback: 用關鍵字映射表做基本轉換
    FALLBACK_MAP = {
        "開心": "happy", "快樂": "happy", "高興": "happy", "愉快": "happy",
        "傷心": "sad", "悲傷": "sad", "難過": "sad", "哭": "cry",
        "心碎": "heartbreak", "痛苦": "depressed",
        "平靜": "calm", "安靜": "quiet", "寧靜": "peaceful", "溫柔": "gentle",
        "放鬆": "chill", "慵懶": "chill", "舒服": "cozy",
        "熱血": "energetic", "衝刺": "energetic", "運動": "workout",
        "激動": "intense", "燃燒": "intense",
        "愛情": "love", "浪漫": "romantic", "甜蜜": "sweet",
        "約會": "date", "心動": "butterflies", "告白": "love",
        "派對": "party", "跳舞": "dance", "慶祝": "celebrate", "夜店": "club",
        "生氣": "angry", "憤怒": "angry", "暴躁": "rage", "不爽": "pissed",
        "專注": "focus", "讀書": "study", "工作": "work", "專心": "concentrate",
        "趕報告": "productive", "考試": "study",
        "史詩": "epic", "壯觀": "grand", "英雄": "heroic", "戰鬥": "battle",
        "懷念": "nostalgic", "回憶": "memory", "想念": "miss",
        "開車": "drive", "兜風": "driving", "公路": "highway",
        "孤獨": "lonely", "寂寞": "alone", "空虛": "empty", "一個人": "alone",
        "好玩": "fun", "有趣": "playful", "活潑": "lively", "興奮": "exciting",
        "深夜": "late night", "夜晚": "night", "早晨": "morning",
        "陽光": "sunny", "分手": "breakup", "想念": "miss",
    }
    fallback_parts = []
    for zh, en in FALLBACK_MAP.items():
        if zh in text:
            fallback_parts.append(en)
    if fallback_parts:
        result = " ".join(fallback_parts)
        print(f"  [翻譯 fallback] {text} → {result}")
        return result

    print(f"  [翻譯] 無法翻譯，使用原文: {text}")
    return text


def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def recommend(mood_description, top_k=5, return_results=False):
    """推薦歌曲"""
    if not mood_description or not mood_description.strip():
        if not return_results:
            print("  ⚠️ 請輸入情緒描述")
        return []

    # 翻譯（如果是中文）
    english = translate_to_english(mood_description)

    # 偵測所有匹配的 moods（同時用中文和英文偵測）
    matched = detect_mood_profiles(english)
    matched_cn = detect_mood_profiles(mood_description)

    # 合併（去重）
    seen = set(m for m, _ in matched)
    for mood, weight in matched_cn:
        if mood not in seen:
            matched.append((mood, weight))
            seen.add(mood)

    if matched:
        target_vector = blend_profiles(matched)
        detected_str = " + ".join(f"{m}({w:.1f})" for m, w in matched)
    else:
        # 預設用 calm（中性，不會太偏）
        target_vector = np.array([
            MOOD_PROFILES["calm"][col] for col in FEATURE_COLS
        ])
        detected_str = "default(calm)"

    if not return_results:
        print(f"  [偵測到] {detected_str}")

    # 計算相似度
    scores = np.array([
        cosine_sim(feature_vectors[i], target_vector)
        for i in range(len(feature_vectors))
    ])

    top_indices = np.argsort(scores)[::-1][:top_k]

    if not return_results:
        print(f"\n🎵 情緒描述：「{mood_description}」")
        if english != mood_description:
            print(f"   English: \"{english}\"")
        for rank, idx in enumerate(top_indices):
            title = song_library.iloc[idx]["title"]
            print(f"  {rank + 1}. {title}  (相似度: {scores[idx]:.3f})")

    return list(zip(top_indices, scores[top_indices]))


# ── 測試 ──────────────────────────────────────────────────
if __name__ == "__main__":
    recommend("傷心的分手之夜")
    recommend("和喜歡的人約會")
    recommend("戀愛情境除了純粹的「甜蜜粉紅泡泡」，有時候更多的是對遠方另一半的想念。這種情緒比較綿長、溫柔，帶點渴望卻不悲傷")