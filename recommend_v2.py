"""
Timbre 推薦引擎 v4（Semantic Matching 版）
使用 sentence-transformers 做語意比對，取代關鍵字匹配

策略：
1. 用戶輸入情緒描述（中英文皆可）
2. 用 sentence-transformer 計算與每個 mood profile 的語意相似度
3. 取相似度超過閾值的 mood → 加權混合目標特徵向量
4. 計算每首歌與目標的 euclidean similarity
5. 排序推薦
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ── 載入 sentence-transformer 模型 ────────────────────────
print("載入語意模型中...")
semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ 語意模型載入完成")

# 載入特徵數據
song_library = pd.read_csv("song_library.csv")
song_features = pd.read_csv("song_features.csv")

# 解決不對齊問題：透過 filename 或 title 進行 Merge，確保 index 絕對一致
song_data = pd.merge(song_library, song_features, on="title", how="inner")

# 用於推薦的特徵欄位（加入 BPM 讓高速/低速歌曲更好區分）
FEATURE_COLS = [
    "valence", "arousal", "bpm",
    "mood_happy", "mood_sad", "mood_aggressive",
    "mood_relaxed", "mood_party", "danceability",
]

# ── 修正 Essentia 模型的偏差 ──────────────────────────────
arousal_norm = (song_data["arousal"] - song_data["arousal"].min()) / \
               (song_data["arousal"].max() - song_data["arousal"].min() + 1e-8)

song_data["mood_relaxed_corrected"] = (
    song_data["mood_relaxed"]
    * (1 - song_data["mood_aggressive"])
    * (1 - arousal_norm * 0.6)
)

song_data["mood_sad_corrected"] = (
    song_data["mood_sad"]
    * (1 - arousal_norm * 0.3)
)

# 正規化特徵到 [0, 1]
feature_matrix = song_data[FEATURE_COLS].copy()
feature_matrix["mood_relaxed"] = song_data["mood_relaxed_corrected"]
feature_matrix["mood_sad"] = song_data["mood_sad_corrected"]
for col in FEATURE_COLS:
    min_val = feature_matrix[col].min()
    max_val = feature_matrix[col].max()
    feature_matrix[col] = (feature_matrix[col] - min_val) / (max_val - min_val + 1e-8)

feature_vectors = feature_matrix.values  # shape: (n_songs, n_features)

# ── 情緒描述 → 目標特徵向量 ──────────────────────────────
# valence, arousal, bpm, mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party, danceability
MOOD_PROFILES = {
    # 1. 高能量正面 (High Arousal, High Valence)
    "party": {
        "valence": 0.85, "arousal": 0.9, "bpm": 0.75,
        "mood_happy": 0.7, "mood_sad": 0.02, "mood_aggressive": 0.15,
        "mood_relaxed": 0.05, "mood_party": 0.95, "danceability": 0.95,
    },
    "euphoric": {
        "valence": 0.95, "arousal": 0.95, "bpm": 0.8,
        "mood_happy": 0.95, "mood_sad": 0.01, "mood_aggressive": 0.1,
        "mood_relaxed": 0.05, "mood_party": 0.8, "danceability": 0.8,
    },
    "romantic_passionate": {
        "valence": 0.8, "arousal": 0.75, "bpm": 0.6,
        "mood_happy": 0.75, "mood_sad": 0.1, "mood_aggressive": 0.1,
        "mood_relaxed": 0.2, "mood_party": 0.4, "danceability": 0.5,
    },
    "triumphant": {
        "valence": 0.85, "arousal": 0.85, "bpm": 0.7,
        "mood_happy": 0.6, "mood_sad": 0.05, "mood_aggressive": 0.3,
        "mood_relaxed": 0.05, "mood_party": 0.5, "danceability": 0.4,
    },

    # 2. 高能量負面 (High Arousal, Low Valence)
    "angry": {
        "valence": 0.1, "arousal": 0.95, "bpm": 0.85,
        "mood_happy": 0.02, "mood_sad": 0.15, "mood_aggressive": 0.95,
        "mood_relaxed": 0.02, "mood_party": 0.15, "danceability": 0.35,
    },
    "epic_dark": {
        "valence": 0.3, "arousal": 0.9, "bpm": 0.65,
        "mood_happy": 0.1, "mood_sad": 0.2, "mood_aggressive": 0.8,
        "mood_relaxed": 0.05, "mood_party": 0.1, "danceability": 0.2,
    },
    "anxious": {
        "valence": 0.2, "arousal": 0.85, "bpm": 0.8,
        "mood_happy": 0.05, "mood_sad": 0.3, "mood_aggressive": 0.6,
        "mood_relaxed": 0.02, "mood_party": 0.1, "danceability": 0.2,
    },

    # 3. 低能量正面 (Low Arousal, High Valence)
    "relaxed": {
        "valence": 0.6, "arousal": 0.25, "bpm": 0.3,
        "mood_happy": 0.4, "mood_sad": 0.1, "mood_aggressive": 0.01,
        "mood_relaxed": 0.95, "mood_party": 0.05, "danceability": 0.3,
    },
    "romantic_tender": {
        "valence": 0.65, "arousal": 0.35, "bpm": 0.35,
        "mood_happy": 0.6, "mood_sad": 0.2, "mood_aggressive": 0.02,
        "mood_relaxed": 0.7, "mood_party": 0.05, "danceability": 0.3,
    },
    "hopeful": {
        "valence": 0.75, "arousal": 0.45, "bpm": 0.45,
        "mood_happy": 0.7, "mood_sad": 0.1, "mood_aggressive": 0.05,
        "mood_relaxed": 0.6, "mood_party": 0.1, "danceability": 0.4,
    },
    "nostalgic": {
        "valence": 0.5, "arousal": 0.3, "bpm": 0.3,
        "mood_happy": 0.3, "mood_sad": 0.5, "mood_aggressive": 0.02,
        "mood_relaxed": 0.7, "mood_party": 0.05, "danceability": 0.2,
    },

    # 4. 低能量負面 (Low Arousal, Low Valence)
    "sad": {
        "valence": 0.15, "arousal": 0.2, "bpm": 0.25,
        "mood_happy": 0.05, "mood_sad": 0.9, "mood_aggressive": 0.05,
        "mood_relaxed": 0.5, "mood_party": 0.05, "danceability": 0.1,
    },
    "melancholic": {
        "valence": 0.3, "arousal": 0.25, "bpm": 0.25,
        "mood_happy": 0.1, "mood_sad": 0.7, "mood_aggressive": 0.05,
        "mood_relaxed": 0.6, "mood_party": 0.02, "danceability": 0.1,
    },
    "lonely": {
        "valence": 0.2, "arousal": 0.15, "bpm": 0.2,
        "mood_happy": 0.05, "mood_sad": 0.8, "mood_aggressive": 0.02,
        "mood_relaxed": 0.6, "mood_party": 0.02, "danceability": 0.05,
    },
    "dark_ambient": {
        "valence": 0.1, "arousal": 0.1, "bpm": 0.1,
        "mood_happy": 0.02, "mood_sad": 0.6, "mood_aggressive": 0.1,
        "mood_relaxed": 0.8, "mood_party": 0.01, "danceability": 0.05,
    },

    # 中性
    "focused": {
        "valence": 0.5, "arousal": 0.5, "bpm": 0.5,
        "mood_happy": 0.2, "mood_sad": 0.2, "mood_aggressive": 0.1,
        "mood_relaxed": 0.6, "mood_party": 0.1, "danceability": 0.4,
    },
}

# ── 語意描述（給 sentence-transformer 用）──────────────────
# 每個 mood 用多種表達方式描述，涵蓋中英文、同義詞、場景描述
MOOD_DESCRIPTIONS = {
    # 1. 高能量正面
    "party": "party dance club disco rave celebrate festival nightlife DJ lit groove 派對 跳舞 慶祝 夜店 KTV 嗨 狂歡 節慶",
    "euphoric": "euphoric ecstatic peak experience ultimate joy pure bliss absolute happiness amazing 狂喜 頂點 高峰體驗 極度快樂 超爽",
    "romantic_passionate": "passionate romance deep love intense desire fiery kiss burning love infatuation 熱戀 激情 渴望 熱烈的愛情 深愛 狂熱",
    "triumphant": "triumphant winning victory success champion overcoming heroic epic win glory 勝利 成就感 成功 冠軍 榮耀 克服 達成",

    # 2. 高能量負面
    "angry": "angry rage fury furious frustrated destroy pissed off mad aggressive violent 生氣 憤怒 暴躁 不爽 氣炸 崩潰 攻擊",
    "epic_dark": "epic dark cinematic tense intense boss battle intense war dramatic orchestral threat 史詩 黑暗 對決 緊張 危機 威脅 戰鬥",
    "anxious": "anxious panic nervous stressful tense uneasy racing thoughts worry jittery 焦慮 緊繃 恐慌 緊張 擔憂 神經質 壓力",

    # 3. 低能量正面
    "relaxed": "chill lofi vibe laid back mellow cozy relaxed lazy afternoon coffee quiet peace 放鬆 慵懶 舒服 悠閒 平靜 寧靜 休息",
    "romantic_tender": "tender romance gentle love sweetheart soft affection cuddling sweet warm 溫柔的愛情 輕柔 甜蜜 依偎 溫馨 浪漫",
    "hopeful": "hopeful optimistic bright future warming sunrise believing inspiring uplifting 希望 溫暖 期待 黎明 曙光 樂觀 振奮",
    "nostalgic": "nostalgic memories remembering missing the past childhood old times bittersweet 懷念 想念 回憶 以前 過去 逝去的美好",

    # 4. 低能量負面
    "sad": "sad depressed heartbroken crying grief mourning feeling down blue tears broken 傷心 悲傷 難過 哭 心碎 痛苦 悲痛",
    "melancholic": "melancholy contemplative wistful pensive gloomy rainy day sorrow reflective 憂鬱 惆悵 沉思 陰天 遺憾 傷感",
    "lonely": "lonely alone solitude empty hollow isolated solitary longing missing someone 孤獨 寂寞 空洞 孤直 一個人 沒人陪",
    "dark_ambient": "dark ambient heavy oppressive bleak scary haunting cold void abyss 黑暗 壓抑 沉重 深淵 冰冷 窒息 詭異",

    # 中性
    "focused": "focused studying concentration productive coding working deep work in the zone 專注 讀書 工作 專心 趕報告 集中精神 穩重",
}

# 預計算 mood description 的 embeddings
print("預計算語意向量中...")
mood_names = list(MOOD_DESCRIPTIONS.keys())
mood_texts = [MOOD_DESCRIPTIONS[m] for m in mood_names]
mood_embeddings = semantic_model.encode(mood_texts, convert_to_tensor=True)
print("✅ 語意向量準備完成\n")

# ── 語意匹配閾值設定 ──────────────────────────────────────
SIMILARITY_THRESHOLD = 0.25   # 最低相似度才會被視為匹配
TOP_MOODS = 3                 # 最多取前 N 個 mood


def detect_mood_profiles(text):
    """用語意相似度偵測匹配的 mood profiles"""
    query_embedding = semantic_model.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, mood_embeddings)[0].cpu().numpy()

    # 取所有超過閾值的 mood，按相似度排序
    scored = [(mood_names[i], float(cos_scores[i])) for i in range(len(mood_names))]
    scored.sort(key=lambda x: x[1], reverse=True)

    matched = []
    for mood, score in scored[:TOP_MOODS]:
        if score >= SIMILARITY_THRESHOLD:
            matched.append((mood, score))

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


def euclidean_sim(a, b):
    """Euclidean distance converted to a similarity score [0, 1]"""
    dist = np.linalg.norm(a - b)
    return 1 / (1 + dist)


def recommend(mood_description, top_k=5, return_results=False):
    """推薦歌曲"""
    if not mood_description or not mood_description.strip():
        if not return_results:
            print("  ⚠️ 請輸入情緒描述")
        return []

    # 語意匹配（直接支援中英文，不需翻譯）
    matched = detect_mood_profiles(mood_description)

    if matched:
        target_vector = blend_profiles(matched)
        detected_str = " + ".join(f"{m}({w:.2f})" for m, w in matched)
    else:
        # 預設用 focused（中性，不會太偏）
        target_vector = np.array([
            MOOD_PROFILES["focused"][col] for col in FEATURE_COLS
        ])
        detected_str = "default(focused)"

    if not return_results:
        print(f"  [偵測到] {detected_str}")

    # 計算相似度
    scores = np.array([
        euclidean_sim(feature_vectors[i], target_vector)
        for i in range(len(feature_vectors))
    ])

    top_indices = np.argsort(scores)[::-1][:top_k]

    if not return_results:
        print(f"\n🎵 情緒描述：「{mood_description}」")
        for rank, idx in enumerate(top_indices):
            title = song_data.iloc[idx]["title"]
            print(f"  {rank + 1}. {title}  (相似度: {scores[idx]:.3f})")

    return list(zip(top_indices, scores[top_indices]))


# ── 測試 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    recommend("feeling blue after a rainy day")
    print()
    recommend("傷心的分手之夜")
    print()
    recommend("和喜歡的人約會")
    print()
    recommend("I need something to pump me up for the gym")
    print()
    recommend("戀愛情境除了純粹的「甜蜜粉紅泡泡」，有時候更多的是對遠方另一半的想念。這種情緒比較綿長、溫柔，帶點渴望卻不悲傷")