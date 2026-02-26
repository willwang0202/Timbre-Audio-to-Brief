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
song_emotions = pd.read_csv("song_emotions.csv")[["title", "emotion"]].drop_duplicates(subset=["title"])
song_data = pd.merge(song_data, song_emotions, on="title", how="left")
song_data["emotion"] = song_data["emotion"].fillna("focused")

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
EMOTION_DESCRIPTIONS = {
    'sad':        "heartbreak crying alone grief loss breakup tears sorrow",
    'melancholic': "nostalgic bittersweet longing wistful memories fading away",
    'lonely':     "isolated empty alone midnight silence abandoned",
    'relaxed':    "calm peaceful lofi coffee reading sunday morning slow",
    'focused':    "studying working concentration late night deadline anxious determined",
    'hopeful':    "optimistic bright new beginning sunrise warm gentle",
    'romantic_passionate': "falling in love heart racing butterflies first kiss passion",
    'romantic_tender': "slow dance holding hands gentle kiss soft warm intimate",
    'party':      "dancing drinking friends celebration club night out energy",
    'triumphant': "victory achievement powerful epic cinematic hero winning",
    'anxious':    "nervous tense worried stressed panic rushing overwhelmed",
    'angry':      "rage aggressive intense dark heavy metal punk fighting",
}

# 預計算 mood description 的 embeddings
print("預計算情緒語意向量中...")
emotion_embeddings = {
    emotion: semantic_model.encode(desc, convert_to_tensor=True)
    for emotion, desc in EMOTION_DESCRIPTIONS.items()
}
print("✅ 語意向量準備完成\n")

def detect_emotion_semantic(text):
    """將輸入文字對應到最符合的 Emotion 標籤"""
    query_tensor = semantic_model.encode(text, convert_to_tensor=True)
    
    scores = {}
    for emotion, em_tensor in emotion_embeddings.items():
        cos_score = util.cos_sim(query_tensor, em_tensor)[0].cpu().numpy()[0]
        scores[emotion] = float(cos_score)
        
    best = max(scores, key=scores.get)
    return best, scores


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

    # 1. 使用 Sentence-Transformers 將情境描述映射到最接近的 Emotion
    best_emotion, scores = detect_emotion_semantic(mood_description)
    
    if not return_results:
        print(f"  [情緒偵測] {mood_description} → {best_emotion} ({scores[best_emotion]:.3f})")

    # 2. 獲取該情緒對應的目標特徵向量 (供後續排序使用)
    target_vector = np.array([
        MOOD_PROFILES.get(best_emotion, MOOD_PROFILES["focused"])[col] for col in FEATURE_COLS
    ])

    # 3. 計算所有歌曲的 Euclidean 相似度
    sim_scores = np.array([
        euclidean_sim(feature_vectors[i], target_vector)
        for i in range(len(feature_vectors))
    ])
    
    # 4. 根據情緒過濾歌庫 (賦予極高權重，讓符合 emotion 的歌曲優先排在前面)
    is_matching_emotion = (song_data["emotion"] == best_emotion).values
    final_scores = sim_scores + (is_matching_emotion * 100.0)
    
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    if not return_results:
        print(f"\n🎵 情緒描述：「{mood_description}」")
        for rank, idx in enumerate(top_indices):
            title = song_data.iloc[idx]["title"]
            real_score = sim_scores[idx]
            match_mark = "⭐" if is_matching_emotion[idx] else ""
            print(f"  {rank + 1}. {title} {match_mark} (特徵相似度: {real_score:.3f})")

    return list(zip(top_indices, sim_scores[top_indices]))


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