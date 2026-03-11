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

# ── 載入特徵數據（single source of truth）─────────────────
song_data = pd.read_csv("song_features.csv")
song_data["emotion"] = song_data["emotion"].fillna("focused")

# 用於推薦的特徵欄位
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

_feat_min = feature_matrix.min()
_feat_max = feature_matrix.max()
feature_matrix = (feature_matrix - _feat_min) / (_feat_max - _feat_min + 1e-8)

feature_vectors = feature_matrix.values  # shape: (n_songs, n_features)

# ── Derive MOOD_PROFILES from actual data medians ─────────
# This guarantees targets live in the same normalized space as features.
# Hand-tuned fallbacks are used only for categories absent from the data.
_FALLBACK_PROFILES = {
    "party":               {"valence": 0.85, "arousal": 0.9,  "bpm": 0.75, "mood_happy": 0.7,  "mood_sad": 0.02, "mood_aggressive": 0.15, "mood_relaxed": 0.05, "mood_party": 0.95, "danceability": 0.95},
    "euphoric":            {"valence": 0.95, "arousal": 0.95, "bpm": 0.8,  "mood_happy": 0.95, "mood_sad": 0.01, "mood_aggressive": 0.1,  "mood_relaxed": 0.05, "mood_party": 0.8,  "danceability": 0.8},
    "romantic_passionate": {"valence": 0.8,  "arousal": 0.75, "bpm": 0.6,  "mood_happy": 0.75, "mood_sad": 0.1,  "mood_aggressive": 0.1,  "mood_relaxed": 0.2,  "mood_party": 0.4,  "danceability": 0.5},
    "triumphant":          {"valence": 0.85, "arousal": 0.85, "bpm": 0.7,  "mood_happy": 0.6,  "mood_sad": 0.05, "mood_aggressive": 0.3,  "mood_relaxed": 0.05, "mood_party": 0.5,  "danceability": 0.4},
    "angry":               {"valence": 0.1,  "arousal": 0.95, "bpm": 0.85, "mood_happy": 0.02, "mood_sad": 0.15, "mood_aggressive": 0.95, "mood_relaxed": 0.02, "mood_party": 0.15, "danceability": 0.35},
    "epic_dark":           {"valence": 0.3,  "arousal": 0.9,  "bpm": 0.65, "mood_happy": 0.1,  "mood_sad": 0.2,  "mood_aggressive": 0.8,  "mood_relaxed": 0.05, "mood_party": 0.1,  "danceability": 0.2},
    "anxious":             {"valence": 0.2,  "arousal": 0.85, "bpm": 0.8,  "mood_happy": 0.05, "mood_sad": 0.3,  "mood_aggressive": 0.6,  "mood_relaxed": 0.02, "mood_party": 0.1,  "danceability": 0.2},
    "relaxed":             {"valence": 0.6,  "arousal": 0.25, "bpm": 0.3,  "mood_happy": 0.4,  "mood_sad": 0.1,  "mood_aggressive": 0.01, "mood_relaxed": 0.95, "mood_party": 0.05, "danceability": 0.3},
    "romantic_tender":     {"valence": 0.65, "arousal": 0.35, "bpm": 0.35, "mood_happy": 0.6,  "mood_sad": 0.2,  "mood_aggressive": 0.02, "mood_relaxed": 0.7,  "mood_party": 0.05, "danceability": 0.3},
    "hopeful":             {"valence": 0.75, "arousal": 0.45, "bpm": 0.45, "mood_happy": 0.7,  "mood_sad": 0.1,  "mood_aggressive": 0.05, "mood_relaxed": 0.6,  "mood_party": 0.1,  "danceability": 0.4},
    "nostalgic":           {"valence": 0.5,  "arousal": 0.3,  "bpm": 0.3,  "mood_happy": 0.3,  "mood_sad": 0.5,  "mood_aggressive": 0.02, "mood_relaxed": 0.7,  "mood_party": 0.05, "danceability": 0.2},
    "sad":                 {"valence": 0.15, "arousal": 0.2,  "bpm": 0.25, "mood_happy": 0.05, "mood_sad": 0.9,  "mood_aggressive": 0.05, "mood_relaxed": 0.5,  "mood_party": 0.05, "danceability": 0.1},
    "melancholic":         {"valence": 0.3,  "arousal": 0.25, "bpm": 0.25, "mood_happy": 0.1,  "mood_sad": 0.7,  "mood_aggressive": 0.05, "mood_relaxed": 0.6,  "mood_party": 0.02, "danceability": 0.1},
    "lonely":              {"valence": 0.2,  "arousal": 0.15, "bpm": 0.2,  "mood_happy": 0.05, "mood_sad": 0.8,  "mood_aggressive": 0.02, "mood_relaxed": 0.6,  "mood_party": 0.02, "danceability": 0.05},
    "dark_ambient":        {"valence": 0.1,  "arousal": 0.1,  "bpm": 0.1,  "mood_happy": 0.02, "mood_sad": 0.6,  "mood_aggressive": 0.1,  "mood_relaxed": 0.8,  "mood_party": 0.01, "danceability": 0.05},
    "focused":             {"valence": 0.5,  "arousal": 0.5,  "bpm": 0.5,  "mood_happy": 0.2,  "mood_sad": 0.2,  "mood_aggressive": 0.1,  "mood_relaxed": 0.6,  "mood_party": 0.1,  "danceability": 0.4},
}

def _compute_data_profiles():
    """Compute per-category median feature vectors from normalized song data."""
    profiles = {}
    for emotion in song_data["emotion"].unique():
        mask = song_data["emotion"] == emotion
        if mask.sum() < 3:
            continue
        median_vec = feature_matrix.loc[mask].median()
        profiles[emotion] = {col: float(median_vec[col]) for col in FEATURE_COLS}
    return profiles

_data_profiles = _compute_data_profiles()

# Merge: data-derived profiles take priority, fallbacks fill gaps
MOOD_PROFILES = {}
for emotion in _FALLBACK_PROFILES:
    MOOD_PROFILES[emotion] = _data_profiles.get(emotion, _FALLBACK_PROFILES[emotion])

print(f"✅ MOOD_PROFILES: {len(_data_profiles)} from data, "
      f"{len(MOOD_PROFILES) - len(_data_profiles)} from fallback")


# ── Category adjacency for sparse-category fallback ───────
EMOTION_NEIGHBORS = {
    "party":               ["euphoric", "triumphant"],
    "euphoric":            ["party", "hopeful", "romantic_passionate"],
    "romantic_passionate":  ["euphoric", "romantic_tender", "hopeful"],
    "triumphant":          ["epic_dark", "party", "euphoric"],
    "angry":               ["anxious", "epic_dark"],
    "epic_dark":           ["angry", "triumphant", "anxious"],
    "anxious":             ["angry", "epic_dark", "focused"],
    "relaxed":             ["romantic_tender", "hopeful", "nostalgic"],
    "romantic_tender":     ["relaxed", "romantic_passionate", "nostalgic"],
    "hopeful":             ["romantic_tender", "euphoric", "relaxed"],
    "nostalgic":           ["melancholic", "romantic_tender", "relaxed"],
    "sad":                 ["melancholic", "lonely"],
    "melancholic":         ["sad", "nostalgic", "lonely", "dark_ambient"],
    "lonely":              ["melancholic", "sad", "dark_ambient"],
    "dark_ambient":        ["lonely", "melancholic"],
    "focused":             ["nostalgic", "relaxed", "hopeful"],
}


# ── 語意描述（全 16 種情緒皆有對應描述）──────────────────
EMOTION_DESCRIPTIONS = {
    "party":               "dancing drinking friends celebration club night out energy fun",
    "euphoric":            "ecstatic bliss peak joy elation rush adrenaline flying high",
    "romantic_passionate": "falling in love heart racing butterflies first kiss passion desire",
    "triumphant":          "victory achievement powerful epic cinematic hero winning glory",
    "angry":               "rage aggressive intense dark heavy metal punk fighting fury",
    "epic_dark":           "cinematic dark powerful ominous dramatic tension villain battle",
    "anxious":             "nervous tense worried stressed panic rushing overwhelmed uneasy",
    "relaxed":             "calm peaceful lofi coffee reading sunday morning slow gentle",
    "romantic_tender":     "slow dance holding hands gentle kiss soft warm intimate tender",
    "hopeful":             "optimistic bright new beginning sunrise warm uplifting inspiring",
    "nostalgic":           "memories bittersweet longing wistful past fading old times hometown",
    "sad":                 "heartbreak crying alone grief loss breakup tears sorrow pain",
    "melancholic":         "gloomy somber cloudy autumn rain grey pensive reflective blue",
    "lonely":              "isolated empty alone midnight silence abandoned void desolate",
    "dark_ambient":        "atmospheric eerie drone space void minimal dark ambient haunting",
    "focused":             "studying working concentration productive chill background neutral steady",
}

print("預計算情緒語意向量中...")
emotion_embeddings = {
    emotion: semantic_model.encode(desc, convert_to_tensor=True)
    for emotion, desc in EMOTION_DESCRIPTIONS.items()
}
print("✅ 情緒語意向量準備完成\n")

def detect_emotion_semantic(text):
    """將輸入文字對應到最符合的 Emotion 標籤"""
    query_tensor = semantic_model.encode(text, convert_to_tensor=True)

    scores = {}
    for emotion, em_tensor in emotion_embeddings.items():
        cos_score = util.cos_sim(query_tensor, em_tensor)[0].cpu().numpy()[0]
        scores[emotion] = float(cos_score)

    best = max(scores, key=scores.get)
    return best, scores


def recommend(mood_description, top_k=5, return_results=False):
    """推薦歌曲 — returns list of dicts with song metadata + score."""
    if not mood_description or not mood_description.strip():
        if not return_results:
            print("  ⚠️ 請輸入情緒描述")
        return []

    # 1. Semantic emotion detection
    best_emotion, scores = detect_emotion_semantic(mood_description)

    if not return_results:
        print(f"  [情緒偵測] {mood_description} → {best_emotion} ({scores[best_emotion]:.3f})")

    # 2. Target feature vector from data-derived profiles
    target_vector = np.array([
        MOOD_PROFILES.get(best_emotion, MOOD_PROFILES["focused"])[col] for col in FEATURE_COLS
    ])

    # 3. Vectorized Euclidean similarity
    dists = np.linalg.norm(feature_vectors - target_vector, axis=1)
    sim_scores = 1.0 / (1.0 + dists)

    # 4. Tiered emotion boosting (primary + neighbors for sparse categories)
    is_primary = (song_data["emotion"] == best_emotion).values
    neighbors = EMOTION_NEIGHBORS.get(best_emotion, [])
    is_neighbor = song_data["emotion"].isin(neighbors).values

    primary_boost = 100.0
    neighbor_boost = 30.0
    final_scores = sim_scores + (is_primary * primary_boost) + (is_neighbor * neighbor_boost)

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    # 5. Build result dicts with metadata
    results = []
    for idx in top_indices:
        row = song_data.iloc[idx]
        results.append({
            "title":    row["title"],
            "filename": row["filename"],
            "score":    float(sim_scores[idx]),
            "emotion":  row["emotion"],
            "features": {col: float(row[col]) for col in
                         ["bpm", "valence", "arousal", "mood_happy", "mood_sad",
                          "mood_aggressive", "mood_relaxed", "mood_party", "danceability"]},
        })

    if not return_results:
        print(f"\n🎵 情緒描述：「{mood_description}」")
        for rank, r in enumerate(results):
            match_mark = "⭐" if r["emotion"] == best_emotion else ""
            print(f"  {rank + 1}. {r['title']} {match_mark} (特徵相似度: {r['score']:.3f})")

    return results

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
