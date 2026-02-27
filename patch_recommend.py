import pandas as pd
import numpy as np

with open("recommend_v2.py", "r") as f:
    code = f.read()

# 1. Add merging with song_emotions.csv
old_merge = '''song_data = pd.merge(song_library, song_features, on="title", how="inner")'''
new_merge = '''song_data = pd.merge(song_library, song_features, on="title", how="inner")
song_emotions = pd.read_csv("song_emotions.csv")[["title", "emotion"]].drop_duplicates("title")
song_data = pd.merge(song_data, song_emotions, on="title", how="left")
song_data["emotion"] = song_data["emotion"].fillna("focused")'''
code = code.replace(old_merge, new_merge)

code = code.replace("SIMILARITY_THRESHOLD = 0.25   # 最低相似度才會被視為匹配\nTOP_MOODS = 3                 # 最多取前 N 個 mood", "")

# 2. Replace everything from MOOD_DESCRIPTIONS to Euclidean distance
old_block = code[code.find("# ── 語意描述"):code.find("def euclidean_sim")]
new_block = '''# ── 語意描述（用 Sentence-Transformers 預測 12 種 ESSENTIA 情緒） ──
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

print("預計算情緒語意向量中...")
emotion_embeddings = {
    emotion: semantic_model.encode(desc, convert_to_tensor=True)
    for emotion, desc in EMOTION_DESCRIPTIONS.items()
}
print("✅ 情緒語意向量準備完成\\n")

def detect_emotion_semantic(text):
    """將輸入文字對應到最符合的 Emotion 標籤"""
    query_tensor = semantic_model.encode(text, convert_to_tensor=True)
    
    scores = {}
    for emotion, em_tensor in emotion_embeddings.items():
        cos_score = util.cos_sim(query_tensor, em_tensor)[0].cpu().numpy()[0]
        scores[emotion] = float(cos_score)
        
    best = max(scores, key=scores.get)
    return best, scores

'''
code = code.replace(old_block, new_block)

# 3. Replace recommend function
old_rec = code[code.find("def recommend("):code.find("# ── 測試")]
new_rec = '''def recommend(mood_description, top_k=5, return_results=False):
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
    # 這樣既能過濾，又能在該 emotion 歌曲不足 top_k 時，讓其他歌曲補上
    is_matching_emotion = (song_data["emotion"] == best_emotion).values
    final_scores = sim_scores + (is_matching_emotion * 100.0)
    
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    if not return_results:
        print(f"\\n🎵 情緒描述：「{mood_description}」")
        for rank, idx in enumerate(top_indices):
            title = song_data.iloc[idx]["title"]
            real_score = sim_scores[idx]
            match_mark = "⭐" if is_matching_emotion[idx] else ""
            print(f"  {rank + 1}. {title} {match_mark} (特徵相似度: {real_score:.3f})")

    return list(zip(top_indices, sim_scores[top_indices]))

'''
code = code.replace(old_rec, new_rec)

with open("recommend_v2.py", "w") as f:
    f.write(code)

