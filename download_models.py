"""
下載 Essentia 預訓練模型權重
只需要跑一次：python download_models.py
"""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BASE_URL = "https://essentia.upf.edu/models"

# 需要的模型列表
MODELS = {
    # Backbone A: MusiCNN (for DEAM valence/arousal)
    "msd-musicnn-1.pb": f"{BASE_URL}/feature-extractors/musicnn/msd-musicnn-1.pb",

    # Valence/Arousal (DEAM dataset, range [1,9]) on MusiCNN embeddings
    "deam-msd-musicnn-2.pb": f"{BASE_URL}/classification-heads/deam/deam-msd-musicnn-2.pb",

    # Backbone B: Discogs-EffNet (for mood & danceability heads)
    "discogs-effnet-bs64-1.pb": f"{BASE_URL}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",

    # Mood classifiers on EffNet embeddings
    "mood_happy-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
    "mood_sad-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
    "mood_aggressive-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
    "mood_relaxed-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
    "mood_party-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_party/mood_party-discogs-effnet-1.pb",

    # Danceability on EffNet embeddings
    "danceability-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
}

def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            continue
        print(f"  ⬇️  下載模型：{filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ 完成：{filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ❌ 失敗：{filename} - {e}")

if __name__ == "__main__":
    ensure_models()
