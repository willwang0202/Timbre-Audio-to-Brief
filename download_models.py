"""
下載 Essentia 預訓練模型權重（HF Spaces 啟動時自動執行）
"""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BASE_URL = "https://essentia.upf.edu/models"

MODELS = {
    "msd-musicnn-1.pb": f"{BASE_URL}/feature-extractors/musicnn/msd-musicnn-1.pb",
    "deam-msd-musicnn-2.pb": f"{BASE_URL}/classification-heads/deam/deam-msd-musicnn-2.pb",
    "mood_happy-msd-musicnn-1.pb": f"{BASE_URL}/classification-heads/mood_happy/mood_happy-msd-musicnn-1.pb",
    "mood_sad-msd-musicnn-1.pb": f"{BASE_URL}/classification-heads/mood_sad/mood_sad-msd-musicnn-1.pb",
    "mood_aggressive-msd-musicnn-1.pb": f"{BASE_URL}/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1.pb",
    "mood_relaxed-msd-musicnn-1.pb": f"{BASE_URL}/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1.pb",
    "mood_party-msd-musicnn-1.pb": f"{BASE_URL}/classification-heads/mood_party/mood_party-msd-musicnn-1.pb",
    "danceability-msd-musicnn-1.pb": f"{BASE_URL}/classification-heads/danceability/danceability-msd-musicnn-1.pb",
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
