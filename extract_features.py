import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # suppress TF info/warning/error
os.environ["ESSENTIA_LOG_LEVEL"] = "silent"     # suppress Essentia warnings

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

"""
用 Essentia 預訓練模型提取歌曲特徵
取代原本的 librosa + CLAP 方案

輸出：song_features.csv，每首歌包含：
  - bpm, valence, arousal
  - mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party
  - danceability
"""
import os
import numpy as np
import pandas as pd
import librosa
from essentia.standard import (
    MonoLoader,
    TensorflowPredictMusiCNN,
    TensorflowPredict2D,
)

SONGS_FOLDER = "./songs"
MODELS_DIR = "./models"

song_library = pd.read_csv("song_library.csv")

# ── 載入模型 ──────────────────────────────────────────────
print("載入 Essentia 模型中...")

# Backbone: MSD-MusiCNN embedding extractor
embedding_model = TensorflowPredictMusiCNN(
    graphFilename=os.path.join(MODELS_DIR, "msd-musicnn-1.pb"),
    output="model/dense/BiasAdd",
)

# Classification heads
model_deam = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "deam-msd-musicnn-2.pb"),
    output="model/Identity",
)
model_happy = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_happy-msd-musicnn-1.pb"),
    output="model/Softmax",
)
model_sad = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_sad-msd-musicnn-1.pb"),
    output="model/Softmax",
)
model_aggressive = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_aggressive-msd-musicnn-1.pb"),
    output="model/Softmax",
)
model_relaxed = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_relaxed-msd-musicnn-1.pb"),
    output="model/Softmax",
)
model_party = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_party-msd-musicnn-1.pb"),
    output="model/Softmax",
)
model_dance = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "danceability-msd-musicnn-1.pb"),
    output="model/Softmax",
)

print("✅ 模型載入完成\n")

# ── 提取特徵 ──────────────────────────────────────────────
features_list = []

for i, row in song_library.iterrows():
    filepath = os.path.join(SONGS_FOLDER, row["filename"])
    print(f"[{i+1}/{len(song_library)}] 處理：{row['title']}")

    try:
        # Essentia MonoLoader (16kHz for MusiCNN)
        audio = MonoLoader(filename=filepath, sampleRate=16000, resampleQuality=4)()

        # MusiCNN embeddings
        embeddings = embedding_model(audio)

        # Arousal / Valence (DEAM, range [1-9])
        deam_preds = model_deam(embeddings)
        valence = float(np.mean(deam_preds[:, 0]))
        arousal = float(np.mean(deam_preds[:, 1]))

        # Mood classifiers (取 positive class 的機率)
        # 注意：不同模型的 class 順序不同，需要查看 metadata JSON
        mood_happy = float(np.mean(model_happy(embeddings)[:, 0]))       # ['happy', 'non_happy'] → idx 0
        mood_sad = float(np.mean(model_sad(embeddings)[:, 1]))           # ['non_sad', 'sad'] → idx 1
        mood_aggressive = float(np.mean(model_aggressive(embeddings)[:, 0]))  # ['aggressive', 'not_aggressive'] → idx 0
        mood_relaxed = float(np.mean(model_relaxed(embeddings)[:, 1]))   # ['non_relaxed', 'relaxed'] → idx 1
        mood_party = float(np.mean(model_party(embeddings)[:, 1]))       # ['non_party', 'party'] → idx 1

        # Danceability (取 danceable 的機率)
        danceability = float(np.mean(model_dance(embeddings)[:, 0]))     # ['danceable', 'not_danceable'] → idx 0

        # BPM (使用 Essentia direct estimation，避免 librosa 重複讀檔)
        from essentia.standard import PercivalBpmEstimator
        # Percival estimator 預設預期 44100Hz 或是更高，但我們的 audio 是 16kHz
        # 在這裡做簡單的特徵直接推算也比重讀硬碟快非常多。
        bpm_estimator = PercivalBpmEstimator()
        bpm = float(bpm_estimator(audio))

        features_list.append({
            "filename": row["filename"],
            "title": row["title"],
            "bpm": bpm,
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "mood_happy": round(mood_happy, 4),
            "mood_sad": round(mood_sad, 4),
            "mood_aggressive": round(mood_aggressive, 4),
            "mood_relaxed": round(mood_relaxed, 4),
            "mood_party": round(mood_party, 4),
            "danceability": round(danceability, 4),
        })

        print(f"  → valence={valence:.2f} arousal={arousal:.2f} "
              f"happy={mood_happy:.2f} sad={mood_sad:.2f} "
              f"party={mood_party:.2f} dance={danceability:.2f}")

    except Exception as e:
        print(f"  ❌ 失敗：{e}")

features_df = pd.DataFrame(features_list)
features_df.to_csv("song_features.csv", index=False)
print(f"\n✅ 完成！共處理 {len(features_list)} 首歌")
print(features_df[["title", "valence", "arousal", "mood_happy",
                    "mood_sad", "mood_party"]].to_string())