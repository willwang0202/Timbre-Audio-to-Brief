import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ESSENTIA_LOG_LEVEL"] = "silent"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

"""
Timbre – Audio to Brief Engine
Feature extraction using Essentia Discogs-EffNet backbone for mood heads,
and MusiCNN for valence/arousal (no EffNet DEAM model exists).

Models required in ./models/:
  discogs-effnet-bs64-1.pb            ← already downloaded (18MB) ✅
  msd-musicnn-1.pb                    ← already downloaded (3MB)  ✅
  deam-msd-musicnn-2.pb               ← already downloaded (81K)  ✅
  mood_happy-discogs-effnet-1.pb      ← already downloaded (502K) ✅
  mood_sad-discogs-effnet-1.pb        ← already downloaded (502K) ✅
  mood_aggressive-discogs-effnet-1.pb ← already downloaded (502K) ✅
  mood_relaxed-discogs-effnet-1.pb    ← already downloaded (502K) ✅
  mood_party-discogs-effnet-1.pb      ← already downloaded (502K) ✅
  danceability-discogs-effnet-1.pb    ← already downloaded (502K) ✅
"""

import numpy as np
import pandas as pd
from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredictMusiCNN,
    TensorflowPredict2D,
    PercivalBpmEstimator,
)
import essentia
essentia.log.warningActive = False
essentia.log.infoActive = False

SONGS_FOLDER = "./songs"
MODELS_DIR   = "./models"

song_library = pd.read_csv("song_library.csv")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading Essentia models...")

# Backbone A: Discogs-EffNet → mood classification heads (1280-dim)
embedding_model_effnet = TensorflowPredictEffnetDiscogs(
    graphFilename=os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb"),
    output="PartitionedCall:1",
)

# Backbone B: MusiCNN → DEAM valence/arousal only (no EffNet DEAM model exists)
embedding_model_musicnn = TensorflowPredictMusiCNN(
    graphFilename=os.path.join(MODELS_DIR, "msd-musicnn-1.pb"),
    output="model/dense/BiasAdd",
)

# Valence / Arousal — DEAM on MusiCNN embeddings (scale 1–9)
model_deam = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "deam-msd-musicnn-2.pb"),
    output="model/Identity",
)

# Mood heads — all on EffNet embeddings
# Class order per model:
#   mood_happy:      ['happy', 'non_happy']            → positive idx 0
#   mood_sad:        ['non_sad', 'sad']                → positive idx 1
#   mood_aggressive: ['aggressive', 'non_aggressive']  → positive idx 0
#   mood_relaxed:    ['non_relaxed', 'relaxed']        → positive idx 1
#   mood_party:      ['non_party', 'party']            → positive idx 1
#   danceability:    ['danceable', 'non_danceable']    → positive idx 0
model_happy = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_happy-discogs-effnet-1.pb"),
    input="model/Placeholder",
    output="model/Softmax",
)
model_sad = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_sad-discogs-effnet-1.pb"),
    input="model/Placeholder",
    output="model/Softmax",
)
model_aggressive = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_aggressive-discogs-effnet-1.pb"),
    input="model/Placeholder",
    output="model/Softmax",
)
model_relaxed = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_relaxed-discogs-effnet-1.pb"),
    input="model/Placeholder",
    output="model/Softmax",
)
model_party = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "mood_party-discogs-effnet-1.pb"),
    input="model/Placeholder",
    output="model/Softmax",
)
model_dance = TensorflowPredict2D(
    graphFilename=os.path.join(MODELS_DIR, "danceability-discogs-effnet-1.pb"),
    input="model/Placeholder",
    output="model/Softmax",
)

bpm_estimator = PercivalBpmEstimator()

print("✅ Models loaded\n")

# ── Extract features ──────────────────────────────────────────────────────────
features_list = []

for i, row in song_library.iterrows():
    filepath = os.path.join(SONGS_FOLDER, row["filename"])
    print(f"[{i+1}/{len(song_library)}] Processing: {row['title']}")

    try:
        audio = MonoLoader(filename=filepath, sampleRate=16000, resampleQuality=4)()

        # EffNet embeddings → mood heads
        embeddings_effnet  = embedding_model_effnet(audio)

        # MusiCNN embeddings → DEAM only
        embeddings_musicnn = embedding_model_musicnn(audio)

        # Valence / Arousal (MusiCNN path)
        deam_preds = model_deam(embeddings_musicnn)
        valence = float(np.mean(deam_preds[:, 0]))
        arousal = float(np.mean(deam_preds[:, 1]))

        # Mood probabilities (EffNet path)
        mood_happy      = float(np.mean(model_happy(embeddings_effnet)[:, 0]))
        mood_sad        = float(np.mean(model_sad(embeddings_effnet)[:, 1]))
        mood_aggressive = float(np.mean(model_aggressive(embeddings_effnet)[:, 0]))
        mood_relaxed    = float(np.mean(model_relaxed(embeddings_effnet)[:, 1]))
        mood_party      = float(np.mean(model_party(embeddings_effnet)[:, 1]))
        danceability    = float(np.mean(model_dance(embeddings_effnet)[:, 0]))

        bpm = float(bpm_estimator(audio))

        features_list.append({
            "filename":        row["filename"],
            "title":           row["title"],
            "bpm":             round(bpm, 2),
            "valence":         round(valence, 4),
            "arousal":         round(arousal, 4),
            "mood_happy":      round(mood_happy, 4),
            "mood_sad":        round(mood_sad, 4),
            "mood_aggressive": round(mood_aggressive, 4),
            "mood_relaxed":    round(mood_relaxed, 4),
            "mood_party":      round(mood_party, 4),
            "danceability":    round(danceability, 4),
        })

        print(f"  → valence={valence:.2f}  arousal={arousal:.2f}  "
              f"happy={mood_happy:.2f}  sad={mood_sad:.2f}  "
              f"party={mood_party:.2f}  dance={danceability:.2f}  bpm={bpm:.0f}")

    except Exception as e:
        print(f"  ❌ Failed: {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
features_df = pd.DataFrame(features_list)
features_df.to_csv("song_features.csv", index=False)

print(f"\n✅ Done — processed {len(features_list)} / {len(song_library)} songs")
print(features_df[["title", "valence", "arousal", "mood_happy",
                    "mood_sad", "mood_party", "bpm"]].to_string())

# Sanity check: flag songs where no mood score exceeds 0.4
low_confidence = features_df[
    (features_df[["mood_happy","mood_sad","mood_aggressive",
                  "mood_relaxed","mood_party"]].max(axis=1)) < 0.4
]
if len(low_confidence):
    print(f"\n⚠️  {len(low_confidence)} songs have no dominant mood score > 0.4 — check class indices:")
    print(low_confidence[["title","mood_happy","mood_sad","mood_aggressive",
                           "mood_relaxed","mood_party"]].to_string())