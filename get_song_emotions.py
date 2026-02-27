import pandas as pd
import numpy as np

features = pd.read_csv('song_features.csv')

# ── Data-driven thresholds ────────────────────────────────────────────────────
V_HIGH = features['valence'].quantile(0.70)
V_LOW  = features['valence'].quantile(0.30)
A_HIGH = features['arousal'].quantile(0.70)
A_LOW  = features['arousal'].quantile(0.30)
A_TOP  = features['arousal'].quantile(0.90)

print(f"Thresholds: V_HIGH={V_HIGH:.2f}  V_LOW={V_LOW:.2f}  "
      f"A_HIGH={A_HIGH:.2f}  A_LOW={A_LOW:.2f}  A_TOP={A_TOP:.2f}\n")

# ── Classifier ────────────────────────────────────────────────────────────────
def classify_from_essentia(row):
    scores = {
        'party':      row['mood_party'],
        'happy':      row['mood_happy'],
        'sad':        row['mood_sad'],
        'relaxed':    row['mood_relaxed'],
        'aggressive': row['mood_aggressive'],
    }

    if max(scores.values()) < 0.4:
        return 'ambiguous'

    base    = max(scores, key=scores.get)
    valence = row['valence']
    arousal = row['arousal']
    bpm     = row['bpm']
    sad_score = row['mood_sad']

    if base == 'party':
        if arousal > A_HIGH:
            return 'party'
        return 'triumphant'

    elif base == 'happy':
        if valence > V_HIGH and arousal > A_HIGH:
            return 'hopeful'
        if valence > V_HIGH and arousal <= A_HIGH:
            return 'romantic_tender'
        return 'romantic_passionate'

    elif base == 'sad':
        if arousal < A_LOW:
            return 'melancholic'
        if valence < V_LOW:
            return 'lonely'
        return 'sad'

    elif base == 'relaxed':
        if sad_score > 0.7:
            # Strongly sad + relaxed → melancholic/sad
            return 'melancholic' if arousal < A_LOW else 'sad'
        if sad_score > 0.5:
            # Bittersweet zone → nostalgic or sad
            return 'nostalgic' if bpm < 95 else 'sad'
        # Genuinely relaxed (sad_score < 0.5)
        if valence > V_HIGH:
            return 'relaxed'
        if valence < V_LOW:
            return 'melancholic'
        return 'nostalgic' if bpm < 100 else 'focused'

    elif base == 'aggressive':
        if arousal > A_TOP:
            return 'angry'
        return 'anxious'

    return 'nostalgic'


features['emotion'] = features.apply(classify_from_essentia, axis=1)
features.to_csv('song_emotions.csv', index=False)

print("Distribution:")
print(features['emotion'].value_counts())
print(f"\nTotal: {len(features)} songs across {features['emotion'].nunique()} emotions")