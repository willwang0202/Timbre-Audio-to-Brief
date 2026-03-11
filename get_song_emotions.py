"""
Timbre – Emotion classifier
Assigns one of 16 emotion labels to each song based on Essentia features.

Categories (must stay in sync with MOOD_PROFILES in recommend_v2.py
and EMOTION_DESCRIPTIONS in recommend_v2.py):

  High Arousal + High Valence:  party, euphoric, romantic_passionate, triumphant
  High Arousal + Low Valence:   angry, epic_dark, anxious
  Low Arousal + High Valence:   relaxed, romantic_tender, hopeful
  Low Arousal + Low/Mid Valence: sad, melancholic, lonely, nostalgic, dark_ambient
  Neutral:                      focused
"""
import pandas as pd
import numpy as np

features = pd.read_csv('song_features.csv')

# ── Data-driven thresholds ────────────────────────────────────────────────────
V_HIGH = features['valence'].quantile(0.70)
V_LOW  = features['valence'].quantile(0.30)
V_MID  = features['valence'].quantile(0.50)
A_HIGH = features['arousal'].quantile(0.70)
A_LOW  = features['arousal'].quantile(0.30)
A_MID  = features['arousal'].quantile(0.50)
A_TOP  = features['arousal'].quantile(0.90)

print(f"Thresholds: V_HIGH={V_HIGH:.2f}  V_MID={V_MID:.2f}  V_LOW={V_LOW:.2f}  "
      f"A_HIGH={A_HIGH:.2f}  A_MID={A_MID:.2f}  A_LOW={A_LOW:.2f}  A_TOP={A_TOP:.2f}\n")

# ── Classifier ────────────────────────────────────────────────────────────────
def classify_from_essentia(row):
    scores = {
        'party':      row['mood_party'],
        'happy':      row['mood_happy'],
        'sad':        row['mood_sad'],
        'relaxed':    row['mood_relaxed'],
        'aggressive': row['mood_aggressive'],
    }

    # Low-confidence songs → neutral
    if max(scores.values()) < 0.4:
        return 'focused'

    base    = max(scores, key=scores.get)
    valence = row['valence']
    arousal = row['arousal']
    sad_score    = row['mood_sad']
    happy_score  = row['mood_happy']
    dance        = row['danceability']
    aggressive   = row['mood_aggressive']
    relaxed      = row['mood_relaxed']

    # ── Party base ────────────────────────────────────────────
    if base == 'party':
        if arousal > A_HIGH and valence > V_HIGH:
            return 'euphoric'
        if arousal > A_HIGH:
            return 'party'
        # Moderate arousal — euphoric requires both high valence AND happy
        if valence > V_HIGH and happy_score > 0.5:
            return 'euphoric'
        if valence > V_HIGH:
            return 'hopeful'
        if happy_score > 0.4 and valence > V_MID:
            return 'hopeful'
        if sad_score > 0.4:
            return 'nostalgic'
        return 'focused'

    # ── Happy base ────────────────────────────────────────────
    elif base == 'happy':
        if arousal > A_HIGH and valence > V_HIGH:
            return 'euphoric'
        if arousal > A_HIGH:
            return 'romantic_passionate'
        if valence > V_HIGH:
            return 'hopeful'
        if arousal < A_LOW:
            return 'romantic_tender'
        return 'romantic_passionate'

    # ── Sad base ──────────────────────────────────────────────
    elif base == 'sad':
        if valence < V_LOW and arousal < A_LOW:
            return 'lonely'
        if arousal < A_LOW:
            return 'melancholic'
        if arousal > A_HIGH:
            return 'sad'
        # Moderate arousal sadness
        if valence < V_LOW:
            return 'lonely'
        return 'sad'

    # ── Relaxed base (largest group — needs careful splitting) ─
    elif base == 'relaxed':
        # Strongly sad + relaxed
        if sad_score > 0.7:
            if arousal < A_LOW and valence < V_LOW:
                return 'dark_ambient'
            if arousal < A_LOW:
                return 'melancholic'
            return 'sad'

        # Bittersweet (moderate sadness)
        if sad_score > 0.5:
            if valence < V_LOW:
                return 'melancholic'
            if arousal < A_LOW:
                return 'nostalgic'
            return 'nostalgic'

        # Genuinely relaxed (sad_score < 0.5)
        if valence > V_HIGH:
            if happy_score > 0.4:
                return 'relaxed'
            return 'hopeful'

        if valence < V_LOW:
            if arousal < A_LOW and sad_score > 0.3:
                return 'dark_ambient'
            if arousal < A_LOW:
                return 'melancholic'
            return 'nostalgic'

        # Mid valence, relaxed
        if happy_score > 0.3 and arousal > A_MID:
            return 'focused'
        if happy_score > 0.3:
            return 'romantic_tender'
        if arousal < A_LOW:
            return 'nostalgic'
        return 'focused'

    # ── Aggressive base ───────────────────────────────────────
    elif base == 'aggressive':
        if arousal > A_TOP:
            return 'angry'
        if valence > V_HIGH:
            return 'triumphant'
        if arousal > A_HIGH:
            return 'epic_dark'
        if valence > V_MID:
            return 'triumphant'
        return 'anxious'

    return 'focused'


features['emotion'] = features.apply(classify_from_essentia, axis=1)
features.to_csv('song_features.csv', index=False)

print("Distribution:")
print(features['emotion'].value_counts().to_string())
print(f"\nTotal: {len(features)} songs across {features['emotion'].nunique()} emotions")
