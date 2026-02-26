import pandas as pd
import numpy as np

features = pd.read_csv('song_features.csv')

def classify_from_essentia(row):
    scores = {
        'party':      row['mood_party'],
        'happy':      row['mood_happy'],
        'sad':        row['mood_sad'],
        'relaxed':    row['mood_relaxed'],
        'aggressive': row['mood_aggressive'],
    }
    
    base = max(scores, key=scores.get)
    valence = row['valence']
    arousal = row['arousal']
    
    if base == 'party':
        if arousal > 6:
            return 'party'
        return 'triumphant'
    
    elif base == 'happy':
        if valence > 6 and arousal > 5:
            return 'hopeful'
        if valence > 6 and arousal <= 5:
            return 'romantic_tender'
        return 'romantic_passionate'
    
    elif base == 'sad':
        if arousal < 4:
            return 'melancholic'
        if valence < 4:
            return 'lonely'
        return 'sad'
    
    elif base == 'relaxed':
        if row['mood_sad'] > 0.5:  # 從 0.7 降到 0.5
            if arousal < 4:
                return 'melancholic'
            return 'sad'
        if valence > 6:
            return 'relaxed'
        elif valence < 4:
            return 'melancholic'
        else:
            return 'focused'
    
    elif base == 'aggressive':
        if arousal > 7:
            return 'angry'
        return 'anxious'
    
    return 'nostalgic'

features['emotion'] = features.apply(classify_from_essentia, axis=1)
features.to_csv('song_emotions.csv', index=False)

print("分佈：")
print(features['emotion'].value_counts())