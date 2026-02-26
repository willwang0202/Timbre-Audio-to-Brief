---
title: Timbre Audio-to-Brief Engine
emoji: 🎵
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
---

# 🎵 Timbre Audio-to-Brief Engine

**Timbre** bridges the gap between clients and musicians. Describe a mood or scene in natural language, and the engine recommends reference tracks with matching acoustic characteristics — then auto-generates an **Acoustic Specification Brief** that musicians can actually execute.

> 輸入情緒描述，AI 幫你找到最匹配的參考音樂，並自動生成聲學規格建議書。

## Features

### 🎬 Client Mode — Find Reference Music
- Enter a mood description (e.g., *"late night drive, feeling lonely"*)
- Get top-3 matching songs from the library with YouTube search links
- Supports both **English** and **Chinese** input

### 🎸 Musician Mode — Acoustic Spec Brief
- View detailed acoustic features for each recommended track:
  - **BPM**, **Valence**, **Arousal**, **Mood Scores** (Happy / Sad / Aggressive / Relaxed / Party), **Danceability**
- Auto-generated **Acoustic Specification Brief** including:
  - 🎯 Suggested tempo range
  - ⚡ Energy & intensity level
  - 🌈 Emotional tone (bright vs. dark)
  - 🏷️ Style tags
  - 💡 Production suggestions (instruments, harmony, arrangement direction)

## How It Works

1. User inputs a mood/scene description
2. Keywords are detected and mapped to mood profiles (with weighted blending for multi-mood inputs)
3. A target feature vector is computed and compared against the song library via cosine similarity
4. Top matches are returned with full acoustic analysis
5. An actionable acoustic brief is generated from the averaged features

## Tech Stack

- **Audio Analysis**: [Essentia](https://essentia.upf.edu/) (MusicNN models for mood, valence, arousal, danceability)
- **Translation**: [Argos Translate](https://github.com/argosopentech/argos-translate) (Chinese → English)
- **UI**: [Gradio](https://gradio.app/)
- **Deployment**: Hugging Face Spaces

## Local Development

```bash
pip install -r requirements.txt
python download_models.py   # Download Essentia models (first time only)
python app.py
```

## License

MIT
