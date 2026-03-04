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

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/WCA0202/Timbre-Audio-to-Brief)

**Timbre** bridges the gap between clients and musicians. Describe a mood or scene in natural language, and the engine recommends reference tracks with matching acoustic characteristics — then auto-generates an **Acoustic Specification Brief** that musicians can actually execute.

> 輸入情緒描述，AI 幫你找到最匹配的參考音樂，並自動生成聲學規格建議書。

## Features

### 🌌 Emotion Explorer (New)
- An interactive, full-screen **emotion bubble UI** — navigate 4 layers of emotion selection:
  1. **Core emotion** (Joy, Calm, Sadness, Anger, Fear, Anticipation — or blend two)
  2. **Nuance** (fine-grained descriptor)
  3. **Somatic sensation** (where you feel it in the body)
  4. **Action urge** (what the emotion wants to do)
- After completing your constellation, click **"Find Reference Music"** to query the AI engine
- Bilingual: 中文 / English

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

1. Client selects emotions via the **Emotion Explorer** bubble UI (or types a description in Text Search)
2. The 4-layer selection path (e.g., *"Joy → Playful → Tingling in limbs → Expressing Freely"*) is sent to the AI engine
3. Keywords are mapped to mood profiles with weighted blending for multi-mood inputs
4. A target feature vector is compared against the song library via cosine similarity
5. Top matches are returned with full acoustic analysis
6. An actionable acoustic brief is generated from the averaged features

## Tech Stack

- **Emotion UI**: Custom interactive bubble physics engine (vanilla JS, no dependencies)
- **Audio Analysis**: [Essentia](https://essentia.upf.edu/) (MusicNN models for mood, valence, arousal, danceability)
- **Translation**: [Argos Translate](https://github.com/argosopentech/argos-translate) (Chinese → English)
- **UI**: [Gradio](https://gradio.app/) + FastAPI custom route for the Emotion Explorer
- **Deployment**: Hugging Face Spaces

## Local Development

```bash
pip install -r requirements.txt
python download_models.py   # Download Essentia models (first time only)
python app.py
```

## License

MIT
