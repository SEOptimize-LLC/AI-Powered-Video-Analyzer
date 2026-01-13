# AI-Powered Video Analyzer

Upload a video file and get an executive summary with actionable insights — powered by AI.

## What It Does

This app saves you hours by automatically:
- **Transcribing** video audio using OpenAI Whisper API
- **Analyzing** key frames using Claude Vision
- **Generating** executive summaries and actionable insights using Claude Sonnet

**Cost:** ~$0.50-0.80 per hour of video (all processing happens in the cloud)

## Features

- Upload .mp4, .mov, .avi, .mkv, or .webm files
- Full audio transcription with timestamps
- Visual content analysis (slides, diagrams, screen recordings)
- Executive summary with key insights
- Actionable takeaways you can implement immediately
- Notable quotes and key topics
- Questions raised and further research suggestions
- Downloadable reports (Markdown format)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/AI-Powered-Video-Analyzer.git
cd AI-Powered-Video-Analyzer
```

### 2. Install dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg (required for audio extraction)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

### 3. Configure API Keys

**For local development:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and add your API keys
```

**For Streamlit Cloud deployment:**
Add your secrets in the Streamlit Cloud dashboard under Settings > Secrets:
```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| OpenAI | Whisper API (transcription) | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | Claude API (analysis) | [console.anthropic.com](https://console.anthropic.com/) |

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Whisper API | ~$0.006/min (~$0.36/hr) |
| Claude Sonnet (analysis) | ~$0.10-0.30/video |
| Claude Vision (frames) | ~$0.05-0.15/video |
| **Total** | **~$0.50-0.80 per hour of video** |

## Project Structure

```
AI-Powered-Video-Analyzer/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── src/
│   ├── video_processor.py  # Audio & frame extraction
│   ├── transcription.py    # Whisper API integration
│   ├── visual_analyzer.py  # Claude Vision analysis
│   └── analyzer.py         # Summary generation
└── .streamlit/
    └── secrets.toml.example
```

## Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your API keys in Settings > Secrets
5. Deploy!

## How It Works

```
┌──────────────────┐
│   Upload .mp4    │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ Audio  │ │ Key Frames │
│Extract │ │  Extract   │
│(ffmpeg)│ │ (OpenCV)   │
└───┬────┘ └─────┬──────┘
    ▼            ▼
┌────────┐ ┌────────────┐
│Whisper │ │Claude Vision│
│  API   │ │  Analysis   │
└───┬────┘ └─────┬──────┘
    └─────┬──────┘
          ▼
   ┌─────────────┐
   │Claude Sonnet│
   │  Analysis   │
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │   Report    │
   │  Download   │
   └─────────────┘
```

## License

MIT
