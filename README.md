# AI-Powered Video Analyzer

Upload a video file and get an executive summary with actionable insights — powered by AI.

## What It Does

This app saves you hours by automatically:
- **Transcribing** video audio using OpenAI Whisper API
- **Analyzing** key frames using AI vision models
- **Generating** executive summaries and actionable insights

**Cost:** ~$0.50-1.00 per hour of video (all processing happens in the cloud)

## Features

- Upload .mp4, .mov, .avi, .mkv, or .webm files
- Full audio transcription with timestamps
- Visual content analysis (slides, diagrams, screen recordings)
- **Choose your AI model** for analysis and vision tasks
- Executive summary with key insights
- Actionable takeaways you can implement immediately
- Notable quotes and key topics
- Questions raised and further research suggestions
- Downloadable reports (Markdown format)

## Available Models (via OpenRouter)

| Model | Best For |
|-------|----------|
| **Claude Sonnet 4.5** | Deep analysis, nuanced insights |
| **GPT-5 Mini** | Fast, cost-effective analysis |
| **Gemini 3 Flash** | Quick visual analysis, budget-friendly |

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
OPENAI_API_KEY = "sk-..."           # For Whisper transcription
OPENROUTER_API_KEY = "sk-or-..."    # For AI analysis
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
| OpenRouter | AI models (Claude, GPT, Gemini) | [openrouter.ai/keys](https://openrouter.ai/keys) |

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Whisper API | ~$0.006/min (~$0.36/hr) |
| OpenRouter (varies by model) | ~$0.15-0.50/video |
| **Total** | **~$0.50-1.00 per hour of video** |

## Project Structure

```
AI-Powered-Video-Analyzer/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── src/
│   ├── video_processor.py  # Audio & frame extraction
│   ├── transcription.py    # Whisper API integration
│   ├── visual_analyzer.py  # Vision model analysis (OpenRouter)
│   └── analyzer.py         # Summary generation (OpenRouter)
└── .streamlit/
    └── secrets.toml.example
```

## Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your API keys in Settings > Secrets:
   - `OPENAI_API_KEY`
   - `OPENROUTER_API_KEY`
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
┌────────┐ ┌─────────────────┐
│Whisper │ │  OpenRouter     │
│  API   │ │  Vision Model   │
└───┬────┘ └───────┬─────────┘
    └─────┬────────┘
          ▼
   ┌──────────────────┐
   │   OpenRouter     │
   │  Analysis Model  │
   └────────┬─────────┘
            ▼
   ┌─────────────┐
   │   Report    │
   │  Download   │
   └─────────────┘
```

## License

MIT
