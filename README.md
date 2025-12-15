# 🧠 AI-Powered Interview Assessment System

An automated interview assessment system using AI to help HR evaluate candidates objectively and consistently.

## ✨ Key Features

- 📹 **Multiple File Upload** - Support video (mp4, mkv, mov, avi, webm) and audio (mp3, wav, m4a, flac, ogg, aac)
- 🎙️ **Speech-to-Text** - Automatic transcription using Faster-Whisper (large-v3)
- 🤖 **AI Evaluation** - Intelligent scoring using LLM (Llama 3.1 via Ollama)
- 📝 **Custom Rubrics** - Define your own questions and scoring criteria
- 💾 **Template Management** - Save/load question templates for reuse
- 📊 **Detailed Reports** - Export results in JSON and CSV formats
- ⚡ **GPU Acceleration** - Optimized for NVIDIA GPUs (CUDA support)
- 🎨 **Professional UI** - Clean, modern interface without emojis
- ✅ **Auto Decision** - Automatic PASS/NEEDS REVIEW based on threshold

## 📚 Documentation

- **[User Guide](USER_GUIDE.md)** - Complete step-by-step usage instructions
- **[Quick Start](QUICKSTART.md)** - Fast setup instructions

## Quick Start

### 1. Prerequisites

- Python 3.13+
- FFmpeg (`sudo apt install ffmpeg`)
- Ollama with Llama 3.1 model
- (Optional) NVIDIA GPU with CUDA for faster processing

### 2. Install Ollama & Model

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull Llama 3.1 model
ollama pull llama3.1

# Start Ollama server (keep this running)
ollama serve
```

### 3. Setup Project

```bash
# Clone repository
git clone https://github.com/TMTMPST/AI-Powered-Interview-Assessment-System.git
cd AI-Powered-Interview-Assessment-System

# Run startup script (auto setup venv & dependencies)
./start.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run UI.py
```

### 4. Access Web UI

Open browser: **http://localhost:8501**

## 📖 How to Use

### For HR/Recruiters:

#### 1. **Setup Questions**

- Open the application in browser
- In "STEP 1: Questions & Rubric Management"
- Add/edit interview questions
- Define scoring rubrics (score 0-4) for each question
- Click "Download Template JSON" to save for later use

#### 2. **Upload Interview Files**

- In "STEP 2: Upload Interview Videos/Audios"
- Select multiple video/audio files
- **Important**: Files are matched by order (File 1 → Question 1, File 2 → Question 2)
- Preview files to verify correct mapping

#### 3. **Run Assessment**

- Click "Run Assessment" button
- Wait for processing:
  - Audio extraction from videos
  - Speech-to-text transcription
  - AI evaluation for each question
- Review detailed results

#### 4. **Export Results**

- Download JSON for system integration
- Download CSV for spreadsheet analysis

**For detailed instructions, see [USER_GUIDE.md](USER_GUIDE.md)**

## 🛠️ Configuration

### Sidebar Settings

- **Pass Threshold**: Adjust minimum passing percentage (default: 70%)
- **System Information**: View current STT model, LLM, device type

### Advanced Configuration

Edit `UI.py` to customize:

```python
# Whisper Model (STT)
WHISPER_MODEL_SIZE = "large-v3"   # Options: large-v3, medium, small
WHISPER_DEVICE = "cuda"           # cuda (GPU) or cpu
WHISPER_COMPUTE_TYPE = "float16"  # float16 (GPU) or int8 (CPU/low VRAM)

# Ollama (LLM)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"

# Scoring
MAX_SCORE_PER_QUESTION = 4
PASS_THRESHOLD = 70  # Default pass threshold percentage
```

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Video/    │────▶│   Faster-    │────▶│   Ollama    │
│   Audio     │     │   Whisper    │     │   Llama3.1  │
│   Upload    │     │   (STT)      │     │   (LLM)     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │ Transcript   │────▶│  Scoring &  │
                    │   Text       │     │  Evaluation │
                    └──────────────┘     └─────────────┘
```

**Processing Flow:**

1. User uploads video/audio files (one per question)
2. FFmpeg extracts audio from video files
3. Faster-Whisper transcribes audio to text
4. Ollama evaluates transcript against rubric
5. Scores calculated and results displayed
6. Export to JSON/CSV formats

## 📁 Project Structure

```
AI-Powered-Interview-Assessment-System/
├── UI.py                           # Main Streamlit application
├── STT.py                          # Speech-to-text utilities (legacy)
├── taketogether.py                 # Video concatenation tool
├── deploy_ngrok.py                 # Deployment script with ngrok
├── start.sh                        # Quick start script
├── requirements.txt                # Python dependencies
├── requirements-app.txt            # App-specific dependencies
├── USER_GUIDE.md                   # Complete usage documentation
├── DEPLOYMENT.md                   # Deployment guide
├── QUICKSTART.md                   # Fast setup guide
├── questions_template.json         # Question template
├── questions_template_example.json # Example template
├── capstone.ipynb                  # Development notebook
├── deploy_colab.ipynb              # Google Colab deployment
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── vid/                           # Video storage
├── temp_interview_data/           # Temporary audio files
└── venv/                          # Virtual environment
```

## 📊 Output Format

### JSON Export

```json
{
  "assessorProfile": {
    "id": 1,
    "name": "AI Auto-Assessor",
    "photoUrl": "https://ui-avatars.com/api/?name=AI"
  },
  "decision": "PASSED",
  "scoresOverview": {
    "interview": 85.5,
    "total": 85.5
  },
  "reviewChecklistResult": {
    "interviews": {
      "minScore": 0,
      "maxScore": 4,
      "scores": [
        {
          "question": "Tell me about a recent ML project...",
          "rubric": "Score 4: Detailed explanation...",
          "transcript": "I worked on a sentiment analysis...",
          "score": 4,
          "reason": "Candidate provided specific details..."
        }
      ]
    }
  }
}
```

### CSV Export

Simple spreadsheet format with:

- Question text
- Score (0-4)
- Evaluation reasoning

## 🔧 Troubleshooting

### Common Issues

#### Ollama Not Connected

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

#### GPU/CUDA Not Detected

Edit in `UI.py`:

```python
WHISPER_DEVICE = "cpu"  # Force CPU mode
WHISPER_COMPUTE_TYPE = "int8"  # CPU-optimized
```

#### Large File Upload Fails

Edit `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 1000  # Maximum size in MB
```

#### Slow Processing

- Check GPU usage in System Information
- Consider using smaller Whisper model (medium/small)
- Ensure PyTorch with CUDA is installed
- Close other GPU-intensive applications

#### Score Always Same Value

- Temperature and prompt have been optimized (v1.0+)
- Ensure transcripts are accurate
- Verify rubrics are clear and specific
- Check Ollama is responding correctly

For more troubleshooting help, see [USER_GUIDE.md](USER_GUIDE.md#troubleshooting)

## 🚀 Deployment Options

### Free Deployment

#### 1. ngrok (Recommended for testing)

```bash
# Install ngrok from https://ngrok.com
# Set up auth token
ngrok config add-authtoken <your-token>

# Run deployment script
python deploy_ngrok.py
```

#### 2. Google Colab (Free GPU)

- Open `deploy_colab.ipynb`
- Run all cells
- Get public ngrok URL
- Share with users

### Production Deployment

For production environments, see **[DEPLOYMENT.md](DEPLOYMENT.md)** which covers:

- Streamlit Cloud (free tier available)
- VPS/Cloud servers (AWS, GCP, Azure, DigitalOcean)
- Docker containerization
- Nginx reverse proxy setup
- SSL/HTTPS configuration
- Systemd service management

### Self-Hosted Requirements

- 4GB+ RAM (8GB recommended)
- 10GB+ storage
- NVIDIA GPU (optional, for faster processing)
- Ubuntu 20.04+ or similar Linux distro

## 🎯 Use Cases

- **Tech Recruitment** - Evaluate technical interview responses objectively
- **Skills Assessment** - Standardized scoring for certifications and skill tests
- **Mock Interviews** - Practice sessions with automated objective feedback
- **Bulk Screening** - Process multiple candidates efficiently with consistent criteria
- **Training Evaluation** - Assess training program outcomes with measurable metrics
- **Remote Hiring** - Asynchronous interview assessment for distributed teams

## 📈 Performance Benchmarks

### Processing Speed

- **With GPU (RTX 3060)**:
  - 10-minute video: ~2-3 minutes
  - Per question: ~30-60 seconds
- **CPU Only**:
  - 10-minute video: ~10-15 minutes
  - Per question: ~2-5 minutes

### Optimization Options

- Use smaller Whisper models (medium/small) for faster processing
- Enable VAD (Voice Activity Detection) to skip silence
- Adjust compute type based on hardware (float16/int8)
- Parallel processing for multiple questions (already implemented)

### Accuracy

- **Transcription**: Faster-Whisper large-v3 provides near-human accuracy
- **Evaluation**: Llama 3.1 with balanced temperature (0.5) for objective scoring
- **Score Distribution**: Natural variance across 0-4 scale based on answer quality

## 🔐 Security & Privacy

### Data Protection

- **Local Processing**: All data processed on your server/machine
- **No External APIs**: Uses self-hosted Ollama (no data sent to cloud)
- **Upload Limits**: Default 500MB file size limit (configurable)
- **Temporary Storage**: Audio files auto-cleaned after processing

### Production Security Recommendations

- Enable HTTPS/SSL for encrypted communication
- Implement user authentication (e.g., OAuth, LDAP)
- Set up rate limiting to prevent abuse
- Regular security updates for all dependencies
- Firewall rules to restrict access
- Data retention policies for interview files

### Privacy Compliance

- GDPR-compliant when self-hosted
- No data sharing with third parties
- Full control over data storage and deletion
- Audit logs for assessment activities (optional)

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **STT Engine**: Faster-Whisper (optimized Whisper implementation)
- **LLM**: Ollama with Llama 3.1 (local inference)
- **Audio Processing**: FFmpeg
- **GPU Acceleration**: PyTorch with CUDA support
- **Language**: Python 3.13+

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AI-Powered-Interview-Assessment-System.git

# Create development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make changes and test
streamlit run UI.py
```

## 📝 Changelog

### Version 1.0 (December 2025)

- ✨ Professional UI redesign with CSS styling
- 🎨 Removed all emojis for professional appearance
- 🔧 Optimized LLM prompt for balanced scoring (temperature 0.5)
- 🚀 GPU optimization (float16, VAD, parallel processing)
- 📦 Support for 11 file formats (5 video, 6 audio)
- 🎯 Video-to-question mapping architecture
- 📊 Enhanced export formats (JSON, CSV)
- 🔒 Session state fixes for template loading
- 📚 Complete documentation (USER_GUIDE, DEPLOYMENT)
- 🌐 ngrok deployment script for free public access

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Authors & Acknowledgments

**Developer**: Capstone Project Team
**Repository**: [TMTMPST/AI-Powered-Interview-Assessment-System](https://github.com/TMTMPST/AI-Powered-Interview-Assessment-System)

### Special Thanks

- OpenAI for Whisper model
- Ollama team for local LLM deployment
- Streamlit for the web framework
- Meta for Llama models

## 📞 Support

For questions, issues, or feature requests:

- 📖 Check [USER_GUIDE.md](USER_GUIDE.md) for usage help
- 🐛 Report bugs via [GitHub Issues](https://github.com/TMTMPST/AI-Powered-Interview-Assessment-System/issues)
- 💬 Discussions and Q&A on GitHub Discussions
- 📧 Contact: [Your Contact Info]

---

**Made with ❤️ for better, fairer interview assessments**
