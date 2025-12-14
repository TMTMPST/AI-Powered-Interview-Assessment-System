# 🧠 AI-Powered Interview Assessment System

Sistem penilaian interview otomatis menggunakan AI untuk membantu HR dalam mengevaluasi kandidat secara objektif dan konsisten.

## ✨ Fitur Utama

- 📹 **Upload Video Interview** - Support format mp4, mkv, mov
- 🎙️ **Speech-to-Text** - Transkripsi otomatis menggunakan Faster-Whisper
- 🤖 **AI Evaluation** - Penilaian jawaban menggunakan LLM (Llama 3.1)
- 📝 **Custom Rubrics** - HR bisa define pertanyaan dan kriteria penilaian sendiri
- 💾 **Template Management** - Save/load template pertanyaan untuk reuse
- 📊 **Detailed Reports** - Export hasil dalam format JSON dan CSV
- ✅ **Auto Decision** - Keputusan PASS/Need Review otomatis berdasarkan threshold

## Quick Start

### 1. Prerequisites

- Python 3.11+
- FFmpeg (`sudo apt install ffmpeg`)
- Ollama dengan model Llama 3.1
- (Optional) NVIDIA GPU untuk STT lebih cepat

### 2. Install Ollama & Model

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.1

# Start Ollama server
ollama serve
```

### 3. Setup Project

```bash
# Clone repository
git clone <repo-url>
cd AI-Powered-Interview-Assessment-System

# Run startup script (otomatis setup venv & dependencies)
./start.sh
```

Atau manual:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-app.txt

# Run Streamlit
streamlit run UI.py
```

### 4. Access Web UI

Buka browser: **http://localhost:8501**

## 📖 Cara Penggunaan

### Untuk HR/Recruiter:

1. **Setup Pertanyaan**

   - Buka aplikasi di browser
   - Di bagian "Pertanyaan & Rubrik Penilaian", tambah/edit pertanyaan
   - Isi rubrik penilaian (skor 0-4) untuk setiap pertanyaan
   - Klik "Simpan Template" untuk reuse di lain waktu

2. **Upload Video Interview**

   - Di section "Upload Video", pilih file video
   - (Optional) Masukkan project score jika ada

3. **Run Assessment**

   - Klik tombol "🚀 Jalankan AI Assessment"
   - Tunggu proses:
     - Ekstraksi audio dari video
     - Transkripsi speech-to-text
     - Evaluasi AI untuk setiap pertanyaan
   - Review hasil penilaian

4. **Export Hasil**
   - Download JSON untuk integrasi sistem
   - Download CSV untuk analisis lebih lanjut

## 🛠️ Konfigurasi

Edit file `UI.py` di bagian atas untuk customize:

```python
# Whisper Model (STT)
WHISPER_MODEL_SIZE = "large-v3"   # Options: large-v3, medium, small
WHISPER_DEVICE = "cuda"           # cuda (GPU) atau cpu
WHISPER_COMPUTE_TYPE = "int8_float16"  # int8 untuk VRAM lebih kecil

# Ollama (LLM)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"

# Scoring
MAX_SCORE_PER_QUESTION = 4
PASS_THRESHOLD = 75  # Final score untuk auto-pass
```

## 📁 Struktur Project

```
AI-Powered-Interview-Assessment-System/
├── UI.py                           # Main Streamlit application
├── capstone.ipynb                  # Original Jupyter notebook
├── TTS.py                          # Text-to-speech utilities
├── start.sh                        # Quick start script
├── requirements-app.txt            # Python dependencies
├── questions_template_example.json # Contoh template pertanyaan
├── DEPLOYMENT.md                   # Deployment guide
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── vid/                           # Video storage
└── temp_interview_data/           # Temporary audio files
```

## 📊 Format Output

### JSON Export

```json
{
  "assessorProfile": {
    "id": 1,
    "name": "AI Auto-Assessor"
  },
  "decision": "PASSED",
  "scoresOverview": {
    "project": 100,
    "interview": 85,
    "total": 92.5
  },
  "reviewChecklistResult": {
    "interviews": {
      "scores": [
        {
          "question": "...",
          "score": 4,
          "reason": "..."
        }
      ]
    }
  },
  "transcript": "Full interview transcript..."
}
```

## 🔧 Troubleshooting

### Ollama Not Connected

```bash
# Check status
curl http://localhost:11434/api/tags

# Start if not running
ollama serve
```

### GPU/CUDA Issues

Ubah di `UI.py`:

```python
WHISPER_DEVICE = "cpu"
```

### Large Upload Fails

Edit `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 1000  # in MB
```

## 🚀 Deployment

Untuk production deployment, lihat **[DEPLOYMENT.md](DEPLOYMENT.md)** yang mencakup:

- Streamlit Cloud deployment
- VPS/Cloud server setup (AWS/GCP/Azure)
- Docker containerization
- Nginx reverse proxy
- Systemd service configuration

## 🎯 Use Cases

- **Tech Recruitment** - Evaluate technical interview responses
- **Skills Assessment** - Standardized scoring for certifications
- **Mock Interview** - Practice sessions dengan feedback objektif
- **Bulk Screening** - Process multiple candidates efficiently

## 🔐 Security Notes

- Video upload limited (default 500MB)
- Local processing - data tidak keluar dari server
- Untuk production: gunakan HTTPS dan authentication

## 📈 Performance

- **With GPU**: ~2-3 menit untuk video 10 menit
- **CPU Only**: ~10-15 menit untuk video 10 menit
- Bisa di-optimize dengan model lebih kecil (medium/small)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

[Your License Here]

## 👨‍💻 Developer

Capstone Project - AI-Powered Interview Assessment System
