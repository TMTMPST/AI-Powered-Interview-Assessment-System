# 🚀 Deployment Guide - AI Interview Assessment System

## Prerequisites

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Start Ollama Server

Pastikan Ollama sudah running dengan model Llama 3.1:

```bash
# Start Ollama server (di terminal terpisah)
ollama serve

# Pull model jika belum ada
ollama pull llama3.1
```

### 3. Verify GPU/CUDA (Optional)

Jika menggunakan GPU untuk Whisper:

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Running Locally

### Development Mode

```bash
streamlit run UI.py
```

Aplikasi akan berjalan di: `http://localhost:8501`

### Production Mode (dengan custom port)

```bash
streamlit run UI.py --server.port 8080 --server.address 0.0.0.0
```

## Deployment Options

### Option 1: Streamlit Cloud (Recommended for Demo)

1. Push repository ke GitHub
2. Login ke [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy dari GitHub repository
4. **Note**: Streamlit Cloud tidak support GPU, jadi ubah konfigurasi:
   ```python
   WHISPER_DEVICE = "cpu"  # di UI.py
   ```

### Option 2: VPS/Cloud Server (AWS/GCP/Azure)

#### Setup Server

```bash
# Install system dependencies
sudo apt update
sudo apt install ffmpeg python3-pip

# Clone repository
git clone <your-repo-url>
cd AI-Powered-Interview-Assessment-System

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.1
```

#### Run with systemd (Production)

Create service file: `/etc/systemd/system/interview-assessment.service`

```ini
[Unit]
Description=AI Interview Assessment Streamlit App
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/AI-Powered-Interview-Assessment-System
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run UI.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable interview-assessment
sudo systemctl start interview-assessment
sudo systemctl status interview-assessment
```

#### Nginx Reverse Proxy (Optional)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Option 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "UI.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t interview-assessment .
docker run -p 8501:8501 interview-assessment
```

## Configuration

### UI.py Settings

Edit konfigurasi di bagian atas `UI.py`:

```python
# Model STT
WHISPER_MODEL_SIZE = "large-v3"  # atau "medium"/"small" untuk server lebih kecil
WHISPER_DEVICE = "cuda"          # "cpu" jika tanpa GPU
WHISPER_COMPUTE_TYPE = "int8_float16"  # "int8" untuk VRAM lebih kecil

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"

# Scoring
MAX_SCORE_PER_QUESTION = 4
PASS_THRESHOLD = 75
```

## Usage Guide

### For HR Users

1. **Setup Pertanyaan**:

   - Isi tabel pertanyaan dan rubrik penilaian
   - Klik "Simpan Template" untuk reuse di lain waktu
   - Klik "Muat Template" untuk load pertanyaan tersimpan

2. **Upload Video**:

   - Upload file video interview (mp4/mkv/mov)
   - Opsional: tambahkan project score

3. **Run Assessment**:

   - Klik "Jalankan AI Assessment"
   - Tunggu proses transkripsi dan evaluasi
   - Review hasil dan download report

4. **Export Results**:
   - Download JSON untuk integrasi sistem
   - Download CSV untuk analisis di Excel

## Troubleshooting

### Ollama Not Connected

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### CUDA/GPU Issues

```python
# Fallback to CPU in UI.py
WHISPER_DEVICE = "cpu"
```

### Memory Issues

```python
# Use smaller Whisper model
WHISPER_MODEL_SIZE = "medium"  # atau "small"
```

### Large Video Upload Fails

Edit `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 1000  # in MB
```

## Security Notes

- Untuk production, gunakan HTTPS (SSL/TLS)
- Set authentication jika diperlukan
- Limit upload size sesuai kebutuhan
- Monitor resource usage (CPU/GPU/RAM)

## Performance Tips

1. **GPU Acceleration**: Gunakan GPU untuk Whisper (10x lebih cepat)
2. **Smaller Models**: Gunakan `medium` atau `small` jika akurasi masih acceptable
3. **Batch Processing**: Process multiple videos di background
4. **Caching**: Streamlit auto-cache model loading

## Support

Jika ada masalah:

1. Check logs: `streamlit run UI.py --logger.level=debug`
2. Verify all dependencies installed
3. Ensure Ollama is running with correct model
