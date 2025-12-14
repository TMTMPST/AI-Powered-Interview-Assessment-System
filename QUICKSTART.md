# Quick Reference - AI Interview Assessment

## Cara Menjalankan

### Option 1: Automatic (Recommended)

```bash
./start.sh
```

### Option 2: Manual

```bash
source venv/bin/activate
streamlit run UI.py
```

Akses: http://localhost:8501

## Workflow HR

1. **Setup Pertanyaan** → Tambah/edit di table
2. **Simpan Template** → Klik "💾 Simpan Template"
3. **Upload Video** → Pilih file interview
4. **Run** → Klik "🚀 Jalankan AI Assessment"
5. **Review** → Lihat skor & alasan per pertanyaan
6. **Export** → Download JSON/CSV

## Template Pertanyaan

Contoh tersedia di: `questions_template_example.json`

Load dengan: Klik "📂 Muat Template Pertanyaan"

## Konfigurasi Cepat

Edit `UI.py`:

```python
# CPU only (tanpa GPU)
WHISPER_DEVICE = "cpu"

# Model lebih kecil (lebih cepat)
WHISPER_MODEL_SIZE = "medium"

# Threshold lulus
PASS_THRESHOLD = 75  # 0-100
```

## Troubleshooting

| Problem              | Solution                                             |
| -------------------- | ---------------------------------------------------- |
| Ollama not connected | `ollama serve` di terminal baru                      |
| GPU error            | Set `WHISPER_DEVICE = "cpu"`                         |
| Upload gagal         | Increase `maxUploadSize` di `.streamlit/config.toml` |
| Slow processing      | Gunakan model `medium` atau `small`                  |

## File Penting

- `UI.py` - Main app
- `requirements-app.txt` - Dependencies
- `DEPLOYMENT.md` - Production guide
- `.streamlit/config.toml` - Streamlit config

## Support

Check full documentation: [README.md](README.md)
