import os
import subprocess
import time
import sys
from pathlib import Path

# Auto-detect dan set LD_LIBRARY_PATH untuk CUDA libraries
def setup_cuda_env():
    try:
        import site
        site_packages = site.getsitepackages()
        
        for sp in site_packages:
            nvidia_libs = Path(sp) / "nvidia"
            if nvidia_libs.exists():
                # Cari semua folder nvidia-*/lib
                for lib_dir in nvidia_libs.glob("*/lib"):
                    if lib_dir.exists():
                        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                        if str(lib_dir) not in current_ld:
                            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current_ld}"
    except Exception as e:
        pass  # Silent fail, biar gak ganggu kalau gak ada CUDA

setup_cuda_env()

from faster_whisper import WhisperModel
from tqdm import tqdm

class Transcriber:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16"):
        """
        Inisialisasi Model Whisper.
        - model_size: 'large-v3' (Paling akurat untuk Indo/Inggris), 'medium', atau 'small'.
        - device: 'cuda' (Wajib pakai GPU NVIDIA biar cepat), 'cpu' (Lebih lambat tapi aman).
        - compute_type: 'float16' atau 'int8_float16' (Hemat VRAM).
        """
        print(f"Loading Whisper Model ({model_size}) on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("Model Whisper Siap!")
        except RuntimeError as e:
            if "libcublas" in str(e) or "CUDA" in str(e):
                print(f"CUDA Error: {e}")
                print("Mencoba fallback ke CPU...")
                try:
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    print("Model Whisper Siap (Running on CPU)!")
                except Exception as cpu_error:
                    print(f"Gagal load model di CPU: {cpu_error}")
                    raise cpu_error
            else:
                raise e
        except Exception as e:
            print(f"Gagal load model: {e}")
            print("Tips: Pastikan driver NVIDIA & CUDA Toolkit sudah terinstall.")
            raise e

    def convert_video_to_audio(self, video_path):
        """
        Mengambil suara dari video menggunakan FFmpeg.
        Output: file .wav 16kHz mono (Format favorit Whisper).
        """
        # Nama file audio disamakan dengan video tapi ganti ekstensi jadi .wav
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        
        # Cek kalau file audio sudah ada, gak perlu convert ulang (Cache sederhana)
        if os.path.exists(audio_path):
            print(f"ℹAudio file already exists: {audio_path}")
            return audio_path

        print(f"Extracting audio from: {video_path}...")
        
        # Command FFmpeg
        command = [
            "ffmpeg", "-i", video_path,
            "-ar", "16000",       # Sample rate 16kHz
            "-ac", "1",           # Mono channel
            "-c:a", "pcm_s16le",  # Codec WAV standard
            "-vn",                # No Video (Suara saja)
            "-y",                 # Overwrite output tanpa tanya
            audio_path
        ]
        
        # Jalankan FFmpeg diam-diam (suppress output)
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(audio_path):
            return audio_path
        else:
            raise FileNotFoundError("Gagal mengekstrak audio. Cek instalasi FFmpeg.")

    def transcribe(self, file_path):
        """
        Fungsi utama untuk mengubah Audio/Video menjadi Teks.
        """
        start_time = time.time()
        
        # 1. Pastikan inputnya Audio (.wav). Kalau Video, convert dulu.
        if file_path.endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm')):
            audio_path = self.convert_video_to_audio(file_path)
        else:
            audio_path = file_path

        # 2. Proses Transkripsi
        print("Mendengarkan & Mencatat (Transcribing)...")
        # beam_size=5 membuat whisper lebih teliti mencari kemungkinan kata terbaik
        segments, info = self.model.transcribe(audio_path, beam_size=5)

        # 3. Gabungkan hasil potongan teks
        full_text = []
        
        # Convert generator to list untuk mendapatkan total count
        segments_list = list(segments)
        
        # Tampilkan progress bar
        for segment in tqdm(segments_list, desc="Processing segments", unit="segment"):
            # Opsional: Bisa print per kalimat biar kelihatan progressnya
            # print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_text.append(segment.text)
        
        result_text = " ".join(full_text).strip()
        
        duration = time.time() - start_time
        print(f"Selesai dalam {duration:.2f} detik.")
        
        return result_text, info.language

# --- BAGIAN TESTING (Biar bisa ditest langsung lewat terminal) ---
if __name__ == "__main__":
    # Cara pakai: python stt_engine.py
    
    # Ganti ini dengan path video dummy kamu buat ngetest
    TEST_FILE = "interview_question_1.webm" 
    
    if os.path.exists(TEST_FILE):
        # Inisialisasi
        engine = Transcriber(model_size="medium.en") # Ganti 'small' kalau mau ngebut buat test
        
        # Jalankan
        text, lang = engine.transcribe(TEST_FILE)
        
        print("\n=== HASIL TRANSKRIPSI ===")
        print(f"Bahasa Terdeteksi: {lang}")
        print("-" * 30)
        print(text)
        print("-" * 30)
    else:
        print(f"File '{TEST_FILE}' tidak ditemukan. Taruh file video sample dulu di sini buat ngetest.")