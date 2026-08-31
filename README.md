# 🦺 PPE Compliance Monitoring

Sistem monitoring kepatuhan **Alat Pelindung Diri (APD)** secara real-time menggunakan YOLOv8/YOLO + object tracking.

Mendeteksi pekerja yang **tidak memakai topi keselamatan** dan/atau **vest**, mencatat setiap observasi ke CSV, serta mengirim notifikasi pelanggaran ke Telegram beserta bukti foto.

---

## ✨ Fitur Utama

- **Deteksi real-time** class: `person`, `topi`, `vest`
- **Multi-object tracking** (BoT-SORT) agar setiap orang punya ID konsisten
- **Asosiasi APD ke orang** secara cerdas (berdasarkan posisi bounding box)
- **Majority vote** untuk mengurangi false positive dari frame yang hilang sebentar
- **Logging observasi** ke `data/observations.csv`
- **Notifikasi Telegram** otomatis saat pelanggaran terdeteksi (foto asli + annotated)
- **Dukungan hardware fleksibel**:
  - NVIDIA GPU (TensorRT / ONNX / PyTorch)
  - Intel/AMD CPU (OpenVINO)
  - ARM / Raspberry Pi (TFLite / ONNX)
- Versi terpisah: `camera_cpu.py`, `camera_gpu.py`, dan versi pintar `src/camera_flexible.py`

---

## 📁 Struktur Proyek
ppe-compliance/
├── data/                     # Log observasi (observations.csv)
├── models/                   # Model dalam berbagai format
│   ├── pytorch/
│   ├── openvino/
│   └── base/
├── requirements/             # Dependency terpisah
│   ├── requirements_cpu.txt
│   ├── requirements_gpu.txt
│   └── requirements_train.txt
├── src/
│   └── camera_flexible.py    # Versi recommended (auto-detect hardware)
├── tests/
├── camera_cpu.py             # Versi khusus CPU
├── camera_gpu.py             # Versi khusus GPU
├── CONTRIBUTE.md
└── README.md
text---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/Zhidan05/ppe-compliance.git
cd ppe-compliance
2. Buat Virtual Environment
Bashpython -m venv yolo-env

# Windows
yolo-env\Scripts\activate

# Linux / macOS
source yolo-env/bin/activate
3. Install Dependency
CPU saja:
Bashpip install -r requirements/requirements_cpu.txt
GPU (NVIDIA + CUDA):
Bashpip install -r requirements/requirements_gpu.txt
4. Siapkan Model
Letakkan model yang sudah dilatih di folder yang sesuai, contoh:
textmodels/
├── pytorch/best.pt
├── openvino/best.xml (+ .bin)
└── ...
Atau sesuaikan path di kode.
5. Konfigurasi Environment
Buat file .env di root proyek:
envTELEGRAM_BOT_TOKEN=123456:ABCDEF...
TELEGRAM_CHAT_ID=-100xxxxxxxxxx
CAMERA_SOURCE=0                    # atau path video / RTSP URL
Penting: Jangan pernah commit file .env yang berisi credential.
6. Jalankan Program
Versi recommended (auto pilih model terbaik sesuai hardware):
Bashpython src/camera_flexible.py
Atau versi khusus:
Bashpython camera_cpu.py
# atau
python camera_gpu.py
Tekan Q untuk keluar.

📊 Output
1. File Log
Setiap orang yang terdeteksi cukup lama akan dicatat di:
textdata/observations.csv
Kolom:

observation_id
timestamp
track_id
frames_observed
topi / vest (1 = ada, 0 = tidak)
violation (1 = pelanggaran)

2. Notifikasi Telegram
Saat pelanggaran terdeteksi (minimal durasi tertentu), sistem akan mengirim:

Foto asli
Foto dengan bounding box
Informasi Person ID + jenis pelanggaran + timestamp


⚙️ Konfigurasi Penting



































ParameterDefaultKeteranganPPE_CLASSES["topi", "vest"]Class APD yang wajibMIN_FRAMES10Minimal frame sebelum dicatat sebagai observasiVIOLATION_MIN_DURATION3.0 detikMinimal durasi pelanggaran sebelum kirim TelegramTELEGRAM_INTERVAL20 detikJeda antar notifikasiCONFIDENCE0.40Threshold deteksi

🛠️ Requirements

Python 3.10 – 3.12 (disarankan 3.12)
Webcam / CCTV (RTSP) / file video
(Opsional) NVIDIA GPU + CUDA untuk performa terbaik


🤝 Kontribusi
Silakan baca panduan lengkap di CONTRIBUTE.md [blocked].
Ringkasnya:

Fork repository
Buat branch baru (feature/... atau fix/...)
Lakukan perubahan + testing
Buat Pull Request ke branch main


⚠️ Catatan Keamanan
Jangan pernah commit:

Token Telegram
Username/password RTSP
File .env
Dataset besar
Virtual environment (yolo-env/)


📜 License
Proyek ini masih private / belum ditentukan lisensinya. Silakan hubungi pemilik repository untuk informasi lebih lanjut.
