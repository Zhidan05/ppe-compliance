# 🦺 PPE Compliance Monitoring

Sistem monitoring kepatuhan **Alat Pelindung Diri (APD)** secara real-time menggunakan YOLO + object tracking.

Mendeteksi pekerja yang tidak memakai **topi keselamatan** dan/atau **vest**, mencatat setiap observasi ke CSV, serta mengirim notifikasi pelanggaran ke Telegram beserta bukti foto.

---

## ✨ Fitur Utama

- Deteksi real-time class: `person`, `topi`, `vest`
- Multi-object tracking (BoT-SORT) agar setiap orang punya ID konsisten
- Asosiasi APD ke orang secara cerdas berdasarkan posisi bounding box
- Majority vote untuk mengurangi false positive
- Logging observasi ke `data/observations.csv`
- Notifikasi Telegram otomatis saat pelanggaran terdeteksi (foto asli + annotated)
- Dukungan hardware fleksibel:
  - NVIDIA GPU (TensorRT / ONNX / PyTorch)
  - Intel/AMD CPU (OpenVINO)
  - ARM / Raspberry Pi (TFLite / ONNX)
- Tersedia versi `camera_cpu.py`, `camera_gpu.py`, dan versi pintar `src/camera_flexible.py`

---

## 📁 Struktur Proyek

```text
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
