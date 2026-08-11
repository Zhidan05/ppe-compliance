# Panduan Kontribusi

Terima kasih telah berkontribusi pada repository ini.

Dokumen ini menjelaskan cara menyiapkan environment, menjalankan proyek, melakukan perubahan, serta mengirimkan kontribusi ke repository.

---

## 1. Persyaratan

Sebelum menjalankan proyek, pastikan perangkat telah memiliki:

- Windows 10/11
- Python 3.12 atau versi yang sesuai dengan `requirements.txt`
- Git
- NVIDIA GPU dan CUDA jika ingin menggunakan GPU untuk inference atau training
- Webcam atau sumber video lain jika diperlukan oleh program

Periksa versi Python:

```bash
python --version
```

Periksa versi Git:

```bash
git --version
```

---

## 2. Clone Repository

Clone repository ke komputer:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
```

Masuk ke direktori repository:

```bash
cd REPOSITORY
```

Ganti `USERNAME` dan `REPOSITORY` sesuai repository yang digunakan.

---

## 3. Membuat Virtual Environment

Buat virtual environment:

```bash
python -m venv yolo-env
```

Aktifkan virtual environment pada Windows:

```bash
yolo-env\Scripts\activate
```

Jika berhasil, nama environment akan muncul di awal terminal:

```text
(yolo-env) C:\...\REPOSITORY>
```

---

## 4. Install Dependency

Pastikan virtual environment telah aktif.

Kemudian install seluruh dependency:

```bash
pip install -r requirements.txt
```

Jika menggunakan GPU NVIDIA, pastikan versi PyTorch dan CUDA yang digunakan sesuai dengan konfigurasi perangkat.

Periksa instalasi PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

Periksa apakah CUDA tersedia:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Jika hasilnya:

```text
True
```

maka PyTorch dapat menggunakan GPU CUDA.

---

## 5. Menyiapkan Model

Model yang diperlukan oleh proyek harus ditempatkan pada direktori yang sesuai.

Contoh:

```text
models/
└── best.pt
```

Sesuaikan nama dan lokasi model dengan konfigurasi pada source code.

Jika model tidak disediakan di repository, pastikan model diperoleh dari sumber yang telah ditentukan oleh pengelola proyek.

---

## 6. Menyiapkan Dataset

Jika proyek membutuhkan dataset untuk training atau pengujian, tempatkan dataset sesuai struktur yang digunakan oleh konfigurasi proyek.

Contoh:

```text
datasets/
├── train/
├── val/
└── test/
```

Pastikan konfigurasi dataset seperti `data.yaml` mengarah ke lokasi dataset yang benar.

Dataset berukuran besar tidak disimpan di repository kecuali dinyatakan secara khusus.

---

## 7. Menjalankan Program

Setelah environment dan dependency selesai disiapkan, jalankan program menggunakan entry point yang tersedia.

Contoh:

```bash
python main.py
```

Jika program menggunakan file tertentu:

```bash
python detect.py
```

Gunakan file utama yang tercantum pada dokumentasi atau konfigurasi repository.

---

## 8. Menjalankan dengan Webcam

Jika program mendukung webcam, pastikan webcam telah terhubung.

Kemudian jalankan program sesuai konfigurasi source.

Contoh:

```bash
python main.py
```

Nomor kamera dapat berbeda pada setiap perangkat.

Contoh:

```python
cv2.VideoCapture(0)
```

Jika kamera pertama tidak tersedia, dapat mencoba:

```python
cv2.VideoCapture(1)
```

---

## 9. Menjalankan dengan Video

Jika program mendukung input video, letakkan file video pada direktori yang sesuai.

Contoh:

```text
videos/
└── test.mp4
```

Kemudian jalankan program dengan konfigurasi video tersebut.

Contoh:

```bash
python main.py --source videos/test.mp4
```

Sesuaikan parameter dengan implementasi program.

---

## 10. Menjalankan dengan CCTV / RTSP

Jika proyek menggunakan kamera CCTV dengan RTSP, masukkan alamat RTSP sesuai konfigurasi kamera.

Contoh format:

```text
rtsp://username:password@IP_ADDRESS:554/stream
```

Jangan memasukkan username, password, atau credential kamera secara langsung ke repository.

Gunakan environment variable atau file konfigurasi lokal yang tidak di-commit.

Contoh:

```text
RTSP_URL=rtsp://username:password@192.168.1.100:554/stream
```

Jangan commit file yang berisi credential tersebut.

---

# Kontribusi

## 11. Membuat Branch

Jangan melakukan perubahan langsung pada branch `main`.

Buat branch baru sesuai perubahan yang dilakukan.

Untuk fitur baru:

```bash
git checkout -b feature/nama-fitur
```

Contoh:

```bash
git checkout -b feature/person-tracking
```

Untuk perbaikan bug:

```bash
git checkout -b fix/nama-masalah
```

Contoh:

```bash
git checkout -b fix/duplicate-person-id
```

Untuk dokumentasi:

```bash
git checkout -b docs/nama-dokumentasi
```

---

## 12. Melakukan Perubahan

Lakukan perubahan sesuai kebutuhan.

Pastikan:

- Kode dapat dijalankan.
- Tidak menambahkan credential.
- Tidak merusak fungsi yang sudah ada.
- Tidak menambahkan dependency yang tidak diperlukan.
- Perubahan telah diuji sebelum dikirimkan.

---

## 13. Menambahkan Dependency

Jika menggunakan package Python baru, install package tersebut pada virtual environment:

```bash
pip install nama-package
```

Setelah itu perbarui `requirements.txt`:

```bash
pip freeze > requirements.txt
```

Pastikan hanya dependency yang benar-benar diperlukan yang ditambahkan.

---

## 14. Testing

Sebelum melakukan commit, jalankan program dan pastikan perubahan bekerja dengan baik.

Untuk perubahan pada sistem Computer Vision, pengujian dapat dilakukan menggunakan:

- Webcam
- Video
- CCTV/RTSP
- Dataset pengujian

Jika ditemukan error, perbaiki terlebih dahulu sebelum melakukan Pull Request.

---

## 15. Commit

Tambahkan perubahan:

```bash
git add .
```

Buat commit dengan pesan yang jelas:

```bash
git commit -m "Add person tracking"
```

Gunakan commit message yang menggambarkan perubahan.

Contoh:

```text
Add person tracking
Fix duplicate person detection
Update YOLO configuration
Optimize video processing
Docs: update installation guide
Refactor detection module
```

Hindari commit message yang terlalu umum seperti:

```text
update
fix
test
coba
perbaikan
```

---

## 16. Push Branch

Push branch ke repository:

```bash
git push origin nama-branch
```

Contoh:

```bash
git push origin feature/person-tracking
```

Untuk pertama kali melakukan push pada branch:

```bash
git push -u origin feature/person-tracking
```

---

## 17. Pull Request

Setelah branch berhasil di-push, buat Pull Request menuju branch:

```text
main
```

Pull Request sebaiknya menjelaskan:

- Perubahan yang dilakukan.
- Alasan perubahan.
- Cara melakukan pengujian.
- Masalah yang diperbaiki.
- Hal yang perlu diperhatikan.

Contoh:

```text
## Perubahan

- Menambahkan person tracking.
- Mengurangi kemungkinan duplicate ID.
- Memperbarui konfigurasi inference.

## Pengujian

- Pengujian menggunakan webcam.
- Pengujian menggunakan video.
- Tidak ditemukan error pada inference.

## Catatan

Perubahan hanya memengaruhi modul tracking.
```

---

## 18. Update Branch

Sebelum melakukan Pull Request atau ketika branch sudah tertinggal dari `main`, sinkronkan perubahan terbaru.

Pindah ke `main`:

```bash
git checkout main
```

Ambil perubahan terbaru:

```bash
git pull origin main
```

Kembali ke branch pekerjaan:

```bash
git checkout nama-branch
```

Kemudian update branch sesuai kebutuhan.

---

## 19. `.gitignore`

File berikut tidak boleh dimasukkan ke repository:

```text
yolo-env/
__pycache__/
*.pyc
runs/
datasets/
*.cache
.env
.vscode/
.idea/
```

Virtual environment dibuat secara lokal menggunakan:

```bash
python -m venv yolo-env
```

Kemudian dependency dipasang menggunakan:

```bash
pip install -r requirements.txt
```

---

## 20. Keamanan

Jangan commit informasi sensitif seperti:

- Password
- API key
- Access token
- Private key
- Credential database
- Credential Telegram
- Username dan password CCTV
- URL RTSP yang mengandung password

Contoh yang **tidak boleh**:

```python
TELEGRAM_BOT_TOKEN = "123456:ABCDEF..."
```

Gunakan environment variable:

```python
import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
```

Jika credential tidak sengaja ter-commit, segera lakukan revoke atau reset credential tersebut.

---

## 21. Checklist Sebelum Pull Request

Sebelum membuat Pull Request, pastikan:

- [ ] Program dapat dijalankan.
- [ ] Perubahan telah diuji.
- [ ] Tidak ada credential yang ikut ter-commit.
- [ ] `yolo-env/` tidak ikut ter-commit.
- [ ] Dataset tidak ikut ter-commit.
- [ ] File berukuran besar tidak ikut ter-commit tanpa alasan.
- [ ] `requirements.txt` diperbarui jika dependency berubah.
- [ ] Commit message sudah jelas.
- [ ] Branch berasal dari `main`.
- [ ] Pull Request memiliki deskripsi yang jelas.

---

## 22. Pertanyaan dan Masalah

Jika mengalami masalah saat menjalankan atau mengembangkan proyek, silakan buat Issue pada repository.

Sertakan informasi berikut:

- Sistem operasi.
- Versi Python.
- Versi dependency.
- Pesan error.
- Langkah untuk menghasilkan error.
- Screenshot atau log jika diperlukan.

Semakin lengkap informasi yang diberikan, semakin mudah masalah tersebut dianalisis dan diperbaiki.

---

Terima kasih telah berkontribusi! 🚀