# KayuIn — Klasifikasi Suara Jenis Kayu (PTU)

Aplikasi web berbasis **FastAPI** untuk mengklasifikasikan **jenis kayu** dari **suara** menggunakan model **CNN (TensorFlow/Keras)**. Setelah prediksi, aplikasi menghasilkan **kalimat deskriptif** dan **audio respons (Text-to-Speech)** menggunakan **gTTS**.

## Fitur

- Antarmuka web (rekam suara via mikrofon atau upload file audio)
- Ekstraksi fitur **MFCC (librosa)** dengan durasi maksimum 3 detik
- Inferensi model **CNN** dari file `voice_recognition_cnn.h5`
- Menampilkan **confidence** dan **Top-3** prediksi
- Menghasilkan audio respons (`.mp3`) dan di-serve lewat `/static`

## Struktur Proyek

- `main.py` — Backend FastAPI + inferensi + TTS
- `templates/index.html` — UI (rekam & upload) + pemanggilan `POST /predict`
- `static/` — Menyimpan audio respons TTS sementara
- `Dataset_Tubes/` — Dataset audio per kelas (folder per jenis kayu)
- `train_tubes.ipynb` — Notebook training model CNN
- `voice_recognition_cnn.h5` — Model hasil training

## Prasyarat

- Python 3.9+ (disarankan 3.10/3.11)
- **FFmpeg** (wajib untuk `pydub` agar bisa mengonversi `webm/m4a/mp3` ke WAV)
- Koneksi internet (untuk **gTTS** saat membuat audio respons)

### Instal FFmpeg (Windows)

Pilih salah satu:

- Via Chocolatey:
  - `choco install ffmpeg`
- Manual:
  - Unduh FFmpeg, lalu tambahkan folder `bin` FFmpeg ke `PATH`.

Cek instalasi:

- `ffmpeg -version`

## Instalasi Dependensi

Di root folder project:

- `python -m venv .venv`
- `./.venv/Scripts/activate`
- `pip install -r requirements.txt`

## Menjalankan Aplikasi

- `python main.py`

Server akan berjalan di:

- `http://localhost:8000`

UI dapat diakses dari halaman root (`GET /`).

## API

### `GET /`

Menampilkan halaman web (`templates/index.html`).

### `POST /predict`

Menerima file audio via form-data dengan field name: `file`.

- Input yang umum:
  - Rekaman browser: `audio/webm` (MediaRecorder)
  - Upload file: `wav/mp3/m4a/webm` (akan dikonversi ke WAV)

Contoh `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@contoh.wav"
```

Respons sukses (ringkas):

```json
{
  "success": true,
  "class_name": "data_set_Jati_1_25",
  "wood_name": "Jati",
  "sentence": "...",
  "confidence": 97.12,
  "top_predictions": [
    {"wood_name": "Jati", "confidence": 97.12},
    {"wood_name": "Meranti", "confidence": 1.83},
    {"wood_name": "Mahoni", "confidence": 0.61}
  ],
  "audio_url": "/static/response_<uuid>.mp3"
}
```

Catatan:
- File TTS akan disimpan di `static/` dan otomatis dibersihkan bila berumur > 3 menit.

## Dataset & Training

### Format dataset

Dataset disusun per folder kelas, misalnya:

- `Dataset_Tubes/data_set_Jati_1_25/*.wav`
- `Dataset_Tubes/data_set_Eboni_1_25/*.wav`

Nama folder kelas mengikuti daftar `CLASS_LIST` di `main.py`.

### Training

Notebook: `train_tubes.ipynb`

- Pastikan `DATASET_PATH` sesuai lokasi dataset Anda.
  - Jika menjalankan notebook dari root project, biasanya cukup: `DATASET_PATH = "Dataset_Tubes"`.
- Setelah training, model akan disimpan sebagai: `voice_recognition_cnn.h5`.

## Troubleshooting

- **Konversi audio gagal / format tidak didukung**
  - Pastikan FFmpeg terpasang dan bisa dipanggil dari terminal (`ffmpeg -version`).
- **gTTS error / tidak ada audio respons**
  - Pastikan koneksi internet aktif (gTTS melakukan request online).
- **Prediksi error pada audio tertentu**
  - Coba gunakan durasi <= 3 detik dan pastikan file tidak korup.

## Lisensi

Tambahkan lisensi sesuai kebutuhan (misalnya MIT) jika proyek ini akan dipublikasikan.
