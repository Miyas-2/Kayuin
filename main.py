import os
import uuid
import time
import shutil
import tempfile
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from gtts import gTTS

app = FastAPI(
    title="Aplikasi Klasifikasi Suara Jenis Kayu - PTU",
    description="Aplikasi klasifikasi suara jenis kayu menggunakan model CNN, pemetaan kalimat, dan Text-to-Speech (gTTS)."
)

# Membuat folder static jika belum ada untuk menampung respons audio
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount folder static agar bisa di-serve oleh FastAPI
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Pengaturan templates Jinja2
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Path menuju model Keras CNN .h5
MODEL_PATH = os.path.join(os.path.dirname(__file__), "voice_recognition_cnn.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model voice_recognition_cnn.h5 tidak ditemukan di {MODEL_PATH}")

print("Menyiapkan pemuatan model CNN TensorFlow/Keras...")
# Memuat model CNN secara global pada saat inisialisasi server
model = tf.keras.models.load_model(MODEL_PATH)
print("Model CNN berhasil dimuat dan siap digunakan!")

# Parameter Audio & MFCC sesuai konfigurasi training di train_tubes.ipynb
SAMPLE_RATE = 22050      # Standar sample rate librosa
MAX_DURATION = 3.0       # Durasi maksimal rekaman audio (3 detik)
N_MFCC = 40              # Jumlah koefisien MFCC (tinggi matriks)
MAX_PAD_LEN = 130        # Panjang dimensi lebar matriks setelah padding/truncating

# Daftar label kelas yang terdeteksi secara alfabetis dari LabelEncoder di train_tubes.ipynb
# Catatan: 'data_set_ulin_1_25' menggunakan huruf kecil 'u' sehingga berada di urutan akhir ASCII
CLASS_LIST = [
    "data_set_Cendana_1_25",
    "data_set_Eboni_1_25",
    "data_set_Jati_1_25",
    "data_set_Kempas_1_25",
    "data_set_Mahoni_1_25",
    "data_set_Matoa_1_25",
    "data_set_Meranti_1_25",
    "data_set_Nyatoh_1_25",
    "data_set_Pinus_1_25",
    "data_set_ulin_1_25"
]

# Kamus pemetaan kelas kayu ke kalimat Bahasa Indonesia deskriptif
CLASS_TO_TEXT = {
    "data_set_Cendana_1_25": "Kayu cendana berhasil terdeteksi. Kayu ini memiliki aroma wangi alami yang sangat khas dan bernilai jual tinggi.",
    "data_set_Eboni_1_25": "Kayu eboni terdeteksi. Kayu eboni dikenal sangat keras, berat, berwarna hitam legam eksotis, dan berharga mahal.",
    "data_set_Jati_1_25": "Kayu jati berhasil diidentifikasi. Merupakan kayu premium yang sangat kuat, awet, serta tahan terhadap serangan rayap.",
    "data_set_Kempas_1_25": "Kayu kempas terdeteksi. Kayu kempas bertekstur kasar, sangat keras, dan umumnya dipakai untuk struktur jembatan atau bantalan rel.",
    "data_set_Mahoni_1_25": "Kayu mahoni diidentifikasi. Serat kayunya indah kemerahan dengan stabilitas bentuk yang sangat baik untuk perabotan rumah tangga.",
    "data_set_Matoa_1_25": "Kayu matoa terdeteksi. Merupakan jenis kayu khas Papua dengan corak garis indah yang menawan serta kekuatan yang baik.",
    "data_set_Meranti_1_25": "Kayu meranti berhasil dideteksi. Sangat populer untuk bahan bangunan konstruksi perumahan, pembuatan kusen pintu, dan jendela.",
    "data_set_Nyatoh_1_25": "Kayu nyatoh berhasil diidentifikasi. Kayu ini memiliki serat halus berwarna kemerahan yang mudah diolah menjadi panel interior.",
    "data_set_Pinus_1_25": "Kayu pinus diidentifikasi. Kayu pinus ringan dengan warna cerah serta mata kayu yang khas, sangat digemari untuk furniture minimalis.",
    "data_set_ulin_1_25": "Kayu ulin berhasil terdeteksi. Dikenal sebagai kayu besi karena sangat keras, kokoh, dan tahan air laut maupun kelembapan tinggi."
}

def clean_old_static_files():
    """Membersihkan file audio respons yang berumur lebih dari 3 menit agar folder static tetap bersih."""
    now = time.time()
    try:
        for filename in os.listdir(STATIC_DIR):
            if filename.startswith("response_") and filename.endswith(".mp3"):
                filepath = os.path.join(STATIC_DIR, filename)
                # Hapus file jika usianya lebih dari 180 detik
                if os.path.getmtime(filepath) < now - 180:
                    os.remove(filepath)
    except Exception as e:
        print(f"Gagal melakukan pembersihan file static lama: {e}")

def convert_to_wav(input_path, output_path):
    """Mengonversi file audio input apa pun (m4a, webm, mp3) ke format WAV menggunakan Pydub."""
    try:
        # Pydub secara otomatis menggunakan FFmpeg untuk mendeteksi container audio
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Gagal mengonversi audio ke WAV via Pydub: {e}")
        return False

def extract_features_from_audio(file_path):
    """Mengekstrak fitur MFCC dari file WAV sesuai spesifikasi model CNN."""
    try:
        # Load audio dengan librosa sesuai parameter sampel
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        # Ekstraksi koefisien MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Padding atau Truncating agar dimensi lebarnya presisi 130 frame (MAX_PAD_LEN)
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        
        if pad_width > 0:
            # Padding dengan nilai konstan 0 di akhir kolom jika durasi kurang dari 3 detik
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            # Potong jika melebihi 130 frame
            mfcc = mfcc[:, :MAX_PAD_LEN]
            
        # Reshape agar sesuai dengan format input model CNN: (batch_size=1, height=40, width=130, channels=1)
        mfcc_input = mfcc.reshape(1, N_MFCC, MAX_PAD_LEN, 1)
        return mfcc_input
    except Exception as e:
        print(f"Gagal melakukan ekstraksi fitur MFCC menggunakan Librosa: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Endpoint utama untuk memuat antarmuka web."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "class_list": [c.replace("data_set_", "").replace("_1_25", "").capitalize() for c in CLASS_LIST]
    })

@app.post("/predict")
async def predict_voice(file: UploadFile = File(...)):
    """Endpoint API untuk memproses rekaman atau file audio untuk diinferensikan dan dikonversi ke suara respons."""
    # Bersihkan file respons usang dari folder static
    clean_old_static_files()
    
    # 1. Simpan file audio yang diunggah ke temporary file sementara
    temp_in_fd, temp_in_path = tempfile.mkstemp(suffix=f"_{file.filename}")
    temp_out_fd, temp_out_path = tempfile.mkstemp(suffix=".wav")
    
    os.close(temp_in_fd)
    os.close(temp_out_fd)
    
    try:
        # Salin file upload ke file temporer input
        with open(temp_in_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Konversi file input tersebut ke WAV murni
        success = convert_to_wav(temp_in_path, temp_out_path)
        if not success:
            raise HTTPException(status_code=400, detail="Format audio tidak didukung atau modul konversi audio gagal.")
            
        # 3. Ekstraksi fitur MFCC dari file WAV hasil konversi
        mfcc_input = extract_features_from_audio(temp_out_path)
        if mfcc_input is None:
            raise HTTPException(status_code=400, detail="Gagal memproses gelombang suara untuk diekstrak fiturnya.")
            
        # 4. Inferensi Prediksi menggunakan Model CNN (.h5)
        predictions = model.predict(mfcc_input)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx]) * 100
        
        # Dapatkan label kelas terpilih
        predicted_class = CLASS_LIST[class_idx]
        wood_display_name = predicted_class.replace("data_set_", "").replace("_1_25", "").capitalize()
        
        # Dapatkan 3 prediksi kelas teratas untuk ditampilkan
        top_indices = np.argsort(predictions[0])[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            cls_name = CLASS_LIST[idx]
            display_name = cls_name.replace("data_set_", "").replace("_1_25", "").capitalize()
            prob = float(predictions[0][idx]) * 100
            top_predictions.append({
                "wood_name": display_name,
                "confidence": round(prob, 2)
            })
        
        # 5. Pemetaan kelas terpilih ke kalimat deskriptif
        sentence = CLASS_TO_TEXT.get(predicted_class, "Jenis kayu yang disebutkan tidak terdaftar dalam database kami.")
        
        # 6. Menghasilkan suara respons Bahasa Indonesia menggunakan gTTS
        response_filename = f"response_{uuid.uuid4().hex}.mp3"
        response_filepath = os.path.join(STATIC_DIR, response_filename)
        
        tts = gTTS(text=sentence, lang="id")
        tts.save(response_filepath)
        
        # Mengembalikan detail informasi prediksi berupa respons JSON
        return JSONResponse(content={
            "success": True,
            "class_name": predicted_class,
            "wood_name": wood_display_name,
            "sentence": sentence,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
            "audio_url": f"/static/{response_filename}"
        })
        
    except Exception as e:
        print(f"Kesalahan pemrosesan inferensi di server: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "detail": f"Terjadi kegagalan pemrosesan di server: {str(e)}"
        })
    finally:
        # Bersihkan file temporer setelah proses selesai agar tidak menumpuk di disk
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
