from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle

# ── Load semua artefak model ──────────────────────────────────
model = tf.keras.models.load_model("model/classifier.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

MAX_LEN = 10

# ── Schema input dari Frontend/Backend ───────────────────────
class TransaksiInput(BaseModel):
    nama_transaksi: str
    nominal: float
    tanggal: str  # format: "YYYY-MM-DD"

# ── Schema output ─────────────────────────────────────────────
class PrediksiOutput(BaseModel):
    kategori: str
    confidence: float
    budget_alert: bool

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="Finsight ML Service", version="1.0.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "Finsight ML Service is running"}

@app.post("/predict", response_model=PrediksiOutput)
def predict(data: TransaksiInput):
    try:
        # Parse tanggal
        tanggal = datetime.strptime(data.tanggal, "%Y-%m-%d")
        bulan               = tanggal.month
        hari_dalam_seminggu = tanggal.weekday()
        adalah_weekend      = 1 if hari_dalam_seminggu >= 5 else 0

        # Encode teks
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq        = tokenizer.texts_to_sequences([data.nama_transaksi])
        seq_padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        # Normalisasi numerik
        num_raw    = np.array([[data.nominal, bulan, hari_dalam_seminggu, adalah_weekend]])
        num_scaled = scaler.transform(num_raw)

        # Prediksi
        pred       = model([
            tf.constant(seq_padded),
            tf.constant(num_scaled, dtype=tf.float32)
        ], training=False)

        pred_class = int(tf.argmax(pred, axis=1).numpy()[0])
        confidence = float(tf.reduce_max(pred).numpy())
        kategori   = le.inverse_transform([pred_class])[0]

        # Budget alert sederhana — threshold per kategori (Rupiah)
        BUDGET_THRESHOLD = {
            "Makanan & Minuman": 150_000,
            "Transportasi":      100_000,
            "Hiburan":           200_000,
            "Kesehatan":         300_000,
            "Belanja":           500_000,
            "Pendidikan":        300_000,
            "Tagihan":           500_000,
        }
        budget_alert = data.nominal > BUDGET_THRESHOLD.get(kategori, 999_999)

        return PrediksiOutput(
            kategori=kategori,
            confidence=round(confidence, 4),
            budget_alert=budget_alert
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))