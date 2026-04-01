import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Prediksi Air PDAM",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Prediksi Kebutuhan Air PDAM")

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_all():
    model = load_model("model_air.h5")
    scaler = joblib.load("scaler.save")
    data = pd.read_csv("data_air.csv")
    return model, scaler, data

model, scaler, data = load_all()

features = data.drop(columns=["Tanggal"])
data_scaled = scaler.transform(features)

time_steps = 7

# ========================
# FUNCTION PREDIKSI
# ========================
def predict_future(n_days):
    current_seq = data_scaled[-time_steps:].copy()
    predictions = []

    for _ in range(n_days):
        pred = model.predict(
            current_seq.reshape(1, time_steps, current_seq.shape[1]),
            verbose=0
        )

        next_val = pred[0][0]
        predictions.append(next_val)

        new_row = current_seq[-1].copy()
        new_row[0] = next_val  # sesuaikan jika target bukan index 0

        current_seq = np.vstack([current_seq[1:], new_row])

    return predictions

# ========================
# UI
# ========================
n_days = st.slider("Pilih jumlah hari prediksi", 1, 30, 7)

if st.button("🔮 Prediksi"):
    preds = predict_future(n_days)

    df_pred = pd.DataFrame({
        "Hari": range(1, n_days + 1),
        "Prediksi": preds
    })

    st.subheader("📋 Hasil Prediksi")
    st.dataframe(df_pred)

    st.subheader("📈 Grafik Prediksi")
    st.line_chart(df_pred.set_index("Hari"))

    st.metric("Prediksi Hari Terakhir", round(preds[-1], 2))