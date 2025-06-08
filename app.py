import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from fuzzy_risk_module import fuzzy_risk, risiko_to_label

# Load model keras
model = load_model("best_model.h5")

st.title("Prediksi Risiko Studi Mahasiswa")

# Form input semester data
ips = [st.number_input(f"IP Semester {i+1}", min_value=0.0, max_value=4.0, value=3.0, step=0.01) for i in range(6)]
sks_lulus = [st.number_input(f"SKS Lulus Semester {i+1}", min_value=0, max_value=24, value=20) for i in range(6)]
mk_ulang = [st.number_input(f"MK Ulang Semester {i+1}", min_value=0, max_value=10, value=0) for i in range(6)]

# Total SKS selesai cumulative
total_sks = [sum(sks_lulus[:i+1]) for i in range(6)]

fluktuasi_ips = st.selectbox("Fluktuasi IPS", ["Stabil", "Fluktuatif", "Menurun"])
sks_tidak_lulus = st.selectbox("SKS Tidak Lulus", ["Sedikit", "Sedang", "Banyak"])
kehadiran = st.selectbox("Kehadiran", ["Rendah", "Sedang", "Tinggi"])
tugas = st.selectbox("Tugas", ["Sering telat", "Tepat waktu"])

if st.button("Prediksi Risiko"):
    data = {
        **{f"IP_Semester_{i+1}": ips[i] for i in range(6)},
        **{f"SKS_Lulus_Semester_{i+1}": sks_lulus[i] for i in range(6)},
        **{f"MK_Ulang_Semester_{i+1}": mk_ulang[i] for i in range(6)},
        **{f"Total_SKS_Selesai_Semester_{i+1}": total_sks[i] for i in range(6)},
        "Fluktuasi_IPS": fluktuasi_ips,
        "SKS_Tidak_Lulus": sks_tidak_lulus,
        "Kehadiran": kehadiran,
        "Tugas": tugas
    }

    df = pd.DataFrame([data])

    # Hitung fuzzy risk dan label
    df['Risiko_Fuzzy'] = df.apply(fuzzy_risk, axis=1)
    df['Kategori_Fuzzy'] = df['Risiko_Fuzzy'].apply(risiko_to_label)

    # Encoding kategori untuk model keras
    mapping_fluktuasi = {"Stabil": 0, "Fluktuatif": 1, "Menurun": 2}
    mapping_sks = {"Sedikit": 0, "Sedang": 1, "Banyak": 2}
    mapping_kehadiran = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
    mapping_tugas = {"Tepat waktu": 0, "Sering telat": 1}

    df["Fluktuasi_IPS"] = df["Fluktuasi_IPS"].map(mapping_fluktuasi)
    df["SKS_Tidak_Lulus"] = df["SKS_Tidak_Lulus"].map(mapping_sks)
    df["Kehadiran"] = df["Kehadiran"].map(mapping_kehadiran)
    df["Tugas"] = df["Tugas"].map(mapping_tugas)

    # Ambil fitur untuk prediksi model
    features = df.drop(columns=["Risiko_Fuzzy", "Kategori_Fuzzy"])
    features = features.values.reshape(1, -1, 1)  # Contoh reshape jika model RNN/LSTM

    pred = model.predict(features)
    pred_label = np.argmax(pred, axis=1)[0]

    kategori_prediksi = {0: "Lulus Cepat", 1: "Tepat Waktu", 2: "Terlambat/DO"}.get(pred_label, "Tidak diketahui")

    st.write(f"**Prediksi Risiko dengan Model ML:** {kategori_prediksi}")
    st.write(f"**Prediksi Risiko dengan Fuzzy Logic:** {df['Kategori_Fuzzy'].iloc[0]}")
