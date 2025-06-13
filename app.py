import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from fuzzy_utils import apply_fuzzy_and_defuzzy
from data_preparation import prepare_fuzzy_input_for_prediction, prepare_defuzzy_input

# Load model
model_fuzzy = tf.keras.models.load_model("fuzzy_cnn_model.h5")
model_defuzzy = tf.keras.models.load_model("cnn_model_fuzzy_defuzzy.h5")

# (Opsional) Simpan akurasi model (misalnya sudah kamu catat saat training)
model_fuzzy_accuracy = 0.92  # 92% contoh
model_defuzzy_accuracy = 0.89  # 89% contoh

# UI
st.title("Prediksi Kategori Lama Studi Mahasiswa (CNN + Fuzzy)")
st.header("Masukkan Data Mahasiswa")

ip = []
sks_lulus = []
mk_ulang = []
valid_input = True

# Input per semester
for i in range(1, 7):
    st.markdown(f"### Semester {i}")
    col1, col2, col3 = st.columns(3)

    with col1:
        ip_val = st.number_input(f"IP Semester {i}", min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f"ip_{i}")
        ip.append(ip_val)

    with col2:
        sks_val = st.number_input(f"SKS Lulus Semester {i}", min_value=0, max_value=24, value=20, key=f"sks_{i}")
        sks_lulus.append(sks_val)

    with col3:
        mk_val = st.number_input(f"MK Tidak Lulus Semester {i}", min_value=0, max_value=24, value=0, key=f"mk_{i}")
        mk_ulang.append(mk_val)

    if sks_val + mk_val > 24:
        st.error(f"Jumlah SKS Lulus + MK Tidak Lulus di Semester {i} melebihi 24! Periksa kembali.")
        valid_input = False

# Input tambahan
kehadiran = st.number_input("Kehadiran (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
tugas = st.number_input("Ketepatan Tugas (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)

# Hitung fitur tambahan
total_sks_selesai = np.cumsum(sks_lulus)
total_sks_tidak_lulus = sum(mk_ulang)

# Buat dataframe input
data_dict = {
    **{f"IP_Semester_{i+1}": ip[i] for i in range(6)},
    **{f"SKS_Lulus_Semester_{i+1}": sks_lulus[i] for i in range(6)},
    **{f"MK_Ulang_Semester_{i+1}": mk_ulang[i] for i in range(6)},
    **{f"Total_SKS_Selesai_Semester_{i+1}": total_sks_selesai[i] for i in range(6)},
    "Total_SKS_Tidak_Lulus": total_sks_tidak_lulus,
    "Kehadiran_Persen": kehadiran,
    "Ketepatan_Tugas_Persen": tugas
}
df_input = pd.DataFrame([data_dict])

# Tampilkan df awal
st.subheader("Data Awal (Numerik)")
st.dataframe(df_input)

# Tombol Fuzzifikasi
if st.button("Fuzzifikasi"):
    if valid_input:
        df_fuzzy, df_defuzzy = apply_fuzzy_and_defuzzy(df_input)
        st.session_state.df_fuzzy = df_fuzzy
        st.session_state.df_defuzzy = df_defuzzy

        st.subheader("Hasil Fuzzifikasi (μ)")
        st.dataframe(df_fuzzy)

        st.subheader("Hasil Defuzzifikasi")
        defuzzy_cols = [col for col in df_defuzzy.columns if col.startswith("Defuzz_")]
        st.dataframe(df_defuzzy[defuzzy_cols])
    else:
        st.warning("Perbaiki input dulu sebelum melakukan fuzzifikasi.")

# Tombol Prediksi
if st.button("Prediksi"):
    if not valid_input:
        st.warning("Perbaiki input dulu sebelum melakukan prediksi.")
    elif 'df_fuzzy' not in st.session_state or 'df_defuzzy' not in st.session_state:
        st.warning("Lakukan fuzzifikasi terlebih dahulu sebelum prediksi.")
    else:
        try:
            df_fuzzy = st.session_state.df_fuzzy
            df_defuzzy = st.session_state.df_defuzzy

            # Fuzzy CNN prediction
            X_fuzzy_input = prepare_fuzzy_input_for_prediction(df_fuzzy)
            pred_probs_fuzzy = model_fuzzy.predict(X_fuzzy_input)
            pred_class_fuzzy = np.argmax(pred_probs_fuzzy, axis=1)[0]

            # Defuzzy CNN prediction
            X_defuzzy_input = prepare_defuzzy_input(df_defuzzy)
            pred_probs_defuzzy = model_defuzzy.predict(X_defuzzy_input)
            pred_class_defuzzy = np.argmax(pred_probs_defuzzy, axis=1)[0]

            label_map = {0: "Lulus Cepat", 1: "Tepat Waktu", 2: "Terlambat", 3: "Drop Out"}

            st.success(f"Prediksi (Fuzzy + CNN): {label_map.get(pred_class_fuzzy, 'Tidak diketahui')}")
            st.info(f"Akurasi Model Fuzzy CNN: {model_fuzzy_accuracy * 100:.2f}%")

            st.success(f"Prediksi (Defuzzy + CNN): {label_map.get(pred_class_defuzzy, 'Tidak diketahui')}")
            st.info(f"Akurasi Model Defuzzy CNN: {model_defuzzy_accuracy * 100:.2f}%")

        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
