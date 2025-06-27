import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import os

# mendisable eager execution untuk kompatibilitas dengan TensorFlow 1.x
tf.compat.v1.disable_eager_execution()

@st.cache_resource
def load_model():
    try:
        sess = tf.compat.v1.Session()
        
        # Path ke model
        model_dir = "anfis_model"
        
    
        if not os.path.exists(model_dir):
            st.error(f"Model folder not found at: {os.path.abspath(model_dir)}")
            return None, None
            
        # load model yang sudah disimpan
        meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, ["serve"], model_dir)
        
        # mendapatkan default graph
        graph = tf.compat.v1.get_default_graph()
        
        # mendapatkan tensor input dan output
        inputs = graph.get_tensor_by_name('inputs:0')
        pred_class = graph.get_tensor_by_name('ArgMax:0')
        
        return sess, inputs, pred_class
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None, None

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        return load('scaler.joblib')
    except:
      
        scaler = MinMaxScaler()
        # Inisialisasi scaler dengan dummy data
        dummy_data = pd.DataFrame(np.zeros((1, 27)), columns=[
            'IP_Semester_1', 'IP_Semester_2', 'IP_Semester_3', 'IP_Semester_4', 
            'IP_Semester_5', 'IP_Semester_6', 'SKS_Lulus_Semester_1', 
            'SKS_Lulus_Semester_2', 'SKS_Lulus_Semester_3', 'SKS_Lulus_Semester_4', 
            'SKS_Lulus_Semester_5', 'SKS_Lulus_Semester_6', 'MK_Ulang_Semester_1', 
            'MK_Ulang_Semester_2', 'MK_Ulang_Semester_3', 'MK_Ulang_Semester_4', 
            'MK_Ulang_Semester_5', 'MK_Ulang_Semester_6', 'Total_SKS_Selesai_Semester_1', 
            'Total_SKS_Selesai_Semester_2', 'Total_SKS_Selesai_Semester_3', 
            'Total_SKS_Selesai_Semester_4', 'Total_SKS_Selesai_Semester_5', 
            'Total_SKS_Selesai_Semester_6', 'Total_SKS_Tidak_Lulus', 
            'Kehadiran_Persen', 'Ketepatan_Tugas_Persen'
        ])
        scaler.fit(dummy_data)
        return scaler

# UI Setup
st.title("Prediksi Kategori Lama Studi Mahasiswa (ANFIS)")
st.header("Masukkan Data Mahasiswa")

# inisialisasi input lists
ip = []
sks_lulus = []
mk_ulang = []
valid_input = True

# Semester inputs
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
        st.error(f"Total SKS di Semester {i} melebihi 24!")
        valid_input = False

# input tambahan
kehadiran = st.number_input("Kehadiran (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
tugas = st.number_input("Ketepatan Tugas (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)

# hitung total SKS selesai
total_sks_selesai = np.cumsum(sks_lulus)
total_sks_tidak_lulus = sum(mk_ulang)

# buat dictionary untuk input data
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

# tampilkan data input
st.subheader("Data Awal (Numerik)")
st.dataframe(df_input)

# Load scaler dan transform input
scaler = load_scaler()
X_scaled = scaler.transform(df_input)

# Prediction section
if st.button("Prediksi"):
    if not valid_input:
        st.warning("Perbaiki input terlebih dahulu!")
    else:
        try:
            # Debug scaled input
            st.subheader("Data setelah Scaling")
            st.write(X_scaled)
            
            # Load model
            sess, inputs, pred_class = load_model()
            if sess is None:
                st.error("Tidak dapat memuat model!")
                st.stop()
                
            # Get prediction
            prediction = sess.run(pred_class, feed_dict={inputs: X_scaled})
            
            label_map = {
                0: "Lulus Cepat (3.5 tahun)",
                1: "Tepat Waktu (4 tahun)", 
                2: "Terlambat (4.5-7 tahun)",
                3: "Drop Out"
            }
            
            st.success(f"**Hasil Prediksi:** {label_map.get(prediction[0], 'Unknown')}")
            
        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
            st.error("Pastikan:")
            st.error("1. Model sudah terinstall dengan benar")
            st.error("2. Format input sesuai dengan yang diharapkan model")

# Model information
st.markdown("---")
st.subheader("Tentang Model")
st.write("""
Model ANFIS ini memprediksi kategori lama studi berdasarkan:
- IPK per semester
- SKS yang lulus
- Mata kuliah yang diulang
- Kehadiran
- Ketepatan pengumpulan tugas

**Kategori Output:**
1. Lulus Cepat (3.5 tahun)
2. Tepat Waktu (4 tahun)
3. Terlambat (4.5-7 tahun) 
4. Drop Out
""")