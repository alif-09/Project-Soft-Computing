import numpy as np

def prepare_defuzzy_input(df_defuzzy):
    """
    Proses dataframe defuzzy untuk jadi input CNN.
    
    Args:
        df_defuzzy (pd.DataFrame): Data defuzzy hasil proses fuzzifikasi + defuzzifikasi.
    
    Returns:
        np.ndarray: Array siap untuk CNN (samples, 7, 2)
    """
    # Fitur defuzzy
    defuzzy_features = [
        f'Defuzz_IP_Sem_{i}' for i in range(1,7)
    ] + [
        f'Defuzz_SKS_Sem_{i}' for i in range(1,7)
    ] + [
        'Defuzz_Kehadiran', 'Defuzz_Tugas'
    ]
    
    # Ambil X
    X = df_defuzzy[defuzzy_features].values
    
    # Bentuk jadi (samples, 6, 2) untuk IP + SKS
    X_semester = X[:, :12].reshape(-1, 6, 2)
    
    # Bentuk jadi (samples, 1, 2) untuk Kehadiran + Tugas
    X_extra = X[:, 12:].reshape(-1, 1, 2)
    
    # Gabung -> (samples, 7, 2)
    X_cnn = np.concatenate([X_semester, X_extra], axis=1)
    
    return X_cnn

def prepare_fuzzy_input_for_prediction(df_fuzzy):
    """
    Proses dataframe fuzzy jadi input CNN fuzzy untuk prediksi.
    
    Args:
        df_fuzzy (pd.DataFrame): Dataframe hasil fuzzifikasi (mengandung μ_*)

    Returns:
        X_fuzzy_reshaped (np.ndarray): Data siap input ke model CNN fuzzy (1, steps, 3)
    """
    # Ambil kolom μ_* saja
    fuzzy_features = [col for col in df_fuzzy.columns if (
        ('_rendah' in col) or ('_sedang' in col) or ('_tinggi' in col)
    )]
    
    # Ambil nilai fitur
    X_fuzzy = df_fuzzy[fuzzy_features].values  # (1, features)
    
    # Hitung langkah (steps)
    n_steps = len(fuzzy_features) // 3
    
    # Reshape jadi (1, steps, 3)
    X_fuzzy_reshaped = X_fuzzy.reshape((X_fuzzy.shape[0], n_steps, 3))
    
    return X_fuzzy_reshaped