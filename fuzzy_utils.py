import numpy as np
import skfuzzy as fuzz
import pandas as pd

def create_ip_membership_functions():
    x_ip = np.arange(0, 4.01, 0.01)
    rendah = fuzz.trimf(x_ip, [0, 0, 2.5])
    sedang = fuzz.trimf(x_ip, [2, 2.75, 3.5])
    tinggi = fuzz.trimf(x_ip, [3, 4, 4])
    return x_ip, rendah, sedang, tinggi

def create_sks_membership_functions():
    x_sks = np.arange(0, 25, 1)
    rendah = fuzz.trimf(x_sks, [0, 0, 15])
    sedang = fuzz.trimf(x_sks, [10, 18, 24])
    tinggi = fuzz.trimf(x_sks, [18, 24, 24])
    return x_sks, rendah, sedang, tinggi

def create_kehadiran_membership_functions():
    x_khd = np.arange(0, 101, 1)
    rendah = fuzz.trimf(x_khd, [0, 0, 70])
    sedang = fuzz.trimf(x_khd, [60, 80, 100])
    tinggi = fuzz.trimf(x_khd, [80, 100, 100])
    return x_khd, rendah, sedang, tinggi

def create_tugas_membership_functions():
    x_tgs = np.arange(0, 101, 1)
    rendah = fuzz.trimf(x_tgs, [0, 0, 60])
    sedang = fuzz.trimf(x_tgs, [50, 75, 100])
    tinggi = fuzz.trimf(x_tgs, [75, 100, 100])
    return x_tgs, rendah, sedang, tinggi

def create_sks_ulang_membership_functions():
    x_sksu = np.arange(0, 61, 1)
    rendah = fuzz.trimf(x_sksu, [0, 0, 5])
    sedang = fuzz.trimf(x_sksu, [3, 7, 11])
    tinggi = fuzz.trimf(x_sksu, [8, 15, 20])
    return x_sksu, rendah, sedang, tinggi


def fuzzify_value(x_val, x_mf, rendah, sedang, tinggi):
    μ_rendah = fuzz.interp_membership(x_mf, rendah, x_val)
    μ_sedang = fuzz.interp_membership(x_mf, sedang, x_val)
    μ_tinggi = fuzz.interp_membership(x_mf, tinggi, x_val)
    return μ_rendah, μ_sedang, μ_tinggi

def defuzzify(μ_rendah, μ_sedang, μ_tinggi, x_mf, rendah, sedang, tinggi):
    combined = np.fmax(μ_rendah * rendah, np.fmax(μ_sedang * sedang, μ_tinggi * tinggi))
    return fuzz.defuzz(x_mf, combined, 'centroid')

def process_row(row):
    fuzzy_results = {}
    defuzzy_results = {}
    
    # IP semua semester
    x_ip, ip_r, ip_s, ip_t = create_ip_membership_functions()
    for i in range(1, 7):
        μ_ip = fuzzify_value(row[f'IP_Semester_{i}'], x_ip, ip_r, ip_s, ip_t)
        fuzzy_results[f'IP_Sem_{i}_rendah'] = μ_ip[0]
        fuzzy_results[f'IP_Sem_{i}_sedang'] = μ_ip[1]
        fuzzy_results[f'IP_Sem_{i}_tinggi'] = μ_ip[2]
        defz_ip = defuzzify(*μ_ip, x_ip, ip_r, ip_s, ip_t)
        defuzzy_results[f'Defuzz_IP_Sem_{i}'] = defz_ip

    # SKS semua semester
    x_sks, sks_r, sks_s, sks_t = create_sks_membership_functions()
    for i in range(1, 7):
        μ_sks = fuzzify_value(row[f'SKS_Lulus_Semester_{i}'], x_sks, sks_r, sks_s, sks_t)
        fuzzy_results[f'SKS_Sem_{i}_rendah'] = μ_sks[0]
        fuzzy_results[f'SKS_Sem_{i}_sedang'] = μ_sks[1]
        fuzzy_results[f'SKS_Sem_{i}_tinggi'] = μ_sks[2]
        defz_sks = defuzzify(*μ_sks, x_sks, sks_r, sks_s, sks_t)
        defuzzy_results[f'Defuzz_SKS_Sem_{i}'] = defz_sks

    # total sks tidak lulus semua semester
    x_sksu, sksu_r, sksu_s, sksu_t = create_sks_ulang_membership_functions()
    μ_sksu = fuzzify_value(row['Total_SKS_Tidak_Lulus'], x_sksu, sksu_r, sksu_s, sksu_t)
    fuzzy_results['SKS_Ulang_rendah'] = μ_sksu[0]
    fuzzy_results['SKS_Ulang_sedang'] = μ_sksu[1]
    fuzzy_results['SKS_Ulang_tinggi'] = μ_sksu[2]
    defz_sksu = defuzzify(*μ_sksu, x_sksu, sksu_r, sksu_s, sksu_t)
    defuzzy_results['Defuzz_SKS_Ulang'] = defz_sksu

    # Kehadiran
    x_khd, khd_r, khd_s, khd_t = create_kehadiran_membership_functions()
    μ_khd = fuzzify_value(row['Kehadiran_Persen'], x_khd, khd_r, khd_s, khd_t)
    fuzzy_results['Khd_rendah'] = μ_khd[0]
    fuzzy_results['Khd_sedang'] = μ_khd[1]
    fuzzy_results['Khd_tinggi'] = μ_khd[2]
    defz_khd = defuzzify(*μ_khd, x_khd, khd_r, khd_s, khd_t)
    defuzzy_results['Defuzz_Kehadiran'] = defz_khd

    # Tugas
    x_tgs, tgs_r, tgs_s, tgs_t = create_tugas_membership_functions()
    μ_tgs = fuzzify_value(row['Ketepatan_Tugas_Persen'], x_tgs, tgs_r, tgs_s, tgs_t)
    fuzzy_results['Tugas_rendah'] = μ_tgs[0]
    fuzzy_results['Tugas_sedang'] = μ_tgs[1]
    fuzzy_results['Tugas_tinggi'] = μ_tgs[2]
    defz_tgs = defuzzify(*μ_tgs, x_tgs, tgs_r, tgs_s, tgs_t)
    defuzzy_results['Defuzz_Tugas'] = defz_tgs

    return pd.Series({**fuzzy_results, **defuzzy_results})

def apply_fuzzy_and_defuzzy(df):
    processed_df = df.apply(process_row, axis=1)
    
    fuzzy_cols = [col for col in processed_df.columns if 'Defuzz' not in col]
    defuzzy_cols = [col for col in processed_df.columns if 'Defuzz' in col]
    
    df_fuzzy = pd.concat([df, processed_df[fuzzy_cols]], axis=1)
    df_defuzzy = pd.concat([df, processed_df[defuzzy_cols]], axis=1)
    
    return df_fuzzy, df_defuzzy
