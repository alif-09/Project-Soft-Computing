import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
fluktuasi = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'fluktuasi')
sks_tidak_lulus = ctrl.Antecedent(np.arange(0, 16, 1), 'sks_tidak_lulus')
kehadiran = ctrl.Antecedent(np.arange(0, 101, 1), 'kehadiran')
tugas = ctrl.Antecedent(np.arange(0, 101, 1), 'tugas')
risiko = ctrl.Consequent(np.arange(0, 101, 1), 'risiko')

# Membership functions
fluktuasi['stabil'] = fuzz.trimf(fluktuasi.universe, [0, 0, 0.3])
fluktuasi['fluktuatif'] = fuzz.trimf(fluktuasi.universe, [0.2, 0.5, 0.7])
fluktuasi['menurun'] = fuzz.trimf(fluktuasi.universe, [0.6, 1, 1])

sks_tidak_lulus['sedikit'] = fuzz.trimf(sks_tidak_lulus.universe, [0, 0, 4])
sks_tidak_lulus['sedang'] = fuzz.trimf(sks_tidak_lulus.universe, [2, 6, 10])
sks_tidak_lulus['banyak'] = fuzz.trimf(sks_tidak_lulus.universe, [8, 15, 15])

kehadiran['rendah'] = fuzz.trimf(kehadiran.universe, [0, 0, 60])
kehadiran['sedang'] = fuzz.trimf(kehadiran.universe, [50, 70, 85])
kehadiran['tinggi'] = fuzz.trimf(kehadiran.universe, [80, 100, 100])

tugas['sering_telat'] = fuzz.trimf(tugas.universe, [0, 0, 50])
tugas['tepat_waktu'] = fuzz.trimf(tugas.universe, [40, 100, 100])

risiko['rendah'] = fuzz.trimf(risiko.universe, [0, 0, 35])
risiko['sedang'] = fuzz.trimf(risiko.universe, [30, 50, 70])
risiko['tinggi'] = fuzz.trimf(risiko.universe, [60, 100, 100])

# Rules
rule1 = ctrl.Rule(
    fluktuasi['stabil'] & sks_tidak_lulus['sedikit'] & kehadiran['tinggi'] & tugas['tepat_waktu'],
    risiko['rendah']
)
rule2 = ctrl.Rule(
    fluktuasi['fluktuatif'] & sks_tidak_lulus['sedang'] & (kehadiran['sedang'] | kehadiran['tinggi']) & tugas['tepat_waktu'],
    risiko['sedang']
)
rule3 = ctrl.Rule(
    (fluktuasi['menurun'] | fluktuasi['fluktuatif']) &
    (sks_tidak_lulus['sedang'] | sks_tidak_lulus['banyak']) &
    (kehadiran['rendah'] | kehadiran['sedang']) &
    tugas['sering_telat'],
    risiko['tinggi']
)
rule4 = ctrl.Rule(
    fluktuasi['menurun'] & sks_tidak_lulus['banyak'] & kehadiran['rendah'] & tugas['sering_telat'],
    risiko['tinggi']
)
rule5 = ctrl.Rule(fluktuasi['stabil'], risiko['rendah'])
rule6 = ctrl.Rule(fluktuasi['fluktuatif'], risiko['sedang'])
rule7 = ctrl.Rule(fluktuasi['menurun'], risiko['tinggi'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

# Mapping helper functions (string kategori -> nilai fuzzy)
def map_fluktuasi(value):
    mapping = {'Stabil': 0.1, 'Fluktuatif': 0.5, 'Menurun': 0.9}
    return mapping.get(value, 0.5)

def map_sks_tidak_lulus(value):
    mapping = {'Sedikit': 2, 'Sedang': 6, 'Banyak': 12}
    return mapping.get(value, 6)

def map_kehadiran(value):
    mapping = {'Rendah': 30, 'Sedang': 70, 'Tinggi': 90}
    return mapping.get(value, 70)

def map_tugas(value):
    mapping = {'Sering telat': 20, 'Tepat waktu': 80}
    return mapping.get(value, 50)

# Fungsi utama fuzzy inference untuk satu row dataframe
def fuzzy_risk(row):
    risk_sim.input['fluktuasi'] = map_fluktuasi(row['Fluktuasi_IPS'])
    risk_sim.input['sks_tidak_lulus'] = map_sks_tidak_lulus(row['SKS_Tidak_Lulus'])
    risk_sim.input['kehadiran'] = map_kehadiran(row['Kehadiran'])
    risk_sim.input['tugas'] = map_tugas(row['Tugas'])
    risk_sim.compute()
    return risk_sim.output['risiko']

# Klasifikasi hasil fuzzy ke label kategori
def risiko_to_label(val):
    if val < 40:
        return "Lulus Cepat"
    elif val < 60:
        return "Tepat Waktu"
    else:
        return "Terlambat/DO"
