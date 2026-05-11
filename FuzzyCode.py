import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="SPK Tingkat Obesitas - Fuzzy Logic", layout="wide")

# ==========================================
# 1. DEFINISI SISTEM FUZZY (MAMDANI)
# ==========================================
# Definisi Variabel Input (Antecedent)
bmi = ctrl.Antecedent(np.arange(10, 55, 1), 'bmi')
age = ctrl.Antecedent(np.arange(10, 70, 1), 'age')
faf = ctrl.Antecedent(np.arange(0, 4, 0.1), 'faf')
fcvc = ctrl.Antecedent(np.arange(1, 4, 0.1), 'fcvc')
ncp = ctrl.Antecedent(np.arange(1, 5, 0.1), 'ncp')

# Definisi Variabel Output (Consequent)[cite: 1]
# Representasi risiko obesitas (0 - 100)
risiko = ctrl.Consequent(np.arange(0, 101, 1), 'risiko')

# Fungsi Keanggotaan BMI (Berdasarkan min 12.9, max 50.8)[cite: 1]
bmi['kurus'] = fuzz.trapmf(bmi.universe, [10, 10, 18.5, 20])
bmi['normal'] = fuzz.trimf(bmi.universe, [18.5, 22, 25])
bmi['gemuk'] = fuzz.trimf(bmi.universe, [24, 27.5, 30])
bmi['obesitas'] = fuzz.trapmf(bmi.universe, [29, 35, 55, 55])

# Fungsi Keanggotaan Age (Berdasarkan min 14, max 61)[cite: 1]
age['muda'] = fuzz.trapmf(age.universe, [10, 10, 20, 25])
age['dewasa'] = fuzz.trimf(age.universe, [20, 35, 50])
age['tua'] = fuzz.trapmf(age.universe, [45, 55, 70, 70])

# Fungsi Keanggotaan FAF (Aktivitas Fisik: 0 - 3)[cite: 1]
faf['jarang'] = fuzz.trapmf(faf.universe, [0, 0, 0.5, 1.5])
faf['sedang'] = fuzz.trimf(faf.universe, [1, 2, 3])
faf['sering'] = fuzz.trapmf(faf.universe, [2.5, 3, 4, 4])

# Fungsi Keanggotaan FCVC (Konsumsi Sayur: 1 - 3)[cite: 1]
fcvc['kurang'] = fuzz.trimf(fcvc.universe, [1, 1, 2])
fcvc['cukup'] = fuzz.trimf(fcvc.universe, [1.5, 2, 2.5])
fcvc['baik'] = fuzz.trimf(fcvc.universe, [2, 3, 3])

# Fungsi Keanggotaan NCP (Jumlah Makan: 1 - 4)[cite: 1]
ncp['sedikit'] = fuzz.trimf(ncp.universe, [1, 1, 2.5])
ncp['normal'] = fuzz.trimf(ncp.universe, [2, 3, 3.5])
ncp['banyak'] = fuzz.trapmf(ncp.universe, [3, 4, 5, 5])

# Fungsi Keanggotaan Output Risiko Obesitas[cite: 1]
risiko['rendah'] = fuzz.trapmf(risiko.universe, [0, 0, 30, 45])
risiko['sedang'] = fuzz.trimf(risiko.universe, [35, 50, 65])
risiko['tinggi'] = fuzz.trapmf(risiko.universe, [55, 75, 100, 100])

# Dasar Aturan (Rule Base)[cite: 1]
# Dasar Aturan (Rule Base) - Komprehensif (Cover 100% Kemungkinan)[cite: 1]
rules = [
    # ==========================================
    # 1. KELOMPOK BMI KURUS (Mewakili 81 kemungkinan)
    # Logika: Jika BMI kurus, bagaimanapun gaya hidupnya, risiko obesitasnya rendah.
    ctrl.Rule(bmi['kurus'], risiko['rendah']),

    # ==========================================
    # 2. KELOMPOK BMI OBESITAS (Mewakili 81 kemungkinan)
    # Logika: Jika sudah masuk kategori obesitas, risikonya mutlak tinggi.
    ctrl.Rule(bmi['obesitas'], risiko['tinggi']),

    # ==========================================
    # 3. KELOMPOK BMI NORMAL (Mewakili 81 kemungkinan)
    # Normal + Rajin/Lumayan Olahraga -> Aman
    ctrl.Rule(bmi['normal'] & (faf['sering'] | faf['sedang']), risiko['rendah']),

    # Normal + Jarang Olahraga + Makan Sedikit/Normal -> Masih Aman
    ctrl.Rule(bmi['normal'] & faf['jarang'] & (ncp['sedikit'] | ncp['normal']), risiko['rendah']),

    # Normal + Jarang Olahraga + Makan Banyak -> Peringatan (Sedang)
    ctrl.Rule(bmi['normal'] & faf['jarang'] & ncp['banyak'], risiko['sedang']),

    # ==========================================
    # 4. KELOMPOK BMI GEMUK (Mewakili 81 kemungkinan)
    # Gemuk + Sering Olahraga -> Risiko bisa ditekan (Sedang)
    ctrl.Rule(bmi['gemuk'] & faf['sering'], risiko['sedang']),

    # Gemuk + Olahraga Sedang + Sayur Banyak -> Risiko Sedang
    ctrl.Rule(bmi['gemuk'] & faf['sedang'] & fcvc['baik'], risiko['sedang']),

    # Gemuk + Olahraga Sedang + Kurang/Cukup Sayur -> Bahaya (Tinggi)
    ctrl.Rule(bmi['gemuk'] & faf['sedang'] & (fcvc['kurang'] | fcvc['cukup']), risiko['tinggi']),

    # Gemuk + Jarang Olahraga -> Sangat Bahaya (Tinggi)
    ctrl.Rule(bmi['gemuk'] & faf['jarang'], risiko['tinggi']),

    # ==========================================
    # 5. ATURAN SPESIFIK PENGUAT BERDASARKAN UMUR (Metabolisme)
    # Tua + Normal + Jarang Olahraga + Makan Banyak -> Makin rentan naik (Sedang)
    ctrl.Rule(age['tua'] & bmi['normal'] & faf['jarang'] & ncp['banyak'], risiko['sedang']),

    # Tua + Gemuk + Olahraga Kurang/Sedang -> Risiko makin tinggi
    ctrl.Rule(age['tua'] & bmi['gemuk'] & (faf['jarang'] | faf['sedang']), risiko['tinggi']),

    # Muda + Gemuk + Olahraga Sering + Makan Sayur -> Metabolisme masih bagus (Rendah)
    ctrl.Rule(age['muda'] & bmi['gemuk'] & faf['sering'] & fcvc['baik'], risiko['rendah'])
]

# Pembuatan Sistem Kontrol[cite: 1]
risiko_ctrl = ctrl.ControlSystem(rules)
simulasi_risiko = ctrl.ControlSystemSimulation(risiko_ctrl)

# ==========================================
# 2. ANTARMUKA STREAMLIT
# ==========================================

# Navigasi Sidebar
st.sidebar.title("Navigasi SPK")
menu = st.sidebar.radio("Pilih Halaman:", ["Profil & Data", "Simulasi Fuzzy SPK"])

if menu == "Profil & Data":
    st.title("Sistem Pendukung Keputusan: Prediksi Risiko Obesitas")
    st.write("Dibuat dengan Metode **Fuzzy Logic (Mamdani)**")

    st.subheader("Tampilan Dataset")
    # Load data (pastikan file csv ada di direktori yang sama)
    try:
        df = pd.read_csv("../database/ObesityDataSet.csv") # Sesuaikan nama file
        # Feature Engineering BMI
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

        # Menampilkan dataset mentah dalam tabel interaktif[cite: 2]
        st.dataframe(df)
        st.write(f"Total Baris Data: {len(df)}")
    except FileNotFoundError:
        st.warning("Silakan pastikan file CSV dataset berada di folder yang sama dengan script ini.")

elif menu == "Simulasi Fuzzy SPK":
    st.title("Simulasi Perhitungan Fuzzy")
    st.write("Masukkan nilai kriteria pasien untuk melihat tingkat risiko obesitas.")

    col1, col2 = st.columns(2)

    # Input Widget Dinamis[cite: 2]
    with col1:
        st.subheader("Kriteria Fisik")
        val_bmi = st.slider("BMI (Body Mass Index)", 12.0, 55.0, 29.7)
        val_age = st.number_input("Umur (Age)", min_value=14, max_value=70, value=24)

    with col2:
        st.subheader("Gaya Hidup")
        val_faf = st.slider("Aktivitas Fisik (FAF)", 0.0, 3.0, 1.0)
        val_fcvc = st.slider("Konsumsi Sayur (FCVC)", 1.0, 3.0, 2.4)
        val_ncp = st.slider("Jumlah Makan Utama (NCP)", 1.0, 4.0, 3.0)

    # Tombol Eksekusi[cite: 2]
    if st.button("Hitung Risiko Obesitas", type="primary"):
        simulasi_risiko.input['bmi'] = val_bmi
        simulasi_risiko.input['age'] = val_age
        simulasi_risiko.input['faf'] = val_faf
        simulasi_risiko.input['fcvc'] = val_fcvc
        simulasi_risiko.input['ncp'] = val_ncp

        try:
            # Proses penalaran / komputasi
            simulasi_risiko.compute()
            hasil_skor = simulasi_risiko.output['risiko']

            # Tampilan Hasil
            st.success(f"### Skor Risiko Obesitas: {hasil_skor:.2f} / 100")

            # Visualisasi Grafik Fungsi Keanggotaan Output
            st.subheader("Visualisasi Area Defuzzifikasi")
            fig, ax = plt.subplots(figsize=(8, 4))
            risiko.view(sim=simulasi_risiko, ax=ax)
            st.pyplot(fig)

        except KeyError:
            # Menangkap error jika tidak ada rule yang cocok
            st.error("Sistem tidak bisa menghitung! Kombinasi input ini belum ada di dalam Rule Base.")
            st.info("💡 Silakan lengkapi dasar aturan (rules) di kode Python-mu untuk mencakup kondisi ini.")