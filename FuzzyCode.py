import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SPK Obesitas - Fuzzy Mamdani",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
if 'bmi_ow_thresh' not in st.session_state:
    st.session_state.bmi_ow_thresh = 25.0
if 'faf_weight' not in st.session_state:
    st.session_state.faf_weight = 1.0
if 'defuzz_method' not in st.session_state:
    st.session_state.defuzz_method = 'Centroid' # Fix huruf besar
if 'top_n' not in st.session_state:
    st.session_state.top_n = 10
if 'df_data' not in st.session_state:
    st.session_state.df_data = None # Fix memori dataset agar tidak FileNotFoundError

# ==========================================
# FUNGSI CACHE UNTUK LOAD DATA
# ==========================================
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Mencoba load dari file lokal jika tidak ada file yang diupload
            df = pd.read_csv('Kuliah/semester_4/SCPK_Prioritas_Penanganan_Obesitas-_Menggunakan_Metode_Fuzzy_Logic/ObesityDataSet_raw_and_data_sinthetic.csv')

        # Pre-processing: Hitung BMI (Weight / Height^2)
        if 'BMI' not in df.columns:
            df['BMI'] = df['Weight'] / (df['Height'] ** 2)

        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

# ==========================================
# FUNGSI PEMBUAT SISTEM FUZZY DINAMIS
# ==========================================
def build_fuzzy_system(bmi_thresh, defuzz_method):
    # 1. Deklarasi Variabel Antecedent (Input) & Consequent (Output)
    bmi = ctrl.Antecedent(np.arange(0, 61, 0.1), 'bmi')
    faf = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'faf')
    fcvc = ctrl.Antecedent(np.arange(1, 3.1, 0.1), 'fcvc')
    ch2o = ctrl.Antecedent(np.arange(1, 3.1, 0.1), 'ch2o')
    ncp = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'ncp')

    score = ctrl.Consequent(np.arange(0, 101, 1), 'score', defuzzify_method=defuzz_method.lower())

    # 2. Membership Functions (Fuzzifikasi)
    bmi['underweight'] = fuzz.trapmf(bmi.universe, [0, 0, 18.0, 19.5])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18.0, 22.0, bmi_thresh])
    bmi['overweight'] = fuzz.trimf(bmi.universe, [23.0, bmi_thresh, 30.0])
    bmi['obese'] = fuzz.trapmf(bmi.universe, [bmi_thresh + 3, 32.0, 60.0, 60.0])

    faf['low'] = fuzz.trapmf(faf.universe, [0, 0, 1.0, 1.5])
    faf['medium'] = fuzz.trimf(faf.universe, [1.0, 1.5, 2.5])
    faf['high'] = fuzz.trapmf(faf.universe, [2.0, 2.5, 3.0, 3.0])

    fcvc['low'] = fuzz.trapmf(fcvc.universe, [1.0, 1.0, 1.5, 2.0])
    fcvc['medium'] = fuzz.trimf(fcvc.universe, [1.5, 2.0, 2.5])
    fcvc['high'] = fuzz.trapmf(fcvc.universe, [2.0, 2.5, 3.0, 3.0])

    ch2o['low'] = fuzz.trapmf(ch2o.universe, [1.0, 1.0, 1.5, 2.0])
    ch2o['medium'] = fuzz.trimf(ch2o.universe, [1.5, 2.0, 2.5])
    ch2o['high'] = fuzz.trapmf(ch2o.universe, [2.0, 2.5, 3.0, 3.0])

    ncp['low'] = fuzz.trapmf(ncp.universe, [1.0, 1.0, 2.0, 2.5])
    ncp['medium'] = fuzz.trimf(ncp.universe, [2.0, 2.5, 3.5])
    ncp['high'] = fuzz.trapmf(ncp.universe, [3.0, 3.5, 4.0, 4.0])

    score['rendah'] = fuzz.trapmf(score.universe, [0, 0, 20, 30])
    score['sedang'] = fuzz.trimf(score.universe, [20, 40, 60])
    score['tinggi'] = fuzz.trimf(score.universe, [50, 65, 80])
    score['sangat_tinggi'] = fuzz.trapmf(score.universe, [70, 85, 100, 100])

    # 3. Rule Base (Minimal 10 Aturan Logis)
    rule1 = ctrl.Rule(bmi['obese'] & faf['low'], score['sangat_tinggi'])
    rule2 = ctrl.Rule(bmi['obese'] & faf['medium'] & fcvc['low'], score['sangat_tinggi'])
    rule3 = ctrl.Rule(bmi['overweight'] & faf['low'] & ch2o['low'], score['tinggi'])
    rule4 = ctrl.Rule(bmi['overweight'] & faf['high'] & ncp['high'], score['sedang'])
    rule5 = ctrl.Rule(bmi['normal'] & faf['high'] & fcvc['high'], score['rendah'])
    rule6 = ctrl.Rule(bmi['normal'] & faf['low'] & ncp['high'], score['sedang'])
    rule7 = ctrl.Rule(bmi['underweight'] & faf['high'], score['rendah'])
    rule8 = ctrl.Rule(bmi['underweight'] & ncp['low'], score['rendah'])
    rule9 = ctrl.Rule(bmi['obese'] & ch2o['high'] & faf['high'], score['tinggi'])
    rule10 = ctrl.Rule(bmi['normal'] & faf['medium'] & ch2o['medium'], score['rendah'])
    rule11 = ctrl.Rule(bmi['overweight'] & fcvc['low'] & ncp['high'], score['tinggi'])
    rule12 = ctrl.Rule(bmi['normal'] & faf['low'] & ch2o['low'] & fcvc['low'], score['sedang'])

    # 4. Bangun Control System
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12]
    obesity_ctrl = ctrl.ControlSystem(rules)
    obesity_sim = ctrl.ControlSystemSimulation(obesity_ctrl)

    return obesity_sim, bmi, faf, fcvc, ch2o, ncp, score

# Helper untuk mendapatkan kategori teks dari nilai crisp
def get_severity_category(crisp_val):
    if crisp_val <= 30: return "Rendah"
    elif crisp_val <= 60: return "Sedang"
    elif crisp_val <= 80: return "Tinggi"
    else: return "Sangat Tinggi"

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigasi SPK")
menu = st.sidebar.radio("Pilih Halaman:",
                        ["📊 Dataset", "⚙️ Konfigurasi Fuzzy", "🏆 Hitung & Peringkat SPK", "👥 Profil Kelompok"])

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini menggunakan logika Fuzzy Mamdani untuk menghitung skor keparahan obesitas.")

# ==========================================
# HALAMAN 1: DATASET
# ==========================================
if menu == "📊 Dataset":
    st.title("📊 Eksplorasi Dataset Obesitas")

    st.markdown("""
    Silakan upload file CSV dataset (contoh: `ObesityDataSet_raw_and_data_sinthetic.csv`).
    """)

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])

    # Update state dataset jika file diupload
    if uploaded_file is not None:
        st.session_state.df_data = load_data(uploaded_file)
    elif st.session_state.df_data is None:
        # Coba load lokal sekali saja jika memori masih kosong
        st.session_state.df_data = load_data()

    df = st.session_state.df_data

    if df is not None:
        st.success(f"Dataset berhasil dimuat! Jumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}")

        st.subheader("Data Mentah (Interaktif)")
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistik Deskriptif")
            st.write(df.describe())

        with col2:
            st.subheader("Distribusi Label Asli (NObeyesdad)")
            st.markdown("*Label ini hanya sebagai referensi dan **TIDAK** digunakan sebagai input SPK.*")
            if 'NObeyesdad' in df.columns:
                target_counts = df['NObeyesdad'].value_counts()
                st.bar_chart(target_counts)
            else:
                st.warning("Kolom 'NObeyesdad' tidak ditemukan di dataset ini.")
    else:
        st.error("Dataset tidak ditemukan. Silakan upload file CSV melalui uploader di atas.")

# ==========================================
# HALAMAN 2: KONFIGURASI FUZZY
# ==========================================
elif menu == "⚙️ Konfigurasi Fuzzy":
    st.title("⚙️ Konfigurasi Parameter Fuzzy Mamdani")

    st.markdown("### Pengaturan Variabel Interaktif")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.session_state.bmi_ow_thresh = st.slider("Batas Awal Overweight (BMI Threshold)", min_value=23.0, max_value=28.0, value=st.session_state.bmi_ow_thresh, step=0.5)
        st.session_state.faf_weight = st.slider("Bobot Aktivitas Fisik (FAF)", min_value=0.5, max_value=2.0, value=st.session_state.faf_weight, step=0.1)
    with col_w2:
        # FIX ERROR SELECTBOX: Pengecekan aman huruf besar/kecil
        opsi_defuzz = ["Centroid", "Bisector", "MOM"]
        default_idx = opsi_defuzz.index(st.session_state.defuzz_method) if st.session_state.defuzz_method in opsi_defuzz else 0
        st.session_state.defuzz_method = st.selectbox("Metode Defuzzifikasi", options=opsi_defuzz, index=default_idx)

        st.session_state.top_n = st.number_input("Jumlah Top-N Hasil Peringkat", min_value=5, max_value=50, value=st.session_state.top_n, step=5)

    st.markdown("---")

    # Build System sementara untuk visualisasi grafik
    sim, bmi, faf, fcvc, ch2o, ncp, score = build_fuzzy_system(st.session_state.bmi_ow_thresh, st.session_state.defuzz_method)

    st.subheader("📈 Kurva Fungsi Keanggotaan (Membership Functions)")
    viz_option = st.selectbox("Pilih Variabel untuk Divisualisasikan:", ["BMI", "FAF", "FCVC", "CH2O", "NCP", "Output Skor"])

    # FIX ERROR GRAFIK BLANK KOSONG
    plt.close('all') # Bersihkan memori kanvas sebelumnya

    if viz_option == "BMI":
        bmi.view()
        plt.title("Membership Function: BMI")
    elif viz_option == "FAF":
        faf.view()
        plt.title("Membership Function: Frekuensi Aktivitas Fisik (FAF)")
    elif viz_option == "FCVC":
        fcvc.view()
        plt.title("Membership Function: Konsumsi Sayur (FCVC)")
    elif viz_option == "CH2O":
        ch2o.view()
        plt.title("Membership Function: Konsumsi Air (CH2O)")
    elif viz_option == "NCP":
        ncp.view()
        plt.title("Membership Function: Jumlah Makan Utama (NCP)")
    elif viz_option == "Output Skor":
        score.view()
        plt.title("Membership Function: Skor Keparahan (Output)")

    # Tangkap grafik buatan skfuzzy dan lempar ke Streamlit
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📚 Tabel Rule Base (Aturan Fuzzy)")
    rules_data = {
        "Rule": [f"Rule {i}" for i in range(1, 13)],
        "Kondisi (IF)": [
            "BMI is Obese AND FAF is Low",
            "BMI is Obese AND FAF is Medium AND FCVC is Low",
            "BMI is Overweight AND FAF is Low AND CH2O is Low",
            "BMI is Overweight AND FAF is High AND NCP is High",
            "BMI is Normal AND FAF is High AND FCVC is High",
            "BMI is Normal AND FAF is Low AND NCP is High",
            "BMI is Underweight AND FAF is High",
            "BMI is Underweight AND NCP is Low",
            "BMI is Obese AND CH2O is High AND FAF is High",
            "BMI is Normal AND FAF is Medium AND CH2O is Medium",
            "BMI is Overweight AND FCVC is Low AND NCP is High",
            "BMI is Normal AND FAF is Low AND CH2O is Low AND FCVC is Low"
        ],
        "Keputusan (THEN) Skor": [
            "Sangat Tinggi", "Sangat Tinggi", "Tinggi", "Sedang",
            "Rendah", "Sedang", "Rendah", "Rendah",
            "Tinggi", "Rendah", "Tinggi", "Sedang"
        ]
    }
    st.table(pd.DataFrame(rules_data))

# ==========================================
# HALAMAN 3: HITUNG & PERINGKAT SPK
# ==========================================
elif menu == "🏆 Hitung & Peringkat SPK":
    st.title("🏆 Perhitungan SPK Fuzzy Mamdani")

    st.markdown("""
    **Alur Proses:**
    1. **Fuzzifikasi:** Mengubah nilai tegas menjadi nilai fuzzy (derajat keanggotaan).
    2. **Inferensi:** Mencocokkan nilai fuzzy dengan Rule Base menggunakan operator MIN.
    3. **Defuzzifikasi:** Menggabungkan output dan mencari nilai akhir.
    """)

    # FIX FILENOTFOUNDERROR: Ambil data dari Session State, bukan baca file baru!
    df = st.session_state.df_data

    if df is None:
        st.warning("Dataset belum diload! Silakan ke halaman 'Dataset' dan Upload file CSV terlebih dahulu.")
    else:
        if st.button("🚀 Hitung Fuzzy Mamdani", use_container_width=True, type="primary"):

            # Bangun sistem berdasarkan konfigurasi terbaru
            sim, bmi_var, faf_var, fcvc_var, ch2o_var, ncp_var, score_var = build_fuzzy_system(
                st.session_state.bmi_ow_thresh,
                st.session_state.defuzz_method
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            process_data = []

            total_rows = len(df)
            for idx, row in df.iterrows():
                progress_bar.progress((idx + 1) / total_rows)
                status_text.text(f"Menghitung data {idx + 1} dari {total_rows}...")

                val_bmi = row['BMI']
                val_faf = np.clip(row['FAF'] * st.session_state.faf_weight, 0, 3)
                val_fcvc = row['FCVC']
                val_ch2o = row['CH2O']
                val_ncp = row['NCP']

                sim.input['bmi'] = val_bmi
                sim.input['faf'] = val_faf
                sim.input['fcvc'] = val_fcvc
                sim.input['ch2o'] = val_ch2o
                sim.input['ncp'] = val_ncp

                try:
                    sim.compute()
                    output_score = sim.output['score']
                except Exception as e:
                    output_score = 0.0

                severity = get_severity_category(output_score)

                process_data.append({
                    "Index_Asli": idx,
                    "Age": round(row['Age'], 1),
                    "Gender": row['Gender'],
                    "BMI_Value": round(val_bmi, 2),
                    "Skor_Fuzzy": round(output_score, 2),
                    "Kategori_Keparahan": severity,
                    "CALC": row.get('CALC', '-'),
                    "SMOKE": row.get('SMOKE', '-')
                })

            status_text.text("Perhitungan Selesai!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            res_df = pd.DataFrame(process_data)

            st.subheader("🔍 Tabel Proses Fuzzifikasi (Sample 5 Data Pertama)")
            st.dataframe(res_df.head(), use_container_width=True)

            st.markdown("---")
            st.subheader("🏅 Hasil Peringkat Akhir (Diurutkan dari Paling Parah)")

            res_df = res_df.sort_values(by="Skor_Fuzzy", ascending=False).reset_index(drop=True)
            res_df.index = res_df.index + 1
            res_df = res_df.rename_axis("Peringkat")

            top_n = st.session_state.top_n

            def highlight_top(s, n):
                return ['background-color: #ffe066; color: black' if i < n else '' for i in range(len(s))]

            styled_df = res_df.style.apply(highlight_top, n=top_n, axis=0)
            st.dataframe(styled_df, use_container_width=True)

            st.markdown("---")
            st.subheader(f"📊 Bar Chart Top 20 Kasus Obesitas Paling Parah")

            top_20 = res_df.head(20).copy()
            top_20['Label'] = top_20.apply(lambda x: f"Idx {x['Index_Asli']} ({x['Gender']}, {x['Age']} th)", axis=1)

            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_20, x='Skor_Fuzzy', y='Label', ax=ax_bar, palette="Reds_r")
            ax_bar.set_xlabel("Skor Keparahan Fuzzy")
            ax_bar.set_ylabel("Identitas Pasien")
            ax_bar.set_title("Top 20 Skor Fuzzy Tertinggi")

            st.pyplot(fig_bar)

# ==========================================
# HALAMAN 4: PROFIL KELOMPOK
# ==========================================
elif menu == "👥 Profil Kelompok":
    st.title("👥 Profil Pengembang & Informasi Sistem")

    st.markdown("### Anggota Pengembang")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown(
            """
            <div style="background-color:#2e2e2e; width:150px; height:200px; display:flex; align-items:center; justify-content:center; border-radius:10px;">
                <h3 style="color:#ffffff;">FOTO<br>PROFIL</h3>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        st.markdown("""
        **Nama:** Hisyam L Baihaqi
        **Status:** Mahasiswa Teknik Informatika (Semester 4)
        **Role:** Programmer / Data Scientist
        """)

    st.markdown("---")

    st.markdown("### Informasi Dataset")
    st.write("- **Sumber Dataset:** Kaggle (*Obesity based on eating habits and physical condition*)")
    st.write("- **Tujuan:** Klasifikasi dan estimasi tingkat obesitas berdasarkan kebiasaan.")

    if st.session_state.df_data is not None:
        st.write(f"- **Total Data:** {st.session_state.df_data.shape[0]} Baris, {st.session_state.df_data.shape[1]} Kolom")

    st.markdown("### Informasi Metode SPK")
    st.markdown("""
    Sistem ini mengimplementasikan Algoritma **Fuzzy Inference System (FIS) Mamdani**.
    - **Fungsi Keanggotaan:** Menggunakan kombinasi Kurva Segitiga (`trimf`) dan Trapesium (`trapmf`).
    - **Inferensi:** Menggunakan operator *Min* (AND) untuk antecedent, dan agregasi *Max* untuk consequent.
    - **Defuzzifikasi:** Berbasis perhitungan area (mendukung *Centroid, Bisector, MOM*).
    """)