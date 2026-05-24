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
    page_title="SPK Penentuan Tingkat Keparahan Obesitas - Fuzzy Mamdani",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
# Cek apakah variabel 'bmi_ow_thresh' sudah ada di memori sesi atau belum. Kalau belum ada, buat dulu dengan nilai awal 25.0 (batas BMI untuk kategori Overweight)
if 'bmi_ow_thresh' not in st.session_state: st.session_state.bmi_ow_thresh = 25.0

# Cek variabel beberapa variabel bobot kriteria. Kalau belum ada, buat dengan nilai awal 1.0 (artinya tidak ada penambahan/pengurangan bobot)
if 'faf_weight' not in st.session_state: st.session_state.faf_weight = 1.0
if 'fcvc_weight' not in st.session_state: st.session_state.fcvc_weight = 1.0
if 'ch2o_weight' not in st.session_state: st.session_state.ch2o_weight = 1.0
if 'ncp_weight' not in st.session_state: st.session_state.ncp_weight = 1.0

# Cek variabel 'defuzz_method' (metode defuzzifikasi yang dipilih user). Kalau belum ada, set default ke 'Centroid' (metode paling umum)
# Huruf besar penting karena nanti dicek dengan opsi ["Centroid", "Bisector", "MOM"]
if 'defuzz_method' not in st.session_state: st.session_state.defuzz_method = 'Centroid'

# Cek variabel 'top_n' (berapa banyak data teratas yang di-highlight di tabel hasil). Kalau belum ada, default 10 (artinya 10 data paling parah akan di-highlight)
if 'top_n' not in st.session_state: st.session_state.top_n = 10

# Cek variabel 'df_data' (tempat menyimpan dataset yang sudah diupload user). Kalau belum ada, isi dengan None (kosong) dulu.
# Ini penting agar halaman tidak error saat dataset belum diupload
if 'df_data' not in st.session_state: st.session_state.df_data = None # Fix memori dataset agar tidak FileNotFoundError

# ==========================================
# FUNGSI CACHE UNTUK LOAD DATA
# ==========================================
@st.cache_data
# Decorator dari Streamlit untuk "cache" (menyimpan hasil fungsi di memori).
# Artinya kalau fungsi ini dipanggil dengan input yang SAMA, Streamlit tidak akan menjalankannya ulang, langsung pakai hasil yang sudah tersimpan.

def load_data(uploaded_file=None):
    try:
        # Kalau user mengupload file CSV lewat st.file_uploader, maka 'uploaded_file' berisi file tersebut (bukan None). Langsung baca file yang diupload itu.
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Mencoba load dari file lokal jika tidak ada file yang diupload
            df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

        # Pre-processing: Hitung BMI (Weight / Height^2)
        # Cek apakah kolom 'BMI' sudah ada di dataset atau belum.
        if 'BMI' not in df.columns:
            df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        return df

    # Kalau file tidak ditemukan (misal nama file salah atau file tidak ada), kembalikan None (kosong) daripada crash dengan error merah.
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Terjadi error: {e}")
        return None

# ==========================================
# FUNGSI PEMBUAT SISTEM FUZZY
# ==========================================
def fuzzy_system(bmi_thresh, defuzz_method):

    # 1. Deklarasi Variabel Antecedent (Input) & Consequent (Output)
    bmi = ctrl.Antecedent(np.arange(0, 61, 0.1), 'bmi')
    faf = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'faf')
    fcvc = ctrl.Antecedent(np.arange(1, 3.1, 0.1), 'fcvc')
    ch2o = ctrl.Antecedent(np.arange(1, 3.1, 0.1), 'ch2o')
    ncp = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'ncp')

    score = ctrl.Consequent(np.arange(0, 101, 1), 'score',
    defuzzify_method=defuzz_method.lower())

    # 2. Fungsi Keanggotaan
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

    # 3. Rule Base (10 Aturan Logis)
    rule1 = ctrl.Rule(bmi['obese'] & (faf['low'] | fcvc['low'] | ncp['high']), score['sangat_tinggi'])
    # Jika Obese kurang minum air ATAU olahraga sedang
    rule2 = ctrl.Rule(bmi['obese'] & (ch2o['low'] | faf['medium']), score['sangat_tinggi'])
    # Jika Obese tapi rajin olahraga DAN makan sayur (sedikit turun ke tinggi)
    rule3 = ctrl.Rule(bmi['obese'] & faf['high'] & fcvc['high'], score['tinggi'])
    # Catch-all aman untuk Obese agar tidak ada blank spot
    rule4 = ctrl.Rule(bmi['obese'] & ch2o['medium'], score['sangat_tinggi'])

    # --- KELOMPOK OVERWEIGHT (Prioritas Menengah ke Tinggi) ---
    # Jika Overweight dan punya kebiasaan sangat buruk
    rule5 = ctrl.Rule(bmi['overweight'] & (faf['low'] | fcvc['low'] | ncp['high'] | ch2o['low']), score['tinggi'])
    # Jika Overweight tapi kebiasaan cukup baik
    rule6 = ctrl.Rule(bmi['overweight'] & (faf['medium'] | faf['high']) & (fcvc['medium'] | fcvc['high']), score['sedang'])
    # Catch-all aman untuk Overweight
    rule7 = ctrl.Rule(bmi['overweight'] & (ch2o['high'] | ch2o['medium']), score['sedang'])

    # --- KELOMPOK NORMAL (Prioritas Rendah ke Menengah) ---
    # Jika Normal tapi kebiasaan buruk (warning bisa naik berat badan)
    rule8 = ctrl.Rule(bmi['normal'] & (faf['low'] | ncp['high'] | fcvc['low']), score['sedang'])
    # Jika Normal dan kebiasaan baik
    rule9 = ctrl.Rule(bmi['normal'] & (faf['high'] | fcvc['high']), score['rendah'])
    # Jika Normal dan aktivitas sedang
    rule10 = ctrl.Rule(bmi['normal'] & (faf['medium'] | ch2o['medium']), score['rendah'])
    # Catch-all aman untuk Normal
    rule11 = ctrl.Rule(bmi['normal'] & ch2o['high'], score['rendah'])

    # --- KELOMPOK UNDERWEIGHT (Bukan Target Penanganan Obesitas) ---
    # Semua yang underweight langsung mendapat prioritas rendah untuk penanganan *obesitas*
    rule12 = ctrl.Rule(bmi['underweight'], score['rendah'])

    # --- ATURAN PENGUAT KONDISI EKSTREM (Independen dari BMI) ---
    # Jika gaya hidup sangat kacau (makan banyak, no olahraga, no air)
    rule13 = ctrl.Rule(ncp['high'] & faf['low'] & ch2o['low'], score['tinggi'])
    # Jika gaya hidup sangat sehat
    rule14 = ctrl.Rule(faf['high'] & fcvc['high'] & ch2o['high'], score['rendah'])
    # Jika kurang gizi (kurang makan, kurang minum)
    rule15 = ctrl.Rule(ncp['low'] & ch2o['low'], score['rendah'])

    # 4. Control System Fuzzy
    # 4. Control System Fuzzy
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15]
    obesity_ctrl = ctrl.ControlSystem(rules)
    obesity_sim = ctrl.ControlSystemSimulation(obesity_ctrl)

    return obesity_sim, bmi, faf, fcvc, ch2o, ncp, score

# ==========================================
# FUNGSI HELPER
# ==========================================
# Helper untuk mendapatkan kategori teks dari nilai crisp
def tingkat_keparahan(crisp_val):
    if crisp_val <= 30: return "Rendah"
    elif crisp_val <= 60: return "Sedang"
    elif crisp_val <= 80: return "Tinggi"
    else: return "Sangat Tinggi"

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigasi SPK")
menu = st.sidebar.radio("Pilih Halaman:",
                        ["📊 Dataset", "⚙️ Konfigurasi Fuzzy", "🏆 Hitung & Peringkat SPK", "👥 Tentang Program & Kelompok"])

st.sidebar.markdown("---")

# ==========================================
# HALAMAN 1: DATASET
# ==========================================
if menu == "📊 Dataset":
    st.title("📊 Dataset")

    st.markdown("""Silakan upload file CSV dataset.""")
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

        st.subheader("Dataset Mentah")
        # Tabel keterangan kolom supaya user paham arti setiap kolom
        with st.expander("📖 Keterangan Kolom Dataset (klik untuk buka)"):
            keterangan = {
                "Kolom": ["Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS", "NObeyesdad"],
                "Keterangan": [
                    "Jenis kelamin",
                    "Usia",
                    "Tinggi badan",
                    "Berat badan",
                    "Apakah ada anggota keluarga yang pernah/sedang mengalami kelebihan berat badan?",
                    "Apakah sering mengonsumsi makanan tinggi kalori?",
                    "Seberapa sering mengonsumsi sayuran dalam makanan sehari-hari?",
                    "Berapa kali makan besar dalam sehari?",
                    "Apakah makan di antara waktu makan utama?",
                    "Apakah merokok?",
                    "Berapa liter air yang diminum setiap hari?",
                    "Apakah memantau kalori yang dikonsumsi setiap hari?",
                    "Seberapa sering melakukan aktivitas fisik?",
                    "Berapa lama menggunakan perangkat teknologi (HP, TV, komputer, dll) per hari?",
                    "Seberapa sering mengonsumsi alkohol?",
                    "Transportasi yang biasa digunakan sehari-hari",
                    "Label tingkat obesitas"
                ]
            }
            st.table(pd.DataFrame(keterangan))
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistik Deskriptif")
            with st.expander("📖 Penjelasan Indikator Statistik (klik untuk buka)"):
                keteranganStatistik = {
                    "Indikator": ["count", "mean", "std", "min", "25%",
                            "50%", "75%", "max"],
                    "Penjelasan": [
                        "Jumlah data yang tersedia (tidak termasuk yang kosong/NaN)",
                        "Nilai rata-rata dari seluruh data",
                        "Standar deviasi, seberapa jauh data menyebar dari rata-rata",
                        "Nilai terkecil dalam kolom",
                        "Kuartil bawah, 25% data berada di bawah nilai ini",
                        "Median, nilai tengah, 50% data di bawah dan 50% di atas nilai ini",
                        "Kuartil atas, 75% data berada di bawah nilai ini",
                        "Nilai terbesar dalam kolom"
                        ]
                }
                st.table(pd.DataFrame(keteranganStatistik))
            st.write(df.describe())

        with col2:
            st.subheader("Distribusi Label Tingkat Obesitas")
            st.markdown("*Label ini hanya sebagai referensi (menampilkan data) dan **TIDAK** digunakan sebagai input SPK.*")
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

    st.markdown("### Pengaturan Variabel")
    col_w1, col_w2, col_w3= st.columns(3)

    with col_w1:
        st.session_state.bmi_ow_thresh = st.slider("Batas Awal Overweight (BMI Threshold)", min_value=23.0, max_value=28.0, value=st.session_state.bmi_ow_thresh, step=0.5)
        st.caption("📏 Batas nilai BMI seseorang mulai dianggap kelebihan berat badan. Standar Asia = 23.0, Standar WHO = 25.0.")
        st.session_state.faf_weight = st.slider("Bobot Aktivitas Fisik (FAF)", min_value=0.5, max_value=2.0, value=st.session_state.faf_weight, step=0.1)
        st.caption("🏃 Seberapa besar pengaruh aktivitas fisik terhadap hasil penilaian.")

    with col_w2:
        st.session_state.ch2o_weight = st.slider("Bobot Konsumsi Air (CH2O)", min_value=0.5, max_value=2.0, value=st.session_state.ch2o_weight, step=0.1)
        st.caption("💧 Seberapa besar pengaruh kebiasaan minum air putih terhadap hasil penilaian.")
        st.session_state.ncp_weight = st.slider("Bobot Frekuensi Makan (NCP)", min_value=0.5, max_value=2.0, value=st.session_state.ncp_weight, step=0.1)
        st.caption("🍽️ Seberapa besar pengaruh jumlah makan utama per hari terhadap hasil penilaian.")

    with col_w3:
        st.session_state.fcvc_weight = st.slider("Bobot Konsumsi Sayur (FCVC)", min_value=0.5, max_value=2.0, value=st.session_state.fcvc_weight, step=0.1)
        st.caption("🥦 Seberapa besar pengaruh kebiasaan makan sayur terhadap hasil penilaian.")

        # ERROR HANDLING ERROR SELECTBOX: Pengecekan aman huruf besar/kecil
        opsi_defuzz = ["Centroid", "Bisector", "MOM"]
        default_idx = opsi_defuzz.index(st.session_state.defuzz_method) if st.session_state.defuzz_method in opsi_defuzz else 0
        st.session_state.defuzz_method = st.selectbox("Metode Defuzzifikasi", options=opsi_defuzz, index=default_idx)

        st.session_state.top_n = st.number_input("Jumlah Top-N Hasil Peringkat", min_value=5, max_value=50, value=st.session_state.top_n, step=5)

    st.markdown("---")

    # Visualisasi grafik SEMENTARA
    sim, bmi, faf, fcvc, ch2o, ncp, score = fuzzy_system(st.session_state.bmi_ow_thresh, st.session_state.defuzz_method)

    st.subheader("📈 Kurva Fungsi Keanggotaan")
    viz_option = st.selectbox("Pilih Variabel untuk Divisualisasikan:", ["BMI", "FAF", "FCVC", "CH2O", "NCP", "Output Skor"])

    # FIX ERROR GRAFIK BLANK KOSONG
    plt.close('all') # Bersihkan memori kanvas sebelumnya

    if viz_option == "BMI":
        bmi.view()
        plt.title("Fungsi Keanggotaan: BMI")
    elif viz_option == "FAF":
        faf.view()
        plt.title("Fungsi Keanggotaan: Frekuensi Aktivitas Fisik (FAF)")
    elif viz_option == "FCVC":
        fcvc.view()
        plt.title("Fungsi Keanggotaan: Konsumsi Sayur (FCVC)")
    elif viz_option == "CH2O":
        ch2o.view()
        plt.title("Fungsi Keanggotaan: Konsumsi Air (CH2O)")
    elif viz_option == "NCP":
        ncp.view()
        plt.title("Fungsi Keanggotaan: Jumlah Makan Utama (NCP)")
    elif viz_option == "Output Skor":
        score.view()
        plt.title("Fungsi Keanggotaan: Skor Keparahan (Output)")

    # Tangkap grafik buatan skfuzzy dan lempar ke Streamlit
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📚 Tabel Rule Base (Aturan Fuzzy)")
    rules_data = {
        "Rule": [f"Rule {i}" for i in range(1, 16)],
        "Kondisi (IF)": [
            "BMI is Obese AND (FAF is Low OR FCVC is Low OR NCP is High)",
            "BMI is Obese AND (CH2O is Low OR FAF is Medium)",
            "BMI is Obese AND FAF is High AND FCVC is High",
            "BMI is Obese AND CH2O is Medium",
            "BMI is Overweight AND (FAF is Low OR FCVC is Low OR NCP is High OR CH2O is Low)",
            "BMI is Overweight AND (FAF is Medium OR High) AND (FCVC is Medium OR High)",
            "BMI is Overweight AND (CH2O is High OR Medium)",
            "BMI is Normal AND (FAF is Low OR NCP is High OR FCVC is Low)",
            "BMI is Normal AND (FAF is High OR FCVC is High)",
            "BMI is Normal AND (FAF is Medium OR CH2O is Medium)",
            "BMI is Normal AND CH2O is High",
            "BMI is Underweight (Semua Kondisi)",
            "NCP is High AND FAF is Low AND CH2O is Low (Ekstrem Buruk)",
            "FAF is High AND FCVC is High AND CH2O is High (Ekstrem Sehat)",
            "NCP is Low AND CH2O is Low"
        ],
        "Keputusan (THEN) Skor": [
            "Sangat Tinggi", "Sangat Tinggi", "Tinggi", "Sangat Tinggi",
            "Tinggi", "Sedang", "Sedang",
            "Sedang", "Rendah", "Rendah", "Rendah",
            "Rendah", "Tinggi", "Rendah", "Rendah"
        ]
    }
    st.table(pd.DataFrame(rules_data))

# ==========================================
# HALAMAN 3: HITUNG & PERINGKAT SPK
# ==========================================
elif menu == "🏆 Hitung & Peringkat SPK":
    st.title("🏆 Perhitungan SPK Fuzzy Mamdani")

    # FIX FILENOTFOUNDERROR: Ambil data dari Session State, bukan baca file baru!
    df = st.session_state.df_data

    if df is None:
        st.warning("Dataset belum diload! Silakan ke halaman 'Dataset' dan Upload file CSV terlebih dahulu.")
    else:
        if st.button("🚀 Hitung Fuzzy Mamdani", use_container_width=True, type="primary"):

            # Bangun sistem berdasarkan konfigurasi terbaru
            sim, bmi_var, faf_var, fcvc_var, ch2o_var, ncp_var, score_var = fuzzy_system(
                st.session_state.bmi_ow_thresh,
                st.session_state.defuzz_method
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            process_data = []

            total_rows = len(df)
            for i, (idx, row) in enumerate(df.iterrows()):
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Menghitung data {i + 1} dari {total_rows}...")

                val_bmi = row['BMI']
                val_faf = np.clip(row['FAF'] * st.session_state.faf_weight, 0, 3)
                val_fcvc = np.clip(row['FCVC'] * st.session_state.fcvc_weight, 1, 3)
                val_ch2o = np.clip(row['CH2O'] * st.session_state.ch2o_weight, 1, 3)
                val_ncp  = np.clip(row['NCP']  * st.session_state.ncp_weight,  1, 4)

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

                keparahan = tingkat_keparahan(output_score)

                process_data.append({
                    "Index_Asli"         : idx,
                    "Age"                : round(row['Age'], 1),
                    "Gender"             : row['Gender'],
                    "BMI_Value"          : round(val_bmi, 2),
                    "FAF_Value"          : round(val_faf, 2),
                    "FCVC_Value"         : round(val_fcvc, 2),
                    "CH2O_Value"         : round(val_ch2o, 2),
                    "NCP_Value"          : round(val_ncp, 2),
                    "Skor_Fuzzy"         : round(output_score, 2),
                    "Kategori_Keparahan" : keparahan,
                })

            status_text.text("Perhitungan Selesai!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            res_df = pd.DataFrame(process_data)

            # Tabel proses defuzzifikasi
            st.subheader("🔬 Tabel Proses Defuzzifikasi (Sampel 10 Data teratas)")

            defuzz_table = res_df.head(10)[['Index_Asli', 'Age', 'Gender', 'BMI_Value', 'FAF_Value', 'FCVC_Value', 'CH2O_Value', 'NCP_Value', 'Skor_Fuzzy', 'Kategori_Keparahan']].copy()
            defuzz_table = defuzz_table.rename(columns={
                'Index_Asli'         : 'ID Data',
                'Age'                : 'Usia',
                'Gender'             : 'Jenis Kelamin',
                'BMI_Value'          : 'BMI (Input)',
                'FAF_Value'          : 'FAF (Input)',
                'FCVC_Value'         : 'FCVC (Input)',
                'CH2O_Value'         : 'CH2O (Input)',
                'NCP_Value'          : 'NCP (Input)',
                'Skor_Fuzzy'         : 'Skor Crisp (Output Defuzzifikasi)',
                'Kategori_Keparahan' : 'Kategori Keparahan'
            })
            st.dataframe(defuzz_table, use_container_width=True, hide_index=True)

            st.markdown("---")
            # Warning Data Gagal Dihitung
            gagal = sum(1 for d in process_data if d['Skor_Fuzzy'] == 0.0)
            if gagal > 0:
                st.warning(f"⚠️ {gagal} dari {total_rows} data ({gagal/total_rows*100:.1f}%) tidak cocok dengan rules yang ada, skornya otomatis 0.")

            st.subheader("🏅 Hasil Peringkat Akhir (Diurutkan dari Paling Parah)")
            res_df = res_df.sort_values(by="Skor_Fuzzy", ascending=False).reset_index(drop=True)
            res_df.index = res_df.index + 1
            res_df = res_df.rename_axis("Peringkat")

            top_n = st.session_state.top_n
            def highlight_top(s, n):
                return ['background-color: #f79525; color: black' if i < n else '' for i in range(len(s))]

            styled_df = res_df.style.apply(highlight_top, n=top_n, axis=0)
            st.dataframe(styled_df, use_container_width=True)

            st.markdown("---")

            st.subheader("📊 Distribusi Tingkat Keparahan Obesitas (Keseluruhan)")

            # Hitung jumlah per kategori
            kategori_counts = res_df['Kategori_Keparahan'].value_counts().reindex(
                ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'], fill_value=0
            )

            col_pie, col_bar = st.columns(2)

            with col_pie:
                fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
                ax_pie.pie(
                    kategori_counts.values,
                    labels=kategori_counts.index,
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90
                )
                ax_pie.set_title("Proporsi Kategori Keparahan")
                st.pyplot(fig_pie)

            with col_bar:
                fig_dist, ax_dist = plt.subplots(figsize=(5, 5))
                ax_dist.bar(kategori_counts.index, kategori_counts.values, color=colors)
                ax_dist.set_xlabel("Kategori Keparahan")
                ax_dist.set_ylabel("Jumlah Data")
                ax_dist.set_title("Jumlah Data per Kategori Keparahan")
                for i, v in enumerate(kategori_counts.values):
                    ax_dist.text(i, v + 5, str(v), ha='center', fontweight='bold')
                st.pyplot(fig_dist)

# =============================================
# HALAMAN 4: ABOUT PROGRAM DAN ANGGOTA KELOMPOK
# =============================================
elif menu == "👥 Tentang Program & Kelompok":
    st.title("👥 Tentang Program & Kelompok")

    st.markdown("### Anggota Kelompok")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(""" **Nama:** Hisyam L Baihaqi""")
        st.markdown(""" **NIM:** 123240117""")
    with col2:
        st.markdown(""" **Nama:** Muhamad Atallah Alfa Dzaky""")
        st.markdown(""" **NIM:** 123240105""")

    st.markdown("---")

    st.markdown("### Informasi Dataset")
    st.write("- **Sumber Dataset:** Kaggle.com")
    st.write("- **Nama Dataset:** Obesity Levels (Prediction of Obesity Levels Based On Eating Habits and Physical Activites)")
    st.write("- **Author Dataset:** Fatemeh Mehrparvar")
    st.link_button("Kunjungi Link", "https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels/data")

    st.markdown("---")

    st.markdown("### Informasi Metode SPK")
    st.write("- **Tujuan:** Menentukan Tingkat Keparahan Obesitas untuk Penentuan Prioritas Penanganan")
    st.write("- Sistem mengimplementasikan Algoritma **Fuzzy Mamdani**")
    st.write("- **Fungsi Keanggotaan** menggunakan kombinasi Kurva Segitiga (`trimf`) dan Trapesium (`trapmf`)")
    st.write("- Defuzzifikasi berbasis perhitungan area (mendukung *Centroid, Bisector, MOM*)")

    st.markdown("---")

    st.markdown("### **Defuzzifikasi**")
    st.write("Berbasis perhitungan area (*Centroid, Bisector, MOM*)")
    tab1, tab2, tab3 = st.tabs(["📐 Centroid", "✂️ Bisector", "🎯 MOM"])

    with tab1:
        st.markdown("**Centroid (Center of Gravity)**")
        st.markdown("""
        Metode paling umum digunakan. Mencari **titik berat** dari keseluruhan area hasil agregasi output fuzzy.
        Hasilnya adalah nilai tengah yang merepresentasikan 'pusat massa' dari bentuk fuzzy tersebut.
        > *Analogi:* Seperti mencari titik keseimbangan sebuah bangun datar di atas ujung pensil.
        - ⚙️ Default yang digunakan `skfuzzy` jika tidak disebutkan
        """)

    with tab2:
        st.markdown("**Bisector (Garis Pembagi)**")
        st.markdown("""
        Mencari **garis vertikal** yang membagi total area hasil fuzzy menjadi **dua bagian sama besar** (50%-50%).
        Mirip dengan Centroid, namun fokus pada pembagian luas area, bukan titik berat.
        > *Analogi:* Seperti memotong sebuah pizza tidak beraturan menjadi dua bagian dengan luas yang sama.
        """)

    with tab3:
        st.markdown("**MOM (Mean of Maximum)**")
        st.markdown("""
        Mengambil **rata-rata dari semua nilai** yang memiliki derajat keanggotaan output **tertinggi (maksimum)**.
        Hanya memperhatikan bagian puncak dari kurva output, mengabaikan area lainnya.
        > *Analogi:* Dari semua nilai yang "paling yakin benar", ambil rata-ratanya.
        """)