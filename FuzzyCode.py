import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

st.set_page_config(
    page_title="SPK Penentuan Tingkat Keparahan Obesitas - Fuzzy Mamdani",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
# Cek apakah variabel 'bmi_ow_thresh' sudah ada di memori sesi atau belum. Kalau belum ada, buat dulu dengan nilai awal 25.0 (batas BMI untuk kategori Overweight)
if "bmi_ow_thresh" not in st.session_state:
    st.session_state.bmi_ow_thresh = 25.0

# Cek variabel beberapa variabel bobot kriteria. Kalau belum ada, buat dengan nilai awal 1.0 (artinya tidak ada penambahan/pengurangan bobot)
if "faf_weight" not in st.session_state:
    st.session_state.faf_weight = 1.0
if "fcvc_weight" not in st.session_state:
    st.session_state.fcvc_weight = 1.0
if "ch2o_weight" not in st.session_state:
    st.session_state.ch2o_weight = 1.0
if "ncp_weight" not in st.session_state:
    st.session_state.ncp_weight = 1.0

# Cek variabel 'defuzz_method' (metode defuzzifikasi yang dipilih user). Kalau belum ada, set default ke 'Centroid' (metode paling umum)
# Huruf besar penting karena nanti dicek dengan opsi ["Centroid", "Bisector", "MOM"]
if "defuzz_method" not in st.session_state:
    st.session_state.defuzz_method = "Centroid"

# Cek variabel 'top_n' (berapa banyak data teratas yang di-highlight di tabel hasil). Kalau belum ada, default 10 (artinya 10 data paling parah akan di-highlight)
if "top_n" not in st.session_state:
    st.session_state.top_n = 10

# Cek variabel 'df_data' (tempat menyimpan dataset yang sudah diupload user). Kalau belum ada, isi dengan None (kosong) dulu.
# Ini penting agar halaman tidak error saat dataset belum diupload
if "df_data" not in st.session_state:
    st.session_state.df_data = None  # Fix memori dataset agar tidak FileNotFoundError


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
            df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

        # Pre-processing: Hitung BMI (Weight / Height^2)
        # Cek apakah kolom 'BMI' sudah ada di dataset atau belum.
        if "BMI" not in df.columns:
            df["BMI"] = df["Weight"] / (df["Height"] ** 2)
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
    bmi = ctrl.Antecedent(np.arange(0, 61, 0.1), "bmi")
    faf = ctrl.Antecedent(np.arange(0, 3.1, 0.1), "faf")
    fcvc = ctrl.Antecedent(np.arange(1, 3.1, 0.1), "fcvc")
    ch2o = ctrl.Antecedent(np.arange(1, 3.1, 0.1), "ch2o")
    ncp = ctrl.Antecedent(np.arange(1, 4.1, 0.1), "ncp")

    score = ctrl.Consequent(
        np.arange(0, 101, 1), "score", defuzzify_method=defuzz_method.lower()
    )

    # 2. Fungsi Keanggotaan
    bmi["underweight"] = fuzz.trapmf(bmi.universe, [0, 0, 18.0, 19.5])
    bmi["normal"] = fuzz.trimf(bmi.universe, [18.0, 22.0, bmi_thresh])
    bmi["overweight"] = fuzz.trimf(bmi.universe, [23.0, bmi_thresh, 30.0])
    bmi["obese"] = fuzz.trapmf(bmi.universe, [bmi_thresh + 3, 32.0, 60.0, 60.0])

    faf["low"] = fuzz.trapmf(faf.universe, [0, 0, 1.0, 1.5])
    faf["medium"] = fuzz.trimf(faf.universe, [1.0, 1.5, 2.5])
    faf["high"] = fuzz.trapmf(faf.universe, [2.0, 2.5, 3.0, 3.0])

    fcvc["low"] = fuzz.trapmf(fcvc.universe, [1.0, 1.0, 1.5, 2.0])
    fcvc["medium"] = fuzz.trimf(fcvc.universe, [1.5, 2.0, 2.5])
    fcvc["high"] = fuzz.trapmf(fcvc.universe, [2.0, 2.5, 3.0, 3.0])

    ch2o["low"] = fuzz.trapmf(ch2o.universe, [1.0, 1.0, 1.5, 2.0])
    ch2o["medium"] = fuzz.trimf(ch2o.universe, [1.5, 2.0, 2.5])
    ch2o["high"] = fuzz.trapmf(ch2o.universe, [2.0, 2.5, 3.0, 3.0])

    ncp["low"] = fuzz.trapmf(ncp.universe, [1.0, 1.0, 2.0, 2.5])
    ncp["medium"] = fuzz.trimf(ncp.universe, [2.0, 2.5, 3.5])
    ncp["high"] = fuzz.trapmf(ncp.universe, [3.0, 3.5, 4.0, 4.0])

    score["rendah"] = fuzz.trapmf(score.universe, [0, 0, 20, 30])
    score["sedang"] = fuzz.trimf(score.universe, [20, 40, 60])
    score["tinggi"] = fuzz.trimf(score.universe, [50, 65, 80])
    score["sangat_tinggi"] = fuzz.trapmf(score.universe, [70, 85, 100, 100])

    # ============================================================
    # UNDERWEIGHT: 81 rules → 1 rule
    # Semua kombinasi faf/fcvc/ch2o/ncp → rendah, jadi tidak perlu dicantumkan
    # ============================================================
    rule_uw = ctrl.Rule(bmi["underweight"], score["rendah"])

    # ============================================================
    # NORMAL: 81 rules → 15 rules
    # Prinsip: faf=low → selalu sedang (27 rules jadi 1)
    #          faf=med, fcvc=low → selalu sedang (9 rules jadi 1)
    #          dst.
    # ============================================================

    # faf=low: semua 27 kombinasi fcvc/ch2o/ncp → sedang → 1 rule
    rule_nm_faf_low = ctrl.Rule(
        bmi["normal"] & faf["low"], score["sedang"]
    )

    # faf=medium, fcvc=low: semua 9 kombinasi → sedang → 1 rule
    rule_nm_med_low = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["low"], score["sedang"]
    )

    # faf=medium, fcvc=medium, ch2o=low: semua ncp → sedang → 1 rule
    rule_nm_med_med_ch2olow = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["medium"] & ch2o["low"], score["sedang"]
    )

    # faf=medium, fcvc=medium, ch2o=medium: ncp=low/med → rendah, ncp=high → sedang
    rule_nm_med_med_ch2omed_ncplow = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["medium"] & ch2o["medium"] & ncp["low"],
        score["rendah"]
    )
    rule_nm_med_med_ch2omed_ncpmed = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["medium"] & ch2o["medium"] & ncp["medium"],
        score["rendah"]
    )
    rule_nm_med_med_ch2omed_ncphigh = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["medium"] & ch2o["medium"] & ncp["high"],
        score["sedang"]
    )

    # faf=medium, fcvc=medium, ch2o=high: ncp=low/med → rendah, ncp=high → sedang
    rule_nm_med_med_ch2ohigh_ncplow = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["medium"] & ch2o["high"] & ncp["low"],
        score["rendah"]
    )
    rule_nm_med_med_ch2ohigh_ncpmed = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["medium"] & ch2o["high"] & ncp["medium"],
        score["rendah"]
    )
    rule_nm_med_med_ch2ohigh_ncphigh = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["medium"] & ch2o["high"] & ncp["high"],
        score["sedang"]
    )

    # faf=medium, fcvc=high, ch2o=low: semua ncp → sedang → 1 rule
    rule_nm_med_high_ch2olow = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["high"] & ch2o["low"], score["sedang"]
    )

    # faf=medium, fcvc=high, ch2o=medium/high: ncp=low/med → rendah, ncp=high → sedang
    rule_nm_med_high_ch2omed_ncplow = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["high"] & ch2o["medium"] & ncp["low"],
        score["rendah"]
    )
    rule_nm_med_high_ch2omed_ncpmed = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["high"] & ch2o["medium"] & ncp["medium"],
        score["rendah"]
    )
    rule_nm_med_high_ncp_high = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["high"] & ncp["high"], score["sedang"]
    )
    rule_nm_med_high_ch2ohigh_ncplow = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["high"] & ch2o["high"] & ncp["low"],
        score["rendah"]
    )
    rule_nm_med_high_ch2ohigh_ncpmed = ctrl.Rule(
        bmi["normal"] & faf["medium"] & fcvc["high"] & ch2o["high"] & ncp["medium"],
        score["rendah"]
    )

    # faf=high, fcvc=low, ch2o=low: semua ncp → sedang → 1 rule
    rule_nm_high_low_ch2olow = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["low"] & ch2o["low"], score["sedang"]
    )

    # faf=high, fcvc=low, ch2o=medium/high: ncp=low/med → rendah, ncp=high → sedang
    rule_nm_high_low_ch2omed_ncplow = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["low"] & ch2o["medium"] & ncp["low"],
        score["rendah"]
    )
    rule_nm_high_low_ch2omed_ncpmed = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["low"] & ch2o["medium"] & ncp["medium"],
        score["rendah"]
    )
    rule_nm_high_low_ch2o_med_high_ncphigh = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["low"] & ncp["high"], score["sedang"]
    )
    rule_nm_high_low_ch2ohigh_ncplow = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["low"] & ch2o["high"] & ncp["low"],
        score["rendah"]
    )
    rule_nm_high_low_ch2ohigh_ncpmed = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["low"] & ch2o["high"] & ncp["medium"],
        score["rendah"]
    )

    # faf=high, fcvc=medium/high: mayoritas rendah kecuali ncp=high + ch2o=low → sedang
    rule_nm_high_med_ch2olow_ncphigh = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["medium"] & ch2o["low"] & ncp["high"],
        score["sedang"]
    )
    rule_nm_high_med_ch2olow_ncp_low_med = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["medium"] & ch2o["low"] & (ncp["low"] | ncp["medium"]),
        score["rendah"]
    )
    # faf=high, fcvc=medium, ch2o=medium/high: semua ncp → rendah → gabung
    rule_nm_high_med_ch2o_med_high = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["medium"] & (ch2o["medium"] | ch2o["high"]),
        score["rendah"]
    )
    # faf=high, fcvc=high: semua → rendah → 1 rule
    rule_nm_high_high = ctrl.Rule(
        bmi["normal"] & faf["high"] & fcvc["high"], score["rendah"]
    )

    # ============================================================
    # OVERWEIGHT: 81 rules → ~13 rules
    # ============================================================

    # faf=low, fcvc=low: semua → tinggi → 1 rule
    rule_ow_faf_low_fcvc_low = ctrl.Rule(
        bmi["overweight"] & faf["low"] & fcvc["low"], score["tinggi"]
    )

    # faf=low, fcvc=medium: ncp=high → tinggi, sisanya sedang
    rule_ow_faf_low_fcvc_med_ncphigh = ctrl.Rule(
        bmi["overweight"] & faf["low"] & fcvc["medium"] & ncp["high"], score["tinggi"]
    )
    rule_ow_faf_low_fcvc_med_ncp_low_med = ctrl.Rule(
        bmi["overweight"] & faf["low"] & fcvc["medium"] & (ncp["low"] | ncp["medium"]),
        score["sedang"]
    )

    # faf=low, fcvc=high: semua → sedang → 1 rule
    rule_ow_faf_low_fcvc_high = ctrl.Rule(
        bmi["overweight"] & faf["low"] & fcvc["high"], score["sedang"]
    )

    # faf=medium, fcvc=low: ncp=high + ch2o=low → tinggi, sisanya sedang
    rule_ow_faf_med_fcvc_low_ncphigh_ch2olow = ctrl.Rule(
        bmi["overweight"] & faf["medium"] & fcvc["low"] & ch2o["low"] & ncp["high"],
        score["tinggi"]
    )
    rule_ow_faf_med_fcvc_low_sisanya = ctrl.Rule(
        bmi["overweight"] & faf["medium"] & fcvc["low"] & (ch2o["medium"] | ch2o["high"]),
        score["sedang"]
    )
    rule_ow_faf_med_fcvc_low_ch2olow_ncp_low_med = ctrl.Rule(
        bmi["overweight"] & faf["medium"] & fcvc["low"] & ch2o["low"] & (ncp["low"] | ncp["medium"]),
        score["sedang"]
    )

    # faf=medium, fcvc=medium/high: semua → sedang → 2 rules
    rule_ow_faf_med_fcvc_med = ctrl.Rule(
        bmi["overweight"] & faf["medium"] & fcvc["medium"], score["sedang"]
    )
    rule_ow_faf_med_fcvc_high = ctrl.Rule(
        bmi["overweight"] & faf["medium"] & fcvc["high"], score["sedang"]
    )

    # faf=high: semua → sedang → 1 rule
    rule_ow_faf_high = ctrl.Rule(
        bmi["overweight"] & faf["high"], score["sedang"]
    )

    # ============================================================
    # OBESE: 81 rules → ~17 rules
    # ============================================================

    # faf=low: semua → sangat_tinggi → 1 rule
    rule_ob_faf_low = ctrl.Rule(
        bmi["obese"] & faf["low"], score["sangat_tinggi"]
    )

    # faf=medium, fcvc=low, ch2o=low/medium: semua → sangat_tinggi
    rule_ob_med_low_ch2o_low_med = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["low"] & (ch2o["low"] | ch2o["medium"]),
        score["sangat_tinggi"]
    )
    # faf=medium, fcvc=low, ch2o=high: ncp=low → tinggi, ncp=med/high → sangat_tinggi
    rule_ob_med_low_ch2ohigh_ncplow = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["low"] & ch2o["high"] & ncp["low"],
        score["tinggi"]
    )
    rule_ob_med_low_ch2ohigh_ncp_med_high = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["low"] & ch2o["high"] & (ncp["medium"] | ncp["high"]),
        score["sangat_tinggi"]
    )

    # faf=medium, fcvc=medium, ch2o=low: semua → sangat_tinggi
    rule_ob_med_med_ch2olow = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["medium"] & ch2o["low"], score["sangat_tinggi"]
    )
    # faf=medium, fcvc=medium, ch2o=medium: ncp=low → tinggi, sisanya sangat_tinggi
    rule_ob_med_med_ch2omed_ncplow = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["medium"] & ch2o["medium"] & ncp["low"],
        score["tinggi"]
    )
    rule_ob_med_med_ch2omed_ncp_med_high = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["medium"] & ch2o["medium"] & (ncp["medium"] | ncp["high"]),
        score["sangat_tinggi"]
    )
    # faf=medium, fcvc=medium, ch2o=high: ncp=low/med → tinggi, ncp=high → sangat_tinggi
    rule_ob_med_med_ch2ohigh_ncp_low_med = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["medium"] & ch2o["high"] & (ncp["low"] | ncp["medium"]),
        score["tinggi"]
    )
    rule_ob_med_med_ch2ohigh_ncphigh = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["medium"] & ch2o["high"] & ncp["high"],
        score["sangat_tinggi"]
    )

    # faf=medium, fcvc=high, ch2o=low: ncp=low → tinggi, ncp=med/high → sangat_tinggi
    rule_ob_med_high_ch2olow_ncplow = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["high"] & ch2o["low"] & ncp["low"], score["tinggi"]
    )
    rule_ob_med_high_ch2olow_ncp_med_high = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["high"] & ch2o["low"] & (ncp["medium"] | ncp["high"]),
        score["sangat_tinggi"]
    )
    # faf=medium, fcvc=high, ch2o=medium/high: ncp=low/med → tinggi, ncp=high → sangat_tinggi
    rule_ob_med_high_ch2o_med_high_ncp_low_med = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["high"] & (ch2o["medium"] | ch2o["high"]) & (ncp["low"] | ncp["medium"]),
        score["tinggi"]
    )
    rule_ob_med_high_ch2o_med_high_ncphigh = ctrl.Rule(
        bmi["obese"] & faf["medium"] & fcvc["high"] & (ch2o["medium"] | ch2o["high"]) & ncp["high"],
        score["sangat_tinggi"]
    )

    # faf=high, fcvc=low, ch2o=low: semua ncp → sangat_tinggi
    rule_ob_high_low_ch2olow = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["low"] & ch2o["low"], score["sangat_tinggi"]
    )
    # faf=high, fcvc=low, ch2o=medium: ncp=low → tinggi, ncp=med/high → sangat_tinggi
    rule_ob_high_low_ch2omed_ncplow = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["low"] & ch2o["medium"] & ncp["low"], score["tinggi"]
    )
    rule_ob_high_low_ch2omed_ncp_med_high = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["low"] & ch2o["medium"] & (ncp["medium"] | ncp["high"]),
        score["sangat_tinggi"]
    )
    # faf=high, fcvc=low, ch2o=high: ncp=low/med → tinggi, ncp=high → sangat_tinggi
    rule_ob_high_low_ch2ohigh_ncp_low_med = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["low"] & ch2o["high"] & (ncp["low"] | ncp["medium"]),
        score["tinggi"]
    )
    rule_ob_high_low_ch2ohigh_ncphigh = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["low"] & ch2o["high"] & ncp["high"], score["sangat_tinggi"]
    )
    # faf=high, fcvc=medium: pola ncp dominan
    rule_ob_high_med_ch2olow_ncplow = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["medium"] & ch2o["low"] & ncp["low"], score["tinggi"]
    )
    rule_ob_high_med_ch2olow_ncp_med_high = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["medium"] & ch2o["low"] & (ncp["medium"] | ncp["high"]),
        score["sangat_tinggi"]
    )
    rule_ob_high_med_ch2o_med_high_ncp_low_med = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["medium"] & (ch2o["medium"] | ch2o["high"]) & (ncp["low"] | ncp["medium"]),
        score["tinggi"]
    )
    rule_ob_high_med_ch2o_med_high_ncphigh = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["medium"] & (ch2o["medium"] | ch2o["high"]) & ncp["high"],
        score["sangat_tinggi"]
    )
    # faf=high, fcvc=high: ncp=high → sangat_tinggi, sisanya tinggi
    rule_ob_high_high_ncphigh = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["high"] & ncp["high"], score["sangat_tinggi"]
    )
    rule_ob_high_high_ncp_low_med = ctrl.Rule(
        bmi["obese"] & faf["high"] & fcvc["high"] & (ncp["low"] | ncp["medium"]), score["tinggi"]
    )

    # ============================================================
    # KUMPULKAN SEMUA (~46 rules, output identik)
    # ============================================================
    rules = [
        rule_uw,
        # Normal
        rule_nm_faf_low,
        rule_nm_med_low, rule_nm_med_med_ch2olow,
        rule_nm_med_med_ch2omed_ncplow, rule_nm_med_med_ch2omed_ncpmed,
        rule_nm_med_med_ch2omed_ncphigh,
        rule_nm_med_med_ch2ohigh_ncplow, rule_nm_med_med_ch2ohigh_ncpmed,
        rule_nm_med_med_ch2ohigh_ncphigh,
        rule_nm_med_high_ch2olow,
        rule_nm_med_high_ch2omed_ncplow, rule_nm_med_high_ch2omed_ncpmed,
        rule_nm_med_high_ncp_high,
        rule_nm_med_high_ch2ohigh_ncplow, rule_nm_med_high_ch2ohigh_ncpmed,
        rule_nm_high_low_ch2olow,
        rule_nm_high_low_ch2omed_ncplow, rule_nm_high_low_ch2omed_ncpmed,
        rule_nm_high_low_ch2o_med_high_ncphigh,
        rule_nm_high_low_ch2ohigh_ncplow, rule_nm_high_low_ch2ohigh_ncpmed,
        rule_nm_high_med_ch2olow_ncphigh, rule_nm_high_med_ch2olow_ncp_low_med,
        rule_nm_high_med_ch2o_med_high,
        rule_nm_high_high,
        # Overweight
        rule_ow_faf_low_fcvc_low,
        rule_ow_faf_low_fcvc_med_ncphigh, rule_ow_faf_low_fcvc_med_ncp_low_med,
        rule_ow_faf_low_fcvc_high,
        rule_ow_faf_med_fcvc_low_ncphigh_ch2olow,
        rule_ow_faf_med_fcvc_low_sisanya,
        rule_ow_faf_med_fcvc_low_ch2olow_ncp_low_med,
        rule_ow_faf_med_fcvc_med, rule_ow_faf_med_fcvc_high,
        rule_ow_faf_high,
        # Obese
        rule_ob_faf_low,
        rule_ob_med_low_ch2o_low_med,
        rule_ob_med_low_ch2ohigh_ncplow, rule_ob_med_low_ch2ohigh_ncp_med_high,
        rule_ob_med_med_ch2olow,
        rule_ob_med_med_ch2omed_ncplow, rule_ob_med_med_ch2omed_ncp_med_high,
        rule_ob_med_med_ch2ohigh_ncp_low_med, rule_ob_med_med_ch2ohigh_ncphigh,
        rule_ob_med_high_ch2olow_ncplow, rule_ob_med_high_ch2olow_ncp_med_high,
        rule_ob_med_high_ch2o_med_high_ncp_low_med,
        rule_ob_med_high_ch2o_med_high_ncphigh,
        rule_ob_high_low_ch2olow,
        rule_ob_high_low_ch2omed_ncplow, rule_ob_high_low_ch2omed_ncp_med_high,
        rule_ob_high_low_ch2ohigh_ncp_low_med, rule_ob_high_low_ch2ohigh_ncphigh,
        rule_ob_high_med_ch2olow_ncplow, rule_ob_high_med_ch2olow_ncp_med_high,
        rule_ob_high_med_ch2o_med_high_ncp_low_med, rule_ob_high_med_ch2o_med_high_ncphigh,
        rule_ob_high_high_ncphigh, rule_ob_high_high_ncp_low_med,
    ]

    # SESUDAH (benar)
    obesity_ctrl = ctrl.ControlSystem(rules)
    obesity_sim = ctrl.ControlSystemSimulation(obesity_ctrl)
    return obesity_sim, bmi, faf, fcvc, ch2o, ncp, score


def hitung_per_baris(row_data, master_ctrl, weights):
    # CUKUP BIKIN SIMULATORNYA AJA, JANGAN BIKIN RULE DARI AWAL
    sim = ctrl.ControlSystemSimulation(master_ctrl)

    # Ambil bobot dari session state yang dipassing
    faf_w, fcvc_w, ch2o_w, ncp_w = weights

    sim.input["bmi"] = row_data["BMI"]
    sim.input["faf"] = np.clip(row_data["FAF"] * faf_w, 0, 3)
    sim.input["fcvc"] = np.clip(row_data["FCVC"] * fcvc_w, 1, 3)
    sim.input["ch2o"] = np.clip(row_data["CH2O"] * ch2o_w, 1, 3)
    sim.input["ncp"] = np.clip(row_data["NCP"] * ncp_w, 1, 4)

    try:
        sim.compute()
        return sim.output["score"]
    except:
        return 0.0

# ==========================================
# FUNGSI HELPER
# ==========================================
# Helper untuk mendapatkan kategori teks dari nilai crisp
def tingkat_keparahan(crisp_val):
    if crisp_val <= 30:
        return "Rendah"
    elif crisp_val <= 60:
        return "Sedang"
    elif crisp_val <= 80:
        return "Tinggi"
    else:
        return "Sangat Tinggi"


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigasi SPK")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "📊 Dataset",
        "⚙️ Konfigurasi Fuzzy",
        "🏆 Hitung & Peringkat SPK",
        "👥 Tentang Program & Kelompok",
    ],
)

st.sidebar.markdown("---")

# ==========================================
# HALAMAN 1: DATASET
# ==========================================
if menu == "📊 Dataset":
    st.title("📊 Dataset")

    st.markdown("""Silakan upload file CSV dataset.""")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    # Update state dataset jika file diupload
    if uploaded_file is not None:
        st.session_state.df_data = load_data(uploaded_file)
    elif st.session_state.df_data is None:
        # Coba load lokal sekali saja jika memori masih kosong
        st.session_state.df_data = load_data()
    df = st.session_state.df_data

    if df is not None:
        st.success(
            f"Dataset berhasil dimuat! Jumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}"
        )

        st.subheader("Dataset Mentah")
        # Tabel keterangan kolom supaya user paham arti setiap kolom
        with st.expander("📖 Keterangan Kolom Dataset (klik untuk buka)"):
            keterangan = {
                "Kolom": [
                    "Gender",
                    "Age",
                    "Height",
                    "Weight",
                    "family_history_with_overweight",
                    "FAVC",
                    "FCVC",
                    "NCP",
                    "CAEC",
                    "SMOKE",
                    "CH2O",
                    "SCC",
                    "FAF",
                    "TUE",
                    "CALC",
                    "MTRANS",
                    "NObeyesdad",
                ],
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
                    "Label tingkat obesitas",
                ],
            }
            st.table(pd.DataFrame(keterangan))
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistik Deskriptif")
            with st.expander("📖 Penjelasan Indikator Statistik (klik untuk buka)"):
                keteranganStatistik = {
                    "Indikator": [
                        "count",
                        "mean",
                        "std",
                        "min",
                        "25%",
                        "50%",
                        "75%",
                        "max",
                    ],
                    "Penjelasan": [
                        "Jumlah data yang tersedia (tidak termasuk yang kosong/NaN)",
                        "Nilai rata-rata dari seluruh data",
                        "Standar deviasi, seberapa jauh data menyebar dari rata-rata",
                        "Nilai terkecil dalam kolom",
                        "Kuartil bawah, 25% data berada di bawah nilai ini",
                        "Median, nilai tengah, 50% data di bawah dan 50% di atas nilai ini",
                        "Kuartil atas, 75% data berada di bawah nilai ini",
                        "Nilai terbesar dalam kolom",
                    ],
                }
                st.table(pd.DataFrame(keteranganStatistik))
            st.write(df.describe())

        with col2:
            st.subheader("Distribusi Label Tingkat Obesitas")
            st.markdown(
                "*Label ini hanya sebagai referensi (menampilkan data) dan **TIDAK** digunakan sebagai input SPK.*"
            )
            if "NObeyesdad" in df.columns:
                target_counts = df["NObeyesdad"].value_counts()
                st.bar_chart(target_counts)
            else:
                st.warning("Kolom 'NObeyesdad' tidak ditemukan di dataset ini.")
    else:
        st.error(
            "Dataset tidak ditemukan. Silakan upload file CSV melalui uploader di atas."
        )

# ==========================================
# HALAMAN 2: KONFIGURASI FUZZY
# ==========================================
elif menu == "⚙️ Konfigurasi Fuzzy":
    st.title("⚙️ Konfigurasi Parameter Fuzzy Mamdani")

    st.markdown("### Pengaturan Variabel")
    col_w1, col_w2, col_w3 = st.columns(3)

    with col_w1:
        st.session_state.bmi_ow_thresh = st.slider(
            "Batas Awal Overweight (BMI Threshold)",
            min_value=23.0,
            max_value=28.0,
            value=st.session_state.bmi_ow_thresh,
            step=0.5,
        )
        st.caption(
            "📏 Batas nilai BMI seseorang mulai dianggap kelebihan berat badan. Standar Asia = 23.0, Standar WHO = 25.0."
        )
        st.session_state.faf_weight = st.slider(
            "Bobot Aktivitas Fisik (FAF)",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.faf_weight,
            step=0.1,
        )
        st.caption(
            "🏃 Seberapa besar pengaruh aktivitas fisik terhadap hasil penilaian."
        )

    with col_w2:
        st.session_state.ch2o_weight = st.slider(
            "Bobot Konsumsi Air (CH2O)",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.ch2o_weight,
            step=0.1,
        )
        st.caption(
            "💧 Seberapa besar pengaruh kebiasaan minum air putih terhadap hasil penilaian."
        )
        st.session_state.ncp_weight = st.slider(
            "Bobot Frekuensi Makan (NCP)",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.ncp_weight,
            step=0.1,
        )
        st.caption(
            "🍽️ Seberapa besar pengaruh jumlah makan utama per hari terhadap hasil penilaian."
        )

    with col_w3:
        st.session_state.fcvc_weight = st.slider(
            "Bobot Konsumsi Sayur (FCVC)",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.fcvc_weight,
            step=0.1,
        )
        st.caption(
            "🥦 Seberapa besar pengaruh kebiasaan makan sayur terhadap hasil penilaian."
        )

        # ERROR HANDLING ERROR SELECTBOX: Pengecekan aman huruf besar/kecil
        opsi_defuzz = ["Centroid", "Bisector", "MOM"]
        default_idx = (
            opsi_defuzz.index(st.session_state.defuzz_method)
            if st.session_state.defuzz_method in opsi_defuzz
            else 0
        )
        st.session_state.defuzz_method = st.selectbox(
            "Metode Defuzzifikasi", options=opsi_defuzz, index=default_idx
        )

        st.session_state.top_n = st.number_input(
            "Jumlah Top-N Hasil Peringkat",
            min_value=5,
            max_value=50,
            value=st.session_state.top_n,
            step=5,
        )

    st.markdown("---")

    # Visualisasi grafik SEMENTARA
    sim, bmi, faf, fcvc, ch2o, ncp, score = fuzzy_system(
        st.session_state.bmi_ow_thresh, st.session_state.defuzz_method
    )

    st.subheader("📈 Kurva Fungsi Keanggotaan")
    viz_option = st.selectbox(
        "Pilih Variabel untuk Divisualisasikan:",
        ["BMI", "FAF", "FCVC", "CH2O", "NCP", "Output Skor"],
    )

    # FIX ERROR GRAFIK BLANK KOSONG
    plt.close("all")  # Bersihkan memori kanvas sebelumnya

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
        "Kelompok": [
            "Underweight", "Normal", "Normal", "Normal", "Normal",
            "Overweight", "Overweight", "Overweight", "Overweight",
            "Obese", "Obese", "Obese", "Obese", "Obese", "Umum"
        ],
        "Kondisi (IF)": [
            "BMI is Underweight (Apapun kebiasaan makan & olahraga)",
            "BMI is Normal AND FAF is Low (Jarang Olahraga)",
            "BMI is Normal AND FAF is Medium AND (FCVC/CH2O is Medium/High)",
            "BMI is Normal AND FAF is High (Sering Olahraga)",
            "BMI is Normal AND (NCP is High AND CH2O is Low)",
            "BMI is Overweight AND FAF is Low AND FCVC is Low",
            "BMI is Overweight AND FAF is Medium/High",
            "BMI is Overweight AND FAF is Low AND FCVC is High",
            "BMI is Overweight AND (NCP is High AND FAF is Low)",
            "BMI is Obese AND FAF is Low (Faktor Risiko Tertinggi)",
            "BMI is Obese AND FAF is Medium AND (FCVC/CH2O is Low)",
            "BMI is Obese AND FAF is High AND (FCVC/CH2O is High)",
            "BMI is Obese AND NCP is High",
            "BMI is Obese AND FAF is Medium AND (FCVC/CH2O is High)",
            "Semua Kondisi: Gaya Hidup Sangat Sehat (FAF, FCVC, CH2O High)"
        ],
        "Keputusan (THEN) Skor": [
            "Rendah",
            "Sedang",
            "Rendah",
            "Rendah",
            "Sedang",
            "Tinggi",
            "Sedang",
            "Sedang",
            "Tinggi",
            "Sangat Tinggi",
            "Sangat Tinggi",
            "Tinggi",
            "Sangat Tinggi",
            "Tinggi",
            "Rendah"
        ]
    }

    # Untuk menampilkan di Streamlit:
    st.subheader("📚 Ringkasan Rule Base (Representasi 243 Aturan)")
    st.table(pd.DataFrame(rules_data))
    st.table(pd.DataFrame(rules_data))

# ==========================================
# HALAMAN 3: HITUNG & PERINGKAT SPK
# ==========================================
# ==========================================
# HALAMAN 3: HITUNG & PERINGKAT SPK
# ==========================================
# ==========================================
# HALAMAN 3: HITUNG & PERINGKAT SPK
# ==========================================
elif menu == "🏆 Hitung & Peringkat SPK":
    st.title("🏆 Perhitungan SPK (Mode Cepat)")

    df = st.session_state.df_data

    if df is None:
        st.warning(
            "Dataset belum diload! Silakan ke halaman 'Dataset' dan Upload file CSV terlebih dahulu."
        )
    else:
        if st.button("🚀 Hitung Cepat SPK", use_container_width=True, type="primary"):
            t_mulai = time.time()

            # 1. Siapkan Parameter
            weights = (
                st.session_state.faf_weight,
                st.session_state.fcvc_weight,
                st.session_state.ch2o_weight,
                st.session_state.ncp_weight,
            )
            data_rows = df.to_dict("records")

            bmi_thresh_val = st.session_state.bmi_ow_thresh
            defuzz_val = st.session_state.defuzz_method

            # --- Bikin "Otak" Fuzzy SEKALI AJA di luar loop ---
            master_sim, _, _, _, _, _, _ = fuzzy_system(bmi_thresh_val, defuzz_val)
            master_ctrl = master_sim.ctrl

            # 2. Proses Eksekusi (Pakai Looping Biasa, Tanpa Joblib yang bikin stuck)
            skor_hasil = []
            total_data = len(data_rows)

            # Siapkan UI untuk Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, row in enumerate(data_rows):
                # Cukup bikin mesin eksekusinya aja per baris
                sim = ctrl.ControlSystemSimulation(master_ctrl)

                sim.input["bmi"] = row["BMI"]
                sim.input["faf"] = np.clip(row["FAF"] * weights[0], 0, 3)
                sim.input["fcvc"] = np.clip(row["FCVC"] * weights[1], 1, 3)
                sim.input["ch2o"] = np.clip(row["CH2O"] * weights[2], 1, 3)
                sim.input["ncp"] = np.clip(row["NCP"] * weights[3], 1, 4)

                try:
                    sim.compute()
                    skor_hasil.append(sim.output["score"])
                except:
                    skor_hasil.append(0.0)

                # Update progress bar per 100 baris agar Streamlit nggak ngelag render UI
                if i % 100 == 0 or i == total_data - 1:
                    progress_bar.progress((i + 1) / total_data)
                    status_text.text(f"Memproses data {i + 1} dari {total_data}...")

            t_selesai = time.time()
            status_text.text(f"✅ Selesai! Waktu eksekusi: {t_selesai - t_mulai:.2f} detik")

            # 3. Gabungkan Hasil
            res_df = df.copy()
            res_df["Skor_Fuzzy"] = [round(s, 2) for s in skor_hasil]
            res_df["Kategori_Keparahan"] = [tingkat_keparahan(s) for s in skor_hasil]

            # --- TAMPILKAN TABEL HASIL ---
            st.subheader("🏅 Hasil Peringkat Akhir")
            res_df = res_df.sort_values(by="Skor_Fuzzy", ascending=False).reset_index(drop=True)
            res_df.index = res_df.index + 1

            top_n = st.session_state.top_n

            def highlight_top(s, n):
                return [
                    "background-color: #f79525; color: black" if i < n else ""
                    for i in range(len(s))
                ]

            st.dataframe(
                res_df.style.apply(highlight_top, n=top_n, axis=0),
                use_container_width=True,
            )

            # --- TAMPILKAN GRAFIK ---
            col_pie, col_bar = st.columns(2)
            kategori_counts = (
                res_df["Kategori_Keparahan"]
                .value_counts()
                .reindex(["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"], fill_value=0)
            )
            with col_pie:
                fig1, ax1 = plt.subplots()
                ax1.pie(
                    kategori_counts,
                    labels=kategori_counts.index,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                st.pyplot(fig1)
            with col_bar:
                fig2, ax2 = plt.subplots()
                ax2.bar(kategori_counts.index, kategori_counts.values)
                st.pyplot(fig2)

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
    st.write(
        "- **Nama Dataset:** Obesity Levels (Prediction of Obesity Levels Based On Eating Habits and Physical Activites)"
    )
    st.write("- **Author Dataset:** Fatemeh Mehrparvar")
    st.link_button(
        "Kunjungi Link",
        "https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels/data",
    )

    st.markdown("---")

    st.markdown("### Informasi Metode SPK")
    st.write(
        "- **Tujuan:** Menentukan Tingkat Keparahan Obesitas untuk Penentuan Prioritas Penanganan"
    )
    st.write("- Sistem mengimplementasikan Algoritma **Fuzzy Mamdani**")
    st.write(
        "- **Fungsi Keanggotaan** menggunakan kombinasi Kurva Segitiga (`trimf`) dan Trapesium (`trapmf`)"
    )
    st.write(
        "- Defuzzifikasi berbasis perhitungan area (mendukung *Centroid, Bisector, MOM*)"
    )

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
