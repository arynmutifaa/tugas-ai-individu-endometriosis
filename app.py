import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Endometriosis (DNN)", page_icon="🩺")

# ---------- 1. Muat model & informasi model (sekali saja) ----------
@st.cache_resource
def load_model():
    return joblib.load("model_endometriosis.pkl")

@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)

model = load_model()
info = load_info()
FITUR = info["fitur_numerik"] + info["fitur_kategori"]
rentang = info["rentang_fitur"]

# ---------- 2. Sidebar: informasi model ----------
with st.sidebar:
    st.header("Tentang Model")
    st.write(info["nama_model"])
    st.metric("Accuracy (data uji)", f"{info['metrik']['accuracy']:.1%}")
    st.metric("F1 macro (data uji)", f"{info['metrik']['f1_macro']:.3f}")
    st.caption(f"Dievaluasi pada {info['metrik']['n_test']} data uji · "
               f"scikit-learn {info['versi']['scikit-learn']}")

st.title("🩺 Prediksi Endometriosis (DNN)")
st.write("Masukkan data pasien untuk memprediksi: **Normal** atau **Endometriosis**.")
st.caption("⚠️ Aplikasi ini hanya untuk keperluan pembelajaran, bukan alat diagnosis medis.")

tab1, tab2 = st.tabs(["Prediksi Satu Pasien", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            usia = st.number_input("Usia (tahun)", min_value=15, max_value=60, value=30, step=1)
            nyeri = st.slider("Tingkat nyeri kronis (0-10)", 0.0, 10.0, 5.0, 0.1)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=23.0, step=0.1)
        with col2:
            menstruasi = st.radio("Menstruasi tidak teratur?", ["Ya", "Tidak"], horizontal=True)
            hormon = st.radio("Kadar hormon abnormal?", ["Ya", "Tidak"], horizontal=True)
            infertil = st.radio("Infertilitas?", ["Ya", "Tidak"], horizontal=True)
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        ya = lambda s: 1 if s == "Ya" else 0
        data = pd.DataFrame([{
            "Age": usia, "Chronic_Pain_Level": nyeri, "BMI": bmi,
            "Menstrual_Irregularity": ya(menstruasi),
            "Hormone_Level_Abnormality": ya(hormon),
            "Infertility": ya(infertil),
        }])[FITUR]

        luar = [c for c in rentang if not rentang[c][0] <= data[c].iloc[0] <= rentang[c][1]]
        if luar:
            st.warning(f"Nilai {', '.join(luar)} di luar rentang data latih; "
                       "hasil prediksi bisa kurang akurat.")

        hasil = model.predict(data)[0]
        proba = pd.Series(model.predict_proba(data)[0], index=model.classes_)

        warna = {"Normal": st.success, "Endometriosis": st.warning}
        warna[hasil](f"Prediksi: **{hasil}** (keyakinan {proba[hasil]:.0%})")
        st.bar_chart(proba.rename("probabilitas"))

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah file CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR))

    try:
        with open("contoh_data_baru.csv", "rb") as f:
            st.download_button("⬇️ Unduh contoh data (CSV)", f,
                               file_name="contoh_data_baru.csv", mime="text/csv")
    except FileNotFoundError:
        st.info("File contoh_data_baru.csv belum tersedia.")

    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df["prediksi"] = model.predict(df[FITUR])
            st.dataframe(df)
            st.write(df["prediksi"].value_counts())
            st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                               file_name="hasil_prediksi.csv", mime="text/csv")