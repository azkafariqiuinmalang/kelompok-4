import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* ==============================
   GLOBAL
============================== */
.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ==============================
   HEADER
============================== */
.hero {
    background: linear-gradient(135deg, #1e4f7a, #3479b9);
    padding: 34px 40px;
    border-radius: 0 0 24px 24px;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(30, 79, 122, 0.15);
}

.hero-title {
    color: white !important;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-description {
    color: #eaf4ff !important;
    font-size: 16px;
    margin: 0;
}

/* ==============================
   INFORMATION CARDS
============================== */
.info-card {
    background: white;
    padding: 22px 24px;
    border-radius: 16px;
    border: 1px solid #e4e8ef;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.045);
    min-height: 115px;
}

.info-label {
    font-size: 14px;
    color: #64748b !important;
    margin-bottom: 8px;
    font-weight: 500;
}

.info-value {
    font-size: 27px;
    color: #172033 !important;
    font-weight: 700;
}

/* ==============================
   SECTION
============================== */
.section-title {
    color: #172033 !important;
    font-size: 24px;
    font-weight: 750;
    margin-top: 22px;
    margin-bottom: 4px;
}

.section-description {
    color: #697386 !important;
    font-size: 15px;
    margin-bottom: 22px;
}

/* ==============================
   WIDGET LABEL
============================== */
label[data-testid="stWidgetLabel"] p {
    color: #253247 !important;
    font-weight: 650 !important;
    font-size: 15px !important;
}

[data-testid="stSlider"] p {
    color: #253247 !important;
}

/* Slider value */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #2468a2 !important;
}

/* ==============================
   INPUT CONTAINER
============================== */
[data-testid="stForm"] {
    background: white;
    padding: 26px 28px;
    border-radius: 20px;
    border: 1px solid #e3e8ef;
    box-shadow: 0 7px 22px rgba(0, 0, 0, 0.045);
}

/* ==============================
   BUTTON
============================== */
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 52px;
    background: linear-gradient(135deg, #1e5f99, #2e7dbe);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    transition: 0.2s;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #174d7d, #266da7);
    color: white !important;
    border: none;
    transform: translateY(-1px);
}

/* ==============================
   RESULTS
============================== */
.result-success {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border: 1px solid #a7f3d0;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 16px;
}

.result-success-title {
    color: #047857 !important;
    font-size: 30px;
    font-weight: 800;
}

.result-danger {
    background: linear-gradient(135deg, #fff1f2, #fef2f2);
    border: 1px solid #fecaca;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 16px;
}

.result-danger-title {
    color: #b42318 !important;
    font-size: 30px;
    font-weight: 800;
}

.result-text {
    color: #475569 !important;
    font-size: 15px;
}

/* ==============================
   PROBABILITY CARDS
============================== */
.prob-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e4e8ef;
    margin-top: 10px;
}

.prob-label {
    color: #64748b !important;
    font-size: 14px;
}

.prob-value {
    color: #172033 !important;
    font-size: 28px;
    font-weight: 750;
}

/* ==============================
   EXPANDER
============================== */
[data-testid="stExpander"] {
    background: white;
    border-radius: 14px;
    border: 1px solid #e5e9f0;
}

/* ==============================
   FOOTER
============================== */
.footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 13px;
    padding-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    scaler_model = joblib.load("scaler.pkl")

    return model, scaler_model


try:
    rf_model, scaler = load_model()

except Exception as e:
    st.error("Model atau scaler gagal dimuat.")
    st.exception(e)
    st.stop()


# =========================================================
# FEATURE COLUMNS
# =========================================================
feature_columns = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score"
]


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">

    <div class="hero-title">
        🎓 Student Placement Prediction
    </div>

    <p class="hero-description">
        Prediksi status penempatan mahasiswa berdasarkan aktivitas
        akademik menggunakan Random Forest Classifier.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INFORMATION CARDS
# =========================================================
card1, card2, card3 = st.columns(3, gap="medium")

with card1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">
            🤖 Machine Learning Model
        </div>

        <div class="info-value">
            Random Forest
        </div>
    </div>
    """, unsafe_allow_html=True)


with card2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">
            📊 Input Features
        </div>

        <div class="info-value">
            6 Features
        </div>
    </div>
    """, unsafe_allow_html=True)


with card3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">
            🎯 Prediction Target
        </div>

        <div class="info-value">
            Placement
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("""
<div class="section-title">
    📝 Student Information
</div>

<div class="section-description">
    Masukkan informasi mahasiswa pada formulir berikut untuk
    mendapatkan hasil prediksi.
</div>
""", unsafe_allow_html=True)


with st.form("prediction_form"):

    left, right = st.columns(2, gap="large")


    # =====================================================
    # LEFT COLUMN
    # =====================================================
    with left:

        study_hours = st.slider(
            "📚 Study Hours (jam/hari)",
            min_value=1.0,
            max_value=11.0,
            value=6.0,
            step=0.5,
            help="Rata-rata waktu yang digunakan mahasiswa untuk belajar setiap hari."
        )

        attendance = st.slider(
            "🏫 Attendance (%)",
            min_value=40.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            help="Persentase kehadiran mahasiswa."
        )

        sleep_hours = st.slider(
            "😴 Sleep Hours (jam/hari)",
            min_value=4.0,
            max_value=9.0,
            value=7.0,
            step=0.5,
            help="Rata-rata durasi tidur mahasiswa setiap hari."
        )


    # =====================================================
    # RIGHT COLUMN
    # =====================================================
    with right:

        internet_usage = st.slider(
            "🌐 Internet Usage (jam/hari)",
            min_value=1.0,
            max_value=11.0,
            value=6.0,
            step=0.5,
            help="Rata-rata waktu penggunaan internet setiap hari."
        )

        assignments_completed = st.slider(
            "✅ Assignments Completed",
            min_value=0,
            max_value=20,
            value=10,
            step=1,
            help="Jumlah tugas yang berhasil diselesaikan."
        )

        previous_score = st.slider(
            "📈 Previous Score",
            min_value=35.0,
            max_value=95.0,
            value=65.0,
            step=1.0,
            help="Nilai akademik mahasiswa sebelumnya."
        )


    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.form_submit_button(
        "🔍 Predict Student Placement",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================
if predict:

    input_data = pd.DataFrame(
        [{
            "study_hours": study_hours,
            "attendance": attendance,
            "sleep_hours": sleep_hours,
            "internet_usage": internet_usage,
            "assignments_completed": assignments_completed,
            "previous_score": previous_score
        }]
    )


    # Ensure correct feature order
    input_data = input_data[feature_columns]


    try:

        # =================================================
        # SCALING
        # =================================================
        input_scaled = pd.DataFrame(
            scaler.transform(input_data),
            columns=feature_columns
        )


        # =================================================
        # PREDICTION
        # =================================================
        prediction = rf_model.predict(input_scaled)


        st.markdown("""
        <div class="section-title">
            🎯 Prediction Result
        </div>

        <div class="section-description">
            Berikut hasil prediksi berdasarkan informasi mahasiswa.
        </div>
        """, unsafe_allow_html=True)


        # =================================================
        # RESULT CARD
        # =================================================
        if prediction[0] == 1:

            st.markdown("""
            <div class="result-success">

                <div style="font-size:48px">
                    🎉
                </div>

                <div class="result-success-title">
                    PLACED
                </div>

                <div class="result-text">
                    Model memprediksi mahasiswa memiliki
                    status <b>Placed</b>.
                </div>

            </div>
            """, unsafe_allow_html=True)


        else:

            st.markdown("""
            <div class="result-danger">

                <div style="font-size:48px">
                    📌
                </div>

                <div class="result-danger-title">
                    NOT PLACED
                </div>

                <div class="result-text">
                    Model memprediksi mahasiswa memiliki
                    status <b>Not Placed</b>.
                </div>

            </div>
            """, unsafe_allow_html=True)


        # =================================================
        # PREDICTION PROBABILITY
        # =================================================
        if hasattr(rf_model, "predict_proba"):

            probabilities = rf_model.predict_proba(input_scaled)[0]

            class_probability = dict(
                zip(
                    rf_model.classes_,
                    probabilities
                )
            )

            not_placed_probability = (
                class_probability.get(0, 0) * 100
            )

            placed_probability = (
                class_probability.get(1, 0) * 100
            )


            st.markdown("<br>", unsafe_allow_html=True)

            probability1, probability2 = st.columns(2)


            with probability1:

                st.markdown(
                    f"""
                    <div class="prob-card">

                        <div class="prob-label">
                            📌 Probability — Not Placed
                        </div>

                        <div class="prob-value">
                            {not_placed_probability:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with probability2:

                st.markdown(
                    f"""
                    <div class="prob-card">

                        <div class="prob-label">
                            🎓 Probability — Placed
                        </div>

                        <div class="prob-value">
                            {placed_probability:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown("<br>", unsafe_allow_html=True)

            st.progress(
                placed_probability / 100,
                text=(
                    f"Placement Probability "
                    f"{placed_probability:.2f}%"
                )
            )


        # =================================================
        # INPUT SUMMARY
        # =================================================
        with st.expander(
            "📋 Lihat Detail Data Input",
            expanded=False
        ):

            summary = pd.DataFrame({

                "Feature": [
                    "Study Hours",
                    "Attendance",
                    "Sleep Hours",
                    "Internet Usage",
                    "Assignments Completed",
                    "Previous Score"
                ],

                "Value": [
                    f"{study_hours} jam",
                    f"{attendance:.0f}%",
                    f"{sleep_hours} jam",
                    f"{internet_usage} jam",
                    assignments_completed,
                    previous_score
                ]

            })


            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Terjadi kesalahan saat melakukan prediksi."
        )

        st.exception(e)


# =========================================================
# ABOUT APPLICATION
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)


with st.expander(
    "ℹ️ Tentang Sistem Prediksi",
    expanded=False
):

    st.markdown("""
    Aplikasi **Student Placement Prediction** merupakan
    implementasi Machine Learning menggunakan algoritma
    **Random Forest Classifier**.

    Prediksi dilakukan berdasarkan enam variabel:

    - 📚 Study Hours
    - 🏫 Attendance
    - 😴 Sleep Hours
    - 🌐 Internet Usage
    - ✅ Assignments Completed
    - 📈 Previous Score

    Sebelum data digunakan oleh model Random Forest,
    data akan melalui proses **scaling** menggunakan scaler
    yang dibuat pada tahap pelatihan model.
    """)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    🎓 Student Placement Prediction
    &nbsp;•&nbsp;
    Machine Learning Project
</div>
""", unsafe_allow_html=True)
