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
st.markdown(
    """
<style>
/* =========================
   GLOBAL
========================= */

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* =========================
   HERO
========================= */

.hero {
    background: linear-gradient(135deg, #1f5c8f 0%, #347fbd 100%);
    padding: 38px 42px;
    border-radius: 0 0 26px 26px;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(31, 92, 143, 0.16);
}

.hero-title {
    color: #ffffff !important;
    font-size: 38px;
    line-height: 1.2;
    font-weight: 800;
    margin: 0 0 12px 0;
}

.hero-description {
    color: #edf7ff !important;
    font-size: 16px;
    line-height: 1.6;
    margin: 0;
}

/* =========================
   INFO CARDS
========================= */

.info-card {
    background: #ffffff;
    padding: 22px 24px;
    border-radius: 18px;
    border: 1px solid #e1e7ef;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    min-height: 110px;
}

.info-label {
    color: #64748b !important;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
}

.info-value {
    color: #172033 !important;
    font-size: 27px;
    font-weight: 750;
    line-height: 1.2;
}

/* =========================
   SECTION
========================= */

.section-title {
    color: #172033 !important;
    font-size: 26px;
    line-height: 1.3;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 7px;
}

.section-description {
    color: #64748b !important;
    font-size: 15px;
    margin-bottom: 22px;
}

/* =========================
   FORM
========================= */

[data-testid="stForm"] {
    background-color: #ffffff;
    border: 1px solid #e1e7ef;
    border-radius: 22px;
    padding: 25px 30px 30px 30px;
    box-shadow: 0 7px 24px rgba(15, 23, 42, 0.05);
}

/* Widget labels */
label[data-testid="stWidgetLabel"] p {
    color: #172033 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

/* Slider numbers */
[data-testid="stSlider"] p {
    color: #172033 !important;
}

/* =========================
   BUTTON
========================= */

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 54px;
    border: none !important;
    border-radius: 13px !important;
    background: linear-gradient(90deg, #2368a2, #347fbd) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #185783, #286da7) !important;
    color: #ffffff !important;
    border: none !important;
}

/* =========================
   RESULT
========================= */

.result-success {
    background: #ecfdf3;
    border: 1px solid #abefc6;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 8px;
}

.result-success-title {
    color: #067647 !important;
    font-size: 31px;
    font-weight: 800;
    margin: 5px 0;
}

.result-danger {
    background: #fef3f2;
    border: 1px solid #fecdca;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 8px;
}

.result-danger-title {
    color: #b42318 !important;
    font-size: 31px;
    font-weight: 800;
    margin: 5px 0;
}

.result-text {
    color: #475467 !important;
    font-size: 15px;
}

/* =========================
   PROBABILITY CARDS
========================= */

.prob-card {
    background: #ffffff;
    border: 1px solid #e1e7ef;
    border-radius: 17px;
    padding: 22px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.04);
}

.prob-label {
    color: #64748b !important;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 7px;
}

.prob-value {
    color: #172033 !important;
    font-size: 29px;
    font-weight: 800;
}

/* =========================
   EXPANDER
========================= */

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e1e7ef !important;
    border-radius: 14px !important;
}

[data-testid="stExpander"] summary {
    color: #172033 !important;
}

[data-testid="stExpander"] summary p {
    color: #172033 !important;
    font-weight: 650 !important;
}

/* =========================
   DATAFRAME
========================= */

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

/* =========================
   FOOTER
========================= */

.footer {
    color: #94a3b8 !important;
    text-align: center;
    font-size: 13px;
    padding-top: 35px;
    padding-bottom: 10px;
}
</style>
""",
    unsafe_allow_html=True
)


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
# HERO
# PENTING: HTML dibuat tanpa indentasi di dalam string
# =========================================================
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🎓 Student Placement Prediction</div>'
    '<div class="hero-description">'
    'Prediksi status penempatan mahasiswa berdasarkan aktivitas akademik '
    'menggunakan algoritma Random Forest Classifier.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INFORMATION CARDS
# =========================================================
card1, card2, card3 = st.columns(3, gap="medium")


with card1:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">🤖 Machine Learning Model</div>'
        '<div class="info-value">Random Forest</div>'
        '</div>',
        unsafe_allow_html=True
    )


with card2:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">📊 Input Features</div>'
        '<div class="info-value">6 Features</div>'
        '</div>',
        unsafe_allow_html=True
    )


with card3:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">🎯 Prediction Target</div>'
        '<div class="info-value">Placement</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# INPUT TITLE
# =========================================================
st.markdown(
    '<div class="section-title">📝 Student Information</div>'
    '<div class="section-description">'
    'Masukkan informasi mahasiswa pada formulir berikut untuk '
    'mendapatkan hasil prediksi.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT FORM
# =========================================================
with st.form("prediction_form"):

    left, right = st.columns(2, gap="large")

    with left:

        study_hours = st.slider(
            "📚 Study Hours (jam/hari)",
            min_value=1.0,
            max_value=11.0,
            value=6.0,
            step=0.5,
            help="Rata-rata jumlah jam belajar mahasiswa setiap hari."
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


    with right:

        internet_usage = st.slider(
            "🌐 Internet Usage (jam/hari)",
            min_value=1.0,
            max_value=11.0,
            value=6.0,
            step=0.5,
            help="Rata-rata penggunaan internet mahasiswa setiap hari."
        )

        assignments_completed = st.slider(
            "✅ Assignments Completed",
            min_value=0,
            max_value=20,
            value=10,
            step=1,
            help="Jumlah tugas yang telah diselesaikan."
        )

        previous_score = st.slider(
            "📈 Previous Score",
            min_value=35.0,
            max_value=95.0,
            value=65.0,
            step=1.0,
            help="Nilai akademik mahasiswa sebelumnya."
        )


    st.write("")

    predict_button = st.form_submit_button(
        "🔍 Predict Student Placement",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================
if predict_button:

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


    # Menjaga urutan feature agar sama dengan training
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


        st.markdown(
            '<div class="section-title">🎯 Prediction Result</div>'
            '<div class="section-description">'
            'Hasil prediksi berdasarkan data mahasiswa yang telah dimasukkan.'
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # RESULT
        # =================================================
        if prediction[0] == 1:

            st.markdown(
                '<div class="result-success">'
                '<div style="font-size:48px;">🎉</div>'
                '<div class="result-success-title">PLACED</div>'
                '<div class="result-text">'
                'Model memprediksi bahwa mahasiswa memiliki '
                'status <b>Placed</b>.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-danger">'
                '<div style="font-size:48px;">📌</div>'
                '<div class="result-danger-title">NOT PLACED</div>'
                '<div class="result-text">'
                'Model memprediksi bahwa mahasiswa memiliki '
                'status <b>Not Placed</b>.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


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


            st.write("")

            probability1, probability2 = st.columns(
                2,
                gap="medium"
            )


            with probability1:

                st.markdown(
                    f'<div class="prob-card">'
                    f'<div class="prob-label">📌 Not Placed Probability</div>'
                    f'<div class="prob-value">{not_placed_probability:.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


            with probability2:

                st.markdown(
                    f'<div class="prob-card">'
                    f'<div class="prob-label">🎓 Placed Probability</div>'
                    f'<div class="prob-value">{placed_probability:.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


            st.write("")

            st.progress(
                float(placed_probability / 100),
                text=f"Placement Probability: {placed_probability:.2f}%"
            )


        # =================================================
        # INPUT SUMMARY
        # =================================================
        with st.expander(
            "📋 Lihat Detail Data Input",
            expanded=False
        ):

            summary = pd.DataFrame(
                {
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
                }
            )


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
# ABOUT
# =========================================================
st.write("")


with st.expander(
    "ℹ️ Tentang Sistem Prediksi",
    expanded=False
):

    st.markdown(
        """
Aplikasi **Student Placement Prediction** menggunakan algoritma
**Random Forest Classifier** untuk memprediksi status penempatan mahasiswa.

Variabel yang digunakan:

- 📚 Study Hours
- 🏫 Attendance
- 😴 Sleep Hours
- 🌐 Internet Usage
- ✅ Assignments Completed
- 📈 Previous Score

Sebelum dilakukan prediksi, data input diproses menggunakan scaler
yang sama dengan scaler pada tahap pelatihan model.
        """
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">'
    '🎓 Student Placement Prediction &nbsp;•&nbsp; Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
