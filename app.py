import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f9fc;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 32px 36px;
            border-radius: 22px;
            background: linear-gradient(135deg, #1f4e78 0%, #326fa8 100%);
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        }

        .hero h1 {
            font-size: 36px;
            margin: 0 0 8px 0;
            font-weight: 700;
        }

        .hero p {
            margin: 0;
            font-size: 16px;
            opacity: 0.9;
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 4px;
            color: #1f2937;
        }

        .section-desc {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 20px;
        }

        [data-testid="stMetric"] {
            background-color: white;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 12px;
            height: 48px;
            font-size: 16px;
            font-weight: 600;
        }

        .result-success {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            padding: 28px;
            border-radius: 18px;
            text-align: center;
            margin-top: 15px;
        }

        .result-success h2 {
            color: #047857;
            margin: 5px 0;
        }

        .result-danger {
            background: #fef2f2;
            border: 1px solid #fecaca;
            padding: 28px;
            border-radius: 18px;
            text-align: center;
            margin-top: 15px;
        }

        .result-danger h2 {
            color: #b91c1c;
            margin: 5px 0;
        }

        .footer {
            text-align: center;
            color: #9ca3af;
            margin-top: 45px;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_models():
    model = joblib.load("random_forest_model.pkl")
    scaler_model = joblib.load("scaler.pkl")
    return model, scaler_model


try:
    rf_model, scaler = load_models()

except Exception as e:
    st.error("Model gagal dimuat.")
    st.exception(e)
    st.stop()


# ============================================================
# FEATURE COLUMNS
# ============================================================
# placement_status TIDAK dimasukkan karena merupakan target.
feature_columns = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score"
]


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Placement Prediction</h1>
        <p>
            Sistem prediksi peluang penempatan mahasiswa menggunakan
            algoritma Random Forest Classifier.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INFORMATION
# ============================================================
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        label="🤖 Model",
        value="Random Forest"
    )

with metric2:
    st.metric(
        label="📊 Jumlah Fitur",
        value="6 Features"
    )

with metric3:
    st.metric(
        label="🎯 Output",
        value="Placement"
    )


st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">
        📝 Student Information
    </div>
    <div class="section-desc">
        Masukkan informasi akademik dan aktivitas mahasiswa untuk melakukan prediksi.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT FORM
# ============================================================
with st.form("prediction_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:

        study_hours = st.slider(
            "📚 Study Hours",
            min_value=1.0,
            max_value=11.0,
            value=6.0,
            step=0.5,
            help="Jumlah jam belajar mahasiswa."
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
            "😴 Sleep Hours",
            min_value=4.0,
            max_value=9.0,
            value=7.0,
            step=0.5,
            help="Rata-rata jumlah jam tidur."
        )


    with col2:

        internet_usage = st.slider(
            "🌐 Internet Usage",
            min_value=1.0,
            max_value=11.0,
            value=6.0,
            step=0.5,
            help="Rata-rata penggunaan internet per hari."
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
            help="Nilai akademik sebelumnya."
        )


    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.form_submit_button(
        "🔍 Predict Student Placement",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================
if predict_button:

    # Membentuk DataFrame input
    input_data = pd.DataFrame(
        [
            {
                "study_hours": study_hours,
                "attendance": attendance,
                "sleep_hours": sleep_hours,
                "internet_usage": internet_usage,
                "assignments_completed": assignments_completed,
                "previous_score": previous_score,
            }
        ]
    )

    # Memastikan urutan fitur sama seperti data training
    input_data = input_data[feature_columns]

    try:

        # Scaling
        input_scaled = pd.DataFrame(
            scaler.transform(input_data),
            columns=feature_columns
        )

        # Prediction
        prediction = rf_model.predict(input_scaled)

        # ====================================================
        # RESULT SECTION
        # ====================================================
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-title">
                🎯 Prediction Result
            </div>
            <div class="section-desc">
                Hasil prediksi berdasarkan data mahasiswa yang dimasukkan.
            </div>
            """,
            unsafe_allow_html=True
        )


        # Prediction probability
        probability = None

        if hasattr(rf_model, "predict_proba"):
            probability = rf_model.predict_proba(input_scaled)


        if prediction[0] == 1:

            st.markdown(
                """
                <div class="result-success">
                    <div style="font-size:45px;">🎉</div>
                    <h2>PLACED</h2>
                    <p>
                        Model memprediksi bahwa mahasiswa memiliki
                        status <b>Placed</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-danger">
                    <div style="font-size:45px;">📌</div>
                    <h2>NOT PLACED</h2>
                    <p>
                        Model memprediksi bahwa mahasiswa memiliki
                        status <b>Not Placed</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILITY
        # ====================================================
        if probability is not None:

            st.markdown("<br>", unsafe_allow_html=True)

            prob_col1, prob_col2 = st.columns(2)

            not_placed_probability = probability[0][0] * 100
            placed_probability = probability[0][1] * 100

            with prob_col1:
                st.metric(
                    "Probability - Not Placed",
                    f"{not_placed_probability:.2f}%"
                )

            with prob_col2:
                st.metric(
                    "Probability - Placed",
                    f"{placed_probability:.2f}%"
                )

            st.progress(
                int(placed_probability),
                text=f"Placement Probability: {placed_probability:.2f}%"
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================
        with st.expander("📋 Lihat Data Input"):

            display_data = pd.DataFrame(
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
                        study_hours,
                        f"{attendance}%",
                        sleep_hours,
                        internet_usage,
                        assignments_completed,
                        previous_score
                    ]
                }
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Terjadi kesalahan ketika melakukan prediksi."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("ℹ️ Tentang Aplikasi"):

    st.write(
        """
        Aplikasi ini menggunakan **Random Forest Classifier**
        untuk memprediksi status penempatan mahasiswa berdasarkan
        enam variabel:
        
        - Study Hours
        - Attendance
        - Sleep Hours
        - Internet Usage
        - Assignments Completed
        - Previous Score
        
        Data input akan melalui proses scaling menggunakan scaler
        yang telah dibuat pada tahap training sebelum digunakan
        oleh model Random Forest.
        """
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        Student Placement Prediction • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
