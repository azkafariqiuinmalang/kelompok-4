import streamlit as st
import pandas as pd
import joblib

# Load model dan preprocessing
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl") # Load the scaler
loaded_feature_columns = joblib.load("feature_columns.pkl")

# 'placement_status' should not be in features after the fixes
feature_columns = loaded_feature_columns # Assuming feature_columns.pkl already contains only features

st.title("Student Performance Prediction")
st.write(
    "Aplikasi demo Machine Learning menggunakan Random Forest Classifier."
)

# Input fields for all relevant features
study_hours = st.number_input(
    "Study Hours (1-11)", min_value=1.0, max_value=11.0, value=6.0
)

attendance = st.number_input(
    "Attendance (%) (40-100)", min_value=40.0, max_value=100.0, value=70.0
)

sleep_hours = st.number_input(
    "Sleep Hours (4-9)", min_value=4.0, max_value=9.0, value=7.0
)

internet_usage = st.number_input(
    "Internet Usage (hours) (1-11)", min_value=1.0, max_value=11.0, value=6.0
)

assignments_completed = st.number_input(
    "Assignments Completed (0-20)", min_value=0, max_value=20, value=10
)

previous_score = st.number_input(
    "Previous Score (35-95)", min_value=35.0, max_value=95.0, value=65.0
)

if st.button("Predict"):
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

    # Ensure order of features is consistent with training data
    input_data = input_data[feature_columns]

    # Use scaler that was fitted on training data
    input_scaled = pd.DataFrame(
        scaler.transform(input_data),
        columns=feature_columns
    )

    prediction = rf_model.predict(input_scaled)

    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.success("Status Prediksi: Placed")
    else:
        st.error("Status Prediksi: Not Placed")
