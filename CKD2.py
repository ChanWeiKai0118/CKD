import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.preprocessing import RobustScaler

# GitHub Raw URLs for model and scaler
github_model_url = "https://raw.githubusercontent.com/ChanWeiKai0118/CKD/main/LGB%20model2.pkl"
scaler_url = "https://raw.githubusercontent.com/ChanWeiKai0118/CKD/main/scaler_sel2.joblib"

# Load the model
response = requests.get(github_model_url)
with open("LGB%20model2.pkl", "wb") as f:
    f.write(response.content)
new_model = joblib.load("LGB%20model2.pkl")

# Load the scaler
scaler_response = requests.get(scaler_url)
with open("scaler_sel2.joblib", "wb") as scaler_file:
    scaler_file.write(scaler_response.content)
scaler = joblib.load("scaler_sel2.joblib")

# Page styling
st.title("CKD Prediction")

# Feature ranges (training set)
feature_ranges = {
    "BMI": (10, 50),
    "Physical Activity (hours/week)": (0, 10),
    "Diet Quality (1-5)": (0, 10),
    "Sleep Quality (4-10)": (4, 10),
    "SystolicBP": (90, 180),
    "DiastolicBP": (60, 120),
    "Fasting Blood Sugar": (70, 200),
    "HbA1c": (4, 10),
    "Serum Creatinine": (0.5, 5),
    "BUN Levels": (5, 50),
    "Protein in Urine": (0, 5),
    "Serum Electrolytes (Sodium)": (135, 145),
    "Hemoglobin Levels": (10, 18),
    "Cholesterol Total": (100, 300),
    "Cholesterol HDL": (20, 100),
    "Cholesterol Triglycerides": (50, 400),
    "Quality of Life Score (0-100)": (0, 100),
    "Medical Checkups Frequency (per year)": (0, 4),
    "Medication Adherence (0-10)": (0, 10),
    "Health Literacy (0-10)": (0, 10)
}

# Display feature ranges
st.markdown("### Feature Ranges from Training Data:")
for feature, (min_val, max_val) in feature_ranges.items():
    st.markdown(f"- **{feature}**: Min = {min_val}, Max = {max_val}")

st.markdown("### Enter patient details below to predict CKD probability:")

st.write("Scaler feature names:", scaler.feature_names_in_)
st.write("Input feature names:", input_df.columns)

# Input form
with st.form("CKD_form"):
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        BMI = st.number_input("BMI", min_value=0.0, step=0.1)
        PhysicalActivity = st.number_input("Physical Activity (hours/week)", min_value=0, step=1)
        DietQuality = st.slider("Diet Quality (0-10)", min_value=0, max_value=10, step=1)
        SleepQuality = st.slider("Sleep Quality (hours/day)(4-10)", min_value=4, max_value=10, step=1)

    with col2:
        SBP = st.number_input("Systolic Blood Pressure (SBP)", min_value=0, max_value=300, step=1)
        DBP = st.number_input("Diastolic Blood Pressure (DBP)", min_value=0, max_value=200, step=1)
        FastingBloodSugar = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, step=1)
        HbA1c = st.number_input("HbA1c", min_value=0.0, step=0.1)

    with col3:
        SerumCreatinine = st.number_input("Serum Creatinine", min_value=0.0, step=0.1)
        BUNLevels = st.number_input("BUN Levels", min_value=0, max_value=200, step=1)
        ProteinInUrine = st.number_input("Protein in Urine", min_value=0.0, step=0.1)
        SerumElectrolytesSodium = st.number_input("Serum Electrolytes (Sodium)", min_value=100, max_value=200, step=1)

    with col4:
        HemoglobinLevels = st.number_input("Hemoglobin Levels", min_value=0.0, step=0.1)
        CholesterolTotal = st.number_input("Cholesterol Total", min_value=0, max_value=500, step=1)
        CholesterolHDL = st.number_input("Cholesterol HDL", min_value=0, max_value=200, step=1)
        CholesterolTriglycerides = st.number_input("Cholesterol Triglycerides", min_value=0, max_value=500, step=1)
    with col5:
        QualityOfLifeScore = st.slider("Quality of Life Score (0-100)", min_value=0, max_value=100, step=1)
        MedicalCheckupsFrequency = st.slider("Medical Checkups Frequency (per year)", min_value=0, max_value=4, step=1)
        MedicationAdherence = st.slider("Medication Adherence (0-10)", min_value=0, max_value=10, step=1)
        HealthLiteracy = st.slider("Health Literacy (0-10)", min_value=0, max_value=10, step=1)
        MuscleCramps = st.checkbox("Muscle Cramps")
        Itching = st.checkbox("Itching")

    submitted = st.form_submit_button("Predict")



# Prediction logic
if submitted:
    input_data = {
        'BMI': [BMI],
        'PhysicalActivity': [PhysicalActivity],
        'DietQuality': [DietQuality],
        'SleepQuality': [SleepQuality],
        'SystolicBP': [SBP],
        'DiastolicBP': [DBP],
        'FastingBloodSugar': [FastingBloodSugar],
        'HbA1c': [HbA1c],
        'SerumCreatinine': [SerumCreatinine],
        'BUNLevels': [BUNLevels],
        'ProteinInUrine': [ProteinInUrine],
        'SerumElectrolytesSodium': [SerumElectrolytesSodium],
        'HemoglobinLevels': [HemoglobinLevels],
        'CholesterolTotal': [CholesterolTotal],
        'CholesterolHDL': [CholesterolHDL],
        'CholesterolTriglycerides': [CholesterolTriglycerides],
        'QualityOfLifeScore': [QualityOfLifeScore],
        'MedicalCheckupsFrequency': [MedicalCheckupsFrequency],
        'MedicationAdherence': [MedicationAdherence],
        'HealthLiteracy': [HealthLiteracy],
        'MuscleCramps': [int(MuscleCramps)],
        'Itching': [int(Itching)]
    }

    input_df = pd.DataFrame(input_data)

    # Scale numerical data
    input_data_scaled = scaler.transform(input_df)
    input_df_scaled = pd.DataFrame(input_data_scaled, columns=input_df.columns)

    # Make predictions
    y_probabilities = new_model.predict_proba(input_df_scaled)[:, 1]
    for prob in y_probabilities:
        st.subheader("Prediction Results (cutoff value: 50%)")
        percentage_prob = prob * 100
        if prob >= 0.5:
            st.success(f"**Probable CKD** with probability: {percentage_prob:.1f}%")
        else:
            st.info(f"**Unlikely CKD** with probability: {percentage_prob:.1f}%")
