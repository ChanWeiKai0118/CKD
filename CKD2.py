import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.preprocessing import RobustScaler

# GitHub Raw URLs for model and scaler
github_model_url = "https://raw.githubusercontent.com/ChanWeiKai0118/CKD/main/LGB%20model.pkl"
scaler_url = "https://raw.githubusercontent.com/ChanWeiKai0118/CKD/main/scaler_sel.joblib"

# Load the model
response = requests.get(github_model_url)
with open("LGB_model.pkl", "wb") as f:
    f.write(response.content)
LGBM_rg = joblib.load("LGB_model.pkl")

# Load the scaler
scaler_response = requests.get(scaler_url)
with open("robust_scaler.joblib", "wb") as scaler_file:
    scaler_file.write(scaler_response.content)
scaler = joblib.load("robust_scaler.joblib")

# Initialize previous data storage
previous_data = pd.DataFrame(columns=[
    'Gender', 'FamilyHistoryKidneyDisease', 'UrinaryTractInfections',
    'SystolicBP', 'DiastolicBP', 'FastingBloodSugar', 'SerumCreatinine',
    'BUNLevels', 'GFR', 'ProteinInUrine', 'CholesterolHDL', 'MuscleCramps',
    'Itching', 'CKD'
])

# Page 1: Input form
st.title("Chronic Kidney Disease (CKD) Prediction")
st.markdown("### Enter patient details below to predict CKD probability:")

with st.form("CKD_form"):
    SBP = st.number_input("Systolic Blood Pressure (SBP)", min_value=0.0)
    DBP = st.number_input("Diastolic Blood Pressure (DBP)", min_value=0.0)
    Glucose = st.number_input("Fasting Blood Sugar", min_value=0.0)
    SCr = st.number_input("Serum Creatinine", min_value=0.0)
    BUN = st.number_input("BUN Levels", min_value=0.0)
    GFR = st.number_input("GFR", min_value=0.0)
    ProteinInUrine = st.number_input("Protein in Urine", min_value=0.0)
    HDL = st.number_input("Cholesterol HDL", min_value=0.0)

    Gender = st.radio("Gender", options=["Male", "Female"])
    Gender = 1 if Gender == "Male" else 2

    Family_KD = st.checkbox("Family History of Kidney Disease")
    UTI = st.checkbox("Urinary Tract Infections")
    Itching = st.checkbox("Itching")
    MuscleCramps = st.checkbox("Muscle Cramps")

    submitted = st.form_submit_button("Predict")

# Prediction Logic
if submitted:
    input_data1 = {
        'SystolicBP': [SBP],
        'DiastolicBP': [DBP],
        'FastingBloodSugar': [Glucose],
        'SerumCreatinine': [SCr],
        'BUNLevels': [BUN],
        'GFR': [GFR],
        'ProteinInUrine': [ProteinInUrine],
        'CholesterolHDL': [HDL]
    }
    input_data2 = {
        'MuscleCramps': [int(MuscleCramps)],
        'Itching': [int(Itching)],
        'UrinaryTractInfections': [int(UTI)],
        'FamilyHistoryKidneyDisease': [int(Family_KD)],
        'Gender': [Gender]
    }

    input_df1 = pd.DataFrame(input_data1)
    input_df2 = pd.DataFrame(input_data2)

    # Scale numerical data
    input_data_scaled = scaler.transform(input_df1)
    input_df_scaled = pd.DataFrame(input_data_scaled, columns=input_df1.columns)

    # Combine scaled and categorical data
    column_order = ['Gender', 'FamilyHistoryKidneyDisease', 'UrinaryTractInfections',
                    'SystolicBP', 'DiastolicBP', 'FastingBloodSugar', 'SerumCreatinine',
                    'BUNLevels', 'GFR', 'ProteinInUrine', 'CholesterolHDL', 'MuscleCramps',
                    'Itching']
    input_data_final = pd.concat([input_df_scaled, input_df2], axis=1)
    input_data_final = input_data_final[column_order]

    # Make predictions
    y_probabilities = LGBM_rg.predict_proba(input_data_final)[:, 1]
    for prob in y_probabilities:
        st.subheader("Prediction Results")
        if prob >= 0.5:
            st.success(f"**Probable CKD** with probability: {prob:.3f}")
        else:
            st.info(f"**Unlikely CKD** with probability: {prob:.3f}")
