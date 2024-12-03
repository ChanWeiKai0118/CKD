# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:11:10 2024

@author: kevin
"""

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

# Page styling
st.title("CKD Prediction")
st.markdown("### Enter patient details below to predict CKD probability:")

# Feature ranges (training set)
feature_ranges = {
    "SystolicBP (SBP)": (90, 179),
    "DiastolicBP (DBP)": (60, 119),
    "Fasting Blood Sugar": (70, 200),
    "Serum Creatinine": (0.5, 5),
    "BUN Levels": (5, 50),
    "GFR": (15.1, 120),
    "Protein in Urine": (0, 5),
    "Cholesterol HDL": (20, 100)
}

# Input form
with st.form("CKD_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        SBP = st.number_input("Systolic Blood Pressure (SBP)", min_value=0, max_value=300, step=1)
        DBP = st.number_input("Diastolic Blood Pressure (DBP)", min_value=0, max_value=200, step=1)
        Glucose = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, step=1)
    
    with col2:
        SCr = st.number_input("Serum Creatinine", min_value=0.0, step=0.1)
        BUN = st.number_input("BUN Levels", min_value=0, max_value=200, step=1)
        GFR = st.number_input("GFR", min_value=0.0, step=0.1)

    with col3:
        ProteinInUrine = st.number_input("Protein in Urine", min_value=0.0, step=0.1)
        HDL = st.number_input("Cholesterol HDL", min_value=0, max_value=200, step=1)

    with col4:
        Gender = st.radio("Gender", options=["Male", "Female"])
        Gender = 1 if Gender == "Male" else 2
        Family_KD = st.checkbox("Family History of Kidney Disease")
        UTI = st.checkbox("Urinary Tract Infections")
        Itching = st.checkbox("Itching")
        MuscleCramps = st.checkbox("Muscle Cramps")

    submitted = st.form_submit_button("Predict")

# Style for button
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px; /* 圓角邊框 */
        height: 50px;
        width: 200px;
        font-size: 20px;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF0000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction logic
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
        st.subheader("Prediction Results (cutoff value : 50%)")
        percentage_prob = prob * 100
        if prob >= 0.5:
            st.success(f"**Probable CKD** with probability: {percentage_prob:.1f}%")
        else:
            st.info(f"**Unlikely CKD** with probability: {percentage_prob:.1f}%")

    # Feature range display
    st.markdown("---")
    st.markdown("### Feature Ranges from Training Data:")
    for feature, (min_val, max_val) in feature_ranges.items():
        st.markdown(f"- **{feature}**: Min = {min_val}, Max = {max_val}")
