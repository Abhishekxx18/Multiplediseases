import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("C:/Users/abhi0/Desktop/majorproject/LiverDisease/random_forest_model.joblib")


st.title("Liver Disease Prediction App")
st.write("Enter the patient details below to predict the likelihood of liver disease.")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0)
direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.5)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=100)
alanine_aminotransferase = st.number_input("Alanine Aminotransferase", min_value=0, value=25)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, value=30)
total_proteins = st.number_input("Total Proteins", min_value=0.0, value=6.5)
albumin = st.number_input("Albumin", min_value=0.0, value=3.5)
albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0)

# Convert gender to numerical format
gender = 0 if gender == "Male" else 1

# Create input dataframe
input_data = np.array([[
    age, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
    alanine_aminotransferase, aspartate_aminotransferase,
    total_proteins, albumin, albumin_globulin_ratio, gender
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of having liver disease
    result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease Detected"
    st.write(f"### Prediction: {result}")
    