import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress all warnings related to missing ScriptRunContext in the MainThread
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext!.*")
import logging

# Set the logging level to WARNING to suppress lower-level logs like the missing ScriptRunContext
logging.basicConfig(level=logging.WARNING)

# Load the dataset
dataset = pd.read_csv("C:/Users/abhi0/Desktop/majorproject/heart/heart.csv")

# Splitting the dataset
X = dataset.drop(columns=['target'])
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# User input form for predictions (Main screen)
st.title("Heart Disease Prediction")
st.header("Enter the following details:")

# Inputs displayed on the main screen with descriptive options
age = st.number_input("Age", min_value=20, max_value=90, value=60)
sex = st.selectbox("Sex", ["Male", "Female"])  
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "Having ST-T wave abnormality", "Left ventricular hypertrophy"])
thalach = st.slider("Max Heart Rate Achieved", 70, 200, 150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment", ["Up", "Flat", "Down"])
ca = st.slider("Number of Major Vessels", 0, 3, 0)
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

# Button to trigger prediction
if st.button('Predict'):
    # Convert input to numeric values as required by the model
    sex = 1 if sex == "Male" else 0
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp) + 1
    fbs = 1 if fbs == "Yes" else 0
    restecg = ["Normal", "Having ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
    exang = 1 if exang == "Yes" else 0
    slope = ["Up", "Flat", "Down"].index(slope) + 1
    thal = ["Normal", "Fixed Defect", "Reversable Defect"].index(thal) + 3
    
    # Create input array with numeric values
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input data using the trained scaler
    input_data_scaled = scaler.transform(input_data)

    # Prediction using Random Forest
    random_forest_pred = model.predict(input_data_scaled)
    rf_result = "Heart Disease Detected" if random_forest_pred == 1 else "No Heart Disease Detected"
    
    # Prediction using Logistic Regression
    logistic_pred = log_model.predict(input_data_scaled)
    log_result = "Heart Disease Detected" if logistic_pred == 1 else "No Heart Disease Detected"

    # Displaying results with red/green buttons
    if random_forest_pred == 1:
        st.markdown(f'<button style="background-color:red;color:white;font-size:20px;width:300px;height:60px;">{rf_result}</button>', unsafe_allow_html=True)
    else:
        st.markdown(f'<button style="background-color:green;color:white;font-size:20px;width:300px;height:60px;">{rf_result}</button>', unsafe_allow_html=True)

    if logistic_pred == 1:
        st.markdown(f'<button style="background-color:red;color:white;font-size:20px;width:300px;height:60px;">{log_result}</button>', unsafe_allow_html=True)
    else:
        st.markdown(f'<button style="background-color:green;color:white;font-size:20px;width:300px;height:60px;">{log_result}</button>', unsafe_allow_html=True)
