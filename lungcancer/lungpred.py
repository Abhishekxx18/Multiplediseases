import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
lung_data = pd.read_csv("C:\\Users\\abhi0\\Desktop\\majorproject\\lungdataset.csv")

# Encode categorical variables
lung_data['GENDER'] = lung_data['GENDER'].map({"M": 1, "F": 2})
lung_data['LUNG_CANCER'] = lung_data['LUNG_CANCER'].map({"YES": 1, "NO": 2})

# Define dependent and independent variables
x = lung_data.iloc[:, 0:-1]
y = lung_data.iloc[:, -1:]

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(x_train_scaled, y_train.values.ravel())

# Streamlit UI elements for input
st.title("Lung Cancer Prediction")
st.subheader("Enter the details below:")

# Input fields for user to enter data
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 20, 90, 60)  # Slider for age input
smoking = st.selectbox("Smoking", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
yellow_fingers = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
anxiety = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
peer_pressure = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
chronic_disease = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
fatigue = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
allergy = st.selectbox("Allergy", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
wheezing = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
alcohol_consuming = st.selectbox("Alcohol Consuming", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
coughing = st.selectbox("Coughing", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
chest_pain = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

# Convert gender to the corresponding numeric value
gender = 1 if gender == "Male" else 2

# Prepare input data for prediction
input_data = [
    gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, 
    fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, 
    swallowing_difficulty, chest_pain
]

# Scale the input data
input_data_scaled = scaler.transform([input_data])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_text = "Lung Cancer Detected" if prediction == 1 else "No Lung Cancer Detected"
    st.write(f"Prediction: {prediction_text}")
    
    
