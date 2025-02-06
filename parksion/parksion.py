import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (modify this part to use the appropriate dataset)
@st.cache_data
def load_data():
    data = pd.read_csv(r'parkinsons.csv')  # Use raw string to fix path issue
    data = data.select_dtypes(include=[np.number])  # Keep only numeric columns
    return data

data = load_data()

# Preprocessing
scaler = MinMaxScaler()
X = data.drop(columns=['status'], errors='ignore')  # Replacing 'target' with 'status', assuming it's the correct column
Y = data['status'] if 'status' in data.columns else None

if Y is None:
    st.error(f"Target column not found! Available columns: {list(data.columns)}")
else:
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    def predict(input_data):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        return prediction[0]

    # Streamlit UI
    st.title("Parkinson's Disease Detection")
    st.write("Enter the values for prediction:")


    # Dynamic input fields
    input_features = []
    for feature in X.columns:
        value = st.number_input(f"{feature}", value=0.0)
        input_features.append(value)

    if st.button("Predict"):
        result = predict(input_features)
        st.write("Prediction:", "Parkinson's Detected" if result == 1 else "Healthy")
