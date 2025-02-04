import numpy as np
import pickle
import streamlit as st
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyAvPpDiLbuzfvKKnfXoEONneqA9pdFqWLw")
myai = genai.GenerativeModel("gemini-1.5-flash")

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Prediction function
def diabetesprediction(input_data):
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array for a single instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Predict using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Recommendation function
def generate_recommendations(input_data, diagnosis):
    prompt = (
        f"The following health parameters were provided:\n"
        f"Pregnancies: {input_data[0]}, Glucose: {input_data[1]}, "
        f"BloodPressure: {input_data[2]}, SkinThickness: {input_data[3]}, "
        f"Insulin: {input_data[4]}, BMI: {input_data[5]}, "
        f"DiabetesPedigreeFunction: {input_data[6]}, Age: {input_data[7]}.\n\n"
        f"The diagnosis is: {diagnosis}.\n"
        
        """Based on the values given, 
         if the person has diabetic explain the possible causes explain it with a subheading if not skip this
        and then 
        the nextsubheading should contain if the values are too high or low highlight the values and provide the normal ranges for the wrong values,
        then in nextsubheading the matter should give the proper recommendations,
        then in last nextsubheading it should give the proper food which should help with abnormal values 
        """
    )

    try:
        response = myai.generate_content(prompt)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"Error generating recommendations: {e}"

# Main function for Streamlit app
def main():
    # Title of the app
    st.title('Diabetes Prediction')

    # User inputs
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age')

    # Variables for storing results
    diagnosis = ''
    recommendations = ''

    # Button for prediction
    if st.button('Diabetes Test Result'):
        try:
            # Validate and convert inputs to floats
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]
            
            # Call the prediction function
            diagnosis = diabetesprediction(input_data)
            
            # Save the diagnosis in session state
            st.session_state.diagnosis = diagnosis
            
            # Display the results
            st.success(diagnosis)

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
    
    # Separate button for recommendations
    if st.button('Check Out The Recommendations') and 'diagnosis' in st.session_state:
        try:
            # Ensure diagnosis is available before generating recommendations
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            recommendations = generate_recommendations(input_data, st.session_state.diagnosis)
            #st.subheader('Key Recommendations:')
            st.write(recommendations)
        except Exception as e:
            st.error(f"Error: {e}")
    elif 'diagnosis' not in st.session_state:
        st.error("Please perform the test first to generate recommendations.")

# Run the app
if __name__ == '__main__':
    main()
