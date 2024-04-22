import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open("model/best_model.sav", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Function for diabetes prediction
def diabetes_prediction(input_data):
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform([input_data])
    pred = model.predict(scaled_input)

    if pred[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function
def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Diabetes Prediction",
        page_icon=":chart_with_upwards_trend:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    # Set background color and add padding
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #333333;
            padding: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Informational text about the tool
    st.title("Diabetes Prediction Tool")
    st.write("Welcome to the Diabetes Prediction Tool! This tool uses machine learning to predict whether a person is diabetic or not based on their health data.")
    st.write("Please enter the patient's information in the fields below. All values should be non-negative.")
    st.write("If you're unsure about any value, consult a healthcare professional for accurate measurements.")
    st.markdown("---")
    
    # Input fields for user data
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age of the Person (years)", min_value=0)
    
    st.markdown("---")
    
    # Button for diagnosis
    if st.button('Run Diabetes Test', key='diagnosis_button'):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        
        diagnosis = diabetes_prediction(input_data)
        st.success(diagnosis)

if __name__ == '__main__':
    main()
