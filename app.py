import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load pre-trained model only
model = joblib.load("diabetes_model.pkl")

# Page title
st.set_page_config(page_title="AI Diabetes Diagnosis", layout="centered")
st.title("🩺 AI-Based Medical Diagnosis System")
st.subheader("Predict Diabetes Risk using Patient Data")

st.write("Upload your medical report (CSV) **or** fill the form manually:")

# -----------------------------
# Option 1: Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Report", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Report")
    st.dataframe(data)

    try:
        # Scale numeric columns automatically
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        prediction = model.predict(X_scaled)

        if prediction[0] == 1:
            st.error("⚠️ High Risk: Diabetes Detected")
        else:
            st.success("✅ No Diabetes Detected")
    except Exception as e:
        st.error(f"Error: Ensure CSV columns are numeric and match model input. {e}")

st.markdown("---")
st.subheader("Or Enter Details Manually")

# -----------------------------
# Option 2: Manual Input Form
# -----------------------------
with st.form(key="manual_form"):
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 0, 120, 33)
    
    submit_button = st.form_submit_button(label="Predict from Form")

if submit_button:
    user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    # Scale manually entered data
    scaler = StandardScaler()
    user_data_scaled = scaler.fit_transform(user_data)  # Scale single row
    pred = model.predict(user_data_scaled)

    if pred[0] == 1:
        st.error("⚠️ High Risk: Diabetes Detected")
    else:
        st.success("✅ No Diabetes Detected")
