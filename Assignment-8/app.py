# DEPLOYMENT IN STREAMLIT APP

# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model and scaler (cache to avoid reloading on each interaction)
@st.cache_resource
def load_artifacts():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

st.title("Diabetes Prediction — Logistic Regression")

st.markdown("Enter patient features below to predict diabetes outcome (0 = No, 1 = Yes).")

# Input widgets (use the same order your model expects)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=1500, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=33)

if st.button("Predict"):
    X = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                   insulin, bmi, dpf, age]])
    X_scaled = scaler.transform(X)
    pred_proba = model.predict_proba(X_scaled)[0, 1]
    pred = int(model.predict(X_scaled)[0])

    st.write("**Predicted label:**", pred)
    st.write(f"**Probability of diabetes:** {pred_proba:.3f}")

    # Simple interpretation
    if pred == 1:
        st.warning("Model predicts HIGH chance of diabetes. Recommend medical follow-up.")
    else:
        st.success("Model predicts LOW chance of diabetes.")
