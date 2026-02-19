import streamlit as st
import pandas as pd
import joblib

# Load saved items
model = joblib.load("mental_health_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🧠 Mental Health Risk Predictor")

# User Inputs
age = st.number_input("Age", 18, 100, 25)
work_hours = st.slider("Work Hours Per Week", 0, 100, 40)
sleep = st.slider("Sleep Hours Per Night", 0.0, 12.0, 7.0)
stress = st.slider("Work Stress Level (1-10)", 1, 10, 5)
loneliness = st.slider("Loneliness Level (1-10)", 1, 10, 5)
social_support = st.slider("Social Support (1-10)", 1, 10, 5)

if st.button("Predict"):

    # Create empty dataframe with all features
    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0   # Fill all with 0

    # Assign user inputs to correct columns
    input_data["Age"] = age
    input_data["Work_Hours_Per_Week"] = work_hours
    input_data["Sleep_Hours_Night"] = sleep
    input_data["Work_Stress_Level"] = stress
    input_data["Loneliness"] = loneliness
    input_data["Social_Support"] = social_support

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠ High Risk of Mental Health Issue")
    else:
        st.success("✅ Low Risk")
