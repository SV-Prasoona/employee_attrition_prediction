import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Employee Attrition Prediction System")

st.write("Fill employee details to predict attrition.")

age = st.slider("Age", 18, 60)
department = st.selectbox("Department", ["HR", "Sales", "Finance", "IT"])
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
years = st.slider("Years At Company", 0, 40)
income = st.number_input("Monthly Income")
satisfaction = st.slider("Job Satisfaction", 1, 5)
worklife = st.slider("Work Life Balance", 1, 5)
overtime = st.selectbox("OverTime", ["Yes", "No"])
distance = st.number_input("Distance From Home")
promotion = st.selectbox("Promotion Last 5 Years", ["Yes", "No"])
performance = st.slider("Performance Rating", 1, 5)
training = st.slider("Training Hours Last Year", 0, 50)

dept_map = {"HR": 0, "Sales": 1, "Finance": 2, "IT": 3}
yes_no_map = {"No": 0, "Yes": 1}

input_data = np.array([[
    age,
    dept_map[department],
    job_level,
    years,
    income,
    satisfaction,
    worklife,
    yes_no_map[overtime],
    distance,
    yes_no_map[promotion],
    performance,
    training
]])

input_scaled = scaler.transform(input_data)

if st.button("Predict"):

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error(" Employee is likely to leave.")
    else:
        st.success(" Employee is likely to stay.")