import streamlit as st
import pickle
import pandas as pd

# Load saved model
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Employee Attrition Predictor")

# Inputs
age = st.slider("Age", 18, 60)
income = st.number_input("Monthly Income")
overtime = st.selectbox("OverTime", ["Yes", "No"])

# Create input data
input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

if 'Age' in input_data.columns:
    input_data['Age'] = age
if 'MonthlyIncome' in input_data.columns:
    input_data['MonthlyIncome'] = income
if 'OverTime_Yes' in input_data.columns:
    input_data['OverTime_Yes'] = 1 if overtime == "Yes" else 0

# Prediction
if st.button("Predict"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.error("Employee likely to leave ❗")
    else:
        st.success("Employee likely to stay ✅")