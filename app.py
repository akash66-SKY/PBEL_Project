import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("💼 Salary Prediction App")
st.write("Enter employee details to predict the salary:")

# User inputs
st.header("Numeric Inputs")
numerical_inputs = {
    "years_experience": st.number_input("Years of Experience", min_value=0.0),
    "certifications": st.number_input("Certifications", min_value=0.0),
    "age": st.number_input("Age", min_value=0.0),
    "working_hours": st.number_input("Working Hours", min_value=0.0),
}

st.header("Categorical Inputs")
categorical_fields = {
    "education_level": ["High School", "PhD", "Bachelors", "Masters"],
    "job_title": ["Data Scientist", "Software Engineer", "Analyst", "Manager"],
    "industry": ["Education", "IT", "Healthcare", "Finance"],
    "location": ["New York", "London", "Bangalore", "San Francisco"],
    "company_size": ["Medium", "Large", "Small"],
}

cat_inputs = {}
for field, options in categorical_fields.items():
    cat_inputs[field] = st.selectbox(field, options)

# Manual one-hot encoding
row = {}
row.update(numerical_inputs)

# Convert categorical to one-hot
for col, val in cat_inputs.items():
    for option in categorical_fields[col]:
        one_hot_col = f"{col}_{option}"
        row[one_hot_col] = 1 if val == option else 0

# Create dataframe and align with model features
input_df = pd.DataFrame([row])

# Fill missing columns
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features]  # ensure column order

# Predict
if st.button("Predict Salary"):
    salary = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ₹ {salary:,.2f}")
