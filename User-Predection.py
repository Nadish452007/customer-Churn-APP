
import pickle
import streamlit as st
import os
import pandas as pd

loaded_model = pickle.load(open('Customer_churn.sav', 'rb'))


st.markdown("<h1 style='text-align:center;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    customerID = st.text_input("Customer ID")
with col2:
    gender = st.selectbox("Gender", ["", "Male", "Female"])
with col3:
    SeniorCitizen = st.selectbox("Senior Citizen", ["", 0, 1])

with col1:
    Partner = st.selectbox("Partner", ["", "Yes", "No"])
with col2:
    Dependents = st.selectbox("Dependents", ["", "Yes", "No"])
with col3:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)

with col1:
    PhoneService = st.selectbox("Phone Service", ["", "Yes", "No"])
with col2:
    MultipleLines = st.selectbox("Multiple Lines", ["", "Yes", "No", "No phone service"])
with col3:
    InternetService = st.selectbox("Internet Service", ["", "DSL", "Fiber optic", "No"])

with col1:
    OnlineSecurity = st.selectbox("Online Security", ["", "Yes", "No", "No internet service"])
with col2:
    OnlineBackup = st.selectbox("Online Backup", ["", "Yes", "No", "No internet service"])
with col3:
    DeviceProtection = st.selectbox("Device Protection", ["", "Yes", "No", "No internet service"])

with col1:
    TechSupport = st.selectbox("Tech Support", ["", "Yes", "No", "No internet service"])
with col2:
    StreamingTV = st.selectbox("Streaming TV", ["", "Yes", "No", "No internet service"])
with col3:
    StreamingMovies = st.selectbox("Streaming Movies", ["", "Yes", "No", "No internet service"])

with col1:
    Contract = st.selectbox("Contract", ["", "Month-to-month", "One year", "Two year"])
with col2:
    PaperlessBilling = st.selectbox("Paperless Billing", ["", "Yes", "No"])
with col3:
    PaymentMethod = st.selectbox(
        "Payment Method", ["", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.text_input("Total Charges")


def validate_inputs():

    required_fields = [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                       MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                       DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                       Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]

    if "" in required_fields or TotalCharges.strip() == "":
        st.error("Please fill in all fields before prediction.")
        return False
    try:
        float(TotalCharges)
        int(SeniorCitizen)
        int(tenure)
        float(MonthlyCharges)
    except ValueError:
        st.error("Please enter numeric values for Senior Citizen, Tenure, Monthly Charges, and Total Charges.")
        return False

    return True


if st.button("Customer Churn Test Result"):

    if validate_inputs():


        user_input = {
            "SeniorCitizen": [int(SeniorCitizen)],
            "tenure": [int(tenure)],
            "MonthlyCharges": [float(MonthlyCharges)],
            "customerID": [str(customerID) if customerID else "0000-Unknown"],
            "gender": [gender],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines],
            "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity],
            "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection],
            "TechSupport": [TechSupport],
            "StreamingTV": [StreamingTV],
            "StreamingMovies": [StreamingMovies],
            "Contract": [Contract],
            "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod],
            "TotalCharges": [float(TotalCharges)]
        }

        input_df = pd.DataFrame(user_input)


        expected_cols = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'customerID', 'gender', 'Partner',
            'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'TotalCharges'
        ]
        input_df = input_df[expected_cols]


        categorical_cols = [
            'customerID', 'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        for col in categorical_cols:
            input_df[col] = input_df[col].astype('category')

        try:
            prediction = loaded_model.predict(input_df)

            if prediction[0] == 1:
                st.error("Customer is likely to CHURN ❌")
            else:
                st.success("Customer will NOT churn ✔️")

        except Exception as e:
            st.error(f"Error in prediction: {e}")
