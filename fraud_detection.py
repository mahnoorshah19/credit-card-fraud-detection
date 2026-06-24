import streamlit as st
import pandas as pd
import joblib

import sklearn
import sys

print("SKLEARN:", sklearn.__version__)
print("PYTHON:", sys.executable)

# Load Model
model = joblib.load("fraud_detection_model.pkl")

# Streamlit Web App
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Enter the transaction details below to predict whether it is fraudulent or legitimate.")

st.divider()
st.subheader("Transaction Details")

# Input Form Layout

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox("Transaction Type", 
                                        ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT'])
        amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
        oldbalanceOrg = st.number_input("Old Balance (Origin Account)", min_value=0.0, value=10000.0)

    with col2:
        newbalanceOrig = st.number_input("New Balance (Origin Account)", min_value=0.0, value=9000.0)
        oldbalanceDest = st.number_input("Old Balance (Destination Account)", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("New Balance (Destination Account)", min_value=0.0, value=0.0)

    submitted = st.form_submit_button("Predict Fraud")


# Prediction Section

if submitted:
    st.divider()
    st.subheader("Prediction Result")

    # Create DataFrame for model

    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of Fraud

    # Display Output with Stylings

    if prediction == 1:
        st.error(f"**Fraud Detected!**")
        st.write(f"**Fraud Probability:** `{probability:.4f}`")
    else:
        st.success("Transaction is Legitimate")
        st.write(f"**Fraud Probability:** `{probability:.4f}`")

    # Risk Meter

    st.subheader("Fraud Risk Level")

    if probability < 0.20:
        st.progress(0.20)
        st.write("🟢 Low Risk")
    elif probability < 0.60:
        st.progress(0.60)
        st.write("🟡 Medium Risk")
    else:
        st.progress(0.95)
        st.write("🔴 High Risk")

    # Show the Input Again for Transparency

    st.subheader("Input Summary")
    st.dataframe(input_data)

st.divider()

# Footer

st.caption("Credir Card Fraud Detection using Machine Learning + Streamlit Web Application")