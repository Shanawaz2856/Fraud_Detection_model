import streamlit as st
import pandas as pd
import joblib
import sklearn
st.write("sklearn version:", sklearn.__version__)

st.set_page_config(page_title="Fraud Detection", page_icon="🕵️", layout="centered")

st.title("🕵️ Online Payment Fraud Detection")
st.write("Enter transaction details to predict whether it's fraudulent.")

@st.cache_resource
def load_model():
    return joblib.load("fraud_pipeline.pkl")

pipeline = load_model()

txn_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)

st.subheader("Sender Details")
oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=9000.0)

st.subheader("Receiver Details")
oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Transaction looks Legitimate. (Fraud Probability: {proba:.2%})")
