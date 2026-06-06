import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load('fraud_model.pkl')

st.title('Credit Card Fraud Detection')
st.write('Enter transaction details to predict if fraudulent')

col1, col2 = st.columns(2)

with col1:
    time = st.number_input('Time', min_value=0.0, value=0.0)
    amount = st.number_input('Amount ($)', min_value=0.0, value=0.0)

with col2:
    v1 = st.number_input('V1', value=0.0)
    v2 = st.number_input('V2', value=0.0)
    v3 = st.number_input('V3', value=0.0)
    v4 = st.number_input('V4', value=0.0)

st.write('### Other Features (V5-V28)')
cols = st.columns(4)
v_values = []
for i in range(5, 29):
    with cols[(i-5) % 4]:
        v = st.number_input(f'V{i}', value=0.0)
        v_values.append(v)

if st.button('Predict', type='primary'):
    features = [time, v1, v2, v3, v4] + v_values + [amount]
    features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    if prediction == 1:
        st.error(f'⚠️ FRAUDULENT Transaction Detected!')
        st.write(f'Fraud Probability: **{probability[1]*100:.2f}%**')
    else:
        st.success(f'✅ Legitimate Transaction')
        st.write(f'Legitimate Probability: **{probability[0]*100:.2f}%**')

st.divider()
st.write('### About')
st.write('Model: Random Forest Classifier')
st.write('Dataset: Kaggle Credit Card Fraud Detection')
st.write('ROC-AUC Score: 0.9999')