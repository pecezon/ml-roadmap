import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_pipeline.pkl')

st.title('Fraud Detection App')
st.markdown('Enter transaction details to predict if it is fraudulent.')
st.divider()

features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# --- Example Data ---

non_fraud_example = {
    'Time': 406, 'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347,
    'V4': 1.378155, 'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599,
    'V8': 0.098698, 'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600,
    'V12': -0.617801, 'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177,
    'V16': -0.470401, 'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993,
    'V20': 0.251412, 'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474,
    'V24': 0.066928, 'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558,
    'V28': -0.021053, 'Amount': 149.62
}

fraud_example = {
    'Time': 406, 'V1': -2.312227, 'V2': 1.951992, 'V3': -1.609851,
    'V4': 3.997906, 'V5': -0.522188, 'V6': -1.426545, 'V7': -2.537387,
    'V8': 1.391657, 'V9': -2.770089, 'V10': -2.772272, 'V11': 3.202033,
    'V12': -2.899907, 'V13': -0.595222, 'V14': -4.289254, 'V15': 0.389724,
    'V16': -1.140747, 'V17': -2.830056, 'V18': -0.016822, 'V19': 0.416956,
    'V20': 0.126911, 'V21': 0.517232, 'V22': -0.035049, 'V23': -0.465211,
    'V24': 0.320198, 'V25': 0.044519, 'V26': 0.177840, 'V27': 0.261145,
    'V28': -0.143276, 'Amount': 0.00
}

# --- Initialize session state for each feature ---
for feat in features:
    if feat not in st.session_state:
        st.session_state[feat] = 0.0

# --- Buttons ---
col1, col2 = st.columns(2)

if col1.button("🟢 Load Non-Fraud Example"):
    for k, v in non_fraud_example.items():
        st.session_state[k] = v
    st.rerun()

if col2.button("🔴 Load Fraud Example"):
    for k, v in fraud_example.items():
        st.session_state[k] = v
    st.rerun()

st.divider()

# --- Input Fields ---
inputs = {}
cols = st.columns(4)

for i, feat in enumerate(features):
    col = cols[i % 4]
    if feat == 'Time':
        inputs[feat] = col.number_input(
            feat,
            step=1,
            format="%d",
            key=feat
        )
    else:
        inputs[feat] = col.number_input(
            feat,
            step=0.0001,
            format="%.6f",
            key=feat
        )

# --- Prediction ---
if st.button("Predict"):
    X = pd.DataFrame([inputs])[features]
    try:
        prob = model.predict_proba(X)[0][1]*100
        if prob >= 50:
            st.error(f"Fraud probability: {prob:.4f}%")
        else:
            st.success(f"Fraud probability: {prob:.4f}%")
    except Exception:
        pred = model.predict(X)[0]
        st.write("Prediction:", int(pred))
