import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="PPC Risk Predictor", layout="centered")
st.title("Postoperative Pulmonary Complications (PPCs) Risk Predictor")
st.write("Enter patient information to estimate PPC risk after spine surgery.")

# --- Load files (same folder as app.py) ---
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "rf_ppc_model.pkl")
feature_order = joblib.load(BASE_DIR / "feature_order.pkl")

# --- Inputs ---
ASA_cat = st.selectbox("ASA class (1–2 / 3 / 4–5)", options=[1, 2, 3])
Age = st.number_input("Age (years)", min_value=18, max_value=120, value=70)
Op_duration_min = st.number_input("Operation time (min)", min_value=0, max_value=1000, value=180)
EBL_ml = st.number_input("Estimated blood loss (mL)", min_value=0, max_value=5000, value=300)
Na = st.number_input("Serum sodium (mmol/L)", min_value=100.0, max_value=170.0, value=138.0, step=0.1)

SpO2_low = st.selectbox("Preop SpO₂ <96% ?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
Segments_ge3 = st.selectbox("Surgical segments ≥3 ?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
Heart_failure = st.selectbox("Heart failure ?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
Anemia = st.selectbox("Preop anemia ?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
COPD = st.selectbox("COPD ?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
URI_1m = st.selectbox("URI within 1 month ?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("Predict"):
    x = pd.DataFrame([{
        "ASA_cat": ASA_cat,
        "Age": Age,
        "Op_duration_min": Op_duration_min,
        "EBL_ml": EBL_ml,
        "SpO2_low": SpO2_low,
        "Na": Na,
        "Segments_ge3": Segments_ge3,
        "Heart_failure": Heart_failure,
        "Anemia": Anemia,
        "COPD": COPD,
        "URI_1m": URI_1m
    }])

    # enforce column order
    x = x[feature_order]

    prob = float(model.predict_proba(x)[0, 1])
    st.subheader(f"Predicted PPC risk: {prob:.3f}")

    if prob < 0.2:
        st.success("Low risk")
    elif prob < 0.4:
        st.warning("Moderate risk")
    else:
        st.error("High risk")


