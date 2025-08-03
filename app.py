import streamlit as st
import pandas as pd
import joblib

# Load your model and encoders
model = joblib.load("fraud_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Title
st.title("🎓 Scholarship Fraud Detection System")
st.markdown("Upload your scholarship data CSV file and predict whether the application is genuine or fraudulent.")

# Upload section
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Uploaded Data:")
    st.dataframe(df)

    if st.button("🔍 Predict"):
        # Predict fraud for each row
        predictions = model.predict(df)
        df['Prediction'] = ["🛑 FRAUD" if p == 1 else "✅ GENUINE" for p in predictions]

        st.success("🎉 Prediction Complete!")
        st.write("📋 Results:")
        st.dataframe(df)
