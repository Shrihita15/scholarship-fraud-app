import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("fraud_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🎓 Scholarship Fraud Detection System")
st.markdown("Upload your scholarship data CSV file and detect frauds using AI.")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Raw Uploaded Data:")
    st.dataframe(df)

    try:
        # Step 1: Drop Name
        if "Name" in df.columns:
            df.drop(columns=["Name"], inplace=True)



import easyocr
import re
from PIL import Image
import tempfile
import os

st.markdown("---")
st.header("📸 Upload Document for OCR")

ocr_file = st.file_uploader("Upload Income Certificate (Image or PDF)", type=["png", "jpg", "jpeg", "pdf"], key="ocr")

if ocr_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(ocr_file.read())
        temp_path = tmp_file.name

    # Read with EasyOCR
    reader = easyocr.Reader(['en', 'mr'])  # English + Marathi
    text = reader.readtext(temp_path, detail=0, paragraph=True)

    extracted_text = "\n".join(text)
    st.text_area("📄 Extracted Text from Document:", extracted_text, height=300)

    # Try to extract income from the text using regex
    income_match = re.search(r"[\₹Rs\.]?\s?([0-9,]{2,15})", extracted_text.replace(",", ""))
    if income_match:
        income_value = int(income_match.group(1))
        st.success(f"💰 Extracted Income: ₹{income_value}")
    else:
        st.warning("⚠️ Could not detect income amount. Please check document clarity.")
        

        # Step 2: Label Encode
        label_cols = ["Spent_On", "Documents_Verified", "Enrollment_Status", "Application_State"]
        for col in label_cols:
            df[col] = encoders[col].transform(df[col])

        # Step 3: Feature Engineering
        df['Income_Ratio'] = df['Income_Certificate_Amount'] / (df['Actual_Income'] + 1)
        df['Low_Attendance'] = (df['Attendance'] < 60).astype(int)
        df['Fake_Income_Claim'] = (df['Income_Certificate_Amount'] < (df['Actual_Income'] / 2)).astype(int)
        df['Non_Education_Spend'] = (df['Spent_On'] != encoders['Spent_On'].transform(['Education'])[0]).astype(int)

        # Step 4: Define features
        features = ['Income_Certificate_Amount', 'Actual_Income', 'Attendance',
                    'Documents_Verified', 'Enrollment_Status', 'Application_State',
                    'Scholarship_Amount', 'Income_Ratio', 'Low_Attendance',
                    'Fake_Income_Claim', 'Non_Education_Spend']

        X = df[features]

        # Step 5: Predict
        predictions = model.predict(X)
        df['Prediction'] = ["🛑 FRAUD" if p == 1 else "✅ GENUINE" for p in predictions]

        st.success("✅ Prediction complete!")
        st.write(df)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

