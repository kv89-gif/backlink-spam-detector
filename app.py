
import streamlit as st
import pandas as pd
import joblib
from enhanced_feature_extractor import extract_enhanced_features

st.title("🧠 Backlink Spam Classifier v2")

@st.cache_resource
def load_model():
    return joblib.load("spam_model_v2.pkl")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload CSV of URLs", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        features = extract_enhanced_features(df, model_feature_names=model.feature_names_in_)
        preds = model.predict(features)
        df["Prediction"] = ["Spam" if p == 1 else "Not Spam" for p in preds]
        st.dataframe(df)
        st.download_button("📥 Download Results", df.to_csv(index=False).encode("utf-8"), "classified.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
