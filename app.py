import streamlit as st
import joblib
import pandas as pd
import spacy.cli

# ✅ Download model if not already available (important for Streamlit Cloud)
spacy.cli.download("en_core_web_sm")

from utils import preprocessor

# Load the model
model = joblib.load("model.joblib")

# UI
st.title("Sentiment Analyzer")
text_input = st.text_area("Enter a sentence:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        # Convert text to DataFrame for compatibility
        df = pd.Series([text_input])
        clean_df = preprocessor().fit_transform(df)
        prediction = model.predict(clean_df)
        sentiment = "Positive 😀" if prediction[0] == 1 else "Negative 😟"
        st.success(f"Sentiment: {sentiment}")
