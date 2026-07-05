import streamlit as st
import joblib
import os

MODEL_PATH = "model/fake_news_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("Fake News Detector")
st.write("Enter a news headline or article to predict whether it is fake or real.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train the model first.")
    st.stop()

model = load_model()

text = st.text_area("Enter news text:", height=200)

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([text])[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([text])[0]
            confidence = max(probability) * 100
        else:
            confidence = None

        if prediction == "FAKE":
            st.error("Prediction: FAKE")
        else:
            st.success("Prediction: REAL")

        if confidence:
            st.write(f"Confidence: {confidence:.2f}%")
