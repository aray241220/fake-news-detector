import streamlit as st

from train import predict_news


st.set_page_config(page_title="Fake News Detector", page_icon="??", layout="wide")
st.title("Fake News Detector")
st.write("Paste a news article and the model will estimate whether it is fake or real.")

news_text = st.text_area("News content", height=220, placeholder="Enter the article title and body here...")

if st.button("Classify"):
    if not news_text.strip():
        st.warning("Please enter some text before classifying.")
    else:
        result = predict_news(news_text)
        st.success(f"Prediction: {result['label']}")
        st.metric("Confidence", f"{result['confidence']}%")
        st.bar_chart({"Fake": [result["probabilities"]["Fake"]], "Real": [result["probabilities"]["Real"]]})
