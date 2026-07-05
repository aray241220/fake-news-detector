# Fake News Detector

This project trains a text-classification model to distinguish between fake and real news articles using the Kaggle fake/real news dataset.

## Project structure

- data/ - Place the dataset CSV files here (True.csv and False.csv or Fake.csv)
- model/ - Stores the trained pipeline as fake_news_model.pkl
- train.py - Downloads the dataset if it is missing, trains the model, and saves it
- app.py - Streamlit interface for single-article classification
- requirements.txt - Python dependencies

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Notes

- The pipeline uses text cleaning, stop-word removal, stemming, TF-IDF features, and logistic regression.
- The saved model is written to model/fake_news_model.pkl.
