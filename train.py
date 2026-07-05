from __future__ import annotations

import pickle
import re
import shutil
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"

TRUE_FILE_CANDIDATES = ["True.csv", "true.csv"]
FAKE_FILE_CANDIDATES = ["False.csv", "false.csv", "Fake.csv", "fake.csv"]


def ensure_nltk_resources() -> None:
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def find_dataset_file(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = DATA_DIR / candidate
        if path.exists():
            return path
    return None


def ensure_dataset_files() -> None:
    true_path = find_dataset_file(TRUE_FILE_CANDIDATES)
    fake_path = find_dataset_file(FAKE_FILE_CANDIDATES)

    if true_path and fake_path:
        return

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("kagglehub is not installed. Please place the CSV files in the data directory.")

    download_dir = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    downloaded_files = list(Path(download_dir).glob("*.csv"))
    if not downloaded_files:
        raise RuntimeError("The Kaggle dataset download did not contain any CSV files.")

    for csv_path in downloaded_files:
        target_path = DATA_DIR / csv_path.name
        if not target_path.exists():
            shutil.copy2(csv_path, target_path)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokens = [stemmer.stem(token) for token in text.split() if token not in stop_words]
    return " ".join(tokens)


def load_dataset() -> pd.DataFrame:
    ensure_nltk_resources()
    ensure_dataset_files()

    true_path = find_dataset_file(TRUE_FILE_CANDIDATES)
    fake_path = find_dataset_file(FAKE_FILE_CANDIDATES)

    if true_path is None or fake_path is None:
        raise FileNotFoundError("Expected dataset files were not found. Place True.csv and False.csv/Fake.csv in the data folder.")

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["label"] = 1
    fake_df["label"] = 0

    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    combined_df = combined_df.fillna("")

    for column in ["title", "text", "subject", "date"]:
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].astype(str)

    combined_df["content"] = combined_df[[col for col in ["title", "subject", "text"] if col in combined_df.columns]].fillna("").agg(
        lambda row: " ".join([str(item) for item in row if str(item).strip()]), axis=1
    )
    combined_df["cleaned_content"] = combined_df["content"].apply(clean_text)
    return combined_df


def train_model() -> dict[str, float]:
    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_content"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
            ("classifier", LogisticRegression(max_iter=2000, solver="liblinear")),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Validation accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=["Fake", "Real"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(pipeline, handle)

    return {"accuracy": round(float(accuracy), 4)}


def predict_news(text: str) -> dict[str, object]:
    if not MODEL_PATH.exists():
        train_model()

    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)

    cleaned_text = clean_text(text)
    prediction = int(model.predict([cleaned_text])[0])
    probabilities = model.predict_proba([cleaned_text])[0]

    label = "Real" if prediction == 1 else "Fake"
    confidence = round(float(max(probabilities)) * 100, 2)
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "Fake": round(float(probabilities[0]) * 100, 2),
            "Real": round(float(probabilities[1]) * 100, 2),
        },
    }


if __name__ == "__main__":
    train_model()
