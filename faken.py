"""
Fake News Detector - basic ML text classification
Model: TF-IDF features + Logistic Regression (scikit-learn)

Setup:
    pip install scikit-learn pandas joblib

Dataset (free): "Fake and Real News Dataset" on Kaggle (ISOT) -> Fake.csv + True.csv
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Usage:
    python fake_news_detector.py train --fake Fake.csv --true True.csv
    python fake_news_detector.py train --demo            # tiny built-in toy data
    python fake_news_detector.py predict "Paste an article or headline here"
    python fake_news_detector.py predict --file article.txt
"""
import argparse
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("fake_news_model.joblib")
LABELS = {0: "REAL", 1: "FAKE"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Lowercase, drop URLs/punctuation/numbers, and remove the agency tag."""
    text = str(text)
    # Real articles in this dataset start with "CITY (Reuters) -": a giveaway
    # the model would memorise instead of learning real patterns. Remove it.
    text = re.sub(r"^.{0,120}?\(reuters\)\s*-?\s*", " ", text, flags=re.I)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def load_dataset(fake_csv: str, true_csv: str) -> pd.DataFrame:
    fake = pd.read_csv(fake_csv)
    true = pd.read_csv(true_csv)
    fake["label"], true["label"] = 1, 0  # 1 = fake, 0 = real
    df = pd.concat([fake, true], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)
    df = df[df["content"].str.len() > 20].drop_duplicates(subset="content")
    return df[["content", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)


def demo_dataset() -> pd.DataFrame:
    """Tiny toy data so the script runs without downloading anything.
    Do NOT trust a model trained on this; it only demonstrates the pipeline."""
    fake = [
        "SHOCKING secret cure doctors don't want you to know about, share before it's deleted",
        "Celebrity caught in unbelievable scandal, you won't believe what happened next",
        "Government secretly puts mind control chemicals in tap water, insiders reveal",
        "Miracle fruit melts belly fat overnight, big pharma furious",
        "Aliens landed in the capital and officials are covering it up, anonymous source says",
        "BREAKING: famous politician is a robot, leaked video proves it",
        "Scientists HATE this one weird trick that makes you live forever",
        "Banks plan to cancel all cash tomorrow, withdraw your money NOW",
    ]
    real = [
        "The central bank held interest rates steady on Wednesday, citing stable inflation",
        "City council approves budget for new public library and road repairs",
        "Researchers publish study on the effects of sleep on student exam performance",
        "Parliament passes bill to expand funding for rural healthcare clinics",
        "Local team wins regional championship after a close final match on Sunday",
        "Company reports quarterly earnings slightly above analyst expectations",
        "Weather service forecasts heavy rainfall across the coast this weekend",
        "University announces new scholarship program for undergraduate applicants",
    ]
    df = pd.DataFrame({"content": fake + real, "label": [1] * len(fake) + [0] * len(real)})
    df["content"] = df["content"].map(clean_text)
    return df


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_pipeline(min_df: int = 2) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # single words and word pairs
            max_df=0.7,           # ignore words in >70% of documents
            min_df=min_df,        # ignore very rare words
            max_features=50_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=2.0)),
    ])


def show_top_words(model: Pipeline, n: int = 10) -> None:
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    order = coefs.argsort()
    print(f"\nWords pushing toward REAL: {', '.join(vocab[order[:n]])}")
    print(f"Words pushing toward FAKE: {', '.join(vocab[order[-n:]][::-1])}")


def train(df: pd.DataFrame, demo: bool = False) -> None:
    print(f"Loaded {len(df)} articles ({int(df.label.sum())} fake, {int((df.label == 0).sum())} real)")
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    model = build_pipeline(min_df=1 if demo else 2)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted) [REAL, FAKE]:")
    print(confusion_matrix(y_test, preds))
    print("\n" + classification_report(y_test, preds, target_names=["REAL", "FAKE"], zero_division=0))
    show_top_words(model)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def predict(text: str) -> None:
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run the 'train' command first.")
    model = joblib.load(MODEL_PATH)
    cleaned = clean_text(text)
    if len(cleaned) < 10:
        sys.exit("Text is too short to classify.")
    proba = model.predict_proba([cleaned])[0]  # [P(real), P(fake)]
    label = int(proba.argmax())
    print(f"\nPrediction : {LABELS[label]}")
    print(f"Confidence : {proba[label]:.1%}  (real {proba[0]:.1%} | fake {proba[1]:.1%})")
    print("\nNote: this flags writing patterns similar to its training data. "
          "It does not verify facts. Always check reliable sources.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Fake News Detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and save a model")
    t.add_argument("--fake", default="Fake.csv")
    t.add_argument("--true", default="True.csv")
    t.add_argument("--demo", action="store_true", help="use tiny built-in toy data")

    p = sub.add_parser("predict", help="classify a piece of text")
    p.add_argument("text", nargs="?", help="article text or headline")
    p.add_argument("--file", help="read the text from a file instead")

    args = parser.parse_args()

    if args.cmd == "train":
        if args.demo:
            train(demo_dataset(), demo=True)
        else:
            if not (Path(args.fake).exists() and Path(args.true).exists()):
                sys.exit("Fake.csv / True.csv not found. Download the dataset (see top of file) "
                         "or run with --demo.")
            train(load_dataset(args.fake, args.true))
    else:
        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        if not text:
            sys.exit("Provide text or --file.")
        predict(text)


if __name__ == "__main__":
    main()
