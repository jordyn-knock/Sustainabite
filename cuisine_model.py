#!/usr/bin/env python3
"""
cuisine_model.py
Train a cuisine classifier on Food.com RAW_recipes.csv
Handles malformed or missing list strings gracefully.
"""

from pathlib import Path
from typing import List
import ast
import json
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ─────────────────────────────────── CONFIG ───────────────────────────────────
CSV_PATH = Path("RAW_recipes.csv")            # raw Kaggle file in working dir
MODEL_PATH = Path("cuisine_clf.joblib")       # where to save the trained model

KNOWN_CUISINES = [
    "italian", "mexican", "chinese", "indian", "thai", "french",
    "greek", "japanese", "american", "spanish", "moroccan",
    "vietnamese", "korean", "caribbean", "irish", "german",
]

TFIDF_MIN_DF = 5
TFIDF_MAX_FEATURES = 50_000
LOGREG_C = 4.0
# ──────────────────────────────────────────────────────────────────────────────


def safe_eval(obj) -> List[str]:
    """Turn a *stringified* Python/JSON list into a real list.

    * Silently handles NaN/None/float values by returning [].
    * Tries literal_eval first, then a JSON fallback.
    """
    # 1. short‑circuit common non‑string cases
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, list):
        return obj

    text = str(obj)

    # 2. literal_eval attempt
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass  # fall through to JSON

    # 3. JSON attempt after swapping quotes
    try:
        json_text = text.replace("'", '"')
        return json.loads(json_text)
    except json.JSONDecodeError:
        return []


def tag_to_cuisine(tags: List[str]):
    """Return the first KNOWN_CUISINE present in tags (case‑insensitive)."""
    for c in KNOWN_CUISINES:
        if any(c in str(tag).lower() for tag in tags):
            return c
    return None


def build_dataframe(csv_path: Path) -> pd.DataFrame:
    print("📥  Loading CSV …")
    df = pd.read_csv(csv_path, usecols=["name", "ingredients", "tags"])

    for col in ("ingredients", "tags"):
        df[col] = df[col].apply(safe_eval)

    df["cuisine"] = df["tags"].apply(tag_to_cuisine)
    print(f"🥘  Total recipes: {len(df):,}")

    train_df = df.dropna(subset=["cuisine"]).copy()
    print(f"🏷️   Labelled for training: {len(train_df):,}")
    return train_df


def train_model(df: pd.DataFrame) -> Pipeline:
    # Flatten ingredients list to one string per recipe
    df["ing_str"] = df["ingredients"].apply(lambda lst: " ".join(lst).lower())

    X_train, X_val, y_train, y_val = train_test_split(
        df["ing_str"],
        df["cuisine"],
        test_size=0.2,
        stratify=df["cuisine"],
        random_state=42,
    )

    clf = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                min_df=TFIDF_MIN_DF,
                ngram_range=(1, 2),
                max_features=TFIDF_MAX_FEATURES,
            ),
        ),
        (
            "logreg",
            LogisticRegression(
                max_iter=200,
                C=LOGREG_C,
                n_jobs=-1,
                class_weight="balanced",
                multi_class="ovr",
            ),
        ),
    ])

    print("🚂  Fitting TF‑IDF + Logistic Regression …")
    clf.fit(X_train, y_train)

    print("\n🔎  Validation metrics")
    print(classification_report(y_val, clf.predict(X_val)))

    return clf


def save_model(model: Pipeline, path: Path) -> None:
    joblib.dump(model, path)
    print(f"\n✅  Model saved to {path}")

def predict_cuisine(model, ingredients: List[str]) -> str:
    ing_str = " ".join(ingredients).lower()
    return model.predict([ing_str])[0]

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CSV_PATH}. Place RAW_recipes.csv in the working directory."
        )

    df = build_dataframe(CSV_PATH)
    model = train_model(df)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
