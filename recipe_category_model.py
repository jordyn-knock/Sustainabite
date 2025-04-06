#!/usr/bin/env python3
"""
recipe_category_model.py
Train a recipe category classifier on Food.com RAW_recipes.csv.
The classifier predicts whether a recipe is one of:
    breakfast, meals, snacks, sweet treat.
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
# CSV_PATH = Path("RAW_recipes.csv")
#  for jordyn
CSV_PATH = Path("ai model/recipes_ingredients.csv")                # raw recipes file
MODEL_PATH = Path("recipe_category_clf.joblib")   # where to save the trained model

# Define the categories and associated keywords
CATEGORY_KEYWORDS = {
    "breakfast": ["breakfast"],
    "meals": ["meals", "dinner", "lunch"],
    "snacks": ["snacks"],
    "sweet treat": ["sweet treat", "dessert"],
}

TFIDF_MIN_DF = 5
TFIDF_MAX_FEATURES = 50_000
LOGREG_C = 4.0
# ──────────────────────────────────────────────────────────────────────────────

def safe_eval(obj) -> List[str]:
    """
    Convert a stringified list into an actual list.
    Handles malformed strings by returning an empty list.
    """
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, list):
        return obj

    text = str(obj)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass  # fall back to JSON approach

    try:
        json_text = text.replace("'", '"')
        return json.loads(json_text)
    except json.JSONDecodeError:
        return []

def tag_to_category(tags: List[str]) -> str:
    """
    Return the first matching category based on CATEGORY_KEYWORDS.
    Checks each tag for any of the keywords associated with a category (case‑insensitive).
    """
    for category, keywords in CATEGORY_KEYWORDS.items():
        for tag in tags:
            tag_lower = str(tag).lower()
            for keyword in keywords:
                if keyword in tag_lower:
                    return category
    return None

def build_dataframe(csv_path: Path) -> pd.DataFrame:
    print("📥  Loading CSV …")
    df = pd.read_csv(csv_path, usecols=["name", "ingredients", "tags"])

    for col in ("ingredients", "tags"):
        df[col] = df[col].apply(safe_eval)

    # Create a new column for our target categories using the updated mapping
    df["category"] = df["tags"].apply(tag_to_category)
    print(f"🥘  Total recipes: {len(df):,}")

    # Only keep recipes that have a valid category label
    train_df = df.dropna(subset=["category"]).copy()
    print(f"🏷️   Labelled for training: {len(train_df):,}")
    return train_df

def train_model(df: pd.DataFrame) -> Pipeline:
    # Flatten ingredients list to one string per recipe
    df["ing_str"] = df["ingredients"].apply(lambda lst: " ".join(lst).lower())

    X_train, X_val, y_train, y_val = train_test_split(
        df["ing_str"],
        df["category"],
        test_size=0.2,
        stratify=df["category"],
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

def predict_category(model, ingredients: List[str]) -> str:
    """
    Predict the category for a given list of ingredients.
    """
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
