#!/usr/bin/env python3
"""
recipe_total_time_predictor_normal.py
Train a regression model to predict total cooking time (in minutes)
by combining:
  a) Extraction of actionable terms from recipe steps (free‑form text),
  b) Association of default times with these actions,
  c) Extraction of explicit numerical times stated in the steps,
  d) A non‑conservative estimate: sum of explicit times and default times,
integrated with numeric prep and cook times.
"""

import re
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────── CONFIG ───────────────────────────────────
CSV_PATH = Path("recipes.csv")  # New dataset file
MODEL_PATH = Path("recipe_total_time_predictor_normal.joblib")
# ──────────────────────────────────────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm")

DEFAULT_DURATIONS = {
    "bake": 30,
    "simmer": 20,
    "boil": 15,
    "fry": 10,
    "grill": 20,
    "mix": 5,
    "stir": 2,
    "chop": 5,
    "whisk": 3,
    "knead": 10,
    "preheat": 10,
    "marinate": 30,
    "roast": 40,
    "steam": 15,
    "sauté": 5,
    "cook": 10
}

TIME_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)", re.IGNORECASE)

class StepsTimeEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, default_durations=DEFAULT_DURATIONS, time_regex=TIME_REGEX):
        self.default_durations = default_durations
        self.time_regex = time_regex

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        estimated_times = []
        for text in X:
            if not isinstance(text, str):
                estimated_times.append(0.0)
                continue

            explicit_time_sum = 0.0
            for match in self.time_regex.finditer(text):
                value = float(match.group(1))
                unit = match.group(2).lower()
                if "hour" in unit or "hr" in unit:
                    value *= 60
                explicit_time_sum += value

            doc = nlp(text)
            default_time_sum = 0.0
            for token in doc:
                if token.pos_ == "VERB":
                    action = token.lemma_.lower()
                    if action in self.default_durations:
                        default_time_sum += self.default_durations[action]

            estimated = explicit_time_sum + default_time_sum
            estimated_times.append(estimated)
        return np.array(estimated_times).reshape(-1, 1)

def parse_time(time_str):
    if not isinstance(time_str, str):
        return None
    time_str = time_str.lower().strip()
    hr_match = re.search(r'(\d+)\s*hr', time_str)
    min_match = re.search(r'(\d+)\s*min', time_str)
    
    minutes = 0
    if hr_match:
        minutes += int(hr_match.group(1)) * 60
    if min_match:
        minutes += int(min_match.group(1))
    return minutes if minutes > 0 else None

def build_dataframe(csv_path: Path) -> pd.DataFrame:
    print("📥 Loading CSV …")
    df = pd.read_csv(csv_path, usecols=["steps", "prep_time", "cook_time", "total_time"])
    
    # Convert "steps" to a lower-case string.
    df["steps"] = df["steps"].astype(str).str.lower()
    
    # Convert prep_time and cook_time to numeric and fill missing values with 0.
    df["prep_time"] = pd.to_numeric(df["prep_time"], errors="coerce").fillna(0)
    df["cook_time"] = pd.to_numeric(df["cook_time"], errors="coerce").fillna(0)
    
    # Convert total_time:
    # Try converting to float; if that fails, use parse_time on the string.
    def convert_total_time(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return parse_time(str(x))
    
    df["total_time"] = df["total_time"].apply(convert_total_time)
    
    print("Before dropping NA:")
    print(df.head())
    print("Non-null counts:\n", df[['steps', 'prep_time', 'cook_time', 'total_time']].count())
    
    # Drop rows missing steps or total_time.
    df = df.dropna(subset=["steps", "total_time"]).copy()
    print(f"🥘 Total recipes for training after dropna: {len(df):,}")
    return df


def train_model(df: pd.DataFrame) -> Pipeline:
    X = df[["steps", "prep_time", "cook_time"]]
    y = df["total_time"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("steps_estimated_time", StepsTimeEstimator(), "steps"),
            ("numeric", "passthrough", ["prep_time", "cook_time"]),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("🚂 Fitting the regression model …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print("\n🔎 Validation Metrics")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    return pipeline

def save_model(model: Pipeline, path: Path) -> None:
    joblib.dump(model, path)
    print(f"\n✅ Model saved to {path}")

def predict_total_time(model, steps: str, prep_time: float, cook_time: float) -> float:
    steps_text = str(steps).lower()
    input_df = pd.DataFrame({
        "steps": [steps_text],
        "prep_time": [prep_time],
        "cook_time": [cook_time],
    })
    return model.predict(input_df)[0]

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}. Please ensure the CSV is in the working directory.")
    df = build_dataframe(CSV_PATH)
    model = train_model(df)
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()

