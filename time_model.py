#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import ast

# Config
EXCEL_PATH = Path("recipes_ingredients.csv")
MODEL_PATH = Path("time_tag_classifier.joblib")

def clean_and_label(df):
    df.columns = df.columns.str.lower().str.strip()

    def label_from_tag(tag):
        try:
            tags = ast.literal_eval(tag)
        except:
            return "over_60"  # fallback if it's badly formatted

        tags = [t.lower() for t in tags]

        if "15-minutes-or-less" in tags:
            return "under_15"
        elif "30-minutes-or-less" in tags:
            return "under_30"
        elif "60-minutes-or-less" in tags:
            return "under_60"
        else:
            return "over_60"

    df["time_label"] = df["tags"].apply(label_from_tag)
    return df


def main():
    print("📖 Reading Excel file …")
    df = pd.read_csv(EXCEL_PATH)

    df["tags"] = df["tags"].astype(str)

    df = clean_and_label(df)
    
    print("\n🔍 Sample tags:")
    print(df["tags"].head(10))

    print("\n🧪 Unique tags:")
    print(df["tags"].dropna().unique())

    print("\n🍴 Label distribution:")
    print(df["time_label"].value_counts())


    if "name" not in df.columns:
        raise ValueError("Expected a 'name' column for recipe titles.")

    X = df["name"].astype(str)
    y = df["time_label"]

    print("🧪 Splitting data …")
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=300))
    ])

    print("🚂 Training model …")
    clf.fit(X_train, y_train)

    print("📊 Classification report:")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

