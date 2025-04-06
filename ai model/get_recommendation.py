import pandas as pd
import ast
import re
import pickle
import numpy as np
import json
from typing import List
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Import user preferences from your UI module.
from userinputs import get_user_preferences

# -----------------------------
# Substitutions dictionary with many examples
# -----------------------------
SUBSTITUTIONS = {
    "milk": ["almond milk", "soy milk", "oat milk", "coconut milk"],
    "butter": ["margarine", "coconut oil", "olive oil"],
    "sugar": ["honey", "maple syrup", "agave nectar", "brown sugar"],
    "egg": ["egg substitute"],
    "flour": ["almond flour", "coconut flour", "whole wheat flour"],
    "salt": ["sea salt", "kosher salt"],
    "baking powder": ["baking soda"],
    "cheddar": ["colby jack", "monterey jack"],
    "cream": ["coconut cream", "cashew cream", "sour cream"],
    "vanilla extract": ["vanilla bean", "vanilla paste"],
    "oil": ["canola oil", "vegetable oil"],
    # Add more substitutions as needed.
}

def ingredient_match_score(user_ingredients: List[str], recipe_ingredients: List[str], substitutions_allowed: bool) -> float:
    """
    Compute a match score between the user's ingredients and a recipe's ingredients.
    - Direct matches count as 1.
    - Substitute matches (if allowed) count as 0.5.
    Returns a normalized score between 0 and 1.
    """
    user_set = set([i.lower().strip() for i in user_ingredients])
    score = 0
    for ingr in recipe_ingredients:
        ingr_norm = ingr.lower().strip()
        if ingr_norm in user_set:
            score += 1
        elif substitutions_allowed and ingr_norm in SUBSTITUTIONS:
            if any(sub in user_set for sub in SUBSTITUTIONS[ingr_norm]):
                score += 0.5
    return score / len(recipe_ingredients) if recipe_ingredients else 0

# -----------------------------
# Helper function to load saved recommendation model and preprocessing objects.
# -----------------------------
def load_preprocessing_objects():
    model = load_model("recipe_model.h5")
    with open("le_recipe.pkl", "rb") as f:
        le_recipe = pickle.load(f)
    vectorizer = tf.saved_model.load("text_vectorizer")
    try:
        @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
        def call_vectorizer(x):
            return vectorizer(x)
        # Call the wrapped function with a dummy input to warm up the lookup table.
        dummy_out = call_vectorizer(tf.constant(["dummy"]))
        print("Vectorizer warmed up successfully with tf.function wrapper.")
    except Exception as e:
        print("Failed to warm up vectorizer:", e)
    return model, le_recipe, vectorizer



# -----------------------------
# Helper to safely evaluate stringified lists.
# -----------------------------
def safe_eval(obj) -> List[str]:
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, list):
        return obj
    text = str(obj)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    try:
        json_text = text.replace("'", '"')
        return json.loads(json_text)
    except json.JSONDecodeError:
        return []

# -----------------------------
# Main function: Generate Recommendations
# -----------------------------
def get_recommendations():
    # Get user preferences from the UI.
    prefs = get_user_preferences()
    # Expected keys: max_time, cuisine, meal_type, servings, use_grocery, allow_substitutions, and (optionally) ingredients.
    selected_cuisine = prefs["cuisine"].lower()
    USER_TO_CATEGORY = {
        "Breakfast": "breakfast",
        "Full Meal": "meals",
        "Sweet Treat": "sweet treat",
        "Snack": "snacks"
    }
    selected_meal_type = USER_TO_CATEGORY.get(prefs["meal_type"], "unknown")
    max_time = prefs["max_time"]
    # For ingredient matching, expect a list of ingredients.
    user_ingredients = prefs.get("ingredients", [])
    allow_substitutions = prefs.get("allow_substitutions", False)
    use_grocery = prefs.get("use_grocery", False)
    
    # For testing, if no ingredients are provided, use an example list.
    if not user_ingredients:
        user_ingredients = [
            "flour", "milk", "egg", "sugar", "butter",
            "vanilla extract", "baking powder", "salt", "oil"
        ]
        print("Using example user ingredient list.")
    
    # Load raw recipes.
    df = pd.read_csv('ai model/recipes_ingredients.csv')
    print("Total recipes loaded:", df.shape)
    
    # Preprocess raw data.
    for col in ['ingredients', 'ingredients_raw', 'steps']:
        df[col] = df[col].fillna("[]")
    df['ingredients'] = df['ingredients'].apply(safe_eval)
    df['ingredients_raw'] = df['ingredients_raw'].apply(safe_eval)
    df['steps'] = df['steps'].apply(safe_eval)
    
    def join_text(lst):
        return " ".join(lst)
    
    df['ingredients_text'] = df['ingredients'].apply(join_text)
    df['ingredients_raw_text'] = df['ingredients_raw'].apply(join_text)
    df['steps_text'] = df['steps'].apply(join_text)
    df['full_text'] = df['ingredients_text'] + " " + df['ingredients_raw_text'] + " " + df['steps_text']
    df['servings'] = pd.to_numeric(df['servings'], errors='coerce')
    df = df.dropna(subset=['servings'])
    print("Preprocessed recipes shape:", df.shape)
    
    # -----------------------------
    # Annotate recipes with pre-trained classifiers using batch predictions.
    # -----------------------------
    batch_size = 1000
    # Load Cuisine Bot.
    cuisine_clf = joblib.load("cuisine_clf.joblib")
    cuisine_texts = df["ingredients_raw_text"].tolist()
    predicted_cuisines = []
    for i in tqdm(range(0, len(cuisine_texts), batch_size), desc="Predicting Cuisine"):
        batch = cuisine_texts[i:i+batch_size]
        batch_preds = cuisine_clf.predict(batch)
        predicted_cuisines.extend(batch_preds)
    df["predicted_cuisine"] = predicted_cuisines
    
    # Load Meal Bot.
    category_clf = joblib.load("recipe_category_clf.joblib")
    full_texts = df["full_text"].tolist()
    predicted_meal_types = []
    for i in tqdm(range(0, len(full_texts), batch_size), desc="Predicting Meal Type"):
        batch = full_texts[i:i+batch_size]
        batch_preds = category_clf.predict(batch)
        predicted_meal_types.extend(batch_preds)
    df["predicted_meal_type"] = predicted_meal_types
    
    # Filter recipes by selected cuisine and meal type.
    candidates = df[
        (df["predicted_cuisine"].str.lower() == selected_cuisine) &
        (df["predicted_meal_type"].str.lower() == selected_meal_type)
    ]
    print(f"Candidate recipes after filtering: {candidates.shape}")
    
    # (Optional) Filter further by max_time using a Time Bot here.
    
    # Load recommendation model and preprocessing objects.
    model, le_recipe, vectorizer = load_preprocessing_objects()
    
    # Generate model predictions for candidate recipes (optional for ranking).
    X_text = candidates["full_text"].values
    X_servings = candidates["servings"].values.reshape(-1, 1).astype(np.float32)
    preds = model.predict([X_text, X_servings])
    
    # Compute ingredient match score for each candidate.
    scores = []
    for idx, row in candidates.iterrows():
        recipe_ingredients = row["ingredients"]
        score = ingredient_match_score(user_ingredients, recipe_ingredients, substitutions_allowed=allow_substitutions)
        scores.append(score)
    candidates["ingredient_score"] = scores
    
    # If the user isn't willing to buy extra ingredients, filter by a threshold.
    if not use_grocery:
        candidates = candidates[candidates["ingredient_score"] >= 0.8]
    
    # Rank candidates by ingredient score (descending).
    ranked_candidates = candidates.sort_values(by="ingredient_score", ascending=False)
    
    # Select the top recipe and 10 additional recommendations.
    top_recipe = ranked_candidates.iloc[0]
    other_recipes = ranked_candidates.iloc[1:11]
    
    return top_recipe, other_recipes

if __name__ == "__main__":
    best_recipe, recommendations = get_recommendations()
    print("Best Recipe:")
    print(best_recipe)
    print("\nOther Recommendations:")
    print(recommendations)
