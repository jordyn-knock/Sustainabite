import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle
from typing import List, Dict, Tuple, Any
import os
import gzip
import random

from userinputs import get_user_preferences
from test_data import TEST_IMAGES  # <--- Import your hardcoded data

# Substitutions dictionary
SUBSTITUTIONS = {
    "milk": ["almond milk", "soy milk", "oat milk", "coconut milk"],
    "butter": ["margarine", "coconut oil", "olive oil"],
    "sugar": ["honey", "maple syrup", "agave nectar", "brown sugar"],
    "egg": ["egg substitute", "flax egg"],
    "flour": ["almond flour", "coconut flour", "whole wheat flour", "gluten-free flour"],
    "salt": ["sea salt", "kosher salt"],
    "baking powder": ["baking soda"],
    "cream": ["coconut cream", "cashew cream", "sour cream"],
    "vanilla extract": ["vanilla bean", "vanilla paste"],
}

ALIASES = {
    "flour": ["all-purpose flour", "white flour", "whole wheat flour"],
    "milk": ["whole milk", "2% milk", "skim milk"],
    "pasta": ["penne", "rigatoni", "spaghetti", "fettuccine", "farfalle", "macaroni"],
    "onion": ["red onion", "white onion", "yellow onion"],
    "garlic": ["garlic cloves", "minced garlic"],
    "pepper": ["black pepper", "ground pepper"],
    "oil": ["canola oil", "vegetable oil"],
    "cheese": ["colby jack", "monterey jack", "cheddar", "mozzerella", "vegan cheese", "parmesian"],
    "tomato sauce": ["marinara", "pizza sauce"],
    "hot sauce": ["sriracha", "franks hot sauce","buffalo sauce", "chilli oil"]
}

DEFAULT_INGREDIENTS = {"salt", "water", "oil", "pepper", "warm water", "salt and pepper", "salt pepper"}

# Dummy classes for development without model files
class DummyInferenceFunction:
    def __call__(self, **kwargs):
        batch_size = kwargs.get("full_text").shape[0]
        return {"output_0": tf.random.uniform((batch_size, 100))}
    
    @property
    def structured_input_signature(self):
        return ((),
                {
                    'full_text': tf.TensorSpec(shape=(None, 1), dtype=tf.string),
                    'servings':  tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                })
    
    @property
    def structured_outputs(self):
        return {'output_0': tf.TensorSpec(shape=(None, 100), dtype=tf.float32)}

class DummyLabelEncoder:
    @property
    def classes_(self):
        # Create 100 dummy recipe IDs
        return np.array(['recipe' + str(i) for i in range(1, 101)])

class DummyClassifier:
    def predict(self, X):
        # Randomly pick cuisine or time
        cuisines = ["italian", "mexican", "chinese", "indian", "american", "french", "thai", "japanese"]
        times = ["under_15", "under_30", "under_60", "any"]
        
        if len(X) == 0:
            return []
        
        if isinstance(X[0], str) and any(cuisine in X[0].lower() for cuisine in cuisines):
            return [random.choice(cuisines) for _ in range(len(X))]
        else:
            return [random.choice(times) for _ in range(len(X))]

# from get_dummy_recipe_db import get_dummy_recipe_db  # If you have a separate function or keep inline

def expand_with_aliases(ingredient_set: set) -> set:
    expanded = set(ingredient_set)
    for ingr in ingredient_set:
        for key, alias_list in ALIASES.items():
            if ingr == key or ingr in alias_list:
                expanded.add(key)
                expanded.update(alias_list)
    return expanded

def ingredient_match_score(user_ingredients: List[str],
                           recipe_ingredients: List[str],
                           substitutions_allowed: bool,
                           user_willing_to_buy_more: bool,
                           beta: float = 0.2) -> float:
    user_explicit = set(i.lower().strip() for i in user_ingredients)
    augmented_user_set = user_explicit | DEFAULT_INGREDIENTS
    recipe_non_default = {
        r.lower().strip() for r in recipe_ingredients
        if r.lower().strip() not in DEFAULT_INGREDIENTS
    }

    # If user is not willing to buy more, require all non-default ingredients in user's set
    if not user_willing_to_buy_more:
        if not recipe_non_default.issubset(user_explicit):
            return 0.0
        else:
            return 1.0

    total = len(recipe_ingredients)
    if total == 0:
        return 1.0

    matched = 0.0
    for ingr in recipe_ingredients:
        ingr_norm = ingr.lower().strip()
        if ingr_norm in augmented_user_set:
            matched += 1.0
        elif substitutions_allowed and ingr_norm in SUBSTITUTIONS:
            if any(sub.lower().strip() in augmented_user_set for sub in SUBSTITUTIONS[ingr_norm]):
                matched += 0.5

    base_score = matched / total
    if user_explicit:
        used_explicit = sum(1 for ingr in recipe_ingredients if ingr.lower().strip() in user_explicit)
        bonus = used_explicit / len(user_explicit)
    else:
        bonus = 0.0

    bonus_factor = 1 + beta * bonus
    final_score = base_score * bonus_factor
    return min(final_score, 1.0)

def full_coverage(recipe_ingredients, user_explicit):
    non_default = {
        i.lower().strip()
        for i in recipe_ingredients
        if i.lower().strip() not in DEFAULT_INGREDIENTS
    }
    return non_default.issubset(user_explicit)

def find_file(filenames, directories):
    for directory in directories:
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                return path
    return None

def load_model_files():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(this_dir, "recipe_model_tf")
    if not os.path.exists(model_dir):
        raise FileNotFoundError("SavedModel directory not found: " + model_dir)

    model = tf.saved_model.load(model_dir)
    infer = model.signatures["serving_default"]
    print("Signature inputs:", infer.structured_input_signature)
    print("Signature outputs:", infer.structured_outputs)

    models_dir = this_dir
    ai_models_dir = this_dir

    # Label Encoder
    le_path = find_file(["le_recipe.pkl"], [model_dir, ai_models_dir])
    if not le_path:
        raise FileNotFoundError("Label encoder file not found")
    with open(le_path, "rb") as f:
        le_recipe = pickle.load(f)
    print(f"Loaded label encoder with {len(le_recipe.classes_)} classes")

    # Cuisine Classifier
    cuisine_path = find_file(["cuisine_clf.joblib"], [models_dir, ai_models_dir])
    if not cuisine_path:
        raise FileNotFoundError("Cuisine classifier not found")
    cuisine_clf = joblib.load(cuisine_path)
    print("Loaded cuisine classifier")

    # Time Classifier
    time_clf_path = find_file(["time_tag_classifier.joblib"], [models_dir, ai_models_dir])
    if not time_clf_path:
        raise FileNotFoundError("Time classifier not found")
    time_clf = joblib.load(time_clf_path)
    print("Loaded time classifier")

    # Recipe Database
    recipe_db_path = find_file(["recipe_database.pkl.gz"], [models_dir, ai_models_dir])
    print(f"🕵️ Attempting to load recipe database from: {recipe_db_path}")

    if recipe_db_path:
        with gzip.open(recipe_db_path, "rb") as f:
            recipe_db = pickle.load(f)
        print(f"Loaded recipe database from gzip pickle with {len(recipe_db)} recipes")
    else:
        recipe_path = find_file(["recipes_ingredients.csv"], [models_dir, ai_models_dir])
        if not recipe_path:
            raise FileNotFoundError("Recipe database not found")
        recipe_db = pd.read_csv(recipe_path)
        print(f"Loaded recipe database from CSV with {len(recipe_db)} recipes")

    return infer, le_recipe, cuisine_clf, time_clf, recipe_db

def get_recommendations():
    """
    If the user has a recognized test image, return the 5 hard-coded recipes from TEST_IMAGES
    (with the first as 'top recipe' and the remaining four as 'other recipes').
    Otherwise, run the normal ML pipeline for recommendations.
    """
    try:
        prefs = get_user_preferences()
        print(f"User preferences: {prefs}")

        # Check if there's a known test image
        test_image_name = prefs.get("test_image_name")
        if test_image_name and test_image_name in TEST_IMAGES:
            # Hard-coded path: skip the ML pipeline
            print(f"Detected test image: {test_image_name}. Returning 5 hard-coded recipes.")
            
            all_recipes = TEST_IMAGES[test_image_name].get("recipes", [])
            if not all_recipes:
                print("No hard-coded recipes found for this image.")
                return None, None

            # The first item is 'top_recipe'
            top_data = all_recipes[0]
            # Others are 'other_recs'
            others_data = all_recipes[1:]
            
            # Convert them into a structure similar to DB-based pipeline
            # so your UI can display them consistently.

            top_recipe = {
                "name": top_data.get("title", "No Title"),
                "ingredients": [f"{i.get('amount','')} {i.get('name','')}" 
                                for i in top_data.get("ingredients", [])],
                "steps": str(top_data.get("steps", [])),
                "ingredient_score": 1.0,  # Arbitrary
                "final_score": 1.0
            }
            
            # Build "other_recs" as a DataFrame to match your UI approach
            # Each row can have a 'name' and 'ingredient_score'
            rows = []
            score = 0.9  # Decrement for each to mimic a ranking
            for recipe_obj in others_data:
                row = {
                    "name": recipe_obj.get("title", "No Title"),
                    "ingredient_score": score
                }
                score -= 0.1
                rows.append(row)
            other_recs = pd.DataFrame(rows)
            
            return top_recipe, other_recs

        # Otherwise, run the normal ML-based pipeline
        infer, le_recipe, cuisine_clf, time_clf, recipe_db = load_model_files()

        selected_cuisine = prefs.get("cuisine", "any cuisine").lower()
        max_time = prefs.get("max_time", 60)
        user_ingredients = prefs.get("ingredients", [])
        allow_substitutions = prefs.get("allow_substitutions", False)
        use_grocery = prefs.get("use_grocery", False)

        print(f"Looking for {selected_cuisine} recipes under {max_time} minutes")
        print(f"User’s ingredients: {user_ingredients}")

        # Convert stringy 'ingredients' columns, prepare text
        recipe_db['ingredients'] = recipe_db['ingredients'].apply(
            lambda x: eval(x) if isinstance(x, str) else x if isinstance(x, list) else []
        )
        recipe_db['ingredients_text'] = recipe_db['ingredients'].apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )
        recipe_db['steps_text'] = recipe_db['steps'].apply(
            lambda x: " ".join(eval(x)) if isinstance(x, str) else (
                " ".join(x) if isinstance(x, list) else ""
            )
        )
        recipe_db['full_text'] = recipe_db['ingredients_text'] + " " + recipe_db['steps_text']

        # 1. Predict cuisine in batches
        batch_size = 500
        cuisines = []
        for i in range(0, len(recipe_db), batch_size):
            batch = recipe_db['ingredients_text'].iloc[i:i+batch_size].tolist()
            predictions = cuisine_clf.predict(batch)
            cuisines.extend(predictions)
        recipe_db['predicted_cuisine'] = cuisines

        # 2. Predict cooking time
        time_preds = time_clf.predict(recipe_db['name'].astype(str).tolist())
        recipe_db['predicted_time'] = time_preds

        # 3. Convert user’s max_time to a label
        if max_time <= 15:
            time_label = "under_15"
        elif max_time <= 30:
            time_label = "under_30"
        elif max_time <= 60:
            time_label = "under_60"
        else:
            time_label = "any"

        # Filter by predicted cuisine & time
        if selected_cuisine != "any cuisine":
            recipe_db = recipe_db[recipe_db['predicted_cuisine'].str.lower() == selected_cuisine]
        if time_label != "any":
            recipe_db = recipe_db[recipe_db['predicted_time'] == time_label]

        print(f"Filtered to {len(recipe_db)} recipes for cuisine='{selected_cuisine}' and time='{time_label}'")
        if recipe_db.empty:
            print("No matching recipes found")
            return None, None

        # 4. Compute ingredient match scores
        user_explicit = {i.lower().strip() for i in user_ingredients}
        scores, covers = [], []
        for _, recipe_row in recipe_db.iterrows():
            ingr_list = recipe_row['ingredients']
            try:
                raw_ingr = recipe_row.get('ingredients_raw')
                if isinstance(raw_ingr, str):
                    raw_ingr = eval(raw_ingr)
                if isinstance(raw_ingr, list):
                    ingr_list = list(set(ingr_list) | set(raw_ingr))
            except Exception as e:
                print(f"Error processing ingredients_raw for recipe {recipe_row['id']}: {e}")
            
            score = ingredient_match_score(user_ingredients, ingr_list,
                                           allow_substitutions, use_grocery)
            scores.append(score)
            covers.append(full_coverage(ingr_list, user_explicit))

        recipe_db['ingredient_score'] = scores
        recipe_db['full_cover'] = covers

        # 5. If user won't buy more, try perfect covers
        if not use_grocery:
            candidate_recipes = recipe_db[recipe_db['full_cover']]
            if candidate_recipes.empty:
                print("No recipes use only the user’s ingredients – fallback to partial matches.")
                candidate_recipes = recipe_db
        else:
            candidate_recipes = recipe_db

        MIN_INGREDIENT_SCORE = 0.15
        MAX_CANDIDATES = 1500

        if not use_grocery and candidate_recipes is recipe_db:
            candidate_recipes = candidate_recipes[
                candidate_recipes['ingredient_score'] >= MIN_INGREDIENT_SCORE
            ]

        candidate_recipes = candidate_recipes.nlargest(MAX_CANDIDATES, 'ingredient_score').reset_index(drop=True)
        if candidate_recipes.empty:
            print("No recipes met the ingredient-score threshold; fallback to top 500.")
            candidate_recipes = recipe_db.nlargest(500, 'ingredient_score').reset_index(drop=True)
            if candidate_recipes.empty:
                print("No suitable recipes found at all.")
                return None, None

        print(f"→ Passing {len(candidate_recipes)} recipes to the model")
        # 6. Model inference
        X_text = candidate_recipes['full_text'].values
        X_text_tensor = tf.constant(X_text.reshape(-1, 1), dtype=tf.string)
        servings_tensor = tf.constant([[4.0]] * len(candidate_recipes), dtype=tf.float32)

        inputs = {"full_text": X_text_tensor, "servings": servings_tensor}
        pred_matrix = infer(**inputs)["output_0"].numpy()

        # 7. Attach model scores
        recipe_ids = le_recipe.classes_
        candidate_recipes['model_score'] = 0.0

        for i, recipe_id in enumerate(candidate_recipes['id']):
            idx = np.where(recipe_ids == recipe_id)[0]
            if idx.size:
                candidate_recipes.loc[candidate_recipes['id'] == recipe_id, 'model_score'] = pred_matrix[i, idx[0]]

        # 8. Final ranking
        candidate_recipes['final_score'] = (
            candidate_recipes['ingredient_score'] * 0.95 +
            candidate_recipes['model_score']     * 0.05
        )
        ranked_recipes = candidate_recipes.sort_values(by='final_score', ascending=False)

        if not ranked_recipes.empty:
            top_recipe = ranked_recipes.iloc[0]
            other_recipes = ranked_recipes.iloc[1:11]
            return top_recipe, other_recipes
        else:
            print("No suitable recipes found after final ranking.")
            return None, None

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("Testing recipe recommendation system")
    top_recipe, other_recipes = get_recommendations()

    if top_recipe is not None:
        print("\n--- TOP RECOMMENDATION ---")
        print(f"Recipe: {top_recipe['name']}")
        print(f"\nIngredients: {top_recipe['ingredients']}")
        print(f"\nIngredient match score: {top_recipe.get('ingredient_score', 0):.2f}")
        print(f"\nSteps: {top_recipe['steps']}")
        
        if other_recipes is not None and len(other_recipes) > 0:
            print("\n--- OTHER RECOMMENDATIONS ---")
            for _, recipe in other_recipes.iterrows():
                print(f"- {recipe['name']} (Score: {recipe.get('ingredient_score', 0):.2f})")
