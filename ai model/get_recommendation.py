import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle
from typing import List, Dict, Tuple, Any
import os
import gzip

from userinputs import get_user_preferences

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

SAMPLE_INGREDIENTS = ["yeast", "flour"]
DEFAULT_INGREDIENTS = {"salt", "water", "oil", "pepper", "warm water", "salt and pepper", "salt pepper"}

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
    augmented_user_set = expand_with_aliases(user_explicit | DEFAULT_INGREDIENTS)

    recipe_normalized = set()
    for r in recipe_ingredients:
        r_norm = r.lower().strip()
        recipe_normalized.add(r_norm)
        recipe_normalized = expand_with_aliases(recipe_normalized)

    if not user_willing_to_buy_more:
        if not recipe_normalized.issubset(augmented_user_set):
            return 0.0
        return 1.0

    total_recipe_ingredients = len(recipe_normalized)
    if total_recipe_ingredients == 0:
        return 1.0

    matched = 0.0
    for ingr in recipe_normalized:
        if ingr in augmented_user_set:
            matched += 1.0
        elif substitutions_allowed:
            for key, subs in SUBSTITUTIONS.items():
                if ingr == key or ingr in subs:
                    if any(sub in augmented_user_set for sub in subs + [key]):
                        matched += 0.5
                        break

    base_score = matched / total_recipe_ingredients

    # Bonus based on percentage of user's ingredients used
    used_explicit = sum(1 for ingr in user_explicit if ingr in recipe_normalized)
    user_coverage = used_explicit / len(user_explicit) if user_explicit else 0.0
    bonus_factor = 1 + beta * user_coverage

    final_score = base_score * bonus_factor
    return min(final_score, 1.0)


def load_model_files():
    """Load all required model files"""
    # Try different possible paths for files
    models_dir = "."
    ai_models_dir = "ai model"

    model_dir = os.path.join(ai_models_dir, "recipe_model_tf")
    if not os.path.exists(model_dir):
        raise FileNotFoundError("SavedModel directory not found: " + model_dir)
    
    
    # # Load main recommendation model
    # model_path = find_file(["recipe_model.h5"], [models_dir, ai_models_dir])
    # if not model_path:
    #     raise FileNotFoundError("Recipe model file not found")
    
    model = tf.saved_model.load(model_dir)
    # Extract the serving signature as 'infer'
    infer = model.signatures["serving_default"]
    print("Signature inputs:", infer.structured_input_signature)
    print("Signature outputs:", infer.structured_outputs)
   
    
    # Load label encoder
    le_path = find_file(["le_recipe.pkl"], [models_dir, ai_models_dir])
    if not le_path:
        raise FileNotFoundError("Label encoder file not found")
    
    with open(le_path, "rb") as f:
        le_recipe = pickle.load(f)
    print(f"Loaded label encoder with {len(le_recipe.classes_)} classes")

    
    # Load cuisine classifier
    cuisine_path = find_file(["cuisine_clf.joblib"], [models_dir, ai_models_dir])
    if not cuisine_path:
        raise FileNotFoundError("Cuisine classifier not found")
    
    cuisine_clf = joblib.load(cuisine_path)
    print(f"Loaded cuisine classifier")

    # Load time classifier instead of meal type
    time_clf_path = find_file(["time_tag_classifier.joblib"], [models_dir, ai_models_dir])
    if not time_clf_path:
        raise FileNotFoundError("Time classifier not found")
    time_clf = joblib.load(time_clf_path)
    print(f"Loaded time classifier")

    
    # Load meal type classifier
    # meal_path = find_file(["recipe_category_clf.joblib"], [models_dir, ai_models_dir])
    # if not meal_path:
    #     raise FileNotFoundError("Meal type classifier not found")
    
    # meal_clf = joblib.load(meal_path)
    # print(f"Loaded meal type classifier")
    
    # Load recipe database from pickle instead of CSV
    recipe_db_path = find_file(["recipe_database.pkl"], [models_dir, ai_models_dir])
    if recipe_db_path:
        # Use the pickled recipe database if available
        with gzip.open(recipe_db_path, "rb") as f:
            recipe_db = pickle.load(f)
        print(f"Loaded recipe database from pickle with {len(recipe_db)} recipes")
    else:
        # Fall back to CSV if pickle not found (for backward compatibility)
        recipe_path = find_file(["recipes_ingredients.csv"], [models_dir, ai_models_dir])
        if not recipe_path:
            raise FileNotFoundError("Recipe database not found")
        
        recipe_db = pd.read_csv(recipe_path)
        print(f"Loaded recipe database from CSV with {len(recipe_db)} recipes")
    
    return infer, le_recipe, cuisine_clf, meal_clf, recipe_db

def find_file(filenames, directories):
    """Helper function to find a file in multiple possible directories"""
    for directory in directories:
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                return path
    return None

def get_recommendations():
    """Generate recipe recommendations based on user preferences"""
    try:
        # Load all model files
        infer, le_recipe, cuisine_clf, meal_clf, recipe_db = load_model_files()
        
        # Get user preferences
        prefs = get_user_preferences()
        print(f"User preferences: {prefs}")

        # Since there is no ingredients part in user preferences, we use our sample list for testing.
        prefs["ingredients"] = SAMPLE_INGREDIENTS
        print("Using sample ingredients:", prefs["ingredients"])
        
        # Extract key preferences
        # selected_cuisine = prefs["cuisine"].lower()
        # USER_TO_CATEGORY = {
            # "Breakfast": "breakfast",
            # "Full Meal": "meals", 
            # "Sweet Treat": "sweet treat",
            # "Snack": "snacks"
        # }
        # selected_meal_type = USER_TO_CATEGORY.get(prefs["meal_type"], "unknown")
        # requested_servings = float(prefs.get("servings", 4))
        selected_cuisine = prefs["cuisine"].lower()
        user_ingredients = prefs.get("ingredients", [])
        allow_substitutions = prefs.get("allow_substitutions", True)
        use_grocery = prefs.get("use_grocery", True)
        max_time = prefs.get("max_time", 60)

        print(f"Looking for {selected_cuisine} recipes for {max_time} time")
        
        # ------------------------------
        # Step 1: Pre-filter recipes by cuisine and meal type 
        # ------------------------------
        
        # Ensure ingredients column is processed
        recipe_db['ingredients'] = recipe_db['ingredients'].apply(lambda x: 
            eval(x) if isinstance(x, str) else x if isinstance(x, list) else [])
        
        # Create the text features needed for the classifiers
        recipe_db['ingredients_raw_text'] = recipe_db['ingredients_raw'].apply(lambda x: 
            " ".join(eval(x)) if isinstance(x, str) else " ".join(x) if isinstance(x, list) else "")
        
        recipe_db['ingredients_text'] = recipe_db['ingredients'].apply(lambda x: 
            " ".join(x) if isinstance(x, list) else "")
            
        recipe_db['steps_text'] = recipe_db['steps'].apply(lambda x: 
            " ".join(eval(x)) if isinstance(x, str) else " ".join(x) if isinstance(x, list) else "")
            
        recipe_db['full_text'] = recipe_db['ingredients_text'] + " " + recipe_db['ingredients_raw_text'] + " " + recipe_db['steps_text']
        
        # Predict cuisine for all recipes
        print("Predicting cuisine for recipes...")
        batch_size = 500
        cuisines = []
        for i in range(0, len(recipe_db), batch_size):
            batch = recipe_db['ingredients_raw_text'].iloc[i:i+batch_size].tolist()
            predictions = cuisine_clf.predict(batch)
            cuisines.extend(predictions)
        recipe_db['predicted_cuisine'] = cuisines
        
        # Predict meal type for all recipes
        # print("Predicting meal type for recipes...")
        # meal_types = []
        # for i in range(0, len(recipe_db), batch_size):
        #     batch = recipe_db['full_text'].iloc[i:i+batch_size].tolist()
        #     predictions = meal_clf.predict(batch)
        #     meal_types.extend(predictions)
        # recipe_db['predicted_meal_type'] = meal_types

        print("Predicting time for recipes...")
        time_preds = []
        for i in range(0, len(recipe_db), batch_size):
            batch = recipe_db['name'].iloc[i:i+batch_size].tolist()
            batch_preds = time_tag_classifier.joblib.predict(batch)
            time_preds.extend(batch_preds)
        recipe_db['predicted_time'] = time_preds

        if selected_cuisine != "any cuisine":
            recipe_db = recipe_db[recipe_db['predicted_cuisine'].str.lower() == selected_cuisine]

        if max_time <= 15:
            time_label = "under_15"
        elif max_time <= 30:
            time_label = "under_30"
        elif max_time <= 60:
            time_label = "under_60"
        else:
            time_label = "any"

        recipe_db = recipe_db[recipe_db['predicted_time'] == time_label]
        
        # Filter by cuisine and meal type
        matching_recipes = recipe_db
        print(f"Found {len(matching_recipes)} recipes matching {selected_cuisine} {selected_meal_type}")
        
        if len(matching_recipes) == 0:
            print("No matching recipes found")
            return None, None
        
        # ------------------------------
        # Step 2: Calculate ingredient match scores
        # ------------------------------
        # ------------------------------
# Step 2: Calculate ingredient match scores
# ------------------------------
        # ------------------------------
# Step 2: Calculate ingredient match scores
# ------------------------------
        scores = []
        match_text = []
        for _, recipe in matching_recipes.iterrows():
            ingr_list = recipe['ingredients']
            try:
                raw_ingr = recipe.get('ingredients_raw')
                if isinstance(raw_ingr, str):
                    raw_ingr = eval(raw_ingr)
                if isinstance(raw_ingr, list):
                    ingr_list = list(set(ingr_list) | set(raw_ingr))
            except Exception as e:
                print(f"Error processing ingredients_raw for recipe {recipe['id']}: {e}")

            score = ingredient_match_score(user_ingredients, ingr_list, allow_substitutions, use_grocery, beta)
            scores.append(score)

            percent = int(round(score * 100))
            match_text.append(f"{percent}% match")

        matching_recipes['ingredient_score'] = scores
        matching_recipes['match_text'] = match_text

        # ------------------------------
        # Step 3: Rank using trained model
        # ------------------------------
        # For recipes that passed filtering, get model predictions
        X_text = matching_recipes['full_text'].values
        
        # Get model predictions
        print("Getting model predictions...")
        # Ensure X_text_tensor and X_servings_tensor are defined as tensors
        X_text_tensor = tf.constant(X_text.reshape(-1, 1), dtype=tf.string)

        inputs = {
            "full_text": X_text_tensor,
        }
# Call the serving function; adjust "output_0" if your model uses a different output key.
        predictions = infer(**inputs)["output_0"].numpy()

        # Get recipe IDs from model
        recipe_ids = le_recipe.classes_
        
        # Add prediction scores
        matching_recipes['model_score'] = 0
        
        # Match prediction scores to recipes
        for i, recipe_id in enumerate(matching_recipes['id']):
            if recipe_id in recipe_ids:
                idx = np.where(recipe_ids == recipe_id)[0]
                if len(idx) > 0:
                    matching_recipes.loc[matching_recipes['id'] == recipe_id, 'model_score'] = predictions[i, idx[0]]
        
        # Calculate final score as weighted combination of ingredient score and model score
        matching_recipes['final_score'] = (matching_recipes['ingredient_score'] * 0.95) + (matching_recipes['model_score'] * 0.05)
        
        # Sort by final score
        ranked_recipes = matching_recipes.sort_values(by='final_score', ascending=False)
        
        # Return top recipe and next 10 recommendations
        if len(ranked_recipes) > 0:
            top_recipe = ranked_recipes.iloc[0]
            print(f"Top recipe: {top_recipe['name']} (Score: {top_recipe['final_score']:.2f})")
            
            other_recipes = ranked_recipes.iloc[1:11] if len(ranked_recipes) > 1 else pd.DataFrame()
            return top_recipe, other_recipes
        else:
            print("No suitable recipes found")
            return None, None
            
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Entry point for testing
if __name__ == "__main__":
    print("Testing recipe recommendation system")
    top_recipe, other_recipes = get_recommendations()
    
    if top_recipe is not None:
        print("\n--- TOP RECOMMENDATION ---")
        print(f"Recipe: {top_recipe['name']}")
        print(f"\nIngredients: {top_recipe['ingredients']}")
        print(f"\nIngredient match score: {top_recipe['ingredient_score']:.2f}")
        print(f"\nSteps: {top_recipe['steps']}")
        
        if len(other_recipes) > 0:
            print("\n--- OTHER RECOMMENDATIONS ---")
            for _, recipe in other_recipes.iterrows():
                print(f"- {recipe['name']} (Score: {recipe['ingredient_score']:.2f})")