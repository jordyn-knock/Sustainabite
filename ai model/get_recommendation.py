import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle
from typing import List, Dict, Tuple, Any
import os


from userinputs import get_user_preferences

# Substitutions dictionary
SUBSTITUTIONS = {
    "milk": ["almond milk", "soy milk", "oat milk", "coconut milk"],
    "butter": ["margarine", "coconut oil", "olive oil"],
    "sugar": ["honey", "maple syrup", "agave nectar", "brown sugar"],
    "egg": ["egg substitute, flax egg"],
    "flour": ["almond flour", "coconut flour", "whole wheat flour"],
    "salt": ["sea salt", "kosher salt"],
    "baking powder": ["baking soda"],
    "cheddar": ["colby jack", "monterey jack"],
    "cream": ["coconut cream", "cashew cream", "sour cream"],
    "vanilla extract": ["vanilla bean", "vanilla paste"],
    "oil": ["canola oil", "vegetable oil"],
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
    
    # Load meal type classifier
    meal_path = find_file(["recipe_category_clf.joblib"], [models_dir, ai_models_dir])
    if not meal_path:
        raise FileNotFoundError("Meal type classifier not found")
    
    meal_clf = joblib.load(meal_path)
    print(f"Loaded meal type classifier")
    
    # Load recipe database from pickle instead of CSV
    recipe_db_path = find_file(["recipe_database.pkl"], [models_dir, ai_models_dir])
    if recipe_db_path:
        # Use the pickled recipe database if available
        with open(recipe_db_path, "rb") as f:
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
        
        # Extract key preferences
        selected_cuisine = prefs["cuisine"].lower()
        USER_TO_CATEGORY = {
            "Breakfast": "breakfast",
            "Full Meal": "meals", 
            "Sweet Treat": "sweet treat",
            "Snack": "snacks"
        }
        selected_meal_type = USER_TO_CATEGORY.get(prefs["meal_type"], "unknown")
        requested_servings = float(prefs.get("servings", 4))
        user_ingredients = prefs.get("ingredients", [])
        allow_substitutions = prefs.get("allow_substitutions", False)
        use_grocery = prefs.get("use_grocery", False)
        
        print(f"Looking for {selected_cuisine} {selected_meal_type} recipes for {requested_servings} servings")
        
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
        print("Predicting meal type for recipes...")
        meal_types = []
        for i in range(0, len(recipe_db), batch_size):
            batch = recipe_db['full_text'].iloc[i:i+batch_size].tolist()
            predictions = meal_clf.predict(batch)
            meal_types.extend(predictions)
        recipe_db['predicted_meal_type'] = meal_types
        
        # Filter by cuisine and meal type
        matching_recipes = recipe_db[
            (recipe_db['predicted_cuisine'].str.lower() == selected_cuisine) & 
            (recipe_db['predicted_meal_type'].str.lower() == selected_meal_type)
        ]
        
        print(f"Found {len(matching_recipes)} recipes matching {selected_cuisine} {selected_meal_type}")
        
        if len(matching_recipes) == 0:
            print("No matching recipes found")
            return None, None
        
        # ------------------------------
        # Step 2: Calculate ingredient match scores
        # ------------------------------
        scores = []
        for _, recipe in matching_recipes.iterrows():
            ingredients = recipe['ingredients']
            if not isinstance(ingredients, list):
                continue
                
            score = ingredient_match_score(user_ingredients, ingredients, allow_substitutions)
            scores.append(score)
            
        matching_recipes['ingredient_score'] = scores
        
        # If user won't buy more ingredients, filter to high-score recipes
        if not use_grocery:
            filtered = matching_recipes[matching_recipes['ingredient_score'] >= 0.8]
            if len(filtered) > 0:
                matching_recipes = filtered
                print(f"Filtered to {len(matching_recipes)} recipes with ingredient score >= 0.8")
        
        # ------------------------------
        # Step 3: Rank using trained model
        # ------------------------------
        # For recipes that passed filtering, get model predictions
        X_text = matching_recipes['full_text'].values
        X_servings = np.array([[requested_servings]] * len(matching_recipes)).astype(np.float32)
        
        # Get model predictions
        print("Getting model predictions...")
        # Ensure X_text_tensor and X_servings_tensor are defined as tensors
# For example, if X_text and X_servings are numpy arrays:
        X_text_tensor = tf.constant(X_text.reshape(-1, 1), dtype=tf.string)
        X_servings_tensor = tf.constant(X_servings.reshape(-1, 1), dtype=tf.float32)

        inputs = {
            "full_text": X_text_tensor,
            "servings": X_servings_tensor
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
        matching_recipes['final_score'] = (matching_recipes['ingredient_score'] * 0.7) + (matching_recipes['model_score'] * 0.3)
        
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