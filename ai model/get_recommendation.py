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

# Dummy classes for development without model files
class DummyInferenceFunction:
    def __call__(self, **kwargs):
        # Return random predictions for dummy usage
        batch_size = kwargs.get("full_text").shape[0]
        return {"output_0": tf.random.uniform((batch_size, 100))}
    
    @property
    def structured_input_signature(self):
        return ((), {'full_text': tf.TensorSpec(shape=(None, 1), dtype=tf.string), 
                    'servings': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)})
    
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
        # List of cuisine types for random selection
        cuisines = ["italian", "mexican", "chinese", "indian", "american", "french", "thai", "japanese"]
        # List of time labels
        times = ["under_15", "under_30", "under_60", "any"]
        
        if len(X) == 0:
            return []
            
        # Randomly select cuisines or times based on first element's type
        if isinstance(X[0], str) and any(cuisine in X[0].lower() for cuisine in cuisines):
            return [random.choice(cuisines) for _ in range(len(X))]
        else:
            return [random.choice(times) for _ in range(len(X))]

def get_dummy_recipe_db():
    """Create a small dummy recipe database for testing"""
    # Generate a list of sample recipes
    sample_recipes = [
        {
            'id': 'recipe1',
            'name': 'Pasta Carbonara',
            'ingredients': ['pasta', 'eggs', 'cheese', 'bacon'],
            'ingredients_raw': str(['pasta', 'eggs', 'cheese', 'bacon']),
            'steps': str(['Cook pasta', 'Mix eggs and cheese', 'Combine']),
            'predicted_cuisine': 'italian',
            'predicted_time': 'under_30'
        },
        {
            'id': 'recipe2',
            'name': 'Chicken Tacos',
            'ingredients': ['chicken', 'tortillas', 'onion', 'salsa'],
            'ingredients_raw': str(['chicken', 'tortillas', 'onion', 'salsa']),
            'steps': str(['Cook chicken', 'Warm tortillas', 'Assemble tacos']),
            'predicted_cuisine': 'mexican',
            'predicted_time': 'under_15'
        },
        {
            'id': 'recipe3',
            'name': 'Chocolate Cake',
            'ingredients': ['flour', 'sugar', 'cocoa', 'butter', 'eggs'],
            'ingredients_raw': str(['flour', 'sugar', 'cocoa', 'butter', 'eggs']),
            'steps': str(['Mix dry ingredients', 'Add wet ingredients', 'Bake']),
            'predicted_cuisine': 'american',
            'predicted_time': 'under_60'
        }
    ]
    
    # Create more dummy recipes based on common ingredients
    common_ingredients = [
        ['chicken', 'rice', 'broccoli'],
        ['beef', 'potatoes', 'carrots'],
        ['shrimp', 'pasta', 'garlic'],
        ['tofu', 'soy sauce', 'ginger'],
        ['salmon', 'lemon', 'dill'],
        ['bread', 'tomato', 'lettuce'],
        ['flour', 'sugar', 'vanilla']
    ]
    
    for i, ingredients in enumerate(common_ingredients, 4):
        sample_recipes.append({
            'id': f'recipe{i}',
            'name': f'Recipe {i}',
            'ingredients': ingredients,
            'ingredients_raw': str(ingredients),
            'steps': str([f'Step 1 for recipe {i}', f'Step 2 for recipe {i}']),
            'predicted_cuisine': random.choice(['italian', 'mexican', 'chinese', 'indian', 'american']),
            'predicted_time': random.choice(['under_15', 'under_30', 'under_60'])
        })
    
    df = pd.DataFrame(sample_recipes)
    
    # Add necessary text columns
    df['ingredients_raw_text'] = df['ingredients_raw'].apply(lambda x: " ".join(eval(x)))
    df['ingredients_text'] = df['ingredients'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    df['steps_text'] = df['steps'].apply(lambda x: " ".join(eval(x)) if isinstance(x, str) else "")
    df['full_text'] = df['ingredients_text'] + " " + df['ingredients_raw_text'] + " " + df['steps_text']
    
    return df

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
    
    # Add the correct path where your model is actually located
    possible_model_dirs = [
        os.path.join(ai_models_dir, "recipe_model_tf"),  # Original path
        os.path.join("/home/alissah/youCode/youcode/recipe_model_tf"),  # Correct path
        "recipe_model_tf"  # Relative path
    ]
    
    # Try each possible model directory
    model_dir = None
    for dir_path in possible_model_dirs:
        if os.path.exists(dir_path):
            model_dir = dir_path
            print(f"Found model directory at: {model_dir}")
            break
    
    if model_dir is None:
        print(f"WARNING: Model directory not found in any of: {possible_model_dirs}")
        print("Using dummy recommendation system for testing.")
        # Return dummy objects that mimic expected behavior
        return DummyInferenceFunction(), DummyLabelEncoder(), DummyClassifier(), DummyClassifier(), get_dummy_recipe_db()
    
    try:
        # Load recommendation model
        model = tf.saved_model.load(model_dir)
        # Extract the serving signature as 'infer'
        infer = model.signatures["serving_default"]
        print("Loaded recommendation model")
        
        # Load label encoder - search in multiple locations
        possible_dirs = [models_dir, ai_models_dir, model_dir, os.path.dirname(model_dir)]
        
        le_path = find_file(["le_recipe.pkl"], possible_dirs)
        if not le_path:
            raise FileNotFoundError("Label encoder file not found")
        
        with open(le_path, "rb") as f:
            le_recipe = pickle.load(f)
        print(f"Loaded label encoder with {len(le_recipe.classes_)} classes")
        
        # Load cuisine classifier
        cuisine_path = find_file(["cuisine_clf.joblib"], possible_dirs)
        if not cuisine_path:
            raise FileNotFoundError("Cuisine classifier not found")
        
        cuisine_clf = joblib.load(cuisine_path)
        print(f"Loaded cuisine classifier")

        # Load time classifier
        time_clf_path = find_file(["time_tag_classifier.joblib"], possible_dirs)
        if not time_clf_path:
            raise FileNotFoundError("Time classifier not found")
        time_clf = joblib.load(time_clf_path)
        print(f"Loaded time classifier")
        
        # Load recipe database from pickle or CSV
        recipe_db_path = find_file(["recipe_database.pkl"], possible_dirs)
        if recipe_db_path:
            # Use the pickled recipe database if available
            try:
                with gzip.open(recipe_db_path, "rb") as f:
                    recipe_db = pickle.load(f)
                print(f"Loaded recipe database from pickle with {len(recipe_db)} recipes")
            except Exception as e:
                print(f"Error loading pickled database: {e}")
                recipe_db = load_recipe_csv(models_dir, ai_models_dir)
        else:
            recipe_db = load_recipe_csv(models_dir, ai_models_dir)
        
        return infer, le_recipe, cuisine_clf, time_clf, recipe_db
        
    except Exception as e:
        print(f"Error loading model files: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to dummy recommendation system.")
        return DummyInferenceFunction(), DummyLabelEncoder(), DummyClassifier(), DummyClassifier(), get_dummy_recipe_db()


def load_recipe_csv(models_dir, ai_models_dir):
    """Load recipe database from CSV"""
    recipe_path = find_file(["recipes_ingredients.csv"], [models_dir, ai_models_dir])
    if not recipe_path:
        print("Recipe database CSV not found, using dummy data")
        return get_dummy_recipe_db()
    
    try:
        # Use optimized CSV loading with chunksize for large files
        recipe_db = pd.read_csv(recipe_path, low_memory=False)
        print(f"Loaded recipe database from CSV with {len(recipe_db)} recipes")
        return recipe_db
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return get_dummy_recipe_db()


def find_file(filenames, directories):
    """Helper function to find a file in multiple possible directories"""
    for directory in directories:
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                return path
    return None


def process_recipe_batch(batch_df, user_ingredients, allow_substitutions, use_grocery):
    """Process a batch of recipes to calculate ingredient match scores"""
    scores = []
    for _, recipe in batch_df.iterrows():
        # Get the ingredients list
        try:
            ingr_list = recipe['ingredients']
            if isinstance(ingr_list, str):
                ingr_list = eval(ingr_list)
            
            # Try to get additional ingredients from 'ingredients_raw'
            try:
                raw_ingr = recipe.get('ingredients_raw')
                if isinstance(raw_ingr, str):
                    raw_ingr = eval(raw_ingr)
                if isinstance(raw_ingr, list):
                    # Combine both lists and remove duplicates
                    ingr_list = list(set(ingr_list) | set(raw_ingr))
            except Exception:
                pass
            
            score = ingredient_match_score(user_ingredients, ingr_list, allow_substitutions, use_grocery, beta=0.2)
        except Exception as e:
            print(f"Error calculating score for recipe {recipe.get('id', 'unknown')}: {e}")
            score = 0.0
            
        scores.append(score)
    
    return scores


def get_recommendations():
    """Generate recipe recommendations based on user preferences"""
    try:
        # Get user preferences
        prefs = get_user_preferences()
        print(f"User preferences: {prefs}")
        
        # Extract key preferences
        selected_cuisine = prefs.get("cuisine", "Any Cuisine").lower()
        user_ingredients = prefs.get("ingredients", SAMPLE_INGREDIENTS)
        allow_substitutions = prefs.get("allow_substitutions", True)
        use_grocery = prefs.get("use_grocery", True)
        
        # Convert max_time from string to int safely
        max_time_str = prefs.get("max_time", "60")
        
        # Ensure max_time is an integer
        try:
            # Handle 'Any Time' case or other non-numeric values
            max_time = int(max_time_str) if max_time_str not in ["Any Time", "any"] else 999
        except (ValueError, TypeError):
            # Default to 60 minutes if conversion fails
            print(f"Warning: Could not convert max_time '{max_time_str}' to int, using default of 60 minutes")
            max_time = 60
            
        print(f"Max time (converted to int): {max_time}")
        
        # Convert servings from string to float safely
        servings_str = prefs.get("servings", "4")
        try:
            requested_servings = float(servings_str)
        except (ValueError, TypeError):
            requested_servings = 4.0
            print(f"Warning: Could not convert servings '{servings_str}' to float, using default of 4")
        
        # Override sample ingredients if user has provided their own
        if "ingredients" in prefs and prefs["ingredients"]:
            print(f"Using user ingredients: {prefs['ingredients']}")
        else:
            # Fall back to sample ingredients for testing
            prefs["ingredients"] = SAMPLE_INGREDIENTS
            print(f"Using sample ingredients: {SAMPLE_INGREDIENTS}")
        
        print(f"Looking for {selected_cuisine} recipes with max time {max_time} mins")
        
        # Load model files with fallback to dummy system
        infer, le_recipe, cuisine_clf, time_clf, recipe_db = load_model_files()
        
        # Check if we're using dummy objects
        using_dummy = isinstance(infer, DummyInferenceFunction)
        if using_dummy:
            print("Using dummy recommendation system (no model files found)")
        
        # ------------------------------
        # Step 1: Pre-process and filter recipes
        # ------------------------------
        
        # Process ingredients column safely
        def safe_eval(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    return eval(x)
                except:
                    return []
            return []
        
        # Process columns safely with error handling
        try:
            # Ensure ingredients column is processed
            recipe_db['ingredients'] = recipe_db['ingredients'].apply(safe_eval)
            
            # Create text features needed for classifiers
            if 'ingredients_raw' in recipe_db.columns:
                recipe_db['ingredients_raw_text'] = recipe_db['ingredients_raw'].apply(
                    lambda x: " ".join(safe_eval(x)) if not pd.isna(x) else "")
            else:
                recipe_db['ingredients_raw_text'] = ""
            
            recipe_db['ingredients_text'] = recipe_db['ingredients'].apply(
                lambda x: " ".join(x) if isinstance(x, list) else "")
                
            if 'steps' in recipe_db.columns:
                recipe_db['steps_text'] = recipe_db['steps'].apply(
                    lambda x: " ".join(safe_eval(x)) if not pd.isna(x) else "")
            else:
                recipe_db['steps_text'] = ""
                
            recipe_db['full_text'] = (recipe_db['ingredients_text'] + " " + 
                                    recipe_db['ingredients_raw_text'] + " " + 
                                    recipe_db['steps_text'])
        except Exception as e:
            print(f"Error preprocessing recipe data: {e}")
            if using_dummy:
                pass  # Continue with dummy data
            else:
                # Try to recover by using dummy data
                print("Falling back to dummy data due to processing error")
                recipe_db = get_dummy_recipe_db()
                using_dummy = True
        
        # Process in smaller batches to avoid memory issues
        if not using_dummy:
            # Predict cuisine for recipes
            print("Predicting cuisine for recipes...")
            batch_size = min(500, max(1, len(recipe_db) // 10))  # Adjust batch size based on dataset
            cuisines = []
            
            # Process in batches
            for i in range(0, len(recipe_db), batch_size):
                end_idx = min(i + batch_size, len(recipe_db))
                batch = recipe_db['ingredients_raw_text'].iloc[i:end_idx].tolist()
                try:
                    predictions = cuisine_clf.predict(batch)
                    cuisines.extend(predictions)
                except Exception as e:
                    print(f"Error predicting cuisines for batch {i}-{end_idx}: {e}")
                    # Fill with default values on error
                    cuisines.extend(["unknown"] * (end_idx - i))
            
            recipe_db['predicted_cuisine'] = cuisines
            
            # Predict time for recipes
            print("Predicting time for recipes...")
            time_preds = []
            for i in range(0, len(recipe_db), batch_size):
                end_idx = min(i + batch_size, len(recipe_db))
                batch = recipe_db['name'].iloc[i:end_idx].tolist()
                try:
                    batch_preds = time_clf.predict(batch)
                    time_preds.extend(batch_preds)
                except Exception as e:
                    print(f"Error predicting time for batch {i}-{end_idx}: {e}")
                    # Fill with default values on error
                    time_preds.extend(["any"] * (end_idx - i))
            
            recipe_db['predicted_time'] = time_preds
        
        # Filter recipes by cuisine
        if selected_cuisine != "any cuisine" and not using_dummy:
            recipe_db = recipe_db[recipe_db['predicted_cuisine'].str.lower() == selected_cuisine]
        
        # Map time preference to label
        if max_time <= 15:
            time_label = "under_15"
        elif max_time <= 30:
            time_label = "under_30"
        elif max_time <= 60:
            time_label = "under_60"
        else:
            time_label = "any"
        
        # Filter recipes by time
        if time_label != "any" and not using_dummy:
            recipe_db = recipe_db[recipe_db['predicted_time'] == time_label]
        
        # Final selection of recipes after filters
        matching_recipes = recipe_db
        print(f"Found {len(matching_recipes)} recipes matching cuisine and time criteria")
        
        if len(matching_recipes) == 0:
            print("No matching recipes found after filtering")
            return None, None
        
        # ------------------------------
        # Step 2: Calculate ingredient match scores
        # ------------------------------
        # Process in batches to avoid memory issues
        print("Calculating ingredient match scores...")
        
        # For very large datasets, sample a subset
        if len(matching_recipes) > 10000 and not using_dummy:
            print(f"Sampling 10000 recipes from {len(matching_recipes)} total matches")
            matching_recipes = matching_recipes.sample(10000, random_state=42)
        
        # Process in batches
        batch_size = min(500, max(1, len(matching_recipes)))
        all_scores = []
        
        for i in range(0, len(matching_recipes), batch_size):
            end_idx = min(i + batch_size, len(matching_recipes))
            batch_df = matching_recipes.iloc[i:end_idx]
            
            batch_scores = process_recipe_batch(
                batch_df, 
                user_ingredients, 
                allow_substitutions, 
                use_grocery
            )
            all_scores.extend(batch_scores)
        
        # Assign scores to dataframe
        matching_recipes['ingredient_score'] = all_scores
        
        # Filter if not willing to buy groceries
        if not use_grocery:
            matching_recipes = matching_recipes[matching_recipes['ingredient_score'] > 0]
            if matching_recipes.empty:
                print("No recipes found that use only your ingredients.")
                return None, None
        
        # ------------------------------
        # Step 3: Rank using model
        # ------------------------------
        print("Ranking recipes...")
        
        # Default to ingredient scores if using dummy
        if using_dummy:
            matching_recipes['model_score'] = np.random.random(size=len(matching_recipes)) * 0.1
            matching_recipes['final_score'] = matching_recipes['ingredient_score']
        else:
            try:
                # For recipes that passed filtering, get model predictions
                X_text = matching_recipes['full_text'].values
                X_servings = np.array([[requested_servings]] * len(matching_recipes)).astype(np.float32)
                
                # Ensure tensors are correctly formatted
                X_text_tensor = tf.constant(X_text.reshape(-1, 1), dtype=tf.string)
                X_servings_tensor = tf.constant(X_servings.reshape(-1, 1), dtype=tf.float32)
                
                inputs = {
                    "full_text": X_text_tensor,
                    "servings": X_servings_tensor
                }
                
                # Get predictions
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
                            matching_recipes.loc[matching_recipes.index[i], 'model_score'] = predictions[i, idx[0]]
                
                # Calculate final score
                matching_recipes['final_score'] = (matching_recipes['ingredient_score'] * 0.95) + (matching_recipes['model_score'] * 0.05)
                
            except Exception as e:
                print(f"Error during model prediction: {e}")
                import traceback
                traceback.print_exc()
                
                # Fall back to ingredient scores if model fails
                matching_recipes['model_score'] = 0
                matching_recipes['final_score'] = matching_recipes['ingredient_score']
        
        # Sort by final score
        ranked_recipes = matching_recipes.sort_values(by='final_score', ascending=False)
        
        # Return top recipe and next recommendations
        if len(ranked_recipes) > 0:
            top_recipe = ranked_recipes.iloc[0].to_dict()
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
        
        if other_recipes is not None and len(other_recipes) > 0:
            print("\n--- OTHER RECOMMENDATIONS ---")
            for _, recipe in other_recipes.iterrows():
                print(f"- {recipe['name']} (Score: {recipe['ingredient_score']:.2f})")