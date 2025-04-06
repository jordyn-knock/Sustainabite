##import os
import pandas as pd
import ast
import os

# Get the full path to the CSV file in the same folder
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "recipes_ingredients.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find dataset at {csv_path}")

df = pd.read_csv(csv_path)

# Rename columns if needed
# df = df.rename(columns={"minutes": "time-to-make"})

# Safely evaluate stringified lists
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

df['ingredients'] = df['ingredients'].fillna("[]").apply(safe_literal_eval)
df['tags'] = df['tags'].fillna("[]").apply(safe_literal_eval)
df['steps'] = df['steps'].fillna("[]").apply(safe_literal_eval)

# New: estimate time from tags
def extract_time_from_tags(tags):
    time_keywords = {
        "15-minutes-or-less": 15,
        "30-minutes-or-less": 30,
        "60-minutes-or-less": 60,
        "under-15-minutes": 15,
        "under-30-minutes": 30,
        "under-60-minutes": 60
    }
    for tag in tags:
        tag = tag.lower().replace(" ", "-")
        if tag in time_keywords:
            return time_keywords[tag]
    return None  # Unknown

df['estimated_time'] = df['tags'].apply(extract_time_from_tags)

# Select relevant columns
recipes = df[['id', 'name', 'tags', 'ingredients', 'steps', 'estimated_time']]

# Meal type mapping
meal_type_tags = {
    "Breakfast": ["breakfast"],
    "Full Meal": ["lunch", "dinner", "brunch"],
    "Sweet Treat": ["dessert"],
    "Snack": ["snack", "appetizer", "side dish"]
}

def find_recipes(ingredients_input, preferences, use_substitutes=False, top_n=5):
    max_time = preferences["max_time"]
    cuisine = preferences["cuisine"]
    meal_type = preferences["meal_type"]

    matches = []

    for idx, row in recipes.iterrows():
        # Filter by estimated time
        if row['estimated_time'] is None or row['estimated_time'] > max_time:
            continue

        # Filter by cuisine
        if cuisine != "Any cuisine":
            if not any(cuisine.lower() == tag.lower() for tag in row['tags']):
                continue

        # Filter by meal type
        type_tags = meal_type_tags.get(meal_type, [])
        if not any(tag.lower() in row['tags'] for tag in type_tags):
            continue

        # Ingredient matching
        recipe_ingredients = [i.lower() for i in row['ingredients']]
        user_ingredients = [i.lower() for i in ingredients_input]

        if use_substitutes:
            # Accept partial match (60% or more)
            match_count = sum(1 for ing in recipe_ingredients if ing in user_ingredients)
            match_ratio = match_count / len(recipe_ingredients) if recipe_ingredients else 0
            ingredient_match = match_ratio >= 0.6
        else:
            # Strict match (all ingredients must be available)
            ingredient_match = all(ing in user_ingredients for ing in recipe_ingredients)

        if not ingredient_match:
            continue

        matches.append(row)

    return pd.DataFrame(matches).head(top_n)[['name', 'ingredients', 'estimated_time', 'steps']]
##