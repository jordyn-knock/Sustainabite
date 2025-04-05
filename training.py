import pandas as pd
import ast  # Needed to convert stringified lists

# Load the CSV file
df = pd.read_csv(r'recipes_ingredients.csv')
print(df.head())

# Rename the loaded DataFrame so it's consistent
recipes = df[['id', 'name', 'tags', 'ingredients', 'steps']]

# Convert stringified lists to real Python lists
recipes['ingredients'] = recipes['ingredients'].apply(ast.literal_eval)
recipes['tags'] = recipes['tags'].apply(ast.literal_eval)
recipes['steps'] = recipes['steps'].apply(ast.literal_eval)

def find_recipes(ingredients_input, max_time, cuisine_preference, top_n=5):
    matches = []

    for idx, row in recipes.iterrows():
        if row['time-to-make'] is None or row['time-to-make'] > max_time:
            continue

        cuisine_match = any(cuisine_preference.lower() in tag.lower() for tag in row['tags'])
        if not cuisine_match:
            continue

        ingredient_match = all(
            ing.lower() in [i.lower() for i in ingredients_input]
            for ing in row['ingredients']
        )
        if not ingredient_match:
            continue

        matches.append(row)

    results = pd.DataFrame(matches).head(top_n)
    return results[['name', 'ingredients', 'time-to-make', 'steps']]