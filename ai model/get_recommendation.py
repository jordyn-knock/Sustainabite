import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from userinputs import get_servings, get_time, get_cuisine  # These functions return user input values

# Load the saved model and preprocessing objects.
model = load_model("recipe_model.h5")
with open("scaler_servings.pkl", "rb") as f:
    scaler_servings = pickle.load(f)
with open("scaler_time.pkl", "rb") as f:
    scaler_time = pickle.load(f)
with open("le_cuisine.pkl", "rb") as f:
    le_cuisine = pickle.load(f)
with open("le_recipe.pkl", "rb") as f:
    le_recipe = pickle.load(f)

# Define a function to recommend recipes.
def recommend_recipe(top_n=10):
    # Get user inputs via functions in userinputs.py.
    servings_input = get_servings()   # e.g., returns 4
    time_input = get_time()           # e.g., returns 30 (minutes)
    cuisine_input = get_cuisine()     # e.g., returns "italian"

    # Preprocess the inputs.
    servings_scaled = scaler_servings.transform(np.array([[servings_input]]))
    time_scaled = scaler_time.transform(np.array([[time_input]]))
    # Ensure the cuisine is lower-case; if the cuisine is not known, this will throw an error.
    cuisine_label = le_cuisine.transform([cuisine_input.lower()])
    cuisine_onehot = to_categorical(cuisine_label, num_classes=len(le_cuisine.classes_))

    # Predict probabilities for each recipe.
    preds = model.predict([servings_scaled, time_scaled, cuisine_onehot])
    top_indices = np.argsort(preds[0])[::-1][:top_n+1]

    # Here you would map the predicted indices back to your recipes.
    # For demonstration, we'll simply return the top prediction index and the next ones.
    best_recipe_index = top_indices[0]
    other_recipe_indices = top_indices[1:]
    return best_recipe_index, other_recipe_indices

# Example usage:
best_idx, other_idxs = recommend_recipe()
print("Best Recipe Index:", best_idx)
print("Other Recipe Indices:", other_idxs)
