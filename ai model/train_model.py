import pandas as pd
import ast
import re
import pickle
import numpy as np
import json
from typing import List
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Concatenate, TextVectorization
from tensorflow.keras.models import Model

# -----------------------------
# Import user preferences from your UI module
# -----------------------------
from userinputs import get_user_preferences

prefs = get_user_preferences()  # returns a dict with keys: max_time, cuisine, meal_type, servings, use_grocery, allow_substitutions

# Map the UI meal type (user-friendly) to our category keys.
USER_TO_CATEGORY = {
    "Breakfast": "breakfast",
    "Full Meal": "meals",
    "Sweet Treat": "sweet treat",
    "Snack": "snacks"
}
selected_cuisine = prefs["cuisine"].lower()
selected_meal_type = USER_TO_CATEGORY.get(prefs["meal_type"], "unknown")
max_time = prefs["max_time"]  # Not used in this example, but can be used for further filtering.

# -----------------------------
# Define safe_eval function
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
# Step 1: Load and Preprocess Data
# -----------------------------
df = pd.read_csv('ai model/recipes_ingredients.csv')
print("Raw data preview:")
print(df.head())

# Select needed columns
recipes = df[['id', 'name', 'ingredients', 'ingredients_raw', 'steps', 'servings']]

# Fill missing values for text columns with "[]"
for col in ['ingredients', 'ingredients_raw', 'steps']:
    recipes[col] = recipes[col].fillna("[]")

# Convert stringified lists to actual lists
recipes['ingredients'] = recipes['ingredients'].apply(safe_eval)
recipes['ingredients_raw'] = recipes['ingredients_raw'].apply(safe_eval)
recipes['steps'] = recipes['steps'].apply(safe_eval)

# Create text versions by joining list elements
def join_text(lst):
    return " ".join(lst)

recipes['ingredients_text'] = recipes['ingredients'].apply(join_text)
recipes['ingredients_raw_text'] = recipes['ingredients_raw'].apply(join_text)
recipes['steps_text'] = recipes['steps'].apply(join_text)

# Combine text fields into a single full_text description.
recipes['full_text'] = recipes['ingredients_text'] + " " + recipes['ingredients_raw_text'] + " " + recipes['steps_text']

# Drop rows with empty full_text or missing servings.
recipes = recipes[recipes['full_text'].str.strip() != ""]
recipes['servings'] = pd.to_numeric(recipes['servings'], errors='coerce')
recipes = recipes.dropna(subset=['servings'])
print("Preprocessed recipes shape (before filtering):", recipes.shape)

# -----------------------------
# Step 1b: Annotate with Pre-Trained Bots and Filter
# -----------------------------
# Load the pre-trained Cuisine Bot.
from tqdm import tqdm

# Load the pre-trained Cuisine Bot.
cuisine_clf = joblib.load("cuisine_clf.joblib")

# --- Cuisine Predictions with Progress Bar ---
cuisine_texts = recipes["ingredients_raw_text"].tolist()
batch_size = 1000  # Adjust as needed
predicted_cuisines = []
for i in tqdm(range(0, len(cuisine_texts), batch_size), desc="Predicting Cuisine"):
    batch = cuisine_texts[i:i+batch_size]
    batch_preds = cuisine_clf.predict(batch)
    predicted_cuisines.extend(batch_preds)
recipes["predicted_cuisine"] = predicted_cuisines

# Load the pre-trained Recipe Category (Meal) Bot.
category_clf = joblib.load("recipe_category_clf.joblib")

# --- Meal Type Predictions with Progress Bar ---
full_texts = recipes["full_text"].tolist()
predicted_meal_types = []
for i in tqdm(range(0, len(full_texts), batch_size), desc="Predicting Meal Type"):
    batch = full_texts[i:i+batch_size]
    batch_preds = category_clf.predict(batch)
    predicted_meal_types.extend(batch_preds)
recipes["predicted_meal_type"] = predicted_meal_types

# Filter recipes by user-selected cuisine and meal type.
recipes = recipes[
    (recipes["predicted_cuisine"].str.lower() == selected_cuisine) &
    (recipes["predicted_meal_type"].str.lower() == selected_meal_type)
]
print(f"Dataset shape after filtering for cuisine '{selected_cuisine}' and meal type '{selected_meal_type}':", recipes.shape)


# -----------------------------
# Step 2: Create Training Data
# -----------------------------
# Encode the target (recipe id) as an integer label (sparse target).
le_recipe = LabelEncoder()
recipe_ids_encoded = le_recipe.fit_transform(recipes['id'])
num_recipes = len(le_recipe.classes_)
y = recipe_ids_encoded  # shape: (n_samples,)

# Use servings as provided (no scaling).
X_servings = recipes['servings'].values.reshape(-1, 1).astype(np.float32)

# -----------------------------
# Step 3: Build Text Vectorization and Multi-Input Model
# -----------------------------
max_features = 20000  # Limits the vocabulary to the 20,000 most frequent tokens.
sequence_length = 500

# Create a TextVectorization layer for full_text.
vectorizer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorizer.adapt(recipes['full_text'].values)

# Define two inputs:
input_text = Input(shape=(1,), dtype="string", name="full_text")
input_servings = Input(shape=(1,), name="servings")

# Process text input.
x_text = vectorizer(input_text)
x_text = Embedding(max_features, 128)(x_text)
x_text = GlobalAveragePooling1D()(x_text)
x_text = Dense(64, activation="relu")(x_text)

# Process servings input (used directly).
x_servings = Dense(16, activation="relu")(input_servings)

# Concatenate both processed inputs.
x = Concatenate()([x_text, x_servings])
x = Dense(64, activation="relu")(x)
output = Dense(num_recipes, activation="softmax")(x)

model = Model(inputs=[input_text, input_servings], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# Step 4: Train the Model
# -----------------------------
X_text = recipes['full_text'].values
history = model.fit(
    [X_text, X_servings],
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Save the model and preprocessing objects.
model.save("recipe_model.h5")
with open("le_recipe.pkl", "wb") as f:
    pickle.dump(le_recipe, f)
with open("X_servings.pkl", "wb") as f:
    pickle.dump(X_servings, f)
tf.saved_model.save(vectorizer, "text_vectorizer")

print("Model and preprocessing objects saved.")
