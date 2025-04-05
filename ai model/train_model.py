import pandas as pd
import ast
import re
import pickle
import numpy as np
import json
from typing import List
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Define safe_eval function
# -----------------------------
def safe_eval(obj) -> List[str]:
    """Convert a stringified list to a real list.
    
    Silently handles NaN/None/invalid values by returning [].
    Tries ast.literal_eval first, then a JSON fallback.
    """
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, list):
        return obj

    text = str(obj)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass  # fall through to JSON
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

# Safely convert stringified lists to actual lists
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
print("Preprocessed recipes shape:", recipes.shape)

# Encode the target (recipe id) as a class.
le_recipe = LabelEncoder()
recipe_ids_encoded = le_recipe.fit_transform(recipes['id'])
num_recipes = len(le_recipe.classes_)
y = to_categorical(recipe_ids_encoded, num_classes=num_recipes)

# Also, scale the servings.
scaler_servings = StandardScaler()
X_servings = scaler_servings.fit_transform(recipes['servings'].values.reshape(-1, 1))

# -----------------------------
# Step 2: Build Text Vectorization and Multi-Input Model
# -----------------------------
max_features = 20000
sequence_length = 500

# Create a TextVectorization layer for full_text.
vectorizer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorizer.adapt(recipes['full_text'].values)

# Define two inputs:
# 1. A string input for the full_text.
input_text = Input(shape=(1,), dtype="string", name="full_text")
# 2. A numeric input for servings.
input_servings = Input(shape=(1,), name="servings")

# Process text input.
x_text = vectorizer(input_text)
x_text = Embedding(max_features, 128)(x_text)
x_text = GlobalAveragePooling1D()(x_text)
x_text = Dense(64, activation="relu")(x_text)

# Optionally, process servings input through a small dense layer.
x_servings = Dense(16, activation="relu")(input_servings)

# Concatenate the processed text and servings inputs.
x = Concatenate()([x_text, x_servings])
x = Dense(64, activation="relu")(x)
output = Dense(num_recipes, activation="softmax")(x)

model = Model(inputs=[input_text, input_servings], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# Step 3: Train the Model
# -----------------------------
X_text = recipes['full_text'].values  # Text input.
# X_servings already computed.
history = model.fit(
    [X_text, X_servings],
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Save the trained model and preprocessing objects.
model.save("recipe_model.h5")
with open("le_recipe.pkl", "wb") as f:
    pickle.dump(le_recipe, f)
with open("scaler_servings.pkl", "wb") as f:
    pickle.dump(scaler_servings, f)
# Save the vectorizer as a SavedModel.
tf.saved_model.save(vectorizer, "text_vectorizer")

print("Model and preprocessing objects saved.")
