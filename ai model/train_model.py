import pandas as pd
import ast
import re
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model  

# Load the CSV file
df = pd.read_csv('ai model/recipes_ingredients.csv')
print("Raw data preview:")
print(df.head())

#Rename the needed columns
recipes = df[['id', 'name', 'tags', 'ingredients', 'ingredients_raw', 'steps', 'servings']]

#convert stings to lists 
recipes['tags'] = recipes['tags'].apply(ast.literal_eval)
recipes['ingredients'] = recipes['ingredients'].apply(ast.literal_eval)
recipes['ingredients_raw'] = recipes['ingredients_raw'].apply(ast.literal_eval)
recipes['steps'] = recipes['steps'].apply(ast.literal_eval)


# Extract the time-to-make from the tags.
def extract_time_from_tags(tags):
    # Look for the tag 'time-to-make' and assume the next tag holds a time range (e.g., '60-to-90-minutes')
    for i, tag in enumerate(tags):
        if tag.lower() == 'time-to-make':
            if i + 1 < len(tags):
                time_tag = tags[i + 1]
                match = re.search(r'(\d+)', time_tag)
                if match:
                    return int(match.group(1))  # Use the lower bound as the estimate
    return None

recipes['time_to_make'] = recipes['tags'].apply(extract_time_from_tags)

#SOPH IS PROBABLY TO CHANGE THIS 
# Extract cuisine type from tags.
def extract_cuisine(tags):
    for tag in tags:
        if tag.lower().startswith('cuisine'):
            parts = tag.split('-')
            if len(parts) > 1:
                return parts[1].lower()  # e.g., "italian"
    return None

recipes['cuisine'] = recipes['tags'].apply(extract_cuisine)

# Ensure 'servings' is numeric. (If not present, you may need to add or impute this column.)
recipes['servings'] = pd.to_numeric(recipes['servings'], errors='coerce')

# Drop rows missing essential information.
recipes = recipes.dropna(subset=['time_to_make', 'servings', 'cuisine'])
print("Preprocessed recipes shape:", recipes.shape)


# -----------------------------
# Step 2: Create Training Data
# -----------------------------

# Features:
#   - Servings (numeric)
#   - Time-to-make (numeric)
#   - Cuisine (categorical, one-hot encoded)

X_servings = recipes['servings'].values.reshape(-1, 1)
X_time = recipes['time_to_make'].values.reshape(-1, 1)

# Scale numeric features.
scaler_servings = StandardScaler()
scaler_time = StandardScaler()
X_servings_scaled = scaler_servings.fit_transform(X_servings)
X_time_scaled = scaler_time.fit_transform(X_time)

# Encode cuisine.
le_cuisine = LabelEncoder()
cuisine_labels = le_cuisine.fit_transform(recipes['cuisine'])
num_cuisines = len(le_cuisine.classes_)
X_cuisine = to_categorical(cuisine_labels, num_classes=num_cuisines)

# The target is the recipe. We treat each recipe as a separate class.
le_recipe = LabelEncoder()
recipe_ids_encoded = le_recipe.fit_transform(recipes['id'])
num_recipes = len(le_recipe.classes_)
y = to_categorical(recipe_ids_encoded, num_classes=num_recipes)

print("Feature shapes:")
print("Servings:", X_servings_scaled.shape)
print("Time:", X_time_scaled.shape)
print("Cuisine:", X_cuisine.shape)
print("Target (recipes):", y.shape)

# -----------------------------
# Step 3: Build the TensorFlow Model
# -----------------------------

# Define inputs.
input_servings = Input(shape=(1,), name='servings')
input_time = Input(shape=(1,), name='time')
input_cuisine = Input(shape=(num_cuisines,), name='cuisine')

# Concatenate all inputs.
x = Concatenate()([input_servings, input_time, input_cuisine])
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
# Output layer: one neuron per recipe.
output = Dense(num_recipes, activation='softmax')(x)

model = Model(inputs=[input_servings, input_time, input_cuisine], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Step 4: Train the Model
# -----------------------------

history = model.fit(
    [X_servings_scaled, X_time_scaled, X_cuisine],
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Save the model and preprocessing objects.
model.save("recipe_model.h5")
with open("scaler_servings.pkl", "wb") as f:
    pickle.dump(scaler_servings, f)
with open("scaler_time.pkl", "wb") as f:
    pickle.dump(scaler_time, f)
with open("le_cuisine.pkl", "wb") as f:
    pickle.dump(le_cuisine, f)
with open("le_recipe.pkl", "wb") as f:
    pickle.dump(le_recipe, f)

print("Model and preprocessing objects saved.")