import streamlit as st
from image_api import get_recipe_image
from storage import save_favourites
from PIL import Image
import sys
import os
import tempfile

# Add "ai model" directory to the path
ai_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai model"))
sys.path.append(ai_model_path)
print(f"Added to path: {ai_model_path}")

# Try to import from image-recognition as well
image_recognition_path = os.path.abspath(os.path.join(ai_model_path, "image-recognition"))
sys.path.append(image_recognition_path)
print(f"Added to path: {image_recognition_path}")

# Import user preferences and recommendation functions
from userinputs import get_user_preferences
from get_recommendation import get_recommendations

# Import the FoodRecognizer class - with error handling for debugging
try:
    from clip_model import FoodRecognizer
    print("Successfully imported FoodRecognizer")
except ImportError as e:
    print(f"Error importing FoodRecognizer: {e}")
    print(f"Current sys.path: {sys.path}")
    # Fallback to a mock recognizer if import fails
    class FoodRecognizer:
        def __init__(self):
            print("Using mock FoodRecognizer")
        
        def recognize(self, image):
            return [("tomato", 0.9), ("onion", 0.8), ("garlic", 0.7)]

# Initialize FoodRecognizer with proper path to ingredients CSV
try:
    ingredients_csv = os.path.join(image_recognition_path, "data", "top_500_ingredients.csv")
    print(f"Looking for ingredients CSV at: {ingredients_csv}")
    
    if os.path.exists(ingredients_csv):
        print(f"Ingredients CSV found at: {ingredients_csv}")
        recognizer = FoodRecognizer(ingredients_csv=ingredients_csv)
    else:
        print(f"Ingredients CSV not found at: {ingredients_csv}")
        recognizer = FoodRecognizer()  # Fall back to default path
except Exception as e:
    print(f"Error initializing FoodRecognizer: {e}")
    recognizer = FoodRecognizer()  # Fall back to default initialization

def render_generator_tab():
    prefs = get_user_preferences()
    st.header("Upload Ingredients")

    # Defaults
    combined_ingredients = []
    use_substitutes = False

    uploaded_files = st.file_uploader(
        "Upload images of your ingredients",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    detected_ingredients = []
    
    # Container for progress and status
    status_container = st.empty()

    if uploaded_files:
        # Show a progress message
        status_container.info("Analyzing your ingredients...")
        
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            try:
                # Display the uploaded image
                image = Image.open(temp_path).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True, width=300)
                
                # Recognize ingredients from the image
                try:
                    # Try recognize method for uploaded PIL image
                    detected = recognizer.recognize(image)
                except Exception as e:
                    print(f"Error with recognize method: {e}")
                    # Fall back to recognize_from_file method
                    detected = recognizer.recognize_from_file(temp_path)
                
                # Add detected ingredients to the list
                new_ingredients = [ingredient for ingredient, prob in detected]
                detected_ingredients += new_ingredients
                
                # Show what was detected in this image
                st.write(f"Detected in this image: {', '.join(new_ingredients)}")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass

        # Clear the progress message
        status_container.empty()
        
        # Remove duplicates
        detected_ingredients = list(set(detected_ingredients))

        st.subheader("Here's what I see:")
        st.success(", ".join(detected_ingredients))

        user_corrections = st.text_input("Want to add or remove anything? (comma-separated)")
        if user_corrections:
            custom = [x.strip().lower() for x in user_corrections.split(",")]
            detected_ingredients = list(set(detected_ingredients + custom))

        use_substitutes = st.checkbox("I'm willing to accept substitutes")

        # Pantry + optional grocery
        combined_ingredients = detected_ingredients + st.session_state.get("pantry", [])
        if prefs.get("use_grocery", False):
            combined_ingredients += st.session_state.get("grocery", [])

        st.markdown("### Final Ingredients Being Used:")
        st.write(", ".join(set(combined_ingredients)))

    if st.button("Generate Recipe"):
        if not combined_ingredients:
            st.warning("Please upload an image and confirm your ingredients before generating.")
            return

        # Update preferences with ingredients and substitute toggle
        prefs["ingredients"] = combined_ingredients
        prefs["allow_substitutions"] = use_substitutes

        # Show a loading spinner while getting recommendations
        with st.spinner("Finding the perfect recipe for you..."):
            top_recipe, other_recs = get_recommendations()

        if not top_recipe:
            st.warning("No matching recipes found.")
        else:
            st.subheader(f"Top Recipe: {top_recipe['name']}")
            
            # Use your existing get_recipe_image function from image_api.py
            with st.spinner("Fetching recipe image..."):
                img_url = get_recipe_image(top_recipe["name"])
            
            if img_url:
                st.image(img_url, caption=top_recipe["name"], use_column_width=True)
            
            st.write(f"**Ingredient Match Score:** {top_recipe['ingredient_score']:.2f}")
            st.write("### Ingredients")
            st.write(", ".join(top_recipe["ingredients"]))
            st.write("### Steps")
            for i, step in enumerate(top_recipe["steps"], 1):
                st.write(f"{i}. {step}")

            if st.button("Save to Favourites"):
                if "favourites" not in st.session_state:
                    st.session_state.favourites = []

                recipe = {
                    "name": top_recipe["name"],
                    "ingredients": top_recipe["ingredients"],
                    "steps": top_recipe["steps"],
                    "time": "Unknown",  # Update if time is added
                    "cuisine": prefs["cuisine"]
                }

                st.session_state.favourites.append(recipe)
                save_favourites(st.session_state.favourites)
                st.success("Recipe saved to favourites!")

            if other_recs is not None and not other_recs.empty:
                st.markdown("### Other Recommended Recipes")
                for _, recipe in other_recs.iterrows():
                    st.markdown(f"- **{recipe['name']}** (Score: {recipe['ingredient_score']:.2f})")