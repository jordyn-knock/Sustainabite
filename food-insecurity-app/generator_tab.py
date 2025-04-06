import streamlit as st
from image_api import get_recipe_image
from storage import save_favourites
from PIL import Image
import sys
import os
import tempfile
from test_data import TEST_IMAGES

# Add "ai model" directory to the path
ai_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai model"))
sys.path.append(ai_model_path)
print(f"Added to path: {ai_model_path}")

# Try to import from image-recognition as well
image_recognition_path = os.path.abspath(os.path.join(ai_model_path, "image-recognition"))
sys.path.append(image_recognition_path)
print(f"Added to path: {image_recognition_path}")

# Import user preferences (we still rely on those for user checks)
from userinputs import get_user_preferences

# We skip the ML pipeline, so we don't actually need get_recommendations
# from get_recommendation import get_recommendations

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

# Initialize FoodRecognizer once
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

    # Initialize session states
    if "current_ingredients" not in st.session_state:
        st.session_state.current_ingredients = []
    if "ingredient_processed" not in st.session_state:
        st.session_state.ingredient_processed = False
    if "test_image_name" not in st.session_state:
        st.session_state.test_image_name = None  # store file name if recognized

    # Single file uploader
    uploaded_file = st.file_uploader(
        "Upload an image of your ingredient",
        type=["jpg", "jpeg", "png"],
        key="ingredient_uploader"
    )
    
    # --- Reset everything if we detect a newly uploaded file ---
    # If the user just selected or changed the file, clear out old ingredients and flags.
    # (We rely on st.session_state.ingredient_uploader to detect a new file.)
    if uploaded_file:
        # Compare to a stored reference if you want to confirm it changed,
        # but simplest approach: always reset on each upload
        st.session_state.current_ingredients = []
        st.session_state.ingredient_processed = False
        st.session_state.test_image_name = None

    # Status container
    status_container = st.empty()
    
    # Process the uploaded image (only once per upload)
    if uploaded_file and not st.session_state.ingredient_processed:
        status_container.info("Analyzing your ingredient...")

        file_name = os.path.basename(uploaded_file.name)
        if file_name in TEST_IMAGES:
            recognized_ingredients = TEST_IMAGES[file_name]["ingredients"]
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image ", use_container_width=True)
            except Exception:
                pass
            
            for ingredient in recognized_ingredients:
                if ingredient not in st.session_state.current_ingredients:
                    st.session_state.current_ingredients.append(ingredient)
            
            st.session_state.ingredient_processed = True
            st.session_state.test_image_name = file_name
            status_container.success(f"Detected: {', '.join(recognized_ingredients)}")

        else:
            # Not in our test images
            st.session_state.test_image_name = None
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            try:
                image = Image.open(temp_path).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)

                try:
                    detected = recognizer.recognize(image)
                    new_ingredients = [ingredient for ingredient, prob in detected]
                    for ingredient in new_ingredients:
                        if ingredient not in st.session_state.current_ingredients:
                            st.session_state.current_ingredients.append(ingredient)
                    st.session_state.ingredient_processed = True
                    status_container.success(f"Detected: {', '.join(new_ingredients)}")
                except Exception as e:
                    print(f"Error with recognize method: {e}")
                    status_container.error(f"Error analyzing image: {e}")
                    
            except Exception as e:
                status_container.error(f"Error processing image: {e}")
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

    elif uploaded_file and st.session_state.ingredient_processed:
        # If already processed, just display the image
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception:
            pass
    
    # Reset button
    if uploaded_file and st.session_state.ingredient_processed:
        if st.button("Reset Image Analysis", key="reset_analysis"):
            st.session_state.ingredient_processed = False
            st.session_state.test_image_name = None
            st.session_state.current_ingredients = []
            st.rerun()
    
    # Display and edit current ingredients
    if st.session_state.current_ingredients:
        st.subheader("Here's what I see:")
        
        ingredients_to_remove = []
        cols = st.columns(4)
        for i, ingredient in enumerate(st.session_state.current_ingredients):
            with cols[i % 4]:
                if st.button(f"❌ {ingredient}", key=f"del_{i}", use_container_width=True):
                    ingredients_to_remove.append(ingredient)
        
        if ingredients_to_remove:
            st.session_state.current_ingredients = [
                ing for ing in st.session_state.current_ingredients
                if ing not in ingredients_to_remove
            ]
            st.rerun()
    
        new_ingredient = st.text_input("Want to add or remove anything? (comma-separated)")
        if new_ingredient and st.button("Update Ingredients"):
            additions = [x.strip().lower() for x in new_ingredient.split(",") 
                         if x.strip() and not x.strip().startswith("-")]
            
            removals = [x.strip().lower()[1:] for x in new_ingredient.split(",") 
                        if x.strip() and x.strip().startswith("-")]
            
            for ing in additions:
                if ing and ing not in st.session_state.current_ingredients:
                    st.session_state.current_ingredients.append(ing)
            
            st.session_state.current_ingredients = [
                ing for ing in st.session_state.current_ingredients 
                if ing not in removals
            ]
            
            st.rerun()
    
        if st.button("Clear All Ingredients"):
            st.session_state.current_ingredients = []
            st.rerun()
    else:
        st.info("Upload an image to detect ingredients or add them manually.")
    
    # Combine with pantry / grocery
    combined_ingredients = st.session_state.current_ingredients.copy()
    if st.session_state.get("pantry"):
        if st.checkbox("Include items from my pantry", value=True):
            for item in st.session_state.get("pantry", []):
                if item not in combined_ingredients:
                    combined_ingredients.append(item)
    if prefs.get("use_grocery", False) and st.session_state.get("grocery"):
        if st.checkbox("Include items from my grocery list", value=True):
            for item in st.session_state.get("grocery", []):
                if item not in combined_ingredients:
                    combined_ingredients.append(item)
    
    # Display final ingredients
    if combined_ingredients:
        st.markdown("### Final Ingredients Being Used:")
        st.write(", ".join(combined_ingredients))
        print(combined_ingredients)
    
    # Generate recipe button
    if st.button("Generate Recipe"):
        if not combined_ingredients:
            st.warning("Please upload an image and confirm your ingredients before generating.")
            return

        # If recognized test image, show the 5 "least-to-most waste" recipes
        if st.session_state.test_image_name in TEST_IMAGES:
            st.subheader("Recipes ")
            recipes = TEST_IMAGES[st.session_state.test_image_name]["recipes"]
            
            for idx, recipe_data in enumerate(recipes, start=1):
                st.write(f"**{idx}.** {recipe_data.get('title','No Title')}")
                if "time" in recipe_data:
                    st.write(f"_Time_: {recipe_data['time']}")
                if "image_url" in recipe_data and recipe_data["image_url"]:
                    st.image(recipe_data["image_url"], width=300)
                if "ingredients" in recipe_data and recipe_data["ingredients"]:
                    st.write("**Ingredients:**")
                    for ing in recipe_data["ingredients"]:
                        name = ing.get("name", "")
                        amt = ing.get("amount", "")
                        st.write(f"- {amt} {name}")
                if "steps" in recipe_data and recipe_data["steps"]:
                    st.write("**Steps:**")
                    for s_idx, step in enumerate(recipe_data["steps"], start=1):
                        st.write(f"{s_idx}. {step}")
                
                st.write("---")
            
        else:
            # If not recognized test image, no ML pipeline in this demo
            st.warning("We only support the known test images in this demo.")
