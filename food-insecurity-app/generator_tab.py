import streamlit as st
from image_api import get_recipe_image
from storage import save_favourites
from PIL import Image
import sys
import os

# Add "ai model" directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai model")))

from userinputs import get_user_preferences
from get_recommendation import get_recommendations

def render_generator_tab():
    prefs = get_user_preferences()
    st.header("📸 Upload Ingredients")

    # Defaults
    combined_ingredients = []
    use_substitutes = False

    uploaded_files = st.file_uploader(
        "Upload images of your ingredients",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Placeholder: replace with actual image-to-ingredients model
        detected_ingredients = ["garlic", "tomato"]

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

        top_recipe, other_recs = get_recommendations()

        if not top_recipe:
            st.warning("No matching recipes found.")
        else:
            st.subheader(f"Top Recipe: {top_recipe['name']}")
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
                st.markdown("###Other Recommended Recipes")
                for _, recipe in other_recs.iterrows():
                    st.markdown(f"- **{recipe['name']}** (Score: {recipe['ingredient_score']:.2f})")
