#this is where the image upload is
import streamlit as st
from storage import save_favourites
from userinputs import get_user_preferences
from training.training import find_recipes
from PIL import Image

def render_generator_tab():
    prefs = get_user_preferences()
    st.header("📸 Upload Ingredients")

    # Default fallback values
    combined_ingredients = []
    use_substitutes = False

    uploaded_files = st.file_uploader(
        "Upload images of your ingredients",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Placeholder for model extraction
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

    # Always show the button
    if st.button("Generate Recipe"):
        if not combined_ingredients:
            st.warning("Please upload an image and confirm your ingredients before generating.")
            return

        results = find_recipes(combined_ingredients, prefs, use_substitutes=use_substitutes)

        if results.empty:
            st.warning("No matching recipes found.")
        else:
            for idx, row in results.iterrows():
                st.subheader(row['name'])
                st.write(f"**Time:** {row['estimated_time']} mins")
                st.write("### Ingredients")
                st.write(", ".join(row['ingredients']))
                st.write("### Steps")
                for i, step in enumerate(row['steps'], 1):
                    st.write(f"{i}. {step}")

                if st.button("Save to Favourites", key=f"save_{idx}"):
                    if "favourites" not in st.session_state:
                        st.session_state.favourites = []

                    recipe = {
                        "name": row['name'],
                        "ingredients": row['ingredients'],
                        "steps": row['steps'],
                        "time": row['estimated_time'],
                        "cuisine": prefs['cuisine']
                    }

                    st.session_state.favourites.append(recipe)
                    save_favourites(st.session_state.favourites)
                    st.success(f"{recipe['name']} saved to favourites!")
