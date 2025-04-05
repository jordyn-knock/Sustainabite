#this is where the image upload is
import streamlit as st
from PIL import Image

def render_generator_tab():
    st.header("📸 Upload Ingredients")

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

        # Pantry + optional grocery
        combined_ingredients = detected_ingredients + st.session_state.get("pantry", [])
        if st.session_state.get("use_grocery", False):
            combined_ingredients += st.session_state.get("grocery", [])

        st.markdown("### Final Ingredients Being Used:")
        st.write(", ".join(set(combined_ingredients)))

        if st.button("Generate Recipe"):
            # this is where you add the ai logic thingy
            recipe = {
                "name": "Tomato Garlic Soup",
                "steps": ["Chop garlic and onions", "Cook with tomatoes", "Simmer and serve"],
                "time": "30 minutes",
                "cuisine": "Italian"
            }

            st.subheader(recipe["name"])
            st.write(f"**Time:** {recipe['time']} | **Cuisine:** {recipe['cuisine']}")
            st.write("### Steps")
            for i, step in enumerate(recipe["steps"], 1):
                st.write(f"{i}. {step}")

            if st.button("❤️ Save to Favourites"):
                if "favourites" not in st.session_state:
                    st.session_state.favourites = []
                st.session_state.favourites.append(recipe)
