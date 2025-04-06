import streamlit as st
import os
import json

USER_PREFS_DIR = "user_data"
os.makedirs(USER_PREFS_DIR, exist_ok=True)

def get_user_preferences():
    username = st.session_state.get("username", "default_user")
    file_path = os.path.join(USER_PREFS_DIR, f"{username}_prefs.json")

    # Load preferences from disk (only once per session)
    if os.path.exists(file_path) and "user_preferences" not in st.session_state:
        with open(file_path, "r") as f:
            st.session_state["user_preferences"] = json.load(f)

    saved = st.session_state.get("user_preferences", {})

    # UI widgets
    max_time = st.slider("Maximum cooking time (minutes)", 5, 120, saved.get("max_time", 30))

    cuisine_options = sorted([
        "american", "caribbean", "chinese", "french", "german", "greek",
        "indian", "irish", "italian", "japanese", "korean", "mexican",
        "moroccan", "spanish", "thai", "vietnamese"
    ]) + ["Any cuisine"]

    cuisine = st.selectbox("Preferred cuisine", cuisine_options, index=cuisine_options.index(saved.get("cuisine", "Any cuisine")))

    meal_options = ["Breakfast", "Full Meal", "Sweet Treat", "Snack"]
    meal_type = st.radio("Type of meal", meal_options, index=meal_options.index(saved.get("meal_type", "Full Meal")))

    use_grocery = st.checkbox("I'm willing to go to the grocery store to get missing ingredients.", value=saved.get("use_grocery", False))
    allow_substitutions = st.checkbox("I'm okay with ingredient substitutions if needed.", value=saved.get("allow_substitutions", False))

    preferences = {
        "max_time": max_time,
        "cuisine": cuisine,
        "meal_type": meal_type,
        "use_grocery": use_grocery,
        "allow_substitutions": allow_substitutions
    }

    # Only save if preferences changed
    if preferences != saved:
        with open(file_path, "w") as f:
            json.dump(preferences, f, indent=2)
        st.session_state["user_preferences"] = preferences

    return preferences
