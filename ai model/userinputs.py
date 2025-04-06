import streamlit as st
import json
import os

USER_PREFS_DIR = "user_data"
os.makedirs(USER_PREFS_DIR, exist_ok=True)

def get_user_preferences():
    st.markdown("### Customize Your Recipe Preferences")

    # Load existing preferences if available
    if "username" in st.session_state:
        file_path = os.path.join(USER_PREFS_DIR, f"{st.session_state['username']}_prefs.json")
        if os.path.exists(file_path) and "user_preferences" not in st.session_state:
            with open(file_path, "r") as f:
                st.session_state["user_preferences"] = json.load(f)

    # Pre-fill from session state if available
    defaults = st.session_state.get("user_preferences", {})

    max_time = st.slider(
        "Maximum cooking time (minutes)", 
        min_value=5, 
        max_value=120, 
        value=defaults.get("max_time", 30)
    )

    cuisine_options = [
        "italian", "mexican", "chinese", "indian", "thai", "french",
        "greek", "japanese", "american", "spanish", "moroccan",
        "vietnamese", "korean", "caribbean", "irish", "german",
        "Any cuisine"
    ]
    cuisine = st.selectbox("Preferred cuisine", options=cuisine_options, index=cuisine_options.index(defaults.get("cuisine", "Not applicable")))

    meal_type = st.radio(
        "Type of meal",
        options=["Breakfast", "Full Meal", "Sweet Treat", "Snack"],
        horizontal=True,
        index=["Breakfast", "Full Meal", "Sweet Treat", "Snack"].index(defaults.get("meal_type", "Breakfast"))
    )

    use_grocery = st.checkbox(
        "I'm willing to go to the grocery store to get missing ingredients.",
        value=defaults.get("use_grocery", False)
    )

    allow_substitutions = st.checkbox(
        "I'm okay with ingredient substitutions if needed.",
        value=defaults.get("allow_substitutions", False)
    )

    preferences = {
        "max_time": max_time,
        "cuisine": cuisine,
        "meal_type": meal_type,
        "use_grocery": use_grocery,
        "allow_substitutions": allow_substitutions
    }

    # Save in session state
    st.session_state["user_preferences"] = preferences

    # Also save to file if user is logged in
    if "username" in st.session_state:
        with open(file_path, "w") as f:
            json.dump(preferences, f)

    return preferences

if __name__ == "__main__":
    print(get_user_preferences())