import streamlit as st
import json
import os

USER_PREFS_FILE = os.path.join(os.path.dirname(__file__), 'user_preferences.json')

def get_user_preferences():
    # Load existing preferences
    if os.path.exists(USER_PREFS_FILE) and "user_preferences" not in st.session_state:
        with open(USER_PREFS_FILE, "r") as f:
            st.session_state["user_preferences"] = json.load(f)

    saved = st.session_state.get("user_preferences", {})

    time_options = ["15", "30", "60", "Any"]
    max_time = st.selectbox("Maximum cooking time (minutes)", time_options, index=time_options.index(saved.get("max_time", "30")))

    cuisine_options = sorted([
        "american", "caribbean", "chinese", "french", "german", "greek",
        "indian", "irish", "italian", "japanese", "korean", "mexican",
        "moroccan", "spanish", "thai", "vietnamese"
    ]) + ["Any cuisine"]

    cuisine = st.selectbox("Preferred cuisine", cuisine_options, index=cuisine_options.index(saved.get("cuisine", "Any cuisine")))

    use_grocery = st.checkbox("I'm willing to go to the grocery store to get missing ingredients.", value=saved.get("use_grocery", False))
    allow_substitutions = st.checkbox("I'm okay with ingredient substitutions if needed.", value=saved.get("allow_substitutions", False))

    # Correct the preferences dictionary creation
    preferences = {
        "max_time": max_time,
        "cuisine": cuisine,
        "use_grocery": use_grocery,
        "allow_substitutions": allow_substitutions
    }
    
    # Save preferences to session state
    st.session_state["user_prefs"] = preferences
    
    return preferences

def load_user_profile():
    """Load user profile from JSON file."""
    profile_path = "user_profile.json"
    try:
        if os.path.exists(profile_path):
            with open(profile_path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading user profile: {e}")
    return {}

def save_user_profile(profile):
    """Save user profile to JSON file."""
    profile_path = "user_profile.json"
    try:
        with open(profile_path, "w") as f:
            json.dump(profile, f)
    except Exception as e:
        print(f"Error saving user profile: {e}")