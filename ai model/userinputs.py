import streamlit as st
import json
import os

def get_user_preferences():
    """Get and save user preferences for recipe generation"""
    # Default preferences without user input
    prefs = {
        "cuisine": "Any Cuisine",  # Default to any cuisine
        "max_time": "60",          # Default to 60 minutes
        "servings": "4",           # Default to 4 servings
        "dietary": "None",         # No dietary restrictions
        "allow_substitutions": True,
        "use_grocery": True
    }
    
    # If there are saved preferences in session state, use those
    if "user_prefs" in st.session_state:
        prefs.update(st.session_state["user_prefs"])
    
    # Save preferences to session state
    st.session_state["user_prefs"] = prefs
    
    return prefs

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