import streamlit as st

def get_user_preferences():
    st.markdown("### Customize Your Recipe Preferences")

    # Max time
    max_time = st.slider("Maximum cooking time (minutes)", min_value=5, max_value=120, value=30)

    # Cuisine
    cuisine_options = [
    "italian", "mexican", "chinese", "indian", "thai", "french",
    "greek", "japanese", "american", "spanish", "moroccan",
    "vietnamese", "korean", "caribbean", "irish", "german",
    "Not applicable"
    ]

    cuisine = st.selectbox("Preferred cuisine", options=cuisine_options)


    # Type of meal
    meal_type = st.radio(
        "Type of meal",
        options=["Breakfast", "Full Meal", "Sweet Treat", "Snack"],
        horizontal=True
    )

    use_grocery = st.checkbox("I'm willing to go to the grocery store to get missing ingredients.")
    allow_substitutions = st.checkbox("I'm okay with ingredient substitutions if needed.")


    return {
        "max_time": max_time,
        "cuisine": cuisine,
        "meal_type": meal_type,
        "use_grocery": use_grocery,
        "allow_substitutions": allow_substitutions
    }

# Simply print the returned dictionary without altering the function code.
if __name__ == "__main__":
    print(get_user_preferences())
