import streamlit as st

def get_user_preferences():
    st.markdown("### Customize Your Recipe Preferences")

    # Servings (stored but not used yet in filtering)
    servings = st.number_input("How many servings do you need?", min_value=1, max_value=20, value=2)

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

    return {
        "servings": servings,
        "max_time": max_time,
        "cuisine": cuisine,
        "meal_type": meal_type,
        "use_grocery": use_grocery
    }

