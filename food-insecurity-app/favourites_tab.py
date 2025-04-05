#this is for the favourites tab
import streamlit as st

def render_favourites_tab():
    st.header("My Favourite Recipes")
    if "favourites" not in st.session_state:
        st.session_state.favourites = []

    if st.session_state.favourites:
        for i, recipe in enumerate(st.session_state.favourites):
            with st.expander(f"{recipe['name']}"):
                st.write(f"**Time:** {recipe['time']} | **Cuisine:** {recipe['cuisine']}")
                for j, step in enumerate(recipe["steps"], 1):
                    st.write(f"{j}. {step}")
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.favourites.pop(i)
                    st.experimental_rerun()
    else:
        st.write("You haven't saved any recipes yet.")
