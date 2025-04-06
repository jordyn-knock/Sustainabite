import streamlit as st
from pantry_tab import render_pantry_tab
from generator_tab import render_generator_tab
from favourites_tab import render_favourites_tab
from storage import load_favourites
from PIL import Image

st.set_page_config(page_title="Sustainabite", layout="centered")

st.title("I'm Hungry")
st.write("Upload an image of your ingredients and get a recipe!")

# Load favourites if not already loaded
if "favourites" not in st.session_state:
    st.session_state.favourites = load_favourites()

# Define your tabs
tab1, tab2, tab3 = st.tabs(["Find Recipe", "Pantry Cupboard", "Favourites"])

with tab1:
    render_generator_tab()
with tab2:
    render_pantry_tab()
with tab3:
    render_favourites_tab()
