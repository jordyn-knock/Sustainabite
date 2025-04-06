import streamlit as st
from pantry_tab import render_pantry_tab
from generator_tab import render_generator_tab
from favourites_tab import render_favourites_tab
from storage import load_favourites
from auth import login_form  
from PIL import Image

st.set_page_config(page_title="I'm Hungry", layout="centered")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.title("Welcome to I'm Hungry")
    st.subheader("Please log in to continue")
    login_form()
    st.stop()  

st.title("I'm Hungry")
st.write("Upload an image of your ingredients and get a recipe!")

# Load user-specific favourites
if "favourites" not in st.session_state:
    st.session_state.favourites = load_favourites()

# Define your tabs
tab1, tab2, tab3 = st.tabs(["Upload Image", "Pantry & My List", "Favourites"])

with tab1:
    render_generator_tab()
with tab2:
    render_pantry_tab()
with tab3:
    render_favourites_tab()

