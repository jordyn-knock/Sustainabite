import streamlit as st
from pantry_tab import render_pantry_tab
from generator_tab import render_generator_tab
from favourites_tab import render_favourites_tab
from PIL import Image

#this is the main page 
st.title("I'm Hungry")
st.write("Upload an image of your ingredients and get a recipe!") # or whatever you want to write here

# Define your tabs
tab1, tab2, tab3 = st.tabs(["Upload Image", "Pantry & My List", "Favourites"])

with tab1:
    render_generator_tab()
with tab2:
    render_pantry_tab()
with tab3:
    render_favourites_tab()


