import streamlit as st
from pantry_tab import render_pantry_tab
from favourites_tab import render_favourites_tab
from PIL import Image

#this is the main page 
st.title("I'm Hungry")
st.write("Upload an image of your ingredients and get a recipe!") # or whatever you want to write here

# Define your tabs
tab1, tab2 = st.tabs(["Pantry & My List", "Favourites"])

with tab1:
    render_pantry_tab()
with tab2:
    render_favourites_tab()

uploaded_files = st.file_uploader( #this is where the user uploads their images
    "Upload images of your ingredients", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

ingredient_list = []

if uploaded_files: 
    #for loop so that the Ai model can go through all the uploaded images
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption="Your ingredients",  use_container_width=True)
        st.write("We're processing your information")
        # This is where we call the AI model 
        extracted = ["tomato", "garlic"]  #this is just an example!!!
        ingredient_list.extend(extracted)
        st.success("Files have successfully uploaded!")

