import streamlit as st
from PIL import Image

st.title("I'm Hungry")
st.write("Upload an image of your ingredients and get a recipe!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your ingredients", use_column_width=True)
    st.write("Analyzing...")
    # Placeholder for AI model result
    st.success("This is where your recipe will appear!")
