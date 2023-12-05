import streamlit as st
import requests
from PIL import Image

# Importer mes fonctions
#def get_pokemon_type_from_image
#def get_pokemon_capture_rate
#def get_pokemon_evolutions

st.set_page_config(
    page_title="My PokéApp",
    page_icon="🦈",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None)

# Header

## Image
image_path = "images/pokeball.png"
image = Image.open(image_path)
st.image(image, width=50, use_column_width=False)

## Main Title
st.title('''
My PokéApp
''')

# Pokémon Type
with st.container(border=True):
    st.markdown("""
    ### 🩻 You want to know the type of a Pokémon.
    """)

    st.warning("Please take a picture or upload an image.")

    col1, col2 = st.columns(2)
    # Caméra
    with col1:
        uploaded_file = st.camera_input("Take a picture of the Pokémon:")

        if uploaded_file:
            type_image = get_pokemon_type_from_image(uploaded_file)
            st.write(f"The type of the Pokémon in the image is: {type_image['type']}")

    # Image uploadée
    with col2:
        uploaded_file = st.file_uploader("Upload your Pokémon image:", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            type_image = get_pokemon_type_from_image(uploaded_file)
            st.write(f"The type of the Pokémon in the image is: {type_image['type']}")
