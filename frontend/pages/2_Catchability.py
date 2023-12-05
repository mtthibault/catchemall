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

# Capture Rate
with st.container(border=True):
    st.markdown("""
    ### 🪤 You want to know the catchabiliy of a Pokémon.
    """)

    col1, col2 = st.columns(2)
    with col1:
        pokemon_name = st.text_input("Name of the Pokémon:")
    with col2:
        pokemon_type = st.text_input("Type:")
    col1, col2 = st.columns(2)
    with col1:
        hp = st.number_input("Numbers of HP:")
    with col2:
        is_legendary = st.radio("Is this Pokémon legendary ?", ("Yes", "No")) == "Yes"

    if st.button("Compute the catchability"):
        capture_difficulty = calculate_capture_difficulty(pokemon_name, pokemon_type, hp, is_legendary)
        st.write(f"The catchability of {pokemon_name.capitalize()} is : {capture_difficulty}")
