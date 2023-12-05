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

# Titre
st.markdown('''
### Welcome, trainer. What do you want to do ?
''')

# Pokémon Type
if st.button("Know the type of the Pokémon"):
    with st.container(border=True):
        st.markdown("""
        ### 🔥 You want to know the type of a Pokémon.
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

# Capture Rate
if st.button("Know the capture rate of a Pokémon"):
    with st.container(border=True):
        st.markdown("""
        ### 🪤 You want to know the capture rate of a Pokémon.
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

        if st.button("Compute the capture rate"):
            capture_difficulty = calculate_capture_difficulty(pokemon_name, pokemon_type, hp, is_legendary)
            st.write(f"The capture of {pokemon_name.capitalize()} is : {capture_difficulty}")

# Evolutions
if st.button("Know the evolutions of a Pokémon"):
    with st.container(border=True):
        st.markdown("""
        ### 🐨 You want to know the evolutions of a Pokémon.
        """)
        pokemon_name_for_evolutions = st.text_input("Name of the Pokémon:")

        if st.button("Discover the evolutions of this Pokémon"):
            evolutions_info = get_pokemon_evolutions(pokemon_name_for_evolutions)
            if evolutions_info['evolutions']:
                st.write(f"The evolutions of {pokemon_name_for_evolutions} are:")
                for evolution in evolutions_info['evolutions']:
                    st.write(f"{evolution['name']} ({evolution['current_level']}/{evolution['final_level']})")
            else:
                st.write(f"{pokemon_name_for_evolutions.capitalize()} has no evolutions.")
