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

# Evolutions
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
