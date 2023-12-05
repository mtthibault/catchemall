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

col1, col2 = st.columns(2)

with col1:
    # Titre
    st.markdown('''Greetings, **Pokémon Trainers**! 🎉''')
    st.markdown('''
    We are thrilled to present **My PokéApp**, a powerful tool designed to **enhance your Pokémon journey** by providing insights beyond the capabilities of your traditional Pokédex. 🤖
    ''')

with col2:
    # GIF
    gif_url = "https://media.tenor.com/s8QfDEQ7QIgAAAAC/pokemon-hello.gif"
    # Display the GIF
    st.image(gif_url, width=200)

# Description

st.markdown('''
Click one of the three buttons on the left to:
''')
st.markdown(f"🩻 <u>Identify Pokémon Type from an Image:</u> Discover the elemental type of encountered Pokémon through **image analysis**.", unsafe_allow_html=True)
st.markdown(f"🪤 <u>Predict Capture Difficulty:</u> Input Pokédex stats to calculate **capture difficulty**.", unsafe_allow_html=True)
st.markdown(f"🐨 <u>Explore Evolutionary Family:</u> Input a Pokémon's name to unveil its **evolutionary family** and plan strategic team growth.", unsafe_allow_html=True)
