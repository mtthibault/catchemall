import streamlit as st
import requests
from PIL import Image

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

## Caption
st.caption('This webapp is part of our project for Le Wagon, Batch #1437. ✨')

col1, col2 = st.columns(2)

with col1:
    # Titre
    st.markdown('''Greetings, <span style="color:#FFA500; font-weight:bold">Pokémon Trainers</span>! 🎉''', unsafe_allow_html=True)
    st.markdown('''
        We are thrilled to present <span style="color:#FFA500; font-weight:bold">My PokéApp</span>, a powerful tool designed to <span style="color:#FFA500; font-weight:bold">enhance your Pokémon journey</span> by providing insights beyond the capabilities of your traditional Pokédex. 🤖
    ''', unsafe_allow_html=True)

with col2:
    # GIF
    gif_url = "https://media.tenor.com/s8QfDEQ7QIgAAAAC/pokemon-hello.gif"
    # Display the GIF
    st.image(gif_url, width=200)

# Description

st.markdown('''
Click one of the two buttons on the left to:
''')

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"🩻 **<u>Identify Pokémon Type from an Image</u>** \n\n"
                f"Discover the elemental type of encountered Pokémon through <span style='color:#FFA500; font-weight:bold'>image analysis</span>.", unsafe_allow_html=True)

with col2:
    st.markdown(f"🪤 **<u>Predict Catchability</u>** \n\n"
                f"Input Pokédex stats to calculate <span style='color:#FFA500; font-weight:bold'>capture difficulty</span>.", unsafe_allow_html=True)
