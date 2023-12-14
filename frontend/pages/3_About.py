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

st.info(f"We are excited to share with you the culmination of our journey at **Le Wagon** – our **final project**! This work represents the synthesis of the diverse skills we have acquired throughout the intensive bootcamp, showcasing our proficiency in Machine Learning, Deep Learning, Web Scraping, and MLOps.\n\n"
        , icon="🔥")

st.divider()

st.header(f"About Our Final Project")
f"As part of our rigorous training at Le Wagon, we undertook the challenge of developing a project that not only demonstrates our understanding of key concepts but also serves as a testament to our **practical application of acquired skills**. "
f"Our goal was to leverage the full spectrum of knowledge gained during the bootcamp and bring it to life through a comprehensive and innovative project about the **Pokémon Universe**."

# GIF
gif_url = "https://www.gifcen.com/wp-content/uploads/2022/10/pokemon-gif-5.gif"
# Display the GIF
st.image(gif_url, width=300)

github_url = "https://github.com/mtthibault/catchemall"
st.markdown(f"If you would like to check out our work, don't hesitate to look at our [GitHub]({github_url}) ! 🐱")

st.divider()

st.header('Meet our team 🕺🕺🕺💃')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image('https://avatars.githubusercontent.com/u/71439067?v=4', caption='Emile NGUYEN')
with col2:
    st.image('https://avatars.githubusercontent.com/u/114233222?v=4', caption='Caspar RITCHIE')
with col3:
    st.image('https://avatars.githubusercontent.com/u/61580635?v=4', caption='Benjamin PERREAUX')
with col4:
    st.image('https://avatars.githubusercontent.com/u/103380937?v=4', caption='Morgane THIBAULT')
