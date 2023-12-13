import streamlit as st
import requests
from PIL import Image

# Importer mes fonctions
# def get_pokemon_type_from_image
# def get_pokemon_capture_rate
# def get_pokemon_evolutions

st.set_page_config(
    page_title="My PokéApp",
    page_icon="🦈",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Header

## Image
image_path = "images/pokeball.png"
image = Image.open(image_path)
st.image(image, width=50, use_column_width=False)

## Main Title
st.title(
    """
My PokéApp
"""
)

# Pokémon Type
with st.container(border=True):
    st.markdown(
        """
    ### 🩻 You want to know the type of a Pokémon.
    """
    )

    st.warning("Please take a picture or upload an image.")
    # st.warning("Please take a picture or upload an image.")

    col1, col2 = st.columns(2)
    # Caméra
    with col1:
        img_file_buffer = st.camera_input("Take a picture of the Pokémon:")

        if img_file_buffer is not None:
            # type_image = get_pokemon_type_from_image(uploaded_file)
            # st.write(f"The type of the Pokémon in the image is: {type_image['type']}")
            ### Display the image user uploaded
            st.image(
                Image.open(img_file_buffer), caption="Here's the photo you took ☝️"
            )

    # Image uploadée
    with col2:
        uploaded_file = st.file_uploader(
            "Upload your Pokémon image:", type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            ### Display the image user uploaded
            st.image(
                Image.open(uploaded_file), caption="Here's the image you uploaded ☝️"
            )
            ### Get bytes from the file buffer
            img_bytes = uploaded_file.getvalue()

            # Promt results
            st.success("**Your Pokemon's type is : Fire !**", icon="🤖")

            # type_image = get_pokemon_type_from_image(uploaded_file)
            # st.write(f"The type of the Pokémon in the image is: {type_image['type']}")
