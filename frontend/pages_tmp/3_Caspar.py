import streamlit as st
import requests
from PIL import Image
import os

# Importer mes fonctions
# def get_pokemon_type_from_image
# def get_pokemon_capture_rate
# def get_pokemon_evolutions

# "Constantes"
# url = "https://weather.lewagon.com/geo/1.0/direct?q=Barcelona"
PREDICT_API_URL = os.environ.get("PREDICT_API_URL")

POKEMON_TYPE_LIST = [
    "Bug",
    "Dark",
    "Dragon",
    "Electric",
    "Fairy",
    "Fighting",
    "Fire",
    "Flying",
    "Ghost",
    "Grass",
    "Ground",
    "Ice",
    "Normal",
    "Poison",
    "Psychic",
    "Rock",
    "Steel",
    "Water",
]


# params = {"feature1": param1, "feature2": param2}  # 0 for Sunday, 1 for Monday, ...
# response = requests.get(url, params=params)
def call_api():
    # response = requests.get(url).json()
    response = requests.get(PREDICT_API_URL).json()
    city = response[0]
    print(f"Test API - {city['name']}: ({city['lat']}, {city['lon']})")
    api_results = city
    return api_results


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

# Capture Rate
with st.container(border=True):
    st.markdown(
        """
    ### 🪤 You want to know the catchability of a Pokémon.
    """
    )

    # Emile 07.12.23 : Enable reset widgets
    if st.session_state.get("reset_btn"):
        st.session_state["pokemon_name_field"] = ""
        st.session_state["pokemon_type_field"] = None
        st.session_state["number_input_field"] = 0
        st.session_state["is_legendary_radio"] = "Yes"
        # st.session_state[""] = ""
    # if st.session_state.get("prompt_result"):
    #     st.session_state["catchability_field"] = result

    # Parameters, input fields
    col1, col2 = st.columns(2)
    with col1:
        pokemon_name = st.text_input("Pokémon's name:", key="pokemon_name_field")
    with col2:
        pokemon_type = st.selectbox(
            "Type:",
            POKEMON_TYPE_LIST,
            index=None,
            placeholder="Select a type...",
            key="pokemon_type_field",
        )
        # st.write("You selected:", pokemon_type)

    col1, col2 = st.columns(2)
    with col1:
        hp = st.number_input("Numbers of HP:", key="number_input_field")
    with col2:
        is_legendary = (
            st.radio(
                "Is this Pokémon legendary ?", ("Yes", "No"), key="is_legendary_radio"
            )
            == "Yes"
        )

    # Get results
    # --------------------------------------------------------
    # def display_results(value_to_display):
    #     st.session_state.result_field = (value_to_display)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Compute the catchability", key="compute_btn"):

            results = call_api()  # >> OK
            st.write("**The higher, the easier**")
            st.slider(
                "Catch slider label",
                label_visibility="hidden",
                min_value=0,
                max_value=255,
                value=int(results["lat"]),
                # disabled=True,
                key="catch_slider"
            )

        # Emile 07.12.23 : Enable reset widgets
        st.button("Reset inputs", key="reset_btn")
