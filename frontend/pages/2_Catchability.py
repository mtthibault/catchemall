import streamlit as st
import requests
from PIL import Image
import os

# "Constantes"
# url = "https://weather.lewagon.com/geo/1.0/direct?q=Barcelona"
PREDICT_API_URL = os.environ.get("PREDICT_API_URL")
# print("PREDICT_API_URL", PREDICT_API_URL)

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


# Call API
def call_api():
    # response = requests.get(url).json()
    response = requests.get(PREDICT_API_URL + "/predict").json()

    print("Test API", response)
    print(f"Test API - {response['catchability']}")
    print("called url", PREDICT_API_URL + "/predict")

    api_results = response["catchability"]
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
    st.markdown("### 🪤 Work out the catchability of a Pokémon.")

    # List of Pokémon names and image paths
    pokemon_options = [
        ("SUPERCASPICHU", "images/predict-images/steel.png"),
        ("PAUVCASPINOR", "images/predict-images/dark.png"),
        ("MOYENCASPOLO", "images/predict-images/grass.png"),
        ("LEGMOYENCASPO", "images/predict-images/ghost.png"),
        (
            "RANDOMCASPICHU",
            "images/predict-images/poison.png",
        ),
        (
            "SUPERMORGANICA",
            "images/predict-images/dragon.png",
        ),
    ]
    # Emile 07.12.23 : Enable reset widgets
    if st.session_state.get("reset_btn"):
        st.session_state["predict_selection_method_field"] = None
        st.session_state["predict_selected_pokemon_name"] = None
        st.session_state["pokemon_name_field"] = ""
        st.session_state["pokemon_type_field"] = None
        st.session_state["number_input_field"] = 0
        st.session_state["is_legendary_radio"] = "Yes"
        st.session_state["base_total_score"] = ""
        st.session_state["predict_sp_defense_field"] = ""
        st.session_state["predict_defense_field"] = ""
        st.session_state["predict_hp_field"] = 0
        st.session_state["predict_height_field"] = ""
        st.session_state["predict_speed_field"] = ""
        st.session_state["predict_weight_kg_field"] = ""
        st.session_state["predict_attack_field"] = ""
        st.session_state["predict_sp_attack_field"] = ""
        st.session_state["predict_legendary_field"] = ""
        st.session_state["pokemon_name_field"] = ""
        # st.session_state[""] = ""
    # if st.session_state.get("prompt_result"):
    #     st.session_state["catchability_field"] = result

    col1, col2, col3 = st.columns(3)
    selection_method = st.radio(
        "Choose your input method:",
        ("Select a Pokémon", "Enter attributes manually"),
        key="predict_selection_method_field_1",
    )

    # hp = st.number_input("Numbers of HP:", key="number_input_field")
    if selection_method == "Select a Pokémon":
        with col1:  # Extract names for the dropdown
            pokemon_names = [name for name, _ in pokemon_options]

            # Selectbox to choose a Pokémon
            selected_pokemon_name = st.selectbox(
                "Choose a Pokémon:", pokemon_names, key="predict_selected_pokemon_name"
            )
        with col2:
            # Find the selected Pokémon's image and display it
            for name, img_path in pokemon_options:
                if name == selected_pokemon_name:
                    st.image(img_path, width=100)  # Adjust width as needed

    elif selection_method == "Enter attributes manually":
        with col1:
            # selection_method = st.radio("Choose your input method:", ("Select a Pokémon", "Enter attributes manually"), key="predict_selection_method_field_2")
            base_egg_steps = st.slider(
                "Base Egg Steps",
                min_value=0,
                max_value=50000,
                value=20,
                step=100,
                key="predict_base_egg_steps_field",
            )
            hp = st.slider(
                "HP",
                min_value=0,
                max_value=255,
                value=10,
                step=1,
                key="predict_hp_field",
            )
            base_total = st.slider(
                "Base Total",
                min_value=180,
                max_value=780,
                value=10,
                step=1,
                key="predict_base_total_field",
            )
            sp_defense = st.slider(
                "Special Defense",
                min_value=0,
                max_value=200,
                value=10,
                step=1,
                key="predict_sp_defense_field",
            )
        with col2:
            attack = st.slider(
                "Attack",
                min_value=180,
                max_value=780,
                value=10,
                step=1,
                key="predict_attack_field",
            )
            sp_attack = st.slider(
                "Special Attack",
                min_value=0,
                max_value=200,
                value=10,
                step=1,
                key="predict_sp_attack_field",
            )
            defense = st.slider(
                "Defense",
                min_value=0,
                max_value=250,
                value=10,
                step=1,
                key="predict_defense_field",
            )
            is_legendary = st.selectbox(
                "Is Legendary",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="predict_legendary_field",
            )
        with col3:
            speed = st.slider(
                "Speed",
                min_value=0,
                max_value=200,
                value=10,
                step=1,
                key="predict_speed_field",
            )
            height_m = st.slider(
                "Height (m)",
                min_value=0.0,
                max_value=10.0,
                value=0.1,
                step=0.1,
                key="predict_height_field",
            )
            weight_kg = st.slider(
                "Weight (kg)",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                step=0.1,
                key="predict_weight_kg_field",
            )

    # Get results
    # --------------------------------------------------------
    # def display_results(value_to_display):
    #     st.session_state.result_field = (value_to_display)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Compute the catchability", key="compute_btn"):
            results = call_api()  # >> OK
            catch_value = results

            st.write("**The higher, the easier**")
            st.slider(
                "Catch slider label",
                label_visibility="hidden",
                min_value=0,
                max_value=255,
                value=int(catch_value),
                # disabled=True,
                key="catch_slider",
            )

        # Emile 07.12.23 : Enable reset widgets
        st.button("Reset inputs", key="reset_btn")
