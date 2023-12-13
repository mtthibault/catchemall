import os
import numpy as np

##################  VARIABLES  ##################


##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser("~"), ".lewagon", "mlops", "data")
# LOCAL_REGISTRY_PATH = os.path.join(
#     os.path.expanduser("~"), ".lewagon", "mlops", "training_outputs"
# )
# Emile 11.12.2023
CNN_REFERENCE_CSV_FILE = os.environ.get("CNN_REFERENCE_CSV_FILE")

LOCAL_DATA_PATH = os.path.join(
    os.path.expanduser("~"), "code", "mtthibault", "catchemall", "raw_data"
)

LOCAL_REGISTRY_PATH = os.path.join(
    os.path.expanduser("~"),
    "code",
    "mtthibault",
    "catchemall",
    "raw_data",
    "training_outputs",
)

# Types of Pokemon
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

# Emile 12.12.2023
# CNN : size of images
TARGET_SIZE = (120, 120)

# For testing
GRASS = "https://archives.bulbagarden.net/media/upload/0/0c/0810Grookey.png"
FIREBUG = "https://archives.bulbagarden.net/media/upload/8/83/0851Centiskorch.png"
GRASSPSYCHIC = "https://archives.bulbagarden.net/media/upload/8/80/1010Iron_Leaves.png"
NORMALFLYING = "https://archives.bulbagarden.net/media/upload/b/b3/HOME0931W.png"
