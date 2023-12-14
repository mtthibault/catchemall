import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Emile
import random  # for generating random catchability
from prediction.ml_logic.preprocessor import preprocess_features
# from prediction.ml_logic.registry import load_model  # , save_model, save_results
from tensorflow.keras.models import load_model
import io
from PIL import Image

# import json # >> pas utile ici

# Emile 12.12.2023
from prediction.params import *

try:
    TARGET_SIZE
except NameError:
    TARGET_SIZE = None
if TARGET_SIZE is None:
    print("TARGET_SIZE isn't read from params.py, it will be assigned")
    TARGET_SIZE = (120, 120)
    print("TARGET_SIZE assigned", TARGET_SIZE)
else:
    print("OK TARGET_SIZE", TARGET_SIZE)

print("CATCH_PREDICT_CSV_FILE", CATCH_PREDICT_CSV_FILE)
print("POKEMON_TYPE_LIST", POKEMON_TYPE_LIST)
print("URL_IMG_GRASS", URL_IMG_GRASS)

print("CNN_TRAINED_MODEL", CNN_TRAINED_MODEL)

from prediction.img_logic.predict_type import *


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Model path
# local_model_path = os.path.join(
#     LOCAL_DATA_PATH, "computer_vision", "models", CNN_TRAINED_MODEL
# )
# print("local_model_path", local_model_path)

# cwd = os.getcwd()
# print("os.getcwd()", os.getcwd())
# # test_path = os.path.join(
# #     os.getcwd(), "..", "..", "..", "data-context-and-setup/data/csv"
# # )

# print("root_dir", os.path.dirname(os.path.dirname(__file__)))

# parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
# print("parent_dir", parent_dir)

# app_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
# print("app_dir", app_dir)

# model_path = os.path.join(
#     app_dir, "raw_data", "computer_vision", "models", CNN_TRAINED_MODEL
# )
# print("model_path", model_path)

model_path = os.path.join(
    LOCAL_DATA_PATH, "computer_vision", "models", CNN_TRAINED_MODEL
)

# *********************************************************
# Preload the model
# *********************************************************
print("model_path", model_path)
print(f"✅ Load model: {model_path}")
app.state.model = load_model(model_path)
print(f"... >>> {app.state.model}")
print(f"✅ ✅ Model is loaded !")


# predicts from url provided by user
@app.get("/predict_url")
def predict_url(url):
    print("✅ post /predict_url: from source")
    predict_types_result = predictImage(url, app.state.model)
    return predict_types_result
    # return {"Cath'em All": "get /predict_url"}


# predicts from file provided by user
@app.post("/predict_file")
async def receive_image(img: UploadFile = File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    img_decode = Image.open(io.BytesIO(contents))

    print("✅ post /predict_file: from source")
    predict_types_result = predictImage(img_decode, app.state.model)
    return predict_types_result
    # return {"Cath'em All": "post /predict_file"}


@app.get("/predict")
def predict(feature_1: int = 100, fetaure_2: int = 20):
    """
    Compute catchability.
    """
    print(">>>>>>>>> ✅ Dans get /predict")

    # Define X_pred
    # X_pred = pd.DataFrame({"feature_1": 100, "feature_2": 20})
    # print(">>>>>>>>> X_pred")
    # print(X_pred)

    # Compute X_pred_processed
    print(">>>>>>>>> preprocess_features(X_pred)")
    # X_pred_processed = preprocess_features(X_pred)
    # print(".........")
    # print(type(X_pred_processed))
    # print(".........")
    # print(X_pred_processed)

    # Enable faster predictions
    # app = FastAPI()
    # app.state.model = load_model()
    # y_pred = app.state.model.predict(X_pred_processed)

    # Load model OK without optimization
    # model = load_model()
    # assert model is not None
    # y_pred = model.predict(X_pred_processed)

    # print(">>>>>>>>> ✅ Apres predict : y_pred, with shape", y_pred.shape)

    # Load model OK
    # model = load_model()
    # print(">>>>>>>>> Apres load_model()")
    # # return {'Avant assert'}
    # assert model is not None
    # print(">>>>>>>>> Apres assert")
    # # return {'Apres assert'}

    # Predict OK
    # print(">>>>>>>>> Avant model.predict(X_pred_processed)")
    # y_pred = model.predict(X_pred_processed)
    # print(">>>>>>>>> ✅ Apres predict : y_pred, with shape", y_pred.shape)

    # Return y_pred as json
    y_pred = random.randrange(0, 255)
    res_y_pred = round(float(y_pred), 2)
    res_dict = {"catchability": res_y_pred}
    print("res_dict type is ", type(res_dict), res_dict)
    return res_dict

    # return {'fare_amount': float("{:.2f}".format(y_pred[0][0]))}

    # Convert Python to JSON >> Emile : inutile et KO au final ici
    #  >>> {\n    \"fare_amount\": 9.84\n}
    # print(json.dumps(res_dict, indent = 4))
    # return json.dumps(res_dict, indent = 4)


@app.get("/")
def root():
    return {"Cath'em All": "Hello !"}
