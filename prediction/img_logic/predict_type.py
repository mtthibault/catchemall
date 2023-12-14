import requests
from io import BytesIO
import PIL
from PIL import Image
from pathlib import Path
import os

from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess_input,
)
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inception_preprocess_input,
)

from tensorflow.keras.models import load_model
import tensorflow.keras.applications.resnet50 as resnet50

from prediction.params import *

# Check
print("CATCH_PREDICT_CSV_FILE", CATCH_PREDICT_CSV_FILE)
print("POKEMON_TYPE_LIST", POKEMON_TYPE_LIST)
print("URL_IMG_NORMALFLYING", URL_IMG_NORMALFLYING)
# URL_IMG_FIREBUG = (
#     "https://archives.bulbagarden.net/media/upload/8/83/0851Centiskorch.png"
# )

print("TARGET_SIZE", TARGET_SIZE)
# TARGET_SIZE = (120, 120)

print("LOCAL_DATA_PATH", LOCAL_DATA_PATH)
print("CNN_TRAINED_MODEL", CNN_TRAINED_MODEL)


# *********************************************************
# Preload the model
# *********************************************************
# app.state.model = registry.load_selected_model() # doggos

#  taxifare
# app.state.model = load_model()
# y_pred = app.state.model.predict(X_pred_processed)

# loaded_model = load_model("resnet_v3.keras")
# model_cache_path = Path(f"{LOCAL_DATA_PATH}/computer_vision/models/resnet_v3.keras")
model_path = os.path.join(
    LOCAL_DATA_PATH, "computer_vision", "models", CNN_TRAINED_MODEL
)
# print("model_cache_path", model_cache_path)

# print("model_path", model_path)

# print(f"✅ Load model: {model_path}")

# loaded_model = load_model(model_path)
# print(f"... >>> {loaded_model}")


# Emile 13.12.2023
def getImage(source):
    """
    Get an image from an url, or a file
    """
    # upload image
    # if type(source) == PIL.Image.Image:
    if not isinstance(source, str):
        img = source

    # chemin image
    elif os.path.exists(source):
        img = Image.open(source)

    # url image
    else:
        response = requests.get(source)
        img = Image.open(BytesIO(response.content))

    img = img.resize(TARGET_SIZE)
    print("✅ getImage : image from source resized")
    return img


# Emile 13.12.2023
def resize_prediction(source):
    """
    Resize an image, and convert it to RGB.
    """
    img = getImage(source)
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)

    img_resized = img_resized.convert("RGB")
    img_array = img_to_array(img_resized, data_format="channels_last", dtype="uint8")

    print("✅ resize_prediction : image from source resized, and converted to RGB.")
    return img_array


# Emile 12.12.2023
def predictImage(source, model):
    """
    Predict type from an image, with a model.
    """

    # Test
    # url = URL_IMG_FIREBUG

    # Test local => POKEMON_TYPE_LIST
    # types = train_ds.class_names
    # types = POKEMON_TYPE_LIST
    # target_size = TARGET_SIZE

    # Get the image
    resized_image_array = resize_prediction(source)

    # Reshape the image
    img = resized_image_array.reshape((TARGET_SIZE[0], TARGET_SIZE[1], 3))
    img = resnet50.preprocess_input(img)

    # Make predictions
    res = model.predict(np.expand_dims(img, axis=0))

    # types = train_ds.class_names

    top3_probs = np.sort(res, axis=-1)[:, -3:]
    top3_classes = np.argsort(res, axis=1)[:, -3:]

    predicted_index_1 = top3_classes[0][-1]
    predicted_proba_1 = round(top3_probs[0][-1] * 100, 2)
    predicted_index_2 = top3_classes[0][-2]
    predicted_proba_2 = round(top3_probs[0][-2] * 100, 2)
    predicted_index_3 = top3_classes[0][-3]
    predicted_proba_3 = round(top3_probs[0][-3] * 100, 2)

    print("✅ predictImage: from source")
    print(" --> ", source)
    print(
        f"First predicted type: {POKEMON_TYPE_LIST[predicted_index_1]}: {predicted_proba_1}%"
    )
    print(
        f"Second predicted type: {POKEMON_TYPE_LIST[predicted_index_2]}: {predicted_proba_2}%"
    )
    print(
        f"Third predicted type: {POKEMON_TYPE_LIST[predicted_index_3]}: {predicted_proba_3}%"
    )
    # print(f"First predicted type: {types[predicted_index_1]}: {predicted_proba_1}%")
    # print(f"Second predicted type: {types[predicted_index_2]}: {predicted_proba_2}%")
    # print(f"Third predicted type: {types[predicted_index_3]}: {predicted_proba_3}%")

    # Build dict as api result
    keys_list = [
        POKEMON_TYPE_LIST[predicted_index_1],
        POKEMON_TYPE_LIST[predicted_index_2],
        POKEMON_TYPE_LIST[predicted_index_3],
    ]
    # res_dict = dict.fromkeys(keys_list, None)
    values_list = [
        f"{predicted_proba_1}%",
        f"{predicted_proba_2}%",
        f"{predicted_proba_3}%",
    ]
    key_value_pairs = zip(keys_list, values_list)

    # convert the list of key-value pairs to a dictionary
    res_dict = dict(key_value_pairs)

    # print(POKEMON_TYPE_LIST[predicted_index_3])
    # res_dict = dict(
    #     POKEMON_TYPE_LIST[predicted_index_1]=[predicted_proba_1],
    #     POKEMON_TYPE_LIST[predicted_index_2]=[predicted_proba_2],
    #     POKEMON_TYPE_LIST[predicted_index_3]=[predicted_proba_3]
    # ) # KO pourquoi ? ;-)"

    print("res_dict type is ", type(res_dict), res_dict)
    return res_dict


# def compile_model(model):
#     """
#     Compile the model.
#     Args:
#         model: the model to compile
#     Returns:
#         model: the compiled model
#     """
#     opt = optimizers.Adam(learning_rate=1e-4)
#     model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
#     print("✅ Model compiled")
#     return model


# def predict_labels(model, model_type, *args, **kwargs):
#     """
#     Function that will load the latest model from local disk and use it to predict the breed of the dog in the image.
#     Args:
#         url: url of the image to predict
#     Returns:
#         breed_prediction: dictionary with the top 3 breeds predicted
#         score_prediction: dictionary with the top 3 scores predicted
#     """
#     img = getImage(*args, **kwargs)
#     print("✅ Image successfully loaded")
#     img = img_to_array(img)  # shape = (224, 224, 3)
#     img = img.reshape((-1, 224, 224, 3))
#     print("✅ Image successfully reshaped", img.shape)

#     model = compile_model(model)
#     print("✅ Model successfully loaded and compiled")

#     if model_type == "resnet50":
#         img = resnet_preprocess_input(img)
#         print("✅ Image successfully preprocessed (resnet50)")
#     elif model_type == "inception_v3":
#         img = inception_preprocess_input(img)
#         print("✅ Image successfully preprocessed (inception_v3)")
#     print("✅ Predicting breed...")

#     res = model.predict(img)
#     print("✅ Breed predicted")
#     indexes = np.argsort(res)[0][-3:][::-1]
#     predicts = np.sort(res)[0][::-1][0:3]

#     breed_prediction = {
#         "first": breed[indexes[0]],
#         "second": breed[indexes[1]],
#         "third": breed[indexes[2]],
#     }
#     score_prediction = {
#         "first": float(round(predicts[0], 2)),
#         "second": float(round(predicts[1], 2)),
#         "third": float(round(predicts[2], 2)),
#     }
#     output = {"prediction": breed_prediction, "score": score_prediction}
#     return output

# Emile Test local 13.12.2023
# predictImage(URL_IMG_NORMALFLYING, loaded_model)

file_path = os.path.join(LOCAL_DATA_PATH, "computer_vision", "vivillon.png")
# predictImage(file_path, loaded_model)

# file_path = os.path.join(LOCAL_DATA_PATH, "computer_vision", "Abra_4750.png")
# predictImage(file_path, loaded_model)
