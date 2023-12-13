import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

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

from prediction.params import *

# Check
print("CATCH_PREDICT_CSV_FILE", CATCH_PREDICT_CSV_FILE)
print("POKEMON_TYPE_LIST", POKEMON_TYPE_LIST)
# print("URL_IMG_NORMALFLYING", URL_IMG_NORMALFLYING)
URL_IMG_FIREBUG = (
    "https://archives.bulbagarden.net/media/upload/8/83/0851Centiskorch.png"
)

# print("TARGET_SIZE", TARGET_SIZE)
TARGET_SIZE = (120, 120)

# Preload the model
# app.state.model = registry.load_selected_model() # doggos

print("LOCAL_DATA_PATH", LOCAL_DATA_PATH)

# loaded_model = load_model("resnet_v3.keras")
model_cache_path = Path(f"{LOCAL_DATA_PATH}/computer_vision/models/resnet_v3.keras")
model_path = os.path.join(
    LOCAL_DATA_PATH, "computer_vision", "models", "resnet_v3.keras"
)
print("model_cache_path", model_cache_path)
print("model_path", model_path)

print(f"✅ Load model: {model_path}")

loaded_model = load_model(model_path)
print(f"... >>> {loaded_model}")


#  Doggos
def getImage(img=None, url_with_pic: str = "", show=False):
    """
    Get an image provided its url and resize it.
    The size of the image is 224x224.
    """
    print(
        f"✅ get image received: img={img is not None}, url_with_pic={url_with_pic is not None}"
    )
    if url_with_pic:
        response = requests.get(url_with_pic)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(img).convert("RGB")
    if show:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    img = img.resize((224, 224))
    print("✅ image resized")
    return img


# Emile 12.12.2023
def getImage(url):
    """
    Get an image from an url
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    # img = img.resize((120, 120))
    img = img.resize(TARGET_SIZE)
    print("✅ getImage : image from url resized")
    return img


# Emile 12.12.2023
def resize_prediction(url):
    """
    Resize an image from an url, and convert it to RGB.
    """
    img = getImage(url)
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)

    img_resized = img_resized.convert("RGB")
    img_array = img_to_array(img_resized, data_format="channels_last", dtype="uint8")

    print("✅ resize_prediction : image from url resized, and converted to RGB.")
    return img_array


# Emile 12.12.2023
def predictImage(url, model):
    """
    Predict type from an image, with a model.
    """

    # Test
    url = URL_IMG_FIREBUG

    # Get the image
    resized_image_array = resize_prediction(url)

    # Reshape the image
    # img = resized_image_array.reshape((120, 120, 3))
    # with TARGET_SIZE...
    img = resized_image_array.reshape((120, 120, 3))
    img = resnet50.preprocess_input(img)  # !!! à changer en fonction du modèle utilisé

    # Make predictions
    res = model.predict(np.expand_dims(img, axis=0))

    # = POKEMON_TYPE_LIST
    # types = train_ds.class_names
    types = POKEMON_TYPE_LIST

    top3_probs = np.partition(res, -3, axis=1)[:, -3:]
    top3_classes = np.argsort(res, axis=1)[:, -3:]

    predicted_index_1 = top3_classes[0][-1]
    predicted_proba_1 = round(top3_probs[0][-1] * 100, 2)
    predicted_index_2 = top3_classes[0][1]
    predicted_proba_2 = round(top3_probs[0][1] * 100, 2)

    print("✅ predictImage:")
    print(f"First type: {types[predicted_index_1]}: {predicted_proba_1}%")
    print(f"Second type: {types[predicted_index_2]}: {predicted_proba_2}%")


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
