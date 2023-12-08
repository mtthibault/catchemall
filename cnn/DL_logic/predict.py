import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from dogs_prediction.params import *

breed = BREED

def compile_model(model):
    """
    Compile the model.
    Args:
        model: the model to compile
    Returns:
        model: the compiled model
    """
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model compiled")
    return model

def getImage(img=None, url_with_pic:str='', show=False):
    '''
    Get an image provided its url and resize it.
    The size of the image is 224x224.
    '''
    print(f"✅ get image received: img={img is not None}, url_with_pic={url_with_pic is not None}")
    if url_with_pic:
        response = requests.get(url_with_pic)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(img).convert("RGB")
    if show:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    img = img.resize((224, 224))
    print("✅ image resized")
    return img

def predict_labels(model, model_type, *args, **kwargs):
    '''
    Function that will load the latest model from local disk and use it to predict the breed of the dog in the image.
    Args:
        url: url of the image to predict
    Returns:
        breed_prediction: dictionary with the top 3 breeds predicted
        score_prediction: dictionary with the top 3 scores predicted
    '''
    img = getImage(*args, **kwargs)
    print("✅ Image successfully loaded")
    img = img_to_array(img)    #shape = (224, 224, 3)
    img = img.reshape((-1, 224, 224, 3))
    print("✅ Image successfully reshaped", img.shape)

    model = compile_model(model)
    print("✅ Model successfully loaded and compiled")

    if model_type == "resnet50":
        img = resnet_preprocess_input(img)
        print("✅ Image successfully preprocessed (resnet50)")
    elif model_type == "inception_v3":
        img = inception_preprocess_input(img)
        print("✅ Image successfully preprocessed (inception_v3)")
    print("✅ Predicting breed...")

    res = model.predict(img)
    print("✅ Breed predicted")
    indexes = np.argsort(res)[0][-3:][::-1]
    predicts = np.sort(res)[0][::-1][0:3]

    breed_prediction = {
        'first': breed[indexes[0]],
        'second': breed[indexes[1]],
        'third': breed[indexes[2]],
    }
    score_prediction = {
        'first': float(round(predicts[0],2)),
        'second': float(round(predicts[1],2)),
        'third': float(round(predicts[2],2))
    }
    output = {'prediction': breed_prediction,
              'score': score_prediction}
    return  output
