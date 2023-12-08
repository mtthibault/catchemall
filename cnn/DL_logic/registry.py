# registry.py is for loading and storing the model's weights from/to Cloud Storage
import os
from colorama import Fore, Style
from tensorflow.keras.models import load_model as keras_load_model
from dogs_prediction.params import *

def load_latest_model(loading_method = MODEL_TARGET):

    '''
    Function to load the latest (uploaded) model from local disk or Google Cloud Storage.
    Depending on the value set to MODEL_TARGET in params.py.
    For local loading, it requires to have a folder called "models" in the same directory.
    '''
    if loading_method == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
        local_model_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        local_model_files = os.listdir(local_model_directory)
        local_model_paths = [os.path.join(local_model_directory, f) for f in local_model_files if f.endswith('.h5')]
        most_recent_model_path = max(local_model_paths, key=os.path.getctime)
        model_path = os.path.join(local_model_directory, most_recent_model_path)
        latest_model = keras_load_model(model_path, compile = False)
        print("✅ Model loaded from local")
        return latest_model

    # get the latest model from GCP
    elif loading_method == "gcp":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)
        from google.cloud import storage
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = keras_load_model(latest_model_path_to_save, compile = False)
            print("✅ Latest model downloaded from cloud storage")
            return latest_model
        except:
            print(f"\n❌ No model found on GCS bucket {BUCKET_NAME}")
            return None

def load_selected_model(model_name='inception_v3', loading_method=MODEL_TARGET):
    '''
    Function to load a specific model from local disk or Google Cloud Storage.
    By default, it loads INCEPTION_V3 which had the best performance.
    Depending on the value set to MODEL_TARGET in params.py.
    For local loading, it requires to have a folder called "models" in the same directory.
    Args:
        model_name: name of the model to load. Default is 'inception_v3'. Can choose between 'inception_v3', 'Resnet_50_epoch'.
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, '../..', 'models'))

    if loading_method == "local":
        print(Fore.BLUE + f"\nLoad model '{model_name}' from local registry..." + Style.RESET_ALL)
        model_path = os.path.join(models_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            print(f"\n❌ No model with name '{model_name}' found in local models directory")
            return None
        else:
            model = keras_load_model(model_path, compile=False)
            print("✅ Model loaded from local")
            return model

    elif loading_method == "gcp":
        print(Fore.BLUE + f"\nLoad model '{model_name}' from GCS..." + Style.RESET_ALL)
        from google.cloud import storage
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        print(blobs)
        selected_blob = None
        for blob in blobs:
            if blob.name.endswith(f"{model_name}.h5"):
                selected_blob = blob
                break
        if selected_blob is None:
            print(f"\n❌ No model with name '{model_name}' found on GCS bucket {BUCKET_NAME}")
            return None
        else:
            selected_model_path_to_save = os.path.join(models_dir,'..' ,selected_blob.name)
            selected_blob.download_to_filename(selected_model_path_to_save)
            selected_model = keras_load_model(selected_model_path_to_save, compile=False)
            print("✅ Selected model downloaded from cloud storage")
            return selected_model
