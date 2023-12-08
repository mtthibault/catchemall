from dogs_prediction.DL_logic import predict, registry
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

#iniate api
app = FastAPI()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allows all methods
    allow_headers=['*'],  # Allows all headers
)

# preload the model
app.state.model = registry.load_selected_model()



# predicts from url provided by user
@app.get('/predict_url')
def prediction(url_with_pic, model_type='inception_v3'):
    model = app.state.model
    assert model is not None
    prediction = predict.predict_labels(model, model_type, url_with_pic=url_with_pic)
    return prediction


# predicts from file provided by user
@app.post('/predict_file')
def prediction(file: UploadFile, model_type='inception_v3'):
    model = app.state.model
    assert model is not None
    prediction = predict.predict_labels(model, model_type, img=file.file)
    return prediction


#root endpoint
@app.get('/')
def root():
    return dict(greeting='Bark!')
