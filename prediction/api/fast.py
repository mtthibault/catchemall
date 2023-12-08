import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Emile
from prediction.ml_logic.preprocessor import preprocess_features
from prediction.ml_logic.registry import load_model  # , save_model, save_results
# import json # >> pas utile ici


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
    pickup_datetime: str,  # 2013-07-06 17:18:00
    pickup_longitude: float,  # -73.950655
    pickup_latitude: float,  # 40.783282
    dropoff_longitude: float,  # -73.984365
    dropoff_latitude: float,  # 40.769802
    passenger_count: int,
):  # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # Emile - YOUR CODE HERE

    # Define X_pred
    X_pred = pd.DataFrame(
        dict(
            pickup_datetime=[pd.Timestamp(pickup_datetime, tz="UTC")],
            pickup_longitude=[pickup_longitude],
            pickup_latitude=[pickup_latitude],
            dropoff_longitude=[dropoff_longitude],
            dropoff_latitude=[dropoff_latitude],
            passenger_count=[passenger_count],
        )
    )
    print(">>>>>>>>> X_pred")
    print(X_pred)

    # Compute X_pred_processed
    print(">>>>>>>>> preprocess_features(X_pred)")
    X_pred_processed = preprocess_features(X_pred)
    # print(".........")
    # print(type(X_pred_processed))
    # print(".........")
    print(X_pred_processed)

    # Enable faster predictions
    app = FastAPI()
    app.state.model = load_model()
    y_pred = app.state.model.predict(X_pred_processed)

    # Load model OK without optimization
    # model = load_model()
    # assert model is not None
    # y_pred = model.predict(X_pred_processed)

    print(">>>>>>>>> ✅ Apres predict : y_pred, with shape", y_pred.shape)

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

    # Return y_pred as json OK
    # print(".........")
    # print(type(y_pred))
    # print(".........")
    # print(y_pred, y_pred[0], str(y_pred[0]).strip('[]'), round(y_pred[0][0], 2))
    # res_y_pred = str(round(y_pred[0][0], 2)) # OK mais KO
    res_y_pred = float("{:.2f}".format(y_pred[0][0]))
    res_y_pred = round(float(y_pred), 2)
    res_dict = {"fare_amount": res_y_pred}
    print("res_dict type is ", type(res_dict), res_dict)
    return res_dict

    # return {'fare_amount': float("{:.2f}".format(y_pred[0][0]))}

    # Convert Python to JSON >> Emile : inutile et KO au final ici
    #  >>> {\n    \"fare_amount\": 9.84\n}
    # print(json.dumps(res_dict, indent = 4))
    # return json.dumps(res_dict, indent = 4)


@app.get("/")
def root():
    return {"greeting": "Hello"}
