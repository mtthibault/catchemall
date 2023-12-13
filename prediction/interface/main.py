import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from prediction.params import *
from prediction.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from prediction.ml_logic.encoders import encode_features
from prediction.ml_logic.model import (
    initialize_model,
    compile_model,
    train_model,
    evaluate_model,
)
from prediction.ml_logic.preprocessor import preprocess_features
from prediction.ml_logic.registry import load_model, save_model, save_results
from prediction.ml_logic.registry import mlflow_run, mlflow_transition_model


def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw data from BigQuery using `get_data_with_cache`
    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        # ORDER BY pickup_datetime
    """

    # Retrieve data using `get_data_with_cache`
    # Emile 11.12.2023 OK
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "prediction", CATCH_PREDICT_CSV_FILE
    )
    data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True,
    )

    # Process data
    data_clean = clean_data(data_query)

    # ???
    print("Processing data ...")
    X = data_clean.drop("catchability", axis=1)
    y = data_clean[["catchability"]]

    X_processed = encode_features(X)
    print("X_processed.head()")
    print(X_processed.head())
    # X_processed, my_fitted_scaler = preprocess_features(X)

    # Emile
    #  suavegarder vers
    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "processed", f"processed_{CATCH_PREDICT_CSV_FILE}"
    )
    X_processed.to_csv(data_processed_cache_path)

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    # data_processed_with_timestamp = pd.DataFrame(
    #     np.concatenate(
    #         (
    #             data_clean[["pickup_datetime"]],
    #             X_processed,
    #             y,
    #         ),
    #         axis=1,
    #     )
    # )

    # load_data_to_bq(
        # data_processed_with_timestamp,
    #     gcp_project=GCP_PROJECT,
    #     bq_dataset=BQ_DATASET,
    #     table=f"processed_{DATA_SIZE}",
    #     truncate=True,
    # )

    print("✅ preprocess() done \n")


@mlflow_run
def train(test_size=0.2, random_state=42) -> float:
    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT * EXCEPT(_0)
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
        ORDER BY _0 ASC
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "processed", f"processed_{CATCH_PREDICT_CSV_FILE}"
    )
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False,
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None
    split_ratio = 0.2
    # Create (X_train_processed, y_train, X_val_processed, y_val)
    train_length = int(len(data_processed) * (1 - split_ratio))

    data_processed_train = (
        data_processed.iloc[:train_length, :].sample(frac=1).to_numpy()
    )
    data_processed_val = data_processed.iloc[train_length:, :].sample(frac=1).to_numpy()

    X_train_processed = data_processed_train[:, :-1]
    y_train = data_processed_train[:, -1]

    X_val_processed = data_processed_val[:, :-1]
    y_val = data_processed_val[:, -1]

    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model()

    # Emile No
    # model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(
        model,
        X_train_processed,
        y_train,
        test_size=test_size,
        random_state=random_state,
    )
    print("✅ train_model() done \n")

    val_mae = np.min(history.history["val_mae"])

    print("✅ ici \n")
    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using catchemall.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    if MODEL_TARGET == "mlflow":
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")

    return val_mae


@mlflow_run
def evaluate(stage: str = "Production") -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    min_date = parse(min_date).strftime("%Y-%m-%d")  # e.g '2009-01-01'
    max_date = parse(max_date).strftime("%Y-%m-%d")  # e.g '2009-01-01'

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT * EXCEPT(_0)
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
        WHERE _0 BETWEEN '{min_date}' AND '{max_date}'
    """

    data_processed_cache_path = Path(
        f"{LOCAL_DATA_PATH}/processed/processed_{CATCH_PREDICT_CSV_FILE}.csv"
    )
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False,
    )

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    data_processed = data_processed.to_numpy()

    X_new = data_processed[:, :-1]
    y_new = data_processed[:, -1]

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate",  # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new),
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    # if X_pred is None:
    #     X_pred = pd.DataFrame(
    #         dict(
    #             pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz="UTC")],
    #             pickup_longitude=[-73.950655],
    #             pickup_latitude=[40.783282],
    #             dropoff_longitude=[-73.984365],
    #             dropoff_latitude=[40.769802],
    #             passenger_count=[1],
    #         )
    #     )

    # model = load_model()
    # assert model is not None

    # X_processed = preprocess_features(X_pred)
    # y_pred = model.predict(X_processed)

    # print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    # return y_pred


if __name__ == "__main__":
    preprocess()
    train()
    evaluate()
    pred()
