import numpy as np
import time
import pandas as pd

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
# print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
# start = time.perf_counter()

# from tensorflow import keras
# from keras import Model, Sequential, layers, regularizers, optimizers
# from keras.callbacks import EarlyStopping

# end = time.perf_counter()
# print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

# Emile 11.12.2023
print(Fore.BLUE + "\nLoading SciKit Learn..." + Style.RESET_ALL)
start = time.perf_counter()

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

end = time.perf_counter()
print(f"\n✅ SciKit Learn loaded ({round(end - start, 2)}s)")


def initialize_model():  # -> Model:
    """
    Initialize the Random Forest Regressor
    """
    print("Start initialise model")

    # Define base models for stacking
    estimators = [('rf', RandomForestRegressor(n_estimators=250,
                                               max_depth=25,
                                               min_samples_split=5,
                                               random_state=42)),
                  ('gb', GradientBoostingRegressor(random_state=42)),
                  ('svr', SVR()),  # Support Vector Regressor
                  ('knn', KNeighborsRegressor())  # K-Neighbors Regressor
]

    # Initialize Stacking Regressor with a meta-regressor
    model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
    model
    print("✅ Model initialized")

    return model


# def compile_model(model: Model, learning_rate=0.0005) -> Model:
def compile_model(model, learning_rate=0.0005):
    """
    Compile the model
    """
    # optimizer = optimizers.Adam(learning_rate=learning_rate)
    # model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")

    return model


# def train_model(
#     model: Model, X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42
# ) -> Tuple[Model, dict]:
def train_model(
    model, X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42
):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    history = model.fit(X_train, y_train)

    print(
        f"✅ Model trained on {len(X_train)} rows"
        # f"✅ Model trained on {len(X_train)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}"
    )

    return model, history


# def evaluate_model(model: Model, X: np.ndarray, y: np.ndarray) -> Tuple[Model, dict]:
def evaluate_model(model, X: np.ndarray, y: np.ndarray):
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    # Make predictions and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # # Scoring the model
    # model_score = model.score(X_test, y_test)
    # print("Model Score on Test Data:", model_score)

    # # For K-Fold Cross Validation
    # metrics = cross_val_score(model, X, y, cv=5)

    print(f"✅ Model evaluated - Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared Score:", r2)

    metrics = predictions
    return metrics
