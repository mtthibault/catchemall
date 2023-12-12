import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# from prediction.ml_logic.encoders import transform_time_features, transform_lonlat_features, compute_geohash

# Emile 11.12.2023
from sklearn.preprocessing import StandardScaler


def preprocess_features(X: pd.DataFrame, fitted_scaler=None):
    # numerical_cols = X.select_dtypes(include=[np.number])

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    # # Scaling numerical features - example with 'attack' and 'defense'
    # scaler = StandardScaler()
    # X[["attack", "defense"]] = scaler.fit_transform(X[["attack", "defense"]])

    # if fitted_scaler:
    #     # Use the pre-fitted scaler
    #     X[numerical_cols.columns] = fitted_scaler.transform(numerical_cols)
    # else:
    #     # Fit a new scaler if no pre-fitted scaler is provided
    #     fitted_scaler = StandardScaler()
    # X[numerical_cols.columns] = fitted_scaler.fit_transform(numerical_cols)
    # X_processed = X

    preprocessor = StandardScaler()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    # return X_processed, fitted_scaler
    return X_processed
