import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import pickle

# import pygeohash as gh

from prediction.utils import simple_time_and_memory_tracker


def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    pass


def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    pass


def compute_geohash(X: pd.DataFrame, precision: int = 5) -> np.ndarray:
    """
    Add a geohash (ex: "dr5rx") of len "precision" = 5 by default
    corresponding to each (lon, lat) tuple, for pick-up, and drop-off
    """
    pass


def encode_features(df):
    """
    Function to encode various features in the Pokemon data.
    """
    print("Initializing encoders...")
    # # Caspar moved this into data.py 20231213 for troubleshooting
    # Initialize encoders
    ohe_classification = OneHotEncoder()
    mlb_abilities = MultiLabelBinarizer()
    ohe_type1 = OneHotEncoder()
    ohe_type2 = OneHotEncoder()

    # Perform encoding
    classification_encoded = ohe_classification.fit_transform(df[['classfication']])
    abilities_encoded = mlb_abilities.fit_transform(df['abilities'].apply(lambda x: x.strip("[]").replace("'", "").split(", ")))
    type1_encoded = ohe_type1.fit_transform(df[['type1']])
    type2_encoded = ohe_type2.fit_transform(df[['type2']])

    # Save the fitted encoders using pickle
    with open('ohe_classification.pkl', 'wb') as f:
        pickle.dump(ohe_classification, f)
    with open('mlb_abilities.pkl', 'wb') as f:
        pickle.dump(mlb_abilities, f)
    with open('ohe_type1.pkl', 'wb') as f:
        pickle.dump(ohe_type1, f)
    with open('ohe_type2.pkl', 'wb') as f:
        pickle.dump(ohe_type2, f)

    # Create DataFrames for encoded features
    classification_encoded_df = pd.DataFrame(classification_encoded.toarray(), columns=ohe_classification.get_feature_names_out())
    abilities_encoded_df = pd.DataFrame(abilities_encoded, columns=mlb_abilities.classes_)
    type1_encoded_df = pd.DataFrame(type1_encoded.toarray(), columns=ohe_type1.get_feature_names_out(['type1']))
    type2_encoded_df = pd.DataFrame(type2_encoded.toarray(), columns=ohe_type2.get_feature_names_out(['type2']))

    # Combine encoded features with the original dataframe
    df = pd.concat([df, classification_encoded_df, abilities_encoded_df, type1_encoded_df, type2_encoded_df], axis=1)

    # Drop the original columns that were encoded
    df = df.drop(columns=['classfication',  'type1', 'type2', 'abilities', 'percentage_male'])
    print("End of encoding block")
    return df
