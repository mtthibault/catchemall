import pandas as pd
import numpy as np
import pickle
import os


from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from joblib import load
from prediction.params import LOCAL_DATA_PATH

MODEL_DATA_PATH = os.path.join(
    LOCAL_DATA_PATH, "prediction", "regression"
)


def preprocess_pokemon_data(df):

    # Drop unnecessary columns
    df = df.drop(columns=['japanese_name', 'name'])

    # Fill missing values
    df['height_m'] = df['height_m'].fillna(df['height_m'].median())
    df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].median())
    df['type1'].fillna('None', inplace=True)
    df['type2'].fillna('None', inplace=True)
    df['abilities'].fillna('None', inplace=True)
    df['classfication'].fillna('None', inplace=True)

    return df


def encode_features(df):
    """
    Function to encode various features in the Pokemon data.
    """
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
    with open(os.path.join(MODEL_DATA_PATH, 'ohe_classification.pkl'), 'wb') as f:
        pickle.dump(ohe_classification, f)
    with open(os.path.join(MODEL_DATA_PATH, 'mlb_abilities.pkl'), 'wb') as f:
        pickle.dump(mlb_abilities, f)
    with open(os.path.join(MODEL_DATA_PATH, 'ohe_type1.pkl'), 'wb') as f:
        pickle.dump(ohe_type1, f)
    with open(os.path.join(MODEL_DATA_PATH, 'ohe_type2.pkl'), 'wb') as f:
        pickle.dump(ohe_type2, f)

    # Create DataFrames for encoded features
    classification_encoded_df = pd.DataFrame(classification_encoded.toarray(), columns=ohe_classification.get_feature_names_out(), index=df.index)
    abilities_encoded_df = pd.DataFrame(abilities_encoded, columns=mlb_abilities.classes_, index=df.index)
    type1_encoded_df = pd.DataFrame(type1_encoded.toarray(), columns=ohe_type1.get_feature_names_out(['type1']), index=df.index)
    type2_encoded_df = pd.DataFrame(type2_encoded.toarray(), columns=ohe_type2.get_feature_names_out(['type2']), index=df.index)

    # Combine encoded features with the original dataframe
    df = pd.concat([df, classification_encoded_df, abilities_encoded_df, type1_encoded_df, type2_encoded_df], axis=1)

    # Drop the original columns that were encoded
    df = df.drop(columns=['classfication', 'abilities', 'type1', 'type2', 'percentage_male'])

    return df


def predict_catchability(
    base_total,
    attack,
    sp_attack,
    sp_defense,
    defense,
    hp,
    height_m,
    speed,
    weight_kg,
    is_legendary,
    base_egg_steps
):

    df_path = os.path.join(MODEL_DATA_PATH, 'pokemon.csv')
    df = pd.read_csv(df_path)
    model_path = os.path.join(MODEL_DATA_PATH, 'saved_model.joblib')
    model = load(model_path)  # Your trained model

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'base_total': [base_total],
        'attack': [attack],
        'sp_attack': [sp_attack],
        'sp_defense': [sp_defense],
        'defense': [defense],
        'hp': [hp],
        'height_m': [height_m],
        'speed': [speed],
        'weight_kg': [weight_kg],
        'is_legendary': [is_legendary],
        'base_egg_steps': [base_egg_steps]
    })

    # Generate random values for other attributes in a separate DataFrame
    random_data = pd.DataFrame({column: np.random.choice(df[column].dropna().values, size=1)
                                for column in df.columns if column not in input_data.columns})
    # Combine the input and random data
    input_data = pd.concat([input_data, random_data], axis=1)

    # Ensure the input data has the same column order as the training data
    input_data = input_data.reindex(columns=[col for col in df.columns if col != 'capture_rate'])

    # Preprocess the data
    preprocessed_fake_pokemon_df = preprocess_pokemon_data(input_data)

    # Encode the new data using the same encoders
    encoded_fake_pokemon_df = encode_features(preprocessed_fake_pokemon_df)

    # Create X_train columns so it matches shape
    columns_path = os.path.join(MODEL_DATA_PATH, 'X_train_columns.pickle')
    with open(columns_path, 'rb') as file:
        X_train_columns = pickle.load(file)

    # Ensure the new data has the same columns as the training data, in the same order
    missing_cols = set(X_train_columns) - set(encoded_fake_pokemon_df.columns)
    missing_df = pd.DataFrame(0, index=encoded_fake_pokemon_df.index, columns=list(missing_cols))
    encoded_fake_pokemon_df = pd.concat([encoded_fake_pokemon_df, missing_df], axis=1)
    encoded_fake_pokemon_df = encoded_fake_pokemon_df[X_train_columns]

    # Ensure there are no NaN values in the data
    encoded_fake_pokemon_df.fillna(0, inplace=True)
    # Apply the prediction model
    prediction = model.predict(encoded_fake_pokemon_df)

    return prediction[0]


if __name__=='__main__':

    # Example input values
    base_total = 10  # Example value
    attack = 10       # Example value
    sp_attack = 10    # Example value
    sp_defense = 10    # Example value
    defense = 10    # Example value
    hp = 10    # Example value
    height_m = 0.1    # Example value
    speed = 10    # Example value
    weight_kg = 1.0    # Example value
    is_legendary = 0    # Example value
    base_egg_steps = 20    # Example value

    # Predict
    catchability = predict_catchability(base_total,
                                        attack,
                                        sp_attack,
                                        sp_defense,
                                        defense,
                                        hp,
                                        height_m,
                                        speed,
                                        weight_kg,
                                        is_legendary,
                                        base_egg_steps,
                                        )
    print(f"Predicted Catchability: {catchability}")
# /Users/lapiscine/code/mtthibault/catchemall/prediction/params.py
