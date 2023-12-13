import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from prediction.params import *

# Emile 11.12.2023
import ast


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # Emile 11.12.2023

    print("Initial Shape:", df.shape)

    # Fill missing values
    if "height_m" in df.columns:
        df["height_m"].fillna(df["height_m"].median(), inplace=True)
    if "weight_kg" in df.columns:
        df["weight_kg"].fillna(df["weight_kg"].median(), inplace=True)

    # Drop columns
    columns_to_drop = ["percentage_male"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print("After dropping columns:", df.shape)
    print("After dropping columns:", df.head())

    # Convert 'capture_rate' to numeric and create 'catchability'
    df["capture_rate"] = pd.to_numeric(df["capture_rate"], errors="coerce")
    df["catchability"] = df["capture_rate"] / 2.55
    df = df.dropna(subset=["catchability"])
    print("After capture rate conversion to catchability:", df.shape)

    df = df.drop(columns=["capture_rate"])
    # Check shape after dropping columns
    print("Shape after dropping capture rate:", df.shape)
    print(" Head after dropping capture rate:", df.head())

    # Handle 'type2' and create 'combined_type'
    df["type2"].fillna("None", inplace=True)
    df["combined_type"] = df["type1"] + "_" + df["type2"]

    # Check shape after creating combined type
    print("Shape after combined type:", df.shape)
    print("Head after combined type:", df.head)

    # Handle 'abilities'
    if "abilities" in df.columns:
        # Convert string representation of list to actual list
        df["abilities"] = df["abilities"].apply(ast.literal_eval)

        # Collect all unique abilities from the dfset
        all_abilities = set().union(*df["abilities"])

        # Prepare df for new DataFrame
        abilities_dicts = []
        for index, row in df.iterrows():
            abilities_dict = {
                ability: int(ability in row["abilities"]) for ability in all_abilities
            }
            abilities_dicts.append(abilities_dict)

        # Create a DataFrame from list of dictionaries
        abilities_df = pd.DataFrame(abilities_dicts, index=df.index)

        # Concatenate the abilities df
        df = pd.concat([df, abilities_df], axis=1)
        df.drop(columns=["abilities"], inplace=True)
    else:
        print("'abilities' column is missing")
    print("Shape after abilities:", df.shape)
    print("Head after abilities:", df.head)

    # One-hot encoding for 'combined_type' using pd.get_dummies
    df = pd.get_dummies(df, columns=["combined_type"])
    print("Shape after get dummies on combined type:", df.shape)
    print("Head after get dummies on combined type:", df.head)

    # One-hot encoding 'classfication' misspelt field and dropping original columns
    df = pd.get_dummies(df, columns=["classfication"])
    print("Shape after get dummies on classfication:", df.shape)
    print("Head after get dummies on classfication:", df.head)

    df.drop(
        columns=["japanese_name", "name", "base_total", "type1", "type2"], inplace=True
    )
    print("Shape after drop final columns:", df.shape)
    print("Head after drop final columns:", df.head)

    print("✅ Datas are cleaned")

    return df


def get_data_with_cache(
    gcp_project: str, query: str, cache_path: Path, data_has_header=True
) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header="infer" if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df


def load_data_to_bq(
    data: pd.DataFrame, gcp_project: str, bq_dataset: str, table: str, truncate: bool
) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(
        Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL
    )

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    # TODO: simplify this solution if possible, but students may very well choose another way to do it
    # We don't test directly against their own BQ tables, but only the result of their query
    data.columns = [
        f"_{column}"
        if not str(column)[0].isalpha() and not str(column)[0] == "_"
        else str(column)
        for column in data.columns
    ]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(
        f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)"
    )

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
