import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def ohe(df):

    df = df.fillna('')

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse=False, drop='first'), ['Type1', 'Type2'])
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    result = pipeline.fit_transform(df[['Type1', 'Type2']])
    feature_names_out = pipeline.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out(['Type1', 'Type2'])
    onehot_df = pd.DataFrame(result, columns=feature_names_out)
    for col in onehot_df.columns:
        if col.startswith('Type1_'):
            onehot_df.rename(columns={col: col.replace('Type1_', '')}, inplace=True)
        elif col.startswith('Type2_'):
            onehot_df.rename(columns={col: col.replace('Type2_', '')}, inplace=True)
    merged_df = onehot_df.groupby(level=0, axis=1).sum()
    result = pd.concat([df, merged_df], axis=1)
    result = result.drop(['Type1', 'Type2'], axis=1)

    return result
