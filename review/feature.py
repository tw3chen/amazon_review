from readability import getmeasures
from review.dataset import TEXT_COLUMN_NAME
import pandas as pd


def add_features_to_df(X):
    rows_to_add = []
    for _, row in X.iterrows():
        text = row[TEXT_COLUMN_NAME]
        readability_features = getmeasures(text)

        row_to_add = {}
        row_to_add.update(readability_features['readability grades'])

        rows_to_add.append(row_to_add)
    df_to_add = pd.DataFrame(rows_to_add)
    X_merged = pd.concat([X, df_to_add], axis=1)
    return X_merged


def add_features_to_dataset(dataset):
    dataset.X_train = add_features_to_df(dataset.X_train)
    dataset.X_validation = add_features_to_df(dataset.X_validation)
    dataset.X_test = add_features_to_df(dataset.X_test)
    return dataset
