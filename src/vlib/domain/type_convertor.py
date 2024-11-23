import pandas as pd


def convert_features_to_float(X: pd.DataFrame) -> pd.DataFrame:
    """
    convert type of feature columns to float
    :param X: Dataframe with features
    :return: Dataframe with features converted to float
    """
    X_converted = X.copy()
    for col in X_converted.columns:
        X_converted[col] = X_converted[col].astype(float)
    return X_converted
