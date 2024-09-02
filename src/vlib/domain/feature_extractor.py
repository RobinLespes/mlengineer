import pandas as pd


def extract_date_features(df):
    df_with_date_features = df.copy()
    df_with_date_features.datetime = pd.to_datetime(df_with_date_features.datetime)
    df_with_date_features["hour"] = df_with_date_features.datetime.dt.hour
    df_with_date_features["year"] = df_with_date_features.datetime.dt.year
    df_with_date_features["weekday"] = df_with_date_features.datetime.dt.weekday
    df_with_date_features["month"] = df_with_date_features.datetime.dt.month
    df_with_date_features['date'] = df_with_date_features.datetime.dt.date
    df_with_date_features['day'] = df_with_date_features.datetime.dt.day
    return df_with_date_features
