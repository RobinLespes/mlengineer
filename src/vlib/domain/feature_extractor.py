import pandas as pd


def extract_date_features(training_data):
    training_data.datetime = pd.to_datetime(training_data.datetime)
    training_data["hour"] = training_data.datetime.dt.hour
    training_data["year"] = training_data.datetime.dt.year
    training_data["weekday"] = training_data.datetime.dt.weekday
    training_data["month"] = training_data.datetime.dt.month
    return training_data
