import os
from datetime import date
from pathlib import Path

import pandas as pd

from vlib.config.paths import PREDICTIONS_CSV, INFERENCE_DATA_CSV


def get_inference_data():
    inference_data = pd.read_csv(INFERENCE_DATA_CSV)
    inference_data["date"] = pd.to_datetime(inference_data.datetime).dt.date
    file_path = Path(PREDICTIONS_CSV)
    default_date = date(2012, 6, 19)
    if os.path.exists(file_path):
        predictions = pd.read_csv(file_path)
        predictions.date = pd.to_datetime(predictions.datetime).dt.date
        max_date = predictions.date.max()
    else:
        predictions = None
        max_date = default_date
    inference_data = inference_data[inference_data.date > max_date]
    next_date_inference = inference_data[inference_data.date == inference_data.date.min()]
    return next_date_inference, predictions


def append_predictions(next_date_inference, predictions):
    if predictions is not None:
        predictions = pd.concat([predictions, next_date_inference], ignore_index=True)
    else:
        predictions = next_date_inference
    return predictions
