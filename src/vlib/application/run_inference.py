from pathlib import Path
import requests

from vlib.config.const import PREDICTION
from vlib.config.paths import MODEL_ENDPOINT_URL, PREDICTIONS_CSV
from vlib.infrastructure.local_files import get_inference_data, append_predictions
from vlib.config.training_config import FEATURES
from vlib.domain.feature_extractor import extract_date_features
from vlib.domain.type_convertor import convert_features_to_float


def infer():
    """
    Do prediction on next date from inference dataset using model served on flask endpoint
    """
    url = MODEL_ENDPOINT_URL
    next_date_inference, predictions = get_inference_data()
    next_date_inference = extract_date_features(next_date_inference)
    X = next_date_inference[FEATURES]
    X = convert_features_to_float(X)
    response = requests.post(url, json=X.to_dict())
    next_date_inference[PREDICTION] = response.json()[PREDICTION]
    predictions = append_predictions(next_date_inference, predictions)
    predictions.to_csv(Path(PREDICTIONS_CSV), index=False)


if __name__ == '__main__':
    infer()