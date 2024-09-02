from vlib.config.paths import MODEL_ENDPOINT_URL, PREDICTIONS_CSV
from vlib.infrastructure.local_files import get_inference_data, append_predictions

import requests
from vlib.config.training_config import FEATURES
from vlib.domain.feature_extractor import extract_date_features
from vlib.domain.type_convertor import convert_features_to_float


def infer():
    url = MODEL_ENDPOINT_URL
    next_date_inference, predictions = get_inference_data()

    # YOUR CODE HERE
    # you should infer the new column prediction to the nex_date_inference dataframe
    next_date_inference = extract_date_features(next_date_inference)
    X = next_date_inference[FEATURES]
    X = convert_features_to_float(X)
    # Make a POST request
    response = requests.post(url, json=X.to_dict())
    # Print the response
    next_date_inference["prediction"] = response.json()["predictions"]

    predictions = append_predictions(next_date_inference, predictions)
    predictions.to_csv(PREDICTIONS_CSV, index=False)


if __name__ == '__main__':
    infer()