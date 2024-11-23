import pandas as pd
import mlflow
from flask import Flask, request, jsonify

from vlib.config import const

model_name = "vlib_rf"
model_version = "1"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles HTTP POST requests to make predictions using the trained model.
    The model should have been registered in MLFLOW registery and named vlib_rf, version 1.
    This function expects a JSON payload containing input data for prediction.
    The data is converted into a Pandas DataFrame and passed to the preloaded model
    for inference. The predictions are returned as a JSON response.

    Returns:
        Response: A JSON object with either:
            - "prediction": A list of predictions if the request is successful.
            - "error": An error message with HTTP status code 500 in case of an exception.

    Raises:
        Exception: If there is an issue with the input data or during prediction,
                   an error message will be returned in the response.

    HTTP Methods:
        - POST: Required for making predictions.

    Example Usage:
        curl -X POST -H "Content-Type: application/json" \
            -d '[{"feature1": value1, "feature2": value2, ...}]' \
            http://<host>:<port>/predict
    """
    if request.method == 'POST':
        try:
            data = request.get_json()
            df = pd.DataFrame(data)
            predictions = model.predict(df)
            result = predictions.tolist()
            return jsonify({const.PREDICTION: result})

        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return "MLflow Model is running. Use /predict endpoint to get predictions."
