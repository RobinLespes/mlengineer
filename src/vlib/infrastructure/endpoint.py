import pandas as pd
import mlflow
from flask import Flask, request, jsonify

model_name = "vlib_rf"
model_version = "2"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the data from the request
            data = request.get_json()

            # Convert the data into a pandas DataFrame
            df = pd.DataFrame(data)

            # Generate predictions
            predictions = model.predict(df)

            # Convert predictions to a list
            result = predictions.tolist()

            # Respond with the predictions
            return jsonify({'predictions': result})

        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return "MLflow Model is running. Use /predict endpoint to get predictions."
