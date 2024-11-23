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
