import pandas as pd
import mlflow
from flask import Flask, request, jsonify

logged_model = 'runs:/da719023e235419c93c134c6fcf83301/model'

model = mlflow.pyfunc.load_model(logged_model)
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            df = pd.DataFrame(data)
            predictions = model.predict(df)
            result = predictions.tolist()
            return jsonify({'predictions': result})

        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return "MLflow Model is running. Use /predict endpoint to get predictions."
