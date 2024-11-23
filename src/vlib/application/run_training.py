from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from vlib.config.paths import TRACKING_URI, TRAINING_DATA_CSV
from vlib.config.training_config import FEATURES, TARGET, RF_PARAMS
from vlib.domain.feature_extractor import extract_date_features




mlflow.set_tracking_uri(TRACKING_URI)


def train_vlib_model():
    training_data = pd.read_csv(Path(TRAINING_DATA_CSV))
    training_data = extract_date_features(training_data)
    X = training_data[FEATURES]
    Y = training_data[TARGET]
    rf = RandomForestRegressor()
    grid_rf = GridSearchCV(rf, RF_PARAMS, cv=2, scoring='neg_mean_absolute_error')
    for col in X.columns:
        X[col] = X[col].astype(float)

    # Start an MLflow run
    with mlflow.start_run(run_name="rf test"):
        # Fit the model
        grid_rf.fit(X, Y)
        # Log the best parameters
        mlflow.log_params(grid_rf.best_params_)

        # Log the best score
        best_score = -grid_rf.best_score_
        mlflow.log_metric("best_score", best_score)

        print(f"Best Parameters: {grid_rf.best_params_}")
        print(f"MAE CV: {best_score}")

        signature = infer_signature(X, grid_rf.predict(X))

        # Log the model
        mlflow.sklearn.log_model(grid_rf.best_estimator_, "model", signature=signature)


if __name__ == "__main__":
    train_vlib_model()
