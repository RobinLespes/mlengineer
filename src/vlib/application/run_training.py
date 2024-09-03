import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
mlflow.set_tracking_uri("http://127.0.0.1:5000")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

def train_vlib_model():
    training_data = pd.read_csv("data/training_data.csv")

    training_data.datetime = pd.to_datetime(training_data.datetime)
    training_data["hour"] = training_data.datetime.dt.hour
    training_data["year"] = training_data.datetime.dt.year
    training_data["weekday"] = training_data.datetime.dt.weekday
    training_data["month"] = training_data.datetime.dt.month

    FEATURES = ["season","holiday","workingday","weather","weekday","month","year","hour", "temp","humidity","windspeed","atemp"]

    X = training_data[FEATURES]
    Y = training_data['count']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    rf = RandomForestRegressor()
    rf_params = { 'n_estimators':[10, 50, 100],'max_depth':[3,5,7]}

    grid_rf = GridSearchCV(rf, rf_params, cv=2, scoring='neg_mean_absolute_error')


    # Start an MLflow run
    with mlflow.start_run(run_name="rf test"):
        # Fit the model
        grid_rf.fit(X_train, Y_train)

        # Log the best parameters
        mlflow.log_params(grid_rf.best_params_)

        # Log the best score
        best_score = grid_rf.best_score_
        mlflow.log_metric("best_score", -best_score)

        print(f"Best Parameters: {grid_rf.best_params_}")
        print(f"Best Score: {best_score}")

        # Predict on the test set
        test_pred = grid_rf.predict(X_test)
        signature = infer_signature(X_test, grid_rf.predict(X_test))

        # Log the model
        mlflow.sklearn.log_model(grid_rf.best_estimator_, "model", signature=signature)

        # Calculate and log test mae
        test_mae = metrics.mean_absolute_error(Y_test, test_pred)
        mlflow.log_metric("test_mae", test_mae)

        print(f"Test mae: {test_mae}")

if __name__ == "__main__":
    train_vlib_model()