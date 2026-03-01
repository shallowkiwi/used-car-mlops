import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Use LOCAL MLflow tracking
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Used Car Price Prediction")


def train_model():

    print("Starting training...")

    # -----------------------------
    # Create dataset (your synthetic data)
    # -----------------------------
    data = {
        "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
        "km_driven": [50000, 40000, 30000, 20000, 15000, 10000, 8000, 5000],
        "fuel_type": [0, 1, 0, 1, 1, 0, 0, 1],
        "transmission": [0, 1, 0, 1, 0, 1, 0, 1],
        "owner_count": [1, 2, 1, 2, 1, 1, 1, 1],
        "engine_size": [1200, 1500, 1300, 1600, 1400, 1800, 2000, 2200],
        "mileage": [18, 20, 19, 21, 22, 17, 16, 15],
        "seats": [5, 5, 5, 7, 5, 7, 5, 5],
        "price": [500000, 600000, 650000, 750000, 800000, 900000, 1100000, 1300000]
    }

    df = pd.DataFrame(data)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Train model
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    # -----------------------------
    # Log to MLflow (local)
    # -----------------------------
    with mlflow.start_run():

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(model, "model")

    # -----------------------------
    # Save model locally
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_model()