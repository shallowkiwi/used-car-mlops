import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Use LOCAL MLflow tracking
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Used Car Price Prediction")


DATA_PATH = "data/raw/cardekho_dataset.csv"
MODEL_PATH = "models/model.pkl"


def train_model():

    print("Starting training with real dataset...")

    # -----------------------------
    # Load dataset
    # -----------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # -----------------------------
    # Select only numeric features
    # -----------------------------
    selected_columns = [
        "vehicle_age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "selling_price"
    ]

    df = df[selected_columns]

    # Drop missing values
    df = df.dropna()

    # -----------------------------
    # Split features & target
    # -----------------------------
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]

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
    # Log to MLflow
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

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_model()