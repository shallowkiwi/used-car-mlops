import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
import os
import json
import warnings

from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# =========================================================
# Config
# =========================================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Used Car Price Prediction")

DATA_PATH = "data/raw/cardekho_dataset.csv"
MODEL_PATH = "models/model.pkl"
SHADOW_MODEL_PATH = "models/shadow_model.pkl"

DRIFT_ARTIFACT_PATH = "artifacts/drift_reference.json"
LOG_FILE = "logs/predictions.log"

RANDOM_STATE = 42
FEEDBACK_RATIO = 0.2

# 🔥 NEW (adaptive)
ADAPTIVE_MODE_THRESHOLD = 20
ADAPTIVE_WEIGHT = 0.5


# =========================================================
# Load SAFE feedback data
# =========================================================
def load_feedback_data():

    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    records = []

    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                if data.get("actual") is None:
                    continue

                pred = data.get("prediction")
                actual = data.get("actual")

                if pred is None or actual is None:
                    continue

                # only accept ±30%
                if not (0.5 * pred <= actual <= 1.5 * pred):
                    continue

                features = data.get("features", {})
                record = {**features, "selling_price": actual}
                records.append(record)

            except:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df = df.dropna()

    df = df[
        (df["vehicle_age"] >= 0) & (df["vehicle_age"] <= 20) &
        (df["km_driven"] >= 0) & (df["km_driven"] <= 300000) &
        (df["mileage"] >= 5) & (df["mileage"] <= 40) &
        (df["engine"] >= 500) & (df["engine"] <= 5000) &
        (df["max_power"] >= 20) & (df["max_power"] <= 500) &
        (df["seats"] >= 2) & (df["seats"] <= 8) &
        (df["selling_price"] >= 20000) & (df["selling_price"] <= 10000000)
    ]

    print(f"Valid feedback samples used: {len(df)}")
    return df


# =========================================================
# TRAIN CORE
# =========================================================
def train_pipeline(df):

    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]

    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)

    predictions = np.expm1(pred_log)
    actuals = np.expm1(y_test_log)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)

    return model, rmse, r2, X_train


# =========================================================
# MAIN TRAIN FUNCTION (with shadow logic)
# =========================================================
def train_model():

    print("Starting adaptive + shadow training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    selected_columns = [
        "vehicle_age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "selling_price"
    ]

    df = df[selected_columns].dropna()

    # ---------------------------
    # Load feedback
    # ---------------------------
    df_feedback = load_feedback_data()

    if not df_feedback.empty:

        df_feedback = df_feedback.sample(frac=1, random_state=RANDOM_STATE)

        if len(df_feedback) >= ADAPTIVE_MODE_THRESHOLD:
            print("🚀 Adaptive mode ON")

            adaptive_count = int(ADAPTIVE_WEIGHT * len(df))
            adaptive_count = min(adaptive_count, len(df_feedback))

            df_feedback = df_feedback.iloc[:adaptive_count]

        else:
            print("Safe feedback mode")

            max_feedback = min(len(df_feedback), int(FEEDBACK_RATIO * len(df)))
            df_feedback = df_feedback.iloc[:max_feedback]

        df = pd.concat([df, df_feedback], ignore_index=True)

    # =====================================================
    # Train NEW model (candidate)
    # =====================================================
    new_model, new_rmse, new_r2, X_train = train_pipeline(df)

    print(f"New Model RMSE: {new_rmse}")
    print(f"New Model R2: {new_r2}")

    # =====================================================
    # Load EXISTING model (if exists)
    # =====================================================
    old_rmse = None

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            old_model = pickle.load(f)

        # evaluate old model on same data
        X = df.drop("selling_price", axis=1)
        y = df["selling_price"]

        y_log = np.log1p(y)

        preds = old_model.predict(X)
        preds = np.expm1(preds)
        actuals = np.expm1(y_log)

        old_rmse = np.sqrt(mean_squared_error(actuals, preds))
        print(f"Old Model RMSE: {old_rmse}")

    # =====================================================
    # Model Comparison (CRITICAL)
    # =====================================================
    deploy_new = False

    if old_rmse is None:
        deploy_new = True
    elif new_rmse < old_rmse:
        deploy_new = True

    # =====================================================
    # Save models
    # =====================================================
    os.makedirs("models", exist_ok=True)

    if deploy_new:
        print("✅ New model is better → deploying")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)

    else:
        print("⚠️ New model worse → saving as shadow model")

        with open(SHADOW_MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)

    # =====================================================
    # Drift reference
    # =====================================================
    os.makedirs("artifacts", exist_ok=True)

    drift_reference = {
        f"{col}_mean": float(X_train[col].mean())
        for col in X_train.columns
    }

    with open(DRIFT_ARTIFACT_PATH, "w") as f:
        json.dump(drift_reference, f)

    # =====================================================
    # MLflow logging
    # =====================================================
    with mlflow.start_run():

        mlflow.log_metric("rmse", new_rmse)
        mlflow.log_metric("r2_score", new_r2)

        mlflow.log_param("adaptive_mode", len(df_feedback) >= ADAPTIVE_MODE_THRESHOLD)

        signature = infer_signature(X_train, new_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=new_model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

    print("Training pipeline completed.")


if __name__ == "__main__":
    train_model()