import logging
import os
import pandas as pd
import pickle
import sys
import json
import warnings
import time
import uuid
import numpy as np
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from src.drift_detection import check_drift
from src.performance_monitor import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="Used Car Price Prediction API")

MODEL_PATH = "models/model.pkl"
SHADOW_MODEL_PATH = "models/shadow_model.pkl"
PERFORMANCE_FILE = "artifacts/model_performance.json"
PREDICTION_LOG = "logs/predictions.log"

model = None
shadow_model = None


# =========================================================
# Health check
# =========================================================
@app.get("/healthz")
def health():
    return {"status": "ok"}


# =========================================================
# Load Models
# =========================================================
@app.on_event("startup")
def load_models():
    global model, shadow_model

    print("🚀 Starting API...")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("✅ Loading model...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        print("⚠️ Model not found.")

    if os.path.exists(SHADOW_MODEL_PATH):
        with open(SHADOW_MODEL_PATH, "rb") as f:
            shadow_model = pickle.load(f)


# =========================================================
# GitHub Trigger
# =========================================================
def trigger_retraining():
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")

    if not token or not repo:
        print("❌ GitHub config missing")
        return

    url = f"https://api.github.com/repos/{repo}/dispatches"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    data = {"event_type": "retrain_model"}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 204:
        print("🚀 GitHub retraining triggered")
    else:
        print(f"❌ Trigger failed: {response.text}")


# =========================================================
# Schemas
# =========================================================
class CarFeatures(BaseModel):
    vehicle_age: int = Field(..., ge=0, le=20)
    km_driven: int = Field(..., ge=0, le=300000)
    mileage: float = Field(..., ge=5, le=40)
    engine: float = Field(..., ge=500, le=5000)
    max_power: float = Field(..., ge=20, le=500)
    seats: int = Field(..., ge=2, le=8)


class FeedbackInput(BaseModel):
    prediction_id: str
    actual_price: float

    @validator("actual_price")
    def validate_price(cls, v):
        if not (20000 <= v <= 2000000):
            raise ValueError("Unrealistic price value")
        return v


# =========================================================
# Logging
# =========================================================
def log_event(entry):
    os.makedirs("logs", exist_ok=True)
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =========================================================
# Performance Tracking
# =========================================================
def load_performance():
    if not os.path.exists(PERFORMANCE_FILE):
        return {"main_errors": [], "shadow_errors": []}

    with open(PERFORMANCE_FILE, "r") as f:
        return json.load(f)


def save_performance(data):
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(data, f)


def update_performance(main_error, shadow_error):
    data = load_performance()
    data["main_errors"].append(main_error)

    if shadow_error is not None:
        data["shadow_errors"].append(shadow_error)

    save_performance(data)


# =========================================================
# Predict
# =========================================================
@app.post("/predict")
def predict(data: CarFeatures):
    global model, shadow_model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_df = pd.DataFrame([data.dict()])

    prediction = float(np.expm1(model.predict(input_df)[0]))

    shadow_prediction = None
    if shadow_model is not None:
        shadow_prediction = float(
            np.expm1(shadow_model.predict(input_df)[0])
        )

    prediction_id = str(uuid.uuid4())

    log_event({
        "type": "prediction",
        "prediction_id": prediction_id,
        "timestamp": time.time(),
        "features": data.dict(),
        "prediction": prediction,
        "shadow_prediction": shadow_prediction
    })

    drift_result = check_drift()
    metrics = compute_metrics()

    # 🚀 Trigger GitHub retraining
    if drift_result.get("drift_detected"):
        trigger_retraining()

    return {
        "prediction_id": prediction_id,
        "predicted_price": prediction,
        "shadow_price": shadow_prediction,
        "drift": drift_result,
        "mae": metrics.get("mae"),
        "robust_mae": metrics.get("trimmed_mae"),
        "median_error": metrics.get("median_error")
    }


# =========================================================
# Feedback
# =========================================================
@app.post("/feedback")
def add_feedback(data: FeedbackInput):

    if not os.path.exists(PREDICTION_LOG):
        raise HTTPException(status_code=404, detail="No logs found")

    prediction_value = None
    shadow_prediction = None

    with open(PREDICTION_LOG, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                if (
                    entry.get("type") == "prediction"
                    and entry.get("prediction_id") == data.prediction_id
                ):
                    prediction_value = entry.get("prediction")
                    shadow_prediction = entry.get("shadow_prediction")
                    break
            except:
                continue

    if prediction_value is None:
        raise HTTPException(status_code=404, detail="Prediction ID not found")

    actual = data.actual_price

    if not (0.5 * prediction_value <= actual <= 1.5 * prediction_value):
        raise HTTPException(status_code=400, detail="Unrealistic deviation")

    log_event({
        "type": "feedback",
        "prediction_id": data.prediction_id,
        "actual": actual,
        "timestamp": time.time()
    })

    main_error = abs(prediction_value - actual)
    shadow_error = None

    if shadow_prediction is not None:
        shadow_error = abs(shadow_prediction - actual)

    update_performance(main_error, shadow_error)

    return {"message": "Feedback recorded"}