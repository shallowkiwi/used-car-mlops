import logging
import os
import pandas as pd
import pickle
import subprocess
import sys
import json
import warnings
import time
import uuid
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from src.drift_detection import check_drift
from src.performance_monitor import compute_metrics
from src.retrain_pipeline import retrain_if_drift

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="Used Car Price Prediction API")

MODEL_PATH = "models/model.pkl"
SHADOW_MODEL_PATH = "models/shadow_model.pkl"
PERFORMANCE_FILE = "artifacts/model_performance.json"
PREDICTION_LOG = "logs/predictions.log"

model = None
shadow_model = None


# =========================================================
# Load Models
# =========================================================
@app.on_event("startup")
def load_models():
    global model, shadow_model

    if not os.path.exists(MODEL_PATH):
        subprocess.run([sys.executable, "src/train.py"], check=True)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if os.path.exists(SHADOW_MODEL_PATH):
        with open(SHADOW_MODEL_PATH, "rb") as f:
            shadow_model = pickle.load(f)
    else:
        shadow_model = None


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
# Logging helper (APPEND ONLY)
# =========================================================
def log_event(entry):
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =========================================================
# Performance tracking
# =========================================================
def load_performance():
    if not os.path.exists(PERFORMANCE_FILE):
        return {"main_errors": [], "shadow_errors": []}

    with open(PERFORMANCE_FILE, "r") as f:
        return json.load(f)


def save_performance(data):
    os.makedirs("artifacts", exist_ok=True)
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(data, f)


def update_performance(main_error, shadow_error):
    data = load_performance()

    data["main_errors"].append(main_error)
    if shadow_error is not None:
        data["shadow_errors"].append(shadow_error)

    save_performance(data)


def check_promotion():
    data = load_performance()

    if len(data["main_errors"]) < 50:
        return False

    main_mae = np.mean(np.abs(data["main_errors"]))
    shadow_mae = np.mean(np.abs(data["shadow_errors"]))

    return shadow_mae < main_mae


def promote_model():
    global model, shadow_model

    print("🚀 Promoting shadow model to production")

    os.replace(SHADOW_MODEL_PATH, MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    save_performance({"main_errors": [], "shadow_errors": []})


# =========================================================
# Predict
# =========================================================
@app.post("/predict")
def predict(data: CarFeatures):
    global model, shadow_model

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

    # Strict validation
    if not (0.5 * prediction_value <= actual <= 1.5 * prediction_value):
        raise HTTPException(
            status_code=400,
            detail=f"Rejected: unrealistic deviation from prediction ({prediction_value})"
        )

    if not (20000 <= actual <= 2000000):
        raise HTTPException(
            status_code=400,
            detail="Rejected: unrealistic price range"
        )

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

    retrain_if_drift()

    if check_promotion():
        promote_model()

    return {"message": "Feedback recorded"}