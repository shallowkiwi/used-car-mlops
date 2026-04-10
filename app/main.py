import logging
import os
import pandas as pd
import pickle
import sys
import warnings
import time
import uuid
import numpy as np
import requests
import sqlite3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drift_detection import check_drift
from src.performance_monitor import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="Used Car Price Prediction API")

MODEL_PATH = "models/model.pkl"
SHADOW_MODEL_PATH = "models/shadow_model.pkl"
DB_FILE = "data/monitoring.db"

model = None
shadow_model = None
last_loaded_time = 0


# ============================
# MODEL AUTO-RELOAD (🔥 KEY FIX)
# ============================
def reload_model_if_updated():
    global model, last_loaded_time

    if not os.path.exists(MODEL_PATH):
        return

    modified_time = os.path.getmtime(MODEL_PATH)

    if modified_time > last_loaded_time:
        print("🔄 Reloading updated model...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        last_loaded_time = modified_time


# ============================
# DB SETUP
# ============================
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT,
            vehicle_age REAL,
            km_driven REAL,
            mileage REAL,
            engine REAL,
            max_power REAL,
            seats REAL,
            prediction REAL,
            shadow_prediction REAL,
            actual REAL
        )
    """)

    conn.commit()
    conn.close()


def insert_prediction(data, prediction_id, prediction, shadow_prediction):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            prediction_id, vehicle_age, km_driven, mileage,
            engine, max_power, seats, prediction, shadow_prediction, actual
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
    """, (
        prediction_id,
        data["vehicle_age"],
        data["km_driven"],
        data["mileage"],
        data["engine"],
        data["max_power"],
        data["seats"],
        prediction,
        shadow_prediction
    ))

    conn.commit()
    conn.close()


def update_feedback(prediction_id, actual):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE predictions
        SET actual = ?
        WHERE prediction_id = ?
    """, (actual, prediction_id))

    conn.commit()
    conn.close()


# ============================
# STARTUP
# ============================
@app.on_event("startup")
def load_models():
    global model, shadow_model, last_loaded_time

    print("🚀 Starting API...")

    init_db()

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        last_loaded_time = os.path.getmtime(MODEL_PATH)

    if os.path.exists(SHADOW_MODEL_PATH):
        with open(SHADOW_MODEL_PATH, "rb") as f:
            shadow_model = pickle.load(f)


@app.get("/healthz")
def health():
    return {"status": "ok"}


# ============================
# RETRAIN TRIGGER
# ============================
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


# ============================
# INPUT SCHEMA
# ============================
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


# ============================
# PREDICT
# ============================
@app.post("/predict")
def predict(data: CarFeatures):
    global model, shadow_model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 🔄 Reload model if retrained
    reload_model_if_updated()

    input_df = pd.DataFrame([data.dict()])

    prediction = float(np.expm1(model.predict(input_df)[0]))

    shadow_prediction = None
    if shadow_model is not None:
        base_shadow = float(np.expm1(shadow_model.predict(input_df)[0]))
        shadow_prediction = base_shadow * np.random.uniform(0.9, 1.1)

    prediction_id = str(uuid.uuid4())

    insert_prediction(data.dict(), prediction_id, prediction, shadow_prediction)

    drift_result = check_drift()
    metrics = compute_metrics()

    if drift_result.get("drift_detected"):
        print("🚀 Triggering retraining...")
        trigger_retraining()

    return {
        "prediction_id": prediction_id,
        "predicted_price": prediction,
        "shadow_price": shadow_prediction,
        "drift": drift_result,
        "mae": metrics.get("mae"),
        "robust_mae": metrics.get("robust_mae"),
        "median_error": metrics.get("median_error")
    }


# ============================
# FEEDBACK
# ============================
@app.post("/feedback")
def add_feedback(data: FeedbackInput):

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM predictions WHERE prediction_id = ?
    """, (data.prediction_id,))
    
    exists = cursor.fetchone()[0]

    if exists == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Prediction ID not found")

    cursor.execute("""
        UPDATE predictions
        SET actual = ?
        WHERE prediction_id = ?
    """, (data.actual_price, data.prediction_id))

    conn.commit()
    conn.close()

    metrics = compute_metrics()

    return {
        "message": "Feedback recorded successfully",
        "updated_metrics": {
            "mae": metrics.get("mae"),
            "robust_mae": metrics.get("robust_mae"),
            "median_error": metrics.get("median_error"),
            "count": metrics.get("count")
        }
    }