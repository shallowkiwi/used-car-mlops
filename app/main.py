import logging
import os
import pandas as pd
import pickle
import subprocess
import sys
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.drift_detection import check_drift


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(title="Used Car Price Prediction API")


# =========================================================
# Load or Train Model
# =========================================================

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model not found. Training model inside container...")
    subprocess.run([sys.executable, "src/train.py"], check=True)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================================================
# Logging Setup
# =========================================================

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# Automated Retraining Trigger
# =========================================================

def trigger_retraining():
    global model

    logging.warning("Drift detected. Starting retraining pipeline...")

    try:
        subprocess.run(
            [sys.executable, "src/train.py"],
            check=True
        )

        logging.info("Retraining completed successfully.")

        # Reload updated model
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        logging.info("New model loaded successfully.")

    except Exception as e:
        logging.error(f"Retraining failed: {e}")


# =========================================================
# Input Schema
# =========================================================

class CarFeatures(BaseModel):
    year: int = Field(..., ge=1990, le=2025)
    km_driven: int = Field(..., ge=0)
    fuel_type: int
    transmission: int
    owner_count: int = Field(..., ge=0, le=5)
    engine_size: int = Field(..., gt=0)
    mileage: float = Field(..., gt=0)
    seats: int = Field(..., ge=2, le=10)


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def home():
    return {"message": "Used Car Price Prediction API Running"}


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict")
def predict(data: CarFeatures):

    logging.info(f"Received input: {data.dict()}")

    # Log prediction input
    with open("logs/predictions.log", "a") as f:
        f.write(json.dumps(data.dict()) + "\n")

    input_df = pd.DataFrame([{
        "year": data.year,
        "km_driven": data.km_driven,
        "fuel_type": data.fuel_type,
        "transmission": data.transmission,
        "owner_count": data.owner_count,
        "engine_size": data.engine_size,
        "mileage": data.mileage,
        "seats": data.seats
    }])

    input_df = input_df[[
        "year",
        "km_driven",
        "fuel_type",
        "transmission",
        "owner_count",
        "engine_size",
        "mileage",
        "seats"
    ]]

    prediction = model.predict(input_df)[0]

    drift_result = check_drift(data.dict())

    # Automated Retraining Trigger
    if drift_result["drift_detected"]:
        trigger_retraining()

    logging.info(f"Prediction: {prediction} | Drift: {drift_result}")

    return {
        "predicted_price": float(prediction),
        "drift": drift_result
    }