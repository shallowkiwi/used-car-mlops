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
    print("Model not found. Training model...")
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

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        logging.info("New model loaded successfully.")

    except Exception as e:
        logging.error(f"Retraining failed: {e}")


# =========================================================
# Updated Input Schema
# =========================================================

class CarFeatures(BaseModel):
    vehicle_age: int = Field(..., ge=0)
    km_driven: int = Field(..., ge=0)
    mileage: float = Field(..., gt=0)
    engine: float = Field(..., gt=0)
    max_power: float = Field(..., gt=0)
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

    with open("logs/predictions.log", "a") as f:
        f.write(json.dumps(data.dict()) + "\n")

    input_df = pd.DataFrame([{
        "vehicle_age": data.vehicle_age,
        "km_driven": data.km_driven,
        "mileage": data.mileage,
        "engine": data.engine,
        "max_power": data.max_power,
        "seats": data.seats
    }])

    prediction = model.predict(input_df)[0]

    drift_result = check_drift(data.dict())

    if drift_result["drift_detected"]:
        trigger_retraining()

    logging.info(f"Prediction: {prediction} | Drift: {drift_result}")

    return {
        "predicted_price": float(prediction),
        "drift": drift_result
    }