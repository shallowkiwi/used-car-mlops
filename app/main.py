from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_price
from src.drift_detection import detect_drift

app = FastAPI(title="Used Car Price Prediction API")

class CarFeatures(BaseModel):
    year: float
    km_driven: float
    fuel_type: float
    transmission: float
    owner_count: float
    engine_size: float
    mileage: float
    seats: float

@app.get("/")
def home():
    return {"message": "Used Car Price Prediction API Running"}

@app.post("/predict")
def predict(data: CarFeatures):

    features = [
        data.year,
        data.km_driven,
        data.fuel_type,
        data.transmission,
        data.owner_count,
        data.engine_size,
        data.mileage,
        data.seats
    ]

    prediction = predict_price(features)
    drift_status = detect_drift(features)

    return {
        "predicted_price": prediction,
        "drift_status": drift_status
    }