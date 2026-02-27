from fastapi import FastAPI
<<<<<<< HEAD
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
=======
from src.predict import predict_price

app = FastAPI()
>>>>>>> c49679dfd936bc738dd1b4c49795b7daa9a40fd4

@app.get("/")
def home():
    return {"message": "Used Car Price Prediction API Running"}

@app.post("/predict")
<<<<<<< HEAD
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
=======
def predict(data: dict):
    features = list(data.values())
    price = predict_price(features)
    return {"predicted_price": price}
>>>>>>> c49679dfd936bc738dd1b4c49795b7daa9a40fd4
