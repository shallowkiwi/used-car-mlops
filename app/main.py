from fastapi import FastAPI
from used_car_mlops.src.predict import predict_price

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Used Car Price Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    features = list(data.values())
    price = predict_price(features)
    return {"predicted_price": price}
