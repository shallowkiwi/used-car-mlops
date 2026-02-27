import pickle
<<<<<<< HEAD
import os
import numpy as np
import logging

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

def predict_price(features):
    model = load_model()

    prediction = model.predict([features])[0]

    logging.info(f"Input: {features} | Prediction: {prediction}")

    return float(prediction)
=======
import numpy as np

def load_model():
    with open("used_car_mlops/models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict_price(features):
    model = load_model()
    prediction = model.predict([features])
    return prediction[0]
>>>>>>> c49679dfd936bc738dd1b4c49795b7daa9a40fd4
