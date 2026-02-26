import pickle
import numpy as np

def load_model():
    with open("used_car_mlops/models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict_price(features):
    model = load_model()
    prediction = model.predict([features])
    return prediction[0]
