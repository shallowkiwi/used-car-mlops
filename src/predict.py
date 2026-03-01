import pickle
import json
import os

LOG_FILE = "logs/predictions.log"

def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def log_prediction(features):
    # Assuming km_driven is index 1 in your features
    km_driven = features[1]

    os.makedirs("logs", exist_ok=True)

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "km_driven": km_driven
        }) + "\n")

def predict_price(features):
    model = load_model()

    prediction = model.predict([features])[0]

    # 🔥 Log input data for drift monitoring
    log_prediction(features)

    return float(prediction)