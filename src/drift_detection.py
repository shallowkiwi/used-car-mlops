import numpy as np
import os
import json

# Reference mean from training data
REFERENCE_MEAN = 50000
DRIFT_THRESHOLD = 20000

# Make sure this path matches your project structure
LOG_FILE = "logs/predictions.log"


def check_drift(new_data=None):
    """
    Checks drift based on km_driven mean difference
    between logged predictions and reference mean.
    """

    # If log file does not exist
    if not os.path.exists(LOG_FILE):
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "message": "No prediction log found"
        }

    km_values = []

    # Read log file
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "km_driven" in data:
                    km_values.append(float(data["km_driven"]))
            except Exception:
                continue

    # If no valid km values
    if len(km_values) == 0:
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "message": "No valid km_driven data found"
        }

    new_mean = float(np.mean(km_values))
    drift_score = float(abs(new_mean - REFERENCE_MEAN))
    drift_detected = bool(drift_score > DRIFT_THRESHOLD)

    return {
        "drift_score": drift_score,
        "drift_detected": drift_detected,
        "message": "Drift detected!" if drift_detected else "No drift"
    }