import numpy as np
import os
import json
import time

LOG_FILE = "logs/predictions.log"

# ============================
# Config
# ============================
WINDOW_SIZE = 30
MIN_REQUIRED = 30
DRIFT_THRESHOLD = 0.1

# NEW
RETRAIN_COOLDOWN = 3600  # seconds (1 hour)
LAST_RETRAIN_FILE = "logs/last_retrain.txt"

FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


# ============================
# Helpers
# ============================
def load_feature_data():
    if not os.path.exists(LOG_FILE):
        return []

    records = []

    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "features" in data:
                    records.append(data["features"])
            except:
                continue

    return records


def compute_feature_drift(old_vals, new_vals):
    old_vals = np.array(old_vals)
    new_vals = np.array(new_vals)

    if len(old_vals) == 0 or len(new_vals) == 0:
        return None

    old_mean = np.mean(old_vals)
    new_mean = np.mean(new_vals)

    old_std = np.std(old_vals) + 1e-6

    drift = abs(new_mean - old_mean) / old_std
    return drift


def can_retrain():
    if not os.path.exists(LAST_RETRAIN_FILE):
        return True

    with open(LAST_RETRAIN_FILE, "r") as f:
        last_time = float(f.read().strip())

    return (time.time() - last_time) > RETRAIN_COOLDOWN


def mark_retrain_time():
    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(str(time.time()))


# ============================
# Main Drift Function
# ============================
def check_drift():

    records = load_feature_data()

    if len(records) < MIN_REQUIRED:
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "retrain_recommended": False,
            "message": "Not enough data for drift detection"
        }

    split_index = max(0, len(records) - 2 * WINDOW_SIZE)

    old_batch = records[split_index : split_index + WINDOW_SIZE]
    new_batch = records[-WINDOW_SIZE:]

    drift_scores = []

    for feature in FEATURES:
        old_vals = [r[feature] for r in old_batch if feature in r]
        new_vals = [r[feature] for r in new_batch if feature in r]

        drift = compute_feature_drift(old_vals, new_vals)

        if drift is not None:
            drift_scores.append(drift)

    if len(drift_scores) == 0:
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "retrain_recommended": False,
            "message": "No valid features for drift"
        }

    final_score = float(np.mean(drift_scores))
    drift_detected = final_score > DRIFT_THRESHOLD

    # ============================
    # NEW LOGIC
    # ============================
    retrain_allowed = can_retrain()

    retrain_recommended = drift_detected and retrain_allowed

    return {
        "drift_score": final_score,
        "drift_detected": drift_detected,
        "retrain_recommended": retrain_recommended,
        "message": (
            "Retrain recommended"
            if retrain_recommended
            else "Drift detected but cooldown active"
            if drift_detected
            else "No drift"
        )
    }