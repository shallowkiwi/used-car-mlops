import numpy as np
import os
import json

LOG_FILE = "logs/predictions.log"

# ============================
# Config
# ============================
WINDOW_SIZE = 100   # recent batch size
MIN_REQUIRED = 50   # minimum samples to compute drift
DRIFT_THRESHOLD = 0.1   # ✅ UPDATED (was 1.5)

FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


# ============================
# Load structured feature data
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


# ============================
# Drift Calculation
# ============================
def compute_feature_drift(old_vals, new_vals):
    old_vals = np.array(old_vals)
    new_vals = np.array(new_vals)

    if len(old_vals) == 0 or len(new_vals) == 0:
        return None

    old_mean = np.mean(old_vals)
    new_mean = np.mean(new_vals)

    old_std = np.std(old_vals) + 1e-6  # avoid division by zero

    # normalized shift (z-score style)
    drift = abs(new_mean - old_mean) / old_std

    return drift


# ============================
# Main Drift Function
# ============================
def check_drift(new_data=None):

    records = load_feature_data()

    if len(records) < MIN_REQUIRED:
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "message": "Not enough data for drift detection"
        }

    # -----------------------------
    # Split into OLD vs NEW batches
    # -----------------------------
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
            "message": "No valid features for drift"
        }

    final_score = float(np.mean(drift_scores))
    drift_detected = final_score > DRIFT_THRESHOLD

    return {
        "drift_score": final_score,
        "drift_detected": drift_detected,
        "message": "Drift detected!" if drift_detected else "No drift"
    }