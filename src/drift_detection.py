import numpy as np
import sqlite3
import os
import time

DB_FILE = "data/monitoring.db"

WINDOW_SIZE = 30
MIN_REQUIRED = 30
DRIFT_THRESHOLD = 0.1

RETRAIN_COOLDOWN = 3600
LAST_RETRAIN_FILE = "logs/last_retrain.txt"

FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


def get_connection():
    return sqlite3.connect(DB_FILE)


def load_feature_data():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT vehicle_age, km_driven, mileage, engine, max_power, seats
        FROM predictions
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        dict(zip(FEATURES, row))
        for row in rows
    ]


def compute_feature_drift(old_vals, new_vals):
    old_vals = np.array(old_vals)
    new_vals = np.array(new_vals)

    if len(old_vals) == 0 or len(new_vals) == 0:
        return None

    old_mean = np.mean(old_vals)
    new_mean = np.mean(new_vals)
    old_std = np.std(old_vals) + 1e-6

    return abs(new_mean - old_mean) / old_std


def can_retrain():
    if not os.path.exists(LAST_RETRAIN_FILE):
        return True

    with open(LAST_RETRAIN_FILE, "r") as f:
        last_time = float(f.read().strip())

    return (time.time() - last_time) > RETRAIN_COOLDOWN


def check_drift():

    records = load_feature_data()

    if len(records) < MIN_REQUIRED:
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "retrain_recommended": False,
            "message": "Not enough data"
        }

    split_index = max(0, len(records) - 2 * WINDOW_SIZE)

    old_batch = records[split_index:split_index + WINDOW_SIZE]
    new_batch = records[-WINDOW_SIZE:]

    drift_scores = []

    for feature in FEATURES:
        old_vals = [r[feature] for r in old_batch]
        new_vals = [r[feature] for r in new_batch]

        drift = compute_feature_drift(old_vals, new_vals)

        if drift is not None:
            drift_scores.append(drift)

    if not drift_scores:
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "retrain_recommended": False,
            "message": "No valid features"
        }

    final_score = float(np.mean(drift_scores))
    drift_detected = final_score > DRIFT_THRESHOLD

    retrain_recommended = drift_detected and can_retrain()

    return {
        "drift_score": final_score,
        "drift_detected": drift_detected,
        "retrain_recommended": retrain_recommended,
        "message": (
            "Retrain recommended"
            if retrain_recommended
            else "Cooldown active"
            if drift_detected
            else "No drift"
        )
    }