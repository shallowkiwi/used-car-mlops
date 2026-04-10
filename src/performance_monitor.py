import sqlite3
import numpy as np
import os

DB_FILE = "data/monitoring.db"


def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_FILE)


def load_joined_records():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT prediction, actual
        FROM predictions
        WHERE actual IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    return [{"prediction": r[0], "actual": r[1]} for r in rows]


def compute_metrics():

    records = load_joined_records()

    if len(records) == 0:
        return {
            "mae": None,
            "rmse": None,
            "median_error": None,
            "trimmed_mae": None,
            "robust_mae": None,
            "count": 0
        }

    predictions = np.array([r["prediction"] for r in records])
    actuals = np.array([r["actual"] for r in records])

    errors = predictions - actuals
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_error = float(np.median(abs_errors))

    sorted_errors = np.sort(abs_errors)
    trim_ratio = 0.1
    trim_count = int(len(sorted_errors) * trim_ratio)

    if len(sorted_errors) > 10 and trim_count > 0:
        trimmed = sorted_errors[:-trim_count]
        trimmed_mae = float(np.mean(trimmed))
    else:
        trimmed_mae = mae

    return {
        "mae": mae,
        "rmse": rmse,
        "median_error": median_error,
        "robust_mae": trimmed_mae,
        "trimmed_mae": trimmed_mae,
        "count": len(errors)
    }