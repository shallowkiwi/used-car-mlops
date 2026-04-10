import sqlite3
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error
import numpy as np

DB_PATH = "data/predictions.db"

def compute_metrics():
    try:
        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()

        # Ensure actual_price exists
        if "actual_price" not in df.columns:
            return {
                "mae": None,
                "median_error": None,
                "robust_mae": None,
                "message": "actual_price not available yet"
            }

        # Remove rows without actual_price
        df = df.dropna(subset=["actual_price"])

        if len(df) < 10:
            return {
                "mae": None,
                "median_error": None,
                "robust_mae": None,
                "message": "Not enough data"
            }

        y_true = df["actual_price"]
        y_pred = df["predicted_price"]

        mae = mean_absolute_error(y_true, y_pred)
        median_err = median_absolute_error(y_true, y_pred)

        # Robust MAE (remove outliers)
        errors = np.abs(y_true - y_pred)
        trimmed = errors[errors < np.percentile(errors, 90)]

        robust_mae = np.mean(trimmed)

        return {
            "mae": float(mae),
            "median_error": float(median_err),
            "robust_mae": float(robust_mae),
            "count": len(df)
        }

    except Exception as e:
        return {
            "mae": None,
            "median_error": None,
            "robust_mae": None,
            "message": f"Error: {str(e)}"
        }