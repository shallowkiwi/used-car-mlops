import sqlite3
import pandas as pd

DB_PATH = "data/predictions.db"

def check_drift():
    try:
        conn = sqlite3.connect(DB_PATH)

        # Load all predictions
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()

        # Not enough data case
        if len(df) < 5:
            return {
                "drift_score": 0,
                "drift_detected": False,
                "retrain_recommended": False,
                "message": "Not enough data for drift detection"
            }

        # Split old vs recent
        recent = df.tail(5)
        old = df.head(5)

        # Simple drift calculation (on km_driven)
        drift_score = abs(recent["km_driven"].mean() - old["km_driven"].mean())

        drift_detected = drift_score > 1000

        return {
            "drift_score": float(drift_score),
            "drift_detected": drift_detected,
            "retrain_recommended": drift_detected,
            "message": "Retrain recommended" if drift_detected else "No drift"
        }

    except Exception as e:
        return {
            "drift_score": 0,
            "drift_detected": False,
            "retrain_recommended": False,
            "message": f"Error: {str(e)}"
        }