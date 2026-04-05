import json
import os
import numpy as np

LOG_FILE = "logs/predictions.log"


def load_joined_records():
    """
    Join prediction + feedback logs using prediction_id
    """

    if not os.path.exists(LOG_FILE):
        return []

    predictions = {}
    feedbacks = {}

    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                record_type = data.get("type")

                # Prediction records
                if record_type == "prediction":
                    pid = data.get("prediction_id")
                    if pid is not None:
                        predictions[pid] = data

                # Feedback records
                elif record_type == "feedback":
                    pid = data.get("prediction_id")
                    if pid is not None:
                        feedbacks[pid] = data.get("actual")

            except:
                continue

    joined = []

    for pid, pred_data in predictions.items():
        actual = feedbacks.get(pid)

        if actual is None:
            continue

        prediction = pred_data.get("prediction")

        if prediction is None:
            continue

        joined.append({
            "prediction": prediction,
            "actual": actual
        })

    return joined


def compute_metrics():

    records = load_joined_records()

    if len(records) == 0:
        return {
            "mae": None,
            "rmse": None,
            "median_error": None,
            "trimmed_mae": None,
            "count": 0
        }

    predictions = np.array([r["prediction"] for r in records])
    actuals = np.array([r["actual"] for r in records])

    if len(predictions) == 0 or len(actuals) == 0:
        return {
            "mae": None,
            "rmse": None,
            "median_error": None,
            "trimmed_mae": None,
            "count": 0
        }

    errors = predictions - actuals

    if len(errors) == 0:
        return {
            "mae": None,
            "rmse": None,
            "median_error": None,
            "trimmed_mae": None,
            "count": 0
        }

    abs_errors = np.abs(errors)

    # -------------------------
    # Standard metrics
    # -------------------------
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # -------------------------
    # NEW: Robust metrics
    # -------------------------

    # Median Absolute Error
    median_error = float(np.median(abs_errors))

    # Trimmed MAE (remove top 10% largest errors)
    sorted_errors = np.sort(abs_errors)

    trim_ratio = 0.1
    trim_count = int(len(sorted_errors) * trim_ratio)

    if len(sorted_errors) > 10 and trim_count > 0:
        trimmed = sorted_errors[:-trim_count]
        trimmed_mae = float(np.mean(trimmed))
    else:
        trimmed_mae = mae  # fallback

    return {
        "mae": mae,
        "rmse": rmse,
        "median_error": median_error,

    # 🔥 FIX: expose robust_mae (alias)
        "robust_mae": trimmed_mae,

    # keep original for compatibility
        "trimmed_mae": trimmed_mae,

        "count": len(errors)
    }