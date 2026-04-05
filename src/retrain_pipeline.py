import subprocess
import datetime
import sys
import os
import json
import time

from src.drift_detection import check_drift
from src.performance_monitor import compute_metrics

# ============================
# Config
# ============================
DELTA = 0.10  # 10% MAE increase required
ALERT_LOG = "logs/alerts.log"
RETRAIN_LOG = "logs/retraining.log"
LAST_RETRAIN_FILE = "artifacts/last_retrain_time.txt"
BASELINE_FILE = "artifacts/baseline_mae.txt"

COOLDOWN_SECONDS = 300  # 5 minutes


# ============================
# Alert Logging
# ============================
def log_alert(message, mae=None, drift_score=None):
    os.makedirs("logs", exist_ok=True)

    alert = {
        "timestamp": str(datetime.datetime.now()),
        "message": message,
        "mae": mae,
        "drift_score": drift_score
    }

    with open(ALERT_LOG, "a") as f:
        f.write(json.dumps(alert) + "\n")


# ============================
# Cooldown Logic
# ============================
def can_retrain():
    if not os.path.exists(LAST_RETRAIN_FILE):
        return True

    with open(LAST_RETRAIN_FILE, "r") as f:
        last_time = float(f.read().strip())

    if time.time() - last_time < COOLDOWN_SECONDS:
        print("⏳ Skipping retraining (cooldown active)")
        return False

    return True


def update_retrain_time():
    os.makedirs("artifacts", exist_ok=True)

    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(str(time.time()))


# ============================
# Baseline MAE Logic
# ============================
def load_baseline():
    if not os.path.exists(BASELINE_FILE):
        return None

    with open(BASELINE_FILE, "r") as f:
        return float(f.read().strip())


def save_baseline(mae):
    os.makedirs("artifacts", exist_ok=True)

    with open(BASELINE_FILE, "w") as f:
        f.write(str(mae))


# ============================
# Main Function
# ============================
def retrain_if_drift():
    print("Checking for drift and performance...")

    # -------------------------
    # Drift check
    # -------------------------
    drift_result = check_drift()
    drift_score = drift_result.get("drift_score", 0)
    drift_detected = drift_result.get("drift_detected", False)

    print(f"Drift result: {drift_result}")

    # -------------------------
    # Performance check
    # -------------------------
    metrics = compute_metrics()
    
    count = metrics.get("count", 0)

    print(f"Feedback sample count: {count}")

# 🔥 NEW: Minimum data guard
    MIN_SAMPLES = 50

    if count < MIN_SAMPLES:
        print(f"Not enough data for reliable retraining (need {MIN_SAMPLES}, got {count})")
        return

    # 🔥 CHANGE: use robust MAE instead
    robust_mae = metrics.get("robust_mae")
    standard_mae = metrics.get("mae")

    print(f"Robust MAE: {robust_mae}")
    print(f"Standard MAE: {standard_mae}")

    if robust_mae is None:
        print("No valid robust MAE → skipping")
        return

    baseline_mae = load_baseline()

    # -------------------------
    # Initialize baseline
    # -------------------------
    if baseline_mae is None:
        print("📌 Setting initial baseline (robust MAE)")
        save_baseline(robust_mae)
        return

    # -------------------------
    # Check worsening
    # -------------------------
    mae_worsened = robust_mae > baseline_mae * (1 + DELTA)

    # -------------------------
    # Alerting
    # -------------------------
    if drift_detected:
        log_alert("Drift detected", robust_mae, drift_score)

    if mae_worsened:
        log_alert("Robust MAE worsened significantly", robust_mae, drift_score)

    # -------------------------
    # Cooldown
    # -------------------------
    if not can_retrain():
        return

    # -------------------------
    # Final decision
    # -------------------------
    if drift_detected or mae_worsened:

        print("🚨 Retraining triggered!")

        if drift_detected:
            print(f"Reason: Drift detected (score={drift_score})")

        if mae_worsened:
            print(f"Reason: Robust MAE worsened (baseline={baseline_mae}, current={robust_mae})")

        # -------------------------
        # Run training
        # -------------------------
        subprocess.run([sys.executable, "src/train.py"], check=True)

        print("Model retrained successfully.")

        # -------------------------
        # Update timers
        # -------------------------
        update_retrain_time()
        save_baseline(robust_mae)

        # -------------------------
        # Log retraining
        # -------------------------
        with open(RETRAIN_LOG, "a") as f:
            f.write(
                f"{datetime.datetime.now()} - Retrained | "
                f"Drift: {drift_detected} (score={drift_score}) | "
                f"Robust MAE: {robust_mae}\n"
            )

    else:
        print("No retraining needed.")