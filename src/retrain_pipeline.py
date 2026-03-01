import subprocess
import datetime
from drift_detection import check_drift

DRIFT_THRESHOLD = 0.05  # adjust if needed


def retrain_if_drift():
    print("Checking for drift...")

    drift_result = check_drift()   # This returns a dictionary
    print(f"Drift result: {drift_result}")

    # Extract values safely
    drift_score = drift_result.get("drift_score", 0)
    drift_detected = drift_result.get("drift_detected", False)

    if drift_detected:
        print("Drift detected! Retraining model...")

        # Call training script
        subprocess.run(["python", "src/train.py"], check=True)

        print("Model retrained successfully.")

        # Log retraining event
        with open("logs/retraining.log", "a") as f:
            f.write(
                f"{datetime.datetime.now()} - Retrained due to drift score: {drift_score}\n"
            )

    else:
        print("No significant drift detected.")


if __name__ == "__main__":
    retrain_if_drift()