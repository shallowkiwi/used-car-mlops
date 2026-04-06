import requests
import random
import time

BASE_URL = "https://used-car-api-hvc8.onrender.com"

# ----------------------------------------
# Generate input
# ----------------------------------------
def generate_input(drift=False):

    if drift:
        return {
            "vehicle_age": random.randint(0, 1),
            "km_driven": random.randint(1000, 10000),
            "mileage": random.uniform(25, 35),
            "engine": random.uniform(800, 1000),
            "max_power": random.uniform(50, 70),
            "seats": random.randint(4, 5)
        }

    return {
        "vehicle_age": random.randint(1, 12),
        "km_driven": random.randint(10000, 150000),
        "mileage": random.uniform(12, 22),
        "engine": random.uniform(1000, 2000),
        "max_power": random.uniform(70, 150),
        "seats": random.randint(4, 7)
    }


# ----------------------------------------
# GOOD actual price (valid)
# ----------------------------------------
def generate_good_actual(prediction):
    noise = random.uniform(0.8, 1.2)
    return max(20000, min(prediction * noise, 2000000))


# ----------------------------------------
# BAD actual price (should be rejected)
# ----------------------------------------
def generate_bad_actual(prediction):
    # extreme unrealistic deviation
    return prediction * random.uniform(2.5, 4.0)


# ----------------------------------------
# Main runner
# ----------------------------------------
def run_test(n_requests=100, drift_start=60, bad_feedback_ratio=0.2):

    print("\n🚀 Starting STRICT validation test...\n")

    rejected = 0
    accepted = 0

    for i in range(n_requests):

        drift = i >= drift_start
        data = generate_input(drift=drift)

        # -------------------------
        # Predict
        # -------------------------
        res = requests.post(f"{BASE_URL}/predict", json=data)

        if res.status_code != 200:
            print(f"[{i}] ❌ Prediction failed")
            continue

        result = res.json()

        prediction_id = result["prediction_id"]
        prediction = result["predicted_price"]

        # -------------------------
        # Decide GOOD vs BAD feedback
        # -------------------------
        if random.random() < bad_feedback_ratio:
            actual_price = generate_bad_actual(prediction)
            is_bad = True
        else:
            actual_price = generate_good_actual(prediction)
            is_bad = False

        feedback_payload = {
            "prediction_id": prediction_id,
            "actual_price": float(actual_price)
        }

        fb_res = requests.post(f"{BASE_URL}/feedback", json=feedback_payload)

        # -------------------------
        # Result handling
        # -------------------------
        if fb_res.status_code == 200:
            accepted += 1
            status = "✅ ACCEPTED"
        else:
            rejected += 1
            status = "🚫 REJECTED"

        print(
            f"[{i+1}] Pred: {round(prediction)} | "
            f"Actual: {round(actual_price)} | "
            f"{'BAD' if is_bad else 'GOOD'} → {status}"
        )

        time.sleep(0.05)

    print("\n==============================")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print("==============================\n")


if __name__ == "__main__":
    run_test()