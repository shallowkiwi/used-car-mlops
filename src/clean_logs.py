import json
import pandas as pd

INPUT_FILE = "logs/predictions.log"
OUTPUT_FILE = "data/cleaned_data.csv"

clean_records = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        try:
            data = json.loads(line.strip())

            # Skip invalid schema
            if "features" not in data:
                continue

            if data.get("actual") is None:
                continue

            features = data["features"]

            record = {
                **features,
                "selling_price": data["actual"]
            }

            # Hard filtering
            if not (0 <= record["vehicle_age"] <= 20):
                continue
            if not (0 <= record["km_driven"] <= 300000):
                continue
            if not (5 <= record["mileage"] <= 40):
                continue
            if not (500 <= record["engine"] <= 5000):
                continue
            if not (20 <= record["max_power"] <= 500):
                continue
            if not (2 <= record["seats"] <= 8):
                continue
            if not (20000 <= record["selling_price"] <= 2000000):
                continue

            clean_records.append(record)

        except:
            continue

df = pd.DataFrame(clean_records)

print(f"Clean samples: {len(df)}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")