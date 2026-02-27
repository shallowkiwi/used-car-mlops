import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Simulated realistic dataset
data = {
    "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "km_driven": [50000, 40000, 30000, 20000, 15000, 10000, 8000, 5000],
    "fuel_type": [0, 1, 1, 0, 0, 1, 1, 0],
    "transmission": [0, 1, 0, 1, 0, 1, 0, 1],
    "owner_count": [1, 2, 1, 1, 2, 1, 1, 1],
    "engine_size": [1200, 1500, 1300, 1600, 1400, 1800, 2000, 2200],
    "mileage": [18, 20, 19, 21, 22, 17, 16, 15],
    "seats": [5, 5, 5, 5, 7, 5, 7, 5],
    "price": [500000, 600000, 650000, 750000, 800000, 900000, 1100000, 1300000]
}

df = pd.DataFrame(data)

X = df.drop("price", axis=1)
y = df["price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save versioned model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_path = f"models/model_{timestamp}.pkl"

with open(versioned_path, "wb") as f:
    pickle.dump(model, f)

# Save latest model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and versioned successfully!")