import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Creating a simple dataset
data = {
    "year": [2015, 2016, 2017, 2018, 2019],
    "km_driven": [50000, 40000, 30000, 20000, 10000],
    "price": [500000, 600000, 700000, 800000, 900000]
}

df = pd.DataFrame(data)

# Features (inputs)
X = df[["year", "km_driven"]]

# Target (output)
y = df["price"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save trained model
with open("used_car_mlops/models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
