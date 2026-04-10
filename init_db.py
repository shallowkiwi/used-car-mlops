import sqlite3
import os

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect("data/predictions.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_age REAL,
    km_driven REAL,
    mileage REAL,
    engine REAL,
    max_power REAL,
    seats REAL,
    predicted_price REAL,
    actual_price REAL
)
""")

conn.commit()
conn.close()

print("Database created successfully")