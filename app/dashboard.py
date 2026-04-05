import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import sys

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.performance_monitor import compute_metrics

LOG_FILE = "logs/predictions.log"


def load_data():
    data = []

    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "features" in df.columns:
        features_df = pd.json_normalize(df["features"])
        df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)

    df = df.loc[:, ~df.columns.duplicated()]

    if "timestamp" in df.columns:
        df = df[df["timestamp"].notna()]
        df["time"] = df["timestamp"].apply(
            lambda x: datetime.fromtimestamp(float(x))
        )

    return df


st.set_page_config(page_title="Used Car MLOps Dashboard", layout="wide")

st.title("🚗 Used Car Price MLOps Dashboard")

df = load_data()
metrics = compute_metrics()

# =========================================================
# 🔥 METRICS
# =========================================================
col1, col2, col3 = st.columns(3)

col1.metric("Robust MAE", f"{metrics.get('trimmed_mae', 0):,.2f}")
col2.metric("Median Error", f"{metrics.get('median_error', 0):,.2f}")
col3.metric("Feedback Samples", metrics.get("count", 0))

# Optional: show original MAE
st.caption(f"Standard MAE: {metrics.get('mae', 0):,.2f}")

st.divider()

# =========================================================
# 📊 DATA TABLE
# =========================================================
st.subheader("📊 Predictions Data")

if df.empty:
    st.warning("No data available yet.")
else:
    st.dataframe(df.tail(20), use_container_width=True)

# =========================================================
# 📉 ERROR ANALYSIS
# =========================================================
st.subheader("📉 Error Analysis")

if not df.empty and "actual" in df.columns and "prediction" in df.columns:
    df["error"] = abs(df["actual"] - df["prediction"])
    st.line_chart(df["error"])
else:
    st.info("Not enough data for error analysis yet.")