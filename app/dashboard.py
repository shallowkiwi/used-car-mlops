import streamlit as st
import pandas as pd
import sqlite3
import os
import sys

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.performance_monitor import compute_metrics

# ================================
# 📂 DATABASE PATH
# ================================
DB_PATH = "data/predictions.db"


# ================================
# 📥 LOAD DATA FROM SQLITE
# ================================
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    except Exception:
        df = pd.DataFrame()

    conn.close()

    if df.empty:
        return df

    # Convert timestamp if exists
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


# ================================
# 🎯 STREAMLIT UI
# ================================
st.set_page_config(page_title="Used Car MLOps Dashboard", layout="wide")

st.title("🚗 Used Car Price MLOps Dashboard")

df = load_data()
metrics = compute_metrics()

# ================================
# 🔥 METRICS
# ================================
col1, col2, col3 = st.columns(3)

def safe_format(value):
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


col1.metric("Robust MAE", safe_format(metrics.get("robust_mae")))
col2.metric("Median Error", safe_format(metrics.get("median_error")))
col3.metric("Feedback Samples", metrics.get("count", 0) or 0)

st.caption(f"Standard MAE: {safe_format(metrics.get('mae'))}")

st.divider()

# ================================
# 📊 DATA TABLE
# ================================
st.subheader("📊 Predictions Data")

if df.empty:
    st.warning("No data available yet.")
else:
    st.dataframe(df.tail(20), use_container_width=True)

# ================================
# 📉 ERROR ANALYSIS
# ================================
st.subheader("📉 Error Analysis")

if not df.empty and "actual_price" in df.columns and "predicted_price" in df.columns:
    df["error"] = abs(df["actual_price"] - df["predicted_price"])
    st.line_chart(df["error"])
else:
    st.info("Not enough data for error analysis yet.")