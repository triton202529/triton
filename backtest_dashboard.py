# dashboard.py or view_results.py

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Triton Backtest Dashboard", layout="wide")
st.title("📊 Triton Backtest Dashboard")

RESULTS_DIR = "data/results"

# Dynamically list all CSVs in the results folder
csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]

if not csv_files:
    st.warning("No result files found in data/results.")
    st.stop()

selected_file = st.selectbox("📂 Select a results file", csv_files)
file_path = os.path.join(RESULTS_DIR, selected_file)

df = pd.read_csv(file_path)
st.success("✅ File loaded successfully!")

# Preview
st.subheader("🔍 Raw Data Preview")
st.dataframe(df)

# Detect file type based on columns
if {"symbol", "action", "actual_price", "predicted_price", "pnl"}.issubset(df.columns):
    st.subheader("📋 Signal-Based Trade Summary")
    st.dataframe(df[["date", "symbol", "action", "actual_price", "predicted_price", "pnl"]])

    # Plot strategy PnL
    import matplotlib.pyplot as plt
    pnl_by_date = df.groupby("date")["pnl"].sum().cumsum()
    fig, ax = plt.subplots()
    pnl_by_date.plot(ax=ax, label="Cumulative PnL")
    ax.set_title("💰 Cumulative PnL Over Time")
    ax.set_ylabel("PnL ($)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

elif {"cumulative_pnl", "portfolio_value"}.issubset(df.columns):
    st.subheader("📈 Portfolio Growth Over Time")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date")[["portfolio_value", "cumulative_pnl"]].plot(ax=ax)
    ax.set_title("📈 Portfolio Value & PnL Over Time")
    ax.set_ylabel("Value ($)")
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("⚠️ No recognized data columns. Ensure the file contains either portfolio or signal data.")
