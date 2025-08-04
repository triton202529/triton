# backtest_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Triton Backtest Dashboard", layout="centered")
st.markdown("## 📈 Triton Backtest Dashboard")

RESULTS_DIR = "./data/results"

if not os.path.exists(RESULTS_DIR):
    st.error("❌ No results directory found.")
    st.stop()

files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
if not files:
    st.warning("🚫 No result files found. Please run a backtest or simulation.")
    st.stop()

selected_file = st.selectbox("📂 Select a results file", files)
file_path = os.path.join(RESULTS_DIR, selected_file)

try:
    df = pd.read_csv(file_path)

    # Auto-detect date column
    date_col = [col for col in df.columns if col.lower() == "date"]
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]])
    else:
        st.error("❌ No 'date' column found in the selected CSV.")
        st.stop()

    st.success("✅ File loaded successfully!")

    st.write("### 🔍 Raw Data Preview")
    st.dataframe(df.head())

    # ----------- Portfolio Mode -----------
    if "Portfolio Value" in df.columns:
        st.subheader("📊 Portfolio Simulation Performance")

        fig, ax = plt.subplots()
        ax.plot(df[date_col[0]], df["Portfolio Value"], label="Portfolio Value", color="blue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Portfolio Simulation Over Time")
        ax.legend()
        st.pyplot(fig)

    # ----------- Signal-Based Backtest Mode -----------
    elif {"symbol", "action", "actual_price", "predicted_price", "pnl"}.issubset(df.columns):
        st.subheader("📋 Signal-Based Trade Summary")
        st.dataframe(df[["date", "symbol", "action", "actual_price", "predicted_price", "pnl"]])

        st.subheader("💰 Cumulative PnL Over Time")
        df_sorted = df.sort_values(by="date")
        df_sorted["cumulative_pnl"] = df_sorted["pnl"].cumsum()

        fig, ax = plt.subplots()
        ax.plot(df_sorted["date"], df_sorted["cumulative_pnl"], label="Cumulative PnL", color="green")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative PnL ($)")
        ax.set_title("Signal Strategy Performance")
        ax.legend()
        st.pyplot(fig)

        st.download_button("📥 Download PnL Chart", file_name="pnl_chart.png", data=fig_to_image(fig))

        # Optional comparison dropdown
        st.subheader("📊 Market Comparison (Optional)")
        symbol_options = df["symbol"].unique().tolist()
        selected_symbol = st.selectbox("Choose a symbol to compare:", symbol_options)

        symbol_df = df[df["symbol"] == selected_symbol]
        if not symbol_df.empty:
            strategy = symbol_df["pnl"].cumsum()
            market = (symbol_df["actual_price"] - symbol_df["actual_price"].iloc[0])

            fig2, ax2 = plt.subplots()
            ax2.plot(symbol_df["date"], strategy, label="Strategy PnL", color="green")
            ax2.plot(symbol_df["date"], market, label="Market Move", linestyle="--", color="gray")
            ax2.set_title(f"{selected_symbol}: Strategy vs. Market")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("PnL / Price Change")
            ax2.legend()
            st.pyplot(fig2)

    else:
        st.warning("⚠️ No recognized data columns. Ensure the file contains either portfolio or signal data.")

except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")


# Helper to convert matplotlib figure to PNG for download
def fig_to_image(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.read()
