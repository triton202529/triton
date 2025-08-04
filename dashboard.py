import pandas as pd
import streamlit as st

# Load the trading signals CSV
file_path = "data/signals/trading_signals.csv"
df = pd.read_csv(file_path)

# Streamlit app setup
st.set_page_config(page_title="Triton Trading Signals", layout="wide")
st.title("📈 Triton AI Trading Signals Dashboard")

# Optional filters
with st.sidebar:
    st.header("🔍 Filter Signals")
    tickers = st.multiselect("Select tickers:", sorted(df["ticker"].unique()), default=sorted(df["ticker"].unique()))
    signal_types = st.multiselect("Select signal types:", sorted(df["signal"].unique()), default=sorted(df["signal"].unique()))

# Apply filters
filtered_df = df[df["ticker"].isin(tickers) & df["signal"].isin(signal_types)]

# Display the signals
st.dataframe(filtered_df.sort_values(by="date", ascending=False), use_container_width=True)

# Save file for user download (optional)
st.download_button(
    label="📥 Download CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_trading_signals.csv",
    mime="text/csv"
)
