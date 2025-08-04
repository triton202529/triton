import streamlit as st
import pandas as pd
import datetime
import os

st.set_page_config(page_title="TRITON Control Center", layout="wide")

# File path
SIGNALS_FILE = "data/results/signals_with_rationale.csv"

@st.cache_data
def load_signals():
    if os.path.exists(SIGNALS_FILE):
        df = pd.read_csv(SIGNALS_FILE, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # ✅ Drop duplicates to prevent clutter
        df = df.drop_duplicates(subset=["date", "ticker", "signal", "confidence"], keep="last")
        return df
    else:
        return pd.DataFrame()

# Load and filter signals
df = load_signals()
today = datetime.date.today()

st.title("🧠 TRITON Control Center")

if df.empty:
    st.warning("No signal data available yet.")
else:
    # Try to use today's signals, fallback to most recent
    today_signals = df[df["date"] == today]
    if today_signals.empty:
        st.info("No AI signals generated for today. Showing most recent available signals.")
        latest_date = df["date"].max()
        today_signals = df[df["date"] == latest_date]

    if not today_signals.empty:
        st.subheader(f"📋 AI Signals ({today_signals['date'].iloc[0]})")

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, step=0.01)
        with col2:
            rationale_filter = st.text_input("Rationale keyword filter").lower()

        filtered = today_signals[
            (today_signals["confidence"] >= min_conf) &
            (today_signals["rationale"].fillna("").str.lower().str.contains(rationale_filter))
        ]

        st.write(f"✅ Total qualifying signals: {len(filtered)}")

        # Ticker selector
        selected = st.multiselect(
            "Select tickers to execute trades for:",
            options=filtered["ticker"].unique()
        )

        st.dataframe(
            filtered[filtered["ticker"].isin(selected)] if selected else filtered,
            use_container_width=True
        )

        # Manual trade trigger
        if selected:
            if st.button("🚀 Submit Selected Trades"):
                for ticker in selected:
                    row = filtered[filtered["ticker"] == ticker].iloc[0]
                    st.success(f"✅ {row['signal']} order submitted for {ticker} (Confidence: {row['confidence']:.3f})")
    else:
        st.warning("⚠️ No signal rows available for display.")

st.markdown("---")
st.caption("📡 Powered by TRITON AI · Control Center Module")
