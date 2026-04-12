# ui/legacy_tabs.py
# Legacy tabbed dashboard (Phase 1 UI)

import streamlit as st


def render_legacy_tabs():
    # --------------------------------------------------
    # Legacy tab labels
    # --------------------------------------------------
    tab_labels = [
        "🔍 Portfolio Drilldown",  # 0
        "📈 Portfolio History",  # 1
        "📋 Trade Log",  # 2
        "📊 Strategy vs Market",  # 3
        "🧠 AI Signals + Rationale",  # 4
        "📁 Browse Any CSV",  # 5
        "📋 Backtest Summary",  # 6
        "📉 Risk: Portfolio Drawdown",  # 7
        "📊 Strategy Diagnostics",  # 8
        "🏦 Portfolio Allocations",  # 9
        "📽️ Trade Replay",  # 10
        "📘 Fundamental Data",  # 11
        "📈 Stock Scores",  # 12
        "🎯 Top Fundamental Picks",  # 13
        "📰 News Sentiment",  # 14
        "🚨 Smart Alerts",  # 15
        "📆 Economic Calendar",  # 16
        "🔬 Feature Importance",  # 17
        "🎯 SL/TP Performance Analysis",  # 18
        "💬 Sentiment + Signal Fusion",  # 19
        "📊 Model Comparison",  # 20
        "🧠 AI Learning Lab",  # 21
        "🧾 Buffett Orders (current)",  # 22
        "🗂️ Consolidated Orders (ML × Buffett blend)",  # 23
        "🤖 AI Feedback (allocator runs)",  # 24
        "📚 Equal-Weight Portfolio vs Benchmark",  # 25
        "🧮 Smart-Weight Portfolio vs Benchmark",  # 26
        "🧪 Confidence Calibration",  # 27
        "🧪 Confidence-Filtered Portfolio vs Benchmark",  # 28
        "📊 Confidence × Sharpe Portfolio vs Benchmark",  # 29
        "🧪 Stress Test Reports & Runner",  # 30
        "🩺 Market Sentinels",  # 31
    ]

    tabs = st.tabs(tab_labels)

    # --------------------------------------------------
    # IMPORTANT RULE
    # Every legacy section MUST live inside its tab.
    # No global rendering above or below.
    # --------------------------------------------------

    with tabs[0]:
        st.subheader("Portfolio Drilldown")
        # existing drilldown code here

    with tabs[1]:
        st.subheader("Portfolio History")
        # existing portfolio history code here

    with tabs[2]:
        st.subheader("Trade Log")
        # existing trade log code here

    # …continue pattern for remaining tabs
