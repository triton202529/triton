# app.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

REPO_ROOT = Path(__file__).resolve().parent
CONF_PATH = REPO_ROOT / "config" / "alpaca.json"
TRADE_LOG = REPO_ROOT / "data" / "results" / "executed_trades.csv"


# ---------- helpers ----------
@st.cache_resource
def get_client_and_conf():
    conf = json.loads(CONF_PATH.read_text(encoding="utf-8-sig"))
    client = TradingClient(conf["key_id"], conf["secret_key"], paper=conf.get("paper", True))
    return client, conf


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def df_positions(positions) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in positions:
        rows.append(
            {
                "Symbol": p.symbol,
                "Qty": to_float(p.qty),
                "Avg Price": to_float(p.avg_entry_price),
                "Price": to_float(p.current_price),
                "Market Value": to_float(p.market_value),
                "Cost Basis": to_float(p.cost_basis),
                "Unrlzd P/L": to_float(p.unrealized_pl),
                "Unrlzd P/L %": to_float(p.unrealized_plpc),
                "Change Today %": to_float(p.change_today),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Qty",
                "Avg Price",
                "Price",
                "Market Value",
                "Cost Basis",
                "Unrlzd P/L",
                "Unrlzd P/L %",
                "Change Today %",
            ]
        )
    df = pd.DataFrame(rows)
    df["Unrlzd P/L %"] = (df["Unrlzd P/L %"] * 100.0).round(2)
    df["Change Today %"] = (df["Change Today %"] * 100.0).round(2)
    return df.sort_values("Market Value", ascending=False, na_position="last").reset_index(
        drop=True
    )


def df_orders(orders) -> pd.DataFrame:
    rows = []
    for o in orders:
        side = getattr(o.side, "name", str(o.side)).lower()
        tif = getattr(o.time_in_force, "name", str(o.time_in_force)).lower()
        otype = getattr(o.type, "name", str(o.type)).lower()
        qty = getattr(o, "qty", None)
        notional = getattr(o, "notional", None)
        qty_or_notional = (
            str(qty) if qty is not None else (f"${notional}" if notional is not None else "")
        )
        rows.append(
            {
                "Symbol": o.symbol,
                "Side": side,
                "Qty/Notional": qty_or_notional,
                "Type": otype,
                "TIF": tif,
                "Status": getattr(o, "status", ""),
                "Submitted": str(getattr(o, "submitted_at", ""))[:19],
                "Order ID": str(getattr(o, "id", "")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Side",
                "Qty/Notional",
                "Type",
                "TIF",
                "Status",
                "Submitted",
                "Order ID",
            ]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["Symbol", "Submitted"]).reset_index(drop=True)


def run_script_sync(args: list[str]) -> str:
    """Run a repo-local Python script and return combined stdout/stderr."""
    proc = subprocess.run(
        [str(Path().resolve() / ".venv" / "Scripts" / "python.exe")] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        shell=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return out.strip() or f"(exit code {proc.returncode})"


# ---------- UI ----------
st.set_page_config(page_title="Triton Dashboard", layout="wide")

st.title("📈 Triton — Trading Dashboard")

client, conf = get_client_and_conf()
paper_mode = "PAPER" if conf.get("paper", True) else "LIVE"
st.caption(f"Mode: **{paper_mode}** · Config: `config/alpaca.json`")

col1, col2, col3 = st.columns(3)
with col1:
    try:
        clock = client.get_clock()
        if getattr(clock, "is_open", False):
            st.success("🟢 Market is **OPEN**")
        else:
            st.warning("🔴 Market is **CLOSED**")
        st.write(f"Next open: {getattr(clock, 'next_open', '')}")
        st.write(f"Next close: {getattr(clock, 'next_close', '')}")
    except Exception as e:
        st.error(f"Clock error: {e}")

with col2:
    try:
        acct = client.get_account()
        st.metric("Buying Power", f"${acct.buying_power}")
        st.metric("Cash", f"${acct.cash}")
    except Exception as e:
        st.error(f"Account error: {e}")

with col3:
    try:
        acct = client.get_account()
        st.metric("Equity", f"${acct.equity}")
        st.metric("Portfolio Value", f"${acct.portfolio_value}")
    except Exception as e:
        pass

st.divider()

# Actions
st.subheader("⚙️ Actions")
c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    if st.button("▶️ Run Opening Sequence", use_container_width=True):
        out = run_script_sync(["scripts/opening_sequence.py"])
        st.code(out, language="bash")

with c2:
    if st.button("🧹 Cancel ALL Open Orders", use_container_width=True):
        out = run_script_sync(
            [
                "-c",
                "import json;from pathlib import Path;from alpaca.trading.client import TradingClient;"
                "conf=json.loads(Path('config/alpaca.json').read_text(encoding='utf-8-sig'));"
                "c=TradingClient(conf['key_id'],conf['secret_key'],paper=conf.get('paper',True));"
                "c.cancel_orders();print('Canceled ALL open orders')",
            ]
        )
        st.code(out, language="bash")

with c3:
    if st.button("🛑 Flatten (sell everything)", use_container_width=True):
        out = run_script_sync(
            [
                "scripts/auto_execute_signals.py",
                "--paper",
                "--force-close-all",
                "--cancel-all-open",
                "--max-buys",
                "0",
            ]
        )
        st.code(out, language="bash")

st.divider()

# Positions
st.subheader("📦 Positions")
try:
    positions = client.get_all_positions()
    dfp = df_positions(positions)
    st.dataframe(dfp, use_container_width=True, hide_index=True)
    if not dfp.empty:
        total_mv = dfp["Market Value"].sum(skipna=True)
        total_cb = dfp["Cost Basis"].sum(skipna=True)
        total_pl = (
            (total_mv - total_cb) if (total_mv is not None and total_cb is not None) else None
        )
        st.caption(
            f"Total Market Value ≈ **${total_mv:,.2f}** · Total Cost Basis ≈ **${total_cb:,.2f}** · Unrlzd P/L ≈ **${(total_pl or 0):,.2f}**"
        )
except Exception as e:
    st.error(f"Positions error: {e}")

st.divider()

# Open Orders
st.subheader("🧾 Open Orders")
try:
    open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    dfo = df_orders(open_orders)
    st.dataframe(dfo, use_container_width=True, hide_index=True)
    st.caption(f"Open orders: **{len(dfo)}**")
except Exception as e:
    st.error(f"Orders error: {e}")

st.divider()

# Trade log
st.subheader("🗂 Recent Executed Trades (from data/results/executed_trades.csv)")
if TRADE_LOG.exists():
    try:
        dflog = pd.read_csv(TRADE_LOG, encoding="utf-8-sig")
        # best-effort normalization
        cols = [c.strip() for c in dflog.columns]
        dflog.columns = cols
        st.dataframe(dflog.tail(200), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read trade log: {e}")
else:
    st.info("No trade log found yet.")

st.caption("Tip: use the buttons above to manage orders and run your morning playbook.")
