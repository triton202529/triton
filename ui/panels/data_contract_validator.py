# ui/panels/data_contract_validator.py
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _safe_read_csv(path: Path) -> Tuple[bool, Optional[pd.DataFrame], str]:
    if not path.exists():
        return False, None, "Missing file"
    if path.stat().st_size == 0:
        return False, None, "Empty file (0 bytes)"
    try:
        df = pd.read_csv(path)
        if df.empty:
            return True, df, "Loaded (but empty)"
        return True, df, "Loaded"
    except Exception as e:
        return False, None, f"Read error: {e}"


def _last_modified(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "--"


def render_data_contract_validator(
    RESULTS_DIR: Path,
    ORDERS_DIR: Path,
    PRED_DIR: Path,
    STRESS_DIR: Path,
):
    st.markdown(
        '<div class="triton-card"><h3>🧾 Data Contract Validator</h3>'
        '<p class="data-label">Checks whether Triton’s required CSVs/parquets exist, load correctly, and contain expected columns.</p></div>',
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Define “contracts” here
    # -----------------------------
    # filename -> { folder, required_columns, optional_columns }
    contracts: Dict[str, Dict[str, object]] = {
        # Core / Live
        "portfolio_history.csv": {
            "folder": RESULTS_DIR,
            "required": ["timestamp", "equity"],
            "optional": ["cash", "market_value", "buying_power"],
        },
        "live_orders.csv": {
            "folder": RESULTS_DIR,
            "required": ["timestamp", "symbol", "side", "qty", "status"],
            "optional": ["limit_price", "order_type", "note"],
        },
        "guard_snapshot.json": {
            "folder": RESULTS_DIR,
            "required": [],  # validated by existence only here
            "optional": [],
            "type": "json",
        },
        # Trading outputs
        "trade_log.csv": {
            "folder": RESULTS_DIR,
            "required": ["date", "ticker"],
            "optional": ["signal", "side", "qty", "entry_price", "exit_price", "profit", "pnl"],
        },
        "signals_with_rationale.csv": {
            "folder": RESULTS_DIR,
            "required": ["date", "ticker"],
            "optional": [
                "signal",
                "confidence",
                "close",
                "predicted_close",
                "rationale",
                "sentiment",
                "total_score",
            ],
        },
        "signals.csv": {
            "folder": RESULTS_DIR,
            "required": ["date", "ticker"],
            "optional": ["signal", "confidence", "close", "predicted_close"],
        },
        # Research
        "news_sentiment.csv": {
            "folder": RESULTS_DIR,
            "required": ["date", "ticker"],
            "optional": ["sentiment", "title", "url", "source"],
        },
        "economic_calendar.csv": {
            "folder": RESULTS_DIR,
            "required": ["date"],
            "optional": ["event", "impact", "country", "forecast", "previous"],
        },
        # Model comparison / summaries
        "backtest_summary.csv": {
            "folder": RESULTS_DIR,
            "required": [],
            "optional": [
                "ticker",
                "model",
                "strategy",
                "sharpe",
                "cagr",
                "total_return",
                "win_rate",
                "rmse",
                "mae",
            ],
        },
        "strategy_vs_market.csv": {
            "folder": RESULTS_DIR,
            "required": ["date", "ticker"],
            "optional": [
                "strategy_return",
                "market_return",
                "cumulative_strategy",
                "cumulative_market",
            ],
        },
    }

    # Optional: show path context
    with st.expander("📌 Current data folders (resolved)"):
        st.code(
            "\n".join(
                [
                    f"RESULTS_DIR = {RESULTS_DIR}",
                    f"ORDERS_DIR  = {ORDERS_DIR}",
                    f"PRED_DIR    = {PRED_DIR}",
                    f"STRESS_DIR  = {STRESS_DIR}",
                ]
            )
        )

    if st.button("↻ Refresh / re-check files", key="dc_refresh"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()

    rows = []
    details = {}

    for fname, meta in contracts.items():
        folder: Path = meta.get("folder")  # type: ignore
        ftype = str(meta.get("type", "csv"))

        full_path = folder / fname

        if ftype == "json":
            exists = (
                full_path.exists() and full_path.stat().st_size > 0 if full_path.exists() else False
            )
            status = "OK" if exists else "MISSING"
            rows.append(
                {
                    "file": fname,
                    "type": "json",
                    "folder": str(folder),
                    "status": status,
                    "last_modified": _last_modified(full_path) if full_path.exists() else "--",
                    "missing_required_cols": "",
                    "notes": "Exists" if exists else "Missing or empty",
                }
            )
            continue

        ok, df, msg = _safe_read_csv(full_path)

        required: List[str] = list(meta.get("required", []))  # type: ignore
        optional: List[str] = list(meta.get("optional", []))  # type: ignore

        if not ok:
            rows.append(
                {
                    "file": fname,
                    "type": "csv",
                    "folder": str(folder),
                    "status": "MISSING/ERROR",
                    "last_modified": _last_modified(full_path) if full_path.exists() else "--",
                    "missing_required_cols": ", ".join(required) if required else "",
                    "notes": msg,
                }
            )
            continue

        cols = list(df.columns) if df is not None else []
        missing_required = [c for c in required if c not in cols]
        status = "OK" if not missing_required else "BAD SCHEMA"

        rows.append(
            {
                "file": fname,
                "type": "csv",
                "folder": str(folder),
                "status": status,
                "last_modified": _last_modified(full_path),
                "missing_required_cols": ", ".join(missing_required),
                "notes": msg,
            }
        )

        details[fname] = {
            "path": str(full_path),
            "columns": cols,
            "required": required,
            "optional": optional,
            "missing_required": missing_required,
            "head": df.head(25) if df is not None else pd.DataFrame(),
        }

    df_report = pd.DataFrame(rows)

    # Make status readable
    def _badge(s: str) -> str:
        s = str(s)
        if s == "OK":
            return "✅ OK"
        if s == "BAD SCHEMA":
            return "🟠 BAD SCHEMA"
        return "🔴 MISSING/ERROR"

    df_report["status"] = df_report["status"].apply(_badge)

    st.subheader("📋 Contract Status")
    st.dataframe(df_report, use_container_width=True)

    # Drilldown
    st.subheader("🔍 Drill into a file")
    selectable = [r["file"] for r in rows if r.get("type") == "csv"]
    chosen = st.selectbox("Select CSV", options=selectable, key="dc_pick")

    if chosen in details:
        info = details[chosen]

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.caption("Resolved path")
            st.code(info["path"])
        with c2:
            if info["missing_required"]:
                st.warning(f"Missing required: {', '.join(info['missing_required'])}")
            else:
                st.success("All required columns present")

        with st.expander("Columns"):
            st.write(info["columns"])

        with st.expander("Preview (first 25 rows)"):
            st.dataframe(info["head"], use_container_width=True)
