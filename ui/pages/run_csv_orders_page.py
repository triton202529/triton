# ui/pages/run_csv_orders_page.py
"""
TRITON — Run CSV Orders (Phase 2.2)
-----------------------------------
Streamlit page wrapper around place_orders_from_csv.py.

- DRY RUN by default.
- Respects Guard Kill Switch: blocks REAL placement while allowing validation.
- Uses services/run_cmd.py to execute a child python process and show output.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

from services.run_cmd import run_cmd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ui/pages/ -> project root
DATA_ROOT = PROJECT_ROOT / "data"
ORDERS_DIR = DATA_ROOT / "orders"
RESULTS_DIR = DATA_ROOT / "results"
GUARD_SNAPSHOT_PATH = RESULTS_DIR / "guard_snapshot.json"


def _load_guard() -> Dict[str, Any]:
    if not GUARD_SNAPSHOT_PATH.exists() or GUARD_SNAPSHOT_PATH.stat().st_size == 0:
        return {}
    try:
        return json.loads(GUARD_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _kill_switch_on(guard: Dict[str, Any]) -> Tuple[bool, str]:
    if not guard:
        return False, ""
    reason = str(guard.get("reason") or guard.get("message") or "").strip()
    if guard.get("kill_switch") is True:
        return True, reason or "Kill switch enabled."
    mode = str(guard.get("mode") or "").upper().strip()
    if mode in {"KILL_SWITCH", "KILLSWITCH", "FROZEN", "FREEZE", "MAINTENANCE"}:
        return True, reason or "Trading frozen by guard mode."
    if guard.get("blocked") is True:
        return True, reason or "Trading blocked by guard."
    return False, ""


def _list_csv_files() -> List[Path]:
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in ORDERS_DIR.glob("*.csv") if p.is_file()], key=lambda p: p.name.lower()
    )
    return files


def _read_csv_preview(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _build_cmd(
    csv_path: Path,
    mode: str,
    top_n: int,
    order_type: str,
    limit_pad_bps: int,
    tif: str,
    sl_pct: float,
    tp_pct: float,
    really_place: bool,
    verbose: bool,
    min_notional: float,
) -> List[str]:
    script = PROJECT_ROOT / "place_orders_from_csv.py"
    cmd = [
        sys.executable,
        str(script),
        "--mode",
        mode,
        "--csv",
        str(csv_path),
        "--top-n",
        str(int(top_n)),
        "--order-type",
        order_type,
        "--limit-pad-bps",
        str(int(limit_pad_bps)),
        "--tif",
        tif,
        "--sl-pct",
        str(float(sl_pct)),
        "--tp-pct",
        str(float(tp_pct)),
        "--min-notional",
        str(float(min_notional)),
    ]
    if verbose:
        cmd.append("--verbose")
    if really_place:
        cmd.append("--really-place")
    return cmd


def render() -> None:
    st.markdown("### ▶ Run CSV Orders")
    st.caption(
        "Wrapper for `place_orders_from_csv.py`. DRY RUN by default. Real placement requires explicit toggle."
    )

    guard = _load_guard()
    blocked, reason = _kill_switch_on(guard)
    if blocked:
        st.error(f"⛔ **BLOCKED [KILL_SWITCH]**: {reason}", icon="⛔")
        with st.expander("Guard snapshot", expanded=False):
            st.json(guard)
    else:
        st.info("Kill switch is OFF.", icon="✅")

    files = _list_csv_files()
    if not files:
        st.warning("No CSV files found in data/orders/. Create one like orders_today.csv.")
        st.code(str(ORDERS_DIR))
        return

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        file_choice = st.selectbox("Orders CSV", [p.name for p in files], index=0)
        csv_path = next(p for p in files if p.name == file_choice)
    with c2:
        mode = st.selectbox("Mode", ["paper", "live"], index=0)
    with c3:
        verbose = st.toggle("Verbose", value=True)

    preview = _read_csv_preview(csv_path)
    if preview is None or preview.empty:
        st.warning("CSV is missing/empty or cannot be read.")
    else:
        st.markdown("#### CSV Preview")
        st.dataframe(preview.head(200), use_container_width=True, hide_index=True)

    st.markdown("#### Execution Settings")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        top_n = st.number_input("Top N", min_value=1, max_value=500, value=20, step=1)
        min_notional = st.number_input("Min Notional (USD)", min_value=0.0, value=50.0, step=5.0)
    with s2:
        order_type = st.selectbox("Order Type", ["market", "limit"], index=0)
        limit_pad_bps = st.number_input(
            "Limit Pad (bps)", min_value=0, max_value=500, value=10, step=1
        )
    with s3:
        tif = st.selectbox("TIF", ["day", "gtc"], index=0)
        sl_pct = st.number_input(
            "Stop Loss %", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.4f"
        )
    with s4:
        tp_pct = st.number_input(
            "Take Profit %", min_value=0.0, max_value=2.0, value=0.08, step=0.01, format="%.4f"
        )
        really_place = st.toggle("REAL placement (danger)", value=False)

    if blocked and really_place:
        st.warning("Kill switch is ON — forcing REAL placement off.")
        really_place = False

    cmd = _build_cmd(
        csv_path=csv_path,
        mode=mode,
        top_n=int(top_n),
        order_type=order_type,
        limit_pad_bps=int(limit_pad_bps),
        tif=tif,
        sl_pct=float(sl_pct),
        tp_pct=float(tp_pct),
        really_place=really_place,
        verbose=verbose,
        min_notional=float(min_notional),
    )

    st.markdown("#### Command")
    st.code(" ".join(cmd))

    run = st.button("Run", use_container_width=True)

    if run:
        if blocked and really_place:
            st.error("Kill switch is ON — real placement blocked.")
            return

        if not (PROJECT_ROOT / "place_orders_from_csv.py").exists():
            st.error("place_orders_from_csv.py not found in project root.")
            st.code(str(PROJECT_ROOT))
            return

        with st.spinner("Running..."):
            code, out = run_cmd(cmd, timeout=600)

        if code == 0:
            st.success("Completed.")
        else:
            st.error(f"Command failed (exit {code}).")

        st.markdown("#### Output")
        st.code(out or "(no output)")
