# services/execution_pressure_diagnostics.py
"""Read-only execution funnel: lifecycle intent → opportunities → planned orders → fills. Never raises to callers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = DATA / "results"
LIVE = DATA / "live"

OUT_JSON = RESULTS / "execution_pressure.json"

EFFECTIVE_PATH = RESULTS / "signal_lifecycle_effective.csv"
OPPS_PATH = RESULTS / "trade_opportunities.csv"
ORDERS_TODAY_PATH = LIVE / "orders_today.csv"
LIVE_ORDERS_LOG_PATH = RESULTS / "live_orders_log.csv"
EXEC_DROP_JSON = RESULTS / "execution_drop_diagnostics.json"
EXEC_DROP_CSV = RESULTS / "execution_drop_diagnostics.csv"
EXEC_DROP_SUMMARY_JSON = RESULTS / "execution_drop_summary.json"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def _lifecycle_actionable_count(df: pd.DataFrame) -> int:
    if "effective_stance" not in df.columns:
        return 0
    s = df["effective_stance"].fillna("").astype(str).str.strip().str.upper()
    return int(s.isin(["BUY", "ADD", "TRIM", "EXIT"]).sum())


def _orders_executed_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    n = 0
    if "filled_qty" in df.columns:
        fq = pd.to_numeric(df["filled_qty"], errors="coerce").fillna(0)
        n = max(n, int((fq > 0).sum()))
    st_col = None
    for c in ("status", "order_status", "alpaca_status"):
        if c in df.columns:
            st_col = c
            break
    if st_col:
        s = df[st_col].fillna("").astype(str).str.strip().str.lower()
        filled_like = s.isin(
            ["filled", "partially_filled", "partial", "done", "closed", "complete", "completed"]
        ) | (
            s.str.contains("fill", na=False) & ~s.str.contains("cancel|reject|fail|error", na=False)
        )
        n = max(n, int(filled_like.sum()))
    return n


def _read_drop_summary_reason_counts(path: Path) -> Optional[Dict[str, int]]:
    """Prefer execution_drop_summary.json (canonical diagnostic reason_code counts)."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        o = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(o, dict):
            return None
        rc = o.get("reason_counts")
        if not isinstance(rc, dict) or not rc:
            return None
        out: Dict[str, int] = {}
        for k, v in rc.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                out[key] = int(v)
            except Exception:
                out[key] = out.get(key, 0) + 1
        return out if out else None
    except Exception:
        return None


def _aggregate_block_reasons_raw_from_csv(df: Optional[pd.DataFrame]) -> Dict[str, int]:
    """Count reason_code as-is for dropped/blocked/skipped rows (no UNKNOWN bucketing)."""
    out: Dict[str, int] = {}
    if df is None or df.empty or "reason_code" not in df.columns:
        return out
    st = (
        df["status"].fillna("").astype(str).str.lower()
        if "status" in df.columns
        else pd.Series([""] * len(df))
    )
    mask = (
        st.isin(["dropped", "blocked", "skipped"])
        if "status" in df.columns
        else pd.Series([True] * len(df))
    )
    sub = df.loc[mask]
    skip_codes = frozenset({"KEPT", "WRITTEN_TO_ORDERS_TODAY", ""})
    for rc in sub["reason_code"].fillna("").astype(str):
        r = rc.strip()
        if not r or r in skip_codes:
            continue
        out[r] = out.get(r, 0) + 1
    return out


def _merge_drop_json_reasons(
    j: Optional[Dict[str, Any]], block_counts: Dict[str, int]
) -> Dict[str, int]:
    dr = j.get("drop_reasons") if isinstance(j, dict) else None
    if isinstance(dr, dict):
        for k, v in dr.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                block_counts[key] = block_counts.get(key, 0) + int(v)
            except Exception:
                block_counts[key] = block_counts.get(key, 0) + 1
    return block_counts


def refresh_execution_pressure_diagnostics() -> None:
    """Write data/results/execution_pressure.json. Best-effort."""
    notes: List[str] = []
    lifecycle_actionable = 0
    opportunities_created = 0
    orders_planned = 0
    orders_executed = 0
    blocked_orders = 0

    eff = _safe_read_csv(EFFECTIVE_PATH)
    lifecycle_intent_actionable = 0
    if eff is None:
        notes.append("signal_lifecycle_effective.csv missing or empty")
    else:
        lifecycle_actionable = _lifecycle_actionable_count(eff)
        if "lifecycle_action" in eff.columns:
            la = eff["lifecycle_action"].fillna("").astype(str).str.strip().str.upper()
            lifecycle_intent_actionable = int(la.isin(["BUY", "ADD", "TRIM", "EXIT"]).sum())
        elif "stance" in eff.columns:
            la = eff["stance"].fillna("").astype(str).str.strip().str.upper()
            lifecycle_intent_actionable = int(la.isin(["BUY", "ADD", "TRIM", "EXIT"]).sum())

    opps = _safe_read_csv(OPPS_PATH)
    if opps is not None:
        opportunities_created = len(opps)

    ot = _safe_read_csv(ORDERS_TODAY_PATH)
    if ot is not None:
        orders_planned = len(ot)
    else:
        notes.append("data/live/orders_today.csv missing or empty")

    log_df = _safe_read_csv(LIVE_ORDERS_LOG_PATH)
    orders_executed = _orders_executed_count(log_df) if log_df is not None else 0
    if log_df is None:
        notes.append("live_orders_log.csv missing or empty")

    if lifecycle_actionable == 0 and opportunities_created > 0 and lifecycle_intent_actionable > 0:
        notes.append(
            "effective_stance has no BUY/ADD/TRIM/EXIT after reconciliation; "
            f"lifecycle_action BUY/ADD/TRIM/EXIT count={lifecycle_intent_actionable}. "
            "trade_opportunities uses lifecycle_action + effective_position_state (see build_trade_opportunities)."
        )

    drop_j: Optional[Dict[str, Any]] = None
    try:
        if EXEC_DROP_JSON.is_file() and EXEC_DROP_JSON.stat().st_size > 0:
            drop_j = json.loads(EXEC_DROP_JSON.read_text(encoding="utf-8", errors="replace"))
            if isinstance(drop_j, dict):
                blocked_orders = int(drop_j.get("dropped_orders") or 0)
    except Exception:
        pass

    drop_csv = _safe_read_csv(EXEC_DROP_CSV)
    block_reasons: Dict[str, int] = {}
    summary_rc = _read_drop_summary_reason_counts(EXEC_DROP_SUMMARY_JSON)
    if summary_rc:
        block_reasons = dict(summary_rc)
    else:
        block_reasons = _aggregate_block_reasons_raw_from_csv(drop_csv)
        if sum(block_reasons.values()) == 0 and drop_j:
            block_reasons = _merge_drop_json_reasons(drop_j, block_reasons)

    drop_off = {
        "lifecycle_to_opportunity": lifecycle_actionable - opportunities_created,
        "opportunity_to_orders": opportunities_created - orders_planned,
        "orders_to_execution": orders_planned - orders_executed,
    }

    payload: Dict[str, Any] = {
        "timestamp": _utc_iso(),
        "lifecycle_actionable": lifecycle_actionable,
        "lifecycle_intent_actionable": lifecycle_intent_actionable,
        "opportunities_created": opportunities_created,
        "orders_planned": orders_planned,
        "orders_executed": orders_executed,
        "blocked_orders": blocked_orders,
        "block_reasons": {k: block_reasons[k] for k in sorted(block_reasons.keys())},
        "drop_off": drop_off,
        "notes": notes,
    }

    try:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    refresh_execution_pressure_diagnostics()
    try:
        from services.session_fill_pressure import refresh_session_fill_pressure

        refresh_session_fill_pressure()
    except Exception:
        pass
    try:
        print(OUT_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
