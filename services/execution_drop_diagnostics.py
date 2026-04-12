# services/execution_drop_diagnostics.py
"""Best-effort execution drop diagnostics (JSON + CSV + append log). Never raises to callers."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
DROP_JSON = RESULTS / "execution_drop_diagnostics.json"
DROP_CSV = RESULTS / "execution_drop_diagnostics.csv"
DROP_LOG_CSV = RESULTS / "execution_drop_diagnostics_log.csv"
DROP_LOG_FIELDS = [
    "ts_utc",
    "ok",
    "blocked",
    "planned_orders",
    "submitted_orders",
    "in_flight_orders",
    "dropped_orders",
    "top_reasons",
]

CSV_FIELDS = [
    "timestamp",
    "run_mode",
    "ticker",
    "symbol",
    "stance",
    "opportunity_type",
    "confidence",
    "delta_pct",
    "planned_qty",
    "planned_notional",
    "phase",
    "status",
    "reason_code",
    "reason_detail",
    "source",
    "session",
    "client_order_id",
]

DROP_SUMMARY_JSON = RESULTS / "execution_drop_summary.json"


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_row(
    *,
    run_mode: str,
    symbol: str,
    stance: str = "",
    planned_qty: Any = "",
    planned_notional: Any = "",
    phase: str,
    status: str,
    reason_code: str,
    reason_detail: str = "",
    source: str,
    session: str = "",
    client_order_id: str = "",
    timestamp: Optional[str] = None,
    ticker: str = "",
    opportunity_type: str = "",
    confidence: Any = "",
    delta_pct: Any = "",
) -> Dict[str, Any]:
    sym_u = str(symbol or "").strip().upper()
    return {
        "timestamp": timestamp or utc_iso(),
        "run_mode": run_mode,
        "ticker": str(ticker or sym_u).strip().upper(),
        "symbol": sym_u,
        "stance": str(stance or ""),
        "opportunity_type": str(opportunity_type or ""),
        "confidence": confidence if confidence != "" else "",
        "delta_pct": delta_pct if delta_pct != "" else "",
        "planned_qty": planned_qty if planned_qty != "" else "",
        "planned_notional": planned_notional if planned_notional != "" else "",
        "phase": phase,
        "status": status,
        "reason_code": reason_code,
        "reason_detail": (reason_detail or "")[:8000],
        "source": source,
        "session": session or "",
        "client_order_id": client_order_id or "",
    }


def read_json(path: Path = DROP_JSON) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        o = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def aggregate_drop_reasons(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        rc = str(r.get("reason_code") or "").strip()
        st = str(r.get("status") or "").lower()
        if rc and st in ("dropped", "blocked", "skipped"):
            c[rc] += 1
    return dict(c)


def write_drop_summary_json(rows: List[Dict[str, Any]], path: Path = DROP_SUMMARY_JSON) -> None:
    """Per-reason counts for dropped/blocked/skipped rows (diagnostic reason_code)."""
    c: Counter[str] = Counter()
    c_preflight: Counter[str] = Counter()
    for r in rows:
        if not isinstance(r, dict):
            continue
        st = str(r.get("status") or "").lower()
        rc = str(r.get("reason_code") or "").strip()
        if st == "satisfied_in_flight" and rc and rc not in ("KEPT", "WRITTEN_TO_ORDERS_TODAY"):
            c_preflight[rc] += 1
            continue
        if st not in ("dropped", "blocked", "skipped"):
            continue
        if not rc or rc in ("KEPT", "WRITTEN_TO_ORDERS_TODAY"):
            continue
        c[rc] += 1
    payload = {
        "timestamp": utc_iso(),
        "total_dropped": int(sum(c.values())),
        "reason_counts": dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))),
        "preflight_in_flight_satisfied": dict(
            sorted(c_preflight.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def build_summary_payload(
    *,
    mode: str,
    rows: List[Dict[str, Any]],
    blocked: bool = False,
    source_hint: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    planned_like = sum(1 for r in rows if str(r.get("status", "")).lower() in ("planned", "kept"))
    submitted = sum(1 for r in rows if str(r.get("status", "")).lower() == "submitted")
    in_flight_orders = sum(
        1
        for r in rows
        if str(r.get("reason_code", "")).strip() == "IN_FLIGHT_ORDER"
        and str(r.get("status", "")).lower() == "kept"
    )
    dropped = sum(
        1 for r in rows if str(r.get("status", "")).lower() in ("dropped", "blocked", "skipped")
    )
    payload: Dict[str, Any] = {
        "timestamp": utc_iso(),
        "mode": mode,
        "planned_orders": planned_like,
        "submitted_orders": submitted,
        "in_flight_orders": in_flight_orders,
        "dropped_orders": dropped,
        "blocked": blocked,
        "drop_reasons": aggregate_drop_reasons(rows),
        "rows": rows,
    }
    if source_hint:
        payload["source"] = source_hint
    if extra:
        payload.update(extra)
    return payload


def write_json(payload: Dict[str, Any], path: Path = DROP_JSON) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def write_csv_rows(rows: List[Dict[str, Any]], path: Path = DROP_CSV) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in CSV_FIELDS}
                w.writerow(row)
    except Exception:
        pass


def merge_rows_from_file(
    existing_rows: List[Dict[str, Any]],
    path: Path = DROP_JSON,
) -> List[Dict[str, Any]]:
    j = read_json(path)
    if not j:
        return list(existing_rows)
    prior = j.get("rows")
    if not isinstance(prior, list):
        return list(existing_rows)
    # Prefer execute_trades planning rows first, then caller appends placement
    out = [r for r in prior if isinstance(r, dict)]
    out.extend(existing_rows)
    return out


def append_log_run(summary: Dict[str, Any], path: Path = DROP_LOG_CSV) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts_utc": summary.get("timestamp") or utc_iso(),
            "ok": str(not summary.get("blocked", False)).lower(),
            "blocked": str(bool(summary.get("blocked"))).lower(),
            "planned_orders": summary.get("planned_orders", ""),
            "submitted_orders": summary.get("submitted_orders", ""),
            "in_flight_orders": summary.get("in_flight_orders", ""),
            "dropped_orders": summary.get("dropped_orders", ""),
            "top_reasons": json.dumps(summary.get("drop_reasons") or {}, sort_keys=True),
        }
        new_file = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DROP_LOG_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in DROP_LOG_FIELDS})
    except Exception:
        pass


def finalize_artifacts(
    payload: Dict[str, Any],
    *,
    write_log: bool = True,
) -> None:
    write_json(payload, DROP_JSON)
    rows = payload.get("rows")
    if isinstance(rows, list):
        write_csv_rows([r for r in rows if isinstance(r, dict)])
        write_drop_summary_json([r for r in rows if isinstance(r, dict)])
    if write_log:
        append_log_run(payload)


def recompute_summary_counts(
    rows: List[Dict[str, Any]], mode: str, blocked: bool
) -> Dict[str, Any]:
    """Rebuild top-level counters from rows (used after merging placement rows)."""
    p = build_summary_payload(mode=mode, rows=rows, blocked=blocked)
    p["rows"] = rows
    return p
