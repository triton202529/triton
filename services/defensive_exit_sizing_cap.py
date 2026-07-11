"""
defensive_exit_sizing_cap.py
Phase 197H — Batch notional estimation and governed capped defensive-exit preview.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
PLAN_CSV = RESULTS / "manage_positions_plan.csv"
HANDOFF_CSV = LIVE / "manage_orders.csv"
CONFIG_PATH = ROOT / "config" / "manage_positions.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"

PLACE_LIVE_ORDERS_DEFAULT_BATCH_CAP = 5000.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _safe_int(v: Any) -> int:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return 0
        return int(float(v))
    except Exception:
        return 0


def load_manage_positions_caps() -> Dict[str, Any]:
    caps = {
        "max_batch_notional": 50000.0,
        "max_management_orders_per_run": 10,
        "min_order_notional": 50.0,
        "partial_defensive_batch_forbidden": False,
    }
    try:
        if CONFIG_PATH.is_file():
            doc = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            caps["max_batch_notional"] = float(
                doc.get("max_batch_notional", caps["max_batch_notional"])
            )
            caps["max_management_orders_per_run"] = int(
                doc.get("max_management_orders_per_run", caps["max_management_orders_per_run"])
            )
            caps["min_order_notional"] = float(
                doc.get("min_order_notional", caps["min_order_notional"])
            )
            caps["partial_defensive_batch_forbidden"] = bool(
                doc.get("partial_defensive_batch_forbidden", False)
            )
    except Exception:
        pass
    return caps


def _load_plan_priority() -> Dict[str, Dict[str, Any]]:
    if not PLAN_CSV.is_file():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(PLAN_CSV, low_memory=False)
        out: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            if str(row.get("status") or "").strip().lower() != "planned":
                continue
            sym = str(row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            out[sym] = {
                "rank": _safe_int(row.get("rank")) or 9999,
                "reason_code": str(row.get("reason_code") or ""),
                "planned_notional": _safe_float(row.get("planned_notional")),
                "priority_bucket": str(row.get("priority_bucket") or ""),
                "score": _safe_float(row.get("score")),
            }
        return out
    except Exception:
        return {}


def load_defensive_exit_handoff_orders() -> List[Dict[str, Any]]:
    try:
        from services.defensive_exit_preview_handoff import (
            load_defensive_handoff_orders_for_analysis,
        )

        orders, _source = load_defensive_handoff_orders_for_analysis()
        if orders:
            return orders
    except Exception:
        pass
    if not HANDOFF_CSV.is_file():
        return []
    try:
        import pandas as pd

        df = pd.read_csv(HANDOFF_CSV, low_memory=False)
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            sym = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            opp = (
                str(row.get("opportunity_type") or row.get("effective_stance") or "")
                .strip()
                .upper()
            )
            if not sym or opp not in ("EXIT", "TRIM", "ROTATE_EXIT", "SELL"):
                continue
            qty = _safe_int(row.get("qty"))
            price = _safe_float(row.get("limit_price") or row.get("close"))
            notional = qty * price if qty > 0 and price > 0 else 0.0
            rows.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "limit_price": price,
                    "notional": round(notional, 2),
                    "opportunity_type": opp,
                    "effective_stance": str(row.get("effective_stance") or ""),
                    "discipline_allowed": row.get("discipline_allowed"),
                }
            )
        return rows
    except Exception:
        return []


def estimate_batch_notional(orders: List[Dict[str, Any]]) -> float:
    return round(sum(_safe_float(o.get("notional")) for o in orders), 2)


def parse_batch_notional_block(output: str) -> Dict[str, Any]:
    text = output or ""
    parsed: Dict[str, Any] = {
        "batch_notional_block": "BATCH_NOTIONAL_BLOCK" in text,
        "batch_notional_estimate": None,
        "configured_batch_cap": None,
        "cap_excess_amount": None,
    }
    m = re.search(
        r"\[BATCH_NOTIONAL_BLOCK\]\s+est_batch_notional=([\d.]+).*?exceeds max ([\d.]+)",
        text,
    )
    if m:
        est = float(m.group(1))
        cap = float(m.group(2).rstrip("."))
        parsed["batch_notional_estimate"] = est
        parsed["configured_batch_cap"] = cap
        parsed["cap_excess_amount"] = round(max(0.0, est - cap), 2)
    return parsed


def build_capped_batch_preview(
    orders: List[Dict[str, Any]],
    *,
    batch_cap: float,
    max_orders_per_batch: Optional[int] = None,
) -> Dict[str, Any]:
    priority = _load_plan_priority()
    ranked = sorted(
        orders,
        key=lambda o: (
            priority.get(o["symbol"], {}).get("rank", 9999),
            -_safe_float(o.get("notional")),
        ),
    )
    capped: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    running = 0.0
    for order in ranked:
        sym = order["symbol"]
        n = _safe_float(order.get("notional"))
        pri = priority.get(sym, {})
        enriched = {
            **order,
            "rank": pri.get("rank"),
            "reason_code": pri.get("reason_code"),
            "priority_bucket": pri.get("priority_bucket"),
        }
        if max_orders_per_batch is not None and len(capped) >= max_orders_per_batch:
            enriched["defer_reason"] = "max_orders_per_batch"
            deferred.append(enriched)
            continue
        if running + n <= batch_cap + 1e-6:
            capped.append(enriched)
            running += n
        else:
            enriched["defer_reason"] = "batch_notional_cap"
            deferred.append(enriched)
    return {
        "batch_cap": batch_cap,
        "batch_notional_full": estimate_batch_notional(orders),
        "batch_notional_capped_preview": round(running, 2),
        "capped_preview_symbols": [o["symbol"] for o in capped],
        "deferred_symbols": [o["symbol"] for o in deferred],
        "capped_preview_orders": capped,
        "deferred_orders": deferred,
        "priority_order": [o["symbol"] for o in ranked],
        "highest_risk_symbols": [
            o["symbol"]
            for o in sorted(
                ranked,
                key=lambda x: (
                    priority.get(x["symbol"], {}).get("rank", 9999),
                    -_safe_float(x.get("notional")),
                ),
            )[:5]
        ],
        "next_eligible_batch_symbols": [o["symbol"] for o in capped],
        "concentration_risk_note": (
            "First capped batch prioritizes plan rank (high-drag EXIT_HIGH_DRAG_PROTECTION "
            "before lower-priority exits) to reduce drawdown exposure within notional cap."
            if capped
            else "No orders fit within current batch cap."
        ),
    }


def analyze_defensive_exit_sizing(
    *,
    placement_output: str = "",
    use_cli_default_cap: bool = True,
) -> Dict[str, Any]:
    caps = load_manage_positions_caps()
    orders = load_defensive_exit_handoff_orders()
    block = parse_batch_notional_block(placement_output)

    runtime_cap = float(caps["max_batch_notional"])
    cli_cap = PLACE_LIVE_ORDERS_DEFAULT_BATCH_CAP
    effective_cap = cli_cap if use_cli_default_cap else runtime_cap
    if block.get("configured_batch_cap"):
        effective_cap = float(block["configured_batch_cap"])

    full_notional = block.get("batch_notional_estimate") or estimate_batch_notional(orders)
    excess = block.get("cap_excess_amount")
    if excess is None and full_notional > effective_cap:
        excess = round(full_notional - effective_cap, 2)

    partial_forbidden = bool(caps.get("partial_defensive_batch_forbidden"))
    max_orders = int(caps.get("max_management_orders_per_run") or 10)
    partial_authorized = (not partial_forbidden) and max_orders > 1

    placement_block_reason = ""
    handoff_status = "HANDOFF_ROWS_PRESENT"
    if block.get("batch_notional_block"):
        placement_block_reason = "BATCH_NOTIONAL_BLOCK"
        handoff_status = "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED"
    elif full_notional > effective_cap:
        placement_block_reason = "BATCH_NOTIONAL_EXCEEDS_CAP"
        handoff_status = "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED"

    capped_preview = {}
    if partial_authorized and orders:
        capped_preview = build_capped_batch_preview(
            orders,
            batch_cap=effective_cap,
            max_orders_per_batch=max_orders,
        )
        cli_preview = build_capped_batch_preview(
            orders,
            batch_cap=cli_cap,
            max_orders_per_batch=max_orders,
        )
    else:
        capped_preview = {
            "capped_preview_available": False,
            "partial_batch_status": "PARTIAL_DEFENSIVE_BATCH_NOT_AUTHORIZED",
        }
        cli_preview = capped_preview

    runtime_preview = (
        build_capped_batch_preview(
            orders,
            batch_cap=runtime_cap,
            max_orders_per_batch=max_orders,
        )
        if partial_authorized and orders
        else {}
    )

    analysis = {
        "timestamp": _utc_now_iso(),
        "phase": "197H",
        "handoff_rows": len(orders),
        "planned_exit_count": len(orders),
        "handoff_symbols": sorted(o["symbol"] for o in orders),
        "batch_notional_estimate": full_notional,
        "configured_batch_cap_cli_default": cli_cap,
        "configured_batch_cap_manage_positions": runtime_cap,
        "configured_batch_cap_effective": effective_cap,
        "cap_excess_amount": excess,
        "placement_block_reason": placement_block_reason,
        "handoff_status": handoff_status,
        "blocked_symbols": sorted(o["symbol"] for o in orders) if excess and excess > 0 else [],
        "partial_batch_authorized": partial_authorized,
        "partial_batch_status": (
            "PARTIAL_DEFENSIVE_BATCH_NOT_AUTHORIZED"
            if partial_forbidden
            else "PARTIAL_DEFENSIVE_BATCH_AUTHORIZED"
        ),
        "max_management_orders_per_run": max_orders,
        "capped_preview_available": bool(capped_preview.get("capped_preview_symbols")),
        "capped_preview": capped_preview,
        "capped_preview_cli_cap": cli_preview,
        "capped_preview_runtime_cap": runtime_preview,
        "placement_parse": block,
        "broker_orders_submitted": False,
    }
    try:
        from services.partial_defensive_exit_preview import (
            enrich_sizing_analysis_with_partial_preview,
        )

        analysis = enrich_sizing_analysis_with_partial_preview(analysis)
    except Exception:
        analysis.setdefault("partial_preview_available", False)
        analysis.setdefault("partial_preview_analysis_only", True)
    return analysis


def write_capped_batch_preview(analysis: Dict[str, Any]) -> Path:
    PREVIEW_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = PREVIEW_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    import os

    os.replace(tmp, PREVIEW_JSON)
    return PREVIEW_JSON
