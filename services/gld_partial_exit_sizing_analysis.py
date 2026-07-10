"""
gld_partial_exit_sizing_analysis.py
Phase 198C — Analysis-only GLD partial exit sizing under notional caps.

Does not submit broker orders, create approvals, or authorize execution.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
POSITIONS_CSV = RESULTS / "positions_snapshot.csv"
READINESS_JSON = RESULTS / "execution_readiness.json"
MANAGE_CFG = ROOT / "config" / "manage_positions.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OUTPUT_JSON = RESULTS / "gld_partial_exit_sizing_analysis.json"
OUTPUT_CSV = RESULTS / "gld_partial_exit_sizing_candidates.csv"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

DEFAULT_CONFIGURED_CAP = 50000.0
DEFAULT_MIN_ORDER_NOTIONAL = 50.0
SYMBOL = "GLD"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _load_configured_cap() -> Tuple[float, float]:
    cap = DEFAULT_CONFIGURED_CAP
    min_notional = DEFAULT_MIN_ORDER_NOTIONAL
    cfg = _read_json(MANAGE_CFG)
    if cfg:
        cap = _safe_float(cfg.get("max_batch_notional"), cap)
        min_notional = _safe_float(cfg.get("min_order_notional"), min_notional)
    return cap, min_notional


def _load_gld_position() -> Dict[str, Any]:
    if not POSITIONS_CSV.is_file():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(POSITIONS_CSV, low_memory=False)
    except Exception:
        return {}
    if df.empty:
        return {}
    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if not sym_col:
        return {}
    mask = df[sym_col].astype(str).str.strip().str.upper() == SYMBOL
    if not mask.any():
        return {}
    row = df.loc[mask].iloc[0]
    qty = _safe_float(row.get("qty") or row.get("qty_available"))
    market_value = _safe_float(row.get("market_value") or row.get("value"))
    current_price = _safe_float(row.get("current_price"))
    if current_price <= 0 and qty > 0 and market_value > 0:
        current_price = market_value / qty
    unrealized_pl = _safe_float(row.get("unrealized_pl"))
    unrealized_plpc = _safe_float(row.get("unrealized_plpc"))
    avg_entry = _safe_float(row.get("avg_entry_price"))

    total_value = 0.0
    for _, r in df.iterrows():
        total_value += abs(_safe_float(r.get("market_value") or r.get("value")))
    concentration_pct = (market_value / total_value * 100.0) if total_value > 0 else None

    return {
        "symbol": SYMBOL,
        "quantity": qty,
        "market_value": round(market_value, 4),
        "current_price": round(current_price, 6),
        "avg_entry_price": round(avg_entry, 6),
        "unrealized_pl": round(unrealized_pl, 4),
        "unrealized_pl_pct": round(unrealized_plpc * 100.0, 4),
        "portfolio_total_value": round(total_value, 4),
        "concentration_share_pct": (
            round(concentration_pct, 4) if concentration_pct is not None else None
        ),
        "snapshot_ts": str(row.get("snapshot_ts") or ""),
    }


def _estimate_limit_price(position: Dict[str, Any]) -> float:
    """Prefer capped-preview limit if present; else current price with sell buffer."""
    preview = _read_json(PREVIEW_JSON)
    for block_key in ("capped_preview_runtime_cap", "capped_preview", "capped_preview_cli_cap"):
        block = preview.get(block_key) or {}
        for order in list(block.get("deferred_orders") or []) + list(
            block.get("capped_preview_orders") or []
        ):
            if str(order.get("symbol") or "").upper() == SYMBOL:
                px = _safe_float(order.get("limit_price"))
                if px > 0:
                    return round(px, 4)
    cfg = _read_json(MANAGE_CFG)
    bps = _safe_float(cfg.get("limit_price_buffer_sell_bps"), 20.0)
    px = _safe_float(position.get("current_price"))
    if px <= 0:
        return 0.0
    return round(px * (1.0 - bps / 10000.0), 4)


def _classify_scenario(
    *,
    scenario_name: str,
    sell_qty: float,
    full_qty: float,
    estimated_notional: float,
    cap: float,
    min_notional: float,
    cap_pass: bool,
    is_full_exit: bool,
) -> str:
    if is_full_exit and not cap_pass:
        return "FULL_EXIT_EXCEEDS_CAP"
    if not cap_pass or sell_qty <= 0:
        return "BLOCKED_BY_CAP"
    if estimated_notional < min_notional:
        return "TOO_SMALL"
    frac = (sell_qty / full_qty) if full_qty > 0 else 0.0
    if frac < 0.08 or estimated_notional < max(min_notional * 2.0, 500.0):
        return "TOO_SMALL"
    if frac > 0.25:
        return "AGGRESSIVE_PARTIAL"
    return "REASONABLE_PARTIAL"


def _build_scenario(
    *,
    scenario_name: str,
    sell_qty: float,
    full_qty: float,
    limit_price: float,
    market_value: float,
    concentration_pct: Optional[float],
    unrealized_pl: float,
    cap: float,
    min_notional: float,
    is_full_exit: bool = False,
) -> Dict[str, Any]:
    sell_qty = max(0.0, min(float(sell_qty), float(full_qty)))
    # Whole shares for ETF/stock analysis display (GLD trades in whole shares here).
    if sell_qty > 0 and abs(sell_qty - round(sell_qty)) < 1e-9:
        sell_qty = float(int(round(sell_qty)))
    elif sell_qty > 0:
        sell_qty = float(math.floor(sell_qty))

    estimated_notional = round(sell_qty * limit_price, 2) if sell_qty > 0 else 0.0
    remaining_qty = round(full_qty - sell_qty, 6)
    remaining_notional = round(remaining_qty * limit_price, 2)
    cap_used_pct = round((estimated_notional / cap * 100.0), 2) if cap > 0 else None
    cap_pass = bool(sell_qty > 0 and estimated_notional <= cap + 1e-6)

    sold_value_share = (estimated_notional / market_value) if market_value > 0 else 0.0
    concentration_reduction = (
        round(concentration_pct * sold_value_share, 4) if concentration_pct is not None else None
    )
    # Approximate drawdown crystallization: fraction of unrealized loss realized if sold.
    drawdown_reduction = (
        round(abs(unrealized_pl) * sold_value_share, 2) if unrealized_pl < 0 else 0.0
    )

    classification = _classify_scenario(
        scenario_name=scenario_name,
        sell_qty=sell_qty,
        full_qty=full_qty,
        estimated_notional=estimated_notional,
        cap=cap,
        min_notional=min_notional,
        cap_pass=cap_pass,
        is_full_exit=is_full_exit,
    )

    return {
        "scenario_name": scenario_name,
        "sell_qty": sell_qty,
        "estimated_notional": estimated_notional,
        "remaining_qty": remaining_qty,
        "remaining_notional": remaining_notional,
        "cap": cap,
        "cap_used_pct": cap_used_pct,
        "cap_pass": cap_pass,
        "concentration_reduction_estimate_pct_pts": concentration_reduction,
        "drawdown_reduction_estimate_usd": drawdown_reduction,
        "classification": classification,
        "paper_only": True,
        "broker_orders_submitted": False,
        "is_full_exit": is_full_exit,
        "reduction_fraction": round(sell_qty / full_qty, 6) if full_qty > 0 else 0.0,
    }


def _max_qty_under_cap(full_qty: float, limit_price: float, cap: float) -> float:
    if limit_price <= 0 or cap <= 0:
        return 0.0
    return float(min(full_qty, math.floor(cap / limit_price)))


def build_scenarios(
    position: Dict[str, Any],
    *,
    limit_price: float,
    configured_cap: float,
    min_notional: float,
) -> List[Dict[str, Any]]:
    full_qty = _safe_float(position.get("quantity"))
    market_value = _safe_float(position.get("market_value"))
    concentration = position.get("concentration_share_pct")
    unrealized_pl = _safe_float(position.get("unrealized_pl"))

    scenarios: List[Dict[str, Any]] = []

    # Full exit against configured cap
    scenarios.append(
        _build_scenario(
            scenario_name="FULL_EXIT_CONFIGURED_CAP",
            sell_qty=full_qty,
            full_qty=full_qty,
            limit_price=limit_price,
            market_value=market_value,
            concentration_pct=concentration,
            unrealized_pl=unrealized_pl,
            cap=configured_cap,
            min_notional=min_notional,
            is_full_exit=True,
        )
    )

    for pct, name in (
        (0.10, "PARTIAL_10PCT"),
        (0.15, "PARTIAL_15PCT"),
        (0.20, "PARTIAL_20PCT"),
        (0.25, "PARTIAL_25PCT"),
    ):
        scenarios.append(
            _build_scenario(
                scenario_name=name,
                sell_qty=math.floor(full_qty * pct),
                full_qty=full_qty,
                limit_price=limit_price,
                market_value=market_value,
                concentration_pct=concentration,
                unrealized_pl=unrealized_pl,
                cap=configured_cap,
                min_notional=min_notional,
            )
        )

    for cap, name in (
        (configured_cap, "MAX_QTY_UNDER_CONFIGURED_CAP"),
        (25000.0, "MAX_QTY_UNDER_25000_CAP"),
        (10000.0, "MAX_QTY_UNDER_10000_CAP"),
    ):
        scenarios.append(
            _build_scenario(
                scenario_name=name,
                sell_qty=_max_qty_under_cap(full_qty, limit_price, cap),
                full_qty=full_qty,
                limit_price=limit_price,
                market_value=market_value,
                concentration_pct=concentration,
                unrealized_pl=unrealized_pl,
                cap=cap,
                min_notional=min_notional,
            )
        )

    return scenarios


def _recommend(scenarios: List[Dict[str, Any]], configured_cap: float) -> Dict[str, Any]:
    feasible = [
        s
        for s in scenarios
        if s.get("cap_pass")
        and s.get("classification") in ("REASONABLE_PARTIAL", "AGGRESSIVE_PARTIAL")
        and float(s.get("cap") or 0) == float(configured_cap)
    ]
    # Prefer REASONABLE over AGGRESSIVE; among equals prefer higher notional under cap.
    feasible.sort(
        key=lambda s: (
            0 if s.get("classification") == "REASONABLE_PARTIAL" else 1,
            -float(s.get("estimated_notional") or 0),
        )
    )
    if feasible:
        top = feasible[0]
        return {
            "recommended_scenario": top.get("scenario_name"),
            "classification": top.get("classification"),
            "sell_qty": top.get("sell_qty"),
            "estimated_notional": top.get("estimated_notional"),
            "rationale": (
                "Prefer a configured-cap-fitting partial GLD reduction that is large enough "
                "to matter for concentration without requiring a full exit that exceeds the cap."
            ),
            "next_review_option": (
                f"Review {top.get('scenario_name')} for a future paper-only operator packet; "
                "do not execute from this analysis phase."
            ),
        }

    max_under = next(
        (s for s in scenarios if s.get("scenario_name") == "MAX_QTY_UNDER_CONFIGURED_CAP"),
        None,
    )
    if max_under and max_under.get("cap_pass"):
        return {
            "recommended_scenario": max_under.get("scenario_name"),
            "classification": max_under.get("classification"),
            "sell_qty": max_under.get("sell_qty"),
            "estimated_notional": max_under.get("estimated_notional"),
            "rationale": "No percentage band preferred; max qty under configured cap is the review candidate.",
            "next_review_option": (
                f"Review {max_under.get('scenario_name')} for a future paper-only operator packet; "
                "do not execute from this analysis phase."
            ),
        }

    return {
        "recommended_scenario": None,
        "classification": "BLOCKED_BY_CAP",
        "sell_qty": 0,
        "estimated_notional": 0,
        "rationale": "No feasible partial GLD scenario fits the configured notional cap.",
        "next_review_option": "Keep GLD deferred until a partial-sizing policy or higher cap is approved.",
    }


def _safety_proof(*, approval_existed_before: bool, auth_existed_before: bool) -> Dict[str, Any]:
    readiness = _read_json(READINESS_JSON)
    live_permitted = bool(readiness.get("live_execution_permitted", False))
    approval_now = APPROVAL_PATH.is_file()
    auth_now = BROKER_AUTH_PATH.is_file()
    return {
        "analysis_only": True,
        "broker_orders_submitted": False,
        "approval_artifact_created_by_this_phase": False,
        "broker_authorization_artifact_created_by_this_phase": False,
        "approval_artifact_present_before": approval_existed_before,
        "approval_artifact_present_after": approval_now,
        "broker_auth_artifact_present_before": auth_existed_before,
        "broker_auth_artifact_present_after": auth_now,
        "live_execution_permitted": live_permitted,
        "live_execution_remains_blocked": not live_permitted,
        "safety_statement": (
            "ANALYSIS_ONLY_NO_EXECUTION — this module does not submit broker orders, "
            "create operator approval artifacts, or create broker authorization artifacts."
        ),
    }


def _write_candidates_csv(scenarios: List[Dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_name",
        "sell_qty",
        "estimated_notional",
        "remaining_qty",
        "remaining_notional",
        "cap",
        "cap_used_pct",
        "cap_pass",
        "concentration_reduction_estimate_pct_pts",
        "drawdown_reduction_estimate_usd",
        "classification",
        "paper_only",
        "broker_orders_submitted",
        "is_full_exit",
        "reduction_fraction",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scenarios:
            writer.writerow({k: s.get(k) for k in fieldnames})


def run_gld_partial_exit_sizing_analysis(*, verbose: bool = True) -> Dict[str, Any]:
    approval_before = APPROVAL_PATH.is_file()
    auth_before = BROKER_AUTH_PATH.is_file()

    configured_cap, min_notional = _load_configured_cap()
    position = _load_gld_position()
    if not position or _safe_float(position.get("quantity")) <= 0:
        payload = {
            "generated_at": _utc_now_iso(),
            "phase": "198C",
            "status": "FAIL",
            "symbol": SYMBOL,
            "error": "GLD position not found in positions_snapshot.csv",
            "broker_orders_submitted": False,
            "safety": _safety_proof(
                approval_existed_before=approval_before, auth_existed_before=auth_before
            ),
        }
        _atomic_write_json(OUTPUT_JSON, payload)
        return payload

    limit_price = _estimate_limit_price(position)
    full_notional = round(_safe_float(position.get("quantity")) * limit_price, 2)
    full_fits_cap = full_notional <= configured_cap + 1e-6

    scenarios = build_scenarios(
        position,
        limit_price=limit_price,
        configured_cap=configured_cap,
        min_notional=min_notional,
    )
    recommendation = _recommend(scenarios, configured_cap)
    safety = _safety_proof(approval_existed_before=approval_before, auth_existed_before=auth_before)

    top_feasible = [
        s
        for s in scenarios
        if s.get("cap_pass")
        and s.get("classification") in ("REASONABLE_PARTIAL", "AGGRESSIVE_PARTIAL")
    ]
    top_feasible.sort(key=lambda s: -float(s.get("estimated_notional") or 0))

    payload: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "phase": "198C",
        "status": "PASS",
        "mode": "ANALYSIS_ONLY",
        "symbol": SYMBOL,
        "current_position": position,
        "full_exit": {
            "quantity": position.get("quantity"),
            "estimated_limit_price": limit_price,
            "estimated_notional": full_notional,
            "configured_cap": configured_cap,
            "fits_configured_cap": full_fits_cap,
            "cap_excess_amount": (
                round(max(0.0, full_notional - configured_cap), 2) if not full_fits_cap else 0.0
            ),
            "classification": (
                "FULL_EXIT_EXCEEDS_CAP" if not full_fits_cap else "REASONABLE_PARTIAL"
            ),
        },
        "configured_cap": configured_cap,
        "min_order_notional": min_notional,
        "scenarios": scenarios,
        "top_feasible_partial_scenarios": top_feasible[:3],
        "recommendation": recommendation,
        "safety": safety,
        "outputs": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)).replace("\\", "/"),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)).replace("\\", "/"),
        },
        "disclaimer": (
            "ANALYSIS_ONLY_NO_EXECUTION. Does not place orders, create approvals, "
            "or authorize broker execution."
        ),
    }

    _write_candidates_csv(scenarios)
    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(f"[198C-GLD-PARTIAL] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198C-GLD-PARTIAL] wrote {OUTPUT_CSV.relative_to(ROOT)}")
        print(
            f"[198C-GLD-PARTIAL] qty={position.get('quantity')} "
            f"full_notional={full_notional} cap={configured_cap} "
            f"full_fits={full_fits_cap} recommended={recommendation.get('recommended_scenario')}"
        )
        print("[198C-GLD-PARTIAL] ANALYSIS_ONLY_NO_EXECUTION")
    return payload


def main() -> int:
    result = run_gld_partial_exit_sizing_analysis(verbose=True)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
