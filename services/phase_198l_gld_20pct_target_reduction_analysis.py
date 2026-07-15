"""
phase_198l_gld_20pct_target_reduction_analysis.py
Phase 198L — Analysis-only GLD reduction to <=20% of invested exposure.

Does not submit broker orders, create approvals, or authorize execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
POSITIONS_CSV = RESULTS / "positions_snapshot.csv"
OUTPUT_JSON = RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json"
OUTPUT_MD = RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

SYMBOL = "GLD"
TARGET_WEIGHT = 0.20
MARKER = "GLD_20PCT_TARGET_REDUCTION_ANALYSIS_ONLY"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _load_portfolio_snapshot() -> Dict[str, Any]:
    if not POSITIONS_CSV.is_file():
        return {"ok": False, "reason": "positions_snapshot_missing"}
    try:
        import pandas as pd

        df = pd.read_csv(POSITIONS_CSV, low_memory=False)
    except Exception as exc:
        return {"ok": False, "reason": f"positions_snapshot_read_failed:{exc}"}
    if df.empty:
        return {"ok": False, "reason": "positions_snapshot_empty"}

    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if not sym_col:
        return {"ok": False, "reason": "positions_snapshot_missing_symbol_column"}

    total_invested = 0.0
    for _, row in df.iterrows():
        total_invested += abs(_safe_float(row.get("market_value") or row.get("value")))

    mask = df[sym_col].astype(str).str.strip().str.upper() == SYMBOL
    if not mask.any():
        return {
            "ok": False,
            "reason": "gld_position_absent",
            "total_invested_value": round(total_invested, 2),
        }

    row = df.loc[mask].iloc[0]
    qty = _safe_float(row.get("qty") or row.get("qty_available"))
    market_value = abs(_safe_float(row.get("market_value") or row.get("value")))
    current_price = _safe_float(row.get("current_price"))
    if current_price <= 0 and qty > 0 and market_value > 0:
        current_price = market_value / qty

    return {
        "ok": True,
        "symbol": SYMBOL,
        "current_gld_qty": qty,
        "current_gld_price": round(current_price, 6),
        "current_gld_market_value": round(market_value, 2),
        "total_invested_value": round(total_invested, 2),
        "positions_count": int(len(df)),
        "snapshot_ts": str(row.get("snapshot_ts") or ""),
        "source": _rel(POSITIONS_CSV),
    }


def _compute_20pct_reduction(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve for remaining GLD value at target weight of invested exposure after sale:

      remaining_gld <= w * (other + remaining_gld)
      remaining_gld <= other * w / (1 - w)

    For w=0.20: remaining_gld <= other / 4
    """
    gld_qty = float(snapshot["current_gld_qty"])
    gld_price = float(snapshot["current_gld_price"])
    gld_value = float(snapshot["current_gld_market_value"])
    total_invested = float(snapshot["total_invested_value"])
    other_value = round(total_invested - gld_value, 2)
    current_weight = (gld_value / total_invested) if total_invested > 0 else 0.0

    if TARGET_WEIGHT >= 1.0 or TARGET_WEIGHT <= 0.0:
        raise ValueError(f"invalid_target_weight={TARGET_WEIGHT}")

    max_remaining_gld_value = round(other_value * TARGET_WEIGHT / (1.0 - TARGET_WEIGHT), 2)
    # Equivalent form for w=0.20: other / 4
    max_remaining_gld_value_alt = round(other_value / ((1.0 / TARGET_WEIGHT) - 1.0), 2)
    if abs(max_remaining_gld_value - max_remaining_gld_value_alt) > 0.02:
        max_remaining_gld_value = min(max_remaining_gld_value, max_remaining_gld_value_alt)

    if gld_price <= 0:
        max_remaining_shares_exact = 0.0
        safe_remaining_shares = 0
    else:
        max_remaining_shares_exact = max_remaining_gld_value / gld_price
        # Floor to whole shares so projected weight stays <= target.
        safe_remaining_shares = int(math.floor(max_remaining_shares_exact + 1e-12))
        safe_remaining_shares = max(0, min(safe_remaining_shares, int(math.floor(gld_qty))))

    current_qty_int = int(math.floor(gld_qty + 1e-12))
    recommended_sell_qty = max(0, current_qty_int - safe_remaining_shares)
    # Never recommend full liquidation when a non-zero remaining share count is feasible.
    full_liquidation_avoided = bool(
        safe_remaining_shares > 0 and recommended_sell_qty < current_qty_int
    )

    projected_remaining_qty = safe_remaining_shares
    projected_gld_value = round(projected_remaining_qty * gld_price, 2)
    projected_invested = round(other_value + projected_gld_value, 2)
    projected_weight = (projected_gld_value / projected_invested) if projected_invested > 0 else 0.0
    estimated_sell_notional = round(recommended_sell_qty * gld_price, 2)

    return {
        "target_weight": TARGET_WEIGHT,
        "non_gld_invested_exposure": other_value,
        "current_gld_weight": round(current_weight, 6),
        "current_gld_weight_pct": round(current_weight * 100.0, 4),
        "max_remaining_gld_value": max_remaining_gld_value,
        "max_remaining_gld_shares_exact": round(max_remaining_shares_exact, 6),
        "safe_remaining_gld_shares": safe_remaining_shares,
        "recommended_sell_qty": recommended_sell_qty,
        "projected_remaining_qty": projected_remaining_qty,
        "projected_post_sale_gld_value": projected_gld_value,
        "projected_post_sale_invested_exposure": projected_invested,
        "projected_post_sale_gld_weight": round(projected_weight, 6),
        "projected_post_sale_gld_weight_pct": round(projected_weight * 100.0, 4),
        "estimated_sell_notional": estimated_sell_notional,
        "full_liquidation_avoided": full_liquidation_avoided,
        "formula": {
            "other_position_value": "total_invested_value - current_gld_value",
            "max_remaining_gld_value": "other_position_value * target_weight / (1 - target_weight)",
            "target_weight_0_20_form": "other_position_value / 4",
            "safe_remaining_shares": "floor(max_remaining_gld_value / current_gld_price)",
            "recommended_sell_qty": "current_gld_qty_floor - safe_remaining_shares",
        },
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    calc = payload.get("calculation") or {}
    snap = payload.get("snapshot") or {}
    lines = [
        "# Phase 198L — GLD 20% Target Reduction Analysis",
        "",
        "Analysis only. No broker orders. No approval. No broker authorization.",
        "",
        f"- Generated: `{payload.get('generated_at')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Marker: `{payload.get('marker')}`",
        "",
        "## Current snapshot",
        "",
        f"- GLD qty: **{snap.get('current_gld_qty')}**",
        f"- GLD price: **{snap.get('current_gld_price')}**",
        f"- GLD market value: **{snap.get('current_gld_market_value')}**",
        f"- Total invested exposure: **{snap.get('total_invested_value')}**",
        f"- Non-GLD invested exposure: **{calc.get('non_gld_invested_exposure')}**",
        f"- Current GLD weight: **{calc.get('current_gld_weight_pct')}%**",
        "",
        "## Target math (denominator shrinks when GLD is sold)",
        "",
        "```",
        "other = total_invested - gld_value",
        "max_remaining_gld_value <= other * 0.20 / 0.80 = other / 4",
        "safe_remaining_shares = floor(max_remaining_gld_value / gld_price)",
        "recommended_sell_qty = current_qty - safe_remaining_shares",
        "```",
        "",
        "## Recommendation",
        "",
        f"- Target weight: **{float(calc.get('target_weight') or 0) * 100:.0f}%**",
        f"- Max remaining GLD value: **{calc.get('max_remaining_gld_value')}**",
        f"- Max remaining GLD shares (exact): **{calc.get('max_remaining_gld_shares_exact')}**",
        f"- Safe remaining GLD shares: **{calc.get('safe_remaining_gld_shares')}**",
        f"- Recommended sell qty: **{calc.get('recommended_sell_qty')}**",
        f"- Estimated sell notional: **{calc.get('estimated_sell_notional')}**",
        f"- Projected remaining qty: **{calc.get('projected_remaining_qty')}**",
        f"- Projected post-sale GLD value: **{calc.get('projected_post_sale_gld_value')}**",
        f"- Projected post-sale invested exposure: **{calc.get('projected_post_sale_invested_exposure')}**",
        f"- Projected post-sale GLD weight: **{calc.get('projected_post_sale_gld_weight_pct')}%**",
        f"- Full liquidation avoided: **{calc.get('full_liquidation_avoided')}**",
        "",
        "## Safety posture",
        "",
        f"- paper_only: `{payload.get('paper_only')}`",
        f"- live_execution_allowed: `{payload.get('live_execution_allowed')}`",
        f"- broker_orders_submitted: `{payload.get('broker_orders_submitted')}`",
        f"- approval_required: `{payload.get('approval_required')}`",
        f"- broker_authorization_required: `{payload.get('broker_authorization_required')}`",
        f"- approval_artifact_exists: `{payload.get('approval_artifact_exists')}`",
        f"- broker_authorization_artifact_exists: `{payload.get('broker_authorization_artifact_exists')}`",
        "",
        f"{MARKER}",
        "",
    ]
    return "\n".join(lines)


def run_phase_198l_analysis(*, verbose: bool = True) -> Dict[str, Any]:
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.portfolio_history_sync import load_live_execution_flag

    notes: List[str] = ["analysis_only_no_execution"]
    snapshot = _load_portfolio_snapshot()
    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    live_block = evaluate_live_block_status(paper_mode=True)
    live_flag = bool(load_live_execution_flag())
    open_orders = int(criteria.get("open_orders_count") or 0)
    approval_exists = APPROVAL_PATH.is_file()
    auth_exists = BROKER_AUTH_PATH.is_file()

    status = "PASS"
    block_reasons: List[str] = []

    if not snapshot.get("ok"):
        status = "FAIL"
        block_reasons.append(str(snapshot.get("reason") or "snapshot_unavailable"))

    calculation: Dict[str, Any] = {}
    if status == "PASS":
        try:
            calculation = _compute_20pct_reduction(snapshot)
        except Exception as exc:
            status = "FAIL"
            block_reasons.append(f"calculation_failed:{exc}")

    if status == "PASS":
        projected_weight = float(calculation.get("projected_post_sale_gld_weight") or 1.0)
        sell_qty = int(calculation.get("recommended_sell_qty") or 0)
        remaining = int(calculation.get("projected_remaining_qty") or -1)
        if sell_qty <= 0:
            status = "FAIL"
            block_reasons.append("recommended_sell_qty_not_positive")
        if remaining < 0:
            status = "FAIL"
            block_reasons.append("projected_remaining_qty_invalid")
        if projected_weight > TARGET_WEIGHT + 1e-9:
            status = "FAIL"
            block_reasons.append(f"projected_weight_above_target weight={projected_weight}")
        if not bool(calculation.get("full_liquidation_avoided")):
            # Still PASS if math forces near-full exit, but note it.
            notes.append("full_liquidation_not_avoided_by_math")

    if live_flag or not bool(live_block.get("live_blocked")):
        status = "FAIL"
        block_reasons.append("live_execution_not_blocked")
        notes.append("live_block_regression")

    # This phase must never create authority artifacts.
    if approval_exists:
        notes.append("approval_artifact_present_but_not_created_by_198l")
    if auth_exists:
        notes.append("broker_auth_artifact_present_but_not_created_by_198l")

    payload: Dict[str, Any] = {
        "phase": "198L",
        "generated_at": _utc_now_iso(),
        "status": status,
        "symbol": SYMBOL,
        "target_weight": TARGET_WEIGHT,
        "snapshot": {
            "current_gld_qty": snapshot.get("current_gld_qty"),
            "current_gld_price": snapshot.get("current_gld_price"),
            "current_gld_market_value": snapshot.get("current_gld_market_value"),
            "total_invested_value": snapshot.get("total_invested_value"),
            "positions_count": snapshot.get("positions_count"),
            "snapshot_ts": snapshot.get("snapshot_ts"),
            "source": snapshot.get("source") or _rel(POSITIONS_CSV),
            "ok": bool(snapshot.get("ok")),
            "reason": snapshot.get("reason"),
        },
        "calculation": calculation,
        "current_gld_qty": snapshot.get("current_gld_qty"),
        "current_gld_price": snapshot.get("current_gld_price"),
        "current_gld_market_value": snapshot.get("current_gld_market_value"),
        "total_invested_exposure": snapshot.get("total_invested_value"),
        "non_gld_invested_exposure": calculation.get("non_gld_invested_exposure"),
        "current_gld_weight": calculation.get("current_gld_weight"),
        "max_remaining_gld_value": calculation.get("max_remaining_gld_value"),
        "max_remaining_gld_shares": calculation.get("max_remaining_gld_shares_exact"),
        "safe_remaining_gld_shares": calculation.get("safe_remaining_gld_shares"),
        "recommended_sell_qty": calculation.get("recommended_sell_qty"),
        "projected_remaining_qty": calculation.get("projected_remaining_qty"),
        "projected_post_sale_gld_value": calculation.get("projected_post_sale_gld_value"),
        "projected_post_sale_invested_exposure": calculation.get(
            "projected_post_sale_invested_exposure"
        ),
        "projected_post_sale_gld_weight": calculation.get("projected_post_sale_gld_weight"),
        "full_liquidation_avoided": bool(calculation.get("full_liquidation_avoided")),
        "paper_only": True,
        "live_execution_allowed": False,
        "live_blocked": bool(live_block.get("live_blocked")),
        "live_execution_flag": live_flag,
        "broker_orders_submitted": False,
        "open_orders_count": open_orders,
        "approval_required": True,
        "broker_authorization_required": True,
        "approval_artifact_exists": approval_exists,
        "broker_authorization_artifact_exists": auth_exists,
        "approval_artifact_created": False,
        "broker_authorization_artifact_created": False,
        "block_reasons": block_reasons,
        "notes": notes,
        "marker": MARKER,
        "outputs": {
            "json": _rel(OUTPUT_JSON),
            "markdown": _rel(OUTPUT_MD),
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(f"[198L-GLD-20PCT] wrote {_rel(OUTPUT_JSON)}")
        print(f"[198L-GLD-20PCT] wrote {_rel(OUTPUT_MD)}")
        print(
            f"[198L-GLD-20PCT] status={status} "
            f"current_weight={payload.get('current_gld_weight')} "
            f"sell_qty={payload.get('recommended_sell_qty')} "
            f"remaining={payload.get('projected_remaining_qty')} "
            f"projected_weight={payload.get('projected_post_sale_gld_weight')} "
            f"broker_orders_submitted=False"
        )
        print(f"[198L-GLD-20PCT] {MARKER}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 198L analysis-only: compute GLD sell qty to reach <=20% of "
            "invested exposure after the sale. Does not place orders or create "
            "approval/broker-auth artifacts."
        )
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198l_analysis(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
