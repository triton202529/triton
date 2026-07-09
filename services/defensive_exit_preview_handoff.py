"""
defensive_exit_preview_handoff.py
Phase 198A — Read-only defensive exit preview handoff for ORANGE_DEGRADED governance.

Generates preview rows from planned risk-reducing exits without broker execution.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
PLAN_CSV = RESULTS / "manage_positions_plan.csv"
POSITIONS_CSV = RESULTS / "positions_snapshot.csv"
READINESS_JSON = RESULTS / "execution_readiness.json"
EXEC_HANDOFF_CSV = LIVE / "manage_orders.csv"
PREVIEW_CSV = RESULTS / "defensive_exit_preview_handoff.csv"
PREVIEW_JSON = RESULTS / "defensive_exit_preview_handoff.json"

RISK_REDUCING_ACTIONS = frozenset({"EXIT", "TRIM", "ROTATE_EXIT"})
OPERATIONAL_FAILED_CHECKS = frozenset(
    {"data_freshness", "watchdog", "broker", "lifecycle", "signals"}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_held_symbols() -> Set[str]:
    if not POSITIONS_CSV.is_file():
        return set()
    try:
        import pandas as pd

        df = pd.read_csv(POSITIONS_CSV, low_memory=False)
        sym_col = "symbol" if "symbol" in df.columns else "ticker"
        if sym_col not in df.columns:
            return set()
        out: Set[str] = set()
        for _, row in df.iterrows():
            sym = str(row.get(sym_col) or "").strip().upper()
            if not sym:
                continue
            qty = row.get("qty") or row.get("quantity") or 0
            try:
                if float(qty) != 0:
                    out.add(sym)
            except (TypeError, ValueError):
                out.add(sym)
        return out
    except Exception:
        return set()


def evaluate_defensive_preview_eligibility(
    readiness: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "paper",
) -> Dict[str, Any]:
    """
    Separate preview eligibility from execution eligibility.
    preview_generation_allowed=true does not authorize broker orders.
    """
    readiness = readiness if readiness is not None else _read_json(READINESS_JSON)
    failed = [str(x) for x in (readiness.get("failed_checks") or [])]
    governance_mode = str(readiness.get("governance_mode") or "")
    operational_blocked = bool(set(failed) & OPERATIONAL_FAILED_CHECKS)
    governance_only = failed == ["governance"]

    preview_allowed = (
        str(mode).lower() == "paper"
        and not bool(readiness.get("live_execution_permitted", True))
        and bool(readiness.get("paper_execution_permitted", False))
        and governance_mode == "ORANGE_DEGRADED"
        and bool(readiness.get("risk_reducing_paper_actions_allowed", False))
        and not operational_blocked
        and (governance_only or not failed)
    )

    reason = ""
    if preview_allowed:
        reason = "orange_degraded_risk_reducing_preview"
    elif operational_blocked:
        reason = (
            f"operational_checks_failed={','.join(sorted(set(failed) & OPERATIONAL_FAILED_CHECKS))}"
        )
    elif governance_mode != "ORANGE_DEGRADED":
        reason = f"governance_mode={governance_mode or 'UNKNOWN'}"
    elif not readiness.get("risk_reducing_paper_actions_allowed"):
        reason = "risk_reducing_paper_actions_not_allowed"
    elif not governance_only and failed:
        reason = f"failed_checks={','.join(failed)}"

    return {
        "preview_generation_allowed": preview_allowed,
        "execution_allowed": False,
        "governance_mode": governance_mode,
        "readiness_status": str(readiness.get("readiness_status") or "UNKNOWN"),
        "failed_checks": failed,
        "governance_only_readiness_block": governance_only,
        "reason": reason,
    }


def is_orange_degraded_preview_only_block(readiness: Optional[Dict[str, Any]] = None) -> bool:
    """True when readiness is blocked only by ORANGE_DEGRADED governance (preview path OK)."""
    return bool(evaluate_defensive_preview_eligibility(readiness).get("preview_generation_allowed"))


def build_preview_handoff_rows_from_plan(
    *,
    held_symbols: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Build read-only handoff rows from planned risk-reducing exits for held symbols."""
    held = held_symbols if held_symbols is not None else load_held_symbols()
    if not PLAN_CSV.is_file():
        return []
    try:
        import pandas as pd

        df = pd.read_csv(PLAN_CSV, low_memory=False)
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if str(row.get("status") or "").strip().lower() != "planned":
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym or sym not in held:
            continue
        action = str(row.get("final_action") or row.get("management_action") or "").strip().upper()
        if action not in RISK_REDUCING_ACTIONS:
            continue
        opp = "EXIT" if action in ("EXIT", "ROTATE_EXIT") else "TRIM"
        try:
            qty = int(float(row.get("qty") or 0))
        except (TypeError, ValueError):
            qty = 0
        try:
            limit_price = float(row.get("limit_price") or row.get("current_price") or 0)
        except (TypeError, ValueError):
            limit_price = 0.0
        if qty <= 0:
            continue
        rows.append(
            {
                "ticker": sym,
                "symbol": sym,
                "effective_stance": action,
                "effective_position_state": "LONG",
                "opportunity_type": opp,
                "confidence": row.get("score"),
                "delta_pct": row.get("delta_pct"),
                "rationale": str(row.get("rationale") or row.get("reason_detail") or ""),
                "close": limit_price,
                "limit_price": limit_price,
                "qty": qty,
                "exploration_flag": False,
                "discipline_allowed": row.get("discipline_allowed", True),
                "discipline_reason": str(row.get("discipline_reason") or ""),
                "preview_only": True,
                "reason_code": str(row.get("reason_code") or ""),
                "rank": row.get("rank"),
                "priority_bucket": str(row.get("priority_bucket") or ""),
                "planned_notional": row.get("planned_notional"),
            }
        )
    rows.sort(key=lambda r: (int(r.get("rank") or 9999), str(r.get("symbol") or "")))
    return rows


def write_preview_handoff_artifacts(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Path:
    PREVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_csv(PREVIEW_CSV, index=False)
    else:
        PREVIEW_CSV.write_text(
            "ticker,effective_stance,effective_position_state,opportunity_type,"
            "confidence,delta_pct,rationale,close,limit_price,qty,exploration_flag\n",
            encoding="utf-8",
        )
    payload = {
        "generated_at": _utc_now_iso(),
        "phase": "198A",
        "preview_only": True,
        "execution_allowed": False,
        "broker_orders_submitted": False,
        "preview_row_count": len(rows),
        "preview_symbols": sorted({str(r.get("symbol") or r.get("ticker") or "") for r in rows}),
        "preview_path": str(PREVIEW_CSV.relative_to(ROOT)).replace("\\", "/"),
        "exec_handoff_path": str(EXEC_HANDOFF_CSV.relative_to(ROOT)).replace("\\", "/"),
        **meta,
    }
    _atomic_write_json(payload, PREVIEW_JSON)
    return PREVIEW_JSON


def ensure_defensive_exit_preview_handoff(
    *,
    mode: str = "paper",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Ensure read-only preview handoff artifact exists when ORANGE_DEGRADED preview is eligible.
    Never writes to data/live/manage_orders.csv.
    """
    eligibility = evaluate_defensive_preview_eligibility(mode=mode)
    exec_rows = _count_exec_handoff_exit_rows()
    if exec_rows > 0 and not force:
        return {
            **eligibility,
            "action": "skipped_exec_handoff_present",
            "exec_handoff_rows": exec_rows,
            "preview_row_count": 0,
            "preview_symbols": [],
        }

    if not eligibility.get("preview_generation_allowed"):
        return {
            **eligibility,
            "action": "skipped_not_eligible",
            "exec_handoff_rows": exec_rows,
            "preview_row_count": 0,
            "preview_symbols": [],
        }

    held = load_held_symbols()
    rows = build_preview_handoff_rows_from_plan(held_symbols=held)
    write_preview_handoff_artifacts(
        rows,
        {
            **eligibility,
            "held_symbols_count": len(held),
            "action": "preview_written",
        },
    )
    print(
        "[DEFENSIVE_EXIT_PREVIEW_HANDOFF] "
        f"preview_generation_allowed=true execution_allowed=false "
        f"rows={len(rows)} symbols={sorted({r['symbol'] for r in rows})}",
        flush=True,
    )
    return {
        **eligibility,
        "action": "preview_written",
        "exec_handoff_rows": exec_rows,
        "preview_row_count": len(rows),
        "preview_symbols": sorted({str(r.get("symbol") or "") for r in rows}),
        "preview_path": str(PREVIEW_CSV.relative_to(ROOT)).replace("\\", "/"),
    }


def _count_exec_handoff_exit_rows() -> int:
    if not EXEC_HANDOFF_CSV.is_file() or EXEC_HANDOFF_CSV.stat().st_size == 0:
        return 0
    try:
        import pandas as pd

        df = pd.read_csv(EXEC_HANDOFF_CSV, low_memory=False)
        if df.empty or "ticker" not in df.columns and "symbol" not in df.columns:
            return 0
        opp = df.get("opportunity_type", df.get("effective_stance", ""))
        return int(opp.astype(str).str.upper().isin(["EXIT", "TRIM", "ROTATE_EXIT", "SELL"]).sum())
    except Exception:
        return 0


def count_preview_handoff_exit_rows() -> int:
    if not PREVIEW_CSV.is_file():
        return 0
    try:
        import pandas as pd

        df = pd.read_csv(PREVIEW_CSV, low_memory=False)
        if df.empty:
            return 0
        sym_col = "ticker" if "ticker" in df.columns else "symbol"
        if sym_col not in df.columns:
            return 0
        opp = df.get("opportunity_type", df.get("effective_stance", ""))
        return int(opp.astype(str).str.upper().isin(["EXIT", "TRIM", "ROTATE_EXIT", "SELL"]).sum())
    except Exception:
        return 0


def resolve_defensive_handoff_orders_path() -> Tuple[Path, str]:
    """
    Prefer live execution handoff when populated; otherwise read-only preview artifact.
    """
    if _count_exec_handoff_exit_rows() > 0:
        return EXEC_HANDOFF_CSV, "exec_handoff"
    if count_preview_handoff_exit_rows() > 0:
        return PREVIEW_CSV, "preview_handoff"
    ensure_defensive_exit_preview_handoff()
    if count_preview_handoff_exit_rows() > 0:
        return PREVIEW_CSV, "preview_handoff"
    return EXEC_HANDOFF_CSV, "exec_handoff_empty"


def load_defensive_handoff_orders_for_analysis() -> Tuple[List[Dict[str, Any]], str]:
    """Load handoff orders from exec or preview path for sizing/governance analysis."""
    path, source = resolve_defensive_handoff_orders_path()
    if not path.is_file():
        return [], source
    try:
        import pandas as pd

        df = pd.read_csv(path, low_memory=False)
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
            try:
                qty = int(float(row.get("qty") or 0))
            except (TypeError, ValueError):
                qty = 0
            try:
                price = float(row.get("limit_price") or row.get("close") or 0)
            except (TypeError, ValueError):
                price = 0.0
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
                    "preview_only": bool(row.get("preview_only", source == "preview_handoff")),
                }
            )
        return rows, source
    except Exception:
        return [], source
