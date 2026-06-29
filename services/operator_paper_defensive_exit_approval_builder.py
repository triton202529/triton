"""
operator_paper_defensive_exit_approval_builder.py
Phase 197L — Guarded operator paper defensive exit approval template builder.
Default dry-run only; real approval requires explicit --write-real-approval flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
APPROVAL_TEMPLATE_JSON = RESULTS / "operator_paper_defensive_exit_approval_template.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
OPEN_ORDERS_JSON = RESULTS / "open_orders_snapshot.csv"

DEFAULT_EXPIRES_MINUTES = 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _count_open_orders() -> int:
    if not OPEN_ORDERS_JSON.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(OPEN_ORDERS_JSON))
    except Exception:
        return 0


def _parse_packet_expiry(packet: Dict[str, Any]) -> Optional[datetime]:
    raw = str(packet.get("expires_at") or "")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _resolve_symbols(
    packet: Dict[str, Any], symbols_arg: Optional[str]
) -> Tuple[List[str], Optional[str]]:
    capped = [
        str(s).strip().upper() for s in (packet.get("capped_symbols") or []) if str(s).strip()
    ]
    if not symbols_arg:
        return capped, None
    requested = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    capped_set = set(capped)
    extra = sorted(set(requested) - capped_set)
    if extra:
        return requested, f"symbols_not_in_capped_preview extra={','.join(extra)}"
    if not requested:
        return capped, "approved_symbols_empty"
    return requested, None


def _estimate_max_notional(packet: Dict[str, Any], approved_symbols: List[str]) -> float:
    capped = [str(s).upper() for s in (packet.get("capped_symbols") or [])]
    total = float(packet.get("capped_notional") or 0.0)
    if not capped or set(approved_symbols) == set(capped):
        return round(total, 2)
    per_symbol = total / len(capped) if capped else 0.0
    return round(per_symbol * len(approved_symbols), 2)


def build_proposed_contract(
    *,
    packet: Dict[str, Any],
    approved_by: str,
    approved_symbols: List[str],
    expires_minutes: int,
) -> Dict[str, Any]:
    from services.defensive_exit_operator_paper_approval import OPERATOR_DECISION_APPROVE

    now = _utc_now()
    configured_cap = float(packet.get("configured_cap") or 0.0)
    max_notional = _estimate_max_notional(packet, approved_symbols)
    if max_notional > configured_cap:
        max_notional = configured_cap
    return {
        "approval_id": str(uuid.uuid4()),
        "approved_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=expires_minutes)).isoformat(),
        "approved_by": approved_by,
        "approval_mode": "PAPER_ONLY",
        "approved_symbols": approved_symbols,
        "max_notional": max_notional,
        "source_packet_id": str(packet.get("packet_id") or ""),
        "source_preview_hash": str(packet.get("source_preview_hash") or ""),
        "acknowledgement_live_blocked": True,
        "acknowledgement_no_live_execution": True,
        "acknowledgement_paper_only": True,
        "decision": OPERATOR_DECISION_APPROVE,
    }


def preflight_real_write(
    *,
    packet: Dict[str, Any],
    approved_by: Optional[str],
    approved_symbols: List[str],
    proposed_contract: Dict[str, Any],
    broker_snapshot_ok: bool = True,
) -> Tuple[bool, List[str]]:
    """Return (ok, block_reasons) for real approval write."""
    reasons: List[str] = []
    if not approved_by or not str(approved_by).strip():
        reasons.append("approved_by_missing")
    if not str(packet.get("source_preview_hash") or "").strip():
        reasons.append("source_preview_hash_missing")
    packet_exp = _parse_packet_expiry(packet)
    if packet_exp and packet_exp <= _utc_now():
        reasons.append("operator_packet_expired")
    if not broker_snapshot_ok:
        reasons.append("broker_snapshot_not_ok")
    open_orders = _count_open_orders()
    if open_orders > 0:
        reasons.append(f"open_orders_count={open_orders}")
    try:
        from services.portfolio_history_sync import load_live_execution_flag

        readiness = _read_json(RESULTS / "execution_readiness.json")
        if bool(readiness.get("live_execution_permitted", False)) or load_live_execution_flag():
            reasons.append("live_execution_not_blocked")
    except Exception:
        reasons.append("readiness_state_unreadable")
    try:
        _read_json(RESULTS / "governance_authorization.json")
    except Exception:
        reasons.append("governance_state_unreadable")
    capped_set = {str(s).upper() for s in (packet.get("capped_symbols") or [])}
    approved_set = {str(s).upper() for s in approved_symbols}
    if not approved_set.issubset(capped_set):
        extra = sorted(approved_set - capped_set)
        reasons.append(f"symbols_not_subset extra={','.join(extra)}")
    configured_cap = float(packet.get("configured_cap") or 0.0)
    max_notional = float(proposed_contract.get("max_notional") or 0.0)
    if max_notional <= 0 or max_notional > configured_cap:
        reasons.append(f"max_notional_exceeds_cap max={max_notional} cap={configured_cap}")
    if str(proposed_contract.get("approval_mode") or "").upper() != "PAPER_ONLY":
        reasons.append("approval_mode_not_paper_only")
    for field in (
        "acknowledgement_live_blocked",
        "acknowledgement_no_live_execution",
        "acknowledgement_paper_only",
    ):
        if not bool(proposed_contract.get(field)):
            reasons.append(f"{field}_not_true")
    return not reasons, reasons


def run_approval_builder(
    *,
    dry_run: bool = True,
    write_real_approval: bool = False,
    approved_by: Optional[str] = None,
    symbols: Optional[str] = None,
    expires_minutes: int = DEFAULT_EXPIRES_MINUTES,
    broker_snapshot_ok: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build approval template (dry-run default) or write real approval when explicitly requested.
    Never submits broker orders or alters governance/readiness artifacts.
    """
    packet = _read_json(OPERATOR_PACKET_JSON)
    if not packet:
        raise FileNotFoundError(
            f"Operator packet missing: {OPERATOR_PACKET_JSON.relative_to(ROOT)}"
        )

    approved_symbols, symbol_err = _resolve_symbols(packet, symbols)
    if symbol_err:
        raise ValueError(symbol_err)

    operator_name = str(approved_by or "").strip() or "OPERATOR_DRY_RUN_TEMPLATE"
    proposed = build_proposed_contract(
        packet=packet,
        approved_by=operator_name,
        approved_symbols=approved_symbols,
        expires_minutes=expires_minutes,
    )

    real_write_attempted = bool(write_real_approval)
    real_write_blocked_without_flag = False
    real_approval_written = False
    real_block_reasons: List[str] = []

    if write_real_approval:
        ok, real_block_reasons = preflight_real_write(
            packet=packet,
            approved_by=approved_by,
            approved_symbols=approved_symbols,
            proposed_contract=proposed,
            broker_snapshot_ok=broker_snapshot_ok,
        )
        if ok:
            _atomic_write_json(REAL_APPROVAL_PATH, proposed)
            real_approval_written = True
        elif verbose:
            print(
                "[197L-APPROVAL-BUILDER] real write blocked: " + "; ".join(real_block_reasons),
                file=sys.stderr,
            )
    else:
        real_write_blocked_without_flag = True

    template = {
        "generated_at": _utc_now_iso(),
        "phase": "197L",
        "template_only": True,
        "not_runtime_authority": True,
        "disclaimer": (
            "Dry-run approval template only. Does not approve execution. "
            "Real approval requires explicit --write-real-approval with --approved-by."
        ),
        "would_write_real_path": str(REAL_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "write_real_approval_requested": write_real_approval,
        "real_approval_written": real_approval_written,
        "real_write_block_reasons": real_block_reasons,
        "operator_packet_id": packet.get("packet_id"),
        "source_preview_hash": packet.get("source_preview_hash"),
        "approved_symbols": approved_symbols,
        "configured_cap": packet.get("configured_cap"),
        "capped_notional": packet.get("capped_notional"),
        "expires_minutes": expires_minutes,
        "proposed_contract": proposed,
        "human_readable_summary": {
            "action": "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
            "mode": "PAPER_ONLY",
            "symbols": approved_symbols,
            "max_notional": proposed.get("max_notional"),
            "expires_at": proposed.get("expires_at"),
            "packet_id": proposed.get("source_packet_id"),
            "preview_hash": proposed.get("source_preview_hash"),
            "warning": (
                "Creating the real approval file is a separate operator decision and does "
                "not bypass governance, readiness, or paper arm gates."
            ),
        },
    }
    _atomic_write_json(APPROVAL_TEMPLATE_JSON, template)

    result = {
        "dry_run": dry_run or not write_real_approval,
        "template_path": str(APPROVAL_TEMPLATE_JSON.relative_to(ROOT)).replace("\\", "/"),
        "template_written": True,
        "real_approval_contract_path": str(REAL_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "real_approval_written": real_approval_written,
        "write_real_approval_attempted": real_write_attempted,
        "write_real_approval_blocked_without_flag": real_write_blocked_without_flag,
        "real_write_block_reasons": real_block_reasons,
        "proposed_contract": proposed,
        "template": template,
    }
    if verbose:
        mode = "REAL_WRITE" if real_approval_written else "DRY_RUN_TEMPLATE"
        print(
            f"[197L-APPROVAL-BUILDER] mode={mode} template={APPROVAL_TEMPLATE_JSON.relative_to(ROOT)}"
        )
        if real_approval_written:
            print(
                f"[197L-APPROVAL-BUILDER] wrote real approval "
                f"{REAL_APPROVAL_PATH.relative_to(ROOT)}"
            )
        else:
            print(
                "[197L-APPROVAL-BUILDER] real approval NOT written "
                "(use --write-real-approval --approved-by to activate)"
            )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build operator paper defensive exit approval template (dry-run default). "
            "Real approval requires --write-real-approval and --approved-by."
        )
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Write template only under data/results (default).",
    )
    ap.add_argument(
        "--write-real-approval",
        action="store_true",
        help="Explicitly write data/governance/operator_paper_defensive_exit_approval.json.",
    )
    ap.add_argument(
        "--approved-by",
        default="",
        help="Operator identity; required with --write-real-approval.",
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated subset of capped preview symbols (default: all capped).",
    )
    ap.add_argument(
        "--expires-minutes",
        type=int,
        default=DEFAULT_EXPIRES_MINUTES,
        help=f"Approval expiration in minutes (default {DEFAULT_EXPIRES_MINUTES}).",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    dry_run = not args.write_real_approval
    if args.write_real_approval and not str(args.approved_by or "").strip():
        print(
            "[197L-APPROVAL-BUILDER] ERROR: --approved-by is required with --write-real-approval",
            file=sys.stderr,
        )
        return 1

    try:
        run_approval_builder(
            dry_run=dry_run,
            write_real_approval=args.write_real_approval,
            approved_by=str(args.approved_by or "").strip() or None,
            symbols=str(args.symbols or "").strip() or None,
            expires_minutes=max(1, int(args.expires_minutes)),
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[197L-APPROVAL-BUILDER] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
