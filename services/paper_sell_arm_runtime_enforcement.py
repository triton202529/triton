"""
paper_sell_arm_runtime_enforcement.py
Phase 198I / 198I-FIX — Canonical hardened paper sell-arm gate and atomic consumption.

Sole authority for paper TRIM/EXIT/ROTATE_EXIT/FORCE_EXIT (and any paper sell
submission/replacement/reprice/retry) is Phase 198H schema PAPER_SELL_ARM.

Legacy 197W approvals are historical evidence only (separate storage) and cannot
authorize dispatch.

Consumption: durable irreversible consume AFTER final in-lock revalidation and
BEFORE the first possible broker submission attempt.

Broker-facing paper sells require an active dispatch permit issued only after
durable consumption (ContextVar). AlpacaBroker hooks enforce this.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence, Set, Tuple

from services.paper_sell_arm_cap_slice import money_round
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    AUTHORITY_TYPE,
    DEPRECATED_SHARED_APPROVAL_FILENAME,
    HARDENED_AUTHORITY_FILENAME,
    LEGACY_197W_APPROVAL_FILENAME,
    LIVE_APPROVAL_PATH,
    LIVE_BUY_ARM_PATH,
    LIVE_GOVERNANCE,
    SCHEMA_VERSION,
    SellArmState,
    BrokerSnapshotView,
    DispatchDurabilityError,
    PaperSellArmPaths,
    SELL_ACTIONS,
    _assert_writable_root,
    _atomic_write_json,
    _parse_iso,
    _read_json,
    _utc_now,
    append_audit,
    detect_hardened_authority_path_collision,
    evaluate_readiness_packet_freshness,
    is_hardened_sell_arm_contract,
    is_dual_identity_record,
    legacy_ambiguous_slice_authority,
    mark_authority_dispatch_attempt,
    classify_sqlite_json_export_consistency,
    readiness_packet_hash,
    revalidate_before_dispatch,
    source_readiness_packet_hash,
    validate_paper_sell_arm_authority,
    write_state,
    BUY_ACTIONS,
)

ROOT = Path(__file__).resolve().parents[1]

DISPATCH_INTENT_NAME = "paper_sell_arm_dispatch_intent.json"
DURABLE_PERMIT_NAME = "paper_sell_arm_dispatch_permit.json"
LOCK_NAME = "paper_sell_arm_dispatch.lock"

_PROCESS_LOCK = threading.RLock()

LEGACY_197W_RETIRED = "legacy_197w_retired_for_runtime_dispatch"
HARDENED_REQUIRED = "hardened_paper_sell_arm_required"
NO_SELLS = "no_paper_sells_in_batch"


@dataclass
class PaperSellDispatchPermit:
    """One-use permit issued only after durable consumption (memory + disk)."""

    dispatch_intent_id: str
    idempotency_key: str
    store_root: Path
    consumed_durable: bool = True
    symbols: Tuple[str, ...] = ()
    actions: Tuple[Tuple[str, str, float], ...] = ()  # (symbol, action, qty)
    permit_token: str = ""
    sqlite_db_path: Optional[Path] = None


_ACTIVE_PERMIT: ContextVar[Optional[PaperSellDispatchPermit]] = ContextVar(
    "paper_sell_dispatch_permit", default=None
)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def looks_like_legacy_197w_approval(doc: Dict[str, Any]) -> bool:
    if not doc:
        return False
    if is_hardened_sell_arm_contract(doc):
        return False
    if str(doc.get("authority_type") or "") == AUTHORITY_TYPE:
        return False
    decision = str(doc.get("decision") or doc.get("approval_action") or "").upper()
    mode = str(doc.get("approval_mode") or doc.get("approval_scope") or "").upper()
    return decision == "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH" or (
        mode == "PAPER_ONLY" and bool(doc.get("approved_symbols"))
    )


def validate_runtime_sell_authority(
    *,
    contract_path: Optional[Path] = None,
    now: Optional[datetime] = None,
    consumption_path: Optional[Path] = None,
    store_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Sole runtime authority check: hardened 198H only on canonical path."""
    root = store_root or (contract_path.parent if contract_path else LIVE_GOVERNANCE)
    details: Dict[str, Any] = {
        "contract_path": str(contract_path or (root / HARDENED_AUTHORITY_FILENAME)),
        "policy": "hardened_only",
        "phase": "198I-FIX",
        "canonical_authority_filename": HARDENED_AUTHORITY_FILENAME,
        "legacy_197w_filename": LEGACY_197W_APPROVAL_FILENAME,
    }
    # Default path resolution: refuse ambiguous storage layouts.
    if contract_path is None:
        collision, creason = detect_hardened_authority_path_collision(root)
        if collision:
            return False, f"authority_path_collision:{creason}", details
        path = root / HARDENED_AUTHORITY_FILENAME
    else:
        path = contract_path
        if path.name == HARDENED_AUTHORITY_FILENAME:
            collision, creason = detect_hardened_authority_path_collision(path.parent)
            if collision:
                return False, f"authority_path_collision:{creason}", details
    details["contract_path"] = str(path)
    from services.paper_sell_arm_store import (
        PaperSellArmStoreError,
        inspect_sqlite_authority,
        resolve_runtime_sqlite_path,
        sqlite_permit_grade_enabled,
    )

    sqlite_applies = sqlite_permit_grade_enabled() and path.name == HARDENED_AUTHORITY_FILENAME
    if sqlite_applies:
        try:
            inspected = inspect_sqlite_authority(
                resolve_runtime_sqlite_path(db_path=db_path), now=now or _utc_now()
            )
        except PaperSellArmStoreError as exc:
            return False, exc.reason, {**details, **exc.details}
        st = str(inspected.get("state") or "ABSENT")
        if st == "UNUSED":
            sqlite_doc = inspected.get("payload") or {}
            details["authority_source"] = "sqlite"
            details["json_role"] = "AUDIT_EXPORT_ONLY"
            details["authority_type"] = sqlite_doc.get("authority_type")
            details["schema_version"] = sqlite_doc.get("schema_version")
            json_doc = _read_json(path) if path.is_file() else None
            consumption = _read_json(consumption_path) if consumption_path else None
            if not consumption:
                consumption = None
                sibling = path.parent / "paper_sell_arm_consumption.json"
                if sibling.is_file():
                    consumption = _read_json(sibling)
            if not consumption:
                consumption = None
            kind, mismatch_reason = classify_sqlite_json_export_consistency(
                sqlite_state="UNUSED",
                sqlite_payload=sqlite_doc if isinstance(sqlite_doc, dict) else {},
                json_authority=json_doc if isinstance(json_doc, dict) else None,
                consumption=consumption if isinstance(consumption, dict) else None,
            )
            if kind != "ok":
                return False, mismatch_reason, {**details, "sqlite_json_consistency": kind}
            ok, reason, hdetails = validate_paper_sell_arm_authority(
                sqlite_doc, now=now, consumption=None
            )
            details.update(hdetails)
            return ok, reason, details
        if st == "EXPIRED":
            return (
                False,
                "sqlite_authority_expired",
                {**details, "json_role": "AUDIT_EXPORT_ONLY"},
            )
        if st == "CONSUMED":
            return (
                False,
                "authority_consumed",
                {**details, "json_role": "AUDIT_EXPORT_ONLY"},
            )
        if st in {"RESTRICTED", "REVOKED"}:
            return (
                False,
                f"sqlite_authority_{st.lower()}",
                {**details, "json_role": "AUDIT_EXPORT_ONLY"},
            )
        # ABSENT: classify JSON for diagnostics, never grant a permit.
        if not path.is_file():
            return False, "approval_contract_missing", details
        doc = _read_json(path)
        details["authority_type"] = doc.get("authority_type")
        details["schema_version"] = doc.get("schema_version")
        details["json_role"] = "AUDIT_EXPORT_ONLY"
        details["authority_source"] = "legacy_json_non_authoritative"
        if looks_like_legacy_197w_approval(doc):
            return False, LEGACY_197W_RETIRED, details
        if not is_hardened_sell_arm_contract(doc):
            return False, HARDENED_REQUIRED, details
        consumption = _read_json(consumption_path) if consumption_path else None
        if consumption is None:
            sibling = path.parent / "paper_sell_arm_consumption.json"
            if sibling.is_file():
                consumption = _read_json(sibling)
        ok, reason, hdetails = validate_paper_sell_arm_authority(
            doc, now=now, consumption=consumption or None
        )
        details.update(hdetails)
        if ok:
            return (
                False,
                "legacy_json_not_broker_usable",
                {
                    **details,
                    "hint": "re_arm_required_sqlite_is_source_of_truth",
                },
            )
        return False, reason, details

    if not path.is_file():
        return False, "approval_contract_missing", details
    # Runtime never treats the deprecated shared filename as a valid hardened path
    # unless the document itself is inspected and rejected as legacy/non-hardened.
    doc = _read_json(path)
    details["authority_type"] = doc.get("authority_type")
    details["schema_version"] = doc.get("schema_version")
    if looks_like_legacy_197w_approval(doc):
        return False, LEGACY_197W_RETIRED, details
    if not is_hardened_sell_arm_contract(doc):
        return False, HARDENED_REQUIRED, details
    consumption = _read_json(consumption_path) if consumption_path else None
    if consumption is None:
        sibling = path.parent / "paper_sell_arm_consumption.json"
        if sibling.is_file():
            consumption = _read_json(sibling)
    ok, reason, hdetails = validate_paper_sell_arm_authority(
        doc, now=now, consumption=consumption or None
    )
    details.update(hdetails)
    return ok, reason, details


def planned_rows_include_paper_sells(planned: Sequence[Any]) -> bool:
    for p in planned:
        side = str(getattr(p, "side", "") or "").lower()
        stance = str(
            getattr(p, "stance", None)
            or getattr(p, "management_action", None)
            or getattr(p, "opportunity_type", None)
            or ""
        ).upper()
        if side == "sell" or stance in SELL_ACTIONS:
            return True
    return False


def requested_actions_from_planned(planned: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in planned:
        side = str(getattr(p, "side", "") or "").lower()
        stance = str(
            getattr(p, "stance", None)
            or getattr(p, "management_action", None)
            or getattr(p, "opportunity_type", None)
            or ""
        ).upper()
        if side != "sell" and stance not in SELL_ACTIONS:
            continue
        action = stance if stance in SELL_ACTIONS else "EXIT"
        if bool(getattr(p, "force_exit_override", False)):
            action = "FORCE_EXIT"
        sym = str(getattr(p, "symbol", "") or "").upper()
        qty = float(getattr(p, "qty", 0) or 0)
        price = float(getattr(p, "limit_price", None) or getattr(p, "ref_price", None) or 0)
        out.append(
            {
                "symbol": sym,
                "action": action,
                "qty": qty,
                "price": price,
                "notional": qty * price if qty and price else 0.0,
                "client_order_id": str(getattr(p, "client_order_id", "") or ""),
            }
        )
    return out


def _planned_is_sell(p: Any) -> bool:
    side = str(getattr(p, "side", "") or "").lower()
    stance = str(
        getattr(p, "stance", None)
        or getattr(p, "management_action", None)
        or getattr(p, "opportunity_type", None)
        or ""
    ).upper()
    return side == "sell" or stance in SELL_ACTIONS


def _clone_planned_qty(p: Any, qty: float) -> Any:
    from dataclasses import is_dataclass, replace

    qty_val: Any = int(qty) if isinstance(getattr(p, "qty", None), int) else qty
    if is_dataclass(p) and not isinstance(p, type):
        return replace(p, qty=qty_val)
    cloned = copy.copy(p)
    try:
        cloned.qty = qty_val
    except Exception:
        return p
    return cloned


def _price_unavailable(value: Any) -> bool:
    """Fail-closed current sell reference: missing/zero/negative/NaN/Inf/malformed."""
    from decimal import Decimal, InvalidOperation

    if value is None:
        return True
    try:
        if isinstance(value, float):
            import math as _math

            if _math.isnan(value) or _math.isinf(value):
                return True
        price = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError, ArithmeticError):
        return True
    return (not price.is_finite()) or price <= 0


def _downward_runtime_qty(
    authorized_qty: float, stored_max_notional: Any, price: float
) -> Tuple[int, str]:
    """Downward-only whole-share qty so runtime notional cannot exceed stored max.

    runtime_quantity = min(authorized_quantity, floor(max_notional / price)),
    then walked down until money_round(qty * price) <= stored max.
    Never increases quantity. Governing comparisons use Decimal, not float.
    """
    from decimal import Decimal, ROUND_FLOOR

    auth = int(Decimal(str(authorized_qty)).to_integral_value(rounding=ROUND_FLOOR))
    if auth <= 0:
        return 0, "invalid_authorized_qty"
    if stored_max_notional is None:
        return auth, "unchanged_no_stored_max"
    max_n = Decimal(str(money_round(stored_max_notional)))
    px = Decimal(str(price))
    runtime = min(Decimal(auth), (max_n / px).to_integral_value(rounding=ROUND_FLOOR))
    if runtime < 0:
        runtime = Decimal("0")
    while runtime > 0 and Decimal(str(money_round(float(runtime * px)))) > max_n:
        runtime -= 1
    qty = int(runtime)
    if qty > auth:
        return auth, "quantity_expansion_forbidden"
    if qty <= 0:
        return 0, "clipped_to_zero_vs_authorized_max_notional"
    if qty < auth:
        return qty, "downward_clip_vs_authorized_max_notional"
    return qty, "authorized_quantity_fits_current_price"


def _authorized_clip_audit(
    authority: Mapping[str, Any], planned: Sequence[Any]
) -> List[Dict[str, Any]]:
    """JSON-serializable per-line clip audit. Never emits Decimal objects."""
    by_sym = {str(getattr(p, "symbol", "") or "").upper(): p for p in planned}
    rows: List[Dict[str, Any]] = []
    for item in list(
        authority.get("authorized_actions") or authority.get("authorized_scope") or []
    ):
        symbol = str(item.get("symbol") or "").upper()
        action = str(item.get("action") or "").upper()
        auth_qty = float(item.get("max_qty") or 0)
        max_n = item.get("max_notional")
        p = by_sym.get(symbol)
        runtime_qty = float(getattr(p, "qty", 0) or 0) if p is not None else 0.0
        price_raw = None
        if p is not None:
            price_raw = getattr(p, "limit_price", None)
            if price_raw is None:
                price_raw = getattr(p, "ref_price", None)
        try:
            price_f = float(price_raw) if price_raw is not None else None
        except (TypeError, ValueError):
            price_f = None
        runtime_notional = (
            money_round(runtime_qty * price_f) if price_f is not None and runtime_qty else 0.0
        )
        outcome = "UNCHANGED"
        reason = str(getattr(p, "clip_reason", "") or "") if p is not None else ""
        if p is None or runtime_qty <= 0:
            outcome = "SKIPPED_ZERO"
            reason = reason or "clipped_to_zero_vs_authorized_max_notional"
        elif runtime_qty + 1e-9 < auth_qty:
            outcome = "CLIPPED_DOWN"
            reason = reason or "downward_clip_vs_authorized_max_notional"
        else:
            reason = reason or "authorized_quantity_fits_current_price"
        rows.append(
            {
                "symbol": symbol,
                "action": action,
                "authorized_quantity": auth_qty,
                "runtime_quantity": runtime_qty,
                "deferred_quantity": max(auth_qty - runtime_qty, 0.0),
                "current_reference_price": price_f,
                "authorized_max_notional": float(max_n) if max_n is not None else None,
                "runtime_estimated_notional": float(runtime_notional),
                "outcome": outcome,
                "reason": reason,
            }
        )
    return rows


def select_authorized_planned(
    planned: Sequence[Any],
    authority: Mapping[str, Any],
    *,
    reject_unapproved_planned: bool = False,
) -> Tuple[bool, str, Tuple[Any, ...]]:
    """Map full planned SELLs onto exact authorized_scope. Clip qty down; never expand."""
    scope = list(authority.get("authorized_actions") or authority.get("authorized_scope") or [])
    if not scope:
        return False, "empty_authorized_scope", ()
    sells = [p for p in planned if _planned_is_sell(p)]
    by_sym: Dict[str, List[Any]] = {}
    extra_syms = set()
    for p in sells:
        sym = str(getattr(p, "symbol", "") or "").upper()
        by_sym.setdefault(sym, []).append(p)
        extra_syms.add(sym)
    selected: List[Any] = []
    for item in scope:
        symbol = str(item.get("symbol") or "").upper()
        action = str(item.get("action") or "").upper()
        max_qty = float(item.get("max_qty") or 0)
        max_n = item.get("max_notional")
        extra_syms.discard(symbol)
        cands = by_sym.get(symbol) or []
        if not cands:
            return False, f"incomplete_authorized_qty:{symbol}", ()
        p = cands[0]
        stance = str(
            getattr(p, "stance", None)
            or getattr(p, "management_action", None)
            or getattr(p, "opportunity_type", None)
            or ""
        ).upper()
        planned_action = stance if stance in SELL_ACTIONS else "EXIT"
        if bool(getattr(p, "force_exit_override", False)):
            planned_action = "FORCE_EXIT"
        if (
            action
            and planned_action != action
            and not (action == "EXIT" and planned_action in SELL_ACTIONS)
        ):
            return False, f"action_mismatch_vs_authorized_slice:{symbol}", ()
        price = float(getattr(p, "limit_price", None) or getattr(p, "ref_price", None) or 0)
        if _price_unavailable(price):
            return False, f"price_unavailable:{symbol}", ()
        derived_qty, clip_reason = _downward_runtime_qty(max_qty, max_n, price)
        pq = float(getattr(p, "qty", 0) or 0)
        planned_reason = str(getattr(p, "clip_reason", "") or "")
        remaining_cap_reasons = {
            "clipped_to_zero_vs_remaining_batch_cap",
            "downward_clip_vs_remaining_batch_cap",
        }
        if pq <= 0:
            continue
        if derived_qty <= 0:
            continue
        if pq + 1e-9 < float(derived_qty):
            if planned_reason not in remaining_cap_reasons:
                return False, f"incomplete_authorized_qty:{symbol}", ()
            use_qty = int(pq)
            clip_reason = planned_reason
        else:
            use_qty = derived_qty
        notional = money_round(float(use_qty) * price)
        if notional > APPROVAL_CLI_CAP + 1e-6:
            return False, f"notional_expansion_vs_authorized_slice:{symbol}", ()
        if abs(float(use_qty) - float(getattr(p, "qty", 0) or 0)) > 1e-9:
            p = _clone_planned_qty(p, use_qty)
        setattr(p, "clip_reason", clip_reason)
        selected.append(p)
    if reject_unapproved_planned and extra_syms:
        extra = sorted(extra_syms)[0]
        return False, f"unauthorized_symbol_vs_authorized_slice:{extra}", ()
    if not selected:
        return False, "no_executable_authorized_quantity", ()
    total_sel = money_round(
        sum(
            float(getattr(p, "qty", 0) or 0)
            * float(getattr(p, "limit_price", None) or getattr(p, "ref_price", None) or 0)
            for p in selected
        )
    )
    stored_total = authority.get("total_batch_max_notional")
    if stored_total is not None and total_sel > money_round(stored_total) + 1e-9:
        return False, "notional_expansion_vs_authorized_slice_total", ()
    if total_sel > APPROVAL_CLI_CAP + 1e-6:
        return False, "batch_notional_exceeds_approval_cli_cap", ()
    return True, "ok", tuple(selected)


def _load_readiness_packet(results_root: Optional[Path], store_root: Path) -> Dict[str, Any]:
    results_root = results_root or (ROOT / "data" / "results")
    path = results_root / "defensive_exit_approval_readiness.json"
    if (store_root / "defensive_exit_approval_readiness.json").is_file():
        path = store_root / "defensive_exit_approval_readiness.json"
    return _read_json(path) or {}


@dataclass
class DispatchGuardResult:
    permitted: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    dispatch_intent_id: Optional[str] = None
    authority: Optional[Dict[str, Any]] = None
    consumed: bool = False
    permit: Optional[PaperSellDispatchPermit] = None
    authorized_planned: Tuple[Any, ...] = field(default_factory=tuple)


class PaperSellDispatchBlocked(Exception):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


# Phase 198T-AT — unusable SELL permit blocks SELL only. Unknown reasons stay
# fail-closed (hard cycle failure) so ambiguous execution is never softened.
SELL_PERMISSION_SOFT_BLOCK_REASONS = frozenset(
    {
        "stale_readiness_packet",
        "no_unused_authority",
        "authority_expired",
        "sqlite_authority_expired",
        "authority_missing",
        "authority_not_unused",
        "status_not_armed",
        "empty_authorized_scope",
        "trading_date_mismatch",
        "readiness_packet_hash_mismatch",
        "source_readiness_packet_missing",
        "source_readiness_packet_hash_mismatch",
        "source_readiness_packet_hash_changed",
        "invocation_mismatch",
        "generated_at_mismatch",
        "source_plan_hash_mismatch",
        "selected_row_hash_mismatch",
        "skipped_not_paper_or_dry_run",
        "skipped_not_paper",
        "no_paper_sells_in_batch",
        "missing_generated_at",
        "unparseable_generated_at",
        "naive_generated_at",
        "future_generated_at",
        "session_boundary_closed",
        "session_derivation_unknown",
        "no_current_challenge",
        "no_authorized_quantity",
        "no_matching_action_after_clipping",
        "foreign_historical_authority",
    }
)


def sell_permission_soft_block_reason(reason: Optional[str]) -> bool:
    """True when SELL must not submit, but F/F2/BUY may continue."""
    return str(reason or "") in SELL_PERMISSION_SOFT_BLOCK_REASONS


def sell_unusable_should_abort_cycle(reason: Optional[str]) -> bool:
    """Hard cycle abort for ambiguous/corrupt execution; soft otherwise for the allowlist."""
    text = str(reason or "").strip()
    if not text:
        return True
    if sell_permission_soft_block_reason(text):
        return False
    return True


def _lock_path(store_root: Path) -> Path:
    return store_root / LOCK_NAME


def _acquire_lock(store_root: Path) -> Any:
    """Cross-process exclusive lock file (Windows msvcrt / Unix fcntl)."""
    path = _lock_path(store_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            fh.write("0")
            fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise PaperSellDispatchBlocked("dispatch_lock_unavailable", {"error": str(exc)}) from exc
    return fh


def _release_lock(fh: Any) -> None:
    if fh is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


@contextmanager
def _dispatch_lock(store_root: Path) -> Generator[Any, None, None]:
    fh = _acquire_lock(store_root)
    try:
        yield fh
    finally:
        _release_lock(fh)


def resolve_authoritative_live_blocked(
    *,
    paper_mode: bool = True,
    caller_live_blocked: Optional[bool] = None,
    live_override: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Independent live-execution validation.
    Caller-provided booleans are diagnostic only and cannot authorize live
    or override authoritative blocked/enabled/missing/ambiguous state.
    """
    details: Dict[str, Any] = {
        "caller_live_blocked": caller_live_blocked,
        "caller_non_authoritative": True,
    }
    if live_override is not None:
        payload = dict(live_override)
    else:
        from services.live_block_classifier import evaluate_live_block_status

        payload = evaluate_live_block_status(paper_mode=paper_mode)
    details["authoritative"] = {
        k: payload.get(k)
        for k in (
            "live_blocked",
            "live_block_reason",
            "live_explicitly_authorized",
            "live_env_enabled",
            "paper_mode",
        )
    }
    if "live_blocked" not in payload:
        return (
            False,
            "live_execution_state_missing",
            {**details, "live_state_missing": True},
        )
    if payload.get("live_state_ambiguous") is True:
        return (
            False,
            "live_execution_state_ambiguous",
            {**details, "live_state_ambiguous": True},
        )
    if payload.get("live_contradictory") is True:
        return False, "live_execution_state_contradictory", details
    # Explicitly authorized live in paper context is fail-closed for paper sells.
    if payload.get("live_explicitly_authorized") is True and not paper_mode:
        return False, "live_execution_enabled", details
    if payload.get("live_env_enabled") and payload.get("live_blocked") is False:
        return False, "live_execution_enabled", details
    if payload.get("live_blocked") is not True:
        # Caller True cannot override authoritative unblocked/enabled.
        return False, "live_execution_not_blocked", details
    return True, "live_blocked_ok", details


def _paper_arm_valid(buy_arm_path: Path, now: datetime) -> bool:
    if not buy_arm_path.is_file():
        return False
    doc = _read_json(buy_arm_path)
    if not doc:
        return False
    if doc.get("allow_execute") is not True:
        return False
    mode = str(doc.get("mode") or "").upper()
    if mode not in {"ASSISTED", "ARMED", "PAPER"}:
        # Established paper arm uses ASSISTED.
        if mode and mode not in {"ASSISTED"}:
            return False
    exp = _parse_iso(doc.get("expires_at"))
    if not exp or exp <= now:
        return False
    return True


def load_independent_runtime_evidence(
    *,
    store_root: Path,
    results_root: Optional[Path] = None,
    live_root: Optional[Path] = None,
    now: Optional[datetime] = None,
    caller_live_blocked: Optional[bool] = None,
    live_override: Optional[Dict[str, Any]] = None,
    account_fingerprint_override: Optional[str] = None,
    positions: Optional[Dict[str, Dict[str, float]]] = None,
    open_order_symbols: Optional[Set[str]] = None,
    lifecycle_actions: Optional[Dict[str, str]] = None,
    newer_order_symbols: Optional[Set[str]] = None,
    newer_fill_symbols: Optional[Set[str]] = None,
    newer_completed_plan: Optional[bool] = None,
    newer_batch_supersedes: Optional[bool] = None,
    market_session: Optional[str] = None,
) -> BrokerSnapshotView:
    """
    Build BrokerSnapshotView from independent current evidence paths.
    Never copies account/plan/live fields from the authority document.
    """
    now = now or _utc_now()
    results_root = results_root or (ROOT / "data" / "results")
    live_root = live_root or (ROOT / "data" / "live")

    live_ok, live_reason, live_details = resolve_authoritative_live_blocked(
        paper_mode=True,
        caller_live_blocked=caller_live_blocked,
        live_override=live_override,
    )
    auth_payload = (live_details.get("authoritative") or {}) if live_details else {}

    buy_arm_path = live_root / "paper_arm_confirm.json"
    if store_root and (store_root / "paper_arm_confirm.json").is_file():
        buy_arm_path = store_root / "paper_arm_confirm.json"
    paper_arm_ok = _paper_arm_valid(buy_arm_path, now)

    challenge = _read_json(store_root / "paper_sell_confirmation_challenge.json")
    readiness_path = results_root / "defensive_exit_approval_readiness.json"
    if (store_root / "defensive_exit_approval_readiness.json").is_file():
        readiness_path = store_root / "defensive_exit_approval_readiness.json"
    readiness = _read_json(readiness_path)
    readiness_hash = readiness_packet_hash(readiness) if readiness else None

    plan_path = results_root / "manage_positions_plan.json"
    if (store_root / "manage_positions_plan.json").is_file():
        plan_path = store_root / "manage_positions_plan.json"
    plan = _read_json(plan_path)
    plan_hash = str(plan.get("plan_hash") or plan.get("fingerprint") or "") or None
    # Prefer explicit binding files under store_root for isolated tests.
    bind = _read_json(store_root / "independent_dispatch_bindings.json")
    if bind:
        plan_hash = str(bind.get("plan_hash") or plan_hash or "") or plan_hash

    cycle_id = (
        str(bind.get("cycle_id") or plan.get("cycle_id") or readiness.get("cycle_id") or "") or None
    )
    run_id = str(bind.get("run_id") or plan.get("run_id") or readiness.get("run_id") or "") or None
    batch_id = (
        str(bind.get("batch_id") or plan.get("batch_id") or readiness.get("batch_id") or "") or None
    )
    plan_id = (
        str(bind.get("plan_id") or plan.get("plan_id") or readiness.get("plan_id") or "") or None
    )
    if bind.get("plan_hash"):
        plan_hash = str(bind.get("plan_hash"))
    if bind.get("readiness_packet_hash"):
        readiness_hash = str(bind.get("readiness_packet_hash"))
    if bind.get("account_fingerprint"):
        account_fingerprint_override = str(bind.get("account_fingerprint"))
    trading_date = str(
        bind.get("trading_date") or readiness.get("trading_date") or now.date().isoformat()
    )

    # Newer plan/batch detection from bindings or plan timestamps vs authority later.
    if newer_completed_plan is None:
        newer_completed_plan = bool(bind.get("newer_completed_plan", False))
    if newer_batch_supersedes is None:
        newer_batch_supersedes = bool(bind.get("newer_batch_supersedes", False))

    pos_map: Dict[str, Dict[str, float]] = dict(positions or {})
    if not pos_map:
        pos_csv = results_root / "positions_snapshot.csv"
        if (store_root / "positions_snapshot.csv").is_file():
            pos_csv = store_root / "positions_snapshot.csv"
        if pos_csv.is_file():
            for row in csv.DictReader(pos_csv.read_text(encoding="utf-8").splitlines()):
                sym = str(row.get("symbol") or row.get("ticker") or "").upper()
                if not sym:
                    continue
                try:
                    qty = float(row.get("qty") or 0)
                    avail_raw = row.get("qty_available")
                    avail = float(avail_raw) if avail_raw not in (None, "") else qty
                except Exception:
                    continue
                pos_map[sym] = {"qty": qty, "qty_available": avail}

    open_syms: Set[str] = set(open_order_symbols or set())
    if open_order_symbols is None:
        oo_csv = results_root / "open_orders_snapshot.csv"
        if (store_root / "open_orders_snapshot.csv").is_file():
            oo_csv = store_root / "open_orders_snapshot.csv"
        if oo_csv.is_file():
            for row in csv.DictReader(oo_csv.read_text(encoding="utf-8").splitlines()):
                sym = str(row.get("symbol") or row.get("ticker") or "").upper()
                if sym:
                    open_syms.add(sym)

    acct = account_fingerprint_override
    if not acct:
        acct = str(bind.get("account_fingerprint") or "") or None

    return BrokerSnapshotView(
        positions=pos_map,
        open_order_symbols=open_syms,
        mode="paper",
        live_blocked=bool(live_ok),
        trading_date=trading_date,
        account_fingerprint=acct,
        current_plan_hash=plan_hash,
        newer_completed_plan=bool(newer_completed_plan),
        market_session=market_session or str(bind.get("market_session") or ""),
        paper_buy_arm_valid=paper_arm_ok,
        paper_execution_authorized=paper_arm_ok or bool(bind.get("paper_execution_authorized")),
        cycle_id=cycle_id,
        run_id=run_id,
        batch_id=batch_id,
        plan_id=plan_id,
        readiness_packet_hash=readiness_hash,
        confirmation_challenge_id=str(
            challenge.get("challenge_id") or bind.get("confirmation_challenge_id") or ""
        )
        or None,
        confirmation_challenge_hash=str(
            challenge.get("challenge_hash") or bind.get("confirmation_challenge_hash") or ""
        )
        or None,
        newer_batch_supersedes=bool(newer_batch_supersedes),
        lifecycle_actions=lifecycle_actions or dict(bind.get("lifecycle_actions") or {}),
        newer_order_symbols=set(newer_order_symbols or bind.get("newer_order_symbols") or set()),
        newer_fill_symbols=set(newer_fill_symbols or bind.get("newer_fill_symbols") or set()),
        live_state_source=str(live_reason),
        live_state_ambiguous=bool((live_details or {}).get("live_state_ambiguous")),
        live_state_missing=bool((live_details or {}).get("live_state_missing")),
        evidence_independent=True,
    )


def get_active_paper_sell_permit() -> Optional[PaperSellDispatchPermit]:
    return _ACTIVE_PERMIT.get()


def _durable_permit_path(store_root: Path) -> Path:
    return store_root / DURABLE_PERMIT_NAME


def _write_durable_permit(
    permit: PaperSellDispatchPermit,
    *,
    now: datetime,
    require_full_durability: bool = True,
) -> Dict[str, Any]:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": "PERMIT_ISSUED_PRE_SUBMIT",
        "permit_token": permit.permit_token,
        "dispatch_intent_id": permit.dispatch_intent_id,
        "idempotency_key": permit.idempotency_key,
        "symbols": list(permit.symbols),
        "actions": [{"symbol": s, "action": a, "qty": q} for s, a, q in permit.actions],
        "symbols_submitted": [],
        "issued_at": now.isoformat(),
        "replay_forbidden": True,
        "additional_symbols_forbidden": True,
    }
    _atomic_write_json(
        _durable_permit_path(permit.store_root),
        doc,
        require_full_durability=require_full_durability,
    )
    return doc


def derive_nyse_regular_session(now: datetime) -> Tuple[str, Dict[str, Any]]:
    """
    Authoritative NYSE regular-session derivation for paper sell dispatch.
    Delegates to services.market_calendar.regular_session_state (no parallel calendar).
    Returns (OPEN|CLOSED|UNKNOWN, details). No extended-hours permission.
    """
    from zoneinfo import ZoneInfo

    from services.market_calendar import (
        SESSION_CLOSED,
        SESSION_OPEN,
        SESSION_UNKNOWN,
        regular_session_state,
    )
    from services.paper_sell_dispatch_session_evidence import (
        calendar_payload_to_session_fields,
    )

    try:
        payload = regular_session_state(now=now)
    except Exception as exc:  # noqa: BLE001
        return "UNKNOWN", {"reason": "calendar_error", "error": str(exc)}
    fields = calendar_payload_to_session_fields(payload)
    state = str(fields.get("session_state") or SESSION_UNKNOWN)
    details = {
        "reason": fields.get("reason"),
        "session_open": fields.get("session_open"),
        "session_close": fields.get("session_close"),
        "session_date": fields.get("session_date"),
        "source": fields.get("source"),
        "early_close": False,
    }
    close_raw = fields.get("session_close")
    if close_raw:
        close_dt = _parse_iso(close_raw)
        if close_dt is not None:
            details["early_close"] = bool(
                close_dt.astimezone(ZoneInfo("America/New_York")).hour < 16
            )
    if state == SESSION_OPEN:
        return "OPEN", details
    if state == SESSION_CLOSED:
        return "CLOSED", details
    return "UNKNOWN", details


def evaluate_market_session_for_dispatch(
    now: Optional[datetime],
    *,
    caller_session: Optional[str] = None,
    session_evidence: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Fail-closed market-session gate for paper sell dispatch.
    Independently derives NYSE regular session; caller OPEN cannot override CLOSED/UNKNOWN.
    Missing/stale/naive/invalid evidence blocks.
    """
    details: Dict[str, Any] = {}
    if now is None:
        return False, "market_session_timestamp_missing", details
    if now.tzinfo is None:
        return False, "market_session_naive_timestamp", details
    evidence = dict(session_evidence or {})
    tz_name = evidence.get("market_session_timezone")
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(str(tz_name))
        except Exception as exc:  # noqa: BLE001
            return (
                False,
                "market_session_invalid_timezone",
                {"tz": tz_name, "error": str(exc)},
            )

    derived, ddet = derive_nyse_regular_session(now)
    details.update(ddet)
    details["derived"] = derived
    details["caller_session"] = caller_session

    if derived == "UNKNOWN":
        return False, "market_session_derivation_unknown", details
    if derived != "OPEN":
        if str(caller_session or "").upper() in {"MARKET_OPEN", "OPEN"}:
            return False, "market_session_caller_open_authoritative_closed", details
        return False, "market_session_not_open", details

    observed_raw = evidence.get("market_session_observed_at")
    evidence_session = evidence.get("market_session")
    if evidence_session in (None, "") and observed_raw in (None, ""):
        return False, "market_session_evidence_missing", details
    if observed_raw in (None, ""):
        return False, "market_session_evidence_missing_timestamp", details
    observed = _parse_iso(str(observed_raw))
    if observed is None:
        return False, "market_session_evidence_unparseable_timestamp", details
    if observed.tzinfo is None:
        return False, "market_session_evidence_naive_timestamp", details
    age = (now.astimezone(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
    details["evidence_age_sec"] = age
    if age < -60:
        return False, "market_session_evidence_future", details
    if age > 15 * 60:
        return False, "market_session_evidence_stale", details
    if str(evidence_session or "").upper() not in {"MARKET_OPEN", "OPEN"}:
        return False, "market_session_evidence_not_open", details
    schema = str(evidence.get("schema_version") or "")
    if schema == "198R-J-DISPATCH-SESSION-EVIDENCE-1":
        from services.paper_sell_dispatch_session_evidence import (
            validate_dispatch_session_evidence_document,
            resolve_current_invocation_id,
        )
        from services.market_calendar import regular_session_state

        cal = regular_session_state(now=now)
        ok_doc, doc_reason, doc_details = validate_dispatch_session_evidence_document(
            evidence,
            now=now,
            expected_invocation_id=resolve_current_invocation_id(),
            calendar_payload=cal,
        )
        details.update(doc_details)
        if not ok_doc:
            return False, doc_reason, details
    else:
        env_inv = str(os.environ.get("TRITON_MARKET_OPEN_INVOCATION_ID") or "").strip()
        ev_inv = str(evidence.get("market_open_invocation_id") or "").strip()
        if env_inv and ev_inv and env_inv != ev_inv:
            return False, "session_evidence_invocation_mismatch", details
    return True, "market_session_open", details


def enforce_paper_sell_broker_submission(
    *,
    mode: str,
    side: str,
    source: str = "broker",
    symbol: Optional[str] = None,
    qty: Optional[float] = None,
    client_order_id: Optional[str] = None,
) -> None:
    """
    Canonical broker-facing gate for paper sells.
    BUY/ADD are unaffected. Live mode uses other live gates (not paper sell arm).

    After lock release, submission still requires:
    - in-process ContextVar permit matching durable permit token
    - durable consumption + intent on disk
    - symbol/qty within the consumed batch
    - no symbol replay
    """
    if str(side or "").lower() != "sell":
        return
    if str(mode or "").lower() != "paper":
        if str(mode or "").lower() in {"", "unknown"}:
            raise PaperSellDispatchBlocked(
                "paper_sell_mode_ambiguous",
                {"mode": mode, "source": source},
            )
        return
    permit = _ACTIVE_PERMIT.get()
    if permit is None or not permit.consumed_durable or not permit.permit_token:
        raise PaperSellDispatchBlocked(
            "paper_sell_requires_hardened_dispatch_permit",
            {
                "source": source,
                "hint": "call begin_paper_sell_dispatch before broker submit",
            },
        )
    from services.paper_sell_arm_store import (
        PaperSellArmStoreError,
        require_permit_sqlite_state,
        resolve_runtime_sqlite_path,
        sqlite_permit_grade_enabled,
    )

    if sqlite_permit_grade_enabled():
        try:
            require_permit_sqlite_state(
                resolve_runtime_sqlite_path(db_path=permit.sqlite_db_path),
                dispatch_id=str(permit.dispatch_intent_id),
                permit_token=str(permit.permit_token),
                symbol=str(symbol or ""),
                qty=qty,
            )
        except PaperSellArmStoreError as exc:
            mapped = {
                "sqlite_permit_symbol_already_submitted": "paper_sell_permit_symbol_already_submitted",
                "sqlite_permit_symbol_not_authorized": "paper_sell_permit_symbol_not_authorized",
                "sqlite_permit_quantity_exceeds_authorized": "paper_sell_permit_quantity_exceeds_authorized",
                "sqlite_permit_symbol_required": "paper_sell_permit_symbol_required",
                "sqlite_dispatch_not_committed": "paper_sell_durable_consumption_missing_at_submit",
                "sqlite_authority_not_consumed_at_submit": "paper_sell_authority_not_consumed_at_submit",
            }.get(exc.reason, exc.reason)
            raise PaperSellDispatchBlocked(
                mapped, {**exc.details, "source": source, "sqlite_reason": exc.reason}
            ) from exc
        durable_path = _durable_permit_path(permit.store_root)
        if durable_path.is_file():
            durable = _read_json(durable_path)
            if str(durable.get("permit_token") or "") != str(permit.permit_token):
                raise PaperSellDispatchBlocked(
                    "paper_sell_permit_token_mismatch",
                    {"source": source},
                )
            submitted_list = list(durable.get("symbols_submitted") or [])
            submitted_list.append(str(symbol or "").upper())
            durable["symbols_submitted"] = submitted_list
            durable["last_client_order_id"] = client_order_id
            durable["last_submit_source"] = source
            durable["last_submit_at"] = _utc_now_iso()
            durable["json_role"] = "AUDIT_EXPORT_ONLY"
            _atomic_write_json(durable_path, durable, require_full_durability=False)
        return

    cons = permit.store_root / "paper_sell_arm_consumption.json"
    intent_path = permit.store_root / DISPATCH_INTENT_NAME
    durable_path = _durable_permit_path(permit.store_root)
    if not cons.is_file() or not intent_path.is_file() or not durable_path.is_file():
        raise PaperSellDispatchBlocked(
            "paper_sell_durable_consumption_missing_at_submit",
            {"source": source, "store_root": str(permit.store_root)},
        )
    durable = _read_json(durable_path)
    if str(durable.get("permit_token") or "") != str(permit.permit_token):
        raise PaperSellDispatchBlocked(
            "paper_sell_permit_token_mismatch",
            {"source": source},
        )
    if str(durable.get("dispatch_intent_id") or "") != str(permit.dispatch_intent_id):
        raise PaperSellDispatchBlocked(
            "paper_sell_permit_intent_mismatch",
            {"source": source},
        )
    if str(durable.get("status") or "") in {
        "PERMIT_EXHAUSTED",
        "PERMIT_FINALIZED",
        "PERMIT_REVOKED",
    }:
        raise PaperSellDispatchBlocked(
            "paper_sell_permit_exhausted_or_finalized",
            {"source": source, "status": durable.get("status")},
        )
    auth = _read_json(permit.store_root / HARDENED_AUTHORITY_FILENAME)
    if str(auth.get("consumption_status") or "") != "CONSUMED" and str(
        auth.get("status") or ""
    ) not in {SellArmState.CONSUMED.value, SellArmState.DISPATCH_IN_PROGRESS.value}:
        raise PaperSellDispatchBlocked(
            "paper_sell_authority_not_consumed_at_submit",
            {"source": source},
        )
    sym = str(symbol or "").upper()
    if not sym:
        raise PaperSellDispatchBlocked(
            "paper_sell_permit_symbol_required",
            {"source": source},
        )
    allowed = {str(a.get("symbol") or "").upper(): a for a in (durable.get("actions") or [])}
    if sym not in allowed:
        raise PaperSellDispatchBlocked(
            "paper_sell_permit_symbol_not_authorized",
            {"source": source, "symbol": sym},
        )
    submitted = {str(s).upper() for s in (durable.get("symbols_submitted") or [])}
    if sym in submitted:
        raise PaperSellDispatchBlocked(
            "paper_sell_permit_symbol_already_submitted",
            {"source": source, "symbol": sym},
        )
    if qty is not None:
        max_q = float(allowed[sym].get("qty") or 0)
        if float(qty) > max_q + 1e-9:
            raise PaperSellDispatchBlocked(
                "paper_sell_permit_quantity_exceeds_authorized",
                {"source": source, "symbol": sym, "qty": qty, "max_qty": max_q},
            )
    # Mark symbol submitted (one-use per symbol in batch) before returning to broker.
    submitted_list = list(durable.get("symbols_submitted") or [])
    submitted_list.append(sym)
    durable["symbols_submitted"] = submitted_list
    durable["last_client_order_id"] = client_order_id
    durable["last_submit_source"] = source
    durable["last_submit_at"] = _utc_now_iso()
    remaining = set(allowed) - set(submitted_list)
    if not remaining:
        durable["status"] = "PERMIT_EXHAUSTED"
    else:
        durable["status"] = "PERMIT_PARTIALLY_USED"
    _atomic_write_json(durable_path, durable)


def _classify_prior_dispatch_intent(
    intent_path: Path,
    authority: Dict[str, Any],
    *,
    db_path: Optional[Path] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify durable prior-intent evidence against the current authority + SQLite.

    Returns (classification, reason, details). Blocking classifications have a reason.
    foreign_historical_terminal and no_intent are non-blocking.
    """
    from services.paper_sell_arm_store import (
        INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL,
        INTENT_CLASS_NO_INTENT,
        classify_dispatch_intent_for_authority,
    )

    if not intent_path.is_file():
        return INTENT_CLASS_NO_INTENT, "", {"path": str(intent_path)}
    try:
        raw = intent_path.read_text(encoding="utf-8")
    except OSError as exc:
        return (
            "ambiguous",
            "corrupt_dispatch_intent",
            {
                "error": str(exc),
                "path": str(intent_path),
            },
        )
    try:
        existing = json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError as exc:
        return (
            "ambiguous",
            "corrupt_dispatch_intent",
            {
                "error": str(exc),
                "path": str(intent_path),
            },
        )
    if not isinstance(existing, dict):
        return (
            "ambiguous",
            "corrupt_dispatch_intent",
            {
                "error": "intent_not_object",
                "path": str(intent_path),
            },
        )
    required = ("dispatch_intent_id", "idempotency_key", "status", "symbols")
    missing = [k for k in required if existing.get(k) in (None, "", [], {})]
    if missing:
        return (
            "ambiguous",
            "incomplete_dispatch_intent",
            {
                "missing": missing,
                "intent": existing,
            },
        )
    classification, reason, details = classify_dispatch_intent_for_authority(
        existing, authority, db_path=db_path
    )
    details = {**details, "intent": existing, "path": str(intent_path)}
    return classification, reason, details


def find_unused_pre_plan_authority(
    *,
    db_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Read-only: newest UNUSED authority suitable for pre-plan dispatch."""
    from services.paper_sell_arm_store import (
        inspect_sqlite_authority,
        resolve_runtime_sqlite_path,
    )

    now = now or _utc_now()
    path = resolve_runtime_sqlite_path(db_path=db_path)
    status = inspect_sqlite_authority(path, now=now)
    if str(status.get("state") or "") == "UNUSED":
        payload = status.get("payload")
        return payload if isinstance(payload, dict) else None
    return None


def validate_pre_plan_authority_bound_artifacts(
    authority: Mapping[str, Any],
    *,
    results_root: Path,
    now: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Fail-closed validation against on-disk authority-bound artifacts (pre-plan)."""
    now = now or _utc_now()
    details: Dict[str, Any] = {"phase": "pre_plan_artifact_validation"}
    if not authority:
        return False, "authority_missing", details
    if str(authority.get("consumption_status") or "") != "UNUSED":
        return False, "authority_not_unused", details
    if str(authority.get("status") or "") != SellArmState.ARMED.value:
        return False, "status_not_armed", details
    if str(authority.get("mode") or "").upper() != "PAPER":
        return False, "mode_not_paper", details
    if str(authority.get("live_execution_required_state") or "").upper() != "BLOCKED":
        return False, "live_execution_not_blocked", details
    if authority.get("live_authorized") is True:
        return False, "live_authorized_true", details
    expires = _parse_iso(authority.get("expires_at"))
    if expires is None or expires <= now:
        return False, "authority_expired", details
    td = str(authority.get("trading_date") or "")
    if td and td != now.astimezone(timezone.utc).date().isoformat():
        return False, "trading_date_mismatch", details

    scope = list(authority.get("authorized_actions") or authority.get("authorized_scope") or [])
    if not scope:
        return False, "empty_authorized_scope", details
    for item in scope:
        action = str(item.get("action") or "").upper()
        if action in BUY_ACTIONS:
            return False, "buy_action_in_sell_authority", details
        if action not in SELL_ACTIONS:
            return False, f"invalid_sell_action:{action}", details

    try:
        total_stored = float(authority.get("total_batch_max_notional") or 0.0)
    except (TypeError, ValueError):
        return False, "total_notional_unparseable", details
    if total_stored > APPROVAL_CLI_CAP + 1e-9:
        return False, "total_notional_exceeds_approval_cli_cap", details

    packet = _load_readiness_packet(results_root, LIVE_GOVERNANCE)
    if not packet:
        return False, "source_readiness_packet_missing", details
    fresh_ok, fresh_reason, _gen = evaluate_readiness_packet_freshness(packet, now=now)
    if not fresh_ok:
        return False, fresh_reason, details
    if str(readiness_packet_hash(packet)) != str(authority.get("readiness_packet_hash") or ""):
        return False, "readiness_packet_hash_mismatch", details

    if is_dual_identity_record(authority):
        if str(source_readiness_packet_hash(packet)) != str(
            authority.get("source_readiness_packet_hash") or ""
        ):
            return False, "source_readiness_packet_hash_mismatch", details
        inv = packet.get("market_open_invocation_id") or packet.get("invocation_id")
        if str(authority.get("market_open_invocation_id") or "") != str(inv or ""):
            return False, "invocation_mismatch", details
        if str(authority.get("source_generated_at") or "") != str(packet.get("generated_at") or ""):
            return False, "generated_at_mismatch", details
        if str(authority.get("source_plan_sha256") or "") != str(
            packet.get("source_plan_sha256") or ""
        ):
            return False, "source_plan_hash_mismatch", details
        if str(authority.get("selected_row_hash") or "") != str(
            packet.get("selected_row_hash") or ""
        ):
            return False, "selected_row_hash_mismatch", details
        from services.paper_sell_arm_cap_slice import (
            authorized_slice_hash as _slice_hash,
            validate_slice_against_packet,
        )

        slice_doc = {
            "slice_actions": scope,
            "authorized_slice_hash": authority.get("authorized_slice_hash"),
            "slice_hash": authority.get("authorized_slice_hash"),
        }
        ok_slice, slice_reason, validated_scope = validate_slice_against_packet(
            packet,
            slice_doc,
            cap=float(authority.get("applied_cap") or APPROVAL_CLI_CAP),
            cap_source=str(authority.get("applied_cap_source") or "APPROVAL_CLI_CAP"),
        )
        if not ok_slice:
            return False, slice_reason, details
        expected_slice = str(authority.get("authorized_slice_hash") or "")
        got_slice = _slice_hash(
            scope=validated_scope,
            slice_total=round(
                sum(float(s.get("max_notional") or 0.0) for s in validated_scope),
                2,
            ),
            applied_cap=float(authority.get("applied_cap") or APPROVAL_CLI_CAP),
            cap_source=str(authority.get("applied_cap_source") or "APPROVAL_CLI_CAP"),
        )
        if expected_slice and expected_slice != got_slice:
            return False, "authorized_slice_hash_mismatch", details

    from services.manage_positions_cycle_binding import (
        selected_row_hash,
        sha256_file_bytes,
    )

    results_root = Path(results_root)
    csv_path = results_root / "manage_positions_plan.csv"
    json_path = results_root / "manage_positions_plan.json"
    on_disk_source = sha256_file_bytes(csv_path) or sha256_file_bytes(json_path)
    if not on_disk_source:
        return False, "source_plan_artifact_missing", details
    if str(authority.get("source_plan_sha256") or "") != str(on_disk_source):
        return False, "on_disk_source_plan_sha256_mismatch", details
    if csv_path.is_file():
        rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
        row_hash = selected_row_hash(rows)
        if str(authority.get("selected_row_hash") or "") != str(row_hash):
            return False, "on_disk_selected_row_hash_mismatch", details

    return True, "ok", details


def build_authority_bound_broker_view(
    authority: Mapping[str, Any],
    *,
    broker: Any,
    results_root: Path,
    now: Optional[datetime] = None,
) -> BrokerSnapshotView:
    """Independent evidence frozen to authority bindings (pre-plan; no new plan write)."""
    now = now or _utc_now()
    evidence = load_independent_runtime_evidence(
        store_root=LIVE_GOVERNANCE,
        results_root=results_root,
        now=now,
        caller_live_blocked=True,
    )
    return BrokerSnapshotView(
        positions=evidence.positions,
        open_order_symbols=evidence.open_order_symbols,
        mode="paper",
        live_blocked=True,
        trading_date=str(authority.get("trading_date") or now.date().isoformat()),
        account_fingerprint=str(authority.get("account_fingerprint") or "") or None,
        current_plan_hash=str(authority.get("plan_hash") or "") or None,
        newer_completed_plan=False,
        market_session=evidence.market_session or "MARKET_OPEN",
        paper_buy_arm_valid=evidence.paper_buy_arm_valid,
        paper_execution_authorized=evidence.paper_execution_authorized,
        cycle_id=str(authority.get("cycle_id") or "") or None,
        run_id=str(authority.get("run_id") or "") or None,
        batch_id=str(authority.get("batch_id") or "") or None,
        plan_id=str(authority.get("plan_id") or "") or None,
        readiness_packet_hash=str(authority.get("readiness_packet_hash") or "") or None,
        confirmation_challenge_id=str(authority.get("confirmation_challenge_id") or "") or None,
        confirmation_challenge_hash=str(authority.get("confirmation_challenge_hash") or "") or None,
        newer_batch_supersedes=False,
        lifecycle_actions=evidence.lifecycle_actions,
        newer_order_symbols=evidence.newer_order_symbols,
        newer_fill_symbols=evidence.newer_fill_symbols,
        live_state_source=evidence.live_state_source,
        live_state_ambiguous=evidence.live_state_ambiguous,
        live_state_missing=evidence.live_state_missing,
        evidence_independent=True,
    )


def build_pre_plan_planned_orders(
    authority: Mapping[str, Any],
    broker: Any,
    *,
    now: Optional[datetime] = None,
) -> Tuple[bool, str, Tuple[Any, ...]]:
    """Build exact authorized-slice planned rows using current broker evidence."""
    from services.place_live_orders import PlannedOrder, get_ref_price, normalize_symbol

    now = now or _utc_now()
    scope = list(authority.get("authorized_actions") or authority.get("authorized_scope") or [])
    per_qty_cap = {
        str(k).upper(): float(v)
        for k, v in (authority.get("per_symbol_max_quantity") or {}).items()
    }
    per_notional_cap = {
        str(k).upper(): float(v)
        for k, v in (authority.get("per_symbol_max_notional") or {}).items()
    }
    positions: Dict[str, float] = {}
    try:
        for pos in broker.list_positions() or []:
            sym = normalize_symbol(pos.get("symbol"))
            if not sym:
                continue
            try:
                avail = float(pos.get("qty_available") or pos.get("qty") or 0)
            except (TypeError, ValueError):
                avail = 0.0
            positions[sym] = avail
    except Exception:
        pass

    planned: List[Any] = []
    total_notional = 0.0
    stored_total = authority.get("total_batch_max_notional")
    batch_ceiling = APPROVAL_CLI_CAP
    if stored_total is not None:
        batch_ceiling = min(float(batch_ceiling), float(money_round(float(stored_total))))
    for item in scope:
        sym = normalize_symbol(item.get("symbol"))
        if not sym:
            return False, "invalid_symbol_in_scope", ()
        action = str(item.get("action") or "").upper()
        if action in BUY_ACTIONS:
            return False, "buy_action_in_authorized_scope", ()
        max_qty = float(item.get("max_qty") or 0)
        if max_qty <= 0:
            return False, f"invalid_authorized_qty:{sym}", ()
        held = float(positions.get(sym, 0.0))
        if held + 1e-9 < max_qty:
            return False, f"insufficient_current_position:{sym}", ()
        price_raw = get_ref_price(broker, sym, "sell")
        if _price_unavailable(price_raw):
            return False, f"price_unavailable:{sym}", ()
        price = float(price_raw)
        qty_cap = per_qty_cap.get(sym)
        if qty_cap is not None and max_qty > float(qty_cap) + 1e-9:
            return False, f"quantity_expansion_vs_symbol_ceiling:{sym}", ()
        stored_n = item.get("max_notional")
        runtime_qty, clip_reason = _downward_runtime_qty(max_qty, stored_n, price)
        if runtime_qty <= 0:
            client_order_id = f"triton-preplan-{sym}-{action}-{uuid.uuid4().hex[:10]}"
            planned.append(
                PlannedOrder(
                    symbol=sym,
                    side="sell",
                    qty=0,
                    order_type="limit",
                    time_in_force="day",
                    limit_price=price,
                    ref_price=price,
                    client_order_id=client_order_id,
                    discipline_allowed=True,
                )
            )
            setattr(planned[-1], "management_action", action)
            setattr(planned[-1], "authorized_qty", max_qty)
            setattr(planned[-1], "clip_reason", clip_reason)
            continue
        notional = money_round(float(runtime_qty) * price)
        cap_n = per_notional_cap.get(sym)
        if cap_n is not None and notional > money_round(cap_n) + 1e-9:
            runtime_qty, clip_reason = _downward_runtime_qty(runtime_qty, cap_n, price)
            if runtime_qty <= 0:
                client_order_id = f"triton-preplan-{sym}-{action}-{uuid.uuid4().hex[:10]}"
                planned.append(
                    PlannedOrder(
                        symbol=sym,
                        side="sell",
                        qty=0,
                        order_type="limit",
                        time_in_force="day",
                        limit_price=price,
                        ref_price=price,
                        client_order_id=client_order_id,
                        discipline_allowed=True,
                    )
                )
                setattr(planned[-1], "management_action", action)
                setattr(planned[-1], "authorized_qty", max_qty)
                setattr(planned[-1], "clip_reason", clip_reason)
                continue
            notional = money_round(float(runtime_qty) * price)
        remaining = money_round(float(batch_ceiling) - float(total_notional))
        if remaining < 0:
            remaining = 0.0
        if runtime_qty > 0:
            qty_vs_rem, _rem_why = _downward_runtime_qty(runtime_qty, remaining, price)
            if qty_vs_rem < runtime_qty:
                runtime_qty = qty_vs_rem
                if runtime_qty <= 0:
                    clip_reason = "clipped_to_zero_vs_remaining_batch_cap"
                    client_order_id = f"triton-preplan-{sym}-{action}-{uuid.uuid4().hex[:10]}"
                    planned.append(
                        PlannedOrder(
                            symbol=sym,
                            side="sell",
                            qty=0,
                            order_type="limit",
                            time_in_force="day",
                            limit_price=price,
                            ref_price=price,
                            client_order_id=client_order_id,
                            discipline_allowed=True,
                        )
                    )
                    setattr(planned[-1], "management_action", action)
                    setattr(planned[-1], "authorized_qty", max_qty)
                    setattr(planned[-1], "clip_reason", clip_reason)
                    continue
                clip_reason = "downward_clip_vs_remaining_batch_cap"
                notional = money_round(float(runtime_qty) * price)
        if notional > APPROVAL_CLI_CAP + 1e-6:
            return False, "batch_notional_exceeds_approval_cli_cap", ()
        client_order_id = f"triton-preplan-{sym}-{action}-{uuid.uuid4().hex[:10]}"
        planned.append(
            PlannedOrder(
                symbol=sym,
                side="sell",
                qty=int(runtime_qty),
                order_type="limit",
                time_in_force="day",
                limit_price=price,
                ref_price=price,
                client_order_id=client_order_id,
                discipline_allowed=True,
            )
        )
        setattr(planned[-1], "management_action", action)
        setattr(planned[-1], "authorized_qty", max_qty)
        setattr(planned[-1], "clip_reason", clip_reason)
        total_notional = money_round(total_notional + notional)

    if not any(float(getattr(p, "qty", 0) or 0) > 0 for p in planned):
        return False, "no_executable_authorized_quantity", ()
    stored_total = authority.get("total_batch_max_notional")
    if stored_total is not None and total_notional > money_round(float(stored_total)) + 1e-9:
        return False, "notional_expansion_vs_authorized_slice_total", ()
    if total_notional > APPROVAL_CLI_CAP + 1e-6:
        return False, "batch_notional_exceeds_approval_cli_cap", ()
    return True, "ok", tuple(planned)


def attempt_pre_plan_authorized_sell_dispatch(
    *,
    authority: Mapping[str, Any],
    broker: Any,
    results_root: Path,
    now: Optional[datetime] = None,
    explicit_runtime_write: bool = True,
    db_path: Optional[Path] = None,
) -> DispatchGuardResult:
    """Validate authority-bound artifacts and consume for pre-plan dispatch."""
    now = now or _utc_now()
    ok, reason, _details = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=results_root, now=now
    )
    if not ok:
        return DispatchGuardResult(False, reason, _details)
    ok_planned, plan_reason, planned = build_pre_plan_planned_orders(authority, broker, now=now)
    clip_audit = _authorized_clip_audit(authority, planned)
    if not ok_planned or not planned:
        return DispatchGuardResult(
            False,
            plan_reason,
            {"planned_count": len(planned), "clip_audit": clip_audit},
            consumed=False,
        )
    broker_view = build_authority_bound_broker_view(
        authority, broker=broker, results_root=results_root, now=now
    )
    guard = begin_paper_sell_dispatch(
        store_root=LIVE_GOVERNANCE,
        planned=planned,
        broker_view=broker_view,
        mode="paper",
        readiness_packet_hash_current=str(authority.get("readiness_packet_hash") or ""),
        explicit_runtime_write=explicit_runtime_write,
        now=now,
        caller_live_blocked=True,
        results_root=results_root,
        db_path=db_path,
        clip_audit=clip_audit,
    )
    guard.details = {**dict(guard.details or {}), "clip_audit": clip_audit}
    return guard


def begin_paper_sell_dispatch(
    *,
    store_root: Path,
    planned: Sequence[Any],
    broker_view: Optional[BrokerSnapshotView] = None,
    mode: str = "paper",
    readiness_packet_hash_current: Optional[str] = None,
    explicit_runtime_write: bool = False,
    now: Optional[datetime] = None,
    caller_live_blocked: Optional[bool] = None,
    live_override: Optional[Dict[str, Any]] = None,
    results_root: Optional[Path] = None,
    live_root: Optional[Path] = None,
    activate_permit: bool = True,
    db_path: Optional[Path] = None,
    clip_audit: Optional[Sequence[Mapping[str, Any]]] = None,
) -> DispatchGuardResult:
    """
    Final revalidation + durable dispatch intent + irreversible consumption.
    Call only when a broker submission attempt is about to begin.
    """
    now = now or _utc_now()
    if now.tzinfo is None:
        return DispatchGuardResult(
            False,
            "market_session_naive_timestamp",
            {"now": str(now)},
        )
    if str(mode).lower() != "paper":
        return DispatchGuardResult(False, "mode_not_paper", {"mode": mode})
    if not planned_rows_include_paper_sells(planned):
        return DispatchGuardResult(True, NO_SELLS, {"skipped": True}, consumed=False)

    paths = _assert_writable_root(store_root, explicit_runtime_write=explicit_runtime_write)
    from services.paper_sell_arm_store import (
        resolve_runtime_sqlite_path as _resolve_sqlite,
    )

    sqlite_path = _resolve_sqlite(db_path=db_path)

    # Preliminary independent evidence (diagnostic); race-sensitive fields reloaded in lock.
    prelim = broker_view or load_independent_runtime_evidence(
        store_root=paths.root,
        results_root=results_root,
        live_root=live_root,
        now=now,
        caller_live_blocked=caller_live_blocked,
        live_override=live_override,
    )

    with _PROCESS_LOCK:
        fh = _acquire_lock(paths.root)
        try:
            ok, reason, details = validate_runtime_sell_authority(
                contract_path=paths.authority,
                now=now,
                consumption_path=paths.consumption,
                store_root=paths.root,
                db_path=sqlite_path,
            )
            if not ok:
                append_audit(
                    paths,
                    {"event": "dispatch_blocked", "reason": reason, "details": details},
                )
                return DispatchGuardResult(False, reason, details)

            from services.paper_sell_arm_store import (
                load_sqlite_authority_payload,
                sqlite_permit_grade_enabled as _sqlite_enabled,
            )

            authority = _read_json(paths.authority)
            if _sqlite_enabled():
                sqlite_doc = load_sqlite_authority_payload(sqlite_path, now=now)
                if sqlite_doc:
                    authority = sqlite_doc
            req = requested_actions_from_planned(planned)
            packet = _load_readiness_packet(results_root, paths.root)
            selected_planned: Tuple[Any, ...] = tuple(p for p in planned if _planned_is_sell(p))
            if is_dual_identity_record(authority):
                if not packet:
                    return DispatchGuardResult(False, "source_readiness_packet_missing", {})
                current_src = source_readiness_packet_hash(packet)
                if str(authority.get("source_readiness_packet_hash") or "") != current_src:
                    return DispatchGuardResult(False, "source_readiness_packet_hash_changed", {})
                current_full = readiness_packet_hash(packet)
                if str(authority.get("readiness_packet_hash") or "") != current_full:
                    return DispatchGuardResult(False, "readiness_packet_hash_changed", {})
                inv = packet.get("market_open_invocation_id") or packet.get("invocation_id")
                if str(authority.get("market_open_invocation_id") or "") != str(inv or ""):
                    return DispatchGuardResult(False, "invocation_changed", {})
                if str(authority.get("source_plan_sha256") or "") != str(
                    packet.get("source_plan_sha256") or ""
                ):
                    return DispatchGuardResult(False, "source_plan_hash_changed", {})
                if str(authority.get("selected_row_hash") or "") != str(
                    packet.get("selected_row_hash") or ""
                ):
                    return DispatchGuardResult(False, "selected_row_hash_changed", {})
                if str(authority.get("source_generated_at") or "") != str(
                    packet.get("generated_at") or ""
                ):
                    return DispatchGuardResult(False, "generated_at_changed", {})
                from services.paper_sell_arm_cap_slice import (
                    authorized_slice_hash as _slice_hash,
                    validate_slice_against_packet,
                )

                slice_doc = {
                    "slice_actions": authority.get("authorized_actions")
                    or authority.get("authorized_scope")
                    or [],
                    "authorized_slice_hash": authority.get("authorized_slice_hash"),
                    "slice_hash": authority.get("authorized_slice_hash"),
                }
                ok_slice, slice_reason, _scope = validate_slice_against_packet(
                    packet,
                    slice_doc,
                    cap=float(authority.get("applied_cap") or APPROVAL_CLI_CAP),
                    cap_source=str(authority.get("applied_cap_source") or "APPROVAL_CLI_CAP"),
                )
                if not ok_slice:
                    return DispatchGuardResult(False, slice_reason, {})
                expected_slice = str(authority.get("authorized_slice_hash") or "")
                got_slice = _slice_hash(
                    scope=_scope,
                    slice_total=round(sum(float(s.get("max_notional") or 0.0) for s in _scope), 2),
                    applied_cap=float(authority.get("applied_cap") or APPROVAL_CLI_CAP),
                    cap_source=str(authority.get("applied_cap_source") or "APPROVAL_CLI_CAP"),
                )
                if expected_slice and expected_slice != got_slice:
                    return DispatchGuardResult(False, "authorized_slice_hash_changed", {})
                ok_sel, sel_reason, selected_planned = select_authorized_planned(
                    planned,
                    authority,
                    reject_unapproved_planned=clip_audit is not None,
                )
                if not ok_sel:
                    return DispatchGuardResult(False, sel_reason, {})
                req = requested_actions_from_planned(selected_planned)
            else:
                amb = legacy_ambiguous_slice_authority(authority, readiness_packet=packet or None)
                if amb:
                    return DispatchGuardResult(False, amb, {})
            if not req:
                return DispatchGuardResult(True, NO_SELLS, {"skipped": True})

            # Reload independent evidence inside the lock.
            evidence = load_independent_runtime_evidence(
                store_root=paths.root,
                results_root=results_root,
                live_root=live_root,
                now=now,
                caller_live_blocked=caller_live_blocked,
                live_override=live_override,
                positions=prelim.positions if broker_view else None,
                open_order_symbols=prelim.open_order_symbols if broker_view else None,
            )
            # If caller supplied a fully independent broker_view (tests), prefer its
            # race-sensitive fields but still force independent live resolution.
            if broker_view is not None and broker_view.evidence_independent:
                live_ok, live_reason, live_details = resolve_authoritative_live_blocked(
                    paper_mode=True,
                    caller_live_blocked=caller_live_blocked,
                    live_override=live_override,
                )
                evidence = BrokerSnapshotView(
                    positions=broker_view.positions,
                    open_order_symbols=broker_view.open_order_symbols,
                    mode="paper",
                    live_blocked=bool(live_ok),
                    trading_date=broker_view.trading_date,
                    account_fingerprint=broker_view.account_fingerprint,
                    current_plan_hash=broker_view.current_plan_hash,
                    newer_completed_plan=broker_view.newer_completed_plan,
                    market_session=broker_view.market_session,
                    paper_buy_arm_valid=broker_view.paper_buy_arm_valid,
                    paper_execution_authorized=broker_view.paper_execution_authorized
                    or broker_view.paper_buy_arm_valid,
                    cycle_id=broker_view.cycle_id,
                    run_id=broker_view.run_id,
                    batch_id=broker_view.batch_id,
                    plan_id=broker_view.plan_id,
                    readiness_packet_hash=broker_view.readiness_packet_hash,
                    confirmation_challenge_id=broker_view.confirmation_challenge_id,
                    confirmation_challenge_hash=broker_view.confirmation_challenge_hash,
                    newer_batch_supersedes=broker_view.newer_batch_supersedes,
                    lifecycle_actions=broker_view.lifecycle_actions,
                    newer_order_symbols=broker_view.newer_order_symbols,
                    newer_fill_symbols=broker_view.newer_fill_symbols,
                    live_state_source=live_reason,
                    live_state_ambiguous=bool(live_details.get("live_state_ambiguous")),
                    live_state_missing=bool(live_details.get("live_state_missing")),
                    evidence_independent=True,
                )

            total_n = money_round(sum(float(r.get("notional") or 0) for r in req))
            if is_dual_identity_record(authority):
                stored_total = authority.get("total_batch_max_notional")
                if stored_total is not None and total_n > money_round(stored_total) + 1e-9:
                    return DispatchGuardResult(
                        False,
                        "notional_expansion_vs_authorized_slice_total",
                        {"total_notional": total_n, "authorized_total": stored_total},
                    )
            if total_n > APPROVAL_CLI_CAP + 1e-6:
                return DispatchGuardResult(
                    False,
                    "batch_notional_exceeds_approval_cli_cap",
                    {"total_notional": total_n, "cap": APPROVAL_CLI_CAP},
                )

            # Market-session policy: independently derived + fresh evidence; rechecked in lock.
            bind_doc = _read_json(paths.root / "independent_dispatch_bindings.json")
            caller_session = (
                broker_view.market_session if broker_view is not None else evidence.market_session
            )
            sess_ok, sess_reason, sess_details = evaluate_market_session_for_dispatch(
                now,
                caller_session=caller_session,
                session_evidence=bind_doc,
            )
            if not sess_ok:
                append_audit(
                    paths,
                    {
                        "event": "dispatch_blocked_market_session",
                        "reason": sess_reason,
                        "details": sess_details,
                    },
                )
                return DispatchGuardResult(False, sess_reason, sess_details)
            evidence = BrokerSnapshotView(
                positions=evidence.positions,
                open_order_symbols=evidence.open_order_symbols,
                mode=evidence.mode,
                live_blocked=evidence.live_blocked,
                trading_date=evidence.trading_date,
                account_fingerprint=evidence.account_fingerprint,
                current_plan_hash=evidence.current_plan_hash,
                newer_completed_plan=evidence.newer_completed_plan,
                market_session="MARKET_OPEN",
                paper_buy_arm_valid=evidence.paper_buy_arm_valid,
                paper_execution_authorized=evidence.paper_execution_authorized,
                cycle_id=evidence.cycle_id,
                run_id=evidence.run_id,
                batch_id=evidence.batch_id,
                plan_id=evidence.plan_id,
                readiness_packet_hash=evidence.readiness_packet_hash,
                confirmation_challenge_id=evidence.confirmation_challenge_id,
                confirmation_challenge_hash=evidence.confirmation_challenge_hash,
                newer_batch_supersedes=evidence.newer_batch_supersedes,
                lifecycle_actions=evidence.lifecycle_actions,
                newer_order_symbols=evidence.newer_order_symbols,
                newer_fill_symbols=evidence.newer_fill_symbols,
                live_state_source=evidence.live_state_source,
                live_state_ambiguous=evidence.live_state_ambiguous,
                live_state_missing=evidence.live_state_missing,
                evidence_independent=True,
            )

            rok, rreason, rdetails = revalidate_before_dispatch(
                authority,
                broker=evidence,
                requested_actions=req,
                now=now,
                consumption=_read_json(paths.consumption) or None,
                readiness_packet_hash_current=readiness_packet_hash_current
                or evidence.readiness_packet_hash,
                clip_audit=clip_audit,
                runtime_ref_prices=(
                    {
                        str(getattr(p, "symbol", "") or "").upper(): float(
                            getattr(p, "limit_price", None) or getattr(p, "ref_price", None) or 0
                        )
                        for p in planned
                        if _planned_is_sell(p)
                    }
                    if clip_audit is not None
                    else None
                ),
            )
            if not rok:
                append_audit(
                    paths,
                    {"event": "dispatch_revalidation_failed", "reason": rreason},
                )
                return DispatchGuardResult(False, rreason, rdetails)

            intent_path = paths.root / DISPATCH_INTENT_NAME
            from services.paper_sell_arm_store import (
                INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL,
                INTENT_CLASS_NO_INTENT,
            )

            intent_class, intent_reason, intent_details = _classify_prior_dispatch_intent(
                intent_path, authority, db_path=sqlite_path
            )
            preserve_historical_intent = intent_class == INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL
            if intent_class not in {
                INTENT_CLASS_NO_INTENT,
                INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL,
            }:
                append_audit(
                    paths,
                    {
                        "event": "dispatch_blocked_prior_intent",
                        "reason": intent_reason,
                        "classification": intent_class,
                        "details": intent_details,
                    },
                )
                return DispatchGuardResult(False, intent_reason, intent_details)

            intent_id = f"sell-dispatch-{uuid.uuid4().hex[:16]}"
            intent = {
                "schema_version": SCHEMA_VERSION,
                "dispatch_intent_id": intent_id,
                "audit_id": authority.get("audit_id"),
                "authority_id": authority.get("authority_id") or authority.get("audit_id"),
                "idempotency_key": authority.get("idempotency_key"),
                "plan_hash": authority.get("plan_hash"),
                "batch_id": authority.get("batch_id"),
                "symbols": [r["symbol"] for r in req],
                "actions": req,
                "clip_audit": list(clip_audit) if clip_audit is not None else None,
                "recorded_at": now.isoformat(),
                "status": "INTENT_RECORDED_PRE_SUBMIT",
                "broker_submission_started": False,
                "order_intent_ids": [
                    f"sell-oi-{authority.get('audit_id')}-{r['symbol']}-{r['action']}-{r['qty']}"
                    for r in req
                ],
            }
            from services.paper_sell_arm_store import (
                PaperSellArmStoreError,
                batch_hash_from_scope,
                consume_authority_insert_dispatch,
                sqlite_permit_grade_enabled,
            )

            if sqlite_permit_grade_enabled():
                try:
                    committed = consume_authority_insert_dispatch(
                        sqlite_path,
                        authority_id=str(
                            authority.get("authority_id") or authority.get("audit_id") or ""
                        ),
                        plan_hash=str(authority.get("plan_hash") or ""),
                        batch_hash=batch_hash_from_scope(
                            authority.get("authorized_actions")
                            or authority.get("authorized_scope")
                            or []
                        ),
                        expected_scope=authority.get("authorized_actions")
                        or authority.get("authorized_scope")
                        or [],
                        mode=mode,
                        live_blocked=bool(getattr(evidence, "live_blocked", False)),
                        requested=req,
                        now=now,
                        clip_audit=list(clip_audit) if clip_audit is not None else None,
                    )
                    intent_id = str(committed["dispatch_id"])
                    intent["dispatch_intent_id"] = intent_id
                    intent["authority_id"] = committed["authority_id"]
                except PaperSellArmStoreError as exc:
                    append_audit(
                        paths,
                        {
                            "event": "dispatch_blocked_sqlite",
                            "reason": exc.reason,
                            "details": exc.details,
                        },
                    )
                    return DispatchGuardResult(False, exc.reason, exc.details)
                consumption = mark_authority_dispatch_attempt(
                    store_root=paths.root,
                    authority=authority,
                    result="STARTED",
                    explicit_runtime_write=explicit_runtime_write,
                    now=now,
                    require_full_durability=False,
                    db_path=sqlite_path,
                )
                authority_after = _read_json(paths.authority) or authority
                intent["broker_submission_started"] = True
                intent["consumption_status"] = "CONSUMED"
                intent["json_role"] = "AUDIT_EXPORT_ONLY"
                if not preserve_historical_intent:
                    _atomic_write_json(intent_path, intent, require_full_durability=False)
                append_audit(
                    paths,
                    {
                        "event": "dispatch_intent_consumed_pre_submit",
                        "dispatch_intent_id": intent_id,
                        "backend": "sqlite",
                        "plan_hash": authority.get("plan_hash"),
                        "symbols": intent["symbols"],
                        "clip_audit": list(clip_audit) if clip_audit is not None else None,
                        "preserve_historical_intent": preserve_historical_intent,
                    },
                )
                permit_token = f"sell-permit-{uuid.uuid4().hex}"
                permit = PaperSellDispatchPermit(
                    dispatch_intent_id=intent_id,
                    idempotency_key=str(authority.get("idempotency_key") or ""),
                    store_root=paths.root,
                    consumed_durable=True,
                    symbols=tuple(intent["symbols"]),
                    actions=tuple(
                        (str(r["symbol"]), str(r["action"]), float(r["qty"])) for r in req
                    ),
                    permit_token=permit_token,
                    sqlite_db_path=sqlite_path,
                )
                _write_durable_permit(permit, now=now, require_full_durability=False)
                if activate_permit:
                    _ACTIVE_PERMIT.set(permit)
                return DispatchGuardResult(
                    True,
                    "ok",
                    {
                        "revalidation": rdetails,
                        "consumption": consumption,
                        "intent": intent,
                        "market_session": sess_details,
                        "authority_backend": "sqlite",
                    },
                    dispatch_intent_id=intent_id,
                    authority=authority_after,
                    consumed=True,
                    permit=permit,
                    authorized_planned=selected_planned,
                )

            try:
                if not preserve_historical_intent:
                    _atomic_write_json(intent_path, intent, require_full_durability=True)

                consumption = mark_authority_dispatch_attempt(
                    store_root=paths.root,
                    authority=authority,
                    result="STARTED",
                    explicit_runtime_write=explicit_runtime_write,
                    now=now,
                    require_full_durability=True,
                    db_path=sqlite_path,
                )
                authority_after = _read_json(paths.authority)
                intent["broker_submission_started"] = True
                intent["consumption_status"] = "CONSUMED"
                if not preserve_historical_intent:
                    _atomic_write_json(intent_path, intent, require_full_durability=True)
                append_audit(
                    paths,
                    {
                        "event": "dispatch_intent_consumed_pre_submit",
                        "dispatch_intent_id": intent_id,
                        "plan_hash": authority.get("plan_hash"),
                        "symbols": intent["symbols"],
                        "clip_audit": list(clip_audit) if clip_audit is not None else None,
                        "preserve_historical_intent": preserve_historical_intent,
                    },
                )
                permit_token = f"sell-permit-{uuid.uuid4().hex}"
                permit = PaperSellDispatchPermit(
                    dispatch_intent_id=intent_id,
                    idempotency_key=str(authority.get("idempotency_key") or ""),
                    store_root=paths.root,
                    consumed_durable=True,
                    symbols=tuple(intent["symbols"]),
                    actions=tuple(
                        (str(r["symbol"]), str(r["action"]), float(r["qty"])) for r in req
                    ),
                    permit_token=permit_token,
                    sqlite_db_path=sqlite_path,
                )
                _write_durable_permit(permit, now=now, require_full_durability=True)
            except DispatchDurabilityError as dur_exc:
                # Partial durable evidence must never yield a broker-usable permit.
                _ACTIVE_PERMIT.set(None)
                try:
                    cur_auth = _read_json(paths.authority)
                    if cur_auth and str(cur_auth.get("consumption_status") or "") != "CONSUMED":
                        mark_authority_dispatch_attempt(
                            store_root=paths.root,
                            authority=cur_auth or authority,
                            result="DURABILITY_UNCERTAIN",
                            explicit_runtime_write=explicit_runtime_write,
                            now=now,
                            require_full_durability=False,
                        )
                except Exception as restrict_exc:  # noqa: BLE001
                    append_audit(
                        paths,
                        {
                            "event": "durability_restrict_failed",
                            "error": str(restrict_exc),
                        },
                    )
                details = {
                    "durability_error": dur_exc.reason,
                    "durability": dur_exc.details,
                    "intent_path": str(intent_path),
                }
                append_audit(
                    paths,
                    {
                        "event": "dispatch_blocked_durability_uncertain",
                        "details": details,
                    },
                )
                return DispatchGuardResult(False, "dispatch_durability_uncertain", details)
            if activate_permit:
                _ACTIVE_PERMIT.set(permit)
            return DispatchGuardResult(
                True,
                "ok",
                {
                    "revalidation": rdetails,
                    "consumption": consumption,
                    "intent": intent,
                    "market_session": sess_details,
                },
                dispatch_intent_id=intent_id,
                authority=authority_after,
                consumed=True,
                permit=permit,
                authorized_planned=selected_planned,
            )
        finally:
            _release_lock(fh)


def finalize_paper_sell_dispatch(
    *,
    store_root: Path,
    outcome: str,
    explicit_runtime_write: bool = False,
    detail: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record terminal outcome. Never restores ARMED / UNUSED."""
    paths = _assert_writable_root(store_root, explicit_runtime_write=explicit_runtime_write)
    intent_path = paths.root / DISPATCH_INTENT_NAME
    intent = _read_json(intent_path)
    from services.paper_sell_arm_store import (
        PaperSellArmStoreError,
        record_dispatch_outcome,
        resolve_runtime_sqlite_path,
        sqlite_permit_grade_enabled,
    )

    if sqlite_permit_grade_enabled():
        mapped = {
            "SUCCESS": "SUBMITTED",
            "SUBMITTED": "SUBMITTED",
            "REJECTED": "REJECTED",
            "BLOCKED_BEFORE_SUBMIT": "NOT_SUBMITTED",
            "NOT_SUBMITTED": "NOT_SUBMITTED",
            "EXCEPTION": "OUTCOME_UNKNOWN",
            "OUTCOME_UNKNOWN": "OUTCOME_UNKNOWN",
        }
        sqlite_state = mapped.get(str(outcome).upper(), "OUTCOME_UNKNOWN")
        recon = sqlite_state == "OUTCOME_UNKNOWN"
        sqlite_dispatch_id = str((intent or {}).get("dispatch_intent_id") or "")
        try:
            from services.paper_sell_arm_store import inspect_authority_row

            auth_doc = _read_json(paths.authority) or {}
            aid = str(auth_doc.get("authority_id") or auth_doc.get("audit_id") or "")
            arow = inspect_authority_row(resolve_runtime_sqlite_path(db_path=db_path), aid)
            if arow and arow.get("consuming_dispatch_id"):
                sqlite_dispatch_id = str(arow.get("consuming_dispatch_id") or sqlite_dispatch_id)
        except Exception:
            pass
        try:
            record_dispatch_outcome(
                resolve_runtime_sqlite_path(db_path=db_path),
                dispatch_id=sqlite_dispatch_id,
                outcome_state=sqlite_state,
                broker_ref=str((detail or {}).get("broker_ref") or "") or None,
                result_class=str(outcome),
                error_class=str((detail or {}).get("error_class") or "") or None,
                reconciliation_required=recon,
            )
        except PaperSellArmStoreError as exc:
            append_audit(
                paths,
                {
                    "event": "dispatch_finalize_failed",
                    "reason": exc.reason,
                    "details": exc.details,
                    "dispatch_id": sqlite_dispatch_id,
                },
            )
            return {
                "ok": False,
                "reason": "sqlite_dispatch_finalize_failed",
                "recovery_required": True,
                "broker_reconciliation_required": True,
                "authority_reusable": False,
                "details": exc.details,
            }
    authority = _read_json(paths.authority)
    if authority:
        authority["status"] = SellArmState.CONSUMED.value
        authority["consumption_status"] = "CONSUMED"
        authority["dispatch_outcome"] = outcome
        authority["dispatch_finalized_at"] = _utc_now_iso()
        _atomic_write_json(paths.authority, authority, refuse_authority_downgrade=True)
    if intent:
        intent_aid = str(intent.get("authority_id") or intent.get("audit_id") or "")
        current_aid = str(
            (authority or {}).get("authority_id") or (authority or {}).get("audit_id") or ""
        )
        if not intent_aid or intent_aid == current_aid:
            intent["status"] = f"TERMINAL_{outcome}"
            intent["finalized_at"] = _utc_now_iso()
            intent["detail"] = detail or {}
            _atomic_write_json(intent_path, intent)
    durable_path = paths.root / DURABLE_PERMIT_NAME
    if durable_path.is_file():
        durable = _read_json(durable_path)
        durable["status"] = "PERMIT_FINALIZED"
        durable["final_outcome"] = outcome
        durable["finalized_at"] = _utc_now_iso()
        _atomic_write_json(durable_path, durable)
    write_state(paths, SellArmState.CONSUMED, {"outcome": outcome})
    append_audit(
        paths,
        {
            "event": "dispatch_finalized",
            "outcome": outcome,
            "detail": detail or {},
            "authority_restored_to_armed": False,
        },
    )
    _ACTIVE_PERMIT.set(None)
    return {"outcome": outcome, "authority_reusable": False}


@contextmanager
def paper_sell_dispatch_guard(
    *,
    store_root: Path,
    planned: Sequence[Any],
    broker_view: Optional[BrokerSnapshotView] = None,
    mode: str = "paper",
    readiness_packet_hash_current: Optional[str] = None,
    explicit_runtime_write: bool = False,
    live_override: Optional[Dict[str, Any]] = None,
) -> Generator[DispatchGuardResult, None, None]:
    result = begin_paper_sell_dispatch(
        store_root=store_root,
        planned=planned,
        broker_view=broker_view,
        mode=mode,
        readiness_packet_hash_current=readiness_packet_hash_current,
        explicit_runtime_write=explicit_runtime_write,
        live_override=live_override,
    )
    outcome = "SUCCESS"
    try:
        yield result
        if not result.permitted and result.reason != NO_SELLS:
            outcome = "BLOCKED_BEFORE_SUBMIT"
        elif result.details.get("test_forced_outcome"):
            outcome = str(result.details["test_forced_outcome"])
    except Exception as exc:
        outcome = "EXCEPTION"
        result.details["exception"] = str(exc)
        raise
    finally:
        if result.consumed:
            finalize_paper_sell_dispatch(
                store_root=store_root,
                outcome=outcome,
                explicit_runtime_write=explicit_runtime_write,
                detail=result.details,
            )


def build_broker_view_from_runtime_snapshots(
    *,
    store_root: Optional[Path] = None,
    positions_csv: Optional[Path] = None,
    open_orders_csv: Optional[Path] = None,
    account_fingerprint: Optional[str] = None,
    trading_date: Optional[str] = None,
    current_plan_hash: Optional[str] = None,
    newer_completed_plan: bool = False,
    live_blocked: Optional[bool] = None,
    market_session: str = "MARKET_OPEN",
    results_root: Optional[Path] = None,
    live_root: Optional[Path] = None,
    live_override: Optional[Dict[str, Any]] = None,
) -> BrokerSnapshotView:
    """
    Independent snapshot builder. `live_blocked` arg is non-authoritative diagnostic only.
    """
    root = store_root or LIVE_GOVERNANCE
    positions: Dict[str, Dict[str, float]] = {}
    open_syms: set = set()
    pos_path = positions_csv or (ROOT / "data" / "results" / "positions_snapshot.csv")
    oo_path = open_orders_csv or (ROOT / "data" / "results" / "open_orders_snapshot.csv")
    if pos_path.is_file():
        for row in csv.DictReader(pos_path.read_text(encoding="utf-8").splitlines()):
            sym = str(row.get("symbol") or row.get("ticker") or "").upper()
            if not sym:
                continue
            try:
                qty = float(row.get("qty") or 0)
                avail = float(
                    row.get("qty_available") if row.get("qty_available") not in (None, "") else qty
                )
            except Exception:
                continue
            positions[sym] = {"qty": qty, "qty_available": avail}
    if oo_path.is_file():
        for row in csv.DictReader(oo_path.read_text(encoding="utf-8").splitlines()):
            sym = str(row.get("symbol") or row.get("ticker") or "").upper()
            if sym:
                open_syms.add(sym)

    view = load_independent_runtime_evidence(
        store_root=root,
        results_root=results_root,
        live_root=live_root,
        caller_live_blocked=live_blocked,
        live_override=live_override,
        account_fingerprint_override=account_fingerprint,
        positions=positions or None,
        open_order_symbols=open_syms or None,
        newer_completed_plan=newer_completed_plan,
        market_session=market_session,
    )
    # Optional plan hash override only when provided from independent plan evidence.
    if current_plan_hash:
        view.current_plan_hash = current_plan_hash
    if trading_date:
        view.trading_date = trading_date
    return view


def legacy_writer_target_path(store_root: Optional[Path] = None) -> Path:
    """Canonical path for Phase 197W historical writes — never hardened authority."""
    root = store_root or LIVE_GOVERNANCE
    return root / LEGACY_197W_APPROVAL_FILENAME


def refuse_legacy_write_to_hardened_path(target: Path) -> None:
    name = target.name
    if name in {HARDENED_AUTHORITY_FILENAME, DEPRECATED_SHARED_APPROVAL_FILENAME}:
        raise PermissionError(
            f"legacy_197w_refuses_write_to_hardened_or_deprecated_shared_path:{name}"
        )
    if name == "paper_sell_arm_consumption.json" or name == DISPATCH_INTENT_NAME:
        raise PermissionError(f"legacy_197w_refuses_write_to_consumption_or_intent:{name}")
