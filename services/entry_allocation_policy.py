"""Position-aware entry allocation policy.

Read-only helpers for settled-cash evidence, flat-versus-held risk, and the
bounded ranking of OPEN_NEW / ADD candidates. This module does not submit
orders, arm execution, or write production artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

IDENTITY_TOLERANCE = Decimal("0.01")
CASH_MAX_AGE_SECONDS = 120 * 60
DEFAULT_MAX_NEW_NAMES = 10
DEFAULT_MAX_ADDS = 3
DEFAULT_MAX_POSITIONS = 35
DEFAULT_MAX_ORDER_NOTIONAL = 1500.0
LOCAL_CASH_FILES = (
    "account_snapshot.json",
    "equity_snapshot.json",
    "paper_account_snapshot.json",
)
CASH_KEYS = ("settled_cash", "cash", "cash_available")
REASON_UNPROVEN = "settled_cash_unproven"

AccountReader = Callable[[], Mapping[str, Any]]


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "None", "NaN"):
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return number


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def validate_account_evidence(
    doc: Optional[Mapping[str, Any]],
    *,
    observed_at: datetime,
    now: datetime,
    max_age_seconds: int = CASH_MAX_AGE_SECONDS,
    require_identity: bool = False,
) -> Tuple[Optional[float], str]:
    """Return settled cash, or (None, settled_cash_unproven).

    A missing, stale, malformed, or inconsistent document is never coerced
    to zero. An explicit validated cash of zero stays zero.
    """
    if not isinstance(doc, Mapping) or not doc:
        return None, REASON_UNPROVEN
    age = (_aware(now) - _aware(observed_at)).total_seconds()
    if age < 0 or age > max_age_seconds:
        return None, REASON_UNPROVEN
    cash = None
    for key in CASH_KEYS:
        if key in doc:
            cash = _decimal(doc.get(key))
            if cash is not None:
                break
    if cash is None or cash < 0:
        return None, REASON_UNPROVEN
    has_identity = any(key in doc for key in ("equity", "long_market_value", "market_value"))
    if require_identity or has_identity:
        equity = _decimal(doc.get("equity"))
        market = _decimal(doc.get("long_market_value"))
        if market is None:
            market = _decimal(doc.get("market_value"))
        if equity is None or market is None or market < 0:
            return None, REASON_UNPROVEN
        if abs((cash + market) - equity) > IDENTITY_TOLERANCE:
            return None, REASON_UNPROVEN
        if "short_market_value" in doc and doc.get("short_market_value") not in (None, ""):
            short = _decimal(doc.get("short_market_value"))
            if short is None or short != 0:
                return None, REASON_UNPROVEN
    return float(cash), ""


def _parse_stamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _aware(parsed)


def _local_observation(path: Path, doc: Mapping[str, Any]) -> Optional[datetime]:
    for key in ("as_of_utc", "observed_at", "generated_at", "generated_at_utc", "timestamp"):
        stamp = _parse_stamp(doc.get(key))
        if stamp is not None:
            return stamp
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def resolve_settled_cash(
    *,
    results_dir: Path,
    now: datetime,
    account_reader: Optional[AccountReader] = None,
    max_age_seconds: int = CASH_MAX_AGE_SECONDS,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Use a fresh local snapshot, otherwise one read-only account read.

    Missing local evidence is not zero. A failed or inconsistent read stays
    fail-closed and does not fall back to a stale file.
    """
    meta: Dict[str, Any] = {
        "source": None,
        "broker_called": False,
        "reason": "",
        "local_status": "absent",
    }
    for name in LOCAL_CASH_FILES:
        path = Path(results_dir) / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta["local_status"] = "malformed"
            continue
        if not isinstance(doc, dict):
            meta["local_status"] = "malformed"
            continue
        observed = _local_observation(path, doc)
        if observed is None:
            meta["local_status"] = "stale"
            continue
        cash, reason = validate_account_evidence(
            doc,
            observed_at=observed,
            now=now,
            max_age_seconds=max_age_seconds,
            require_identity=False,
        )
        if cash is not None and reason == "":
            meta["source"] = "local"
            meta["local_status"] = "fresh"
            meta["path"] = name
            return cash, meta
        meta["local_status"] = reason or "rejected"
    if account_reader is None:
        meta["reason"] = REASON_UNPROVEN
        return None, meta
    meta["broker_called"] = True
    try:
        fresh = account_reader()
    except Exception:
        meta["reason"] = REASON_UNPROVEN
        meta["source"] = "broker_read_failed"
        return None, meta
    if not isinstance(fresh, Mapping):
        meta["reason"] = REASON_UNPROVEN
        meta["source"] = "broker_read_failed"
        return None, meta
    cash, reason = validate_account_evidence(
        fresh,
        observed_at=now,
        now=now,
        max_age_seconds=max_age_seconds,
        require_identity=True,
    )
    if cash is None:
        meta["reason"] = reason or REASON_UNPROVEN
        meta["source"] = "broker_read_rejected"
        return None, meta
    meta["source"] = "broker_readonly"
    meta["reason"] = ""
    return cash, meta


def read_paper_account_readonly() -> Dict[str, Any]:
    """GET the paper account. Does not submit, cancel, or replace orders."""
    from services.broker_alpaca import AlpacaBroker

    broker = AlpacaBroker(mode="paper")
    base = str(getattr(broker, "base", ""))
    if "paper-api.alpaca.markets" not in base:
        raise RuntimeError("paper_account_reader_refused")
    account = broker.get_account()
    if not isinstance(account, dict):
        raise RuntimeError("paper_account_malformed")
    return account


def entry_limits(config_dir: Path) -> Dict[str, float]:
    """Existing per-order ceiling, new-name cap, and add cap. Read-only."""
    limits = {
        "max_new": float(DEFAULT_MAX_NEW_NAMES),
        "max_adds": float(DEFAULT_MAX_ADDS),
        "max_positions": float(DEFAULT_MAX_POSITIONS),
        "max_notional": float(DEFAULT_MAX_ORDER_NOTIONAL),
    }
    trades = Path(config_dir) / "execute_trades.json"
    guard = Path(config_dir) / "execution_guard.json"
    try:
        raw = json.loads(trades.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if raw.get("max_new_positions_per_run") is not None:
                limits["max_new"] = float(raw["max_new_positions_per_run"])
            if raw.get("max_adds_per_run") is not None:
                limits["max_adds"] = float(raw["max_adds_per_run"])
            if raw.get("max_positions") is not None:
                limits["max_positions"] = float(raw["max_positions"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        raw_guard = json.loads(guard.read_text(encoding="utf-8"))
        if isinstance(raw_guard, dict) and raw_guard.get("max_notional_usd") is not None:
            limits["max_notional"] = float(raw_guard["max_notional_usd"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return limits


def basis_hold_blocked_symbols() -> set[str]:
    """Symbols the basis-unproven hold still blocks. Read-only; fail closed."""
    try:
        from services.basis_unproven_hold import (
            HOLD_SYMBOLS,
            evaluate_basis_unproven_hold_for_packet,
        )

        result = evaluate_basis_unproven_hold_for_packet()
        if not result.get("ok"):
            return {str(symbol).upper() for symbol in HOLD_SYMBOLS}
        blocked = {str(symbol).upper() for symbol in (result.get("blocked_symbols") or [])}
        return blocked or {str(symbol).upper() for symbol in HOLD_SYMBOLS}
    except Exception:
        return {"VOO", "XLK"}


def select_bounded_entries(
    rows: Sequence[Mapping[str, Any]],
    *,
    cash: Optional[float],
    max_new: int,
    max_adds: int,
    max_notional: float,
    open_slots: int,
) -> List[Dict[str, Any]]:
    """Rank OPEN_NEW and ADD without turning every passer into an order.

    Cash of None fail-closes those actions with settled_cash_unproven.
    A proven cash balance of zero does not use that reason.
    """
    prepared: List[Dict[str, Any]] = [dict(row) for row in rows]
    held_back = basis_hold_blocked_symbols()
    for row in prepared:
        if str(row.get("symbol") or "").upper() not in held_back:
            continue
        if row.get("disposition") in ("OPEN_NEW", "ADD", "eligible"):
            row["disposition"] = "hard_block"
            row["reason"] = "hard_block:basis_unproven_hold"
    if cash is None:
        for row in prepared:
            if row.get("disposition") in ("OPEN_NEW", "ADD", "eligible"):
                row["disposition"] = "hard_block"
                row["reason"] = REASON_UNPROVEN
        return prepared

    notional = float(max_notional)
    remaining = float(cash)
    slots = max(0, int(open_slots))
    new_cap = max(0, min(int(max_new), slots))
    add_cap = max(0, int(max_adds))

    def _rank(row: Mapping[str, Any]) -> Tuple[float, str]:
        try:
            score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (-score, str(row.get("symbol") or ""))

    new_rows = [row for row in prepared if row.get("disposition") == "OPEN_NEW"]
    add_rows = [row for row in prepared if row.get("disposition") == "ADD"]
    chosen_new = 0
    for row in sorted(new_rows, key=_rank):
        if chosen_new >= new_cap:
            row["disposition"] = "eligible"
            row["reason"] = "eligible:beyond_max_new_positions_per_run"
            continue
        if notional > remaining + 1e-9:
            row["disposition"] = "hard_block"
            row["reason"] = "hard_block:per_order_notional_exceeds_settled_cash"
            continue
        chosen_new += 1
        remaining -= notional
        row["disposition"] = "OPEN_NEW"
        row["reason"] = "OPEN_NEW"
        row["notional"] = notional
    chosen_add = 0
    for row in sorted(add_rows, key=_rank):
        if chosen_add >= add_cap:
            row["disposition"] = "eligible"
            row["reason"] = "eligible:beyond_max_adds_per_run"
            continue
        if notional > remaining + 1e-9:
            row["disposition"] = "hard_block"
            row["reason"] = "hard_block:per_order_notional_exceeds_settled_cash"
            continue
        chosen_add += 1
        remaining -= notional
        row["disposition"] = "ADD"
        row["reason"] = "ADD"
        row["notional"] = notional
    return prepared


def _optional_float(value: Any) -> Optional[float]:
    number = _decimal(value)
    if number is None:
        return None
    return float(number)


def classify_allocation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cash: Optional[float],
    add_confidence_min: float,
    max_new: int,
    max_adds: int,
    max_notional: float,
    open_slots: int,
) -> List[Dict[str, Any]]:
    """Recompute dispositions from an existing allocation book. Does not write."""
    from services.portfolio_allocation_engine import _decide_action, _risk_components

    classified: List[Dict[str, Any]] = []
    held_back = basis_hold_blocked_symbols()
    for source in rows:
        symbol = str(source.get("ticker") or source.get("symbol") or "").upper()
        held_text = str(source.get("is_currently_held") or "").strip().lower()
        held = held_text in ("true", "1", "yes")
        components = _risk_components(str(source.get("risk_flag") or ""))
        edge = _optional_float(source.get("edge_score"))
        confidence = _optional_float(source.get("confidence"))
        action, reason = _decide_action(
            is_held=held,
            risk_components=components,
            edge_score=edge,
            sizing_tier=str(source.get("sizing_tier") or ""),
            effective_stance=str(source.get("effective_stance") or ""),
            lifecycle_action=str(source.get("lifecycle_action") or ""),
            signal=str(source.get("signal") or ""),
            confidence=confidence,
            delta_pct=_optional_float(source.get("delta_pct")),
            add_confidence_min=add_confidence_min,
        )
        if action in ("EXIT", "TRIM", "BLOCK"):
            disposition = "hard_block"
        elif action == "OPEN_NEW":
            disposition = "OPEN_NEW"
        elif action == "ADD":
            disposition = "ADD"
        elif "soft_caution:" in str(reason):
            disposition = "soft_caution"
        else:
            disposition = action
        if symbol in held_back and action not in ("EXIT", "TRIM", "BLOCK"):
            action = "BLOCK"
            reason = "hard_block:basis_unproven_hold"
            disposition = "hard_block"
        elif symbol in held_back:
            disposition = "hard_block"
        classified.append(
            {
                "symbol": symbol,
                "score": _optional_float(source.get("allocation_score")) or 0.0,
                "action": action,
                "disposition": disposition,
                "reason": reason,
                "detail": reason,
                "signal": str(source.get("signal") or ""),
                "held": held,
            }
        )
    return select_bounded_entries(
        classified,
        cash=cash,
        max_new=max_new,
        max_adds=max_adds,
        max_notional=max_notional,
        open_slots=open_slots,
    )
