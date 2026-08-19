"""Deterministic cap-compliant paper SELL slice (Phase 198N-C/F).

Pure helper: no production I/O, SQLite, broker, env credentials, or network.
Does not import authority, runtime, console, store, or production paths.
Callers pass cap/cap_source. Does not write packets or challenges.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SELL_ACTIONS = frozenset({"TRIM", "EXIT", "ROTATE_EXIT", "FORCE_EXIT"})
BUY_ACTIONS = frozenset({"BUY", "ADD", "ENTRY", "ROTATE_ENTRY"})

STATUS_SLICE_READY = "SLICE_READY"
STATUS_AUTHORITY_CAP_EXCEEDED = "AUTHORITY_CAP_EXCEEDED"
STATUS_PACKET_INVALID = "PACKET_INVALID"
UNSLICEABLE = "UNSLICEABLE_UNDER_AUTHORITY_CAP"
DEFAULT_SLICE_CAP = 5000.0
DEFAULT_SLICE_CAP_SOURCE = "APPROVAL_CLI_CAP"

RECOGNIZED_ENTRY_ONLY = frozenset(set(BUY_ACTIONS) | {"OPEN_NEW"})
CONTRADICTORY_BUY_SIDES = frozenset({"BUY", "B", "BID", "LONG_ENTRY", "ENTRY"})
_EPS = 1e-9
_MAX_SLICE_ESTIMATE = 64


def money_round(value: Any) -> float:
    """Canonical two-decimal notional rounding used by paper SELL authority."""
    return round(float(value) + 1e-12, 2)


def canonical_json_hash(doc: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(doc), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _whole_share_qty(qty: float) -> int:
    """Largest whole-share count that does not round original quantity upward."""
    return int(math.floor(float(qty) + 1e-12))


def source_readiness_packet_hash(packet: Mapping[str, Any]) -> str:
    """Full canonical F2 source identity. Does not redefine readiness_packet_hash."""
    actions = []
    for row in packet.get("approval_ready_actions") or []:
        if not isinstance(row, dict):
            continue
        actions.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "action": str(row.get("action") or "").upper(),
                "qty": row.get("qty"),
                "est_notional": money_round(row.get("est_notional") or 0.0),
            }
        )
    material = {
        "schema_version": packet.get("schema_version"),
        "status": packet.get("status"),
        "mode": packet.get("mode"),
        "cycle_id": packet.get("cycle_id"),
        "run_id": packet.get("run_id"),
        "batch_id": packet.get("batch_id"),
        "plan_id": packet.get("plan_id"),
        "plan_hash": packet.get("plan_hash"),
        "generated_at": packet.get("generated_at"),
        "invocation_id": packet.get("invocation_id") or packet.get("market_open_invocation_id"),
        "market_open_invocation_id": packet.get("market_open_invocation_id")
        or packet.get("invocation_id"),
        "source_plan_sha256": packet.get("source_plan_sha256"),
        "selected_row_hash": packet.get("selected_row_hash"),
        "approval_ready_actions": actions,
        "approval_ready_symbols": list(packet.get("approval_ready_symbols") or []),
        "total_proposed_notional": money_round(packet.get("total_proposed_notional") or 0.0),
    }
    return canonical_json_hash(material)


def packet_fingerprint(packet: Mapping[str, Any]) -> str:
    """In-memory fingerprint of the full reviewed packet (not a slice)."""
    return source_readiness_packet_hash(packet)


def authorized_slice_hash(
    *,
    scope: Sequence[Mapping[str, Any]],
    slice_total: float,
    applied_cap: float,
    cap_source: str,
) -> str:
    material = {
        "applied_cap": money_round(applied_cap),
        "applied_cap_source": str(cap_source),
        "slice_total_max_notional": money_round(slice_total),
        "ordered_actions": [
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "action": str(row.get("action") or "").upper(),
                "max_qty": float(
                    row.get("max_qty") if row.get("max_qty") is not None else row.get("qty") or 0
                ),
                "max_notional": money_round(
                    row.get("max_notional")
                    if row.get("max_notional") is not None
                    else row.get("est_notional") or 0.0
                ),
            }
            for row in scope
        ],
    }
    return canonical_json_hash(material)


def slice_hash(
    *,
    actions: Sequence[Mapping[str, Any]],
    slice_total: float,
    applied_cap: float,
    cap_source: str,
) -> str:
    return authorized_slice_hash(
        scope=actions,
        slice_total=slice_total,
        applied_cap=applied_cap,
        cap_source=cap_source,
    )


def presentation_token(
    *,
    packet: Mapping[str, Any],
    slice_result: Mapping[str, Any],
    packet_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-secret render-to-click token. Bind Arm Reviewed Sells to this exact view."""
    slice_actions = list(slice_result.get("slice_actions") or [])
    return {
        "packet_sha256": str(packet_sha256 or slice_result.get("source_packet_fingerprint") or ""),
        "packet_fingerprint": str(slice_result.get("source_packet_fingerprint") or ""),
        "source_readiness_packet_hash": str(
            slice_result.get("source_readiness_packet_hash")
            or slice_result.get("source_packet_fingerprint")
            or ""
        ),
        "generated_at": str(packet.get("generated_at") or ""),
        "invocation_id": str(
            packet.get("market_open_invocation_id")
            or packet.get("invocation_id")
            or packet.get("cycle_id")
            or ""
        ),
        "slice_hash": str(slice_result.get("slice_hash") or ""),
        "authorized_slice_hash": str(
            slice_result.get("authorized_slice_hash") or slice_result.get("slice_hash") or ""
        ),
        "slice_total": money_round(slice_result.get("slice_total_notional") or 0.0),
        "slice_identity": [
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "action": str(row.get("action") or "").upper(),
                "qty": row.get("qty"),
                "est_notional": money_round(row.get("est_notional") or 0.0),
            }
            for row in slice_actions
        ],
    }


def tokens_match(expected: Any, actual: Any) -> bool:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    return canonical_json_hash(expected) == canonical_json_hash(actual)


def _validate_action_row(row: Any, *, index: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(row, dict):
        return None, "action_row_not_object"
    symbol = str(row.get("symbol") or "").upper().strip()
    action = str(row.get("action") or "").upper().strip()
    side = str(row.get("side") or "").upper().strip()
    if not symbol:
        return None, "action_missing_symbol"
    if action in RECOGNIZED_ENTRY_ONLY or action in BUY_ACTIONS:
        return None, "buy_or_add_in_sell_scope"
    if action not in SELL_ACTIONS:
        return None, "unknown_sell_action"
    if side in CONTRADICTORY_BUY_SIDES:
        return None, "contradictory_side_action"
    if not _finite_positive(row.get("qty")):
        return None, "quantity_not_finite_positive"
    if not _finite_positive(row.get("est_notional") or row.get("max_notional")):
        return None, "notional_not_finite_positive"
    original_qty = float(row.get("qty"))
    notional = float(
        row.get("est_notional") if row.get("est_notional") is not None else row.get("max_notional")
    )
    unit = notional / original_qty
    if not math.isfinite(unit) or unit <= 0.0:
        return None, "unit_notional_invalid"
    return {
        "index": index,
        "symbol": symbol,
        "action": action,
        "original_qty": original_qty,
        "original_notional": money_round(notional),
        "unit_notional": unit,
        "est_price": row.get("est_price"),
        "remaining_qty": float(_whole_share_qty(original_qty)),
    }, None


def _normalized_actions(packet: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if str(packet.get("mode") or "").upper() != "PAPER":
        return [], "readiness_mode_not_paper"
    if str(packet.get("live_execution_state") or "").upper() != "BLOCKED":
        return [], "readiness_live_not_blocked"
    if not packet.get("plan_hash") or not packet.get("batch_id") or not packet.get("cycle_id"):
        return [], "readiness_missing_binding_ids"
    raw = list(packet.get("approval_ready_actions") or [])
    if not raw:
        return [], "readiness_has_no_approval_ready_actions"
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(raw):
        item, err = _validate_action_row(row, index=index)
        if err:
            return [], err
        assert item is not None
        if item["remaining_qty"] < 1:
            return [], "quantity_not_whole_share"
        out.append(item)
    return out, None


def _take_slice(
    remaining: Sequence[Dict[str, Any]],
    *,
    cap: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy scan in stable packet order. Mutates remaining_qty copies."""
    leftover = cap
    slice_rows: List[Dict[str, Any]] = []
    unsliceable: List[Dict[str, Any]] = []
    per_symbol = {}
    working = [dict(row) for row in remaining]

    for row in working:
        unit = float(row["unit_notional"])
        left = float(row["remaining_qty"])
        if left < 1:
            continue
        if money_round(unit) > cap + _EPS:
            unsliceable.append(
                {
                    "symbol": row["symbol"],
                    "action": row["action"],
                    "original_qty": row["original_qty"],
                    "deferred_qty": left,
                    "unit_notional": money_round(unit),
                    "reason": UNSLICEABLE,
                }
            )
            continue
        symbol_used = float(per_symbol.get(row["symbol"]) or 0.0)
        symbol_room = cap - symbol_used
        room = min(leftover, symbol_room)
        if room <= _EPS:
            continue
        max_qty = min(int(left), int(math.floor((room + 1e-12) / unit)))
        if max_qty < 1:
            continue
        notional = money_round(max_qty * unit)
        if notional <= 0 or notional > room + _EPS or notional > cap + _EPS:
            continue
        if leftover - notional < -_EPS:
            continue
        slice_rows.append(
            {
                "symbol": row["symbol"],
                "action": row["action"],
                "qty": max_qty,
                "est_notional": notional,
                "est_price": row.get("est_price"),
                "original_qty": row["original_qty"],
                "slice_qty": max_qty,
                "deferred_qty": left - max_qty,
                "unit_notional": money_round(unit),
            }
        )
        row["remaining_qty"] = left - max_qty
        leftover = money_round(leftover - notional)
        per_symbol[row["symbol"]] = money_round(symbol_used + notional)

    deferred: List[Dict[str, Any]] = []
    for row in working:
        left = float(row["remaining_qty"])
        if left < 1:
            continue
        if any(
            u.get("symbol") == row["symbol"] and u.get("reason") == UNSLICEABLE for u in unsliceable
        ):
            continue
        deferred.append(
            {
                "symbol": row["symbol"],
                "action": row["action"],
                "original_qty": row["original_qty"],
                "deferred_qty": left,
                "est_notional": money_round(left * float(row["unit_notional"])),
                "unit_notional": money_round(row["unit_notional"]),
                "reason": "DEFERRED_UNDER_AUTHORITY_CAP",
            }
        )
    deferred.extend(unsliceable)
    return slice_rows, deferred, working


def _empty_result(
    packet: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    cap: float,
    cap_source: str,
    actions: Optional[Sequence[Dict[str, Any]]] = None,
    unsliceable: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    full_n = money_round(
        sum(float(a.get("original_notional") or 0.0) for a in (actions or []))
        if actions
        else float(packet.get("total_proposed_notional") or 0.0)
    )
    return {
        "status": status,
        "reason": reason,
        "applied_cap": money_round(cap),
        "cap_source": cap_source,
        "source_packet_fingerprint": packet_fingerprint(packet),
        "source_generated_at": packet.get("generated_at"),
        "source_invocation_id": packet.get("market_open_invocation_id")
        or packet.get("invocation_id")
        or packet.get("cycle_id"),
        "source_plan_hash": packet.get("plan_hash"),
        "source_selected_row_hash": packet.get("selected_row_hash"),
        "source_plan_sha256": packet.get("source_plan_sha256"),
        "full_action_count": len(list(packet.get("approval_ready_actions") or [])),
        "full_total_notional": money_round(packet.get("total_proposed_notional") or full_n),
        "slice_actions": [],
        "slice_action_count": 0,
        "slice_total_notional": 0.0,
        "deferred_actions": list(unsliceable or []),
        "deferred_action_count": len(list(unsliceable or [])),
        "deferred_total_notional": money_round(
            sum(float(d.get("est_notional") or 0.0) for d in (unsliceable or []))
        ),
        "unsliceable_actions": list(unsliceable or []),
        "slice_hash": "",
        "estimated_remaining_slice_count": 0,
        "estimated_remaining_slice_count_is_estimate": True,
    }


def _count_remaining_slices(working: Sequence[Dict[str, Any]], *, cap: float) -> int:
    rows = [dict(r) for r in working]
    count = 0
    while count < _MAX_SLICE_ESTIMATE:
        if all(float(r.get("remaining_qty") or 0) < 1 for r in rows):
            break
        if all(
            money_round(r["unit_notional"]) > cap + _EPS
            for r in rows
            if float(r.get("remaining_qty") or 0) >= 1
        ):
            break
        taken, _, rows = _take_slice(rows, cap=cap)
        if not taken:
            break
        count += 1
    return count


def slice_reviewed_sells(
    packet: Mapping[str, Any],
    *,
    cap: Optional[float] = None,
    cap_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the current cap-compliant slice for a classified reviewed SELL packet."""
    cap_value = float(DEFAULT_SLICE_CAP if cap is None else cap)
    cap_name = str(DEFAULT_SLICE_CAP_SOURCE if cap_source is None else cap_source)
    actions, err = _normalized_actions(packet)
    if err:
        return _empty_result(
            packet, status=STATUS_PACKET_INVALID, reason=err, cap=cap_value, cap_source=cap_name
        )

    slice_rows, deferred, working = _take_slice(actions, cap=cap_value)
    unsliceable = [d for d in deferred if d.get("reason") == UNSLICEABLE]
    if not slice_rows:
        return _empty_result(
            packet,
            status=STATUS_AUTHORITY_CAP_EXCEEDED,
            reason="authority_cap_exceeded",
            cap=cap_value,
            cap_source=cap_name,
            actions=actions,
            unsliceable=unsliceable or deferred,
        )

    slice_total = money_round(sum(float(r["est_notional"]) for r in slice_rows))
    if slice_total > cap_value + _EPS:
        return _empty_result(
            packet,
            status=STATUS_AUTHORITY_CAP_EXCEEDED,
            reason="authority_cap_exceeded",
            cap=cap_value,
            cap_source=cap_name,
            actions=actions,
        )

    remaining_after = _count_remaining_slices(working, cap=cap_value)
    fingerprint = packet_fingerprint(packet)
    identity_hash = slice_hash(
        actions=slice_rows,
        slice_total=slice_total,
        applied_cap=cap_value,
        cap_source=cap_name,
    )
    deferred_n = money_round(sum(float(d.get("est_notional") or 0.0) for d in deferred))
    return {
        "status": STATUS_SLICE_READY,
        "reason": "cap_compliant_slice_ready",
        "applied_cap": money_round(cap_value),
        "cap_source": cap_name,
        "source_packet_fingerprint": fingerprint,
        "source_readiness_packet_hash": fingerprint,
        "source_generated_at": packet.get("generated_at"),
        "source_invocation_id": packet.get("market_open_invocation_id")
        or packet.get("invocation_id")
        or packet.get("cycle_id"),
        "source_plan_hash": packet.get("plan_hash"),
        "source_selected_row_hash": packet.get("selected_row_hash"),
        "source_plan_sha256": packet.get("source_plan_sha256"),
        "full_action_count": len(actions),
        "full_total_notional": money_round(packet.get("total_proposed_notional") or 0.0),
        "slice_actions": slice_rows,
        "slice_action_count": len(slice_rows),
        "slice_total_notional": slice_total,
        "deferred_actions": deferred,
        "deferred_action_count": len(deferred),
        "deferred_total_notional": deferred_n,
        "unsliceable_actions": unsliceable,
        "slice_hash": identity_hash,
        "authorized_slice_hash": identity_hash,
        "estimated_remaining_slice_count": remaining_after,
        "estimated_remaining_slice_count_is_estimate": True,
    }


def iter_cap_compliant_slices(
    packet: Mapping[str, Any],
    *,
    cap: Optional[float] = None,
    cap_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Model remaining-quantity slices. Does not mutate the source packet."""
    applied = float(DEFAULT_SLICE_CAP if cap is None else cap)
    source = str(DEFAULT_SLICE_CAP_SOURCE if cap_source is None else cap_source)
    actions, err = _normalized_actions(packet)
    if err:
        return []
    working = [dict(a) for a in actions]
    out: List[Dict[str, Any]] = []
    for _ in range(_MAX_SLICE_ESTIMATE):
        taken, _, working = _take_slice(working, cap=float(applied))
        if not taken:
            break
        total = money_round(sum(float(r["est_notional"]) for r in taken))
        out.append(
            {
                "slice": len(out) + 1,
                "actions": taken,
                "total": total,
                "le_cap": total <= float(applied) + _EPS,
                "cap_source": source,
            }
        )
    return out


def build_sliced_readiness_packet(
    packet: Mapping[str, Any], slice_result: Mapping[str, Any]
) -> Dict[str, Any]:
    """In-memory readiness clone whose approval_ready_* fields are the current slice.

    Does not write the canonical F2 packet. Preserves whole-plan identity fields.
    When the slice is the full original quantity set, original action rows are kept
    so readiness_packet_hash stays identical for under-cap packets.
    """
    clone = dict(packet)
    original = [
        dict(row) for row in (packet.get("approval_ready_actions") or []) if isinstance(row, dict)
    ]
    orig_by_key = {}
    for row in original:
        key = (str(row.get("symbol") or "").upper(), str(row.get("action") or "").upper())
        orig_by_key[key] = row
    actions = []
    full_identity = True
    for row in slice_result.get("slice_actions") or []:
        symbol = str(row.get("symbol") or "").upper()
        action = str(row.get("action") or "").upper()
        orig = orig_by_key.get((symbol, action)) or {}
        slice_qty = row.get("qty")
        slice_notional = money_round(row.get("est_notional") or 0.0)
        orig_qty = orig.get("qty")
        orig_notional = money_round(orig.get("est_notional") or 0.0) if orig else None
        if (
            orig
            and orig_qty is not None
            and float(slice_qty) == float(orig_qty)
            and orig_notional == slice_notional
        ):
            actions.append(dict(orig))
            continue
        full_identity = False
        item = dict(orig) if orig else {}
        item["symbol"] = symbol
        item["action"] = action
        item["qty"] = slice_qty
        item["est_notional"] = slice_notional
        if row.get("est_price") is not None:
            item["est_price"] = row.get("est_price")
        actions.append(item)
    if full_identity and len(actions) == len(original):
        clone["approval_ready_actions"] = actions
        clone["approval_ready_symbols"] = list(packet.get("approval_ready_symbols") or [])
        clone["total_proposed_notional"] = packet.get("total_proposed_notional")
    else:
        clone["approval_ready_actions"] = actions
        clone["approval_ready_symbols"] = [str(a["symbol"]) for a in actions]
        clone["total_proposed_notional"] = money_round(
            slice_result.get("slice_total_notional") or 0.0
        )
    clone["cap_compliant_slice"] = True
    clone["source_packet_fingerprint"] = slice_result.get("source_packet_fingerprint")
    clone["slice_hash"] = slice_result.get("slice_hash")
    return clone


def authorized_scope_from_slice(slice_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for row in slice_result.get("slice_actions") or slice_result.get("authorized_scope") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        action = str(row.get("action") or "").upper()
        qty = row.get("max_qty")
        if qty is None:
            qty = row.get("qty")
        notional = row.get("max_notional")
        if notional is None:
            notional = row.get("est_notional")
        key = (symbol, action)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "symbol": symbol,
                "action": action,
                "max_qty": float(qty or 0),
                "max_notional": money_round(notional or 0.0),
            }
        )
    return out


def validate_slice_against_packet(
    packet: Mapping[str, Any],
    slice_result: Mapping[str, Any],
    *,
    cap: float,
    cap_source: str,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Prove authorized_slice is a strict cap-compliant subset of the full packet."""
    cap_value = money_round(cap)
    if cap_value > DEFAULT_SLICE_CAP + _EPS:
        return False, "applied_cap_raises_cli_limit", []
    full_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in packet.get("approval_ready_actions") or []:
        if not isinstance(row, dict):
            return False, "action_row_not_object", []
        symbol = str(row.get("symbol") or "").upper()
        action = str(row.get("action") or "").upper()
        if not symbol:
            return False, "action_missing_symbol", []
        if action in RECOGNIZED_ENTRY_ONLY or action in BUY_ACTIONS:
            return False, "buy_or_add_in_sell_scope", []
        if action not in SELL_ACTIONS:
            return False, "unknown_sell_action", []
        key = (symbol, action)
        if key in full_rows:
            return False, "duplicate_symbol_action_in_source", []
        qty = float(row.get("qty") or 0)
        notional = money_round(row.get("est_notional") or 0.0)
        if qty <= 0 or notional <= 0:
            return False, "source_row_not_finite_positive", []
        full_rows[key] = {"qty": qty, "notional": notional, "action": action}

    scope = authorized_scope_from_slice(slice_result)
    if not scope:
        return False, "empty_scope", []
    seen: set[Tuple[str, str]] = set()
    total = 0.0
    for item in scope:
        symbol = str(item.get("symbol") or "").upper()
        action = str(item.get("action") or "").upper()
        key = (symbol, action)
        if key in seen:
            return False, "duplicate_symbol_action_in_slice", []
        seen.add(key)
        if action in BUY_ACTIONS or action in RECOGNIZED_ENTRY_ONLY:
            return False, "buy_or_add_in_sell_scope", []
        if action not in SELL_ACTIONS:
            return False, "unknown_sell_action", []
        src = full_rows.get(key)
        if src is None:
            return False, f"slice_symbol_not_in_source:{symbol}", []
        if src["action"] != action:
            return False, f"action_mismatch_vs_source:{symbol}", []
        qty = float(item.get("max_qty") or 0)
        notional = money_round(item.get("max_notional") or 0.0)
        if qty <= 0:
            return False, f"slice_quantity_not_positive:{symbol}", []
        if qty > float(src["qty"]) + _EPS:
            return False, f"slice_quantity_exceeds_source:{symbol}", []
        if notional > float(src["notional"]) + _EPS:
            return False, f"slice_notional_exceeds_source:{symbol}", []
        if notional > cap_value + _EPS:
            return False, f"symbol_exceeds_cap:{symbol}", []
        total = money_round(total + notional)
    if total > cap_value + _EPS:
        return False, "scope_exceeds_approval_cli_cap", []
    expected = str(
        slice_result.get("authorized_slice_hash") or slice_result.get("slice_hash") or ""
    )
    recomputed = authorized_slice_hash(
        scope=scope,
        slice_total=total,
        applied_cap=cap_value,
        cap_source=cap_source,
    )
    if expected and expected != recomputed:
        return False, "authorized_slice_hash_changed", []
    return True, "ok", scope
