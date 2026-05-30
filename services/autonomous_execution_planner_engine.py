"""
Autonomous Execution Planner Engine -- Step 24.

Reads:
    data/results/autonomous_execution_summary.json         (Step 23)
    data/results/autonomous_execution_authorization.json   (Step 23)
    data/results/autonomous_action_permissions.json        (Step 22)
    data/results/portfolio_execution_intents.csv           (Step 7)
    data/results/portfolio_rebalance_plan.csv              (Step 6)
    data/results/runtime_policy_governed.json              (Step 18)
    data/results/autonomous_committee_summary.json         (Step 15)
    data/results/adaptive_regime.json                      (Step 10)

Writes:
    data/results/autonomous_execution_plan.json
    data/results/autonomous_execution_plan.csv
    data/results/autonomous_execution_plan.md
    data/results/autonomous_execution_plan_summary.json

Purpose
-------
Step 23 produced an authorization *state* (whether execution is
allowed today, and what kind). Step 24 takes that authorization
together with Step 6's actual rebalance plan and Step 7's per-
ticker execution intents, then filters every candidate action
through three concentric checks:

    1. Step 22 per-action permission gate
    2. Mode-specific gate (EXIT_ONLY blocks all buys, etc.)
    3. Runtime policy thresholds (confidence/persistence/deployment)

The output is the *final* authorized execution plan -- the one
artifact any future execution engine reads. Each per-ticker row
is tagged ``allowed=True/False`` with a structured
``blocked_reason`` for every denial so the audit trail is
complete.

This is NOT execution. There are no broker calls anywhere in this
module. The plan is written to disk and the engine returns.
Whether any future engine actually consumes it is a separate
decision.

Five execution modes (spec section 1)
-------------------------------------
    NO_EXECUTION         empty plan; every candidate denied
    EXIT_ONLY            sells/trims only; all buys denied
    DEFENSIVE_DEPLOYMENT selective buys + defensive rotation; +0.05
                          confidence floor on top of base threshold
    SELECTIVE_DEPLOYMENT limited buys + rebalance; standard thresholds
    FULL_DEPLOYMENT      all approved actions allowed; standard thresholds

Map from Step 23 authorization_state:
    BLOCKED              -> NO_EXECUTION
    ANALYSIS_ONLY        -> NO_EXECUTION
    EXIT_ONLY            -> EXIT_ONLY
    DEFENSIVE_EXECUTION  -> DEFENSIVE_DEPLOYMENT
    SELECTIVE_DEPLOYMENT -> SELECTIVE_DEPLOYMENT
    FULL_AUTONOMY        -> FULL_DEPLOYMENT

Safety
------
* READ ONLY. Absolutely no broker calls. No execution mutation.
  The words "place_order", "submit", and "client" do not appear.
* Atomic writes (.tmp + os.replace) for all four outputs.
* Missing inputs warn-and-continue. With zero candidate rows the
  plan is empty; with zero authorization the mode is NO_EXECUTION
  and every (hypothetical) action is denied.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_AUTH_SUMMARY = RESULTS_DIR / "autonomous_execution_summary.json"
DEFAULT_AUTH_FULL = RESULTS_DIR / "autonomous_execution_authorization.json"
DEFAULT_ACTION_PERMS = RESULTS_DIR / "autonomous_action_permissions.json"
DEFAULT_EXEC_INTENTS_CSV = RESULTS_DIR / "portfolio_execution_intents.csv"
DEFAULT_REBALANCE_CSV = RESULTS_DIR / "portfolio_rebalance_plan.csv"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_execution_plan.json"
DEFAULT_OUT_CSV = RESULTS_DIR / "autonomous_execution_plan.csv"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_execution_plan.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_execution_plan_summary.json"


# -----------------------------------------------------------
# Modes & state mapping
# -----------------------------------------------------------
MODE_NO_EXECUTION = "NO_EXECUTION"
MODE_EXIT_ONLY = "EXIT_ONLY"
MODE_DEFENSIVE_DEPLOYMENT = "DEFENSIVE_DEPLOYMENT"
MODE_SELECTIVE_DEPLOYMENT = "SELECTIVE_DEPLOYMENT"
MODE_FULL_DEPLOYMENT = "FULL_DEPLOYMENT"

ALL_MODES: Tuple[str, ...] = (
    MODE_NO_EXECUTION,
    MODE_EXIT_ONLY,
    MODE_DEFENSIVE_DEPLOYMENT,
    MODE_SELECTIVE_DEPLOYMENT,
    MODE_FULL_DEPLOYMENT,
)

AUTH_STATE_TO_MODE: Dict[str, str] = {
    "BLOCKED": MODE_NO_EXECUTION,
    "ANALYSIS_ONLY": MODE_NO_EXECUTION,
    "EXIT_ONLY": MODE_EXIT_ONLY,
    "DEFENSIVE_EXECUTION": MODE_DEFENSIVE_DEPLOYMENT,
    "SELECTIVE_DEPLOYMENT": MODE_SELECTIVE_DEPLOYMENT,
    "FULL_AUTONOMY": MODE_FULL_DEPLOYMENT,
}

# Rebalance-action vocabularies (Step 6 emits the canonical names below;
# we keep a permissive alias set so column-name drift never silently
# drops a candidate).
BUY_NEW_ACTIONS = {"BUY_NEW", "OPEN_NEW", "OPEN_POSITION", "BUY"}
ADD_ACTIONS = {"ADD", "ADD_TO_POSITION"}
TRIM_ACTIONS = {"TRIM"}
SELL_ACTIONS = {"SELL", "EXIT", "FULL_EXIT"}
HOLD_ACTIONS = {"HOLD", "NO_ACTION", "WAIT", ""}
ROTATION_ACTIONS = {"ROTATION", "ROTATE"}

# Execution intent vocabulary (Step 7)
INTENT_EXECUTE = "EXECUTE_NOW"
INTENT_TERMINAL_NON_EXEC = {"DELAY", "SKIP", "BLOCK", "BLOCKED", "REJECT"}

# Mode-specific elevated confidence floors (additive to base threshold)
DEFENSIVE_CONFIDENCE_BUMP = 0.05

# Default confidence threshold when the runtime policy is missing the
# field entirely (matches the spec's institutional defaults).
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_PERSISTENCE_THRESHOLD = 0.70
DEFAULT_DEPLOYMENT_THRESHOLD = 0.70
DEFAULT_MAX_POSITION_PCT = 8.0
DEFAULT_MAX_NEW_POSITIONS = 3

# Plan confidence contributor weights (spec section 5)
PLAN_CONFIDENCE_WEIGHTS = {
    "authorization_confidence": 0.40,
    "avg_intent_score": 0.25,
    "runtime_confidence_floor": 0.20,
    "committee_confidence": 0.15,
}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[EXECUTION_PLAN_WARN] {msg}", flush=True)


def _safe_read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        _warn(f"failed to parse {label} ({path}): {type(e).__name__}: {e}")
        return {}


def _safe_read_csv_rows(path: Path, *, label: str) -> List[Dict[str, str]]:
    try:
        if not path.is_file():
            return []
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]
    except Exception as e:
        _warn(f"failed to parse {label} ({path}): {type(e).__name__}: {e}")
        return []


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_safe)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_csv(rows: List[Dict[str, Any]], path: Path, *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            safe_row = {k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames}
            w.writerow(safe_row)
    os.replace(tmp, path)


def _json_safe(o: Any) -> Any:
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    try:
        return float(o)
    except Exception:
        return str(o)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------
# Coercion / lookups
# -----------------------------------------------------------
def _to_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _to_int(x: Any, default: int = 0) -> int:
    v = _to_float(x)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _to_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    if not s:
        return default
    return s in {"true", "1", "yes", "y", "t"}


def _norm_symbol(s: Any) -> str:
    return str(s or "").strip().upper()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _first(row: Dict[str, Any], keys: Iterable[str], *, default: Any = None) -> Any:
    """Return the first non-empty value across a list of candidate keys."""
    for k in keys:
        if k in row:
            v = row[k]
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none", "null"):
                return v
    return default


def _action_class(rebal: str) -> str:
    r = (rebal or "").strip().upper()
    if r in BUY_NEW_ACTIONS:
        return "buy_new"
    if r in ADD_ACTIONS:
        return "add"
    if r in TRIM_ACTIONS:
        return "trim"
    if r in SELL_ACTIONS:
        return "sell"
    if r in ROTATION_ACTIONS:
        return "rotation"
    if r in HOLD_ACTIONS:
        return "hold"
    return "unknown"


# -----------------------------------------------------------
# Mode resolution
# -----------------------------------------------------------
def _resolve_mode(
    *,
    auth_summary: Dict[str, Any],
    auth_full: Dict[str, Any],
) -> Tuple[str, str, bool, float]:
    """
    Returns (mode, authorization_state, execution_authorized, auth_conf).
    """
    state = (
        str(
            (auth_summary or {}).get("authorization_state")
            or (auth_full or {}).get("authorization_state")
            or ""
        )
        .strip()
        .upper()
    )
    authorized = bool(
        (auth_summary or {}).get(
            "execution_authorized",
            (auth_full or {}).get("authorization_booleans", {}).get("execution_authorized", False),
        )
    )
    conf = _to_float(
        (auth_summary or {}).get("authorization_confidence")
        or (auth_full or {}).get("authorization_confidence")
    )
    auth_conf = 0.0 if conf is None else _clamp(conf, 0.0, 1.0)

    mode = AUTH_STATE_TO_MODE.get(state, MODE_NO_EXECUTION)
    if not authorized:
        mode = MODE_NO_EXECUTION
    return mode, state or "UNKNOWN", authorized, auth_conf


# -----------------------------------------------------------
# Candidate merge (rebalance plan + execution intents)
# -----------------------------------------------------------
def _merge_candidates(
    rebal_rows: List[Dict[str, str]],
    intent_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    by_ticker: Dict[str, Dict[str, Any]] = {}

    for r in rebal_rows:
        t = _norm_symbol(r.get("ticker"))
        if not t:
            continue
        by_ticker.setdefault(t, {"ticker": t})
        rec = by_ticker[t]
        rec["rebalance_action"] = (
            str(
                _first(
                    r,
                    [
                        "rebalance_action",
                        "portfolio_action",
                        "action",
                    ],
                )
                or ""
            )
            .strip()
            .upper()
        )
        rec["rebalance_amount_usd"] = _to_float(
            _first(
                r,
                [
                    "rebalance_amount_usd",
                    "target_position_size_usd",
                    "estimated_notional",
                    "amount_usd",
                ],
            )
        )
        rec["priority"] = (
            _to_float(
                _first(
                    r,
                    [
                        "priority",
                        "deploy_priority",
                        "rebalance_priority",
                    ],
                )
            )
            or 0.0
        )
        rec["target_weight_pct"] = _to_float(
            _first(
                r,
                [
                    "target_weight_pct",
                    "target_weight",
                ],
            )
        )
        rec["execution_ready"] = _to_bool(_first(r, ["execution_ready"], default=True), True)

    for r in intent_rows:
        t = _norm_symbol(r.get("ticker"))
        if not t:
            continue
        rec = by_ticker.setdefault(t, {"ticker": t})
        rec["execution_intent"] = (
            str(
                _first(
                    r,
                    [
                        "execution_intent",
                        "intent",
                    ],
                )
                or ""
            )
            .strip()
            .upper()
        )
        # Intent CSV often carries the most up-to-date scoring fields,
        # so let it overwrite rebalance row defaults where present.
        for key, src_keys in (
            ("confidence", ["confidence", "latest_confidence"]),
            ("persistence_score", ["persistence_score", "latest_persistence"]),
            ("intent_score", ["intent_score"]),
            ("delta_pct", ["delta_pct", "latest_delta_pct"]),
        ):
            v = _to_float(_first(r, src_keys))
            if v is not None:
                rec[key] = v
        # Backfill amount/priority/rebalance_action from intents if
        # absent on the rebalance side.
        if rec.get("rebalance_amount_usd") is None:
            rec["rebalance_amount_usd"] = _to_float(
                _first(
                    r,
                    [
                        "rebalance_amount_usd",
                        "estimated_notional",
                    ],
                )
            )
        if not rec.get("rebalance_action"):
            rec["rebalance_action"] = (
                str(
                    _first(
                        r,
                        [
                            "rebalance_action",
                            "action",
                        ],
                    )
                    or ""
                )
                .strip()
                .upper()
            )
        if rec.get("priority") in (None, 0.0):
            pv = _to_float(_first(r, ["priority"]))
            if pv is not None:
                rec["priority"] = pv

    # Default numeric scores when neither input supplied them
    for rec in by_ticker.values():
        for k in ("confidence", "persistence_score", "intent_score"):
            if rec.get(k) is None:
                rec[k] = 0.0
        if rec.get("execution_intent") is None:
            # No intents row means we never saw an execute decision for
            # this ticker -- treat as SKIP so it's filtered at the intent
            # gate (rebalance plan exists but execution intent engine
            # never authorised it).
            rec["execution_intent"] = "SKIP"

    return list(by_ticker.values())


# -----------------------------------------------------------
# Per-action evaluation
# -----------------------------------------------------------
def _build_thresholds(policy: Dict[str, Any]) -> Dict[str, float]:
    return {
        "confidence_threshold": _to_float(policy.get("confidence_threshold"))
        or DEFAULT_CONFIDENCE_THRESHOLD,
        "persistence_threshold": _to_float(policy.get("persistence_threshold"))
        or DEFAULT_PERSISTENCE_THRESHOLD,
        "deployment_threshold": _to_float(policy.get("deployment_threshold"))
        or DEFAULT_DEPLOYMENT_THRESHOLD,
        "max_position_pct": _to_float(policy.get("max_position_pct")) or DEFAULT_MAX_POSITION_PCT,
        "max_new_positions_per_cycle": _to_int(
            policy.get("max_new_positions_per_cycle"), default=DEFAULT_MAX_NEW_POSITIONS
        ),
        "target_cash_pct": _to_float(policy.get("target_cash_pct")) or 0.0,
    }


def _check_permission(action_cls: str, perms: Dict[str, bool]) -> Optional[str]:
    if action_cls in ("buy_new", "add"):
        if not perms.get("allow_new_buys", False):
            return "permission:allow_new_buys=False"
        return None
    if action_cls == "sell":
        if not perms.get("allow_sell_exits", False):
            return "permission:allow_sell_exits=False"
        return None
    if action_cls == "trim":
        if not perms.get("allow_rebalance", False) and not perms.get("allow_sell_exits", False):
            return "permission:allow_rebalance=False+allow_sell_exits=False"
        return None
    if action_cls == "rotation":
        if not perms.get("allow_rotation", False):
            return "permission:allow_rotation=False"
        return None
    if action_cls == "hold":
        return "no_action_required:HOLD"
    if action_cls == "unknown":
        return "unknown_action_class"
    return None


def _check_mode_gate(action_cls: str, mode: str) -> Optional[str]:
    if mode == MODE_NO_EXECUTION:
        return "mode:NO_EXECUTION"
    if mode == MODE_EXIT_ONLY and action_cls in ("buy_new", "add", "rotation"):
        return f"mode:EXIT_ONLY blocks {action_cls}"
    if mode == MODE_DEFENSIVE_DEPLOYMENT and action_cls == "rotation":
        # Defensive mode permits *defensive* rotation but not general
        # opportunistic rotation. Action-class "rotation" is the
        # general flavour; defensive rotation typically rides through
        # as a buy_new with defensive context, not a rotation row.
        return "mode:DEFENSIVE_DEPLOYMENT blocks generic rotation"
    return None


def _check_runtime_thresholds(
    rec: Dict[str, Any],
    action_cls: str,
    mode: str,
    thresholds: Dict[str, float],
) -> Optional[str]:
    if action_cls not in ("buy_new", "add"):
        return None
    conf = float(rec.get("confidence") or 0.0)
    persist = float(rec.get("persistence_score") or 0.0)
    intent = float(rec.get("intent_score") or 0.0)
    target_w = _to_float(rec.get("target_weight_pct"))

    conf_floor = thresholds["confidence_threshold"]
    if mode == MODE_DEFENSIVE_DEPLOYMENT:
        conf_floor = min(0.95, conf_floor + DEFENSIVE_CONFIDENCE_BUMP)

    if conf < conf_floor:
        return f"runtime:confidence {conf:.3f}<{conf_floor:.3f}"
    if persist < thresholds["persistence_threshold"]:
        return f"runtime:persistence {persist:.3f}<{thresholds['persistence_threshold']:.3f}"
    if intent < thresholds["deployment_threshold"]:
        return f"runtime:intent_score {intent:.3f}<{thresholds['deployment_threshold']:.3f}"
    if target_w is not None and target_w > thresholds["max_position_pct"]:
        return f"runtime:target_weight {target_w:.2f}%>{thresholds['max_position_pct']:.2f}%"
    return None


def _evaluate_candidate(
    rec: Dict[str, Any],
    *,
    mode: str,
    perms: Dict[str, bool],
    thresholds: Dict[str, float],
    auth_state: str,
) -> Dict[str, Any]:
    rebal = str(rec.get("rebalance_action") or "").strip().upper()
    intent = str(rec.get("execution_intent") or "").strip().upper()
    action_cls = _action_class(rebal)

    allowed = False
    blocked_reason: Optional[str] = None
    rationale = ""

    # Order matters: mode first (NO_EXECUTION shortcuts everything),
    # then intent, then permission, then runtime thresholds.
    mode_block = _check_mode_gate(action_cls, mode)
    if mode_block:
        blocked_reason = mode_block
        rationale = f"Blocked by execution mode: {mode_block}"
    elif intent and intent != INTENT_EXECUTE:
        blocked_reason = f"intent:{intent or 'EMPTY'}"
        rationale = f"Blocked at upstream intent gate (intent={intent or 'EMPTY'}; only EXECUTE_NOW is actionable)"
    else:
        perm_block = _check_permission(action_cls, perms)
        if perm_block:
            blocked_reason = perm_block
            rationale = f"Blocked by Step 22 action gate: {perm_block}"
        else:
            thresh_block = _check_runtime_thresholds(rec, action_cls, mode, thresholds)
            if thresh_block:
                blocked_reason = thresh_block
                rationale = f"Blocked by Step 18 runtime policy: {thresh_block}"
            else:
                allowed = True
                rationale = (
                    f"Authorized: action_class={action_cls}, intent=EXECUTE_NOW, "
                    f"confidence={float(rec.get('confidence') or 0.0):.2f}, "
                    f"intent_score={float(rec.get('intent_score') or 0.0):.2f}"
                )

    return {
        "ticker": rec["ticker"],
        "action": action_cls,
        "authorization_state": auth_state,
        "execution_mode": mode,
        "rebalance_action": rebal,
        "execution_intent": intent or "EMPTY",
        "confidence": float(rec.get("confidence") or 0.0),
        "persistence_score": float(rec.get("persistence_score") or 0.0),
        "intent_score": float(rec.get("intent_score") or 0.0),
        "target_weight_pct": _to_float(rec.get("target_weight_pct")),
        "estimated_notional_usd": _to_float(rec.get("rebalance_amount_usd")),
        "priority": float(rec.get("priority") or 0.0),
        "allowed": bool(allowed),
        "blocked_reason": blocked_reason,
        "rationale": rationale,
    }


# -----------------------------------------------------------
# Position-cap enforcement (post-evaluation)
# -----------------------------------------------------------
def _enforce_position_cap(
    rows: List[Dict[str, Any]],
    *,
    max_new: int,
) -> List[Dict[str, Any]]:
    if max_new is None or max_new < 0:
        return rows
    new_pos_rows = [r for r in rows if r["allowed"] and r["action"] == "buy_new"]
    if len(new_pos_rows) <= max_new:
        return rows
    new_pos_rows.sort(key=lambda r: (-r["priority"], -r["intent_score"], -r["confidence"]))
    keep_tickers = {r["ticker"] for r in new_pos_rows[:max_new]}
    for r in rows:
        if r["allowed"] and r["action"] == "buy_new" and r["ticker"] not in keep_tickers:
            r["allowed"] = False
            r["blocked_reason"] = f"runtime:max_new_positions_per_cycle={max_new}"
            r["rationale"] = (
                f"Blocked by position cap: only {max_new} new position(s) "
                f"per cycle; lower-priority candidate dropped"
            )
    return rows


# -----------------------------------------------------------
# Plan confidence
# -----------------------------------------------------------
def _plan_confidence(
    *,
    auth_conf: float,
    committee_conf: float,
    authorized_buy_rows: List[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    if authorized_buy_rows:
        avg_intent = mean(r["intent_score"] for r in authorized_buy_rows)
        avg_conf = mean(r["confidence"] for r in authorized_buy_rows)
        floor = thresholds["confidence_threshold"]
        if floor < 1.0:
            runtime_floor = _clamp((avg_conf - floor) / (1.0 - floor), 0.0, 1.0)
        else:
            runtime_floor = 1.0 if avg_conf >= floor else 0.0
    else:
        # No buy actions to enforce thresholds against -- runtime floor
        # signal is uninformative; mirror authorization_confidence so
        # the blend stays well-defined and an exits-only plan still
        # gets a meaningful plan_confidence.
        avg_intent = auth_conf
        runtime_floor = auth_conf

    contributors = {
        "authorization_confidence": _clamp(auth_conf, 0.0, 1.0),
        "avg_intent_score": _clamp(avg_intent, 0.0, 1.0),
        "runtime_confidence_floor": _clamp(runtime_floor, 0.0, 1.0),
        "committee_confidence": _clamp(committee_conf, 0.0, 1.0),
    }
    total_w = sum(PLAN_CONFIDENCE_WEIGHTS.values()) or 1.0
    blended = (
        sum(PLAN_CONFIDENCE_WEIGHTS[k] * contributors[k] for k in PLAN_CONFIDENCE_WEIGHTS) / total_w
    )
    return _clamp(blended, 0.0, 1.0), {k: round(v, 6) for k, v in contributors.items()}


# -----------------------------------------------------------
# Aggregations
# -----------------------------------------------------------
def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    allowed = [r for r in rows if r["allowed"]]
    denied = [r for r in rows if not r["allowed"]]
    buy_rows = [r for r in allowed if r["action"] in ("buy_new", "add")]
    sell_rows = [r for r in allowed if r["action"] in ("sell", "trim")]
    rebal_rows = [r for r in allowed if r["action"] in ("trim", "rotation")]
    blocked_rows = denied

    est_deploy = sum((r["estimated_notional_usd"] or 0.0) for r in buy_rows)
    est_exit = sum(abs(r["estimated_notional_usd"] or 0.0) for r in sell_rows)

    return {
        "n_candidates": len(rows),
        "authorized_actions": len(allowed),
        "denied_actions": len(denied),
        "buy_actions": len(buy_rows),
        "sell_actions": len(sell_rows),
        "rebalance_actions": len(rebal_rows),
        "blocked_actions": len(blocked_rows),
        "estimated_total_deployment_usd": round(est_deploy, 2),
        "estimated_total_exit_usd": round(est_exit, 2),
        "top_authorized_tickers": [
            r["ticker"]
            for r in sorted(allowed, key=lambda r: (-r["priority"], -r["intent_score"]))[:5]
        ],
        "top_blocked_tickers": [
            r["ticker"] for r in sorted(denied, key=lambda r: -r["priority"])[:5]
        ],
    }


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    mode: str,
    rows: List[Dict[str, Any]],
    agg: Dict[str, Any],
    thresholds: Dict[str, float],
    plan_conf: float,
) -> List[str]:
    recs: List[str] = []
    if mode == MODE_NO_EXECUTION:
        recs.append(
            "No execution authorized today -- treat plan as advisory only "
            "and do not place any orders."
        )
    elif mode == MODE_EXIT_ONLY:
        recs.append(
            f"Execute the {agg['sell_actions']} authorized exit/trim action(s) "
            "only; do not place new buys."
        )
    elif mode == MODE_DEFENSIVE_DEPLOYMENT:
        recs.append(
            f"Limit deployment to the {agg['buy_actions']} high-conviction "
            f"defensive candidate(s) that cleared the elevated "
            f"+{DEFENSIVE_CONFIDENCE_BUMP:.2f} confidence floor."
        )
    elif mode == MODE_SELECTIVE_DEPLOYMENT:
        recs.append(
            f"Permit selective deployment of the {agg['buy_actions']} authorized buy(s) "
            "and the corresponding rebalance actions."
        )
    elif mode == MODE_FULL_DEPLOYMENT:
        recs.append(
            f"Full deployment authorized: {agg['buy_actions']} buy(s), "
            f"{agg['sell_actions']} sell(s), within runtime policy bounds."
        )

    if agg["blocked_actions"] > 0:
        # Surface the most common blocked_reason to nudge the operator
        # toward the bottleneck (low conviction, missing intent, etc.)
        reasons: Dict[str, int] = {}
        for r in rows:
            if not r["allowed"] and r["blocked_reason"]:
                family = r["blocked_reason"].split(":", 1)[0]
                reasons[family] = reasons.get(family, 0) + 1
        if reasons:
            top = max(reasons.items(), key=lambda kv: kv[1])
            recs.append(
                f"{agg['blocked_actions']} candidate action(s) blocked; "
                f"dominant family={top[0]} ({top[1]} row(s))."
            )

    if plan_conf < 0.40 and mode != MODE_NO_EXECUTION:
        recs.append(
            f"Plan confidence weak ({plan_conf:.2f}) -- treat as preliminary; "
            "consider waiting for next cycle before honouring."
        )
    if agg["buy_actions"] > 0 and thresholds["target_cash_pct"] >= 30.0:
        recs.append(
            f"Cash reserve target elevated ({thresholds['target_cash_pct']:.1f}%) "
            "-- verify any new buys do not breach the reserve floor."
        )

    if not recs:
        recs.append("No recommendations beyond mode default.")
    return recs


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    mode: str,
    auth_state: str,
    execution_authorized: bool,
    plan_rows: List[Dict[str, Any]],
    agg: Dict[str, Any],
    thresholds: Dict[str, float],
    plan_conf: float,
    plan_contribs: Dict[str, float],
    recommendations: List[str],
) -> str:
    def yn(b: bool) -> str:
        return "yes" if b else "no"

    def fmt_money(v: Optional[float]) -> str:
        if v is None:
            return "-"
        return f"${v:,.0f}"

    def fmt_pct(v: Optional[float]) -> str:
        if v is None:
            return "-"
        return f"{v:.2f}%"

    allowed = [r for r in plan_rows if r["allowed"]]
    denied = [r for r in plan_rows if not r["allowed"]]

    lines: List[str] = []
    lines.append("# Triton Autonomous Execution Plan")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Execution Mode")
    lines.append("")
    lines.append(f"**{mode}**")
    lines.append("")
    lines.append(f"- upstream authorization_state: {auth_state}")
    lines.append(f"- execution_authorized: **{yn(execution_authorized)}**")
    lines.append(f"- plan_confidence: **{plan_conf:.3f}**")
    lines.append(f"- candidates evaluated: {agg['n_candidates']}")
    lines.append(f"- authorized: {agg['authorized_actions']} | denied: {agg['denied_actions']}")
    lines.append("")

    lines.append("## Authorized Actions")
    lines.append("")
    if allowed:
        lines.append("| ticker | action | confidence | intent | target_w | notional |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(allowed, key=lambda r: (-r["priority"], -r["intent_score"])):
            lines.append(
                f"| {r['ticker']} | {r['action']} | {r['confidence']:.2f} | "
                f"{r['intent_score']:.2f} | {fmt_pct(r['target_weight_pct'])} | "
                f"{fmt_money(r['estimated_notional_usd'])} |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append("## Denied Actions")
    lines.append("")
    if denied:
        lines.append("| ticker | proposed_action | blocked_reason |")
        lines.append("|---|---|---|")
        for r in sorted(denied, key=lambda r: -r["priority"]):
            lines.append(f"| {r['ticker']} | {r['action']} | {r['blocked_reason'] or '-'} |")
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append("## Runtime Constraints")
    lines.append("")
    lines.append("| threshold | value |")
    lines.append("|---|---|")
    for k in (
        "confidence_threshold",
        "persistence_threshold",
        "deployment_threshold",
        "max_position_pct",
        "max_new_positions_per_cycle",
        "target_cash_pct",
    ):
        v = thresholds.get(k)
        if k.endswith("_pct"):
            lines.append(f"| {k} | {fmt_pct(v)} |")
        elif k == "max_new_positions_per_cycle":
            lines.append(f"| {k} | {int(v) if v is not None else '-'} |")
        else:
            lines.append(f"| {k} | {v:.3f} |" if v is not None else f"| {k} | - |")
    if mode == MODE_DEFENSIVE_DEPLOYMENT:
        lines.append(
            f"| defensive_confidence_floor | {thresholds['confidence_threshold'] + DEFENSIVE_CONFIDENCE_BUMP:.3f} |"
        )
    lines.append("")

    lines.append("## Deployment Totals")
    lines.append("")
    lines.append(
        f"- estimated_total_deployment: {fmt_money(agg['estimated_total_deployment_usd'])}"
    )
    lines.append(f"- estimated_total_exit: {fmt_money(agg['estimated_total_exit_usd'])}")
    lines.append(f"- buy_actions: {agg['buy_actions']}")
    lines.append(f"- sell_actions: {agg['sell_actions']}")
    lines.append(f"- rebalance_actions: {agg['rebalance_actions']}")
    lines.append(f"- blocked_actions: {agg['blocked_actions']}")
    lines.append("")
    lines.append("**Plan confidence contributors:**")
    lines.append("")
    lines.append("| contributor | score | weight |")
    lines.append("|---|---|---|")
    for k, w in PLAN_CONFIDENCE_WEIGHTS.items():
        lines.append(f"| {k} | {plan_contribs[k]:.3f} | {w:.2f} |")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    if mode == MODE_NO_EXECUTION:
        narrative = (
            f"Execution is not authorized today (authorization state {auth_state}). "
            f"The plan contains {agg['n_candidates']} candidate row(s); every "
            "candidate is denied. No orders should be placed."
        )
    else:
        narrative = (
            f"Execution mode {mode} permits {agg['authorized_actions']} of "
            f"{agg['n_candidates']} candidate action(s) ({agg['buy_actions']} buy(s), "
            f"{agg['sell_actions']} sell(s)) within runtime policy bounds. "
            f"Estimated deployment {fmt_money(agg['estimated_total_deployment_usd'])}, "
            f"estimated exit {fmt_money(agg['estimated_total_exit_usd'])}. "
            f"Plan confidence {plan_conf:.2f}."
        )
    lines.append(narrative)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
PLAN_CSV_FIELDS: List[str] = [
    "ticker",
    "action",
    "authorization_state",
    "execution_mode",
    "rebalance_action",
    "execution_intent",
    "confidence",
    "persistence_score",
    "intent_score",
    "target_weight_pct",
    "estimated_notional_usd",
    "priority",
    "allowed",
    "blocked_reason",
    "rationale",
]


def build_execution_plan(
    *,
    auth_summary: Dict[str, Any],
    auth_full: Dict[str, Any],
    action_permissions: Dict[str, Any],
    intent_rows: List[Dict[str, str]],
    rebal_rows: List[Dict[str, str]],
    runtime_policy: Dict[str, Any],
    committee_summary: Dict[str, Any],
    regime_json: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, Dict[str, Any]]:
    mode, auth_state, execution_authorized, auth_conf = _resolve_mode(
        auth_summary=auth_summary,
        auth_full=auth_full,
    )
    perms = dict((action_permissions or {}).get("permissions") or {})
    thresholds = _build_thresholds(runtime_policy)

    candidates = _merge_candidates(rebal_rows, intent_rows)
    plan_rows = [
        _evaluate_candidate(
            rec,
            mode=mode,
            perms=perms,
            thresholds=thresholds,
            auth_state=auth_state,
        )
        for rec in candidates
    ]
    plan_rows = _enforce_position_cap(
        plan_rows,
        max_new=thresholds["max_new_positions_per_cycle"],
    )

    committee_conf = _to_float((committee_summary or {}).get("recommendation_confidence")) or 0.5
    authorized_buy_rows = [
        r for r in plan_rows if r["allowed"] and r["action"] in ("buy_new", "add")
    ]
    plan_conf, plan_contribs = _plan_confidence(
        auth_conf=auth_conf,
        committee_conf=committee_conf,
        authorized_buy_rows=authorized_buy_rows,
        thresholds=thresholds,
    )

    agg = _aggregate(plan_rows)
    recommendations = _build_recommendations(
        mode=mode,
        rows=plan_rows,
        agg=agg,
        thresholds=thresholds,
        plan_conf=plan_conf,
    )

    now_iso = _now_iso_utc()
    regime = (
        str((regime_json or {}).get("regime") or (runtime_policy or {}).get("regime") or "")
        .strip()
        .upper()
        or "UNKNOWN"
    )

    plan_json: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_planner_engine",
        "engine_version": 1,
        "execution_mode": mode,
        "authorization_state": auth_state,
        "execution_authorized": execution_authorized,
        "regime": regime,
        "plan_confidence": round(plan_conf, 6),
        "plan_confidence_contributors": plan_contribs,
        "plan_confidence_weights": PLAN_CONFIDENCE_WEIGHTS,
        "runtime_thresholds": {
            k: (int(v) if k == "max_new_positions_per_cycle" else round(v, 6))
            for k, v in thresholds.items()
        },
        "aggregates": agg,
        "actions": plan_rows,
        "recommendations": recommendations,
        "upstream_context": {
            "authorization_confidence": round(auth_conf, 6),
            "committee_confidence": round(committee_conf, 6),
            "permissions": perms,
            "regime": regime,
        },
        "inputs_seen": {
            "autonomous_execution_summary": bool(auth_summary),
            "autonomous_execution_authorization": bool(auth_full),
            "autonomous_action_permissions": bool(action_permissions),
            "portfolio_execution_intents_rows": len(intent_rows),
            "portfolio_rebalance_plan_rows": len(rebal_rows),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_committee_summary": bool(committee_summary),
            "adaptive_regime": bool(regime_json),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_planner_engine",
        "execution_mode": mode,
        "execution_authorized": execution_authorized,
        "authorization_state": auth_state,
        "plan_confidence": round(plan_conf, 6),
        "authorized_actions": agg["authorized_actions"],
        "denied_actions": agg["denied_actions"],
        "buy_actions": agg["buy_actions"],
        "sell_actions": agg["sell_actions"],
        "rebalance_actions": agg["rebalance_actions"],
        "blocked_actions": agg["blocked_actions"],
        "estimated_total_deployment_usd": agg["estimated_total_deployment_usd"],
        "estimated_total_exit_usd": agg["estimated_total_exit_usd"],
        "top_authorized_tickers": agg["top_authorized_tickers"],
        "top_blocked_tickers": agg["top_blocked_tickers"],
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        generated_at=now_iso,
        mode=mode,
        auth_state=auth_state,
        execution_authorized=execution_authorized,
        plan_rows=plan_rows,
        agg=agg,
        thresholds=thresholds,
        plan_conf=plan_conf,
        plan_contribs=plan_contribs,
        recommendations=recommendations,
    )
    return plan_json, plan_rows, md, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous execution planner (Step 24). Filters "
            "Step 6/7 candidate actions through Step 22 permissions + "
            "Step 18 runtime thresholds, gated by Step 23 authorization, "
            "to produce the final authorized execution plan. Writes JSON, "
            "CSV, markdown, and a compact summary. Places no orders."
        ),
    )
    p.add_argument("--auth-summary", default=str(DEFAULT_AUTH_SUMMARY))
    p.add_argument("--auth-full", default=str(DEFAULT_AUTH_FULL))
    p.add_argument("--action-permissions", default=str(DEFAULT_ACTION_PERMS))
    p.add_argument("--exec-intents", default=str(DEFAULT_EXEC_INTENTS_CSV))
    p.add_argument("--rebalance-plan", default=str(DEFAULT_REBALANCE_CSV))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[EXECUTION_PLAN] starting (read-only -- final authorized plan)", flush=True)

    auth_summary = _safe_read_json(
        Path(args.auth_summary), label="autonomous_execution_summary.json"
    )
    auth_full = _safe_read_json(
        Path(args.auth_full), label="autonomous_execution_authorization.json"
    )
    action_permissions = _safe_read_json(
        Path(args.action_permissions), label="autonomous_action_permissions.json"
    )
    intent_rows = _safe_read_csv_rows(
        Path(args.exec_intents), label="portfolio_execution_intents.csv"
    )
    rebal_rows = _safe_read_csv_rows(
        Path(args.rebalance_plan), label="portfolio_rebalance_plan.csv"
    )
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_committee_summary.json"
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")

    plan_json, plan_rows, md, summary = build_execution_plan(
        auth_summary=auth_summary,
        auth_full=auth_full,
        action_permissions=action_permissions,
        intent_rows=intent_rows,
        rebal_rows=rebal_rows,
        runtime_policy=runtime_policy,
        committee_summary=committee_summary,
        regime_json=regime_json,
    )

    try:
        _atomic_write_json(plan_json, Path(args.out_json))
    except Exception as e:
        _warn(f"failed to write {args.out_json}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_csv(plan_rows, Path(args.out_csv), fieldnames=PLAN_CSV_FIELDS)
    except Exception as e:
        _warn(f"failed to write {args.out_csv}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_text(md, Path(args.out_md))
    except Exception as e:
        _warn(f"failed to write {args.out_md}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2

    agg = plan_json["aggregates"]
    print(
        "[EXECUTION_PLAN] "
        f"mode={plan_json['execution_mode']} "
        f"actions={agg['authorized_actions']} "
        f"blocked={agg['blocked_actions']} "
        f"confidence={plan_json['plan_confidence']:.3f} "
        f"deployment=${agg['estimated_total_deployment_usd']:,.0f}",
        flush=True,
    )
    if summary["top_authorized_tickers"]:
        print(
            "[EXECUTION_PLAN_TOP] " + ",".join(summary["top_authorized_tickers"]),
            flush=True,
        )
    print(
        f"[EXECUTION_PLAN_OUT] json={Path(args.out_json).as_posix()} "
        f"csv={Path(args.out_csv).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
