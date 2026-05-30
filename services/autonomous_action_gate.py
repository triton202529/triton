"""
Autonomous Action Gate Engine -- Step 22.

Reads:
    data/results/autonomous_readiness_summary.json     (Step 21)
    data/results/autonomous_system_health_summary.json (Step 20)
    data/results/runtime_policy_governed.json          (Step 18)
    data/results/meta_decision_intelligence.json       (Step 13)
    data/results/governance_trust_feedback.json        (Step 17)

Writes:
    data/results/autonomous_action_permissions.json
    data/results/autonomous_action_summary.json

Purpose
-------
Step 21 produces a five-state readiness verdict. Step 22 is the
*one* universal authorization layer that every future autonomous
component must consult:

    "What actions is Triton allowed to take right now?"

A downstream automation engine does not re-derive permissions from
the readiness state, the health flags, the runtime policy, or any
combination -- it queries exactly one file
(``autonomous_action_permissions.json``) and inspects the eight
boolean fields it contains. That single source of truth eliminates
the risk of two engines disagreeing on whether new buys are
allowed today.

Permission matrix (spec section 2)
----------------------------------
    state          new_buys  sell_exits  rebalance  rotation  aggressive  defensive_rot  read_only  review_required
    BLOCKED        F         F           F          F         F           F              T          T
    READ_ONLY      F         F           F          F         F           F              T          T
    NOT_READY      F         T           T          F         F           F              T          F
    READY_LIMITED  T         T           T          T         F           T              T          F
    READY          T         T           T          T         T           T              T          F

The matrix above represents the *baseline* from Step 21's state.
On top of it, this engine applies safety overrides drawn from
Steps 13, 17, 18, and 20 that can only ever DOWNGRADE a permission
from True to False -- never the other way around. This keeps the
layer additive and impossible to use as an autonomy *expansion*
mechanism.

Safety
------
* READ ONLY. No broker calls, no engine state mutation. The
  permissions are advisory but authoritative; downstream engines
  are responsible for honouring them.
* Overrides are monotonic (T -> F only); the matrix encodes the
  upper bound for each state.
* Atomic writes (.tmp + os.replace).
* Missing inputs warn-and-continue. With zero inputs the gate
  defaults to readiness_state=BLOCKED, which sets every
  ``allow_*`` field to False and forces operator review.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_READINESS_SUMMARY = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_GOV_FEEDBACK = RESULTS_DIR / "governance_trust_feedback.json"

DEFAULT_OUT_PERMISSIONS = RESULTS_DIR / "autonomous_action_permissions.json"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_action_summary.json"


# -----------------------------------------------------------
# Permission keys & matrix
# -----------------------------------------------------------
ALLOW_KEYS: Tuple[str, ...] = (
    "allow_new_buys",
    "allow_sell_exits",
    "allow_rebalance",
    "allow_rotation",
    "allow_aggressive_deployment",
    "allow_defensive_rotation",
    "allow_read_only_analysis",
)
REVIEW_KEY = "require_operator_review"
PERMISSION_KEYS: Tuple[str, ...] = ALLOW_KEYS + (REVIEW_KEY,)

STATE_READY = "READY"
STATE_READY_LIMITED = "READY_LIMITED"
STATE_NOT_READY = "NOT_READY"
STATE_READ_ONLY = "READ_ONLY"
STATE_BLOCKED = "BLOCKED"

ALL_STATES: Tuple[str, ...] = (
    STATE_BLOCKED,
    STATE_READ_ONLY,
    STATE_NOT_READY,
    STATE_READY_LIMITED,
    STATE_READY,
)

# Spec section 2 matrix. Keys/values exactly as documented in the
# module docstring; do not edit one without updating the other.
PERMISSION_MATRIX: Dict[str, Dict[str, bool]] = {
    STATE_BLOCKED: {
        "allow_new_buys": False,
        "allow_sell_exits": False,
        "allow_rebalance": False,
        "allow_rotation": False,
        "allow_aggressive_deployment": False,
        "allow_defensive_rotation": False,
        "allow_read_only_analysis": True,
        "require_operator_review": True,
    },
    STATE_READ_ONLY: {
        "allow_new_buys": False,
        "allow_sell_exits": False,
        "allow_rebalance": False,
        "allow_rotation": False,
        "allow_aggressive_deployment": False,
        "allow_defensive_rotation": False,
        "allow_read_only_analysis": True,
        "require_operator_review": True,
    },
    STATE_NOT_READY: {
        "allow_new_buys": False,
        "allow_sell_exits": True,
        "allow_rebalance": True,
        "allow_rotation": False,
        "allow_aggressive_deployment": False,
        "allow_defensive_rotation": False,
        "allow_read_only_analysis": True,
        "require_operator_review": False,
    },
    STATE_READY_LIMITED: {
        "allow_new_buys": True,  # selective; enforced upstream
        "allow_sell_exits": True,
        "allow_rebalance": True,
        "allow_rotation": True,
        "allow_aggressive_deployment": False,
        "allow_defensive_rotation": True,
        "allow_read_only_analysis": True,
        "require_operator_review": False,
    },
    STATE_READY: {
        "allow_new_buys": True,
        "allow_sell_exits": True,
        "allow_rebalance": True,
        "allow_rotation": True,
        "allow_aggressive_deployment": True,
        "allow_defensive_rotation": True,
        "allow_read_only_analysis": True,
        "require_operator_review": False,
    },
}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ACTION_GATE_WARN] {msg}", flush=True)


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


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_safe)
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
# Coercion
# -----------------------------------------------------------
def _to_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


# -----------------------------------------------------------
# Override engine (monotonic T -> F only)
# -----------------------------------------------------------
def _apply_overrides(
    permissions: Dict[str, bool],
    *,
    health_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    meta_intel: Dict[str, Any],
    feedback: Dict[str, Any],
) -> Tuple[Dict[str, bool], List[Dict[str, Any]]]:
    """
    Apply belt-and-suspenders downgrades that may *only* turn a
    True permission into False. The matrix already encodes the
    upper bound; overrides act as an additional safety net.

    Returns (updated_permissions, override_records[]).
    """
    out = copy.deepcopy(permissions)
    records: List[Dict[str, Any]] = []

    def downgrade(field: str, reason: str, source: str) -> None:
        if field in out and out[field] is True:
            out[field] = False
            records.append(
                {
                    "field": field,
                    "from": True,
                    "to": False,
                    "source": source,
                    "reason": reason,
                }
            )

    health_flags = set(map(str, (health_summary or {}).get("blocking_flags") or []))
    if "BLOCK_NEW_BUYS" in health_flags:
        downgrade(
            "allow_new_buys",
            "Step 20 set BLOCK_NEW_BUYS",
            "autonomous_system_health_summary.blocking_flags",
        )
    if "BLOCK_AUTONOMOUS_DEPLOYMENT" in health_flags:
        downgrade(
            "allow_aggressive_deployment",
            "Step 20 set BLOCK_AUTONOMOUS_DEPLOYMENT",
            "autonomous_system_health_summary.blocking_flags",
        )

    max_new = _to_int((runtime_policy or {}).get("max_new_positions_per_cycle"))
    if max_new is not None and max_new <= 0:
        downgrade(
            "allow_new_buys",
            f"runtime_policy.max_new_positions_per_cycle={max_new} <= 0",
            "runtime_policy_governed.max_new_positions_per_cycle",
        )

    gov_trust = str((feedback or {}).get("governance_trust_level") or "").strip().upper()
    if gov_trust == "COLLAPSED":
        downgrade(
            "allow_aggressive_deployment",
            "governance_trust_level=COLLAPSED",
            "governance_trust_feedback.governance_trust_level",
        )
        downgrade(
            "allow_new_buys",
            "governance_trust_level=COLLAPSED",
            "governance_trust_feedback.governance_trust_level",
        )

    meta_trust = str((meta_intel or {}).get("trust_level") or "").strip().upper()
    if meta_trust == "VERY_LOW":
        downgrade(
            "allow_aggressive_deployment",
            "meta_trust_level=VERY_LOW",
            "meta_decision_intelligence.trust_level",
        )

    return out, records


# -----------------------------------------------------------
# Rationale builder
# -----------------------------------------------------------
def _per_permission_rationale(
    *,
    state: str,
    permissions: Dict[str, bool],
    overrides: List[Dict[str, Any]],
) -> Dict[str, str]:
    overridden = {rec["field"] for rec in overrides}
    out: Dict[str, str] = {}
    for key in PERMISSION_KEYS:
        granted = permissions[key]
        if key in overridden:
            override_reason = next(rec["reason"] for rec in overrides if rec["field"] == key)
            out[key] = f"{key}={granted}: state={state}, overridden -- {override_reason}"
            continue
        if granted:
            out[key] = f"{key}={granted}: state={state} permits this action"
        else:
            out[key] = f"{key}={granted}: state={state} does not permit this action"
    return out


def _build_rationale(
    *,
    state: str,
    permissions: Dict[str, bool],
    overrides: List[Dict[str, Any]],
    readiness_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
) -> Tuple[str, str]:
    allowed = [k for k in ALLOW_KEYS if permissions[k]]
    denied = [k for k in ALLOW_KEYS if not permissions[k]]
    review = permissions[REVIEW_KEY]

    readiness_short = str((readiness_summary or {}).get("rationale_short") or "").strip()
    health_status = str((health_summary or {}).get("overall_status") or "UNKNOWN").strip().upper()

    rationale_short = (
        f"State={state}: {len(allowed)}/{len(ALLOW_KEYS)} permissions allowed; "
        f"operator review {'required' if review else 'not required'}."
    )

    bullets: List[str] = []
    if denied:
        bullets.append(f"denied: {', '.join(denied)}")
    if allowed:
        bullets.append(f"allowed: {', '.join(allowed)}")
    if overrides:
        bullets.append(
            f"{len(overrides)} safety override(s) applied: "
            + "; ".join(f"{r['field']} <- {r['reason']}" for r in overrides)
        )
    if readiness_short:
        bullets.append(f"upstream readiness: {readiness_short}")
    bullets.append(f"upstream health: {health_status}")

    rationale_long = "Action gate decision -- " + ". ".join(bullets) + "."
    return rationale_short, rationale_long


# -----------------------------------------------------------
# State extraction
# -----------------------------------------------------------
def _resolve_state(readiness_summary: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Returns (state, was_valid). When the upstream readiness state is
    missing or unrecognised, we fall back to BLOCKED -- the safest
    possible default.
    """
    raw = str((readiness_summary or {}).get("readiness_state") or "").strip().upper()
    if raw in PERMISSION_MATRIX:
        return raw, True
    if raw:
        _warn(f"unrecognised readiness_state {raw!r}; defaulting to BLOCKED")
    return STATE_BLOCKED, False


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_action_permissions(
    *,
    readiness_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    meta_intel: Dict[str, Any],
    feedback: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state, state_valid = _resolve_state(readiness_summary)
    baseline = copy.deepcopy(PERMISSION_MATRIX[state])
    final, overrides = _apply_overrides(
        baseline,
        health_summary=health_summary,
        runtime_policy=runtime_policy,
        meta_intel=meta_intel,
        feedback=feedback,
    )
    per_action_rationale = _per_permission_rationale(
        state=state,
        permissions=final,
        overrides=overrides,
    )
    rationale_short, rationale_long = _build_rationale(
        state=state,
        permissions=final,
        overrides=overrides,
        readiness_summary=readiness_summary,
        health_summary=health_summary,
    )

    n_allowed = sum(1 for k in ALLOW_KEYS if final[k])
    n_denied = len(ALLOW_KEYS) - n_allowed
    review_required = final[REVIEW_KEY]
    now_iso = _now_iso_utc()

    permissions: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_action_gate",
        "engine_version": 1,
        "readiness_state": state,
        "readiness_state_valid": state_valid,
        "permissions": final,
        "baseline_permissions": baseline,
        "applied_overrides": overrides,
        "per_action_rationale": per_action_rationale,
        "rationale_short": rationale_short,
        "rationale_long": rationale_long,
        "upstream_context": {
            "readiness_score": (readiness_summary or {}).get("readiness_score"),
            "readiness_state_source": "autonomous_readiness_summary.json",
            "health_overall_status": (health_summary or {}).get("overall_status"),
            "health_blocking_flags": list((health_summary or {}).get("blocking_flags") or []),
            "runtime_policy_regime": (runtime_policy or {}).get("regime"),
            "meta_trust_level": (meta_intel or {}).get("trust_level"),
            "governance_trust_level": (feedback or {}).get("governance_trust_level"),
            "governance_active": bool((feedback or {}).get("active", False)),
        },
        "inputs_seen": {
            "autonomous_readiness_summary": bool(readiness_summary),
            "autonomous_system_health_summary": bool(health_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "meta_decision_intelligence": bool(meta_intel),
            "governance_trust_feedback": bool(feedback),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_action_gate",
        "readiness_state": state,
        "permissions": final,
        "rationale_short": rationale_short,
        "n_allowed_actions": n_allowed,
        "n_denied_actions": n_denied,
        "n_overrides_applied": len(overrides),
        "operator_review_required": bool(review_required),
        "any_buy_allowed": final["allow_new_buys"] or final["allow_aggressive_deployment"],
        "any_sell_allowed": final["allow_sell_exits"]
        or final["allow_rebalance"]
        or final["allow_rotation"],
    }
    return permissions, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous action gate (Step 22). Produces eight "
            "boolean permissions that any future autonomous component must "
            "consult before acting. Applies a hard-coded permission matrix "
            "per readiness state and additional safety overrides that can "
            "only downgrade T -> F."
        ),
    )
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUMMARY))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--meta-intel", default=str(DEFAULT_META_INTEL))
    p.add_argument("--gov-feedback", default=str(DEFAULT_GOV_FEEDBACK))
    p.add_argument("--out-permissions", default=str(DEFAULT_OUT_PERMISSIONS))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[ACTION_GATE] starting (readiness + safety -> universal permissions)", flush=True)

    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    meta_intel = _safe_read_json(Path(args.meta_intel), label="meta_decision_intelligence.json")
    feedback = _safe_read_json(Path(args.gov_feedback), label="governance_trust_feedback.json")

    permissions, summary = build_action_permissions(
        readiness_summary=readiness_summary,
        health_summary=health_summary,
        runtime_policy=runtime_policy,
        meta_intel=meta_intel,
        feedback=feedback,
    )

    try:
        _atomic_write_json(permissions, Path(args.out_permissions))
    except Exception as e:
        _warn(f"failed to write {args.out_permissions}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2

    p = permissions["permissions"]
    print(
        "[ACTION_GATE] "
        f"state={permissions['readiness_state']} "
        f"new_buys={p['allow_new_buys']} "
        f"rebalance={p['allow_rebalance']} "
        f"aggressive={p['allow_aggressive_deployment']} "
        f"review={p[REVIEW_KEY]}",
        flush=True,
    )
    if permissions["applied_overrides"]:
        for rec in permissions["applied_overrides"]:
            print(
                f"[ACTION_GATE_OVERRIDE] {rec['field']}: True -> False " f"({rec['reason']})",
                flush=True,
            )
    print(
        f"[ACTION_GATE_OUT] permissions={Path(args.out_permissions).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
