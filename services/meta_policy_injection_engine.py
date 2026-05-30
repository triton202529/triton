"""
Meta Policy Injection Engine -- Step 14 (self-trust -> policy bridge).

Reads:
    data/results/runtime_policy.json
    data/results/meta_decision_intelligence.json
    data/results/meta_runtime_adjustments.json
    data/results/adaptive_policy.json     (optional, provenance fallback)

Writes:
    data/results/runtime_policy_meta.json
    data/results/meta_policy_summary.json

Purpose
-------
Step 11 (policy_override_engine) compiles the regime-driven runtime
policy. Step 13 (meta_decision_engine) measures how much Triton
should trust itself today and emits six bounded modifiers. This
engine is the bridge that *applies* those modifiers to the policy,
turning meta self-confidence into concrete threshold movement:

    "How should Triton alter policy based on trust in itself?"

Modifier application (spec section 2)
-------------------------------------
    confidence_threshold += confidence_modifier
    persistence_threshold += persistence_modifier
    deployment_threshold  -= deployment_modifier
    execution_threshold   -= execution_modifier
    target_cash_pct       += cash_modifier * 100
    max_position_pct      += aggressiveness_modifier * 2

The signs match Step 13's convention: positive modifier => stricter
gate / more cash / more aggressive, depending on the field; high
trust pushes the opposite direction. Each output field is then
clamped into the safe operating range from spec section 3:

    confidence_threshold / persistence_threshold : 0.45 .. 0.85
    deployment_threshold / execution_threshold   : 0.40 .. 0.90
    target_cash_pct                              : 5    .. 40
    max_position_pct                             : 2    .. 10

`execution_threshold` is a *new* top-level field (no Step 11 analogue);
it is also mirrored into ``aliases.min_execute_intent_score`` so any
downstream that prefers the alias picks it up cleanly. The
``deployment_threshold`` field (the slot Step 11 uses as the
``min_execute_intent_score`` source) retains its spec-literal
modifier application; downstream engines that read it directly
still get the meta-blended value.

Provenance (spec section 4)
---------------------------
Every output carries:
    * adaptive_regime / regime
    * meta_trust_level
    * self_confidence_score
    * modifier_summary  (the six numeric inputs)
    * generated_at_utc
    * policy_version

Plus a per-field ``meta_changes`` diff so the summary file is a
self-explaining audit trail.

Safety
------
* READ ONLY. Modifiers are *additive* on top of the regime policy.
  No execution mutation, no engine state mutation.
* Every modifier-application result is clamped to the spec range.
* Atomic writes (.tmp + os.replace) for both outputs.
* Missing inputs warn-and-continue. If runtime_policy.json is absent
  the engine still emits a degraded output (FALLBACK_POLICY) and
  marks ``degraded=True`` so consumers can refuse to use it.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
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

# Read the regime-only baseline first (preserved by Step 11 each cycle
# as `runtime_policy_base.json`); fall back to the canonical
# `runtime_policy.json` if the base snapshot is absent. This prevents
# the meta engine from compounding its own prior overlay across
# cycles when run after Step 11's overlay has already swapped the
# meta values into runtime_policy.json.
DEFAULT_RUNTIME_POLICY_BASE = RESULTS_DIR / "runtime_policy_base.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy.json"
DEFAULT_META_INTELLIGENCE = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_META_ADJ = RESULTS_DIR / "meta_runtime_adjustments.json"
DEFAULT_ADAPTIVE_POLICY = RESULTS_DIR / "adaptive_policy.json"

DEFAULT_OUT_POLICY = RESULTS_DIR / "runtime_policy_meta.json"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "meta_policy_summary.json"

META_POLICY_VERSION = 1

# -----------------------------------------------------------
# Tunables (spec section 3 -- safe clamp ranges)
# -----------------------------------------------------------
CLAMP_RANGES: Dict[str, Tuple[float, float]] = {
    "confidence_threshold": (0.45, 0.85),
    "persistence_threshold": (0.40, 0.85),
    "deployment_threshold": (0.40, 0.90),
    "execution_threshold": (0.40, 0.90),
    "target_cash_pct": (5.0, 40.0),
    "max_position_pct": (2.0, 10.0),
}

# Per-field modifier wiring. Tuple = (modifier_field, sign, scale).
#   adjusted = base + sign * scale * modifier
#
# sign  +1: spec uses ``field += modifier``
# sign  -1: spec uses ``field -= modifier``
#
# scale  1.0: spec uses raw modifier
# scale 100.0: spec multiplies modifier by 100  (cash_modifier -> pct)
# scale   2.0: spec multiplies modifier by 2    (aggressiveness)
MODIFIER_WIRING: Tuple[Tuple[str, str, int, float], ...] = (
    ("confidence_threshold", "confidence_modifier", +1, 1.0),
    ("persistence_threshold", "persistence_modifier", +1, 1.0),
    ("deployment_threshold", "deployment_modifier", -1, 1.0),
    ("execution_threshold", "execution_modifier", -1, 1.0),
    ("target_cash_pct", "cash_modifier", +1, 100.0),
    ("max_position_pct", "aggressiveness_modifier", +1, 2.0),
)

# Fallback runtime policy mirrors Step 11's FALLBACK_POLICY (neutral
# regime). Used only when runtime_policy.json is missing entirely --
# the output is then marked degraded.
FALLBACK_POLICY: Dict[str, Any] = {
    "schema_version": META_POLICY_VERSION,
    "policy_version": META_POLICY_VERSION,
    "engine": "meta_policy_injection_engine",
    "regime": "NEUTRAL",
    "max_position_pct": 6.0,
    "min_position_pct": 0.5,
    "max_sector_pct": 25.0,
    "max_cluster_pct": 30.0,
    "min_cash_reserve_pct": 10.0,
    "max_cash_reserve_pct": 20.0,
    "target_cash_pct": 15.0,
    "max_new_positions_per_cycle": 3,
    "deployment_threshold": 0.55,
    "confidence_threshold": 0.55,
    "persistence_threshold": 0.60,
    "rebalance_frequency": "DAILY",
    "rotation_pressure": 0.50,
    "diversification_aggressiveness": 0.60,
    "risk_tolerance": 0.50,
    "block_strict_mode": False,
}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[META_POLICY_WARN] {msg}", flush=True)


def _safe_read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); continuing without it")
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
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------------------------------------
# Core: apply modifiers
# -----------------------------------------------------------
def _base_value(base_policy: Dict[str, Any], field: str) -> float:
    """
    Pull the starting value for a field from the base runtime policy.

    Special-case: `execution_threshold` does not exist in Step 11's
    schema, so we synthesise it from `deployment_threshold` (the slot
    the existing schema uses as the execution intent floor).
    """
    if field in base_policy and base_policy[field] is not None:
        v = _to_float(base_policy[field])
        if v is not None:
            return v
    if field == "execution_threshold":
        v = _to_float(base_policy.get("deployment_threshold"))
        if v is not None:
            return v
    # Last resort: spec-bound midpoint of the clamp range so a missing
    # base never explodes the output. This only fires for an empty
    # base_policy (degraded path).
    lo, hi = CLAMP_RANGES.get(field, (0.0, 1.0))
    return (lo + hi) / 2.0


def apply_meta_modifiers(
    base_policy: Dict[str, Any],
    modifiers: Dict[str, float],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Apply the six meta modifiers to ``base_policy`` and return
    (new_policy, change_diff). Both are plain dicts. ``change_diff``
    is keyed by the policy field name and contains
    ``{before, after, delta, modifier, modifier_value, clamped}``.

    Pure function -- no IO, no global state. Easy to unit-test.
    """
    new_policy: Dict[str, Any] = dict(base_policy)
    changes: Dict[str, Dict[str, Any]] = {}

    for field, mod_name, sign, scale in MODIFIER_WIRING:
        mod_value = _to_float(modifiers.get(mod_name))
        if mod_value is None:
            mod_value = 0.0
        before = _base_value(base_policy, field)
        delta = sign * scale * mod_value
        proposed = before + delta
        lo, hi = CLAMP_RANGES[field]
        after = _clamp(proposed, lo, hi)
        clamped = after != proposed
        new_policy[field] = round(after, 6)
        changes[field] = {
            "before": round(before, 6),
            "after": round(after, 6),
            "delta": round(after - before, 6),
            "raw_delta": round(delta, 6),
            "proposed_before_clamp": round(proposed, 6),
            "clamp_lo": lo,
            "clamp_hi": hi,
            "clamped": bool(clamped),
            "modifier": mod_name,
            "modifier_value": round(mod_value, 6),
            "modifier_scale": scale,
            "modifier_sign": sign,
        }

    # Cash discipline ordering: keep min <= target <= max after the cash bump.
    cmin = _to_float(new_policy.get("min_cash_reserve_pct"))
    ctgt = _to_float(new_policy.get("target_cash_pct"))
    cmax = _to_float(new_policy.get("max_cash_reserve_pct"))
    if cmin is not None and ctgt is not None and ctgt < cmin:
        new_policy["min_cash_reserve_pct"] = ctgt
    if cmax is not None and ctgt is not None and cmax < ctgt:
        new_policy["max_cash_reserve_pct"] = ctgt

    # Refresh aliases so downstream alias-consumers see the meta values.
    aliases = dict(base_policy.get("aliases") or {})
    aliases.update(
        {
            "max_single_position_pct": new_policy["max_position_pct"],
            "min_deploy_confidence": new_policy["confidence_threshold"],
            "deploy_persistence_floor": new_policy["persistence_threshold"],
            "min_execute_confidence": new_policy["confidence_threshold"],
            "min_execute_persistence": new_policy["persistence_threshold"],
            # Step 14 introduces execution_threshold as the dedicated
            # final-gate floor; alias mirrors it.
            "min_execute_intent_score": new_policy["execution_threshold"],
            "target_cash_reserve_pct": new_policy["target_cash_pct"],
        }
    )
    new_policy["aliases"] = aliases

    return new_policy, changes


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_meta_policy(
    *,
    runtime_policy: Dict[str, Any],
    meta_intelligence: Dict[str, Any],
    meta_adjustments: Dict[str, Any],
    adaptive_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Produce (runtime_policy_meta, meta_policy_summary).

    Both outputs are JSON-ready dicts. The caller writes them to disk.
    """
    now_iso = _now_iso_utc()

    base = dict(runtime_policy) if runtime_policy else dict(FALLBACK_POLICY)
    degraded = not bool(runtime_policy)

    modifiers: Dict[str, float] = {}
    raw_mods = (meta_adjustments or {}).get("modifiers") or {}
    for _, mod_name, _, _ in MODIFIER_WIRING:
        v = _to_float(raw_mods.get(mod_name))
        modifiers[mod_name] = 0.0 if v is None else float(v)

    new_policy, changes = apply_meta_modifiers(base, modifiers)

    regime = (
        str(
            base.get("regime")
            or (meta_intelligence or {}).get("regime")
            or (adaptive_policy or {}).get("regime")
            or "UNKNOWN"
        )
        .strip()
        .upper()
        or "UNKNOWN"
    )

    trust_level = (
        str(
            (meta_intelligence or {}).get("trust_level")
            or (meta_adjustments or {}).get("trust_level")
            or "MODERATE"
        )
        .strip()
        .upper()
    )
    self_conf = _to_float(
        (meta_intelligence or {}).get("self_confidence_score")
        or (meta_adjustments or {}).get("self_confidence_score")
    )
    self_conf = 0.50 if self_conf is None else _clamp(self_conf, 0.0, 1.0)

    # Stamp Step-14 provenance on top of the base policy contents.
    new_policy.update(
        {
            "schema_version": META_POLICY_VERSION,
            "policy_version": META_POLICY_VERSION,
            "generated_at_utc": now_iso,
            "engine": "meta_policy_injection_engine",
            "regime": regime,
            "meta_trust_level": trust_level,
            "self_confidence_score": round(self_conf, 6),
            "meta_modifier_summary": {k: round(v, 6) for k, v in modifiers.items()},
            "meta_changes": changes,
            "meta_inputs_seen": {
                "runtime_policy": bool(runtime_policy),
                "meta_intelligence": bool(meta_intelligence),
                "meta_adjustments": bool(meta_adjustments),
                "adaptive_policy": bool(adaptive_policy),
            },
            "degraded": bool(degraded),
            "base_engine": base.get("engine", "policy_override_engine"),
            "base_generated_at_utc": base.get("generated_at_utc"),
            "base_rationale_short": base.get("rationale_short"),
        }
    )

    # ---- Summary diff ----
    clamp_events = [f for f, c in changes.items() if c["clamped"]]
    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "meta_policy_injection_engine",
        "engine_version": META_POLICY_VERSION,
        "regime": regime,
        "meta_trust_level": trust_level,
        "self_confidence_score": round(self_conf, 6),
        "modifier_summary": {k: round(v, 6) for k, v in modifiers.items()},
        "field_changes": changes,
        "clamp_events": clamp_events,
        "fields_changed": [f for f, c in changes.items() if c["delta"] != 0.0],
        "degraded": bool(degraded),
        "inputs_seen": new_policy["meta_inputs_seen"],
        "thresholds": {
            "clamp_ranges": {k: list(v) for k, v in CLAMP_RANGES.items()},
        },
        "base_runtime_policy": {
            "engine": base.get("engine"),
            "generated_at_utc": base.get("generated_at_utc"),
            "regime": base.get("regime"),
            "policy_version": base.get("policy_version"),
        },
        "rationale_short": (
            f"Meta-trust={trust_level} (self_confidence={self_conf:.2f}) "
            f"-> confidence={new_policy['confidence_threshold']:.2f}, "
            f"persistence={new_policy['persistence_threshold']:.2f}, "
            f"deployment={new_policy['deployment_threshold']:.2f}, "
            f"execution={new_policy['execution_threshold']:.2f}, "
            f"cash={new_policy['target_cash_pct']:.1f}%, "
            f"max_position={new_policy['max_position_pct']:.1f}%."
        ),
    }
    return new_policy, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only meta policy injection engine (Step 14). Reads the "
            "regime-driven runtime policy (Step 11) and the meta runtime "
            "modifiers (Step 13), applies the modifiers under spec-bound "
            "clamps, and emits runtime_policy_meta.json + a diff summary."
        ),
    )
    p.add_argument(
        "--runtime-policy",
        default=None,
        help=(
            "Path to the base runtime policy. If omitted, prefers "
            "runtime_policy_base.json (the regime-only snapshot Step 11 "
            "writes each cycle) and falls back to runtime_policy.json."
        ),
    )
    p.add_argument("--meta-intelligence", default=str(DEFAULT_META_INTELLIGENCE))
    p.add_argument("--meta-adjustments", default=str(DEFAULT_META_ADJ))
    p.add_argument("--adaptive-policy", default=str(DEFAULT_ADAPTIVE_POLICY))
    p.add_argument("--out-policy", default=str(DEFAULT_OUT_POLICY))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def _resolve_runtime_policy_path(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    if DEFAULT_RUNTIME_POLICY_BASE.is_file():
        return DEFAULT_RUNTIME_POLICY_BASE
    return DEFAULT_RUNTIME_POLICY


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[META_POLICY] starting (read-only self-trust -> policy bridge)", flush=True)

    runtime_policy_path = _resolve_runtime_policy_path(args.runtime_policy)
    runtime_policy = _safe_read_json(runtime_policy_path, label=runtime_policy_path.name)
    meta_intel = _safe_read_json(
        Path(args.meta_intelligence), label="meta_decision_intelligence.json"
    )
    meta_adj = _safe_read_json(Path(args.meta_adjustments), label="meta_runtime_adjustments.json")
    adaptive_policy = _safe_read_json(Path(args.adaptive_policy), label="adaptive_policy.json")

    new_policy, summary = build_meta_policy(
        runtime_policy=runtime_policy,
        meta_intelligence=meta_intel,
        meta_adjustments=meta_adj,
        adaptive_policy=adaptive_policy,
    )

    try:
        _atomic_write_json(new_policy, Path(args.out_policy))
    except Exception as e:
        _warn(f"failed to write {args.out_policy}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2

    print(
        "[META_POLICY] "
        f"trust={new_policy['meta_trust_level']} "
        f"confidence_threshold={new_policy['confidence_threshold']:.2f} "
        f"cash={new_policy['target_cash_pct']:.1f}% "
        f"max_position={new_policy['max_position_pct']:.1f}% "
        f"deployment={new_policy['deployment_threshold']:.2f}",
        flush=True,
    )
    if summary["clamp_events"]:
        print(
            "[META_POLICY_CLAMPED] " + ", ".join(summary["clamp_events"]),
            flush=True,
        )
    print(
        f"[META_POLICY_OUT] runtime_policy_meta={Path(args.out_policy).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}"
        + (" [DEGRADED]" if new_policy["degraded"] else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
