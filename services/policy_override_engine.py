"""
Policy Override Integration Engine — Step 11 (runtime policy compiler).

Reads:
    data/results/adaptive_policy.json
    data/results/adaptive_regime.json

Writes:
    data/results/runtime_policy.json

Purpose
-------
Step 10 (adaptive_regime_engine) emits a *structured* policy keyed by
regime. This engine flattens that into a single
``runtime_policy.json`` that downstream operational engines
(capital_deployment, portfolio_construction, portfolio_rebalance,
portfolio_execution_intent) can read at the top of ``main()`` and use
to override their built-in constants for the cycle.

The downstream engines do *not* re-implement regime logic. They simply
ask: "is there a runtime_policy.json? if yes, override these fields;
if no, use my defaults". This keeps each engine isolated, testable,
and reversible — deleting ``runtime_policy.json`` restores every
engine to its baked-in baseline.

The flat schema is intentionally a superset of the construction /
deployment / rebalance / intent engines' constant names so a future
integration is just dict-lookup.

Safety
------
* Read-only. No broker calls, no execution-state mutation.
* If either input is missing, the engine still writes a neutral
  fallback ``runtime_policy.json`` (so downstream engines never see a
  half-broken state) and records the degradation in metadata.
* Atomic writes (``.tmp`` + ``os.replace``).
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

DEFAULT_ADAPTIVE_POLICY_JSON = RESULTS_DIR / "adaptive_policy.json"
DEFAULT_ADAPTIVE_REGIME_JSON = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "runtime_policy.json"

# Step 14 (optional, additive): if meta_policy_injection_engine has
# emitted a meta-blended policy for the *current* regime, prefer it
# over the regime-only policy so downstream engines pick up the
# self-trust adjustments. Mismatched-regime or missing meta files are
# silently ignored -- the regime-only policy is the safe fallback.
DEFAULT_META_POLICY_JSON = RESULTS_DIR / "runtime_policy_meta.json"

# Step 14 (optional, additive): a regime-only snapshot preserved each
# cycle so the meta policy engine has a clean baseline to overlay --
# preventing the meta engine from compounding its own prior overlay
# on top of itself across cycles.
DEFAULT_BASE_POLICY_JSON = RESULTS_DIR / "runtime_policy_base.json"

POLICY_SCHEMA_VERSION = 1

# Fallback policy used when adaptive inputs are missing or empty.
# Mirrors the NEUTRAL row of the regime → policy table.
FALLBACK_POLICY: Dict[str, Any] = {
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

# Fields that *every* runtime_policy.json must surface (used for
# validation and for the downstream engines' overrides).
REQUIRED_RUNTIME_FIELDS: Tuple[str, ...] = (
    "max_position_pct",
    "min_position_pct",
    "max_sector_pct",
    "max_cluster_pct",
    "min_cash_reserve_pct",
    "max_cash_reserve_pct",
    "target_cash_pct",
    "max_new_positions_per_cycle",
    "deployment_threshold",
    "confidence_threshold",
    "persistence_threshold",
    "rebalance_frequency",
    "rotation_pressure",
    "diversification_aggressiveness",
    "risk_tolerance",
    "block_strict_mode",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[POLICY_OVERRIDE_WARN] {msg}", flush=True)


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


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


# -----------------------------------------------------------
# Builder
# -----------------------------------------------------------
def build_runtime_policy(
    *,
    adaptive_policy: Dict[str, Any],
    adaptive_regime: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flatten the adaptive policy into a single executable dict.

    The returned object always contains every key in
    ``REQUIRED_RUNTIME_FIELDS`` plus engine-named aliases plus
    provenance metadata.
    """
    policy_block: Dict[str, Any] = dict(adaptive_policy.get("policy") or {})
    regime: str = (
        str(adaptive_policy.get("regime") or adaptive_regime.get("regime") or "NEUTRAL")
        .strip()
        .upper()
        or "NEUTRAL"
    )
    trigger_reasons: List[str] = list(
        adaptive_policy.get("trigger_reasons") or adaptive_regime.get("trigger_reasons") or []
    )
    rationale_long = str(
        adaptive_regime.get("rationale") or adaptive_policy.get("rationale_short") or ""
    )
    rationale_short = str(adaptive_policy.get("rationale_short") or "")

    # Defensive merge — start from FALLBACK, overlay live policy fields.
    merged: Dict[str, Any] = dict(FALLBACK_POLICY)
    for k, v in policy_block.items():
        if v is None:
            continue
        merged[k] = v

    # Type-coerce numerics defensively (defaults survive bad input).
    for k in (
        "max_position_pct",
        "min_position_pct",
        "max_sector_pct",
        "max_cluster_pct",
        "min_cash_reserve_pct",
        "max_cash_reserve_pct",
        "target_cash_pct",
        "deployment_threshold",
        "confidence_threshold",
        "persistence_threshold",
        "rotation_pressure",
        "diversification_aggressiveness",
        "risk_tolerance",
    ):
        v = _to_float(merged.get(k))
        if v is not None:
            merged[k] = float(v)
    new_pos = _to_float(merged.get("max_new_positions_per_cycle"))
    if new_pos is not None:
        merged["max_new_positions_per_cycle"] = int(new_pos)
    merged["block_strict_mode"] = _to_bool(merged.get("block_strict_mode"))
    if "rebalance_frequency" in merged:
        merged["rebalance_frequency"] = str(merged["rebalance_frequency"]).strip().upper()

    # Sanity: cash discipline must be ordered.
    cmin = _to_float(merged.get("min_cash_reserve_pct"))
    ctgt = _to_float(merged.get("target_cash_pct"))
    cmax = _to_float(merged.get("max_cash_reserve_pct"))
    if cmin is not None and ctgt is not None and ctgt < cmin:
        merged["target_cash_pct"] = cmin
        ctgt = cmin
    if cmax is not None and ctgt is not None and cmax < ctgt:
        merged["max_cash_reserve_pct"] = ctgt
        cmax = ctgt

    # Pull source scores for full provenance.
    ev = (adaptive_regime.get("evidence") or {}).get("committee") or {}
    source_scores: Dict[str, Any] = {
        "portfolio_health_score": _to_float(ev.get("portfolio_health_score")),
        "deployment_readiness_score": _to_float(ev.get("deployment_readiness_score")),
        "conviction_score": _to_float(ev.get("conviction_score")),
        "diversification_score": _to_float(ev.get("diversification_score")),
        "governance_score": _to_float(ev.get("governance_score")),
    }
    risk_ev = (adaptive_regime.get("evidence") or {}).get("risk_overlay") or {}
    source_evidence_summary: Dict[str, Any] = {
        "n_force_exit": int(risk_ev.get("n_force_exit") or 0),
        "n_block_new_buy": int(risk_ev.get("n_block_new_buy") or 0),
        "n_flagged": int(risk_ev.get("n_flagged") or 0),
        "n_symbols": int(risk_ev.get("n_symbols") or 0),
        "flag_ratio": _to_float(risk_ev.get("flag_ratio")) or 0.0,
    }

    degraded = not bool(adaptive_policy and adaptive_regime)

    runtime_policy: Dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_version": POLICY_SCHEMA_VERSION,
        "generated_at_utc": _now_iso_utc(),
        "engine": "policy_override_engine",
        "regime": regime,
        "rationale_short": rationale_short,
        "rationale_long": rationale_long,
        "trigger_reasons": trigger_reasons,
        "source_scores": source_scores,
        "source_evidence_summary": source_evidence_summary,
        "degraded": degraded,
        "inputs_seen": {
            "adaptive_policy": bool(adaptive_policy),
            "adaptive_regime": bool(adaptive_regime),
        },
    }
    # Flatten the policy into the top-level so downstream engines can
    # do a one-shot dict lookup.
    for k in REQUIRED_RUNTIME_FIELDS:
        runtime_policy[k] = merged[k]

    # Aliases mirroring the engine constants verbatim so downstream
    # patches can use whichever name reads more naturally.
    runtime_policy["aliases"] = {
        "max_single_position_pct": merged["max_position_pct"],
        "min_deploy_confidence": merged["confidence_threshold"],
        "deploy_persistence_floor": merged["persistence_threshold"],
        "min_execute_confidence": merged["confidence_threshold"],
        "min_execute_persistence": merged["persistence_threshold"],
        "min_execute_intent_score": merged["deployment_threshold"],
        "max_new_positions_per_cycle": merged["max_new_positions_per_cycle"],
        "target_cash_reserve_pct": merged["target_cash_pct"],
        "min_cash_reserve_pct": merged["min_cash_reserve_pct"],
        "max_cash_reserve_pct": merged["max_cash_reserve_pct"],
        "max_sector_pct": merged["max_sector_pct"],
        "max_cluster_pct": merged["max_cluster_pct"],
    }
    return runtime_policy


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only policy override integration engine (Step 11). Flattens "
            "the adaptive regime policy into a single runtime_policy.json that "
            "downstream operational engines can consume as a JSON override."
        ),
    )
    p.add_argument("--adaptive-policy", default=str(DEFAULT_ADAPTIVE_POLICY_JSON))
    p.add_argument("--adaptive-regime", default=str(DEFAULT_ADAPTIVE_REGIME_JSON))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def _maybe_apply_meta_overlay(
    runtime_policy: Dict[str, Any],
    output_path: Path,
    meta_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Step 14 optional integration. If a meta-blended policy file exists
    *and* its regime matches the just-computed regime, swap it into
    ``output_path`` so downstream engines see the self-trust-adjusted
    policy. Returns (applied, reason).

    Mismatched regimes and any read/parse failure fall through to the
    regime-only policy on disk (which was just written by the caller).
    This keeps the integration strictly additive -- removing the meta
    file or breaking it never breaks Step 11.
    """
    path = meta_path if meta_path is not None else DEFAULT_META_POLICY_JSON
    try:
        if not path.is_file():
            return False, "no_meta_policy_file"
    except OSError as e:
        _warn(f"meta overlay stat failed ({path}): {type(e).__name__}: {e}")
        return False, "meta_stat_failed"
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    except Exception as e:
        _warn(f"meta overlay unreadable ({path}): {type(e).__name__}: {e}")
        return False, "meta_unreadable"
    if not isinstance(meta, dict) or not meta:
        return False, "meta_empty"
    meta_regime = str(meta.get("regime") or "").strip().upper()
    cur_regime = str(runtime_policy.get("regime") or "").strip().upper()
    if not meta_regime or meta_regime != cur_regime:
        _warn(
            f"meta overlay skipped: regime mismatch "
            f"(current={cur_regime!r}, meta={meta_regime!r})"
        )
        return False, "regime_mismatch"
    try:
        _atomic_write_json(meta, output_path)
    except Exception as e:
        _warn(f"meta overlay write failed ({output_path}): " f"{type(e).__name__}: {e}")
        return False, "meta_write_failed"
    return True, "overlay_applied"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[POLICY_OVERRIDE] starting (read-only runtime policy compiler)", flush=True)

    adaptive_policy = _safe_read_json(Path(args.adaptive_policy), label="adaptive_policy.json")
    adaptive_regime = _safe_read_json(Path(args.adaptive_regime), label="adaptive_regime.json")

    runtime_policy = build_runtime_policy(
        adaptive_policy=adaptive_policy,
        adaptive_regime=adaptive_regime,
    )

    out_json = Path(args.out_json)
    try:
        _atomic_write_json(runtime_policy, out_json)
    except Exception as e:
        _warn(f"failed to write {out_json}: {type(e).__name__}: {e}")
        return 2

    # Step 14 hand-off: preserve the regime-only snapshot for the meta
    # engine so it can compute fresh adjustments from a clean baseline
    # rather than compounding its own prior overlay.
    try:
        _atomic_write_json(runtime_policy, DEFAULT_BASE_POLICY_JSON)
    except Exception as e:
        _warn(
            f"failed to write base snapshot {DEFAULT_BASE_POLICY_JSON}: "
            f"{type(e).__name__}: {e} (overlay still proceeds)"
        )

    print(
        "[POLICY_OVERRIDE] "
        f"regime={runtime_policy['regime']} "
        f"max_position={runtime_policy['max_position_pct']:.1f}% "
        f"cash={runtime_policy['target_cash_pct']:.1f}% "
        f"confidence>={runtime_policy['confidence_threshold']:.2f} "
        f"persistence>={runtime_policy['persistence_threshold']:.2f} "
        f"new_positions={runtime_policy['max_new_positions_per_cycle']}"
        + (" [DEGRADED]" if runtime_policy["degraded"] else ""),
        flush=True,
    )

    applied, reason = _maybe_apply_meta_overlay(runtime_policy, out_json)
    if applied:
        print(
            f"[POLICY_OVERRIDE_META] overlay applied "
            f"({DEFAULT_META_POLICY_JSON.as_posix()}) reason={reason}",
            flush=True,
        )
    else:
        print(f"[POLICY_OVERRIDE_META] overlay skipped reason={reason}", flush=True)

    print(
        f"[POLICY_OVERRIDE_OUT] runtime_policy={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
