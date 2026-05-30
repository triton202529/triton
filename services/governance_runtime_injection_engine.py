"""
Governance Runtime Injection Engine -- Step 18.

Reads:
    data/results/runtime_policy_meta.json      (preferred base)
    data/results/runtime_policy.json           (fallback base)
    data/results/governance_trust_feedback.json
    data/results/governance_runtime_adjustments.json
    data/results/autonomous_strategy_diagnostics.json

Writes:
    data/results/runtime_policy_governed.json
    data/results/governance_policy_summary.json

Purpose
-------
Step 14 (meta_policy_injection_engine) layered Triton's self-trust
on top of the regime policy. Step 18 layers *proven governance
effectiveness* on top of that, producing the final policy:

    regime    + meta self-trust + proven governance
    (Step 11)   (Step 14)        (Step 18)

This engine answers: "How should proven governance performance
modify Triton behaviour?"

Layering precedence
-------------------
* Base policy comes from ``runtime_policy_meta.json`` when present
  (so meta self-trust is preserved), else from ``runtime_policy.json``
  (regime-only). The output is *always* written separately as
  ``runtime_policy_governed.json`` so downstream engines can decide
  whether to opt in. This file is intentionally NOT auto-promoted
  to ``runtime_policy.json`` -- adoption is an explicit downstream
  choice, exactly like Step 14's hand-off pattern.

Modifier wiring (spec section 2)
--------------------------------
   field                       arithmetic                  delta
   --------------------------  --------------------------  -------------------
   confidence_threshold        += confidence_delta         confidence_delta
   deployment_threshold        -= deployment_delta         deployment_delta
   target_cash_pct             += cash_delta * 100         cash_delta
   max_position_pct            += aggressiveness_delta*2   aggressiveness_delta
   skepticism_threshold        += skepticism_delta         skepticism_delta

``trust_delta`` is the master signal -- it is recorded in
provenance (``modifier_summary``) but does not modify any single
threshold directly. The five deltas above already encode it.

Clamping (spec section 3)
-------------------------
   confidence_threshold:  [0.45, 0.90]
   deployment_threshold:  [0.40, 0.90]
   target_cash_pct:       [5.0, 40.0]
   max_position_pct:      [2.0, 10.0]
   skepticism_threshold:  [0.0, 1.0]

Dormancy
--------
When Step 17 reports ``active=false`` (insufficient labelled
history) every delta is exactly zero by construction, so the
governance overlay is a no-op. The engine still emits the
governed policy file (== base policy + provenance) so downstream
consumers always see a consistent schema. The summary's
``governance_active`` flag and the log's ``[DORMANT]`` suffix
make the dormant state obvious to operators.

Safety
------
* READ ONLY. No broker calls, no engine state mutation.
* Additive overlay only -- the base policy is preserved field-for-field
  except for the five spec-required threshold fields.
* All five output fields hard-clamped to spec section 3 bounds.
* Atomic writes (.tmp + os.replace).
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

DEFAULT_BASE_META = RESULTS_DIR / "runtime_policy_meta.json"
DEFAULT_BASE_RUNTIME = RESULTS_DIR / "runtime_policy.json"
DEFAULT_FEEDBACK = RESULTS_DIR / "governance_trust_feedback.json"
DEFAULT_ADJUSTMENTS = RESULTS_DIR / "governance_runtime_adjustments.json"
DEFAULT_DIAGNOSTICS = RESULTS_DIR / "autonomous_strategy_diagnostics.json"

DEFAULT_OUT_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "governance_policy_summary.json"

POLICY_VERSION = 1

# -----------------------------------------------------------
# Modifier wiring -- mirrors Step 14's MODIFIER_WIRING shape so
# operators can diff the two side-by-side.
#
#   (policy_field,        delta_field,            sign, scale,
#    clamp_lo, clamp_hi)
#
#   sign = +1  -> policy_field += sign * scale * delta
#   sign = -1  -> policy_field -= scale * delta
#
# scale is the *multiplier* applied to the (already-bounded) delta
# before it touches the threshold. The spec only requires non-unit
# scales on the two percent fields:
#
#   target_cash_pct      += cash_delta * 100
#   max_position_pct     += aggressiveness_delta * 2
#
# Every other modifier moves the threshold by the raw delta (which
# is itself capped at +/-0.05 by Step 17).
# -----------------------------------------------------------
MODIFIER_WIRING: Tuple[Tuple[str, str, int, float, float, float], ...] = (
    ("confidence_threshold", "confidence_delta", +1, 1.0, 0.45, 0.90),
    ("deployment_threshold", "deployment_delta", -1, 1.0, 0.40, 0.90),
    ("target_cash_pct", "cash_delta", +1, 100.0, 5.00, 40.0),
    ("max_position_pct", "aggressiveness_delta", +1, 2.0, 2.00, 10.0),
    ("skepticism_threshold", "skepticism_delta", +1, 1.0, 0.00, 1.0),
)

# Default value for skepticism_threshold when the base policy
# omits it (it's a new field introduced by Step 18).
SKEPTICISM_THRESHOLD_DEFAULT = 0.50

# Aliases that mirror canonical fields. When a canonical field is
# overlaid we keep its alias in sync so legacy downstream readers
# don't see a stale value.
ALIAS_MIRRORS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("confidence_threshold", ("min_deploy_confidence", "min_execute_confidence")),
    ("max_position_pct", ("max_single_position_pct",)),
    ("target_cash_pct", ("target_cash_reserve_pct",)),
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[GOVERNANCE_RUNTIME_WARN] {msg}", flush=True)


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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm01(x: Any, default: float = 0.50) -> float:
    v = _to_float(x)
    if v is None:
        return default
    return _clamp(v, 0.0, 1.0)


# -----------------------------------------------------------
# Base policy selection
# -----------------------------------------------------------
def _load_base_policy(
    *,
    meta_path: Path,
    runtime_path: Path,
) -> Tuple[Dict[str, Any], str]:
    """
    Prefer meta runtime policy; fall back to regime-only runtime
    policy; return ({}, "none") if neither exists.
    """
    meta = _safe_read_json(meta_path, label="runtime_policy_meta.json")
    if meta:
        return meta, "runtime_policy_meta.json"
    base = _safe_read_json(runtime_path, label="runtime_policy.json")
    if base:
        return base, "runtime_policy.json"
    return {}, "none"


# -----------------------------------------------------------
# Overlay application
# -----------------------------------------------------------
def _apply_overlay(
    base: Dict[str, Any],
    deltas: Dict[str, float],
    *,
    governance_active: bool,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Apply the governance overlay onto a (copy of) the base policy.
    Returns (governed_policy, per_field_changes).

    When ``governance_active`` is False the deltas are zero by
    construction, so the policy is unchanged. The per-field changes
    map still gets populated with before==after for transparency.
    """
    governed = copy.deepcopy(base)
    changes: Dict[str, Dict[str, Any]] = {}

    for field, delta_key, sign, scale, lo, hi in MODIFIER_WIRING:
        before_raw = governed.get(field)
        if before_raw is None and field == "skepticism_threshold":
            before_raw = SKEPTICISM_THRESHOLD_DEFAULT
        before_f = _to_float(before_raw)
        if before_f is None:
            # Field absent and we don't know a sane default -- skip
            # rather than fabricate one.
            continue

        raw_delta = _to_float(deltas.get(delta_key)) or 0.0
        scaled = sign * scale * raw_delta
        proposed = before_f + scaled
        after = _clamp(proposed, lo, hi)
        clamped = proposed != after

        governed[field] = after
        changes[field] = {
            "before": before_f,
            "after": after,
            "delta": after - before_f,
            "raw_delta": raw_delta,
            "scaled_delta": scaled,
            "proposed_before_clamp": proposed,
            "clamp_lo": lo,
            "clamp_hi": hi,
            "clamped": bool(clamped),
            "delta_field": delta_key,
            "modifier_sign": sign,
            "modifier_scale": scale,
        }

    # Keep aliases consistent with their canonical fields.
    aliases = governed.get("aliases")
    if not isinstance(aliases, dict):
        aliases = {}
        governed["aliases"] = aliases
    for canonical, mirror_keys in ALIAS_MIRRORS:
        if canonical in governed:
            for mk in mirror_keys:
                aliases[mk] = governed[canonical]

    return governed, changes


# -----------------------------------------------------------
# Provenance builder
# -----------------------------------------------------------
def _build_modifier_summary(deltas: Dict[str, float]) -> Dict[str, float]:
    keys = (
        "trust_delta",
        "confidence_delta",
        "aggressiveness_delta",
        "skepticism_delta",
        "deployment_delta",
        "cash_delta",
    )
    return {k: round(_to_float(deltas.get(k)) or 0.0, 6) for k in keys}


def _annotate_provenance(
    governed: Dict[str, Any],
    *,
    base_source: str,
    feedback: Dict[str, Any],
    deltas: Dict[str, float],
    diagnostics: Dict[str, Any],
    changes: Dict[str, Dict[str, Any]],
) -> None:
    """In-place annotation onto the governed policy."""
    now_iso = _now_iso_utc()

    governed["engine"] = "governance_runtime_injection_engine"
    governed["policy_version"] = POLICY_VERSION
    governed["generated_at_utc"] = now_iso

    trust_level = str(feedback.get("governance_trust_level") or "STABLE").strip().upper()
    governance_active = bool(feedback.get("active", False))
    gov_health = _norm01(feedback.get("governance_health_score"))
    decision_quality = _norm01(
        (feedback.get("scores") or {}).get("decision_quality_score")
        or diagnostics.get("decision_quality_score")
    )

    governed["governance_trust_level"] = trust_level
    governed["governance_health_score"] = round(gov_health, 6)
    governed["decision_quality_score"] = round(decision_quality, 6)
    governed["governance_active"] = governance_active
    governed["governance_modifier_summary"] = _build_modifier_summary(deltas)
    governed["governance_changes"] = changes
    governed["governance_inputs_seen"] = {
        "base_policy_source": base_source,
        "governance_trust_feedback": bool(feedback),
        "governance_runtime_adjustments": bool(deltas),
        "autonomous_strategy_diagnostics": bool(diagnostics),
    }
    governed["governance_rationale_short"] = str(
        feedback.get("rationale_short")
        or (
            "Governance overlay dormant; base policy preserved."
            if not governance_active
            else f"Governance overlay applied at {trust_level}."
        )
    )


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_governed_policy(
    *,
    base: Dict[str, Any],
    base_source: str,
    feedback: Dict[str, Any],
    adjustments: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    deltas_blob = adjustments.get("deltas") if isinstance(adjustments, dict) else None
    if not isinstance(deltas_blob, dict):
        # Fallback to deltas embedded inside the feedback file.
        deltas_blob = feedback.get("deltas") if isinstance(feedback, dict) else {}
    if not isinstance(deltas_blob, dict):
        deltas_blob = {}

    governance_active = bool(feedback.get("active", False))
    governed, changes = _apply_overlay(base, deltas_blob, governance_active=governance_active)
    _annotate_provenance(
        governed,
        base_source=base_source,
        feedback=feedback,
        deltas=deltas_blob,
        diagnostics=diagnostics,
        changes=changes,
    )

    summary: Dict[str, Any] = {
        "engine": "governance_runtime_injection_engine",
        "policy_version": POLICY_VERSION,
        "generated_at_utc": governed["generated_at_utc"],
        "base_policy_source": base_source,
        "governance_trust_level": governed["governance_trust_level"],
        "governance_active": governance_active,
        "governance_health_score": governed["governance_health_score"],
        "decision_quality_score": governed["decision_quality_score"],
        "modifier_summary": governed["governance_modifier_summary"],
        "field_changes": {
            f: {
                "before": c["before"],
                "after": c["after"],
                "delta": c["delta"],
                "clamped": c["clamped"],
            }
            for f, c in changes.items()
        },
        "governed_thresholds": {f: governed.get(f) for f, *_ in MODIFIER_WIRING if f in governed},
        "rationale_short": governed["governance_rationale_short"],
    }
    return governed, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only governance runtime injection engine (Step 18). "
            "Overlays Step 17 governance trust feedback onto the meta/regime "
            "runtime policy, producing runtime_policy_governed.json. Dormant "
            "governance leaves the policy unchanged."
        ),
    )
    p.add_argument("--base-meta", default=str(DEFAULT_BASE_META))
    p.add_argument("--base-runtime", default=str(DEFAULT_BASE_RUNTIME))
    p.add_argument("--feedback", default=str(DEFAULT_FEEDBACK))
    p.add_argument("--adjustments", default=str(DEFAULT_ADJUSTMENTS))
    p.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    p.add_argument("--out-policy", default=str(DEFAULT_OUT_POLICY))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[GOVERNANCE_RUNTIME] starting (governance feedback -> runtime policy overlay)", flush=True
    )

    base, base_source = _load_base_policy(
        meta_path=Path(args.base_meta),
        runtime_path=Path(args.base_runtime),
    )
    feedback = _safe_read_json(Path(args.feedback), label="governance_trust_feedback.json")
    adjustments = _safe_read_json(
        Path(args.adjustments), label="governance_runtime_adjustments.json"
    )
    diagnostics = _safe_read_json(
        Path(args.diagnostics), label="autonomous_strategy_diagnostics.json"
    )

    if not base:
        _warn(
            "no base runtime policy available (meta + regime both missing); "
            "writing degraded governed policy with only governance provenance"
        )

    governed, summary = build_governed_policy(
        base=base,
        base_source=base_source,
        feedback=feedback,
        adjustments=adjustments,
        diagnostics=diagnostics,
    )

    try:
        _atomic_write_json(governed, Path(args.out_policy))
    except Exception as e:
        _warn(f"failed to write {args.out_policy}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2

    trust = governed.get("governance_trust_level", "STABLE")
    cash = governed.get("target_cash_pct")
    deployment = governed.get("deployment_threshold")
    confidence = governed.get("confidence_threshold")
    max_pos = governed.get("max_position_pct")
    dormant_tag = "" if governed.get("governance_active") else " [DORMANT]"
    print(
        "[GOVERNANCE_RUNTIME] "
        f"trust={trust} "
        f"cash={_fmt(cash)} "
        f"deployment={_fmt(deployment)} "
        f"confidence={_fmt(confidence)} "
        f"aggressiveness={_fmt(max_pos)}" + dormant_tag,
        flush=True,
    )
    print(
        f"[GOVERNANCE_RUNTIME_OUT] base={base_source} "
        f"policy={Path(args.out_policy).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


def _fmt(x: Any) -> str:
    v = _to_float(x)
    return f"{v:.4f}" if v is not None else "NA"


if __name__ == "__main__":
    raise SystemExit(main())
