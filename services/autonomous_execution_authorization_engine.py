"""
Autonomous Execution Authorization Engine -- Step 23.

Reads:
    data/results/autonomous_action_permissions.json     (Step 22)
    data/results/autonomous_action_summary.json         (Step 22)
    data/results/autonomous_readiness_summary.json      (Step 21)
    data/results/autonomous_committee_summary.json      (Step 15)
    data/results/runtime_policy_governed.json           (Step 18)
    data/results/autonomous_system_health_summary.json  (Step 20)
    data/results/meta_decision_intelligence.json        (Step 13)

Writes:
    data/results/autonomous_execution_authorization.json
    data/results/autonomous_execution_summary.json
    data/results/autonomous_execution_authorization.md

Purpose
-------
Step 22 produced eight atomic permission booleans. Step 23 is the
*final* machine-facing authorization layer below any future
execution engine. It collapses those booleans + regime + trust
context into one of six high-level execution authorization states:

    "What execution behaviour is Triton authorized to perform right now?"

This is NOT execution -- this is *authorization*. Execution remains
a separate (still-to-be-built) engine that will read exactly one
file (``autonomous_execution_authorization.json``) and refuse to
proceed unless ``execution_authorized=true`` and the relevant
per-action booleans permit the specific action it wants to take.

Six execution authorization states (spec section 1)
---------------------------------------------------
    ANALYSIS_ONLY       read-only; no execution, no rebalance, no buys
    EXIT_ONLY           sells/trims allowed; no buys; no aggressive
    DEFENSIVE_EXECUTION selective + defensive rotation; aggressive denied
    SELECTIVE_DEPLOYMENT limited buys + rebalance; runtime policy restrictive
    FULL_AUTONOMY       all permissions true; READY + HEALTHY + trust ok
    BLOCKED             nothing allowed; operator review required

Distinguishing READY_LIMITED -> DEFENSIVE_EXECUTION vs
SELECTIVE_DEPLOYMENT depends on regime and committee context
(defensive regime or defensive committee decision => DEFENSIVE).

Authorization confidence (spec section 3)
-----------------------------------------
A 0-1 weighted blend tuned for execution risk -- emphasises
operational health and overall readiness over the strategic
quality signals. The six contributors mirror those listed in the
spec; weights sum to 1.0 by construction.

Safety
------
* READ ONLY. Absolutely no broker calls. No execution mutation
  anywhere in this module (the words "place_order" and "submit"
  do not appear).
* The output is a *contract*; downstream automation is responsible
  for honouring it. The per-action booleans here are deliberate
  copies of Step 22's matrix (post-override) so a single check in
  the future execution engine is sufficient.
* Atomic writes (.tmp + os.replace) for all three outputs.
* Missing inputs warn-and-continue. With zero inputs the engine
  defaults to ANALYSIS_ONLY (the safest authorisation state that
  still lets operators triage), or BLOCKED if even Step 22's
  permissions blob is unavailable.
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

DEFAULT_ACTION_PERMS = RESULTS_DIR / "autonomous_action_permissions.json"
DEFAULT_ACTION_SUMMARY = RESULTS_DIR / "autonomous_action_summary.json"
DEFAULT_READINESS_SUMMARY = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_execution_authorization.json"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_execution_summary.json"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_execution_authorization.md"


# -----------------------------------------------------------
# State constants
# -----------------------------------------------------------
STATE_BLOCKED = "BLOCKED"
STATE_ANALYSIS_ONLY = "ANALYSIS_ONLY"
STATE_EXIT_ONLY = "EXIT_ONLY"
STATE_DEFENSIVE_EXECUTION = "DEFENSIVE_EXECUTION"
STATE_SELECTIVE_DEPLOYMENT = "SELECTIVE_DEPLOYMENT"
STATE_FULL_AUTONOMY = "FULL_AUTONOMY"

ALL_AUTH_STATES: Tuple[str, ...] = (
    STATE_BLOCKED,
    STATE_ANALYSIS_ONLY,
    STATE_EXIT_ONLY,
    STATE_DEFENSIVE_EXECUTION,
    STATE_SELECTIVE_DEPLOYMENT,
    STATE_FULL_AUTONOMY,
)

# States that authorise *any* execution at all
EXECUTION_AUTHORIZED_STATES = {
    STATE_EXIT_ONLY,
    STATE_DEFENSIVE_EXECUTION,
    STATE_SELECTIVE_DEPLOYMENT,
    STATE_FULL_AUTONOMY,
}

# Regimes / committee decisions that trigger DEFENSIVE_EXECUTION
DEFENSIVE_REGIMES = {"DEFENSIVE", "RISK_OFF", "HIGH_VOLATILITY", "CAPITAL_PRESERVATION"}
DEFENSIVE_COMMITTEE_DECISIONS = {
    "CAPITAL_PRESERVATION",
    "DEFENSIVE_ROTATION",
    "DELEVER",
}

# Trust-level -> numeric (matches Step 19/21 mapping)
TRUST_LEVEL_NUMERIC: Dict[str, float] = {
    "VERY_LOW": 0.10,
    "LOW": 0.30,
    "MODERATE": 0.55,
    "HIGH": 0.78,
    "VERY_HIGH": 0.92,
}

# Step 20 overall_status -> numeric
HEALTH_STATUS_NUMERIC: Dict[str, float] = {
    "HEALTHY": 1.00,
    "DEGRADED": 0.70,
    "STALE": 0.40,
    "CRITICAL": 0.10,
    "OFFLINE": 0.00,
}

# Confidence-contributor weights -- spec section 3 (sums to 1.0)
SCORE_WEIGHTS: Dict[str, float] = {
    "system_health": 0.25,
    "readiness_score": 0.20,
    "governance_health": 0.15,
    "trust_level": 0.15,
    "committee_confidence": 0.15,
    "runtime_freshness": 0.10,
}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[EXECUTION_AUTH_WARN] {msg}", flush=True)


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


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
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


def _to_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    if not s:
        return default
    return s in {"true", "1", "yes", "y", "t"}


# -----------------------------------------------------------
# State classification
# -----------------------------------------------------------
def _classify_authorization(
    *,
    permissions: Dict[str, bool],
    readiness_state: str,
    regime: str,
    committee_decision: str,
) -> Tuple[str, List[str]]:
    """
    Strict precedence cascade. Permissions are *authoritative* (they
    already encode Step 22's matrix + safety overrides) -- we only
    use readiness_state, regime, and committee_decision as
    *tie-breakers* between READY_LIMITED-shaped permission sets.
    """
    reasons: List[str] = []
    p = permissions

    # --- 1. BLOCKED: hard block from upstream readiness ---
    if readiness_state == "BLOCKED":
        reasons.append("upstream readiness=BLOCKED")
        return STATE_BLOCKED, reasons

    # --- 2. ANALYSIS_ONLY: no execution actions enabled at all ---
    exec_actions_enabled = sum(
        int(p.get(k, False))
        for k in (
            "allow_sell_exits",
            "allow_rebalance",
            "allow_rotation",
            "allow_new_buys",
            "allow_aggressive_deployment",
            "allow_defensive_rotation",
        )
    )
    if exec_actions_enabled == 0:
        if p.get("allow_read_only_analysis", False):
            reasons.append("zero execution permissions; only analysis allowed")
            return STATE_ANALYSIS_ONLY, reasons
        # No execution AND no analysis -- treat as BLOCKED (should
        # never happen because the matrix always grants analysis).
        reasons.append("zero execution permissions and no analysis flag")
        return STATE_BLOCKED, reasons

    # --- 3. FULL_AUTONOMY: every actionable permission true ---
    full_clear = (
        p.get("allow_new_buys", False)
        and p.get("allow_sell_exits", False)
        and p.get("allow_rebalance", False)
        and p.get("allow_rotation", False)
        and p.get("allow_aggressive_deployment", False)
        and p.get("allow_defensive_rotation", False)
        and not p.get("require_operator_review", True)
    )
    if full_clear:
        reasons.append("all six execution permissions granted + no review required")
        return STATE_FULL_AUTONOMY, reasons

    # --- 4. EXIT_ONLY: sells/rebalance allowed but no buys ---
    if (
        not p.get("allow_new_buys", False)
        and not p.get("allow_aggressive_deployment", False)
        and (p.get("allow_sell_exits", False) or p.get("allow_rebalance", False))
    ):
        reasons.append("exits/rebalance permitted but no buy permissions")
        return STATE_EXIT_ONLY, reasons

    # --- 5. DEFENSIVE_EXECUTION vs SELECTIVE_DEPLOYMENT ---
    # At this point: some buy permission is True but aggressive is
    # False (or some other partial-grant). Distinguish by regime /
    # committee context.
    is_defensive_context = (
        regime in DEFENSIVE_REGIMES or committee_decision in DEFENSIVE_COMMITTEE_DECISIONS
    )
    if is_defensive_context:
        reasons.append(
            f"selective grant under defensive context "
            f"(regime={regime or 'UNKNOWN'}, committee={committee_decision or 'UNKNOWN'})"
        )
        return STATE_DEFENSIVE_EXECUTION, reasons

    reasons.append(
        f"selective grant under neutral/opportunistic context "
        f"(regime={regime or 'UNKNOWN'}, committee={committee_decision or 'UNKNOWN'})"
    )
    return STATE_SELECTIVE_DEPLOYMENT, reasons


# -----------------------------------------------------------
# Authorization booleans
# -----------------------------------------------------------
def _build_authorization_booleans(
    *,
    state: str,
    permissions: Dict[str, bool],
) -> Dict[str, bool]:
    """
    Spec section 2 fields. Most are direct copies of Step 22's
    permissions (post-override), with two new derived fields:
    ``execution_authorized`` (the top-level "is any execution
    permitted?" flag) and ``analysis_only``.
    """
    p = permissions
    return {
        "execution_authorized": state in EXECUTION_AUTHORIZED_STATES,
        "analysis_only": state == STATE_ANALYSIS_ONLY,
        "allow_new_buys": bool(p.get("allow_new_buys", False)),
        "allow_sell_exits": bool(p.get("allow_sell_exits", False)),
        "allow_rebalance": bool(p.get("allow_rebalance", False)),
        "allow_rotation": bool(p.get("allow_rotation", False)),
        "allow_aggressive_deployment": bool(p.get("allow_aggressive_deployment", False)),
        "allow_defensive_rotation": bool(p.get("allow_defensive_rotation", False)),
        "require_operator_review": bool(p.get("require_operator_review", True)),
    }


# -----------------------------------------------------------
# Confidence
# -----------------------------------------------------------
def _build_confidence_contributors(
    *,
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    committee_summary: Dict[str, Any],
    meta_intel: Dict[str, Any],
    governance_feedback: Dict[str, Any],
) -> Dict[str, float]:
    health_status = str((health_summary or {}).get("overall_status") or "OFFLINE").strip().upper()
    sys_health = HEALTH_STATUS_NUMERIC.get(health_status, 0.0)

    readiness_score = _norm01((readiness_summary or {}).get("readiness_score"), default=0.0)

    # Committee confidence preferentially read from autonomous committee
    # summary (Step 15); readiness summary also stores the same value
    # under a slightly different key for convenience.
    committee_conf = _to_float((committee_summary or {}).get("recommendation_confidence"))
    if committee_conf is None:
        committee_conf = _to_float(
            ((readiness_summary or {}).get("raw_inputs") or {}).get("recommendation_confidence")
        )
    committee_conf = _norm01(committee_conf if committee_conf is not None else 0.50)

    trust_level = str((meta_intel or {}).get("trust_level") or "MODERATE").strip().upper()
    trust_score = TRUST_LEVEL_NUMERIC.get(trust_level, 0.55)

    if bool((governance_feedback or {}).get("active", False)):
        gov_health = _norm01((governance_feedback or {}).get("governance_health_score"))
    else:
        # Fall back to readiness contributor when feedback dormant.
        gov_health = _norm01(
            ((readiness_summary or {}).get("headline_contributors") or {}).get("governance_health"),
            default=0.50,
        )

    # Runtime freshness mirrors Step 21's derivation.
    stale = set(map(str, (health_summary or {}).get("stale_artifacts") or []))
    missing = set(map(str, (health_summary or {}).get("missing_artifacts") or []))
    if "runtime_policy_governed" in missing:
        runtime_fresh = 0.0
    elif "runtime_policy_governed" in stale:
        runtime_fresh = 0.30
    else:
        runtime_fresh = 1.0

    return {
        "system_health": round(sys_health, 6),
        "readiness_score": round(readiness_score, 6),
        "governance_health": round(gov_health, 6),
        "trust_level": round(trust_score, 6),
        "committee_confidence": round(committee_conf, 6),
        "runtime_freshness": round(runtime_fresh, 6),
    }


def _authorization_confidence(contributors: Dict[str, float]) -> float:
    total_w = 0.0
    weighted = 0.0
    for k, w in SCORE_WEIGHTS.items():
        weighted += w * _clamp(contributors.get(k, 0.0), 0.0, 1.0)
        total_w += w
    if total_w <= 0.0:
        return 0.0
    return _clamp(weighted / total_w, 0.0, 1.0)


# -----------------------------------------------------------
# Rationale + recommendations
# -----------------------------------------------------------
def _build_rationale(
    *,
    state: str,
    reasons: List[str],
    booleans: Dict[str, bool],
    confidence: float,
    contributors: Dict[str, float],
    readiness_state: str,
    regime: str,
    committee_decision: str,
) -> Tuple[str, str]:
    allowed_keys = [
        k
        for k in (
            "allow_new_buys",
            "allow_sell_exits",
            "allow_rebalance",
            "allow_rotation",
            "allow_aggressive_deployment",
            "allow_defensive_rotation",
        )
        if booleans.get(k)
    ]
    denied_keys = [
        k
        for k in (
            "allow_new_buys",
            "allow_sell_exits",
            "allow_rebalance",
            "allow_rotation",
            "allow_aggressive_deployment",
            "allow_defensive_rotation",
        )
        if not booleans.get(k)
    ]

    if state == STATE_BLOCKED:
        flavour = "all execution denied; operator review required"
    elif state == STATE_ANALYSIS_ONLY:
        flavour = "analysis only; no execution authorised"
    elif state == STATE_EXIT_ONLY:
        flavour = "exits and rebalance only; no new buys, no aggressive deployment"
    elif state == STATE_DEFENSIVE_EXECUTION:
        flavour = "selective deployment under defensive constraints; aggressive denied"
    elif state == STATE_SELECTIVE_DEPLOYMENT:
        flavour = "limited buys + rebalance; runtime policy restrictive"
    elif state == STATE_FULL_AUTONOMY:
        flavour = "all execution permissions granted; full autonomous deployment"
    else:
        flavour = "unknown state"

    rationale_short = (
        f"Triton execution authorization: {state} (confidence {confidence:.2f}) -- {flavour}."
    )

    bullet = "; ".join(reasons) if reasons else "no specific trigger recorded"
    weak = sorted(
        (k for k, v in contributors.items() if v < 0.40),
        key=lambda k: contributors[k],
    )
    strong = sorted(
        (k for k, v in contributors.items() if v >= 0.65),
        key=lambda k: -contributors[k],
    )
    rationale_long = (
        f"State {state} assigned because: {bullet}. "
        f"Upstream readiness state {readiness_state or 'UNKNOWN'}, "
        f"regime {regime or 'UNKNOWN'}, "
        f"committee {committee_decision or 'UNKNOWN'}. "
        f"Authorization confidence is {confidence:.3f} "
        f"(system_health={contributors['system_health']:.2f}, "
        f"readiness={contributors['readiness_score']:.2f}, "
        f"committee_confidence={contributors['committee_confidence']:.2f}, "
        f"trust={contributors['trust_level']:.2f}, "
        f"governance={contributors['governance_health']:.2f}, "
        f"runtime_freshness={contributors['runtime_freshness']:.2f}). "
        f"Allowed: {allowed_keys or 'none'}. "
        f"Denied: {denied_keys or 'none'}. "
        f"Weakest contributors: {weak or 'none'}. "
        f"Strongest contributors: {strong or 'none'}."
    )
    return rationale_short, rationale_long


def _build_recommendations(
    *,
    state: str,
    contributors: Dict[str, float],
    booleans: Dict[str, bool],
) -> List[str]:
    recs: List[str] = []
    if state == STATE_BLOCKED:
        recs.append("Require operator approval before deployment -- system is BLOCKED.")
        recs.append("Refresh full pipeline and re-run system health monitor (Step 20).")
    elif state == STATE_ANALYSIS_ONLY:
        recs.append("Continue analysis only -- no execution permitted today.")
        recs.append("Refresh stale pipeline artifacts before any deployment can resume.")
    elif state == STATE_EXIT_ONLY:
        recs.append("Permit exit and trim execution only -- no new buys today.")
        recs.append("Reassess deployment readiness on the next cycle.")
    elif state == STATE_DEFENSIVE_EXECUTION:
        recs.append("Permit defensive rotation and selective deployment only.")
        recs.append("Restrict deployment to high-conviction defensive assets.")
        recs.append("Honour elevated cash discipline from the governed runtime policy.")
    elif state == STATE_SELECTIVE_DEPLOYMENT:
        recs.append("Permit selective deployment under confidence filters.")
        recs.append("Block aggressive deployment until governance health improves.")
    elif state == STATE_FULL_AUTONOMY:
        recs.append("Proceed with full autonomous deployment within policy bounds.")
        recs.append(
            "Continue monitoring for regime or governance changes that would tighten authorization."
        )

    # Targeted per-contributor hints
    if contributors["runtime_freshness"] < 0.50:
        recs.append("Refresh runtime policy (Steps 11 + 14 + 18) before any new deployment.")
    if contributors["system_health"] < 0.40:
        recs.append("System health degraded -- run the system health monitor (Step 20) and triage.")
    if contributors["governance_health"] < 0.30:
        recs.append(
            "Governance health below WEAK threshold -- review governance diagnostics (Step 16)."
        )
    if contributors["trust_level"] < 0.30:
        recs.append("Meta trust LOW or VERY_LOW -- restrict deployment to defensive assets only.")

    # De-duplicate while preserving order
    seen: List[str] = []
    for r in recs:
        if r not in seen:
            seen.append(r)
    return seen


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    booleans: Dict[str, bool],
    confidence: float,
    contributors: Dict[str, float],
    rationale_short: str,
    rationale_long: str,
    recommendations: List[str],
    readiness_state: str,
    regime: str,
    committee_decision: str,
) -> str:
    def yn(b: bool) -> str:
        return "yes" if b else "no"

    lines: List[str] = []
    lines.append("# Triton Execution Authorization")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Authorization State")
    lines.append("")
    lines.append(f"**{state}**")
    lines.append("")
    lines.append(f"- upstream readiness: {readiness_state or 'UNKNOWN'}")
    lines.append(f"- regime: {regime or 'UNKNOWN'}")
    lines.append(f"- committee decision: {committee_decision or 'UNKNOWN'}")
    lines.append(f"- execution authorized: **{yn(booleans['execution_authorized'])}**")
    lines.append(f"- analysis only: {yn(booleans['analysis_only'])}")
    lines.append("")

    lines.append("## Execution Permissions")
    lines.append("")
    lines.append("| permission | granted |")
    lines.append("|---|---|")
    for k in (
        "execution_authorized",
        "allow_new_buys",
        "allow_sell_exits",
        "allow_rebalance",
        "allow_rotation",
        "allow_aggressive_deployment",
        "allow_defensive_rotation",
        "require_operator_review",
        "analysis_only",
    ):
        lines.append(f"| {k} | {yn(booleans[k])} |")
    lines.append("")

    lines.append("## Authorization Confidence")
    lines.append("")
    lines.append(f"**{confidence:.3f}** (0.000 = no authorization, 1.000 = full clear)")
    lines.append("")
    lines.append("| contributor | score | weight |")
    lines.append("|---|---|---|")
    for k, w in SCORE_WEIGHTS.items():
        lines.append(f"| {k} | {contributors[k]:.3f} | {w:.2f} |")
    lines.append("")

    lines.append("## Why")
    lines.append("")
    lines.append(rationale_long)
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    lines.append(rationale_short)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_execution_authorization(
    *,
    action_permissions: Dict[str, Any],
    action_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    committee_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    health_summary: Dict[str, Any],
    meta_intel: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    permissions = dict((action_permissions or {}).get("permissions") or {})

    readiness_state = (
        str(
            (readiness_summary or {}).get("readiness_state")
            or (action_permissions or {}).get("readiness_state")
            or "UNKNOWN"
        )
        .strip()
        .upper()
    )
    regime = (
        str(
            (runtime_policy or {}).get("regime")
            or ((readiness_summary or {}).get("raw_inputs") or {}).get("regime")
            or ""
        )
        .strip()
        .upper()
    )
    committee_decision = (
        str(
            (committee_summary or {}).get("decision")
            or (committee_summary or {}).get("committee_decision")
            or ""
        )
        .strip()
        .upper()
    )

    # Pull governance feedback indirectly from the action_permissions
    # upstream_context (Step 22 already cached it) -- keeps Step 23's
    # input set small.
    governance_feedback = (action_permissions or {}).get("upstream_context") or {}
    governance_feedback = {
        "governance_trust_level": governance_feedback.get("governance_trust_level"),
        "active": governance_feedback.get("governance_active", False),
        # governance_health_score is not in upstream_context; fall back
        # to readiness contributor below.
    }

    state, reasons = _classify_authorization(
        permissions=permissions,
        readiness_state=readiness_state,
        regime=regime,
        committee_decision=committee_decision,
    )
    booleans = _build_authorization_booleans(state=state, permissions=permissions)

    contributors = _build_confidence_contributors(
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        committee_summary=committee_summary,
        meta_intel=meta_intel,
        governance_feedback=governance_feedback,
    )
    confidence = _authorization_confidence(contributors)

    rationale_short, rationale_long = _build_rationale(
        state=state,
        reasons=reasons,
        booleans=booleans,
        confidence=confidence,
        contributors=contributors,
        readiness_state=readiness_state,
        regime=regime,
        committee_decision=committee_decision,
    )
    recommendations = _build_recommendations(
        state=state,
        contributors=contributors,
        booleans=booleans,
    )

    now_iso = _now_iso_utc()
    authorization: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_authorization_engine",
        "engine_version": 1,
        "authorization_state": state,
        "authorization_confidence": round(confidence, 6),
        "authorization_booleans": booleans,
        "reasons": reasons,
        "rationale_short": rationale_short,
        "rationale_long": rationale_long,
        "recommendations": recommendations,
        "confidence_contributors": contributors,
        "confidence_weights": SCORE_WEIGHTS,
        "upstream_context": {
            "readiness_state": readiness_state,
            "regime": regime,
            "committee_decision": committee_decision,
            "health_overall_status": (health_summary or {}).get("overall_status"),
            "meta_trust_level": (meta_intel or {}).get("trust_level"),
            "governance_trust_level": governance_feedback.get("governance_trust_level"),
            "action_permission_overrides": len(
                (action_permissions or {}).get("applied_overrides") or []
            ),
        },
        "inputs_seen": {
            "autonomous_action_permissions": bool(action_permissions),
            "autonomous_action_summary": bool(action_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "autonomous_committee_summary": bool(committee_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_system_health_summary": bool(health_summary),
            "meta_decision_intelligence": bool(meta_intel),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_authorization_engine",
        "authorization_state": state,
        "authorization_confidence": round(confidence, 6),
        "execution_authorized": booleans["execution_authorized"],
        "analysis_only": booleans["analysis_only"],
        "allow_new_buys": booleans["allow_new_buys"],
        "allow_sell_exits": booleans["allow_sell_exits"],
        "allow_rebalance": booleans["allow_rebalance"],
        "allow_rotation": booleans["allow_rotation"],
        "allow_aggressive_deployment": booleans["allow_aggressive_deployment"],
        "allow_defensive_rotation": booleans["allow_defensive_rotation"],
        "require_operator_review": booleans["require_operator_review"],
        "rationale_short": rationale_short,
        "n_recommendations": len(recommendations),
        "n_action_permission_overrides": (
            authorization["upstream_context"]["action_permission_overrides"]
        ),
    }

    md = _render_markdown(
        generated_at=now_iso,
        state=state,
        booleans=booleans,
        confidence=confidence,
        contributors=contributors,
        rationale_short=rationale_short,
        rationale_long=rationale_long,
        recommendations=recommendations,
        readiness_state=readiness_state,
        regime=regime,
        committee_decision=committee_decision,
    )
    return authorization, summary, md


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous execution authorization engine (Step 23). "
            "Final machine-facing authorization layer below any future "
            "execution engine. Produces one of six authorization states, "
            "nine boolean fields, a 0-1 confidence score, and operator-"
            "actionable recommendations."
        ),
    )
    p.add_argument("--action-permissions", default=str(DEFAULT_ACTION_PERMS))
    p.add_argument("--action-summary", default=str(DEFAULT_ACTION_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUMMARY))
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--meta-intel", default=str(DEFAULT_META_INTEL))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[EXECUTION_AUTH] starting (read-only authorization gate)", flush=True)

    action_permissions = _safe_read_json(
        Path(args.action_permissions), label="autonomous_action_permissions.json"
    )
    action_summary = _safe_read_json(
        Path(args.action_summary), label="autonomous_action_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_committee_summary.json"
    )
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    meta_intel = _safe_read_json(Path(args.meta_intel), label="meta_decision_intelligence.json")

    authorization, summary, md = build_execution_authorization(
        action_permissions=action_permissions,
        action_summary=action_summary,
        readiness_summary=readiness_summary,
        committee_summary=committee_summary,
        runtime_policy=runtime_policy,
        health_summary=health_summary,
        meta_intel=meta_intel,
    )

    try:
        _atomic_write_json(authorization, Path(args.out_json))
    except Exception as e:
        _warn(f"failed to write {args.out_json}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_text(md, Path(args.out_md))
    except Exception as e:
        _warn(f"failed to write {args.out_md}: {type(e).__name__}: {e}")
        return 2

    b = authorization["authorization_booleans"]
    print(
        "[EXECUTION_AUTH] "
        f"state={authorization['authorization_state']} "
        f"authorized={b['execution_authorized']} "
        f"new_buys={b['allow_new_buys']} "
        f"rebalance={b['allow_rebalance']} "
        f"aggressive={b['allow_aggressive_deployment']} "
        f"review={b['require_operator_review']}",
        flush=True,
    )
    print(
        f"[EXECUTION_AUTH_OUT] json={Path(args.out_json).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()} "
        f"md={Path(args.out_md).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
