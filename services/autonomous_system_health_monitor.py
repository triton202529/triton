"""
Autonomous System Health Monitor -- Step 20.

Reads:
    data/results/autonomous_governance_scorecard.json   (Step 19)
    data/results/autonomous_committee_summary.json      (Step 15)
    data/results/runtime_policy_governed.json           (Step 18)
    data/results/adaptive_regime.json                   (Step 10)
    data/results/meta_decision_intelligence.json        (Step 13)
    data/results/governance_trust_feedback.json         (Step 17)
    data/results/autonomous_strategy_diagnostics.json   (Step 16)
    data/results/investment_committee_summary.json      (Step 9)
    data/results/portfolio_execution_summary.json       (Step 7)
    data/results/portfolio_rebalance_summary.json       (Step 6)

Writes:
    data/results/autonomous_system_health.json
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_system_health.md

Purpose
-------
Steps 1-19 built Triton's autonomous intelligence stack. Step 20
adds the operational gatekeeper that any future automation must
consult before trusting that stack:

    "Is Triton's autonomous system healthy enough to trust today?"

The monitor performs three concentric checks:

1. *Pipeline freshness* -- per-artifact existence + mtime age,
   tagged FRESH/STALE/MISSING.
2. *Critical subsystem health* -- per-subsystem combined check of
   freshness AND structural validity (expected fields present),
   tagged HEALTHY/DEGRADED/STALE/MISSING.
3. *Overall status* -- single label drawn from
   HEALTHY/DEGRADED/STALE/CRITICAL/OFFLINE based on a strict
   precedence cascade.

The output blocking flags are the *contract* downstream automation
consumes. ``BLOCK_AUTONOMOUS_DEPLOYMENT`` set means: "do not
auto-execute today, regardless of what the committee says".

Safety
------
* READ ONLY. No broker calls, no engine state mutation. The
  blocking flags are advisory -- enforcing them is each downstream
  engine's responsibility.
* Atomic writes (.tmp + os.replace) for all three outputs.
* Missing inputs warn-and-continue. With zero inputs the monitor
  reports overall_status=OFFLINE and sets every blocking flag.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_GOV_FEEDBACK = RESULTS_DIR / "governance_trust_feedback.json"
DEFAULT_DIAGNOSTICS = RESULTS_DIR / "autonomous_strategy_diagnostics.json"
DEFAULT_IC_SUMMARY = RESULTS_DIR / "investment_committee_summary.json"
DEFAULT_EXEC_SUMMARY = RESULTS_DIR / "portfolio_execution_summary.json"
DEFAULT_REBALANCE_SUMMARY = RESULTS_DIR / "portfolio_rebalance_summary.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_system_health.json"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_system_health.md"


# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
# Default freshness thresholds. Triton runs at most a few times a
# trading day, so 8h is "within today's cycle" and 48h is "way past".
FRESH_MAX_MIN = 8 * 60  # 480 min  -- still within today's cycle
STALE_MAX_MIN = 48 * 60  # 2880 min -- beyond two trading days

# Status sentinels
FRESH = "FRESH"
STALE = "STALE"
MISSING = "MISSING"

HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
# (STALE and MISSING reused at the subsystem layer)

# Overall-status cascade (in decreasing severity)
OVERALL_OFFLINE = "OFFLINE"
OVERALL_CRITICAL = "CRITICAL"
OVERALL_STALE = "STALE"
OVERALL_DEGRADED = "DEGRADED"
OVERALL_HEALTHY = "HEALTHY"

# Blocking flags
FLAG_BLOCK_AUTONOMOUS_DEPLOYMENT = "BLOCK_AUTONOMOUS_DEPLOYMENT"
FLAG_BLOCK_NEW_BUYS = "BLOCK_NEW_BUYS"
FLAG_REQUIRE_PIPELINE_REFRESH = "REQUIRE_PIPELINE_REFRESH"
FLAG_REQUIRE_OPERATOR_REVIEW = "REQUIRE_OPERATOR_REVIEW"
FLAG_ALLOW_READ_ONLY_ANALYSIS = "ALLOW_READ_ONLY_ANALYSIS"


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[SYSTEM_HEALTH_WARN] {msg}", flush=True)


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


def _safe_mtime_utc(path: Path) -> Optional[datetime]:
    try:
        if not path.is_file():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError as e:
        _warn(f"stat failed for {path}: {type(e).__name__}: {e}")
        return None


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


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------
# Artifact catalogue
#
# Each entry: (key, default_path, expected_fields)
#
# expected_fields is the minimum set of keys that must be present
# inside the JSON for the artifact to be considered structurally
# valid. A fresh file missing one of these is DEGRADED, not
# HEALTHY.
# -----------------------------------------------------------
ARTIFACT_SPECS: Tuple[Tuple[str, Path, Tuple[str, ...]], ...] = (
    (
        "autonomous_governance_scorecard",
        DEFAULT_SCORECARD,
        ("system_state", "scores", "scores_known", "narrative"),
    ),
    (
        "autonomous_committee_summary",
        DEFAULT_COMMITTEE_SUMMARY,
        # Step 15's summary canonically uses `decision`; the full
        # decision blob uses `committee_decision`. Accept either via the
        # score check below; expected_fields stays empty here so neither
        # name being absent triggers DEGRADED on its own.
        (),
    ),
    (
        "runtime_policy_governed",
        DEFAULT_GOV_POLICY,
        ("confidence_threshold", "deployment_threshold", "target_cash_pct", "max_position_pct"),
    ),
    ("adaptive_regime", DEFAULT_REGIME, ("regime",)),
    ("meta_decision_intelligence", DEFAULT_META_INTEL, ("trust_level", "self_confidence_score")),
    (
        "governance_trust_feedback",
        DEFAULT_GOV_FEEDBACK,
        ("governance_trust_level", "active", "deltas"),
    ),
    ("autonomous_strategy_diagnostics", DEFAULT_DIAGNOSTICS, ("scores", "scores_known")),
    (
        "investment_committee_summary",
        DEFAULT_IC_SUMMARY,
        (),
    ),  # no hard schema -- presence alone is enough
    ("portfolio_execution_summary", DEFAULT_EXEC_SUMMARY, ()),
    ("portfolio_rebalance_summary", DEFAULT_REBALANCE_SUMMARY, ()),
)

# Subsystem health derives from artifact health. Some subsystems
# blend multiple artifacts and apply additional score floors.
#
# Each subsystem: (subsystem_key, primary_artifact, score_check_fn)
#   score_check_fn(payload) -> Optional[str]: returns a reason string
#   if payload fails the score floor (downgrades HEALTHY -> DEGRADED),
#   or None if the score floor is met or no score is available.
SubsystemSpec = Tuple[str, str, Optional[Callable[[Dict[str, Any]], Optional[str]]]]


def _score_governance_scorecard(payload: Dict[str, Any]) -> Optional[str]:
    s = (payload or {}).get("scores") or {}
    ih = _to_float(s.get("intelligence_health_score"))
    if ih is None or ih < 0.30:
        return f"intelligence_health_score={ih if ih is not None else 'NA'} below floor 0.30"
    return None


def _score_committee_decision(payload: Dict[str, Any]) -> Optional[str]:
    # Accept either `decision` (Step 15 summary canonical name) or
    # `committee_decision` (the full decision blob's canonical name);
    # matches the tolerance pattern used by Step 19's scorecard.
    decision = (
        str((payload or {}).get("decision") or (payload or {}).get("committee_decision") or "")
        .strip()
        .upper()
    )
    if not decision:
        return "decision/committee_decision absent"
    return None


def _score_runtime_policy(payload: Dict[str, Any]) -> Optional[str]:
    cash = _to_float((payload or {}).get("target_cash_pct"))
    max_pos = _to_float((payload or {}).get("max_position_pct"))
    if cash is None or not (0.0 <= cash <= 100.0):
        return f"target_cash_pct out of range ({cash})"
    if max_pos is None or not (0.0 <= max_pos <= 100.0):
        return f"max_position_pct out of range ({max_pos})"
    return None


def _score_regime(payload: Dict[str, Any]) -> Optional[str]:
    if not str((payload or {}).get("regime") or "").strip():
        return "regime label absent"
    return None


def _score_meta_trust(payload: Dict[str, Any]) -> Optional[str]:
    if not str((payload or {}).get("trust_level") or "").strip():
        return "trust_level absent"
    sc = _to_float((payload or {}).get("self_confidence_score"))
    if sc is None or not (0.0 <= sc <= 1.0):
        return f"self_confidence_score out of range ({sc})"
    return None


def _score_strategy_diagnostics(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance((payload or {}).get("scores"), dict):
        return "scores dict missing"
    return None


def _score_execution_intent(payload: Dict[str, Any]) -> Optional[str]:
    # Presence-only check; no hard score floor.
    return None


SUBSYSTEM_SPECS: Tuple[SubsystemSpec, ...] = (
    ("governance_scorecard_health", "autonomous_governance_scorecard", _score_governance_scorecard),
    ("committee_decision_health", "autonomous_committee_summary", _score_committee_decision),
    ("runtime_policy_health", "runtime_policy_governed", _score_runtime_policy),
    ("regime_health", "adaptive_regime", _score_regime),
    ("meta_trust_health", "meta_decision_intelligence", _score_meta_trust),
    ("strategy_diagnostics_health", "autonomous_strategy_diagnostics", _score_strategy_diagnostics),
    ("execution_intent_health", "portfolio_execution_summary", _score_execution_intent),
)

# Subset of subsystems considered "critical" -- if any of these are
# MISSING the overall status escalates to CRITICAL or OFFLINE.
CRITICAL_SUBSYSTEMS = {
    "governance_scorecard_health",
    "committee_decision_health",
    "runtime_policy_health",
    "regime_health",
}


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


# -----------------------------------------------------------
# Freshness
# -----------------------------------------------------------
def _freshness_for(path: Path, *, now: datetime) -> Dict[str, Any]:
    mtime = _safe_mtime_utc(path)
    if mtime is None:
        return {
            "path": str(path),
            "file_exists": False,
            "age_minutes": None,
            "mtime_utc": None,
            "freshness_status": MISSING,
            "very_stale": False,
        }
    age_sec = (now - mtime).total_seconds()
    age_min = max(0.0, age_sec / 60.0)
    if age_min <= FRESH_MAX_MIN:
        status = FRESH
    else:
        status = STALE
    return {
        "path": str(path),
        "file_exists": True,
        "age_minutes": round(age_min, 2),
        "mtime_utc": _iso(mtime),
        "freshness_status": status,
        "very_stale": age_min > STALE_MAX_MIN,
    }


# -----------------------------------------------------------
# Subsystem health
# -----------------------------------------------------------
def _subsystem_status(
    *,
    artifact_health: Dict[str, Any],
    payload: Dict[str, Any],
    expected_fields: Tuple[str, ...],
    score_check: Optional[Callable[[Dict[str, Any]], Optional[str]]],
) -> Tuple[str, List[str]]:
    """Returns (status, reasons[])."""
    reasons: List[str] = []
    fresh_status = artifact_health.get("freshness_status")

    if fresh_status == MISSING:
        return MISSING, ["artifact file missing"]
    if fresh_status == STALE:
        reasons.append(f"artifact stale ({artifact_health.get('age_minutes')} min old)")
        return STALE, reasons

    missing_fields = [f for f in expected_fields if f not in (payload or {})]
    if missing_fields:
        reasons.append(f"missing required fields: {missing_fields}")
        return DEGRADED, reasons

    if score_check is not None:
        msg = score_check(payload or {})
        if msg:
            reasons.append(msg)
            return DEGRADED, reasons

    return HEALTHY, reasons


# -----------------------------------------------------------
# Overall status cascade
# -----------------------------------------------------------
def _overall_status(subsystem_statuses: Dict[str, str]) -> str:
    counts = {HEALTHY: 0, DEGRADED: 0, STALE: 0, MISSING: 0}
    for s in subsystem_statuses.values():
        counts[s] = counts.get(s, 0) + 1

    total = len(subsystem_statuses) or 1
    missing_count = counts[MISSING]

    if missing_count >= max(5, total - 2):
        return OVERALL_OFFLINE

    critical_missing = [k for k in CRITICAL_SUBSYSTEMS if subsystem_statuses.get(k) == MISSING]
    if critical_missing:
        return OVERALL_CRITICAL

    if counts[STALE] > 0 or any(subsystem_statuses.get(k) == STALE for k in CRITICAL_SUBSYSTEMS):
        return OVERALL_STALE

    if counts[DEGRADED] > 0 or missing_count > 0:
        return OVERALL_DEGRADED

    return OVERALL_HEALTHY


# -----------------------------------------------------------
# Blocking flags
# -----------------------------------------------------------
def _blocking_flags(
    *,
    overall: str,
    subsystem_statuses: Dict[str, str],
    artifact_healths: Dict[str, Dict[str, Any]],
) -> List[str]:
    flags: List[str] = []

    if overall in (OVERALL_CRITICAL, OVERALL_OFFLINE, OVERALL_STALE):
        flags.append(FLAG_BLOCK_AUTONOMOUS_DEPLOYMENT)

    if (
        overall in (OVERALL_CRITICAL, OVERALL_OFFLINE)
        or subsystem_statuses.get("committee_decision_health") in (MISSING, STALE)
        or subsystem_statuses.get("runtime_policy_health") in (MISSING, STALE)
    ):
        flags.append(FLAG_BLOCK_NEW_BUYS)

    if any(a.get("freshness_status") == STALE for a in artifact_healths.values()):
        flags.append(FLAG_REQUIRE_PIPELINE_REFRESH)

    if overall in (OVERALL_CRITICAL, OVERALL_OFFLINE) or any(
        a.get("freshness_status") == MISSING for a in artifact_healths.values()
    ):
        flags.append(FLAG_REQUIRE_OPERATOR_REVIEW)

    # Read-only analysis is always allowed -- the scorecard itself is
    # read-only and we want operators to be able to triage even when
    # the rest of the stack is offline.
    flags.append(FLAG_ALLOW_READ_ONLY_ANALYSIS)

    # Stable, de-duplicated order
    seen: List[str] = []
    for f in flags:
        if f not in seen:
            seen.append(f)
    return seen


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    overall: str,
    subsystem_statuses: Dict[str, Dict[str, Any]],  # value = {"status": ..., "reasons": [...]}
    artifact_healths: Dict[str, Dict[str, Any]],
    flags: List[str],
) -> List[str]:
    recs: List[str] = []

    stale_arts = sorted(
        k for k, v in artifact_healths.items() if v.get("freshness_status") == STALE
    )
    missing_arts = sorted(
        k for k, v in artifact_healths.items() if v.get("freshness_status") == MISSING
    )

    if overall == OVERALL_OFFLINE:
        recs.append(
            "Pipeline is OFFLINE -- run the full Step 1..19 sequence end-to-end "
            "before consuming any autonomous output."
        )
    elif overall == OVERALL_CRITICAL:
        missing_critical = sorted(
            k for k in CRITICAL_SUBSYSTEMS if subsystem_statuses.get(k, {}).get("status") == MISSING
        )
        recs.append(
            "Critical subsystems missing -- regenerate "
            f"{', '.join(missing_critical) or 'core artifacts'} before resuming autonomy."
        )
    elif overall == OVERALL_STALE:
        recs.append(
            "Refresh full pipeline -- one or more critical artifacts are beyond the "
            f"freshness window ({FRESH_MAX_MIN} min)."
        )
    elif overall == OVERALL_DEGRADED:
        degraded = sorted(k for k, v in subsystem_statuses.items() if v.get("status") == DEGRADED)
        recs.append(
            "Investigate degraded subsystems before trusting autonomous output: "
            f"{', '.join(degraded) or 'see subsystem report'}."
        )
    else:
        recs.append("All subsystems HEALTHY -- continue normal cycle cadence.")

    # Targeted per-subsystem hints
    if subsystem_statuses.get("runtime_policy_health", {}).get("status") in (
        MISSING,
        STALE,
        DEGRADED,
    ):
        recs.append("Rebuild runtime policy (Steps 11 + 14 + 18) before any new deployment.")
    if subsystem_statuses.get("committee_decision_health", {}).get("status") in (
        MISSING,
        STALE,
        DEGRADED,
    ):
        recs.append("Rerun investment committee report (Step 15) -- last decision is unusable.")
    if subsystem_statuses.get("governance_scorecard_health", {}).get("status") in (
        MISSING,
        STALE,
        DEGRADED,
    ):
        recs.append("Rerun governance scorecard (Step 19) to refresh self-evaluation.")
    if subsystem_statuses.get("regime_health", {}).get("status") in (MISSING, STALE):
        recs.append("Recompute adaptive regime (Step 10) -- regime context is unreliable.")
    if subsystem_statuses.get("strategy_diagnostics_health", {}).get("status") in (MISSING, STALE):
        recs.append("Refresh strategy diagnostics (Step 16) -- governance signals lag the data.")

    if missing_arts:
        recs.append(
            f"Missing artifacts ({len(missing_arts)}): "
            f"{', '.join(missing_arts)} -- regenerate or accept degraded mode."
        )
    if stale_arts and not any(r.startswith("Refresh full pipeline") for r in recs):
        recs.append(
            f"Stale artifacts ({len(stale_arts)}): "
            f"{', '.join(stale_arts)} -- run their owning engines."
        )

    if FLAG_REQUIRE_OPERATOR_REVIEW in flags:
        recs.append(
            "Operator review required before execution -- do not auto-trade until "
            "all blocking flags clear."
        )
    if overall in (OVERALL_HEALTHY, OVERALL_DEGRADED):
        recs.append("Continue read-only monitoring while diagnostics accumulate.")

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
    overall: str,
    deployment_allowed: bool,
    review_required: bool,
    artifact_healths: Dict[str, Dict[str, Any]],
    subsystem_statuses: Dict[str, Dict[str, Any]],
    flags: List[str],
    recommendations: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Triton Autonomous System Health")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_  ")
    lines.append(f"_Overall status: **{overall}**_  ")
    lines.append(f"_Autonomous deployment allowed: **{deployment_allowed}**_  ")
    lines.append(f"_Operator review required: **{review_required}**_")
    lines.append("")

    lines.append("## Pipeline Freshness")
    lines.append("")
    lines.append("| artifact | exists | age (min) | status |")
    lines.append("|---|---|---|---|")
    for key, h in artifact_healths.items():
        age = h.get("age_minutes")
        age_str = f"{age:.1f}" if isinstance(age, (int, float)) else "NA"
        very = " (VERY_STALE)" if h.get("very_stale") else ""
        lines.append(
            f"| {key} | {h.get('file_exists')} | {age_str} | {h.get('freshness_status')}{very} |"
        )
    lines.append("")

    lines.append("## Critical Subsystem Health")
    lines.append("")
    lines.append("| subsystem | status | reasons |")
    lines.append("|---|---|---|")
    for key, s in subsystem_statuses.items():
        reasons = "; ".join(s.get("reasons") or []) or "-"
        lines.append(f"| {key} | {s.get('status')} | {reasons} |")
    lines.append("")

    lines.append("## Blocking Flags")
    lines.append("")
    if flags:
        for f in flags:
            lines.append(f"- `{f}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_health_report(
    *,
    now: datetime,
    artifact_paths: Dict[str, Path],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    # 1. Freshness per artifact
    artifact_healths: Dict[str, Dict[str, Any]] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    for key, _path, _fields in ARTIFACT_SPECS:
        p = artifact_paths.get(key) or _path
        artifact_healths[key] = _freshness_for(p, now=now)
        payloads[key] = _safe_read_json(p, label=key)

    # 2. Subsystem health
    subsystem_statuses: Dict[str, Dict[str, Any]] = {}
    expected_fields_by_key = {key: fields for key, _p, fields in ARTIFACT_SPECS}
    for sub_key, art_key, score_fn in SUBSYSTEM_SPECS:
        status, reasons = _subsystem_status(
            artifact_health=artifact_healths[art_key],
            payload=payloads.get(art_key) or {},
            expected_fields=expected_fields_by_key[art_key],
            score_check=score_fn,
        )
        subsystem_statuses[sub_key] = {
            "status": status,
            "reasons": reasons,
            "primary_artifact": art_key,
        }

    # 3. Overall status
    status_only = {k: v["status"] for k, v in subsystem_statuses.items()}
    overall = _overall_status(status_only)

    # 4. Blocking flags
    flags = _blocking_flags(
        overall=overall,
        subsystem_statuses=status_only,
        artifact_healths=artifact_healths,
    )

    deployment_allowed = FLAG_BLOCK_AUTONOMOUS_DEPLOYMENT not in flags
    read_only_allowed = FLAG_ALLOW_READ_ONLY_ANALYSIS in flags
    operator_review_required = FLAG_REQUIRE_OPERATOR_REVIEW in flags

    # 5. Recommendations
    recommendations = _build_recommendations(
        overall=overall,
        subsystem_statuses=subsystem_statuses,
        artifact_healths=artifact_healths,
        flags=flags,
    )

    stale_artifacts = sorted(
        k for k, v in artifact_healths.items() if v.get("freshness_status") == STALE
    )
    missing_artifacts = sorted(
        k for k, v in artifact_healths.items() if v.get("freshness_status") == MISSING
    )

    now_iso = _iso(now) or ""

    health: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_system_health_monitor",
        "engine_version": 1,
        "overall_status": overall,
        "autonomous_deployment_allowed": deployment_allowed,
        "read_only_allowed": read_only_allowed,
        "operator_review_required": operator_review_required,
        "blocking_flags": flags,
        "freshness_thresholds": {
            "fresh_max_minutes": FRESH_MAX_MIN,
            "stale_max_minutes": STALE_MAX_MIN,
        },
        "artifact_freshness": artifact_healths,
        "subsystem_health": subsystem_statuses,
        "stale_artifacts": stale_artifacts,
        "missing_artifacts": missing_artifacts,
        "recommendations": recommendations,
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_system_health_monitor",
        "overall_status": overall,
        "autonomous_deployment_allowed": deployment_allowed,
        "read_only_allowed": read_only_allowed,
        "operator_review_required": operator_review_required,
        "stale_artifacts": stale_artifacts,
        "missing_artifacts": missing_artifacts,
        "blocking_flags": flags,
        "recommendations": recommendations,
        "n_artifacts_total": len(artifact_healths),
        "n_artifacts_fresh": sum(
            1 for v in artifact_healths.values() if v.get("freshness_status") == FRESH
        ),
        "n_artifacts_stale": len(stale_artifacts),
        "n_artifacts_missing": len(missing_artifacts),
        "n_subsystems_total": len(subsystem_statuses),
        "n_subsystems_healthy": sum(
            1 for v in subsystem_statuses.values() if v["status"] == HEALTHY
        ),
        "n_subsystems_degraded": sum(
            1 for v in subsystem_statuses.values() if v["status"] == DEGRADED
        ),
        "n_subsystems_stale": sum(1 for v in subsystem_statuses.values() if v["status"] == STALE),
        "n_subsystems_missing": sum(
            1 for v in subsystem_statuses.values() if v["status"] == MISSING
        ),
    }

    md = _render_markdown(
        generated_at=now_iso,
        overall=overall,
        deployment_allowed=deployment_allowed,
        review_required=operator_review_required,
        artifact_healths=artifact_healths,
        subsystem_statuses=subsystem_statuses,
        flags=flags,
        recommendations=recommendations,
    )
    return health, summary, md


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous system health monitor (Step 20). "
            "Reports per-artifact freshness, per-subsystem health, an "
            "overall status label, blocking flags, and operator-actionable "
            "recommendations. Acts as the operational gatekeeper for any "
            "future automation that consumes Triton's autonomous output."
        ),
    )
    for key, default, _fields in ARTIFACT_SPECS:
        p.add_argument(f"--{key.replace('_', '-')}", default=str(default))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument(
        "--now",
        default=None,
        help="Override 'now' for freshness calculation (ISO-8601 UTC). "
        "Used by tests to simulate age windows deterministically.",
    )
    return p.parse_args(argv)


def _resolve_now(now_str: Optional[str]) -> datetime:
    if not now_str:
        return _now_utc()
    try:
        s = now_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0)
    except Exception as e:
        _warn(f"failed to parse --now ({now_str!r}): {type(e).__name__}: {e}; using system time")
        return _now_utc()


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[SYSTEM_HEALTH] starting (pipeline freshness + subsystem consistency)", flush=True)

    artifact_paths: Dict[str, Path] = {}
    for key, _default, _fields in ARTIFACT_SPECS:
        arg_attr = key.replace("-", "_")
        artifact_paths[key] = Path(getattr(args, arg_attr))

    now = _resolve_now(args.now)
    health, summary, md = build_health_report(now=now, artifact_paths=artifact_paths)

    try:
        _atomic_write_json(health, Path(args.out_json))
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

    print(
        "[SYSTEM_HEALTH] "
        f"status={health['overall_status']} "
        f"deployment_allowed={health['autonomous_deployment_allowed']} "
        f"review_required={health['operator_review_required']} "
        f"stale={len(health['stale_artifacts'])} "
        f"missing={len(health['missing_artifacts'])}",
        flush=True,
    )
    if health["blocking_flags"]:
        print(
            "[SYSTEM_HEALTH_FLAGS] " + ",".join(health["blocking_flags"]),
            flush=True,
        )
    print(
        f"[SYSTEM_HEALTH_OUT] json={Path(args.out_json).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()} "
        f"md={Path(args.out_md).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
