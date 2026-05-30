"""
ARM Autonomy Constitution Engine -- Step 32.

Reads:
    data/results/arm_autonomy_graduation_summary.json    (Step 31)
    data/results/arm_mode_governance_summary.json        (Step 28)
    data/results/autonomous_execution_certificate_summary.json   (Step 27)
    data/results/autonomous_readiness_summary.json       (Step 21)
    data/results/autonomous_system_health_summary.json   (Step 20)
    data/results/runtime_policy_governed.json            (Step 18)
    data/results/autonomous_governance_scorecard.json    (Step 19)
    data/results/meta_decision_intelligence.json         (Step 13)

Writes:
    data/results/arm_autonomy_constitution.json
    data/results/arm_autonomy_constitution.md
    data/results/arm_autonomy_constitution_summary.json
    data/results/arm_constitution_violation_memory.csv
    data/results/arm_constitution_violation_memory.parquet

Purpose
-------
This engine answers:

    "What rules may Triton NEVER violate?"

It codifies a small set of **immutable constitutional laws** that
future ARM Runtime systems MUST obey. The engine itself is pure
governance -- it never executes anything -- but its outputs are
the contract that constrains every other system above and below
it on the autonomy stack.

Constitutional rules override autonomy. Even AUTO_ALLOWED_APPROVED
graduation from Step 31 must yield to a CONSTITUTION_VIOLATED or
CONSTITUTION_LOCKDOWN outcome here.

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens (defined in Step 29 spec) are never written
literally; an import-time self-check raises if they ever appear.

State cascade (strict precedence)
---------------------------------
    1. CONSTITUTION_LOCKDOWN   -- severe failure, all autonomy denied
    2. CONSTITUTION_VIOLATED   -- at least one CRITICAL law breach
    3. CONSTITUTION_DEFENSIVE  -- defensive regime / elevated posture
    4. CONSTITUTION_RESTRICTED -- elevated uncertainty (WARNING only)
    5. CONSTITUTION_CLEAR      -- all laws pass

Immutable laws (eight)
----------------------
    LAW_001_CAPITAL_PRESERVATION         elevated cash under uncertainty
    LAW_002_CERTIFICATE_REQUIRED         valid execution certificate
    LAW_003_MANUAL_OVERRIDE_SUPREMACY    operator override always wins (axiom)
    LAW_004_STALE_INTELLIGENCE_PROHIBITION  no autonomy under stale health
    LAW_005_GOVERNANCE_COLLAPSE_PROHIBITION autonomy prohibited if gov collapsed
    LAW_006_DEFENSIVE_POSTURE_REQUIREMENT defensive regime -> defensive mode
    LAW_007_POSITION_RISK_LIMIT          max_position_pct constitutional ceiling
    LAW_008_AUTONOMY_REVOCATION_ENFORCEMENT  revoked autonomy -> MANUAL only

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only violation memory keyed by (cycle_id, law_id).
* Missing inputs warn-and-continue. With nothing to evaluate the
  engine emits CONSTITUTION_LOCKDOWN as the safe default since the
  inputs it needs to authorize anything are not present.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
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

DEFAULT_GRAD_SUMMARY = RESULTS_DIR / "arm_autonomy_graduation_summary.json"
DEFAULT_ARM_SUMMARY = RESULTS_DIR / "arm_mode_governance_summary.json"
DEFAULT_CERT_SUMMARY = RESULTS_DIR / "autonomous_execution_certificate_summary.json"
DEFAULT_READINESS = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_HEALTH = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_META_DECISION = RESULTS_DIR / "meta_decision_intelligence.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_autonomy_constitution.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_autonomy_constitution.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_autonomy_constitution_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_constitution_violation_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_constitution_violation_memory.parquet"


# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
STATE_LOCKDOWN = "CONSTITUTION_LOCKDOWN"
STATE_VIOLATED = "CONSTITUTION_VIOLATED"
STATE_DEFENSIVE = "CONSTITUTION_DEFENSIVE"
STATE_RESTRICTED = "CONSTITUTION_RESTRICTED"
STATE_CLEAR = "CONSTITUTION_CLEAR"

SEV_INFO = "INFO"
SEV_WARNING = "WARNING"
SEV_CRITICAL = "CRITICAL"
SEV_LOCKDOWN = "LOCKDOWN"

SEV_RANK: Dict[str, int] = {
    SEV_INFO: 0,
    SEV_WARNING: 1,
    SEV_CRITICAL: 2,
    SEV_LOCKDOWN: 3,
}

# Constitutional ceiling for any single position. Runtime policy
# values above this immediately constitute a LAW_007 breach.
CONSTITUTIONAL_MAX_POSITION_PCT = 10.0

# Defensive regimes that trigger LAW_006
DEFENSIVE_REGIMES = frozenset({"DEFENSIVE", "HIGH_VOLATILITY", "RISK_OFF"})

# Governance states that imply collapse
GOVERNANCE_COLLAPSE_STATES = frozenset({"COLLAPSED", "GOVERNANCE_COLLAPSED"})
GOVERNANCE_WEAK_STATES = frozenset({"WEAK", "GOVERNANCE_WEAK"})

# System health categories that prohibit autonomy
PROHIBITED_HEALTH_STATES = frozenset({"STALE", "CRITICAL", "OFFLINE"})

# Certificate states that fail LAW_002
BLOCKED_CERT_STATES = frozenset({"EXECUTION_BLOCKED", "EXECUTION_DENIED"})

# Meta trust levels that imply elevated uncertainty
LOW_TRUST_LEVELS = frozenset({"VERY_LOW", "LOW"})

VIOLATION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "cycle_id",
    "law_id",
    "law_name",
    "severity",
    "reason",
    "autonomy_state",
    "system_state",
    "certificate_state",
    "constitution_state",
    "regime",
    "resolved",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_CONSTITUTION_WARN] {msg}", flush=True)


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
            return [dict(r) for r in csv.DictReader(f)]
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


def _atomic_write_csv(rows: List[Dict[str, Any]], path: Path, *, columns: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    os.replace(tmp, path)


def _atomic_write_parquet(rows: List[Dict[str, Any]], path: Path) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        _warn(f"pandas unavailable for parquet write: {type(e).__name__}: {e}")
        return False
    try:
        df = pd.DataFrame(rows, columns=list(VIOLATION_MEMORY_COLUMNS))
        for col in ("resolved",):
            if col in df.columns:
                df[col] = df[col].map(_to_bool_optional)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        _warn(f"parquet write failed for {path}: {type(e).__name__}: {e}")
        return False


def _to_bool_optional(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


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


def _to_int(x: Any) -> Optional[int]:
    v = _to_float(x)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _norm_upper(x: Any, default: str = "UNKNOWN") -> str:
    s = str(x or "").strip().upper()
    return s or default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------------------------------------
# Evidence extraction
# -----------------------------------------------------------
def _extract_evidence(
    *,
    grad_summary: Dict[str, Any],
    arm_summary: Dict[str, Any],
    cert_summary: Dict[str, Any],
    readiness: Dict[str, Any],
    health: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    meta_decision: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten upstream artefacts into a single evidence dict."""
    arm_mode = _norm_upper((arm_summary or {}).get("arm_mode"))
    cert_state = _norm_upper((cert_summary or {}).get("certification_state"))
    cert_valid = bool(
        (cert_summary or {}).get(
            "certificate_valid",
            cert_state in ("EXECUTION_CERTIFIED", "EXECUTION_CERTIFIED_LIMITED"),
        )
    )

    grad_state = _norm_upper((grad_summary or {}).get("graduation_state"))
    autonomy_earned = bool((grad_summary or {}).get("autonomy_promotion_earned"))
    auto_approved = bool((grad_summary or {}).get("auto_mode_approved"))
    autonomy_revoked = bool((grad_summary or {}).get("autonomy_revoked"))
    observations = _to_int((grad_summary or {}).get("observations")) or 0

    health_status = _norm_upper((health or {}).get("overall_status"))
    health_score = _to_float((health or {}).get("system_health_score"))
    if health_score is None:
        health_score = {
            "HEALTHY": 0.90,
            "DEGRADED": 0.55,
            "STALE": 0.35,
            "CRITICAL": 0.10,
            "OFFLINE": 0.00,
        }.get(health_status, 0.50)

    readiness_state = _norm_upper((readiness or {}).get("readiness_state"))
    readiness_score = _to_float((readiness or {}).get("readiness_score")) or 0.5

    governance_state = _norm_upper(
        (scorecard or {}).get("system_state") or (arm_summary or {}).get("governance_state")
    )
    governance_quality = _to_float(
        (scorecard or {}).get("governance_quality_score")
        or (scorecard or {}).get("intelligence_health_score")
    )
    if governance_quality is None:
        governance_quality = 0.5

    trust_level = _norm_upper((meta_decision or {}).get("trust_level"))
    self_confidence = _to_float((meta_decision or {}).get("self_confidence_score")) or 0.5

    regime = _norm_upper((runtime_policy or {}).get("regime") or (cert_summary or {}).get("regime"))

    return {
        # Autonomy / graduation
        "arm_mode": arm_mode,
        "graduation_state": grad_state,
        "autonomy_promotion_earned": autonomy_earned,
        "auto_mode_approved": auto_approved,
        "autonomy_revoked": autonomy_revoked,
        "observations": observations,
        # Certificate
        "certification_state": cert_state,
        "certificate_valid": cert_valid,
        "certificate_confidence": _clamp(
            _to_float((cert_summary or {}).get("certificate_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        # Readiness
        "readiness_state": readiness_state,
        "readiness_score": _clamp(readiness_score, 0.0, 1.0),
        # Health
        "system_health_status": health_status,
        "system_health_score": _clamp(health_score, 0.0, 1.0),
        # Governance
        "governance_state": governance_state,
        "governance_quality": _clamp(governance_quality, 0.0, 1.0),
        # Meta trust
        "trust_level": trust_level,
        "self_confidence_score": _clamp(self_confidence, 0.0, 1.0),
        # Regime + policy
        "regime": regime,
        "max_position_pct": _to_float((runtime_policy or {}).get("max_position_pct")),
        "target_cash_pct": _to_float((runtime_policy or {}).get("target_cash_pct")),
        "runtime_policy_version": (runtime_policy or {}).get("policy_version"),
    }


# -----------------------------------------------------------
# Law catalogue
# -----------------------------------------------------------
@dataclass
class LawResult:
    triggered: bool
    severity: str
    reason: str


@dataclass
class Law:
    law_id: str
    name: str
    description: str
    check: Callable[[Dict[str, Any]], LawResult]
    is_axiom: bool = False  # axioms never produce violation entries


def _check_law_001_capital_preservation(e: Dict[str, Any]) -> LawResult:
    """Elevated cash posture must be maintained when uncertainty is elevated."""
    cash = e.get("target_cash_pct")
    gov = e.get("governance_quality") or 0.0
    trust = e.get("trust_level")
    uncertain = (gov < 0.50) or (trust in LOW_TRUST_LEVELS)
    if not uncertain:
        return LawResult(
            False, SEV_INFO, "uncertainty within tolerance; capital posture acceptable"
        )
    if cash is None:
        return LawResult(
            True,
            SEV_WARNING,
            "uncertainty elevated but target_cash_pct unknown; " "enforce defensive cash posture",
        )
    if cash < 10.0:
        return LawResult(
            True,
            SEV_CRITICAL,
            f"uncertainty elevated (gov={gov:.2f}, trust={trust}) but "
            f"target_cash_pct={cash:.1f}% below 10% floor",
        )
    if cash < 20.0:
        return LawResult(
            True,
            SEV_WARNING,
            f"uncertainty elevated (gov={gov:.2f}, trust={trust}); "
            f"target_cash_pct={cash:.1f}% modest -- defensive posture advised",
        )
    return LawResult(
        False, SEV_INFO, f"target_cash_pct={cash:.1f}% adequate for elevated uncertainty"
    )


def _check_law_002_certificate_required(e: Dict[str, Any]) -> LawResult:
    """No execution without a valid execution certificate."""
    cert_state = e["certification_state"]
    if cert_state in BLOCKED_CERT_STATES:
        return LawResult(
            True,
            SEV_CRITICAL,
            f"certification_state={cert_state}; execution prohibited",
        )
    if not e["certificate_valid"]:
        # Treat absent/unknown certificate as a CRITICAL block too.
        return LawResult(
            True,
            SEV_CRITICAL,
            f"certificate_valid=False (state={cert_state}); execution prohibited",
        )
    return LawResult(False, SEV_INFO, f"certificate valid (state={cert_state})")


def _check_law_003_manual_override_supremacy(e: Dict[str, Any]) -> LawResult:
    """Operator override always wins. Axiom -- never a violation."""
    return LawResult(
        False, SEV_INFO, "operator override is axiomatic; no autonomous override permitted"
    )


def _check_law_004_stale_intelligence(e: Dict[str, Any]) -> LawResult:
    """No autonomy under stale / critical / offline system health."""
    health = e["system_health_status"]
    if health in PROHIBITED_HEALTH_STATES:
        sev = SEV_LOCKDOWN if health in ("CRITICAL", "OFFLINE") else SEV_CRITICAL
        return LawResult(
            True,
            sev,
            f"system_health={health}; autonomy under stale intelligence prohibited",
        )
    return LawResult(False, SEV_INFO, f"system_health={health}; intelligence is fresh")


def _check_law_005_governance_collapse(e: Dict[str, Any]) -> LawResult:
    """Autonomy prohibited if governance is collapsed or weak."""
    gov_state = e["governance_state"]
    gov_q = e["governance_quality"]
    if gov_state in GOVERNANCE_COLLAPSE_STATES or gov_q < 0.30:
        return LawResult(
            True,
            SEV_LOCKDOWN,
            f"governance_state={gov_state}, governance_quality={gov_q:.2f}; "
            "autonomy prohibited under governance collapse",
        )
    if gov_state in GOVERNANCE_WEAK_STATES or gov_q < 0.45:
        return LawResult(
            True,
            SEV_CRITICAL,
            f"governance_state={gov_state}, governance_quality={gov_q:.2f}; "
            "autonomy prohibited under weak governance",
        )
    return LawResult(False, SEV_INFO, f"governance_quality={gov_q:.2f} adequate")


def _check_law_006_defensive_posture(e: Dict[str, Any]) -> LawResult:
    """Uncertainty -> defensive mode."""
    regime = e["regime"]
    if regime in DEFENSIVE_REGIMES:
        return LawResult(
            True,
            SEV_WARNING,
            f"regime={regime}; defensive posture enforced "
            "(elevated cash, restricted deployment)",
        )
    return LawResult(False, SEV_INFO, f"regime={regime}; no defensive trigger")


def _check_law_007_position_risk_limit(e: Dict[str, Any]) -> LawResult:
    """max_position_pct constitutional ceiling."""
    mp = e.get("max_position_pct")
    if mp is None:
        return LawResult(False, SEV_INFO, "max_position_pct unset; ceiling inactive")
    if mp > CONSTITUTIONAL_MAX_POSITION_PCT:
        return LawResult(
            True,
            SEV_CRITICAL,
            f"max_position_pct={mp:.1f}% exceeds constitutional ceiling "
            f"{CONSTITUTIONAL_MAX_POSITION_PCT:.1f}%",
        )
    return LawResult(False, SEV_INFO, f"max_position_pct={mp:.1f}% within ceiling")


def _check_law_008_autonomy_revocation(e: Dict[str, Any]) -> LawResult:
    """Revoked autonomy forces MANUAL mode."""
    if not e["autonomy_revoked"]:
        return LawResult(False, SEV_INFO, "autonomy not revoked")
    if e["arm_mode"] != "MANUAL":
        return LawResult(
            True,
            SEV_LOCKDOWN,
            f"autonomy revoked but arm_mode={e['arm_mode']} (must be MANUAL)",
        )
    return LawResult(
        True,
        SEV_WARNING,
        "autonomy revoked; MANUAL mode enforced -- recovery required",
    )


LAWS: Tuple[Law, ...] = (
    Law(
        "LAW_001_CAPITAL_PRESERVATION",
        "Capital Preservation Doctrine",
        "Elevated cash posture must be maintained when uncertainty is elevated.",
        _check_law_001_capital_preservation,
    ),
    Law(
        "LAW_002_CERTIFICATE_REQUIRED",
        "Certificate Required",
        "No execution without a valid execution certificate.",
        _check_law_002_certificate_required,
    ),
    Law(
        "LAW_003_MANUAL_OVERRIDE_SUPREMACY",
        "Manual Override Supremacy",
        "Operator override always wins. Axiom -- never overridden by autonomy.",
        _check_law_003_manual_override_supremacy,
        is_axiom=True,
    ),
    Law(
        "LAW_004_STALE_INTELLIGENCE_PROHIBITION",
        "Stale Intelligence Prohibition",
        "No autonomy permitted under stale / critical / offline system health.",
        _check_law_004_stale_intelligence,
    ),
    Law(
        "LAW_005_GOVERNANCE_COLLAPSE_PROHIBITION",
        "Governance Collapse Prohibition",
        "Autonomy prohibited if governance is collapsed or weak.",
        _check_law_005_governance_collapse,
    ),
    Law(
        "LAW_006_DEFENSIVE_POSTURE_REQUIREMENT",
        "Defensive Posture Requirement",
        "Defensive regime requires defensive deployment posture.",
        _check_law_006_defensive_posture,
    ),
    Law(
        "LAW_007_POSITION_RISK_LIMIT",
        "Position Risk Limit",
        f"No single-position weight may exceed {CONSTITUTIONAL_MAX_POSITION_PCT:.1f}%.",
        _check_law_007_position_risk_limit,
    ),
    Law(
        "LAW_008_AUTONOMY_REVOCATION_ENFORCEMENT",
        "Autonomy Revocation Enforcement",
        "Revoked autonomy forces MANUAL mode until recovery is earned.",
        _check_law_008_autonomy_revocation,
    ),
)


# -----------------------------------------------------------
# State classification
# -----------------------------------------------------------
def _classify_state(
    evaluations: List[Tuple[Law, LawResult]],
) -> Tuple[str, List[str]]:
    """Apply state cascade based on the worst-severity trigger."""
    reasons: List[str] = []
    triggered = [(law, res) for law, res in evaluations if res.triggered and not law.is_axiom]

    if any(res.severity == SEV_LOCKDOWN for _, res in triggered):
        for law, res in triggered:
            if res.severity == SEV_LOCKDOWN:
                reasons.append(f"{law.law_id} LOCKDOWN: {res.reason}")
        return STATE_LOCKDOWN, reasons

    if any(res.severity == SEV_CRITICAL for _, res in triggered):
        for law, res in triggered:
            if res.severity == SEV_CRITICAL:
                reasons.append(f"{law.law_id} CRITICAL: {res.reason}")
        return STATE_VIOLATED, reasons

    # Defensive regime takes precedence over generic RESTRICTED
    if any(
        law.law_id == "LAW_006_DEFENSIVE_POSTURE_REQUIREMENT" and res.triggered
        for law, res in evaluations
    ):
        for law, res in triggered:
            if law.law_id == "LAW_006_DEFENSIVE_POSTURE_REQUIREMENT":
                reasons.append(f"{law.law_id}: {res.reason}")
        # Include any other warnings as context
        for law, res in triggered:
            if (
                law.law_id != "LAW_006_DEFENSIVE_POSTURE_REQUIREMENT"
                and res.severity == SEV_WARNING
            ):
                reasons.append(f"{law.law_id} WARNING: {res.reason}")
        return STATE_DEFENSIVE, reasons

    if any(res.severity == SEV_WARNING for _, res in triggered):
        for law, res in triggered:
            if res.severity == SEV_WARNING:
                reasons.append(f"{law.law_id} WARNING: {res.reason}")
        return STATE_RESTRICTED, reasons

    reasons.append("all constitutional laws pass")
    return STATE_CLEAR, reasons


# -----------------------------------------------------------
# Booleans
# -----------------------------------------------------------
def _booleans(
    state: str,
    evidence: Dict[str, Any],
    evaluations: List[Tuple[Law, LawResult]],
) -> Dict[str, bool]:
    violated = state in (STATE_VIOLATED, STATE_LOCKDOWN)
    lockdown = state == STATE_LOCKDOWN
    defensive_active = state == STATE_DEFENSIVE or any(
        law.law_id == "LAW_006_DEFENSIVE_POSTURE_REQUIREMENT" and res.triggered
        for law, res in evaluations
    )
    autonomy_allowed = (
        not violated and not lockdown and bool(evidence.get("autonomy_promotion_earned"))
    )
    execution_allowed = not violated and not lockdown and bool(evidence.get("certificate_valid"))
    # Operator override required everywhere except CLEAR with full earned + approved autonomy
    op_required = not (state == STATE_CLEAR and bool(evidence.get("auto_mode_approved")))
    return {
        "constitution_clear": state == STATE_CLEAR,
        "autonomy_constitutionally_allowed": autonomy_allowed,
        "execution_constitutionally_allowed": execution_allowed,
        "operator_override_required": op_required,
        "defensive_constraints_required": defensive_active,
        "lockdown_required": lockdown,
        "constitution_violated": violated,
    }


# -----------------------------------------------------------
# Confidence
# -----------------------------------------------------------
def _confidence(
    evidence: Dict[str, Any],
    evaluations: List[Tuple[Law, LawResult]],
) -> float:
    """Weighted blend with severity penalty.

    Base components (sum to 1.0):
      0.25 governance_quality
      0.20 readiness_score
      0.20 certificate_confidence
      0.15 system_health_score
      0.20 autonomy_maturity (observations / 150 capped)

    Penalty:
      -0.10 per CRITICAL trigger
      -0.05 per WARNING trigger
      -0.30 if any LOCKDOWN trigger
    """
    maturity = _clamp(evidence["observations"] / 150.0, 0.0, 1.0)
    base = (
        0.25 * evidence["governance_quality"]
        + 0.20 * evidence["readiness_score"]
        + 0.20 * evidence["certificate_confidence"]
        + 0.15 * evidence["system_health_score"]
        + 0.20 * maturity
    )
    penalty = 0.0
    for law, res in evaluations:
        if not res.triggered or law.is_axiom:
            continue
        if res.severity == SEV_LOCKDOWN:
            penalty += 0.30
        elif res.severity == SEV_CRITICAL:
            penalty += 0.10
        elif res.severity == SEV_WARNING:
            penalty += 0.05
    return _clamp(base - penalty, 0.0, 1.0)


# -----------------------------------------------------------
# Violation memory
# -----------------------------------------------------------
def _build_violation_rows(
    *,
    cycle_id: str,
    state: str,
    evidence: Dict[str, Any],
    evaluations: List[Tuple[Law, LawResult]],
) -> List[Dict[str, Any]]:
    """Materialize a memory row per *triggered, non-axiom, WARNING+* law."""
    rows: List[Dict[str, Any]] = []
    for law, res in evaluations:
        if law.is_axiom or not res.triggered:
            continue
        if SEV_RANK.get(res.severity, 0) < SEV_RANK[SEV_WARNING]:
            continue
        rows.append(
            {
                "timestamp_utc": cycle_id,
                "cycle_id": cycle_id,
                "law_id": law.law_id,
                "law_name": law.name,
                "severity": res.severity,
                "reason": res.reason,
                "autonomy_state": evidence["graduation_state"],
                "system_state": evidence["system_health_status"],
                "certificate_state": evidence["certification_state"],
                "constitution_state": state,
                "regime": evidence["regime"],
                "resolved": False,
            }
        )
    return rows


def _merge_violation_memory(
    existing: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    *,
    current_cycle_id: str,
) -> List[Dict[str, Any]]:
    """
    Append-only memory keyed by (cycle_id, law_id). Re-running the
    same cycle replaces partial writes for that cycle; rows from
    older cycles are preserved.
    """
    keyed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in existing:
        key = (str(r.get("cycle_id", "")), str(r.get("law_id", "")))
        keyed[key] = r
    # Clear any previous entries for the current cycle so a re-run
    # accurately reflects the current evaluation (no stale violations
    # that have since been resolved).
    for k in list(keyed.keys()):
        if k[0] == current_cycle_id:
            del keyed[k]
    for r in new_rows:
        key = (str(r.get("cycle_id", "")), str(r.get("law_id", "")))
        keyed[key] = r
    out = list(keyed.values())
    for r in out:
        for c in VIOLATION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _recommendations(
    state: str,
    evidence: Dict[str, Any],
    violations: List[Dict[str, Any]],
) -> List[str]:
    recs: List[str] = []
    if state == STATE_LOCKDOWN:
        recs.append("Constitutional LOCKDOWN: maintain manual supervision.")
        recs.append("Halt all autonomous execution; require operator approval for any action.")
        recs.append("Refresh stale intelligence and restore governance before re-evaluating.")
        return recs
    if state == STATE_VIOLATED:
        recs.append("Constitutional violation detected: deny autonomy escalation.")
        recs.append("Require operator override before any execution attempt.")
        for v in violations:
            if v.get("severity") == SEV_CRITICAL:
                recs.append(f"Resolve {v['law_id']} ({v['law_name']}) before re-evaluating.")
        return recs
    if state == STATE_DEFENSIVE:
        recs.append("Defensive constitutional posture: enforce defensive deployment only.")
        recs.append("Maintain elevated cash target until regime clears.")
        recs.append("Require operator approval for any aggressive deployment.")
        return recs
    if state == STATE_RESTRICTED:
        recs.append("Constitutional posture is RESTRICTED: limited deployment only.")
        recs.append("Maintain operator supervision; do not promote autonomy until warnings clear.")
        return recs
    # CLEAR
    recs.append("All constitutional laws pass; standard governance applies.")
    if not evidence.get("auto_mode_approved"):
        recs.append("Maintain operator approval until graduation reaches AUTO_ALLOWED_APPROVED.")
    return recs


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    reasons: List[str],
    evaluations: List[Tuple[Law, LawResult]],
    booleans: Dict[str, bool],
    confidence: float,
    evidence: Dict[str, Any],
    violations: List[Dict[str, Any]],
    recommendations: List[str],
) -> str:
    def fmt(x: Optional[float], spec: str = ".3f") -> str:
        if x is None:
            return "-"
        return format(x, spec)

    lines: List[str] = []
    lines.append("# Triton ARM Constitution")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Constitutional State")
    lines.append("")
    lines.append(f"**{state}**")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| confidence | {confidence:.3f} |")
    lines.append(
        f"| autonomy_constitutionally_allowed | {booleans['autonomy_constitutionally_allowed']} |"
    )
    lines.append(
        f"| execution_constitutionally_allowed | {booleans['execution_constitutionally_allowed']} |"
    )
    lines.append(f"| operator_override_required | {booleans['operator_override_required']} |")
    lines.append(
        f"| defensive_constraints_required | {booleans['defensive_constraints_required']} |"
    )
    lines.append(f"| lockdown_required | {booleans['lockdown_required']} |")
    lines.append(f"| constitution_violated | {booleans['constitution_violated']} |")
    lines.append("")
    lines.append("| evidence | value |")
    lines.append("|---|---|")
    lines.append(f"| graduation_state | {evidence['graduation_state']} |")
    lines.append(f"| arm_mode | {evidence['arm_mode']} |")
    lines.append(f"| certification_state | {evidence['certification_state']} |")
    lines.append(f"| readiness_state | {evidence['readiness_state']} |")
    lines.append(f"| system_health_status | {evidence['system_health_status']} |")
    lines.append(f"| governance_state | {evidence['governance_state']} |")
    lines.append(f"| regime | {evidence['regime']} |")
    lines.append(f"| max_position_pct | {fmt(evidence.get('max_position_pct'), '.1f')} |")
    lines.append(f"| target_cash_pct | {fmt(evidence.get('target_cash_pct'), '.1f')} |")
    lines.append("")

    lines.append("## Immutable Laws")
    lines.append("")
    lines.append("| law_id | name | status | severity | rationale |")
    lines.append("|---|---|---|---|---|")
    for law, res in evaluations:
        if law.is_axiom:
            status = "AXIOM"
            sev = "-"
        elif res.triggered:
            status = "TRIGGERED"
            sev = res.severity
        else:
            status = "PASS"
            sev = "-"
        lines.append(
            f"| {law.law_id} | {law.name} | {status} | {sev} | " f"{res.reason.replace('|', ' ')} |"
        )
    lines.append("")

    lines.append("## Violations")
    lines.append("")
    if not violations:
        lines.append("_(no constitutional violations this cycle)_")
    else:
        lines.append("| law_id | severity | reason |")
        lines.append("|---|---|---|")
        for v in violations:
            lines.append(
                f"| {v['law_id']} | {v['severity']} | " f"{str(v['reason']).replace('|', ' ')} |"
            )
    lines.append("")

    lines.append("## Why")
    lines.append("")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    lines.append(_narrative(state, evidence, confidence, violations))
    lines.append("")
    return "\n".join(lines)


def _narrative(
    state: str,
    evidence: Dict[str, Any],
    confidence: float,
    violations: List[Dict[str, Any]],
) -> str:
    if state == STATE_LOCKDOWN:
        return (
            f"Triton entered CONSTITUTION_LOCKDOWN. {len(violations)} "
            f"constitutional rule(s) breached; all autonomous execution "
            f"is denied until the underlying failures resolve. "
            f"Confidence {confidence:.2f}."
        )
    if state == STATE_VIOLATED:
        return (
            f"Triton entered CONSTITUTION_VIOLATED because at least one "
            f"CRITICAL constitutional rule was breached. Autonomy escalation "
            f"is denied and execution requires operator override. "
            f"Confidence {confidence:.2f}."
        )
    if state == STATE_DEFENSIVE:
        return (
            f"Triton entered CONSTITUTION_DEFENSIVE because regime "
            f"{evidence['regime']} elevated capital preservation requirements. "
            f"Defensive deployment posture is enforced. "
            f"Confidence {confidence:.2f}."
        )
    if state == STATE_RESTRICTED:
        return (
            f"Triton entered CONSTITUTION_RESTRICTED. Uncertainty is "
            f"elevated and operator supervision remains required for any "
            f"autonomous action. Confidence {confidence:.2f}."
        )
    return (
        f"All constitutional laws pass: Triton is CONSTITUTION_CLEAR. "
        f"Standard graduation-driven governance applies; "
        f"constitutional rules remain in force as the floor. "
        f"Confidence {confidence:.2f}."
    )


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_constitution(
    *,
    grad_summary: Dict[str, Any],
    arm_summary: Dict[str, Any],
    cert_summary: Dict[str, Any],
    readiness: Dict[str, Any],
    health: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    meta_decision: Dict[str, Any],
    existing_memory_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    evidence = _extract_evidence(
        grad_summary=grad_summary,
        arm_summary=arm_summary,
        cert_summary=cert_summary,
        readiness=readiness,
        health=health,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        meta_decision=meta_decision,
    )

    # Defensive default: if essentially nothing is wired up, force lockdown.
    have_any_evidence = any(
        bool(x)
        for x in (
            grad_summary,
            cert_summary,
            health,
            scorecard,
            runtime_policy,
        )
    )

    evaluations: List[Tuple[Law, LawResult]] = []
    for law in LAWS:
        try:
            res = law.check(evidence)
        except Exception as e:
            _warn(f"law {law.law_id} check failed: {type(e).__name__}: {e}")
            res = LawResult(True, SEV_CRITICAL, f"law check error: {type(e).__name__}")
        evaluations.append((law, res))

    state, reasons = _classify_state(evaluations)

    if not have_any_evidence and state != STATE_LOCKDOWN:
        # Safe default: no inputs means no basis for any permission.
        state = STATE_LOCKDOWN
        reasons.insert(0, "no upstream evidence present; defaulting to LOCKDOWN")

    booleans = _booleans(state, evidence, evaluations)
    confidence = _confidence(evidence, evaluations)

    violations = _build_violation_rows(
        cycle_id=timestamp,
        state=state,
        evidence=evidence,
        evaluations=evaluations,
    )
    merged_memory = _merge_violation_memory(
        existing_memory_rows,
        violations,
        current_cycle_id=timestamp,
    )

    recommendations = _recommendations(state, evidence, violations)

    severity = SEV_INFO
    if booleans["lockdown_required"]:
        severity = SEV_LOCKDOWN
    elif booleans["constitution_violated"]:
        severity = SEV_CRITICAL
    elif state == STATE_DEFENSIVE or state == STATE_RESTRICTED:
        severity = SEV_WARNING

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        reasons=reasons,
        evaluations=evaluations,
        booleans=booleans,
        confidence=confidence,
        evidence=evidence,
        violations=violations,
        recommendations=recommendations,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_autonomy_constitution_engine",
        "engine_version": 1,
        "constitution_state": state,
        "severity": severity,
        "reasons": reasons,
        "confidence": round(confidence, 6),
        "evidence": evidence,
        "constitutional_booleans": booleans,
        "violations": violations,
        "violation_count": len(violations),
        "laws": [
            {
                "law_id": law.law_id,
                "name": law.name,
                "description": law.description,
                "is_axiom": law.is_axiom,
                "triggered": res.triggered,
                "severity": res.severity if res.triggered else SEV_INFO,
                "reason": res.reason,
            }
            for law, res in evaluations
        ],
        "recommendations": recommendations,
        "violation_memory_size_after_append": len(merged_memory),
        "constitutional_ceilings": {
            "max_position_pct_ceiling": CONSTITUTIONAL_MAX_POSITION_PCT,
            "defensive_regimes": sorted(DEFENSIVE_REGIMES),
            "blocked_certificate_states": sorted(BLOCKED_CERT_STATES),
            "prohibited_health_states": sorted(PROHIBITED_HEALTH_STATES),
        },
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "constitutional_only": True,
        },
        "inputs_seen": {
            "arm_autonomy_graduation_summary": bool(grad_summary),
            "arm_mode_governance_summary": bool(arm_summary),
            "autonomous_execution_certificate_summary": bool(cert_summary),
            "autonomous_readiness_summary": bool(readiness),
            "autonomous_system_health_summary": bool(health),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_governance_scorecard": bool(scorecard),
            "meta_decision_intelligence": bool(meta_decision),
            "existing_violation_memory_rows": len(existing_memory_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_autonomy_constitution_engine",
        "constitution_state": state,
        "severity": severity,
        "violation_count": len(violations),
        "violated_laws": [v["law_id"] for v in violations],
        "confidence": record["confidence"],
        "constitution_clear": booleans["constitution_clear"],
        "autonomy_constitutionally_allowed": booleans["autonomy_constitutionally_allowed"],
        "execution_constitutionally_allowed": booleans["execution_constitutionally_allowed"],
        "operator_override_required": booleans["operator_override_required"],
        "defensive_constraints_required": booleans["defensive_constraints_required"],
        "lockdown_required": booleans["lockdown_required"],
        "constitution_violated": booleans["constitution_violated"],
        "graduation_state": evidence["graduation_state"],
        "arm_mode": evidence["arm_mode"],
        "certification_state": evidence["certification_state"],
        "regime": evidence["regime"],
        "n_recommendations": len(recommendations),
        "violation_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM autonomy constitution engine (Step 32). "
            "Codifies Triton's immutable constitutional laws. No broker "
            "calls; no portfolio mutation."
        ),
    )
    p.add_argument("--grad-summary", default=str(DEFAULT_GRAD_SUMMARY))
    p.add_argument("--arm-summary", default=str(DEFAULT_ARM_SUMMARY))
    p.add_argument("--cert-summary", default=str(DEFAULT_CERT_SUMMARY))
    p.add_argument("--readiness", default=str(DEFAULT_READINESS))
    p.add_argument("--health", default=str(DEFAULT_HEALTH))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--meta-decision", default=str(DEFAULT_META_DECISION))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_CONSTITUTION] starting (read-only constitution; no broker calls)",
        flush=True,
    )

    grad_summary = _safe_read_json(
        Path(args.grad_summary), label="arm_autonomy_graduation_summary.json"
    )
    arm_summary = _safe_read_json(Path(args.arm_summary), label="arm_mode_governance_summary.json")
    cert_summary = _safe_read_json(
        Path(args.cert_summary), label="autonomous_execution_certificate_summary.json"
    )
    readiness = _safe_read_json(Path(args.readiness), label="autonomous_readiness_summary.json")
    health = _safe_read_json(Path(args.health), label="autonomous_system_health_summary.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    meta_decision = _safe_read_json(
        Path(args.meta_decision), label="meta_decision_intelligence.json"
    )
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_constitution_violation_memory.csv"
    )

    record, summary, md, merged_memory = build_constitution(
        grad_summary=grad_summary,
        arm_summary=arm_summary,
        cert_summary=cert_summary,
        readiness=readiness,
        health=health,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        meta_decision=meta_decision,
        existing_memory_rows=existing_mem,
    )

    try:
        _atomic_write_json(record, Path(args.out_json))
    except Exception as e:
        _warn(f"failed to write {args.out_json}: {type(e).__name__}: {e}")
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
    try:
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=VIOLATION_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    print(
        "[ARM_CONSTITUTION] "
        f"state={record['constitution_state']} "
        f"violations={record['violation_count']} "
        f"severity={record['severity']} "
        f"lockdown={record['constitutional_booleans']['lockdown_required']} "
        f"confidence={record['confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_CONSTITUTION_SAFETY] broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_CONSTITUTION_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()} "
        f"memory_csv={Path(args.out_mem_csv).as_posix()} "
        f"memory_parquet={Path(args.out_mem_parquet).as_posix() if parquet_ok else 'SKIPPED'}",
        flush=True,
    )
    return 0


# -----------------------------------------------------------
# Self-inspection: enforce the no-broker safety rule at import time
# -----------------------------------------------------------
_FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "Alpaca" + "Broker",
    "place" + "_order",
    "submit" + "_order",
    "execute" + "_trades",
    "place" + "_live_orders",
    "broker" + "_client",
)


def _self_check_no_broker_tokens() -> None:
    """Refuse to import if any forbidden broker token appears in source."""
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return
    for tok in _FORBIDDEN_TOKENS:
        if tok in src:
            raise RuntimeError(
                f"[ARM_CONSTITUTION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
