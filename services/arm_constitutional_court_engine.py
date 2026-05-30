"""
ARM Constitutional Court Engine -- Step 33.

Reads:
    data/results/arm_autonomy_constitution_summary.json   (Step 32)
    data/results/arm_autonomy_constitution.json           (Step 32)
    data/results/arm_autonomy_graduation_summary.json     (Step 31)
    data/results/autonomous_execution_risk_committee_summary.json  (Step 26)
    data/results/autonomous_execution_certificate_summary.json     (Step 27)
    data/results/arm_mode_governance_summary.json         (Step 28)
    data/results/autonomous_readiness_summary.json        (Step 21)
    data/results/autonomous_system_health_summary.json    (Step 20)
    data/results/runtime_policy_governed.json             (Step 18)

Writes:
    data/results/arm_constitutional_court.json
    data/results/arm_constitutional_court.md
    data/results/arm_constitutional_court_summary.json
    data/results/arm_constitutional_precedent_memory.csv
    data/results/arm_constitutional_precedent_memory.parquet

Purpose
-------
This engine answers:

    "Should constitutional law overrule autonomy?"

It is Triton's final judicial authority -- a read-only governance
layer that adjudicates conflicts between autonomy graduation, ARM
mode governance, execution certificates, risk committee verdicts,
and the immutable constitutional laws from Step 32.

Constitutional law supersedes autonomy. The court's ruling is the
machine-readable contract that future ARM Runtime systems MUST
validate before any action.

Judicial precedence (strict order, highest wins)
------------------------------------------------
    1. Constitution
    2. Capital Preservation Doctrine
    3. Operator supremacy
    4. Risk committee
    5. Execution certificate
    6. Autonomy graduation
    7. ARM governance

Ruling cascade
--------------
    1. COURT_LOCKDOWN      severe constitutional breach / collapse
    2. COURT_OVERRULED     constitution overrides autonomy / cert
    3. COURT_ESCALATED     conflicting signals, operator review
    4. COURT_APPROVED_LIMITED  defensive / restricted / assisted only
    5. COURT_APPROVED      all layers aligned, no material conflict

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only precedent memory keyed by (case_id).
* Missing inputs warn-and-continue; absent evidence defaults to
  COURT_LOCKDOWN as the safe judicial posture.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

DEFAULT_CONST_SUMMARY = RESULTS_DIR / "arm_autonomy_constitution_summary.json"
DEFAULT_CONST_RECORD = RESULTS_DIR / "arm_autonomy_constitution.json"
DEFAULT_GRAD_SUMMARY = RESULTS_DIR / "arm_autonomy_graduation_summary.json"
DEFAULT_COMMITTEE = RESULTS_DIR / "autonomous_execution_risk_committee_summary.json"
DEFAULT_CERT_SUMMARY = RESULTS_DIR / "autonomous_execution_certificate_summary.json"
DEFAULT_ARM_SUMMARY = RESULTS_DIR / "arm_mode_governance_summary.json"
DEFAULT_READINESS = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_HEALTH = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_constitutional_court.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_constitutional_court.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_constitutional_precedent_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_constitutional_precedent_memory.parquet"


# -----------------------------------------------------------
# Ruling constants
# -----------------------------------------------------------
RULING_APPROVED = "COURT_APPROVED"
RULING_APPROVED_LIMITED = "COURT_APPROVED_LIMITED"
RULING_ESCALATED = "COURT_ESCALATED"
RULING_OVERRULED = "COURT_OVERRULED"
RULING_LOCKDOWN = "COURT_LOCKDOWN"

CONSTITUTION_LOCKDOWN = "CONSTITUTION_LOCKDOWN"
CONSTITUTION_VIOLATED = "CONSTITUTION_VIOLATED"
CONSTITUTION_DEFENSIVE = "CONSTITUTION_DEFENSIVE"
CONSTITUTION_RESTRICTED = "CONSTITUTION_RESTRICTED"
CONSTITUTION_CLEAR = "CONSTITUTION_CLEAR"

SEV_LOCKDOWN = "LOCKDOWN"
SEV_CRITICAL = "CRITICAL"

DEFENSIVE_REGIMES = frozenset({"DEFENSIVE", "HIGH_VOLATILITY", "RISK_OFF"})
PROHIBITED_HEALTH = frozenset({"STALE", "CRITICAL", "OFFLINE"})
BLOCKED_CERT = frozenset({"EXECUTION_BLOCKED", "EXECUTION_DENIED"})

# Judicial precedence layer names (for reporting)
PRECEDENCE_LAYERS: Tuple[str, ...] = (
    "constitution",
    "capital_preservation",
    "operator_supremacy",
    "risk_committee",
    "execution_certificate",
    "autonomy_graduation",
    "arm_governance",
)

PRECEDENT_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "case_id",
    "ruling",
    "violated_laws",
    "conflict_type",
    "precedent_reason",
    "autonomy_state",
    "certificate_state",
    "constitutional_state",
    "judicial_confidence",
    "conflict_count",
    "constitutional_override_triggered",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_COURT_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(PRECEDENT_MEMORY_COLUMNS))
        for col in ("judicial_confidence", "conflict_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "constitutional_override_triggered" in df.columns:
            df["constitutional_override_triggered"] = df["constitutional_override_triggered"].map(
                _to_bool_optional
            )
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


def _norm_upper(x: Any, default: str = "UNKNOWN") -> str:
    s = str(x or "").strip().upper()
    return s or default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _case_id(timestamp: str, ruling: str) -> str:
    digest = hashlib.sha256(f"{timestamp}|{ruling}".encode("utf-8")).hexdigest()[:12]
    return f"COURT-{digest.upper()}"


# -----------------------------------------------------------
# Evidence extraction
# -----------------------------------------------------------
def _extract_evidence(
    *,
    const_summary: Dict[str, Any],
    const_record: Dict[str, Any],
    grad_summary: Dict[str, Any],
    committee: Dict[str, Any],
    cert_summary: Dict[str, Any],
    arm_summary: Dict[str, Any],
    readiness: Dict[str, Any],
    health: Dict[str, Any],
    runtime_policy: Dict[str, Any],
) -> Dict[str, Any]:
    const_state = _norm_upper(
        const_summary.get("constitution_state") or const_record.get("constitution_state")
    )
    const_severity = _norm_upper(const_summary.get("severity") or const_record.get("severity"))
    const_confidence = (
        _to_float(const_summary.get("confidence") or const_record.get("confidence")) or 0.0
    )

    violated_laws: List[str] = list(
        const_summary.get("violated_laws")
        or [v.get("law_id") for v in (const_record.get("violations") or []) if v.get("law_id")]
    )

    grad_state = _norm_upper(grad_summary.get("graduation_state"))
    autonomy_earned = bool(grad_summary.get("autonomy_promotion_earned"))
    auto_approved = bool(grad_summary.get("auto_mode_approved"))
    autonomy_revoked = bool(grad_summary.get("autonomy_revoked"))

    committee_verdict = _norm_upper(committee.get("committee_verdict"))
    committee_approved = bool(committee.get("execution_approved"))
    committee_limited = bool(committee.get("limited_execution_only"))
    committee_escalate = committee_verdict == "ESCALATE"
    committee_confidence = _to_float(committee.get("approval_confidence")) or 0.0

    cert_state = _norm_upper(
        cert_summary.get("certification_state") or const_summary.get("certification_state")
    )
    cert_valid = bool(
        cert_summary.get(
            "certificate_valid",
            cert_state in ("EXECUTION_CERTIFIED", "EXECUTION_CERTIFIED_LIMITED"),
        )
    )
    cert_confidence = _to_float(cert_summary.get("certificate_confidence")) or 0.0

    arm_mode = _norm_upper(arm_summary.get("arm_mode") or const_summary.get("arm_mode"))
    arm_auto_allowed = bool(arm_summary.get("autonomous_execution_allowed"))

    readiness_state = _norm_upper(readiness.get("readiness_state"))
    readiness_score = _to_float(readiness.get("readiness_score")) or 0.5

    health_status = _norm_upper(
        health.get("overall_status") or const_summary.get("system_health_status")
    )
    health_score = _to_float(health.get("system_health_score"))
    if health_score is None:
        health_score = {
            "HEALTHY": 0.90,
            "DEGRADED": 0.55,
            "STALE": 0.35,
            "CRITICAL": 0.10,
            "OFFLINE": 0.00,
        }.get(health_status, 0.50)

    regime = _norm_upper((runtime_policy or {}).get("regime") or const_summary.get("regime"))
    target_cash = _to_float((runtime_policy or {}).get("target_cash_pct"))

    governance_quality = _to_float(grad_summary.get("governance_quality")) or 0.5

    return {
        "constitution_state": const_state,
        "constitution_severity": const_severity,
        "constitution_confidence": _clamp(const_confidence, 0.0, 1.0),
        "constitution_clear": bool(const_summary.get("constitution_clear")),
        "constitution_violated": bool(const_summary.get("constitution_violated")),
        "lockdown_required": bool(const_summary.get("lockdown_required")),
        "defensive_constraints_required": bool(const_summary.get("defensive_constraints_required")),
        "autonomy_constitutionally_allowed": bool(
            const_summary.get("autonomy_constitutionally_allowed")
        ),
        "execution_constitutionally_allowed": bool(
            const_summary.get("execution_constitutionally_allowed")
        ),
        "operator_override_required": bool(const_summary.get("operator_override_required")),
        "violated_laws": violated_laws,
        "graduation_state": grad_state,
        "autonomy_promotion_earned": autonomy_earned,
        "auto_mode_approved": auto_approved,
        "autonomy_revoked": autonomy_revoked,
        "governance_quality": _clamp(governance_quality, 0.0, 1.0),
        "committee_verdict": committee_verdict,
        "committee_approved": committee_approved,
        "committee_limited": committee_limited,
        "committee_escalate": committee_escalate,
        "committee_confidence": _clamp(committee_confidence, 0.0, 1.0),
        "certification_state": cert_state,
        "certificate_valid": cert_valid,
        "certificate_confidence": _clamp(cert_confidence, 0.0, 1.0),
        "arm_mode": arm_mode,
        "arm_auto_allowed": arm_auto_allowed,
        "readiness_state": readiness_state,
        "readiness_score": _clamp(readiness_score, 0.0, 1.0),
        "system_health_status": health_status,
        "system_health_score": _clamp(health_score, 0.0, 1.0),
        "regime": regime,
        "target_cash_pct": target_cash,
    }


# -----------------------------------------------------------
# Conflict detection
# -----------------------------------------------------------
def _detect_conflicts(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of conflict dicts with type, severity, reason."""
    conflicts: List[Dict[str, Any]] = []

    # autonomy_vs_constitution
    autonomy_claims = (
        evidence["autonomy_promotion_earned"]
        or evidence["auto_mode_approved"]
        or evidence["arm_auto_allowed"]
    )
    if autonomy_claims and (
        evidence["constitution_violated"]
        or evidence["lockdown_required"]
        or not evidence["autonomy_constitutionally_allowed"]
    ):
        conflicts.append(
            {
                "conflict_type": "autonomy_vs_constitution",
                "severity": SEV_CRITICAL if evidence["lockdown_required"] else SEV_CRITICAL,
                "reason": (
                    f"autonomy layer claims promotion (grad={evidence['graduation_state']}, "
                    f"arm={evidence['arm_mode']}) but constitution={evidence['constitution_state']} "
                    "denies autonomy"
                ),
            }
        )

    # certificate_vs_constitution
    if evidence["certificate_valid"] and not evidence["execution_constitutionally_allowed"]:
        conflicts.append(
            {
                "conflict_type": "certificate_vs_constitution",
                "severity": SEV_CRITICAL,
                "reason": (
                    f"certificate_state={evidence['certification_state']} valid but "
                    f"constitution={evidence['constitution_state']} prohibits execution"
                ),
            }
        )
    if not evidence["certificate_valid"] and evidence["committee_approved"]:
        conflicts.append(
            {
                "conflict_type": "certificate_vs_constitution",
                "severity": SEV_CRITICAL,
                "reason": (
                    f"risk committee approved but certificate_state="
                    f"{evidence['certification_state']} invalid"
                ),
            }
        )

    # governance_vs_risk
    if evidence["committee_approved"] and evidence["governance_quality"] < 0.45:
        conflicts.append(
            {
                "conflict_type": "governance_vs_risk",
                "severity": SEV_CRITICAL,
                "reason": (
                    f"committee_approved=True but governance_quality="
                    f"{evidence['governance_quality']:.2f} is weak"
                ),
            }
        )
    if evidence["committee_escalate"] and evidence["autonomy_promotion_earned"]:
        conflicts.append(
            {
                "conflict_type": "governance_vs_risk",
                "severity": "WARNING",
                "reason": ("committee ESCALATE conflicts with earned autonomy promotion"),
            }
        )

    # execution_vs_capital_preservation
    capital_threatened = (
        evidence["regime"] in DEFENSIVE_REGIMES
        or evidence["defensive_constraints_required"]
        or (evidence.get("target_cash_pct") is not None and evidence["target_cash_pct"] >= 25.0)
    )
    if capital_threatened and (
        evidence["committee_approved"] and not evidence["committee_limited"]
    ):
        conflicts.append(
            {
                "conflict_type": "execution_vs_capital_preservation",
                "severity": "WARNING",
                "reason": (
                    f"full committee approval under defensive posture "
                    f"(regime={evidence['regime']}, cash="
                    f"{evidence.get('target_cash_pct')})"
                ),
            }
        )

    # stale_intelligence_conflict
    if evidence["system_health_status"] in PROHIBITED_HEALTH:
        lower_layers_ok = (
            evidence["committee_approved"]
            or evidence["autonomy_promotion_earned"]
            or evidence["arm_auto_allowed"]
        )
        if lower_layers_ok:
            conflicts.append(
                {
                    "conflict_type": "stale_intelligence_conflict",
                    "severity": (
                        SEV_LOCKDOWN
                        if evidence["system_health_status"] in ("CRITICAL", "OFFLINE")
                        else SEV_CRITICAL
                    ),
                    "reason": (
                        f"system_health={evidence['system_health_status']} but lower "
                        "governance layers suggest action"
                    ),
                }
            )

    return conflicts


# -----------------------------------------------------------
# Precedence resolution
# -----------------------------------------------------------
def _precedence_stack(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the precedence stack showing which layer wins for autonomy
    and execution permission.
    """
    layers: List[Dict[str, Any]] = []

    # 1. Constitution
    const_allows_autonomy = evidence["autonomy_constitutionally_allowed"]
    const_allows_exec = evidence["execution_constitutionally_allowed"]
    layers.append(
        {
            "layer": "constitution",
            "rank": 1,
            "state": evidence["constitution_state"],
            "autonomy_allowed": const_allows_autonomy,
            "execution_allowed": const_allows_exec,
            "note": f"severity={evidence['constitution_severity']}",
        }
    )

    # 2. Capital preservation
    cap_ok = not (
        evidence["regime"] in DEFENSIVE_REGIMES and evidence["defensive_constraints_required"]
    )
    layers.append(
        {
            "layer": "capital_preservation",
            "rank": 2,
            "state": evidence["regime"],
            "autonomy_allowed": cap_ok,
            "execution_allowed": cap_ok or evidence["committee_limited"],
            "note": f"target_cash_pct={evidence.get('target_cash_pct')}",
        }
    )

    # 3. Operator supremacy (axiom -- always requires override unless fully approved)
    op_required = evidence["operator_override_required"]
    layers.append(
        {
            "layer": "operator_supremacy",
            "rank": 3,
            "state": "OPERATOR_OVERRIDE" if op_required else "OPERATOR_OPTIONAL",
            "autonomy_allowed": not op_required or evidence["auto_mode_approved"],
            "execution_allowed": not op_required,
            "note": "operator override always wins (axiom)",
        }
    )

    # 4. Risk committee
    layers.append(
        {
            "layer": "risk_committee",
            "rank": 4,
            "state": evidence["committee_verdict"],
            "autonomy_allowed": evidence["committee_approved"],
            "execution_allowed": evidence["committee_approved"],
            "note": f"limited={evidence['committee_limited']}",
        }
    )

    # 5. Execution certificate
    layers.append(
        {
            "layer": "execution_certificate",
            "rank": 5,
            "state": evidence["certification_state"],
            "autonomy_allowed": evidence["certificate_valid"],
            "execution_allowed": evidence["certificate_valid"],
            "note": f"confidence={evidence['certificate_confidence']:.2f}",
        }
    )

    # 6. Autonomy graduation
    layers.append(
        {
            "layer": "autonomy_graduation",
            "rank": 6,
            "state": evidence["graduation_state"],
            "autonomy_allowed": evidence["autonomy_promotion_earned"]
            and not evidence["autonomy_revoked"],
            "execution_allowed": evidence["auto_mode_approved"],
            "note": f"revoked={evidence['autonomy_revoked']}",
        }
    )

    # 7. ARM governance
    layers.append(
        {
            "layer": "arm_governance",
            "rank": 7,
            "state": evidence["arm_mode"],
            "autonomy_allowed": evidence["arm_auto_allowed"],
            "execution_allowed": evidence["arm_auto_allowed"],
            "note": "",
        }
    )

    return layers


def _constitutional_override_triggered(
    evidence: Dict[str, Any],
    conflicts: List[Dict[str, Any]],
) -> bool:
    """True when constitution layer overrules a lower layer."""
    if evidence["lockdown_required"] or evidence["constitution_violated"]:
        return True
    override_types = {
        "autonomy_vs_constitution",
        "certificate_vs_constitution",
        "stale_intelligence_conflict",
    }
    return any(c["conflict_type"] in override_types for c in conflicts)


# -----------------------------------------------------------
# Ruling classification
# -----------------------------------------------------------
def _classify_ruling(
    *,
    evidence: Dict[str, Any],
    conflicts: List[Dict[str, Any]],
    have_evidence: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not have_evidence:
        reasons.append("no upstream judicial evidence; defaulting to LOCKDOWN")
        return RULING_LOCKDOWN, reasons

    # ---- 1. COURT_LOCKDOWN ----
    if (
        evidence["lockdown_required"]
        or evidence["constitution_state"] == CONSTITUTION_LOCKDOWN
        or evidence["constitution_severity"] == SEV_LOCKDOWN
        or evidence["system_health_status"] in ("CRITICAL", "OFFLINE")
        or evidence["autonomy_revoked"]
    ):
        if evidence["lockdown_required"]:
            reasons.append("constitution lockdown_required=True")
        if evidence["constitution_state"] == CONSTITUTION_LOCKDOWN:
            reasons.append(f"constitution_state={CONSTITUTION_LOCKDOWN}")
        if evidence["system_health_status"] in ("CRITICAL", "OFFLINE"):
            reasons.append(f"system_health={evidence['system_health_status']}")
        if evidence["autonomy_revoked"]:
            reasons.append("autonomy_revoked=True; manual-only enforced")
        return RULING_LOCKDOWN, reasons

    # ---- 2. COURT_OVERRULED ----
    override = _constitutional_override_triggered(evidence, conflicts)
    cert_invalid = (
        not evidence["certificate_valid"] or evidence["certification_state"] in BLOCKED_CERT
    )
    capital_threatened = (
        evidence["constitution_violated"]
        or evidence["constitution_state"] == CONSTITUTION_VIOLATED
        or cert_invalid
        or override
    )
    if capital_threatened:
        if evidence["constitution_violated"]:
            reasons.append(f"constitution_violated=True (state={evidence['constitution_state']})")
        if cert_invalid:
            reasons.append(f"certificate invalid (state={evidence['certification_state']})")
        if override:
            reasons.append("constitutional override supersedes lower governance layers")
        if not reasons:
            reasons.append("capital preservation doctrine threatened")
        return RULING_OVERRULED, reasons

    # ---- 3. COURT_ESCALATED ----
    material_conflicts = [c for c in conflicts if c["severity"] in (SEV_CRITICAL, "WARNING")]
    if (
        evidence["committee_escalate"]
        or len(material_conflicts) >= 2
        or (material_conflicts and evidence["operator_override_required"])
    ):
        if evidence["committee_escalate"]:
            reasons.append("risk committee verdict=ESCALATE")
        if material_conflicts:
            reasons.append(f"{len(material_conflicts)} governance conflict(s) detected")
        if evidence["operator_override_required"]:
            reasons.append("operator review required under elevated uncertainty")
        return RULING_ESCALATED, reasons

    # ---- 4. COURT_APPROVED_LIMITED ----
    limited_path = (
        evidence["constitution_state"] in (CONSTITUTION_DEFENSIVE, CONSTITUTION_RESTRICTED)
        or evidence["defensive_constraints_required"]
        or evidence["committee_limited"]
        or evidence["arm_mode"] in ("ASSISTED", "AUTO_DISABLED")
        or not evidence["autonomy_promotion_earned"]
    )
    if limited_path:
        if evidence["constitution_state"] in (CONSTITUTION_DEFENSIVE, CONSTITUTION_RESTRICTED):
            reasons.append(f"constitution_state={evidence['constitution_state']}")
        if evidence["defensive_constraints_required"]:
            reasons.append("defensive constitutional constraints active")
        if evidence["committee_limited"]:
            reasons.append("risk committee APPROVED_LIMITED")
        if evidence["arm_mode"] in ("ASSISTED", "AUTO_DISABLED"):
            reasons.append(f"arm_mode={evidence['arm_mode']}; assisted autonomy only")
        if not reasons:
            reasons.append("limited deployment posture required")
        return RULING_APPROVED_LIMITED, reasons

    # ---- 5. COURT_APPROVED ----
    all_aligned = (
        evidence["constitution_clear"]
        and evidence["certificate_valid"]
        and evidence["committee_approved"]
        and evidence["autonomy_promotion_earned"]
        and not conflicts
    )
    if all_aligned:
        reasons.append(
            "constitution clear, certificate valid, committee approved, "
            "autonomy earned, no material conflicts"
        )
        return RULING_APPROVED, reasons

    # Fallback: if constitution clear but minor gaps, limited approval
    if evidence["constitution_clear"] and evidence["certificate_valid"]:
        reasons.append(
            "constitution and certificate pass; minor governance gaps -> limited approval"
        )
        return RULING_APPROVED_LIMITED, reasons

    reasons.append("governance layers not fully aligned; defaulting to escalation posture")
    return RULING_ESCALATED, reasons


# -----------------------------------------------------------
# Court booleans
# -----------------------------------------------------------
def _court_booleans(
    ruling: str,
    evidence: Dict[str, Any],
    override: bool,
    conflicts: List[Dict[str, Any]],
) -> Dict[str, bool]:
    lockdown = ruling == RULING_LOCKDOWN
    overruled = ruling == RULING_OVERRULED
    escalated = ruling == RULING_ESCALATED
    approved = ruling in (RULING_APPROVED, RULING_APPROVED_LIMITED)

    autonomy_allowed = (
        approved
        and not lockdown
        and not overruled
        and evidence["autonomy_constitutionally_allowed"]
        and (ruling == RULING_APPROVED or ruling == RULING_APPROVED_LIMITED)
    )
    execution_allowed = (
        approved
        and not lockdown
        and not overruled
        and evidence["execution_constitutionally_allowed"]
        and evidence["certificate_valid"]
        and evidence["committee_approved"]
    )
    if ruling == RULING_APPROVED_LIMITED:
        execution_allowed = execution_allowed and evidence["committee_limited"]

    precedent = bool(conflicts) or override or lockdown

    return {
        "autonomy_judicially_allowed": autonomy_allowed,
        "execution_judicially_allowed": execution_allowed,
        "constitutional_override_triggered": override or overruled,
        "operator_review_required": (
            lockdown or overruled or escalated or evidence["operator_override_required"]
        ),
        "lockdown_required": lockdown,
        "precedent_created": precedent,
    }


# -----------------------------------------------------------
# Judicial confidence
# -----------------------------------------------------------
def _judicial_confidence(
    evidence: Dict[str, Any],
    conflicts: List[Dict[str, Any]],
) -> float:
    base = (
        0.30 * evidence["constitution_confidence"]
        + 0.20 * evidence["governance_quality"]
        + 0.15 * evidence["readiness_score"]
        + 0.15 * evidence["certificate_confidence"]
        + 0.20 * evidence["system_health_score"]
    )
    penalty = 0.0
    for c in conflicts:
        sev = c.get("severity", "WARNING")
        if sev == SEV_LOCKDOWN:
            penalty += 0.25
        elif sev == SEV_CRITICAL:
            penalty += 0.10
        else:
            penalty += 0.05
    if evidence["constitution_severity"] == SEV_CRITICAL:
        penalty += 0.10
    if evidence["constitution_severity"] == SEV_LOCKDOWN:
        penalty += 0.25
    return _clamp(base - penalty, 0.0, 1.0)


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _recommendations(ruling: str, evidence: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    if ruling == RULING_LOCKDOWN:
        recs.append("Maintain manual supervision; court enforces LOCKDOWN.")
        recs.append("Halt all autonomous execution pending constitutional recovery.")
        recs.append("Refresh stale intelligence and restore governance before re-adjudication.")
        return recs
    if ruling == RULING_OVERRULED:
        recs.append("Constitutional override remains active; execution denied.")
        recs.append("Resolve constitutional violations before any autonomy escalation.")
        if evidence["certification_state"] in BLOCKED_CERT:
            recs.append("Obtain valid execution certificate before re-adjudication.")
        return recs
    if ruling == RULING_ESCALATED:
        recs.append("Escalate to operator review; conflicting governance signals detected.")
        recs.append("Do not proceed with autonomous deployment until conflicts resolve.")
        return recs
    if ruling == RULING_APPROVED_LIMITED:
        recs.append("Continue assisted-only operation under constitutional constraints.")
        recs.append("Maintain elevated cash posture and defensive deployment discipline.")
        recs.append("Require operator approval for any aggressive deployment.")
        return recs
    recs.append("Court approves governance alignment; standard constitutional floor applies.")
    recs.append("Continue shadow evaluation in parallel as regression guard.")
    return recs


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    ruling: str,
    reasons: List[str],
    evidence: Dict[str, Any],
    precedence: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    confidence: float,
    recommendations: List[str],
    case_id: str,
) -> str:
    lines: List[str] = []
    lines.append("# Triton Constitutional Court")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Judicial Ruling")
    lines.append("")
    lines.append(f"**{ruling}**")
    lines.append("")
    lines.append(f"Case ID: `{case_id}`")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| judicial_confidence | {confidence:.3f} |")
    lines.append(f"| autonomy_judicially_allowed | {booleans['autonomy_judicially_allowed']} |")
    lines.append(f"| execution_judicially_allowed | {booleans['execution_judicially_allowed']} |")
    lines.append(
        f"| constitutional_override_triggered | {booleans['constitutional_override_triggered']} |"
    )
    lines.append(f"| operator_review_required | {booleans['operator_review_required']} |")
    lines.append(f"| lockdown_required | {booleans['lockdown_required']} |")
    lines.append(f"| precedent_created | {booleans['precedent_created']} |")
    lines.append("")

    lines.append("## Constitutional Precedence")
    lines.append("")
    lines.append("| rank | layer | state | autonomy | execution | note |")
    lines.append("|---|---|---|---|---|---|")
    for layer in precedence:
        lines.append(
            f"| {layer['rank']} | {layer['layer']} | {layer['state']} | "
            f"{layer['autonomy_allowed']} | {layer['execution_allowed']} | "
            f"{layer.get('note', '')} |"
        )
    lines.append("")

    lines.append("## Conflicts")
    lines.append("")
    if not conflicts:
        lines.append("_(no governance conflicts detected)_")
    else:
        lines.append("| type | severity | reason |")
        lines.append("|---|---|---|")
        for c in conflicts:
            lines.append(
                f"| {c['conflict_type']} | {c['severity']} | "
                f"{str(c['reason']).replace('|', ' ')} |"
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
    lines.append(_narrative(ruling, evidence, confidence, booleans, conflicts))
    lines.append("")
    return "\n".join(lines)


def _narrative(
    ruling: str,
    evidence: Dict[str, Any],
    confidence: float,
    booleans: Dict[str, bool],
    conflicts: List[Dict[str, Any]],
) -> str:
    n_conflicts = len(conflicts)
    if ruling == RULING_LOCKDOWN:
        return (
            f"The court entered {RULING_LOCKDOWN}. Severe constitutional or "
            f"operational failure requires manual-only operation. "
            f"{n_conflicts} conflict(s) noted. Confidence {confidence:.2f}."
        )
    if ruling == RULING_OVERRULED:
        return (
            f"The court {RULING_OVERRULED} autonomous deployment because "
            f"constitutional law and capital preservation superseded execution "
            f"authorization (constitution={evidence['constitution_state']}, "
            f"cert={evidence['certification_state']}). Confidence {confidence:.2f}."
        )
    if ruling == RULING_ESCALATED:
        return (
            f"The court {RULING_ESCALATED} due to {n_conflicts} conflicting "
            f"governance signal(s). Operator review is required before any "
            f"autonomous action. Confidence {confidence:.2f}."
        )
    if ruling == RULING_APPROVED_LIMITED:
        return (
            f"The court {RULING_APPROVED_LIMITED} under defensive/restricted "
            f"posture. Assisted autonomy only; constitutional constraints remain "
            f"active. Confidence {confidence:.2f}."
        )
    return (
        f"The court {RULING_APPROVED}. Constitution, certificate, committee, "
        f"and graduation layers are aligned with no material conflicts. "
        f"Confidence {confidence:.2f}."
    )


# -----------------------------------------------------------
# Precedent memory
# -----------------------------------------------------------
def _build_precedent_row(
    *,
    timestamp: str,
    case_id: str,
    ruling: str,
    evidence: Dict[str, Any],
    conflicts: List[Dict[str, Any]],
    reasons: List[str],
    confidence: float,
    override: bool,
) -> Dict[str, Any]:
    primary_conflict = conflicts[0]["conflict_type"] if conflicts else ""
    return {
        "timestamp_utc": timestamp,
        "case_id": case_id,
        "ruling": ruling,
        "violated_laws": "|".join(evidence.get("violated_laws") or []),
        "conflict_type": primary_conflict,
        "precedent_reason": " | ".join(reasons),
        "autonomy_state": evidence["graduation_state"],
        "certificate_state": evidence["certification_state"],
        "constitutional_state": evidence["constitution_state"],
        "judicial_confidence": round(confidence, 6),
        "conflict_count": len(conflicts),
        "constitutional_override_triggered": override,
    }


def _merge_precedent_memory(
    existing: List[Dict[str, Any]],
    new_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Append-only keyed by case_id."""
    keyed: Dict[str, Dict[str, Any]] = {}
    for r in existing:
        cid = str(r.get("case_id", ""))
        if cid:
            keyed[cid] = r
    keyed[str(new_row.get("case_id", ""))] = new_row
    out = list(keyed.values())
    for r in out:
        for c in PRECEDENT_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_court_ruling(
    *,
    const_summary: Dict[str, Any],
    const_record: Dict[str, Any],
    grad_summary: Dict[str, Any],
    committee: Dict[str, Any],
    cert_summary: Dict[str, Any],
    arm_summary: Dict[str, Any],
    readiness: Dict[str, Any],
    health: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    existing_precedent_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    evidence = _extract_evidence(
        const_summary=const_summary,
        const_record=const_record,
        grad_summary=grad_summary,
        committee=committee,
        cert_summary=cert_summary,
        arm_summary=arm_summary,
        readiness=readiness,
        health=health,
        runtime_policy=runtime_policy,
    )

    have_evidence = any(
        bool(x)
        for x in (
            const_summary,
            const_record,
            grad_summary,
            cert_summary,
            health,
        )
    )

    conflicts = _detect_conflicts(evidence) if have_evidence else []
    precedence = _precedence_stack(evidence)
    override = _constitutional_override_triggered(evidence, conflicts)

    ruling, reasons = _classify_ruling(
        evidence=evidence,
        conflicts=conflicts,
        have_evidence=have_evidence,
    )
    booleans = _court_booleans(ruling, evidence, override, conflicts)
    confidence = _judicial_confidence(evidence, conflicts)
    recommendations = _recommendations(ruling, evidence)
    cid = _case_id(timestamp, ruling)

    precedent_row = _build_precedent_row(
        timestamp=timestamp,
        case_id=cid,
        ruling=ruling,
        evidence=evidence,
        conflicts=conflicts,
        reasons=reasons,
        confidence=confidence,
        override=override,
    )
    merged_precedent = _merge_precedent_memory(existing_precedent_rows, precedent_row)

    md = _render_markdown(
        generated_at=timestamp,
        ruling=ruling,
        reasons=reasons,
        evidence=evidence,
        precedence=precedence,
        conflicts=conflicts,
        booleans=booleans,
        confidence=confidence,
        recommendations=recommendations,
        case_id=cid,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_constitutional_court_engine",
        "engine_version": 1,
        "case_id": cid,
        "judicial_ruling": ruling,
        "ruling_reasons": reasons,
        "judicial_confidence": round(confidence, 6),
        "evidence": evidence,
        "constitutional_precedence": precedence,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "court_booleans": booleans,
        "constitutional_override_triggered": booleans["constitutional_override_triggered"],
        "recommendations": recommendations,
        "precedent_memory_size_after_append": len(merged_precedent),
        "precedence_order": list(PRECEDENCE_LAYERS),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "judicial_only": True,
        },
        "inputs_seen": {
            "arm_autonomy_constitution_summary": bool(const_summary),
            "arm_autonomy_constitution_record": bool(const_record),
            "arm_autonomy_graduation_summary": bool(grad_summary),
            "autonomous_execution_risk_committee_summary": bool(committee),
            "autonomous_execution_certificate_summary": bool(cert_summary),
            "arm_mode_governance_summary": bool(arm_summary),
            "autonomous_readiness_summary": bool(readiness),
            "autonomous_system_health_summary": bool(health),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_precedent_memory_rows": len(existing_precedent_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_constitutional_court_engine",
        "case_id": cid,
        "judicial_ruling": ruling,
        "judicial_confidence": record["judicial_confidence"],
        "conflict_count": len(conflicts),
        "conflict_types": [c["conflict_type"] for c in conflicts],
        "constitutional_override_triggered": booleans["constitutional_override_triggered"],
        "autonomy_judicially_allowed": booleans["autonomy_judicially_allowed"],
        "execution_judicially_allowed": booleans["execution_judicially_allowed"],
        "operator_review_required": booleans["operator_review_required"],
        "lockdown_required": booleans["lockdown_required"],
        "precedent_created": booleans["precedent_created"],
        "constitution_state": evidence["constitution_state"],
        "graduation_state": evidence["graduation_state"],
        "certification_state": evidence["certification_state"],
        "committee_verdict": evidence["committee_verdict"],
        "arm_mode": evidence["arm_mode"],
        "n_recommendations": len(recommendations),
        "precedent_memory_size": len(merged_precedent),
    }

    return record, summary, md, merged_precedent


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM constitutional court engine (Step 33). "
            "Adjudicates governance conflicts; constitutional law "
            "supersedes autonomy. No broker calls; no portfolio mutation."
        ),
    )
    p.add_argument("--const-summary", default=str(DEFAULT_CONST_SUMMARY))
    p.add_argument("--const-record", default=str(DEFAULT_CONST_RECORD))
    p.add_argument("--grad-summary", default=str(DEFAULT_GRAD_SUMMARY))
    p.add_argument("--committee", default=str(DEFAULT_COMMITTEE))
    p.add_argument("--cert-summary", default=str(DEFAULT_CERT_SUMMARY))
    p.add_argument("--arm-summary", default=str(DEFAULT_ARM_SUMMARY))
    p.add_argument("--readiness", default=str(DEFAULT_READINESS))
    p.add_argument("--health", default=str(DEFAULT_HEALTH))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_COURT] starting (read-only constitutional court; no broker calls)",
        flush=True,
    )

    const_summary = _safe_read_json(
        Path(args.const_summary), label="arm_autonomy_constitution_summary.json"
    )
    const_record = _safe_read_json(Path(args.const_record), label="arm_autonomy_constitution.json")
    grad_summary = _safe_read_json(
        Path(args.grad_summary), label="arm_autonomy_graduation_summary.json"
    )
    committee = _safe_read_json(
        Path(args.committee), label="autonomous_execution_risk_committee_summary.json"
    )
    cert_summary = _safe_read_json(
        Path(args.cert_summary), label="autonomous_execution_certificate_summary.json"
    )
    arm_summary = _safe_read_json(Path(args.arm_summary), label="arm_mode_governance_summary.json")
    readiness = _safe_read_json(Path(args.readiness), label="autonomous_readiness_summary.json")
    health = _safe_read_json(Path(args.health), label="autonomous_system_health_summary.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_constitutional_precedent_memory.csv"
    )

    record, summary, md, merged_precedent = build_court_ruling(
        const_summary=const_summary,
        const_record=const_record,
        grad_summary=grad_summary,
        committee=committee,
        cert_summary=cert_summary,
        arm_summary=arm_summary,
        readiness=readiness,
        health=health,
        runtime_policy=runtime_policy,
        existing_precedent_rows=existing_mem,
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
        _atomic_write_csv(
            merged_precedent, Path(args.out_mem_csv), columns=PRECEDENT_MEMORY_COLUMNS
        )
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_precedent, Path(args.out_mem_parquet))

    booleans = record["court_booleans"]
    print(
        "[ARM_COURT] "
        f"ruling={record['judicial_ruling']} "
        f"override={booleans['constitutional_override_triggered']} "
        f"lockdown={booleans['lockdown_required']} "
        f"confidence={record['judicial_confidence']:.3f} "
        f"conflicts={record['conflict_count']}",
        flush=True,
    )
    print(
        "[ARM_COURT_SAFETY] broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_COURT_OUT] json={Path(args.out_json).as_posix()} "
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
            raise RuntimeError(f"[ARM_COURT_SAFETY] forbidden broker token detected: {tok!r}")


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
