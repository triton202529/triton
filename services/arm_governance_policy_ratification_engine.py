"""
ARM Governance Policy Ratification Engine -- Step 41.

Reads:
    data/results/arm_governance_policy_review_board_summary.json  (Step 40)
    data/results/arm_governance_policy_review_board.json          (Step 40)
    data/results/arm_governance_policy_review_memory.csv          (Step 40)
    data/results/arm_governance_policy_evolution_summary.json     (Step 39)
    data/results/arm_governance_learning_summary.json             (Step 38)
    data/results/arm_constitutional_court_summary.json              (Step 33)
    data/results/runtime_policy_governed.json                     (Step 18)
    data/results/autonomous_governance_scorecard.json               (Step 19)

Writes:
    data/results/arm_governance_policy_ratification.json
    data/results/arm_governance_policy_ratification.md
    data/results/arm_governance_policy_ratification_summary.json
    data/results/arm_governance_ratification_memory.csv
    data/results/arm_governance_ratification_memory.parquet

Purpose
-------
This engine answers:

    "Which governance policies become institutional doctrine?"

It converts institutionally approved governance policy proposals into formally
ratified governance doctrine. Approved != Ratified. Ratified != Activated.
Ratification NEVER mutates runtime policy. Constitutional law remains supreme.

Ratification state cascade
--------------------------
    1. POLICY_RATIFICATION_INSTITUTIONAL  mature repeatable doctrine formation
    2. POLICY_RATIFICATION_CONSERVATIVE   elevated pressure; defensive only
    3. POLICY_RATIFICATION_ACTIVE         policies ratified normally
    4. POLICY_RATIFICATION_FORMING        approved policies emerging; weak confidence
    5. POLICY_RATIFICATION_DORMANT        insufficient approved policies

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* runtime_mutation_allowed and activation_allowed are ALWAYS false.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only ratification memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to POLICY_RATIFICATION_DORMANT.
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_REVIEW_SUMMARY = RESULTS_DIR / "arm_governance_policy_review_board_summary.json"
DEFAULT_REVIEW_RECORD = RESULTS_DIR / "arm_governance_policy_review_board.json"
DEFAULT_REVIEW_MEM = RESULTS_DIR / "arm_governance_policy_review_memory.csv"
DEFAULT_EVOLUTION_SUM = RESULTS_DIR / "arm_governance_policy_evolution_summary.json"
DEFAULT_LEARNING_SUM = RESULTS_DIR / "arm_governance_learning_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_policy_ratification.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_policy_ratification.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_policy_ratification_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_ratification_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_ratification_memory.parquet"


# -----------------------------------------------------------
# Ratification state constants
# -----------------------------------------------------------
RATIFICATION_DORMANT = "POLICY_RATIFICATION_DORMANT"
RATIFICATION_FORMING = "POLICY_RATIFICATION_FORMING"
RATIFICATION_ACTIVE = "POLICY_RATIFICATION_ACTIVE"
RATIFICATION_CONSERVATIVE = "POLICY_RATIFICATION_CONSERVATIVE"
RATIFICATION_INSTITUTIONAL = "POLICY_RATIFICATION_INSTITUTIONAL"

DECISION_RATIFIED = "RATIFIED"
DECISION_RATIFIED_LIMITED = "RATIFIED_LIMITED"
DECISION_DEFERRED = "DEFERRED"
DECISION_REJECTED = "REJECTED"
DECISION_OPERATOR_RATIFICATION = "OPERATOR_RATIFICATION_REQUIRED"

BOARD_APPROVED = frozenset({"APPROVED", "APPROVED_LIMITED"})
BOARD_OPERATOR = frozenset({"OPERATOR_REVIEW_REQUIRED"})

DEFENSIVE_POLICIES = frozenset(
    {
        "confidence_threshold",
        "deployment_threshold",
        "target_cash_pct",
        "max_position_pct",
        "persistence_threshold",
        "min_observations_before_graduation",
        "autonomy_readiness_threshold",
        "skepticism_threshold",
        "auto_lock_manual_after_overruling",
        "governance_monitoring_frequency_multiplier",
    }
)

LOOSENING_POLICIES = frozenset(
    {
        "autonomy_loosen_threshold",
        "reduce_confidence_threshold",
        "reduce_target_cash_pct",
    }
)

HIGH_IMPACT_POLICIES = frozenset(
    {
        "confidence_threshold",
        "deployment_threshold",
        "autonomy_readiness_threshold",
        "min_observations_before_graduation",
        "auto_lock_manual_after_overruling",
        "skepticism_threshold",
        "persistence_threshold",
    }
)

RATIFICATION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "ratification_state",
    "ratified_count",
    "deferred_count",
    "rejected_count",
    "operator_required_count",
    "ratification_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_POLICY_RATIFICATION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(RATIFICATION_MEMORY_COLUMNS))
        for col in ("ratification_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in (
            "ratified_count",
            "deferred_count",
            "rejected_count",
            "operator_required_count",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        _warn(f"parquet write failed for {path}: {type(e).__name__}: {e}")
        return False


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


def _is_defensive_policy(name: str, current: Any, proposed: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return False
    if name not in DEFENSIVE_POLICIES:
        return False
    if name == "max_position_pct":
        c, p = _to_float(current), _to_float(proposed)
        return c is not None and p is not None and p < c
    if name == "auto_lock_manual_after_overruling":
        return proposed is True or str(proposed).lower() == "true"
    c, p = _to_float(current), _to_float(proposed)
    if c is not None and p is not None:
        return p >= c
    return True


def _is_aggressive_policy(name: str, current: Any, proposed: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return True
    if name == "max_position_pct":
        c, p = _to_float(current), _to_float(proposed)
        return c is not None and p is not None and p > c
    if name in (
        "confidence_threshold",
        "deployment_threshold",
        "target_cash_pct",
        "persistence_threshold",
        "skepticism_threshold",
        "autonomy_readiness_threshold",
    ):
        c, p = _to_float(current), _to_float(proposed)
        return c is not None and p is not None and p < c
    return False


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    review_summary: Dict[str, Any],
    review_record: Dict[str, Any],
    review_mem: List[Dict[str, str]],
    evolution_summary: Dict[str, Any],
    learning_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    mem_confidences = [
        _to_float(r.get("review_confidence"))
        for r in review_mem
        if _to_float(r.get("review_confidence")) is not None
    ]
    review_memory_depth = len(review_mem)
    review_memory_stability = 0.0
    if len(mem_confidences) >= 2:
        mean_c = sum(mem_confidences) / len(mem_confidences)
        var = sum((c - mean_c) ** 2 for c in mem_confidences) / len(mem_confidences)
        review_memory_stability = _clamp(1.0 - min(var * 4.0, 1.0), 0.0, 1.0)

    return {
        "review_state": _norm_upper(
            review_summary.get("review_state") or review_record.get("review_state")
        ),
        "board_confidence": _clamp(
            _to_float(review_summary.get("review_confidence"))
            or _to_float(review_record.get("review_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "learning_score": _clamp(
            _to_float(learning_summary.get("learning_score")) or 0.0, 0.0, 1.0
        ),
        "learning_state": _norm_upper(learning_summary.get("learning_state")),
        "evolution_state": _norm_upper(evolution_summary.get("evolution_state")),
        "evolution_score": _clamp(
            _to_float(evolution_summary.get("evolution_score")) or 0.0, 0.0, 1.0
        ),
        "constitutional_pressure": _clamp(
            _to_float(evolution_summary.get("constitutional_pressure"))
            or _to_float(learning_summary.get("constitutional_pressure"))
            or 0.0,
            0.0,
            1.0,
        ),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "recovery_effectiveness": _clamp(
            0.65 if learning_summary.get("recovery_lessons_detected") else 0.35,
            0.0,
            1.0,
        ),
        "court_ruling": _norm_upper(court_summary.get("judicial_ruling")),
        "constitution_violated": _norm_upper(court_summary.get("constitution_state"))
        == "CONSTITUTION_VIOLATED",
        "court_override": bool(court_summary.get("constitutional_override_triggered")),
        "operator_pressure": bool(
            court_summary.get("operator_review_required")
            or review_summary.get("operator_review_required")
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "reviewed_proposals": review_record.get("reviewed_proposals") or [],
        "review_memory_depth": review_memory_depth,
        "review_memory_stability": review_memory_stability,
    }


# -----------------------------------------------------------
# Ratification decisions
# -----------------------------------------------------------
def _ratify_proposal(
    proposal: Dict[str, Any],
    *,
    conservative: bool,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(proposal.get("policy_name", ""))
    board_decision = _norm_upper(proposal.get("decision"))
    conf = _to_float(proposal.get("confidence")) or 0.0
    const_safe = bool(proposal.get("constitutional_safe", True))
    op_flag = bool(proposal.get("operator_review_required"))
    current = proposal.get("current_value")
    proposed = proposal.get("proposed_value")
    reason = str(proposal.get("reason", ""))
    defensive = _is_defensive_policy(name, current, proposed)
    aggressive = _is_aggressive_policy(name, current, proposed)

    decision = DECISION_DEFERRED
    ratification_rationale = "insufficient institutional evidence for doctrine ratification"

    if board_decision in ("REJECTED",):
        decision = DECISION_REJECTED
        ratification_rationale = "rejected: board rejected proposal; ratification blocked"
    elif board_decision in BOARD_OPERATOR or op_flag:
        decision = DECISION_OPERATOR_RATIFICATION
        ratification_rationale = (
            "escalated to operator ratification due to high-impact governance doctrine sensitivity"
        )
    elif board_decision not in BOARD_APPROVED:
        decision = DECISION_DEFERRED
        ratification_rationale = "deferred: board has not approved this proposal for ratification"
    elif not const_safe:
        decision = DECISION_REJECTED
        ratification_rationale = "rejected due to constitutional safety concern"
    elif aggressive:
        decision = DECISION_REJECTED
        ratification_rationale = "rejected: doctrine would loosen governance under instability"
    elif conservative and not defensive:
        decision = DECISION_DEFERRED
        ratification_rationale = (
            "deferred: conservative ratification permits only defensive doctrine"
        )
    elif name in HIGH_IMPACT_POLICIES and (
        ctx["constitutional_pressure"] >= 0.55 or ctx["constitution_violated"]
    ):
        decision = DECISION_OPERATOR_RATIFICATION
        ratification_rationale = (
            "operator ratification required for high-impact doctrine under constitutional pressure"
        )
    elif board_decision == "APPROVED_LIMITED" or conf < 0.65:
        decision = DECISION_RATIFIED_LIMITED
        ratification_rationale = (
            "ratified with limited scope; gradual doctrine adoption recommended"
        )
    elif (
        board_decision == "APPROVED"
        and conf >= 0.70
        and defensive
        and ctx["board_confidence"] >= 0.35
    ):
        if conservative:
            decision = DECISION_RATIFIED_LIMITED
            ratification_rationale = (
                "ratified conservatively under elevated pressure; defensive doctrine only"
            )
        else:
            decision = DECISION_RATIFIED
            ratification_rationale = (
                f"ratified: {name} doctrine supports repeatable governance benefit "
                f"with confidence {conf:.2f}"
            )
    elif board_decision == "APPROVED" and conf >= 0.58 and const_safe:
        decision = DECISION_RATIFIED_LIMITED
        ratification_rationale = "ratified with limited institutional scope pending stability"
    else:
        decision = DECISION_DEFERRED
        ratification_rationale = "deferred pending stronger institutional evidence"

    # Autonomy-sensitive doctrine under violation always needs operator
    if (
        name
        in (
            "autonomy_readiness_threshold",
            "min_observations_before_graduation",
            "auto_lock_manual_after_overruling",
        )
        and ctx["constitution_violated"]
        and decision == DECISION_RATIFIED
    ):
        decision = DECISION_OPERATOR_RATIFICATION
        ratification_rationale = "operator ratification required for autonomy-sensitive doctrine under constitutional pressure"

    return {
        "policy_name": name,
        "ratification_decision": decision,
        "current_value": current,
        "proposed_value": proposed,
        "reason": reason,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "operator_ratification_required": (op_flag or decision == DECISION_OPERATOR_RATIFICATION),
        "ratification_rationale": ratification_rationale,
        "activation_allowed": False,
        "board_decision": board_decision,
    }


def _ratify_all(ctx: Dict[str, Any], conservative: bool) -> List[Dict[str, Any]]:
    return [
        _ratify_proposal(p, conservative=conservative, ctx=ctx) for p in ctx["reviewed_proposals"]
    ]


# -----------------------------------------------------------
# Ratification confidence and state
# -----------------------------------------------------------
def _ratification_confidence(
    ctx: Dict[str, Any],
    ratified: List[Dict[str, Any]],
) -> float:
    avg_conf = 0.0
    if ratified:
        avg_conf = sum(r["confidence"] for r in ratified) / len(ratified)

    const_safety = sum(1 for r in ratified if r["constitutional_safe"]) / max(len(ratified), 1)

    raw = (
        ctx["board_confidence"] * 0.25
        + ctx["learning_score"] * 0.20
        + const_safety * 0.15
        + ctx["governance_quality"] * 0.15
        + ctx["recovery_effectiveness"] * 0.10
        + avg_conf * 0.10
        + ctx["review_memory_stability"] * 0.05
    )

    penalty = ctx["constitutional_pressure"] * 0.18
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.07
    if ctx["court_override"]:
        penalty += 0.03
    rejected = sum(1 for r in ratified if r["ratification_decision"] == DECISION_REJECTED)
    if ratified and rejected / len(ratified) > 0.25:
        penalty += 0.05
    if ctx["board_confidence"] < 0.30:
        penalty += 0.05

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _approved_for_ratification(reviewed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in reviewed if _norm_upper(r.get("decision")) in BOARD_APPROVED]


def _classify_ratification_state(
    *,
    ctx: Dict[str, Any],
    ratification_confidence: float,
    ratified: List[Dict[str, Any]],
    approved_count: int,
) -> Tuple[str, List[str], bool]:
    reasons: List[str] = []

    if approved_count == 0:
        reasons.append("insufficient board-approved policies for ratification")
        return RATIFICATION_DORMANT, reasons, False

    conservative = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
        or ctx["review_state"] == "POLICY_REVIEW_CONSERVATIVE"
    )

    n_ratified = sum(
        1
        for r in ratified
        if r["ratification_decision"] in (DECISION_RATIFIED, DECISION_RATIFIED_LIMITED)
    )

    if (
        ctx["review_state"] == "POLICY_REVIEW_INSTITUTIONAL"
        and ratification_confidence >= 0.68
        and n_ratified >= 2
        and ctx["review_memory_depth"] >= 3
    ):
        reasons.append("mature governance doctrine formation with repeatable ratification quality")
        return RATIFICATION_INSTITUTIONAL, reasons, conservative

    if conservative:
        reasons.append("constitutional pressure elevated; only defensive ratification permitted")
        return RATIFICATION_CONSERVATIVE, reasons, True

    if ratification_confidence < 0.35 or approved_count < 2:
        reasons.append("approved policies emerging; institutional confidence weak")
        return RATIFICATION_FORMING, reasons, False

    if ratification_confidence >= 0.50 and n_ratified >= 1:
        reasons.append("policies ratified under normal institutional doctrine process")
        return RATIFICATION_ACTIVE, reasons, False

    reasons.append("ratification process forming institutional doctrine posture")
    return RATIFICATION_FORMING, reasons, conservative


# -----------------------------------------------------------
# Booleans, recommendations, rationale, doctrine
# -----------------------------------------------------------
def _ratification_booleans(
    state: str,
    ratified: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, bool]:
    doctrine = [
        r
        for r in ratified
        if r["ratification_decision"] in (DECISION_RATIFIED, DECISION_RATIFIED_LIMITED)
    ]
    op_required = any(
        r["ratification_decision"] == DECISION_OPERATOR_RATIFICATION
        or r["operator_ratification_required"]
        for r in ratified
    )
    return {
        "ratified_policy_available": len(doctrine) > 0,
        "institutional_doctrine_forming": state
        in (
            RATIFICATION_FORMING,
            RATIFICATION_ACTIVE,
            RATIFICATION_INSTITUTIONAL,
        ),
        "operator_ratification_required": op_required or ctx["operator_pressure"],
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "activation_allowed": False,
        "ratification_memory_reliable": state == RATIFICATION_INSTITUTIONAL,
    }


def _governance_doctrine(ratified: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doctrine: List[Dict[str, Any]] = []
    for r in ratified:
        if r["ratification_decision"] not in (DECISION_RATIFIED, DECISION_RATIFIED_LIMITED):
            continue
        doctrine.append(
            {
                "policy_name": r["policy_name"],
                "doctrine_value": r["proposed_value"],
                "prior_value": r["current_value"],
                "ratification_scope": r["ratification_decision"],
                "confidence": r["confidence"],
                "activation_allowed": False,
                "ratification_rationale": r["ratification_rationale"],
            }
        )
    return doctrine


def _recommendations(ratified: List[Dict[str, Any]], state: str) -> List[str]:
    recs: List[str] = []
    by_name = {r["policy_name"]: r for r in ratified}

    if by_name.get("target_cash_pct", {}).get("ratification_decision") in (
        DECISION_RATIFIED,
        DECISION_RATIFIED_LIMITED,
    ):
        recs.append("Maintain elevated cash doctrine")
    if by_name.get("auto_lock_manual_after_overruling", {}).get("ratification_decision") in (
        DECISION_RATIFIED,
        DECISION_RATIFIED_LIMITED,
        DECISION_OPERATOR_RATIFICATION,
    ):
        recs.append("Continue manual lock doctrine")
    if any(
        r["ratification_decision"] == DECISION_DEFERRED
        for r in ratified
        if _norm_upper(r.get("board_decision", "")) not in ("REJECTED",)
    ):
        recs.append("Defer autonomy loosening doctrine")
    if any(r["ratification_decision"] == DECISION_OPERATOR_RATIFICATION for r in ratified):
        recs.append("Escalate sensitive governance doctrine to operator ratification")
    if state in (RATIFICATION_FORMING, RATIFICATION_ACTIVE, RATIFICATION_INSTITUTIONAL):
        recs.append("Continue governance learning accumulation")
    if state == RATIFICATION_CONSERVATIVE:
        recs.append("Restrict ratification to defensive governance doctrine only")
    if not recs:
        recs.append("Continue accumulating board approvals before doctrine ratification")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(ratified: List[Dict[str, Any]], state: str) -> str:
    doctrine = [
        r
        for r in ratified
        if r["ratification_decision"] in (DECISION_RATIFIED, DECISION_RATIFIED_LIMITED)
    ]
    if any(r["policy_name"] == "target_cash_pct" for r in doctrine):
        return (
            "Triton ratified elevated cash posture doctrine because repeated instability "
            "improved under defensive capital preservation."
        )
    if doctrine:
        names = ", ".join(r["policy_name"] for r in doctrine[:3])
        return f"Triton ratified institutional governance doctrine for: {names}."
    if state == RATIFICATION_CONSERVATIVE:
        return (
            "Ratification operates in conservative mode due to elevated constitutional "
            "pressure; only defensive doctrine may be formalized."
        )
    deferred = sum(1 for r in ratified if r["ratification_decision"] == DECISION_DEFERRED)
    if deferred:
        return (
            f"Ratification deferred {deferred} proposal(s) pending stronger institutional evidence."
        )
    return "Ratification completed evaluation of board-approved governance policies."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    ratification_confidence: float,
    ratified: List[Dict[str, Any]],
    doctrine: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Policy Ratification",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Ratification State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| ratification_confidence | {ratification_confidence:.3f} |",
        f"| ratified | {counts['ratified']} |",
        f"| deferred | {counts['deferred']} |",
        f"| rejected | {counts['rejected']} |",
        f"| operator_ratification | {counts['operator_required']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        f"| activation_allowed | {booleans['activation_allowed']} |",
        "",
        "## Ratified Policies",
        "",
    ]
    ratified_policies = [
        r
        for r in ratified
        if r["ratification_decision"] in (DECISION_RATIFIED, DECISION_RATIFIED_LIMITED)
    ]
    if ratified_policies:
        lines.append("| policy | decision | confidence | activation_allowed |")
        lines.append("|---|---|---|---|")
        for r in ratified_policies:
            lines.append(
                f"| {r['policy_name']} | {r['ratification_decision']} | {r['confidence']:.2f} | "
                f"{r['activation_allowed']} |"
            )
        lines.append("")
        for r in ratified_policies:
            lines.append(
                f"- **{r['policy_name']}** ({r['ratification_decision']}): "
                f"{r['ratification_rationale']}"
            )
    else:
        lines.append("_No policies ratified this cycle._")

    deferred = [r for r in ratified if r["ratification_decision"] == DECISION_DEFERRED]
    lines.extend(["", "## Deferred Policies", ""])
    if deferred:
        for r in deferred:
            lines.append(f"- {r['policy_name']}: {r['ratification_rationale']}")
    else:
        lines.append("_None deferred._")

    lines.extend(["", "## Governance Doctrine", ""])
    if doctrine:
        for d in doctrine:
            lines.append(
                f"- **{d['policy_name']}** ({d['ratification_scope']}): "
                f"{d['prior_value']} → {d['doctrine_value']} "
                f"(activation_allowed={d['activation_allowed']})"
            )
    else:
        lines.append("_No institutional doctrine formalized this cycle._")

    lines.extend(["", "## Recommendations", ""])
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.extend(["", "## Why", ""])
    for r in reasons:
        lines.append(f"- {r}")
    lines.extend(
        [
            "",
            "## Narrative",
            "",
            rationale,
            "",
            "Ratification formalizes doctrine only. Approved ≠ Ratified. Ratified ≠ Activated. "
            "This engine never mutates runtime policy. Constitutional law, court rulings, "
            "capital preservation doctrine, and operator supremacy remain supreme.",
            "",
        ]
    )
    return "\n".join(lines)


def _count_decisions(ratified: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "ratified": sum(
            1
            for r in ratified
            if r["ratification_decision"] in (DECISION_RATIFIED, DECISION_RATIFIED_LIMITED)
        ),
        "deferred": sum(1 for r in ratified if r["ratification_decision"] == DECISION_DEFERRED),
        "rejected": sum(1 for r in ratified if r["ratification_decision"] == DECISION_REJECTED),
        "operator_required": sum(
            1 for r in ratified if r["ratification_decision"] == DECISION_OPERATOR_RATIFICATION
        ),
    }


# -----------------------------------------------------------
# Ratification memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    ratification_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "ratification_state": state,
        "ratified_count": counts["ratified"],
        "deferred_count": counts["deferred"],
        "rejected_count": counts["rejected"],
        "operator_required_count": counts["operator_required"],
        "ratification_confidence": round(ratification_confidence, 6),
        "rationale": rationale,
    }


def _merge_memory(
    existing: List[Dict[str, Any]],
    new_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    keyed: Dict[str, Dict[str, Any]] = {}
    for r in existing:
        ts = str(r.get("timestamp", ""))
        if ts:
            keyed[ts] = r
    keyed[str(new_row.get("timestamp", ""))] = new_row
    out = list(keyed.values())
    for r in out:
        for c in RATIFICATION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_policy_ratification(
    *,
    review_summary: Dict[str, Any],
    review_record: Dict[str, Any],
    review_mem: List[Dict[str, str]],
    evolution_summary: Dict[str, Any],
    learning_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    existing_ratification_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        review_summary=review_summary,
        review_record=review_record,
        review_mem=review_mem,
        evolution_summary=evolution_summary,
        learning_summary=learning_summary,
        court_summary=court_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
    )

    approved_count = len(_approved_for_ratification(ctx["reviewed_proposals"]))

    pre_conservative = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
    )
    ratified = _ratify_all(ctx, conservative=pre_conservative)
    ratification_confidence = _ratification_confidence(ctx, ratified)

    state, reasons, conservative = _classify_ratification_state(
        ctx=ctx,
        ratification_confidence=ratification_confidence,
        ratified=ratified,
        approved_count=approved_count,
    )

    if conservative != pre_conservative:
        ratified = _ratify_all(ctx, conservative=conservative)
        ratification_confidence = _ratification_confidence(ctx, ratified)

    counts = _count_decisions(ratified)
    booleans = _ratification_booleans(state, ratified, ctx)
    doctrine = _governance_doctrine(ratified)
    recommendations = _recommendations(ratified, state)
    rationale = _build_rationale(ratified, state)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        ratification_confidence=ratification_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_ratification_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        ratification_confidence=ratification_confidence,
        ratified=ratified,
        doctrine=doctrine,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_policy_ratification_engine",
        "engine_version": 1,
        "ratification_state": state,
        "ratification_confidence": ratification_confidence,
        "ratification_reasons": reasons,
        "ratified_policies": ratified,
        "governance_doctrine": doctrine,
        "decision_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "approved_vs_ratified_note": (
            "Approved ≠ Ratified. Board approval is necessary but not sufficient for doctrine. "
            "Ratified ≠ Activated. Ratified doctrine is not applied to runtime by this engine."
        ),
        "constitutional_supremacy_note": (
            "Ratification formalizes doctrine only. It NEVER mutates runtime policy. "
            "Ratification cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "ratification_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "activation_allowed": False,
            "ratification_only": True,
        },
        "inputs_seen": {
            "arm_governance_policy_review_board_summary": bool(review_summary),
            "arm_governance_policy_review_board_record": bool(review_record),
            "arm_governance_policy_review_memory_rows": len(review_mem),
            "arm_governance_policy_evolution_summary": bool(evolution_summary),
            "arm_governance_learning_summary": bool(learning_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_governance_scorecard": bool(scorecard),
            "existing_ratification_memory_rows": len(existing_ratification_mem),
            "board_approved_count": approved_count,
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_policy_ratification_engine",
        "ratification_state": state,
        "ratification_confidence": ratification_confidence,
        "ratified_policy_available": booleans["ratified_policy_available"],
        "institutional_doctrine_forming": booleans["institutional_doctrine_forming"],
        "operator_ratification_required": booleans["operator_ratification_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "activation_allowed": booleans["activation_allowed"],
        "ratification_memory_reliable": booleans["ratification_memory_reliable"],
        "ratified_count": counts["ratified"],
        "deferred_count": counts["deferred"],
        "rejected_count": counts["rejected"],
        "operator_required_count": counts["operator_required"],
        "n_policies_evaluated": len(ratified),
        "n_doctrine_entries": len(doctrine),
        "n_recommendations": len(recommendations),
        "ratification_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance policy ratification engine (Step 41). "
            "Converts board-approved policies into institutional doctrine. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--review-summary", default=str(DEFAULT_REVIEW_SUMMARY))
    p.add_argument("--review-record", default=str(DEFAULT_REVIEW_RECORD))
    p.add_argument("--review-mem", default=str(DEFAULT_REVIEW_MEM))
    p.add_argument("--evolution-summary", default=str(DEFAULT_EVOLUTION_SUM))
    p.add_argument("--learning-summary", default=str(DEFAULT_LEARNING_SUM))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_POLICY_RATIFICATION] starting "
        "(read-only governance doctrine ratification; no runtime mutation; no broker calls)",
        flush=True,
    )

    review_summary = _safe_read_json(
        Path(args.review_summary), label="arm_governance_policy_review_board_summary.json"
    )
    review_record = _safe_read_json(
        Path(args.review_record), label="arm_governance_policy_review_board.json"
    )
    review_mem = _safe_read_csv_rows(
        Path(args.review_mem), label="arm_governance_policy_review_memory.csv"
    )
    evolution_summary = _safe_read_json(
        Path(args.evolution_summary), label="arm_governance_policy_evolution_summary.json"
    )
    learning_summary = _safe_read_json(
        Path(args.learning_summary), label="arm_governance_learning_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_ratification_memory.csv"
    )

    record, summary, md, merged_memory = build_policy_ratification(
        review_summary=review_summary,
        review_record=review_record,
        review_mem=review_mem,
        evolution_summary=evolution_summary,
        learning_summary=learning_summary,
        court_summary=court_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        existing_ratification_mem=existing_mem,
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
            merged_memory, Path(args.out_mem_csv), columns=RATIFICATION_MEMORY_COLUMNS
        )
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["decision_counts"]
    print(
        "[ARM_POLICY_RATIFICATION] "
        f"state={record['ratification_state']} "
        f"ratified={counts['ratified']} "
        f"deferred={counts['deferred']} "
        f"rejected={counts['rejected']} "
        f"confidence={record['ratification_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_POLICY_RATIFICATION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False "
        "runtime_mutation=False activation=False",
        flush=True,
    )
    print(
        f"[ARM_POLICY_RATIFICATION_OUT] json={Path(args.out_json).as_posix()} "
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
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return
    for tok in _FORBIDDEN_TOKENS:
        if tok in src:
            raise RuntimeError(
                f"[ARM_POLICY_RATIFICATION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
