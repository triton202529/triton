"""
ARM Governance Policy Review Board Engine -- Step 40.

Reads:
    data/results/arm_governance_policy_evolution_summary.json  (Step 39)
    data/results/arm_governance_policy_evolution.json          (Step 39)
    data/results/arm_governance_policy_evolution_memory.csv    (Step 39)
    data/results/arm_governance_learning_summary.json          (Step 38)
    data/results/arm_constitutional_court_summary.json         (Step 33)
    data/results/arm_supreme_governance_council_summary.json   (Step 34)
    data/results/runtime_policy_governed.json                  (Step 18)
    data/results/autonomous_governance_scorecard.json            (Step 19)

Writes:
    data/results/arm_governance_policy_review_board.json
    data/results/arm_governance_policy_review_board.md
    data/results/arm_governance_policy_review_board_summary.json
    data/results/arm_governance_policy_review_memory.csv
    data/results/arm_governance_policy_review_memory.parquet

Purpose
-------
This engine answers:

    "Which governance policy proposals should be approved?"

It is Triton's governance committee -- read-only evaluation of policy evolution
proposals for institutional approval. The board NEVER mutates runtime policy.
Constitutional law remains supreme.

Review board state cascade
--------------------------
    1. POLICY_REVIEW_INSTITUTIONAL  mature repeatable approval quality
    2. POLICY_REVIEW_CONSERVATIVE   elevated pressure; defensive only
    3. POLICY_REVIEW_ACTIVE         proposals evaluated normally
    4. POLICY_REVIEW_FORMING        proposals emerging; weak confidence
    5. POLICY_REVIEW_DORMANT        insufficient proposals

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* runtime_mutation_allowed is ALWAYS false.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only review memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to POLICY_REVIEW_DORMANT.
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

DEFAULT_EVOLUTION_SUMMARY = RESULTS_DIR / "arm_governance_policy_evolution_summary.json"
DEFAULT_EVOLUTION_RECORD = RESULTS_DIR / "arm_governance_policy_evolution.json"
DEFAULT_EVOLUTION_MEM = RESULTS_DIR / "arm_governance_policy_evolution_memory.csv"
DEFAULT_LEARNING_SUMMARY = RESULTS_DIR / "arm_governance_learning_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_policy_review_board.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_policy_review_board.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_policy_review_board_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_policy_review_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_policy_review_memory.parquet"


# -----------------------------------------------------------
# Review state constants
# -----------------------------------------------------------
REVIEW_DORMANT = "POLICY_REVIEW_DORMANT"
REVIEW_FORMING = "POLICY_REVIEW_FORMING"
REVIEW_ACTIVE = "POLICY_REVIEW_ACTIVE"
REVIEW_CONSERVATIVE = "POLICY_REVIEW_CONSERVATIVE"
REVIEW_INSTITUTIONAL = "POLICY_REVIEW_INSTITUTIONAL"

DECISION_APPROVED = "APPROVED"
DECISION_APPROVED_LIMITED = "APPROVED_LIMITED"
DECISION_DEFERRED = "DEFERRED"
DECISION_REJECTED = "REJECTED"
DECISION_OPERATOR_REVIEW = "OPERATOR_REVIEW_REQUIRED"

# Defensive proposals permitted under conservative review
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

REVIEW_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "review_state",
    "approved_count",
    "deferred_count",
    "rejected_count",
    "operator_review_count",
    "review_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_POLICY_REVIEW_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(REVIEW_MEMORY_COLUMNS))
        for col in ("review_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("approved_count", "deferred_count", "rejected_count", "operator_review_count"):
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


def _is_defensive_proposal(name: str, delta: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return False
    if name not in DEFENSIVE_POLICIES:
        return False
    if name == "max_position_pct":
        d = _to_float(delta)
        return d is not None and d < 0
    if name == "auto_lock_manual_after_overruling":
        return delta is True or str(delta).lower() == "true"
    d = _to_float(delta)
    if d is not None:
        return d >= 0
    return True


def _is_aggressive_proposal(name: str, delta: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return True
    if name == "max_position_pct":
        d = _to_float(delta)
        return d is not None and d > 0
    if name in (
        "confidence_threshold",
        "deployment_threshold",
        "target_cash_pct",
        "persistence_threshold",
        "skepticism_threshold",
        "autonomy_readiness_threshold",
    ):
        d = _to_float(delta)
        return d is not None and d < 0
    return False


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    evolution_summary: Dict[str, Any],
    evolution_record: Dict[str, Any],
    learning_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    return {
        "evolution_state": _norm_upper(
            evolution_summary.get("evolution_state") or evolution_record.get("evolution_state")
        ),
        "evolution_score": _to_float(
            evolution_summary.get("evolution_score") or evolution_record.get("evolution_score")
        )
        or 0.0,
        "learning_score": _to_float(learning_summary.get("learning_score")) or 0.0,
        "learning_state": _norm_upper(learning_summary.get("learning_state")),
        "governance_confidence": _clamp(
            _to_float(evolution_summary.get("governance_confidence")) or 0.0,
            0.0,
            1.0,
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
        "court_ruling": _norm_upper(court_summary.get("judicial_ruling")),
        "constitution_state": _norm_upper(
            court_summary.get("constitution_state") or council_summary.get("constitution_state")
        ),
        "constitution_violated": _norm_upper(
            court_summary.get("constitution_state") or council_summary.get("constitution_state")
        )
        == "CONSTITUTION_VIOLATED",
        "court_override": bool(court_summary.get("constitutional_override_triggered")),
        "council_ruling": _norm_upper(council_summary.get("governance_ruling")),
        "operator_pressure": bool(
            court_summary.get("operator_review_required")
            or council_summary.get("operator_supervision_required")
            or evolution_summary.get("operator_review_required")
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "proposals": evolution_record.get("policy_evolution_proposals") or [],
    }


# -----------------------------------------------------------
# Proposal review
# -----------------------------------------------------------
def _review_proposal(
    proposal: Dict[str, Any],
    *,
    conservative: bool,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(proposal.get("policy_name", ""))
    conf = _to_float(proposal.get("confidence")) or 0.0
    const_safe = bool(proposal.get("constitutional_safe", True))
    op_flag = bool(proposal.get("operator_review_required"))
    delta = proposal.get("delta")
    reason = str(proposal.get("reason", ""))
    defensive = _is_defensive_proposal(name, delta)
    aggressive = _is_aggressive_proposal(name, delta)

    decision = DECISION_DEFERRED
    board_rationale = "insufficient evidence for board approval at this time"

    if not const_safe:
        decision = DECISION_REJECTED
        board_rationale = "rejected due to constitutional safety concern"
    elif aggressive:
        decision = DECISION_REJECTED
        board_rationale = "rejected: proposal loosens governance under elevated instability"
    elif conservative and not defensive:
        decision = DECISION_DEFERRED
        board_rationale = "deferred: conservative board permits only defensive policy changes"
    elif conf < 0.50:
        decision = DECISION_DEFERRED
        board_rationale = "deferred due to low proposal confidence"
    elif op_flag and (ctx["constitutional_pressure"] >= 0.55 or ctx["operator_pressure"]):
        decision = DECISION_OPERATOR_REVIEW
        board_rationale = "escalated to operator review due to high-impact governance sensitivity"
    elif conf >= 0.70 and const_safe and defensive:
        decision = DECISION_APPROVED
        board_rationale = (
            f"approved: {name} supports governance stability with confidence {conf:.2f}"
        )
    elif conf >= 0.55 and const_safe:
        decision = DECISION_APPROVED_LIMITED
        board_rationale = "approved with limited scope; gradual adoption recommended"
    else:
        decision = DECISION_DEFERRED
        board_rationale = "deferred pending additional institutional evidence"

    # High-impact autonomy changes always need operator under pressure
    if (
        name
        in (
            "autonomy_readiness_threshold",
            "min_observations_before_graduation",
            "auto_lock_manual_after_overruling",
        )
        and ctx["constitution_violated"]
        and decision == DECISION_APPROVED
    ):
        decision = DECISION_OPERATOR_REVIEW
        board_rationale = (
            "operator review required for autonomy-sensitive policy under constitutional pressure"
        )

    return {
        "policy_name": name,
        "decision": decision,
        "current_value": proposal.get("current_value"),
        "proposed_value": proposal.get("proposed_value"),
        "reason": reason,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "operator_review_required": op_flag or decision == DECISION_OPERATOR_REVIEW,
        "board_rationale": board_rationale,
    }


def _review_all_proposals(ctx: Dict[str, Any], conservative: bool) -> List[Dict[str, Any]]:
    return [_review_proposal(p, conservative=conservative, ctx=ctx) for p in ctx["proposals"]]


# -----------------------------------------------------------
# Review confidence and board state
# -----------------------------------------------------------
def _review_confidence(
    ctx: Dict[str, Any],
    reviewed: List[Dict[str, Any]],
) -> float:
    avg_prop_conf = 0.0
    if reviewed:
        avg_prop_conf = sum(r["confidence"] for r in reviewed) / len(reviewed)

    const_safety = sum(1 for r in reviewed if r["constitutional_safe"]) / max(len(reviewed), 1)

    raw = (
        ctx["learning_score"] * 0.25
        + avg_prop_conf * 0.25
        + const_safety * 0.15
        + ctx["governance_quality"] * 0.15
        + ctx["evolution_score"] * 0.10
        + (1.0 - ctx["constitutional_pressure"]) * 0.10
    )

    penalty = ctx["constitutional_pressure"] * 0.15
    if ctx["constitution_violated"]:
        penalty += 0.08
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.05
    rejected = sum(1 for r in reviewed if r["decision"] == DECISION_REJECTED)
    if reviewed and rejected / len(reviewed) > 0.3:
        penalty += 0.05

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _classify_review_state(
    *,
    ctx: Dict[str, Any],
    review_confidence: float,
    reviewed: List[Dict[str, Any]],
    have_evidence: bool,
) -> Tuple[str, List[str], bool]:
    """Returns (state, reasons, conservative_mode)."""
    reasons: List[str] = []
    n = len(reviewed)

    if not have_evidence or n == 0:
        reasons.append("insufficient policy evolution proposals for review")
        return REVIEW_DORMANT, reasons, False

    conservative = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
        or ctx["council_ruling"] in ("GOVERNANCE_REVOKE_AUTONOMY", "GOVERNANCE_LOCKDOWN")
    )

    if ctx["evolution_state"] == "POLICY_EVOLUTION_INSTITUTIONAL" and review_confidence >= 0.70:
        reasons.append("mature policy review process with repeatable approval quality")
        return REVIEW_INSTITUTIONAL, reasons, conservative

    if conservative:
        reasons.append("constitutional pressure elevated; only defensive changes permitted")
        return REVIEW_CONSERVATIVE, reasons, True

    if review_confidence < 0.35 or ctx["evolution_state"] == "POLICY_EVOLUTION_DORMANT":
        reasons.append("proposals emerging; review confidence weak")
        return REVIEW_FORMING, reasons, False

    if review_confidence >= 0.55 and n >= 2:
        reasons.append("proposals evaluated under normal board review")
        return REVIEW_ACTIVE, reasons, False

    reasons.append("review board forming institutional evaluation posture")
    return REVIEW_FORMING, reasons, conservative


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _board_booleans(
    state: str,
    reviewed: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, bool]:
    approved = [
        r for r in reviewed if r["decision"] in (DECISION_APPROVED, DECISION_APPROVED_LIMITED)
    ]
    op_required = any(
        r["decision"] == DECISION_OPERATOR_REVIEW or r["operator_review_required"] for r in reviewed
    )
    return {
        "approved_policy_changes_available": len(approved) > 0,
        "operator_review_required": op_required or ctx["operator_pressure"],
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "governance_policy_escalation_required": (
            state == REVIEW_CONSERVATIVE
            or any(r["decision"] == DECISION_OPERATOR_REVIEW for r in reviewed)
        ),
        "runtime_mutation_allowed": False,
        "policy_review_memory_reliable": state == REVIEW_INSTITUTIONAL,
    }


def _recommendations(reviewed: List[Dict[str, Any]], state: str) -> List[str]:
    recs: List[str] = []
    by_name = {r["policy_name"]: r for r in reviewed}

    if by_name.get("confidence_threshold", {}).get("decision") in (
        DECISION_APPROVED,
        DECISION_APPROVED_LIMITED,
    ):
        recs.append("Gradually raise confidence threshold")
    if by_name.get("target_cash_pct", {}).get("decision") in (
        DECISION_APPROVED,
        DECISION_APPROVED_LIMITED,
    ):
        recs.append("Preserve elevated cash posture")
    if by_name.get("auto_lock_manual_after_overruling", {}).get("decision") in (
        DECISION_APPROVED,
        DECISION_APPROVED_LIMITED,
        DECISION_OPERATOR_REVIEW,
    ):
        recs.append("Maintain manual lock after repeated overruling")
    if any(r["decision"] == DECISION_DEFERRED for r in reviewed):
        recs.append("Defer autonomy loosening until governance stabilizes")
    if any(r["decision"] == DECISION_OPERATOR_REVIEW for r in reviewed):
        recs.append("Escalate sensitive policy proposals to operator review")
    if state == REVIEW_CONSERVATIVE:
        recs.append("Restrict approvals to defensive governance adjustments only")
    if not recs:
        recs.append("Continue accumulating proposals before board action")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(reviewed: List[Dict[str, Any]], state: str) -> str:
    approved = [r for r in reviewed if r["decision"] == DECISION_APPROVED]
    if approved and approved[0]["policy_name"] == "confidence_threshold":
        return (
            "The board approved tighter confidence thresholds because repeated "
            "governance instability emerged during low-confidence conditions."
        )
    if approved:
        names = ", ".join(r["policy_name"] for r in approved[:3])
        return f"The board approved defensive policy adjustments: {names}."
    if state == REVIEW_CONSERVATIVE:
        return (
            "The board operates in conservative mode due to elevated constitutional "
            "pressure; only defensive governance changes may proceed."
        )
    deferred = sum(1 for r in reviewed if r["decision"] == DECISION_DEFERRED)
    if deferred:
        return f"The board deferred {deferred} proposal(s) pending stronger institutional evidence."
    return "The review board completed evaluation of governance policy evolution proposals."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    review_confidence: float,
    reviewed: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Policy Review Board",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Review State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| review_confidence | {review_confidence:.3f} |",
        f"| approved | {counts['approved']} |",
        f"| deferred | {counts['deferred']} |",
        f"| rejected | {counts['rejected']} |",
        f"| operator_review | {counts['operator_review']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Proposal Decisions",
        "",
    ]
    if reviewed:
        lines.append("| policy | decision | confidence | operator_review |")
        lines.append("|---|---|---|---|")
        for r in reviewed:
            lines.append(
                f"| {r['policy_name']} | {r['decision']} | {r['confidence']:.2f} | "
                f"{r['operator_review_required']} |"
            )
        lines.append("")
        for r in reviewed:
            lines.append(f"- **{r['policy_name']}** ({r['decision']}): {r['board_rationale']}")
    else:
        lines.append("_No proposals to review._")

    approved = [
        r for r in reviewed if r["decision"] in (DECISION_APPROVED, DECISION_APPROVED_LIMITED)
    ]
    deferred = [r for r in reviewed if r["decision"] == DECISION_DEFERRED]
    lines.extend(["", "## Approved Policies", ""])
    if approved:
        for r in approved:
            lines.append(
                f"- {r['policy_name']}: {r['current_value']} → {r['proposed_value']} ({r['decision']})"
            )
    else:
        lines.append("_None approved this cycle._")

    lines.extend(["", "## Deferred Policies", ""])
    if deferred:
        for r in deferred:
            lines.append(f"- {r['policy_name']}: {r['board_rationale']}")
    else:
        lines.append("_None deferred._")

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
            "The review board evaluates proposals only. It never mutates runtime policy. "
            "All approved changes remain subject to constitutional law, court rulings, "
            "capital preservation doctrine, and operator supremacy.",
            "",
        ]
    )
    return "\n".join(lines)


def _count_decisions(reviewed: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "approved": sum(
            1 for r in reviewed if r["decision"] in (DECISION_APPROVED, DECISION_APPROVED_LIMITED)
        ),
        "deferred": sum(1 for r in reviewed if r["decision"] == DECISION_DEFERRED),
        "rejected": sum(1 for r in reviewed if r["decision"] == DECISION_REJECTED),
        "operator_review": sum(1 for r in reviewed if r["decision"] == DECISION_OPERATOR_REVIEW),
    }


# -----------------------------------------------------------
# Review memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    review_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "review_state": state,
        "approved_count": counts["approved"],
        "deferred_count": counts["deferred"],
        "rejected_count": counts["rejected"],
        "operator_review_count": counts["operator_review"],
        "review_confidence": round(review_confidence, 6),
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
        for c in REVIEW_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_policy_review(
    *,
    evolution_summary: Dict[str, Any],
    evolution_record: Dict[str, Any],
    evolution_mem: List[Dict[str, str]],
    learning_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    existing_review_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        evolution_summary=evolution_summary,
        evolution_record=evolution_record,
        learning_summary=learning_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
    )

    have_evidence = bool(evolution_record or evolution_summary)

    # Preliminary conservative flag for first-pass review
    pre_conservative = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
    )
    reviewed = _review_all_proposals(ctx, conservative=pre_conservative)
    review_confidence = _review_confidence(ctx, reviewed)

    state, reasons, conservative = _classify_review_state(
        ctx=ctx,
        review_confidence=review_confidence,
        reviewed=reviewed,
        have_evidence=have_evidence,
    )

    # Re-review under final conservative determination if changed
    if conservative != pre_conservative:
        reviewed = _review_all_proposals(ctx, conservative=conservative)
        review_confidence = _review_confidence(ctx, reviewed)

    counts = _count_decisions(reviewed)
    booleans = _board_booleans(state, reviewed, ctx)
    recommendations = _recommendations(reviewed, state)
    rationale = _build_rationale(reviewed, state)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        review_confidence=review_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_review_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        review_confidence=review_confidence,
        reviewed=reviewed,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_policy_review_board_engine",
        "engine_version": 1,
        "review_state": state,
        "review_confidence": review_confidence,
        "review_reasons": reasons,
        "reviewed_proposals": reviewed,
        "decision_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "constitutional_supremacy_note": (
            "The review board evaluates proposals only. It NEVER mutates runtime policy. "
            "Board decisions cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "review_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "review_board_only": True,
        },
        "inputs_seen": {
            "arm_governance_policy_evolution_summary": bool(evolution_summary),
            "arm_governance_policy_evolution_record": bool(evolution_record),
            "arm_governance_policy_evolution_memory_rows": len(evolution_mem),
            "arm_governance_learning_summary": bool(learning_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_governance_scorecard": bool(scorecard),
            "existing_review_memory_rows": len(existing_review_mem),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_policy_review_board_engine",
        "review_state": state,
        "review_confidence": review_confidence,
        "approved_policy_changes_available": booleans["approved_policy_changes_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "governance_policy_escalation_required": booleans["governance_policy_escalation_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "policy_review_memory_reliable": booleans["policy_review_memory_reliable"],
        "approved_count": counts["approved"],
        "deferred_count": counts["deferred"],
        "rejected_count": counts["rejected"],
        "operator_review_count": counts["operator_review"],
        "n_proposals_reviewed": len(reviewed),
        "n_recommendations": len(recommendations),
        "review_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance policy review board engine (Step 40). "
            "Evaluates policy evolution proposals for approval. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--evolution-summary", default=str(DEFAULT_EVOLUTION_SUMMARY))
    p.add_argument("--evolution-record", default=str(DEFAULT_EVOLUTION_RECORD))
    p.add_argument("--evolution-mem", default=str(DEFAULT_EVOLUTION_MEM))
    p.add_argument("--learning-summary", default=str(DEFAULT_LEARNING_SUMMARY))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
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
        "[ARM_POLICY_REVIEW] starting "
        "(read-only governance policy review board; no runtime mutation; no broker calls)",
        flush=True,
    )

    evolution_summary = _safe_read_json(
        Path(args.evolution_summary), label="arm_governance_policy_evolution_summary.json"
    )
    evolution_record = _safe_read_json(
        Path(args.evolution_record), label="arm_governance_policy_evolution.json"
    )
    evolution_mem = _safe_read_csv_rows(
        Path(args.evolution_mem), label="arm_governance_policy_evolution_memory.csv"
    )
    learning_summary = _safe_read_json(
        Path(args.learning_summary), label="arm_governance_learning_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_policy_review_memory.csv"
    )

    record, summary, md, merged_memory = build_policy_review(
        evolution_summary=evolution_summary,
        evolution_record=evolution_record,
        evolution_mem=evolution_mem,
        learning_summary=learning_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        existing_review_mem=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=REVIEW_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["decision_counts"]
    print(
        "[ARM_POLICY_REVIEW] "
        f"state={record['review_state']} "
        f"approved={counts['approved']} "
        f"deferred={counts['deferred']} "
        f"rejected={counts['rejected']} "
        f"confidence={record['review_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_POLICY_REVIEW_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_POLICY_REVIEW_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_POLICY_REVIEW_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
