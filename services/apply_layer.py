# services/apply_layer.py
"""
TRITON — Controlled Apply Layer (Phase 1: registry only).

Purpose
-------
Take *explicitly approved* adaptation proposals and write them into a
separate, additive, auditable, reversible applied-adjustments registry.

This module is the safe bridge between:
    services/adaptation_layer.py  (advisory proposals, advisory_only=True)
        ↓
    data/results/applied_adjustments.csv  (this layer's authoritative state)

Hard contract — Phase 1
-----------------------
1. This module never mutates broker, lifecycle, signal, execution, portfolio,
   or risk surfaces. It only reads from data/results/ and writes to
   data/results/.
2. A proposal becomes "applied" only when an explicit approval source marks it
   APPROVED. With no approvals on disk, this layer applies *nothing*.
3. Every applied row is bounded by the proposal's own [min_allowed_value,
   max_allowed_value] guardrails; out-of-range proposals are rejected.
4. Application is idempotent: re-running with the same approved proposals
   produces zero new APPLY events.
5. Supersession is non-destructive: previously-active rows are flagged
   INACTIVE with `superseded_by_application_id` pointing at the new row;
   nothing is deleted.
6. Rollback is non-destructive: rows are flagged ROLLED_BACK with
   active_flag=False and the apply log gets a ROLLBACK event.

Inputs (best-effort, never required):
    data/results/adaptation_proposals.csv     (primary)
    data/results/adaptation_review_queue.csv  (optional approval source #2)
    data/results/adaptation_summary.json      (governance metadata)
    data/results/approval_queue.csv           (optional, primary approval source)
    data/results/applied_adjustments.csv      (existing state, for idempotency)
    data/results/applied_adjustments.json     (existing state mirror)

Outputs:
    data/results/applied_adjustments.csv      (registry — current truth)
    data/results/applied_adjustments.json     (registry mirror w/ active+all)
    data/results/apply_log.csv                (append-only event log)
    data/results/apply_summary.json           (counts + status snapshot)

Runtime consumption (execute path):
    services/adaptation_simulation.apply_runtime_score_influence() loads this CSV,
    keeps rows that are ACTIVE/APPLIED with active_flag, and applies bounded score
    deltas during execute_trades planning (same rule predicates as simulation).

Run
---
    python -m services.apply_layer
or
    python services/apply_layer.py
Optional flags:
    --approval-file PATH    Override approval_queue.csv path
    --dry-run               Compute and log everything, write nothing
    --rollback-id ID        Roll back a specific application_id and exit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

INPUT_PATHS: Dict[str, Path] = {
    "adaptation_proposals": RESULTS / "adaptation_proposals.csv",
    "adaptation_review_queue": RESULTS / "adaptation_review_queue.csv",
    "adaptation_summary": RESULTS / "adaptation_summary.json",
    "approval_queue": RESULTS / "approval_queue.csv",
    "applied_adjustments_csv": RESULTS / "applied_adjustments.csv",
    "applied_adjustments_json": RESULTS / "applied_adjustments.json",
}

APPLIED_CSV = RESULTS / "applied_adjustments.csv"
APPLIED_JSON = RESULTS / "applied_adjustments.json"
APPLY_LOG_CSV = RESULTS / "apply_log.csv"
APPLY_SUMMARY_JSON = RESULTS / "apply_summary.json"

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Proposal/applied-row statuses understood by this layer.
STATUS_PROPOSED = "PROPOSED"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"
STATUS_APPLIED = "APPLIED"
STATUS_ROLLED_BACK = "ROLLED_BACK"
STATUS_INACTIVE = "INACTIVE"

# Apply-log event types.
EVENT_APPLY = "APPLY"
EVENT_SKIP = "SKIP"
EVENT_ROLLBACK = "ROLLBACK"
EVENT_SUPERSEDE = "SUPERSEDE"
EVENT_NOOP = "NOOP"

# What "applied_by" gets recorded as when not provided through approval row.
DEFAULT_APPLIED_BY = "apply_layer"

# Phase tag echoed into outputs.
PHASE = "1-applied-registry-only"

# Stable, full-schema column order — used to make every output deterministic.
APPLIED_COLUMNS: List[str] = [
    # Identity
    "application_id",
    "proposal_id",
    "generated_at_utc",
    "applied_at_utc",
    "adaptation_target",
    "proposal_type",
    # Proposal context
    "proposal_direction",
    "proposal_strength",
    "proposal_confidence",
    "evidence_count",
    "evidence_strength",
    "recommendation_type",
    # Applied values
    "current_value",
    "proposed_value",
    "proposed_delta",
    "effective_value",
    "effective_delta",
    "min_allowed_value",
    "max_allowed_value",
    "bounded_change_applied",
    # Control
    "status",
    "active_flag",
    "rollback_eligible",
    "rollback_parent_application_id",
    "superseded_by_application_id",
    # Audit
    "apply_reason",
    "apply_note",
    "source_file",
    "source_status",
    "applied_by",
    "advisory_origin",
    # Supersession identity (helps reviewers join across rows)
    "related_bucket",
    "related_flag",
    "related_style",
]

APPLY_LOG_COLUMNS: List[str] = [
    "event_time_utc",
    "event_type",
    "application_id",
    "proposal_id",
    "adaptation_target",
    "result",
    "reason",
    "note",
]

# Bool columns that get serialized as '1'/'0'/'' for CSV stability.
_BOOL_COLS_APPLIED = {
    "bounded_change_applied",
    "active_flag",
    "rollback_eligible",
}


# ─────────────────────────────────────────────────────────────
# Safe IO helpers (mirrors services/adaptation_layer.py contracts)
# ─────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_csv_safe(path: Path) -> Tuple[pd.DataFrame, str]:
    """Best-effort CSV loader. Returns (df, status); never raises."""
    try:
        if not path.exists():
            return pd.DataFrame(), "missing"
        try:
            if path.stat().st_size == 0:
                return pd.DataFrame(), "empty"
        except OSError:
            pass
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df is None or df.empty:
            return (df if df is not None else pd.DataFrame()), "empty"
        df.columns = [str(c).strip() for c in df.columns]
        return df, "ok"
    except Exception as e:
        return pd.DataFrame(), f"error:{type(e).__name__}:{str(e)[:120]}"


def load_json_safe(path: Path) -> Tuple[Optional[Any], str]:
    """Best-effort JSON loader. Returns (obj_or_None, status); never raises."""
    try:
        if not path.exists():
            return None, "missing"
        try:
            if path.stat().st_size == 0:
                return None, "empty"
        except OSError:
            pass
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None, "empty"
        obj = json.loads(text)
        return obj, "ok"
    except Exception as e:
        return None, f"error:{type(e).__name__}:{str(e)[:120]}"


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        if isinstance(x, str) and not x.strip():
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    v = _safe_float(x, None)
    return int(v) if v is not None else default


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none") else s


def _safe_bool(x: Any, default: Optional[bool] = None) -> Optional[bool]:
    """
    Tri-state boolean parse: True / False / None (unknown).
    Accepts standard truthy/falsy strings and pandas-style empties.
    """
    if x is None:
        return default
    try:
        if isinstance(x, float) and math.isnan(x):
            return default
    except Exception:
        pass
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("", "nan", "none"):
        return default
    if s in ("true", "1", "yes", "t", "y"):
        return True
    if s in ("false", "0", "no", "f", "n"):
        return False
    return default


def _clamp(v: float, lo: float, hi: float) -> Tuple[float, bool]:
    """Clamp v into [lo, hi]. Returns (clamped_value, was_bounded)."""
    if v < lo:
        return lo, True
    if v > hi:
        return hi, True
    return v, False


# ─────────────────────────────────────────────────────────────
# Loader bundle
# ─────────────────────────────────────────────────────────────


@dataclass
class ApplyInputs:
    proposals: pd.DataFrame = field(default_factory=pd.DataFrame)
    review_queue: pd.DataFrame = field(default_factory=pd.DataFrame)
    approval_queue: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Optional[Any] = None
    existing_applied: pd.DataFrame = field(default_factory=pd.DataFrame)
    existing_applied_json: Optional[Any] = None
    status: Dict[str, str] = field(default_factory=dict)
    paths_used: Dict[str, str] = field(default_factory=dict)

    def missing(self) -> List[str]:
        return [name for name, s in self.status.items() if s != "ok"]


def load_inputs(approval_file: Optional[Path] = None) -> ApplyInputs:
    """
    Load every potential apply-layer input. Existing applied state is loaded
    here so idempotency and supersession can use it.

    `approval_file` overrides INPUT_PATHS["approval_queue"] when provided.
    """
    inp = ApplyInputs()
    paths = dict(INPUT_PATHS)
    if approval_file is not None:
        paths["approval_queue"] = Path(approval_file)

    inp.proposals, inp.status["adaptation_proposals"] = load_csv_safe(paths["adaptation_proposals"])
    inp.review_queue, inp.status["adaptation_review_queue"] = load_csv_safe(
        paths["adaptation_review_queue"]
    )
    inp.approval_queue, inp.status["approval_queue"] = load_csv_safe(paths["approval_queue"])
    inp.summary, inp.status["adaptation_summary"] = load_json_safe(paths["adaptation_summary"])
    inp.existing_applied, inp.status["applied_adjustments_csv"] = load_csv_safe(
        paths["applied_adjustments_csv"]
    )
    inp.existing_applied_json, inp.status["applied_adjustments_json"] = load_json_safe(
        paths["applied_adjustments_json"]
    )

    inp.paths_used = {k: str(v) for k, v in paths.items()}
    return inp


# ─────────────────────────────────────────────────────────────
# Approval resolution
# ─────────────────────────────────────────────────────────────


def _norm_status(s: Any) -> str:
    """Normalize free-form approval status strings into the canonical set."""
    raw = _safe_str(s).upper()
    if not raw:
        return ""
    if raw in (
        STATUS_PROPOSED,
        STATUS_APPROVED,
        STATUS_REJECTED,
        STATUS_APPLIED,
        STATUS_ROLLED_BACK,
        STATUS_INACTIVE,
    ):
        return raw
    if raw in ("APPROVE", "OK", "GREEN", "ACCEPT", "ACCEPTED"):
        return STATUS_APPROVED
    if raw in ("REJECT", "DECLINED", "DENY", "DENIED", "RED"):
        return STATUS_REJECTED
    if raw in ("PENDING", "REVIEW", "OPEN", "QUEUED"):
        return STATUS_PROPOSED
    return raw  # leave as-is so callers can see oddities


def _approvals_from_queue(approval_q: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build {proposal_id: approval-info} from approval_queue.csv.
    Recognized columns (all optional except proposal_id):
        proposal_id, approval_status, approval_note, approved_by, approved_at_utc
    Any unrecognized status normalizes via `_norm_status`.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if approval_q is None or approval_q.empty or "proposal_id" not in approval_q.columns:
        return out
    for _, row in approval_q.iterrows():
        pid = _safe_str(row.get("proposal_id"))
        if not pid:
            continue
        status = _norm_status(row.get("approval_status"))
        out[pid] = {
            "status": status,
            "note": _safe_str(row.get("approval_note")),
            "approved_by": _safe_str(row.get("approved_by")) or DEFAULT_APPLIED_BY,
            "approved_at_utc": _safe_str(row.get("approved_at_utc")) or _utc_now_iso(),
            "source_file": "approval_queue.csv",
        }
    return out


def _approvals_from_inline(df: pd.DataFrame, source_label: str) -> Dict[str, Dict[str, Any]]:
    """
    Some pipelines mark approval directly on the proposal/review-queue row via
    one of: approval_status, status, approved (True/False).
    We treat status==APPROVED on those rows as eligibility.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty or "proposal_id" not in df.columns:
        return out

    has_apprstat = "approval_status" in df.columns
    has_status = "status" in df.columns
    has_approved = "approved" in df.columns
    if not (has_apprstat or has_status or has_approved):
        return out

    for _, row in df.iterrows():
        pid = _safe_str(row.get("proposal_id"))
        if not pid:
            continue
        status = ""
        if has_apprstat:
            status = _norm_status(row.get("approval_status"))
        if not status and has_status:
            # Only treat 'status' as approval if it explicitly says APPROVED;
            # the proposals file's default 'PROPOSED' must NOT auto-approve.
            cand = _norm_status(row.get("status"))
            if cand == STATUS_APPROVED:
                status = STATUS_APPROVED
            elif cand == STATUS_REJECTED:
                status = STATUS_REJECTED
        if not status and has_approved:
            b = _safe_bool(row.get("approved"))
            if b is True:
                status = STATUS_APPROVED
            elif b is False:
                status = STATUS_REJECTED
        if not status:
            continue
        out[pid] = {
            "status": status,
            "note": _safe_str(row.get("approval_note")),
            "approved_by": _safe_str(row.get("approved_by")) or DEFAULT_APPLIED_BY,
            "approved_at_utc": _safe_str(row.get("approved_at_utc")) or _utc_now_iso(),
            "source_file": source_label,
        }
    return out


def resolve_approvals(inp: ApplyInputs) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Find approvals using the documented priority order:
      1. approval_queue.csv
      2. adaptation_review_queue.csv (if it carries inline approval columns)
      3. adaptation_proposals.csv    (if it carries inline approval columns)
    Returns ({proposal_id: info}, source_label). Source label "none" when
    no approval information exists anywhere.
    """
    if not inp.approval_queue.empty:
        approvals = _approvals_from_queue(inp.approval_queue)
        if approvals:
            return approvals, "approval_queue.csv"

    approvals = _approvals_from_inline(inp.review_queue, "adaptation_review_queue.csv")
    if approvals:
        return approvals, "adaptation_review_queue.csv"

    approvals = _approvals_from_inline(inp.proposals, "adaptation_proposals.csv")
    if approvals:
        return approvals, "adaptation_proposals.csv"

    return {}, "none"


# ─────────────────────────────────────────────────────────────
# Eligibility
# ─────────────────────────────────────────────────────────────


@dataclass
class EligibilityResult:
    eligible: bool
    reason: str  # short machine code
    detail: str  # human-readable diagnostics


def _eligibility(prop: Dict[str, Any], approval: Dict[str, Any]) -> EligibilityResult:
    """Apply the Phase-1 eligibility rules to a single proposal+approval pair."""
    pid = _safe_str(prop.get("proposal_id"))
    if not pid:
        return EligibilityResult(False, "missing_proposal_id", "Proposal row has no proposal_id.")
    target = _safe_str(prop.get("adaptation_target"))
    if not target:
        return EligibilityResult(
            False, "missing_adaptation_target", f"Proposal {pid} has no adaptation_target."
        )
    ptype = _safe_str(prop.get("proposal_type"))
    if not ptype:
        return EligibilityResult(
            False, "missing_proposal_type", f"Proposal {pid} has no proposal_type."
        )

    status = _norm_status(approval.get("status"))
    if status != STATUS_APPROVED:
        return EligibilityResult(
            False, "not_approved", f"Approval status is {status or 'BLANK'}; need APPROVED."
        )

    # advisory_only must be True or blank — the only blocker is an explicit False.
    adv = _safe_bool(prop.get("advisory_only"), default=True)
    if adv is False:
        return EligibilityResult(
            False, "advisory_only_false", f"Proposal {pid} carries advisory_only=False."
        )

    delta = _safe_float(prop.get("proposed_delta"))
    if delta is None:
        return EligibilityResult(
            False, "missing_proposed_delta", f"Proposal {pid} has no numeric proposed_delta."
        )
    lo = _safe_float(prop.get("min_allowed_value"))
    hi = _safe_float(prop.get("max_allowed_value"))
    if lo is not None and hi is not None and lo > hi:
        return EligibilityResult(
            False, "invalid_bounds", f"Proposal {pid} bounds inverted: min={lo} > max={hi}."
        )
    if lo is not None and delta < lo - 1e-9:
        return EligibilityResult(
            False, "delta_below_min", f"Proposal {pid} proposed_delta={delta} < min_allowed={lo}."
        )
    if hi is not None and delta > hi + 1e-9:
        return EligibilityResult(
            False, "delta_above_max", f"Proposal {pid} proposed_delta={delta} > max_allowed={hi}."
        )

    return EligibilityResult(True, "ok", "Eligible.")


# ─────────────────────────────────────────────────────────────
# Existing applied state — index helpers
# ─────────────────────────────────────────────────────────────


def _supersede_key(prop_or_row: Dict[str, Any]) -> str:
    """
    Identity used to detect "another active row addresses the same surface".
    Keyed on adaptation_target + related_bucket / flag / style so distinct
    related contexts (e.g. two sizing buckets) don't supersede each other.
    """
    parts = [
        _safe_str(prop_or_row.get("adaptation_target")),
        _safe_str(prop_or_row.get("related_bucket")),
        _safe_str(prop_or_row.get("related_flag")),
        _safe_str(prop_or_row.get("related_style")),
    ]
    return "|".join(parts)


def _index_existing_applied(existing: pd.DataFrame) -> Dict[str, Any]:
    """
    Build lookup structures from any prior applied_adjustments.csv:
      - by_pid_active:    {proposal_id: row_dict}    (only ACTIVE rows)
      - by_key_active:    {supersede_key: row_dict}  (only ACTIVE rows)
      - all_rows:         list of dicts (full history, untouched)
    """
    empty = {"by_pid_active": {}, "by_key_active": {}, "all_rows": []}
    if existing is None or existing.empty:
        return empty

    df = existing.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "active_flag" in df.columns:
        active_mask = (
            df["active_flag"]
            .apply(lambda v: _safe_bool(v, default=False))
            .fillna(False)
            .astype(bool)
        )
    else:
        active_mask = pd.Series([False] * len(df), index=df.index)

    by_pid: Dict[str, Dict[str, Any]] = {}
    by_key: Dict[str, Dict[str, Any]] = {}
    all_rows: List[Dict[str, Any]] = df.to_dict(orient="records")

    for _, row in df[active_mask].iterrows():
        d = row.to_dict()
        pid = _safe_str(d.get("proposal_id"))
        if pid:
            by_pid[pid] = d
        by_key[_supersede_key(d)] = d
    return {"by_pid_active": by_pid, "by_key_active": by_key, "all_rows": all_rows}


# ─────────────────────────────────────────────────────────────
# Application engine
# ─────────────────────────────────────────────────────────────


def _application_id() -> str:
    return f"APPLY-{uuid.uuid4().hex[:12]}"


def _build_applied_row(
    prop: Dict[str, Any],
    approval: Dict[str, Any],
    *,
    application_id: str,
    applied_at_utc: str,
    source_file: str,
    source_status: str,
    reason: str = "Applied via apply_layer.",
) -> Dict[str, Any]:
    """
    Build a fully-populated applied-row dict using the full APPLIED_COLUMNS
    schema. Keys missing from `prop` get safe blanks. effective_value /
    effective_delta apply the Phase-1 rule documented in the module docstring.
    """
    # Effective value/delta: Phase 1 never invents a current value.
    proposed_val = _safe_str(prop.get("proposed_value"))
    current_val = _safe_str(prop.get("current_value"))
    proposed_delta = _safe_float(prop.get("proposed_delta"))
    if current_val:
        effective_value = current_val
    elif proposed_val:
        effective_value = proposed_val
    else:
        effective_value = ""
    effective_delta = proposed_delta

    note_parts: List[str] = []
    appr_note = _safe_str(approval.get("note"))
    if appr_note:
        note_parts.append(f"approval_note: {appr_note}")
    prop_note = _safe_str(prop.get("proposal_note"))
    if prop_note:
        note_parts.append(f"proposal_note: {prop_note}")
    apply_note = " | ".join(note_parts)

    advisory_origin = _safe_bool(prop.get("advisory_only"), default=True)

    row: Dict[str, Any] = {
        "application_id": application_id,
        "proposal_id": _safe_str(prop.get("proposal_id")),
        "generated_at_utc": _safe_str(prop.get("generated_at_utc")),
        "applied_at_utc": applied_at_utc,
        "adaptation_target": _safe_str(prop.get("adaptation_target")),
        "proposal_type": _safe_str(prop.get("proposal_type")),
        "proposal_direction": _safe_str(prop.get("proposal_direction")),
        "proposal_strength": _safe_str(prop.get("proposal_strength")),
        "proposal_confidence": _safe_float(prop.get("proposal_confidence"), 0.0),
        "evidence_count": _safe_int(prop.get("evidence_count"), 0),
        "evidence_strength": _safe_str(prop.get("evidence_strength")),
        "recommendation_type": _safe_str(prop.get("recommendation_type")),
        "current_value": current_val,
        "proposed_value": proposed_val,
        "proposed_delta": proposed_delta if proposed_delta is not None else "",
        "effective_value": effective_value,
        "effective_delta": effective_delta if effective_delta is not None else "",
        "min_allowed_value": _safe_float(prop.get("min_allowed_value")),
        "max_allowed_value": _safe_float(prop.get("max_allowed_value")),
        "bounded_change_applied": _safe_bool(prop.get("bounded_change_applied"), default=False),
        "status": STATUS_APPLIED,
        "active_flag": True,
        "rollback_eligible": True,
        "rollback_parent_application_id": "",
        "superseded_by_application_id": "",
        "apply_reason": reason,
        "apply_note": apply_note,
        "source_file": source_file,
        "source_status": source_status,
        "applied_by": _safe_str(approval.get("approved_by")) or DEFAULT_APPLIED_BY,
        "advisory_origin": True if advisory_origin is None else bool(advisory_origin),
        "related_bucket": _safe_str(prop.get("related_bucket")),
        "related_flag": _safe_str(prop.get("related_flag")),
        "related_style": _safe_str(prop.get("related_style")),
    }

    # Defensive: fill any missing schema column with a blank.
    for col in APPLIED_COLUMNS:
        row.setdefault(col, "")
    return row


def _log_event(
    event_type: str,
    *,
    application_id: str = "",
    proposal_id: str = "",
    adaptation_target: str = "",
    result: str = "",
    reason: str = "",
    note: str = "",
) -> Dict[str, Any]:
    return {
        "event_time_utc": _utc_now_iso(),
        "event_type": event_type,
        "application_id": application_id,
        "proposal_id": proposal_id,
        "adaptation_target": adaptation_target,
        "result": result,
        "reason": reason,
        "note": note,
    }


@dataclass
class ApplyResult:
    new_active_rows: List[Dict[str, Any]] = field(default_factory=list)
    superseded_rows: List[Dict[str, Any]] = field(default_factory=list)
    untouched_rows: List[Dict[str, Any]] = field(default_factory=list)
    log_events: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(
        default_factory=lambda: {
            "proposals_seen": 0,
            "proposals_approved": 0,
            "proposals_applied": 0,
            "proposals_skipped": 0,
            "proposals_rolled_back": 0,
            "supersessions": 0,
        }
    )


def apply_proposals(inp: ApplyInputs) -> ApplyResult:
    """
    Pure planner: compute the next applied-state and the events to log,
    without touching disk. write_outputs() persists the result.
    """
    res = ApplyResult()
    proposals = inp.proposals
    res.counts["proposals_seen"] = 0 if proposals is None else int(proposals.shape[0])

    approvals, source_file = resolve_approvals(inp)
    source_status = inp.status.get(
        {
            "approval_queue.csv": "approval_queue",
            "adaptation_review_queue.csv": "adaptation_review_queue",
            "adaptation_proposals.csv": "adaptation_proposals",
        }.get(source_file, "approval_queue"),
        "missing",
    )

    # Index any existing applied state. We start the next-state from the
    # untouched history so non-affected rows survive verbatim.
    existing_idx = _index_existing_applied(inp.existing_applied)
    res.untouched_rows = list(existing_idx["all_rows"])
    by_pid_active: Dict[str, Dict[str, Any]] = dict(existing_idx["by_pid_active"])
    by_key_active: Dict[str, Dict[str, Any]] = dict(existing_idx["by_key_active"])

    if not approvals:
        res.log_events.append(
            _log_event(
                EVENT_NOOP,
                reason="no_approved_proposals_found",
                note=(
                    "No approval source produced APPROVED rows "
                    "(checked approval_queue.csv, adaptation_review_queue.csv, "
                    "adaptation_proposals.csv)."
                ),
            )
        )
        return res

    if proposals is None or proposals.empty:
        res.log_events.append(
            _log_event(
                EVENT_NOOP,
                reason="no_proposals_loaded",
                note=(
                    "Approval source produced rows but adaptation_proposals.csv "
                    "is empty or missing — nothing to match against."
                ),
            )
        )
        return res

    res.counts["proposals_approved"] = sum(
        1 for a in approvals.values() if _norm_status(a.get("status")) == STATUS_APPROVED
    )

    # We process approved proposals deterministically by proposal_id so a
    # rerun produces an identical sequence. If the same proposal_id appears
    # twice in the proposals frame, we keep the *last* row (most recent).
    prop_by_id: Dict[str, Dict[str, Any]] = {}
    for _, row in proposals.iterrows():
        pid = _safe_str(row.get("proposal_id"))
        if pid:
            prop_by_id[pid] = row.to_dict()

    apply_iso = _utc_now_iso()

    for pid in sorted(approvals.keys()):
        approval = approvals[pid]
        prop = prop_by_id.get(pid)
        if prop is None:
            res.counts["proposals_skipped"] += 1
            res.log_events.append(
                _log_event(
                    EVENT_SKIP,
                    proposal_id=pid,
                    result="skipped",
                    reason="proposal_not_found",
                    note=(
                        "Approval references a proposal_id that is not present "
                        "in adaptation_proposals.csv."
                    ),
                )
            )
            continue

        elig = _eligibility(prop, approval)
        if not elig.eligible:
            res.counts["proposals_skipped"] += 1
            res.log_events.append(
                _log_event(
                    EVENT_SKIP,
                    proposal_id=pid,
                    adaptation_target=_safe_str(prop.get("adaptation_target")),
                    result="skipped",
                    reason=elig.reason,
                    note=elig.detail,
                )
            )
            continue

        # Idempotency: same proposal_id already active → no-op SKIP, log it.
        if pid in by_pid_active:
            res.counts["proposals_skipped"] += 1
            existing_row = by_pid_active[pid]
            res.log_events.append(
                _log_event(
                    EVENT_SKIP,
                    application_id=_safe_str(existing_row.get("application_id")),
                    proposal_id=pid,
                    adaptation_target=_safe_str(prop.get("adaptation_target")),
                    result="skipped",
                    reason="already_applied",
                    note="Active applied row exists for this proposal_id.",
                )
            )
            continue

        # Supersession: a different proposal_id is active for the same surface.
        new_app_id = _application_id()
        sup_key = _supersede_key(prop)
        if sup_key in by_key_active:
            old = by_key_active[sup_key]
            old_app_id = _safe_str(old.get("application_id"))

            # Patch the untouched-history copy in place: status→INACTIVE,
            # active_flag→False, supersede pointer set to the new application.
            for hist_row in res.untouched_rows:
                if _safe_str(hist_row.get("application_id")) == old_app_id:
                    hist_row["status"] = STATUS_INACTIVE
                    hist_row["active_flag"] = False
                    hist_row["superseded_by_application_id"] = new_app_id
                    res.superseded_rows.append(hist_row)
                    break

            # Drop from active indices so the new row can take over.
            old_pid = _safe_str(old.get("proposal_id"))
            if old_pid and old_pid in by_pid_active:
                del by_pid_active[old_pid]
            if sup_key in by_key_active:
                del by_key_active[sup_key]

            res.counts["supersessions"] += 1
            res.log_events.append(
                _log_event(
                    EVENT_SUPERSEDE,
                    application_id=old_app_id,
                    proposal_id=old_pid,
                    adaptation_target=_safe_str(old.get("adaptation_target")),
                    result="superseded",
                    reason="newer_approved_proposal_for_same_target",
                    note=f"Superseded by application_id={new_app_id} (proposal_id={pid}).",
                )
            )

        new_row = _build_applied_row(
            prop,
            approval,
            application_id=new_app_id,
            applied_at_utc=apply_iso,
            source_file=source_file,
            source_status=source_status,
        )
        res.new_active_rows.append(new_row)
        by_pid_active[pid] = new_row
        by_key_active[sup_key] = new_row
        res.counts["proposals_applied"] += 1
        res.log_events.append(
            _log_event(
                EVENT_APPLY,
                application_id=new_app_id,
                proposal_id=pid,
                adaptation_target=_safe_str(prop.get("adaptation_target")),
                result="applied",
                reason="approved_and_eligible",
                note=f"Sourced from {source_file}.",
            )
        )

    return res


# ─────────────────────────────────────────────────────────────
# Rollback
# ─────────────────────────────────────────────────────────────


@dataclass
class RollbackResult:
    rolled_back: bool
    application_id: str
    log_events: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""


def rollback_application(
    application_id: str, reason: str = "manual rollback", *, dry_run: bool = False
) -> RollbackResult:
    """
    Mark a specific application_id as ROLLED_BACK.
    Non-destructive: history is preserved; only status / active_flag flip.
    Always appends a ROLLBACK event to apply_log.csv (unless dry_run).
    """
    aid = _safe_str(application_id)
    res = RollbackResult(rolled_back=False, application_id=aid, reason=reason)

    existing, status = load_csv_safe(APPLIED_CSV)
    if existing is None or existing.empty:
        res.reason = "no_registry"
        res.log_events.append(
            _log_event(
                EVENT_NOOP,
                application_id=aid,
                result="noop",
                reason="rollback_failed_no_registry",
                note="applied_adjustments.csv is missing or empty.",
            )
        )
        if not dry_run:
            _append_log_events(res.log_events)
        return res

    if not aid:
        res.reason = "missing_application_id"
        res.log_events.append(
            _log_event(
                EVENT_NOOP,
                result="noop",
                reason="rollback_failed_missing_application_id",
                note="rollback_application called without an application_id.",
            )
        )
        if not dry_run:
            _append_log_events(res.log_events)
        return res

    rows = existing.to_dict(orient="records")
    found = False
    target_pid = ""
    target_target = ""
    for row in rows:
        if _safe_str(row.get("application_id")) != aid:
            continue
        found = True
        target_pid = _safe_str(row.get("proposal_id"))
        target_target = _safe_str(row.get("adaptation_target"))
        active = _safe_bool(row.get("active_flag"), default=False)
        if not active:
            res.reason = "already_inactive"
            res.log_events.append(
                _log_event(
                    EVENT_NOOP,
                    application_id=aid,
                    proposal_id=target_pid,
                    adaptation_target=target_target,
                    result="noop",
                    reason="rollback_failed_not_active",
                    note="application_id is already inactive (rolled back or superseded).",
                )
            )
            break
        row["status"] = STATUS_ROLLED_BACK
        row["active_flag"] = False
        row["rollback_eligible"] = False
        # Stamp the parent pointer to itself so consumers can join history.
        if not _safe_str(row.get("rollback_parent_application_id")):
            row["rollback_parent_application_id"] = aid
        res.rolled_back = True
        res.log_events.append(
            _log_event(
                EVENT_ROLLBACK,
                application_id=aid,
                proposal_id=target_pid,
                adaptation_target=target_target,
                result="rolled_back",
                reason="manual_rollback",
                note=reason,
            )
        )
        break

    if not found:
        res.reason = "not_found"
        res.log_events.append(
            _log_event(
                EVENT_NOOP,
                application_id=aid,
                result="noop",
                reason="rollback_failed_not_found",
                note="application_id not present in applied_adjustments.csv.",
            )
        )

    res.history = rows

    if not dry_run:
        _write_applied(rows)
        _append_log_events(res.log_events)
        _write_summary_after_rollback(rows, source_status=status, reason=reason)

    return res


# ─────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────


def _df_for_csv(df: pd.DataFrame, bool_cols: Optional[set] = None) -> pd.DataFrame:
    """Stable CSV cast — bool → '1'/'0'/''. Avoid duplicate columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
    cols = set(bool_cols or [])
    for col in list(out.columns):
        if col in cols or out[col].dtype == bool:
            s = out[col]
            out = out.drop(columns=[col])
            try:
                norm = s.apply(lambda v: _safe_bool(v, default=None))
                out[col] = norm.map(lambda b: "" if b is None else ("1" if b else "0"))
            except Exception:
                out[col] = s
    return out


def _empty_applied_df() -> pd.DataFrame:
    return pd.DataFrame(columns=APPLIED_COLUMNS)


def _empty_log_df() -> pd.DataFrame:
    return pd.DataFrame(columns=APPLY_LOG_COLUMNS)


def _normalize_rows_for_schema(rows: List[Dict[str, Any]], cols: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    return df[cols]


def _write_applied(all_rows: List[Dict[str, Any]]) -> None:
    APPLIED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = _normalize_rows_for_schema(all_rows, APPLIED_COLUMNS)
    _df_for_csv(df, bool_cols=_BOOL_COLS_APPLIED).to_csv(APPLIED_CSV, index=False)
    _write_applied_json(all_rows)


def _write_applied_json(all_rows: List[Dict[str, Any]]) -> None:
    """Mirror the registry into JSON with active+all separation."""
    norm: List[Dict[str, Any]] = []
    for r in all_rows:
        d = dict(r)
        d["active_flag"] = bool(_safe_bool(d.get("active_flag"), default=False))
        d["rollback_eligible"] = bool(_safe_bool(d.get("rollback_eligible"), default=False))
        d["bounded_change_applied"] = bool(
            _safe_bool(d.get("bounded_change_applied"), default=False)
        )
        norm.append(d)
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "schema_version": 1,
        "phase": PHASE,
        "active_adjustments": [r for r in norm if r.get("active_flag")],
        "all_adjustments": norm,
        "notes": [
            "Phase 1: this registry is the authoritative applied state but is "
            "not yet consumed by live trading code.",
        ],
    }
    APPLIED_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _append_log_events(events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    APPLY_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_new = _normalize_rows_for_schema(events, APPLY_LOG_COLUMNS)
    if APPLY_LOG_CSV.exists() and APPLY_LOG_CSV.stat().st_size > 0:
        try:
            df_new.to_csv(APPLY_LOG_CSV, mode="a", header=False, index=False)
            return
        except Exception:
            pass  # fall through to full rewrite
    # First write — start the file with a clean header row.
    df_new.to_csv(APPLY_LOG_CSV, index=False)


def _build_summary(
    res: ApplyResult, inp: ApplyInputs, approvals_source: str, notes: List[str]
) -> Dict[str, Any]:
    next_state = list(res.untouched_rows) + list(res.new_active_rows)
    active = [r for r in next_state if _safe_bool(r.get("active_flag"), default=False)]
    inactive = [r for r in next_state if not _safe_bool(r.get("active_flag"), default=False)]

    src_rows: Dict[str, Dict[str, Any]] = {}
    for name, path in INPUT_PATHS.items():
        rows = 0
        if name == "adaptation_proposals":
            rows = int(inp.proposals.shape[0]) if inp.proposals is not None else 0
        elif name == "adaptation_review_queue":
            rows = int(inp.review_queue.shape[0]) if inp.review_queue is not None else 0
        elif name == "approval_queue":
            rows = int(inp.approval_queue.shape[0]) if inp.approval_queue is not None else 0
        elif name == "applied_adjustments_csv":
            rows = int(inp.existing_applied.shape[0]) if inp.existing_applied is not None else 0
        elif name == "adaptation_summary":
            rows = 0 if inp.summary is None else 1
        elif name == "applied_adjustments_json":
            rows = 0 if inp.existing_applied_json is None else 1
        src_rows[name] = {
            "status": inp.status.get(name, "missing"),
            "rows": rows,
            "path": str(path),
        }

    rolled_back_count = sum(
        1 for r in next_state if _safe_str(r.get("status")) == STATUS_ROLLED_BACK
    )
    res.counts["proposals_rolled_back"] = rolled_back_count

    top_active: List[Dict[str, Any]] = []
    if active:
        try:
            tmp = pd.DataFrame(active)
            tmp["__conf"] = pd.to_numeric(tmp.get("proposal_confidence"), errors="coerce").fillna(
                0.0
            )
            tmp = tmp.sort_values("__conf", ascending=False).drop(columns=["__conf"])
            keep = [
                c
                for c in (
                    "application_id",
                    "proposal_id",
                    "adaptation_target",
                    "proposal_type",
                    "proposal_direction",
                    "proposal_strength",
                    "proposal_confidence",
                    "evidence_strength",
                    "evidence_count",
                    "effective_delta",
                    "applied_at_utc",
                    "source_file",
                    "related_bucket",
                    "related_flag",
                    "related_style",
                )
                if c in tmp.columns
            ]
            top_active = tmp[keep].head(10).to_dict(orient="records") if keep else []
        except Exception:
            top_active = []

    return {
        "generated_at_utc": _utc_now_iso(),
        "schema_version": 1,
        "phase": PHASE,
        "advisory_only_source": True,
        "auto_apply_allowed": False,
        "approvals_source": approvals_source,
        "source_availability": src_rows,
        "missing_inputs": [n for n, s in inp.status.items() if s != "ok"],
        "proposals_seen": int(res.counts["proposals_seen"]),
        "proposals_approved": int(res.counts["proposals_approved"]),
        "proposals_applied": int(res.counts["proposals_applied"]),
        "proposals_skipped": int(res.counts["proposals_skipped"]),
        "proposals_rolled_back": int(rolled_back_count),
        "supersessions": int(res.counts["supersessions"]),
        "active_adjustments_count": len(active),
        "inactive_adjustments_count": len(inactive),
        "top_active_adjustments": top_active,
        "notes": notes,
    }


def write_outputs(
    res: ApplyResult, inp: ApplyInputs, *, approvals_source: str, notes: Optional[List[str]] = None
) -> Dict[str, str]:
    notes = notes or []
    written: Dict[str, str] = {}
    APPLIED_CSV.parent.mkdir(parents=True, exist_ok=True)

    next_state = list(res.untouched_rows) + list(res.new_active_rows)
    try:
        if next_state:
            _write_applied(next_state)
        else:
            _empty_applied_df().to_csv(APPLIED_CSV, index=False)
            _write_applied_json([])
        written["applied_csv"] = str(APPLIED_CSV)
        written["applied_json"] = str(APPLIED_JSON)
    except Exception as e:
        written["applied_error"] = f"{type(e).__name__}:{e}"

    try:
        if res.log_events:
            _append_log_events(res.log_events)
        elif not APPLY_LOG_CSV.exists():
            _empty_log_df().to_csv(APPLY_LOG_CSV, index=False)
        written["apply_log_csv"] = str(APPLY_LOG_CSV)
    except Exception as e:
        written["apply_log_error"] = f"{type(e).__name__}:{e}"

    try:
        summary = _build_summary(res, inp, approvals_source, notes)
        APPLY_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        written["apply_summary_json"] = str(APPLY_SUMMARY_JSON)
    except Exception as e:
        written["apply_summary_error"] = f"{type(e).__name__}:{e}"

    return written


def _write_summary_after_rollback(
    all_rows: List[Dict[str, Any]], source_status: str, reason: str
) -> None:
    """Lightweight summary refresh used by the rollback path."""
    active = [r for r in all_rows if _safe_bool(r.get("active_flag"), default=False)]
    inactive = [r for r in all_rows if not _safe_bool(r.get("active_flag"), default=False)]
    rolled = [r for r in all_rows if _safe_str(r.get("status")) == STATUS_ROLLED_BACK]
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "schema_version": 1,
        "phase": PHASE,
        "advisory_only_source": True,
        "auto_apply_allowed": False,
        "approvals_source": "rollback_path",
        "source_availability": {
            "applied_adjustments_csv": {
                "status": source_status,
                "rows": len(all_rows),
                "path": str(APPLIED_CSV),
            }
        },
        "missing_inputs": [],
        "proposals_seen": 0,
        "proposals_approved": 0,
        "proposals_applied": 0,
        "proposals_skipped": 0,
        "proposals_rolled_back": len(rolled),
        "supersessions": 0,
        "active_adjustments_count": len(active),
        "inactive_adjustments_count": len(inactive),
        "top_active_adjustments": [],
        "notes": [f"Summary refreshed after rollback. reason: {reason}"],
    }
    APPLY_SUMMARY_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Top-level runner / CLI
# ─────────────────────────────────────────────────────────────


def run_apply_layer(
    *, approval_file: Optional[Path] = None, dry_run: bool = False, verbose: bool = True
) -> Dict[str, Any]:
    """Run the full apply pipeline. Always returns a result dict, never raises."""
    inp = load_inputs(approval_file=approval_file)
    notes: List[str] = []
    if inp.missing():
        notes.append(
            "Missing or unreadable input(s): "
            + ", ".join(inp.missing())
            + ". Apply layer ran on best-effort partial data."
        )

    approvals, source_file = resolve_approvals(inp)
    if not approvals:
        notes.append(
            "No approved proposals were found in any source. "
            "Phase 1 requires explicit APPROVED status; "
            "create approval_queue.csv with `proposal_id,approval_status,...` "
            "or mark rows in adaptation_review_queue.csv with approval_status=APPROVED."
        )

    try:
        result = apply_proposals(inp)
    except Exception as e:
        result = ApplyResult()
        notes.append(f"apply_proposals error: {type(e).__name__}: {e}")

    if dry_run:
        notes.append("Dry-run: no files were written.")
        if verbose:
            print(f"[apply_layer] DRY-RUN counts={result.counts}")
            for ev in result.log_events:
                print(
                    f"[apply_layer] DRY-RUN event: {ev['event_type']} "
                    f"proposal_id={ev.get('proposal_id','')} "
                    f"reason={ev.get('reason','')}"
                )
        return {
            "result": result,
            "approvals_source": source_file,
            "written": {},
            "notes": notes,
            "source_status": inp.status,
        }

    written = write_outputs(result, inp, approvals_source=source_file, notes=notes)

    if verbose:
        print(f"[apply_layer] sources_ok={[n for n,s in inp.status.items() if s=='ok']}")
        print(f"[apply_layer] missing={inp.missing()}")
        print(
            f"[apply_layer] approvals_source={source_file} "
            f"approved={result.counts['proposals_approved']} "
            f"applied={result.counts['proposals_applied']} "
            f"skipped={result.counts['proposals_skipped']} "
            f"superseded={result.counts['supersessions']}"
        )
        for k, v in written.items():
            print(f"[apply_layer] {k}: {v}")

    return {
        "result": result,
        "approvals_source": source_file,
        "written": written,
        "notes": notes,
        "source_status": inp.status,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apply_layer",
        description=(
            "Apply approved adaptation proposals to the registry. "
            "Phase 1: writes data/results/applied_adjustments.* only."
        ),
    )
    p.add_argument(
        "--approval-file", type=str, default=None, help="Override the approval_queue.csv path."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Compute and log everything without writing files."
    )
    p.add_argument(
        "--rollback-id", type=str, default=None, help="Roll back the given application_id and exit."
    )
    p.add_argument(
        "--rollback-reason",
        type=str,
        default="manual rollback",
        help="Reason recorded in the rollback log event.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress informational stdout.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    verbose = not args.quiet
    try:
        if args.rollback_id:
            r = rollback_application(
                args.rollback_id,
                reason=args.rollback_reason,
                dry_run=args.dry_run,
            )
            if verbose:
                print(
                    f"[apply_layer] rollback application_id={args.rollback_id} "
                    f"rolled_back={r.rolled_back} reason={r.reason}"
                )
            return 0 if r.rolled_back or r.reason in ("already_inactive",) else 2

        approval_path = Path(args.approval_file) if args.approval_file else None
        run_apply_layer(approval_file=approval_path, dry_run=args.dry_run, verbose=verbose)
        return 0
    except Exception as e:
        print(f"[apply_layer] FATAL {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
