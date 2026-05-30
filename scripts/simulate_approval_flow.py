"""
scripts/simulate_approval_flow.py — Triton governance-loop simulation.

Purpose
-------
Populate `data/results/` with a realistic, simulation-only adaptation /
approval / applied-state dataset so the following dashboards have meaningful
content end-to-end:

    adaptation proposal  →  approval  →  apply layer  →  applied adjustments

What this script writes (simulation only):
    data/results/adaptation_proposals.csv       (overwritten — simulation set)
    data/results/adaptation_review_queue.csv    (overwritten — priority subset)
    data/results/adaptation_summary.json        (overwritten — counts + phase)
    data/results/approval_queue.csv             (overwritten per step)
    data/results/applied_adjustments.csv        (reset, then written by apply layer)
    data/results/applied_adjustments.json       (reset, then written by apply layer)
    data/results/apply_log.csv                  (reset, then written by apply layer)
    data/results/apply_summary.json             (reset, then written by apply layer)

Hard non-goals
--------------
* NO live trading logic changes.
* NO broker / execution / risk / lifecycle modifications.
* Every emitted proposal stays advisory_only=True, auto_apply_allowed=False,
  requires_manual_review=True.
* Only files under data/results/ are touched.

Run
---
    python scripts/simulate_approval_flow.py
or  python -m scripts.simulate_approval_flow
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Path bootstrap so this script can also be run as `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported once ROOT is on sys.path.
from services import apply_layer as al  # noqa: E402

RESULTS = ROOT / "data" / "results"

PROPOSALS_CSV = RESULTS / "adaptation_proposals.csv"
REVIEW_QUEUE_CSV = RESULTS / "adaptation_review_queue.csv"
ADAPTATION_SUMMARY_JSON = RESULTS / "adaptation_summary.json"
APPROVAL_QUEUE_CSV = RESULTS / "approval_queue.csv"
APPLIED_CSV = RESULTS / "applied_adjustments.csv"
APPLIED_JSON = RESULTS / "applied_adjustments.json"
APPLY_LOG_CSV = RESULTS / "apply_log.csv"
APPLY_SUMMARY_JSON = RESULTS / "apply_summary.json"


# ──────────────────────────────────────────────────────────────
# Proposal schema — must match services/adaptation_layer._empty_proposals_df()
# ──────────────────────────────────────────────────────────────

PROPOSAL_COLUMNS: List[str] = [
    "proposal_id",
    "generated_at_utc",
    "adaptation_target",
    "proposal_type",
    "recommendation_type",
    "source_recommendation_text",
    "evidence_count",
    "evidence_strength",
    "recommendation_confidence",
    "observed_group",
    "observed_metric",
    "observed_value",
    "baseline_value",
    "effect_direction",
    "current_value",
    "proposed_value",
    "proposed_delta",
    "proposal_direction",
    "proposal_strength",
    "proposal_confidence",
    "min_allowed_value",
    "max_allowed_value",
    "bounded_change_applied",
    "requires_manual_review",
    "auto_apply_allowed",
    "proposal_reason",
    "proposal_note",
    "advisory_only",
    "status",
    "review_priority",
    "thin_data_flag",
    "related_bucket",
    "related_flag",
    "related_style",
]

APPROVAL_COLUMNS: List[str] = [
    "proposal_id",
    "approval_status",
    "approval_note",
    "approved_by",
    "approved_at_utc",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ──────────────────────────────────────────────────────────────
# Simulation inputs — realistic 6-proposal set
# ──────────────────────────────────────────────────────────────


def _proposal(**overrides: Any) -> Dict[str, Any]:
    """Build a full proposal row dict. Unspecified fields get safe defaults."""
    base: Dict[str, Any] = {
        # Identity
        "proposal_id": "",
        "generated_at_utc": _utc_now_iso(),
        "adaptation_target": "",
        "proposal_type": "",
        # Evidence
        "recommendation_type": "",
        "source_recommendation_text": "",
        "evidence_count": 0,
        "evidence_strength": "MEDIUM",
        "recommendation_confidence": 0.70,
        "observed_group": "",
        "observed_metric": "metric_snapshot",
        "observed_value": "",
        "baseline_value": "",
        "effect_direction": "INCREASE",
        # Proposed change
        "current_value": "",
        "proposed_value": "",
        "proposed_delta": 0.0,
        "proposal_direction": "INCREASE",
        "proposal_strength": "MODERATE",
        "proposal_confidence": 0.65,
        # Guardrails
        "min_allowed_value": 0.0,
        "max_allowed_value": 0.20,
        "bounded_change_applied": False,
        "requires_manual_review": True,
        "auto_apply_allowed": False,
        # Explanation
        "proposal_reason": "",
        "proposal_note": "",
        "advisory_only": True,
        # Status (the apply layer itself keys off approval_queue.csv; this
        # field is the proposal-lifecycle state, not the apply-layer state)
        "status": "PROPOSED",
        "review_priority": "MEDIUM",
        "thin_data_flag": False,
        "related_bucket": "",
        "related_flag": "",
        "related_style": "",
    }
    base.update(overrides)
    # Defensive: ensure every schema column is present exactly once.
    for col in PROPOSAL_COLUMNS:
        base.setdefault(col, "")
    return {col: base.get(col, "") for col in PROPOSAL_COLUMNS}


def _build_simulation_proposals() -> pd.DataFrame:
    """
    Realistic 6-proposal simulation set covering every governance state.

    The apply-layer keys "same surface" off
        (adaptation_target, related_bucket, related_flag, related_style).
    ADAPT-1001 and ADAPT-1002 intentionally share the same surface so that
    approving 1002 after 1001 supersedes it.
    """
    rows: List[Dict[str, Any]] = [
        # 1) Will be APPROVED then SUPERSEDED by 1002.
        _proposal(
            proposal_id="ADAPT-1001",
            adaptation_target="wide_spread_entry_penalty",
            proposal_type="INCREASE_PENALTY",
            recommendation_type="SPREAD_CAUTION",
            source_recommendation_text=(
                "TOO_WIDE-spread entries showed systematically worse slippage "
                "(avg_slip_bps=62 across 14 rows)."
            ),
            evidence_count=14,
            evidence_strength="MEDIUM",
            recommendation_confidence=0.74,
            observed_group="TOO_WIDE",
            observed_value="n=14, avg_pnl=-38, avg_slip_bps=62",
            effect_direction="INCREASE",
            proposed_delta=0.08,
            proposal_direction="INCREASE",
            proposal_strength="MODERATE",
            proposal_confidence=0.66,
            min_allowed_value=0.0,
            max_allowed_value=0.20,
            proposal_reason=(
                "Mapped from SPREAD_CAUTION; penalize entries " "when spread_bucket==TOO_WIDE."
            ),
            proposal_note="Penalize / defer wide-spread entries more strongly.",
            review_priority="MEDIUM",
            related_bucket="TOO_WIDE",
        ),
        # 2) Supersedes 1001 — same surface, stronger evidence.
        _proposal(
            proposal_id="ADAPT-1002",
            adaptation_target="wide_spread_entry_penalty",
            proposal_type="INCREASE_PENALTY",
            recommendation_type="SPREAD_CAUTION",
            source_recommendation_text=(
                "TOO_WIDE-spread entries continue to underperform after two "
                "additional weeks (n=28, avg_slip_bps=71)."
            ),
            evidence_count=28,
            evidence_strength="HIGH",
            recommendation_confidence=0.86,
            observed_group="TOO_WIDE",
            observed_value="n=28, avg_pnl=-52, avg_slip_bps=71",
            effect_direction="INCREASE",
            proposed_delta=0.14,
            proposal_direction="INCREASE",
            proposal_strength="STRONG",
            proposal_confidence=0.81,
            min_allowed_value=0.0,
            max_allowed_value=0.20,
            proposal_reason=(
                "Mapped from SPREAD_CAUTION; reinforce penalty "
                "on TOO_WIDE entries with fresh evidence."
            ),
            proposal_note="Penalize / defer wide-spread entries more strongly.",
            review_priority="HIGH",
            related_bucket="TOO_WIDE",
        ),
        # 3) Different target — applied independently.
        _proposal(
            proposal_id="ADAPT-1003",
            adaptation_target="trim_profit_threshold",
            proposal_type="DECREASE_TRIM_THRESHOLD",
            recommendation_type="EDGE_VALIDATION",
            source_recommendation_text=(
                "HIGH_CONVICTION positions trimmed too late; lowering the "
                "trim threshold captured +8% more realized pnl in backtest."
            ),
            evidence_count=22,
            evidence_strength="HIGH",
            recommendation_confidence=0.82,
            observed_group="HIGH_CONVICTION",
            observed_value="n=22, avg_pnl=+184, realized_vs_peak=-5.8%",
            effect_direction="DECREASE",
            proposed_delta=-0.04,
            proposal_direction="DECREASE",
            proposal_strength="MODERATE",
            proposal_confidence=0.76,
            min_allowed_value=-0.10,
            max_allowed_value=0.0,
            proposal_reason=(
                "Tighten trim threshold for high-conviction "
                "buckets to capture more realized pnl."
            ),
            proposal_note="Lower trim_profit_threshold by a small amount.",
            review_priority="HIGH",
            related_bucket="HIGH_CONVICTION",
        ),
        # 4) Proposed only — never approved, should stay unapplied.
        _proposal(
            proposal_id="ADAPT-1004",
            adaptation_target="stale_quote_penalty",
            proposal_type="INCREASE_CAUTION",
            recommendation_type="QUOTE_FRESHNESS_WARNING",
            source_recommendation_text=(
                "Entries with stale quotes showed elevated slippage (n=9, "
                "avg_slip_bps=48); evidence is still thin."
            ),
            evidence_count=9,
            evidence_strength="MEDIUM",
            recommendation_confidence=0.58,
            observed_group="STALE",
            observed_value="n=9, avg_pnl=-14, avg_slip_bps=48",
            effect_direction="INCREASE",
            proposed_delta=0.06,
            proposal_direction="INCREASE",
            proposal_strength="WEAK",
            proposal_confidence=0.52,
            min_allowed_value=0.0,
            max_allowed_value=0.20,
            proposal_reason=(
                "Mapped from QUOTE_FRESHNESS_WARNING; tentative " "caution on stale-quote entries."
            ),
            proposal_note="Increase caution when entry quotes are stale.",
            review_priority="MEDIUM",
            related_flag="STALE",
        ),
        # 5) Approved but out-of-bounds → SKIP delta_above_max.
        _proposal(
            proposal_id="ADAPT-1005",
            adaptation_target="add_score_threshold",
            proposal_type="INCREASE_SCORE_THRESHOLD",
            recommendation_type="SIGNAL_CAUTION",
            source_recommendation_text=(
                "Low-score 'ADD' signals underperform; proposal suggests a "
                "large threshold bump, but that exceeds the allow-listed range."
            ),
            evidence_count=7,
            evidence_strength="MEDIUM",
            recommendation_confidence=0.62,
            observed_group="WEAK_ADD",
            observed_value="n=7, avg_pnl=-22",
            effect_direction="INCREASE",
            # Intentionally out-of-bounds: max_allowed_value=0.10 but delta=0.25.
            proposed_delta=0.25,
            proposal_direction="INCREASE",
            proposal_strength="STRONG",
            proposal_confidence=0.60,
            min_allowed_value=0.0,
            max_allowed_value=0.10,
            proposal_reason=(
                "Raise ADD-signal score threshold for weak "
                "additions — intentionally out-of-bounds to "
                "demonstrate the apply-layer guardrail."
            ),
            proposal_note="Raise add_score_threshold only inside [0.0, 0.10].",
            review_priority="MEDIUM",
            related_flag="WEAK_ADD",
        ),
        # 6) Approved & applied, then rolled back in a later step.
        _proposal(
            proposal_id="ADAPT-1006",
            adaptation_target="low_confidence_entry_penalty",
            proposal_type="INCREASE_PENALTY",
            recommendation_type="SIGNAL_CAUTION",
            source_recommendation_text=(
                "Low-confidence entries (conf<0.55) show higher fail rates."
            ),
            evidence_count=18,
            evidence_strength="HIGH",
            recommendation_confidence=0.78,
            observed_group="LOW_CONF",
            observed_value="n=18, avg_pnl=-29, win_rate=0.41",
            effect_direction="INCREASE",
            proposed_delta=0.10,
            proposal_direction="INCREASE",
            proposal_strength="MODERATE",
            proposal_confidence=0.72,
            min_allowed_value=0.0,
            max_allowed_value=0.20,
            proposal_reason=(
                "Mapped from SIGNAL_CAUTION; penalize entries "
                "with recommendation_confidence < 0.55."
            ),
            proposal_note="Increase penalty on low-confidence entries.",
            review_priority="HIGH",
            related_flag="LOW_CONF",
        ),
        # Bonus: one extra for richer dashboards — proposed only.
        _proposal(
            proposal_id="ADAPT-1007",
            adaptation_target="position_cooldown_bias",
            proposal_type="ADJUST_COOLDOWN",
            recommendation_type="EDGE_VALIDATION",
            source_recommendation_text=(
                "Re-entries within 2 sessions of a stop-out slightly "
                "underperform; modest cooldown bias proposed."
            ),
            evidence_count=12,
            evidence_strength="MEDIUM",
            recommendation_confidence=0.66,
            observed_group="POST_STOP_OUT",
            observed_value="n=12, avg_pnl=-8",
            effect_direction="INCREASE",
            proposed_delta=0.05,
            proposal_direction="INCREASE",
            proposal_strength="MODERATE",
            proposal_confidence=0.58,
            min_allowed_value=0.0,
            max_allowed_value=0.15,
            proposal_reason=(
                "Mapped from EDGE_VALIDATION; slightly bias " "cooldown after stop-out events."
            ),
            proposal_note="Modest cooldown bias for post-stop-out reentries.",
            review_priority="MEDIUM",
            related_flag="POST_STOP_OUT",
        ),
    ]
    return pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)


# ──────────────────────────────────────────────────────────────
# Approval queue helpers
# ──────────────────────────────────────────────────────────────


def _approval(pid: str, status: str, by: str, note: str) -> Dict[str, Any]:
    return {
        "proposal_id": pid,
        "approval_status": status,
        "approval_note": note,
        "approved_by": by,
        "approved_at_utc": _utc_now_iso(),
    }


def _write_approval_queue(rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows, columns=APPROVAL_COLUMNS)
    df.to_csv(APPROVAL_QUEUE_CSV, index=False)


# ──────────────────────────────────────────────────────────────
# Summary / review queue helpers
# ──────────────────────────────────────────────────────────────


def _write_review_queue(proposals: pd.DataFrame) -> None:
    """Priority-ordered subset for the review-queue CSV."""
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "": 3}
    df = proposals.copy()
    df["__pri"] = df["review_priority"].astype(str).str.upper().map(order).fillna(3)
    df["__conf"] = pd.to_numeric(df["proposal_confidence"], errors="coerce").fillna(0.0)
    df = (
        df.sort_values(["__pri", "__conf"], ascending=[True, False])
        .drop(columns=["__pri", "__conf"])
        .head(20)
        .reset_index(drop=True)
    )
    df.to_csv(REVIEW_QUEUE_CSV, index=False)


def _write_adaptation_summary(proposals: pd.DataFrame) -> None:
    """Match the shape services/adaptation_layer writes, so the dashboard is happy."""

    def _vc(col: str) -> Dict[str, int]:
        try:
            return proposals[col].astype(str).value_counts(dropna=False).to_dict()
        except Exception:
            return {}

    thin = 0
    try:
        thin = int(proposals["thin_data_flag"].astype(bool).sum())
    except Exception:
        thin = 0

    payload = {
        "generated_at_utc": _utc_now_iso(),
        "schema_version": 1,
        "advisory_only": True,
        "auto_apply_allowed": False,
        "phase": "1-advisory-proposals-only",
        "source_availability": {
            "feedback_recommendations": {
                "status": "simulated",
                "rows": int(proposals.shape[0]),
                "path": str(PROPOSALS_CSV),
            },
        },
        "missing_inputs": [],
        "proposal_count": int(proposals.shape[0]),
        "thin_data_proposal_count": thin,
        "proposal_count_by_target": _vc("adaptation_target"),
        "proposal_count_by_type": _vc("proposal_type"),
        "proposal_count_by_priority": _vc("review_priority"),
        "proposal_count_by_evidence_strength": _vc("evidence_strength"),
        "top_proposals": [],
        "adaptation_targets": [],
        "notes": [
            "This adaptation_summary.json was written by "
            "scripts/simulate_approval_flow.py for governance-loop "
            "end-to-end testing. No live trading logic was modified."
        ],
    }
    ADAPTATION_SUMMARY_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Apply-layer output reset (simulation only)
# ──────────────────────────────────────────────────────────────


def _reset_apply_layer_outputs() -> None:
    """Delete simulation-scope apply-layer artifacts so each simulation run starts clean."""
    for path in (APPLIED_CSV, APPLIED_JSON, APPLY_LOG_CSV, APPLY_SUMMARY_JSON):
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            print(f"  [reset] could not delete {path.name}: {e}")


# ──────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────


def _find_application_id_for(proposal_id: str) -> Optional[str]:
    """Best-effort lookup of the active application_id for a proposal_id."""
    if not APPLIED_CSV.exists():
        return None
    df, _ = al.load_csv_safe(APPLIED_CSV)
    if df is None or df.empty:
        return None
    try:
        mask_pid = df["proposal_id"].astype(str) == str(proposal_id)
        if "active_flag" in df.columns:
            mask_active = (
                df["active_flag"]
                .apply(lambda v: al._safe_bool(v, default=False))
                .fillna(False)
                .astype(bool)
            )
            hit = df[mask_pid & mask_active]
        else:
            hit = df[mask_pid]
        if hit.empty:
            return None
        return str(hit.iloc[0].get("application_id") or "")
    except Exception:
        return None


def _dump_counts(label: str, result: al.ApplyResult) -> None:
    c = result.counts
    print(
        f"  [{label}] "
        f"seen={c.get('proposals_seen',0)} "
        f"approved={c.get('proposals_approved',0)} "
        f"applied={c.get('proposals_applied',0)} "
        f"skipped={c.get('proposals_skipped',0)} "
        f"superseded={c.get('supersessions',0)}"
    )


def _step_a() -> al.ApplyResult:
    print("\n=== Step A: initial approvals ===")
    approvals = [
        _approval(
            "ADAPT-1001",
            "APPROVED",
            "akim",
            "Approve TOO_WIDE spread penalty after SPREAD_CAUTION review.",
        ),
        _approval(
            "ADAPT-1003",
            "APPROVED",
            "operator_review",
            "Approve tighter trim threshold for HIGH_CONVICTION.",
        ),
        _approval(
            "ADAPT-1005",
            "APPROVED",
            "risk_review",
            "Approve add-score threshold bump (bounds will catch it).",
        ),
        _approval("ADAPT-1006", "APPROVED", "akim", "Approve low-confidence entry penalty."),
    ]
    _write_approval_queue(approvals)
    out = al.run_apply_layer(verbose=False)
    res: al.ApplyResult = out["result"]
    _dump_counts("Step A", res)
    return res


def _step_b() -> al.ApplyResult:
    print("\n=== Step B: add ADAPT-1002 (supersedes ADAPT-1001) ===")
    approvals = [
        _approval(
            "ADAPT-1001", "APPROVED", "akim", "Initially approved (now expected to be superseded)."
        ),
        _approval(
            "ADAPT-1002",
            "APPROVED",
            "operator_review",
            "Stronger evidence — approve and supersede ADAPT-1001.",
        ),
        _approval(
            "ADAPT-1003",
            "APPROVED",
            "operator_review",
            "Re-approved (idempotent — already applied).",
        ),
        _approval(
            "ADAPT-1005",
            "APPROVED",
            "risk_review",
            "Re-approved (still out-of-bounds, will skip again).",
        ),
        _approval("ADAPT-1006", "APPROVED", "akim", "Re-approved (idempotent — already applied)."),
    ]
    _write_approval_queue(approvals)
    out = al.run_apply_layer(verbose=False)
    res: al.ApplyResult = out["result"]
    _dump_counts("Step B", res)
    return res


def _step_c() -> Optional[al.RollbackResult]:
    print("\n=== Step C: rollback ADAPT-1006 ===")
    app_id = _find_application_id_for("ADAPT-1006")
    if not app_id:
        print("  [Step C] SKIPPED: could not locate active application for ADAPT-1006.")
        return None
    print(f"  [Step C] rolling back application_id={app_id}")
    rb = al.rollback_application(
        app_id,
        reason="simulation: operator retracted approval after review",
    )
    print(f"  [Step C] rolled_back={rb.rolled_back} reason={rb.reason}")
    return rb


# ──────────────────────────────────────────────────────────────
# Post-run verification
# ──────────────────────────────────────────────────────────────


def _verify_outputs() -> Tuple[bool, List[str]]:
    """Check the invariants listed in the spec. Returns (all_ok, messages)."""
    msgs: List[str] = []
    ok = True

    # Applied CSV invariants
    if not APPLIED_CSV.exists():
        msgs.append("FAIL: applied_adjustments.csv not written.")
        return False, msgs
    applied = pd.read_csv(APPLIED_CSV)
    statuses = (
        applied["status"].astype(str).str.upper().value_counts().to_dict()
        if "status" in applied.columns
        else {}
    )
    msgs.append(f"applied_adjustments.csv → {applied.shape[0]} rows; status counts={statuses}")

    if statuses.get("APPLIED", 0) < 1:
        ok = False
        msgs.append("FAIL: no APPLIED rows found.")
    if statuses.get("INACTIVE", 0) < 1:
        ok = False
        msgs.append("FAIL: no INACTIVE rows (supersession didn't happen).")
    if statuses.get("ROLLED_BACK", 0) < 1:
        ok = False
        msgs.append("FAIL: no ROLLED_BACK rows.")

    # Apply log invariants
    if not APPLY_LOG_CSV.exists():
        ok = False
        msgs.append("FAIL: apply_log.csv not written.")
    else:
        log_df = pd.read_csv(APPLY_LOG_CSV)
        ev = (
            log_df["event_type"].astype(str).str.upper().value_counts().to_dict()
            if "event_type" in log_df.columns
            else {}
        )
        msgs.append(f"apply_log.csv → {log_df.shape[0]} rows; event counts={ev}")
        if ev.get("APPLY", 0) < 1:
            ok = False
            msgs.append("FAIL: no APPLY events in apply_log.csv.")
        if ev.get("SKIP", 0) < 1:
            ok = False
            msgs.append("FAIL: no SKIP events in apply_log.csv.")
        if ev.get("SUPERSEDE", 0) < 1:
            ok = False
            msgs.append("FAIL: no SUPERSEDE events in apply_log.csv.")
        if ev.get("ROLLBACK", 0) < 1:
            ok = False
            msgs.append("FAIL: no ROLLBACK events in apply_log.csv.")
        # Confirm at least one SKIP has a guardrail reason.
        skips = (
            log_df[log_df.get("event_type").astype(str).str.upper() == "SKIP"]
            if "event_type" in log_df.columns
            else pd.DataFrame()
        )
        guardrail_hits = (
            skips["reason"]
            .astype(str)
            .isin(
                [
                    "delta_above_max",
                    "delta_below_min",
                    "invalid_bounds",
                ]
            )
            .sum()
            if not skips.empty and "reason" in skips.columns
            else 0
        )
        if guardrail_hits < 1:
            ok = False
            msgs.append("FAIL: no guardrail-reason SKIP event found.")
        else:
            msgs.append(f"guardrail SKIP count: {guardrail_hits}")

    # Apply summary invariants
    if not APPLY_SUMMARY_JSON.exists():
        ok = False
        msgs.append("FAIL: apply_summary.json not written.")
    else:
        summary = json.loads(APPLY_SUMMARY_JSON.read_text(encoding="utf-8"))
        sa = summary.get("source_availability") or {}
        msgs.append(
            f"apply_summary.json → source_availability keys={list(sa.keys())}; "
            f"advisory_only_source={summary.get('advisory_only_source')}, "
            f"auto_apply_allowed={summary.get('auto_apply_allowed')}, "
            f"approvals_source={summary.get('approvals_source')}"
        )
        if summary.get("advisory_only_source") is not True:
            ok = False
            msgs.append("FAIL: advisory_only_source is not True.")
        if summary.get("auto_apply_allowed") is not False:
            ok = False
            msgs.append("FAIL: auto_apply_allowed is not False.")
        if not sa:
            ok = False
            msgs.append("FAIL: source_availability is empty.")

    # Applied JSON mirror
    if not APPLIED_JSON.exists():
        ok = False
        msgs.append("FAIL: applied_adjustments.json not written.")
    else:
        mirror = json.loads(APPLIED_JSON.read_text(encoding="utf-8"))
        active = len(mirror.get("active_adjustments") or [])
        allr = len(mirror.get("all_adjustments") or [])
        msgs.append(f"applied_adjustments.json → active={active}, all={allr}")

    return ok, msgs


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def main() -> int:
    print("== Triton governance-loop simulation ==")
    print(f"results dir: {RESULTS}")

    print("\n[1/4] writing simulation inputs")
    RESULTS.mkdir(parents=True, exist_ok=True)
    proposals = _build_simulation_proposals()
    proposals.to_csv(PROPOSALS_CSV, index=False)
    print(f"  wrote {PROPOSALS_CSV.name} ({proposals.shape[0]} proposals)")
    _write_review_queue(proposals)
    print(f"  wrote {REVIEW_QUEUE_CSV.name}")
    _write_adaptation_summary(proposals)
    print(f"  wrote {ADAPTATION_SUMMARY_JSON.name}")

    print("\n[2/4] resetting apply-layer outputs for a clean simulation run")
    _reset_apply_layer_outputs()

    print("\n[3/4] running apply-layer in three steps")
    _step_a()
    _step_b()
    _step_c()

    print("\n[4/4] verifying outputs")
    ok, msgs = _verify_outputs()
    for m in msgs:
        print("  " + m)
    print("\nVERIFICATION:", "PASS" if ok else "FAIL")

    print("\nFiles written:")
    for p in (
        PROPOSALS_CSV,
        REVIEW_QUEUE_CSV,
        ADAPTATION_SUMMARY_JSON,
        APPROVAL_QUEUE_CSV,
        APPLIED_CSV,
        APPLIED_JSON,
        APPLY_LOG_CSV,
        APPLY_SUMMARY_JSON,
    ):
        exists = "ok" if p.exists() else "missing"
        print(f"  {exists:>7}  {p}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
