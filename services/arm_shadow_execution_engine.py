"""
ARM Shadow Execution Engine -- Step 29.

Reads:
    data/results/arm_mode_governance_summary.json                 (Step 28)
    data/results/arm_mode_governance.json                         (Step 28)
    data/results/autonomous_execution_certificate_summary.json    (Step 27)
    data/results/autonomous_execution_plan.json                   (Step 24)
    data/results/autonomous_execution_plan_summary.json           (Step 24)
    data/results/autonomous_execution_simulation_summary.json     (Step 25)
    data/results/runtime_policy_governed.json                     (Step 18)
    data/results/adaptive_regime.json                             (Step 10)
    data/results/meta_decision_intelligence.json                  (Step 13)

Writes:
    data/results/arm_shadow_execution.json
    data/results/arm_shadow_execution.md
    data/results/arm_shadow_execution_summary.json
    data/results/arm_shadow_execution_memory.csv
    data/results/arm_shadow_execution_memory.parquet

Purpose
-------
This engine answers:

    "What would Triton have done autonomously?"

It is a *pure observational layer*. It places no orders, contacts
no broker, mutates no portfolio state, and never imports any
trade-execution module. Its sole purpose is to project what an
ARM-mode execution engine *would* have done at this point in time,
persist that projection in append-only memory, and surface learning
hooks for future enrichment with realized forward returns.

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
The exact forbidden token list is defined in the Step 29 spec
(Triton evolution plan); to keep the file grep-clean those tokens
are never written literally in this source.

Self-inspection at import time (``_self_check_no_broker_tokens``)
verifies the file body contains zero matches against a tokenized
deny-list, raising at import time if the contract is ever violated.

Shadow execution states (spec section 1)
----------------------------------------
    SHADOW_DISABLED      MANUAL mode + blocked cert + stale system,
                         or no plan to evaluate. Nothing to simulate.
    SHADOW_OBSERVATION   MANUAL / ASSISTED / AUTO_DISABLED with plan
                         rows present. Log hypothetical actions with
                         would_execute=False.
    SHADOW_ASSISTED      ASSISTED mode with at least one allowed plan
                         row. Simulate selective deployment with
                         operator-confirmation flag set.
    SHADOW_AUTO          AUTO_ALLOWED with at least one allowed plan
                         row. Simulate fully autonomous execution
                         (would_execute=True on allowed rows).

Memory schema (append-only, deduplicated by cycle_id+ticker+action)
-------------------------------------------------------------------
    cycle_id                 issuance timestamp (anchor for dedup)
    timestamp_utc            row timestamp (== cycle_id in practice)
    ticker
    action                   buy_new, add, sell, trim, etc.
    target_weight
    estimated_notional
    plan_confidence
    shadow_mode              SHADOW_AUTO / SHADOW_ASSISTED / ...
    rationale                lifted from plan row if present
    execution_mode           Step 24 plan execution_mode
    authorization_state      Step 23 authorization_state
    would_execute            True only in SHADOW_AUTO on allowed rows
    requires_operator_confirmation
    blocked_reason           when would_execute=False
    regime
    trust_level
    governance_state
    runtime_policy_snapshot_json
    future_return_1d         null until future enrichment
    future_return_5d         null until future enrichment
    future_return_20d        null until future enrichment
    outcome_known            False at write time

Safety
------
* STRICT READ ONLY. No broker calls, no execution mutation, no
  trade-API imports.
* Atomic writes (.tmp + os.replace) for JSON, MD, CSV, Parquet.
* Append-only memory with deduplication.
* Missing inputs warn-and-continue. With no plan the state is
  SHADOW_DISABLED and zero rows are emitted.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_ARM_SUMMARY = RESULTS_DIR / "arm_mode_governance_summary.json"
DEFAULT_ARM_RECORD = RESULTS_DIR / "arm_mode_governance.json"
DEFAULT_CERT_SUMMARY = RESULTS_DIR / "autonomous_execution_certificate_summary.json"
DEFAULT_PLAN_JSON = RESULTS_DIR / "autonomous_execution_plan.json"
DEFAULT_PLAN_SUMMARY = RESULTS_DIR / "autonomous_execution_plan_summary.json"
DEFAULT_SIM_SUMMARY = RESULTS_DIR / "autonomous_execution_simulation_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_META_DECISION = RESULTS_DIR / "meta_decision_intelligence.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_shadow_execution.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_shadow_execution.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_shadow_execution_summary.json"
DEFAULT_MEMORY_CSV = RESULTS_DIR / "arm_shadow_execution_memory.csv"
DEFAULT_MEMORY_PARQUET = RESULTS_DIR / "arm_shadow_execution_memory.parquet"


# -----------------------------------------------------------
# State constants
# -----------------------------------------------------------
SHADOW_DISABLED = "SHADOW_DISABLED"
SHADOW_OBSERVATION = "SHADOW_OBSERVATION"
SHADOW_ASSISTED = "SHADOW_ASSISTED"
SHADOW_AUTO = "SHADOW_AUTO"

ALL_STATES: Tuple[str, ...] = (
    SHADOW_DISABLED,
    SHADOW_OBSERVATION,
    SHADOW_ASSISTED,
    SHADOW_AUTO,
)

ARM_MANUAL = "MANUAL"
ARM_ASSISTED = "ASSISTED"
ARM_AUTO_DISABLED = "AUTO_DISABLED"
ARM_AUTO_ALLOWED = "AUTO_ALLOWED"

CERT_BLOCKED = "EXECUTION_BLOCKED"
CERT_DENIED = "EXECUTION_DENIED"

# Memory schema (CSV header order)
MEMORY_COLUMNS: Tuple[str, ...] = (
    "cycle_id",
    "timestamp_utc",
    "ticker",
    "action",
    "target_weight",
    "estimated_notional",
    "plan_confidence",
    "shadow_mode",
    "rationale",
    "execution_mode",
    "authorization_state",
    "would_execute",
    "requires_operator_confirmation",
    "blocked_reason",
    "regime",
    "trust_level",
    "governance_state",
    "runtime_policy_snapshot_json",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "outcome_known",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_SHADOW_WARN] {msg}", flush=True)


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
            row_out = {c: ("" if r.get(c) is None else r.get(c)) for c in columns}
            w.writerow(row_out)
    os.replace(tmp, path)


def _atomic_write_parquet(rows: List[Dict[str, Any]], path: Path) -> bool:
    """
    Best-effort parquet write. Returns True on success.

    We avoid hard-depending on pandas/pyarrow -- if either is missing,
    we warn-and-continue (CSV mirror is the authoritative artefact).
    """
    if not rows:
        # Still emit an empty parquet for downstream consistency
        rows = []
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        _warn(f"pandas unavailable for parquet write: {type(e).__name__}: {e}")
        return False
    try:
        df = pd.DataFrame(rows, columns=list(MEMORY_COLUMNS))
        # Coerce numeric columns explicitly so pyarrow doesn't choke
        # on mixed-dtype object columns (NaN vs empty string).
        for col in (
            "target_weight",
            "estimated_notional",
            "plan_confidence",
            "future_return_1d",
            "future_return_5d",
            "future_return_20d",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("would_execute", "requires_operator_confirmation", "outcome_known"):
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


def _norm_upper(x: Any, default: str = "UNKNOWN") -> str:
    s = str(x or "").strip().upper()
    return s or default


def _norm_symbol(s: Any) -> str:
    return str(s or "").strip().upper()


# -----------------------------------------------------------
# Shadow state classification
# -----------------------------------------------------------
def _classify_shadow_state(
    *,
    arm_mode: str,
    cert_state: str,
    system_health: str,
    has_plan: bool,
    n_allowed_rows: int,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    # 1. SHADOW_DISABLED
    if not has_plan:
        reasons.append("no execution plan to simulate")
        return SHADOW_DISABLED, reasons
    if cert_state in (CERT_BLOCKED, CERT_DENIED):
        reasons.append(f"certification_state={cert_state}")
        return SHADOW_DISABLED, reasons
    if system_health in ("CRITICAL", "OFFLINE"):
        reasons.append(f"system_health={system_health}")
        return SHADOW_DISABLED, reasons

    # 2. SHADOW_AUTO -- fully autonomous simulation
    if arm_mode == ARM_AUTO_ALLOWED and n_allowed_rows > 0:
        reasons.append(f"arm_mode=AUTO_ALLOWED with {n_allowed_rows} authorized plan row(s)")
        return SHADOW_AUTO, reasons

    # 3. SHADOW_ASSISTED -- hypothetical selective deployment
    if arm_mode == ARM_ASSISTED and n_allowed_rows > 0:
        reasons.append(f"arm_mode=ASSISTED with {n_allowed_rows} authorized plan row(s)")
        return SHADOW_ASSISTED, reasons

    # 4. SHADOW_OBSERVATION -- plan present but no autonomy
    reasons.append(f"arm_mode={arm_mode}; logging plan rows without execution authority")
    return SHADOW_OBSERVATION, reasons


# -----------------------------------------------------------
# Per-row blocked-reason derivation
# -----------------------------------------------------------
def _derive_blocked_reason(
    *,
    shadow_state: str,
    arm_mode: str,
    row_allowed: bool,
    plan_blocked_reason: Optional[str],
) -> Optional[str]:
    if shadow_state == SHADOW_AUTO and row_allowed:
        return None  # would_execute is True
    if not row_allowed:
        return f"plan_row_not_allowed:{plan_blocked_reason or 'unspecified'}"
    if shadow_state == SHADOW_DISABLED:
        return f"shadow_disabled (arm_mode={arm_mode})"
    if shadow_state == SHADOW_OBSERVATION:
        return f"shadow_observation_only (arm_mode={arm_mode}); no autonomy"
    if shadow_state == SHADOW_ASSISTED:
        return "shadow_assisted (operator confirmation required for execution)"
    return f"shadow_state={shadow_state}; no execution authority"


# -----------------------------------------------------------
# Runtime policy snapshot (compact)
# -----------------------------------------------------------
def _runtime_policy_snapshot(runtime_policy: Dict[str, Any]) -> Dict[str, Any]:
    if not runtime_policy:
        return {}
    return {
        "regime": runtime_policy.get("regime"),
        "policy_version": runtime_policy.get("policy_version"),
        "max_position_pct": runtime_policy.get("max_position_pct"),
        "target_cash_pct": runtime_policy.get("target_cash_pct"),
        "confidence_threshold": runtime_policy.get("confidence_threshold"),
        "persistence_threshold": runtime_policy.get("persistence_threshold"),
        "deployment_threshold": runtime_policy.get("deployment_threshold"),
        "skepticism_threshold": runtime_policy.get("skepticism_threshold"),
        "generated_at_utc": runtime_policy.get("generated_at_utc"),
    }


# -----------------------------------------------------------
# Build shadow records
# -----------------------------------------------------------
def _build_shadow_records(
    *,
    cycle_id: str,
    plan: Dict[str, Any],
    plan_summary: Dict[str, Any],
    arm_mode: str,
    shadow_state: str,
    auth_state: str,
    regime: str,
    trust_level: str,
    governance_state: str,
    runtime_policy_snapshot: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = plan.get("actions") or []
    plan_confidence = (
        _to_float(plan.get("plan_confidence"))
        or _to_float(plan_summary.get("plan_confidence"))
        or 0.0
    )
    execution_mode = str(plan.get("execution_mode") or plan_summary.get("execution_mode") or "")
    snapshot_json = json.dumps(runtime_policy_snapshot, default=_json_safe, sort_keys=True)

    rows: List[Dict[str, Any]] = []
    for action in actions:
        ticker = _norm_symbol(action.get("ticker"))
        if not ticker:
            continue
        action_name = str(action.get("action") or "").strip()
        row_allowed = bool(action.get("allowed"))
        plan_blocked_reason = action.get("blocked_reason")

        blocked_reason = _derive_blocked_reason(
            shadow_state=shadow_state,
            arm_mode=arm_mode,
            row_allowed=row_allowed,
            plan_blocked_reason=(
                plan_blocked_reason if isinstance(plan_blocked_reason, str) else None
            ),
        )
        would_execute = shadow_state == SHADOW_AUTO and row_allowed
        requires_operator = shadow_state == SHADOW_ASSISTED and row_allowed

        rows.append(
            {
                "cycle_id": cycle_id,
                "timestamp_utc": cycle_id,
                "ticker": ticker,
                "action": action_name,
                "target_weight": _to_float(action.get("target_weight")),
                "estimated_notional": _to_float(action.get("estimated_notional_usd")),
                "plan_confidence": plan_confidence,
                "shadow_mode": shadow_state,
                "rationale": str(action.get("rationale") or "").strip(),
                "execution_mode": execution_mode,
                "authorization_state": auth_state,
                "would_execute": would_execute,
                "requires_operator_confirmation": requires_operator,
                "blocked_reason": blocked_reason or "",
                "regime": regime,
                "trust_level": trust_level,
                "governance_state": governance_state,
                "runtime_policy_snapshot_json": snapshot_json,
                # Learning hooks (null until future enrichment)
                "future_return_1d": None,
                "future_return_5d": None,
                "future_return_20d": None,
                "outcome_known": False,
            }
        )
    return rows


# -----------------------------------------------------------
# Append-only memory dedup
# -----------------------------------------------------------
def _merge_memory(
    existing_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Append new shadow rows to the existing memory. New observations
    for the same (cycle_id, ticker, action) tuple overwrite older
    entries (so re-running a cycle replaces partial writes).
    """
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in existing_rows:
        key = (str(r.get("cycle_id", "")), str(r.get("ticker", "")), str(r.get("action", "")))
        out[key] = r
    for r in new_rows:
        key = (str(r.get("cycle_id", "")), str(r.get("ticker", "")), str(r.get("action", "")))
        out[key] = r
    # Preserve insertion order: existing first, then new replaces in place
    merged = list(out.values())
    # Ensure every row has every memory column
    for r in merged:
        for c in MEMORY_COLUMNS:
            r.setdefault(c, None)
    return merged


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    shadow_state: str,
    arm_mode: str,
    n_rows: int,
    n_would_execute: int,
    confidence: float,
) -> List[str]:
    recs: List[str] = []
    if shadow_state == SHADOW_DISABLED:
        recs.append("Continue manual supervision -- no hypothetical execution recorded.")
        recs.append("Refresh the pipeline before any future shadow cycle.")
        return recs
    if shadow_state == SHADOW_OBSERVATION:
        recs.append("Continue shadow observation -- log hypothetical actions only.")
        recs.append("Compare shadow decisions against the live portfolio for divergence analysis.")
        if arm_mode == ARM_AUTO_DISABLED:
            recs.append("Increase governance maturity before lifting autonomy.")
        return recs
    if shadow_state == SHADOW_ASSISTED:
        recs.append(
            "Permit assisted-only observation -- record selective deployment with operator confirmation."
        )
        recs.append(
            f"{n_rows} hypothetical actions captured; {n_would_execute} would execute with confirmation."
        )
        return recs
    # SHADOW_AUTO
    recs.append("Shadow autonomous execution recorded -- compare against live portfolio outcomes.")
    if confidence < 0.70:
        recs.append(
            "Shadow confidence is moderate -- monitor decision quality before lifting to real execution."
        )
    recs.append(f"{n_would_execute}/{n_rows} hypothetical actions marked would_execute=True.")
    return recs


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    shadow_state: str,
    arm_mode: str,
    rows: List[Dict[str, Any]],
    n_would_execute: int,
    reasons: List[str],
    recommendations: List[str],
    confidence: float,
    regime: str,
    auth_state: str,
    execution_mode: str,
    trust_level: str,
    governance_state: str,
) -> str:
    def fmt_money(x: Optional[float]) -> str:
        try:
            v = float(x or 0.0)
        except Exception:
            return "-"
        return f"${v:,.0f}"

    def fmt_pct(x: Optional[float]) -> str:
        try:
            v = float(x or 0.0) * 100.0
        except Exception:
            return "-"
        return f"{v:.2f}%"

    lines: List[str] = []
    lines.append("# Triton ARM Shadow Execution")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Shadow State")
    lines.append("")
    lines.append(f"**{shadow_state}**")
    lines.append("")
    lines.append("| input | value |")
    lines.append("|---|---|")
    lines.append(f"| arm_mode | {arm_mode} |")
    lines.append(f"| plan_execution_mode | {execution_mode or '-'} |")
    lines.append(f"| authorization_state | {auth_state or '-'} |")
    lines.append(f"| regime | {regime} |")
    lines.append(f"| trust_level | {trust_level} |")
    lines.append(f"| governance_state | {governance_state} |")
    lines.append(f"| shadow_confidence | {confidence:.3f} |")
    lines.append(f"| total_rows | {len(rows)} |")
    lines.append(f"| would_execute_rows | {n_would_execute} |")
    lines.append("")

    lines.append("## Hypothetical Actions")
    lines.append("")
    if rows:
        lines.append(
            "| ticker | action | target_w | notional | would_execute | requires_confirm | blocked_reason |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['ticker']} | {r['action']} | "
                f"{fmt_pct(r.get('target_weight'))} | "
                f"{fmt_money(r.get('estimated_notional'))} | "
                f"{r.get('would_execute')} | "
                f"{r.get('requires_operator_confirmation')} | "
                f"{(r.get('blocked_reason') or '-').replace('|', ' ')} |"
            )
    else:
        lines.append("_(no plan rows to evaluate)_")
    lines.append("")

    lines.append("## What Triton Would Have Done")
    lines.append("")
    if shadow_state == SHADOW_DISABLED:
        lines.append(
            "- Triton would have remained observational. No hypothetical "
            "actions are recorded because the plan is empty or the "
            "certificate is blocked."
        )
    elif shadow_state == SHADOW_OBSERVATION:
        lines.append(
            f"- Triton would have *considered* {len(rows)} action(s) but "
            "taken none autonomously. All rows are logged as observational only."
        )
    elif shadow_state == SHADOW_ASSISTED:
        lines.append(
            f"- Triton would have proposed {n_would_execute} hypothetical "
            "selective deployment action(s), each requiring operator confirmation "
            "before any real execution could occur."
        )
    elif shadow_state == SHADOW_AUTO:
        lines.append(
            f"- Triton would have autonomously executed {n_would_execute} "
            "action(s) within the active certificate window. "
            "Forward returns will be backfilled in a future enrichment pass."
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
    if shadow_state == SHADOW_DISABLED:
        narrative = (
            f"Shadow execution disabled. There is no plan to simulate, or the "
            f"certificate/system blocks even hypothetical actions. Confidence "
            f"{confidence:.2f}."
        )
    elif shadow_state == SHADOW_OBSERVATION:
        narrative = (
            f"Shadow observation only. Triton would have considered {len(rows)} "
            f"action(s) but executed none, because ARM mode {arm_mode} grants no "
            f"autonomous authority. Confidence {confidence:.2f}."
        )
    elif shadow_state == SHADOW_ASSISTED:
        narrative = (
            f"Shadow assisted execution recorded. {n_would_execute} hypothetical "
            f"action(s) would have proceeded with operator confirmation under "
            f"ARM mode {arm_mode}. Confidence {confidence:.2f}."
        )
    else:
        narrative = (
            f"Shadow autonomous execution recorded under ARM mode {arm_mode}. "
            f"{n_would_execute}/{len(rows)} hypothetical action(s) would have "
            f"executed; forward returns pending enrichment. Confidence "
            f"{confidence:.2f}."
        )
    lines.append(narrative)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Confidence (mirrors Step 28 autonomy confidence when available)
# -----------------------------------------------------------
def _shadow_confidence(
    *,
    arm_record: Dict[str, Any],
    cert_summary: Dict[str, Any],
    plan: Dict[str, Any],
    plan_summary: Dict[str, Any],
) -> float:
    c = _to_float(arm_record.get("autonomy_confidence"))
    if c is None:
        c = _to_float(cert_summary.get("certificate_confidence"))
    if c is None:
        c = _to_float(plan.get("plan_confidence")) or _to_float(plan_summary.get("plan_confidence"))
    if c is None:
        c = 0.0
    return max(0.0, min(1.0, c))


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_shadow_execution(
    *,
    arm_summary: Dict[str, Any],
    arm_record: Dict[str, Any],
    cert_summary: Dict[str, Any],
    plan: Dict[str, Any],
    plan_summary: Dict[str, Any],
    sim_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    regime_json: Dict[str, Any],
    meta_decision: Dict[str, Any],
    existing_memory_rows: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    cycle_id = _now_iso_utc()

    arm_mode = _norm_upper(arm_summary.get("arm_mode") or arm_record.get("arm_mode"))
    cert_state = _norm_upper(cert_summary.get("certification_state"))
    system_health = _norm_upper(
        cert_summary.get("system_health") or sim_summary.get("system_health")
    )
    auth_state = _norm_upper(
        (plan.get("authorization_state"))
        or (plan_summary.get("authorization_state"))
        or (sim_summary.get("authorization_state"))
    )
    regime = _norm_upper(
        (regime_json or {}).get("regime")
        or (runtime_policy or {}).get("regime")
        or plan_summary.get("regime")
    )
    trust_level = _norm_upper((meta_decision or {}).get("trust_level"))
    governance_state = str(
        (
            arm_summary.get("governance_state")
            or (arm_record.get("upstream_context") or {}).get("governance_trust_level")
            or "UNKNOWN"
        )
    )

    actions: List[Dict[str, Any]] = plan.get("actions") or []
    n_allowed = sum(1 for a in actions if a.get("allowed"))
    has_plan = bool(actions)

    shadow_state, reasons = _classify_shadow_state(
        arm_mode=arm_mode,
        cert_state=cert_state,
        system_health=system_health,
        has_plan=has_plan,
        n_allowed_rows=n_allowed,
    )

    runtime_snapshot = _runtime_policy_snapshot(runtime_policy)

    new_rows: List[Dict[str, Any]] = []
    if shadow_state != SHADOW_DISABLED:
        new_rows = _build_shadow_records(
            cycle_id=cycle_id,
            plan=plan,
            plan_summary=plan_summary,
            arm_mode=arm_mode,
            shadow_state=shadow_state,
            auth_state=auth_state,
            regime=regime,
            trust_level=trust_level,
            governance_state=governance_state,
            runtime_policy_snapshot=runtime_snapshot,
        )

    n_would_execute = sum(1 for r in new_rows if r.get("would_execute"))
    confidence = _shadow_confidence(
        arm_record=arm_record,
        cert_summary=cert_summary,
        plan=plan,
        plan_summary=plan_summary,
    )

    recommendations = _build_recommendations(
        shadow_state=shadow_state,
        arm_mode=arm_mode,
        n_rows=len(new_rows),
        n_would_execute=n_would_execute,
        confidence=confidence,
    )

    merged_memory = _merge_memory(list(existing_memory_rows), new_rows)

    record: Dict[str, Any] = {
        "generated_at_utc": cycle_id,
        "engine": "arm_shadow_execution_engine",
        "engine_version": 1,
        "shadow_state": shadow_state,
        "shadow_reasons": reasons,
        "arm_mode": arm_mode,
        "certification_state": cert_state,
        "authorization_state": auth_state,
        "execution_mode": str(
            plan.get("execution_mode") or plan_summary.get("execution_mode") or ""
        ),
        "regime": regime,
        "trust_level": trust_level,
        "governance_state": governance_state,
        "shadow_confidence": round(confidence, 6),
        "n_rows": len(new_rows),
        "n_would_execute": n_would_execute,
        "n_requires_operator_confirmation": sum(
            1 for r in new_rows if r.get("requires_operator_confirmation")
        ),
        "n_blocked": sum(1 for r in new_rows if not r.get("would_execute")),
        "runtime_policy_snapshot": runtime_snapshot,
        "shadow_rows": new_rows,
        "recommendations": recommendations,
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "shadow_only": True,
        },
        "inputs_seen": {
            "arm_mode_governance_summary": bool(arm_summary),
            "arm_mode_governance": bool(arm_record),
            "autonomous_execution_certificate_summary": bool(cert_summary),
            "autonomous_execution_plan": bool(plan),
            "autonomous_execution_plan_summary": bool(plan_summary),
            "autonomous_execution_simulation_summary": bool(sim_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "adaptive_regime": bool(regime_json),
            "meta_decision_intelligence": bool(meta_decision),
        },
        "memory_size_after_append": len(merged_memory),
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": cycle_id,
        "engine": "arm_shadow_execution_engine",
        "shadow_state": shadow_state,
        "arm_mode": arm_mode,
        "n_rows": len(new_rows),
        "n_would_execute": n_would_execute,
        "n_requires_operator_confirmation": sum(
            1 for r in new_rows if r.get("requires_operator_confirmation")
        ),
        "shadow_confidence": round(confidence, 6),
        "certification_state": cert_state,
        "authorization_state": auth_state,
        "execution_mode": record["execution_mode"],
        "regime": regime,
        "memory_size_after_append": len(merged_memory),
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        generated_at=cycle_id,
        shadow_state=shadow_state,
        arm_mode=arm_mode,
        rows=new_rows,
        n_would_execute=n_would_execute,
        reasons=reasons,
        recommendations=recommendations,
        confidence=confidence,
        regime=regime,
        auth_state=auth_state,
        execution_mode=record["execution_mode"],
        trust_level=trust_level,
        governance_state=governance_state,
    )
    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM shadow execution engine (Step 29). "
            "Simulates what Triton would have done autonomously, "
            "without touching the broker or mutating any portfolio "
            "state. Pure observational layer with append-only memory."
        ),
    )
    p.add_argument("--arm-summary", default=str(DEFAULT_ARM_SUMMARY))
    p.add_argument("--arm-record", default=str(DEFAULT_ARM_RECORD))
    p.add_argument("--cert-summary", default=str(DEFAULT_CERT_SUMMARY))
    p.add_argument("--plan", default=str(DEFAULT_PLAN_JSON))
    p.add_argument("--plan-summary", default=str(DEFAULT_PLAN_SUMMARY))
    p.add_argument("--sim-summary", default=str(DEFAULT_SIM_SUMMARY))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--meta-decision", default=str(DEFAULT_META_DECISION))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--memory-csv", default=str(DEFAULT_MEMORY_CSV))
    p.add_argument("--memory-parquet", default=str(DEFAULT_MEMORY_PARQUET))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[ARM_SHADOW] starting (read-only shadow execution; no broker calls)", flush=True)

    arm_summary = _safe_read_json(Path(args.arm_summary), label="arm_mode_governance_summary.json")
    arm_record = _safe_read_json(Path(args.arm_record), label="arm_mode_governance.json")
    cert_summary = _safe_read_json(
        Path(args.cert_summary), label="autonomous_execution_certificate_summary.json"
    )
    plan = _safe_read_json(Path(args.plan), label="autonomous_execution_plan.json")
    plan_summary = _safe_read_json(
        Path(args.plan_summary), label="autonomous_execution_plan_summary.json"
    )
    sim_summary = _safe_read_json(
        Path(args.sim_summary), label="autonomous_execution_simulation_summary.json"
    )
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    meta_decision = _safe_read_json(
        Path(args.meta_decision), label="meta_decision_intelligence.json"
    )

    existing_rows = _safe_read_csv_rows(
        Path(args.memory_csv), label="arm_shadow_execution_memory.csv"
    )

    record, summary, md, merged_memory = build_shadow_execution(
        arm_summary=arm_summary,
        arm_record=arm_record,
        cert_summary=cert_summary,
        plan=plan,
        plan_summary=plan_summary,
        sim_summary=sim_summary,
        runtime_policy=runtime_policy,
        regime_json=regime_json,
        meta_decision=meta_decision,
        existing_memory_rows=existing_rows,
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

    # Persist append-only memory (CSV authoritative; parquet best-effort)
    try:
        _atomic_write_csv(merged_memory, Path(args.memory_csv), columns=MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write memory CSV {args.memory_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.memory_parquet))

    print(
        "[ARM_SHADOW] "
        f"state={record['shadow_state']} "
        f"mode={record['arm_mode']} "
        f"rows={record['n_rows']} "
        f"would_execute={record['n_would_execute']} "
        f"confidence={record['shadow_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_SHADOW_SAFETY] broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_SHADOW_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()} "
        f"memory_csv={Path(args.memory_csv).as_posix()} "
        f"memory_parquet={Path(args.memory_parquet).as_posix() if parquet_ok else 'SKIPPED'}",
        flush=True,
    )
    return 0


# -----------------------------------------------------------
# Self-inspection: enforce the no-broker safety rule at import time
# -----------------------------------------------------------
# Forbidden tokens are constructed from disjoint fragments so the raw
# strings never appear in this source file's body -- this keeps a
# strict ``grep`` against this file returning zero hits while still
# letting us check for accidental reintroduction at import time.
_FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "Alpaca" + "Broker",
    "place" + "_order",
    "submit" + "_order",
    "execute" + "_trades",
    "place" + "_live_orders",
    "broker" + "_client",
)


def _self_check_no_broker_tokens() -> None:
    """
    Assert that this module's source text does not contain any of the
    forbidden broker/execution tokens listed in the Step 29 spec. This
    is a defensive check against accidental future edits that would
    violate the strict shadow-only contract.
    """
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return
    for tok in _FORBIDDEN_TOKENS:
        if tok in src:
            raise RuntimeError(f"[ARM_SHADOW_SAFETY] forbidden broker token detected: {tok!r}")


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
