"""
ARM Governance Doctrine Activation Authorization Engine -- Step 51.

Reads:
    data/results/arm_governance_doctrine_approval_board_summary.json          (Step 50)
    data/results/arm_governance_doctrine_approval_board.json                  (Step 50)
    data/results/arm_governance_doctrine_approval_board_memory.csv            (Step 50)
    data/results/arm_governance_doctrine_activation_consideration_summary.json (Step 49)
    data/results/arm_governance_doctrine_readiness_summary.json               (Step 48)
    data/results/arm_governance_doctrine_institutional_trust_summary.json     (Step 47)
    data/results/arm_constitutional_court_summary.json                        (Step 33)
    data/results/arm_supreme_governance_council_summary.json                 (Step 34)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/runtime_policy_governed.json                                 (Step 18)

Writes:
    data/results/arm_governance_doctrine_activation_authorization.json
    data/results/arm_governance_doctrine_activation_authorization.md
    data/results/arm_governance_doctrine_activation_authorization_summary.json
    data/results/arm_governance_doctrine_activation_authorization_memory.csv
    data/results/arm_governance_doctrine_activation_authorization_memory.parquet

Purpose
-------
This engine answers:

    "Is this approved doctrine authorized for future activation?"

It converts institutional doctrine approval into activation authorization.
Approved != authorized. Authorized != activated. Authorization != runtime mutation.
Authorization NEVER activates doctrine or mutates runtime policy.

Authorization state cascade
---------------------------
    1. DOCTRINE_AUTHORIZATION_INSTITUTIONAL  stable authorization quality; mature governance
    2. DOCTRINE_AUTHORIZATION_READY         full authorization for future activation
    3. DOCTRINE_AUTHORIZATION_LIMITED       limited authorization; constitutionally safe
    4. DOCTRINE_AUTHORIZATION_OBSERVE       observe only; authorization immature
    5. DOCTRINE_AUTHORIZATION_DORMANT       insufficient approval for authorization

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
* Append-only authorization memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_AUTHORIZATION_DORMANT.
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

DEFAULT_APPROVAL_SUM = RESULTS_DIR / "arm_governance_doctrine_approval_board_summary.json"
DEFAULT_APPROVAL_REC = RESULTS_DIR / "arm_governance_doctrine_approval_board.json"
DEFAULT_APPROVAL_MEM = RESULTS_DIR / "arm_governance_doctrine_approval_board_memory.csv"
DEFAULT_CONSIDERATION_SUM = (
    RESULTS_DIR / "arm_governance_doctrine_activation_consideration_summary.json"
)
DEFAULT_READINESS_SUM = RESULTS_DIR / "arm_governance_doctrine_readiness_summary.json"
DEFAULT_TRUST_SUM = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_AUTONOMOUS_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_activation_authorization.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_activation_authorization.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_activation_authorization_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_activation_authorization_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_activation_authorization_memory.parquet"


# -----------------------------------------------------------
# Authorization state constants
# -----------------------------------------------------------
AUTH_DORMANT = "DOCTRINE_AUTHORIZATION_DORMANT"
AUTH_OBSERVE = "DOCTRINE_AUTHORIZATION_OBSERVE"
AUTH_LIMITED = "DOCTRINE_AUTHORIZATION_LIMITED"
AUTH_READY = "DOCTRINE_AUTHORIZATION_READY"
AUTH_INSTITUTIONAL = "DOCTRINE_AUTHORIZATION_INSTITUTIONAL"

CLASS_NOT_AUTHORIZED = "NOT_AUTHORIZED"
CLASS_OBSERVE_AUTH = "OBSERVE_ONLY_AUTHORIZATION"
CLASS_LIMITED_AUTH = "LIMITED_AUTHORIZATION"
CLASS_FULL_AUTH = "FULL_AUTHORIZATION"

APPROVAL_NOT = "NOT_APPROVED"
APPROVAL_OBSERVE = "OBSERVE_CONTINUED"
APPROVAL_LIMITED = "LIMITED_APPROVAL"
APPROVAL_INSTITUTIONAL = "INSTITUTIONAL_APPROVAL"

AUTH_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "authorization_state",
    "observe_count",
    "limited_count",
    "full_count",
    "authorization_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_AUTHORIZATION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(AUTH_MEMORY_COLUMNS))
        for col in ("authorization_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("observe_count", "limited_count", "full_count"):
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


def _system_health_score(health: Dict[str, Any]) -> float:
    status = _norm_upper(health.get("overall_status"))
    n_total = _to_float(health.get("n_artifacts_total")) or 1.0
    n_fresh = _to_float(health.get("n_artifacts_fresh")) or 0.0
    n_sub = _to_float(health.get("n_subsystems_total")) or 1.0
    n_healthy = _to_float(health.get("n_subsystems_healthy")) or 0.0
    base = (n_fresh / max(n_total, 1.0)) * 0.55 + (n_healthy / max(n_sub, 1.0)) * 0.45
    if status == "STALE":
        base *= 0.55
    elif status == "DEGRADED":
        base *= 0.75
    elif status == "HEALTHY":
        base = max(base, 0.75)
    return _clamp(base, 0.0, 1.0)


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    approval_summary: Dict[str, Any],
    approval_record: Dict[str, Any],
    approval_mem: List[Dict[str, str]],
    consideration_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    trust_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    authorization_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "approval_state": _norm_upper(
            approval_summary.get("approval_state") or approval_record.get("approval_state")
        ),
        "approval_confidence": _clamp(
            _to_float(
                approval_summary.get("approval_confidence")
                or approval_record.get("approval_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "doctrine_approval": approval_record.get("doctrine_approval") or [],
        "approval_available": bool(approval_summary.get("doctrine_approval_available")),
        "approval_memory_depth": len(approval_mem),
        "authorization_memory_depth": len(authorization_mem),
        "consideration_confidence": _clamp(
            _to_float(consideration_summary.get("consideration_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "readiness_confidence": _clamp(
            _to_float(readiness_summary.get("readiness_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "trust_confidence": _clamp(
            _to_float(trust_summary.get("trust_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "observation_cycles": max(
            _to_float(approval_summary.get("observation_cycles")) or 0,
            _to_float(consideration_summary.get("observation_cycles")) or 0,
            len(approval_mem),
            1,
        ),
        "constitutional_pressure": _clamp(
            0.75 if constitution_state == "CONSTITUTION_VIOLATED" else 0.30,
            0.0,
            1.0,
        ),
        "constitution_violated": constitution_state == "CONSTITUTION_VIOLATED",
        "court_ruling": _norm_upper(court_summary.get("judicial_ruling")),
        "council_ruling": _norm_upper(council_summary.get("governance_ruling")),
        "operator_pressure": bool(
            court_summary.get("operator_review_required")
            or council_summary.get("operator_supervision_required")
            or approval_summary.get("operator_review_required")
            or consideration_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "autonomous_readiness_score": _clamp(
            _to_float(autonomous_readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "governance_quality": _clamp(
            _to_float((runtime_policy.get("scores") or {}).get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime")),
    }
    return ctx


def _prior_authorization_map(prior_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_record.get("doctrine_authorization") or []:
        name = str(row.get("policy_name", ""))
        if name:
            out[name] = row
    return out


# -----------------------------------------------------------
# Per-doctrine authorization
# -----------------------------------------------------------
def _compute_authorization_score(da: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    approval_score = _to_float(da.get("approval_score")) or 0.0
    const_safe = 1.0 if bool(da.get("constitutional_safe")) else 0.0
    raw = approval_score * 0.80 + const_safe * 0.20
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.28
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_authorization(
    *,
    da: Dict[str, Any],
    authorization_score: float,
    ctx: Dict[str, Any],
) -> str:
    approval_class = _norm_upper(da.get("approval_classification"))
    const_safe = bool(da.get("constitutional_safe"))

    if not const_safe:
        return CLASS_NOT_AUTHORIZED

    if approval_class == APPROVAL_INSTITUTIONAL and authorization_score >= 0.55:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED_AUTH
        return CLASS_FULL_AUTH

    if approval_class == APPROVAL_LIMITED and authorization_score >= 0.40:
        return CLASS_LIMITED_AUTH

    if approval_class == APPROVAL_OBSERVE:
        return CLASS_OBSERVE_AUTH

    return CLASS_NOT_AUTHORIZED


def _authorization_rationale(name: str, classification: str, authorization_score: float) -> str:
    templates = {
        CLASS_FULL_AUTH: (
            f"full authorization: {name} is authorized for future activation deliberation"
        ),
        CLASS_LIMITED_AUTH: (
            f"limited authorization: {name} receives limited activation authorization"
        ),
        CLASS_OBSERVE_AUTH: (
            f"observe only: {name} remains under observation; not authorized for activation"
        ),
        CLASS_NOT_AUTHORIZED: (
            f"not authorized: {name} lacks institutional approval for activation authorization"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_AUTHORIZED])
    return f"{base} (authorization_score={authorization_score:.2f})"


def _build_doctrine_authorization(
    da: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(da.get("policy_name", ""))
    approval_class = _norm_upper(da.get("approval_classification"))
    approval_score = _to_float(da.get("approval_score")) or 0.0
    conf = _to_float(da.get("confidence")) or 0.0
    const_safe = bool(da.get("constitutional_safe"))

    authorization_score = _compute_authorization_score(da, ctx)
    if prior:
        prior_score = _to_float(prior.get("authorization_score")) or authorization_score
        authorization_score = round(
            _clamp(authorization_score * 0.65 + prior_score * 0.35, 0.0, 1.0),
            4,
        )

    classification = _classify_authorization(
        da=da,
        authorization_score=authorization_score,
        ctx=ctx,
    )

    future_candidate = bool(da.get("future_activation_candidate"))
    authorized = classification in (CLASS_LIMITED_AUTH, CLASS_FULL_AUTH)

    return {
        "policy_name": name,
        "authorization_classification": classification,
        "authorization_score": authorization_score,
        "approval_classification": approval_class,
        "approval_score": round(approval_score, 4),
        "future_activation_candidate": future_candidate,
        "authorized_for_activation": authorized,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "authorization_rationale": _authorization_rationale(
            name, classification, authorization_score
        ),
    }


def _build_all_authorization(
    ctx: Dict[str, Any],
    prior_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for da in ctx["doctrine_approval"]:
        name = str(da.get("policy_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(_build_doctrine_authorization(da, prior_map.get(name), ctx=ctx))

    for name, prior in prior_map.items():
        if name not in seen:
            rows.append(
                {
                    "policy_name": name,
                    "authorization_classification": prior.get(
                        "authorization_classification", CLASS_NOT_AUTHORIZED
                    ),
                    "authorization_score": _to_float(prior.get("authorization_score")) or 0.0,
                    "approval_classification": prior.get("approval_classification", APPROVAL_NOT),
                    "approval_score": _to_float(prior.get("approval_score")) or 0.0,
                    "future_activation_candidate": False,
                    "authorized_for_activation": prior.get("authorization_classification")
                    in (
                        CLASS_LIMITED_AUTH,
                        CLASS_FULL_AUTH,
                    ),
                    "confidence": _to_float(prior.get("confidence")) or 0.0,
                    "constitutional_safe": bool(prior.get("constitutional_safe")),
                    "runtime_mutation_allowed": False,
                    "authorization_rationale": "prior authorization retained; no new approval this cycle",
                }
            )
    return rows


# -----------------------------------------------------------
# Authorization confidence and state
# -----------------------------------------------------------
def _constitutional_safety_aggregate(auth_rows: List[Dict[str, Any]]) -> float:
    if not auth_rows:
        return 0.0
    vals = [1.0 if bool(r.get("constitutional_safe")) else 0.0 for r in auth_rows]
    return sum(vals) / len(vals)


def _authorization_confidence(ctx: Dict[str, Any], auth_rows: List[Dict[str, Any]]) -> float:
    if not auth_rows:
        return 0.0

    avg_auth = sum(r["authorization_score"] for r in auth_rows) / len(auth_rows)

    raw = (
        ctx["approval_confidence"] * 0.22
        + ctx["consideration_confidence"] * 0.18
        + ctx["readiness_confidence"] * 0.16
        + ctx["trust_confidence"] * 0.14
        + _constitutional_safety_aggregate(auth_rows) * 0.15
        + ctx["system_health_score"] * 0.10
        + ctx["autonomous_readiness_score"] * 0.05
    )
    raw += avg_auth * 0.05

    penalty = ctx["constitutional_pressure"] * 0.28
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.08
    if ctx["system_health_stale"]:
        penalty += 0.07
    if ctx["observation_cycles"] < 2:
        penalty += 0.06

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _count_authorization(auth_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "not_authorized": sum(
            1 for r in auth_rows if r["authorization_classification"] == CLASS_NOT_AUTHORIZED
        ),
        "observe": sum(
            1 for r in auth_rows if r["authorization_classification"] == CLASS_OBSERVE_AUTH
        ),
        "limited": sum(
            1 for r in auth_rows if r["authorization_classification"] == CLASS_LIMITED_AUTH
        ),
        "full": sum(1 for r in auth_rows if r["authorization_classification"] == CLASS_FULL_AUTH),
    }


def _classify_authorization_state(
    *,
    ctx: Dict[str, Any],
    authorization_confidence: float,
    auth_rows: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not auth_rows or not ctx["approval_available"]:
        reasons.append("insufficient approval for activation authorization")
        return AUTH_DORMANT, reasons

    if counts["full"] >= 1 and ctx["authorization_memory_depth"] >= 2:
        reasons.append("stable authorization quality with institutional governance maturity")
        return AUTH_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("full authorization granted for future activation")
        return AUTH_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("limited activation authorization; constitutionally safe")
        return AUTH_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observe only; authorization immature")
        return AUTH_OBSERVE, reasons

    if counts["not_authorized"] >= 1 or ctx["approval_state"] == "DOCTRINE_APPROVAL_DORMANT":
        reasons.append("insufficient approval for activation authorization")
        return AUTH_DORMANT, reasons

    reasons.append("insufficient approval for activation authorization")
    return AUTH_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _authorization_booleans(
    state: str,
    auth_rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_authorization_available": len(auth_rows) > 0,
        "limited_authorization_available": counts["limited"] > 0 or counts["full"] > 0,
        "full_authorization_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"] or any(r.get("authorized_for_activation") for r in auth_rows)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "authorization_memory_reliable": state == AUTH_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue governance observation",
        "Maintain defensive doctrine monitoring",
        "Avoid premature activation assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["full"] > 0 or counts["limited"] > 0:
        recs.append("Escalate authorized doctrine to operator review before any activation")
    if state == AUTH_DORMANT:
        recs.append("Accumulate more approval before activation authorization")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(auth_rows: List[Dict[str, Any]], state: str, ctx: Dict[str, Any]) -> str:
    limited = [
        r
        for r in auth_rows
        if r["policy_name"] == "target_cash_pct"
        and r["authorization_classification"] in (CLASS_LIMITED_AUTH, CLASS_FULL_AUTH)
    ]
    if limited and ctx["constitution_violated"]:
        return (
            "Triton grants limited activation authorization to elevated cash doctrine "
            "because repeated governance stabilization persisted under constitutional stress."
        )
    full = [r for r in auth_rows if r["authorization_classification"] == CLASS_FULL_AUTH]
    if full:
        names = ", ".join(r["policy_name"] for r in full[:3])
        return f"Triton grants full activation authorization for: {names}."
    lim = [r for r in auth_rows if r["authorization_classification"] == CLASS_LIMITED_AUTH]
    if lim:
        names = ", ".join(r["policy_name"] for r in lim[:3])
        return f"Triton grants limited activation authorization for: {names}."
    if state == AUTH_OBSERVE:
        return "Activation authorization remains observe-only; approval is immature."
    return "Governance doctrine activation authorization completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    authorization_confidence: float,
    auth_rows: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Activation Authorization",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Authorization State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| authorization_confidence | {authorization_confidence:.3f} |",
        f"| not_authorized | {counts['not_authorized']} |",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| full | {counts['full']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Authorization",
        "",
    ]
    if auth_rows:
        lines.append(
            "| policy | classification | authorization | approval | authorized_for_activation |"
        )
        lines.append("|---|---|---|---|---|")
        for r in auth_rows:
            lines.append(
                f"| {r['policy_name']} | {r['authorization_classification']} | {r['authorization_score']:.2f} | "
                f"{r['approval_classification']} | {r['authorized_for_activation']} |"
            )
        lines.append("")
        for r in auth_rows:
            lines.append(
                f"- **{r['policy_name']}** ({r['authorization_classification']}): {r['authorization_rationale']}"
            )
    else:
        lines.append("_No doctrine authorization assessments this cycle._")

    limited_full = [
        r
        for r in auth_rows
        if r["authorization_classification"] in (CLASS_LIMITED_AUTH, CLASS_FULL_AUTH)
    ]
    lines.extend(["", "## Limited or Full Authorization", ""])
    if limited_full:
        for r in limited_full:
            lines.append(
                f"- {r['policy_name']}: {r['authorization_classification']} "
                f"(authorization_score={r['authorization_score']:.2f}, "
                f"authorized_for_activation={r['authorized_for_activation']})"
            )
    else:
        lines.append("_No limited or full authorization yet._")

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
            "Activation authorization is governance only. Approved != authorized. "
            "Authorized != activated. Authorization != runtime mutation. "
            "No runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Authorization memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    authorization_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "authorization_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "authorization_confidence": round(authorization_confidence, 6),
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
        for c in AUTH_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_activation_authorization(
    *,
    approval_summary: Dict[str, Any],
    approval_record: Dict[str, Any],
    approval_mem: List[Dict[str, str]],
    consideration_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    trust_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    authorization_mem: List[Dict[str, str]],
    prior_authorization_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        approval_summary=approval_summary,
        approval_record=approval_record,
        approval_mem=approval_mem,
        consideration_summary=consideration_summary,
        readiness_summary=readiness_summary,
        trust_summary=trust_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        health_summary=health_summary,
        autonomous_readiness_summary=autonomous_readiness_summary,
        runtime_policy=runtime_policy,
        authorization_mem=authorization_mem,
    )

    prior_map = _prior_authorization_map(prior_authorization_record)
    auth_rows = _build_all_authorization(ctx, prior_map)
    authorization_confidence = _authorization_confidence(ctx, auth_rows)
    counts = _count_authorization(auth_rows)

    state, reasons = _classify_authorization_state(
        ctx=ctx,
        authorization_confidence=authorization_confidence,
        auth_rows=auth_rows,
        counts=counts,
    )

    booleans = _authorization_booleans(state, auth_rows, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(auth_rows, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        authorization_confidence=authorization_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(authorization_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        authorization_confidence=authorization_confidence,
        auth_rows=auth_rows,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_activation_authorization_engine",
        "engine_version": 1,
        "authorization_state": state,
        "authorization_confidence": authorization_confidence,
        "authorization_reasons": reasons,
        "doctrine_authorization": auth_rows,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "approved_vs_authorized_note": (
            "Approved != authorized. Authorized != activated. "
            "Authorization != runtime mutation. Authorization never activates doctrine."
        ),
        "constitutional_supremacy_note": (
            "Activation authorization cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "authorization_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "authorization_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_approval_board_summary": bool(approval_summary),
            "arm_governance_doctrine_approval_board_record": bool(approval_record),
            "arm_governance_doctrine_approval_board_memory_rows": len(approval_mem),
            "arm_governance_doctrine_activation_consideration_summary": bool(consideration_summary),
            "arm_governance_doctrine_readiness_summary": bool(readiness_summary),
            "arm_governance_doctrine_institutional_trust_summary": bool(trust_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(autonomous_readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_authorization_memory_rows": len(authorization_mem),
            "prior_doctrine_authorization_entries": len(prior_map),
            "n_doctrines_assessed": len(auth_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_activation_authorization_engine",
        "authorization_state": state,
        "authorization_confidence": authorization_confidence,
        "doctrine_authorization_available": booleans["doctrine_authorization_available"],
        "limited_authorization_available": booleans["limited_authorization_available"],
        "full_authorization_available": booleans["full_authorization_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "authorization_memory_reliable": booleans["authorization_memory_reliable"],
        "not_authorized_count": counts["not_authorized"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "n_doctrines_tracked": len(auth_rows),
        "n_recommendations": len(recommendations),
        "authorization_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine activation authorization engine (Step 51). "
            "Converts approval into activation authorization. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--approval-summary", default=str(DEFAULT_APPROVAL_SUM))
    p.add_argument("--approval-record", default=str(DEFAULT_APPROVAL_REC))
    p.add_argument("--approval-mem", default=str(DEFAULT_APPROVAL_MEM))
    p.add_argument("--consideration-summary", default=str(DEFAULT_CONSIDERATION_SUM))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--trust-summary", default=str(DEFAULT_TRUST_SUM))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--autonomous-readiness-summary", default=str(DEFAULT_AUTONOMOUS_READINESS_SUM))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_DOCTRINE_AUTHORIZATION] starting "
        "(read-only activation authorization; no runtime mutation; no broker calls)",
        flush=True,
    )

    approval_summary = _safe_read_json(
        Path(args.approval_summary), label="arm_governance_doctrine_approval_board_summary.json"
    )
    approval_record = _safe_read_json(
        Path(args.approval_record), label="arm_governance_doctrine_approval_board.json"
    )
    approval_mem = _safe_read_csv_rows(
        Path(args.approval_mem), label="arm_governance_doctrine_approval_board_memory.csv"
    )
    consideration_summary = _safe_read_json(
        Path(args.consideration_summary),
        label="arm_governance_doctrine_activation_consideration_summary.json",
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="arm_governance_doctrine_readiness_summary.json"
    )
    trust_summary = _safe_read_json(
        Path(args.trust_summary), label="arm_governance_doctrine_institutional_trust_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    autonomous_readiness_summary = _safe_read_json(
        Path(args.autonomous_readiness_summary), label="autonomous_readiness_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    authorization_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_activation_authorization_memory.csv"
    )
    prior_authorization_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_governance_doctrine_activation_authorization.json"
    )

    record, summary, md, merged_memory = build_doctrine_activation_authorization(
        approval_summary=approval_summary,
        approval_record=approval_record,
        approval_mem=approval_mem,
        consideration_summary=consideration_summary,
        readiness_summary=readiness_summary,
        trust_summary=trust_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        health_summary=health_summary,
        autonomous_readiness_summary=autonomous_readiness_summary,
        runtime_policy=runtime_policy,
        authorization_mem=authorization_mem,
        prior_authorization_record=prior_authorization_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=AUTH_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[ARM_DOCTRINE_AUTHORIZATION] "
        f"state={record['authorization_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['authorization_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_AUTHORIZATION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_AUTHORIZATION_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_AUTHORIZATION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
