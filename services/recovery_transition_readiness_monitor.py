"""
TRITON Phase 193 — Recovery Transition Live Authorization Readiness Monitor
(read-only).

Continuously monitors Stage 1 recovery authorization gates and determines
whether Triton is currently eligible for Stage 1 authorization. Does NOT
grant approval, activate exceptions, or execute trades.

SAFETY: monitoring only — no rule changes, no broker interaction, no state mutation.

Outputs (data/results/):
    recovery_transition_readiness.json
    recovery_transition_readiness_history.csv  (appended each run)
    recovery_transition_readiness_alerts.csv   (appended on new alerts)

Run:
    python -m services.recovery_transition_readiness_monitor
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.concentration_governance_analysis import (
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.diversification_feasibility_analysis import _lifecycle_row
from services.guard_auto import is_market_open_et
from services.pnl_ledger_builder import _to_float

RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUT_JSON = RESULTS / "recovery_transition_readiness.json"
OUT_HISTORY = RESULTS / "recovery_transition_readiness_history.csv"
OUT_ALERTS = RESULTS / "recovery_transition_readiness_alerts.csv"

STAGE_1_SYMBOL = "PEP"
BUY_STANCES = {"BUY", "ADD"}
MIN_CONF = 0.45
APPROVAL_MAX_AGE_HOURS = 48


def _log(msg: str) -> None:
    print(f"[recovery_transition_readiness_monitor] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_context() -> Dict[str, Any]:
    return {
        "simulation": _read_json(RESULTS / "final_authorization_simulation.json"),
        "playbook": _read_json(RESULTS / "stage1_governance_playbook.json"),
        "approval_package": _read_json(RESULTS / "recovery_exception_registry_migration.json"),
        "auth_assessment": _read_json(
            RESULTS / "recovery_transition_authorization_assessment.json"
        ),
        "human_approvals": _read_json(RESULTS / "human_approval_queue.json"),
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "exec_readiness": _read_json(RESULTS / "execution_readiness.json"),
        "auth": _read_json(RESULTS / "governance_authorization.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "guard": _read_json(RESULTS / "guard_snapshot.json"),
        "positions": _load_positions(),
        "stale_orders": _read_csv(RESULTS / "stale_open_orders.csv"),
        "manage_orders": _read_csv(LIVE / "manage_orders.csv"),
        "previous": _read_json(OUT_JSON) if OUT_JSON.is_file() else {},
    }


def _current_status(ctx: Dict[str, Any], ts: str) -> Dict[str, Any]:
    sim = ctx["simulation"].get("current_state") or {}
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions) or sim.get("portfolio_value") or 0
    weights: Dict[str, float] = {}
    if total_mv > 0:
        for p in positions:
            sym = p["symbol"]
            weights[sym] = weights.get(sym, 0.0) + p["market_value"] / total_mv * 100.0
    if not weights:
        weights = sim.get("weights") or {"QQQ": 100.0}

    cps = int(sim.get("current_cps") or ctx["cpi"].get("capital_preservation_score") or 68)
    esc = str(
        sim.get("escalation") or ctx["cpe"].get("escalation_state") or _escalation_from_cps(cps)
    ).upper()

    prev_ts = ctx["previous"].get("generated_at")
    last_val = prev_ts or ts
    since = None
    if prev_ts:
        prev_dt = _parse_ts(prev_ts)
        now_dt = _parse_ts(ts)
        if prev_dt and now_dt:
            since = _f((now_dt - prev_dt).total_seconds() / 60.0, 1)

    return {
        "timestamp": ts,
        "weights": {k: _f(v, 2) for k, v in weights.items()},
        "current_cps": cps,
        "escalation": esc,
        "governance_status": "BLOCKED",
        "stage_1_target": {"QQQ": 90.0, "PEP": 10.0},
        "portfolio_value": _f(total_mv),
        "last_validation_at": last_val,
        "minutes_since_last_validation": since,
    }


def _recovery_approval_records(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve operator / risk office recovery transition signatures from checklist + queue."""
    checklist = ctx["approval_package"].get("readiness_checklist") or []
    auth_req = ctx["playbook"].get("authorization_requirements") or {}

    def _from_checklist(label: str) -> Tuple[bool, Optional[str]]:
        row = next((c for c in checklist if c.get("approval_item") == label), {})
        ok = row.get("status") == "COMPLETE"
        return ok, row.get("reviewed_at") or row.get("timestamp")

    op_ok, op_ts = _from_checklist("Operator signature")
    risk_ok, risk_ts = _from_checklist("Risk office signature")

    if auth_req.get("operator_signature") is True:
        op_ok = True
    if auth_req.get("risk_office_signature") is True:
        risk_ok = True

    # Search human approval queue for recovery-specific approvals
    requests = ctx["human_approvals"].get("requests") or []
    recovery_keywords = {"RECOVERY", "STAGE_1", "STAGE 1", "RECOVERY_TRANSITION", "RECOVERY BASKET"}
    for req in requests:
        action = str(req.get("candidate_action") or "").upper()
        policy = str(req.get("policy_type") or "").upper()
        if not any(k in action or k in policy for k in recovery_keywords):
            continue
        if req.get("status") != "APPROVED":
            continue
        reviewed = req.get("reviewed_at")
        role = str(req.get("approver_role") or req.get("reviewer_role") or "").upper()
        if "RISK" in role or "RISK" in policy:
            risk_ok = True
            risk_ts = risk_ts or reviewed
        else:
            op_ok = True
            op_ts = op_ts or reviewed

    now = datetime.now(timezone.utc)

    def _ts_valid(approved: bool, ts_str: Optional[str]) -> bool:
        if not approved:
            return False
        if not ts_str:
            return True  # signature recorded without explicit timestamp
        dt = _parse_ts(ts_str)
        if not dt:
            return True
        return (now - dt) <= timedelta(hours=APPROVAL_MAX_AGE_HOURS)

    return (
        {
            "signature_recorded": op_ok,
            "approval_timestamp": op_ts,
            "timestamp_valid": _ts_valid(op_ok, op_ts),
        },
        {
            "signature_recorded": risk_ok,
            "approval_timestamp": risk_ts,
            "timestamp_valid": _ts_valid(risk_ok, risk_ts),
        },
    )


def _gate_operator(op: Dict[str, Any]) -> Dict[str, Any]:
    passed = op.get("signature_recorded") and op.get("timestamp_valid", True)
    return {
        "gate_id": 1,
        "gate_name": "Operator Approval",
        "status": "PASS" if passed else "FAIL",
        "checks": {
            "operator_signature_recorded": op.get("signature_recorded") is True,
            "approval_timestamp_valid": op.get("timestamp_valid", True),
        },
        "detail": op.get("approval_timestamp") or "PENDING",
        "blocking_reason": None if passed else "Operator DUAL_APPROVAL signature not recorded",
    }


def _gate_risk_office(risk: Dict[str, Any]) -> Dict[str, Any]:
    passed = risk.get("signature_recorded") and risk.get("timestamp_valid", True)
    return {
        "gate_id": 2,
        "gate_name": "Risk Office Approval",
        "status": "PASS" if passed else "FAIL",
        "checks": {
            "risk_office_signature_recorded": risk.get("signature_recorded") is True,
            "approval_timestamp_valid": risk.get("timestamp_valid", True),
        },
        "detail": risk.get("approval_timestamp") or "PENDING",
        "blocking_reason": None if passed else "Risk Office DUAL_APPROVAL signature not recorded",
    }


def _gate_market(ctx: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    market_open = is_market_open_et(now)
    passed = market_open is True
    return {
        "gate_id": 3,
        "gate_name": "Market Session",
        "status": "PASS" if passed else "FAIL",
        "checks": {
            "market_open": market_open,
            "trading_session_active": market_open,
        },
        "detail": f"NYSE regular session open={market_open}",
        "blocking_reason": None if passed else "Market session CLOSED",
    }


def _pep_status(ctx: Dict[str, Any]) -> Dict[str, Any]:
    lc = _lifecycle_row(ctx["lifecycle_eff"], STAGE_1_SYMBOL)
    effective = str(lc.get("effective_stance") or "").upper()
    signal = str(lc.get("signal") or "").upper()
    conf = _to_float(lc.get("confidence"), 0.0)
    risk_flag = str(lc.get("risk_flag") or "OK").upper()
    risk_block = risk_flag not in ("OK", "")
    valid = effective in BUY_STANCES and signal != "SELL" and conf >= MIN_CONF and not risk_block
    return {
        "symbol": STAGE_1_SYMBOL,
        "effective_stance": effective,
        "signal": signal,
        "confidence": _f(conf, 4),
        "risk_flag": risk_flag,
        "lifecycle_valid": valid,
        "blockers": [] if valid else [f"stance={effective}", f"conf={conf}"],
    }


def _gate_lifecycle(pep: Dict[str, Any]) -> Dict[str, Any]:
    passed = pep.get("lifecycle_valid") is True
    return {
        "gate_id": 4,
        "gate_name": "PEP Lifecycle Validation",
        "status": "PASS" if passed else "FAIL",
        "checks": {
            "effective_stance_buy_add": str(pep.get("effective_stance", "")).upper() in BUY_STANCES,
            "confidence_above_minimum": _to_float(pep.get("confidence"), 0) >= MIN_CONF,
            "no_critical_risk_blocks": str(pep.get("risk_flag", "OK")).upper() in ("OK", ""),
        },
        "detail": f"{pep.get('effective_stance')} conf={pep.get('confidence')}",
        "blocking_reason": None if passed else f"PEP lifecycle invalid: {pep.get('blockers')}",
    }


def _stale_order_count(ctx: Dict[str, Any]) -> int:
    stale = ctx["stale_orders"]
    if not stale.empty:
        return len(stale)
    manage = ctx["manage_orders"]
    if manage.empty:
        return 0
    status_col = next((c for c in manage.columns if "status" in c.lower()), None)
    if status_col:
        return int((manage[status_col].astype(str).str.upper() == "STALE").sum())
    return 0


def _gate_execution(ctx: Dict[str, Any]) -> Dict[str, Any]:
    er = ctx["exec_readiness"]
    checks_er = er.get("checks") or {}
    guard = ctx["guard"]
    guard_blocked = bool(guard.get("blocked") or guard.get("kill_switch_active"))
    stale_n = _stale_order_count(ctx)
    lock = guard_blocked or not er.get("paper_execution_permitted")

    sub = {
        "broker_connection_healthy": checks_er.get("broker") is True,
        "paper_execution_authorized": er.get("paper_execution_permitted") is True,
        "no_stale_orders": stale_n == 0,
        "no_execution_lock": not guard_blocked,
        "lifecycle_data_current": checks_er.get("lifecycle") is True,
    }
    passed = all(sub.values())
    failures = [k for k, v in sub.items() if not v]
    return {
        "gate_id": 5,
        "gate_name": "Execution Infrastructure",
        "status": "PASS" if passed else "FAIL",
        "checks": sub,
        "detail": er.get("readiness_status") or "—",
        "blocking_reason": None if passed else ", ".join(failures),
        "stale_orders_count": stale_n,
        "execution_lock_active": lock,
    }


def _readiness_scores(gates: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _pct(indices: List[int]) -> float:
        passed = sum(1 for i in indices if gates[i]["status"] == "PASS")
        return _f(passed / len(indices) * 100.0, 1)

    total_pass = sum(1 for g in gates if g["status"] == "PASS")
    score = _f(total_pass / len(gates) * 100.0, 1)

    return {
        "authorization_readiness_score": score,
        "governance_readiness": _pct([0, 1]),
        "market_readiness": _pct([2]),
        "lifecycle_readiness": _pct([3]),
        "execution_readiness": _pct([4]),
        "gates_passing": total_pass,
        "gates_total": len(gates),
    }


def _classify(gates: List[Dict[str, Any]], scores: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    all_pass = all(g["status"] == "PASS" for g in gates)
    if all_pass:
        return "STAGE_1_READY_FOR_AUTHORIZATION"

    failed = {g["gate_name"]: g["status"] for g in gates if g["status"] == "FAIL"}
    if failed.get("Operator Approval") or failed.get("Risk Office Approval"):
        return "WAITING_FOR_APPROVAL"
    if failed.get("Market Session"):
        return "WAITING_FOR_MARKET"
    if failed.get("PEP Lifecycle Validation"):
        return "WAITING_FOR_LIFECYCLE"
    if failed.get("Execution Infrastructure"):
        return "WAITING_FOR_INFRASTRUCTURE"

    sim_ready = (
        ctx["simulation"].get("executive_recommendation", {}).get("simulation_readiness")
        == "CONTROLS_VALIDATED_AWAITING_REAL_APPROVAL"
    )
    if sim_ready and scores.get("authorization_readiness_score", 0) >= 40:
        return "READY_FOR_REVIEW"

    return "NOT_READY"


def _blockers(gates: List[Dict[str, Any]], classification: str) -> List[Dict[str, Any]]:
    rows = []
    for g in gates:
        if g["status"] == "FAIL":
            rows.append(
                {
                    "gate": g["gate_name"],
                    "severity": "CRITICAL" if g["gate_id"] <= 2 else "HIGH",
                    "reason": g.get("blocking_reason"),
                    "required_action": g.get("detail"),
                }
            )
    if classification != "STAGE_1_READY_FOR_AUTHORIZATION":
        rows.append(
            {
                "gate": "authorization",
                "severity": "INFO",
                "reason": f"Classification: {classification}",
                "required_action": "Resolve blocking gates before Stage 1 authorization",
            }
        )
    return rows


def _generate_alerts(
    ctx: Dict[str, Any],
    ts: str,
    pep: Dict[str, Any],
    gates: List[Dict[str, Any]],
    classification: str,
) -> List[Dict[str, Any]]:
    prev = ctx["previous"]
    alerts: List[Dict[str, Any]] = []
    prev_pep = (prev.get("pep_lifecycle_watch") or {}).get("effective_stance")
    cur_pep = pep.get("effective_stance")
    prev_market = (prev.get("authorization_gates") or {}).get("market_session", {}).get("status")
    cur_market = gates[2]["status"]
    prev_op = (prev.get("authorization_gates") or {}).get("operator_approval", {}).get("status")
    prev_risk = (
        (prev.get("authorization_gates") or {}).get("risk_office_approval", {}).get("status")
    )
    prev_exec = (
        (prev.get("authorization_gates") or {}).get("execution_infrastructure", {}).get("status")
    )
    cur_exec = gates[4]["status"]

    if prev_pep and prev_pep != cur_pep:
        sev = "CRITICAL" if cur_pep not in BUY_STANCES else "WARNING"
        alerts.append(
            {
                "timestamp": ts,
                "severity": sev,
                "event": f"PEP lifecycle change: {prev_pep} -> {cur_pep}",
                "required_action": "HALT Stage 1 authorization if not BUY/ADD",
            }
        )

    for transition, label in [
        (("FAIL", "PASS"), "market CLOSED -> OPEN"),
        (("PASS", "FAIL"), "market OPEN -> CLOSED"),
    ]:
        if prev_market == transition[0] and cur_market == transition[1]:
            sev = "INFO" if transition[1] == "PASS" else "WARNING"
            alerts.append(
                {
                    "timestamp": ts,
                    "severity": sev,
                    "event": f"Market session: {label}",
                    "required_action": (
                        "Stage 1 may proceed if all other gates pass"
                        if transition[1] == "PASS"
                        else "Defer Stage 1 until market opens"
                    ),
                }
            )

    for gate_name, prev_st, cur_st in [
        ("Operator approval", prev_op, gates[0]["status"]),
        ("Risk Office approval", prev_risk, gates[1]["status"]),
    ]:
        if prev_st == "FAIL" and cur_st == "PASS":
            alerts.append(
                {
                    "timestamp": ts,
                    "severity": "INFO",
                    "event": f"{gate_name}: PENDING -> APPROVED",
                    "required_action": "Verify DUAL_APPROVAL complete before authorization",
                }
            )

    if prev_exec == "PASS" and cur_exec == "FAIL":
        alerts.append(
            {
                "timestamp": ts,
                "severity": "CRITICAL",
                "event": "Execution infrastructure: HEALTHY -> DEGRADED",
                "required_action": "HALT authorization until infrastructure restored",
            }
        )

    if not prev and classification != "STAGE_1_READY_FOR_AUTHORIZATION":
        alerts.append(
            {
                "timestamp": ts,
                "severity": "INFO",
                "event": f"Readiness monitor initialized — {classification}",
                "required_action": "Resolve blocking gates per executive recommendation",
            }
        )

    return alerts


def _executive_report(
    status: Dict[str, Any],
    gates: List[Dict[str, Any]],
    scores: Dict[str, Any],
    classification: str,
    blockers: List[Dict[str, Any]],
    pep: Dict[str, Any],
    market_gate: Dict[str, Any],
    exec_gate: Dict[str, Any],
) -> Dict[str, Any]:
    all_pass = classification == "STAGE_1_READY_FOR_AUTHORIZATION"
    primary_block = next((b["reason"] for b in blockers if b.get("gate") != "authorization"), None)

    if all_pass:
        action = (
            "STAGE_1_READY_FOR_AUTHORIZATION — all gates pass. Operator may proceed to "
            "formal authorization ceremony per Phase 191 playbook. Monitor does NOT grant approval."
        )
    elif classification == "WAITING_FOR_APPROVAL":
        action = "Obtain Operator and Risk Office DUAL_APPROVAL signatures for recovery transition."
    elif classification == "WAITING_FOR_MARKET":
        action = "Wait for NYSE regular session OPEN; re-run monitor at session open."
    elif classification == "WAITING_FOR_LIFECYCLE":
        action = f"PEP lifecycle invalid ({pep.get('effective_stance')}); do not authorize Stage 1."
    elif classification == "WAITING_FOR_INFRASTRUCTURE":
        action = f"Restore execution infrastructure: {exec_gate.get('blocking_reason')}"
    elif classification == "READY_FOR_REVIEW":
        action = "Controls validated — complete remaining gates before authorization."
    else:
        action = f"NOT_READY — resolve: {primary_block or 'multiple blockers'}"

    return {
        "report_type": "RECOVERY_TRANSITION_READINESS_REPORT",
        "current_stage_1_status": {
            "classification": classification,
            "authorization_readiness_score": scores.get("authorization_readiness_score"),
            "stage_1_eligible": all_pass,
            "authorization_granted": False,
        },
        "authorization_gate_dashboard": {g["gate_name"]: g["status"] for g in gates},
        "readiness_score": scores,
        "blocking_conditions": blockers,
        "lifecycle_status": pep,
        "market_status": {
            "open": market_gate["checks"].get("market_open"),
            "gate_status": market_gate["status"],
        },
        "execution_health": {
            "status": exec_gate["status"],
            "readiness": exec_gate.get("detail"),
            "stale_orders": exec_gate.get("stale_orders_count", 0),
        },
        "time_since_last_validation_minutes": status.get("minutes_since_last_validation"),
        "recommended_operator_action": action,
        "executive_questions": {
            "stage_1_eligible_for_authorization": all_pass,
            "blocking_gate": primary_block,
            "pep_valid_buy_candidate": pep.get("lifecycle_valid"),
            "market_open_and_execution_healthy": (
                market_gate["status"] == "PASS" and exec_gate["status"] == "PASS"
            ),
            "human_approvals_complete": gates[0]["status"] == "PASS"
            and gates[1]["status"] == "PASS",
            "can_legally_proceed_to_authorization": all_pass,
        },
    }


def _append_history(
    ts: str, scores: Dict[str, Any], classification: str, blockers: List[Dict[str, Any]]
):
    primary = next(
        (b["reason"] for b in blockers if b.get("severity") in ("CRITICAL", "HIGH")),
        classification,
    )
    row = {
        "timestamp": ts,
        "readiness_score": scores.get("authorization_readiness_score"),
        "authorization_status": classification,
        "blocking_reason": primary,
        "recommended_action": classification,
    }
    if OUT_HISTORY.is_file():
        hist = pd.read_csv(OUT_HISTORY)
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(OUT_HISTORY, index=False, encoding="utf-8")


def _append_alerts(alerts: List[Dict[str, Any]]) -> None:
    if not alerts:
        return
    df_new = pd.DataFrame(alerts)
    if OUT_ALERTS.is_file():
        existing = pd.read_csv(OUT_ALERTS)
        df = pd.concat([existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(OUT_ALERTS, index=False, encoding="utf-8")


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    status = _current_status(ctx, ts)

    op_meta, risk_meta = _recovery_approval_records(ctx)
    gates = [
        _gate_operator(op_meta),
        _gate_risk_office(risk_meta),
        _gate_market(ctx),
        _gate_lifecycle(_pep_status(ctx)),
        _gate_execution(ctx),
    ]
    pep = _pep_status(ctx)
    scores = _readiness_scores(gates)
    classification = _classify(gates, scores, ctx)
    blockers = _blockers(gates, classification)
    alerts = _generate_alerts(ctx, ts, pep, gates, classification)
    executive = _executive_report(
        status, gates, scores, classification, blockers, pep, gates[2], gates[4]
    )

    history_summary = {}
    if OUT_HISTORY.is_file():
        hist = pd.read_csv(OUT_HISTORY)
        history_summary = {
            "observations": len(hist),
            "latest_score": _to_float(hist.iloc[-1]["readiness_score"], 0) if len(hist) else None,
            "latest_status": str(hist.iloc[-1]["authorization_status"]) if len(hist) else None,
        }

    summary = {
        "classification": classification,
        "authorization_readiness_score": scores.get("authorization_readiness_score"),
        "stage_1_eligible": classification == "STAGE_1_READY_FOR_AUTHORIZATION",
        "authorization_granted": False,
        "exception_activated": False,
        "gates_passing": scores.get("gates_passing"),
        "alerts_generated": len(alerts),
    }

    report = {
        "generated_at": ts,
        "phase": "193",
        "mode": "recovery_transition_readiness_monitor",
        "disclaimer": (
            "Monitoring only — does NOT grant DUAL_APPROVAL, activate exceptions, modify "
            "governance/lifecycle/execution/broker/allocation/risk controls, or place trades."
        ),
        "current_status": status,
        "authorization_gates": {
            "operator_approval": gates[0],
            "risk_office_approval": gates[1],
            "market_session": gates[2],
            "pep_lifecycle": gates[3],
            "execution_infrastructure": gates[4],
            "all_pass": all(g["status"] == "PASS" for g in gates),
        },
        "readiness_scores": scores,
        "readiness_classification": classification,
        "blockers": blockers,
        "alerts": alerts,
        "pep_lifecycle_watch": pep,
        "history_summary": history_summary,
        "executive_recommendation": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only monitor — no authorization granted",
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _append_history(ts, scores, classification, blockers)
    _append_alerts(alerts)

    _log(
        f"wrote {OUT_JSON.name}; classification={classification} score={scores['authorization_readiness_score']}"
    )
    _log(f"history appended; alerts={len(alerts)}")
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
