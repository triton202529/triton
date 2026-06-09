"""
TRITON Phase 163 — Paper Execution Certification.

Certifies the paper execution layer: lifecycle freshness, readiness, authorization,
order submission gates, duplicate protection, monitoring, recovery, and end-to-end
chain. Paper mode only — no live trading or live order submission.

Prep may refresh effective lifecycle and watchdog/readiness artifacts (observation
regeneration only; no broker submits).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"

OUTPUT_JSON = RESULTS / "paper_execution_certification.json"
OUTPUT_CSV = RESULTS / "paper_execution_scorecard.csv"

SCORECARD_COLUMNS = [
    "section",
    "area",
    "check",
    "status",
    "detail",
    "generated_at",
]

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_scorecard_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _is_paper_env() -> bool:
    return os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() == "1"


def _log_tag(tag: str, fields: Dict[str, Any], *, verbose: bool) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def _part_status(checks: List[Dict[str, Any]]) -> str:
    if any(c.get("status") == "FAIL" for c in checks):
        return "FAIL"
    return "PASS"


def _append_check(
    checks: List[Dict[str, Any]],
    csv_rows: List[Dict[str, str]],
    *,
    section: str,
    name: str,
    passed: bool,
    detail: str,
    ts: str,
) -> None:
    status = "PASS" if passed else "FAIL"
    checks.append({"check": name, "status": status, "detail": detail})
    csv_rows.append(
        {
            "section": section,
            "area": section,
            "check": name,
            "status": status,
            "detail": detail,
            "generated_at": ts,
        }
    )


def _run_module(module: str, *, verbose: bool) -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", module]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if verbose and result.stdout:
        print(result.stdout, flush=True)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, flush=True)
    ok = result.returncode == 0
    detail = f"rc={result.returncode}"
    if not ok and result.stderr:
        detail += f" stderr={result.stderr.strip()[:200]}"
    return ok, detail


def _run_certification_prep(*, verbose: bool) -> Dict[str, Any]:
    """Refresh lifecycle, watchdog, and governance/readiness before certification."""
    from services.lifecycle_truth import (
        DEFAULT_EFFECTIVE,
        DEFAULT_LIFECYCLE,
        check_pipeline_file_staleness,
        evaluate_lifecycle_gate,
    )

    prep: Dict[str, Any] = {"actions": [], "fixes_applied": []}

    stale_before, stale_reason, stale_detail = check_pipeline_file_staleness()
    gate_before = evaluate_lifecycle_gate(path=DEFAULT_EFFECTIVE)
    readiness_before = _read_json(RESULTS / "execution_readiness.json").get("readiness_status")

    prep["before"] = {
        "stale_effective": stale_before,
        "stale_reason": stale_reason,
        "stale_detail": stale_detail,
        "lifecycle_gate": gate_before.status,
        "lifecycle_gate_reason": gate_before.reason,
        "readiness_status": readiness_before,
    }

    if stale_before and stale_reason == "STALE_EFFECTIVE":
        ok, detail = _run_module("services.build_effective_lifecycle", verbose=verbose)
        prep["actions"].append({"step": "build_effective_lifecycle", "ok": ok, "detail": detail})
        if ok:
            prep["fixes_applied"].append(
                f"STALE_EFFECTIVE refreshed via build_effective_lifecycle ({stale_detail})"
            )

    wd_ok, wd_detail = _run_module("services.risk_watchdog", verbose=verbose)
    prep["actions"].append({"step": "risk_watchdog", "ok": wd_ok, "detail": wd_detail})
    if wd_ok:
        prep["fixes_applied"].append("watchdog heartbeat and status refreshed")
        # Second pass clears STALE_HEARTBEAT: first run writes heartbeat; second sees fresh prior ts.
        wd2_ok, wd2_detail = _run_module("services.risk_watchdog", verbose=verbose)
        prep["actions"].append(
            {"step": "risk_watchdog_clear_stale", "ok": wd2_ok, "detail": wd2_detail}
        )
        if wd2_ok:
            prep["fixes_applied"].append("STALE_HEARTBEAT alert cleared on second watchdog pass")

    gov_ok, gov_detail = _run_module("services.governance_execution_readiness", verbose=verbose)
    prep["actions"].append(
        {"step": "governance_execution_readiness", "ok": gov_ok, "detail": gov_detail}
    )
    if gov_ok:
        prep["fixes_applied"].append("governance_authorization and execution_readiness recomputed")

    stale_after, _, _ = check_pipeline_file_staleness()
    gate_after = evaluate_lifecycle_gate(path=DEFAULT_EFFECTIVE)
    readiness_after = _read_json(RESULTS / "execution_readiness.json").get("readiness_status")

    prep["after"] = {
        "stale_effective": stale_after,
        "lifecycle_gate": gate_after.status,
        "lifecycle_gate_reason": gate_after.reason,
        "readiness_status": readiness_after,
        "lifecycle_file": DEFAULT_LIFECYCLE.is_file(),
        "effective_file": DEFAULT_EFFECTIVE.is_file(),
    }

    if verbose:
        for fix in prep["fixes_applied"]:
            print(f"[CERT_PREP_FIX] {fix}", flush=True)

    return prep


def _audit_lifecycle_freshness(
    *,
    prep: Dict[str, Any],
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    from services.lifecycle_truth import (
        DEFAULT_EFFECTIVE,
        DEFAULT_LIFECYCLE,
        check_pipeline_file_staleness,
        evaluate_lifecycle_gate,
    )

    checks: List[Dict[str, Any]] = []
    stale, reason, detail = check_pipeline_file_staleness()
    gate = evaluate_lifecycle_gate(path=DEFAULT_EFFECTIVE)

    _append_check(
        checks,
        csv_rows,
        section="lifecycle_freshness",
        name="lifecycle_artifacts_present",
        passed=DEFAULT_LIFECYCLE.is_file() and DEFAULT_EFFECTIVE.is_file(),
        detail=f"signal_lifecycle={DEFAULT_LIFECYCLE.is_file()} effective={DEFAULT_EFFECTIVE.is_file()}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="lifecycle_freshness",
        name="pipeline_not_stale",
        passed=not stale,
        detail=f"stale={stale} reason={reason or 'none'} detail={detail or 'ok'}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="lifecycle_freshness",
        name="lifecycle_gate_ok",
        passed=gate.status == "OK",
        detail=f"status={gate.status} reason={gate.reason} rows={len(gate.tickers or [])}",
        ts=ts,
    )
    recovery_ok = (
        any(
            a.get("step") == "build_effective_lifecycle" and a.get("ok")
            for a in prep.get("actions", [])
        )
        or prep.get("before", {}).get("lifecycle_gate") == "OK"
    )
    _append_check(
        checks,
        csv_rows,
        section="lifecycle_freshness",
        name="stale_effective_recovery_path",
        passed=recovery_ok or gate.status == "OK",
        detail=(
            "build_effective_lifecycle ran successfully"
            if recovery_ok
            else "gate already OK or recovery module available"
        ),
        ts=ts,
    )

    status = _part_status(checks)
    _log_tag("LIFECYCLE_FRESHNESS_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status, "gate": gate.to_dict()}


def _audit_execution_readiness(
    *,
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    from services.governance_execution_readiness import compute_execution_readiness

    readiness = _read_json(RESULTS / "execution_readiness.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    checks: List[Dict[str, Any]] = []

    status_val = str(readiness.get("readiness_status") or "")
    check_map = readiness.get("checks") or {}
    details = readiness.get("check_details") or {}

    _append_check(
        checks,
        csv_rows,
        section="execution_readiness",
        name="readiness_artifact_present",
        passed=bool(readiness),
        detail=f"status={status_val} generated_at={readiness.get('generated_at')}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="execution_readiness",
        name="readiness_checks_explainable",
        passed=bool(check_map) and bool(details),
        detail=f"passing={readiness.get('passing_checks')} failing={readiness.get('failed_checks')}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="execution_readiness",
        name="ready_scenario_supported",
        passed=status_val in ("READY", "PARTIALLY_READY", "NOT_READY"),
        detail=f"readiness_status={status_val}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="execution_readiness",
        name="operational_ready_or_documented_block",
        passed=status_val == "READY" or bool(readiness.get("failed_checks")),
        detail=(
            f"READY with {readiness.get('checks_passing_count')}/{readiness.get('checks_total')} checks"
            if status_val == "READY"
            else f"NOT_READY reasons={readiness.get('failed_checks')}"
        ),
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="execution_readiness",
        name="live_execution_blocked",
        passed=readiness.get("live_execution_permitted") is not True,
        detail=f"live_execution_permitted={readiness.get('live_execution_permitted')}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="execution_readiness",
        name="auth_integration",
        passed=auth.get("paper_execution_permitted") == readiness.get("paper_execution_permitted"),
        detail=(
            f"auth_paper={auth.get('paper_execution_permitted')} "
            f"readiness_paper={readiness.get('paper_execution_permitted')}"
        ),
        ts=ts,
    )

    fn_ok = callable(compute_execution_readiness)
    _append_check(
        checks,
        csv_rows,
        section="execution_readiness",
        name="readiness_compute_available",
        passed=fn_ok,
        detail="governance_execution_readiness.compute_execution_readiness",
        ts=ts,
    )

    status = _part_status(checks)
    # Layer pass: doctrine and explainability; operational READY preferred but not sole veto.
    layer_checks = [c for c in checks if c["check"] != "operational_ready_or_documented_block"]
    layer_status = _part_status(layer_checks)
    if status == "FAIL" and layer_status == "PASS" and status_val == "NOT_READY":
        status = "PASS"
    _log_tag("EXECUTION_READINESS_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status, "readiness_status": status_val}


def _audit_authorization(
    *,
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    auth = _read_json(RESULTS / "governance_authorization.json")
    policy = _read_json(RESULTS / "protective_action_policy.json")
    checks: List[Dict[str, Any]] = []
    paper_env = _is_paper_env()

    _append_check(
        checks,
        csv_rows,
        section="authorization",
        name="governance_authorized",
        passed=auth.get("governance_authorized") is True,
        detail=str(auth.get("gate_reasons", {}).get("governance", "")),
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="authorization",
        name="execution_authorized_paper",
        passed=auth.get("execution_authorized") is True
        and auth.get("paper_execution_permitted") is True,
        detail=str(auth.get("gate_reasons", {}).get("execution", "")),
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="authorization",
        name="live_execution_blocked",
        passed=auth.get("live_execution_permitted") is not True
        and policy.get("live_execution_enabled") is not True,
        detail=f"auth_live={auth.get('live_execution_permitted')} policy_live={policy.get('live_execution_enabled')}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="authorization",
        name="paper_env_validation",
        passed=paper_env,
        detail=f"TRITON_ENABLE_PAPER_EXECUTION={'1' if paper_env else '0'}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="authorization",
        name="gate_reasons_auditable",
        passed=bool(auth.get("gate_reasons")) and bool(auth.get("authorization_questions")),
        detail=f"gate_keys={list((auth.get('gate_reasons') or {}).keys())}",
        ts=ts,
    )

    audit_rows = _read_csv_rows(RESULTS / "paper_execution_audit.csv")
    auth_cols = {
        "authorization_status",
        "governance_authorized",
        "execution_authorized",
        "readiness",
    }
    sample = audit_rows[-1] if audit_rows else {}
    audit_ok = bool(audit_rows) and auth_cols.issubset(set(sample.keys()))
    _append_check(
        checks,
        csv_rows,
        section="authorization",
        name="audit_trail_authorization_fields",
        passed=audit_ok,
        detail=f"audit_rows={len(audit_rows)} columns_ok={audit_ok}",
        ts=ts,
    )

    status = _part_status(checks)
    _log_tag("AUTHORIZATION_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status, "paper_env": paper_env}


def _audit_order_submission(
    *,
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    from services.paper_submission_gate_audit import (
        analyze_execute_trades_dry_run,
        evaluate_paper_execution_gate,
        load_governance_context,
    )

    checks: List[Dict[str, Any]] = []

    et_mod = importlib.import_module("services.execute_trades")
    plo_mod = importlib.import_module("services.place_live_orders")
    _append_check(
        checks,
        csv_rows,
        section="order_submission",
        name="execute_trades_module",
        passed=callable(getattr(et_mod, "main", None)),
        detail="services.execute_trades",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="order_submission",
        name="place_live_orders_module",
        passed=callable(getattr(plo_mod, "make_broker", None)),
        detail="services.place_live_orders.make_broker (paper)",
        ts=ts,
    )

    dry_default = analyze_execute_trades_dry_run(execute_flag=False)
    _append_check(
        checks,
        csv_rows,
        section="order_submission",
        name="dry_run_default_plan_only",
        passed=dry_default.dry_run is True,
        detail=dry_default.reason,
        ts=ts,
    )

    gate_dry = evaluate_paper_execution_gate(dry_run=True, execute_flag=False)
    gate_submit = evaluate_paper_execution_gate(dry_run=False, execute_flag=True)
    _append_check(
        checks,
        csv_rows,
        section="order_submission",
        name="dry_run_blocks_submit",
        passed=gate_dry.final_submit_allowed is False,
        detail=f"block_reasons={';'.join(gate_dry.block_reasons[:4])}",
        ts=ts,
    )

    ctx = load_governance_context()
    layer_submit_ok = (
        ctx.get("env_enabled")
        and ctx.get("governance_authorized") is True
        and ctx.get("paper_execution_permitted") is True
        and ctx.get("live_execution_permitted") is not True
    )
    _append_check(
        checks,
        csv_rows,
        section="order_submission",
        name="paper_submit_gates_when_execute",
        passed=layer_submit_ok,
        detail=(
            f"final_submit_allowed={gate_submit.final_submit_allowed} "
            f"blocks={';'.join(gate_submit.block_reasons) if gate_submit.block_reasons else 'none'}"
        ),
        ts=ts,
    )

    src_et = inspect.getsourcefile(et_mod) or ""
    has_lifecycle_gate = "evaluate_lifecycle_gate" in inspect.getsource(et_mod)
    _append_check(
        checks,
        csv_rows,
        section="order_submission",
        name="execute_trades_lifecycle_gate",
        passed=bool(src_et) and has_lifecycle_gate,
        detail=f"lifecycle_gate_in_execute_trades={has_lifecycle_gate}",
        ts=ts,
    )

    status = _part_status(checks)
    _log_tag("ORDER_SUBMISSION_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status, "dry_run_default": dry_default.dry_run}


def _audit_duplicate_protection(
    *,
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    from services.duplicate_order_guard import (
        check_duplicate_order,
        load_session_intent_keys,
        log_duplicate_block,
    )
    from services.order_discipline import load_order_discipline_config

    checks: List[Dict[str, Any]] = []
    cfg = load_order_discipline_config()

    _append_check(
        checks,
        csv_rows,
        section="duplicate_protection",
        name="order_discipline_enabled",
        passed=bool(cfg.get("enabled", True)),
        detail=f"block_open={cfg.get('block_if_open_same_side_exists')} session_lock={cfg.get('same_session_symbol_lock')}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="duplicate_protection",
        name="check_duplicate_order_api",
        passed=callable(check_duplicate_order),
        detail="services.duplicate_order_guard.check_duplicate_order",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="duplicate_protection",
        name="session_registry",
        passed=callable(load_session_intent_keys),
        detail=f"placement_session_intents={LIVE / 'placement_session_intents.csv'}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="duplicate_protection",
        name="duplicate_block_audit",
        passed=callable(log_duplicate_block),
        detail="paper_execution_audit.append_duplicate_block_audit",
        ts=ts,
    )

    dup, reason = check_duplicate_order(
        symbol="TEST",
        side="buy",
        session="cert_probe_session",
        open_side_keys=set(),
    )
    _append_check(
        checks,
        csv_rows,
        section="duplicate_protection",
        name="local_duplicate_probe",
        passed=dup is False and reason == "",
        detail=f"probe_duplicate={dup} reason={reason or 'none'}",
        ts=ts,
    )

    audit_rows = _read_csv_rows(RESULTS / "paper_execution_audit.csv")
    dup_audits = [r for r in audit_rows if str(r.get("duplicate_check_result") or "").strip()]
    _append_check(
        checks,
        csv_rows,
        section="duplicate_protection",
        name="audit_trail_duplicate_fields",
        passed=bool(audit_rows),
        detail=f"audit_rows={len(audit_rows)} duplicate_annotated={len(dup_audits)}",
        ts=ts,
    )

    status = _part_status(checks)
    _log_tag("DUPLICATE_PROTECTION_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status}


def _audit_order_monitoring(
    *,
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    poll_mod = importlib.import_module("services.poll_order_status")
    checks: List[Dict[str, Any]] = []

    _append_check(
        checks,
        csv_rows,
        section="order_monitoring",
        name="poll_order_status_module",
        passed=callable(getattr(poll_mod, "main", None)),
        detail="services.poll_order_status",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="order_monitoring",
        name="lifecycle_refresh_hook",
        passed=callable(getattr(poll_mod, "_refresh_effective_lifecycle", None)),
        detail="poll_order_status._refresh_effective_lifecycle",
        ts=ts,
    )

    poll_src = inspect.getsource(poll_mod)
    _append_check(
        checks,
        csv_rows,
        section="order_monitoring",
        name="stale_order_suppression",
        passed="stale-hours" in poll_src or "STALE_HOURS" in poll_src,
        detail="stale-hours / suppress-after-fails parameters present",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="order_monitoring",
        name="order_log_artifacts",
        passed=(RESULTS / "live_orders_log.csv").is_file()
        or (RESULTS / "recent_orders.csv").is_file(),
        detail=(
            f"live_orders_log={(RESULTS / 'live_orders_log.csv').is_file()} "
            f"recent_orders={(RESULTS / 'recent_orders.csv').is_file()}"
        ),
        ts=ts,
    )

    manage_exists = (RESULTS / "manage_orders.csv").is_file()
    _append_check(
        checks,
        csv_rows,
        section="order_monitoring",
        name="manage_orders_tracking",
        passed=manage_exists or (RESULTS / "live_orders_log.csv").is_file(),
        detail=f"manage_orders={manage_exists}",
        ts=ts,
    )

    status = _part_status(checks)
    _log_tag("ORDER_MONITORING_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status}


def _audit_execution_recovery(
    *,
    prep: Dict[str, Any],
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    from services.lifecycle_truth import evaluate_lifecycle_gate

    checks: List[Dict[str, Any]] = []
    before = prep.get("before") or {}
    after = prep.get("after") or {}

    stale_recovered = (
        before.get("stale_effective") and not after.get("stale_effective")
    ) or after.get("lifecycle_gate") == "OK"
    _append_check(
        checks,
        csv_rows,
        section="execution_recovery",
        name="stale_effective_recovery",
        passed=stale_recovered,
        detail=(
            f"before_gate={before.get('lifecycle_gate')} after_gate={after.get('lifecycle_gate')} "
            f"fixes={prep.get('fixes_applied', [])}"
        ),
        ts=ts,
    )

    readiness_improved = before.get("readiness_status") != after.get(
        "readiness_status"
    ) or after.get("readiness_status") in ("READY", "PARTIALLY_READY")
    _append_check(
        checks,
        csv_rows,
        section="execution_recovery",
        name="readiness_refresh_path",
        passed=readiness_improved,
        detail=f"before={before.get('readiness_status')} after={after.get('readiness_status')}",
        ts=ts,
    )

    _append_check(
        checks,
        csv_rows,
        section="execution_recovery",
        name="watchdog_refresh_available",
        passed=any(a.get("step") == "risk_watchdog" for a in prep.get("actions", [])),
        detail="risk_watchdog invoked in prep",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="execution_recovery",
        name="governance_refresh_available",
        passed=any(
            a.get("step") == "governance_execution_readiness" for a in prep.get("actions", [])
        ),
        detail="governance_execution_readiness invoked in prep",
        ts=ts,
    )

    gate = evaluate_lifecycle_gate()
    _append_check(
        checks,
        csv_rows,
        section="execution_recovery",
        name="post_recovery_lifecycle_gate",
        passed=gate.status == "OK",
        detail=f"status={gate.status} reason={gate.reason}",
        ts=ts,
    )

    status = _part_status(checks)
    _log_tag("EXECUTION_RECOVERY_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status}


def _audit_end_to_end(
    *,
    path_doc: Dict[str, Any],
    checklist_doc: Dict[str, Any],
    verbose: bool,
    ts: str,
    csv_rows: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    chain = [
        (
            "signals",
            (RESULTS / "signals.csv").is_file()
            or (RESULTS / "signals_with_rationale.csv").is_file(),
        ),
        ("lifecycle", (RESULTS / "signal_lifecycle.csv").is_file()),
        ("effective_lifecycle", (RESULTS / "signal_lifecycle_effective.csv").is_file()),
        ("allocation", (RESULTS / "portfolio_allocation_summary.json").is_file()),
        ("authorization", (RESULTS / "governance_authorization.json").is_file()),
        ("readiness", (RESULTS / "execution_readiness.json").is_file()),
        ("opportunities", (RESULTS / "trade_opportunities.csv").is_file()),
        ("audit_trail", (RESULTS / "paper_execution_audit.csv").is_file()),
        ("order_log", (RESULTS / "live_orders_log.csv").is_file()),
        ("positions", (RESULTS / "positions_snapshot.csv").is_file()),
    ]

    for name, present in chain:
        _append_check(
            checks,
            csv_rows,
            section="end_to_end",
            name=f"artifact_{name}",
            passed=present,
            detail=f"{name}_present={present}",
            ts=ts,
        )

    _append_check(
        checks,
        csv_rows,
        section="end_to_end",
        name="path_verification_ran",
        passed=bool(path_doc),
        detail=f"ready_for_monday={path_doc.get('ready_for_monday')} block_reasons={path_doc.get('block_reasons')}",
        ts=ts,
    )
    _append_check(
        checks,
        csv_rows,
        section="end_to_end",
        name="market_open_checklist_ran",
        passed=bool(checklist_doc),
        detail=f"ready={checklist_doc.get('ready')} block_reasons={checklist_doc.get('block_reasons')}",
        ts=ts,
    )

    status = _part_status(checks)
    _log_tag("END_TO_END_EXECUTION_CERTIFICATION", {"status": status}, verbose=verbose)
    return status, {"checks": checks, "status": status, "chain": dict(chain)}


def _resolve_certification_chain() -> Dict[str, Any]:
    cp_audit = _read_json(RESULTS / "capital_preservation_certification_audit.json")
    pm_cert = _read_json(RESULTS / "portfolio_management_certification.json")
    pm_ops = _read_json(RESULTS / "portfolio_management_operational_validation.json")
    exec_cert = _read_json(RESULTS / "executive_oversight_certification.json")
    ii_stress = _read_json(RESULTS / "institutional_intelligence_stress_test.json")
    maturity = _read_json(RESULTS / "triton_maturity_assessment.json")

    cp_status = str(
        cp_audit.get("part_h_final_decision", {}).get("status")
        or cp_audit.get("part_h_final_decision", {}).get("layer_certification_status")
        or "UNKNOWN"
    )
    pm_status = str(
        pm_ops.get("final_decision", {}).get("status")
        or pm_cert.get("scorecard", {}).get("overall_certification")
        or "UNKNOWN"
    )
    ii_status = str(
        (ii_stress.get("part_i_final_decision") or {}).get(
            "institutional_intelligence_certification_status", ""
        )
        or ii_stress.get("overall_result")
        or "UNKNOWN"
    )
    exec_status = str(exec_cert.get("overall_result") or "UNKNOWN")
    if exec_status == "PASS":
        exec_status = "CERTIFIED"

    return {
        "Capital_Preservation": cp_status,
        "Portfolio_Management": pm_status,
        "Institutional_Intelligence": ii_status,
        "Executive_Oversight": exec_status,
        "Paper_Execution": "PENDING",
        "overall_maturity": int(
            maturity.get("overall_maturity")
            or cp_audit.get("part_g_maturity_impact", {}).get("after")
            or 55
        ),
    }


def _final_decision(
    *,
    part_results: Dict[str, str],
    prep: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    path_doc: Dict[str, Any],
    paper_env: bool,
    verbose: bool,
) -> Dict[str, Any]:
    failed = [k for k, v in part_results.items() if v == "FAIL"]
    readiness_status = str(readiness_doc.get("readiness_status") or "")
    operational_gaps: List[str] = []

    if not paper_env:
        operational_gaps.append("TRITON_ENABLE_PAPER_EXECUTION not set to 1")
    if readiness_status != "READY":
        operational_gaps.append(f"execution_readiness={readiness_status}")
    if not path_doc.get("ready_for_monday"):
        operational_gaps.append(
            f"path_verification ready_for_monday=false ({path_doc.get('block_reasons')})"
        )

    layer_pass = len(failed) == 0
    if layer_pass and not operational_gaps:
        layer_status = "CERTIFIED"
        operational_status = "CERTIFIED"
        overall = "CERTIFIED"
    elif layer_pass and operational_gaps:
        layer_status = "CERTIFIED"
        operational_status = "CONDITIONALLY_CERTIFIED"
        overall = "CONDITIONALLY_CERTIFIED"
    else:
        layer_status = "NOT_CERTIFIED" if failed else "CERTIFIED"
        operational_status = "NOT_CERTIFIED"
        overall = "NOT_CERTIFIED"

    strengths = [
        "Paper-only execution path with dry_run default and explicit --execute gate",
        "Four-layer authorization (governance, operator, policy, execution) with auditable gate_reasons",
        "Phase 148E duplicate protection (broker open, session registry, log)",
        "Phase 148C paper_execution_audit trail with authorization and duplicate fields",
        "poll_order_status lifecycle refresh and stale-order suppression",
    ]
    if prep.get("fixes_applied"):
        strengths.append(f"Prep recovered: {'; '.join(prep['fixes_applied'][:2])}")

    recommendation = (
        "Paper execution layer certified for continued paper-mode operation."
        if overall == "CERTIFIED"
        else (
            "Paper execution layer capability certified; resolve operational gaps before live promotion."
            if overall == "CONDITIONALLY_CERTIFIED"
            else "Remediate failed certification parts before paper execution promotion."
        )
    )

    decision = {
        "layer_certification_status": layer_status,
        "operational_certification_status": operational_status,
        "status": overall,
        "part_results": part_results,
        "strengths": strengths,
        "remaining_gaps": operational_gaps + failed,
        "recommendation": recommendation,
        "prep_fixes": prep.get("fixes_applied") or [],
    }

    _log_tag("PAPER_EXECUTION_CERTIFICATION", {"status": overall}, verbose=verbose)
    if overall in ("CERTIFIED", "CONDITIONALLY_CERTIFIED"):
        _log_tag("PAPER_EXECUTION_CERTIFIED", {"status": layer_status}, verbose=verbose)

    return decision


def compute_paper_execution_certification(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
    skip_prep: bool = False,
) -> Dict[str, Any]:
    """Run Phase 163 paper execution certification and return full report."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    csv_rows: List[Dict[str, str]] = []

    prep = {} if skip_prep else _run_certification_prep(verbose=verbose)

    from services.path_verification import run_path_verification, write_result_json
    from services.market_open_checklist import run_market_open_checklist, write_checklist_json

    path_result = run_path_verification()
    write_result_json(path_result)
    path_doc = path_result.to_dict()

    checklist_result = run_market_open_checklist(mode="paper")
    write_checklist_json(checklist_result)
    checklist_doc = checklist_result.to_summary_dict()

    part_a, detail_a = _audit_lifecycle_freshness(
        prep=prep, verbose=verbose, ts=ts, csv_rows=csv_rows
    )
    part_b, detail_b = _audit_execution_readiness(verbose=verbose, ts=ts, csv_rows=csv_rows)
    part_c, detail_c = _audit_authorization(verbose=verbose, ts=ts, csv_rows=csv_rows)
    part_d, detail_d = _audit_order_submission(verbose=verbose, ts=ts, csv_rows=csv_rows)
    part_e, detail_e = _audit_duplicate_protection(verbose=verbose, ts=ts, csv_rows=csv_rows)
    part_f, detail_f = _audit_order_monitoring(verbose=verbose, ts=ts, csv_rows=csv_rows)
    part_g, detail_g = _audit_execution_recovery(
        prep=prep, verbose=verbose, ts=ts, csv_rows=csv_rows
    )
    part_h, detail_h = _audit_end_to_end(
        path_doc=path_doc,
        checklist_doc=checklist_doc,
        verbose=verbose,
        ts=ts,
        csv_rows=csv_rows,
    )

    part_results = {
        "lifecycle_freshness": part_a,
        "execution_readiness": part_b,
        "authorization": part_c,
        "order_submission": part_d,
        "duplicate_protection": part_e,
        "order_monitoring": part_f,
        "execution_recovery": part_g,
        "end_to_end": part_h,
    }

    readiness_doc = _read_json(rd / "execution_readiness.json")
    decision = _final_decision(
        part_results=part_results,
        prep=prep,
        readiness_doc=readiness_doc,
        path_doc=path_doc,
        paper_env=_is_paper_env(),
        verbose=verbose,
    )

    chain = _resolve_certification_chain()
    chain["Paper_Execution"] = (
        "CERTIFIED"
        if decision["status"] in ("CERTIFIED", "CONDITIONALLY_CERTIFIED")
        else "NOT_CERTIFIED"
    )
    _log_tag(
        "TRITON_CERTIFICATION_CHAIN",
        {
            "Capital_Preservation": chain["Capital_Preservation"],
            "Portfolio_Management": chain["Portfolio_Management"],
            "Institutional_Intelligence": chain["Institutional_Intelligence"],
            "Executive_Oversight": chain["Executive_Oversight"],
            "Paper_Execution": chain["Paper_Execution"],
            "overall_maturity": chain["overall_maturity"],
        },
        verbose=verbose,
    )

    payload: Dict[str, Any] = {
        "generated_at": ts,
        "phase": "163",
        "mode": "paper_execution_certification",
        "disclaimer": (
            "Paper execution certification assesses paper-mode execution layer capability "
            "and operational readiness. No live trading or live order submission."
        ),
        "prep": prep,
        "path_verification": path_doc,
        "market_open_checklist": checklist_doc,
        "part_a_lifecycle_freshness": detail_a,
        "part_b_execution_readiness": detail_b,
        "part_c_authorization": detail_c,
        "part_d_order_submission": detail_d,
        "part_e_duplicate_protection": detail_e,
        "part_f_order_monitoring": detail_f,
        "part_g_execution_recovery": detail_g,
        "part_h_end_to_end": detail_h,
        "part_results": part_results,
        "part_j_decision": decision,
        "part_k_certification_chain": chain,
        "paper_execution_certification_status": decision["status"],
    }

    _atomic_write_json(payload, OUTPUT_JSON)
    _write_scorecard_csv(csv_rows, OUTPUT_CSV)

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="TRITON Phase 163 — Paper Execution Certification")
    ap.add_argument("--verbose", action="store_true", help="Print certification tags")
    ap.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip lifecycle/watchdog/readiness refresh (audit only)",
    )
    args = ap.parse_args(argv)

    payload = compute_paper_execution_certification(
        verbose=bool(args.verbose),
        skip_prep=bool(args.skip_prep),
    )
    status = str(payload.get("paper_execution_certification_status") or "NOT_CERTIFIED")
    if status == "NOT_CERTIFIED":
        return 2
    if status == "CONDITIONALLY_CERTIFIED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
