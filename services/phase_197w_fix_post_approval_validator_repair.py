"""
phase_197w_fix_post_approval_validator_repair.py
Phase 197W-FIX — Repair post-approval validator classification for paper-only approval lifecycle.
Read-only validation; does not submit broker orders or execute trades.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197w_fix_post_approval_validator_repair.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
EXEC_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
SOAK_JSON = RESULTS / "phase_197t_safe_continuous_observation_soak.json"
W197_CREATE_JSON = RESULTS / "phase_197w_create_paper_operator_approval_artifact.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
PREVIOUS_COMMIT = "a4d11c7e"
PREVIOUS_PHASE = "197Z-CAP"

FINAL_VALID_CURRENT = "PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN"
FINAL_EXPIRED_CLEANUP = "PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED"
FINAL_INVALID_CLEANUP = "PAPER_OPERATOR_APPROVAL_INVALID_CLEANUP_REQUIRED"
FINAL_ABSENT_OPERATOR_REQUIRED = "PAPER_OPERATOR_APPROVAL_ABSENT_OPERATOR_APPROVAL_REQUIRED"

ACCEPTABLE_PAPER_REASON_WHEN_APPROVAL_VALID = frozenset(
    {"explicit_broker_execution_phase_required"}
)
ACCEPTABLE_197N_WHEN_APPROVAL_VALID = frozenset(
    {
        "GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT",
        "GOV_REVIEW_BLOCKED_BY_STALE_PACKET",
    }
)
ACCEPTABLE_197O_WHEN_APPROVAL_VALID = frozenset(
    {
        "MARKET_HOURS_RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION",
        "MARKET_HOURS_RECHECK_BLOCKED_BY_STALE_PACKET",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _git_branch() -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


def _git_clean() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return not (proc.stdout or "").strip()


def _soak_execution(soak_doc: Dict[str, Any]) -> Tuple[int, int, bool]:
    exits = trims = 0
    broker_submit = False
    for cycle in soak_doc.get("cycles") or []:
        exits = max(exits, int(cycle.get("exits_executed") or 0))
        trims = max(trims, int(cycle.get("trims_executed") or 0))
        broker_submit = broker_submit or bool(cycle.get("broker_orders_submitted"))
    return exits, trims, broker_submit


def _run_paper_entry_dry_run() -> Tuple[str, str]:
    env = os.environ.copy()
    env["TRITON_ENABLE_PAPER_EXECUTION"] = "1"
    env["TRITON_ENABLE_AUTO_PAPER_ENTRIES"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "services.paper_entry_loop_stage", "--verbose", "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    gate = "UNKNOWN"
    reason = ""
    for line in text.splitlines():
        if "[PAPER_ENTRY_GATE]" in line:
            if "status=" in line:
                gate = line.split("status=", 1)[-1].split()[0]
            if "reason=" in line:
                reason = line.split("reason=", 1)[-1].strip()
        elif "[PAPER_ENTRY_EXECUTION_BLOCKED]" in line and "reason=" in line and not reason:
            reason = line.split("reason=", 1)[-1].strip()
    if reason.endswith(" (SAFE_BLOCK)"):
        reason = reason[: -len(" (SAFE_BLOCK)")]
    return gate, reason


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _validate_downstream_chain(
    *,
    approval_valid: bool,
    open_orders: int,
    live_blocked: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """Read-only downstream chain checks for 197X/197Z/197Y without broker submission."""
    chain_failures: List[str] = []
    chain_observations: List[str] = []
    from services.paper_broker_execution_authorization_state import (
        PAPER_BROKER_EXECUTION_AUTH_ABSENT,
        classify_paper_broker_execution_authorization,
    )
    from services.phase_197y_loop_mediated_paper_execution_gate import (
        GATE_BLOCKED_EXECUTION_AUTH_REQUIRED,
        evaluate_loop_mediated_paper_execution_gate,
    )
    from services.phase_197z_paper_broker_execution_authorization_creator import run_creator

    w197_create = _read_json(W197_CREATE_JSON)
    x197 = _read_json(X197_JSON)
    z197 = run_creator(create=False, verbose=False)
    y197 = evaluate_loop_mediated_paper_execution_gate(write_output=False)
    exec_state, _, _ = classify_paper_broker_execution_authorization(open_orders_count=open_orders)
    exec_auth_exists = EXEC_AUTH_PATH.is_file()

    chain: Dict[str, Any] = {
        "197w_create_status": w197_create.get("status"),
        "197w_create_final_state": w197_create.get("final_state"),
        "197x_dry_run_status": x197.get("status"),
        "197x_projected_order_count": x197.get("projected_order_count"),
        "197z_creator_status": z197.get("status"),
        "197z_would_create_authorization": z197.get("would_create_authorization"),
        "197z_real_authorization_created": z197.get("real_authorization_created"),
        "197y_gate_final_state": y197.get("final_state"),
        "197y_eligible_for_execution": bool(y197.get("eligible_for_execution")),
        "execution_authorization_state": exec_state,
        "execution_authorization_artifact_exists": exec_auth_exists,
    }

    if approval_valid and w197_create.get("status") not in (None, "PASS", "PASS_SAFE_BLOCKED"):
        chain_failures.append(f"197w_create_status={w197_create.get('status')}")
    if approval_valid and x197.get("status") not in (None, "PASS", "PASS_SAFE_BLOCKED"):
        chain_failures.append(f"197x_dry_run_status={x197.get('status')}")
    if z197.get("real_authorization_created"):
        chain_failures.append("197z_creator_must_not_create_real_authorization_in_validation")
    if z197.get("status") not in ("PASS_DRY_RUN", "PASS_SAFE_BLOCKED", "PASS"):
        chain_failures.append(f"197z_creator_status={z197.get('status')}")
    if not exec_auth_exists and exec_state == PAPER_BROKER_EXECUTION_AUTH_ABSENT:
        if str(y197.get("final_state") or "") != GATE_BLOCKED_EXECUTION_AUTH_REQUIRED:
            chain_failures.append(
                f"197y_expected_execution_auth_required got {y197.get('final_state')}"
            )
    elif exec_auth_exists:
        chain_observations.append("execution_authorization_present_197y_gate_may_be_ready")

    chain["chain_failures"] = chain_failures
    chain["chain_observations"] = chain_observations
    chain["chain_ok"] = not chain_failures
    return chain, chain_failures


def _validate_197w_contract_safety(
    contract: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    scope = str(contract.get("approval_scope") or contract.get("approval_mode") or "").upper()
    if scope != "PAPER_ONLY":
        reasons.append(f"approval_scope={scope}")
    if contract.get("broker_execution_authorized") is True:
        reasons.append("broker_execution_authorized=true")
    if contract.get("live_authorized") is True:
        reasons.append("live_authorized=true")
    if contract.get("raw_execution_authorized") is True:
        reasons.append("raw_execution_authorized=true")
    return not reasons, reasons


def run_phase_197w_fix_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_governance_review_criteria import (
        GOV_REVIEW_BLOCKED_BY_EXPIRED_APPROVAL,
        GOV_REVIEW_BLOCKED_BY_INVALID_APPROVAL,
        GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED,
        GOV_REVIEW_BLOCKED_BY_STALE_PACKET,
        GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED,
        GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT,
        evaluate_governance_review_criteria,
    )
    from services.defensive_exit_dispatch_interlock import load_paper_approved_execution_scope
    from services.defensive_exit_market_hours_governance_recheck import (
        RECHECK_BLOCKED_BY_EXPIRED_APPROVAL,
        RECHECK_BLOCKED_BY_INVALID_APPROVAL,
        RECHECK_BLOCKED_BY_OPERATOR_APPROVAL,
        RECHECK_BLOCKED_BY_STALE_PACKET,
        RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION,
        RECHECK_WAITING,
        run_market_hours_governance_recheck,
    )
    from services.live_block_classifier import scan_live_confirm_artifacts
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_ABSENT,
        PAPER_APPROVAL_EXPIRED,
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
        paper_approval_cleanup_required,
    )
    from services.phase_197u_paper_runtime_governance_packet_refresh import (
        GOV_PACKET_CURRENT_OPERATOR_REVIEW_REQUIRED,
        GOV_PACKET_CURRENT_PAPER_APPROVAL_PRESENT,
        GOV_PACKET_REFRESH_BLOCKED_APPROVAL_EXPIRED,
        GOV_PACKET_REFRESH_BLOCKED_APPROVAL_INVALID,
        classify_packet_refresh_state,
    )

    snapshot_ok = False
    try:
        snap = subprocess.run(
            [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        jstart = (snap.stdout or "").find("{")
        if jstart >= 0:
            snapshot_ok = bool(json.loads(snap.stdout[jstart:]).get("ok"))
    except Exception:
        pass

    soak = {}
    if SOAK_JSON.is_file():
        try:
            soak = json.loads(SOAK_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    exits_executed, trims_executed, broker_orders_submitted = _soak_execution(soak)

    criteria = evaluate_governance_review_criteria(
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )
    recheck = run_market_hours_governance_recheck(
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )

    ap_state, ap_reason, ap_details = classify_paper_operator_approval_artifact(
        broker_snapshot_ok=snapshot_ok
    )
    contract = ap_details.get("contract") or {}
    approval_valid = ap_state == PAPER_APPROVAL_VALID_CURRENT
    approval_expired = ap_state == PAPER_APPROVAL_EXPIRED
    cleanup_required = paper_approval_cleanup_required(broker_snapshot_ok=snapshot_ok)
    contract_safe, contract_failures = _validate_197w_contract_safety(contract)
    scope_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()
    confirm_scan = scan_live_confirm_artifacts()
    valid_live_confirm_count = int(confirm_scan.get("valid_live_confirm_count") or 0)

    live_blocked = bool(criteria.get("live_blocked"))
    open_orders = int(criteria.get("open_orders_count") or 0)
    gov_197n = str(criteria.get("governance_review_state") or "")
    gov_197o = str(recheck.get("market_hours_recheck_state") or "")

    u197_state, _ = classify_packet_refresh_state(
        soak_ok=True,
        live_blocked=live_blocked,
        real_approval_exists=APPROVAL_PATH.is_file(),
        open_orders_count=open_orders,
        broker_orders_submitted=broker_orders_submitted,
        exits_executed=exits_executed,
        trims_executed=trims_executed,
        packet_refreshed=False,
    )

    paper_gate_status, paper_gate_reason = _run_paper_entry_dry_run()
    market_open = bool(criteria.get("market_open"))

    if ap_state == PAPER_APPROVAL_ABSENT:
        final_state = FINAL_ABSENT_OPERATOR_REQUIRED
        expected_197u = GOV_PACKET_CURRENT_OPERATOR_REVIEW_REQUIRED
        if market_open:
            expected_197n = GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED
            expected_197o = RECHECK_BLOCKED_BY_OPERATOR_APPROVAL
            expected_paper_reason = "operator_approval_required"
        else:
            expected_197n = GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED
            expected_197o = RECHECK_WAITING
            expected_paper_reason = "market is closed"
    elif approval_expired:
        final_state = FINAL_EXPIRED_CLEANUP
        expected_paper_reason = "approval_expired"
        expected_197u = GOV_PACKET_REFRESH_BLOCKED_APPROVAL_EXPIRED
        expected_197n = GOV_REVIEW_BLOCKED_BY_EXPIRED_APPROVAL
        expected_197o = RECHECK_BLOCKED_BY_EXPIRED_APPROVAL
    elif approval_valid:
        final_state = FINAL_VALID_CURRENT
        expected_paper_reason = "explicit_broker_execution_phase_required"
        expected_197u = GOV_PACKET_CURRENT_PAPER_APPROVAL_PRESENT
        expected_197n = GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT
        expected_197o = RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION
    else:
        final_state = FINAL_INVALID_CLEANUP
        expected_paper_reason = "approval_invalid"
        expected_197u = GOV_PACKET_REFRESH_BLOCKED_APPROVAL_INVALID
        expected_197n = GOV_REVIEW_BLOCKED_BY_INVALID_APPROVAL
        expected_197o = RECHECK_BLOCKED_BY_INVALID_APPROVAL

    status = "PASS"
    if approval_expired or ap_state == PAPER_APPROVAL_ABSENT:
        status = "PASS_SAFE_BLOCKED"
    elif approval_valid:
        status = "PASS_SAFE_BLOCKED"

    if broker_orders_submitted or exits_executed or trims_executed:
        status = "FAIL"
        notes.append("unexpected execution detected")
    if not live_blocked:
        status = "FAIL"
        notes.append("live_blocked=false")
    if open_orders != 0:
        status = "FAIL"
        notes.append("open_orders!=0")
    if criteria.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if valid_live_confirm_count > 0:
        status = "FAIL"
        notes.append(f"valid_live_confirm_count={valid_live_confirm_count}")

    if approval_valid:
        if not contract_safe:
            status = "FAIL"
            notes.extend(contract_failures)
        if not scope_valid or not scope_symbols:
            status = "FAIL"
            notes.append(f"approved_scope_invalid source={scope_source}")
        if paper_gate_status != "SAFE_BLOCK":
            status = "FAIL"
            notes.append(f"paper_entry expected SAFE_BLOCK got {paper_gate_status}")
        if paper_gate_reason not in ACCEPTABLE_PAPER_REASON_WHEN_APPROVAL_VALID:
            status = "FAIL"
            notes.append(
                f"paper_entry expected one of {sorted(ACCEPTABLE_PAPER_REASON_WHEN_APPROVAL_VALID)} "
                f"got {paper_gate_reason}"
            )
        if gov_197n not in ACCEPTABLE_197N_WHEN_APPROVAL_VALID:
            status = "FAIL"
            notes.append(f"197N unexpected state {gov_197n}")
        elif gov_197n == GOV_REVIEW_BLOCKED_BY_STALE_PACKET:
            notes.append("197N_stale_packet_observation_only")
        if gov_197o not in ACCEPTABLE_197O_WHEN_APPROVAL_VALID:
            status = "FAIL"
            notes.append(f"197O unexpected state {gov_197o}")
        elif gov_197o == RECHECK_BLOCKED_BY_STALE_PACKET:
            notes.append("197O_stale_packet_observation_only")
        if u197_state != expected_197u:
            status = "FAIL"
            notes.append(f"197U expected {expected_197u} got {u197_state}")
        if final_state != FINAL_VALID_CURRENT:
            status = "FAIL"
            notes.append(f"final_state_expected={FINAL_VALID_CURRENT}")
    else:
        if gov_197n != expected_197n:
            status = "FAIL"
            notes.append(f"197N expected {expected_197n} got {gov_197n}")
        if gov_197o != expected_197o:
            status = "FAIL"
            notes.append(f"197O expected {expected_197o} got {gov_197o}")
        if u197_state != expected_197u:
            status = "FAIL"
            notes.append(f"197U expected {expected_197u} got {u197_state}")
        if paper_gate_reason != expected_paper_reason:
            status = "FAIL"
            notes.append(
                f"paper_entry expected reason={expected_paper_reason} got {paper_gate_reason}"
            )
        if paper_gate_status != "SAFE_BLOCK":
            status = "FAIL"
            notes.append(f"paper_entry expected SAFE_BLOCK got {paper_gate_status}")

    if gov_197n == "GOV_REVIEW_BLOCKED_BY_READINESS" and APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("generic readiness block with approval present")

    downstream_chain, chain_failures = _validate_downstream_chain(
        approval_valid=approval_valid,
        open_orders=open_orders,
        live_blocked=live_blocked,
    )
    if approval_valid and chain_failures:
        status = "FAIL"
        notes.extend(chain_failures)
    notes.extend(downstream_chain.get("chain_observations") or [])

    payload: Dict[str, Any] = {
        "phase": "197W-VF",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "approval_artifact_state": ap_state,
        "approval_artifact_reason": ap_reason,
        "approval_valid": approval_valid,
        "approval_expired": approval_expired,
        "cleanup_required": cleanup_required,
        "open_orders": open_orders,
        "live_blocked": live_blocked,
        "broker_orders_submitted": broker_orders_submitted,
        "exits_executed": exits_executed,
        "trims_executed": trims_executed,
        "197u_state": u197_state,
        "197n_state": gov_197n,
        "197o_state": gov_197o,
        "paper_entry_dry_run_status": paper_gate_status,
        "paper_entry_dry_run_reason": paper_gate_reason,
        "final_state": final_state,
        "contract_safe": contract_safe,
        "contract_failures": contract_failures,
        "approved_scope_symbols": scope_symbols,
        "approved_scope_source": scope_source,
        "approved_scope_valid": scope_valid,
        "valid_live_confirm_count": valid_live_confirm_count,
        "downstream_chain": downstream_chain,
        "cleanup_guidance": (
            "python -m services.phase_197w_fix_expired_approval_cleanup --execute"
            if cleanup_required
            else None
        ),
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197W-FIX] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197W-FIX] status={status} ap_state={ap_state} final_state={final_state} "
            f"paper_reason={paper_gate_reason}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197W-FIX post-approval validator repair")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197w_fix_validation(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
