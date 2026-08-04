"""DO-007 runtime recovery guidance tests — mocks/fixtures only; no real processes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from dashboard.operator_console import DEFAULT_OPERATOR_PAGE, OPERATOR_PAGES
from services.operator_runtime_recovery_guidance import (
    ACTION_BLOCKED,
    ACTION_CONSIDER_RESTART,
    ACTION_CONSIDER_START,
    ACTION_MANUAL,
    ACTION_NO_ACTION,
    ACTION_WAIT,
    CLASS_ACTIVE_CONTROL_CONFLICT,
    CLASS_FULLY_STOPPED,
    CLASS_HEALTHY,
    CLASS_INSUFFICIENT_EVIDENCE,
    CLASS_LIVE_EXECUTION_ENABLED_INCIDENT,
    CLASS_PARTIALLY_RUNNING,
    CLASS_RESTART_IN_PROGRESS,
    CLASS_RESTART_PARTIAL_OR_TIMED_OUT,
    CLASS_START_IN_PROGRESS,
    CLASS_START_PARTIAL_OR_TIMED_OUT,
    CLASS_STOP_IN_PROGRESS,
    CLASS_STOP_PARTIAL_OR_TIMED_OUT,
    CLASS_STALE_OR_INCONSISTENT_LOCK,
    CLASS_UNVERIFIED_IDENTITY,
    append_guidance_audit,
    build_runtime_recovery_guidance,
)

ROOT = Path(__file__).resolve().parents[1]


def _loop(
    state: str,
    *,
    pids=None,
    lock_exists: bool = False,
    lock_pid=None,
) -> dict:
    return {
        "state": state,
        "process_pids": list(pids or []),
        "lock_exists": lock_exists,
        "lock_pid": lock_pid,
        "explanation": state,
    }


def _obs(**kwargs) -> dict:
    base = {
        "classifications": {
            "continuous": _loop("NOT_RUNNING"),
            "watchdog": _loop("NOT_RUNNING"),
        },
        "start_eligibility": {
            "decision": "START_ALLOWED",
            "start_button_enabled": True,
            "guard_active": False,
            "live_execution_blocked": True,
            "paper_arm_status": "ARMED",
        },
        "stop_eligibility": {
            "decision": "STOP_ALREADY_STOPPED",
            "stop_button_enabled": False,
            "ui_state": "STOP_UNAVAILABLE",
            "live_execution_blocked": True,
        },
        "restart_eligibility": {
            "decision": "RESTART_ALREADY_STOPPED",
            "restart_button_enabled": True,
            "ui_state": "RESTART_CONFIRMATION_REQUIRED",
            "live_execution_blocked": True,
        },
        "live_execution_blocked": True,
        "live_execution_appears_enabled": False,
        "paper_arm_status": "ARMED",
        "treat_unblocked_as_incident": True,
    }
    base.update(kwargs)
    return base


def _guide(observation: dict, **kwargs):
    return build_runtime_recovery_guidance(observation=observation, **kwargs)


def _src_guidance() -> str:
    return (ROOT / "services" / "operator_runtime_recovery_guidance.py").read_text(encoding="utf-8")


def _src_dashboard() -> str:
    return (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")


def test_01_healthy_runtime_recommends_no_action():
    o = _obs(
        classifications={
            "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
            "watchdog": _loop("HEALTHY_RUNNING", pids=[2], lock_exists=True, lock_pid=2),
        },
        start_eligibility={
            "decision": "ALREADY_RUNNING",
            "start_button_enabled": False,
            "guard_active": False,
            "live_execution_blocked": True,
            "paper_arm_status": "ARMED",
        },
        restart_eligibility={
            "decision": "RESTART_ELIGIBLE_RUNNING",
            "restart_button_enabled": True,
            "live_execution_blocked": True,
        },
    )
    r = _guide(o)
    assert r["runtime_overall_status"] == CLASS_HEALTHY
    assert r["safe_next_action"] == ACTION_NO_ACTION


def test_02_fully_stopped_may_recommend_consider_start():
    r = _guide(_obs())
    assert r["runtime_overall_status"] == CLASS_FULLY_STOPPED
    assert r["safe_next_action"] == ACTION_CONSIDER_START
    assert r["eligible_protected_control"] == "START_TRITON"


def test_03_fully_stopped_guidance_does_not_invoke_start():
    start = Mock(return_value={"decision": "START_ALLOWED", "start_button_enabled": True})
    r = build_runtime_recovery_guidance(
        observation=_obs(),
        start_eligibility_fn=start,
    )
    # Injected observation path does not call start_eligibility_fn when key present
    assert r["execution_attempted"] is False
    assert r["control_transaction_created"] is False
    assert "execute_start_triton" not in _src_guidance()


def test_04_partial_runtime_classified_honestly():
    o = _obs(
        classifications={
            "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
            "watchdog": _loop("NOT_RUNNING"),
        },
        restart_eligibility={
            "decision": "RESTART_ELIGIBLE_CONTINUOUS_ONLY",
            "restart_button_enabled": True,
            "live_execution_blocked": True,
        },
    )
    r = _guide(o)
    assert r["runtime_overall_status"] == CLASS_PARTIALLY_RUNNING
    assert "continuous" in r["affected_targets"]


def test_05_missing_process_identified():
    o = _obs(report_missing_process=True)
    r = _guide(o)
    assert any("missing process" in n for n in r["incident_notes"])


def test_06_unexpected_process_evidence_identified():
    o = _obs(
        classifications={
            "continuous": _loop("DUPLICATE_PROCESSES", pids=[1, 2], lock_exists=True, lock_pid=1),
            "watchdog": _loop("NOT_RUNNING"),
        }
    )
    r = _guide(o)
    assert any("extra process" in n for n in r["incident_notes"])


def test_07_pid_mismatch_requires_investigation():
    o = _obs(
        classifications={
            "continuous": _loop("PID_MISMATCH", pids=[1], lock_exists=True, lock_pid=9),
            "watchdog": _loop("NOT_RUNNING"),
        }
    )
    r = _guide(o)
    assert r["runtime_overall_status"] == CLASS_UNVERIFIED_IDENTITY
    assert r["manual_investigation_required"] is True


def test_08_creation_time_mismatch_requires_investigation():
    r = _guide(_obs(creation_time_mismatch=True, unverified_identity=True))
    assert r["runtime_overall_status"] == CLASS_UNVERIFIED_IDENTITY
    assert r["manual_investigation_required"] is True


def test_09_command_fingerprint_mismatch_requires_investigation():
    o = _obs(
        classifications={
            "continuous": _loop("COMMAND_MISMATCH", pids=[1], lock_exists=True, lock_pid=1),
            "watchdog": _loop("NOT_RUNNING"),
        }
    )
    r = _guide(o)
    assert r["runtime_overall_status"] == CLASS_UNVERIFIED_IDENTITY


def test_10_missing_owner_lock_reported():
    o = _obs(
        classifications={
            "continuous": _loop("PROCESS_WITHOUT_VALID_LOCK", pids=[1], lock_exists=False),
            "watchdog": _loop("NOT_RUNNING"),
        }
    )
    r = _guide(o)
    assert any("missing owner lock" in n for n in r["incident_notes"])


def test_11_stale_lock_is_not_deleted():
    o = _obs(
        classifications={
            "continuous": _loop("STALE_LOCK_NO_PROCESS", lock_exists=True, lock_pid=1),
            "watchdog": _loop("NOT_RUNNING"),
        },
        stale_lock=True,
    )
    r = _guide(o)
    assert r["runtime_locks_deleted"] is False
    assert "delete_runtime_lock" in r["prohibited_actions"]


def test_12_conflicting_lock_ownership_blocks_recovery_advice():
    r = _guide(_obs(conflicting_lock_owner=True, stale_lock=True))
    assert r["runtime_overall_status"] == CLASS_STALE_OR_INCONSISTENT_LOCK
    assert r["safe_next_action"] in (ACTION_MANUAL, ACTION_BLOCKED)


def test_13_malformed_evidence_produces_insufficient_evidence():
    r = _guide(_obs(malformed=True))
    assert r["runtime_overall_status"] == CLASS_INSUFFICIENT_EVIDENCE


def test_14_missing_diagnostics_does_not_guess_action():
    r = _guide(_obs(missing_diagnostics=True, insufficient_evidence=True))
    assert r["runtime_overall_status"] == CLASS_INSUFFICIENT_EVIDENCE
    assert r["eligible_protected_control"] is None


def test_15_stop_in_progress_blocks_start():
    r = _guide(_obs(stop_in_progress=True))
    assert r["runtime_overall_status"] == CLASS_STOP_IN_PROGRESS
    assert "start_blocked" in r["blockers"]
    assert r["eligible_protected_control"] is None


def test_16_stop_in_progress_blocks_duplicate_stop():
    r = _guide(_obs(stop_in_progress=True))
    assert "stop_duplicate_blocked" in r["blockers"]


def test_17_stop_in_progress_blocks_restart_recommendation():
    r = _guide(_obs(stop_in_progress=True))
    assert "restart_blocked" in r["blockers"]
    assert r["safe_next_action"] == ACTION_WAIT


def test_18_stop_ack_alone_does_not_mean_completion():
    r = _guide(_obs(stop_ack_only=True))
    assert r["runtime_overall_status"] == CLASS_STOP_PARTIAL_OR_TIMED_OUT


def test_19_stop_partial_completion_blocks_start():
    r = _guide(_obs(stop_partial_or_timed_out=True))
    assert "start_prohibited_until_stop_complete" in r["blockers"]


def test_20_stop_timeout_does_not_recommend_force_termination():
    r = _guide(_obs(stop_partial_or_timed_out=True, stop_timed_out=True))
    assert r["forced_termination_used"] is False
    assert "force_terminate" in r["prohibited_actions"]
    assert "force termination" in r["operator_message"].lower() or any(
        "force termination" in n for n in r["incident_notes"]
    )


def test_21_start_in_progress_blocks_duplicate_start():
    r = _guide(_obs(start_in_progress=True))
    assert r["runtime_overall_status"] == CLASS_START_IN_PROGRESS
    assert "start_duplicate_blocked" in r["blockers"]


def test_22_ordinary_start_in_progress_is_not_called_timed_out():
    r = _guide(_obs(start_in_progress=True, start_timed_out=False))
    assert r["runtime_overall_status"] == CLASS_START_IN_PROGRESS
    assert r["runtime_overall_status"] != CLASS_START_PARTIAL_OR_TIMED_OUT


def test_23_explicit_start_timeout_reported_honestly():
    r = _guide(_obs(start_timed_out=True, start_partial_or_timed_out=True))
    assert r["runtime_overall_status"] == CLASS_START_PARTIAL_OR_TIMED_OUT
    assert any("Start timeout" in n for n in r["incident_notes"])


def test_24_start_timeout_does_not_automatically_retry():
    r = _guide(_obs(start_timed_out=True, start_partial_or_timed_out=True))
    assert "no_automatic_start_retry" in r["blockers"]
    assert r["execution_attempted"] is False


def test_25_partial_start_verification_not_reported_as_healthy():
    r = _guide(
        _obs(
            start_partial_or_timed_out=True,
            classifications={
                "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
                "watchdog": _loop("HEALTHY_RUNNING", pids=[2], lock_exists=True, lock_pid=2),
            },
        )
    )
    assert r["runtime_overall_status"] != CLASS_HEALTHY


def test_26_restart_stop_stage_displayed_separately():
    r = _guide(
        _obs(
            restart_in_progress=True,
            restart_transaction={
                "stage": "STOP_IN_PROGRESS",
                "stop_result": "STOP_IN_PROGRESS",
                "start_result": None,
            },
        )
    )
    assert r["restart_stop_stage"] is not None
    assert (
        "STOP" in str(r["restart_stop_stage"]).upper() or r["restart_stage"] == "STOP_IN_PROGRESS"
    )


def test_27_restart_start_stage_displayed_separately():
    r = _guide(
        _obs(
            restart_in_progress=True,
            restart_transaction={
                "stage": "START_ELIGIBILITY_CONFIRMED",
                "stop_result": "STOP_COMPLETED",
                "start_result": None,
            },
        )
    )
    assert r["restart_stage"] == "START_ELIGIBILITY_CONFIRMED"
    assert "START" in str(r["restart_start_stage"] or r["restart_stage"]).upper()


def test_28_restart_in_progress_blocks_duplicate_actions():
    r = _guide(_obs(restart_in_progress=True, restart_transaction={"stage": "STOP_REQUESTED"}))
    assert r["runtime_overall_status"] == CLASS_RESTART_IN_PROGRESS
    assert "restart_duplicate_blocked" in r["blockers"]


def test_29_restart_partial_does_not_recommend_automatic_retry():
    r = _guide(
        _obs(
            restart_partial_or_timed_out=True,
            restart_transaction={
                "result": "RESTART_PARTIAL_COMPLETION",
                "stop_result": "STOP_COMPLETED",
            },
        )
    )
    assert r["runtime_overall_status"] == CLASS_RESTART_PARTIAL_OR_TIMED_OUT
    assert (
        "automatically" in r["operator_message"].lower() or r["safe_next_action"] == ACTION_MANUAL
    )


def test_30_restart_timeout_does_not_recommend_force_termination():
    r = _guide(
        _obs(
            restart_partial_or_timed_out=True,
            restart_timed_out=True,
            restart_transaction={"result": "RESTART_TIMED_OUT"},
        )
    )
    assert r["forced_termination_used"] is False
    assert "force_terminate" in r["prohibited_actions"]


def test_31_transaction_conflict_blocks_new_action_recommendations():
    r = _guide(_obs(control_conflict=True, start_in_progress=True, stop_in_progress=True))
    assert r["runtime_overall_status"] == CLASS_ACTIVE_CONTROL_CONFLICT
    assert r["safe_next_action"] == ACTION_BLOCKED


def test_32_guidance_never_creates_start_transactions():
    r = _guide(_obs())
    assert r["control_transaction_created"] is False
    assert "execute_start_triton" not in _src_guidance()


def test_33_guidance_never_creates_stop_requests():
    assert "execute_stop_triton" not in _src_guidance()
    assert "prepare_cooperative_shutdown(" not in _src_guidance()


def test_34_guidance_never_creates_restart_transactions():
    assert "execute_restart_triton" not in _src_guidance()


def test_35_guidance_never_creates_preflight_tokens():
    r = _guide(_obs())
    assert r["preflight_created"] is False
    assert "issue_restart_preflight" not in _src_guidance()
    assert "issue_stop_preflight" not in _src_guidance()


def test_36_guidance_id_cannot_be_used_as_control_authority():
    r = _guide(_obs())
    assert r["guidance_id_is_authority"] is False


def test_37_existing_control_tokens_are_not_exposed():
    r = _guide(_obs())
    blob = json.dumps(r)
    assert "preflight_token" not in blob
    assert "RESTART TRITON" not in blob


def test_38_frontend_cannot_provide_pid():
    r = build_runtime_recovery_guidance(observation=_obs(), pid=1)
    assert r["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_39_frontend_cannot_provide_target():
    assert (
        build_runtime_recovery_guidance(observation=_obs(), target="continuous")["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_40_frontend_cannot_provide_path():
    assert (
        build_runtime_recovery_guidance(observation=_obs(), path="C:\\x")["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_41_frontend_cannot_provide_command():
    assert (
        build_runtime_recovery_guidance(observation=_obs(), command="evil")["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_42_frontend_cannot_provide_arguments():
    assert (
        build_runtime_recovery_guidance(observation=_obs(), arguments=["x"])["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_43_frontend_cannot_provide_environment():
    assert (
        build_runtime_recovery_guidance(observation=_obs(), environment={"A": "1"})["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_44_frontend_cannot_provide_timeout():
    assert (
        build_runtime_recovery_guidance(observation=_obs(), timeout=1)["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_45_frontend_cannot_provide_lock_path():
    assert (
        build_runtime_recovery_guidance(observation=_obs(), lock_path="x.lock")["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_46_frontend_cannot_select_controller_function():
    assert (
        build_runtime_recovery_guidance(
            observation=_obs(), controller_function="execute_start_triton"
        )["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_47_page_load_does_not_trigger_recovery():
    src = _src_dashboard()
    block = src.split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    assert "execute_start_triton" not in block
    assert "execute_stop_triton" not in block
    assert "execute_restart_triton" not in block
    assert "build_runtime_recovery_guidance()" in block


def test_48_refresh_does_not_trigger_recovery():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "st.rerun()" not in block


def test_49_query_parameters_do_not_trigger_recovery():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "query_params" not in block and "experimental_get_query_params" not in block


def test_50_url_fragments_do_not_trigger_recovery():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "fragment" not in block.lower()


def test_51_session_restoration_does_not_trigger_recovery():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "session_state[" not in block or "recovery_auto" not in block


def test_52_navigation_to_protected_control_does_not_submit_it():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "Confirm Start" not in block
    assert "Confirm Stop" not in block
    assert "Confirm Restart" not in block


def test_53_protected_start_eligibility_freshly_evaluated():
    start = Mock(
        return_value={
            "decision": "START_ALLOWED",
            "start_button_enabled": True,
            "guard_active": False,
            "live_execution_blocked": True,
            "paper_arm_status": "ARMED",
        }
    )
    r = build_runtime_recovery_guidance(
        observation={
            "classifications": {
                "continuous": _loop("NOT_RUNNING"),
                "watchdog": _loop("NOT_RUNNING"),
            },
            "live_execution_blocked": True,
            "live_execution_appears_enabled": False,
            "paper_arm_status": "ARMED",
            # omit start/stop/restart eligibility to force fresh calls
            "stop_eligibility": {
                "decision": "STOP_ALREADY_STOPPED",
                "stop_button_enabled": False,
                "live_execution_blocked": True,
            },
            "restart_eligibility": {
                "decision": "RESTART_ALREADY_STOPPED",
                "restart_button_enabled": True,
                "live_execution_blocked": True,
            },
        },
        start_eligibility_fn=start,
    )
    assert start.call_count == 1
    assert r["fresh_start_eligibility_decision"] == "START_ALLOWED"


def test_54_protected_stop_eligibility_freshly_evaluated():
    stop = Mock(
        return_value={
            "decision": "STOP_ELIGIBLE_BOTH",
            "stop_button_enabled": True,
            "live_execution_blocked": True,
        }
    )
    build_runtime_recovery_guidance(
        observation={
            "classifications": {
                "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
                "watchdog": _loop("HEALTHY_RUNNING", pids=[2], lock_exists=True, lock_pid=2),
            },
            "start_eligibility": {
                "decision": "ALREADY_RUNNING",
                "start_button_enabled": False,
                "guard_active": False,
                "live_execution_blocked": True,
                "paper_arm_status": "ARMED",
            },
            "restart_eligibility": {
                "decision": "RESTART_ELIGIBLE_RUNNING",
                "restart_button_enabled": True,
                "live_execution_blocked": True,
            },
            "live_execution_blocked": True,
            "live_execution_appears_enabled": False,
            "paper_arm_status": "ARMED",
        },
        stop_eligibility_fn=stop,
    )
    assert stop.call_count == 1


def test_55_protected_restart_eligibility_freshly_evaluated():
    restart = Mock(
        return_value={
            "decision": "RESTART_ELIGIBLE_RUNNING",
            "restart_button_enabled": True,
            "live_execution_blocked": True,
        }
    )
    build_runtime_recovery_guidance(
        observation={
            "classifications": {
                "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
                "watchdog": _loop("HEALTHY_RUNNING", pids=[2], lock_exists=True, lock_pid=2),
            },
            "start_eligibility": {
                "decision": "ALREADY_RUNNING",
                "start_button_enabled": False,
                "guard_active": False,
                "live_execution_blocked": True,
                "paper_arm_status": "ARMED",
            },
            "stop_eligibility": {
                "decision": "STOP_ELIGIBLE_BOTH",
                "stop_button_enabled": True,
                "live_execution_blocked": True,
            },
            "live_execution_blocked": True,
            "live_execution_appears_enabled": False,
            "paper_arm_status": "ARMED",
        },
        restart_eligibility_fn=restart,
    )
    assert restart.call_count == 1


def test_56_stale_eligibility_evidence_is_rejected():
    # When observation omits eligibility, fresh fn results are used (not stale cached tokens)
    start = Mock(
        return_value={
            "decision": "START_REFUSED_UNSAFE_STATE",
            "start_button_enabled": False,
            "guard_active": False,
            "live_execution_blocked": True,
            "paper_arm_status": "ARMED",
        }
    )
    r = build_runtime_recovery_guidance(
        observation={
            "classifications": {
                "continuous": _loop("NOT_RUNNING"),
                "watchdog": _loop("NOT_RUNNING"),
            },
            "stop_eligibility": {
                "decision": "STOP_ALREADY_STOPPED",
                "stop_button_enabled": False,
                "live_execution_blocked": True,
            },
            "restart_eligibility": {
                "decision": "RESTART_REFUSED_UNSAFE_STATE",
                "restart_button_enabled": False,
                "live_execution_blocked": True,
            },
            "live_execution_blocked": True,
            "live_execution_appears_enabled": False,
            "paper_arm_status": "ARMED",
        },
        start_eligibility_fn=start,
    )
    assert r["fresh_start_eligibility_decision"] == "START_REFUSED_UNSAFE_STATE"
    assert r["eligible_protected_control"] is None


def test_57_guidance_does_not_bypass_do003():
    assert "execute_start_triton" not in _src_guidance()
    r = _guide(_obs())
    assert r["advisory_only"] is True


def test_58_guidance_does_not_bypass_do005():
    assert "execute_stop_triton" not in _src_guidance()


def test_59_guidance_does_not_bypass_do006():
    assert "execute_restart_triton" not in _src_guidance()


def test_60_start_stop_restart_mutual_exclusion_preserved():
    r = _guide(_obs(start_in_progress=True, stop_in_progress=True, control_conflict=True))
    assert r["runtime_overall_status"] == CLASS_ACTIVE_CONTROL_CONFLICT
    assert r["eligible_protected_control"] is None


def test_61_paper_authorization_missing_reported_separately():
    r = _guide(_obs(paper_arm_status="MISSING"))
    assert r["paper_authorization_missing"] is True
    assert any("paper" in w.lower() for w in r["warnings"]) or any(
        "paper" in n.lower() for n in r["incident_notes"]
    )


def test_62_guidance_does_not_create_paper_authorization():
    r = _guide(_obs(paper_arm_status="MISSING"))
    assert r["paper_arm_mutated"] is False
    assert "arm_paper" in r["prohibited_actions"]


def test_63_guidance_does_not_extend_paper_authorization():
    assert "extend_paper_authorization" in _guide(_obs())["prohibited_actions"]


def test_64_live_execution_blocked_displayed_as_safe_state():
    r = _guide(_obs(live_execution_blocked=True, live_execution_appears_enabled=False))
    assert r["live_execution_blocked"] is True
    assert any("LIVE_EXECUTION_BLOCKED" in w for w in r["warnings"])


def test_65_possible_live_enablement_becomes_critical_incident():
    r = _guide(
        _obs(
            live_execution_blocked=False,
            live_execution_appears_enabled=True,
            classifications={
                "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
                "watchdog": _loop("HEALTHY_RUNNING", pids=[2], lock_exists=True, lock_pid=2),
            },
        )
    )
    assert r["runtime_overall_status"] == CLASS_LIVE_EXECUTION_ENABLED_INCIDENT
    assert r["severity"] == "CRITICAL"
    assert r["escalation_required"] is True


def test_66_guidance_does_not_modify_live_execution_flags():
    r = _guide(_obs(live_execution_appears_enabled=True, live_execution_blocked=False))
    assert r["runtime_artifact_modified"] is False
    assert "enable_live_execution" in r["prohibited_actions"]


def test_67_no_broker_call_occurs():
    assert _guide(_obs())["broker_called"] is False


def test_68_no_order_submitted():
    assert _guide(_obs())["orders_submitted"] is False


def test_69_no_order_cancelled():
    assert _guide(_obs())["orders_cancelled"] is False


def test_70_no_approval_created():
    assert _guide(_obs())["approvals_created"] is False


def test_71_no_authorization_created():
    assert _guide(_obs())["authorizations_created"] is False


def test_72_governance_unchanged():
    assert _guide(_obs())["governance_modified"] is False


def test_73_lifecycle_unchanged():
    assert _guide(_obs())["lifecycle_modified"] is False


def test_74_allocation_unchanged():
    assert _guide(_obs())["allocation_modified"] is False


def test_75_risk_unchanged():
    assert _guide(_obs())["risk_modified"] is False


def test_76_scheduled_tasks_unchanged():
    assert _guide(_obs())["scheduled_task_modified"] is False
    src = _src_guidance()
    assert "schtasks" not in src and "Start-ScheduledTask" not in src


def test_77_no_process_control_api_invoked():
    """Validation-layer source inspection (not a production runtime scanner)."""
    src = _src_guidance()
    assert "import subprocess" not in src
    assert "Popen(" not in src
    assert "os.kill(" not in src
    assert "process.kill(" not in src
    assert "taskkill.exe" not in src.lower()
    assert "Stop-Process -" not in src
    # Guidance result fields remain non-mutating
    r = _guide(_obs())
    assert r["execution_attempted"] is False
    assert r["forced_termination_used"] is False
    assert r["broker_called"] is False


def test_78_no_forced_termination_exists():
    src = _src_guidance()
    assert "os.kill(" not in src
    assert "taskkill.exe" not in src.lower()
    assert "Stop-Process -" not in src
    assert "SIGKILL" not in src and "SIGTERM" not in src


def test_79_no_lock_deletion_exists():
    src = _src_guidance()
    assert ".unlink(" not in src and "os.remove(" not in src and "rmtree" not in src


def test_80_no_lock_repair_exists():
    assert "repair_lock" not in _src_guidance()
    assert "Clear-OrphanLoopLock" not in _src_guidance()


def test_81_audit_output_is_append_only_if_enabled(tmp_path):
    path = tmp_path / "guidance.jsonl"
    assert append_guidance_audit({"guidance_id": "g1", "token": "secret"}, path=path)["ok"]
    assert append_guidance_audit({"guidance_id": "g2"}, path=path)["ok"]
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_82_audit_output_is_sanitized(tmp_path):
    path = tmp_path / "guidance.jsonl"
    append_guidance_audit(
        {"preflight_token": "abc", "confirmation_phrase": "RESTART TRITON", "x": 1},
        path=path,
    )
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["preflight_token"] == "<redacted>"
    assert row["confirmation_phrase"] == "<redacted>"


def test_83_raw_tokens_never_recorded(tmp_path):
    path = tmp_path / "g.jsonl"
    r = build_runtime_recovery_guidance(observation=_obs(), write_audit=True, audit_path=path)
    text = path.read_text(encoding="utf-8")
    assert r["guidance_id"] in text
    assert "preflight_token" not in text or "<redacted>" in text


def test_84_credentials_never_recorded(tmp_path):
    path = tmp_path / "g.jsonl"
    append_guidance_audit({"password": "x", "api_key": "y"}, path=path)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["password"] == "<redacted>"
    assert row["api_key"] == "<redacted>"


def test_85_confirmation_input_never_recorded(tmp_path):
    path = tmp_path / "g.jsonl"
    append_guidance_audit({"confirmation_phrase": "START TRITON"}, path=path)
    assert (
        json.loads(path.read_text(encoding="utf-8").strip())["confirmation_phrase"] == "<redacted>"
    )


def test_86_guidance_deterministic_for_same_evidence():
    o = _obs()
    a = _guide(o)
    b = _guide(o)
    assert a["runtime_overall_status"] == b["runtime_overall_status"]
    assert a["safe_next_action"] == b["safe_next_action"]
    assert a["severity"] == b["severity"]


def test_87_classification_precedence_is_conservative():
    r = _guide(
        _obs(
            control_conflict=True,
            start_in_progress=True,
            stop_in_progress=True,
            classifications={
                "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
                "watchdog": _loop("HEALTHY_RUNNING", pids=[2], lock_exists=True, lock_pid=2),
            },
        )
    )
    assert r["runtime_overall_status"] == CLASS_ACTIVE_CONTROL_CONFLICT


def test_88_contradictory_evidence_cannot_produce_healthy():
    r = _guide(
        _obs(
            unverified_identity=True,
            creation_time_mismatch=True,
            classifications={
                "continuous": _loop("HEALTHY_RUNNING", pids=[1], lock_exists=True, lock_pid=1),
                "watchdog": _loop("HEALTHY_RUNNING", pids=[2], lock_exists=True, lock_pid=2),
            },
        )
    )
    assert r["runtime_overall_status"] != CLASS_HEALTHY


def test_89_unknown_target_evidence_cannot_select_operational_target():
    o = _obs()
    o["classifications"]["evil"] = _loop("HEALTHY_RUNNING", pids=[1])
    r = _guide(o)
    assert "evil" not in r["affected_targets"]
    assert r["runtime_overall_status"] == CLASS_INSUFFICIENT_EVIDENCE or r.get("source_evidence")


def test_90_advisory_only_notice_is_rendered():
    src = _src_dashboard()
    assert "Recovery guidance is advisory only" in src
    assert "does not start, stop, restart, terminate" in src


def test_91_operations_page_shows_classification_and_severity():
    src = _src_dashboard()
    assert "Runtime Recovery Guidance" in src
    assert "runtime_overall_status" in src
    assert "severity" in src


def test_92_operations_page_shows_affected_targets():
    assert "Affected targets:" in _src_dashboard()


def test_93_operations_page_shows_evidence_summary():
    assert "Recovery evidence summary" in _src_dashboard()


def test_94_operations_page_shows_recommended_safe_action():
    assert "Recommended safe action:" in _src_dashboard()


def test_95_operations_page_shows_blockers_and_warnings():
    src = _src_dashboard()
    assert "Blockers:" in src and "Warnings:" in src


def test_96_operations_page_shows_manual_investigation_status():
    assert "Manual investigation required:" in _src_dashboard()


def test_97_operations_page_shows_paper_and_live_separately():
    src = _src_dashboard()
    assert "Live execution blocked:" in src
    assert "Paper authorization:" in src


def test_98_operations_page_does_not_promise_successful_recovery():
    assert "does not promise successful recovery" in _src_dashboard()


def test_99_operations_page_does_not_describe_restart_as_immediate():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "does not promise successful recovery, immediate restart" in block
    assert "Restart is immediate" not in block
    assert "immediate completion" not in block.lower()


def test_100_operations_page_does_not_imply_forced_termination():
    assert (
        "forced termination"
        in _src_dashboard()
        .split("def _render_runtime_recovery_guidance", 1)[1]
        .split("\ndef ", 1)[0]
    )


def test_101_operations_page_does_not_imply_lock_repair():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "lock repair" in block


def test_102_operations_page_does_not_imply_paper_authorization():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "does not create or extend authorization" in block


def test_103_operations_page_does_not_imply_live_enablement():
    block = (
        _src_dashboard().split("def _render_runtime_recovery_guidance", 1)[1].split("\ndef ", 1)[0]
    )
    assert "never enable live" in block
    assert "live enablement" in block


def test_104_no_real_process_started_during_tests():
    assert _guide(_obs())["execution_attempted"] is False


def test_105_no_real_process_stopped_during_tests():
    assert "prepare_cooperative_shutdown(" not in _src_guidance()


def test_106_no_real_process_restarted_during_tests():
    assert "execute_restart_triton" not in _src_guidance()


def test_107_no_real_runtime_lock_modified_during_tests(tmp_path):
    live = ROOT / "data" / "live"
    # Guidance with tmp observation must not touch real locks
    before = sorted(p.name for p in live.glob("*.lock")) if live.is_dir() else []
    _guide(_obs())
    after = sorted(p.name for p in live.glob("*.lock")) if live.is_dir() else []
    assert before == after
    assert _guide(_obs())["runtime_locks_deleted"] is False


def test_108_no_real_control_request_created_during_tests():
    assert _guide(_obs())["control_transaction_created"] is False


def test_109_no_real_acknowledgement_created_during_tests():
    assert "write_ack" not in _src_guidance()


def test_110_do002_operator_surface_remains_intact():
    assert OPERATOR_PAGES[0] == "Operations" and DEFAULT_OPERATOR_PAGE == "Operations"


def test_111_do003_start_surface_remains_intact():
    src = _src_dashboard()
    assert "execute_start_triton" in src and "Confirm Start" in src


def test_112_do004_controller_surface_remains_intact():
    from services.operator_cooperative_shutdown import ACTION

    assert ACTION == "REQUEST_COOPERATIVE_SHUTDOWN"


def test_113_do004_loop_tests_file_remains():
    assert (ROOT / "tests/test_loop_cooperative_shutdown_regression_do004.py").is_file()


def test_114_do005_stop_surface_remains_intact():
    from services.operator_stop_action import ACTION_STOP, CONFIRMATION_PHRASE

    assert ACTION_STOP == "STOP_TRITON" and CONFIRMATION_PHRASE == "STOP TRITON"


def test_115_do006_restart_surface_remains_intact():
    from services.operator_restart_action import ACTION_RESTART, CONFIRMATION_PHRASE

    assert ACTION_RESTART == "RESTART_TRITON" and CONFIRMATION_PHRASE == "RESTART TRITON"


def test_116_combined_regression_marker_and_advisory_contract():
    r = _guide(_obs())
    assert r["advisory_only"] is True
    assert r["phase"] == "DO-007"
    assert r["guidance_id_is_authority"] is False
    assert (ROOT / "tests/test_operator_protected_restart_do006.py").is_file()
    assert (ROOT / "tests/test_operator_protected_stop_do005.py").is_file()
    assert (ROOT / "tests/test_operator_runtime_controller_do003.py").is_file()
    assert (ROOT / "tests/test_operator_console_do002.py").is_file()
