"""
operator_runtime_recovery_guidance.py
Phase DO-007 — Read-only Operator Runtime Recovery and Incident Guidance.

Observes, classifies, explains, and recommends.
Never Starts, Stops, Restarts, terminates, repairs, deletes, authorizes, or modifies.
Never creates preflights, tokens, requests, acknowledgements, or control transactions.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
AUDIT_PATH = RESULTS / "operator_runtime_recovery_guidance.jsonl"

PHASE = "DO-007"
CONTROLLER_VERSION = "do007-recovery-guidance-v1"
SCHEMA_VERSION = 1
ADVISORY_ONLY = True

TYPED_TARGETS = frozenset({"continuous", "watchdog"})

# Runtime overall classifications (Part 4)
CLASS_HEALTHY = "HEALTHY"
CLASS_FULLY_STOPPED = "FULLY_STOPPED"
CLASS_PARTIALLY_RUNNING = "PARTIALLY_RUNNING"
CLASS_STALE_OR_INCONSISTENT_LOCK = "STALE_OR_INCONSISTENT_LOCK_EVIDENCE"
CLASS_UNVERIFIED_IDENTITY = "UNVERIFIED_PROCESS_IDENTITY"
CLASS_STOP_IN_PROGRESS = "STOP_IN_PROGRESS"
CLASS_STOP_PARTIAL_OR_TIMED_OUT = "STOP_PARTIAL_OR_TIMED_OUT"
CLASS_START_IN_PROGRESS = "START_IN_PROGRESS"
CLASS_START_PARTIAL_OR_TIMED_OUT = "START_PARTIAL_OR_TIMED_OUT"
CLASS_RESTART_IN_PROGRESS = "RESTART_IN_PROGRESS"
CLASS_RESTART_PARTIAL_OR_TIMED_OUT = "RESTART_PARTIAL_OR_TIMED_OUT"
CLASS_ACTIVE_CONTROL_CONFLICT = "ACTIVE_CONTROL_TRANSACTION_CONFLICT"
CLASS_PAPER_AUTHORIZATION_MISSING = "PAPER_AUTHORIZATION_MISSING"
CLASS_LIVE_EXECUTION_BLOCKED = "LIVE_EXECUTION_BLOCKED"
CLASS_LIVE_EXECUTION_ENABLED_INCIDENT = "LIVE_EXECUTION_ENABLED_INCIDENT"
CLASS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# Conservative precedence (lower index = higher priority). Paper/live are separate facts.
CLASSIFICATION_PRECEDENCE: Sequence[str] = (
    CLASS_ACTIVE_CONTROL_CONFLICT,
    CLASS_LIVE_EXECUTION_ENABLED_INCIDENT,
    CLASS_UNVERIFIED_IDENTITY,
    CLASS_STALE_OR_INCONSISTENT_LOCK,
    CLASS_INSUFFICIENT_EVIDENCE,
    CLASS_STOP_IN_PROGRESS,
    CLASS_RESTART_IN_PROGRESS,
    CLASS_STOP_PARTIAL_OR_TIMED_OUT,
    CLASS_START_PARTIAL_OR_TIMED_OUT,
    CLASS_RESTART_PARTIAL_OR_TIMED_OUT,
    CLASS_START_IN_PROGRESS,
    CLASS_PARTIALLY_RUNNING,
    CLASS_FULLY_STOPPED,
    CLASS_HEALTHY,
)

# Advisory safe-action labels only (never execute)
ACTION_NO_ACTION = "NO_ACTION_CONTINUE_MONITORING"
ACTION_CONSIDER_START = "CONSIDER_PROTECTED_START"
ACTION_CONSIDER_STOP = "CONSIDER_PROTECTED_STOP"
ACTION_CONSIDER_RESTART = "CONSIDER_PROTECTED_RESTART"
ACTION_WAIT = "WAIT_FOR_ACTIVE_TRANSACTION"
ACTION_REVIEW_DIAGNOSTICS = "REVIEW_RUNTIME_DIAGNOSTICS"
ACTION_MANUAL = "MANUAL_INVESTIGATION_REQUIRED"
ACTION_ESCALATE = "ESCALATE_TO_OPERATOR"
ACTION_BLOCKED = "RECOVERY_ACTION_BLOCKED"

PROHIBITED_ALWAYS = (
    "start_process",
    "stop_process",
    "restart_process",
    "force_terminate",
    "delete_runtime_lock",
    "repair_runtime_lock",
    "create_preflight",
    "create_control_token",
    "create_start_transaction",
    "create_stop_request",
    "create_restart_transaction",
    "invoke_scheduled_task",
    "arm_paper",
    "extend_paper_authorization",
    "enable_live_execution",
    "submit_order",
    "cancel_order",
    "create_approval",
    "create_authorization",
    "modify_governance",
    "modify_lifecycle",
    "modify_allocation",
    "modify_risk",
)

# Rejected frontend injection fields
REJECTED_FRONTEND_FIELDS = frozenset(
    {
        "pid",
        "target",
        "targets",
        "target_name",
        "path",
        "command",
        "arguments",
        "env",
        "environment",
        "timeout",
        "lock_path",
        "controller_function",
        "controller_operation",
        "process_control_operation",
        "operation",
    }
)

UNSAFE_LOOP_STATES = frozenset(
    {
        "PID_MISMATCH",
        "COMMAND_MISMATCH",
        "DUPLICATE_PROCESSES",
        "AMBIGUOUS",
        "UNKNOWN",
        "PROCESS_WITHOUT_VALID_LOCK",
    }
)
STALE_LOCK_STATES = frozenset({"STALE_LOCK_NO_PROCESS"})
HEALTHY_STATES = frozenset({"HEALTHY_RUNNING"})
ABSENT_STATES = frozenset({"NOT_RUNNING"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).isoformat().replace("+00:00", "Z")


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            lk = str(k).lower()
            if any(
                s in lk
                for s in (
                    "token",
                    "secret",
                    "password",
                    "credential",
                    "cookie",
                    "api_key",
                    "authorization",
                    "confirmation_phrase",
                    "preflight",
                )
            ):
                out[k] = "<redacted>"
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, str) and len(value) > 240:
        return value[:240] + "…"
    return value


def append_guidance_audit(record: Dict[str, Any], *, path: Optional[Path] = None) -> Dict[str, Any]:
    """Append-only sanitized advisory audit. Never execution authority."""
    path = path or AUDIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _sanitize(
        {
            **record,
            "schema_version": SCHEMA_VERSION,
            "controller_version": CONTROLLER_VERSION,
            "advisory_only": True,
            "execution_attempted": False,
            "recorded_at": _iso(),
        }
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {"ok": True, "path": str(path)}


def _loop_state(classifications: Dict[str, Any], name: str) -> str:
    loop = classifications.get(name) or {}
    return str(loop.get("state") or "UNKNOWN")


def _affected_from_states(classifications: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for name in ("continuous", "watchdog"):
        st = _loop_state(classifications, name)
        if st and st != "UNKNOWN":
            out.append(name)
    return out or ["continuous", "watchdog"]


def _collect_conditions(observation: Dict[str, Any]) -> List[str]:
    """Detect all applicable classification conditions (unordered)."""
    conditions: List[str] = []
    classifications = observation.get("classifications") or {}
    start_el = observation.get("start_eligibility") or {}
    stop_el = observation.get("stop_eligibility") or {}
    restart_el = observation.get("restart_eligibility") or {}

    if observation.get("insufficient_evidence") or observation.get("malformed"):
        conditions.append(CLASS_INSUFFICIENT_EVIDENCE)

    if observation.get("live_execution_appears_enabled") is True or (
        observation.get("live_execution_blocked") is False
        and observation.get("live_execution_appears_enabled") is not False
        and observation.get("force_live_blocked_only") is not True
    ):
        # Explicit enablement incident when evidence says live may be enabled.
        if (
            observation.get("live_execution_appears_enabled") is True
            or observation.get("live_execution_blocked") is False
        ):
            if observation.get("live_execution_appears_enabled") is True or (
                observation.get("live_execution_blocked") is False
                and observation.get("treat_unblocked_as_incident", True)
            ):
                conditions.append(CLASS_LIVE_EXECUTION_ENABLED_INCIDENT)

    # Control transaction conflict: multiple active control surfaces
    active_flags = [
        bool(observation.get("start_guard_active") or start_el.get("guard_active")),
        bool(
            observation.get("stop_in_progress")
            or stop_el.get("decision") == "STOP_IN_PROGRESS"
            or stop_el.get("ui_state") == "STOP_IN_PROGRESS"
        ),
        bool(
            observation.get("restart_guard_active")
            or restart_el.get("restart_guard_active")
            or str(restart_el.get("decision") or "").startswith("RESTART_IN_PROGRESS")
            or str(restart_el.get("ui_state") or "")
            in ("RESTART_IN_PROGRESS", "RESTART_STOP_IN_PROGRESS")
        ),
    ]
    if observation.get("control_conflict") or sum(1 for x in active_flags if x) >= 2:
        conditions.append(CLASS_ACTIVE_CONTROL_CONFLICT)

    # Restart / Stop / Start progress from observation or eligibility
    rtx = observation.get("restart_transaction") or {}
    r_stage = str(rtx.get("stage") or rtx.get("result") or "")
    r_result = str(rtx.get("result") or "")
    if observation.get("restart_in_progress") or r_stage in (
        "STOP_REQUESTED",
        "STOP_IN_PROGRESS",
        "START_FRESH_CLASSIFICATION",
        "START_ELIGIBILITY_CONFIRMED",
        "START_TRANSACTION_CREATED",
        "RESTART_IN_PROGRESS",
        "RESTART_STOP_IN_PROGRESS",
    ):
        conditions.append(CLASS_RESTART_IN_PROGRESS)
    if observation.get("restart_partial_or_timed_out") or r_result in (
        "RESTART_PARTIAL_COMPLETION",
        "RESTART_TIMED_OUT",
        "RESTART_REFUSED_STOP_INCOMPLETE",
        "RESTART_REFUSED_START_INELIGIBLE",
        "RESTART_AUDIT_FAILED",
    ):
        conditions.append(CLASS_RESTART_PARTIAL_OR_TIMED_OUT)

    if (
        observation.get("stop_in_progress")
        or stop_el.get("decision") == "STOP_IN_PROGRESS"
        or stop_el.get("ui_state") == "STOP_IN_PROGRESS"
    ):
        conditions.append(CLASS_STOP_IN_PROGRESS)
    if observation.get("stop_partial_or_timed_out") or str(stop_el.get("decision") or "") in (
        "STOP_PARTIAL_COMPLETION",
        "STOP_TIMED_OUT",
    ):
        conditions.append(CLASS_STOP_PARTIAL_OR_TIMED_OUT)
    # Ack alone never means completed Stop
    if observation.get("stop_ack_only"):
        conditions.append(CLASS_STOP_PARTIAL_OR_TIMED_OUT)

    if observation.get("start_in_progress") or start_el.get("guard_active"):
        # Distinguish ordinary in-progress from explicit timeout
        if observation.get("start_timed_out") or observation.get("start_partial_or_timed_out"):
            conditions.append(CLASS_START_PARTIAL_OR_TIMED_OUT)
        else:
            conditions.append(CLASS_START_IN_PROGRESS)
    elif observation.get("start_partial_or_timed_out") or observation.get("start_timed_out"):
        conditions.append(CLASS_START_PARTIAL_OR_TIMED_OUT)

    cont = _loop_state(classifications, "continuous")
    wd = _loop_state(classifications, "watchdog")
    states = {cont, wd}

    # Observation flags and unsafe loop states both contribute (flags alone are sufficient).
    if (
        observation.get("unverified_identity")
        or observation.get("creation_time_mismatch")
        or observation.get("command_fingerprint_mismatch")
        or states
        & {
            "PID_MISMATCH",
            "COMMAND_MISMATCH",
            "DUPLICATE_PROCESSES",
            "AMBIGUOUS",
            "UNKNOWN",
        }
    ):
        conditions.append(CLASS_UNVERIFIED_IDENTITY)
    if (
        observation.get("conflicting_lock_owner")
        or observation.get("stale_lock")
        or states & STALE_LOCK_STATES
        or states & {"PROCESS_WITHOUT_VALID_LOCK"}
    ):
        conditions.append(CLASS_STALE_OR_INCONSISTENT_LOCK)

    if observation.get("missing_diagnostics"):
        conditions.append(CLASS_INSUFFICIENT_EVIDENCE)

    healthy = cont in HEALTHY_STATES and wd in HEALTHY_STATES
    stopped = cont in ABSENT_STATES and wd in ABSENT_STATES
    partial = (cont in HEALTHY_STATES) ^ (wd in HEALTHY_STATES) or (
        (cont in HEALTHY_STATES or wd in HEALTHY_STATES)
        and (cont in ABSENT_STATES or wd in ABSENT_STATES)
    )

    if healthy and not (
        set(conditions)
        & {
            CLASS_UNVERIFIED_IDENTITY,
            CLASS_STALE_OR_INCONSISTENT_LOCK,
            CLASS_INSUFFICIENT_EVIDENCE,
            CLASS_ACTIVE_CONTROL_CONFLICT,
            CLASS_LIVE_EXECUTION_ENABLED_INCIDENT,
        }
    ):
        conditions.append(CLASS_HEALTHY)
    elif stopped and not (
        set(conditions)
        & {
            CLASS_UNVERIFIED_IDENTITY,
            CLASS_STALE_OR_INCONSISTENT_LOCK,
            CLASS_INSUFFICIENT_EVIDENCE,
            CLASS_STOP_IN_PROGRESS,
            CLASS_RESTART_IN_PROGRESS,
            CLASS_START_IN_PROGRESS,
        }
    ):
        conditions.append(CLASS_FULLY_STOPPED)
    elif partial:
        conditions.append(CLASS_PARTIALLY_RUNNING)

    # Deduplicate while preserving order of first occurrence
    seen = set()
    ordered: List[str] = []
    for c in conditions:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _select_primary(conditions: Sequence[str]) -> str:
    if not conditions:
        return CLASS_INSUFFICIENT_EVIDENCE
    rank = {name: i for i, name in enumerate(CLASSIFICATION_PRECEDENCE)}
    return sorted(conditions, key=lambda c: rank.get(c, 999))[0]


def _severity(primary: str) -> str:
    if primary in (
        CLASS_LIVE_EXECUTION_ENABLED_INCIDENT,
        CLASS_ACTIVE_CONTROL_CONFLICT,
        CLASS_UNVERIFIED_IDENTITY,
    ):
        return "CRITICAL"
    if primary in (
        CLASS_STALE_OR_INCONSISTENT_LOCK,
        CLASS_STOP_PARTIAL_OR_TIMED_OUT,
        CLASS_START_PARTIAL_OR_TIMED_OUT,
        CLASS_RESTART_PARTIAL_OR_TIMED_OUT,
        CLASS_INSUFFICIENT_EVIDENCE,
        CLASS_PARTIALLY_RUNNING,
    ):
        return "HIGH"
    if primary in (
        CLASS_STOP_IN_PROGRESS,
        CLASS_START_IN_PROGRESS,
        CLASS_RESTART_IN_PROGRESS,
        CLASS_PAPER_AUTHORIZATION_MISSING,
    ):
        return "MEDIUM"
    if primary == CLASS_FULLY_STOPPED:
        return "INFO"
    return "INFO"


def _incident_notes(observation: Dict[str, Any], classifications: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    for name in ("continuous", "watchdog"):
        loop = classifications.get(name) or {}
        st = str(loop.get("state") or "")
        if st == "NOT_RUNNING" and observation.get("report_missing_process"):
            notes.append(f"{name}: missing process")
        if st == "DUPLICATE_PROCESSES":
            notes.append(f"{name}: unexpected extra process evidence")
        if st == "PID_MISMATCH":
            notes.append(f"{name}: mismatched PID / lock ownership")
        if st == "COMMAND_MISMATCH":
            notes.append(f"{name}: command fingerprint mismatch")
        if st == "PROCESS_WITHOUT_VALID_LOCK":
            notes.append(f"{name}: missing owner lock")
        if st == "STALE_LOCK_NO_PROCESS":
            notes.append(f"{name}: stale lock without process")
        if observation.get("creation_time_mismatch"):
            notes.append(f"{name}: changed creation time (investigate)")
        if observation.get("conflicting_lock_owner"):
            notes.append(f"{name}: conflicting lock owner")
    if observation.get("malformed"):
        notes.append("malformed runtime evidence")
    if observation.get("missing_diagnostics"):
        notes.append("unavailable diagnostics")
    if observation.get("stop_in_progress"):
        notes.append("active cooperative Stop")
    if observation.get("start_in_progress"):
        notes.append("active protected Start")
    if observation.get("restart_in_progress"):
        notes.append("active protected Restart")
    if observation.get("stop_timed_out"):
        notes.append("Stop timeout — no force termination")
    if observation.get("start_timed_out"):
        notes.append("Start timeout — no automatic retry")
    if observation.get("restart_timed_out"):
        notes.append("Restart timeout — no force termination or lock repair")
    if observation.get("partial_replacement"):
        notes.append("partial replacement verification")
    paper = str(observation.get("paper_arm_status") or "")
    if paper in ("MISSING", "EXPIRED", "INVALID"):
        notes.append("missing or invalid paper authorization (separate prerequisite)")
    if observation.get("live_execution_appears_enabled"):
        notes.append("CRITICAL: live execution unexpectedly reported as enabled")
    return notes


def _recommend(
    primary: str,
    *,
    start_el: Dict[str, Any],
    stop_el: Dict[str, Any],
    restart_el: Dict[str, Any],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Map primary classification to advisory labels + optional eligible control name."""
    eligible: Optional[str] = None
    action = ACTION_MANUAL
    manual = True
    escalate = False
    blockers: List[str] = []
    warnings: List[str] = []
    message = ""

    if primary == CLASS_HEALTHY:
        action = ACTION_NO_ACTION
        manual = False
        message = (
            "Runtime appears healthy with verified process and lock ownership. "
            "Continue monitoring. Unnecessary Start, Stop, or Restart is not recommended."
        )
    elif primary == CLASS_FULLY_STOPPED:
        if start_el.get("start_button_enabled"):
            action = ACTION_CONSIDER_START
            eligible = "START_TRITON"
            manual = False
            message = (
                "Runtime appears fully stopped. Protected Start (DO-003) may be considered. "
                "Guidance does not invoke Start and does not guarantee Start will succeed."
            )
        else:
            action = ACTION_REVIEW_DIAGNOSTICS
            blockers.append(str(start_el.get("decision") or "START_NOT_ELIGIBLE"))
            message = (
                "Runtime appears fully stopped, but fresh DO-003 Start eligibility is not enabled. "
                "Review Diagnostics before acting."
            )
            manual = True
    elif primary == CLASS_PARTIALLY_RUNNING:
        if restart_el.get("restart_button_enabled"):
            action = ACTION_CONSIDER_RESTART
            eligible = "RESTART_TRITON"
            manual = True
            warnings.append(
                "Partial runtime: investigate before acting. Restart eligibility is advisory only."
            )
            message = (
                "Some expected loops are verified while others are absent. "
                "Protected Restart may be considered only if DO-006 eligibility permits; "
                "guidance does not create a Restart preflight or transaction."
            )
        else:
            action = ACTION_MANUAL
            message = "Partial runtime detected. Manual investigation is required before any protected action."
    elif primary == CLASS_STALE_OR_INCONSISTENT_LOCK:
        action = ACTION_MANUAL
        escalate = True
        blockers.append("lock_evidence_inconsistent")
        message = (
            "Stale or inconsistent lock evidence. Manual investigation required. "
            "Do not delete or repair locks. Do not bypass ownership safeguards."
        )
    elif primary == CLASS_UNVERIFIED_IDENTITY:
        action = ACTION_BLOCKED
        escalate = True
        blockers.append("unverified_process_identity")
        message = (
            "Process identity cannot be safely verified (PID, creation time, command fingerprint, "
            "or lock ownership). Automated recovery advice is refused. Never force-terminate."
        )
    elif primary == CLASS_STOP_IN_PROGRESS:
        action = ACTION_WAIT
        manual = False
        blockers.extend(["start_blocked", "stop_duplicate_blocked", "restart_blocked"])
        message = (
            "A valid cooperative Stop remains in progress. Continue observing. "
            "Do not Start, do not issue another Stop, and do not Restart. "
            "Bounded work may need to reach its documented safe boundary."
        )
    elif primary == CLASS_STOP_PARTIAL_OR_TIMED_OUT:
        action = ACTION_MANUAL
        escalate = True
        blockers.append("start_prohibited_until_stop_complete")
        message = (
            "Stop was acknowledged or initiated, but verified completion was not satisfied. "
            "Start remains prohibited. No forced termination. No lock deletion or repair. "
            "Investigate remaining processes and owner-held locks."
        )
    elif primary == CLASS_START_IN_PROGRESS:
        action = ACTION_WAIT
        manual = False
        blockers.extend(["start_duplicate_blocked", "restart_blocked"])
        message = (
            "A protected Start transaction remains unresolved. "
            "Do not recommend another Start or Restart. "
            "Ordinary in-progress is not a timeout unless explicit timeout evidence exists."
        )
    elif primary == CLASS_START_PARTIAL_OR_TIMED_OUT:
        action = ACTION_MANUAL
        blockers.append("no_automatic_start_retry")
        message = (
            "Start did not achieve complete replacement-process and lock-ownership verification. "
            "Do not automatically retry. Investigate before another protected action."
        )
    elif primary == CLASS_RESTART_IN_PROGRESS:
        action = ACTION_WAIT
        manual = False
        rtx = observation.get("restart_transaction") or {}
        stage = str(rtx.get("stage") or "unknown")
        blockers.extend(["start_blocked", "stop_duplicate_blocked", "restart_duplicate_blocked"])
        message = (
            f"Protected Restart is in progress (current stage: {stage}). "
            "Stop and Start stages are separate verified transactions. "
            "Do not recommend duplicate actions. Completion is not immediate."
        )
    elif primary == CLASS_RESTART_PARTIAL_OR_TIMED_OUT:
        action = ACTION_MANUAL
        escalate = True
        rtx = observation.get("restart_transaction") or {}
        message = (
            "Protected Restart did not reach fully verified completion "
            f"(result={rtx.get('result') or 'partial/timeout'}, "
            f"stop_result={rtx.get('stop_result')}, start_result={rtx.get('start_result')}). "
            "Do not retry automatically. Do not delete locks or terminate processes."
        )
    elif primary == CLASS_ACTIVE_CONTROL_CONFLICT:
        action = ACTION_BLOCKED
        escalate = True
        blockers.append("control_transaction_conflict")
        message = (
            "Start, Stop, or Restart transaction evidence conflicts or indicates concurrent "
            "control activity. Block further operational recommendations. Investigate. "
            "Preserve mutual exclusion."
        )
    elif primary == CLASS_LIVE_EXECUTION_ENABLED_INCIDENT:
        action = ACTION_ESCALATE
        escalate = True
        manual = True
        blockers.append("live_execution_not_blocked")
        message = (
            "CRITICAL SAFETY INCIDENT: live execution may be enabled. "
            "Immediate manual investigation is required. "
            "DO-007 does not modify execution flags and does not attempt remediation."
        )
    elif primary == CLASS_INSUFFICIENT_EVIDENCE:
        action = ACTION_MANUAL
        blockers.append("insufficient_evidence")
        message = (
            "Required evidence is missing, unreadable, malformed, stale, or contradictory. "
            "Do not guess. No protected action is presented as eligible."
        )
    else:
        action = ACTION_MANUAL
        message = "Conservative default: manual investigation required."

    # Paper authorization is a separate safety fact — never overrides identity concerns
    paper = str(observation.get("paper_arm_status") or "")
    paper_missing = paper in ("MISSING", "EXPIRED", "INVALID", "")
    if paper_missing and primary in (
        CLASS_HEALTHY,
        CLASS_FULLY_STOPPED,
        CLASS_PARTIALLY_RUNNING,
    ):
        warnings.append(
            "Paper authorization is missing or invalid — separate operational prerequisite; "
            "runtime recovery guidance does not create or extend paper authorization."
        )

    return {
        "safe_next_action": action,
        "eligible_protected_control": eligible,
        "manual_investigation_required": manual,
        "escalation_required": escalate,
        "operator_message": message,
        "blockers": blockers,
        "warnings": warnings,
        "paper_authorization_missing": paper_missing,
    }


def build_runtime_recovery_guidance(
    *,
    root: Optional[Path] = None,
    observation: Optional[Dict[str, Any]] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    start_eligibility_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    stop_eligibility_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    restart_eligibility_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    classify_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    write_audit: bool = False,
    audit_path: Optional[Path] = None,
    # Explicitly rejected frontend fields
    pid: Any = None,
    target: Any = None,
    targets: Any = None,
    target_name: Any = None,
    path: Any = None,
    command: Any = None,
    arguments: Any = None,
    env: Any = None,
    environment: Any = None,
    timeout: Any = None,
    lock_path: Any = None,
    controller_function: Any = None,
    controller_operation: Any = None,
    process_control_operation: Any = None,
    operation: Any = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build read-only recovery guidance. Never mutates runtime or invokes controllers."""
    guidance_id = str(uuid.uuid4())
    injected = any(
        v is not None
        for v in (
            pid,
            target,
            targets,
            target_name,
            path,
            command,
            arguments,
            env,
            environment,
            timeout,
            lock_path,
            controller_function,
            controller_operation,
            process_control_operation,
            operation,
        )
    ) or bool(extra)
    if injected:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "guidance_id": guidance_id,
            "generated_at": _iso(),
            "runtime_overall_status": CLASS_INSUFFICIENT_EVIDENCE,
            "affected_targets": [],
            "observed_conditions": [CLASS_INSUFFICIENT_EVIDENCE],
            "evidence_summary": ["Frontend operational inputs are refused."],
            "severity": "HIGH",
            "operator_message": (
                "Recovery guidance refused: frontend must not supply PID, target, path, "
                "command, arguments, environment, timeout, lock path, or controller selection."
            ),
            "safe_next_action": ACTION_BLOCKED,
            "eligible_protected_control": None,
            "prohibited_actions": list(PROHIBITED_ALWAYS),
            "manual_investigation_required": True,
            "escalation_required": False,
            "confidence": "none",
            "warnings": ["FRONTEND_INJECTION_REFUSED"],
            "blockers": ["FRONTEND_INJECTION_REFUSED"],
            "source_evidence": {},
            "advisory_only": True,
            "execution_attempted": False,
            "runtime_artifact_modified": False,
            "guidance_id_is_authority": False,
            "error_code": "FRONTEND_INJECTION_REFUSED",
            "paper_arm_status": "UNKNOWN",
            "live_execution_blocked": True,
            "restart_stage": None,
            "incident_notes": ["frontend_injection_refused"],
        }

    root = Path(root) if root is not None else ROOT
    obs = dict(observation or {})

    # Fresh read-only eligibility from existing controllers unless fully injected
    if observation is None or not observation.get("classifications"):
        from services.operator_runtime_classifier import classify_runtime_loops

        classify = classify_fn or classify_runtime_loops
        classifications = classify(root=root, process_snapshot=process_snapshot)
    else:
        classifications = observation.get("classifications") or {}

    if observation is None or "start_eligibility" not in observation:
        from services.operator_runtime_controller import start_eligibility

        start_fn = start_eligibility_fn or start_eligibility
        start_el = start_fn(root=root, process_snapshot=process_snapshot)
    else:
        start_el = observation.get("start_eligibility") or {}

    if observation is None or "stop_eligibility" not in observation:
        from services.operator_stop_action import stop_eligibility

        stop_fn = stop_eligibility_fn or stop_eligibility
        stop_el = stop_fn(root=root, process_snapshot=process_snapshot)
    else:
        stop_el = observation.get("stop_eligibility") or {}

    if observation is None or "restart_eligibility" not in observation:
        from services.operator_restart_action import restart_eligibility

        restart_fn = restart_eligibility_fn or restart_eligibility
        restart_el = restart_fn(root=root, process_snapshot=process_snapshot)
    else:
        restart_el = observation.get("restart_eligibility") or {}

    # Fill observation defaults for classification from live eligibility when not injected
    if "start_guard_active" not in obs:
        obs["start_guard_active"] = bool(start_el.get("guard_active"))
    if "stop_in_progress" not in obs:
        obs["stop_in_progress"] = (
            str(stop_el.get("decision") or "") == "STOP_IN_PROGRESS"
            or str(stop_el.get("ui_state") or "") == "STOP_IN_PROGRESS"
        )
    if "restart_in_progress" not in obs:
        obs["restart_in_progress"] = str(restart_el.get("decision") or "") in (
            "RESTART_IN_PROGRESS",
            "RESTART_STOP_IN_PROGRESS",
        ) or str(restart_el.get("ui_state") or "") in (
            "RESTART_IN_PROGRESS",
            "RESTART_STOP_IN_PROGRESS",
        )
    if "live_execution_blocked" not in obs:
        obs["live_execution_blocked"] = bool(
            start_el.get("live_execution_blocked", True)
            and stop_el.get("live_execution_blocked", True)
        )
    if "live_execution_appears_enabled" not in obs:
        obs["live_execution_appears_enabled"] = not bool(obs.get("live_execution_blocked", True))
    if "paper_arm_status" not in obs:
        obs["paper_arm_status"] = str(start_el.get("paper_arm_status") or "UNKNOWN")

    obs["classifications"] = classifications
    obs["start_eligibility"] = start_el
    obs["stop_eligibility"] = stop_el
    obs["restart_eligibility"] = restart_el

    # Reject unknown / non-allowlisted target names in injected classifications
    for name in list(classifications.keys()):
        if name in ("generated_at", "detection_confidence"):
            continue
        if name not in TYPED_TARGETS:
            obs["insufficient_evidence"] = True
            obs["unknown_target"] = name

    conditions = _collect_conditions(obs)
    primary = _select_primary(conditions)
    # Contradictory evidence cannot produce HEALTHY
    if primary == CLASS_HEALTHY and (
        CLASS_UNVERIFIED_IDENTITY in conditions
        or CLASS_STALE_OR_INCONSISTENT_LOCK in conditions
        or CLASS_INSUFFICIENT_EVIDENCE in conditions
        or CLASS_LIVE_EXECUTION_ENABLED_INCIDENT in conditions
    ):
        primary = _select_primary(
            [c for c in conditions if c != CLASS_HEALTHY] or [CLASS_INSUFFICIENT_EVIDENCE]
        )

    rec = _recommend(
        primary,
        start_el=start_el,
        stop_el=stop_el,
        restart_el=restart_el,
        observation=obs,
    )
    severity = _severity(primary)
    incidents = _incident_notes(obs, classifications)
    rtx = obs.get("restart_transaction") or {}
    restart_stage = rtx.get("stage")

    evidence_summary = [
        f"continuous={_loop_state(classifications, 'continuous')}",
        f"watchdog={_loop_state(classifications, 'watchdog')}",
        f"start_decision={start_el.get('decision')}",
        f"stop_decision={stop_el.get('decision')}",
        f"restart_decision={restart_el.get('decision')}",
    ]
    if incidents:
        evidence_summary.extend(incidents[:8])

    confidence = "high"
    if primary in (CLASS_INSUFFICIENT_EVIDENCE, CLASS_UNVERIFIED_IDENTITY):
        confidence = "low"
    elif primary in (CLASS_PARTIALLY_RUNNING, CLASS_STOP_PARTIAL_OR_TIMED_OUT):
        confidence = "medium"

    warnings = list(rec.get("warnings") or [])
    if bool(obs.get("live_execution_blocked", True)):
        warnings.append("LIVE_EXECUTION_BLOCKED (required safe state)")
    else:
        warnings.append("LIVE_EXECUTION_NOT_BLOCKED")

    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "controller_version": CONTROLLER_VERSION,
        "guidance_id": guidance_id,
        "generated_at": _iso(),
        "runtime_overall_status": primary,
        "affected_targets": [
            t for t in _affected_from_states(classifications) if t in TYPED_TARGETS
        ],
        "observed_conditions": conditions,
        "evidence_summary": evidence_summary,
        "severity": severity,
        "operator_message": rec["operator_message"],
        "safe_next_action": rec["safe_next_action"],
        "eligible_protected_control": rec["eligible_protected_control"],
        "prohibited_actions": list(PROHIBITED_ALWAYS),
        "manual_investigation_required": bool(rec["manual_investigation_required"]),
        "escalation_required": bool(rec["escalation_required"]),
        "confidence": confidence,
        "warnings": warnings,
        "blockers": list(rec.get("blockers") or []),
        "source_evidence": _sanitize(
            {
                "classifications": {
                    k: classifications.get(k)
                    for k in (
                        "continuous",
                        "watchdog",
                        "generated_at",
                        "detection_confidence",
                    )
                    if k in classifications
                },
                "start_decision": start_el.get("decision"),
                "stop_decision": stop_el.get("decision"),
                "restart_decision": restart_el.get("decision"),
                "restart_stage": restart_stage,
                "start_button_enabled": start_el.get("start_button_enabled"),
                "stop_button_enabled": stop_el.get("stop_button_enabled"),
                "restart_button_enabled": restart_el.get("restart_button_enabled"),
            }
        ),
        "advisory_only": True,
        "execution_attempted": False,
        "runtime_artifact_modified": False,
        "guidance_id_is_authority": False,
        "paper_arm_status": obs.get("paper_arm_status") or "UNKNOWN",
        "paper_authorization_missing": bool(rec.get("paper_authorization_missing")),
        "live_execution_blocked": bool(obs.get("live_execution_blocked", True)),
        "live_execution_appears_enabled": bool(obs.get("live_execution_appears_enabled")),
        "restart_stage": restart_stage,
        "restart_stop_stage": (
            restart_stage
            if restart_stage and "STOP" in str(restart_stage).upper()
            else (rtx.get("stop_result") if rtx else None)
        ),
        "restart_start_stage": (
            restart_stage
            if restart_stage and "START" in str(restart_stage).upper()
            else (rtx.get("start_result") if rtx else None)
        ),
        "incident_notes": incidents,
        "classification_precedence": list(CLASSIFICATION_PRECEDENCE),
        "fresh_start_eligibility_decision": start_el.get("decision"),
        "fresh_stop_eligibility_decision": stop_el.get("decision"),
        "fresh_restart_eligibility_decision": restart_el.get("decision"),
        "orders_submitted": False,
        "orders_cancelled": False,
        "approvals_created": False,
        "authorizations_created": False,
        "paper_arm_mutated": False,
        "governance_modified": False,
        "lifecycle_modified": False,
        "allocation_modified": False,
        "risk_modified": False,
        "scheduled_task_modified": False,
        "forced_termination_used": False,
        "runtime_locks_deleted": False,
        "runtime_locks_repaired": False,
        "broker_called": False,
        "control_transaction_created": False,
        "preflight_created": False,
    }

    if write_audit:
        append_guidance_audit(
            {
                "guidance_id": guidance_id,
                "runtime_overall_status": primary,
                "safe_next_action": result["safe_next_action"],
                "severity": severity,
                "eligible_protected_control": result["eligible_protected_control"],
            },
            path=audit_path,
        )

    return result
