"""
phase_196b_auto_paper_approval_runtime_alignment_validation.py
Phase 196B — Validate automatic paper approval and runtime lock alignment.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "phase_196b_auto_paper_approval_runtime_alignment_validation.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _sample_pending_request() -> Dict[str, Any]:
    exp = (datetime.now(timezone.utc) + timedelta(hours=24)).replace(microsecond=0).isoformat()
    return {
        "approval_id": "eg-apr-test196b0001",
        "approval_status": "PENDING",
        "authority_level_name": "GOVERNANCE_COUNCIL_APPROVAL",
        "execution_path": "capital_deployment",
        "requested_action": "capital_deployment",
        "execution_gate_decision": "ALLOW",
        "symbol(s)": "AAPL,MSFT",
        "governance_context": {
            "context_summary": {"paperMode": True, "liveMode": False},
        },
        "expires_at": exp,
    }


def _sample_live_request() -> Dict[str, Any]:
    req = _sample_pending_request()
    req["approval_id"] = "eg-apr-test196blive01"
    req["execution_path"] = "live_execution"
    req["requested_action"] = "live_execution"
    req["governance_context"] = {"context_summary": {"paperMode": False, "liveMode": True}}
    return req


def _ready_readiness() -> Dict[str, Any]:
    return {
        "readiness_status": "READY",
        "live_execution_permitted": False,
        "paper_execution_permitted": True,
    }


def run_phase_196b_validation(*, verbose: bool = True) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    from services.auto_paper_approval import attempt_auto_paper_approval
    from services.execution_approval_workflow import validate_paper_capital_deployment_approvable
    from services.runtime_lock_classifier import (
        classify_runtime_lock,
        is_lock_healthy,
    )

    # 1. Live auto approval impossible
    live_ok, live_reason = validate_paper_capital_deployment_approvable(_sample_live_request())
    checks["live_auto_approval_impossible"] = live_ok is False
    details["live_refusal_reason"] = live_reason

    symbols = ["AAPL", "MSFT"]
    base_env = {
        "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
        "TRITON_ENABLE_PAPER_EXECUTION": "1",
        "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
    }

    # 2. Refuses when env flag missing
    with patch.dict(
        os.environ,
        {k: "1" for k in base_env if k != "TRITON_ENABLE_AUTO_PAPER_APPROVAL"},
        clear=False,
    ):
        os.environ.pop("TRITON_ENABLE_AUTO_PAPER_APPROVAL", None)
        ok, reason, _ = attempt_auto_paper_approval(symbols, verbose=False)
    checks["refuses_missing_auto_approval_flag"] = not ok and "AUTO_PAPER_APPROVAL" in reason

    # 3. Refuses when market closed
    with patch.dict(os.environ, base_env, clear=False):
        with patch("services.auto_paper_approval._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=False):
                ok, reason, _ = attempt_auto_paper_approval(symbols, verbose=False)
    checks["refuses_market_closed"] = not ok and reason == "MARKET_CLOSED"

    # 4. Refuses when ARM expired
    with patch.dict(os.environ, base_env, clear=False):
        with patch("services.auto_paper_approval._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.auto_paper_approval._file_age_minutes", return_value=5.0):
                    with patch(
                        "services.arm_mode.is_paper_arm_confirm_valid",
                        return_value=False,
                    ):
                        with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                            with patch("services.arm_mode.load_arm_config", return_value={}):
                                ok, reason, _ = attempt_auto_paper_approval(symbols, verbose=False)
    checks["refuses_arm_expired"] = not ok and "ARM" in reason

    # 5. Refuses when readiness NOT_READY
    with patch.dict(os.environ, base_env, clear=False):
        with patch(
            "services.auto_paper_approval._load_json",
            return_value={"readiness_status": "NOT_READY", "live_execution_permitted": False},
        ):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                ok, reason, _ = attempt_auto_paper_approval(symbols, verbose=False)
    checks["refuses_readiness_not_ready"] = not ok and "READINESS" in reason

    # 6. Refuses stale broker snapshot
    with patch.dict(os.environ, base_env, clear=False):
        with patch("services.auto_paper_approval._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.auto_paper_approval._file_age_minutes", return_value=200.0):
                    with patch("services.arm_mode.is_paper_arm_confirm_valid", return_value=True):
                        with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                            with patch("services.arm_mode.load_arm_config", return_value={}):
                                ok, reason, _ = attempt_auto_paper_approval(symbols, verbose=False)
    checks["refuses_stale_broker_snapshot"] = not ok and "BROKER_SNAPSHOT" in reason

    # 7. Refuses duplicate open orders
    with patch.dict(os.environ, base_env, clear=False):
        with patch("services.auto_paper_approval._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.auto_paper_approval._file_age_minutes", return_value=5.0):
                    with patch("services.arm_mode.is_paper_arm_confirm_valid", return_value=True):
                        with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                            with patch("services.arm_mode.load_arm_config", return_value={}):
                                with patch(
                                    "services.paper_entry_loop_stage._load_open_buy_keys",
                                    return_value={("AAPL", "BUY")},
                                ):
                                    with patch(
                                        "services.paper_entry_loop_stage._load_held_symbols",
                                        return_value=set(),
                                    ):
                                        ok, reason, _ = attempt_auto_paper_approval(
                                            symbols, verbose=False
                                        )
    checks["refuses_duplicate_open_orders"] = not ok and "OPEN_ORDER" in reason

    # 8. Refuses already-held positions
    with patch.dict(os.environ, base_env, clear=False):
        with patch("services.auto_paper_approval._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.auto_paper_approval._file_age_minutes", return_value=5.0):
                    with patch("services.arm_mode.is_paper_arm_confirm_valid", return_value=True):
                        with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                            with patch("services.arm_mode.load_arm_config", return_value={}):
                                with patch(
                                    "services.paper_entry_loop_stage._load_open_buy_keys",
                                    return_value=set(),
                                ):
                                    with patch(
                                        "services.paper_entry_loop_stage._load_held_symbols",
                                        return_value={"MSFT"},
                                    ):
                                        ok, reason, _ = attempt_auto_paper_approval(
                                            symbols, verbose=False
                                        )
    checks["refuses_already_held_positions"] = not ok and "ALREADY_HELD" in reason

    # 9. Approves valid paper request
    approved_record = dict(_sample_pending_request())
    approved_record["approval_status"] = "APPROVED"
    with patch.dict(os.environ, base_env, clear=False):
        with patch("services.auto_paper_approval._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.auto_paper_approval._file_age_minutes", return_value=5.0):
                    with patch("services.arm_mode.is_paper_arm_confirm_valid", return_value=True):
                        with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                            with patch("services.arm_mode.load_arm_config", return_value={}):
                                with patch(
                                    "services.paper_entry_loop_stage._load_open_buy_keys",
                                    return_value=set(),
                                ):
                                    with patch(
                                        "services.paper_entry_loop_stage._load_held_symbols",
                                        return_value=set(),
                                    ):
                                        with patch(
                                            "services.execution_approval_workflow.find_approved_request",
                                            side_effect=[None, approved_record],
                                        ):
                                            with patch(
                                                "services.auto_paper_approval._ensure_pending_request",
                                                return_value=(
                                                    _sample_pending_request(),
                                                    "EXISTING_PENDING",
                                                ),
                                            ):
                                                with patch(
                                                    "services.execution_approval_workflow.approve_request",
                                                    return_value=(approved_record, "OK"),
                                                ):
                                                    ok, reason, det = attempt_auto_paper_approval(
                                                        symbols, verbose=False
                                                    )
    checks["approves_valid_paper_request"] = ok

    # 10. Approved request consumed by paper_entry_loop_stage governance check
    with patch.dict(os.environ, {**base_env}, clear=False):
        with patch("services.paper_entry_loop_stage._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.paper_entry_loop_stage._file_age_minutes", return_value=5.0):
                    with patch(
                        "services.paper_entry_loop_stage._find_execute_trades_running",
                        return_value=(False, []),
                    ):
                        with patch(
                            "services.arm_mode.is_paper_arm_confirm_valid", return_value=True
                        ):
                            with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                                with patch("services.arm_mode.load_arm_config", return_value={}):
                                    with patch(
                                        "services.arm_mode.resolve_arm_mutation_permissions",
                                        return_value=({"execute_trades": True}, []),
                                    ):
                                        with patch(
                                            "services.paper_entry_loop_stage._load_entry_candidates",
                                            return_value=(
                                                [
                                                    {
                                                        "ticker": "AAPL",
                                                        "eligible_for_auto_entry": True,
                                                    }
                                                ],
                                                None,
                                            ),
                                        ):
                                            with patch(
                                                "services.execution_approval_workflow.find_approved_request",
                                                return_value=approved_record,
                                            ):
                                                from services.paper_entry_loop_stage import (
                                                    evaluate_paper_entry_gates,
                                                )

                                                gate_ok, gate_reason, _ = (
                                                    evaluate_paper_entry_gates(verbose=False)
                                                )
    checks["paper_entry_stage_consumes_approval"] = gate_ok and gate_reason == "OK"

    # 11. Healthy lock classification
    healthy_lock = {
        "exists": True,
        "pid": 1234,
        "stale": True,
        "lock_freshness_age_minutes": 400.0,
    }
    cls = classify_runtime_lock(
        healthy_lock,
        True,
        [1234],
        loop_kind="watchdog",
        heartbeat_age_minutes=2.0,
    )
    checks["lock_classifier_healthy_active"] = is_lock_healthy(cls)

    # 12. Stale orphan lock
    orphan_cls = classify_runtime_lock(
        {"exists": True, "pid": 9999, "stale": True},
        False,
        [],
        loop_kind="continuous",
        heartbeat_age_minutes=500.0,
    )
    checks["stale_orphan_lock_classified"] = orphan_cls == "STALE_ORPHAN_LOCK"

    # 13. Live execution remains blocked
    readiness_doc = {}
    try:
        readiness_doc = json.loads(
            (RESULTS / "execution_readiness.json").read_text(encoding="utf-8")
        )
    except Exception:
        pass
    checks["live_execution_permitted_false"] = not bool(
        readiness_doc.get("live_execution_permitted", True)
    )

    phase_pass = all(checks.values())
    payload = {
        "phase": "196B",
        "generated_at": _utc_now_iso(),
        "phase_pass": phase_pass,
        "validation_status": "PASS" if phase_pass else "FAIL",
        "checks": checks,
        "details": details,
    }
    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(f"[196B] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196B] validation_status={payload['validation_status']}")
        if not phase_pass:
            failed = [k for k, v in checks.items() if not v]
            print(f"[196B] failed checks: {failed}")

    return payload


def main() -> int:
    result = run_phase_196b_validation(verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
