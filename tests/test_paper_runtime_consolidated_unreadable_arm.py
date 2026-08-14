"""Narrow consolidated-verification case: unreadable paper BUY/ADD arm.

Existing session-gating tests cover missing, expired, malformed, unknown-classifier,
and valid-then-invalid-before-launch. They do not execute an I/O failure on
paper_arm_confirm.json. This file adds only that required Protected Start case.

Isolated temp root, fake launcher, credentials emptied, network blocked.
Does not read or mutate the real paper arm.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.operator_runtime_controller import execute_start_triton
from services.paper_arm_status import classify_paper_arm_confirm_status

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_network_and_credentials(monkeypatch):
    for key in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_BASE_URL",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")

    def _blocked(*_a, **_k):
        raise RuntimeError("network blocked in unreadable-arm verification")

    monkeypatch.setattr(socket, "create_connection", _blocked)


def _lock_text(pid: int) -> str:
    now = datetime.now(timezone.utc) - timedelta(minutes=1)
    return (
        f"started_at={now.isoformat()}\n"
        f"refreshed_at={now.isoformat()}\n"
        f"pid={pid}\n"
        f"source=test\n"
        f"cycle=1\n"
    )


def _write_runtime_fixture(root: Path) -> None:
    live = root / "data" / "live"
    results = root / "data" / "results"
    scripts = root / "scripts"
    live.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "run_watchdog_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (scripts / "run_continuous_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    (live / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "continuous_loop_running": False,
                "watchdog_loop_running": False,
                "detected_continuous_process_ids": [],
                "detected_watchdog_process_ids": [],
                "process_detection_confidence": "high",
                "command_lines": {},
            }
        ),
        encoding="utf-8",
    )


def _write_valid_arm(root: Path) -> Path:
    live = root / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    path = live / "paper_arm_confirm.json"
    clock = datetime.now(timezone.utc)
    path.write_text(
        json.dumps(
            {
                "timestamp": clock.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "expires_at": (clock + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": "ASSISTED",
                "allow_execute": True,
                "confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _start(root: Path, launched: list):
    state = {"cont": [], "wd": []}

    def snapshot():
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    def launch(component):
        launched.append(component)
        pid = 222 if component == "watchdog" else 111
        if component == "watchdog":
            state["wd"] = [pid]
            (root / "data" / "live" / "watchdog_loop.lock").write_text(
                _lock_text(pid), encoding="utf-8"
            )
        else:
            state["cont"] = [pid]
            (root / "data" / "live" / "continuous_loop.lock").write_text(
                _lock_text(pid), encoding="utf-8"
            )
        return {"ok": True, "component": component}

    return execute_start_triton(
        root=root,
        process_snapshot=snapshot,
        launch_fn=launch,
        audit_path=root / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=root / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
    )


def test_unreadable_arm_refuses_protected_start_with_zero_launches(tmp_path: Path, monkeypatch):
    """I/O failure reading paper_arm_confirm.json must fail closed as INVALID."""
    _write_runtime_fixture(tmp_path)
    arm_path = _write_valid_arm(tmp_path)
    original = Path.read_text

    def _maybe_unreadable(self, *args, **kwargs):
        if Path(self) == arm_path:
            raise PermissionError("unreadable paper_arm_confirm.json")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _maybe_unreadable)
    assert classify_paper_arm_confirm_status(root=tmp_path) == "INVALID"
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_MALFORMED"
    assert rec["result"] == "START_REFUSED_PAPER_ARM_MALFORMED"
    assert launched == []
    assert rec.get("launched_components") in ([], None)
    assert not (tmp_path / "data" / "live" / "continuous_loop.lock").is_file()
    assert not (tmp_path / "data" / "live" / "watchdog_loop.lock").is_file()
    assert rec.get("paper_arm_mutated") is False
    # Synthetic file bytes are unchanged; real production arm was never opened.
    restored = original(arm_path, encoding="utf-8")
    assert "allow_execute" in restored
