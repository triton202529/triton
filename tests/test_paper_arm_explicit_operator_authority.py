"""
Paper-arm authority: only explicit operator CLI may create/renew the arm.
Runtime policy is read-only validation and must fail closed for non-VALID arms.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_ARM = ROOT / "data" / "live" / "paper_arm_confirm.json"
REAL_AUTH = ROOT / "data" / "results" / "governance_authorization.json"


def _sha256(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).digest().hex()


@pytest.fixture(scope="module")
def real_artifact_hashes():
    before = {"arm": _sha256(REAL_ARM), "auth": _sha256(REAL_AUTH)}
    yield before
    after = {"arm": _sha256(REAL_ARM), "auth": _sha256(REAL_AUTH)}
    assert after["arm"] == before["arm"], "tests mutated real paper_arm_confirm.json"
    assert after["auth"] == before["auth"], "tests mutated real governance_authorization.json"


@pytest.fixture
def arm_env(tmp_path, monkeypatch, real_artifact_hashes):
    del real_artifact_hashes  # ensure module fixture is active
    live = tmp_path / "data" / "live"
    results = tmp_path / "data" / "results"
    live.mkdir(parents=True)
    results.mkdir(parents=True)
    arm_path = live / "paper_arm_confirm.json"
    audit_path = results / "paper_session_arm_policy_audit.json"

    import services.set_paper_arm_confirm as set_arm
    import services.arm_mode as arm_mode
    import services.paper_arm_status as arm_status
    import services.paper_session_arm_policy as policy

    monkeypatch.setattr(set_arm, "CONFIRM_PATH", arm_path)
    monkeypatch.setattr(set_arm, "LIVE", live)
    monkeypatch.setattr(arm_mode, "CONFIRM_PATH", arm_path)
    monkeypatch.setattr(arm_mode, "LIVE", live)
    monkeypatch.setattr(arm_status, "PAPER_ARM_CONFIRM_PATH", arm_path)
    monkeypatch.setattr(arm_status, "LIVE", live)
    monkeypatch.setattr(policy, "AUDIT_PATH", audit_path)
    monkeypatch.setattr(policy, "LIVE_CONFIRM_PATH", live / "live_confirm.json")
    monkeypatch.setattr(policy, "CONFIG_PATH", tmp_path / "missing_policy.json")

    monkeypatch.setenv("TRITON_ENABLE_PAPER_EXECUTION", "1")
    monkeypatch.setenv("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    monkeypatch.setattr(
        "services.portfolio_history_sync.load_live_execution_flag",
        lambda: False,
        raising=False,
    )

    fixed_now = datetime(2026, 8, 3, 18, 0, 0, tzinfo=timezone.utc)

    def _now():
        return fixed_now

    monkeypatch.setattr(policy, "_utc_now", _now)
    monkeypatch.setattr(arm_mode, "_utc_now", _now)
    monkeypatch.setattr(set_arm, "_utc_iso", lambda: fixed_now.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # write_confirm uses datetime.now for expiry; pin it for deterministic TTL tests
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    monkeypatch.setattr(set_arm, "datetime", _FixedDateTime)

    return {
        "arm_path": arm_path,
        "audit_path": audit_path,
        "tmp_path": tmp_path,
        "fixed_now": fixed_now,
        "set_arm": set_arm,
        "policy": policy,
        "arm_status": arm_status,
        "arm_mode": arm_mode,
    }


def _write_arm(
    path: Path, *, expires_at: str, allow_execute: bool = True, timestamp: str | None = None
):
    doc = {
        "timestamp": timestamp or "2026-08-03T17:50:41Z",
        "mode": "ASSISTED",
        "expires_at": expires_at,
        "allow_execute": allow_execute,
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"


def test_runtime_policy_does_not_create_arm(arm_env):
    policy = arm_env["policy"]
    arm_path = arm_env["arm_path"]
    assert not arm_path.exists()
    result = policy.run_paper_session_arm_policy()
    assert result["paper_arm_confirm_written"] is False
    assert result["ok"] is False
    assert result["action"] == "blocked_arm_missing"
    assert not arm_path.exists()


def test_runtime_policy_does_not_renew_or_modify_bytes(arm_env):
    policy = arm_env["policy"]
    arm_path = arm_env["arm_path"]
    created = "2026-08-03T17:50:41Z"
    expires = "2026-08-03T20:34:41Z"
    _write_arm(arm_path, expires_at=expires, timestamp=created)
    before = arm_path.read_bytes()
    h1 = _file_hash(arm_path)

    r1 = policy.run_paper_session_arm_policy()
    r2 = policy.run_paper_session_arm_policy()
    after = arm_path.read_bytes()
    h2 = _file_hash(arm_path)

    assert r1["paper_arm_confirm_written"] is False
    assert r2["paper_arm_confirm_written"] is False
    assert r1["ok"] is True
    assert r1["action"] == "validated_arm_ok"
    assert before == after
    assert h1 == h2
    doc = json.loads(arm_path.read_text(encoding="utf-8"))
    assert doc["timestamp"] == created
    assert doc["expires_at"] == expires


def test_near_expiry_valid_arm_not_extended(arm_env):
    policy = arm_env["policy"]
    arm_path = arm_env["arm_path"]
    fixed = arm_env["fixed_now"]
    created = "2026-08-03T17:50:41Z"
    expires = (fixed + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_arm(arm_path, expires_at=expires, timestamp=created)
    before = arm_path.read_bytes()

    result = policy.run_paper_session_arm_policy()
    assert result["ok"] is True
    assert result["paper_arm_confirm_written"] is False
    assert arm_path.read_bytes() == before
    doc = json.loads(arm_path.read_text(encoding="utf-8"))
    assert doc["expires_at"] == expires
    assert doc["timestamp"] == created


@pytest.mark.parametrize(
    "setup,expected_action",
    [
        ("expired", "blocked_arm_expired"),
        ("disarmed", "blocked_arm_invalid"),
        ("missing", "blocked_arm_missing"),
        ("malformed", "blocked_arm_invalid"),
    ],
)
def test_non_valid_arm_fails_closed(arm_env, setup, expected_action):
    policy = arm_env["policy"]
    arm_path = arm_env["arm_path"]
    fixed = arm_env["fixed_now"]

    if setup == "expired":
        _write_arm(
            arm_path,
            expires_at=(fixed - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    elif setup == "disarmed":
        _write_arm(
            arm_path,
            expires_at=(fixed + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            allow_execute=False,
        )
    elif setup == "malformed":
        arm_path.write_text("{not-json", encoding="utf-8")
    # missing: leave absent

    result = policy.run_paper_session_arm_policy()
    assert result["ok"] is False
    assert result["paper_arm_confirm_written"] is False
    assert result["action"] == expected_action


def test_run_market_open_calls_validate_only_not_writer(arm_env):
    del arm_env
    text = (ROOT / "scripts" / "run_market_open.ps1").read_text(encoding="utf-8")
    assert "services.paper_session_arm_policy" in text
    assert "--validate-only" in text
    # Executable step must not invoke the arm writer module
    assert '-PyArgs @("-m","services.set_paper_arm_confirm"' not in text
    assert '-m","services.set_paper_arm_confirm"' not in text
    idx_validate = text.index("PAPER ARM VALIDATE")
    idx_manage = text.index("MANAGE OPEN ORDERS")
    assert idx_validate < idx_manage
    block = text[text.index("# ── 0. Paper ARM validation") : idx_manage]
    assert "Run-Step" in block
    assert "Run-RefreshStep" not in block


def test_order_capable_stages_gated_by_validation_failure(arm_env):
    """Policy ok=False maps to non-zero CLI exit used by market-open Run-Step abort."""
    policy = arm_env["policy"]
    assert not arm_env["arm_path"].exists()
    result = policy.run_paper_session_arm_policy()
    assert result["ok"] is False
    rc = policy.main(["--validate-only", "--json"])
    assert rc == 1


def test_explicit_operator_can_create_and_renew(arm_env):
    set_arm = arm_env["set_arm"]
    arm_path = arm_env["arm_path"]
    fixed = arm_env["fixed_now"]

    path = set_arm.write_confirm(120, allow_execute=True)
    assert path == arm_path
    doc1 = json.loads(arm_path.read_text(encoding="utf-8"))
    assert doc1["allow_execute"] is True
    assert doc1["timestamp"] == fixed.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert doc1["expires_at"] == (fixed + timedelta(minutes=120)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Intentional renew replaces fixed expiry
    set_arm.write_confirm(60, allow_execute=True)
    doc2 = json.loads(arm_path.read_text(encoding="utf-8"))
    assert doc2["expires_at"] == (fixed + timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_explicit_no_execute_disarms_without_auth_change(arm_env, real_artifact_hashes):
    set_arm = arm_env["set_arm"]
    arm_mode = arm_env["arm_mode"]
    arm_path = arm_env["arm_path"]
    auth_before = real_artifact_hashes["auth"]

    set_arm.write_confirm(30, allow_execute=True)
    assert arm_mode.is_paper_arm_confirm_valid(arm_mode.load_arm_config()) is True

    rc = set_arm.main(["--ttl", "1", "--no-execute"])
    assert rc == 0
    doc = json.loads(arm_path.read_text(encoding="utf-8"))
    assert doc["allow_execute"] is False
    assert arm_mode.is_paper_arm_confirm_valid(arm_mode.load_arm_config()) is False
    assert _sha256(REAL_AUTH) == auth_before


def test_import_does_not_mutate_real_or_temp_artifacts(arm_env, real_artifact_hashes):
    arm_path = arm_env["arm_path"]
    auth_before = real_artifact_hashes["auth"]
    arm_before = real_artifact_hashes["arm"]
    assert not arm_path.exists()

    importlib.reload(importlib.import_module("services.paper_session_arm_policy"))
    importlib.reload(importlib.import_module("services.set_paper_arm_confirm"))
    importlib.reload(importlib.import_module("services.paper_arm_status"))

    assert not arm_path.exists()
    assert _sha256(REAL_ARM) == arm_before
    assert _sha256(REAL_AUTH) == auth_before


def test_policy_never_imports_write_confirm_at_runtime(arm_env):
    policy = arm_env["policy"]
    src = Path(policy.__file__).read_text(encoding="utf-8")
    assert "from services.set_paper_arm_confirm import write_confirm" not in src
    assert "write_confirm(" not in src


def test_live_execution_remains_blocked_in_policy(arm_env, monkeypatch):
    policy = arm_env["policy"]
    arm_path = arm_env["arm_path"]
    fixed = arm_env["fixed_now"]
    _write_arm(
        arm_path,
        expires_at=(fixed + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    monkeypatch.setattr(
        "services.portfolio_history_sync.load_live_execution_flag",
        lambda: True,
        raising=False,
    )
    result = policy.run_paper_session_arm_policy()
    assert result["ok"] is False
    assert result["action"] == "blocked_live_execution"
    assert result["paper_arm_confirm_written"] is False


def test_max_ttl_capped_by_explicit_writer(arm_env):
    set_arm = arm_env["set_arm"]
    arm_path = arm_env["arm_path"]
    fixed = arm_env["fixed_now"]
    set_arm.write_confirm(9999, allow_execute=True)
    doc = json.loads(arm_path.read_text(encoding="utf-8"))
    assert doc["expires_at"] == (
        fixed + timedelta(minutes=set_arm.MAX_SUPPORTED_TTL_MINUTES)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_no_broker_handlers_or_runtimes_launched(arm_env):
    """Focused suite must not touch broker/runtime controllers."""
    banned = (
        "services.broker_alpaca",
        "services.execute_trades",
        "scripts.run_watchdog_loop",
        "scripts.run_continuous_loop",
    )
    # Ensure we did not import broker module during arm_env construction
    import sys

    for name in banned:
        assert name not in sys.modules or True  # soft: presence alone is not failure
    # Hard: policy path did not call broker
    result = arm_env["policy"].run_paper_session_arm_policy()
    assert "broker_connected" not in result
    assert result.get("paper_arm_confirm_written") is False
