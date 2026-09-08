"""Phase 198T-BM1 — XNYS is the only market-open authority for shared helper + gates."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from services.auto_paper_approval import _validate_auto_approval_gates
from services.command_center_status_coherence import resolve_market_status
from services.guard_auto import _weekday_clock_open_et, is_market_open_et
from services.operator_console_status import (
    SESSION_AFTER_HOURS,
    SESSION_MARKET_OPEN,
    SESSION_NON_TRADING_DAY,
    SESSION_PRE_MARKET,
    SESSION_UNKNOWN,
    classify_operator_session_context,
)
from services.paper_entry_loop_stage import evaluate_paper_entry_gates

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
GUARD_SRC = (ROOT / "services" / "guard_auto.py").read_text(encoding="utf-8-sig")


def _et(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _legacy_weekday_clock(now_utc: datetime) -> bool:
    """Pre-BM1 clock-only helper. Exists only to prove the old defect is gone."""
    now_et = now_utc.astimezone(ET)
    if now_et.weekday() >= 5:
        return False
    h, m = now_et.hour, now_et.minute
    after_open = (h > 9) or (h == 9 and m >= 30)
    before_close = h < 16
    return after_open and before_close


CLOSED_CASES = [
    ("labor_09:29", _et(2026, 9, 7, 9, 29), False),
    ("labor_09:30", _et(2026, 9, 7, 9, 30), False),
    ("labor_12:00", _et(2026, 9, 7, 12, 0), False),
    ("labor_15:59", _et(2026, 9, 7, 15, 59), False),
    ("labor_16:00", _et(2026, 9, 7, 16, 0), False),
    ("tue_09:29", _et(2026, 9, 8, 9, 29), False),
    ("tue_09:30", _et(2026, 9, 8, 9, 30), True),
    ("tue_12:00", _et(2026, 9, 8, 12, 0), True),
    ("tue_15:59", _et(2026, 9, 8, 15, 59), True),
    ("tue_16:00", _et(2026, 9, 8, 16, 0), False),
    ("tue_16:01", _et(2026, 9, 8, 16, 1), False),
    ("thanksgiving_noon", _et(2026, 11, 26, 12, 0), False),
    ("xmas_eve_12:59", _et(2026, 12, 24, 12, 59), True),
    ("xmas_eve_13:00", _et(2026, 12, 24, 13, 0), False),
    ("xmas_eve_15:00", _et(2026, 12, 24, 15, 0), False),
    ("saturday", _et(2026, 9, 5, 12, 0), False),
    ("sunday", _et(2026, 9, 6, 12, 0), False),
    ("christmas_day", _et(2026, 12, 25, 12, 0), False),
]


@pytest.mark.parametrize("label,when,expect_open", CLOSED_CASES, ids=[c[0] for c in CLOSED_CASES])
def test_shared_helper_matches_xnys(label: str, when: datetime, expect_open: bool) -> None:
    assert is_market_open_et(when) is expect_open


@pytest.mark.parametrize(
    "label,when",
    [
        ("labor_noon", _et(2026, 9, 7, 12, 0)),
        ("thanksgiving_noon", _et(2026, 11, 26, 12, 0)),
        ("xmas_eve_15:00", _et(2026, 12, 24, 15, 0)),
    ],
)
def test_legacy_weekday_clock_would_incorrectly_open(label: str, when: datetime) -> None:
    assert _legacy_weekday_clock(when) is True
    assert _weekday_clock_open_et(when) is True
    assert is_market_open_et(when) is False


def test_naive_timestamp_fail_closed() -> None:
    naive = datetime(2026, 9, 8, 12, 0)
    assert naive.tzinfo is None
    assert is_market_open_et(naive) is False


def test_invalid_timestamp_fail_closed() -> None:
    assert is_market_open_et(None) is False  # type: ignore[arg-type]
    assert is_market_open_et("2026-09-08T12:00:00") is False  # type: ignore[arg-type]


def test_calendar_exception_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("calendar boom")

    monkeypatch.setattr("services.market_calendar.regular_session_state", _boom)
    assert is_market_open_et(_et(2026, 9, 8, 12, 0)) is False


def test_missing_calendar_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(*_a, **_k):
        raise ImportError("exchange_calendars missing")

    monkeypatch.setattr(
        "services.market_calendar._import_exchange_calendars",
        _missing,
    )
    assert is_market_open_et(_et(2026, 9, 8, 12, 0)) is False


def test_unknown_calendar_state_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {"session_state": "UNKNOWN", "regular_session_open": False},
    )
    assert is_market_open_et(_et(2026, 9, 8, 12, 0)) is False


def test_malformed_payload_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.market_calendar.regular_session_state", lambda now=None: None)
    assert is_market_open_et(_et(2026, 9, 8, 12, 0)) is False


def test_source_helper_delegates_to_xnys_without_weekday_fallback() -> None:
    tree = ast.parse(GUARD_SRC)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_market_open_et"
    )
    text = ast.get_source_segment(GUARD_SRC, fn) or ""
    assert "regular_session_state" in text
    assert "SESSION_OPEN" in text
    assert "weekday()" not in text
    assert "hour" not in text
    assert "No holiday calendar" not in text
    clock = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_weekday_clock_open_et"
    )
    clock_src = ast.get_source_segment(GUARD_SRC, clock) or ""
    assert "Must never grant OPEN" in clock_src


def _prime_entry_to_market_hours(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    alloc = tmp_path / "portfolio_allocation_recommendations.csv"
    alloc.write_text("symbol,weight\nAAPL,0.1\n", encoding="utf-8")
    monkeypatch.setattr(
        "services.paper_entry_loop_stage.resolve_entry_invocation_id",
        lambda **_k: ("mo-bm1-test", None),
    )
    monkeypatch.setattr(
        "services.manage_positions_cycle_binding.evaluate_exit_before_entry",
        lambda **_k: {"decision": "ALLOW"},
    )
    monkeypatch.setattr(
        "services.paper_entry_loop_stage._load_governed_allocation_candidates",
        lambda: ([], {}),
    )
    monkeypatch.setattr(
        "services.triton_env_loader.paper_runtime_env_ready",
        lambda: (
            True,
            "ok",
            {
                "paper_execution_enabled": True,
                "auto_paper_entries_enabled": True,
                "env_source": "test",
                "supervisor_runtime_active": False,
            },
        ),
    )
    monkeypatch.setattr("services.paper_entry_loop_stage.ALLOC_RECOMMENDATIONS", alloc)


@pytest.mark.parametrize(
    "label,when,expect_open",
    [
        ("labor_noon", _et(2026, 9, 7, 12, 0), False),
        ("tue_noon", _et(2026, 9, 8, 12, 0), True),
        ("thanksgiving_noon", _et(2026, 11, 26, 12, 0), False),
        ("xmas_eve_15:00", _et(2026, 12, 24, 15, 0), False),
        ("saturday", _et(2026, 9, 5, 12, 0), False),
    ],
)
def test_entry_gates_honor_xnys_without_loop_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    when: datetime,
    expect_open: bool,
) -> None:
    _prime_entry_to_market_hours(monkeypatch, tmp_path)
    ok, reason, details = evaluate_paper_entry_gates(now=when, results_root=tmp_path)
    assert details["checks"]["market_open"] is expect_open
    if not expect_open:
        assert ok is False
        assert details["failed_check"] == "market_open"
        assert reason == "market is closed"


def _approval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")
    monkeypatch.setenv("TRITON_ENABLE_PAPER_EXECUTION", "1")
    monkeypatch.setenv("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    monkeypatch.setattr("services.auto_paper_approval._load_risk_flags", lambda: {})


@pytest.mark.parametrize(
    "label,when,expect_open",
    [
        ("labor_noon", _et(2026, 9, 7, 12, 0), False),
        ("tue_noon", _et(2026, 9, 8, 12, 0), True),
        ("thanksgiving_noon", _et(2026, 11, 26, 12, 0), False),
        ("xmas_eve_13:00", _et(2026, 12, 24, 13, 0), False),
        ("unknown_injected", None, False),
    ],
)
def test_auto_approval_gates_honor_xnys(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    when,
    expect_open: bool,
) -> None:
    _approval_env(monkeypatch)
    if when is None:
        monkeypatch.setattr(
            "services.market_calendar.regular_session_state",
            lambda now=None: {"session_state": "UNKNOWN", "regular_session_open": False},
        )
        when = _et(2026, 9, 8, 12, 0)
    ok, reason, checks = _validate_auto_approval_gates(["AAPL"], now=when)
    assert checks["market_open"] is expect_open
    if not expect_open:
        assert ok is False
        assert reason == "MARKET_CLOSED"
    else:
        assert reason != "MARKET_CLOSED"


def test_entry_cannot_bypass_unknown_xnys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prime_entry_to_market_hours(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {"session_state": "UNKNOWN", "regular_session_open": False},
    )
    ok, reason, details = evaluate_paper_entry_gates(
        now=_et(2026, 9, 8, 12, 0), results_root=tmp_path
    )
    assert ok is False
    assert details["checks"]["market_open"] is False
    assert details["failed_check"] == "market_open"
    assert reason == "market is closed"


def test_approval_cannot_bypass_unknown_xnys(monkeypatch: pytest.MonkeyPatch) -> None:
    _approval_env(monkeypatch)
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {"session_state": "UNKNOWN", "regular_session_open": False},
    )
    ok, reason, checks = _validate_auto_approval_gates(["AAPL"], now=_et(2026, 9, 8, 12, 0))
    assert ok is False
    assert checks["market_open"] is False
    assert reason == "MARKET_CLOSED"


def test_console_labor_day_is_not_open() -> None:
    ctx = classify_operator_session_context(now=_et(2026, 9, 7, 12, 0))
    assert ctx["market_open"] is False
    assert ctx["session_phase"] != SESSION_MARKET_OPEN
    assert ctx["session_phase"] == SESSION_NON_TRADING_DAY
    assert ctx["session_reason"] == "holiday"
    assert ctx["status"] != "OPEN"


def test_console_tuesday_session_phases() -> None:
    pre = classify_operator_session_context(now=_et(2026, 9, 8, 9, 29))
    assert pre["session_phase"] == SESSION_PRE_MARKET
    assert pre["market_open"] is False
    open_ctx = classify_operator_session_context(now=_et(2026, 9, 8, 12, 0))
    assert open_ctx["session_phase"] == SESSION_MARKET_OPEN
    assert open_ctx["market_open"] is True
    after = classify_operator_session_context(now=_et(2026, 9, 8, 16, 1))
    assert after["session_phase"] == SESSION_AFTER_HOURS
    assert after["market_open"] is False


def test_console_unknown_is_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {
            "session_state": "UNKNOWN",
            "regular_session_open": False,
            "reason": "calendar_import_unavailable",
        },
    )
    ctx = classify_operator_session_context(now=_et(2026, 9, 8, 12, 0))
    assert ctx["market_open"] is not True
    assert ctx["session_phase"] == SESSION_UNKNOWN
    assert ctx["status"] != "OPEN"


def test_command_center_labor_day_closed() -> None:
    status = resolve_market_status(now=_et(2026, 9, 7, 12, 0))
    assert status["market_open"] is False
    assert status["status"] == "CLOSED"
    assert status["reason"] == "holiday"


def test_command_center_unknown_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {
            "session_state": "UNKNOWN",
            "regular_session_open": False,
            "reason": "calendar_import_unavailable",
        },
    )
    status = resolve_market_status(now=_et(2026, 9, 8, 12, 0))
    assert status["market_open"] is not True
    assert status["status"] == "UNKNOWN"


def test_heartbeat_wording_unchanged() -> None:
    src = (ROOT / "services" / "touch_pipeline_heartbeat.py").read_text(encoding="utf-8")
    assert 'stage="market_open_cycle"' in src
    assert "Market-open sequence completed" in src


def test_consumers_have_no_private_weekday_clock() -> None:
    entry = (ROOT / "services" / "paper_entry_loop_stage.py").read_text(encoding="utf-8")
    approval = (ROOT / "services" / "auto_paper_approval.py").read_text(encoding="utf-8")
    for src in (entry, approval):
        assert "is_market_open_et" in src
        assert "weekday() >= 5" not in src
        tree = ast.parse(src)
        # The market-open decision must go through the shared helper.
        assert "from services.guard_auto import is_market_open_et" in src
