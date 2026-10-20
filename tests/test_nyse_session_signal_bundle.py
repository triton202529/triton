"""NYSE session selection and fail-closed signal bundle publication."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services.apply_signal_lifecycle import _normalize_pair
from services.coherent_signal_bundle import (
    BUNDLE_FILES,
    assert_held_positions_stay_long,
    publish_coherent_bundle,
)
from services.governance_degraded_mode import entry_blocked_by_active_drawdown
from services.nyse_signal_session import required_signal_session

NY = ZoneInfo("America/New_York")
GENERATED = "2026-10-16T20:05:00Z"
SESSION = "2026-10-16"


def test_monday_after_weekend_uses_friday_session():
    monday_preopen = datetime(2026, 10, 19, 8, 0, tzinfo=NY)
    sunday = datetime(2026, 10, 18, 12, 0, tzinfo=NY)
    assert required_signal_session(monday_preopen) == date(2026, 10, 16)
    assert required_signal_session(sunday) == date(2026, 10, 16)


def test_holiday_and_early_close_use_latest_completed_session():
    thanksgiving = datetime(2026, 11, 26, 12, 0, tzinfo=NY)
    before_early_close = datetime(2026, 11, 27, 12, 0, tzinfo=NY)
    after_early_close = datetime(2026, 11, 27, 14, 0, tzinfo=NY)
    assert required_signal_session(thanksgiving) == date(2026, 11, 25)
    assert required_signal_session(before_early_close) == date(2026, 11, 25)
    assert required_signal_session(after_early_close) == date(2026, 11, 27)


def test_tuesday_session_uses_completed_monday():
    tuesday_open = datetime(2026, 10, 20, 12, 56, tzinfo=NY)
    assert required_signal_session(tuesday_open) == date(2026, 10, 19)


def test_missing_bar_preserves_prior_bundle(tmp_path, monkeypatch):
    import services.generate_signals as gs

    results = tmp_path / "results"
    processed = tmp_path / "processed"
    predictions = tmp_path / "predictions"
    config = tmp_path / "config"
    for path in (results, processed, predictions, config):
        path.mkdir()
    (config / "tickers.txt").write_text("AAA\n", encoding="utf-8")
    pd.DataFrame({"date": ["2026-10-16"], "close": [10.0]}).to_parquet(
        processed / "AAA.parquet", index=False
    )
    prior = {
        "signals.csv": b"PRIOR-SIGNALS\n",
        "signals_with_rationale.csv": b"PRIOR-RATIONALE\n",
        "signal_lifecycle.csv": b"PRIOR-LIFECYCLE\n",
        "signal_lifecycle_effective.csv": b"PRIOR-EFFECTIVE\n",
        "portfolio_allocation_recommendations.csv": b"PRIOR-ALLOCATION\n",
        "execution_readiness.json": b'{"risk_increasing_actions_blocked": true}\n',
    }
    for name, body in prior.items():
        (results / name).write_bytes(body)
    authority = tmp_path / "paper_sell_arm_authority.json"
    authority.write_text('{"status":"ARMED","consumption":"UNUSED"}\n', encoding="utf-8")
    authority_bytes = authority.read_bytes()

    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(processed))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(predictions))
    monkeypatch.setattr(gs, "TICKERS_PATH", str(config / "tickers.txt"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.setattr(gs, "_now_utc_ts", lambda: pd.Timestamp("2026-10-20T16:56:00Z"))

    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        gs.main()

    for name, body in prior.items():
        assert (results / name).read_bytes() == body
    receipt = json.loads(
        (results / "signal_bundle_failure_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["published_signal_artifacts"] is False
    assert receipt["session_as_of_date"] == "2026-10-19"
    assert (results / "signals_asof_diagnostics.json").is_file()
    assert authority.read_bytes() == authority_bytes
    assert not (results / "signals.csv").read_text(encoding="utf-8").startswith("date,")


def _stamped_csv(rows: list[dict]) -> str:
    frame = pd.DataFrame(rows)
    frame["bundle_generated_at_utc"] = GENERATED
    frame["bundle_session_as_of_date"] = SESSION
    return frame.to_csv(index=False)


def _candidate(staged, *, header_only: bool = False, omit: str | None = None) -> None:
    staged.mkdir(parents=True, exist_ok=True)
    signal_rows = [] if header_only else [{"ticker": "AAA", "signal": "HOLD"}]
    lifecycle_rows = [
        {"ticker": "AAA", "position_state": "FLAT", "lifecycle_action": "WAIT"},
        {"ticker": "NVDA", "position_state": "LONG", "lifecycle_action": "EXIT"},
    ]
    bodies = {
        "signals.csv": _stamped_csv(signal_rows),
        "signals_with_rationale.csv": _stamped_csv(signal_rows),
        "signal_lifecycle.csv": _stamped_csv(lifecycle_rows),
        "signal_lifecycle_effective.csv": _stamped_csv(lifecycle_rows),
        "portfolio_allocation_recommendations.csv": _stamped_csv(
            [{"ticker": "NVDA", "recommendation": "EXIT"}]
        ),
        "execution_readiness.json": json.dumps(
            {
                "bundle_generated_at_utc": GENERATED,
                "bundle_session_as_of_date": SESSION,
                "live_execution_permitted": False,
                "risk_increasing_actions_blocked": True,
                "paper_only": True,
            }
        )
        + "\n",
    }
    for name, body in bodies.items():
        if name == omit:
            continue
        (staged / name).write_text(body, encoding="utf-8")


def test_incomplete_candidate_does_not_replace_bundle(tmp_path):
    production = tmp_path / "production"
    staged = tmp_path / "staged"
    production.mkdir()
    prior = {name: f"OLD-{name}\n".encode("utf-8") for name in BUNDLE_FILES}
    for name, body in prior.items():
        (production / name).write_bytes(body)
    _candidate(staged, omit="signal_lifecycle.csv")
    with pytest.raises(RuntimeError, match="incomplete"):
        publish_coherent_bundle(production, staged, generated_at=GENERATED, session=SESSION)
    for name, body in prior.items():
        assert (production / name).read_bytes() == body


def test_header_only_candidate_is_not_published(tmp_path):
    production = tmp_path / "production"
    staged = tmp_path / "staged"
    production.mkdir()
    prior = {name: f"OLD-{name}\n".encode("utf-8") for name in BUNDLE_FILES}
    for name, body in prior.items():
        (production / name).write_bytes(body)
    _candidate(staged, header_only=True)
    with pytest.raises(RuntimeError, match="header-only"):
        publish_coherent_bundle(production, staged, generated_at=GENERATED, session=SESSION)
    for name, body in prior.items():
        assert (production / name).read_bytes() == body


def test_coherent_publish_shares_one_generation_and_keeps_nvda_long_exit(tmp_path):
    production = tmp_path / "production"
    staged = tmp_path / "staged"
    production.mkdir()
    authority = tmp_path / "paper_sell_arm_authority.json"
    authority.write_text('{"status":"ARMED","orders_submitted": false}\n', encoding="utf-8")
    authority_bytes = authority.read_bytes()
    _candidate(staged)
    publish_coherent_bundle(production, staged, generated_at=GENERATED, session=SESSION)

    stamps = set()
    for name in (
        "signals.csv",
        "signals_with_rationale.csv",
        "signal_lifecycle.csv",
        "signal_lifecycle_effective.csv",
        "portfolio_allocation_recommendations.csv",
    ):
        frame = pd.read_csv(production / name)
        stamps.update(frame["bundle_generated_at_utc"].astype(str))
        assert set(frame["bundle_session_as_of_date"].astype(str)) == {SESSION}
    readiness = json.loads((production / "execution_readiness.json").read_text(encoding="utf-8"))
    stamps.add(readiness["bundle_generated_at_utc"])
    assert stamps == {GENERATED}
    assert readiness["bundle_session_as_of_date"] == SESSION
    assert readiness["risk_increasing_actions_blocked"] is True
    assert readiness["live_execution_permitted"] is False

    position, action, _changes = _normalize_pair("FLAT", "EXIT", broker_held=True)
    assert (position, action) == ("LONG", "EXIT")
    lifecycle = pd.read_csv(production / "signal_lifecycle.csv")
    nvda = lifecycle[lifecycle["ticker"] == "NVDA"].iloc[0]
    assert nvda["position_state"] == "LONG"
    assert nvda["lifecycle_action"] == "EXIT"
    assert_held_positions_stay_long(lifecycle, {"NVDA"})
    assert authority.read_bytes() == authority_bytes


def test_active_nvda_drawdown_blocks_buy_add():
    blocked = entry_blocked_by_active_drawdown(
        [
            {
                "alert_type": "EXCESS_POSITION_DRAWDOWN",
                "status": "ACTIVE",
                "symbol": "NVDA",
                "first_seen": "2026-10-19T19:44:12Z",
            }
        ],
        datetime(2026, 10, 20, 16, 56, tzinfo=timezone.utc),
    )
    assert blocked is True
