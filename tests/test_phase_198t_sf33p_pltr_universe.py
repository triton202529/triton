"""PLTR universe onboarding and lifecycle fallback parser consistency."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from services.execute_trades import (
    PlannedOrder,
    REASON_DISPATCH_CONTRACT_REQUIRED,
    REASON_EDGE_EVIDENCE_UNPROVEN,
    REASON_INDEPENDENT_BLOCK_NEW_BUY,
    REASON_LIFECYCLE_STANCE_NOT_ENTRY,
    apply_paper_buy_add_entry_boundary,
    maybe_execute_plan,
)
from services.generate_signals import (
    SKIP_INVALID_PROCESSED_SESSION_BAR,
    SKIP_MISSING_EXPECTED_SESSION_BAR,
    _inspect_processed_session,
    _is_index_ticker,
    _load_optional_tickers,
)
from services.lifecycle_artifact_guard import (
    DuplicateUniverseTicker,
    load_universe_ticker_count,
    parse_universe_ticker_identities,
    preferred_expected_rows,
)
from services.sector_exposure import UNKNOWN_SECTOR_LABEL, evaluate_sector_cap, get_sector

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ROOT / "data" / "config" / "tickers.txt"


def _ids(path: Path | None = None) -> list[str]:
    return parse_universe_ticker_identities(path or TICKERS)


def _tradables(ids: list[str]) -> list[str]:
    return [t for t in ids if not t.startswith("^")]


def _indexes(ids: list[str]) -> list[str]:
    return [t for t in ids if t.startswith("^")]


def test_pltr_present_once_and_normalized_counts():
    ids = _ids()
    governed = _load_optional_tickers(str(TICKERS))
    assert ids == governed
    assert ids.count("PLTR") == 1
    assert len(_tradables(ids)) == 63
    assert len(ids) == 67
    assert _indexes(ids) == ["^DJI", "^GSPC", "^IXIC", "^VIX"]
    assert len(ids) == len(_tradables(ids)) + 4
    assert load_universe_ticker_count(TICKERS) == len(ids)
    assert not any(token.startswith("#") or "TRITON" in token or "\ufeff" in token for token in ids)
    assert not _is_index_ticker("PLTR")
    assert get_sector("PLTR") == "Tech"
    assert get_sector("PLTR") != UNKNOWN_SECTOR_LABEL


def test_bom_comment_and_blanks_are_not_counted(tmp_path: Path):
    raw = TICKERS.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    sample = tmp_path / "tickers.txt"
    sample.write_bytes(b"\xef\xbb\xbf# comment\r\n\r\nPLTR\r\n^VIX\r\n")
    ids = _ids(sample)
    assert ids == ["PLTR", "^VIX"]
    assert load_universe_ticker_count(sample) == 2


def test_duplicate_pltr_is_rejected(tmp_path: Path):
    sample = tmp_path / "tickers.txt"
    sample.write_text("# c\nPLTR\nPLTR\n", encoding="utf-8")
    with pytest.raises(DuplicateUniverseTicker, match="DUPLICATE_TICKER:PLTR"):
        parse_universe_ticker_identities(sample)


def test_fallback_count_matches_parser_when_diagnostics_absent(monkeypatch, tmp_path: Path):
    sample = tmp_path / "tickers.txt"
    sample.write_bytes(b"\xef\xbb\xbf# note\nAAPL\nPLTR\n^VIX\n")
    monkeypatch.setattr(
        "services.lifecycle_artifact_guard.load_asof_diagnostics",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "services.lifecycle_artifact_guard.TICKERS_FILE",
        sample,
    )
    assert preferred_expected_rows() == load_universe_ticker_count(sample) == 3


def _bars(session: date, *, bad: bool = False, rows: int = 30) -> pd.DataFrame:
    out = []
    for i in range(rows):
        d = session - timedelta(days=rows - 1 - i)
        close = 20.0 + i
        high = close + (0.5 if not bad else -1.0)
        low = close - 0.4
        out.append(
            {
                "date": d.isoformat(),
                "open": close - 0.1,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + i,
            }
        )
    return pd.DataFrame(out)


def test_valid_exact_session_pltr_is_eligible(tmp_path: Path, monkeypatch):
    session = date(2026, 9, 21)
    path = tmp_path / "PLTR.parquet"
    _bars(session).to_parquet(path, index=False)
    monkeypatch.setattr(
        "services.generate_signals._processed_path_for_ticker",
        lambda ticker: str(path) if ticker == "PLTR" else None,
    )
    view = _inspect_processed_session("PLTR", session)
    assert view.ok
    assert view.actual == "2026-09-21"
    assert view.reason is None


def test_missing_stale_and_malformed_pltr_are_dated_skips(tmp_path: Path, monkeypatch):
    session = date(2026, 9, 21)
    monkeypatch.setattr(
        "services.generate_signals._processed_path_for_ticker",
        lambda ticker: None,
    )
    missing = _inspect_processed_session("PLTR", session)
    assert missing.ok is False
    assert missing.reason == SKIP_MISSING_EXPECTED_SESSION_BAR

    stale_path = tmp_path / "stale.parquet"
    _bars(session - timedelta(days=5)).to_parquet(stale_path, index=False)
    monkeypatch.setattr(
        "services.generate_signals._processed_path_for_ticker",
        lambda ticker: str(stale_path),
    )
    stale = _inspect_processed_session("PLTR", session)
    assert stale.ok is False
    assert stale.reason == SKIP_MISSING_EXPECTED_SESSION_BAR
    assert stale.actual != "2026-09-21"

    bad_path = tmp_path / "bad.parquet"
    _bars(session, bad=True).to_parquet(bad_path, index=False)
    monkeypatch.setattr(
        "services.generate_signals._processed_path_for_ticker",
        lambda ticker: str(bad_path),
    )
    bad = _inspect_processed_session("PLTR", session)
    assert bad.ok is False
    assert bad.reason == SKIP_INVALID_PROCESSED_SESSION_BAR


def test_rf_file_does_not_restore_missing_pltr(tmp_path: Path, monkeypatch):
    session = date(2026, 9, 21)
    pred = tmp_path / "PLTR_predictions.parquet"
    pd.DataFrame({"date": [session.isoformat()], "rf_delta_pct": [5.0]}).to_parquet(
        pred, index=False
    )
    monkeypatch.setattr(
        "services.generate_signals._processed_path_for_ticker",
        lambda ticker: None,
    )
    view = _inspect_processed_session("PLTR", session)
    assert view.ok is False
    assert view.reason == SKIP_MISSING_EXPECTED_SESSION_BAR


def _entry_frame() -> pd.DataFrame:
    return pd.DataFrame([{"ticker": "PLTR", "opportunity_type": "ENTRY", "score": 0.7}])


def test_hold_block_and_allocation_cannot_authorize_pltr(monkeypatch):
    monkeypatch.setattr(
        "services.edge_sizing_engine.load_proven_edge_sizing_map",
        lambda *args, **kwargs: ({}, {"status": "FAIL_CLOSED", "reason": "unproven"}),
    )
    monkeypatch.setattr(
        "services.execute_trades.load_independent_current_session_lifecycle_stances",
        lambda: ({"PLTR": "HOLD"}, {"ok": True}),
    )
    monkeypatch.setattr(
        "services.execute_trades.load_independent_block_new_buy_symbols",
        lambda: (set(), {"ok": True}),
    )
    kept, meta = apply_paper_buy_add_entry_boundary(_entry_frame(), allocation_used=True)
    assert kept.empty
    assert "PLTR" not in meta["allowed"]
    assert REASON_LIFECYCLE_STANCE_NOT_ENTRY in meta["rejected"]["PLTR"]
    assert REASON_EDGE_EVIDENCE_UNPROVEN in meta["rejected"]["PLTR"]

    monkeypatch.setattr(
        "services.execute_trades.load_independent_current_session_lifecycle_stances",
        lambda: ({"PLTR": "BUY"}, {"ok": True}),
    )
    monkeypatch.setattr(
        "services.execute_trades.load_independent_block_new_buy_symbols",
        lambda: ({"PLTR"}, {"ok": True}),
    )
    monkeypatch.setattr(
        "services.edge_sizing_engine.load_proven_edge_sizing_map",
        lambda *args, **kwargs: ({"PLTR": {"status": "proven"}}, {"status": "VALID"}),
    )
    kept, meta = apply_paper_buy_add_entry_boundary(_entry_frame(), allocation_used=True)
    assert kept.empty
    assert meta["rejected"]["PLTR"] == [REASON_INDEPENDENT_BLOCK_NEW_BUY]


def test_pltr_sector_is_tech_and_not_unknown():
    blocked, reason, _cur, _proj, sector = evaluate_sector_cap(
        "PLTR",
        "BUY",
        1.0,
        {"sector_values": {"Tech": 1.0}, "total_value": 100.0},
        pending_sector_add={},
        pending_total_add=0.0,
        cfg={"allow_unknown_sector": False, "sector_caps_enabled": True},
    )
    assert sector == "Tech"
    assert reason != "UNKNOWN_SECTOR_BLOCK"
    assert blocked is False


def test_contract_free_execute_blocks_before_broker(tmp_path: Path):
    planned = PlannedOrder(
        symbol="PLTR",
        side="buy",
        qty=1,
        order_type="limit",
        time_in_force="day",
        limit_price=10.0,
        stance="BUY",
        rationale="temp",
        confidence=0.7,
        delta_pct=0.0,
        planned_notional=10.0,
        current_position_qty=0.0,
        mode="paper",
        generated_at="2026-09-22T00:00:00Z",
    )
    calls = []

    class Broker:
        def submit_order(self, *args, **kwargs):
            calls.append("submit_order")
            raise AssertionError("broker mutation")

    rc, submitted, reasons = maybe_execute_plan(
        [planned],
        "paper",
        False,
        {},
        Broker(),
        False,
        False,
        tmp_path / "orders.csv",
        dispatch_contract=None,
    )
    assert rc == 2
    assert submitted == 0
    assert reasons == [REASON_DISPATCH_CONTRACT_REQUIRED]
    assert calls == []


def test_parser_does_not_write(tmp_path: Path):
    sample = tmp_path / "tickers.txt"
    sample.write_text("PLTR\n", encoding="utf-8")
    before = sample.read_bytes()
    parse_universe_ticker_identities(sample)
    assert sample.read_bytes() == before
