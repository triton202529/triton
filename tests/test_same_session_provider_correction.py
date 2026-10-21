"""Same-session provider corrections. Temporary fixtures only. No network."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

from services.fetch_raw_data import (
    classify_provider_payload,
    session_correction_row_sha256,
)

SESSION = date(2026, 10, 20)
PRIOR = "2026-10-19"


def _row(day: str, **overrides) -> dict:
    row = {
        "date": day,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.5,
        "volume": 100,
    }
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _frame(rows: list[dict], symbol: str, **attrs) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.attrs["provider"] = attrs.get("provider", "yfinance")
    frame.attrs["adjustment"] = attrs.get("adjustment", "unadjusted")
    frame.attrs["interval"] = attrs.get("interval", "1d")
    frame.attrs["auto_adjust"] = attrs.get("auto_adjust", False)
    frame.attrs["symbol"] = attrs.get("symbol", symbol)
    return frame


def _mrk_local() -> list[dict]:
    return [
        _row(PRIOR, open=140.0, high=142.0, low=139.0, close=141.0, adj_close=141.0, volume=1000),
        _row(
            SESSION.isoformat(),
            open=146.2899932861328,
            high=149.36000061035156,
            low=146.3000030517578,
            close=148.3000030517578,
            adj_close=148.3000030517578,
            volume=9899397,
        ),
    ]


def _mrk_provider_rows() -> list[dict]:
    rows = _mrk_local()
    rows[1] = _row(
        SESSION.isoformat(),
        open=146.2899932861328,
        high=149.36000061035156,
        low=146.1199951171875,
        close=148.3000030517578,
        adj_close=148.3000030517578,
        volume=9902800,
    )
    return rows


def test_identical_same_session_bars_remain_already_current(tmp_path: Path) -> None:
    import services.fetch_raw_data as fetch

    rows = [_row(PRIOR, close=10.25, adj_close=10.25), _row(SESSION.isoformat())]
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, rows)
    before = path.read_bytes()
    decision = classify_provider_payload(_frame(rows, "AAA"), "AAA", SESSION, path)
    assert decision["reason"] == "local_already_current"
    assert decision["action"] == "keep"
    assert path.read_bytes() == before
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": _frame(rows, "AAA")}
    )
    assert receipt["promoted"] == []
    assert receipt["decisions"][0]["reason"] == "local_already_current"
    assert path.read_bytes() == before


def test_valid_provider_correction_is_promotable_without_classification_write(
    tmp_path: Path,
) -> None:
    import services.fetch_raw_data as fetch

    local = [_row(PRIOR), _row(SESSION.isoformat(), low=9.0)]
    corrected = [_row(PRIOR), _row(SESSION.isoformat(), low=8.5)]
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    decision = classify_provider_payload(_frame(corrected, "AAA"), "AAA", SESSION, path)
    assert decision["action"] == "promote"
    assert decision["reason"] == "same_session_provider_correction"
    assert path.read_bytes() == before
    assert not (tmp_path / "data" / "results").exists()
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": _frame(corrected, "AAA")}
    )
    assert receipt["promoted"] == ["AAA"]
    assert receipt["decisions"][0]["reason"] == "same_session_provider_correction"
    written = pd.read_csv(path)
    assert float(written.iloc[0]["low"]) == 9.0
    assert float(written.iloc[1]["low"]) == 8.5
    assert list(written["date"]) == [PRIOR, SESSION.isoformat()]


def test_mrk_style_invalid_local_promotes_valid_provider_bar(tmp_path: Path) -> None:
    import services.fetch_raw_data as fetch

    path = tmp_path / "data" / "raw" / "MRK.csv"
    _write(path, _mrk_local())
    before = path.read_bytes()
    decision = classify_provider_payload(_frame(_mrk_provider_rows(), "MRK"), "MRK", SESSION, path)
    assert decision["reason"] == "same_session_provider_correction"
    assert decision["reason"] != "local_already_current"
    assert path.read_bytes() == before
    record = decision["same_session_correction"]
    assert record["before"]["low"] == 146.3000030517578
    assert record["after"]["low"] == 146.1199951171875
    assert record["ohlc_violation"] is None
    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["MRK"],
        asof=SESSION,
        frames={"MRK": _frame(_mrk_provider_rows(), "MRK")},
    )
    assert receipt["promoted"] == ["MRK"]
    written = pd.read_csv(path)
    assert float(written.iloc[-1]["low"]) == 146.1199951171875
    assert float(written.iloc[-1]["open"]) == 146.2899932861328
    assert int(written.iloc[-1]["volume"]) == 9902800
    assert float(written.iloc[0]["close"]) == 141.0


def test_invalid_provider_does_not_overwrite_or_count(tmp_path: Path) -> None:
    import services.fetch_raw_data as fetch

    local = [_row(PRIOR), _row(SESSION.isoformat())]
    invalid = [_row(PRIOR), _row(SESSION.isoformat(), low=10.02)]
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    decision = classify_provider_payload(_frame(invalid, "AAA"), "AAA", SESSION, path)
    assert decision["reason"] == "invalid_provider_session_ohlc"
    assert decision["counts_toward_coverage"] is False
    assert decision["same_session_correction"]["ohlc_violation"]["relationship"] == "low_gt_open"
    assert path.read_bytes() == before
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": _frame(invalid, "AAA")}
    )
    assert receipt["promoted"] == []
    assert receipt["tradable_coverage"]["covered"] == 0
    assert path.read_bytes() == before


def test_adjusted_response_fails_closed(tmp_path: Path) -> None:
    import services.fetch_raw_data as fetch

    local = [_row(PRIOR), _row(SESSION.isoformat())]
    other = [_row(PRIOR), _row(SESSION.isoformat(), low=8.5, close=10.4, adj_close=9.9)]
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    frame = _frame(other, "AAA", adjustment="adjusted", auto_adjust=True)
    decision = classify_provider_payload(frame, "AAA", SESSION, path)
    assert decision["reason"] == "incompatible_provider_convention"
    assert decision["counts_toward_coverage"] is False
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": frame}
    )
    assert receipt["promoted"] == []
    assert receipt["tradable_coverage"]["covered"] == 0
    assert path.read_bytes() == before


def test_missing_provenance_fails_closed(tmp_path: Path) -> None:
    import services.fetch_raw_data as fetch

    local = [_row(PRIOR), _row(SESSION.isoformat())]
    other = [_row(PRIOR), _row(SESSION.isoformat(), low=8.5)]
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    frame = pd.DataFrame(other)
    decision = classify_provider_payload(frame, "AAA", SESSION, path)
    assert decision["reason"] == "missing_provider_provenance"
    assert decision["counts_toward_coverage"] is False
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": frame}
    )
    assert receipt["promoted"] == []
    assert path.read_bytes() == before


def test_same_session_conflict_records_values_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "data" / "raw" / "MRK.csv"
    _write(path, _mrk_local())
    decision = classify_provider_payload(_frame(_mrk_provider_rows(), "MRK"), "MRK", SESSION, path)
    record = decision["same_session_correction"]
    assert record["before"]["open"] == 146.2899932861328
    assert record["after"]["low"] == 146.1199951171875
    assert record["after"]["volume"] == 9902800
    assert record["before_row_sha256"] == session_correction_row_sha256(record["before"])
    assert record["after_row_sha256"] == session_correction_row_sha256(record["after"])
    assert record["before_raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["after_raw_sha256"] == hashlib.sha256(decision["candidate_bytes"]).hexdigest()
    assert record["provider"] == "yfinance"
    assert record["adjustment"] == "unadjusted"
    assert record["price_tolerance"] == 1e-6
    assert path.read_bytes()


def test_agreeing_invalid_local_bar_is_not_already_current(tmp_path: Path) -> None:
    rows = [_row(PRIOR), _row(SESSION.isoformat(), low=10.02)]
    path = tmp_path / "data" / "raw" / "T.csv"
    _write(path, rows)
    before = path.read_bytes()
    decision = classify_provider_payload(_frame(rows, "T"), "T", SESSION, path)
    assert decision["reason"] == "invalid_local_session_ohlc"
    assert decision["counts_toward_coverage"] is False
    assert path.read_bytes() == before
