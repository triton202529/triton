"""SF37 monotonic daily-refresh safety. TEMP fixtures and fake frames only."""

from __future__ import annotations

import hashlib
import socket
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SESSION = date(2026, 9, 22)
PRIOR = date(2026, 9, 21)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(day: date, price: float, *, symbol: str | None = None, volume=100) -> pd.DataFrame:
    row = {
        "date": day.isoformat(),
        "open": price,
        "high": price + 1.0,
        "low": price - 0.5,
        "close": price + 0.25,
        "volume": volume,
    }
    if symbol is not None:
        row["ticker"] = symbol
    return pd.DataFrame([row])


def _write_history(path: Path, last: date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    days = [last - timedelta(days=2), last - timedelta(days=1), last]
    pd.DataFrame(
        {
            "date": [d.isoformat() for d in days],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 110, 120],
        }
    ).to_csv(path, index=False)


def _governed() -> list[str]:
    from services.lifecycle_artifact_guard import parse_universe_ticker_identities

    return parse_universe_ticker_identities(ROOT / "data" / "config" / "tickers.txt")


def _seed(tmp: Path, symbols: list[str], *, skip: set[str] | None = None) -> None:
    absent = skip or set()
    (tmp / "data" / "config").mkdir(parents=True)
    (tmp / "data" / "config" / "tickers.txt").write_text(
        "\n".join(symbols) + "\n", encoding="utf-8"
    )
    for symbol in symbols:
        if symbol in absent:
            continue
        _write_history(tmp / "data" / "raw" / f"{symbol}.csv", SESSION)
        processed = tmp / "data" / "processed" / f"{symbol}.parquet"
        processed.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": [SESSION.isoformat()], "close": [1.0]}).to_parquet(
            processed, index=False
        )
        if symbol == "SPCX":
            continue
        pred = tmp / "data" / "predictions" / f"{symbol}_predictions.parquet"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "date": [SESSION.isoformat()],
                "close": [1.0],
                "predicted_close": [1.01],
                "signal": ["HOLD"],
                "ticker": [symbol],
            }
        ).to_parquet(pred, index=False)


def _hashes(root: Path, symbols: list[str], kind: str) -> dict[str, str]:
    out = {}
    for symbol in symbols:
        if kind == "raw":
            path = root / "data" / "raw" / f"{symbol}.csv"
        elif kind == "processed":
            path = root / "data" / "processed" / f"{symbol}.parquet"
        else:
            path = root / "data" / "predictions" / f"{symbol}_predictions.parquet"
        if path.is_file():
            out[symbol] = _sha(path)
    return out


def test_sf36_stale_provider_does_not_replace_or_create_pltr(tmp_path, monkeypatch):
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")),
    )
    import services.fetch_raw_data as fetch

    monkeypatch.setattr(
        fetch.yf, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("yahoo"))
    )
    symbols = _governed()
    tradables = [s for s in symbols if not s.startswith("^")]
    indexes = [s for s in symbols if s.startswith("^")]
    assert len(tradables) == 63 and "PLTR" in tradables and len(indexes) == 4
    _seed(tmp_path, symbols, skip={"PLTR"})
    originals = [s for s in tradables if s != "PLTR"]
    before_raw = _hashes(tmp_path, originals + indexes, "raw")
    before_proc = _hashes(tmp_path, originals + indexes, "processed")
    before_pred = _hashes(tmp_path, [s for s in originals + indexes if s != "SPCX"], "predictions")
    frames = {s: _frame(PRIOR, 4.0) for s in tradables}
    frames.update({s: _frame(SESSION, 8.0) for s in indexes})
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=symbols, asof=SESSION, frames=frames
    )
    assert receipt["passed"] is True
    assert receipt["promoted"] == []
    assert receipt["tradable_coverage"]["covered"] == 62
    assert receipt["tradable_coverage"]["denominator"] == 63
    assert receipt["tradable_coverage"]["ratio"] >= 0.90
    assert _hashes(tmp_path, originals + indexes, "raw") == before_raw
    assert _hashes(tmp_path, originals + indexes, "processed") == before_proc
    assert (
        _hashes(tmp_path, [s for s in originals + indexes if s != "SPCX"], "predictions")
        == before_pred
    )
    assert not (tmp_path / "data" / "raw" / "PLTR.csv").exists()
    assert {
        "symbol": "PLTR",
        "session": "2026-09-22",
        "reason": "MISSING_EXPECTED_SESSION_BAR",
        "operationally_eligible": False,
    } in receipt["dated_skips"]

    from services.daily_prediction_refresh_validate import validate_refresh
    from services.stale_data_gate import assess_raw_sessions

    gate = assess_raw_sessions(tmp_path, SESSION)
    assert gate["passed"] is True
    assert "PLTR" in gate["skipped"]
    result = validate_refresh(
        repo_root=tmp_path,
        target_session=SESSION,
        run_id="sf37",
        run_started_at="2026-09-23T00:00:00Z",
        require_advancement=False,
    )
    assert "PLTR" not in result["predictions"]["missing_required_symbols"]
    assert result["tradable_coverage"]["covered"] == 62
    assert result["tradable_coverage"]["ratio"] >= 0.90
    assert result["skipped_symbols_remain_ineligible"][0]["operationally_eligible"] is False


def test_exact_session_pltr_bar_is_promoted_and_others_stay(tmp_path, monkeypatch):
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")),
    )
    import services.fetch_raw_data as fetch

    monkeypatch.setattr(
        fetch.yf, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("yahoo"))
    )
    symbols = _governed()
    tradables = [s for s in symbols if not s.startswith("^")]
    indexes = [s for s in symbols if s.startswith("^")]
    _seed(tmp_path, symbols, skip={"PLTR"})
    originals = [s for s in tradables if s != "PLTR"]
    before = _hashes(tmp_path, originals + indexes, "raw")
    frames = {s: _frame(PRIOR, 4.0) for s in originals}
    frames["PLTR"] = _frame(SESSION, 7.0, symbol="PLTR")
    frames.update({s: _frame(SESSION, 8.0) for s in indexes})
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=symbols, asof=SESSION, frames=frames
    )
    assert receipt["passed"] is True
    assert receipt["promoted"] == ["PLTR"]
    assert (tmp_path / "data" / "raw" / "PLTR.csv").is_file()
    assert _hashes(tmp_path, originals + indexes, "raw") == before
    assert "PLTR" not in {row["symbol"] for row in receipt["dated_skips"]}


def test_exact_session_new_symbol_can_be_promoted(tmp_path):
    import services.fetch_raw_data as fetch

    symbols = ["AAA", "BBB", "^DJI"]
    _seed(tmp_path, symbols, skip={"BBB"})
    frames = {
        "AAA": _frame(PRIOR, 1.0),
        "BBB": _frame(SESSION, 6.0, symbol="BBB"),
        "^DJI": _frame(PRIOR, 2.0),
    }
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=symbols, asof=SESSION, frames=frames
    )
    assert receipt["promoted"] == ["BBB"]
    assert (tmp_path / "data" / "raw" / "BBB.csv").is_file()
    assert _sha(tmp_path / "data" / "raw" / "AAA.csv")


def test_provider_older_than_local_does_not_replace(tmp_path):
    import services.fetch_raw_data as fetch

    _seed(tmp_path, ["AAA"])
    before = _sha(tmp_path / "data" / "raw" / "AAA.csv")
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": _frame(PRIOR, 3.0)}
    )
    assert receipt["promoted"] == []
    assert _sha(tmp_path / "data" / "raw" / "AAA.csv") == before
    assert receipt["decisions"][0]["reason"] == "provider_older_than_local"


@pytest.mark.parametrize(
    "mutator",
    ["duplicate", "malformed", "partial", "future", "wrong_symbol", "empty", "mixed", "relabel"],
)
def test_invalid_provider_payload_does_not_replace(tmp_path, mutator):
    import services.fetch_raw_data as fetch

    _seed(tmp_path, ["AAA"])
    before = _sha(tmp_path / "data" / "raw" / "AAA.csv")
    history = pd.read_csv(tmp_path / "data" / "raw" / "AAA.csv")
    if mutator == "duplicate":
        frame = pd.concat([history, history.iloc[[-1]]], ignore_index=True)
    elif mutator == "malformed":
        frame = history.copy()
        frame.loc[frame.index[-1], "high"] = 0.1
    elif mutator == "partial":
        frame = history.copy()
        frame.loc[frame.index[-1], "volume"] = None
    elif mutator == "future":
        frame = history.copy()
        frame.loc[frame.index[-1], "date"] = "2026-09-23"
    elif mutator == "wrong_symbol":
        frame = history.copy()
        frame["ticker"] = "ZZZ"
    elif mutator == "empty":
        frame = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    elif mutator == "mixed":
        frame = history.copy()
        extra = history.iloc[[-1]].copy()
        extra["date"] = "2026-09-23"
        frame = pd.concat([frame, extra], ignore_index=True)
    else:
        frame = history.copy()
        frame.loc[frame.index[-1], ["open", "high", "low", "close", "volume"]] = frame.loc[
            frame.index[0], ["open", "high", "low", "close", "volume"]
        ].values
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": frame}
    )
    assert receipt["promoted"] == []
    assert _sha(tmp_path / "data" / "raw" / "AAA.csv") == before


def test_absent_local_and_stale_provider_creates_nothing(tmp_path):
    import services.fetch_raw_data as fetch

    (tmp_path / "data" / "raw").mkdir(parents=True)
    _write_history(tmp_path / "data" / "raw" / "AAA.csv", SESSION)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["AAA", "BBB"],
        asof=SESSION,
        frames={"AAA": _frame(PRIOR, 1.0), "BBB": _frame(PRIOR, 2.0)},
    )
    assert not (tmp_path / "data" / "raw" / "BBB.csv").exists()
    assert any(
        row["symbol"] == "BBB" and row["reason"] == "MISSING_EXPECTED_SESSION_BAR"
        for row in receipt["dated_skips"]
    )


def test_coverage_below_minimum_promotes_nothing(tmp_path):
    import services.fetch_raw_data as fetch

    symbols = [f"S{i}" for i in range(10)]
    for symbol in symbols[:8]:
        _write_history(tmp_path / "data" / "raw" / f"{symbol}.csv", SESSION)
    before = _sha(tmp_path / "data" / "raw" / "S0.csv")
    history = pd.read_csv(tmp_path / "data" / "raw" / "S0.csv")
    frames = {s: history.copy() for s in symbols[:8]}
    frames["S8"] = _frame(PRIOR, 1.0)
    frames["S9"] = _frame(PRIOR, 1.0)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=symbols, asof=SESSION, frames=frames
    )
    assert receipt["passed"] is False
    assert receipt["failure"] == "coverage_below_minimum"
    assert receipt["promoted"] == []
    assert _sha(tmp_path / "data" / "raw" / "S0.csv") == before


def test_failure_after_staging_writes_no_raw_file(tmp_path):
    import services.fetch_raw_data as fetch

    _seed(tmp_path, ["AAA"], skip={"BBB"})
    before = _sha(tmp_path / "data" / "raw" / "AAA.csv")

    def _boom():
        raise RuntimeError("staged")

    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["AAA", "BBB"],
        asof=SESSION,
        frames={"AAA": _frame(PRIOR, 1), "BBB": _frame(SESSION, 5)},
        before_promote=_boom,
    )
    assert receipt["failure"] == "staged_before_promotion"
    assert receipt["promoted"] == []
    assert not (tmp_path / "data" / "raw" / "BBB.csv").exists()
    assert _sha(tmp_path / "data" / "raw" / "AAA.csv") == before


def test_failed_atomic_replacement_does_not_retry(tmp_path):
    import services.fetch_raw_data as fetch

    _seed(tmp_path, ["AAA"], skip={"BBB"})
    calls = {"n": 0}

    def _boom(src, dest):
        calls["n"] += 1
        raise OSError("disk")

    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["AAA", "BBB"],
        asof=SESSION,
        frames={"AAA": _frame(PRIOR, 1), "BBB": _frame(SESSION, 5)},
        replace_impl=_boom,
    )
    assert calls["n"] == 1
    assert receipt["failure"] == "atomic_replacement_failed"
    assert receipt["durable_state"] == "promotion_stopped_no_retry"
    assert not (tmp_path / "data" / "raw" / "BBB.csv").exists()


def test_skipped_symbol_prediction_is_not_required_and_cannot_restore_eligibility(tmp_path):
    from services.daily_prediction_refresh_validate import validate_refresh
    import json

    symbols = ["AAA", "^DJI"]
    _seed(tmp_path, symbols, skip=set())
    (tmp_path / "data" / "predictions" / "NEW_predictions.parquet").parent.mkdir(
        parents=True, exist_ok=True
    )
    skips = {
        "target_session": SESSION.isoformat(),
        "skips": [
            {
                "symbol": "NEW",
                "session": SESSION.isoformat(),
                "reason": "MISSING_EXPECTED_SESSION_BAR",
                "operationally_eligible": False,
            }
        ],
    }
    (tmp_path / "data" / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "results" / "daily_refresh_input_skips.json").write_text(
        json.dumps(skips), encoding="utf-8"
    )
    (tmp_path / "data" / "config" / "tickers.txt").write_text("AAA\nNEW\n^DJI\n", encoding="utf-8")
    pd.DataFrame(
        {
            "date": [SESSION.isoformat()],
            "close": [1.0],
            "predicted_close": [9.0],
            "signal": ["BUY"],
            "ticker": ["NEW"],
        }
    ).to_parquet(tmp_path / "data" / "predictions" / "NEW_predictions.parquet", index=False)
    result = validate_refresh(
        repo_root=tmp_path,
        target_session=SESSION,
        require_advancement=False,
    )
    assert "NEW" not in result["predictions"]["missing_required_symbols"]
    assert result["skipped_symbols_remain_ineligible"][0]["prediction_present"] is True
    assert result["skipped_symbols_remain_ineligible"][0]["operationally_eligible"] is False

    (tmp_path / "data" / "predictions" / "AAA_predictions.parquet").unlink()
    (tmp_path / "data" / "results" / "daily_refresh_input_skips.json").unlink()
    (tmp_path / "data" / "config" / "tickers.txt").write_text("AAA\n^DJI\n", encoding="utf-8")
    missing = validate_refresh(
        repo_root=tmp_path,
        target_session=SESSION,
        require_advancement=False,
    )
    assert missing["passed"] is False
    assert "AAA" in missing["predictions"]["missing_required_symbols"]
    assert missing["first_failure"] == "predictions_missing_required"


def test_rf_shadow_flag_is_unchanged():
    text = (ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    assert "RF_SHADOW_ONLY = True" in text
