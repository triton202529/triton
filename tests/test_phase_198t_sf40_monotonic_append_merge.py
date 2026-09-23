"""SF40A monotonic append. Synthetic frames only. No network and no production writes."""

from __future__ import annotations

import hashlib
import os
import socket
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

SESSION = date(2026, 9, 23)
PRIOR = [
    date(2026, 9, 16),
    date(2026, 9, 17),
    date(2026, 9, 18),
    date(2026, 9, 21),
    date(2026, 9, 22),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table(days: list[date], *, close0: float = 10.0, symbol: str | None = None) -> pd.DataFrame:
    rows = []
    for offset, day in enumerate(days):
        price = close0 + offset
        row = {
            "date": day.isoformat(),
            "open": price,
            "high": price + 1.0,
            "low": price - 0.5,
            "close": price + 0.25,
            "volume": 100 + offset,
        }
        if symbol is not None:
            row["ticker"] = symbol
        rows.append(row)
    return pd.DataFrame(rows)


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _values(path: Path) -> list[tuple]:
    frame = pd.read_csv(path)
    return [tuple(row) for row in frame.itertuples(index=False, name=None)]


def _last_xnys_sessions(end: date, count: int) -> list[date]:
    import exchange_calendars as xc

    start = end - timedelta(days=int(count * 2) + 30)
    labels = xc.get_calendar("XNYS").sessions_in_range(start.isoformat(), end.isoformat())
    days = [pd.Timestamp(item).date() for item in labels]
    assert days[-1] == end
    assert len(days) >= count
    return days[-count:]


def _block_network(monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")),
    )


def test_shorter_history_appends_one_newer_session(tmp_path, monkeypatch):
    _block_network(monkeypatch)
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    provider = pd.concat([local.iloc[1:], _table([SESSION], close0=40.0)], ignore_index=True)
    assert len(provider) == len(local)
    assert provider["date"].iloc[0] != local["date"].iloc[0]
    assert provider["date"].iloc[-1] == SESSION.isoformat()
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["decisions"][0]["reason"] == "monotonic_append"
    assert receipt["decisions"][0]["reason"] != "provider_would_truncate"
    assert path.read_bytes().startswith(before)
    assert _values(path)[-1][0] == SESSION.isoformat()


def test_larger_non_subset_frame_is_not_provider_would_truncate(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = _values(path)
    provider = pd.concat(
        [
            local.iloc[1:],
            _table([date(2026, 9, 19)], close0=19.0),
            _table([SESSION], close0=40.0),
        ],
        ignore_index=True,
    )
    assert len(provider) > len(local)
    assert set(local["date"]) - set(provider["date"]) == {"2026-09-16"}
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["decisions"][0]["reason"] == "monotonic_append"
    assert receipt["decisions"][0]["omitted_prefix_count"] == 1
    assert _values(path)[: len(before)] == before


def test_omitted_old_prefix_preserves_local_rows(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    provider = pd.concat([local.iloc[2:], _table([SESSION], close0=40.0)], ignore_index=True)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["decisions"][0]["omitted_prefix_count"] == 2
    assert receipt["promoted"] == ["AAA"]
    assert path.read_bytes().startswith(before)


def test_overlap_conflict_is_diagnostic_only(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    provider = pd.concat([local.iloc[1:], _table([SESSION], close0=40.0)], ignore_index=True)
    provider.loc[provider.index[0], "volume"] = 999
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["decisions"][0]["reason"] == "monotonic_append"
    assert receipt["decisions"][0]["overlap_conflicts"] == ["2026-09-17"]
    assert path.read_bytes().startswith(before)


def test_local_only_internal_date_is_preserved(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = _values(path)
    kept = local[local["date"] != "2026-09-17"]
    provider = pd.concat([kept, _table([SESSION], close0=40.0)], ignore_index=True)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["decisions"][0]["reason"] == "monotonic_append"
    assert "2026-09-17" in receipt["decisions"][0]["local_only_internal_dates"]
    assert _values(path)[: len(before)] == before


def test_provider_only_internal_date_is_not_inserted(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    extra = _table([date(2026, 9, 19)], close0=19.0)
    provider = pd.concat([local, extra, _table([SESSION], close0=40.0)], ignore_index=True)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["decisions"][0]["reason"] == "monotonic_append"
    assert receipt["decisions"][0]["provider_only_internal_dates"] == ["2026-09-19"]
    assert "2026-09-19" not in [row[0] for row in _values(path)]
    assert [row[0] for row in _values(path)] == [day.isoformat() for day in PRIOR] + [
        SESSION.isoformat()
    ]


def test_missing_required_session_between_local_max_and_target(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR[:3])
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = _sha(path)
    provider = pd.concat([local, _table([SESSION], close0=40.0)], ignore_index=True)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["promoted"] == []
    assert receipt["decisions"][0]["reason"] == "missing_required_session"
    assert receipt["decisions"][0]["missing_required_sessions"] == ["2026-09-21", "2026-09-22"]
    assert _sha(path) == before


def test_stale_maximum_does_not_append(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR + [SESSION])
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = _sha(path)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": _table(PRIOR)}
    )
    assert receipt["decisions"][0]["reason"] == "provider_older_than_local"
    assert _sha(path) == before


def test_exact_session_absent_keeps_local(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = _sha(path)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": local.iloc[1:].copy()}
    )
    assert receipt["decisions"][0]["reason"] == "MISSING_EXPECTED_SESSION_BAR"
    assert _sha(path) == before


def test_duplicate_exact_session_rows_fail_closed(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = _sha(path)
    provider = pd.concat(
        [local, _table([SESSION], close0=40.0), _table([SESSION], close0=41.0)],
        ignore_index=True,
    )
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["promoted"] == []
    assert "duplicate_session" in receipt["decisions"][0]["problems"]
    assert _sha(path) == before


@pytest.mark.parametrize(
    "mutator",
    ["wrong_symbol", "future", "nonpositive", "bad_geometry", "bad_volume"],
)
def test_invalid_provider_rows_fail_closed(tmp_path, mutator):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = _sha(path)
    provider = pd.concat([local, _table([SESSION], close0=40.0)], ignore_index=True)
    if mutator == "wrong_symbol":
        provider["ticker"] = "ZZZ"
    elif mutator == "future":
        provider.loc[provider.index[-1], "date"] = "2026-09-24"
    elif mutator == "nonpositive":
        provider.loc[provider.index[-1], "open"] = 0
    elif mutator == "bad_geometry":
        provider.loc[provider.index[-1], "high"] = 0.1
    else:
        provider.loc[provider.index[-1], "volume"] = -1
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["promoted"] == []
    assert _sha(path) == before


def test_local_already_current_is_not_rewritten(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR + [SESSION])
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    provider = local.iloc[1:].copy()
    provider.loc[provider.index[0], "volume"] = 999
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["promoted"] == []
    assert receipt["decisions"][0]["reason"] == "local_already_current"
    assert receipt["decisions"][0]["overlap_conflicts"]
    assert path.read_bytes() == before


def test_multiple_valid_newer_sessions_append_in_order(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR[:3])
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    newer = [date(2026, 9, 21), date(2026, 9, 22), SESSION]
    provider = pd.concat([local, _table(newer, close0=40.0)], ignore_index=True)
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=["AAA"], asof=SESSION, frames={"AAA": provider}
    )
    assert receipt["decisions"][0]["appended_dates"] == [day.isoformat() for day in newer]
    assert path.read_bytes().startswith(before)


def test_weekend_or_holiday_bar_does_not_create_pltr(tmp_path):
    import services.fetch_raw_data as fetch

    weekend = pd.concat(
        [_table(PRIOR + [SESSION], symbol="PLTR"), _table([date(2026, 9, 19)], close0=19.0)],
        ignore_index=True,
    )
    holiday = pd.concat(
        [_table([date(2026, 9, 7)], close0=7.0, symbol="PLTR"), _table([SESSION], close0=23.0)],
        ignore_index=True,
    )
    saturday_only = _table([date(2026, 9, 19)], symbol="PLTR")
    for frame in (weekend, holiday, saturday_only):
        root = tmp_path / frame["date"].iloc[0]
        receipt = fetch.apply_provider_fetch(
            root=root, tickers=["PLTR"], asof=SESSION, frames={"PLTR": frame}
        )
        assert receipt["promoted"] == []
        assert not (root / "data" / "raw" / "PLTR.csv").exists()
        assert receipt["decisions"][0]["reason"] in {
            "non_session_bar",
            "MISSING_EXPECTED_SESSION_BAR",
        }


def test_new_pltr_file_uses_its_own_frame(tmp_path):
    import services.fetch_raw_data as fetch

    _write(tmp_path / "data" / "raw" / "AAA.csv", _table(PRIOR + [SESSION]))
    pltr = _table(PRIOR + [SESSION], close0=20.0, symbol="PLTR")
    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["AAA", "PLTR"],
        asof=SESSION,
        frames={"AAA": _table(PRIOR), "PLTR": pltr},
    )
    assert receipt["promoted"] == ["PLTR"]
    assert receipt["decisions"][1]["reason"] == "exact_session"
    written = pd.read_csv(tmp_path / "data" / "raw" / "PLTR.csv")
    assert written["date"].tolist() == [day.isoformat() for day in PRIOR + [SESSION]]
    assert (written["date"] == SESSION.isoformat()).sum() == 1


def test_below_minimum_coverage_writes_nothing_including_pltr(tmp_path):
    import services.fetch_raw_data as fetch

    symbols = [f"S{i}" for i in range(10)] + ["PLTR"]
    frames = {}
    for symbol in symbols[:8]:
        _write(tmp_path / "data" / "raw" / f"{symbol}.csv", _table(PRIOR + [SESSION]))
        frames[symbol] = _table(PRIOR)
    frames["S8"] = _table(PRIOR)
    frames["S9"] = _table(PRIOR)
    frames["PLTR"] = _table([SESSION], symbol="PLTR")
    before = _sha(tmp_path / "data" / "raw" / "S0.csv")
    receipt = fetch.apply_provider_fetch(
        root=tmp_path, tickers=symbols, asof=SESSION, frames=frames
    )
    assert receipt["failure"] == "coverage_below_minimum"
    assert receipt["promoted"] == []
    assert receipt["tradable_coverage"]["ratio"] < 0.90
    assert not (tmp_path / "data" / "raw" / "PLTR.csv").exists()
    assert _sha(tmp_path / "data" / "raw" / "S0.csv") == before


def test_one_invalid_symbol_promotes_no_partial_batch(tmp_path):
    import services.fetch_raw_data as fetch

    local = _table(PRIOR)
    path = tmp_path / "data" / "raw" / "AAA.csv"
    _write(path, local)
    before = path.read_bytes()
    malformed = pd.concat([local, _table([SESSION], close0=40.0)], ignore_index=True)
    malformed.loc[malformed.index[-1], "high"] = 0.1
    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["AAA", "PLTR"],
        asof=SESSION,
        frames={"AAA": malformed, "PLTR": _table([SESSION], symbol="PLTR")},
    )
    assert receipt["promoted"] == []
    assert receipt["failure"] == "coverage_below_minimum"
    assert not (tmp_path / "data" / "raw" / "PLTR.csv").exists()
    assert path.read_bytes() == before


def test_failed_atomic_replacement_does_not_retry(tmp_path):
    import services.fetch_raw_data as fetch

    _write(tmp_path / "data" / "raw" / "AAA.csv", _table(PRIOR + [SESSION]))
    calls = {"n": 0}

    def _boom(src, dest):
        calls["n"] += 1
        raise OSError("disk")

    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["AAA", "PLTR"],
        asof=SESSION,
        frames={"AAA": _table(PRIOR), "PLTR": _table([SESSION], symbol="PLTR")},
        replace_impl=_boom,
    )
    assert calls["n"] == 1
    assert receipt["failure"] == "atomic_replacement_failed"
    assert receipt["durable_state"] == "promotion_stopped_no_retry"
    assert not (tmp_path / "data" / "raw" / "PLTR.csv").exists()


def test_partial_batch_stops_without_retry(tmp_path):
    import services.fetch_raw_data as fetch

    _write(tmp_path / "data" / "raw" / "AAA.csv", _table(PRIOR + [SESSION]))
    calls = {"n": 0}

    def _replace(src, dest):
        calls["n"] += 1
        if calls["n"] == 1:
            os.replace(src, dest)
            return
        raise OSError("disk")

    receipt = fetch.apply_provider_fetch(
        root=tmp_path,
        tickers=["AAA", "BBB", "CCC"],
        asof=SESSION,
        frames={
            "AAA": _table(PRIOR),
            "BBB": _table([SESSION], symbol="BBB"),
            "CCC": _table([SESSION], symbol="CCC"),
        },
        replace_impl=_replace,
    )
    assert calls["n"] == 2
    assert receipt["promoted"] == ["BBB"]
    assert receipt["failure"] == "atomic_replacement_failed"
    assert (tmp_path / "data" / "raw" / "BBB.csv").is_file()
    assert not (tmp_path / "data" / "raw" / "CCC.csv").exists()


def test_reconstructed_sf39a_shape_covers_63_and_is_deterministic(tmp_path, monkeypatch):
    _block_network(monkeypatch)
    import services.fetch_raw_data as fetch

    equal_count = [f"E{i:02d}" for i in range(35)]
    plus_one = [f"P{i:02d}" for i in range(27)]
    indexes = ["^DJI", "^GSPC", "^IXIC", "^VIX"]
    symbols = equal_count + plus_one + ["PLTR"] + indexes

    def _run(root: Path) -> dict[str, str]:
        frames = {}
        originals = {}
        for symbol in equal_count + indexes:
            local = _table(PRIOR, close0=11.0)
            _write(root / "data" / "raw" / f"{symbol}.csv", local)
            originals[symbol] = (root / "data" / "raw" / f"{symbol}.csv").read_bytes()
            frames[symbol] = pd.concat(
                [local.iloc[1:], _table([SESSION], close0=30.0)], ignore_index=True
            )
            assert len(frames[symbol]) == len(local)
            assert frames[symbol]["date"].iloc[-1] == SESSION.isoformat()
        for symbol in plus_one:
            local = _table(PRIOR, close0=12.0)
            _write(root / "data" / "raw" / f"{symbol}.csv", local)
            originals[symbol] = (root / "data" / "raw" / f"{symbol}.csv").read_bytes()
            frames[symbol] = pd.concat([local, _table([SESSION], close0=50.0)], ignore_index=True)
        pltr_days = _last_xnys_sessions(SESSION, 1502)
        frames["PLTR"] = _table(pltr_days, close0=21.0, symbol="PLTR")
        assert len(frames["PLTR"]) == 1502
        assert frames["PLTR"]["date"].iloc[-1] == SESSION.isoformat()
        assert int((frames["PLTR"]["date"] == SESSION.isoformat()).sum()) == 1
        assert all(day.weekday() < 5 for day in pltr_days)
        receipt = fetch.apply_provider_fetch(
            root=root, tickers=symbols, asof=SESSION, frames=frames
        )
        assert receipt["passed"] is True
        assert receipt["tradable_coverage"]["covered"] == 63
        assert receipt["tradable_coverage"]["denominator"] == 63
        pltr_decision = next(item for item in receipt["decisions"] if item["symbol"] == "PLTR")
        assert pltr_decision["reason"] == "exact_session"
        assert sum(item["reason"] == "monotonic_append" for item in receipt["decisions"]) == 66
        rolling = {
            item["symbol"]
            for item in receipt["decisions"]
            if item["symbol"] in set(equal_count + indexes)
        }
        assert rolling == set(equal_count + indexes)
        assert all(
            item["appended_dates"] == [SESSION.isoformat()]
            for item in receipt["decisions"]
            if item["symbol"] in rolling
        )
        assert "provider_would_truncate" not in {item["reason"] for item in receipt["decisions"]}
        for symbol, blob in originals.items():
            assert (root / "data" / "raw" / f"{symbol}.csv").read_bytes().startswith(blob)
        return {
            symbol: _sha(root / "data" / "raw" / f"{symbol}.csv")
            for symbol in symbols
            if (root / "data" / "raw" / f"{symbol}.csv").is_file()
        }

    assert _run(tmp_path / "a") == _run(tmp_path / "b")
