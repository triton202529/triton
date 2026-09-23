"""Current paper holdings follow positions_snapshot.csv across reruns."""

from __future__ import annotations

import csv
import hashlib
import inspect
from pathlib import Path

import pytest

from dashboard import operator_modern
from dashboard import portfolio_position_snapshot as snap

SIX = (
    {"symbol": "TSLA", "qty": "0.137141758"},
    {"symbol": "MA", "qty": "5"},
    {"symbol": "DIA", "qty": "1"},
    {"symbol": "DBA", "qty": "39"},
    {"symbol": "BITO", "qty": "1326"},
    {"symbol": "ARKK", "qty": "32"},
)
FOURTEEN = (
    {"symbol": "XLK", "qty": "7"},
    {"symbol": "VOO", "qty": "2"},
    {"symbol": "TSLA", "qty": "0.137141758"},
    {"symbol": "QQQ", "qty": "2"},
    {"symbol": "META", "qty": "1"},
    {"symbol": "MA", "qty": "5"},
    {"symbol": "INTC", "qty": "12"},
    {"symbol": "GOOGL", "qty": "4"},
    {"symbol": "DIA", "qty": "1"},
    {"symbol": "DBA", "qty": "39"},
    {"symbol": "BITO", "qty": "1326"},
    {"symbol": "ARKK", "qty": "32"},
    {"symbol": "AMD", "qty": "2"},
    {"symbol": "AAPL", "qty": "4"},
)
TS_SIX = "2026-09-22T13:42:16Z"
TS_FOURTEEN = "2026-09-22T20:02:29Z"


@pytest.fixture(autouse=True)
def _clear_holdings_cache():
    snap.cached_parse.clear()
    snap._PARSE_KEYS.clear()
    yield
    snap.cached_parse.clear()
    snap._PARSE_KEYS.clear()


def _write(path: Path, rows, snapshot_ts: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["snapshot_ts", "symbol", "ticker", "qty", "market_value"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "snapshot_ts": snapshot_ts,
                    "symbol": row["symbol"],
                    "ticker": row["symbol"],
                    "qty": row["qty"],
                    "market_value": "1",
                }
            )


def _pairs(loaded):
    return [(row["symbol"], row["qty"]) for row in snap.holdings_quantity_pairs(loaded)]


def _sorted_pairs(rows):
    return sorted((row["symbol"], row["qty"]) for row in rows)


def test_six_row_snapshot_renders_six_exact_quantities(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    loaded = snap.load_portfolio_holdings(path)
    assert loaded["ok"] is True
    assert loaded["title"] == "Current paper holdings"
    assert loaded["source_label"] == "Source: positions_snapshot.csv"
    assert loaded["as_of_label"] == f"Snapshot as of: {TS_SIX}"
    assert len(loaded["rows"]) == 6
    assert _pairs(loaded) == _sorted_pairs(SIX)
    assert dict(_pairs(loaded))["TSLA"] == "0.137141758"


def test_replacing_with_fourteen_updates_without_restart(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    first = snap.load_portfolio_holdings(path)
    _write(path, FOURTEEN, TS_FOURTEEN)
    second = snap.load_portfolio_holdings(path)
    assert len(first["rows"]) == 6
    assert len(second["rows"]) == 14
    assert _pairs(second) == _sorted_pairs(FOURTEEN)
    assert dict(_pairs(second))["TSLA"] == "0.137141758"
    assert dict(_pairs(second))["VOO"] == "2"
    assert first["identity"]["sha256"] != second["identity"]["sha256"]
    assert second["as_of_label"] == f"Snapshot as of: {TS_FOURTEEN}"


def test_same_row_count_new_quantity_changes_hash_and_cache_key(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    first = snap.load_portfolio_holdings(path)
    changed = tuple(
        {"symbol": row["symbol"], "qty": "33" if row["symbol"] == "ARKK" else row["qty"]}
        for row in SIX
    )
    _write(path, changed, TS_SIX)
    second = snap.load_portfolio_holdings(path)
    assert len(first["rows"]) == len(second["rows"]) == 6
    assert first["identity"]["sha256"] != second["identity"]["sha256"]
    assert first["identity"]["mtime_ns"] != second["identity"]["mtime_ns"]
    assert dict(_pairs(second))["ARKK"] == "33"
    assert [key[4] for key in snap._PARSE_KEYS] == [
        first["identity"]["sha256"],
        second["identity"]["sha256"],
    ]
    params = list(inspect.signature(snap.cached_parse).parameters)
    assert params == [
        "raw_bytes",
        "file_exists",
        "resolved_path",
        "mtime_ns",
        "size",
        "sha256",
        "snapshot_ts",
    ]
    assert all(not name.startswith("_") for name in params)
    parser_source = inspect.getsource(snap.cached_parse)
    assert "hash_funcs" not in parser_source
    assert snap.cached_parse.__name__ == "cached_parse"
    for banned in ("read_bytes", "read_text", ".open(", ".stat(", "Path("):
        assert banned not in parser_source
    assert [key[0] for key in snap._PARSE_KEYS] == [True, True]
    assert [key[3] for key in snap._PARSE_KEYS] == [
        first["identity"]["byte_size"],
        second["identity"]["byte_size"],
    ]


def test_unchanged_bytes_remain_cached(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    first = snap.load_portfolio_holdings(path)
    second = snap.load_portfolio_holdings(path)
    assert first["identity"]["sha256"] == second["identity"]["sha256"]
    assert _pairs(first) == _pairs(second)
    assert len(snap._PARSE_KEYS) == 1
    assert snap._PARSE_KEYS[0][0] is True
    assert snap._PARSE_KEYS[0][1] == first["identity"]["resolved_path"]
    assert snap._PARSE_KEYS[0][2] == first["identity"]["mtime_ns"]
    assert snap._PARSE_KEYS[0][3] == first["identity"]["byte_size"]
    assert snap._PARSE_KEYS[0][4] == first["identity"]["sha256"]
    assert snap._PARSE_KEYS[0][5] == TS_SIX
    assert snap._PARSE_KEYS[0][6] == first["identity"]["sha256"]
    assert all(row["snapshot_ts"] == first["snapshot_ts"] for row in first["rows"])


def test_missing_duplicate_index_and_malformed_fail_visibly(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    missing = snap.load_portfolio_holdings(path)
    assert missing["ok"] is False
    assert missing["rows"] == []
    assert missing["status"] == "missing"
    assert missing["identity"]["file_exists"] is False
    assert missing["message"].startswith("Current holdings unavailable")
    assert missing["snapshot_ts"] is None
    assert snap._PARSE_KEYS == []
    empty_sha = hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValueError):
        snap.cached_parse(b"", False, str(path), 0, 0, empty_sha, "")
    with pytest.raises(ValueError):
        snap.cached_parse(b"", True, str(path), 0, 0, empty_sha, "")
    assert snap._PARSE_KEYS == []

    _write(path, SIX + ({"symbol": "ARKK", "qty": "1"},), TS_SIX)
    duplicate = snap.load_portfolio_holdings(path)
    assert duplicate["ok"] is False
    assert duplicate["rows"] == []
    assert "duplicate" in duplicate["message"].lower()

    _write(path, SIX + ({"symbol": "^VIX", "qty": "1"},), TS_SIX)
    indexed = snap.load_portfolio_holdings(path)
    assert indexed["ok"] is False
    assert indexed["rows"] == []
    assert "index" in indexed["message"].lower()

    _write(path, ({"symbol": "AAPL", "qty": "-1"},), TS_SIX)
    negative = snap.load_portfolio_holdings(path)
    assert negative["ok"] is False
    assert negative["rows"] == []

    path.write_text("snapshot_ts,symbol,qty\nyesterday,AAPL,1\n", encoding="utf-8")
    bad_ts = snap.load_portfolio_holdings(path)
    assert bad_ts["ok"] is False
    assert bad_ts["rows"] == []
    assert bad_ts["message"].startswith("Current holdings unavailable")
    assert snap._PARSE_KEYS == []


def test_unstable_read_does_not_show_rows(tmp_path, monkeypatch):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    stamps = [(1, 10), (2, 11), (3, 12), (4, 13)]

    def _moving(_path):
        return stamps.pop(0)

    monkeypatch.setattr(snap, "_stamp", _moving)
    loaded = snap.load_portfolio_holdings(path)
    assert loaded["ok"] is False
    assert loaded["status"] == "unstable"
    assert loaded["rows"] == []
    assert loaded["message"].startswith("Current holdings unavailable")
    assert snap._PARSE_KEYS == []


def test_cached_parser_rejects_bytes_for_a_different_hash(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    raw = path.read_bytes()
    with pytest.raises(ValueError):
        snap.cached_parse(raw, True, str(path.resolve()), 1, len(raw), "0" * 64, TS_SIX)


def _other_snapshot_bytes() -> bytes:
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(
        handle,
        fieldnames=["snapshot_ts", "symbol", "ticker", "qty", "market_value"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "snapshot_ts": "2026-09-22T18:00:00Z",
            "symbol": "AAPL",
            "ticker": "AAPL",
            "qty": "9",
            "market_value": "1",
        }
    )
    return handle.getvalue().encode("utf-8")


def test_replacement_between_stat_and_read_is_not_shown_as_the_old_book(tmp_path, monkeypatch):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    prior = path.read_bytes()
    prior_sha = hashlib.sha256(prior).hexdigest()
    replacement = _other_snapshot_bytes()
    state = {"reads": 0}

    def _read(target):
        state["reads"] += 1
        if state["reads"] == 1:
            target.write_bytes(replacement)
        return target.read_bytes()

    monkeypatch.setattr(snap, "_read_file_bytes", _read)
    loaded = snap.load_portfolio_holdings(path)
    assert state["reads"] == 2
    assert loaded["ok"] is True
    assert loaded["identity"]["file_exists"] is True
    assert _pairs(loaded) == [("AAPL", "9")]
    assert loaded["snapshot_ts"] == "2026-09-22T18:00:00Z"
    assert loaded["snapshot_ts"] != TS_SIX
    assert "TSLA" not in dict(_pairs(loaded))
    assert loaded["identity"]["sha256"] == hashlib.sha256(replacement).hexdigest()
    assert loaded["identity"]["sha256"] != prior_sha
    assert snap._PARSE_KEYS[-1][5] == loaded["snapshot_ts"]
    assert snap._PARSE_KEYS[-1][6] == loaded["identity"]["sha256"]


def test_replacement_after_read_before_post_stat_discards_rows(tmp_path, monkeypatch):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    six_raw = path.read_bytes()
    fourteen_path = tmp_path / "fourteen.csv"
    _write(fourteen_path, FOURTEEN, TS_FOURTEEN)
    fourteen_raw = fourteen_path.read_bytes()
    reads = {"n": 0}
    pres = {"n": 0}
    posts = {"n": 0}

    def _before(_path):
        pres["n"] += 1
        if pres["n"] == 1:
            return (100, len(six_raw))
        return (200, len(fourteen_raw))

    def _read(_path):
        reads["n"] += 1
        if reads["n"] == 1:
            return six_raw
        return fourteen_raw

    def _after(_path):
        posts["n"] += 1
        if posts["n"] == 1:
            return (101, len(fourteen_raw))
        return (201, len(six_raw))

    monkeypatch.setattr(snap, "_stamp_before", _before)
    monkeypatch.setattr(snap, "_read_file_bytes", _read)
    monkeypatch.setattr(snap, "_stamp_after", _after)
    loaded = snap.load_portfolio_holdings(path)
    assert reads["n"] == 2
    assert pres["n"] == 2
    assert posts["n"] == 2
    assert loaded["ok"] is False
    assert loaded["status"] == "unstable"
    assert loaded["rows"] == []
    assert loaded["snapshot_ts"] is None
    assert loaded["as_of_label"] is None
    assert loaded["message"].startswith("Current holdings unavailable")
    assert snap._PARSE_KEYS == []


def test_change_after_stable_read_keeps_captured_bytes(tmp_path, monkeypatch):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    six_raw = path.read_bytes()
    fourteen_path = tmp_path / "fourteen.csv"
    _write(fourteen_path, FOURTEEN, TS_FOURTEEN)
    fourteen_raw = fourteen_path.read_bytes()
    phase = {"n": 0}

    def _before(_path):
        phase["n"] += 1
        if phase["n"] == 1:
            return (100, len(six_raw))
        return (200, len(fourteen_raw))

    def _read(_path):
        phase["n"] += 1
        if phase["n"] == 2:
            path.write_bytes(fourteen_raw)
            return six_raw
        return fourteen_raw

    def _after(_path):
        phase["n"] += 1
        if phase["n"] == 3:
            return (100, len(six_raw))
        return (200, len(fourteen_raw))

    monkeypatch.setattr(snap, "_stamp_before", _before)
    monkeypatch.setattr(snap, "_read_file_bytes", _read)
    monkeypatch.setattr(snap, "_stamp_after", _after)
    first = snap.load_portfolio_holdings(path)
    assert phase["n"] == 3
    assert path.read_bytes() == fourteen_raw
    assert first["ok"] is True
    assert first["snapshot_ts"] == TS_SIX
    assert first["identity"]["sha256"] == hashlib.sha256(six_raw).hexdigest()
    assert first["identity"]["byte_size"] == len(six_raw)
    assert _pairs(first) == _sorted_pairs(SIX)
    assert "VOO" not in dict(_pairs(first))
    assert all(row["snapshot_ts"] == TS_SIX for row in first["rows"])
    second = snap.load_portfolio_holdings(path)
    assert phase["n"] == 6
    assert second["ok"] is True
    assert second["snapshot_ts"] == TS_FOURTEEN
    assert second["identity"]["sha256"] == hashlib.sha256(fourteen_raw).hexdigest()
    assert _pairs(second) == _sorted_pairs(FOURTEEN)
    assert dict(_pairs(second))["VOO"] == "2"
    assert first["identity"]["sha256"] != second["identity"]["sha256"]


def test_same_mtime_and_size_with_different_sha_invalidates(tmp_path, monkeypatch):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    first_raw = path.read_bytes()
    changed = tuple(
        {"symbol": row["symbol"], "qty": "99" if row["symbol"] == "ARKK" else row["qty"]}
        for row in SIX
    )
    changed_path = tmp_path / "changed.csv"
    _write(changed_path, changed, TS_SIX)
    second_raw = changed_path.read_bytes()
    assert len(first_raw) == len(second_raw)
    assert hashlib.sha256(first_raw).hexdigest() != hashlib.sha256(second_raw).hexdigest()
    payloads = {"n": 0, "raw": (first_raw, second_raw)}

    def _read(_target):
        raw = payloads["raw"][payloads["n"]]
        payloads["n"] += 1
        return raw

    def _pinned(_target):
        return (424242, len(first_raw))

    monkeypatch.setattr(snap, "_read_file_bytes", _read)
    monkeypatch.setattr(snap, "_stamp", _pinned)
    first = snap.load_portfolio_holdings(path)
    second = snap.load_portfolio_holdings(path)
    assert first["identity"]["mtime_ns"] == second["identity"]["mtime_ns"] == 424242
    assert first["identity"]["byte_size"] == second["identity"]["byte_size"] == len(first_raw)
    assert first["identity"]["sha256"] != second["identity"]["sha256"]
    assert dict(_pairs(second))["ARKK"] == "99"
    assert second["snapshot_ts"] == TS_SIX
    assert [key[4] for key in snap._PARSE_KEYS] == [
        first["identity"]["sha256"],
        second["identity"]["sha256"],
    ]
    assert [key[6] for key in snap._PARSE_KEYS] == [
        hashlib.sha256(first_raw).hexdigest(),
        hashlib.sha256(second_raw).hexdigest(),
    ]
    snap.cached_parse.clear()
    snap._PARSE_KEYS.clear()
    identity = (
        True,
        first["identity"]["resolved_path"],
        424242,
        len(first_raw),
        first["identity"]["sha256"],
        TS_SIX,
    )
    cached_rows = snap.cached_parse(first_raw, *identity)
    assert [(row["symbol"], row["qty"]) for row in cached_rows] == _sorted_pairs(SIX)
    with pytest.raises(ValueError):
        snap.cached_parse(second_raw, *identity)
    assert len(snap._PARSE_KEYS) == 1
    assert snap._PARSE_KEYS[0][6] == hashlib.sha256(first_raw).hexdigest()


def test_displayed_timestamp_matches_the_parsed_bytes(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    loaded = snap.load_portfolio_holdings(path)
    raw = path.read_bytes()
    assert loaded["identity"]["file_exists"] is True
    assert loaded["identity"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert loaded["snapshot_ts"] == TS_SIX
    assert TS_SIX.encode("utf-8") in raw
    assert all(row["snapshot_ts"] == loaded["snapshot_ts"] for row in loaded["rows"])
    snap.cached_parse.clear()
    snap._PARSE_KEYS.clear()
    fourteen_path = tmp_path / "fourteen.csv"
    _write(fourteen_path, FOURTEEN, TS_FOURTEEN)
    path.write_bytes(fourteen_path.read_bytes())

    def _boom(*_args, **_kwargs):
        raise AssertionError("cached parser reopened the path")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Path, "read_bytes", _boom)
    monkeypatch.setattr(Path, "read_text", _boom)
    monkeypatch.setattr(Path, "open", _boom)
    monkeypatch.setattr(Path, "stat", _boom)
    try:
        rows = snap.cached_parse(
            raw,
            True,
            loaded["identity"]["resolved_path"],
            loaded["identity"]["mtime_ns"],
            loaded["identity"]["byte_size"],
            loaded["identity"]["sha256"],
            loaded["snapshot_ts"],
        )
    finally:
        monkeypatch.undo()
    assert [(row["symbol"], row["qty"]) for row in rows] == _sorted_pairs(SIX)
    assert dict((row["symbol"], row["qty"]) for row in rows)["TSLA"] == "0.137141758"
    assert "VOO" not in {row["symbol"] for row in rows}
    assert snap._PARSE_KEYS[-1][6] == hashlib.sha256(raw).hexdigest()
    assert snap.cached_parse.__name__ == "cached_parse"


def test_empty_bytes_and_missing_file_fail_closed(tmp_path):
    path = tmp_path / "positions_snapshot.csv"
    missing = snap.load_portfolio_holdings(path)
    assert missing["ok"] is False
    assert missing["rows"] == []
    assert missing["identity"]["file_exists"] is False
    assert missing["snapshot_ts"] is None
    assert missing["message"].startswith("Current holdings unavailable")
    empty_sha = hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValueError):
        snap.cached_parse(b"", False, str(path), 0, 0, empty_sha, "")
    with pytest.raises(ValueError):
        snap.cached_parse(b"", True, str(path), 0, 0, empty_sha, "")
    assert snap._PARSE_KEYS == []
    path.write_bytes(b"")
    empty_file = snap.load_portfolio_holdings(path)
    assert empty_file["ok"] is False
    assert empty_file["rows"] == []
    assert empty_file["snapshot_ts"] is None
    assert snap._PARSE_KEYS == []


def test_certified_metrics_stay_outside_the_current_table():
    portfolio = inspect.getsource(operator_modern.render_portfolio_page)
    current = inspect.getsource(operator_modern._render_canonical_holdings)
    assert "_portfolio_metrics(" in portfolio
    assert 'st.expander("Certified close holdings"' in portfolio
    assert "_certified_holdings_table(" in portfolio
    assert snap.CURRENT_HOLDINGS_TITLE == "Current paper holdings"
    assert snap.CURRENT_HOLDINGS_SOURCE == "Source: positions_snapshot.csv"
    assert "CURRENT_HOLDINGS_TITLE" in current
    assert "CURRENT_HOLDINGS_SOURCE" in current
    assert "as_of_label" in current
    assert "certified" not in current.lower()
    assert "_certified_holdings_table" not in current


def test_loader_does_not_write_call_broker_or_refresh(tmp_path, monkeypatch):
    path = tmp_path / "positions_snapshot.csv"
    _write(path, SIX, TS_SIX)
    source = Path(snap.__file__).read_text(encoding="utf-8")
    page = inspect.getsource(operator_modern._render_canonical_holdings)
    for text in (source, page):
        lowered = text.lower()
        assert "broker_alpaca" not in lowered
        assert "submit_order" not in lowered
        assert "cache_data.clear" not in lowered
        assert 'st.button("refresh data"' not in lowered
        assert "session_state" not in lowered
        assert "paper_arm" not in lowered
        assert "start_triton" not in lowered
    writes = []
    original = Path.write_bytes

    def _guard(self, data):
        if Path(self) == path:
            writes.append(self)
            raise AssertionError("snapshot write")
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", _guard)
    loaded = snap.load_portfolio_holdings(path)
    assert loaded["ok"] is True
    assert writes == []


def test_streamlit_rerun_replaces_six_with_fourteen(tmp_path):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "positions_snapshot.csv"
    app = tmp_path / "holdings_app.py"
    _write(path, SIX, TS_SIX)
    app.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import streamlit as st",
                "from dashboard.portfolio_position_snapshot import load_portfolio_holdings",
                f"loaded = load_portfolio_holdings(Path({str(path)!r}))",
                "st.write(loaded['title'])",
                "st.write(loaded['source_label'])",
                "if not loaded['ok']:",
                "    st.write(loaded['message'])",
                "else:",
                "    st.write(loaded['as_of_label'])",
                "    for row in loaded['rows']:",
                "        st.write(row['symbol'] + '=' + row['qty'])",
            ]
        ),
        encoding="utf-8",
    )
    first = AppTest.from_file(str(app)).run()
    first_text = "\n".join(item.value for item in first.markdown)
    assert "Current paper holdings" in first_text
    assert "Source: positions_snapshot.csv" in first_text
    assert f"Snapshot as of: {TS_SIX}" in first_text
    assert "TSLA=0.137141758" in first_text
    assert first_text.count("=") == 6
    _write(path, FOURTEEN, TS_FOURTEEN)
    second = AppTest.from_file(str(app)).run()
    second_text = "\n".join(item.value for item in second.markdown)
    assert f"Snapshot as of: {TS_FOURTEEN}" in second_text
    assert "VOO=2" in second_text
    assert "TSLA=0.137141758" in second_text
    assert second_text.count("=") == 14
