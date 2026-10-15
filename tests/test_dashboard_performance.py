"""Dashboard reads stay fresh when files change and do not repeat a Portfolio render."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from dashboard import operator_modern
from dashboard.operator_read_cache import (
    clear_dashboard_read_cache,
    read_jsonl_tail,
    read_local_bytes,
    source_stamp,
)
from services.manual_portfolio_exit import (
    build_manual_exit_review,
    provisional_gain_loss_text,
)
from services.operator_console_status import build_portfolio_summary
from services.portfolio_effective_action import (
    BASIS_UNPROVEN_HOLD_STATUS,
    resolve_portfolio_effective_actions,
)
from services.runtime_process_detector import _find_loop_processes, open_process_scan_scope

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "ledger" / "ledger.parquet",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
)
SNAPSHOT = (
    "symbol,qty,snapshot_ts,current_price,avg_entry_price\n"
    "VOO,1,2026-10-14T20:00:00Z,10,9\n"
    "XLK,2,2026-10-14T20:00:00Z,20,18\n"
    "GOOGL,1,2026-10-14T20:00:00Z,100,80\n"
)


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    before = {path: _fingerprint(path) for path in PRODUCTION}
    yield
    after = {path: _fingerprint(path) for path in PRODUCTION}
    assert after == before


@pytest.fixture(autouse=True)
def _clear_read_cache():
    clear_dashboard_read_cache()
    yield
    clear_dashboard_read_cache()


def _snapshot_root(tmp_path: Path) -> Path:
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    (results / "positions_snapshot.csv").write_text(SNAPSHOT, encoding="utf-8")
    (results / "portfolio_history.csv").write_text(
        "date,total_value\n2026-10-14,1\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "data" / "ledger"
    ledger.mkdir()
    (ledger / "ledger.parquet").write_bytes(b"not-read")
    (results / "paper_order_events.jsonl").write_text('{"event":"old"}\n', encoding="utf-8")
    return tmp_path


def test_repeated_reads_reuse_the_cache_until_the_stamp_changes(tmp_path, monkeypatch):
    path = tmp_path / "local.json"
    path.write_text('{"value":1}', encoding="utf-8")
    reads = {"n": 0}
    real = Path.read_bytes

    def counting(self, *args, **kwargs):
        if self.resolve() == path.resolve():
            reads["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting)
    stamp = source_stamp(path)
    assert read_local_bytes(str(path.resolve()), stamp[0], stamp[1]) == b'{"value":1}'
    assert read_local_bytes(str(path.resolve()), stamp[0], stamp[1]) == b'{"value":1}'
    assert reads["n"] == 1
    path.write_text('{"value":22}', encoding="utf-8")
    changed = source_stamp(path)
    assert changed != stamp
    assert read_local_bytes(str(path.resolve()), changed[0], changed[1]) == b'{"value":22}'
    assert reads["n"] == 2


def test_refresh_data_invalidates_the_dashboard_cache(tmp_path, monkeypatch):
    path = tmp_path / "local.json"
    path.write_text("[]", encoding="utf-8")
    reads = {"n": 0}
    real = Path.read_bytes

    def counting(self, *args, **kwargs):
        if self.resolve() == path.resolve():
            reads["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting)
    stamp = source_stamp(path)
    read_local_bytes(str(path.resolve()), stamp[0], stamp[1])
    read_local_bytes(str(path.resolve()), stamp[0], stamp[1])
    clear_dashboard_read_cache()
    read_local_bytes(str(path.resolve()), stamp[0], stamp[1])
    assert reads["n"] == 2
    source = (ROOT / "view_results.py").read_text(encoding="utf-8")
    button = source.index('st.button("Refresh data"')
    assert "clear_dashboard_read_cache" in source[button : button + 500]


def test_jsonl_summary_reads_only_the_requested_tail(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"i": index}) for index in range(30)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reads = {"n": 0}
    real = Path.read_bytes

    def counting(self, *args, **kwargs):
        if self.resolve() == path.resolve():
            reads["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting)
    stamp = source_stamp(path)
    rows = read_jsonl_tail(str(path.resolve()), stamp[0], stamp[1], 4)
    assert [row["i"] for row in rows] == [26, 27, 28, 29]
    assert read_jsonl_tail(str(path.resolve()), stamp[0], stamp[1], 4) == rows
    assert reads["n"] == 1


def test_portfolio_loading_does_not_require_unrelated_history(tmp_path, monkeypatch):
    root = _snapshot_root(tmp_path)
    touched: list[str] = []
    real_read = Path.read_bytes
    real_open = Path.open

    def track_read(self, *args, **kwargs):
        touched.append(str(self))
        return real_read(self, *args, **kwargs)

    def track_open(self, *args, **kwargs):
        touched.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", track_read)
    monkeypatch.setattr(Path, "open", track_open)
    data = operator_modern.load_portfolio_page_data(root, blocked_symbols=["VOO", "XLK"])
    assert [row["symbol"] for row in data["rows"]] == ["GOOGL", "VOO", "XLK"]
    blob = "\n".join(touched)
    assert "portfolio_history.csv" not in blob
    assert "paper_order_events.jsonl" not in blob
    assert "ledger.parquet" not in blob
    page = inspect.getsource(operator_modern.render_portfolio_page)
    loader = inspect.getsource(operator_modern.load_portfolio_page_data)
    assert "collect_activity_events" not in page
    assert "operator_reports" not in page
    assert "portfolio_history" not in loader
    assert "paper_order_events" not in loader
    assert "ledger.parquet" not in loader


def test_portfolio_hold_decision_is_fresh_on_every_load(tmp_path, monkeypatch):
    root = _snapshot_root(tmp_path)
    calls = {"n": 0}

    def fake(blocked_symbols=None):
        calls["n"] += 1
        assert blocked_symbols is None
        return {"VOO", "XLK"}

    monkeypatch.setattr("services.manual_portfolio_exit.blocked_manual_exit_symbols", fake)
    first = operator_modern.load_portfolio_page_data(root)
    second = operator_modern.load_portfolio_page_data(root)
    assert calls["n"] == 2
    assert first["blocked_symbols"] == ["VOO", "XLK"]
    assert second["blocked_symbols"] == ["VOO", "XLK"]


def test_portfolio_summary_skips_history_until_requested(tmp_path, monkeypatch):
    root = _snapshot_root(tmp_path)
    seen: list[str] = []
    import pandas as pd

    real = pd.read_csv

    def wrapped(path, *args, **kwargs):
        seen.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", wrapped)
    build_portfolio_summary(root=root, include_history=False)
    assert not any("portfolio_history.csv" in item for item in seen)
    build_portfolio_summary(root=root, include_history=True)
    assert any("portfolio_history.csv" in item for item in seen)


def test_shared_block_preserves_effective_action_and_manual_exit():
    rows = [
        {
            "symbol": symbol,
            "qty": "1",
            "current_position_qty": "1",
            "current_price": "10",
            "avg_entry_price": "9",
        }
        for symbol in ("VOO", "XLK", "GOOGL", "AAPL", "MSFT")
    ]
    blocked = ["VOO", "XLK"]
    decisions = resolve_portfolio_effective_actions(
        rows,
        {row["symbol"]: "HOLD" for row in rows},
        blocked_symbols=blocked,
        authority={},
    )
    voo = build_manual_exit_review(rows[0], blocked_symbols=blocked)
    googl = build_manual_exit_review(
        {
            "symbol": "GOOGL",
            "qty": "1",
            "current_price": "100",
            "avg_entry_price": "80",
        },
        blocked_symbols=blocked,
    )
    assert decisions["VOO"]["effective_action"] == "HOLD"
    assert decisions["VOO"]["operator_status"] == BASIS_UNPROVEN_HOLD_STATUS
    assert decisions["GOOGL"]["effective_action"] == "HOLD"
    assert voo["eligible"] is False
    assert voo["unavailable_reason"] == "Basis-unproven hold"
    assert googl["eligible"] is True
    text = provisional_gain_loss_text(googl).lower()
    assert "gross provisional" in text
    for banned in ("realized", "certified", "guaranteed", " net "):
        assert banned not in text
    holdings = inspect.getsource(operator_modern._render_canonical_holdings)
    exits = inspect.getsource(operator_modern._render_manual_portfolio_exits)
    assert "resolve_portfolio_effective_actions" in holdings
    assert "session_state" not in holdings
    assert "submit_order" not in holdings
    assert "Confirm paper exit" in exits
    assert "blocked_symbols" in exits


def test_process_scan_runs_once_per_dashboard_render_and_again_on_the_next(monkeypatch):
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1

        class Result:
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("services.runtime_process_detector.sys.platform", "win32")
    monkeypatch.setattr("services.runtime_process_detector.subprocess.run", fake_run)
    scope = open_process_scan_scope("render-1")
    try:
        _find_loop_processes("run_watchdog_loop.ps1")
        _find_loop_processes("run_continuous_loop.ps1")
        _find_loop_processes("run_watchdog_loop.ps1")
        assert calls["n"] == 1
    finally:
        scope.close()
    scope = open_process_scan_scope("render-2")
    try:
        _find_loop_processes("run_continuous_loop.ps1")
        assert calls["n"] == 2
    finally:
        scope.close()
    _find_loop_processes("run_watchdog_loop.ps1")
    _find_loop_processes("run_watchdog_loop.ps1")
    assert calls["n"] == 4
