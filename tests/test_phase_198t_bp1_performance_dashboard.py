"""BP1 isolated Performance dashboard. Presentation only. No production writes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from dashboard.operator_modern import UI_A_RENDERERS
from dashboard.operator_nav import all_destinations
from dashboard.performance_presentation import (
    BUILDING_HISTORY_NOTE,
    EARLIEST_FIRST_RETURN_DATE,
    EXPECTED_ACCOUNTING_SHA256,
    EXPECTED_EVENT_COUNT,
    EXPECTED_SESSION_CLOSE_DATE,
    FAIL_CLOSED_TITLE,
    GROSS_PROVISIONAL_LABEL,
    KIND_CERTIFIED,
    KIND_OPENING,
    NET_PL_NOTE,
    PAGE_HEADING,
    PAGE_SUPPORTING_COPY,
    SECOND_CLOSE_NOTE,
    STATUS_BUILDING,
    UNAVAILABLE_LABEL,
    PerformanceDataUnavailable,
    file_sha256,
    load_performance_view,
    overview_performance_summary,
)
from dashboard.preview_data import preview_capture_label
from tests.fixtures.bp1.two_close import TEST_ONLY_LABEL, write_two_close_ledger

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")
FROZEN_LEDGER = ISO_ROOT / "tests" / "fixtures" / "bp1" / "performance_accounting_events.jsonl"
BM6A_ROOT = Path(r"C:\Users\akimw\triton-bm6a-ui-isolated")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_ledger(dest_root: Path) -> Path:
    dest = dest_root / "data" / "results" / "performance_accounting_events.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(FROZEN_LEDGER.read_bytes())
    return dest


def test_performance_nav_appears_after_portfolio():
    pages = all_destinations()
    assert pages.index("Performance") == pages.index("Portfolio") + 1
    assert "Performance" in UI_A_RENDERERS
    assert UI_A_RENDERERS["Performance"].__name__ == "render_performance_page"


def test_frozen_ledger_hash_and_event_count():
    assert FROZEN_LEDGER.is_file()
    assert file_sha256(FROZEN_LEDGER) == EXPECTED_ACCOUNTING_SHA256
    assert _sha256(FROZEN_LEDGER) == EXPECTED_ACCOUNTING_SHA256
    lines = [ln for ln in FROZEN_LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == EXPECTED_EVENT_COUNT


def test_displayed_numbers_come_from_copied_ledger(tmp_path: Path):
    ledger = _copy_ledger(tmp_path)
    view = load_performance_view(
        root=tmp_path,
        expected_sha256=EXPECTED_ACCOUNTING_SHA256,
        expected_event_count=EXPECTED_EVENT_COUNT,
    )
    assert view["ledger_sha256"] == EXPECTED_ACCOUNTING_SHA256
    assert view["event_count"] == EXPECTED_EVENT_COUNT
    assert view["cutover_count"] == 1
    assert view["certified_close_count"] == 1
    assert view["latest_session_date"] == EXPECTED_SESSION_CLOSE_DATE
    assert view["latest_equity"] == "$92,006.00"
    assert view["cash"] == "$66,447.22"
    assert view["market_value"] == "$25,558.78"
    assert view["accounting_status"] == STATUS_BUILDING
    assert view["unrealized_pl"] == "+$3,607.22"
    assert view["building_note"] == BUILDING_HISTORY_NOTE
    assert view["latest_equity_raw"] == "92006.00"
    assert view["ledger_path"] == str(ledger)
    assert view["uses_portfolio_history"] is False


def test_one_cutover_plus_one_close_checkpoints(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    kinds = [p["kind"] for p in view["checkpoints"]]
    assert kinds == [KIND_OPENING, KIND_CERTIFIED]
    assert view["checkpoints"][0]["timestamp"] == "2026-09-10T23:22:50+00:00"
    assert view["checkpoints"][1]["timestamp"] == "2026-09-11T22:47:19+00:00"
    assert view["checkpoints"][0]["equity"] == "91871.49"
    assert view["checkpoints"][1]["equity"] == "92006.00"
    assert view["checkpoints"][0]["not_a_return"] is True
    assert view["checkpoints"][0]["axis_label"] == "Sep 10 · Opening checkpoint"
    assert view["checkpoints"][1]["axis_label"] == "Sep 11 · Certified close"
    assert view["checkpoint_axis_labels"] == [
        "Sep 10 · Opening checkpoint",
        "Sep 11 · Certified close",
    ]
    assert view["chart_tone"] == "positive"
    assert view["cutover_is_not_a_return"] is True


def test_twr_and_multi_close_metrics_unavailable(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    for key in (
        "time_weighted_return",
        "maximum_drawdown",
        "spy_comparison",
        "qqq_comparison",
        "net_realized_pl",
        "win_rate",
        "average_winner",
        "average_loser",
        "profit_factor",
        "exposure_adjusted_return",
        "decision_slippage",
        "historical_return",
    ):
        assert view[key] == UNAVAILABLE_LABEL
        assert view[key] != "0"
        assert view[key] != "$0.00"
        assert view[key] != "0.00"
    assert view["shared_unavailable_note"] == SECOND_CLOSE_NOTE
    assert view["earliest_first_return_date"] == EARLIEST_FIRST_RETURN_DATE
    assert view["fees"] == "Pending fee reconciliation"
    assert view["fees"] != "$0.00"


def test_dba_proceeds_are_not_profit(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    dba = next(row for row in view["activity"] if row["symbol"] == "DBA")
    assert dba["side"] == "SELL"
    assert dba["quantity"] == "12"
    assert dba["fill_price"] == "$29.03"
    assert dba["proceeds"] == "$348.36"
    assert dba["proceeds_is_not_profit"] is True
    assert dba["position"] == "51 → 39"
    assert dba["gross_result"] == "$26.22"
    assert dba["gross_label"] == GROSS_PROVISIONAL_LABEL
    assert dba["net_realized_pl"] == UNAVAILABLE_LABEL
    assert dba["broker_order_id"] == "528bdc13-e2b5-44da-895f-0e252da3beff"
    assert NET_PL_NOTE


def test_spy_qqq_same_september_11_date(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    dates = {row["session_date"] for row in view["benchmarks"]}
    assert dates == {EXPECTED_SESSION_CLOSE_DATE}
    spy = next(row for row in view["benchmarks"] if row["symbol"] == "SPY")
    qqq = next(row for row in view["benchmarks"] if row["symbol"] == "QQQ")
    assert spy["adjusted_close"] == "$764.29"
    assert qqq["adjusted_close"] == "$714.88"
    assert spy["methodology"] == "Total return evidence"
    assert qqq["methodology"] == "Total return evidence"
    assert spy["comparison"] == UNAVAILABLE_LABEL


def test_planning_rows_never_appear_as_fills(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert all(row["side"] in {"BUY", "SELL"} for row in view["activity"])
    assert all("plan" not in str(row.get("logical_key") or "").lower() for row in view["activity"])
    src = (ISO_ROOT / "dashboard" / "performance_presentation.py").read_text(encoding="utf-8")
    assert "paper_execution_audit" not in src


def test_missing_or_tampered_ledger_fails_closed(tmp_path: Path):
    with pytest.raises(PerformanceDataUnavailable):
        load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    ledger = _copy_ledger(tmp_path)
    ledger.write_text('{"event_id":"synthetic"}\n', encoding="utf-8")
    with pytest.raises(PerformanceDataUnavailable):
        load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    _copy_ledger(tmp_path)
    with pytest.raises(PerformanceDataUnavailable):
        load_performance_view(root=tmp_path, expected_sha256="0" * 64)
    from dashboard.performance_page import FAIL_CLOSED_TITLE as PAGE_FAIL

    assert PAGE_FAIL == FAIL_CLOSED_TITLE
    pres = (ISO_ROOT / "dashboard" / "performance_presentation.py").read_text(encoding="utf-8")
    assert "never opens portfolio_history.csv" in pres


def test_no_synthetic_fallback_or_portfolio_history_authority():
    pres = (ISO_ROOT / "dashboard" / "performance_presentation.py").read_text(encoding="utf-8")
    page = (ISO_ROOT / "dashboard" / "performance_page.py").read_text(encoding="utf-8")
    assert "portfolio_history.csv as a performance authority" in pres
    assert "reduce_events" in pres
    assert "alpaca" not in pres.lower()
    assert "alpaca" not in page.lower()
    assert "publish_session_close" not in pres
    assert "publish_ledger_atomic" not in pres
    assert "publish_session_close" not in page
    from dashboard.performance_page import PAGE_HEADING as PAGE_H
    from dashboard.performance_page import PAGE_SUPPORTING_COPY as PAGE_COPY

    assert PAGE_H == PAGE_HEADING
    assert PAGE_COPY == PAGE_SUPPORTING_COPY


def test_import_and_view_load_create_no_files(tmp_path: Path):
    before = {p: p.stat().st_mtime for p in PROD_ROOT.joinpath("data").rglob("*") if p.is_file()}
    iso_before = {p: p.stat().st_mtime for p in ISO_ROOT.joinpath("data").rglob("*") if p.is_file()}
    import dashboard.performance_page as perf_page
    import dashboard.performance_presentation as perf_pres

    assert perf_page.PAGE_HEADING == PAGE_HEADING
    assert perf_pres.EXPECTED_ACCOUNTING_SHA256 == EXPECTED_ACCOUNTING_SHA256
    _copy_ledger(tmp_path)
    load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    after = {p: p.stat().st_mtime for p in PROD_ROOT.joinpath("data").rglob("*") if p.is_file()}
    iso_after = {p: p.stat().st_mtime for p in ISO_ROOT.joinpath("data").rglob("*") if p.is_file()}
    assert before == after
    assert iso_before == iso_after


def test_no_broker_request_or_accounting_append(tmp_path: Path):
    _copy_ledger(tmp_path)

    def _fail(*_a, **_k):
        raise AssertionError("broker or write invoked")

    with patch("urllib.request.urlopen", side_effect=_fail):
        with patch("services.performance_accounting.append_events", create=True, side_effect=_fail):
            view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert view["certified_close_count"] == 1


def test_isolated_preview_refuses_production_paths(tmp_path: Path):
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "ui_a_isolation_marker.txt").write_text("bp1\n", encoding="utf-8")
    with pytest.raises(PerformanceDataUnavailable):
        load_performance_view(
            root=tmp_path,
            events_path=PROD_ROOT / "data" / "results" / "performance_accounting_events.jsonl",
            expected_sha256=EXPECTED_ACCOUNTING_SHA256,
        )


def test_buy_sell_and_start_contracts_unchanged():
    sell = (ISO_ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    buy = (ISO_ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    assert 'key="paper_sell_arm_reviewed"' in sell
    assert 'key="paper_sell_arm_confirm"' in sell
    assert "CONFIRM PAPER SELLS" in sell
    assert 'key="paper_buy_add_arm_submit"' in buy
    assert "ARM PAPER ENTRIES" in buy
    assert "REVOKE PAPER ENTRIES" in buy
    assert "_render_start_controls" in modern
    assert "render_paper_buy_add_arm_controls" in modern
    assert "render_paper_sell_arm_controls" in modern


def test_preview_banner_uses_evidence_timestamp():
    label = preview_capture_label("2026-09-11T22:47:19+00:00")
    assert label.startswith("Preview data captured at ")
    assert "2026-09-11" in label
    assert "ET" in label


def test_two_close_test_only_fixture_renders_future_metrics(tmp_path: Path):
    dest = tmp_path / "two_certified_closes_TEST_ONLY.jsonl"
    info = write_two_close_ledger(FROZEN_LEDGER, dest)
    assert info["label"] == TEST_ONLY_LABEL
    view = load_performance_view(root=tmp_path, events_path=dest)
    assert view["certified_close_count"] == 2
    assert view["time_weighted_return"] != UNAVAILABLE_LABEL
    assert view["time_weighted_return"] != "$0.00" or view["certified_close_count"] == 2
    assert any(
        p["session_date"] == "2026-09-14"
        for p in view["checkpoints"]
        if p["kind"] == KIND_CERTIFIED
    )
    assert dest.name.endswith("TEST_ONLY.jsonl")


def test_overview_summary_is_compact(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    summary = overview_performance_summary(view)
    assert summary["latest_certified_equity"] == "$92,006.00"
    assert summary["certified_close_count"] == 1
    assert summary["status"] == STATUS_BUILDING
    assert "Performance" in summary["guidance"]
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    assert "render_overview_performance_summary" in modern


def test_holdings_include_dba_weight(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    symbols = [row["symbol"] for row in view["holdings"]]
    assert symbols == ["ARKK", "BITO", "DBA", "DIA", "GE", "GS", "MA", "TSLA"]
    dba = next(row for row in view["holdings"] if row["symbol"] == "DBA")
    assert dba["quantity"] == "39"
    assert dba["weight"].endswith("%")


def test_page_source_keeps_raw_evidence_collapsed():
    page = (ISO_ROOT / "dashboard" / "performance_page.py").read_text(encoding="utf-8")
    assert 'st.expander("Technical details", expanded=False)' in page
    assert 'st.expander("More performance metrics", expanded=False)' in page
    assert "st.plotly_chart" in page
    assert '"type": "category"' in page
    assert '"type": "date"' not in page


def test_humanized_visible_labels_have_no_underscores(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    visible = [
        view["accounting_status"],
        view["time_weighted_return"],
        view["fees"],
        view["activity"][0]["gross_label"],
        view["benchmarks"][0]["methodology"],
    ]
    for item in visible:
        assert "_" not in str(item)


def test_bm6a_worktree_untouched_by_this_module():
    assert BM6A_ROOT.is_dir()
    before = (BM6A_ROOT / "dashboard" / "operator_modern.py").stat().st_mtime
    import dashboard.performance_presentation as _

    del _
    assert (BM6A_ROOT / "dashboard" / "operator_modern.py").stat().st_mtime == before
