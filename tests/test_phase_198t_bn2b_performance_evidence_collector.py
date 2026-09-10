"""Phase 198T-BN2B — isolated passive performance-evidence collector."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from services.performance_accounting import (
    EVENTS_RELATIVE,
    PROPOSED_CUTOVER_CASH,
    PROPOSED_CUTOVER_EQUITY,
    PROPOSED_CUTOVER_UTC,
    UNAVAILABLE,
    q_cash,
)
from services.performance_evidence_collector import (
    FrozenBn1EvidenceSource,
    InjectedSessionGate,
    MemoryEvidenceSource,
    PerformanceEvidenceCollector,
    collect_safe,
    observe_without_affecting_execution,
    run_cli,
)
from tests.isolation_guard import ISO, PROD

pytestmark = pytest.mark.bn2a_isolation

BN1 = Path(r"C:\Users\akimw\AppData\Local\Temp\phase_198T_BN1_20260909_132114")


def _account(**extra):
    cash = extra.pop("cash", format(PROPOSED_CUTOVER_CASH, "f"))
    equity = extra.pop("equity", format(PROPOSED_CUTOVER_EQUITY, "f"))
    mv = extra.pop("market_value", format(q_cash(Decimal(equity) - Decimal(cash)), "f"))
    snap = {
        "as_of_utc": extra.pop("as_of_utc", "2026-09-10T20:05:00+00:00"),
        "session_date": extra.pop("session_date", "2026-09-10"),
        "cash": cash,
        "market_value": mv,
        "equity": equity,
        "buying_power": extra.pop("buying_power", "158021.63"),
        "account_structure": "long_only_paper",
        "positions": extra.pop(
            "positions",
            [
                {
                    "symbol": "GE",
                    "qty": "1",
                    "market_value": mv,
                    "average_entry_price": "280.1925",
                    "unrealized_pl": "0",
                    "cost_basis": "280.1925",
                    "side": "long",
                }
            ],
        ),
    }
    snap.update(extra)
    return snap


def _closed(kind: str = "regular_close", session_date: str = "2026-09-10") -> InjectedSessionGate:
    return InjectedSessionGate(
        {
            "session_state": "CLOSED",
            "reason": kind,
            "session_date": session_date,
            "session_close_et": f"{session_date}T16:00:00-04:00",
            "calendar": "XNYS",
        }
    )


def _collector(tmp_path: Path, source, gate=None, persist: bool = False):
    return PerformanceEvidenceCollector(
        events_path=tmp_path / "performance_accounting_events.jsonl",
        lock_path=tmp_path / "performance_accounting_events.lock",
        source=source,
        session_gate=gate or _closed(),
        persist=persist,
    )


def test_import_creates_no_files_and_skips_broker():
    import services.performance_evidence_collector as mod

    text = Path(mod.__file__).read_text(encoding="utf-8")
    assert "alpaca" not in text.lower()
    assert "requests." not in text
    assert "decide_paper_entry" not in text
    assert not (PROD / EVENTS_RELATIVE).exists()
    assert not (ISO / "data" / "results" / "performance_accounting_events.jsonl").exists()


def test_successful_cutover_from_bn1_freeze(tmp_path: Path):
    account = _account(
        as_of_utc=PROPOSED_CUTOVER_UTC,
        session_date="2026-09-09",
        positions=None,
    )
    account.pop("positions")
    source = FrozenBn1EvidenceSource(BN1, account)
    report = _collector(tmp_path, source).propose_cutover()
    assert report["checks_passed"] is True
    assert report["writes_performed"] is False
    assert report["publication_permitted"] is False
    assert report["blocked_reason"] == "isolated_phase_no_publication_mode"
    assert report["cutover_is_not_a_return"] is True
    assert report["proposed_events"][0]["event_type"] == "ACCOUNTING_CUTOVER"
    assert report["logical_keys"][0].startswith("cutover:")
    assert "positions_snapshot.csv" in report["source_hashes"]
    assert not (tmp_path / "performance_accounting_events.jsonl").exists()


def test_successful_normal_close(tmp_path: Path):
    source = MemoryEvidenceSource(
        snapshot=_account(),
        hashes={"positions_snapshot.csv": "abc"},
        benchmarks={
            "SPY:2026-09-10": {"close": "650", "adj_close": "650"},
            "QQQ:2026-09-10": {"close": "470", "adj_close": "470"},
        },
    )
    report = _collector(tmp_path, source, _closed()).collect_session_close()
    types = [e["event_type"] for e in report["proposed_events"]]
    assert "SESSION_CLOSE" in types
    assert "POSITION_SNAPSHOT" in types
    assert "BENCHMARK_CLOSE" in types
    assert report["close_kind"] == "regular_close"
    assert report["writes_performed"] is False
    assert report["publication_permitted"] is False
    assert report["missing_evidence"] == []


def test_early_close(tmp_path: Path):
    source = MemoryEvidenceSource(snapshot=_account())
    report = _collector(tmp_path, source, _closed("early_close")).collect_session_close()
    assert report["checks_passed"] is True
    assert report["close_kind"] == "early_close"


def test_duplicate_rerun_is_idempotent(tmp_path: Path):
    source = MemoryEvidenceSource(snapshot=_account())
    first = _collector(tmp_path, source, persist=True).collect_session_close()
    assert first["duplicate_idempotency"]["appended"] >= 1
    second = _collector(tmp_path, source, persist=True).collect_session_close()
    assert second["duplicate_idempotency"]["skipped_duplicate"] >= 1
    assert second["duplicate_idempotency"]["appended"] == 0


def test_delayed_fees_after_prior_capture(tmp_path: Path):
    fill = {
        "side": "SELL",
        "symbol": "GE",
        "qty": "1",
        "price": "330",
        "broker_order_id": "ord-1",
        "broker_fill_id": "ord-1-fill",
        "event_timestamp": "2026-09-10T20:10:00+00:00",
    }
    first_source = MemoryEvidenceSource(snapshot=_account(), fills=[fill])
    _collector(tmp_path, first_source, persist=True).collect_session_close()
    fee = {
        "event_type": "FEE",
        "amount": "0.02",
        "activity_id": "fee-reg-1",
        "fee_component": "REG",
        "allocates_to_order_id": "ord-1",
        "event_timestamp": "2026-09-10T21:00:00+00:00",
    }
    later = MemoryEvidenceSource(snapshot=_account(), activities=[fee])
    report = _collector(tmp_path, later, persist=True).collect_session_close()
    assert "fee:fee-reg-1" in report["logical_keys"]
    assert report["duplicate_idempotency"]["appended"] >= 1


def test_missing_benchmark_is_unavailable_not_inferred(tmp_path: Path):
    source = MemoryEvidenceSource(snapshot=_account(), benchmarks={})
    report = _collector(tmp_path, source).collect_session_close()
    assert "benchmark:SPY:2026-09-10" in report["missing_evidence"]
    assert "benchmark:QQQ:2026-09-10" in report["missing_evidence"]
    assert all(e["event_type"] != "BENCHMARK_CLOSE" for e in report["proposed_events"])
    assert report["checks_passed"] is True


def test_stale_snapshot_blocked(tmp_path: Path):
    source = MemoryEvidenceSource(
        snapshot=_account(as_of_utc="2026-09-08T20:00:00+00:00", session_date="2026-09-08")
    )
    report = _collector(tmp_path, source).safe_collect_session_close()
    assert report["checks_passed"] is False
    assert report["blocked_reason"] == "stale_snapshot"
    assert report["writes_performed"] is False


def test_open_market_blocked(tmp_path: Path):
    gate = InjectedSessionGate(
        {"session_state": "OPEN", "reason": "regular_session_open", "session_date": "2026-09-10"}
    )
    report = _collector(
        tmp_path, MemoryEvidenceSource(snapshot=_account()), gate
    ).safe_collect_session_close()
    assert report["blocked_reason"] == "session_open"
    assert report["publication_permitted"] is False


def test_incomplete_positions_blocked(tmp_path: Path):
    snap = _account(
        positions=[{"symbol": "GE", "qty": "1", "market_value": "325.14", "side": "long"}]
    )
    report = _collector(tmp_path, MemoryEvidenceSource(snapshot=snap)).safe_collect_session_close()
    assert report["blocked_reason"] == "incomplete_positions"


def test_reconciliation_failure_blocked(tmp_path: Path):
    snap = _account(cash="65000.00", market_value="27000.00", equity="100000.00")
    report = _collector(tmp_path, MemoryEvidenceSource(snapshot=snap)).safe_propose_cutover()
    assert report["blocked_reason"] == "reconciliation_failure"


def test_forbidden_production_path():
    with pytest.raises(Exception) as exc:
        PerformanceEvidenceCollector(
            events_path=PROD / EVENTS_RELATIVE,
            lock_path=PROD / "data" / "results" / "performance_accounting_events.lock",
            source=MemoryEvidenceSource(snapshot=_account()),
            session_gate=_closed(),
        )
    reason = getattr(exc.value, "reason", "")
    assert reason == "forbidden_production_path"


def test_collector_failure_does_not_affect_execution():
    def boom():
        raise RuntimeError("simulated_collector_failure")

    execution = {"action": "skip_market_closed", "orders_submitted": 0}

    def decide():
        return dict(execution)

    observed = observe_without_affecting_execution(boom, decide)
    assert observed["execution"] == execution
    assert observed["collection"]["blocked_reason"] == "collector_failure"
    assert collect_safe(boom, execution) == execution


def test_cli_dry_run_cutover_writes_nothing(tmp_path: Path):
    snap = tmp_path / "snap.json"
    snap.write_text(
        __import__("json").dumps(
            _account(as_of_utc=PROPOSED_CUTOVER_UTC, session_date="2026-09-09")
        ),
        encoding="utf-8",
    )
    events = tmp_path / "events.jsonl"
    lock = tmp_path / "events.lock"
    report = run_cli(
        [
            "--dry-run-cutover",
            "--events-path",
            str(events),
            "--lock-path",
            str(lock),
            "--snapshot-json",
            str(snap),
        ]
    )
    assert report["mode"] == "dry-run-cutover"
    assert report["writes_performed"] is False
    assert not events.exists()


def test_cli_rejects_production_events_path():
    report = run_cli(
        [
            "--validate-ledger",
            "--events-path",
            str(PROD / EVENTS_RELATIVE),
            "--lock-path",
            str(PROD / "data" / "results" / "performance_accounting_events.lock"),
        ]
    )
    assert report["blocked_reason"] == "forbidden_production_path"
    assert report["writes_performed"] is False


def test_marks_preserved_when_present(tmp_path: Path):
    fill = {
        "side": "SELL",
        "symbol": "GE",
        "qty": "1",
        "price": "330",
        "broker_order_id": "ord-marks",
        "broker_fill_id": "ord-marks-fill",
        "event_timestamp": "2026-09-10T20:10:00+00:00",
    }
    source = MemoryEvidenceSource(
        snapshot=_account(),
        fills=[fill],
        marks={
            "ord-marks": {
                "decision_price": "329.10",
                "submit_mark": "329.50",
                "bid": "329.00",
                "ask": "329.20",
            }
        },
    )
    report = _collector(tmp_path, source).collect_session_close()
    assert any(e["event_type"] == "PRICE_MARKS" for e in report["proposed_events"])
    assert any(e["logical_key"] == "marks:ord-marks" for e in report["proposed_events"])
