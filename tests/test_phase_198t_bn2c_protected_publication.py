"""Phase 198T-BN2C — isolated protected accounting-publication path."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from services.performance_accounting import (
    EVENT_CUTOVER,
    PerformanceAccountingError,
    load_events,
    sha256_file,
)
from services.performance_evidence_collector import (
    CONFIRM_CUTOVER,
    CONFIRM_SESSION_CLOSE,
    LEVERAGE_MARGIN_CAPABLE_UNBORROWED,
    LEVERAGE_MARGIN_IN_USE,
    LEVERAGE_SHORT_EXPOSURE,
    MemoryEvidenceSource,
    PerformanceEvidenceCollector,
    PublicationContext,
    classify_leverage,
    run_cli,
)

pytestmark = pytest.mark.bn2a_isolation


def _account(**extra):
    cash = extra.pop("cash", "65663.61")
    equity = extra.pop("equity", "92358.02")
    mv = extra.pop("market_value", format(Decimal(equity) - Decimal(cash), "f"))
    snap = {
        "as_of_utc": extra.pop("as_of_utc", "2026-09-10T20:05:00+00:00"),
        "session_date": extra.pop("session_date", "2026-09-10"),
        "cash": cash,
        "market_value": mv,
        "equity": equity,
        "buying_power": extra.pop("buying_power", "158021.63"),
        "account_structure": extra.pop("account_structure", "long_only_paper"),
        "short_market_value": extra.pop("short_market_value", "0"),
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


def _hashes():
    return {
        "account": "acc-stable",
        "positions": "pos-stable",
        "orders": "ord-stable",
        "activities": "act-stable",
        "SPY": "spy-stable",
        "QQQ": "qqq-stable",
    }


def _benches(session_date: str = "2026-09-10"):
    return {
        f"SPY:{session_date}": {"close": "650", "adj_close": "650"},
        f"QQQ:{session_date}": {"close": "470", "adj_close": "470"},
    }


def _closed(kind: str = "regular_close", session_date: str = "2026-09-10", hour: int = 16):
    from services.performance_evidence_collector import InjectedSessionGate

    return InjectedSessionGate(
        {
            "session_state": "CLOSED",
            "reason": kind,
            "session_date": session_date,
            "session_close_et": f"{session_date}T{hour:02d}:00:00-04:00",
            "calendar": "XNYS",
        }
    )


def _source(snapshot=None, **extra):
    return MemoryEvidenceSource(
        snapshot=snapshot or _account(),
        hashes=extra.pop("hashes", _hashes()),
        benchmarks=extra.pop("benchmarks", _benches()),
        **extra,
    )


def _collector(tmp_path: Path, source, gate=None):
    return PerformanceEvidenceCollector(
        events_path=tmp_path / "performance_accounting_events.jsonl",
        lock_path=tmp_path / "performance_accounting_events.lock",
        source=source,
        session_gate=gate or _closed(),
        persist=False,
    )


def _ctx(**extra):
    return PublicationContext(**extra)


def test_successful_create_only_cutover(tmp_path: Path):
    collector = _collector(tmp_path, _source())
    report = collector.publish_cutover(confirm=CONFIRM_CUTOVER, context=_ctx())
    ledger = tmp_path / "performance_accounting_events.jsonl"
    assert report["writes_performed"] is True
    assert report["publication_permitted"] is True
    assert report["blocked_reason"] is None
    assert report["opening_checkpoint"] is True
    assert report["creates_historical_return"] is False
    assert report["creates_realized_pl"] is False
    assert report["creates_deposit"] is False
    assert report["creates_withdrawal"] is False
    assert report["leverage_class"] == LEVERAGE_MARGIN_CAPABLE_UNBORROWED
    assert ledger.is_file()
    events = load_events(ledger)
    assert len(events) == 1
    assert events[0]["event_type"] == EVENT_CUTOVER
    body = events[0]["body"]
    assert body["opening_checkpoint"] is True
    assert body["opening_basis_only"] is True
    assert body["not_a_return"] is True
    assert body["not_a_deposit"] is True
    assert body["not_a_withdrawal"] is True
    assert body["not_profit_or_loss"] is True
    assert body["historical_return"] is None
    assert body["realized_pl"] is None
    assert body["deposit"] is None
    assert body["withdrawal"] is None
    assert body["measurement_boundary"] == "earliest_future"
    assert body["position_count"] == 1
    assert body["positions"][0]["average_entry_price"] == "280.1925"
    assert not (tmp_path / "portfolio_history.csv").exists()


def test_existing_ledger_refusal(tmp_path: Path):
    ledger = tmp_path / "performance_accounting_events.jsonl"
    ledger.write_text('{"unknown": true}\n', encoding="utf-8")
    before = ledger.read_bytes()
    report = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert report["blocked_reason"] == "existing_ledger_refused"
    assert report["writes_performed"] is False
    assert ledger.read_bytes() == before


def test_interrupted_publication_leaves_ledger_unchanged(tmp_path: Path):
    def boom(stage, path, payload):
        if stage == "before_rename":
            raise PerformanceAccountingError("publication_interrupted")

    collector = _collector(tmp_path, _source())
    report = collector.safe_publish_cutover(confirm=CONFIRM_CUTOVER, context=_ctx(fail_hook=boom))
    assert report["writes_performed"] is False
    assert report["blocked_reason"] == "publication_interrupted"
    assert not (tmp_path / "performance_accounting_events.jsonl").exists()
    assert not (tmp_path / "performance_accounting_events.jsonl.tmp").exists()


def test_hash_mismatch_aborts_without_rename(tmp_path: Path):
    def corrupt(stage, path, payload):
        if stage == "after_fsync":
            path.write_bytes(path.read_bytes() + b"CORRUPT")

    report = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx(fail_hook=corrupt)
    )
    assert report["blocked_reason"] == "publication_hash_mismatch"
    assert report["writes_performed"] is False
    assert not (tmp_path / "performance_accounting_events.jsonl").exists()


def test_held_lock_refuses_publication(tmp_path: Path):
    lock = tmp_path / "performance_accounting_events.lock"
    handle = open(lock, "a+b")
    handle.seek(0)
    if handle.read(1) == b"":
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    import msvcrt

    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        report = _collector(tmp_path, _source()).safe_publish_cutover(
            confirm=CONFIRM_CUTOVER, context=_ctx()
        )
        assert report["blocked_reason"] == "accounting_lock_unavailable"
        assert report["writes_performed"] is False
        assert not (tmp_path / "performance_accounting_events.jsonl").exists()
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def test_open_market_blocks_publication(tmp_path: Path):
    from services.performance_evidence_collector import InjectedSessionGate

    gate = InjectedSessionGate(
        {"session_state": "OPEN", "reason": "regular_session_open", "session_date": "2026-09-10"}
    )
    report = _collector(tmp_path, _source(), gate).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert report["blocked_reason"] == "session_open"
    assert report["writes_performed"] is False
    assert not (tmp_path / "performance_accounting_events.jsonl").exists()


def test_stale_snapshot_blocks_publication(tmp_path: Path):
    source = _source(_account(as_of_utc="2026-09-10T15:00:00+00:00"))
    report = _collector(tmp_path, source).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert report["blocked_reason"] == "stale_snapshot"
    assert report["writes_performed"] is False


def test_changing_evidence_blocks_publication(tmp_path: Path):
    source = _source(hashes_next={**_hashes(), "positions": "changed"})
    report = _collector(tmp_path, source).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert report["blocked_reason"] == "changing_evidence"
    assert report["writes_performed"] is False
    assert not (tmp_path / "performance_accounting_events.jsonl").exists()


def test_writers_active_and_freshness_lock_block(tmp_path: Path):
    busy = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx(writers_idle=False)
    )
    assert busy["blocked_reason"] == "writers_active"
    locked = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx(freshness_lock_present=True)
    )
    assert locked["blocked_reason"] == "freshness_lock_present"


def test_normal_and_early_close_publication(tmp_path: Path):
    cutover = _collector(tmp_path, _source()).publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert cutover["writes_performed"] is True
    regular = _collector(tmp_path, _source(), _closed("regular_close")).publish_session_close(
        confirm=CONFIRM_SESSION_CLOSE, context=_ctx()
    )
    assert regular["writes_performed"] is True
    assert regular["close_kind"] == "regular_close"
    early_root = tmp_path / "early"
    early_root.mkdir()
    early_collector = _collector(early_root, _source(), _closed("early_close", hour=13))
    early_collector.publish_cutover(confirm=CONFIRM_CUTOVER, context=_ctx())
    early = early_collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=_ctx())
    assert early["close_kind"] == "early_close"
    assert early["writes_performed"] is True


def test_margin_capable_unborrowed_is_permitted(tmp_path: Path):
    snap = _account(
        margin=True,
        account_type="margin",
        multiplier="2",
        buying_power="184716.04",
        short_market_value="0",
        margin_debit="0",
        borrowed="0",
        used_margin="0",
    )
    assert classify_leverage(snap) == LEVERAGE_MARGIN_CAPABLE_UNBORROWED
    report = _collector(tmp_path, _source(snap)).publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert report["writes_performed"] is True
    assert report["leverage_class"] == LEVERAGE_MARGIN_CAPABLE_UNBORROWED


def test_actual_margin_use_rejected(tmp_path: Path):
    snap = _account(
        cash="-1000.00", market_value="93358.02", equity="92358.02", margin_debit="1000"
    )
    report = _collector(tmp_path, _source(snap)).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert report["blocked_reason"] == LEVERAGE_MARGIN_IN_USE
    assert report["writes_performed"] is False
    assert not (tmp_path / "performance_accounting_events.jsonl").exists()


def test_short_exposure_rejected(tmp_path: Path):
    snap = _account(
        cash="100000.00",
        market_value="-8000.00",
        equity="92000.00",
        short_market_value="-8000.00",
        positions=[
            {
                "symbol": "GE",
                "qty": "-1",
                "market_value": "-8000",
                "average_entry_price": "1",
                "side": "short",
            }
        ],
    )
    assert classify_leverage(snap) == LEVERAGE_SHORT_EXPOSURE
    report = _collector(tmp_path, _source(snap)).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert report["blocked_reason"] == LEVERAGE_SHORT_EXPOSURE
    assert report["writes_performed"] is False


def test_duplicate_session_close_does_not_write_again(tmp_path: Path):
    collector = _collector(tmp_path, _source())
    collector.publish_cutover(confirm=CONFIRM_CUTOVER, context=_ctx())
    first = collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=_ctx())
    assert first["writes_performed"] is True
    ledger = tmp_path / "performance_accounting_events.jsonl"
    digest = sha256_file(ledger)
    second = collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=_ctx())
    assert second["blocked_reason"] == "duplicate_session_close"
    assert second["writes_performed"] is False
    assert sha256_file(ledger) == digest


def test_wrong_confirmation_phrase(tmp_path: Path):
    report = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm="PLEASE PUBLISH", context=_ctx()
    )
    assert report["blocked_reason"] == "wrong_confirmation_phrase"
    assert report["writes_performed"] is False
    close = _collector(tmp_path, _source()).safe_publish_session_close(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    assert close["blocked_reason"] == "wrong_confirmation_phrase"


def test_production_and_live_mode_rejected(tmp_path: Path):
    live = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx(live_execution_permitted=True)
    )
    assert live["blocked_reason"] == "production_or_live_mode"
    prod = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx(mode="LIVE")
    )
    assert prod["blocked_reason"] == "production_or_live_mode"
    mutate = _collector(tmp_path, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx(can_mutate_orders=True)
    )
    assert mutate["blocked_reason"] == "order_mutation_capability"
    snap = tmp_path / "snap.json"
    sess = tmp_path / "sess.json"
    snap.write_text(json.dumps(_account()), encoding="utf-8")
    sess.write_text(
        json.dumps(
            {
                "session_state": "CLOSED",
                "reason": "regular_close",
                "session_date": "2026-09-10",
                "session_close_et": "2026-09-10T16:00:00-04:00",
            }
        ),
        encoding="utf-8",
    )
    cli = run_cli(
        [
            "--publish-cutover",
            "--events-path",
            str(tmp_path / "events.jsonl"),
            "--lock-path",
            str(tmp_path / "events.lock"),
            "--snapshot-json",
            str(snap),
            "--session-json",
            str(sess),
            "--confirm",
            CONFIRM_CUTOVER,
            "--live-execution-permitted",
        ]
    )
    assert cli["blocked_reason"] == "production_or_live_mode"
    assert cli["writes_performed"] is False


def test_account_id_redacted_from_report_and_errors(tmp_path: Path):
    snap = _account(account_id="PA9XYZ1234", note="linked to PA9XYZ1234")
    report = _collector(tmp_path, _source(snap)).publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_ctx()
    )
    dumped = json.dumps(report, default=str)
    assert "PA9XYZ1234" not in dumped
    events = load_events(tmp_path / "performance_accounting_events.jsonl")
    assert "PA9XYZ1234" not in json.dumps(events)
    assert "account_id" not in events[0]["body"]


def test_idempotent_retry_planning_does_not_write(tmp_path: Path):
    collector = _collector(tmp_path, _source())
    collector.publish_cutover(confirm=CONFIRM_CUTOVER, context=_ctx())
    collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=_ctx())
    ledger = tmp_path / "performance_accounting_events.jsonl"
    digest = sha256_file(ledger)
    mtime = ledger.stat().st_mtime_ns
    planned = collector.publish_session_close(
        confirm=CONFIRM_SESSION_CLOSE, context=_ctx(), plan_only=True
    )
    assert planned["writes_performed"] is False
    assert planned["mode"] == "plan-session-close"
    assert planned["planned_append_count"] == 0
    assert sha256_file(ledger) == digest
    assert ledger.stat().st_mtime_ns == mtime
