"""Focused checks for the October 16 entry-allocation repair."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from services.entry_allocation_policy import (
    classify_allocation_rows,
    entry_limits,
    resolve_settled_cash,
    validate_account_evidence,
)
from services.performance_risk_overlay import build_overlay
from services.portfolio_allocation_engine import _decide_action
from services.signal_lifecycle import (
    LifecycleConfig,
    LifecycleLogicConfig,
    decide_action,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 16, 23, 33, tzinfo=timezone.utc)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _account(**overrides):
    doc = {
        "cash": "1000.00",
        "long_market_value": "50.00",
        "equity": "1050.00",
        "short_market_value": "0",
    }
    doc.update(overrides)
    return doc


def test_missing_cash_reads_fresh_account_and_is_not_zero(tmp_path: Path):
    seen = {}

    def reader():
        seen["called"] = True
        return _account()

    cash, meta = resolve_settled_cash(results_dir=tmp_path, now=NOW, account_reader=reader)
    assert seen.get("called") is True
    assert meta["broker_called"] is True
    assert meta["source"] == "broker_readonly"
    assert cash == 1000.0
    assert cash != 0.0


def test_failed_fresh_read_stays_unproven_and_does_not_use_stale_cash(tmp_path: Path):
    stale = NOW - timedelta(hours=3)
    (tmp_path / "account_snapshot.json").write_text(
        json.dumps({"cash": "1.00", "as_of_utc": stale.isoformat()}),
        encoding="utf-8",
    )

    def reader():
        raise RuntimeError("paper get failed")

    cash, meta = resolve_settled_cash(results_dir=tmp_path, now=NOW, account_reader=reader)
    assert cash is None
    assert meta["reason"] == "settled_cash_unproven"
    assert meta["broker_called"] is True

    rows = classify_allocation_rows(
        [
            {
                "ticker": "PG",
                "is_currently_held": "False",
                "risk_flag": "OK",
                "edge_score": 0.9,
                "sizing_tier": "STRONG_EDGE",
                "effective_stance": "BUY",
                "lifecycle_action": "BUY",
                "signal": "BUY",
                "confidence": 0.8,
                "allocation_score": 0.5,
            }
        ],
        cash=cash,
        add_confidence_min=0.52,
        max_new=10,
        max_adds=3,
        max_notional=1500,
        open_slots=10,
    )
    assert rows[0]["disposition"] == "hard_block"
    assert rows[0]["reason"] == "settled_cash_unproven"


def test_verified_cash_permits_ranked_allocation_not_every_name():
    rows = []
    for index in range(12):
        rows.append(
            {
                "ticker": f"N{index:02d}",
                "is_currently_held": "False",
                "risk_flag": "OK",
                "edge_score": 0.8,
                "sizing_tier": "STRONG_EDGE",
                "effective_stance": "BUY",
                "lifecycle_action": "BUY",
                "signal": "BUY",
                "confidence": 0.8,
                "allocation_score": 1 - index / 100,
            }
        )
    ranked = classify_allocation_rows(
        rows,
        cash=75271.72,
        add_confidence_min=0.52,
        max_new=10,
        max_adds=3,
        max_notional=1500,
        open_slots=21,
    )
    opened = [row["symbol"] for row in ranked if row["disposition"] == "OPEN_NEW"]
    waiting = [row["symbol"] for row in ranked if row["disposition"] == "eligible"]
    assert opened == [f"N{index:02d}" for index in range(10)]
    assert waiting == ["N10", "N11"]
    assert all(
        row["reason"] == "eligible:beyond_max_new_positions_per_run"
        for row in ranked
        if row["symbol"] in waiting
    )


def _open_row(symbol: str, **overrides):
    row = {
        "ticker": symbol,
        "is_currently_held": "False",
        "risk_flag": "OK",
        "edge_score": 0.8,
        "sizing_tier": "STRONG_EDGE",
        "effective_stance": "BUY",
        "lifecycle_action": "BUY",
        "signal": "BUY",
        "confidence": 0.8,
        "allocation_score": 0.5,
    }
    row.update(overrides)
    return row


def _blocked_entries(cash):
    ranked = classify_allocation_rows(
        [_open_row("PG", allocation_score=0.9), _open_row("UNH", allocation_score=0.8)],
        cash=cash,
        add_confidence_min=0.52,
        max_new=10,
        max_adds=3,
        max_notional=1500,
        open_slots=10,
    )
    assert ranked
    assert all(row["disposition"] == "hard_block" for row in ranked)
    assert all(row["reason"] == "settled_cash_unproven" for row in ranked)


def test_account_failures_block_every_entry(tmp_path: Path):
    stale = NOW - timedelta(hours=3)
    consistent = _account()
    cash, reason = validate_account_evidence(
        consistent, observed_at=stale, now=NOW, require_identity=True
    )
    assert cash is None and reason == "settled_cash_unproven"
    _blocked_entries(cash)

    malformed, malformed_reason = validate_account_evidence(
        {"cash": "not-a-number", "equity": "1", "long_market_value": "1"},
        observed_at=NOW,
        now=NOW,
        require_identity=True,
    )
    assert malformed is None and malformed_reason == "settled_cash_unproven"
    _blocked_entries(malformed)

    inconsistent, inconsistent_reason = validate_account_evidence(
        {"cash": "10", "long_market_value": "1", "equity": "100", "short_market_value": "0"},
        observed_at=NOW,
        now=NOW,
        require_identity=True,
    )
    assert inconsistent is None and inconsistent_reason == "settled_cash_unproven"
    _blocked_entries(inconsistent)

    def fail():
        raise RuntimeError("refresh failed")

    (tmp_path / "paper_account_snapshot.json").write_text(
        json.dumps({"buying_power": "347024.04", "as_of_utc": NOW.isoformat()}),
        encoding="utf-8",
    )
    failed_cash, meta = resolve_settled_cash(results_dir=tmp_path, now=NOW, account_reader=fail)
    assert meta["broker_called"] is True
    assert failed_cash is None
    assert meta["reason"] == "settled_cash_unproven"
    _blocked_entries(failed_cash)


def test_buying_power_never_substitutes_for_settled_cash():
    only_power, reason = validate_account_evidence(
        {"buying_power": "347024.04"},
        observed_at=NOW,
        now=NOW,
    )
    assert only_power is None
    assert reason == "settled_cash_unproven"
    cash, cash_reason = validate_account_evidence(
        _account(buying_power="347024.04"),
        observed_at=NOW,
        now=NOW,
        require_identity=True,
    )
    assert cash_reason == ""
    assert cash == 1000.0


def test_missing_local_cash_triggers_one_readonly_refresh(tmp_path, monkeypatch):
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        return _account()

    monkeypatch.setattr("services.paper_entry_loop_stage.RESULTS", tmp_path)
    from services.paper_entry_loop_stage import _load_settled_paper_cash

    cash, meta = _load_settled_paper_cash(account_reader=reader, now=NOW)
    assert calls["n"] == 1
    assert meta["broker_called"] is True
    assert cash == 1000.0
    source = (ROOT / "services" / "entry_allocation_policy.py").read_text(encoding="utf-8")
    reader_src = source.split("def read_paper_account_readonly", 1)[1].split("\ndef ", 1)[0]
    assert "get_account" in reader_src
    assert "submit_order" not in reader_src
    assert "_post" not in reader_src
    assert "buying_power" not in reader_src


def test_flat_trim_is_not_a_position_action_and_held_risk_blocks_add():
    flat, flat_reason = _decide_action(
        is_held=False,
        risk_components=["TRIM_PRIORITY", "BLOCK_NEW_BUY"],
        edge_score=0.2,
        sizing_tier="WEAK_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.9,
        delta_pct=None,
        add_confidence_min=0.52,
    )
    assert flat not in ("EXIT", "TRIM", "BLOCK")
    assert flat_reason.startswith("soft_caution:historical_loss;")
    for flag in (["BLOCK_NEW_BUY"], ["TRIM_PRIORITY"], ["FORCE_EXIT"]):
        action, reason = _decide_action(
            is_held=True,
            risk_components=flag,
            edge_score=0.9,
            sizing_tier="STRONG_EDGE",
            effective_stance="HOLD",
            lifecycle_action="HOLD",
            signal="BUY",
            confidence=0.9,
            delta_pct=None,
            add_confidence_min=0.52,
        )
        assert action in ("EXIT", "TRIM", "BLOCK")
        assert action != "ADD"
        assert "basis" not in reason


def test_voo_and_xlk_stay_blocked_and_adds_respect_cash():
    qualifying = dict(
        is_currently_held="True",
        risk_flag="OK",
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="BUY",
        confidence=0.9,
        allocation_score=0.4,
    )
    ranked = classify_allocation_rows(
        [
            _open_row("PG", allocation_score=0.9),
            {"ticker": "VOO", **qualifying},
            {"ticker": "XLK", **qualifying, "risk_flag": "TRIM_PRIORITY|BLOCK_NEW_BUY"},
            {"ticker": "MA", **qualifying, "allocation_score": 0.2},
        ],
        cash=1500.0,
        add_confidence_min=0.52,
        max_new=10,
        max_adds=3,
        max_notional=1500,
        open_slots=5,
    )
    by_symbol = {row["symbol"]: row for row in ranked}
    assert by_symbol["PG"]["disposition"] == "OPEN_NEW"
    assert by_symbol["PG"]["notional"] == 1500
    assert by_symbol["VOO"]["disposition"] == "hard_block"
    assert by_symbol["VOO"]["reason"] == "hard_block:basis_unproven_hold"
    assert by_symbol["XLK"]["disposition"] == "hard_block"
    assert by_symbol["XLK"]["action"] == "TRIM"
    assert by_symbol["MA"]["disposition"] == "hard_block"
    assert by_symbol["MA"]["reason"] == "hard_block:per_order_notional_exceeds_settled_cash"


def test_malformed_account_is_unproven():
    cash, reason = validate_account_evidence(
        {"cash": "10", "long_market_value": "1", "equity": "100", "short_market_value": "0"},
        observed_at=NOW,
        now=NOW,
        require_identity=True,
    )
    assert cash is None
    assert reason == "settled_cash_unproven"
    buying_power_only, buying_reason = validate_account_evidence(
        {"buying_power": "347024.04"},
        observed_at=NOW,
        now=NOW,
    )
    assert buying_power_only is None
    assert buying_reason == "settled_cash_unproven"


def test_flat_historical_loss_is_caution_and_held_risk_still_exits():
    flat, flat_reason = _decide_action(
        is_held=False,
        risk_components=["FORCE_EXIT", "BLOCK_NEW_BUY"],
        edge_score=-0.27,
        sizing_tier="BLOCKED",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.83,
        delta_pct=None,
        add_confidence_min=0.52,
    )
    assert flat != "EXIT"
    assert flat != "BLOCK"
    assert flat_reason.startswith("soft_caution:historical_loss;")
    held, held_reason = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT", "BLOCK_NEW_BUY"],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.9,
        delta_pct=None,
        add_confidence_min=0.52,
    )
    assert held == "EXIT"
    assert "FORCE_EXIT" in held_reason


def test_held_buy_adds_only_when_configured_floor_passes():
    logic = LifecycleLogicConfig(enabled=True, add_confidence_min=0.52)
    cfg = LifecycleConfig()
    added, added_reason = decide_action(
        cfg=cfg,
        logic=logic,
        signal="BUY",
        delta_pct=float("nan"),
        confidence=0.52,
        pos_state="LONG",
        last_change_date=None,
        cooldown_until=None,
        as_of_date=NOW.date(),
    )
    weak, weak_reason = decide_action(
        cfg=cfg,
        logic=logic,
        signal="BUY",
        delta_pct=0.02,
        confidence=0.51,
        pos_state="LONG",
        last_change_date=None,
        cooldown_until=None,
        as_of_date=NOW.date(),
    )
    assert added == "ADD"
    assert added_reason == "long_strong_buy_to_add"
    assert weak == "HOLD"
    assert weak_reason == "long_weak_buy_to_hold"
    below, _ = _decide_action(
        is_held=True,
        risk_components=[],
        edge_score=None,
        sizing_tier="",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="BUY",
        confidence=0.51,
        delta_pct=None,
        add_confidence_min=0.52,
    )
    above, above_reason = _decide_action(
        is_held=True,
        risk_components=[],
        edge_score=None,
        sizing_tier="",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="BUY",
        confidence=0.52,
        delta_pct=None,
        add_confidence_min=0.52,
    )
    assert below == "HOLD"
    assert above == "ADD"
    assert "held_buy_conf>=0.52" in above_reason


def test_overlay_suppresses_flat_loss_and_keeps_open_loss():
    frame = pd.DataFrame(
        [
            {"ticker": "HD", "total_pl": -551.49, "unrealized_pl": 0.0, "open_qty": 0},
            {"ticker": "XLK", "total_pl": -154.72, "unrealized_pl": -85.17, "open_qty": 7},
        ]
    )
    overlay, _counts = build_overlay(frame)
    flags = {row.ticker: row.risk_flag for row in overlay.itertuples()}
    assert "FORCE_EXIT" not in flags["HD"]
    assert "BLOCK_NEW_BUY" not in flags["HD"]
    assert (
        "FORCE_EXIT" in flags["XLK"]
        or "TRIM_PRIORITY" in flags["XLK"]
        or "BLOCK_NEW_BUY" in flags["XLK"]
    )


def test_october16_replay_ranks_without_writing_or_submitting(monkeypatch):
    def _broker_called(*_args, **_kwargs):
        raise AssertionError("broker called")

    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker.__init__", _broker_called)
    monkeypatch.setattr(
        "services.entry_allocation_policy.read_paper_account_readonly",
        _broker_called,
    )
    protected = [
        ROOT / "data/results/portfolio_allocation_recommendations.csv",
        ROOT / "data/results/performance_accounting_events.jsonl",
        ROOT / "data/results/performance_risk_overlay.csv",
        ROOT / "data/governance/basis_unproven_hold.json",
        ROOT / "data/governance/basis_unproven_hold_audit.jsonl",
    ]
    before = {path: _sha(path) for path in protected}
    book = pd.read_csv(protected[0])
    close = None
    for line in protected[1].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        body = event.get("body") or {}
        if event.get("event_type") == "SESSION_CLOSE" and body.get("session_date") == "2026-10-16":
            close = body
            break
    assert close is not None
    cash, reason = validate_account_evidence(
        {
            "cash": close["cash"],
            "market_value": close["market_value"],
            "equity": close["equity"],
            "short_market_value": "0",
        },
        observed_at=NOW,
        now=NOW,
        require_identity=True,
    )
    assert reason == ""
    assert cash == pytest.approx(75271.72)
    limits = entry_limits(ROOT / "config")
    held = int((book["is_currently_held"].astype(str).str.lower() == "true").sum())
    slots = int(limits["max_positions"]) - held
    ranked = classify_allocation_rows(
        book.to_dict(orient="records"),
        cash=cash,
        add_confidence_min=0.52,
        max_new=int(limits["max_new"]),
        max_adds=int(limits["max_adds"]),
        max_notional=float(limits["max_notional"]),
        open_slots=slots,
    )

    def _by_rank(disposition: str) -> list:
        chosen = [row for row in ranked if row["disposition"] == disposition]
        return [
            row["symbol"] for row in sorted(chosen, key=lambda row: (-row["score"], row["symbol"]))
        ]

    opened = _by_rank("OPEN_NEW")
    added = _by_rank("ADD")
    eligible = _by_rank("eligible")
    assert opened == ["PG", "UNH", "WMT", "DIS", "V", "BRK.B", "XLP", "XLV", "KO", "ABBV"]
    assert added == ["MSFT", "MA"]
    assert eligible == ["XLB", "BLK", "CVX", "XLY", "NFLX", "XLE", "PFE", "XLU"]
    by_symbol = {row["symbol"]: row for row in ranked}
    for symbol in ("HD", "UNG", "USO", "SLV", "GBTC"):
        assert by_symbol[symbol]["disposition"] == "soft_caution"
        assert by_symbol[symbol]["action"] not in ("EXIT", "BLOCK")
    for symbol in ("NVDA", "PLTR", "XLK", "TSLA", "VOO"):
        assert by_symbol[symbol]["disposition"] == "hard_block"
        assert symbol not in opened
        assert symbol not in added
    assert by_symbol["VOO"]["reason"] == "hard_block:basis_unproven_hold"
    assert by_symbol["XLK"]["action"] in ("TRIM", "EXIT", "BLOCK")
    assert by_symbol["HD"]["reason"].startswith("soft_caution:historical_loss;")
    assert "submitted" not in json.dumps(ranked)
    after = {path: _sha(path) for path in protected}
    assert after == before
