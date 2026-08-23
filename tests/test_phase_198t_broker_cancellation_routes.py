"""Phase 198T-B — guarded paper cancellation routes (N1/N2) and remaining SELL POSTs.

PAPER ONLY. Fake transport/broker. Isolated sell-store state. No real broker.
Pure cancellation must not consume reviewed-SELL authority or create a dispatch row.
"""

from __future__ import annotations

import ast
import hashlib
import math
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from services.paper_order_cancellation_guard import (
    CAP_MONITOR,
    CAP_RUNTIME_VERIFY,
    PaperOrderCancelBlocked,
    classify_delete_http_outcome,
    evaluate_paper_order_cancellation,
    live_execution_is_enabled,
)
from services.paper_sell_arm_runtime_enforcement import (
    APPROVAL_CLI_CAP,
    PaperSellDispatchBlocked,
    _ACTIVE_PERMIT,
)
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    NetworkFakeAlpacaBroker,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    install_boundary_trace,
    isolate_sell_store,
    seed_paper_sell_store,
    stub_unrelated_pre_gates,
)

ROOT = Path(__file__).resolve().parents[1]


class RecordingTransport:
    def __init__(self, status_code: int = 204, text: str = "", exc: Optional[BaseException] = None):
        self.status_code = status_code
        self.text = text
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(
            {"method": "DELETE", "url": url, "headers_keys": sorted((headers or {}).keys())}
        )
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(status_code=self.status_code, text=self.text)

    def post(self, *a, **k):
        self.calls.append({"method": "POST"})
        raise AssertionError("POST is forbidden on the cancellation transport")

    def patch(self, *a, **k):
        self.calls.append({"method": "PATCH"})
        raise AssertionError("PATCH is forbidden on the cancellation transport")

    @property
    def deletes(self) -> int:
        return sum(1 for c in self.calls if c.get("method") == "DELETE")

    @property
    def posts(self) -> int:
        return sum(1 for c in self.calls if c.get("method") == "POST")

    @property
    def patches(self) -> int:
        return sum(1 for c in self.calls if c.get("method") == "PATCH")


class StatusFakeBroker(NetworkFakeAlpacaBroker):
    def __init__(self, *a, order_status_by_id: Optional[Dict[str, str]] = None, **k):
        super().__init__(*a, **k)
        self.order_status_by_id = dict(order_status_by_id or {})
        self.patches: List[str] = []

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if path.startswith("/v2/orders/") and path != "/v2/orders":
            oid = path.rsplit("/", 1)[-1]
            side = self.order_side_by_id.get(oid, "sell")
            status = self.order_status_by_id.get(oid, "open")
            self.gets.append(path)
            return {
                "id": oid,
                "side": side,
                "symbol": "TSLA",
                "qty": "8",
                "status": status,
                "type": "limit",
            }
        return super()._get(path, params)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dispatch_count(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sell_dispatches'"
        ).fetchone()
        if row is None:
            return 0
        return int(conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0])
    finally:
        conn.close()


def _authority_states(db_path: Path) -> List[str]:
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sell_authorities'"
        ).fetchone()
        if row is None:
            return []
        return [str(r[0]) for r in conn.execute("SELECT state FROM sell_authorities").fetchall()]
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _safety(monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    block_real_network(monkeypatch)
    monkeypatch.setenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "paper-secret")
    monkeypatch.setenv("APCA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-secret")
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


def _isolate_empty(monkeypatch, tmp_path: Path) -> Path:
    empty = tmp_path / "empty_store"
    empty.mkdir(exist_ok=True)
    isolate_sell_store(monkeypatch, empty)
    stub_unrelated_pre_gates(monkeypatch)
    return empty


def _seed_then_isolate(monkeypatch, tmp_path: Path, **seed_kw) -> Path:
    store = tmp_path / "sell_store"
    store.mkdir(exist_ok=True)
    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(store, **seed_kw)
    isolate_sell_store(monkeypatch, store)
    stub_unrelated_pre_gates(monkeypatch)
    return store


def _prep_monitor(monkeypatch, tmp_path: Path):
    import monitor_open_orders as mon

    isolate_sell_store(monkeypatch, tmp_path)
    monkeypatch.setattr(mon, "get_alpaca_headers", lambda: {"APCA-API-KEY-ID": "paper-key"})
    monkeypatch.setattr(mon, "get_alpaca_base", lambda: "https://paper-api.alpaca.markets")
    return mon


def test_n1_valid_paper_buy_cancel(tmp_path, monkeypatch):
    mon = _prep_monitor(monkeypatch, tmp_path)
    db = tmp_path / "paper_sell_arm.sqlite3"
    before_disp = _dispatch_count(db)
    before_auth = _authority_states(db)
    transport = RecordingTransport()
    ok, note = mon.cancel_order(
        "ord-buy-1", order_side="buy", order_status="open", transport=transport
    )
    assert ok is True
    assert "CANCELLED" in note
    assert transport.deletes == 1
    assert transport.posts == 0
    assert transport.patches == 0
    assert _dispatch_count(db) == before_disp
    assert _authority_states(db) == before_auth


def test_n1_valid_paper_sell_cancel_does_not_consume_authority(tmp_path, monkeypatch):
    store = _seed_then_isolate(monkeypatch, tmp_path, consume=False)
    mon = _prep_monitor(monkeypatch, store)
    db = store / "paper_sell_arm.sqlite3"
    before = _sha256(db)
    before_disp = _dispatch_count(db)
    transport = RecordingTransport()
    ok, note = mon.cancel_order(
        "ord-sell-1", order_side="sell", order_status="open", transport=transport
    )
    assert ok is True
    assert "CANCELLED" in note
    assert transport.deletes == 1
    assert transport.posts == 0
    assert _dispatch_count(db) == before_disp == 0
    assert _sha256(db) == before
    assert _ACTIVE_PERMIT.get() is None


def test_n1_malformed_and_missing_order_id(tmp_path, monkeypatch):
    mon = _prep_monitor(monkeypatch, tmp_path)
    transport = RecordingTransport()
    for oid in ("", "bad/id", "..", "id with space"):
        ok, note = mon.cancel_order(oid, order_side="buy", transport=transport)
        assert ok is False
        assert "BLOCKED" in note
    assert transport.deletes == 0


def test_n1_unknown_side_fail_closed_never_submit(tmp_path, monkeypatch):
    mon = _prep_monitor(monkeypatch, tmp_path)
    transport = RecordingTransport()
    ok, note = mon.cancel_order("ord-1", order_side=None, transport=transport)
    assert ok is False
    assert "order_side_unknown" in note
    assert transport.deletes == 0
    assert transport.posts == 0


def test_n1_live_flag_zero_delete(tmp_path, monkeypatch):
    mon = _prep_monitor(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "1")
    transport = RecordingTransport()
    ok, note = mon.cancel_order("ord-1", order_side="buy", transport=transport)
    assert ok is False
    assert "live_execution_blocked" in note
    assert transport.deletes == 0


def test_n1_environment_mismatch_zero_delete(tmp_path, monkeypatch):
    mon = _prep_monitor(monkeypatch, tmp_path)
    monkeypatch.setattr(mon, "get_alpaca_base", lambda: "https://api.alpaca.markets")
    transport = RecordingTransport()
    ok, note = mon.cancel_order("ord-1", order_side="buy", transport=transport)
    assert ok is False
    assert "broker_environment_mismatch" in note
    assert transport.deletes == 0


def test_n1_already_canceled_idempotent(tmp_path, monkeypatch):
    mon = _prep_monitor(monkeypatch, tmp_path)
    transport = RecordingTransport()
    ok, note = mon.cancel_order(
        "ord-1", order_side="sell", order_status="canceled", transport=transport
    )
    assert ok is True
    assert "IDEMPOTENT_TERMINAL" in note
    assert transport.deletes == 0
    ok2, _ = mon.cancel_order(
        "ord-1", order_side="sell", order_status="filled", transport=transport
    )
    assert ok2 is True
    assert transport.deletes == 0
    assert transport.posts == 0


def test_n1_timeout_outcome_unknown_no_retry_loop(tmp_path, monkeypatch):
    mon = _prep_monitor(monkeypatch, tmp_path)
    transport = RecordingTransport(exc=TimeoutError("timeout"))
    ok, note = mon.cancel_order("ord-1", order_side="buy", transport=transport)
    assert ok is False
    assert "OUTCOME_UNKNOWN" in note
    assert transport.deletes == 1
    # The route does not automatically repeat DELETE.
    assert transport.deletes == 1


def test_n2_requires_really_cancel_capability(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    from services import runtime_verify as rv

    calls: List[tuple] = []

    def _req(method, path, **_k):
        calls.append((method, path))
        return None

    with pytest.raises(PaperOrderCancelBlocked) as ei:
        rv.cancel_order("ord-1", order_side="buy", request_fn=_req)
    assert "cancellation_capability" in ei.value.reason
    assert calls == []


def test_n2_paper_buy_and_sell_pure_cancel(tmp_path, monkeypatch):
    store = _seed_then_isolate(monkeypatch, tmp_path, consume=False)
    db = store / "paper_sell_arm.sqlite3"
    before = _sha256(db)
    from services import runtime_verify as rv

    calls: List[tuple] = []

    def _req(method, path, **_k):
        calls.append((method, path))
        if method != "DELETE":
            raise AssertionError(f"unexpected {method}")
        return {"status": "canceled"}

    rv.cancel_order(
        "ord-buy-1",
        order_side="buy",
        cancellation_capability=CAP_RUNTIME_VERIFY,
        request_fn=_req,
    )
    rv.cancel_order(
        "ord-sell-1",
        order_side="sell",
        cancellation_capability=CAP_RUNTIME_VERIFY,
        request_fn=_req,
    )
    assert calls == [
        ("DELETE", "/v2/orders/ord-buy-1"),
        ("DELETE", "/v2/orders/ord-sell-1"),
    ]
    assert _sha256(db) == before
    assert _dispatch_count(db) == 0
    assert _ACTIVE_PERMIT.get() is None


def test_n2_verify_open_orders_dry_run_zero_delete(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    from services import runtime_verify as rv

    calls: List[tuple] = []

    def _req(method, path, **_k):
        calls.append((method, path))
        return []

    monkeypatch.setattr(
        rv,
        "fetch_open_orders",
        lambda **_k: [
            {
                "id": "day-1",
                "side": "buy",
                "symbol": "AAPL",
                "time_in_force": "day",
                "status": "open",
            }
        ],
    )
    monkeypatch.setattr(rv, "_request", _req)
    report = rv.verify_open_orders(cancel_non_gtc=True, dry_run=True)
    assert report["dry_run"] is True
    assert report["cancelled"] == []
    assert not any(m == "DELETE" for m, _p in calls)


def test_n2_verify_open_orders_really_cancel_one_delete(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    from services import runtime_verify as rv

    calls: List[tuple] = []

    def _req(method, path, **_k):
        calls.append((method, path))
        return {"status": "canceled"}

    monkeypatch.setattr(
        rv,
        "fetch_open_orders",
        lambda **_k: [
            {
                "id": "day-sell-1",
                "side": "sell",
                "symbol": "TSLA",
                "time_in_force": "day",
                "status": "open",
            }
        ],
    )
    monkeypatch.setattr(rv, "_request", _req)
    report = rv.verify_open_orders(cancel_non_gtc=True, dry_run=False)
    assert report["cancelled"] == ["day-sell-1"]
    assert calls == [("DELETE", "/v2/orders/day-sell-1")]
    assert all(o.get("retry") is False for o in report.get("cancel_outcomes") or [])


def test_n2_malformed_live_mismatch_zero_delete(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    from services import runtime_verify as rv

    calls: List[tuple] = []

    def _req(method, path, **_k):
        calls.append((method, path))
        return None

    with pytest.raises(PaperOrderCancelBlocked):
        rv.cancel_order(
            "bad/id",
            order_side="buy",
            cancellation_capability=CAP_RUNTIME_VERIFY,
            request_fn=_req,
        )
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "1")
    with pytest.raises(PaperOrderCancelBlocked):
        rv.cancel_order(
            "ord-1",
            order_side="buy",
            cancellation_capability=CAP_RUNTIME_VERIFY,
            request_fn=_req,
        )
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(PaperOrderCancelBlocked):
        rv.cancel_order(
            "ord-1",
            order_side="buy",
            cancellation_capability=CAP_RUNTIME_VERIFY,
            request_fn=_req,
        )
    assert calls == []


def test_n2_replay_idempotent_and_outcome_unknown(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    from services import runtime_verify as rv

    calls: List[tuple] = []

    def _req(method, path, **_k):
        calls.append((method, path))
        return {"status": "canceled"}

    first = rv.cancel_order(
        "ord-1",
        order_side="sell",
        order_status="open",
        cancellation_capability=CAP_RUNTIME_VERIFY,
        request_fn=_req,
    )
    assert first.get("status") == "canceled"
    second = rv.cancel_order(
        "ord-1",
        order_side="sell",
        order_status="canceled",
        cancellation_capability=CAP_RUNTIME_VERIFY,
        request_fn=_req,
    )
    assert second.get("idempotent") is True
    assert calls == [("DELETE", "/v2/orders/ord-1")]

    def _boom(method, path, **_k):
        calls.append((method, path))
        raise TimeoutError("timeout")

    with pytest.raises(RuntimeError) as ei:
        rv.cancel_order(
            "ord-2",
            order_side="buy",
            cancellation_capability=CAP_RUNTIME_VERIFY,
            request_fn=_boom,
        )
    assert "OUTCOME_UNKNOWN" in str(ei.value)
    assert calls.count(("DELETE", "/v2/orders/ord-2")) == 1


def _isolate_manage_paths(monkeypatch, tmp_path: Path):
    import services.manage_open_orders as moo

    monkeypatch.setattr(moo, "RESULTS", tmp_path)
    monkeypatch.setattr(moo, "MANAGE_ORDERS_CSV", tmp_path / "manage_orders.csv")
    monkeypatch.setattr(moo, "SMART_MANAGER_JSON", tmp_path / "smart.json")
    monkeypatch.setattr(moo, "STALE_CSV", tmp_path / "stale.csv")
    monkeypatch.setattr(moo, "OUT_JSON", tmp_path / "out.json")
    monkeypatch.setattr(moo, "OUT_CSV", tmp_path / "out.csv")
    monkeypatch.setattr(moo, "OUT_LOG", tmp_path / "out.log")
    monkeypatch.setattr(moo, "LIVE_LOG", tmp_path / "live.log")
    monkeypatch.setattr(
        moo,
        "load_smart_order_manager_config",
        lambda: {
            "stale_after_seconds": 1.0,
            "very_stale_after_seconds": 2.0,
            "distance_reprice_abs": 0.0,
            "reprice_buy_mult": 0.9985,
            "reprice_sell_mult": 1.0015,
            "max_slippage_pct": 0.05,
            "spread_no_cross_buffer_bps": 2.0,
            "min_tick": 0.01,
            "quote_max_age_seconds": 3600.0,
        },
    )
    return moo


def _stale_limit_order(
    *,
    oid: str,
    side: str,
    symbol: str = "TSLA",
    qty: str = "8",
    limit_price: str = "350.00",
) -> Dict[str, Any]:
    ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    return {
        "id": oid,
        "symbol": symbol,
        "side": side,
        "type": "limit",
        "status": "open",
        "qty": qty,
        "filled_qty": "0",
        "limit_price": limit_price,
        "time_in_force": "day",
        "created_at": ts,
        "submitted_at": ts,
        "updated_at": ts,
    }


def _bind_broker(monkeypatch, *, open_orders, order_side_by_id):
    created: Dict[str, Any] = {}

    class _Bound(StatusFakeBroker):
        def __init__(self, mode="paper", **kwargs):
            super().__init__(
                mode=mode,
                open_orders=list(open_orders),
                order_side_by_id=dict(order_side_by_id),
            )
            created["broker"] = self

    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker", _Bound)
    return created


def test_cancel_and_replace_buy_follows_buy_path(tmp_path, monkeypatch):
    store = _seed_then_isolate(monkeypatch, tmp_path, consume=False)
    moo = _isolate_manage_paths(monkeypatch, tmp_path)
    orders = [
        _stale_limit_order(oid="buy-1", side="buy", symbol="AAPL", qty="1", limit_price="300.00")
    ]
    created = _bind_broker(monkeypatch, open_orders=orders, order_side_by_id={"buy-1": "buy"})
    rc = moo.run_smart_order_manager("paper", dry_run=False, ignore_market_closed=True)
    assert rc == 0
    fake = created["broker"]
    assert fake.deletes == ["/v2/orders/buy-1"]
    assert len(fake.submit_payloads) == 1
    assert fake.submit_payloads[0]["side"] == "buy"
    assert _dispatch_count(store / "paper_sell_arm.sqlite3") == 0


def test_cancel_and_replace_sell_blocked_without_authority(tmp_path, monkeypatch):
    _isolate_empty(monkeypatch, tmp_path)
    moo = _isolate_manage_paths(monkeypatch, tmp_path)
    orders = [_stale_limit_order(oid="sell-1", side="sell")]
    created = _bind_broker(monkeypatch, open_orders=orders, order_side_by_id={"sell-1": "sell"})
    rc = moo.run_smart_order_manager(
        "paper", dry_run=False, ignore_market_closed=True, verbose=True
    )
    assert rc == 0
    fake = created["broker"]
    assert fake.deletes == ["/v2/orders/sell-1"]
    assert fake.submit_payloads == []
    csv_text = (tmp_path / "manage_orders.csv").read_text(encoding="utf-8")
    assert "replace_failed" in csv_text


def test_cancel_and_replace_sell_exact_authority_only(tmp_path, monkeypatch):
    _seed_then_isolate(monkeypatch, tmp_path, consume=True)
    moo = _isolate_manage_paths(monkeypatch, tmp_path)
    orders = [_stale_limit_order(oid="sell-1", side="sell", qty="8")]
    created = _bind_broker(monkeypatch, open_orders=orders, order_side_by_id={"sell-1": "sell"})
    rc = moo.run_smart_order_manager("paper", dry_run=False, ignore_market_closed=True)
    assert rc == 0
    fake = created["broker"]
    assert fake.deletes == ["/v2/orders/sell-1"]
    assert len(fake.submit_payloads) == 1
    payload = fake.submit_payloads[0]
    assert payload["side"] == "sell"
    assert payload["symbol"] == "TSLA"
    assert str(payload["qty"]) in {"8", "8.0"}


def test_cancel_decision_cannot_be_reused_as_submit_authority(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = StatusFakeBroker(order_side_by_id={"sell-1": "sell"})
    fake.cancel_order("sell-1")
    assert fake.deletes == ["/v2/orders/sell-1"]
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order(symbol="TSLA", qty=8, side="sell")
    assert fake.submit_payloads == []


def test_changed_sell_slice_fails_closed(tmp_path, monkeypatch):
    _seed_then_isolate(monkeypatch, tmp_path, consume=True)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order(symbol="AAPL", qty=8, side="sell")
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order(symbol="TSLA", qty=99, side="sell")
    fake2 = NetworkFakeAlpacaBroker()
    buy_out = fake2.submit_order(symbol="TSLA", qty=8, side="buy")
    assert buy_out.get("status") == "accepted"
    assert fake.submit_payloads == []
    assert fake2.submit_payloads[0]["side"] == "buy"


def test_n4_cancel_all_guarded(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    fake = NetworkFakeAlpacaBroker(order_side_by_id={"a": "sell"})
    fake.open_orders = [{"id": "a", "side": "sell", "symbol": "TSLA", "qty": "8"}]
    fake.cancel_all_orders()
    assert fake.deletes == ["/v2/orders"]
    assert fake.submit_payloads == []
    fake.mode = "live"
    fake.deletes.clear()
    with pytest.raises(PaperOrderCancelBlocked):
        fake.cancel_all_orders()
    assert fake.deletes == []


def _assert_sell_route_matrix(fake_factory, submit_fn, tmp_path, monkeypatch):
    _isolate_empty(monkeypatch, tmp_path)
    missing = fake_factory()
    with pytest.raises(PaperSellDispatchBlocked):
        submit_fn(missing, side="sell")
    assert missing.submit_payloads == []

    _seed_then_isolate(monkeypatch, tmp_path, consume=True)
    ok_fake = fake_factory()
    out = submit_fn(ok_fake, side="sell")
    assert out.get("status") == "accepted"
    assert len(ok_fake.submit_payloads) == 1
    payload = ok_fake.submit_payloads[0]
    assert payload["side"] == "sell"
    assert payload["symbol"] == "TSLA"
    if "notional" in payload:
        assert float(payload["notional"]) <= APPROVAL_CLI_CAP
    replay = fake_factory()
    with pytest.raises(PaperSellDispatchBlocked):
        submit_fn(replay, side="sell")
    assert replay.submit_payloads == []


def test_n5_submit_order_notional_sell_matrix(tmp_path, monkeypatch):
    def submit(fake, side):
        return fake.submit_order_notional(symbol="TSLA", notional=2660.0, side=side)

    _assert_sell_route_matrix(NetworkFakeAlpacaBroker, submit, tmp_path, monkeypatch)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order_notional(symbol="AAPL", notional=2660.0, side="sell")
    assert fake.submit_payloads == []


def test_n6_submit_bracket_order_notional_sell_matrix(tmp_path, monkeypatch):
    def submit(fake, side):
        return fake.submit_bracket_order_notional(
            symbol="TSLA",
            notional=2660.0,
            side=side,
            take_profit={"limit_price": 300.0},
            stop_loss={"stop_price": 340.0},
        )

    _assert_sell_route_matrix(NetworkFakeAlpacaBroker, submit, tmp_path, monkeypatch)


def test_n7_place_bracket_market_sell_matrix(tmp_path, monkeypatch):
    def submit(fake, side):
        return fake.place_bracket_market(
            symbol="TSLA",
            qty=8,
            side=side,
            take_profit_price=300.0,
            stop_loss_price=340.0,
        )

    _assert_sell_route_matrix(NetworkFakeAlpacaBroker, submit, tmp_path, monkeypatch)


def test_n8_pages_order_desk_notional_path(tmp_path, monkeypatch):
    src = (ROOT / "pages_order_desk.py").read_text(encoding="utf-8")
    assert "broker.submit_order_notional(" in src
    assert "broker.place_bracket_market(" in src
    _isolate_empty(monkeypatch, tmp_path)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order_notional(symbol="TSLA", notional=2660.0, side="sell")
    assert fake.submit_payloads == []
    _seed_then_isolate(monkeypatch, tmp_path, consume=True)
    ok = NetworkFakeAlpacaBroker()
    out = ok.submit_order_notional(symbol="TSLA", notional=2660.0, side="sell")
    assert out.get("status") == "accepted"
    assert ok.submit_payloads[0]["side"] == "sell"
    assert float(ok.submit_payloads[0]["notional"]) <= APPROVAL_CLI_CAP


def test_n9_place_orders_from_csv_sell_matrix(tmp_path, monkeypatch):
    _isolate_empty(monkeypatch, tmp_path)
    from place_orders_from_csv import OrderRow, place_order

    row = OrderRow(
        symbol="TSLA",
        side="SELL",
        qty=8,
        close=332.5,
        order_type="market",
        tif="day",
        client_order_id="csv-sell-1",
    )
    created: Dict[str, Any] = {}

    class _Bound(NetworkFakeAlpacaBroker):
        def __init__(self, mode="paper", **kwargs):
            super().__init__(mode=mode)
            created["broker"] = self

    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker", _Bound)
    with pytest.raises((PaperSellDispatchBlocked, RuntimeError)):
        place_order(row, dry_run=False)
    assert created["broker"].submit_payloads == []

    _seed_then_isolate(monkeypatch, tmp_path, consume=True)
    created.clear()
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker", _Bound)
    status, oid, raw, payload = place_order(row, dry_run=False)
    assert len(created["broker"].submit_payloads) == 1
    assert created["broker"].submit_payloads[0]["side"] == "sell"
    created.clear()
    with pytest.raises((PaperSellDispatchBlocked, RuntimeError)):
        place_order(row, dry_run=False)
    assert created["broker"].submit_payloads == []


def test_buy_notional_not_blocked_by_pending_sell_authority(tmp_path, monkeypatch):
    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(tmp_path, consume=False)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = NetworkFakeAlpacaBroker()
    out = fake.submit_order_notional(symbol="AAPL", notional=100.0, side="buy")
    assert out.get("status") == "accepted"
    assert fake.submit_payloads[0]["side"] == "buy"


def test_buy_row_cannot_become_sell(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = NetworkFakeAlpacaBroker()
    fake.submit_order(symbol="AAPL", qty=1, side="buy")
    assert fake.submit_payloads[0]["side"] == "buy"
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order(symbol="TSLA", qty=8, side="sell")
    assert [p["side"] for p in fake.submit_payloads] == ["buy"]


def test_exact_5000_authorized_sell_notional(tmp_path, monkeypatch):
    _seed_then_isolate(monkeypatch, tmp_path, qty=10.0, price=500.0, consume=True)
    fake = NetworkFakeAlpacaBroker(
        positions=[{"symbol": "TSLA", "qty": "10", "qty_available": "10", "avg_entry_price": "200"}]
    )
    out = fake.submit_order_notional(symbol="TSLA", notional=5000.0, side="sell")
    assert out.get("status") == "accepted"
    assert len(fake.submit_payloads) == 1
    payload = fake.submit_payloads[0]
    assert payload["side"] == "sell"
    assert payload["symbol"] == "TSLA"
    assert float(payload["notional"]) == APPROVAL_CLI_CAP


def test_over_cap_sell_notional_blocked_before_post(tmp_path, monkeypatch):
    store = _seed_then_isolate(monkeypatch, tmp_path, consume=False)
    db = store / "paper_sell_arm.sqlite3"
    before_disp = _dispatch_count(db)
    before_auth = _authority_states(db)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked) as excinfo:
        fake.submit_order_notional(symbol="TSLA", notional=5000.01, side="sell")
    assert excinfo.value.reason == "notional_exceeds_approval_cli_cap"
    assert fake.submit_payloads == []
    assert fake.posts == []
    assert _dispatch_count(db) == before_disp
    assert _authority_states(db) == before_auth
    assert before_auth == ["UNUSED"]


def test_over_cap_bracket_sell_notional_blocked_before_post(tmp_path, monkeypatch):
    store = _seed_then_isolate(monkeypatch, tmp_path, consume=False)
    db = store / "paper_sell_arm.sqlite3"
    before_disp = _dispatch_count(db)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked) as excinfo:
        fake.submit_bracket_order_notional(
            symbol="TSLA",
            notional=5000.01,
            side="sell",
            take_profit={"limit_price": 300.0},
            stop_loss={"stop_price": 340.0},
        )
    assert excinfo.value.reason == "notional_exceeds_approval_cli_cap"
    assert fake.submit_payloads == []
    assert fake.posts == []
    assert _dispatch_count(db) == before_disp
    assert _authority_states(db) == ["UNUSED"]


def test_invalid_notional_fail_closed_before_post(tmp_path, monkeypatch):
    _isolate_empty(monkeypatch, tmp_path)
    fake = NetworkFakeAlpacaBroker()
    invalids = [None, "", "abc", " ", math.nan, math.inf, -math.inf, -1, 0, "5000.01"]
    for value in invalids:
        fake.posts.clear()
        with pytest.raises(PaperSellDispatchBlocked) as excinfo:
            fake.submit_order_notional(symbol="TSLA", notional=value, side="sell")
        assert excinfo.value.reason in {
            "notional_exceeds_approval_cli_cap",
            "notional_missing_or_malformed",
            "notional_non_finite",
            "notional_not_positive",
        }
        assert fake.submit_payloads == []
        assert fake.posts == []


def test_notional_rounding_cannot_exceed_cap(tmp_path, monkeypatch):
    _isolate_empty(monkeypatch, tmp_path)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked) as excinfo:
        fake.submit_order_notional(symbol="TSLA", notional=5000.001, side="sell")
    assert excinfo.value.reason == "notional_exceeds_approval_cli_cap"
    assert fake.posts == []
    with pytest.raises(PaperSellDispatchBlocked) as excinfo2:
        fake.submit_order_notional(symbol="TSLA", notional="5000.010", side="sell")
    assert excinfo2.value.reason == "notional_exceeds_approval_cli_cap"
    assert fake.posts == []


def test_buy_over_cli_cap_not_subject_to_sell_authority(tmp_path, monkeypatch):
    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(tmp_path, consume=False)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = NetworkFakeAlpacaBroker()
    out = fake.submit_order_notional(symbol="AAPL", notional=5000.01, side="buy")
    assert out.get("status") == "accepted"
    assert fake.submit_payloads[0]["side"] == "buy"
    assert float(fake.submit_payloads[0]["notional"]) == 5000.01


def _powershell_parse_error_count(path: Path) -> int:
    quoted = str(path).replace("'", "''")
    cmd = (
        "$e = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{quoted}', [ref]$null, [ref]$e); "
        "if ($null -eq $e) { '0' } else { [string]@($e).Count }"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return int((proc.stdout or "1").strip().splitlines()[-1])


def test_modified_verification_scripts_parse_and_fail_closed():
    root_ps1 = ROOT / "verify_open_orders.ps1"
    tools_ps1 = ROOT / "tools" / "Verify-OpenOrders.ps1"
    assert _powershell_parse_error_count(root_ps1) == 0
    assert _powershell_parse_error_count(tools_ps1) == 0
    for path in (root_ps1, tools_ps1):
        src = path.read_text(encoding="utf-8-sig")
        assert "DISABLED_FAIL_CLOSED" in src
        assert "Invoke-RestMethod -Method DELETE" not in src


def test_static_inventory_reconciliation():
    import importlib.util

    path = ROOT / "scripts" / "evidence" / "phase_198i_sell_remediation_inventory.py"
    spec = importlib.util.spec_from_file_location("phase_198i_sell_remediation_inventory", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    APPENDED_PHASE_198T = mod.APPENDED_PHASE_198T
    FINAL = mod.FINAL
    FORMERLY_UNRESOLVED = mod.FORMERLY_UNRESOLVED
    build_appended_198t_rows = mod.build_appended_198t_rows

    assert list(FORMERLY_UNRESOLVED) == [
        "R3",
        "R4",
        "R5",
        "R6",
        "R23",
        "R33",
        "R34",
        "R36",
        "R37",
        "R38",
        "R39",
        "R40",
        "R41",
        "R42",
        "R43",
    ]
    assert len(FINAL) == 61
    assert "N1" in APPENDED_PHASE_198T and "N2" in APPENDED_PHASE_198T
    rows = build_appended_198t_rows()
    assert all(r["final_disposition"] not in {"UNKNOWN", "BLOCKED_UNRESOLVED"} for r in rows)
    kinds = {r["kind"] for r in rows}
    assert kinds <= {"SUBMIT", "REPLACE_SUBMIT", "PURE_CANCEL", "CANCEL_ALL", "DEAD", "TEST_ONLY"}
    n1 = APPENDED_PHASE_198T["N1"]
    assert n1["kind"] == "PURE_CANCEL"
    assert n1["sell_submission_authority"] is False
    assert n1["cancellation_authorization"] is True


def test_no_remaining_raw_production_delete_without_guard():
    skip_parts = {
        ".venv",
        "venv",
        "tests",
        "tmp",
        "data",
        "models",
        "__pycache__",
        "site-packages",
        "node_modules",
    }
    unguarded = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if any(part in skip_parts or part.startswith("tmp") for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "exit_if_deprecated" in text and "_deprecation_guard()" in text:
            continue
        has_delete = (
            "requests.delete(" in text
            or '_request("DELETE"' in text
            or "_request('DELETE'" in text
            or "method, url" in text
            and "DELETE" in text
            and "cancel_order" in text
        )
        if "requests.delete(" in text or '_request("DELETE"' in text or "_request('DELETE'" in text:
            if "evaluate_paper_order_cancellation" not in text:
                unguarded.append(rel)
        elif "self._delete(" in text and path.name == "broker_alpaca.py":
            if "evaluate_paper_order_cancellation" not in text:
                unguarded.append(rel)
    assert unguarded == []

    for ps1 in (ROOT / "verify_open_orders.ps1", ROOT / "tools" / "Verify-OpenOrders.ps1"):
        src = ps1.read_text(encoding="utf-8", errors="replace")
        assert "Invoke-RestMethod -Method DELETE" not in src
        assert "DISABLED_FAIL_CLOSED" in src


def test_no_remaining_production_sell_post_outside_hardened_boundary():
    """SELL POSTs must still go through AlpacaBroker.submit* which calls enforce_paper_sell."""
    src = (ROOT / "services" / "broker_alpaca.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    submit_fns = {
        "submit_order",
        "submit_order_notional",
        "submit_bracket_order_notional",
        "place_bracket_market",
        "place_bracket",
    }
    found = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AlpacaBroker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in submit_fns:
                    dumped = ast.dump(item)
                    assert "enforce_paper_sell_broker_submission" in dumped or item.name in {
                        "place_bracket_market",
                        "place_bracket",
                    }
                    found.add(item.name)
    assert submit_fns <= found or {"submit_order", "submit_order_notional"} <= found
    # place_bracket_market / place_bracket delegate to submit_order.
    assert "return self.submit_order(" in src


def test_guard_sanitizes_secrets_and_does_not_broker_io():
    from services.paper_order_cancellation_guard import (
        CancellationRequest,
        sanitize_diagnostics,
    )

    dirty = sanitize_diagnostics(
        {"APCA-API-SECRET-KEY": "super-secret", "account_number": "PA-1", "route_id": "N1"}
    )
    assert dirty["APCA-API-SECRET-KEY"] == "[redacted]"
    assert dirty["account_number"] == "[redacted]"
    assert dirty["route_id"] == "N1"
    assert live_execution_is_enabled({"TRITON_ENABLE_LIVE_EXECUTION": "0"}) is False
    decision = evaluate_paper_order_cancellation(
        CancellationRequest(
            broker_mode="paper",
            live_execution_enabled=False,
            route_id="N1",
            cancellation_capability=CAP_MONITOR,
            broker_order_id="ord-1",
            order_side="buy",
        )
    )
    assert decision.allow_delete is True
    unknown = classify_delete_http_outcome(None, timeout=True)
    assert unknown.classification == "OUTCOME_UNKNOWN"
    assert unknown.details.get("retry") is False


def test_cancel_must_not_authorize_submit_intent():
    from services.paper_order_cancellation_guard import CancellationRequest

    blocked = evaluate_paper_order_cancellation(
        CancellationRequest(
            broker_mode="paper",
            live_execution_enabled=False,
            route_id="N1",
            cancellation_capability=CAP_MONITOR,
            broker_order_id="ord-1",
            order_side="sell",
            intent="SUBMIT",
        )
    )
    assert blocked.allow_delete is False
