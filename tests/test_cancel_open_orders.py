"""Age-only cancel_open_orders must not wipe orders newer than the cutoff."""

from datetime import datetime, timedelta, timezone

from services.broker_alpaca import AlpacaBroker


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Broker:
    def __init__(self, orders):
        self.orders = orders
        self.cancelled = []

    def list_orders(self, **_kwargs):
        return list(self.orders)

    def _normalize_order(self, order):
        return dict(order)

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {}


def _orders():
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "fresh",
            "symbol": "AAPL",
            "submitted_at": _iso(now - timedelta(minutes=5)),
        },
        {
            "id": "stale",
            "symbol": "MSFT",
            "submitted_at": _iso(now - timedelta(days=10)),
        },
        {
            "id": "fresh-wl",
            "symbol": "V",
            "submitted_at": _iso(now - timedelta(minutes=5)),
        },
        {
            "id": "stale-wl",
            "symbol": "KO",
            "submitted_at": _iso(now - timedelta(days=10)),
        },
    ]


def _ids(rows):
    return sorted(row["id"] for row in rows)


def test_no_filters_cancels_nothing():
    broker = _Broker(_orders())
    out = AlpacaBroker.cancel_open_orders(broker, dry_run=True)
    assert out == []
    assert broker.cancelled == []


def test_older_than_days_keeps_fresh_orders():
    broker = _Broker(_orders())
    out = AlpacaBroker.cancel_open_orders(broker, older_than_days=1, dry_run=True)
    assert _ids(out) == ["stale", "stale-wl"]
    assert all(row.get("dry_run") is True for row in out)
    assert broker.cancelled == []


def test_cutoff_keeps_orders_after_cutoff():
    broker = _Broker(_orders())
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(days=1))
    out = AlpacaBroker.cancel_open_orders(broker, cutoff=cutoff, dry_run=True)
    assert _ids(out) == ["stale", "stale-wl"]


def test_whitelist_only_keeps_named_symbols():
    broker = _Broker(_orders())
    out = AlpacaBroker.cancel_open_orders(broker, whitelist=["V", "KO"], dry_run=True)
    assert _ids(out) == ["fresh", "stale"]


def test_whitelist_and_age_cancels_old_whitelist_and_all_others():
    broker = _Broker(_orders())
    out = AlpacaBroker.cancel_open_orders(
        broker, whitelist=["V", "KO"], older_than_days=1, dry_run=True
    )
    assert _ids(out) == ["fresh", "stale", "stale-wl"]


def test_unparsed_timestamp_is_not_treated_as_stale():
    broker = _Broker(
        [{"id": "notime", "symbol": "AAPL", "submitted_at": "not-a-timestamp"}]
    )
    out = AlpacaBroker.cancel_open_orders(broker, older_than_days=0, dry_run=True)
    assert out == []


def test_live_cancel_only_hits_selected_orders():
    broker = _Broker(_orders())
    AlpacaBroker.cancel_open_orders(broker, older_than_days=1, dry_run=False)
    assert sorted(broker.cancelled) == ["stale", "stale-wl"]
