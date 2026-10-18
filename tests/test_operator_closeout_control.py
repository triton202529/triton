"""Focused tests for the protected closeout control. Temporary paths and injected fixtures only."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from services.operator_closeout_control import (
    CONFIRMATION_PHRASE,
    HAPPY_PATH,
    ORDER_MUTATION_METHODS,
    STATE_ALREADY_CERTIFIED,
    STATE_BROKER_READ_FAILED,
    STATE_CERTIFIED,
    STATE_DRAIN_FAILED,
    STATE_PUBLICATION_FAILED,
    STATE_RECONCILE_FAILED,
    STATE_REFUSED,
    STATE_STOP_FAILED,
    STATE_STREAMLIT_STOP_FAILED,
    STATE_WAITING,
    ReadOnlyBroker,
    classify_dashboard_processes,
    dashboard_stop_argv,
    execute_closeout,
    exit_current_dashboard,
    launch_detached_worker,
    review_closeout,
    run_publishing_worker,
)
from services.performance_evidence_collector import (
    CONFIRM_CUTOVER,
    CONFIRM_SESSION_CLOSE,
    InjectedSessionGate,
    MemoryEvidenceSource,
    PerformanceEvidenceCollector,
    PublicationContext,
)
from services.performance_publication_authority import passing_runtime_state

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 17, 21, 0, tzinfo=timezone.utc)
AS_OF = datetime(2026, 10, 16, 20, 30, tzinfo=timezone.utc)
CLOSE = "2026-10-16T20:00:00+00:00"


class Broker:
    def __init__(
        self, *, orders=None, cash="100.00", equity="150.00", mv="50.00", qty="1", boom=False
    ):
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self.orders = [] if orders is None else orders
        self.cash = cash
        self.equity = equity
        self.mv = mv
        self.qty = qty
        self.boom = boom
        self.calls = []

    def get_account(self):
        self.calls.append("get_account")
        if self.boom:
            raise RuntimeError("broker down")
        return {
            "id": "PAPER-ACCOUNT",
            "cash": self.cash,
            "equity": self.equity,
            "long_market_value": self.mv,
            "buying_power": self.cash,
        }

    def get_positions(self):
        self.calls.append("get_positions")
        side = "short" if str(self.qty).startswith("-") else "long"
        return [
            {
                "symbol": "SPCX",
                "qty": self.qty,
                "market_value": self.mv,
                "avg_entry_price": "40",
                "side": side,
            }
        ]

    def get_open_orders(self):
        self.calls.append("get_open_orders")
        return list(self.orders)

    def submit_order(self, *args, **kwargs):
        self.calls.append("submit_order")

    def cancel_order(self, *args, **kwargs):
        self.calls.append("cancel_order")

    def replace_order(self, *args, **kwargs):
        self.calls.append("replace_order")

    def close_position(self, *args, **kwargs):
        self.calls.append("close_position")

    def place_order(self, *args, **kwargs):
        self.calls.append("place_order")


def _session(reason="regular_close", state="CLOSED", date="2026-10-16", close=CLOSE):
    return {
        "calendar": "XNYS",
        "session_state": state,
        "reason": reason,
        "session_date": date,
        "session_close_et": close,
    }


def _bars(date="2026-10-16"):
    return {
        "SPY": {"symbol": "SPY", "session_date": date, "close": "500"},
        "QQQ": {"symbol": "QQQ", "session_date": date, "close": "400"},
    }


def _git():
    return {"git_identity_valid": True, "index_empty": True, "failure_reason": None, "head": "abc"}


def _classes(state="NOT_RUNNING"):
    return {"continuous": {"state": state}, "watchdog": {"state": state}}


def _refresh(date="2026-10-16"):
    return {"target_session": date, "passed": True}


def _run(root, broker, publisher, killer, **overrides):
    published = []
    stopped = []

    def shutdown(**kwargs):
        stopped.append(kwargs)
        return {
            "result": "ALREADY_STOPPED",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
        }

    def worker(path):
        run_publishing_worker(
            root=root,
            receipt_path=path,
            runtime_inspector=lambda _root: passing_runtime_state(),
            git_inspector=lambda _root: _git(),
            processes=[{"pid": 42, "command_line": "python -m streamlit run view_results.py"}],
            killer=killer,
            publisher=publisher,
            own_pid=999,
        )

    def default_publisher(_root, receipt):
        published.append(receipt["session_date"])
        return {"writes_performed": True, "payload_hash": "hash-1", "blocked_reason": None}

    kwargs = dict(
        confirmation_phrase=CONFIRMATION_PHRASE,
        root=root,
        session=_session(),
        broker=broker,
        benchmarks=_bars(),
        classifications=_classes(),
        live_execution_blocked=True,
        daily_refresh=_refresh(),
        manual_gate={"ok": True, "reason": "store_absent"},
        duplicate_close=False,
        git_identity=_git(),
        as_of=AS_OF,
        now=NOW,
        shutdown_fn=shutdown,
        worker_runner=worker,
    )
    kwargs.update(overrides)
    if overrides.get("publisher") is not None:
        publisher = overrides.pop("publisher")
        kwargs.pop("publisher", None)
    result = execute_closeout(**kwargs)
    return result, stopped, published, broker.calls


def _publisher_box(calls):
    def publisher(_root, receipt):
        calls.append("publish")
        return {"writes_performed": True, "payload_hash": "hash-1", "blocked_reason": None}

    return publisher


def test_read_only_broker_has_no_callable_mutation_method():
    broker = Broker()
    wrapped = ReadOnlyBroker(broker)
    for name in ORDER_MUTATION_METHODS:
        assert callable(getattr(wrapped, name, None)) is False
    assert wrapped.get_open_orders() == []
    assert "submit_order" not in broker.calls


def test_dashboard_stop_is_not_a_force_kill_and_worker_is_not_streamlit():
    argv = dashboard_stop_argv(15)
    assert argv == ["taskkill", "/PID", "15"]
    assert "/F" not in argv
    launched = {}

    def popen(argv, **kwargs):
        launched["argv"] = argv
        launched["flags"] = kwargs.get("creationflags")

        class Proc:
            pid = 7

        return Proc()

    with TemporaryDirectory() as tmp:
        launch_detached_worker(Path(tmp), Path(tmp) / "receipt.json", popen=popen)
    assert "streamlit" not in " ".join(launched["argv"][1:]).lower()
    assert "--worker" in launched["argv"]
    plan = classify_dashboard_processes(
        [
            {"pid": 1, "command_line": "powershell run_continuous_loop.ps1"},
            {"pid": 2, "command_line": "python -m streamlit run view_results.py"},
            {"pid": 3, "command_line": "python -m streamlit run_continuous_loop.ps1"},
        ]
    )
    assert plan["ok"] is False
    assert plan["dashboard_pids"] == [2]
    refused = exit_current_dashboard(
        command="python run_continuous_loop.ps1", exit_fn=lambda code: None
    )
    assert refused["ok"] is False
    exited = {}
    allowed = exit_current_dashboard(
        command="python -m streamlit run view_results.py",
        exit_fn=lambda code: exited.setdefault("code", code),
    )
    assert allowed["ok"] is True
    assert exited["code"] == 0


def test_regular_and_early_close_visit_every_state_and_retry_is_idempotent():
    for reason in ("regular_close", "early_close"):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            killed = []
            broker = Broker()
            result, stopped, _published, broker_calls = _run(
                root,
                broker,
                _publisher_box(calls),
                lambda pid: killed.append(pid),
                session=_session(reason=reason),
            )
            assert result["state"] == STATE_CERTIFIED
            assert result["states_visited"] == list(HAPPY_PATH)
            assert result["close_kind"] == reason
            assert result["session_close_display"].endswith("ET")
            assert result["orders_submitted"] is False
            assert result["orders_cancelled"] is False
            assert result["orders_replaced"] is False
            assert result["orders_modified"] is False
            assert result["forced_termination_used"] is False
            assert killed == [42]
            assert calls == ["publish"]
            assert stopped
            assert not any(name in ORDER_MUTATION_METHODS for name in broker_calls)
            again, _, _, calls_again = _run(
                root,
                Broker(),
                _publisher_box(calls),
                lambda pid: None,
                session=_session(reason=reason),
            )
            assert again["replayed"] is True
            assert again["closeout_id"] == result["closeout_id"]
            assert again["state"] == STATE_CERTIFIED
            assert calls == ["publish"]
            assert not any(name in ORDER_MUTATION_METHODS for name in calls_again)


def test_session_states_other_than_certified_close_are_refused():
    cases = [
        ("regular_session_open", "OPEN", "2026-10-19"),
        ("pre_open", "CLOSED", "2026-10-19"),
        ("weekend", "CLOSED", None),
        ("holiday", "CLOSED", None),
    ]
    for reason, state, date in cases:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            result, stopped, _, _ = _run(
                root,
                Broker(),
                _publisher_box(calls),
                lambda pid: None,
                session=_session(reason=reason, state=state, date=date),
            )
            assert result["state"] == STATE_REFUSED
            assert result["reason"] == "session_not_certified_close"
            assert calls == []
            assert stopped == []


def test_open_orders_stale_lock_unsafe_process_and_active_guard_do_not_stop():
    cases = [
        ({"broker": Broker(orders=[{"id": "open-1"}])}, "open_orders", STATE_REFUSED),
        (
            {"classifications": _classes("STALE_LOCK_NO_PROCESS")},
            "stale_lock",
            STATE_REFUSED,
        ),
        ({"classifications": _classes("AMBIGUOUS")}, "unsafe_process", STATE_REFUSED),
        ({"start_guard_active": True}, "start_guard_active", STATE_REFUSED),
        ({"shutdown_guard_active": True}, "shutdown_guard_active", STATE_REFUSED),
    ]
    for overrides, reason, state in cases:
        with TemporaryDirectory() as tmp:
            calls = []
            broker = overrides.pop("broker", Broker())
            result, stopped, _, broker_calls = _run(
                Path(tmp),
                broker,
                _publisher_box(calls),
                lambda pid: None,
                **overrides,
            )
            assert result["state"] == state
            assert result["reason"] == reason
            assert calls == []
            assert stopped == []
            assert not any(name in ORDER_MUTATION_METHODS for name in broker_calls)


def test_broker_read_identity_snapshot_refresh_manual_and_benchmark_failures():
    old = NOW - timedelta(hours=48)
    cases = [
        ({"broker": Broker(boom=True)}, STATE_BROKER_READ_FAILED, "broker_read_failed"),
        (
            {"broker": Broker(cash="100.00", equity="200.00", mv="50.00")},
            STATE_RECONCILE_FAILED,
            "account_identity",
        ),
        (
            {"broker": Broker(qty="-2", mv="-50.00", equity="50.00", cash="100.00")},
            STATE_RECONCILE_FAILED,
            "short_structure",
        ),
        (
            {"as_of": datetime(2026, 10, 16, 19, 0, tzinfo=timezone.utc)},
            STATE_REFUSED,
            "snapshot_before_session_close",
        ),
        (
            {
                "as_of": old,
                "now": NOW,
                "session": _session(close="2026-10-14T20:00:00+00:00", date="2026-10-14"),
                "benchmarks": _bars("2026-10-14"),
                "daily_refresh": _refresh("2026-10-14"),
            },
            STATE_REFUSED,
            "snapshot_older_than_36h",
        ),
        (
            {"daily_refresh": {"target_session": "2026-10-16", "passed": False}},
            STATE_WAITING,
            "daily_refresh_not_passed",
        ),
        ({"daily_refresh": {}}, STATE_WAITING, "daily_refresh_receipt_missing"),
        (
            {"manual_gate": {"ok": False, "reason": "incomplete_terminal_evidence"}},
            STATE_REFUSED,
            "manual_exit_gate",
        ),
        (
            {"benchmarks": {"SPY": {"close": "1", "session_date": "2026-10-16"}, "QQQ": {}}},
            STATE_REFUSED,
            "missing_benchmark",
        ),
    ]
    for overrides, state, reason in cases:
        with TemporaryDirectory() as tmp:
            calls = []
            broker = overrides.pop("broker", Broker())
            result, stopped, _, broker_calls = _run(
                Path(tmp), broker, _publisher_box(calls), lambda pid: None, **overrides
            )
            assert result["state"] == state, result
            assert result["reason"] == reason
            assert calls == []
            assert stopped == []
            assert not any(name in ORDER_MUTATION_METHODS for name in broker_calls)


def test_drain_and_stop_failures_do_not_publish():
    def drain(**kwargs):
        return {
            "result": "FAILED",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
        }

    def timeout(**kwargs):
        return {
            "result": "SHUTDOWN_TIMED_OUT",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
        }

    expectations = [
        (drain, STATE_DRAIN_FAILED, "FAILED"),
        (timeout, STATE_STOP_FAILED, "SHUTDOWN_TIMED_OUT"),
    ]
    for shutdown, state, reason in expectations:
        with TemporaryDirectory() as tmp:
            calls = []
            result, _, _, broker_calls = _run(
                Path(tmp),
                Broker(),
                _publisher_box(calls),
                lambda pid: None,
                shutdown_fn=shutdown,
            )
            assert result["state"] == state
            assert result["reason"] == reason
            assert calls == []
            assert not any(name in ORDER_MUTATION_METHODS for name in broker_calls)


def test_dashboard_still_present_and_failed_publisher_fail_closed():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        calls = []

        def worker(path):
            run_publishing_worker(
                root=root,
                receipt_path=path,
                runtime_inspector=lambda _root: {
                    **passing_runtime_state(),
                    "streamlit_stopped": False,
                    "ports_closed": False,
                },
                git_inspector=lambda _root: _git(),
                processes=[],
                killer=lambda pid: None,
                publisher=lambda _root, receipt: calls.append("publish"),
            )

        result, _, _, broker_calls = _run(
            root, Broker(), None, lambda pid: None, worker_runner=worker
        )
        assert result["state"] == STATE_STREAMLIT_STOP_FAILED
        assert result["reason"] == "dashboard_or_ports_still_present"
        assert calls == []
        assert not any(name in ORDER_MUTATION_METHODS for name in broker_calls)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        calls = []

        def publisher(_root, receipt):
            calls.append("publish")
            return {"writes_performed": False, "blocked_reason": "publisher_rejected"}

        result, _, _, _ = _run(root, Broker(), publisher, lambda pid: None)
        assert result["state"] == STATE_PUBLICATION_FAILED
        assert result["reason"] == "publisher_rejected"
        assert result["writes_performed"] is False
        assert calls == ["publish"]


def test_duplicate_session_close_never_publishes_a_second_close():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        calls = []
        result, stopped, _, broker_calls = _run(
            root,
            Broker(),
            _publisher_box(calls),
            lambda pid: None,
            duplicate_close=True,
        )
        assert result["state"] == STATE_ALREADY_CERTIFIED
        assert result["writes_performed"] is False
        assert calls == []
        assert stopped == []
        assert not any(name in ORDER_MUTATION_METHODS for name in broker_calls)
        again, _, _, _ = _run(
            root, Broker(), _publisher_box(calls), lambda pid: None, duplicate_close=True
        )
        assert again["replayed"] is True
        assert calls == []


def test_real_publisher_duplicate_appends_nothing_further():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        events = root / "events.jsonl"
        lock = root / "events.lock"
        session_date = "2026-10-16"
        snap = {
            "cash": "100.00",
            "market_value": "50.00",
            "equity": "150.00",
            "buying_power": "100.00",
            "as_of_utc": "2026-10-16T20:30:00Z",
            "session_date": session_date,
            "account_structure": "long_only_paper",
            "positions": [
                {
                    "symbol": "SPCX",
                    "qty": "1",
                    "market_value": "50.00",
                    "average_entry_price": "40",
                    "unrealized_pl": "10",
                }
            ],
        }
        source = MemoryEvidenceSource(
            snapshot=snap,
            benchmarks={
                "SPY:2026-10-16": {"close": "500", "session_date": session_date},
                "QQQ:2026-10-16": {"close": "400", "session_date": session_date},
            },
            hashes={
                "account": "a",
                "positions": "b",
                "orders": "c",
                "activities": "d",
                "SPY": "e",
                "QQQ": "f",
            },
        )
        gate = InjectedSessionGate(
            {
                "session_state": "CLOSED",
                "reason": "regular_close",
                "session_date": session_date,
                "session_close_et": CLOSE,
                "calendar": "XNYS",
            }
        )
        collector = PerformanceEvidenceCollector(
            events_path=events,
            lock_path=lock,
            source=source,
            session_gate=gate,
            persist=False,
        )
        context = PublicationContext(
            mode="PAPER_ONLY",
            live_execution_permitted=False,
            broker_adapter_mode="GET_ONLY",
            can_mutate_orders=False,
            writers_idle=True,
            freshness_lock_present=False,
        )
        cutover = collector.publish_cutover(confirm=CONFIRM_CUTOVER, context=context)
        assert cutover["writes_performed"] is True
        first = collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=context)
        second = collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=context)
        assert first["writes_performed"] is True
        assert second["blocked_reason"] == "duplicate_session_close"
        assert second["writes_performed"] is False
        text = events.read_text(encoding="utf-8")
        assert text.count(f"session_close:{session_date}") == 1


def test_review_lists_blockers_and_writes_nothing():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        before = sorted(path.name for path in root.rglob("*"))
        review = review_closeout(
            root=root,
            session=_session(reason="weekend", state="CLOSED", date=None),
            broker=Broker(),
            classifications=_classes(),
            benchmarks=_bars(),
            daily_refresh=_refresh(),
            manual_gate={"ok": True, "reason": "store_absent"},
            git_identity=_git(),
            as_of=AS_OF,
            now=NOW,
        )
        after = sorted(path.name for path in root.rglob("*"))
        assert before == after
        assert review["writes_performed"] is False
        assert review["ok"] is False
        assert any(
            row["code"] == "session_regular_or_early_close" and not row["ok"]
            for row in review["conditions"]
        )
        assert review["confirmation_phrase"] == "CLOSE OUT AND STOP TRITON"


def test_phrase_mismatch_does_not_write_or_mutate():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        broker = Broker()
        result, stopped, _, broker_calls = _run(
            root, broker, lambda *_args: None, lambda pid: None, confirmation_phrase="stop"
        )
        assert result["state"] == STATE_REFUSED
        assert result["reason"] == "phrase_mismatch"
        assert result["writes_performed"] is False
        assert stopped == []
        assert broker_calls == []
        assert list(root.rglob("*.json")) == []


def test_live_execution_and_non_paper_broker_are_refused():
    with TemporaryDirectory() as tmp:
        result, stopped, _, _ = _run(
            Path(tmp), Broker(), lambda *_a: None, lambda pid: None, live_execution_blocked=False
        )
        assert result["state"] == STATE_REFUSED
        assert result["reason"] == "live_execution_not_blocked"
        assert stopped == []
    with TemporaryDirectory() as tmp:
        broker = Broker()
        broker.mode = "live"
        result, stopped, _, calls = _run(Path(tmp), broker, lambda *_a: None, lambda pid: None)
        assert result["reason"] == "paper_broker_required"
        assert calls == []
        assert stopped == []


def test_console_exposes_the_closeout_control():
    console = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    screen = (ROOT / "dashboard" / "operator_closeout.py").read_text(encoding="utf-8")
    assert "render_closeout_control" in console
    assert "Close out & stop TRITON" in screen
    assert "CLOSE OUT AND STOP TRITON" in screen
    module = (ROOT / "services" / "operator_closeout_control.py").read_text(encoding="utf-8")
    for name in ORDER_MUTATION_METHODS:
        assert f".{name}(" not in module
    assert "taskkill" in module
    assert "CREATE_NO_WINDOW" in module or "DETACHED_PROCESS" in module


def test_worker_skips_publisher_when_ledger_already_has_the_close():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        events = root / "data" / "results" / "performance_accounting_events.jsonl"
        events.parent.mkdir(parents=True)
        events.write_text(
            '{"logical_key": "session_close:2026-10-16"}\n',
            encoding="utf-8",
        )
        calls = []
        killed = []
        result, _, _, _ = _run(
            root,
            Broker(),
            lambda _root, receipt: calls.append("publish"),
            lambda pid: killed.append(pid),
        )
        assert result["state"] == STATE_ALREADY_CERTIFIED
        assert result["writes_performed"] is False
        assert calls == []
        assert killed == [42]


def test_receipt_json_has_no_secret_fields():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        result, _, _, _ = _run(root, Broker(), _publisher_box([]), lambda pid: None)
        text = json.dumps(result)
        assert "PAPER-ACCOUNT" not in text
        assert "api_key" not in text
        assert result["account_fingerprint"]
