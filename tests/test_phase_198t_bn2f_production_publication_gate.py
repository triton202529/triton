"""Phase 198T-BN2F2 — production publication authority and gate validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.performance_accounting import (
    EVENT_CUTOVER,
    EVENT_SESSION_CLOSE,
    EVENTS_RELATIVE,
    LOCK_RELATIVE,
    PerformanceAccountingError,
    build_event,
    encode_ledger_suffix,
    load_events,
    plan_ledger_writes,
    publish_ledger_atomic,
    reduce_events,
    sha256_file,
)
from services.performance_evidence_collector import (
    CONFIRM_CUTOVER,
    CONFIRM_SESSION_CLOSE,
    MemoryEvidenceSource,
    PerformanceEvidenceCollector,
    PublicationContext,
    run_cli,
)
from services.performance_publication_authority import (
    BN2E_CUTOVER_TIMESTAMP,
    BN2E_LEDGER_SHA256,
    COMMAND_CUTOVER,
    COMMAND_SESSION_CLOSE,
    EMPTY_LEDGER_SHA256,
    EXPECTED_PRODUCTION_HEAD,
    PRODUCTION_ROOT,
    PublicationAuthority,
    PublicationAuthorityError,
    canonical_path,
    destination_requires_authority,
    is_real_production_path,
    mint_publication_authority,
    passing_git_identity,
    passing_runtime_state,
    passing_session,
    sha256_bytes,
)
from tests.isolation_guard import ISO

pytestmark = pytest.mark.bn2a_isolation

PROD_LEDGER = PRODUCTION_ROOT / EVENTS_RELATIVE
PROD_LOCK = PRODUCTION_ROOT / LOCK_RELATIVE


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


def _source(snapshot=None, session_date: str = "2026-09-10", **extra):
    return MemoryEvidenceSource(
        snapshot=snapshot or _account(),
        hashes=extra.pop("hashes", _hashes()),
        benchmarks=extra.pop("benchmarks", _benches(session_date)),
        **extra,
    )


def _prod_ctx(root: Path, **extra):
    payload = {
        "production_root": root,
        "expected_git_head": EXPECTED_PRODUCTION_HEAD,
        "git_identity": passing_git_identity(),
        "runtime_state": passing_runtime_state(),
    }
    payload.update(extra)
    return PublicationContext(**payload)


def _fake_prod(tmp_path: Path):
    root = tmp_path / "fake_prod"
    events = root / EVENTS_RELATIVE
    lock = root / LOCK_RELATIVE
    events.parent.mkdir(parents=True, exist_ok=True)
    return root, events, lock


def _collector(root: Path, events: Path, lock: Path, source, gate=None):
    return PerformanceEvidenceCollector(
        events_path=events,
        lock_path=lock,
        source=source,
        session_gate=gate or _closed(),
        persist=False,
        allow_production_destination=True,
        production_root=root,
    )


def _readonly_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    fd = os.open(str(path), flags)
    try:
        chunks = []
        while True:
            piece = os.read(fd, 1024 * 1024)
            if not piece:
                break
            chunks.append(piece)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _copy_production_ledger(dest: Path) -> bytes:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = _readonly_file_bytes(PROD_LEDGER)
    dest.write_bytes(data)
    return data


def _cutover_event():
    return build_event(
        EVENT_CUTOVER,
        event_timestamp="2026-09-10T20:05:00+00:00",
        session_date="2026-09-10",
        source="phase_198t_bn2f_test",
        body={
            "as_of_utc": "2026-09-10T20:05:00+00:00",
            "cash": "65663.61",
            "equity": "92358.02",
            "market_value": "26694.41",
            "buying_power": "158021.63",
            "positions": [
                {
                    "symbol": "GE",
                    "qty": "1",
                    "market_value": "26694.41",
                    "average_entry_price": "280.1925",
                    "cost_basis": "280.1925",
                }
            ],
            "not_a_deposit": True,
            "not_a_withdrawal": True,
            "not_profit_or_loss": True,
            "not_a_return": True,
            "account_structure": "long_only_paper",
        },
    )


def _session_close_from_cutover(cutover: dict, session_date: str = "2026-09-11") -> dict:
    body = cutover["body"]
    return build_event(
        EVENT_SESSION_CLOSE,
        event_timestamp=f"{session_date}T20:05:00+00:00",
        session_date=session_date,
        source="phase_198t_bn2f_test",
        body={
            "session_date": session_date,
            "close_kind": "regular_close",
            "cash": body["cash"],
            "market_value": body["market_value"],
            "equity": body["equity"],
            "buying_power": body.get("buying_power") or "158021.63",
            "positions": [
                {
                    "symbol": row["symbol"],
                    "qty": row["qty"],
                    "market_value": row["market_value"],
                    "average_entry_price": row.get("average_entry_price")
                    or row.get("avg_entry_price")
                    or "0",
                    "unrealized_pl": row.get("unrealized_pl") or "0",
                }
                for row in body.get("positions") or []
            ],
            "account_structure": "long_only_paper",
            "source_hashes": {"test": "1"},
        },
    )


def _mint(
    root: Path,
    events_path: Path,
    lock_path: Path,
    records,
    *,
    mode: str,
    command: str,
    confirm: str,
    prior: bytes = b"",
    **extra,
):
    planned, _ = plan_ledger_writes(records, load_events(events_path) if prior else [])
    payload = prior + encode_ledger_suffix(planned)
    kwargs = {
        "resolved_root": root,
        "expected_git_head": EXPECTED_PRODUCTION_HEAD,
        "events_path": events_path,
        "lock_path": lock_path,
        "command": command,
        "publication_mode": mode,
        "confirm_phrase": confirm,
        "payload_sha256": sha256_bytes(payload),
        "existing_ledger_sha256": sha256_bytes(prior) if prior else EMPTY_LEDGER_SHA256,
        "existing_ledger_nbytes": len(prior),
        "required_prefix": prior if mode == "append" else b"",
        "git_identity": passing_git_identity(),
        "runtime_state": passing_runtime_state(),
        "session": passing_session(),
        "evidence_hashes": _hashes(),
    }
    kwargs.update(extra)
    return mint_publication_authority(**kwargs)


def _publish(records, events_path, lock_path, root, authority, *, mode, command, confirm, **extra):
    return publish_ledger_atomic(
        records,
        events_path=events_path,
        lock_path=lock_path,
        mode=mode,
        publication_authority=authority,
        production_root=root,
        expected_git_head=EXPECTED_PRODUCTION_HEAD,
        publication_command=command,
        confirm=confirm,
        **extra,
    )


def test_import_creates_no_files_and_skips_broker():
    import services.performance_accounting as accounting
    import services.performance_evidence_collector as collector
    import services.performance_publication_authority as auth

    for mod in (auth, collector, accounting):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "alpaca" not in text.lower()
        assert "requests." not in text
    assert not (ISO / EVENTS_RELATIVE).exists()
    assert "unforgeable" not in Path(auth.__file__).read_text(encoding="utf-8").lower()


def test_direct_production_call_rejected_without_opening_writable():
    before = _readonly_file_bytes(PROD_LEDGER)
    assert hashlib.sha256(before).hexdigest() == BN2E_LEDGER_SHA256
    with pytest.raises(PerformanceAccountingError) as exc:
        publish_ledger_atomic(
            [_cutover_event()],
            events_path=PROD_LEDGER,
            lock_path=PROD_LOCK,
            mode="create",
        )
    assert exc.value.reason == "missing_publication_authority"
    after = _readonly_file_bytes(PROD_LEDGER)
    assert after == before
    assert "nonce" not in json.dumps(exc.value.details)


def test_authorized_call_succeeds_against_fake_temp_production_root(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    report = _collector(root, events, lock, _source()).publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_prod_ctx(root)
    )
    assert report["writes_performed"] is True
    assert report["publication_authority_issued"] is True
    assert report["publication_schema"] == "198T-BN2F-PRODUCTION-PUBLICATION-GATE-1"
    loaded = load_events(events)
    assert len(loaded) == 1
    assert loaded[0]["event_type"] == EVENT_CUTOVER
    assert not (tmp_path / "portfolio_history.csv").exists()


def test_missing_dummy_forged_reconstructed_serialized_copied_subclassed(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    event = _cutover_event()
    with pytest.raises(PerformanceAccountingError) as missing:
        publish_ledger_atomic(
            [event],
            events_path=events,
            lock_path=lock,
            mode="create",
            production_root=root,
            expected_git_head=EXPECTED_PRODUCTION_HEAD,
        )
    assert missing.value.reason == "missing_publication_authority"

    with pytest.raises(PublicationAuthorityError) as forged_ctor:
        PublicationAuthority(resolved_root=root, events_path=events, lock_path=lock)
    assert forged_ctor.value.reason == "forged_publication_authority"

    dummy = SimpleNamespace(_nonce="forged")
    with pytest.raises(PerformanceAccountingError) as dummy_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            dummy,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert dummy_exc.value.reason == "forged_publication_authority"

    live = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    with pytest.raises(PublicationAuthorityError):
        pickle.dumps(live)
    with pytest.raises(PublicationAuthorityError):
        copy.copy(live)
    with pytest.raises(PublicationAuthorityError):
        copy.deepcopy(live)
    dumped = json.dumps({"command": COMMAND_CUTOVER, "mode": "create"})
    reconstructed = json.loads(dumped)
    with pytest.raises(PerformanceAccountingError) as recon:
        _publish(
            [event],
            events,
            lock,
            root,
            reconstructed,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert recon.value.reason == "forged_publication_authority"

    class Sub(PublicationAuthority):
        def __init__(self, **_kwargs):
            pass

    with pytest.raises(PerformanceAccountingError) as sub_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            Sub(),
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert sub_exc.value.reason == "forged_publication_authority"
    assert "nonce" not in repr(live)
    assert live._nonce not in repr(live)
    assert live._nonce not in str(PublicationAuthorityError("forged_publication_authority"))


def test_expired_reused_and_binding_mismatches_rejected(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    other, other_events, other_lock = _fake_prod(tmp_path / "other")
    event = _cutover_event()
    now = datetime(2026, 9, 10, 23, 30, tzinfo=timezone.utc)
    expired = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
        ttl_seconds=0,
        now=now,
    )
    with pytest.raises(PerformanceAccountingError) as expired_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            expired,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
            now=now,
        )
    assert expired_exc.value.reason == "expired_publication_authority"

    live = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    first = _publish(
        [event],
        events,
        lock,
        root,
        live,
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    assert first["writes_performed"] is True
    with pytest.raises(PerformanceAccountingError) as reused:
        _publish(
            [event],
            events,
            lock,
            root,
            live,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert reused.value.reason == "reused_publication_authority"

    wrong_root = _mint(
        other,
        other_events,
        other_lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    with pytest.raises(PerformanceAccountingError) as root_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            wrong_root,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert root_exc.value.reason == "publication_authority_wrong_root"

    wrong_head = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
        expected_git_head="deadbeef" * 5,
    )
    with pytest.raises(PerformanceAccountingError) as head_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            wrong_head,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert head_exc.value.reason == "publication_authority_wrong_head"


def test_wrong_mode_command_phrase_payload_existing_and_prefix(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    event = _cutover_event()
    cutover_auth = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    with pytest.raises(PerformanceAccountingError) as mode_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            cutover_auth,
            mode="append",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert mode_exc.value.reason == "publication_authority_wrong_mode"

    close_auth = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    with pytest.raises(PerformanceAccountingError) as cmd_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            close_auth,
            mode="create",
            command=COMMAND_SESSION_CLOSE,
            confirm=CONFIRM_CUTOVER,
        )
    assert cmd_exc.value.reason == "publication_authority_wrong_command"

    phrase_auth = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    with pytest.raises(PerformanceAccountingError) as phrase_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            phrase_auth,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_SESSION_CLOSE,
        )
    assert phrase_exc.value.reason == "publication_authority_wrong_phrase"

    payload_auth = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
        payload_sha256="0" * 64,
    )
    with pytest.raises(PerformanceAccountingError) as payload_exc:
        _publish(
            [event],
            events,
            lock,
            root,
            payload_auth,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert payload_exc.value.reason == "publication_authority_wrong_payload"


def test_cutover_authority_cannot_publish_session_close(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    event = _cutover_event()
    auth = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    with pytest.raises(PerformanceAccountingError) as exc:
        _publish(
            [_session_close_from_cutover({"body": event["body"]})],
            events,
            lock,
            root,
            auth,
            mode="append",
            command=COMMAND_SESSION_CLOSE,
            confirm=CONFIRM_SESSION_CLOSE,
        )
    assert exc.value.reason in {
        "publication_authority_wrong_mode",
        "publication_authority_wrong_command",
        "publication_authority_wrong_phrase",
    }
    assert not events.exists()


def test_session_close_authority_cannot_create_cutover(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    original = _copy_production_ledger(events)
    close = _session_close_from_cutover(load_events(events)[0])
    auth = _mint(
        root,
        events,
        lock,
        [close],
        mode="append",
        command=COMMAND_SESSION_CLOSE,
        confirm=CONFIRM_SESSION_CLOSE,
        prior=original,
    )
    with pytest.raises(PerformanceAccountingError) as exc:
        _publish(
            [_cutover_event()],
            events,
            lock,
            root,
            auth,
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=CONFIRM_CUTOVER,
        )
    assert exc.value.reason in {
        "publication_authority_wrong_mode",
        "publication_authority_wrong_command",
        "publication_authority_wrong_phrase",
    }
    assert events.read_bytes() == original


def test_wrong_existing_hash_length_and_prefix(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    original = _copy_production_ledger(events)
    close = _session_close_from_cutover(load_events(events)[0])
    auth = _mint(
        root,
        events,
        lock,
        [close],
        mode="append",
        command=COMMAND_SESSION_CLOSE,
        confirm=CONFIRM_SESSION_CLOSE,
        prior=original,
        existing_ledger_sha256="1" * 64,
    )
    with pytest.raises(PerformanceAccountingError) as hash_exc:
        _publish(
            [close],
            events,
            lock,
            root,
            auth,
            mode="append",
            command=COMMAND_SESSION_CLOSE,
            confirm=CONFIRM_SESSION_CLOSE,
        )
    assert hash_exc.value.reason == "publication_authority_wrong_existing_hash"
    length_auth = _mint(
        root,
        events,
        lock,
        [close],
        mode="append",
        command=COMMAND_SESSION_CLOSE,
        confirm=CONFIRM_SESSION_CLOSE,
        prior=original,
        existing_ledger_nbytes=len(original) + 9,
    )
    with pytest.raises(PerformanceAccountingError) as len_exc:
        _publish(
            [close],
            events,
            lock,
            root,
            length_auth,
            mode="append",
            command=COMMAND_SESSION_CLOSE,
            confirm=CONFIRM_SESSION_CLOSE,
        )
    assert len_exc.value.reason == "publication_authority_wrong_existing_length"
    prefix_auth = _mint(
        root,
        events,
        lock,
        [close],
        mode="append",
        command=COMMAND_SESSION_CLOSE,
        confirm=CONFIRM_SESSION_CLOSE,
        prior=original,
        required_prefix=original + b"x",
    )
    with pytest.raises(PerformanceAccountingError) as prefix_exc:
        _publish(
            [close],
            events,
            lock,
            root,
            prefix_auth,
            mode="append",
            command=COMMAND_SESSION_CLOSE,
            confirm=CONFIRM_SESSION_CLOSE,
        )
    assert prefix_exc.value.reason == "publication_authority_wrong_prefix"
    assert events.read_bytes() == original


def test_concurrent_reuse_allows_at_most_one_attempt(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    event = _cutover_event()
    auth = _mint(
        root,
        events,
        lock,
        [event],
        mode="create",
        command=COMMAND_CUTOVER,
        confirm=CONFIRM_CUTOVER,
    )
    outcomes: list[str] = []

    def worker() -> None:
        try:
            result = _publish(
                [event],
                events,
                lock,
                root,
                auth,
                mode="create",
                command=COMMAND_CUTOVER,
                confirm=CONFIRM_CUTOVER,
            )
            outcomes.append("wrote" if result.get("writes_performed") else "noop")
        except PerformanceAccountingError as exc:
            outcomes.append(exc.reason)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("wrote") == 1
    assert outcomes.count("reused_publication_authority") == 7
    assert events.is_file()


def test_state_change_after_gate_prevents_write(tmp_path: Path, monkeypatch):
    root, events, lock = _fake_prod(tmp_path)
    writes = {"n": 0}

    def track_mkdir(self, *args, **kwargs):
        writes["n"] += 1
        raise AssertionError("mkdir_after_failed_reconfirm")

    monkeypatch.setattr(Path, "mkdir", track_mkdir, raising=False)
    report = _collector(root, events, lock, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER,
        context=_prod_ctx(
            root,
            runtime_inspector=lambda _root: {
                **passing_runtime_state(),
                "runtime_stopped": False,
            },
        ),
    )
    assert report["writes_performed"] is False
    assert report["blocked_reason"] == "runtime_state_changed"
    assert writes["n"] == 0
    assert not events.exists()


def test_wrong_confirmation_phrase_rejected(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    report = _collector(root, events, lock, _source()).safe_publish_cutover(
        confirm="PLEASE PUBLISH", context=_prod_ctx(root)
    )
    assert report["blocked_reason"] == "wrong_confirmation_phrase"
    assert report["writes_performed"] is False
    assert not events.exists()


@pytest.mark.parametrize(
    "ctx_extra,reason",
    [
        ({"expected_git_head": ""}, "expected_git_head_required"),
        ({"git_identity": passing_git_identity("0" * 40)}, "git_head_mismatch"),
        (
            {
                "git_identity": {
                    "head": EXPECTED_PRODUCTION_HEAD,
                    "branch": "topic",
                    "index_empty": True,
                }
            },
            "git_branch_not_main",
        ),
        (
            {
                "git_identity": {
                    "head": EXPECTED_PRODUCTION_HEAD,
                    "branch": "main",
                    "index_empty": False,
                }
            },
            "git_index_not_empty",
        ),
        (
            {"runtime_state": {**passing_runtime_state(), "runtime_stopped": False}},
            "runtime_not_stopped",
        ),
        (
            {"runtime_state": {**passing_runtime_state(), "watchdog_stopped": False}},
            "watchdog_not_stopped",
        ),
        (
            {"runtime_state": {**passing_runtime_state(), "streamlit_stopped": False}},
            "streamlit_not_stopped",
        ),
        (
            {"runtime_state": {**passing_runtime_state(), "ports_closed": False}},
            "production_ports_open",
        ),
        ({"writers_idle": False}, "writers_active"),
        ({"freshness_lock_present": True}, "freshness_lock_present"),
        ({"live_execution_permitted": True}, "production_or_live_mode"),
        ({"can_mutate_orders": True}, "order_mutation_capability"),
    ],
)
def test_failed_gate_prevents_publisher_invocation(tmp_path: Path, monkeypatch, ctx_extra, reason):
    root, events, lock = _fake_prod(tmp_path)
    called = {"n": 0}

    def forbidden(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("publisher_invoked")

    monkeypatch.setattr("services.performance_evidence_collector.publish_ledger_atomic", forbidden)
    report = _collector(root, events, lock, _source()).safe_publish_cutover(
        confirm=CONFIRM_CUTOVER, context=_prod_ctx(root, **ctx_extra)
    )
    assert report["blocked_reason"] == reason
    assert report["writes_performed"] is False
    assert called["n"] == 0
    assert not events.exists()


def test_path_aliases_cannot_bypass_production_detection(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    dotted = events.parent / ".." / "results" / events.name
    assert destination_requires_authority(dotted, lock, root)
    alias = tmp_path / "alias_prod"
    created = os.system(f'mklink /J "{alias}" "{root}"')
    if created != 0:
        pytest.skip("junction creation unavailable")
    alias_events = alias / EVENTS_RELATIVE
    assert destination_requires_authority(alias_events, lock, root)
    assert canonical_path(alias_events) == canonical_path(events)
    foreign = tmp_path / "other_root"
    foreign.mkdir()
    assert not is_real_production_path(alias_events)
    assert not destination_requires_authority(alias_events, lock, foreign)


def test_copied_production_ledger_hash_reducer_and_prefix(tmp_path: Path):
    copied = tmp_path / "copied_ledger" / "performance_accounting_events.jsonl"
    lock = tmp_path / "copied_ledger" / "performance_accounting_events.lock"
    original = _copy_production_ledger(copied)
    assert hashlib.sha256(original).hexdigest() == BN2E_LEDGER_SHA256
    assert sha256_file(copied) == BN2E_LEDGER_SHA256
    events = load_events(copied)
    assert len(events) == 1
    assert events[0]["event_type"] == EVENT_CUTOVER
    assert events[0]["event_timestamp"] == BN2E_CUTOVER_TIMESTAMP
    state = reduce_events(events)
    assert state["cutover"] is not None
    assert state["cutover"]["publication_status"] == "PUBLISHED_OPENING_CHECKPOINT"
    with pytest.raises(PerformanceAccountingError) as second:
        publish_ledger_atomic(
            [_cutover_event()],
            events_path=copied,
            lock_path=lock,
            mode="create",
        )
    assert second.value.reason == "existing_ledger_refused"
    assert copied.read_bytes() == original
    with pytest.raises(PerformanceAccountingError) as dup:
        reduce_events(events + [_cutover_event()])
    assert dup.value.reason == "duplicate_cutover"
    appended = publish_ledger_atomic(
        [_session_close_from_cutover(events[0])],
        events_path=copied,
        lock_path=lock,
        mode="append",
    )
    assert appended["writes_performed"] is True
    after = copied.read_bytes()
    assert after.startswith(original)
    assert after != original


def test_successful_authorized_append_preserves_exact_prefix(tmp_path: Path):
    root, events, lock = _fake_prod(tmp_path)
    original = _copy_production_ledger(events)
    session_date = "2026-09-11"
    collector = _collector(
        root,
        events,
        lock,
        _source(
            _account(as_of_utc="2026-09-11T20:05:00+00:00", session_date=session_date),
            session_date=session_date,
        ),
        _closed(session_date=session_date),
    )
    report = collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=_prod_ctx(root))
    assert report["writes_performed"] is True
    assert events.read_bytes().startswith(original)
    assert hashlib.sha256(_readonly_file_bytes(PROD_LEDGER)).hexdigest() == BN2E_LEDGER_SHA256


def test_interrupted_append_leaves_destination_unchanged(tmp_path: Path):
    copied = tmp_path / "interrupt" / "performance_accounting_events.jsonl"
    lock = tmp_path / "interrupt" / "performance_accounting_events.lock"
    original = _copy_production_ledger(copied)
    events = load_events(copied)

    def boom(stage, path, payload):
        if stage == "before_rename":
            raise PerformanceAccountingError("publication_interrupted")

    with pytest.raises(PerformanceAccountingError) as exc:
        publish_ledger_atomic(
            [_session_close_from_cutover(events[0])],
            events_path=copied,
            lock_path=lock,
            mode="append",
            fail_hook=boom,
        )
    assert exc.value.reason == "publication_interrupted"
    assert copied.read_bytes() == original
    assert not copied.with_name(copied.name + ".tmp").exists()


def test_default_dry_run_writes_nothing(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    lock = tmp_path / "events.lock"
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
                "calendar": "XNYS",
            }
        ),
        encoding="utf-8",
    )
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
    assert report["writes_performed"] is False
    assert report["publication_permitted"] is False
    assert not events.exists()


def test_production_cutover_remains_unchanged():
    digest = hashlib.sha256(_readonly_file_bytes(PROD_LEDGER)).hexdigest()
    assert digest == BN2E_LEDGER_SHA256
    text = _readonly_file_bytes(PROD_LEDGER).decode("utf-8")
    assert text.count("ACCOUNTING_CUTOVER") == 1
    assert BN2E_CUTOVER_TIMESTAMP in text
    assert not (ISO / EVENTS_RELATIVE).exists()


def test_temp_publication_still_works_without_authority(tmp_path: Path):
    events = tmp_path / "performance_accounting_events.jsonl"
    lock = tmp_path / "performance_accounting_events.lock"
    collector = PerformanceEvidenceCollector(
        events_path=events,
        lock_path=lock,
        source=_source(),
        session_gate=_closed(),
        persist=False,
    )
    report = collector.publish_cutover(confirm=CONFIRM_CUTOVER, context=PublicationContext())
    assert report["writes_performed"] is True
    assert report.get("publication_authority_issued") is False
    assert load_events(events)[0]["event_type"] == EVENT_CUTOVER


def test_cli_production_path_without_root_still_rejected():
    report = run_cli(
        [
            "--publish-cutover",
            "--events-path",
            str(PROD_LEDGER),
            "--lock-path",
            str(PROD_LOCK),
            "--confirm",
            CONFIRM_CUTOVER,
        ]
    )
    assert report["blocked_reason"] == "forbidden_production_path"
    assert report["writes_performed"] is False
    assert hashlib.sha256(_readonly_file_bytes(PROD_LEDGER)).hexdigest() == BN2E_LEDGER_SHA256
