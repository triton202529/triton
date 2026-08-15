"""Phase 198I-YR — canonical live DB path, concurrent arm, read-only snapshot.

PAPER ONLY. Isolated tmp_path. No repository data/live or data/governance SQLite.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.paper_sell_arm_authority import (
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    READINESS_PACKET_NAME,
    arm_reviewed_sells,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT
from services.paper_sell_arm_store import (
    SCHEMA_USER_VERSION,
    inspect_sqlite_arm_snapshot,
    open_connection,
    persist_open_challenge,
    resolve_runtime_sqlite_path,
    write_connection,
)
from tests._phase_198i_sell_fakes import SESSION_OPEN_NOW, block_real_network, readiness_packet

ROOT = Path(__file__).resolve().parents[1]
NOW = SESSION_OPEN_NOW
REPO_LIVE_DB = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
REPO_GOV_DB = ROOT / "data" / "governance" / "paper_sell_arm.sqlite3"


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    for key in (
        "TRITON_PAPER_SELL_ARM_STORE_ROOT",
        "TRITON_PAPER_SELL_ARM_RESULTS_ROOT",
        "TRITON_PAPER_SELL_ARM_DB",
        "RESULTS_ROOT",
        "TRITON_PAPER_SELL_ARM_STORE",
    ):
        monkeypatch.delenv(key, raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)
    assert not REPO_LIVE_DB.is_file()
    assert not REPO_GOV_DB.is_file()


def _packet(results: Path, now=NOW):
    pkt = readiness_packet(now=now)
    pkt["generated_at"] = now.isoformat()
    (results / READINESS_PACKET_NAME).write_text(__import__("json").dumps(pkt), encoding="utf-8")
    return pkt


def test_canonical_resolver_is_data_live():
    from services.paper_sell_arm_store import DEFAULT_DB_NAME, ROOT

    expected = ROOT / "data" / "live" / DEFAULT_DB_NAME
    assert expected.as_posix().endswith("data/live/paper_sell_arm.sqlite3")
    isolated = Path("C:/tmp/isolated/paper_sell_arm.sqlite3")
    assert resolve_runtime_sqlite_path(db_path=isolated) == isolated


def test_readonly_snapshot_absent_does_not_create_files(tmp_path: Path):
    target = tmp_path / "empty_dir"
    target.mkdir()
    before = {p.name for p in target.iterdir()}
    snap = inspect_sqlite_arm_snapshot(target / "paper_sell_arm.sqlite3", now=NOW)
    after = {p.name for p in target.iterdir()}
    assert snap["reason"] == "sqlite_absent"
    assert snap["db_present"] is False
    assert before == after


def test_readonly_snapshot_existing_does_not_mutate(tmp_path: Path):
    pkt = _packet(tmp_path)
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=pkt,
        now=NOW,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    db = tmp_path / "paper_sell_arm.sqlite3"
    with write_connection(db) as conn:
        ch_before = conn.execute("SELECT state FROM sell_challenges").fetchone()["state"]
        audit_before = conn.execute("SELECT COUNT(*) FROM sell_audit_events").fetchone()[0]
    names_before = sorted(p.name for p in tmp_path.iterdir())
    size_before = db.stat().st_size
    for _ in range(3):
        snap = inspect_sqlite_arm_snapshot(db, now=NOW)
        assert snap["latest_challenge"]["state"] == "OPEN"
        assert snap["authority_count"] == 0
    names_after = sorted(p.name for p in tmp_path.iterdir())
    assert names_before == names_after
    assert db.stat().st_size == size_before
    with write_connection(db) as conn:
        ch_after = conn.execute("SELECT state FROM sell_challenges").fetchone()["state"]
        audit_after = conn.execute("SELECT COUNT(*) FROM sell_audit_events").fetchone()[0]
    assert ch_before == ch_after == "OPEN"
    assert audit_before == audit_after
    assert chal["challenge_id"]


def test_concurrent_arm_one_open_challenge(tmp_path: Path):
    pkt = _packet(tmp_path)
    chal_template = prepare_confirmation_challenge(
        store_root=tmp_path / "template",
        readiness_packet=pkt,
        now=NOW,
        db_path=tmp_path / "template" / "paper_sell_arm.sqlite3",
    )
    db = tmp_path / "paper_sell_arm.sqlite3"
    results: list = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _click(idx: int):
        doc = dict(chal_template)
        doc["challenge_id"] = f"sell-chal-race-{idx:04d}-" + "x" * 8
        barrier.wait(timeout=10)
        try:
            persist_open_challenge(db, challenge=doc, now=NOW)
            with lock:
                results.append(("ok", idx))
        except Exception as exc:
            with lock:
                results.append(("err", getattr(exc, "reason", type(exc).__name__)))

    t1 = threading.Thread(target=_click, args=(1,))
    t2 = threading.Thread(target=_click, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    snap = inspect_sqlite_arm_snapshot(db, now=NOW)
    assert snap["challenge_count"] == 1
    assert snap["open_challenges"][0]["state"] == "OPEN"
    assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    assert _ACTIVE_PERMIT.get() is None
    statuses = [r[0] for r in results]
    assert statuses.count("ok") == 1
    assert any(
        r[0] == "err" and r[1] == "sqlite_duplicate_open_challenge" for r in results
    ), results


def test_concurrent_button_callbacks(tmp_path: Path):
    _packet(tmp_path)
    outcomes = []
    lock = threading.Lock()

    def _arm():
        r = arm_reviewed_sells(
            store_root=tmp_path,
            results_root=tmp_path,
            now=NOW,
            paper_mode=True,
            live_blocked=True,
            db_path=tmp_path / "paper_sell_arm.sqlite3",
        )
        with lock:
            outcomes.append(r)

    t1 = threading.Thread(target=_arm)
    t2 = threading.Thread(target=_arm)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    snap = inspect_sqlite_arm_snapshot(tmp_path / "paper_sell_arm.sqlite3", now=NOW)
    assert snap["challenge_count"] == 1
    assert snap["authority_count"] == 0
    oks = [o for o in outcomes if o.get("ok")]
    dens = [o for o in outcomes if not o.get("ok")]
    assert len(oks) == 1
    assert dens and dens[0]["reason"] == "duplicate_click"


def test_override_poisoning_blocks_production_buttons(tmp_path: Path, monkeypatch):
    from services.paper_sell_arm_store import PRODUCTION_PATH_OVERRIDE_ENVS

    for key in PRODUCTION_PATH_OVERRIDE_ENVS:
        for other in PRODUCTION_PATH_OVERRIDE_ENVS:
            monkeypatch.delenv(other, raising=False)
        override_dir = tmp_path / f"override_{key.lower()}"
        override_dir.mkdir(exist_ok=True)
        monkeypatch.setenv(key, str(override_dir))
        r1 = arm_reviewed_sells(now=NOW, paper_mode=True, live_blocked=True)
        r2 = confirm_and_arm_paper_sells(
            confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
            now=NOW,
            paper_mode=True,
            live_blocked=True,
        )
        assert r1["ok"] is False and r1["reason"] == "configuration_blocked", key
        assert r2["ok"] is False and r2["reason"] == "configuration_blocked", key
        assert not (override_dir / "paper_sell_arm.sqlite3").exists()
        assert not (tmp_path / "paper_sell_arm.sqlite3").exists()
        assert not REPO_LIVE_DB.is_file()
        assert not REPO_GOV_DB.is_file()
        msg = (r1.get("message") or "") + (r2.get("message") or "")
        assert str(override_dir) not in msg
        assert "override" not in msg.lower()
        monkeypatch.delenv(key, raising=False)


def test_dependency_injection_still_works_when_env_set(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRITON_PAPER_SELL_ARM_STORE_ROOT", str(tmp_path / "poison"))
    _packet(tmp_path)
    db = tmp_path / "paper_sell_arm.sqlite3"
    r1 = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=db,
    )
    assert r1["ok"] is True
    r2 = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=r1["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=db,
    )
    assert r2["ok"] is True
    assert not (tmp_path / "poison" / "paper_sell_arm.sqlite3").exists()


def test_schema_user_version_is_v2(tmp_path: Path):
    db = tmp_path / "paper_sell_arm.sqlite3"
    conn = open_connection(db)
    ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
    conn.close()
    assert ver == SCHEMA_USER_VERSION == 2
