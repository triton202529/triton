"""Publisher tolerates only a verified manual-exit store. Temporary roots only."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from services.basis_unproven_hold import enforce_basis_unproven_hold_on_order_post
from services.manual_portfolio_exit import (
    STORE_DB_NAME,
    build_manual_exit_review,
    confirm_manual_paper_exit,
    evaluate_manual_exit_close_gate,
    note_proven_terminal_receipts,
)
from services.paper_sell_arm_runtime_enforcement import enforce_paper_sell_broker_submission
from services.paper_sell_arm_store import close_manual_exit_sqlite, write_connection
from services.performance_evidence_collector import (
    CollectorError,
    InjectedSessionGate,
    PerformanceEvidenceCollector,
    PublicationContext,
)
from services.performance_publication_authority import (
    CLASS_VERIFIED_MANUAL_EXIT_STORE,
    MANUAL_EXIT_STORE_RELATIVE,
    REASON_ARTIFACT_UNEXPLAINED,
    _manual_exit_store_publication_decision,
    evaluate_git_publication_identity,
    passing_runtime_state,
)

ROOT = Path(__file__).resolve().parents[1]
HEAD = "830947bce6e81493732af3e41bbd9c6f27409644"
STORE_LINE = f"?? {MANUAL_EXIT_STORE_RELATIVE}/"
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "ledger" / "ledger.parquet",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
)
PRODUCTION_STORE = ROOT / "data" / "governance" / "manual_portfolio_exit"


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _tree(root: Path) -> dict[str, tuple[str, int]]:
    if not root.exists():
        return {}
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[path.relative_to(root).as_posix()] = _fingerprint(path)
    return found


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    before = {path: _fingerprint(path) for path in PRODUCTION}
    before_store = _tree(PRODUCTION_STORE)
    yield
    assert {path: _fingerprint(path) for path in PRODUCTION} == before
    assert _tree(PRODUCTION_STORE) == before_store


class Broker:
    def __init__(self):
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self.orders = []

    def get_positions(self):
        return [{"symbol": "AAPL", "qty": "10", "current_price": "20"}]

    def submit_order(self, **kwargs):
        enforce_paper_sell_broker_submission(
            mode=self.mode,
            side=kwargs["side"],
            source="broker_alpaca.submit_order",
            symbol=kwargs["symbol"],
            qty=float(kwargs["qty"]),
            client_order_id=kwargs.get("client_order_id"),
        )
        enforce_basis_unproven_hold_on_order_post(
            {
                "symbol": kwargs["symbol"],
                "side": kwargs["side"],
                "qty": kwargs["qty"],
                "client_order_id": kwargs.get("client_order_id"),
            },
            source="broker_alpaca._post",
        )
        self.orders.append(kwargs)
        return {
            "id": "ord-1",
            "status": "accepted",
            "client_order_id": kwargs.get("client_order_id"),
        }

    def get_order(self, order_id):
        raise AssertionError(order_id)


def _recovered(tmp_path: Path):
    root = tmp_path / "repo"
    store = root / MANUAL_EXIT_STORE_RELATIVE
    review = build_manual_exit_review(
        {"symbol": "AAPL", "qty": "10", "current_price": "20", "avg_entry_price": "12"},
        blocked_symbols=[],
    )
    done = confirm_manual_paper_exit(
        review,
        broker=Broker(),
        store_root=store,
        ledger_root=root,
        live_execution_permitted=False,
        blocked_symbols=[],
    )
    assert done["submitted"] is True
    aligned = note_proven_terminal_receipts(
        [
            {
                "event_type": "ORDER_TERMINAL_RECONCILED",
                "terminal_status_proven": True,
                "broker_terminal_status": "FILLED",
                "broker_order_id": done["broker_order_id"],
                "client_order_id": done["client_order_id"],
                "symbol": done["symbol"],
                "filled_qty": done["selected_qty"],
                "extra": {"raw_broker_status": "filled"},
            }
        ],
        root=root,
    )
    assert aligned["ok"] is True
    assert evaluate_manual_exit_close_gate(store)["reason"] == "manual_exit_store_consistent"
    return root, store, done


def _identity(root: Path, lines: list[str]):
    return evaluate_git_publication_identity(
        root=root,
        head=HEAD,
        branch="main",
        index_empty=True,
        porcelain_lines=lines,
        staged_paths=[],
        verify_filesystem=True,
    )


def _preflight(root: Path, identity: dict, *, streamlit_stopped: bool, ports_closed: bool):
    events = root / "data" / "results" / "performance_accounting_events.jsonl"
    lock = root / "data" / "results" / "performance_accounting_events.lock"
    events.parent.mkdir(parents=True, exist_ok=True)
    collector = PerformanceEvidenceCollector(
        events_path=events,
        lock_path=lock,
        source=object(),
        session_gate=InjectedSessionGate(
            {
                "session_state": "CLOSED",
                "reason": "regular_close",
                "session_date": "2026-10-16",
                "calendar": "XNYS",
            }
        ),
        persist=False,
        allow_production_destination=True,
        production_root=root,
    )
    runtime = passing_runtime_state()
    runtime["streamlit_stopped"] = streamlit_stopped
    runtime["ports_closed"] = ports_closed
    context = PublicationContext(
        production_root=root,
        expected_git_head=HEAD,
        git_identity=identity,
        runtime_state=runtime,
    )
    return collector._assert_production_publication_gates(
        context,
        {
            "session_state": "CLOSED",
            "reason": "regular_close",
            "session_date": "2026-10-16",
            "calendar": "XNYS",
        },
    )


def _assert_store_blocks(root: Path, store: Path, reason: str) -> None:
    gate = evaluate_manual_exit_close_gate(store)
    assert gate["ok"] is False
    assert gate["reason"] == reason
    before = _tree(store)
    identity = _identity(root, [STORE_LINE])
    assert identity["git_identity_valid"] is False
    assert identity["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED
    assert identity["rejected_paths"][0]["path"].rstrip("/") == MANUAL_EXIT_STORE_RELATIVE
    with pytest.raises(CollectorError) as exc:
        _preflight(root, identity, streamlit_stopped=True, ports_closed=True)
    assert exc.value.reason == REASON_ARTIFACT_UNEXPLAINED
    assert _tree(store) == before


def test_decision_reads_only_the_exact_directory():
    source = inspect.getsource(_manual_exit_store_publication_decision)
    assert "norm != MANUAL_EXIT_STORE_RELATIVE" in source
    for banned in (
        "note_proven_terminal_receipts",
        "write_connection",
        "close_manual_exit_sqlite",
        "startswith(",
    ):
        assert banned not in source


def test_recovered_store_passes_artifact_gate_and_keeps_runtime_stops(tmp_path):
    root, store, _done = _recovered(tmp_path)
    before = _tree(store)
    identity = _identity(root, [STORE_LINE])
    assert identity["git_identity_valid"] is True
    assert identity["failure_reason"] is None
    assert identity["tolerated_paths"] == [
        {
            "path": f"{MANUAL_EXIT_STORE_RELATIVE}/",
            "classification": CLASS_VERIFIED_MANUAL_EXIT_STORE,
            "x": "?",
            "y": "?",
        }
    ]
    with pytest.raises(CollectorError) as streamlit:
        _preflight(root, identity, streamlit_stopped=False, ports_closed=True)
    assert streamlit.value.reason == "streamlit_not_stopped"
    with pytest.raises(CollectorError) as ports:
        _preflight(root, identity, streamlit_stopped=True, ports_closed=False)
    assert ports.value.reason == "production_ports_open"
    assert _tree(store) == before


def test_stale_identity_blocks_publication(tmp_path):
    root, store, done = _recovered(tmp_path)
    path = store / "order_identities" / f"{done['broker_order_id']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["broker_status"] = "pending_new"
    payload["reconciliation_state"] = "submission_recorded"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _assert_store_blocks(root, store, "stale_or_mismatched_state")


def test_open_challenge_blocks_publication(tmp_path):
    root, store, _done = _recovered(tmp_path)
    db_path = store / STORE_DB_NAME
    with write_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE sell_challenges
            SET state='OPEN', consumed_at=NULL, consuming_authority_id=NULL
            """
        )
    close_manual_exit_sqlite(db_path)
    _assert_store_blocks(root, store, "stale_or_mismatched_state")


def test_pre_submit_permit_blocks_publication(tmp_path):
    root = tmp_path / "repo"
    store = root / MANUAL_EXIT_STORE_RELATIVE
    review = build_manual_exit_review(
        {"symbol": "AAPL", "qty": "10", "current_price": "20", "avg_entry_price": "12"},
        blocked_symbols=[],
    )
    done = confirm_manual_paper_exit(
        review,
        broker=Broker(),
        store_root=store,
        ledger_root=root,
        live_execution_permitted=False,
        blocked_symbols=[],
    )
    assert done["submitted"] is True
    _assert_store_blocks(root, store, "mutable_pre_submit_permit")


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_sqlite_sidecar_blocks_publication(tmp_path, suffix):
    root, store, _done = _recovered(tmp_path)
    Path(str(store / STORE_DB_NAME) + suffix).write_bytes(b"")
    _assert_store_blocks(root, store, "sqlite_sidecar_unexplained")


def test_unexpected_child_blocks_publication(tmp_path):
    root, store, _done = _recovered(tmp_path)
    (store / "notes.txt").write_text("unrelated", encoding="utf-8")
    _assert_store_blocks(root, store, "unrelated_child")


def test_unrelated_untracked_artifact_still_blocks(tmp_path):
    root, store, _done = _recovered(tmp_path)
    other = root / "data" / "governance" / "not_manual_exit.txt"
    other.write_text("x", encoding="utf-8")
    before = _tree(store)
    identity = _identity(
        root,
        [STORE_LINE, "?? data/governance/not_manual_exit.txt"],
    )
    assert identity["git_identity_valid"] is False
    assert identity["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED
    assert any(
        row["classification"] == CLASS_VERIFIED_MANUAL_EXIT_STORE
        for row in identity["tolerated_paths"]
    )
    assert any(
        row["path"] == "data/governance/not_manual_exit.txt" for row in identity["rejected_paths"]
    )
    with pytest.raises(CollectorError) as exc:
        _preflight(root, identity, streamlit_stopped=True, ports_closed=True)
    assert exc.value.reason == REASON_ARTIFACT_UNEXPLAINED
    assert _tree(store) == before


def test_parent_and_nested_paths_are_not_allowlisted(tmp_path):
    root, store, _done = _recovered(tmp_path)
    receipt = next((store / "requests").glob("*.json"))
    nested = receipt.relative_to(root).as_posix()
    for line in ("?? data/", "?? data/governance/"):
        identity = _identity(root, [line])
        assert identity["git_identity_valid"] is False
        assert identity["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED
        assert identity["tolerated_paths"] == []
    both = _identity(root, [STORE_LINE, f"?? {nested}"])
    assert both["git_identity_valid"] is False
    assert any(row["path"] == nested for row in both["rejected_paths"])
    assert any(
        row["classification"] == CLASS_VERIFIED_MANUAL_EXIT_STORE for row in both["tolerated_paths"]
    )


def test_absent_manual_exit_directory_preserves_the_old_path(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert not (root / MANUAL_EXIT_STORE_RELATIVE).exists()
    clean = _identity(root, [])
    assert clean["git_identity_valid"] is True
    assert clean["tolerated_paths"] == []
    operational = _identity(root, ["?? data/governance/execution_gate_decisions.json"])
    assert operational["git_identity_valid"] is True
    assert operational["tolerated_paths"][0]["classification"] == "KNOWN_MUTABLE_OPERATIONAL_DATA"
    mystery = _identity(root, ["?? data/results/mystery.bin"])
    assert mystery["git_identity_valid"] is False
    assert mystery["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED
    listed_but_missing = _identity(root, [STORE_LINE])
    assert listed_but_missing["git_identity_valid"] is False
    assert listed_but_missing["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED
    assert listed_but_missing["tolerated_paths"] == []
