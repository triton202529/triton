"""Phase 198T-BN2H1 — Git publication-gate semantics. TEMP repositories only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from services.performance_accounting import (
    EVENT_CUTOVER,
    EVENT_SESSION_CLOSE,
    EVENTS_RELATIVE,
    LOCK_RELATIVE,
    load_events,
    plan_ledger_writes,
    reduce_events,
    sha256_file,
)
from services.performance_evidence_collector import (
    CONFIRM_CUTOVER,
    MemoryEvidenceSource,
    PerformanceEvidenceCollector,
    PublicationContext,
    run_cli,
)
from services.performance_publication_authority import (
    CLASS_KNOWN_ACCOUNTING_LEDGER_OR_LOCK,
    CLASS_KNOWN_MUTABLE_OPERATIONAL_DATA,
    CLASS_KNOWN_PAPER_ORDER_LEDGER_OR_LOCK,
    CLASS_KNOWN_TEMP_EVIDENCE,
    CLASS_UNTRACKED_TEST,
    EXPECTED_PRODUCTION_HEAD,
    REASON_ARTIFACT_UNEXPLAINED,
    REASON_INDEX_NOT_EMPTY,
    REASON_STATUS_CONFLICT,
    REASON_STATUS_UNPARSEABLE,
    REASON_TRACKED_CODE_DIRTY,
    REASON_UNTRACKED_CODE,
    REASON_UNTRACKED_TEST,
    evaluate_git_publication_identity,
    inspect_git_identity,
    passing_git_identity,
    passing_runtime_state,
)
from tests.isolation_guard import ISO

pytestmark = pytest.mark.bn2a_isolation

BN2H = Path(r"C:\Users\akimw\AppData\Local\Temp\phase_198T_BN2H_20260911_184130")
PROD = Path(r"C:\Users\akimw\triton")
FIVE_OPS = (
    "data/governance/execution_gate_authority_matrix.json",
    "data/governance/execution_gate_decisions.json",
    "data/governance/execution_gate_evidence.json",
    "data/governance/symbol_hydration_quarantine.json",
    "data/results/exit_logic_refinement_validation.json",
)
HEAD = "ba40497ea6ed8ad6997df017cf098e690db5828a"
LEDGER_SHA = "42868be678c4ed0bb75a3b527fa412d7e7c96b2061eb69b51f410bf8e061ec95"
APPROVED_FINAL = "9227fb2c802261434877860d250df4a45575c141a6ec6605a955afd5b11a73d6"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "services").mkdir()
    (repo / "scripts").mkdir()
    (repo / "dashboard").mkdir()
    (repo / "tests").mkdir()
    (repo / "services" / "performance_accounting.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("ok\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.email=bn2h1@example.invalid",
        "-c",
        "user.name=BN2H1",
        "commit",
        "-m",
        "init",
    )
    return repo


def _eval(
    porcelain: list[str],
    *,
    staged: list[str] | None = None,
    index_empty: bool | None = None,
    root: Path | None = None,
    verify: bool = False,
    head: str = HEAD,
):
    staged_paths = staged if staged is not None else []
    empty = (not staged_paths) if index_empty is None else index_empty
    return evaluate_git_publication_identity(
        root=root or Path("."),
        head=head,
        branch="main",
        index_empty=empty,
        porcelain_lines=porcelain,
        staged_paths=staged_paths,
        verify_filesystem=verify,
    )


def _source():
    return MemoryEvidenceSource(
        snapshot={
            "as_of_utc": "2026-09-11T22:47:19+00:00",
            "session_date": "2026-09-11",
            "cash": "66447.22",
            "market_value": "25558.78",
            "equity": "92006.00",
            "buying_power": "337313.45",
            "positions": [
                {
                    "symbol": "GE",
                    "qty": "1",
                    "market_value": "25558.78",
                    "average_entry_price": "1",
                    "unrealized_pl": "0",
                    "side": "long",
                }
            ],
        },
        hashes={
            "account": "a",
            "positions": "p",
            "orders": "o",
            "activities": "c",
            "SPY": "s",
            "QQQ": "q",
        },
        benchmarks={
            "SPY:2026-09-11": {"close": "1", "adj_close": "1"},
            "QQQ:2026-09-11": {"close": "1", "adj_close": "1"},
        },
    )


def test_isolated_worktree_name():
    assert "isolated" in ISO.name.lower()


def test_clean_index_and_worktree_passes():
    identity = _eval([])
    assert identity["index_empty"] is True
    assert identity["tracked_worktree_safe"] is True
    assert identity["untracked_worktree_safe"] is True
    assert identity["publication_code_matches_head"] is True
    assert identity["git_identity_valid"] is True
    assert identity["failure_reason"] is None


def test_five_known_dirty_operational_jsons_pass():
    porcelain = [f" M {path}" for path in FIVE_OPS]
    identity = _eval(porcelain)
    assert identity["index_empty"] is True
    assert identity["git_identity_valid"] is True
    assert {row["path"] for row in identity["tolerated_paths"]} == set(FIVE_OPS)
    assert all(
        row["classification"] == CLASS_KNOWN_MUTABLE_OPERATIONAL_DATA
        for row in identity["tolerated_paths"]
    )


def test_accounting_and_paper_ledger_fixtures_pass():
    porcelain = [
        " M data/results/performance_accounting_events.jsonl",
        " M data/results/performance_accounting_events.lock",
        " M data/results/paper_order_events.jsonl",
        " M data/results/paper_order_events.lock",
    ]
    identity = _eval(porcelain)
    assert identity["git_identity_valid"] is True
    classes = {row["classification"] for row in identity["tolerated_paths"]}
    assert CLASS_KNOWN_ACCOUNTING_LEDGER_OR_LOCK in classes
    assert CLASS_KNOWN_PAPER_ORDER_LEDGER_OR_LOCK in classes


def test_classified_temp_evidence_passes():
    porcelain = ["?? tmp/phase_198t_bn2h/evidence.json", "?? tmp/notes.txt"]
    identity = _eval(porcelain)
    assert identity["git_identity_valid"] is True
    assert all(
        row["classification"] == CLASS_KNOWN_TEMP_EVIDENCE for row in identity["tolerated_paths"]
    )


def test_staged_source_fails_index_not_empty():
    identity = _eval(["M  services/foo.py"], staged=["services/foo.py"])
    assert identity["index_empty"] is False
    assert identity["failure_reason"] == REASON_INDEX_NOT_EMPTY
    assert identity["git_identity_valid"] is False


def test_staged_data_file_fails_index_not_empty():
    identity = _eval(
        ["A  data/governance/execution_gate_authority_matrix.json"],
        staged=["data/governance/execution_gate_authority_matrix.json"],
    )
    assert identity["index_empty"] is False
    assert identity["failure_reason"] == REASON_INDEX_NOT_EMPTY


def test_unstaged_tracked_service_fails_tracked_code_dirty():
    identity = _eval([" M services/performance_accounting.py"])
    assert identity["index_empty"] is True
    assert identity["tracked_worktree_safe"] is False
    assert identity["failure_reason"] == REASON_TRACKED_CODE_DIRTY
    assert identity["publication_code_matches_head"] is False


def test_unstaged_tracked_script_and_dashboard_fail():
    for path in ("scripts/run_continuous_loop.ps1", "dashboard/app.py"):
        identity = _eval([f" M {path}"])
        assert identity["index_empty"] is True
        assert identity["tracked_worktree_safe"] is False
        assert identity["failure_reason"] == REASON_TRACKED_CODE_DIRTY


def test_untracked_service_script_dashboard_hook_and_root_executable_fail():
    cases = (
        "?? services/extra.py",
        "?? scripts/helper.ps1",
        "?? dashboard/page.py",
        "?? hooks/pre-commit",
        "?? view_results.py",
    )
    for line in cases:
        identity = _eval([line])
        assert identity["index_empty"] is True
        assert identity["untracked_worktree_safe"] is False
        assert identity["failure_reason"] == REASON_UNTRACKED_CODE


def test_untracked_test_has_distinct_reason():
    identity = _eval(["?? tests/test_phase_195h_phase_195g_ancestry.py"])
    assert identity["index_empty"] is True
    assert identity["failure_reason"] == REASON_UNTRACKED_TEST
    assert identity["rejected_paths"][0]["classification"] == CLASS_UNTRACKED_TEST


def test_unexplained_artifact_fails():
    identity = _eval(["?? manage_orders.csv"])
    assert identity["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED
    assert identity["git_identity_valid"] is False


def test_conflict_rename_deletion_typechange_malformed_and_submodule_fail():
    assert _eval(["UU services/conflict.py"])["failure_reason"] == REASON_STATUS_CONFLICT
    assert (
        _eval(["R  services/old.py -> services/new.py"])["failure_reason"] == REASON_STATUS_CONFLICT
    )
    assert _eval([" D services/gone.py"])["failure_reason"] == REASON_STATUS_CONFLICT
    assert _eval(["T  services/changed.py"])["failure_reason"] == REASON_STATUS_CONFLICT
    assert _eval(["XX"])["failure_reason"] == REASON_STATUS_UNPARSEABLE
    assert _eval(["S  vendor/lib"])["failure_reason"] == REASON_STATUS_CONFLICT


def test_tmp_python_is_untracked_code_not_temp_evidence():
    identity = _eval(["?? tmp/_extract_198i_routes_once.py"])
    assert identity["failure_reason"] == REASON_UNTRACKED_CODE
    assert identity["index_empty"] is True


def test_bn2h_empty_index_is_not_mislabeled_index_failure():
    porcelain = [f" M {path}" for path in FIVE_OPS] + ["?? tmp/ok.json"]
    identity = _eval(porcelain)
    assert identity["index_empty"] is True
    assert identity["failure_reason"] is None
    assert identity["git_identity_valid"] is True


def test_fake_212_classified_mutable_only_passes():
    porcelain = [f" M {path}" for path in FIVE_OPS]
    porcelain += [
        " M data/results/performance_accounting_events.jsonl",
        " M data/results/paper_order_events.jsonl",
    ]
    porcelain += [f"?? tmp/evidence_{i:03d}.json" for i in range(205)]
    assert len(porcelain) == 212
    identity = _eval(porcelain)
    assert identity["index_empty"] is True
    assert identity["git_identity_valid"] is True
    assert identity["failure_reason"] is None
    assert len(identity["tolerated_paths"]) == 212


def test_fake_212_with_untracked_test_fails():
    porcelain = [f"?? tmp/evidence_{i:03d}.json" for i in range(211)]
    porcelain.append("?? tests/test_uir003f_dba_trim_trace_contracts.py")
    identity = _eval(porcelain)
    assert identity["index_empty"] is True
    assert identity["failure_reason"] == REASON_UNTRACKED_TEST
    assert identity["git_identity_valid"] is False


def test_traversal_path_rejected(tmp_path: Path):
    identity = _eval(
        ["?? data/results/../../services/shadow.py"],
        root=tmp_path,
        verify=True,
    )
    assert identity["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED


def test_junction_allowed_path_fails(tmp_path: Path):
    repo = tmp_path / "jroot"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text("{}", encoding="utf-8")
    (repo / "tmp").mkdir(parents=True)
    link = repo / "tmp" / "escaped"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stdout
    identity = _eval(["?? tmp/escaped/secret.json"], root=repo, verify=True)
    assert identity["git_identity_valid"] is False
    assert identity["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED


def test_inspect_live_temp_repo_clean_and_staged(tmp_path: Path):
    repo = _init_repo(tmp_path)
    identity = inspect_git_identity(repo)
    assert identity["index_empty"] is True
    assert identity["git_identity_valid"] is True
    extra = repo / "services" / "new.py"
    extra.write_text("Y = 2\n", encoding="utf-8")
    _git(repo, "add", "services/new.py")
    dirty = inspect_git_identity(repo)
    assert dirty["index_empty"] is False
    assert dirty["failure_reason"] == REASON_INDEX_NOT_EMPTY


def test_failed_git_gate_never_invokes_publisher(tmp_path: Path, monkeypatch):
    root = tmp_path / "fake_prod"
    events = root / EVENTS_RELATIVE
    lock = root / LOCK_RELATIVE
    events.parent.mkdir(parents=True)
    called = {"n": 0}

    def forbidden(*_a, **_k):
        called["n"] += 1
        raise AssertionError("publisher_invoked")

    monkeypatch.setattr("services.performance_evidence_collector.publish_ledger_atomic", forbidden)
    identity = _eval(["?? tests/test_extra.py"], head=EXPECTED_PRODUCTION_HEAD)
    ctx = PublicationContext(
        production_root=root,
        expected_git_head=EXPECTED_PRODUCTION_HEAD,
        git_identity=identity,
        runtime_state=passing_runtime_state(),
    )
    from services.performance_evidence_collector import InjectedSessionGate

    collector = PerformanceEvidenceCollector(
        events_path=events,
        lock_path=lock,
        source=_source(),
        session_gate=InjectedSessionGate(
            {
                "session_state": "CLOSED",
                "reason": "regular_close",
                "session_date": "2026-09-11",
                "calendar": "XNYS",
                "session_close_et": "2026-09-11T16:00:00-04:00",
            }
        ),
        persist=False,
        allow_production_destination=True,
        production_root=root,
    )
    report = collector.safe_publish_cutover(confirm=CONFIRM_CUTOVER, context=ctx)
    assert report["blocked_reason"] == REASON_UNTRACKED_TEST
    assert report["writes_performed"] is False
    assert called["n"] == 0
    assert not events.exists()


def test_default_dry_run_writes_nothing(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    lock = tmp_path / "events.lock"
    session = tmp_path / "session.json"
    snap = tmp_path / "snap.json"
    session.write_text(
        json.dumps(
            {
                "session_state": "CLOSED",
                "reason": "regular_close",
                "session_date": "2026-09-11",
                "calendar": "XNYS",
                "session_close_et": "2026-09-11T16:00:00-04:00",
            }
        ),
        encoding="utf-8",
    )
    snap.write_text(json.dumps(_source().snapshot), encoding="utf-8")
    report = run_cli(
        [
            "--dry-run-session-close",
            "--events-path",
            str(events),
            "--lock-path",
            str(lock),
            "--snapshot-json",
            str(snap),
            "--session-json",
            str(session),
        ]
    )
    assert report["writes_performed"] is False
    assert not events.exists()


def test_copied_production_ledger_prefix_unchanged_and_read_only():
    ledger = BN2H / "ops_backup" / "data__results__performance_accounting_events.jsonl"
    assert sha256_file(ledger) == LEDGER_SHA
    events = load_events(ledger)
    assert sum(1 for e in events if e["event_type"] == EVENT_CUTOVER) == 1
    assert sum(1 for e in events if e["event_type"] == EVENT_SESSION_CLOSE) == 0
    live = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-FileHash -Algorithm SHA256 "
            f"'{PROD}\\data\\results\\performance_accounting_events.jsonl').Hash.ToLower()",
        ],
        text=True,
    ).strip()
    assert live == LEDGER_SHA


def test_revalidate_frozen_bn2h_proposal_in_temp():
    suffix = (BN2H / "dryrun" / "approved_suffix.jsonl").read_bytes()
    planned = [json.loads(line) for line in suffix.decode("utf-8").splitlines() if line.strip()]
    existing = load_events(
        BN2H / "ops_backup" / "data__results__performance_accounting_events.jsonl"
    )
    prefix = (
        BN2H / "ops_backup" / "data__results__performance_accounting_events.jsonl"
    ).read_bytes()
    assert hashlib.sha256(prefix).hexdigest() == LEDGER_SHA
    to_write, skipped = plan_ledger_writes(planned, existing)
    assert skipped == 0
    assert len(to_write) == 13
    final = prefix + suffix
    assert hashlib.sha256(final).hexdigest() == APPROVED_FINAL
    reduced = reduce_events(existing + to_write)
    close = reduced["session_closes"][0]
    assert close["session_date"] == "2026-09-11"
    assert close["evidence_timestamp"] == "2026-09-11T22:47:19+00:00"
    assert close["equity"] == "92006.00"
    assert close["cash"] == "66447.22"
    assert close["market_value"] == "25558.78"
    assert close["external_cash_flow_since_prior"] == "0"
    assert len(close["positions"]) == 8
    assert len(reduced["sales"]) == 1
    assert reduced["sales"][0]["symbol"] == "DBA"
    assert Decimal(reduced["sales"][0]["qty"]) == Decimal("12")
    assert len(reduced["session_closes"]) == 1
    metrics = reduced["metrics"]
    assert metrics["time_weighted_return"] == "UNAVAILABLE"
    assert metrics["one_period_return_available"] is False
    assert metrics["historical_return"] is None


def test_passing_identity_includes_new_fields():
    identity = passing_git_identity()
    assert identity["index_empty"] is True
    assert identity["tracked_worktree_safe"] is True
    assert identity["untracked_worktree_safe"] is True
    assert identity["publication_code_matches_head"] is True
    assert identity["git_identity_valid"] is True
