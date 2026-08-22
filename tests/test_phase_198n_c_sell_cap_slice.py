"""Phase 198N-C — cap-compliant paper SELL slice (pure slicer + console + isolated proof).

PAPER ONLY. Isolated tmp_path SQLite/artifacts. No production packet/SQLite writes,
no real broker/network, no TRITON runtime, no Streamlit HTTP server.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest

from dashboard.paper_sell_arm_controls import (
    COLOR_AMBER,
    COLOR_GREEN,
    LABEL_ARMED,
    LABEL_READY,
    LABEL_UNAVAILABLE,
    map_sell_arm_visual_status,
)
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    BrokerSnapshotView,
    prepare_confirmation_challenge,
    readiness_packet_hash,
    READINESS_REFERENCE_TTL_MINUTES,
)
from services.paper_sell_arm_cap_slice import (
    STATUS_AUTHORITY_CAP_EXCEEDED,
    STATUS_PACKET_INVALID,
    STATUS_SLICE_READY,
    UNSLICEABLE,
    build_sliced_readiness_packet,
    iter_cap_compliant_slices,
    money_round,
    presentation_token,
    slice_reviewed_sells,
    source_readiness_packet_hash,
    tokens_match,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    READINESS_PACKET_NAME,
    arm_reviewed_sells,
    build_presentation,
    confirm_and_arm_paper_sells,
    operator_fail_message,
)
from services.paper_sell_arm_runtime_enforcement import (
    PaperSellDispatchBlocked,
    _ACTIVE_PERMIT,
    begin_paper_sell_dispatch,
)
from services.paper_sell_arm_store import (
    db_path_for_store_root,
    inspect_sqlite_arm_snapshot,
)
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    SESSION_OPEN_NOW,
    NetworkFakeAlpacaBroker,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    inprocess_place_live_orders,
    install_boundary_trace,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = SESSION_OPEN_NOW
PRODUCTION_ROOT = Path(r"C:\Users\akimw\triton")
PROD_PACKET = PRODUCTION_ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"
PROD_PACKET_SHA = "25c6cdb1a51fb24eec3d0b4edd786a248aa4447e25353d1024629040a0544795"
BUY_ARM = PRODUCTION_ROOT / "data" / "live" / "paper_arm_confirm.json"
BUY_ARM_SHA = "7b74bd1881b2e7918f31626f7376028c37a5b93e954eb00ca96613c0acef86f1"
REPO_SQLITE = PRODUCTION_ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
REPO_SQLITE_SHA = "d4886298a538610f1e7950b4cf48dc88126a5664b0d4d0bce94a4d9a4f2e13aa"
ISOLATED_REGRESSION = os.environ.get("TRITON_ISOLATED_REGRESSION") == "1" or (
    ROOT.resolve() != PRODUCTION_ROOT.resolve()
)
# Production-shaped seven-action F2 snapshot. Tests never open production paths
# when running in the isolated git-archive tree.
_SEVEN_ACTION_F2 = json.loads(
    r"""{"schema_version":"198I-ZA-SELL-READINESS-1","status":"UNAPPROVED_APPROVAL_READINESS_ONLY","generation_status":"ok","producer":"services.defensive_exit_approval_readiness","generated_at":"2026-08-17T19:44:13.496612+00:00","valid_until":"2026-08-17T20:14:13.496612+00:00","freshness_ttl_minutes":30,"mode":"PAPER","live_execution_state":"BLOCKED","creates_no_authority":true,"orders_submitted":false,"approval_created":false,"cycle_id":"mo-20260817_154341-24d873a6df6f4ff3a1a8e55a8a9722ed","run_id":"2026-08-17T19:44:09+00:00","batch_id":"manage-mo-20260817_1543","plan_id":"manage_positions_plan@2026-08-17T19:44:09+00:00","plan_hash":"f74f270a9de448863c68cd63c8aec377d44131747d6f6cfc797f90f9463a0a91","approval_ready_symbols":["TSLA","GLD","HD","BLK","DBA","MA","GE"],"approval_ready_actions":[{"symbol":"TSLA","action":"EXIT","qty":8.0,"est_price":339.27,"est_notional":2709.04},{"symbol":"GLD","action":"EXIT","qty":16.0,"est_price":405.3,"est_notional":6472.32},{"symbol":"HD","action":"EXIT","qty":40.0,"est_price":338.895,"est_notional":13520.8},{"symbol":"BLK","action":"EXIT","qty":2.0,"est_price":1154.04,"est_notional":2302.96},{"symbol":"DBA","action":"EXIT","qty":88.0,"est_price":28.1199,"est_notional":2469.2799999999997},{"symbol":"MA","action":"EXIT","qty":7.0,"est_price":561.88,"est_notional":3925.3199999999997},{"symbol":"GE","action":"EXIT","qty":9.0,"est_price":368.5,"est_notional":3310.11}],"total_proposed_notional":34709.83,"packet_generation_timestamp":"2026-08-17T19:44:13.496612+00:00","invocation_id":"mo-20260817_154341-24d873a6df6f4ff3a1a8e55a8a9722ed","market_open_invocation_id":"mo-20260817_154341-24d873a6df6f4ff3a1a8e55a8a9722ed","plan_generated_at":"2026-08-17T19:44:09+00:00","f_stage_completed_at":"2026-08-17T19:44:13.121211+00:00","f_stage_status":"COMPLETE","source_plan_sha256":"8d7ebc0316f80efb185f63280bf09cbe31678c0317dd1372bd8eeb407788c337","selected_row_hash":"f74f270a9de448863c68cd63c8aec377d44131747d6f6cfc797f90f9463a0a91","positions_snapshot_sha256":"74249851b33344e5cbaeaaba4bbc84ef7b9f562bf7188bd4734ff924659c7e5e","open_orders_snapshot_sha256":"a9203c9b29b2cbd7c520330a9f018b5e91fb5b0f82a7a17b286701d167c84226","row_classifications":[{"symbol":"BITO","action":"NONE","qty":"","selected_for_execution":false,"reason_code":"NO_ACTION_HOLD","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"NOT_SELECTED"},{"symbol":"BRK.B","action":"","qty":"","selected_for_execution":false,"reason_code":"STALE_LIFECYCLE_NO_POSITION","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"NOT_SELECTED"},{"symbol":"DIA","action":"NONE","qty":"","selected_for_execution":false,"reason_code":"NO_ACTION_HOLD","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"NOT_SELECTED"},{"symbol":"GS","action":"NONE","qty":"","selected_for_execution":false,"reason_code":"NO_ACTION_HOLD","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"NOT_SELECTED"},{"symbol":"TSLA","action":"EXIT","qty":"8","selected_for_execution":true,"reason_code":"EXIT_HIGH_DRAG_PROTECTION","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY"},{"symbol":"GLD","action":"EXIT","qty":"16","selected_for_execution":true,"reason_code":"EXIT_HIGH_DRAG_PROTECTION","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY"},{"symbol":"HD","action":"EXIT","qty":"40","selected_for_execution":true,"reason_code":"EXIT_HIGH_DRAG_PROTECTION","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY"},{"symbol":"BLK","action":"EXIT","qty":"2","selected_for_execution":true,"reason_code":"EXIT_APPROVED","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY"},{"symbol":"DBA","action":"EXIT","qty":"88","selected_for_execution":true,"reason_code":"EXIT_APPROVED_STALE_WEAK_CYCLES","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY"},{"symbol":"MA","action":"EXIT","qty":"7","selected_for_execution":true,"reason_code":"EXIT_APPROVED_STALE_WEAK_CYCLES","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY"},{"symbol":"GE","action":"EXIT","qty":"9","selected_for_execution":true,"reason_code":"EXIT_APPROVED","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY"},{"symbol":"ARKK","action":"ADD","qty":"8","selected_for_execution":true,"reason_code":"ADD_APPROVED","force_exit_override":null,"rotation_id":null,"risk_objective":null,"classification":"INVALID","reason":"buy_or_add_in_sell_cycle"}],"excluded_entry_only_count":1,"excluded_entry_only_symbols":["ARKK"],"excluded_entry_only_reason":"recognized_non_sell_row","arm_eligibility":"AUTHORITY_READY","remaining_blocker":"MISSING_PAPER_SELL_AUTHORITY","packet_status":"AUTHORITY_READY"}"""
)
EXPECTED_FIRST = (("TSLA", 8), ("GLD", 5), ("DBA", 9))
EXPECTED_FIRST_TOTAL = 4984.18
EXPECTED_SEQUENCE = [
    (("TSLA", 8), ("GLD", 5), ("DBA", 9), 4984.18),
    (("GLD", 11), ("HD", 1), ("DBA", 7), 4984.16),
    (("HD", 14), ("DBA", 9), 4984.82),
    (("HD", 14), ("DBA", 9), 4984.82),
    (("HD", 11), ("BLK", 1), ("DBA", 4), 4981.94),
    (("BLK", 1), ("DBA", 50), ("MA", 4), 4797.52),
    (("MA", 3), ("GE", 9), 4992.39),
]


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    for key in (
        "TRITON_PAPER_SELL_ARM_STORE_ROOT",
        "TRITON_PAPER_SELL_ARM_RESULTS_ROOT",
        "TRITON_PAPER_SELL_ARM_DB",
        "RESULTS_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)
    if ISOLATED_REGRESSION:
        return
    if REPO_SQLITE.is_file():
        assert hashlib.sha256(REPO_SQLITE.read_bytes()).hexdigest() == REPO_SQLITE_SHA
    assert hashlib.sha256(PROD_PACKET.read_bytes()).hexdigest() == PROD_PACKET_SHA
    if BUY_ARM.is_file():
        assert hashlib.sha256(BUY_ARM.read_bytes()).hexdigest() == BUY_ARM_SHA


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_price(row: Mapping[str, Any]) -> float:
    qty = float(row.get("qty") or 0)
    notional = float(
        row.get("est_notional")
        if row.get("est_notional") is not None
        else row.get("max_notional") or 0
    )
    if qty <= 0:
        return float(row.get("est_price") or 0)
    return notional / qty


def _load_prod_copy(*, generated_at: Optional[str] = None) -> Dict[str, Any]:
    packet = copy.deepcopy(_SEVEN_ACTION_F2)
    if generated_at is not None:
        packet["generated_at"] = generated_at
        packet["packet_generation_timestamp"] = generated_at
    return packet


def _write_packet(results_root: Path, packet: Dict[str, Any]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / READINESS_PACKET_NAME).write_text(
        json.dumps(packet, indent=2), encoding="utf-8"
    )


def _fresh_under_cap(**overrides) -> Dict[str, Any]:
    pkt = {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_198n_c",
        "run_id": "run_198n_c",
        "batch_id": "batch_198n_c",
        "plan_id": "plan_198n_c",
        "plan_hash": "planhash_198n_c",
        "schema_version": "198G-UNAPPROVED-APPROVAL-READINESS-1",
        "generated_at": NOW.isoformat(),
        "approval_ready_symbols": ["TSLA"],
        "approval_ready_actions": [
            {
                "symbol": "TSLA",
                "action": "EXIT",
                "qty": 8.0,
                "est_notional": 2660.0,
                "est_price": 332.5,
            }
        ],
        "total_proposed_notional": 2660.0,
        "trading_date": NOW.date().isoformat(),
    }
    pkt.update(overrides)
    return pkt


def _db(tmp_path: Path) -> Path:
    return tmp_path / "paper_sell_arm.sqlite3"


def _pres(tmp_path: Path, packet: Dict[str, Any], *, now=NOW) -> Dict[str, Any]:
    _write_packet(tmp_path, packet)
    return build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=now,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )


def _snap(tmp_path: Path, *, now=NOW) -> Dict[str, Any]:
    return inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=now)


def _scope_syms(payload: Dict[str, Any]) -> List[tuple]:
    per = payload.get("per_symbol_max_quantity")
    actions = list(payload.get("authorized_actions") or payload.get("authorized_scope") or [])
    if actions:
        out = []
        for row in actions:
            symbol = str(row.get("symbol") or "")
            qty = row.get("max_qty")
            if qty is None and isinstance(per, dict):
                qty = per.get(symbol)
            if qty is None:
                qty = row.get("qty")
            out.append((symbol, float(qty or 0)))
        return out
    if isinstance(per, dict) and per:
        return [(str(k), float(v)) for k, v in per.items()]
    return []


@dataclass
class FakePlanned:
    symbol: str
    side: str
    qty: float
    limit_price: float = 100.0
    stance: str = "EXIT"
    client_order_id: str = ""
    force_exit_override: bool = False
    ref_price: Optional[float] = None


def _planned_from_slice(actions: List[Dict[str, Any]]) -> List[FakePlanned]:
    out: List[FakePlanned] = []
    for row in actions:
        qty = float(row["qty"])
        notional = float(row["est_notional"])
        price = notional / qty
        out.append(
            FakePlanned(
                symbol=str(row["symbol"]),
                side="sell",
                qty=qty,
                limit_price=price,
                stance=str(row.get("action") or "EXIT"),
                client_order_id=f"coid-{row['symbol']}-slice",
                ref_price=price,
            )
        )
    return out


def _broker(auth: Dict[str, Any], positions: Dict[str, Dict[str, float]], **overrides):
    view = BrokerSnapshotView(
        positions=positions,
        open_order_symbols=set(),
        mode="paper",
        live_blocked=True,
        trading_date=str(auth.get("trading_date")),
        account_fingerprint=str(auth.get("account_fingerprint")),
        current_plan_hash=str(auth.get("plan_hash")),
        newer_completed_plan=False,
        market_session="MARKET_OPEN",
        paper_buy_arm_valid=True,
        paper_execution_authorized=True,
        cycle_id=str(auth.get("cycle_id")),
        run_id=str(auth.get("run_id")),
        batch_id=str(auth.get("batch_id")),
        plan_id=str(auth.get("plan_id")),
        readiness_packet_hash=str(auth.get("readiness_packet_hash")),
        confirmation_challenge_id=str(auth.get("confirmation_challenge_id")),
        confirmation_challenge_hash=str(auth.get("confirmation_challenge_hash")),
        evidence_independent=True,
    )
    for key, value in overrides.items():
        setattr(view, key, value)
    return view


class PacketQuotedFakeBroker(NetworkFakeAlpacaBroker):
    """Fake HTTP broker quoting stored unit notionals (est_notional/qty)."""

    def __init__(self, packet: Dict[str, Any], **kwargs: Any):
        prices = {
            str(row["symbol"]).upper(): _unit_price(row)
            for row in packet.get("approval_ready_actions") or []
        }
        first = next(iter(prices.values()), 330.0)
        super().__init__(quote_bid=first, quote_ask=first, **kwargs)
        self._packet_prices = prices

    def _px(self, symbol: str) -> float:
        return float(self._packet_prices.get(str(symbol).upper(), self.quote_ask))

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        px = self._px(symbol)
        return {
            "symbol": str(symbol).upper(),
            "bid": px,
            "ask": px,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        px = self._px(symbol)
        return {
            "symbol": str(symbol).upper(),
            "last": px,
            "ts": datetime.now(timezone.utc).isoformat(),
        }


def _write_buy_arm(tmp_path: Path, *, now=NOW) -> None:
    (tmp_path / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(),
                "mode": "ASSISTED",
                "expires_at": (now + timedelta(hours=4)).isoformat(),
                "allow_execute": True,
            }
        ),
        encoding="utf-8",
    )


def _write_market_session_evidence(tmp_path: Path, *, now=NOW) -> None:
    """Session freshness only. Does not override readiness_packet_hash."""
    path = tmp_path / "independent_dispatch_bindings.json"
    doc: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                doc = loaded
        except json.JSONDecodeError:
            doc = {}
    doc["market_session"] = "MARKET_OPEN"
    doc["market_session_observed_at"] = now.isoformat()
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_slicer_is_pure_and_side_effect_free():
    src = (ROOT / "services" / "paper_sell_arm_cap_slice.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {
        "open",
        "urlopen",
        "request",
        "sqlite3",
        "connect",
        "getenv",
        "Popen",
        "system",
        "submit_order",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (names & banned)
    assert "paper_sell_arm_authority" not in src
    assert "paper_sell_arm_runtime_enforcement" not in src
    assert "paper_sell_arm_console" not in src
    assert "paper_sell_arm_store" not in src
    assert APPROVAL_CLI_CAP == 5000.0
    cap_src = (ROOT / "services" / "paper_sell_arm_authority.py").read_text(encoding="utf-8")
    assert "APPROVAL_CLI_CAP = 5000.0" in cap_src


def test_current_packet_first_slice_and_full_coverage():
    packet = _load_prod_copy()
    assert money_round(packet["total_proposed_notional"]) == 34709.83
    result = slice_reviewed_sells(packet)
    assert result["status"] == STATUS_SLICE_READY
    assert result["applied_cap"] == 5000.0
    assert result["cap_source"] == "APPROVAL_CLI_CAP"
    got = tuple((a["symbol"], int(a["qty"])) for a in result["slice_actions"])
    assert got == EXPECTED_FIRST
    assert result["slice_total_notional"] == EXPECTED_FIRST_TOTAL
    assert result["slice_total_notional"] <= 5000.0
    assert all(money_round(a["est_notional"]) <= 5000.0 for a in result["slice_actions"])
    assert all(a["symbol"] != "ARKK" for a in result["slice_actions"])
    assert all(a.get("action") != "ADD" for a in result["slice_actions"])
    seq = iter_cap_compliant_slices(packet)
    assert len(seq) == 7
    for idx, expected in enumerate(EXPECTED_SEQUENCE):
        *pairs, total = expected
        got_pairs = tuple((a["symbol"], int(a["qty"])) for a in seq[idx]["actions"])
        assert got_pairs == tuple(pairs)
        assert seq[idx]["total"] == total
        assert seq[idx]["total"] <= 5000.0
        assert seq[idx]["le_cap"] is True
    need = {a["symbol"]: float(a["qty"]) for a in packet["approval_ready_actions"]}
    got_qty: Dict[str, float] = {k: 0.0 for k in need}
    for item in seq:
        for row in item["actions"]:
            got_qty[row["symbol"]] += float(row["qty"])
    assert got_qty == need
    twice = slice_reviewed_sells(packet)
    assert twice["slice_hash"] == result["slice_hash"]
    assert packet["approval_ready_actions"][1]["qty"] == 16.0


def test_presentation_ready_and_callback_same_slice(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    pres = _pres(tmp_path, packet)
    visual = map_sell_arm_visual_status(pres)
    assert visual["label"] == LABEL_READY
    assert visual["color"] == COLOR_AMBER
    assert visual["reason"] == "Cap-compliant reviewed SELL slice is ready."
    assert pres["arm_enabled"] is True
    assert pres["action_count"] == 7
    assert money_round(pres["full_total_notional"]) == 34709.83
    got = tuple((a["symbol"], int(a["qty"])) for a in pres["slice_actions"])
    assert got == EXPECTED_FIRST
    assert pres["slice_total_notional"] == EXPECTED_FIRST_TOTAL
    assert pres["presentation_token"]
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert armed["ok"] is True
    assert armed["counts"]["broker"] == 0
    assert armed["counts"]["network"] == 0
    snap = _snap(tmp_path)
    assert snap["challenge_count"] == 1
    assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    payload = (snap["open_challenges"][0] or {}).get("payload") or {}
    assert _scope_syms(payload) == [("TSLA", 8.0), ("GLD", 5.0), ("DBA", 9.0)]
    assert payload.get("applied_cap") == 5000.0
    assert payload.get("identity_schema") == "198N-E-DUAL-IDENTITY-SELL-1"
    assert payload.get("source_readiness_packet_hash") == source_readiness_packet_hash(packet)
    assert payload.get("readiness_packet_hash") == readiness_packet_hash(packet)
    assert payload.get("authorized_slice_hash")
    assert payload.get("source_readiness_packet_hash") != payload.get("authorized_slice_hash")
    on_disk = json.loads((tmp_path / READINESS_PACKET_NAME).read_text(encoding="utf-8"))
    assert len(on_disk["approval_ready_actions"]) == 7
    assert money_round(on_disk["total_proposed_notional"]) == 34709.83


def test_isolated_confirm_authority_and_hardened_boundary(tmp_path: Path, monkeypatch):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    pres = _pres(tmp_path, packet)
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert armed["ok"] is True
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert confirmed["ok"] is True
    assert confirmed["counts"]["broker"] == 0
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=1))
    assert snap["latest_challenge"]["state"] == "CONSUMED"
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert snap["authority_count"] == 1
    assert snap["dispatch_count"] == 0
    auth_payload = snap["latest_authority"]["payload"]
    assert _scope_syms(auth_payload) == [("TSLA", 8.0), ("GLD", 5.0), ("DBA", 9.0)]
    assert auth_payload.get("identity_schema") == "198N-E-DUAL-IDENTITY-SELL-1"
    assert auth_payload.get("readiness_packet_hash") == readiness_packet_hash(packet)
    assert auth_payload.get("source_readiness_packet_hash")
    assert auth_payload.get("authorized_slice_hash")
    assert auth_payload.get("source_readiness_packet_hash") != auth_payload.get(
        "authorized_slice_hash"
    )
    deferred = {a["symbol"] for a in pres["deferred_actions"]}
    authorized = {s for s, _ in _scope_syms(auth_payload)}
    assert "HD" in deferred and "HD" not in authorized
    assert "BLK" in deferred and "BLK" not in authorized
    armed_pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    visual = map_sell_arm_visual_status(armed_pres)
    assert visual["label"] == LABEL_ARMED
    assert visual["color"] == COLOR_GREEN

    _write_buy_arm(tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    stub_unrelated_pre_gates(monkeypatch)
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    positions = {
        str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
        for a in packet["approval_ready_actions"]
    }
    view = _broker(auth_payload, positions)
    extra = _planned_from_slice(packet["approval_ready_actions"])
    g_ok = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=extra,
        broker_view=view,
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth_payload["readiness_packet_hash"],
    )
    assert g_ok.consumed, f"{g_ok.reason} {g_ok.details}"
    auth_syms = [str(getattr(p, "symbol", "")).upper() for p in g_ok.authorized_planned]
    auth_qty = [int(getattr(p, "qty", 0)) for p in g_ok.authorized_planned]
    assert auth_syms == ["TSLA", "GLD", "DBA"]
    assert auth_qty == [8, 5, 9]
    assert "HD" not in auth_syms
    g_replay = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=extra,
        broker_view=view,
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth_payload["readiness_packet_hash"],
    )
    assert not g_replay.consumed
    fake = PacketQuotedFakeBroker(
        packet,
        positions=[{"symbol": k, "qty": str(v["qty"])} for k, v in positions.items()],
    )
    trace = OrderedTrace()
    from tests._phase_198i_sell_fakes import install_boundary_trace

    install_boundary_trace(monkeypatch, trace)
    for row in g_ok.authorized_planned:
        resp = fake.submit_order(
            symbol=row.symbol,
            qty=row.qty,
            side="sell",
            order_type="limit",
            time_in_force="day",
            limit_price=row.limit_price,
            client_order_id=row.client_order_id,
        )
        assert resp.get("status") == "accepted"
    assert len(fake.submit_payloads) == 3
    from services.paper_sell_arm_runtime_enforcement import PaperSellDispatchBlocked

    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order(
            symbol="HD",
            qty=1,
            side="sell",
            order_type="limit",
            time_in_force="day",
            limit_price=338.02,
            client_order_id="coid-hd-after",
        )
    snap2 = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap2["latest_authority"]["state"] == "CONSUMED"
    assert snap2["authority_count"] == 1


def test_raw_over_cap_packet_still_refused_by_authority(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    with pytest.raises(ValueError, match="scope_exceeds_approval_cli_cap"):
        prepare_confirmation_challenge(
            store_root=tmp_path,
            readiness_packet=packet,
            account_raw="acct-198n-c",
            trading_date=NOW.date().isoformat(),
            now=NOW,
            db_path=_db(tmp_path),
        )
    snap = _snap(tmp_path)
    assert snap["challenge_count"] == 0
    assert snap["authority_count"] == 0
    assert APPROVAL_CLI_CAP == 5000.0


def test_render_to_click_refusals(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    pres = _pres(tmp_path, packet)
    token = copy.deepcopy(pres["presentation_token"])
    changed = copy.deepcopy(packet)
    changed["generated_at"] = (NOW + timedelta(seconds=5)).isoformat()
    _write_packet(tmp_path, changed)
    denied = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=token,
    )
    assert denied["ok"] is False
    assert denied["reason"] == "packet_changed_since_render"
    assert operator_fail_message("packet_changed_since_render") == (
        "Reviewed SELL packet changed. Refresh and review the new slice."
    )
    visual = map_sell_arm_visual_status(
        {**pres, "arm_enabled": False, "arm_deny_reason": "packet_changed_since_render"}
    )
    assert visual["label"] == LABEL_UNAVAILABLE
    assert visual["reason"] == operator_fail_message("packet_changed_since_render")
    assert _snap(tmp_path)["challenge_count"] == 0

    _write_packet(tmp_path, packet)
    pres = _pres(tmp_path, packet)
    inv = copy.deepcopy(packet)
    inv["invocation_id"] = "other-invocation"
    inv["market_open_invocation_id"] = "other-invocation"
    _write_packet(tmp_path, inv)
    denied_inv = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert denied_inv["reason"] == "packet_changed_since_render"
    assert _snap(tmp_path)["challenge_count"] == 0

    _write_packet(tmp_path, packet)
    pres = _pres(tmp_path, packet)
    reordered = copy.deepcopy(packet)
    reordered["approval_ready_actions"] = list(reversed(reordered["approval_ready_actions"]))
    reordered["approval_ready_symbols"] = [a["symbol"] for a in reordered["approval_ready_actions"]]
    _write_packet(tmp_path, reordered)
    denied_order = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert denied_order["reason"] == "packet_changed_since_render"

    _write_packet(tmp_path, packet)
    pres = _pres(tmp_path, packet)
    qty_changed = copy.deepcopy(packet)
    qty_changed["approval_ready_actions"][0]["qty"] = 7.0
    qty_changed["approval_ready_actions"][0]["est_notional"] = 2370.41
    _write_packet(tmp_path, qty_changed)
    denied_qty = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert denied_qty["reason"] == "packet_changed_since_render"

    _write_packet(tmp_path, packet)
    pres = _pres(tmp_path, packet)
    bad_token = copy.deepcopy(pres["presentation_token"])
    bad_token["slice_hash"] = "0" * 64
    denied_hash = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=bad_token,
    )
    assert denied_hash["reason"] == "packet_changed_since_render"
    plan = copy.deepcopy(packet)
    plan["source_plan_sha256"] = "a" * 64
    _write_packet(tmp_path, plan)
    denied_plan = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert denied_plan["reason"] == "packet_changed_since_render"
    row = copy.deepcopy(packet)
    row["selected_row_hash"] = "b" * 64
    _write_packet(tmp_path, row)
    denied_row = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert denied_row["reason"] == "packet_changed_since_render"
    missing_inv = copy.deepcopy(packet)
    missing_inv.pop("invocation_id", None)
    missing_inv.pop("market_open_invocation_id", None)
    _write_packet(tmp_path, missing_inv)
    denied_missing = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert denied_missing["reason"] == "packet_changed_since_render"
    assert _snap(tmp_path)["challenge_count"] == 0


def test_stale_future_malformed_and_unknown_action(tmp_path: Path):
    stale = _load_prod_copy(
        generated_at=(NOW - timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES + 1)).isoformat()
    )
    pres = _pres(tmp_path, stale, now=NOW)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "stale_readiness_packet"
    future = _load_prod_copy(generated_at=(NOW + timedelta(minutes=5)).isoformat())
    pres_f = _pres(tmp_path, future, now=NOW)
    assert pres_f["arm_deny_reason"] == "future_generated_at"
    malformed = _fresh_under_cap()
    malformed["approval_ready_actions"] = "nope"
    pres_m = _pres(tmp_path, malformed)
    assert pres_m["arm_enabled"] is False
    unknown = _fresh_under_cap()
    unknown["approval_ready_actions"][0]["action"] = "HEDGE"
    pres_u = _pres(tmp_path, unknown)
    assert pres_u["arm_enabled"] is False
    assert pres_u["arm_deny_reason"] == "malformed_packet"


def test_buy_add_contradictory_and_nonfinite(tmp_path: Path):
    buy = _fresh_under_cap()
    buy["approval_ready_actions"].append(
        {"symbol": "ARKK", "action": "ADD", "qty": 1, "est_notional": 50.0}
    )
    buy["approval_ready_symbols"].append("ARKK")
    pres = _pres(tmp_path, buy)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "malformed_packet"
    contra = _fresh_under_cap()
    contra["approval_ready_actions"][0]["side"] = "BUY"
    sliced = slice_reviewed_sells(contra)
    assert sliced["status"] == STATUS_PACKET_INVALID
    for qty in (0, -1, float("nan"), float("inf")):
        bad = _fresh_under_cap()
        bad["approval_ready_actions"][0]["qty"] = qty
        assert slice_reviewed_sells(bad)["status"] == STATUS_PACKET_INVALID
    for notional in (0, -1, float("nan"), float("inf")):
        bad = _fresh_under_cap()
        bad["approval_ready_actions"][0]["est_notional"] = notional
        assert slice_reviewed_sells(bad)["status"] == STATUS_PACKET_INVALID


def test_unit_over_cap_and_no_safe_slice():
    only = _fresh_under_cap()
    only["approval_ready_actions"] = [
        {
            "symbol": "NVDA",
            "action": "EXIT",
            "qty": 1.0,
            "est_notional": 5000.01,
            "est_price": 5000.01,
        }
    ]
    only["approval_ready_symbols"] = ["NVDA"]
    only["total_proposed_notional"] = 5000.01
    result = slice_reviewed_sells(only)
    assert result["status"] == STATUS_AUTHORITY_CAP_EXCEEDED
    assert result["unsliceable_actions"]
    assert result["unsliceable_actions"][0]["reason"] == UNSLICEABLE
    mixed = _fresh_under_cap()
    mixed["approval_ready_actions"] = [
        {
            "symbol": "TSLA",
            "action": "EXIT",
            "qty": 8.0,
            "est_notional": 2660.0,
            "est_price": 332.5,
        },
        {
            "symbol": "NVDA",
            "action": "EXIT",
            "qty": 1.0,
            "est_notional": 5000.01,
            "est_price": 5000.01,
        },
    ]
    mixed["approval_ready_symbols"] = ["TSLA", "NVDA"]
    mixed["total_proposed_notional"] = 7660.01
    mixed_result = slice_reviewed_sells(mixed)
    assert mixed_result["status"] == STATUS_SLICE_READY
    assert [a["symbol"] for a in mixed_result["slice_actions"]] == ["TSLA"]
    assert mixed_result["unsliceable_actions"][0]["symbol"] == "NVDA"


def test_no_safe_slice_presentation(tmp_path: Path):
    only = _fresh_under_cap()
    only["approval_ready_actions"] = [
        {
            "symbol": "NVDA",
            "action": "EXIT",
            "qty": 1.0,
            "est_notional": 5000.01,
            "est_price": 5000.01,
        }
    ]
    only["approval_ready_symbols"] = ["NVDA"]
    only["total_proposed_notional"] = 5000.01
    pres = _pres(tmp_path, only)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "authority_cap_exceeded"
    assert operator_fail_message("authority_cap_exceeded") == (
        "No reviewed SELL quantity fits the $5,000 authority cap."
    )
    visual = map_sell_arm_visual_status(pres)
    assert visual["label"] == LABEL_UNAVAILABLE
    assert visual["reason"] == operator_fail_message("authority_cap_exceeded")
    denied = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert denied["ok"] is False
    assert _snap(tmp_path)["challenge_count"] == 0


def test_exact_cap_boundary_and_over_by_cent(tmp_path: Path):
    exact = _fresh_under_cap()
    exact["approval_ready_actions"] = [
        {
            "symbol": "TSLA",
            "action": "EXIT",
            "qty": 10.0,
            "est_notional": 5000.0,
            "est_price": 500.0,
        }
    ]
    exact["approval_ready_symbols"] = ["TSLA"]
    exact["total_proposed_notional"] = 5000.0
    result = slice_reviewed_sells(exact)
    assert result["status"] == STATUS_SLICE_READY
    assert result["slice_total_notional"] == 5000.0
    over = copy.deepcopy(exact)
    over["approval_ready_actions"][0]["est_notional"] = 5000.01
    over["approval_ready_actions"][0]["est_price"] = 500.001
    over["total_proposed_notional"] = 5000.01
    sliced_over = slice_reviewed_sells(over)
    assert sliced_over["status"] == STATUS_SLICE_READY
    assert sliced_over["slice_actions"][0]["qty"] == 9
    assert sliced_over["slice_total_notional"] <= 5000.0
    with pytest.raises(ValueError, match="scope_exceeds_approval_cli_cap"):
        prepare_confirmation_challenge(
            store_root=tmp_path,
            readiness_packet=over,
            account_raw="acct-198n-c",
            now=NOW,
            db_path=_db(tmp_path),
        )


def test_sqlite_failure_and_outcome_unknown(tmp_path: Path):
    packet = _fresh_under_cap()
    _write_packet(tmp_path, packet)
    blocker = tmp_path / "blocker_file"
    blocker.write_text("not-a-dir", encoding="utf-8")
    denied = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=blocker / "paper_sell_arm.sqlite3",
    )
    assert denied["ok"] is False
    unknown = _fresh_under_cap(packet_status="OUTCOME_UNKNOWN")
    pres = _pres(tmp_path, unknown)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "sell_outcome_unknown"


def test_expired_consumed_mismatched_authority_and_no_auto_second_challenge(
    tmp_path: Path,
):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    pres = _pres(tmp_path, packet)
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert confirmed["ok"] is True
    before = _snap(tmp_path, now=NOW + timedelta(minutes=1))
    assert before["challenge_count"] == 1
    assert before["authority_count"] == 1
    rerun = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert rerun["arm_enabled"] is False
    auto = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=rerun.get("presentation_token"),
    )
    assert auto["ok"] is False
    after = _snap(tmp_path, now=NOW + timedelta(minutes=1))
    assert after["challenge_count"] == 1
    assert after["authority_count"] == 1


def test_partial_fill_modeled_next_slice_requires_new_packet(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    first = slice_reviewed_sells(packet)
    remaining = copy.deepcopy(packet)
    remaining_actions = []
    taken = {a["symbol"]: float(a["qty"]) for a in first["slice_actions"]}
    for row in packet["approval_ready_actions"]:
        left = float(row["qty"]) - taken.get(row["symbol"], 0.0)
        if left < 1:
            continue
        unit = float(row["est_notional"]) / float(row["qty"])
        nxt = dict(row)
        nxt["qty"] = left
        nxt["est_notional"] = money_round(left * unit)
        remaining_actions.append(nxt)
    remaining["approval_ready_actions"] = remaining_actions
    remaining["approval_ready_symbols"] = [a["symbol"] for a in remaining_actions]
    remaining["total_proposed_notional"] = money_round(
        sum(float(a["est_notional"]) for a in remaining_actions)
    )
    remaining["generated_at"] = (NOW + timedelta(minutes=2)).isoformat()
    second = slice_reviewed_sells(remaining)
    got = tuple((a["symbol"], int(a["qty"])) for a in second["slice_actions"])
    assert got == (("GLD", 11), ("HD", 1), ("DBA", 7))
    assert second["slice_hash"] != first["slice_hash"]
    _write_packet(tmp_path, remaining)
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW + timedelta(minutes=2),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert pres["arm_enabled"] is True
    assert tuple((a["symbol"], int(a["qty"])) for a in pres["slice_actions"]) == got
    assert _snap(tmp_path)["challenge_count"] == 0


def test_dependency_deferred_sells_remain_blocking():
    from services.paper_sell_entry_dependency import (
        ALLOW_EXISTING_GATES,
        BLOCK_SAME_SYMBOL_PENDING_SELL,
        filter_entry_candidates,
    )

    packet = _load_prod_copy()
    first = slice_reviewed_sells(packet)
    pending = [{"symbol": a["symbol"], "qty": a["qty"]} for a in packet["approval_ready_actions"]]
    allowed, decisions = filter_entry_candidates(
        [
            {"ticker": "ARKK", "allocation_action": "ADD", "est_notional": 200.0},
            {"ticker": "GLD", "allocation_action": "ADD", "est_notional": 200.0},
            {"ticker": "MSFT", "allocation_action": "OPEN_NEW", "est_notional": 500.0},
        ],
        pending_sells=pending,
        current_cash=10000.0,
    )
    by_sym = {d["symbol"]: d["decision"] for d in decisions}
    assert by_sym["GLD"] == BLOCK_SAME_SYMBOL_PENDING_SELL
    assert by_sym["ARKK"] == ALLOW_EXISTING_GATES or by_sym["MSFT"] == ALLOW_EXISTING_GATES
    assert "GLD" not in [c.get("ticker") or c.get("symbol") for c in allowed]
    assert first["deferred_action_count"] > 0
    assert packet["approval_ready_actions"][1]["qty"] == 16.0


def test_under_cap_packet_still_arms_full_scope(tmp_path: Path):
    packet = _fresh_under_cap()
    pres = _pres(tmp_path, packet)
    assert pres["arm_enabled"] is True
    assert pres["slice_action_count"] == 1
    assert pres["slice_actions"][0]["qty"] == 8
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert armed["ok"] is True
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase="CONFIRM PAPER SELLS",
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert confirmed["ok"] is True
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=1))
    assert _scope_syms(snap["latest_authority"]["payload"]) == [("TSLA", 8.0)]


def test_tokens_match_and_sliced_packet_does_not_rewrite_identity():
    packet = _load_prod_copy()
    result = slice_reviewed_sells(packet)
    sliced = build_sliced_readiness_packet(packet, result)
    assert sliced["plan_hash"] == packet["plan_hash"]
    assert sliced["selected_row_hash"] == packet["selected_row_hash"]
    assert sliced["source_plan_sha256"] == packet["source_plan_sha256"]
    assert sliced["cycle_id"] == packet["cycle_id"]
    assert sliced["batch_id"] == packet["batch_id"]
    assert len(sliced["approval_ready_actions"]) == 3
    assert readiness_packet_hash(sliced) != readiness_packet_hash(packet)
    under = _fresh_under_cap()
    under_slice = slice_reviewed_sells(under)
    under_cloned = build_sliced_readiness_packet(under, under_slice)
    assert readiness_packet_hash(under_cloned) == readiness_packet_hash(under)
    token = presentation_token(packet=packet, slice_result=result, packet_sha256=PROD_PACKET_SHA)
    assert tokens_match(token, token)
    other = dict(token)
    other["slice_total"] = 1.0
    assert not tokens_match(token, other)


def test_callback_does_not_map_cap_to_malformed(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    _write_packet(tmp_path, packet)
    from services.paper_sell_arm_console import _map_arm_exception

    assert _map_arm_exception(ValueError("challenge_blocked:scope_exceeds_approval_cli_cap")) == (
        "authority_cap_exceeded"
    )
    assert _map_arm_exception(ValueError("challenge_blocked:symbol_exceeds_cap:GLD")) == (
        "authority_cap_exceeded"
    )
    assert operator_fail_message("authority_cap_exceeded") != operator_fail_message(
        "malformed_packet"
    )


def test_dashboard_labels_and_phrase_preserved():
    src = (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    assert 'ARM_BUTTON_LABEL = "Arm Reviewed Sells"' in src
    assert 'CONFIRM_BUTTON_LABEL = "Confirm & Arm Paper Sells"' in src
    assert "CONFIRM PAPER SELLS" in src
    assert "Full source packet:" in src
    assert "Current authorized slice:" in src
    assert "Button clicks do not submit orders." in (
        ROOT / "services" / "paper_sell_arm_console.py"
    ).read_text(encoding="utf-8")
    assert "expected_presentation_token" in src


def _arm_and_confirm_current_slice(tmp_path: Path, packet: Dict[str, Any]) -> Dict[str, Any]:
    pres = _pres(tmp_path, packet)
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert armed["ok"] is True
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert confirmed["ok"] is True
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=1))
    return {"pres": pres, "armed": armed, "confirmed": confirmed, "snap": snap}


def test_dual_identity_authority_binds_full_source_and_slice(tmp_path: Path):
    """Console arms the full F2 packet; authorized_scope is the cap slice only."""
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_and_confirm_current_slice(tmp_path, packet)
    on_disk = json.loads((tmp_path / READINESS_PACKET_NAME).read_text(encoding="utf-8"))
    assert len(on_disk["approval_ready_actions"]) == 7
    auth = seeded["snap"]["latest_authority"]["payload"]
    sliced = build_sliced_readiness_packet(packet, seeded["pres"]["slice"])
    assert auth.get("identity_schema") == "198N-E-DUAL-IDENTITY-SELL-1"
    assert auth.get("readiness_packet_hash") == readiness_packet_hash(on_disk)
    assert auth.get("source_readiness_packet_hash") == source_readiness_packet_hash(on_disk)
    assert auth.get("authorized_slice_hash") == seeded["pres"]["slice"]["slice_hash"]
    assert auth.get("readiness_packet_hash") != readiness_packet_hash(sliced)
    assert _scope_syms(auth) == [("TSLA", 8.0), ("GLD", 5.0), ("DBA", 9.0)]
    assert not (tmp_path / "independent_dispatch_bindings.json").is_file()


def test_full_seven_action_plan_filters_to_authorized_slice(tmp_path: Path, monkeypatch):
    """Dual identity: full planned SELL set is mapped onto the $5,000 slice."""
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_and_confirm_current_slice(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    _write_buy_arm(tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    planned = _planned_from_slice(packet["approval_ready_actions"])
    positions = {
        str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
        for a in packet["approval_ready_actions"]
    }
    view = _broker(auth, positions)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=view,
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert guard.consumed, f"{guard.reason} {guard.details}"
    assert [p.symbol for p in guard.authorized_planned] == ["TSLA", "GLD", "DBA"]
    assert [int(p.qty) for p in guard.authorized_planned] == [8, 5, 9]


def test_slice_plan_matches_full_source_hash(tmp_path: Path, monkeypatch):
    """readiness_packet_hash keeps full-packet meaning; slice-sized plan still matches."""
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_and_confirm_current_slice(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    full_hash = readiness_packet_hash(packet)
    assert full_hash == auth["readiness_packet_hash"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    planned = _planned_from_slice(seeded["pres"]["slice"]["slice_actions"])
    positions = {
        str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
        for a in packet["approval_ready_actions"]
    }
    view = _broker(auth, positions, readiness_packet_hash=full_hash)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=view,
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=full_hash,
    )
    assert guard.consumed, f"{guard.reason} {guard.details}"
    assert [p.symbol for p in guard.authorized_planned] == ["TSLA", "GLD", "DBA"]


def test_missing_authorized_symbol_fails_closed(tmp_path: Path, monkeypatch):
    """Incomplete authorized slice cannot dispatch."""
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_and_confirm_current_slice(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    planned = _planned_from_slice(seeded["pres"]["slice"]["slice_actions"][:2])
    positions = {
        str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
        for a in packet["approval_ready_actions"]
    }
    view = _broker(auth, positions)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=view,
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not guard.consumed
    assert "incomplete_authorized" in str(guard.reason) or "mismatch" in str(guard.reason)
