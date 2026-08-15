"""Shared fake-broker + paper-sell seed helpers for Phase 198I SELL route proof.

Fake sits at AlpacaBroker HTTP helpers (_get/_post/_delete/_data_get). Production
submit_order / cancel_order / cancel_all_orders and the hardened SELL boundary
still execute. No credentials, no real network, no live execution.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

from services.broker_alpaca import (
    AlpacaBroker,
    BrokerAuthorizationRequired,
    _broker_order_authorized,
)
from services.paper_sell_arm_authority import (
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
    readiness_packet_hash,
)
from services.paper_sell_arm_runtime_enforcement import (
    begin_paper_sell_dispatch,
    enforce_paper_sell_broker_submission,
)

LIVE_BLOCKED = {
    "live_blocked": True,
    "live_block_reason": "TEST_LIVE_BLOCKED",
    "live_explicitly_authorized": False,
    "live_env_enabled": False,
    "paper_mode": True,
}

# Tuesday 2026-08-11 12:00 ET — regular NYSE session (same instant used by prior 198I tests).
SESSION_OPEN_NOW = datetime(2026, 8, 11, 16, 0, 0, tzinfo=timezone.utc)

FAKE_BROKER_BASE = "http://phase198i-fake-broker.test"


@dataclass
class OrderedTrace:
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, event: str, **kwargs: Any) -> None:
        row = {"event": event, **kwargs}
        self.events.append(row)

    def names(self) -> List[str]:
        return [str(e.get("event")) for e in self.events]

    def count(self, event: str) -> int:
        return sum(1 for e in self.events if e.get("event") == event)


class NetworkFakeAlpacaBroker(AlpacaBroker):
    """AlpacaBroker stand-in: production mutation methods, fake HTTP only."""

    def __init__(
        self,
        mode: str = "paper",
        *,
        positions: Optional[List[Dict[str, Any]]] = None,
        open_orders: Optional[List[Dict[str, Any]]] = None,
        quote_bid: float = 330.0,
        quote_ask: float = 332.5,
        order_side_by_id: Optional[Dict[str, str]] = None,
    ):
        self.mode = str(mode or "paper").lower()
        self.base = FAKE_BROKER_BASE
        self.data_base = FAKE_BROKER_BASE
        self.timeout = 1
        self.key = "FAKE_PAPER_KEY"
        self.secret = "FAKE_PAPER_SECRET"
        self.session = None
        self._auth_diagnostics = {"fake": True}
        self.posts: List[Dict[str, Any]] = []
        self.deletes: List[str] = []
        self.gets: List[str] = []
        self.data_gets: List[str] = []
        self.positions = positions or [
            {"symbol": "TSLA", "qty": "8", "qty_available": "8", "avg_entry_price": "200"}
        ]
        self.open_orders = list(open_orders or [])
        self.quote_bid = quote_bid
        self.quote_ask = quote_ask
        self.order_side_by_id = dict(order_side_by_id or {})

    @property
    def fake_broker_call_count(self) -> int:
        return len(self.posts) + len(self.deletes)

    @property
    def submit_payloads(self) -> List[Dict[str, Any]]:
        return [p["payload"] for p in self.posts if p.get("path") == "/v2/orders"]

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self.gets.append(path)
        if path == "/v2/clock":
            return {"is_open": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        if path == "/v2/account":
            return {
                "account_number": "PA-FAKE-198I",
                "status": "ACTIVE",
                "buying_power": "100000",
                "cash": "100000",
                "trading_blocked": False,
            }
        if path == "/v2/positions":
            return list(self.positions)
        if path.startswith("/v2/positions/"):
            sym = path.rsplit("/", 1)[-1].upper()
            for p in self.positions:
                if str(p.get("symbol") or "").upper() == sym:
                    return p
            return {"symbol": sym, "qty": "0", "qty_available": "0"}
        if path == "/v2/orders":
            return list(self.open_orders)
        if path.startswith("/v2/orders/"):
            oid = path.rsplit("/", 1)[-1]
            side = self.order_side_by_id.get(oid, "sell")
            return {
                "id": oid,
                "side": side,
                "symbol": "TSLA",
                "qty": "8",
                "status": "open",
                "type": "limit",
            }
        return {}

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        if path == "/v2/orders" and not _broker_order_authorized.get():
            raise BrokerAuthorizationRequired(
                "BROKER FINAL CHECK: POST /v2/orders blocked — execution authorization "
                "evidence missing. All orders must pass enforce_broker_submission_gate."
            )
        rec = {"path": path, "payload": dict(payload)}
        self.posts.append(rec)
        oid = f"fake-ord-{len(self.posts)}"
        out = {"id": oid, "status": "accepted", **payload}
        return out

    def _delete(self, path: str) -> Any:
        self.deletes.append(path)
        return {"status": "canceled", "id": path.rsplit("/", 1)[-1], "path": path}

    def _data_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self.data_gets.append(path)
        return {
            "quote": {
                "bp": self.quote_bid,
                "ap": self.quote_ask,
                "t": datetime.now(timezone.utc).isoformat(),
            },
            "trade": {"p": self.quote_ask, "t": datetime.now(timezone.utc).isoformat()},
        }

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        return {
            "symbol": str(symbol).upper(),
            "bid": self.quote_bid,
            "ask": self.quote_ask,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        return {
            "symbol": str(symbol).upper(),
            "last": self.quote_ask,
            "ts": datetime.now(timezone.utc).isoformat(),
        }


def block_real_network(monkeypatch) -> None:
    for key in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_BASE_URL",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")

    def _blocked(*_a, **_k):
        raise RuntimeError("network blocked in phase 198I SELL tests")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


def readiness_packet(
    *, symbol: str = "TSLA", qty: float = 8.0, price: float = 332.5, now: datetime
):
    return {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_198i_sell",
        "run_id": "run_198i_sell",
        "batch_id": "batch_198i_sell",
        "plan_id": "plan_198i_sell",
        "plan_hash": "planhash_198i_sell",
        "schema_version": "198G-UNAPPROVED-APPROVAL-READINESS-1",
        "approval_ready_symbols": [symbol],
        "approval_ready_actions": [
            {
                "symbol": symbol,
                "action": "EXIT",
                "qty": qty,
                "est_notional": float(qty) * float(price),
                "est_price": price,
            }
        ],
        "total_proposed_notional": float(qty) * float(price),
        "trading_date": now.date().isoformat(),
    }


def seed_paper_sell_store(
    tmp_path: Path,
    *,
    now: Optional[datetime] = None,
    symbol: str = "TSLA",
    qty: float = 8.0,
    price: float = 332.5,
    consume: bool = False,
) -> Dict[str, Any]:
    now = now or SESSION_OPEN_NOW
    readiness = readiness_packet(symbol=symbol, qty=qty, price=price, now=now)
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
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-198i-sell",
        trading_date=now.date().isoformat(),
        now=now,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=now,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    bindings = {
        "plan_hash": readiness["plan_hash"],
        "cycle_id": readiness["cycle_id"],
        "run_id": readiness["run_id"],
        "batch_id": readiness["batch_id"],
        "plan_id": readiness["plan_id"],
        "trading_date": now.date().isoformat(),
        "account_fingerprint": auth["account_fingerprint"],
        "readiness_packet_hash": auth["readiness_packet_hash"],
        "confirmation_challenge_id": auth["confirmation_challenge_id"],
        "confirmation_challenge_hash": auth["confirmation_challenge_hash"],
        "paper_execution_authorized": True,
        "market_session": "MARKET_OPEN",
        "market_session_observed_at": now.isoformat(),
        "lifecycle_actions": {symbol: "EXIT"},
    }
    (tmp_path / "independent_dispatch_bindings.json").write_text(
        json.dumps(bindings), encoding="utf-8"
    )
    (tmp_path / "defensive_exit_approval_readiness.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )
    (tmp_path / "positions_snapshot.csv").write_text(
        f"symbol,qty,qty_available\n{symbol},{qty},{qty}\n", encoding="utf-8"
    )
    (tmp_path / "open_orders_snapshot.csv").write_text("symbol,side,qty\n", encoding="utf-8")
    out = {
        "readiness": readiness,
        "auth": auth,
        "now": now,
        "symbol": symbol,
        "qty": qty,
        "price": price,
    }
    if consume:
        from dataclasses import dataclass

        @dataclass
        class _Planned:
            symbol: str
            side: str
            qty: float
            limit_price: float = price
            stance: str = "EXIT"
            client_order_id: str = "coid-198i"
            force_exit_override: bool = False
            ref_price: Optional[float] = price
            management_action: Optional[str] = "EXIT"

        from services.paper_sell_arm_authority import BrokerSnapshotView

        view = BrokerSnapshotView(
            positions={symbol: {"qty": float(qty), "qty_available": float(qty)}},
            open_order_symbols=set(),
            mode="paper",
            live_blocked=True,
            trading_date=str(auth.get("trading_date")),
            account_fingerprint=str(auth.get("account_fingerprint")),
            current_plan_hash=str(auth.get("plan_hash")),
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
            market_session="MARKET_OPEN",
        )
        guard = begin_paper_sell_dispatch(
            store_root=tmp_path,
            planned=[_Planned(symbol, "sell", qty, price)],
            broker_view=view,
            live_override=LIVE_BLOCKED,
            now=now + timedelta(minutes=1),
            results_root=tmp_path,
            live_root=tmp_path,
            db_path=tmp_path / "paper_sell_arm.sqlite3",
        )
        out["guard"] = guard
        assert guard.consumed, f"seed consume failed: {guard.reason} {guard.details}"
    return out


def install_boundary_trace(monkeypatch, trace: OrderedTrace) -> None:
    import services.paper_sell_arm_runtime_enforcement as enf

    real_enforce = enf.enforce_paper_sell_broker_submission
    real_begin = enf.begin_paper_sell_dispatch

    def _enforce(**kwargs):
        trace.add(
            "hardened_sell_boundary_entered",
            fn="enforce_paper_sell_broker_submission",
            side=kwargs.get("side"),
            symbol=kwargs.get("symbol"),
            qty=kwargs.get("qty"),
            source=kwargs.get("source"),
            mode=kwargs.get("mode"),
        )
        try:
            real_enforce(**kwargs)
            trace.add("validation_result", ok=True, reason="permitted")
        except Exception as exc:
            reason = getattr(exc, "reason", str(exc))
            trace.add("validation_result", ok=False, reason=reason)
            raise

    def _begin(**kwargs):
        trace.add("hardened_sell_boundary_entered", fn="begin_paper_sell_dispatch")
        result = real_begin(**kwargs)
        trace.add(
            "begin_paper_sell_dispatch_result",
            permitted=result.permitted,
            consumed=result.consumed,
            reason=result.reason,
        )
        return result

    monkeypatch.setattr(enf, "enforce_paper_sell_broker_submission", _enforce)
    monkeypatch.setattr(enf, "begin_paper_sell_dispatch", _begin)


def freeze_dispatch_clock(monkeypatch, now: datetime = SESSION_OPEN_NOW) -> None:
    import services.paper_sell_arm_authority as auth_mod
    import services.paper_sell_arm_runtime_enforcement as enf

    monkeypatch.setattr(auth_mod, "_utc_now", lambda: now)
    if hasattr(enf, "_utc_now"):
        monkeypatch.setattr(enf, "_utc_now", lambda: now)
    monkeypatch.setattr(enf, "_utc_now_iso", lambda: now.strftime("%Y-%m-%dT%H:%M:%SZ"))


def isolate_sell_store(monkeypatch, tmp_path: Path) -> None:
    import services.duplicate_order_guard as dog
    import services.manage_positions as mp
    import services.paper_execution_audit as pea
    import services.paper_sell_arm_authority as auth_mod
    import services.paper_sell_arm_runtime_enforcement as enf
    import services.place_live_orders as plo

    freeze_dispatch_clock(monkeypatch, SESSION_OPEN_NOW)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    monkeypatch.setattr(auth_mod, "LIVE_GOVERNANCE", tmp_path)
    monkeypatch.setattr(enf, "LIVE_GOVERNANCE", tmp_path)
    monkeypatch.setattr(plo, "IDEMPOTENCY_STATE_PATH", tmp_path / "idempotency_state.json")
    monkeypatch.setattr(plo, "DEFAULT_LOG_CSV", tmp_path / "live_orders_log.csv")
    monkeypatch.setattr(plo, "SIGNAL_STATE_PATH", tmp_path / "signal_state.json")
    monkeypatch.setattr(plo, "EXECUTION_INTELLIGENCE_CSV", tmp_path / "execution_intelligence.csv")
    monkeypatch.setattr(
        plo, "PERFORMANCE_RISK_OVERLAY_CSV", tmp_path / "performance_risk_overlay.csv"
    )
    monkeypatch.setattr(
        mp, "PERFORMANCE_RISK_OVERLAY_CSV", tmp_path / "performance_risk_overlay.csv"
    )
    monkeypatch.setattr(dog, "SESSION_INTENTS_CSV", tmp_path / "placement_session_intents.csv")
    monkeypatch.setattr(dog, "DEFAULT_LOG_PATH", tmp_path / "live_orders_log.csv")
    monkeypatch.setattr(pea, "AUDIT_CSV", tmp_path / "paper_execution_audit.csv")
    monkeypatch.setattr(plo, "_load_block_new_buy_set", lambda path=None: frozenset())
    import services.execution_drop_diagnostics as edd

    monkeypatch.setattr(edd, "DROP_JSON", tmp_path / "execution_drop_diagnostics.json")
    monkeypatch.setattr(edd, "DROP_CSV", tmp_path / "execution_drop_diagnostics.csv")
    monkeypatch.setattr(edd, "DROP_LOG_CSV", tmp_path / "execution_drop_diagnostics_log.csv")

    real_build = enf.build_broker_view_from_runtime_snapshots

    def _build(**kwargs):
        kwargs.setdefault("live_override", LIVE_BLOCKED)
        kwargs["store_root"] = tmp_path
        kwargs["results_root"] = tmp_path
        kwargs["live_root"] = tmp_path
        return real_build(**kwargs)

    monkeypatch.setattr(enf, "build_broker_view_from_runtime_snapshots", _build)


def stub_unrelated_pre_gates(monkeypatch) -> None:
    """Isolate master/pipeline/interlock/197x so SELL hardening can be reached.

    Does not stub place_live_orders production routing or the hardened SELL boundary.
    """
    from services.master_execution_gate import GateDecision

    def _ok_gate(self, mode="paper", **_k):
        if str(mode).lower() != "paper":
            return GateDecision(
                ok=False,
                status="BLOCKED",
                mode=mode,
                checked_at=datetime.now(timezone.utc).isoformat(),
                reasons=["LIVE_EXECUTION_BLOCKED"],
                warnings=[],
                summary="Live execution blocked.",
                details={"test_isolated": True, "live_blocked": True},
            )
        return GateDecision(
            ok=True,
            status="READY",
            mode=mode,
            checked_at=datetime.now(timezone.utc).isoformat(),
            reasons=[],
            warnings=["TEST_ISOLATED_PRE_GATE"],
            summary="Execution allowed (paper) — test isolated pre-gate.",
            details={"test_isolated": True},
        )

    monkeypatch.setattr(
        "services.master_execution_gate.MasterExecutionGate.evaluate",
        _ok_gate,
    )

    def _pipeline(**kwargs):
        mode = str(kwargs.get("mode") or "paper").lower()
        if mode != "paper":
            from services.triton_execution_gate import ExecutionGateBlocked

            raise ExecutionGateBlocked(
                {"allowed": False, "decision": "BLOCK", "reason": "live_blocked"}
            )
        return {"allowed": True, "decision": "ALLOW"}

    def _broker_gate(**kwargs):
        mode = str(getattr(kwargs.get("broker"), "mode", "paper") or "paper").lower()
        if mode != "paper":
            from services.triton_execution_gate import ExecutionGateBlocked

            raise ExecutionGateBlocked(
                {"allowed": False, "decision": "BLOCK", "reason": "live_blocked"}
            )
        return {"allowed": True, "decision": "ALLOW"}

    monkeypatch.setattr("services.triton_execution_gate.enforce_pipeline_gate", _pipeline)
    monkeypatch.setattr(
        "services.triton_execution_gate.enforce_broker_submission_gate",
        _broker_gate,
    )
    monkeypatch.setattr(
        "services.paper_execution_handoff_scope.should_enforce_approved_197x_handoff_scope",
        lambda **_k: False,
    )

    def _permit(**_k):
        return {
            "dispatch_permitted": True,
            "dispatch_interlock_state": "DISPATCH_PERMITTED",
            "dispatch_interlock_reason": "TEST_ISOLATED",
        }

    monkeypatch.setattr(
        "services.defensive_exit_dispatch_interlock.evaluate_defensive_exit_dispatch_interlock",
        _permit,
    )
    monkeypatch.setattr(
        "services.defensive_exit_dispatch_interlock.assert_defensive_exit_dispatch_permitted",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "services.defensive_exit_dispatch_interlock.constrain_handoff_to_approved_paper_scope",
        lambda planned, **_k: (list(planned), {"enforced": False}),
    )
    monkeypatch.setattr(
        "services.master_execution_gate.write_snapshot",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "services.master_execution_gate.append_gate_log_csv",
        lambda *_a, **_k: None,
    )


def inprocess_place_live_orders(cmd, **_kwargs):
    """Run the same argv place_live_orders.main would receive, in-process."""
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout

    args = list(cmd)
    # Expected: [python, -m, services.place_live_orders, ...flags]
    if "-m" in args:
        i = args.index("-m")
        argv = args[i + 2 :]
    else:
        argv = args[1:]
    old = sys.argv[:]
    buf = io.StringIO()
    sys.argv = ["services.place_live_orders"] + argv
    try:
        from services.place_live_orders import main

        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                main()
                rc = 0
            except SystemExit as exc:
                rc = int(exc.code or 0)
        text = buf.getvalue()
        return SimpleNamespace(returncode=rc, stdout=text, stderr=text)
    finally:
        sys.argv = old
