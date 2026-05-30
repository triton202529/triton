# services/execution_guard.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from services.broker_alpaca import AlpacaBroker  # uses your existing broker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "execution_guard.json"
EXECUTE_TRADES_CONFIG_PATH = PROJECT_ROOT / "config" / "execute_trades.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KILL_SWITCH_PATH = RESULTS_DIR / "kill_switch.json"
EXEC_AUDIT_PATH = RESULTS_DIR / "execution_audit.csv"


@dataclass
class GuardDecision:
    ok: bool
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "code": self.code, "message": self.message, "context": self.context}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "" or str(x).lower() == "null":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "" or str(x).lower() == "null":
            return None
        return int(float(x))
    except Exception:
        return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_audit_row(row: Dict[str, Any]) -> None:
    """
    Append an audit record (best-effort).
    Safe: never throws.
    """
    try:
        df = pd.DataFrame([row])
        if EXEC_AUDIT_PATH.exists() and EXEC_AUDIT_PATH.stat().st_size > 0:
            df.to_csv(EXEC_AUDIT_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(EXEC_AUDIT_PATH, mode="w", header=True, index=False)
    except Exception:
        pass


class ExecutionGuard:
    """
    Phase 2.2 — Execution Guardrails

    This module is the *single gate* in front of all order placement:
    - Manual Order Desk
    - CSV Runner
    - (future) auto-execution

    It blocks trades when:
    - kill switch is ON
    - symbol is not allowed
    - buying power insufficient (optional strict)
    - duplicate/cooldown violations
    - order sizing violates max limits
    - bracket prices invalid (TP/SL sanity)

    Defaults are conservative and configurable via config/execution_guard.json.
    """

    def __init__(self, broker: AlpacaBroker):
        self.broker = broker
        self.cfg = self._load_config()

    # -----------------------------
    # Config / switches
    # -----------------------------
    def _default_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            # Safety switches
            "block_live_mode": True,  # block if broker.mode == "live"
            "require_market_open": False,  # optional; if True blocks when market closed (based on broker clock)
            "kill_switch_path": str(KILL_SWITCH_PATH),
            # Symbols
            "allowlist": [],  # if non-empty, ONLY these can trade
            "denylist": [],  # always blocked
            # Size / exposure
            "min_notional_usd": 25.0,  # block tiny trades
            "max_notional_usd": 1500.0,  # per-order cap
            "max_qty": 200,  # per-order share cap (qty orders)
            "max_open_orders": 25,  # block if too many open orders
            "max_positions": 25,  # block if too many open positions
            # Buying power rules
            "enforce_buying_power": True,  # if True, block if cost > buying_power (estimate if needed)
            "bp_buffer_pct": 0.05,  # keep 5% buffer (capital preservation)
            # Duplicate / cooldown
            "cooldown_minutes": 10,  # block same symbol+side repeats inside this window
            "audit_path": str(EXEC_AUDIT_PATH),
        }

    def _load_config(self) -> Dict[str, Any]:
        cfg = self._default_config()
        try:
            if CONFIG_PATH.exists() and CONFIG_PATH.stat().st_size > 0:
                user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(user_cfg, dict):
                    cfg.update(user_cfg)
        except Exception:
            # keep defaults
            pass
        self._merge_execute_trades_position_cap(cfg)
        return cfg

    @staticmethod
    def _merge_execute_trades_position_cap(cfg: Dict[str, Any]) -> None:
        """
        Prefer max_positions from config/execute_trades.json (trading config).
        Legacy: max_portfolio_positions if max_positions is absent.
        """
        try:
            if not EXECUTE_TRADES_CONFIG_PATH.is_file():
                return
            u = json.loads(EXECUTE_TRADES_CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(u, dict):
                return
            mp = u.get("max_positions")
            if mp is None:
                mp = u.get("max_portfolio_positions")
            if mp is not None:
                cfg["max_positions"] = max(1, int(mp))
                cfg["_position_cap_source"] = "execute_trades"
        except Exception:
            pass

    def kill_switch_on(self) -> Tuple[bool, str]:
        """
        kill_switch.json example:
        { "enabled": true, "reason": "maintenance", "updated_at": "2025-12-27T00:00:00Z" }
        """
        ks_path = Path(self.cfg.get("kill_switch_path", str(KILL_SWITCH_PATH)))
        ks = _load_json(ks_path)
        if not ks:
            return False, ""
        enabled = bool(ks.get("enabled", False))
        reason = str(ks.get("reason", "") or "")
        return enabled, reason

    # -----------------------------
    # Core checks
    # -----------------------------
    def _check_mode(self) -> Optional[GuardDecision]:
        if not self.cfg.get("enabled", True):
            return GuardDecision(
                True,
                "GUARD_DISABLED",
                "Guard disabled in config; allowing order (NOT RECOMMENDED).",
            )

        if (
            self.cfg.get("block_live_mode", True)
            and getattr(self.broker, "mode", "paper") == "live"
        ):
            return GuardDecision(
                False, "LIVE_BLOCKED", "Live mode blocked by execution guard config."
            )

        ks_on, ks_reason = self.kill_switch_on()
        if ks_on:
            return GuardDecision(
                False,
                "KILL_SWITCH",
                f"Kill switch enabled. {ks_reason}".strip(),
                {"reason": ks_reason},
            )

        if self.cfg.get("require_market_open", False):
            try:
                clock = self.broker.get_clock()
                if not bool(clock.get("is_open", False)):
                    next_open = str(clock.get("next_open") or "")
                    return GuardDecision(
                        False,
                        "MARKET_CLOSED",
                        "Market is closed (guard requires open).",
                        {"next_open": next_open},
                    )
            except Exception as e:
                return GuardDecision(
                    False, "CLOCK_FAIL", "Unable to verify market clock.", {"error": str(e)}
                )

        return None

    def _check_symbol(self, symbol: str) -> Optional[GuardDecision]:
        sym = (symbol or "").upper().strip()
        if not sym:
            return GuardDecision(False, "NO_SYMBOL", "Missing symbol.")

        deny = {s.upper() for s in (self.cfg.get("denylist") or []) if isinstance(s, str)}
        if sym in deny:
            return GuardDecision(
                False, "SYMBOL_DENYLIST", f"{sym} is blocked by denylist.", {"symbol": sym}
            )

        allow = [s.upper() for s in (self.cfg.get("allowlist") or []) if isinstance(s, str)]
        if allow and sym not in set(allow):
            return GuardDecision(
                False,
                "SYMBOL_NOT_ALLOWED",
                f"{sym} not in allowlist.",
                {"symbol": sym, "allowlist": allow},
            )

        return None

    @staticmethod
    def _is_manage_exposure_reduce(payload: Optional[Dict[str, Any]]) -> bool:
        """
        True only for trim/exit sells from manage_positions (explicit markers).
        Used to skip entry-style MAX_POSITIONS; does not apply to BUY/ADD or generic sells.
        """
        if not payload or not isinstance(payload, dict):
            return False
        if str(payload.get("side", "")).lower().strip() != "sell":
            return False
        if str(payload.get("order_source", "")).lower().strip() != "manage_positions":
            return False
        ma = str(payload.get("management_action", "")).upper().strip()
        return ma in ("EXIT", "TRIM")

    def _check_open_counts(
        self, payload: Optional[Dict[str, Any]] = None
    ) -> Optional[GuardDecision]:
        # Positions (entry-style cap: skip for verified manage_positions EXIT/TRIM sells)
        skip_max_pos = self._is_manage_exposure_reduce(payload)
        if not skip_max_pos:
            try:
                positions = self.broker.get_positions()
                max_pos = int(self.cfg.get("max_positions", 25))
                if isinstance(positions, list) and len(positions) >= max_pos:
                    return GuardDecision(
                        False,
                        "MAX_POSITIONS",
                        f"Too many open positions ({len(positions)} >= {max_pos}).",
                    )
            except Exception:
                # don't hard-fail; execution should still be possible in degraded state if you want
                pass

        # Open orders
        try:
            open_orders = self.broker.list_orders(status="open", limit=500, direction="desc")
            max_oo = int(self.cfg.get("max_open_orders", 25))
            if isinstance(open_orders, list) and len(open_orders) >= max_oo:
                return GuardDecision(
                    False,
                    "MAX_OPEN_ORDERS",
                    f"Too many open orders ({len(open_orders)} >= {max_oo}).",
                )
        except Exception:
            pass

        return None

    def _estimate_notional(self, payload: Dict[str, Any]) -> Optional[float]:
        """
        Return notional if explicitly provided, else try qty * latest_price (best-effort).
        """
        notional = _safe_float(payload.get("notional"))
        if notional is not None:
            return abs(notional)

        qty = _safe_float(payload.get("qty"))
        if qty is None:
            return None

        sym = str(payload.get("symbol", "")).upper().strip()
        px = self.broker.get_latest_price(sym) if sym else None
        if px is None:
            return None
        return abs(float(qty) * float(px))

    def _check_size_limits(self, payload: Dict[str, Any]) -> Optional[GuardDecision]:
        # Qty limit
        qty = _safe_float(payload.get("qty"))
        if qty is not None:
            max_qty = _safe_float(self.cfg.get("max_qty", 200)) or 200.0
            if qty > max_qty:
                return GuardDecision(
                    False,
                    "MAX_QTY",
                    f"Qty {qty:g} exceeds max_qty {max_qty:g}.",
                    {"qty": qty, "max_qty": max_qty},
                )

        # Notional limits
        est_notional = self._estimate_notional(payload)
        if est_notional is not None:
            mn = _safe_float(self.cfg.get("min_notional_usd", 25.0)) or 0.0
            mx = _safe_float(self.cfg.get("max_notional_usd", 1500.0)) or float("inf")

            if est_notional < mn:
                return GuardDecision(
                    False,
                    "MIN_NOTIONAL",
                    f"Estimated notional ${est_notional:,.2f} below min ${mn:,.2f}.",
                    {"notional": est_notional, "min_notional": mn},
                )
            if est_notional > mx and not self._is_manage_exposure_reduce(payload):
                return GuardDecision(
                    False,
                    "MAX_NOTIONAL",
                    f"Estimated notional ${est_notional:,.2f} exceeds max ${mx:,.2f}.",
                    {"notional": est_notional, "max_notional": mx},
                )

        return None

    def _check_buying_power(self, payload: Dict[str, Any]) -> Optional[GuardDecision]:
        if not self.cfg.get("enforce_buying_power", True):
            return None

        est_notional = self._estimate_notional(payload)
        # If we can't estimate (no notional + no price), don't hard-block.
        if est_notional is None:
            return None

        try:
            acct = self.broker.get_account()
            bp = _safe_float(acct.get("buying_power"))
            if bp is None:
                return None
            buffer_pct = _safe_float(self.cfg.get("bp_buffer_pct", 0.05)) or 0.0
            allowed = bp * (1.0 - buffer_pct)
            if est_notional > allowed:
                return GuardDecision(
                    False,
                    "INSUFFICIENT_BP",
                    f"Estimated notional ${est_notional:,.2f} exceeds allowed buying power ${allowed:,.2f} (bp=${bp:,.2f}, buffer={buffer_pct:.0%}).",
                    {
                        "notional": est_notional,
                        "bp": bp,
                        "allowed": allowed,
                        "buffer_pct": buffer_pct,
                    },
                )
        except Exception as e:
            return GuardDecision(
                False, "BP_CHECK_FAIL", "Could not verify buying power.", {"error": str(e)}
            )

        return None

    def _check_cooldown(self, payload: Dict[str, Any]) -> Optional[GuardDecision]:
        cooldown_min = _safe_int(self.cfg.get("cooldown_minutes", 10)) or 0
        if cooldown_min <= 0:
            return None

        sym = str(payload.get("symbol", "")).upper().strip()
        side = str(payload.get("side", "")).lower().strip()
        if not sym or side not in ("buy", "sell"):
            return None

        try:
            # look at recent open + recently closed
            recent = self.broker.list_orders(status="all", limit=200, direction="desc")
            if not isinstance(recent, list) or not recent:
                return None

            cutoff = _utc_now() - timedelta(minutes=cooldown_min)

            for o in recent:
                if str(o.get("symbol", "")).upper().strip() != sym:
                    continue
                if str(o.get("side", "")).lower().strip() != side:
                    continue

                ts_raw = o.get("submitted_at") or o.get("created_at") or o.get("updated_at")
                if not ts_raw:
                    continue
                # Alpaca ISO is usually Z
                try:
                    ts = pd.to_datetime(str(ts_raw), utc=True, errors="coerce")
                    if ts is pd.NaT:
                        continue
                    if ts.to_pydatetime() > cutoff:
                        return GuardDecision(
                            False,
                            "COOLDOWN",
                            f"Cooldown active for {sym} {side}. Recent order within {cooldown_min}m.",
                            {"symbol": sym, "side": side, "cooldown_minutes": cooldown_min},
                        )
                except Exception:
                    continue

        except Exception:
            # if this fails, don't block trading
            return None

        return None

    def _check_bracket_sanity(self, payload: Dict[str, Any]) -> Optional[GuardDecision]:
        """
        If order_class == 'bracket', enforce basic sanity:
        - take_profit.limit_price exists and is > 0
        - stop_loss.stop_price exists and is > 0
        - buy: TP > entry_estimate, SL < entry_estimate
        - sell: TP < entry_estimate, SL > entry_estimate
        For market orders, entry_estimate uses latest price best-effort.
        """
        oc = str(payload.get("order_class") or "").lower().strip()
        if oc != "bracket":
            return None

        side = str(payload.get("side", "")).lower().strip()
        sym = str(payload.get("symbol", "")).upper().strip()

        tp = payload.get("take_profit") or {}
        sl = payload.get("stop_loss") or {}

        tp_px = _safe_float(tp.get("limit_price"))
        sl_px = _safe_float(sl.get("stop_price"))

        if tp_px is None or tp_px <= 0:
            return GuardDecision(
                False,
                "BRACKET_NO_TP",
                "Bracket requires take_profit.limit_price.",
                {"take_profit": tp},
            )
        if sl_px is None or sl_px <= 0:
            return GuardDecision(
                False, "BRACKET_NO_SL", "Bracket requires stop_loss.stop_price.", {"stop_loss": sl}
            )

        # Entry estimate
        entry_est = None
        if payload.get("type") == "limit":
            entry_est = _safe_float(payload.get("limit_price"))
        if entry_est is None and sym:
            entry_est = self.broker.get_latest_price(sym)

        if entry_est is None:
            # Can't sanity-check relationship without an entry estimate; allow.
            return None

        if side == "buy":
            if not (tp_px > entry_est):
                return GuardDecision(
                    False,
                    "BRACKET_TP_INVALID",
                    f"BUY bracket TP must be > entry_est ({tp_px} <= {entry_est}).",
                    {"tp": tp_px, "entry_est": entry_est},
                )
            if not (sl_px < entry_est):
                return GuardDecision(
                    False,
                    "BRACKET_SL_INVALID",
                    f"BUY bracket SL must be < entry_est ({sl_px} >= {entry_est}).",
                    {"sl": sl_px, "entry_est": entry_est},
                )
        elif side == "sell":
            if not (tp_px < entry_est):
                return GuardDecision(
                    False,
                    "BRACKET_TP_INVALID",
                    f"SELL bracket TP must be < entry_est ({tp_px} >= {entry_est}).",
                    {"tp": tp_px, "entry_est": entry_est},
                )
            if not (sl_px > entry_est):
                return GuardDecision(
                    False,
                    "BRACKET_SL_INVALID",
                    f"SELL bracket SL must be > entry_est ({sl_px} <= {entry_est}).",
                    {"sl": sl_px, "entry_est": entry_est},
                )

        return None

    # -----------------------------
    # Public API
    # -----------------------------
    def validate(self, payload: Dict[str, Any]) -> GuardDecision:
        """
        Validate an order payload (Alpaca order schema-ish).
        Does NOT place the order.
        """
        # Mode / kill switch
        d = self._check_mode()
        if d is not None and not d.ok:
            return d

        # Basic fields
        sym = str(payload.get("symbol", "")).upper().strip()
        d = self._check_symbol(sym)
        if d is not None:
            return d

        # Account state (pass payload so manage_positions trim/exit can skip MAX_POSITIONS)
        d = self._check_open_counts(payload)
        if d is not None:
            return d

        # Size / BP / cooldown
        d = self._check_size_limits(payload)
        if d is not None:
            return d

        d = self._check_buying_power(payload)
        if d is not None:
            return d

        d = self._check_cooldown(payload)
        if d is not None:
            return d

        d = self._check_bracket_sanity(payload)
        if d is not None:
            return d

        return GuardDecision(True, "OK", "Approved by Execution Guard.", {"symbol": sym})

    def validate_and_audit(self, payload: Dict[str, Any], source: str) -> GuardDecision:
        """
        Validate and write an audit row (approve/deny).
        """
        decision = self.validate(payload)

        row = {
            "ts_utc": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": getattr(self.broker, "mode", "unknown"),
            "source": source,
            "symbol": str(payload.get("symbol", "")).upper().strip(),
            "side": str(payload.get("side", "")).lower().strip(),
            "type": str(payload.get("type", "")).lower().strip(),
            "tif": str(payload.get("time_in_force", "")).lower().strip(),
            "order_class": str(payload.get("order_class", "")).lower().strip(),
            "qty": payload.get("qty"),
            "notional": payload.get("notional"),
            "limit_price": payload.get("limit_price"),
            "stop_price": payload.get("stop_price"),
            "tp": (
                (payload.get("take_profit") or {}).get("limit_price")
                if isinstance(payload.get("take_profit"), dict)
                else None
            ),
            "sl": (
                (payload.get("stop_loss") or {}).get("stop_price")
                if isinstance(payload.get("stop_loss"), dict)
                else None
            ),
            "ok": bool(decision.ok),
            "code": decision.code,
            "message": decision.message,
        }
        _write_audit_row(row)
        return decision
