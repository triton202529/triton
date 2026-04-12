# services/broker_alpaca.py
from dotenv import load_dotenv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Force load .env file (project root; works regardless of cwd / IDE terminal settings)
load_dotenv(os.path.join(ROOT, ".env"))
load_dotenv(os.path.join(ROOT, "config", ".env"))

print("[ENV CHECK] ALPACA_KEY_ID =", os.getenv("ALPACA_KEY_ID"))
print("[ENV CHECK] APCA_API_KEY_ID =", os.getenv("APCA_API_KEY_ID"))

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Iterable
from zoneinfo import ZoneInfo

import requests

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE = "https://api.alpaca.markets"
ALPACA_DATA_BASE = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


def _ts() -> str:
    """UTC timestamp (ISO8601, Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_env_files() -> None:
    """
    Best-effort env loader:

    - Use already-set env vars if present
    - Else try ROOT/.env, ROOT/config/.env, ROOT/config/alpaca.json
    - Support both ALPACA_* and APCA_* keys
    """
    have_keys = (os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")) and (
        os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    )
    if have_keys:
        return

    def _load_env_file_plain(path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and os.getenv(k) is None:
                        os.environ[k] = v
        except FileNotFoundError:
            pass

    def load_json(path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in ("ALPACA_API_KEY_ID", "APCA_API_KEY_ID"):
                if k in data and os.getenv(k) is None:
                    os.environ[k] = str(data[k])
            for k in ("ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY"):
                if k in data and os.getenv(k) is None:
                    os.environ[k] = str(data[k])
        except FileNotFoundError:
            pass
        except Exception:
            # ignore malformed JSON
            pass

    _load_env_file_plain(os.path.join(ROOT, ".env"))
    _load_env_file_plain(os.path.join(ROOT, "config", ".env"))
    load_json(os.path.join(ROOT, "config", "alpaca.json"))


class AlpacaBroker:
    """
    Minimal Alpaca REST wrapper using requests.Session.
    Provides account/positions/orders endpoints + convenience helpers.
    """

    def __init__(self, mode: str = "paper", timeout: int = 20):
        _load_env_files()
        self.key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        self.secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not self.key or not self.secret:
            raise AlpacaError("Missing Alpaca API credentials in env.")
        self.base = ALPACA_PAPER_BASE if mode.lower() == "paper" else ALPACA_LIVE_BASE
        self.data_base = ALPACA_DATA_BASE
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self.timeout = timeout
        self.mode = mode.lower()

    # ---------- HTTP helpers ----------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base}{path}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code >= 300:
            raise AlpacaError(f"GET {path} failed {r.status_code}: {r.text}")
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base}{path}"
        r = self.session.post(url, data=json.dumps(payload), timeout=self.timeout)
        if r.status_code >= 300:
            raise AlpacaError(f"POST {path} failed {r.status_code}: {r.text}")
        return r.json()

    def _delete(self, path: str) -> Any:
        url = f"{self.base}{path}"
        r = self.session.delete(url, timeout=self.timeout)
        if r.status_code >= 300:
            raise AlpacaError(f"DELETE {path} failed {r.status_code}: {r.text}")
        if r.text:
            try:
                return r.json()
            except Exception:
                return {"status": r.status_code, "text": r.text}
        return {"status": r.status_code}

    # ---------- Data HTTP helpers ----------
    def _data_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        GET against Alpaca Market Data base.
        Uses the same session/headers (key+secret).
        """
        url = f"{self.data_base}{path}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code >= 300:
            raise AlpacaError(f"DATA GET {path} failed {r.status_code}: {r.text}")
        return r.json()

    # ---------- Normalizers ----------
    @staticmethod
    def _normalize_order(o: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact, consistent order dict suitable for logs/UI."""

        def _flt(x: Any) -> Optional[float]:
            try:
                return None if x in (None, "", "null") else float(x)
            except Exception:
                return None

        def _int_or_str(x: Any) -> Union[int, str, None]:
            if x in (None, "", "null"):
                return None
            try:
                xi = int(float(x))
                if str(xi) == str(x):
                    return xi
                return x
            except Exception:
                return x

        return {
            "id": o.get("id"),
            "client_order_id": o.get("client_order_id"),
            "symbol": o.get("symbol"),
            "side": o.get("side"),
            "type": o.get("type"),
            "time_in_force": o.get("time_in_force"),
            "status": o.get("status"),
            "qty": _int_or_str(o.get("qty")),
            "filled_qty": _int_or_str(o.get("filled_qty")),
            "notional": _flt(o.get("notional")),
            "filled_avg_price": _flt(o.get("filled_avg_price")),
            "limit_price": _flt(o.get("limit_price")),
            "stop_price": _flt(o.get("stop_price")),
            "submitted_at": str(o.get("submitted_at")),
            "created_at": str(o.get("created_at")),
            "updated_at": str(o.get("updated_at")),
        }

    # ---------- Account / market ----------
    def get_account(self) -> Dict[str, Any]:
        return self._get("/v2/account")

    def get_clock(self) -> Dict[str, Any]:
        return self._get("/v2/clock")

    def get_positions(self) -> List[Dict[str, Any]]:
        return self._get("/v2/positions")

    # Back-compat aliases used elsewhere
    def list_positions(self) -> List[Dict[str, Any]]:
        return self.get_positions()

    def account(self) -> Dict[str, Any]:
        return self.get_account()

    def clock(self) -> Dict[str, Any]:
        return self.get_clock()

    # ---------- Orders (raw) ----------
    def list_orders(
        self,
        status: str = "open",
        limit: int = 100,
        after: Optional[str] = None,
        until: Optional[str] = None,
        direction: str = "desc",
        nested: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Raw Alpaca orders list (no normalization).

        status: 'open' | 'closed' | 'all'
        direction: 'asc' | 'desc'
        nested: include legs for bracket/oco orders
        """
        params: Dict[str, Any] = {
            "status": status,
            "limit": limit,
            "direction": direction,
        }
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        if nested:
            params["nested"] = "true"
        return self._get("/v2/orders", params=params)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._get(f"/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._delete(f"/v2/orders/{order_id}")

    # Optional alias some wrappers expect
    def cancel(self, order_id: str) -> Dict[str, Any]:
        return self.cancel_order(order_id)

    def cancel_all_orders(self) -> Any:
        """Cancel all open orders."""
        return self._delete("/v2/orders")

    # ---------- Data (quotes / trades / prices) ----------
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Alpaca Data v2: latest quote for symbol.
        Returns dict with bid/ask + timestamp, or None on error.
        """
        sym = str(symbol).upper().strip()
        try:
            data = self._data_get(f"/v2/stocks/{sym}/quotes/latest")
            q = data.get("quote") or {}
            return {
                "symbol": sym,
                "bid": q.get("bp"),
                "ask": q.get("ap"),
                "bid_size": q.get("bs"),
                "ask_size": q.get("as"),
                "ts": q.get("t"),
                "raw": data,
            }
        except Exception:
            return None

    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Alpaca Data v2: latest trade for symbol.
        Returns dict with last price + timestamp, or None on error.
        """
        sym = str(symbol).upper().strip()
        try:
            data = self._data_get(f"/v2/stocks/{sym}/trades/latest")
            t = data.get("trade") or {}
            return {
                "symbol": sym,
                "last": t.get("p"),
                "size": t.get("s"),
                "exchange": t.get("x"),
                "ts": t.get("t"),
                "raw": data,
            }
        except Exception:
            return None

    # Friendly names expected by hardening code
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.get_latest_quote(symbol)

    def get_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.get_latest_trade(symbol)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Alpaca Data v2: latest trade price for symbol.
        Returns float or None on error.
        """
        t = self.get_latest_trade(symbol)
        if not t:
            return None
        try:
            return float(t["last"])
        except Exception:
            return None

    # ---------- Convenience: open orders ----------
    def get_open_orders(
        self,
        symbols: Optional[Iterable[str]] = None,
        limit: int = 200,
        nested: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return normalized open orders. Optional `symbols` filter.
        nested=True includes legs for bracket orders.
        """
        raw = self.list_orders(status="open", limit=limit, direction="desc", nested=nested)
        if symbols:
            symset = {s.upper() for s in symbols}
            raw = [o for o in raw if str(o.get("symbol", "")).upper() in symset]
        return [self._normalize_order(o) for o in raw]

    # ---------- Maintenance: cancel with whitelist / cutoff ----------
    def cancel_open_orders(
        self,
        whitelist: Optional[Iterable[str]] = None,
        cutoff: Optional[str] = None,  # ISO date/time, e.g. "2025-09-01" or "2025-09-01T15:30:00"
        older_than_days: Optional[int] = None,  # 0 = older than today's start-of-day (US/Eastern)
        limit: int = 500,
        dry_run: bool = False,
        nested: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Cancel open orders, keeping a whitelist and/or anything newer than a cutoff.

        If neither `whitelist` nor (`cutoff` or `older_than_days`) is provided, does nothing.

        Logic:
          - Cancel if NOT in whitelist, OR submitted BEFORE cutoff/start-of-day-window.
          - `older_than_days=0` uses today's 00:00 US/Eastern as cutoff; 1 ⇒ yesterday 00:00 ET, etc.

        Returns a summary list of affected orders (normalized). If dry_run=True, adds {"dry_run": True}.
        """
        if whitelist is None and cutoff is None and older_than_days is None:
            return []

        def _parse_ts(s: Any) -> Optional[datetime]:
            if not s:
                return None
            try:
                t = str(s).replace("Z", "+00:00")
                # Trim excessive fractional seconds to microseconds if present
                if "." in t:
                    # split off timezone if present
                    if "+" in t[t.index(".") :] or "-" in t[t.index(".") :]:
                        head, rest = t.split(".", 1)
                        # locate sign of tz offset in remainder
                        if "+" in rest:
                            frac, tz = rest.split("+", 1)
                            t = f"{head}.{frac[:6]}+{tz}"
                        elif "-" in rest:
                            frac, tz = rest.split("-", 1)
                            t = f"{head}.{frac[:6]}-{tz}"
                    else:
                        head, frac = t.split(".", 1)
                        t = f"{head}.{frac[:6]}"
                return datetime.fromisoformat(t)
            except Exception:
                return None

        wl = {s.upper() for s in whitelist} if whitelist else None

        # Determine cutoff
        if cutoff:
            cutoff_dt = _parse_ts(cutoff)
            if cutoff_dt is not None:
                cutoff_dt = cutoff_dt.replace(tzinfo=None)
        elif older_than_days is not None:
            # Use start-of-day in US/Eastern to mirror trading sessions
            now_et = datetime.now(ZoneInfo("America/New_York"))
            sod_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
            sod_et -= timedelta(days=older_than_days)
            # compare with naive UTC
            cutoff_dt = sod_et.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            cutoff_dt = None

        raw = self.list_orders(status="open", limit=limit, direction="desc", nested=nested)
        victims = []
        for o in raw:
            sym = str(o.get("symbol", "")).upper()
            ts = _parse_ts(o.get("submitted_at")) or _parse_ts(o.get("created_at"))
            ts_naive = ts.replace(tzinfo=None) if ts is not None else None

            keep_by_wl = wl is not None and sym in wl
            too_old = cutoff_dt is not None and ts_naive is not None and ts_naive < cutoff_dt

            # Cancel if not whitelisted OR too old
            cancel = (wl is None or not keep_by_wl) or too_old
            if cancel:
                victims.append(o)

        out = [self._normalize_order(o) for o in victims]
        if dry_run:
            for d in out:
                d["dry_run"] = True
            return out

        for o in victims:
            try:
                self.cancel_order(o.get("id"))
            except Exception as e:
                # record error but continue
                for d in out:
                    if d["id"] == o.get("id"):
                        d["cancel_error"] = str(e)
                        break

        for d in out:
            d["cancelled"] = "cancel_error" not in d
        return out

    # ---------- Core order APIs (qty-based) ----------
    def submit_order(
        self,
        symbol: str,
        qty: Union[int, float],
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        order_class: Optional[str] = None,
        take_profit: Optional[Dict[str, Any]] = None,
        stop_loss: Optional[Dict[str, Any]] = None,
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """
        Generic submit_order. For fractional qty, pass a float (if your acct supports it).

        Alpaca expects either qty OR notional (not both). This method is qty-based.
        """
        payload: Dict[str, Any] = {
            "symbol": symbol.upper().strip(),
            "qty": qty,
            "side": side.lower().strip(),
            "type": order_type.lower().strip(),
            "time_in_force": time_in_force.lower().strip(),
            "extended_hours": bool(extended_hours),
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        if limit_price is not None:
            payload["limit_price"] = round(float(limit_price), 4)
        if stop_price is not None:
            payload["stop_price"] = round(float(stop_price), 4)
        if order_class:
            payload["order_class"] = str(order_class).lower().strip()
        if take_profit:
            payload["take_profit"] = take_profit
        if stop_loss:
            payload["stop_loss"] = stop_loss
        return self._post("/v2/orders", payload)

    def submit_market_order(
        self,
        symbol: str,
        qty: Union[int, float],
        side: str,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """Convenience: market order by quantity."""
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="market",
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            extended_hours=extended_hours,
        )

    def submit_limit_order(
        self,
        symbol: str,
        qty: Union[int, float],
        side: str,
        limit_price: float,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """Convenience: limit order by quantity."""
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="limit",
            time_in_force=time_in_force,
            limit_price=limit_price,
            client_order_id=client_order_id,
            extended_hours=extended_hours,
        )

    # ---------- Notional order helpers (market) ----------
    def submit_order_notional(
        self,
        symbol: str,
        notional: float,
        side: str,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """
        Generic notional market order. Alpaca supports 'notional' instead of 'qty'.
        """
        payload: Dict[str, Any] = {
            "symbol": symbol.upper().strip(),
            "notional": round(float(abs(notional)), 2),
            "side": side.lower().strip(),
            "type": "market",
            "time_in_force": time_in_force.lower().strip(),
            "extended_hours": bool(extended_hours),
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        return self._post("/v2/orders", payload)

    # Aliases other modules use:
    def place_order_notional(
        self, symbol: str, notional: float, side: str, tif: str = "day"
    ) -> Dict[str, Any]:
        return self.submit_order_notional(
            symbol=symbol, notional=notional, side=side, time_in_force=tif
        )

    def submit_market_order_notional(
        self, symbol: str, notional: float, side: str, tif: str = "day"
    ) -> Dict[str, Any]:
        return self.submit_order_notional(
            symbol=symbol, notional=notional, side=side, time_in_force=tif
        )

    # Backward-compat alias some codebases use:
    def place_order(
        self,
        symbol: str,
        qty: Union[int, float],
        side: str,
        order_type: str = "market",
        tif: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            time_in_force=tif,
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )

    # ---------- Bracket orders ----------
    def place_bracket_market(
        self,
        symbol: str,
        qty: Union[int, float],
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        tif: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Market bracket (TP/SL) by quantity.
        """
        payload: Dict[str, Any] = {
            "symbol": symbol.upper().strip(),
            "qty": qty,
            "side": side.lower().strip(),
            "type": "market",
            "time_in_force": tif.lower().strip(),
            "order_class": "bracket",
            "take_profit": {"limit_price": round(float(take_profit_price), 4)},
            "stop_loss": {"stop_price": round(float(stop_loss_price), 4)},
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        return self._post("/v2/orders", payload)

    def place_bracket(
        self,
        symbol: str,
        side: str,
        qty: Union[int, float],
        entry_type: str = "market",  # "market" or "limit"
        entry_limit_price: Optional[float] = None,
        take_profit_price: float = 0.0,
        stop_loss_price: float = 0.0,
        tif: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Bracket order with MARKET or LIMIT entry (qty-based).

        - entry_type="market": parent fills at market
        - entry_type="limit": parent rests at entry_limit_price until filled

        take_profit_price -> TP leg limit_price
        stop_loss_price   -> SL leg stop_price
        """
        entry_type = (entry_type or "market").lower().strip()
        if entry_type not in ("market", "limit"):
            raise AlpacaError("entry_type must be 'market' or 'limit'")

        if float(take_profit_price) <= 0 or float(stop_loss_price) <= 0:
            raise AlpacaError("take_profit_price and stop_loss_price must be > 0")

        if entry_type == "limit":
            if entry_limit_price is None or float(entry_limit_price) <= 0:
                raise AlpacaError("entry_limit_price required for limit entry")

        # Optional safety sanity (not blocking, but avoids obvious mistakes)
        side_l = (side or "").lower().strip()
        if side_l == "buy" and not (float(stop_loss_price) < float(take_profit_price)):
            raise AlpacaError("BUY bracket requires stop_loss_price < take_profit_price")
        if side_l == "sell" and not (float(take_profit_price) < float(stop_loss_price)):
            raise AlpacaError("SELL bracket requires take_profit_price < stop_loss_price")

        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=entry_type,
            time_in_force=tif,
            limit_price=entry_limit_price if entry_type == "limit" else None,
            client_order_id=client_order_id,
            order_class="bracket",
            take_profit={"limit_price": round(float(take_profit_price), 4)},
            stop_loss={"stop_price": round(float(stop_loss_price), 4)},
            extended_hours=False,
        )
