# services/place_live_orders.py
"""
TRITON — Order Placement (Paper/Live) with guardrails + idempotency + quote sanity.

Default: --dry-run (no broker.submit_order). Use --no-dry-run to submit.

HARDENING:
- Uses AlpacaBroker.submit_order signature correctly
- Duplicate open-order guard (broker-based; prevents pileups)
- Market-data sanity checks using broker.get_quote/get_trade
- Ref-price selection handles bad quotes (ask=0) by preferring last trade
- Limit clamp (open-safe buffer)
- Max batch notional cap (applies to LIMIT + MARKET via ref estimate)
- Idempotency only records batch OK if placed>0 and failed==0

NEW SAFETY:
- Position-aware illegal SELL protection (no accidental shorts):
  * Default: block entire batch if any SELL has no available position
  * --drop-illegal-sells: drop illegal SELL rows and continue
  * --allow-shorts: allow shorts (explicit override)
- Market clock gate for --mode live:
  * Default: block if market is closed
  * --ignore-market-closed: override (use carefully)
- Optional extended-hours flag:
  * --extended-hours sets extended_hours=True on submitted orders

TICK-SIZE / PRICE INCREMENT HARDENING (FIXES Alpaca 422 sub-penny errors):
- Enforces valid limit-price increments:
  * price >= 1.00 -> tick = 0.01
  * price <  1.00 -> tick = 0.0001 (safe default)
- BUY limits quantized DOWN to tick (never raises your max)
- SELL limits quantized UP to tick (never lowers your min)

SESSION + POLL COMPAT (IMPORTANT):
- Writes rows to data/results/live_orders_log.csv using SAME schema poll_order_status expects:
  timestamp, session, action, symbol, side, qty, type, limit_price, order_id, status,
  filled_qty, filled_avg_price, client_order_id, tp_limit, sl_stop
- Placement rows are logged with action="submit"
- Cancel-dupes rows are logged with action="cancel" (audit trail)

PATCH (2026-04-01):
- ✅ Execution source priority (default path): trade_opportunities.csv → signal_lifecycle_effective.csv → signal_lifecycle.csv
- ✅ Explicit --orders path uses that file; effective_* column mapping unchanged for lifecycle sources

PATCH (2026-02-01):
- ✅ CRITICAL: log writer is schema-safe and snapshot-guarded:
    - refuses to write to a file that looks like a snapshot artifact (snapshot_ts header)
    - upgrades by COLUMN NAMES (not positional)
    - tolerates extra legacy columns without shifting
- ✅ Always logs broker order UUID in order_id when available
  (prevents the "order_id became limit_price" corruption seen previously)
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from services.execution_drop_diagnostics import (
    DROP_JSON,
    finalize_artifacts,
    make_row,
    read_json as read_drop_json,
    recompute_summary_counts,
)
from services.execution_intelligence import (
    ExecutionIntelligenceConfig,
    annotate_order as _ei_annotate_order,
    ANNOTATE_ORDER_KEYS as _EI_KEYS,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIFECYCLE_CSV = ROOT / "data" / "results" / "signal_lifecycle.csv"
EFFECTIVE_LIFECYCLE_PATH = ROOT / "data" / "results" / "signal_lifecycle_effective.csv"
TRADE_OPPORTUNITIES_PATH = ROOT / "data" / "results" / "trade_opportunities.csv"
DEFAULT_ORDERS_CSV = DEFAULT_LIFECYCLE_CSV
DEFAULT_MANAGE_ORDERS_CSV = ROOT / "data" / "live" / "manage_orders.csv"
PERFORMANCE_RISK_OVERLAY_CSV = ROOT / "data" / "results" / "performance_risk_overlay.csv"

# IMPORTANT: poll_order_status now writes/reads live_orders_log.csv
DEFAULT_LOG_CSV = ROOT / "data" / "results" / "live_orders_log.csv"

IDEMPOTENCY_STATE_PATH = ROOT / "data" / "live" / "idempotency_state.json"
SIGNAL_STATE_PATH = ROOT / "data" / "results" / "signal_state.json"

ACTIONABLE_STANCES = {"BUY", "ADD", "TRIM", "EXIT", "ROTATE_EXIT"}

# Cap exploratory rows from trade_opportunities.csv only (strict + lifecycle paths unchanged).
MAX_EXPLORATION_POSITIONS = 3

SYMBOL_ALIASES = {
    "BRK-B": "BRK.B",
    "BF-B": "BF.B",
}

LOG_FIELDS = [
    "timestamp",
    "session",
    "action",
    "symbol",
    "side",
    "qty",
    "type",
    "limit_price",
    "order_id",
    "status",
    "filled_qty",
    "filled_avg_price",
    "client_order_id",
    "tp_limit",
    "sl_stop",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


# Placement diagnostics (best-effort; never raises from hooks)
_PLACE_DIAG_ROWS: List[Dict[str, Any]] = []
_PLACE_DIAG_FINALIZED = False
_PLACE_CTX: Dict[str, Any] = {}


def _reset_place_diag() -> None:
    global _PLACE_DIAG_ROWS, _PLACE_DIAG_FINALIZED, _PLACE_CTX
    _PLACE_DIAG_ROWS = []
    _PLACE_DIAG_FINALIZED = False
    _PLACE_CTX = {}


def _finalize_place_diag_once(
    blocked: bool = False, extra: Optional[Dict[str, Any]] = None
) -> None:
    global _PLACE_DIAG_FINALIZED
    if _PLACE_DIAG_FINALIZED:
        return
    _PLACE_DIAG_FINALIZED = True
    try:
        mode = str(_PLACE_CTX.get("mode") or "paper")
        rows = list(_PLACE_DIAG_ROWS)
        p = recompute_summary_counts(rows, mode, blocked=blocked)
        if extra:
            p.update(extra)
        p.setdefault("source", "place_live_orders")
        finalize_artifacts(p, write_log=True)
    except Exception:
        pass


def _append_die_row_from_msg(msg: str, code: int) -> None:
    if not _PLACE_CTX:
        return
    rc = "UNKNOWN_DROP_REASON"
    detail = msg
    if "[BATCH_EMPTY]" in msg or "BATCH_EMPTY" in msg:
        rc = "BATCH_EMPTY"
    elif "[MASTER_GATE]" in msg or "MASTER_GATE" in msg:
        rc = "PLACEMENT_BLOCKED"
    elif "[DUPLICATE_OPEN_BLOCK]" in msg:
        rc = "DUPLICATE_OPEN_ORDER"
    elif "[ILLEGAL_SELL_BLOCK]" in msg or "[ILLEGAL_SELL" in msg:
        rc = "ILLEGAL_SELL"
    elif "[BATCH_NOTIONAL_BLOCK]" in msg or "BATCH_NOTIONAL" in msg:
        rc = "MAX_NOTIONAL"
    elif "[BAD_LIMIT]" in msg:
        rc = "FILTERED_BY_PLACE_LIVE_ORDERS"
    elif "[MARKETDATA_BLOCK]" in msg or "MARKETDATA" in msg:
        rc = "PRICE_UNAVAILABLE"
    elif "[LIMIT_SANITY_BLOCK]" in msg:
        rc = "FILTERED_BY_PLACE_LIVE_ORDERS"
    elif "[IDEMPOTENCY_BLOCK]" in msg:
        rc = "ORDER_ALREADY_PENDING"
    elif "[AUTH_BLOCK]" in msg:
        rc = "BROKER_SUBMIT_ERROR"
    elif "[MISSING]" in msg or "[EMPTY]" in msg or "[NO_VALID_ROWS]" in msg:
        rc = "EMPTY_AFTER_VALIDATION"
    elif "[MARKET_CLOSED_BLOCK]" in msg or "[MARKET_CLOCK_BLOCK]" in msg:
        rc = "PLACEMENT_BLOCKED"
    _PLACE_DIAG_ROWS.append(
        make_row(
            run_mode=str(_PLACE_CTX.get("mode") or "paper"),
            symbol="",
            stance="",
            phase="placement_validation",
            status="blocked" if code == 2 else "dropped",
            reason_code=rc,
            reason_detail=detail[:4000],
            source="place_live_orders",
            session=str(_PLACE_CTX.get("session") or ""),
        )
    )


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    try:
        _append_die_row_from_msg(msg, code)
        _finalize_place_diag_once(blocked=(code == 2))
    except Exception:
        pass
    raise SystemExit(code)


def safe_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def normalize_side(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("buy", "b"):
        return "buy"
    if s in ("sell", "s"):
        return "sell"
    return None


def normalize_symbol(x: Any) -> Optional[str]:
    if x is None:
        return None
    sym = str(x).strip().upper()
    if not sym:
        return None
    return SYMBOL_ALIASES.get(sym, sym)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def short_hash(s: str, n: int = 20) -> str:
    return sha256_hex(s)[:n]


def clamp_float(x: Optional[float], ndp: int = 6) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(f"{float(x):.{ndp}f}")
    except Exception:
        return x


def tick_size(price: float) -> float:
    return 0.01 if price >= 1.0 else 0.0001


def quantize_to_tick(price: float, side: str) -> float:
    t = tick_size(price)
    if t <= 0:
        return price
    k = price / t
    if side == "buy":
        q = math.floor(k + 1e-12) * t
    else:
        q = math.ceil(k - 1e-12) * t
    ndp = 2 if q >= 1.0 else 4
    return float(f"{q:.{ndp}f}")


def is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("401" in msg) or ("unauthorized" in msg) or ("forbidden" in msg) or ("403" in msg)


def make_broker(mode: str):
    from services.broker_alpaca import AlpacaBroker  # type: ignore

    return AlpacaBroker(mode=mode)


def read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class OrderRow:
    symbol: str
    side: str
    qty: int
    limit_price: Optional[float]
    # True iff this row originated from a performance-risk-overlay FORCE_EXIT
    # decision in manage_positions. Carried through to PlannedOrder so that
    # downstream batch caps (notional, future per-run counts) can exempt it.
    force_exit_override: bool = False


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: int
    order_type: str
    time_in_force: str
    limit_price: Optional[float]
    ref_price: Optional[float]
    client_order_id: str
    discipline_allowed: bool = True
    discipline_reason: str = ""
    # Execution-intelligence quote snapshot (best-effort; may all be None).
    bid: Optional[float] = None
    ask: Optional[float] = None
    quote_ts: Optional[str] = None
    intended_price: Optional[float] = None
    # See OrderRow.force_exit_override.
    force_exit_override: bool = False


def _load_block_new_buy_set(path: Path = PERFORMANCE_RISK_OVERLAY_CSV) -> "frozenset[str]":
    """
    Return the frozenset of normalized symbols carrying a ``BLOCK_NEW_BUY``
    component in the performance-risk-overlay's ``risk_flag`` column.

    The overlay CSV is fully optional. Missing / empty / malformed files
    produce an empty set with no warnings (per spec: skip silently).

    The ``risk_flag`` column may be a pipe-joined union — e.g.
    ``FORCE_EXIT|BLOCK_NEW_BUY`` — so the test is component-wise, not
    equality. Symbols are normalized through :func:`normalize_symbol` so
    they match the same key shape used downstream in the BUY branch.
    """
    empty: "frozenset[str]" = frozenset()
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return empty
    except OSError:
        return empty
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return empty
    if df is None or df.empty:
        return empty
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    sym_col: Optional[str] = None
    for cand in ("ticker", "symbol"):
        if cand in df.columns:
            sym_col = cand
            break
    if sym_col is None or "risk_flag" not in df.columns:
        return empty
    blocked: set[str] = set()
    for _, r in df.iterrows():
        raw = str(r.get("risk_flag") or "").strip().upper()
        if not raw:
            continue
        parts = {p.strip() for p in raw.split("|") if p.strip()}
        if "BLOCK_NEW_BUY" not in parts:
            continue
        sym = normalize_symbol(r.get(sym_col))
        if sym:
            blocked.add(sym)
    return frozenset(blocked)


def _row_force_exit_flag(row) -> bool:
    """
    Defensive truthy-check for the `force_exit_override` column on a CSV
    row. The column is optional; older `manage_orders.csv` files may not
    contain it. Pandas with keep_default_na=False yields strings for
    boolean-ish values, so accept both real bools and stringy variants.
    """
    try:
        if hasattr(row, "get"):
            val = row.get("force_exit_override", False)
        else:
            val = False
    except Exception:
        return False
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


# -----------------------------
# Log safety (schema + guards)
# -----------------------------


def _first_line(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            return (f.readline() or "").strip()
    except Exception:
        return ""


def _guard_event_log(path: Path) -> None:
    """
    Refuse to write into snapshot-style CSVs (e.g., overwritten by snapshot_live_orders).
    """
    if not path.exists():
        return
    header = _first_line(path).lower()
    if header.startswith("snapshot_ts"):
        die(
            f"[LOG_GUARD] Refusing to write to {path} because it looks like a SNAPSHOT file (snapshot_ts header).\n"
            f"Fix: point --log to data/results/live_orders_log.csv (event log), "
            f"or restore the correct event log file."
        )


def ensure_log_schema(path: Path) -> None:
    """
    Schema-safe: upgrades by COLUMN NAMES (not positional).
    Tolerates extra/legacy columns; writes only LOG_FIELDS order.
    """
    if not path.exists():
        return

    _guard_event_log(path)

    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", keep_default_na=False)

    cols = [str(c).strip() for c in df.columns]
    if cols == LOG_FIELDS:
        return

    ts = utc_now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.backup.{ts}{path.suffix}")
    try:
        shutil_ok = True
        path.replace(backup)
    except Exception:
        shutil_ok = False

    # If replace failed, still try to continue without risking corruption
    if not shutil_ok:
        die(f"[LOG_SCHEMA_BLOCK] Could not rotate log for schema upgrade: {path}")

    try:
        df = pd.read_csv(backup, keep_default_na=False)
    except Exception:
        df = pd.read_csv(backup, engine="python", on_bad_lines="skip", keep_default_na=False)

    df.columns = [str(c).strip() for c in df.columns]

    # helpful aliases
    rename: Dict[str, str] = {}
    if "order_type" in df.columns and "type" not in df.columns:
        rename["order_type"] = "type"
    if "id" in df.columns and "order_id" not in df.columns:
        rename["id"] = "order_id"
    df = df.rename(columns=rename)

    for c in LOG_FIELDS:
        if c not in df.columns:
            df[c] = ""

    df2 = df[LOG_FIELDS].copy()
    df2.to_csv(path, index=False, encoding="utf-8")
    print(f"[LOG_SCHEMA] Upgraded log schema safely. Backup saved as: {backup.name}")


def append_log_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _guard_event_log(path)
    if not path.exists():
        # create with correct header
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    ensure_log_schema(path)

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


# -----------------------------
# Orders parsing / planning
# -----------------------------


def load_orders_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"[MISSING] Orders file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        die(f"[EMPTY] Orders file is empty: {path}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "symbol" not in df.columns and "ticker" in df.columns:
        df["symbol"] = df["ticker"]
    if "symbol" not in df.columns and "sym" in df.columns:
        df["symbol"] = df["sym"]
    if "limit_price" not in df.columns and "close" in df.columns:
        df["limit_price"] = df["close"]
    if "qty" not in df.columns and "quantity" in df.columns:
        df["qty"] = df["quantity"]
    if "qty" not in df.columns:
        df["qty"] = 1

    if "symbol" not in df.columns or "side" not in df.columns:
        die(f"[BAD_SCHEMA] Need symbol/ticker and side. Columns={list(df.columns)}")

    df["symbol"] = df["symbol"].apply(normalize_symbol)
    df["side"] = df["side"].apply(normalize_side)
    df["qty"] = df["qty"].apply(safe_int)

    if "limit_price" in df.columns:
        df["limit_price"] = df["limit_price"].apply(safe_float)
    else:
        df["limit_price"] = None

    df = df[df["symbol"].notna() & (df["symbol"] != "")]
    df = df[df["side"].notna()]
    df = df[df["qty"].notna() & (df["qty"] > 0)]

    if df.empty:
        die("[NO_VALID_ROWS] No valid rows after sanitization.")

    return df


def to_order_rows(df: pd.DataFrame) -> List[OrderRow]:
    out: List[OrderRow] = []
    for _, r in df.iterrows():
        out.append(
            OrderRow(
                symbol=str(r["symbol"]).upper().strip(),
                side=str(r["side"]).lower().strip(),
                qty=int(r["qty"]),
                limit_price=safe_float(r.get("limit_price")),
            )
        )
    return out


def load_signal_lifecycle_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"[MISSING] signal_lifecycle not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        die(f"[READ_FAIL] {path}: {e}")
    if df is None or df.empty:
        die(f"[EMPTY] {path}")
    df.columns = [str(c).strip() for c in df.columns]
    if "ticker" not in df.columns:
        die(f"[BAD_SCHEMA] signal_lifecycle.csv must include ticker. Columns={list(df.columns)}")
    return df


def apply_effective_lifecycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer effective_stance / effective_position_state into stance / position_state when present.
    Missing effective cells fall back to original columns.
    """
    out = df.copy()
    if "effective_stance" in out.columns:
        if "stance" in out.columns:
            out["stance"] = out["effective_stance"].fillna(out["stance"])
        else:
            out["stance"] = out["effective_stance"]
    if "effective_position_state" in out.columns:
        if "position_state" in out.columns:
            out["position_state"] = out["effective_position_state"].fillna(out["position_state"])
        else:
            out["position_state"] = out["effective_position_state"]
    return out


def resolve_execution_source(cli_path: Path) -> Tuple[Path, str]:
    """
    Resolve CSV path and execution kind.
    Kind: 'trade_opps' | 'effective' | 'raw'
    Default lifecycle path: trade_opportunities.csv > signal_lifecycle_effective.csv > signal_lifecycle.csv
    """
    try:
        cli_r = cli_path.resolve()
        trade_r = TRADE_OPPORTUNITIES_PATH.resolve()
        eff_r = EFFECTIVE_LIFECYCLE_PATH.resolve()
        raw_r = DEFAULT_LIFECYCLE_CSV.resolve()
    except Exception:
        cli_r = cli_path
        trade_r = TRADE_OPPORTUNITIES_PATH
        eff_r = EFFECTIVE_LIFECYCLE_PATH
        raw_r = DEFAULT_LIFECYCLE_CSV

    if cli_r == trade_r:
        return TRADE_OPPORTUNITIES_PATH, "trade_opps"
    if cli_r == eff_r:
        return EFFECTIVE_LIFECYCLE_PATH, "effective"
    if cli_r != raw_r:
        return cli_path, "raw"

    if TRADE_OPPORTUNITIES_PATH.exists() and TRADE_OPPORTUNITIES_PATH.stat().st_size > 0:
        try:
            tdf = pd.read_csv(TRADE_OPPORTUNITIES_PATH)
            if tdf is not None and len(tdf) > 0:
                return TRADE_OPPORTUNITIES_PATH, "trade_opps"
        except Exception:
            pass

    if EFFECTIVE_LIFECYCLE_PATH.exists() and EFFECTIVE_LIFECYCLE_PATH.stat().st_size > 0:
        try:
            edf = pd.read_csv(EFFECTIVE_LIFECYCLE_PATH)
            if edf is not None and len(edf) > 0:
                return EFFECTIVE_LIFECYCLE_PATH, "effective"
        except Exception:
            pass

    return DEFAULT_LIFECYCLE_CSV, "raw"


def load_trade_opportunities_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"[MISSING] trade_opportunities not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        die(f"[READ_FAIL] {path}: {e}")
    if df is None or df.empty:
        die(f"[EMPTY] {path}")
    df.columns = [str(c).strip() for c in df.columns]
    if "ticker" not in df.columns or "opportunity_type" not in df.columns:
        die(
            f"[BAD_SCHEMA] trade_opportunities.csv needs ticker, opportunity_type. Columns={list(df.columns)}"
        )
    return df


def prepare_trade_opportunities_for_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map opportunity_type → stance for shared lifecycle_to_order_rows; merge close from effective/raw lifecycle.
    """
    out = df.copy()
    ot = out["opportunity_type"].fillna("").astype(str).str.strip().str.upper()
    type_to_stance = {"ENTRY": "BUY", "ADD": "ADD", "EXIT": "EXIT", "TRIM": "TRIM"}
    mapped = ot.map(type_to_stance).fillna("")
    if "effective_stance" in out.columns:
        es = out["effective_stance"].fillna("").astype(str).str.strip().str.upper()
        out["stance"] = es.where(es != "", mapped)
    else:
        out["stance"] = mapped

    if "effective_position_state" in out.columns:
        out["position_state"] = out["effective_position_state"]
    elif "position_state" not in out.columns:
        out["position_state"] = ""

    out["lifecycle_action"] = out["stance"]
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()

    merged_close = False
    for price_path in (EFFECTIVE_LIFECYCLE_PATH, DEFAULT_LIFECYCLE_CSV):
        if not price_path.exists() or price_path.stat().st_size == 0:
            continue
        try:
            ref = pd.read_csv(price_path)
            ref.columns = [str(c).strip() for c in ref.columns]
            if "ticker" not in ref.columns or "close" not in ref.columns:
                continue
            ref["ticker"] = ref["ticker"].astype(str).str.strip().str.upper()
            ref = ref.drop_duplicates(subset=["ticker"], keep="last")[["ticker", "close"]]
            out = out.drop(columns=["close"], errors="ignore").merge(ref, on="ticker", how="left")
            merged_close = True
            break
        except Exception:
            continue

    if not merged_close and "close" not in out.columns:
        out["close"] = float("nan")

    if "exploration_flag" not in out.columns:
        out["exploration_flag"] = False

    return out


def _parse_iso_date_only(x: Any) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    try:
        s = str(x).strip()
        if not s or s.lower() in ("nan", "none"):
            return None
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _in_buy_cooldown(ticker: str, signal_state: dict) -> bool:
    """True while calendar date is strictly before cooldown_until (matches lifecycle engine)."""
    t = signal_state.get(str(ticker).upper()) or signal_state.get(ticker)
    if not isinstance(t, dict):
        return False
    cu = t.get("cooldown_until")
    if cu in (None, "", "None"):
        return False
    cd = _parse_iso_date_only(cu)
    if cd is None:
        return False
    today = datetime.now(timezone.utc).date()
    return today < cd


def _stance_from_row(row: pd.Series) -> str:
    if "stance" in row.index:
        v = row["stance"]
        if not (isinstance(v, float) and pd.isna(v)):
            return str(v).strip().upper()
    if "lifecycle_action" in row.index:
        v = row.get("lifecycle_action")
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return str(v).strip().upper()
    return ""


def _placement_stance_from_row(row: pd.Series) -> str:
    """
    Stance for order mapping. ROTATE_EXIT (managed rotation partial/full exit) must be sell —
    detect from stance, lifecycle_action, management_action, action, or effective_stance.
    """
    for col in ("stance", "lifecycle_action", "management_action", "action", "effective_stance"):
        if col not in row.index:
            continue
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip().upper()
        if s == "ROTATE_EXIT":
            return "ROTATE_EXIT"
    return _stance_from_row(row)


def _row_position_state(row: pd.Series) -> str:
    if "position_state" not in row.index:
        return ""
    v = row.get("position_state")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip().upper()


def _row_limit_from_lifecycle(row: pd.Series) -> Optional[float]:
    for c in ("limit_price", "close", "last", "ref_price"):
        if c in row.index:
            v = safe_float(row.get(c))
            if v is not None and v > 0:
                return v
    return None


def _row_qty_preference(row: pd.Series, default: int) -> int:
    for c in ("qty", "quantity", "shares"):
        if c in row.index:
            q = safe_int(row.get(c))
            if q is not None and q > 0:
                return int(q)
    return int(max(1, default))


def _trade_opportunities_confidence_qty(row: pd.Series) -> Tuple[int, Any]:
    """
    Size BUY/ADD from confidence when processing trade_opportunities.csv only.
    >= 0.70 -> 2, >= 0.60 -> 1, else -> 1; hard cap 2; missing/invalid confidence -> 1.
    """
    max_q = 2
    if "confidence" not in row.index:
        return 1, None
    c = safe_float(row.get("confidence"))
    if c is None:
        return 1, None
    if c >= 0.70:
        q = 2
    elif c >= 0.60:
        q = 1
    else:
        q = 1
    return min(int(q), max_q), c


def _exploration_from_series(row: pd.Series) -> bool:
    if "exploration_flag" not in row.index:
        return False
    v = row.get("exploration_flag")
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "t")


def cap_exploration_trade_opportunities(
    df: pd.DataFrame, max_expl: int
) -> Tuple[pd.DataFrame, int, int, List[Dict[str, Any]]]:
    """
    Keep row order. All non-exploratory rows pass through. At most max_expl rows with
    exploration_flag True are kept (first wins); additional exploratory rows are dropped.
    Returns (df_capped, exploratory_rows_kept, exploratory_rows_skipped, exploration_drop_rows).
    """
    drop_meta: List[Dict[str, Any]] = []
    if max_expl < 0 or df.empty or "exploration_flag" not in df.columns:
        return df.copy(), 0, 0, drop_meta

    keep_idx: List[Any] = []
    expl_kept = 0
    expl_total = 0
    for idx in df.index:
        row = df.loc[idx]
        if _exploration_from_series(row):
            expl_total += 1
            if expl_kept < max_expl:
                keep_idx.append(idx)
                expl_kept += 1
            else:
                sym = (
                    normalize_symbol(row.get("ticker"))
                    or str(row.get("ticker") or "").strip().upper()
                )
                st = _stance_from_row(row)
                drop_meta.append({"symbol": sym, "stance": st, "exploration": True})
        else:
            keep_idx.append(idx)
    skipped = expl_total - expl_kept
    out = df.loc[keep_idx].copy().reset_index(drop=True)
    return out, expl_kept, skipped, drop_meta


def _fmt_exec_log(
    ticker: str,
    stance: str,
    action: str,
    qty: int,
    decision_reason: str,
    exploration: bool = False,
) -> str:
    ex = "True" if exploration else "False"
    return f"{ticker} | {stance} | {action} | {qty} | {decision_reason} | {ex}"


def lifecycle_to_order_rows(
    df: pd.DataFrame,
    *,
    pos_map: Dict[str, float],
    signal_state: dict,
    default_qty: int,
    trim_qty: int,
    trade_opportunities_mode: bool = False,
    run_mode: str = "paper",
    session: str = "",
) -> Tuple[List[OrderRow], List[str], List[Dict[str, Any]]]:
    """
    Map lifecycle STATE rows to broker OrderRows. Logs one line per actionable row (including skips).
    BUY/ADD → buy; TRIM/EXIT → sell (qty from position for exits / trim cap).
    If trade_opportunities_mode, BUY/ADD qty uses confidence tiers (cap 2); lifecycle unchanged when False.
    Third return value: structured drop diagnostics (placement_input phase).
    """
    out: List[OrderRow] = []
    lines: List[str] = []
    diag_out: List[Dict[str, Any]] = []
    block_new_buy_set = _load_block_new_buy_set()
    if block_new_buy_set:
        print(
            f"[BLOCK_NEW_BUY_OVERLAY_LOAD] count={len(block_new_buy_set)} "
            f"symbols={sorted(block_new_buy_set)}",
            flush=True,
        )

    def _diag(
        sym: str,
        stance: str,
        qty: Any,
        phase: str,
        status: str,
        reason_code: str,
        detail: str = "",
        planned_notional: Any = "",
    ) -> None:
        diag_out.append(
            make_row(
                run_mode=run_mode,
                symbol=sym,
                stance=stance,
                planned_qty=qty,
                planned_notional=planned_notional,
                phase=phase,
                status=status,
                reason_code=reason_code,
                reason_detail=detail,
                source="place_live_orders",
                session=session,
            )
        )

    work = df.copy()
    work["_stance"] = work.apply(_placement_stance_from_row, axis=1)
    work = work[work["_stance"].isin(ACTIONABLE_STANCES)]
    work = work.drop(columns=["_stance"], errors="ignore")

    for _, row in work.iterrows():
        stance = _placement_stance_from_row(row)
        ex_flag = _exploration_from_series(row)
        sym_in = row.get("ticker")
        sym = normalize_symbol(sym_in) or ""
        if not sym:
            lines.append(_fmt_exec_log(str(sym_in), stance, "skip", 0, "bad_ticker", ex_flag))
            _diag(
                "",
                stance,
                0,
                "placement_input",
                "dropped",
                "BAD_TICKER",
                "Missing or invalid ticker",
            )
            continue

        if stance == "ROTATE_EXIT":
            print(f"[MAP] ROTATE_EXIT -> SELL for {sym}")

        pos_broker = float(pos_map.get(sym, 0.0) or 0.0)
        csv_pos = _row_position_state(row)
        is_long_csv = csv_pos == "LONG"
        is_long_broker = pos_broker > 1e-9
        lim = _row_limit_from_lifecycle(row)

        if stance == "BUY":
            if sym in block_new_buy_set:
                # Performance-risk-overlay says this asset is underperforming.
                # Block NEW BUY entries only — ADD/TRIM/EXIT/ROTATE_EXIT branches
                # are unaffected (they are separate elif branches below) so the
                # system can still actively exit / trim / scale-in on existing
                # positions in the same name.
                print(
                    f"[BLOCK_NEW_BUY] symbol={sym} reason=underperforming_asset",
                    flush=True,
                )
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", 0, "block_new_buy_overlay", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "BLOCK_NEW_BUY_OVERLAY",
                    "Performance risk overlay: underperforming asset",
                )
                continue
            if _in_buy_cooldown(sym, signal_state):
                lines.append(_fmt_exec_log(sym, stance, "skip", 0, "buy_cooldown_active", ex_flag))
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "COOL_DOWN_ACTIVE",
                    "Buy cooldown active",
                )
                continue
            if is_long_csv or is_long_broker:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", 0, "already_long_no_new_buy", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "ALREADY_LONG",
                    "Already long (CSV or broker)",
                )
                continue
            if trade_opportunities_mode:
                q, conf_log = _trade_opportunities_confidence_qty(row)
                print("[SIZING] ticker=", sym, "confidence=", conf_log, "qty=", q)
            else:
                q = _row_qty_preference(row, default_qty)
            if lim is None:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", q, "missing_limit_price_close", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    q,
                    "placement_input",
                    "dropped",
                    "PRICE_UNAVAILABLE",
                    "Missing limit/close price",
                )
                continue
            out.append(OrderRow(symbol=sym, side="buy", qty=q, limit_price=lim))
            lines.append(_fmt_exec_log(sym, stance, "buy", q, "open_new_long", ex_flag))
            _diag(
                sym,
                stance,
                q,
                "placement_input",
                "kept",
                "KEPT",
                "Mapped to buy order row",
                planned_notional=round(float(q) * float(lim), 4),
            )

        elif stance == "ADD":
            if not is_long_broker:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", 0, "add_requires_existing_position", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "ADD_WITHOUT_POSITION",
                    "ADD requires existing broker position",
                )
                continue
            if trade_opportunities_mode:
                q, conf_log = _trade_opportunities_confidence_qty(row)
                print("[SIZING] ticker=", sym, "confidence=", conf_log, "qty=", q)
            else:
                q = _row_qty_preference(row, default_qty)
            if lim is None:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", q, "missing_limit_price_close", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    q,
                    "placement_input",
                    "dropped",
                    "PRICE_UNAVAILABLE",
                    "Missing limit/close price",
                )
                continue
            out.append(OrderRow(symbol=sym, side="buy", qty=q, limit_price=lim))
            lines.append(_fmt_exec_log(sym, stance, "buy", q, "increase_long", ex_flag))
            _diag(
                sym,
                stance,
                q,
                "placement_input",
                "kept",
                "KEPT",
                "Mapped to buy order row",
                planned_notional=round(float(q) * float(lim), 4),
            )

        elif stance == "TRIM":
            if pos_broker <= 1e-9:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", 0, "trim_requires_long_position", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "TRIM_WITHOUT_POSITION",
                    "TRIM requires long position",
                )
                continue
            q_user = safe_int(row.get("qty")) if "qty" in row.index else None
            if q_user is not None and q_user > 0:
                q = min(int(q_user), int(math.floor(pos_broker)))
            else:
                q = max(1, min(int(trim_qty), int(math.floor(pos_broker))))
            if lim is None:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", q, "missing_limit_price_close", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    q,
                    "placement_input",
                    "dropped",
                    "PRICE_UNAVAILABLE",
                    "Missing limit/close price",
                )
                continue
            out.append(OrderRow(symbol=sym, side="sell", qty=q, limit_price=lim))
            lines.append(_fmt_exec_log(sym, stance, "sell", q, "reduce_position", ex_flag))
            _diag(
                sym,
                stance,
                q,
                "placement_input",
                "kept",
                "KEPT",
                "Mapped to sell order row",
                planned_notional=round(float(q) * float(lim), 4),
            )

        elif stance == "EXIT":
            if pos_broker <= 1e-9:
                lines.append(
                    _fmt_exec_log(
                        sym, stance, "skip", 0, "exit_requires_long_position_flat", ex_flag
                    )
                )
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "EXIT_WITHOUT_POSITION",
                    "EXIT requires long position",
                )
                continue
            force_exit = _row_force_exit_flag(row)
            q_user = safe_int(row.get("qty")) if "qty" in row.index else None
            full_pos_qty = int(math.floor(pos_broker))
            if force_exit:
                # FORCE_EXIT: ignore the row's qty cap (MAX_QTY) and always
                # close the full broker position. Notional cap is exempted
                # downstream when this flag rides on the PlannedOrder.
                q = full_pos_qty
            elif q_user is not None and q_user > 0:
                q = min(int(q_user), full_pos_qty)
            else:
                q = full_pos_qty
            if q < 1:
                lines.append(
                    _fmt_exec_log(
                        sym, stance, "skip", 0, "exit_requires_long_position_flat", ex_flag
                    )
                )
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "ZERO_QTY_AFTER_ROUNDING",
                    "EXIT qty resolved to zero",
                )
                continue
            if lim is None:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", q, "missing_limit_price_close", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    q,
                    "placement_input",
                    "dropped",
                    "PRICE_UNAVAILABLE",
                    "Missing limit/close price",
                )
                continue
            out.append(
                OrderRow(
                    symbol=sym,
                    side="sell",
                    qty=q,
                    limit_price=lim,
                    force_exit_override=force_exit,
                )
            )
            if force_exit:
                print(
                    f"[FORCE_EXIT_EXECUTION_OVERRIDE] symbol={sym} qty={q} "
                    f"bypassed_limits=MAX_QTY,BATCH_NOTIONAL_CAP",
                    flush=True,
                )
            lines.append(_fmt_exec_log(sym, stance, "sell", q, "close_full_position", ex_flag))
            _diag(
                sym,
                stance,
                q,
                "placement_input",
                "kept",
                "KEPT",
                "Mapped to sell order row",
                planned_notional=round(float(q) * float(lim), 4),
            )

        elif stance == "ROTATE_EXIT":
            if pos_broker <= 1e-9:
                lines.append(
                    _fmt_exec_log(
                        sym, stance, "skip", 0, "rotate_exit_requires_long_position", ex_flag
                    )
                )
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "ROTATE_EXIT_WITHOUT_POSITION",
                    "ROTATE_EXIT requires long position",
                )
                continue
            q_user = safe_int(row.get("qty")) if "qty" in row.index else None
            if q_user is not None and q_user > 0:
                q = min(int(q_user), int(math.floor(pos_broker)))
            else:
                q = int(math.floor(pos_broker))
            if q < 1:
                lines.append(_fmt_exec_log(sym, stance, "skip", 0, "rotate_exit_zero_qty", ex_flag))
                _diag(
                    sym,
                    stance,
                    0,
                    "placement_input",
                    "dropped",
                    "ZERO_QTY_AFTER_ROUNDING",
                    "ROTATE_EXIT qty resolved to zero",
                )
                continue
            if lim is None:
                lines.append(
                    _fmt_exec_log(sym, stance, "skip", q, "missing_limit_price_close", ex_flag)
                )
                _diag(
                    sym,
                    stance,
                    q,
                    "placement_input",
                    "dropped",
                    "PRICE_UNAVAILABLE",
                    "Missing limit/close price",
                )
                continue
            out.append(OrderRow(symbol=sym, side="sell", qty=q, limit_price=lim))
            lines.append(_fmt_exec_log(sym, stance, "sell", q, "rotate_exit_sell", ex_flag))
            _diag(
                sym,
                stance,
                q,
                "placement_input",
                "kept",
                "KEPT",
                "Mapped ROTATE_EXIT to sell order row",
                planned_notional=round(float(q) * float(lim), 4),
            )

        else:
            lines.append(_fmt_exec_log(sym, stance, "skip", 0, "not_actionable", ex_flag))
            _diag(
                sym,
                stance,
                0,
                "placement_input",
                "dropped",
                "NON_ACTIONABLE_STANCE",
                f"Stance={stance!r}",
            )

    return out, lines, diag_out


def list_open_orders(broker) -> List[dict]:
    return broker.list_orders(status="open", nested=True, limit=500)


def build_open_index(open_orders: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    idx: Dict[Tuple[str, str], List[dict]] = {}
    for o in open_orders:
        sym = normalize_symbol(o.get("symbol")) or ""
        side = str(o.get("side") or "").lower().strip()
        if not sym or side not in ("buy", "sell"):
            continue
        idx.setdefault((sym, side), []).append(o)
    return idx


def get_ref_price(broker, symbol: str, side: str) -> Optional[float]:
    sym = normalize_symbol(symbol) or ""
    s = (side or "").strip().lower()

    bid = ask = None
    try:
        q = broker.get_quote(sym) or {}
        bid = safe_float(q.get("bid"))
        ask = safe_float(q.get("ask"))
        if bid is not None and bid <= 0:
            bid = None
        if ask is not None and ask <= 0:
            ask = None
    except Exception:
        pass

    last = None
    try:
        t = broker.get_trade(sym) or {}
        last = safe_float(t.get("last"))
        if last is not None and last <= 0:
            last = None
    except Exception:
        pass

    if s == "buy":
        return ask or last or bid
    if s == "sell":
        return bid or last or ask
    return last or bid or ask


def get_quote_snapshot(broker, symbol: str) -> Dict[str, Any]:
    """
    Best-effort snapshot of top-of-book + last trade for a single symbol.

    Always returns a dict with keys {bid, ask, last, ts}. Any field may be None.
    Never raises.
    """
    sym = normalize_symbol(symbol) or ""
    out: Dict[str, Any] = {"bid": None, "ask": None, "last": None, "ts": None}
    if not sym:
        return out
    try:
        q = broker.get_quote(sym) or {}
        b = safe_float(q.get("bid"))
        a = safe_float(q.get("ask"))
        if b is not None and b > 0:
            out["bid"] = float(b)
        if a is not None and a > 0:
            out["ask"] = float(a)
        out["ts"] = q.get("timestamp") or q.get("ts") or q.get("t") or None
    except Exception:
        pass
    try:
        t = broker.get_trade(sym) or {}
        last = safe_float(t.get("last"))
        if last is not None and last > 0:
            out["last"] = float(last)
    except Exception:
        pass
    return out


# ── Execution-intelligence sidecar log ─────────────────────────────────────
EXECUTION_INTELLIGENCE_CSV = ROOT / "data" / "results" / "execution_intelligence.csv"

EI_FIELDS: List[str] = [
    "timestamp",
    "session",
    "action",
    "symbol",
    "side",
    "qty",
    "order_id",
    "client_order_id",
    "status",
] + list(_EI_KEYS)


def append_execution_intelligence_row(row: Dict[str, Any]) -> None:
    """Best-effort additive sidecar log; never raises."""
    try:
        path = EXECUTION_INTELLIGENCE_CSV
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=EI_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in EI_FIELDS})
    except Exception:
        pass


def clamp_limit(limit_price: float, ref: float, side: str, open_buffer_pct: float) -> float:
    buffer = max(0.0, open_buffer_pct) / 100.0
    if side == "buy":
        return min(limit_price, ref * (1.0 + buffer))
    return max(limit_price, ref * (1.0 - buffer))


def deviation_pct(a: float, b: float) -> float:
    return abs(a - b) / max(1e-9, b) * 100.0


def load_idempotency() -> dict:
    st = read_json(IDEMPOTENCY_STATE_PATH)
    if not isinstance(st, dict):
        st = {}
    st.setdefault("last_batches", [])
    return st


def record_batch(
    st: dict, fingerprint: str, ok: bool, placed: int, failed: int, session: str
) -> None:
    st.setdefault("last_batches", [])
    st["last_batches"].append(
        {
            "ts_utc": utc_now_iso(),
            "session": session,
            "fingerprint": fingerprint,
            "ok": bool(ok),
            "placed": int(placed),
            "failed": int(failed),
        }
    )


def idempotency_block(st: dict, fingerprint: str, ttl_min: int, force: bool) -> None:
    if force:
        return
    cutoff = utc_now().timestamp() - ttl_min * 60
    for b in st.get("last_batches", []):
        try:
            ts = datetime.fromisoformat(str(b.get("ts_utc")).replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        if ts < cutoff:
            continue
        if b.get("fingerprint") == fingerprint and b.get("ok") is True:
            die(
                f"[IDEMPOTENCY_BLOCK] Identical batch already completed within TTL ({ttl_min}m). "
                f"fingerprint={fingerprint[:12]}... Use --force to override."
            )


def build_position_map(broker) -> Dict[str, float]:
    # NOTE: keep your existing behavior. Some brokers expose list_positions; some use get_positions.
    # We try list_positions first, fall back to get_positions.
    pos = []
    try:
        pos = broker.list_positions()
    except Exception:
        try:
            pos = broker.get_positions()
        except Exception:
            pos = []

    out: Dict[str, float] = {}
    for p in pos or []:
        sym = normalize_symbol(p.get("symbol"))
        if not sym:
            continue
        qa = safe_float(p.get("qty_available"))
        q = safe_float(p.get("qty"))
        qty = qa if qa is not None else q
        if qty is None:
            continue
        out[sym] = float(qty)
    return out


def enforce_market_open(broker, mode: str, ignore_market_closed: bool) -> None:
    if mode != "live":
        return
    if ignore_market_closed:
        return
    try:
        clk = broker.get_clock() or {}
        if not bool(clk.get("is_open")):
            die(
                "[MARKET_CLOSED_BLOCK] Market is closed (live mode). Use --ignore-market-closed to override."
            )
    except Exception as e:
        die(f"[MARKET_CLOCK_BLOCK] Could not fetch market clock in live mode: {e}")


def default_session_tag(mode: str) -> str:
    return f"{utc_now().strftime('%Y-%m-%d')}_{mode.upper()}"


def _coerce_status(resp: Any) -> str:
    st = ""
    if isinstance(resp, dict):
        st = str(resp.get("status") or "").strip().lower()
    return st or "submitted"


def _coerce_order_id(resp: Any) -> str:
    """
    Must be the broker UUID (critical for poller).
    """
    if not isinstance(resp, dict):
        return ""
    oid = str(resp.get("id") or resp.get("order_id") or "").strip()
    return oid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument(
        "--orders",
        type=str,
        default=str(DEFAULT_ORDERS_CSV),
        help="Lifecycle/trade CSV path. Default signal_lifecycle.csv: auto-picks trade_opportunities.csv then signal_lifecycle_effective.csv when non-empty.",
    )
    ap.add_argument(
        "--default-qty",
        type=int,
        default=1,
        help="Share count for BUY/ADD when lifecycle row has no qty/quantity column.",
    )
    ap.add_argument(
        "--trim-qty",
        type=int,
        default=1,
        help="Max shares to sell for TRIM (capped by broker position).",
    )
    ap.add_argument("--log", type=str, default=str(DEFAULT_LOG_CSV))
    ap.add_argument(
        "--session",
        type=str,
        default="",
        help="Placement session tag (for client_order_id uniqueness).",
    )
    ap.add_argument(
        "--log-session",
        type=str,
        default="",
        help="Session tag used in log rows (stable grouping).",
    )
    ap.set_defaults(dry_run=True)
    ap.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print lifecycle decisions + planned orders only; never call broker.submit_order (default).",
    )
    ap.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Submit orders (calls broker.submit_order). Overrides default dry-run.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--idempotency-ttl-min", type=int, default=720)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--block-on-duplicates", action="store_true")
    ap.add_argument("--cancel-duplicates", action="store_true")
    ap.add_argument("--max-batch-notional", type=float, default=5000.0)
    ap.add_argument("--require-marketdata", action="store_true")
    ap.add_argument("--max-limit-deviation-pct", type=float, default=15.0)
    ap.add_argument("--open-buffer-pct", type=float, default=0.75)
    ap.add_argument("--drop-illegal-sells", action="store_true")
    ap.add_argument("--allow-shorts", action="store_true")
    ap.add_argument("--ignore-market-closed", action="store_true")
    ap.add_argument("--extended-hours", action="store_true")
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--time-in-force", type=str, default="day")

    args = ap.parse_args()

    mode = args.mode
    orders_path, exec_source = resolve_execution_source(Path(args.orders))
    log_path = Path(args.log)

    placement_session = (args.session or "").strip() or default_session_tag(mode)
    log_session = (args.log_session or "").strip() or placement_session

    _reset_place_diag()
    _PLACE_CTX["mode"] = mode
    _PLACE_CTX["session"] = placement_session
    _PLACE_CTX["orders_path"] = str(orders_path)

    # Make session truth obvious (prevents “why is polling empty?” confusion)
    print(
        f"[PLACE] mode={mode} dry_run={args.dry_run} placement_session={placement_session} log_session={log_session} orders={orders_path.name}"
    )
    print(f"[PLACE] log={log_path}")

    # Guard log file early
    _guard_event_log(log_path)

    broker = make_broker(mode=mode)
    enforce_market_open(broker, mode=mode, ignore_market_closed=args.ignore_market_closed)

    if not args.dry_run:
        from services.master_execution_gate import (
            MasterExecutionGate,
            append_gate_log_csv,
            write_snapshot,
        )

        _mg = MasterExecutionGate(project_root=ROOT).evaluate(
            mode=mode,
            broker=broker,
            verbose=args.verbose,
            require_market_open=(False if (mode == "live" and args.ignore_market_closed) else None),
        )
        write_snapshot(_mg)
        append_gate_log_csv(_mg)
        if not _mg.ok:
            print(f"[MASTER_GATE_BLOCK] {_mg.summary}")
            for _r in _mg.reasons:
                print(f"  reason={_r}")
                _PLACE_DIAG_ROWS.append(
                    make_row(
                        run_mode=mode,
                        symbol="",
                        stance="",
                        phase="placement_validation",
                        status="blocked",
                        reason_code="PLACEMENT_BLOCKED",
                        reason_detail=str(_r),
                        source="place_live_orders",
                        session=placement_session,
                    )
                )
            die("[MASTER_GATE] Execution blocked.", 2)

    # Merge planning rows from execute_trades (same run_id) before placement pipeline
    _rid = os.environ.get("TRITON_EXEC_TRADES_RUN_ID")
    _prior = read_drop_json()
    if (
        _rid
        and isinstance(_prior, dict)
        and _prior.get("run_id") == _rid
        and isinstance(_prior.get("rows"), list)
    ):
        for _r in _prior["rows"]:
            if isinstance(_r, dict):
                _PLACE_DIAG_ROWS.append(dict(_r))

    pos_map = build_position_map(broker)

    if exec_source == "trade_opps":
        top_df = load_trade_opportunities_csv(orders_path)
        lc_df = prepare_trade_opportunities_for_lifecycle(top_df)
        lc_df, _expl_kept, _expl_skip, _expl_drop_meta = cap_exploration_trade_opportunities(
            lc_df, MAX_EXPLORATION_POSITIONS
        )
        for ed in _expl_drop_meta:
            _PLACE_DIAG_ROWS.append(
                make_row(
                    run_mode=mode,
                    symbol=str(ed.get("symbol") or ""),
                    stance=str(ed.get("stance") or ""),
                    phase="placement_input",
                    status="dropped",
                    reason_code="BATCH_TRIMMED",
                    reason_detail="Exploration cap exceeded (exploration_flag)",
                    source="place_live_orders",
                    session=placement_session,
                )
            )
        print("[EXPLORATION_CAP] allowed=", _expl_kept, "skipped=", _expl_skip)
    else:
        lc_df = load_signal_lifecycle_csv(orders_path)
        lc_df = apply_effective_lifecycle_columns(lc_df)

    sig_state = read_json(SIGNAL_STATE_PATH) or {}
    if not isinstance(sig_state, dict):
        sig_state = {}

    rows, lifecycle_lines, lifecycle_diag = lifecycle_to_order_rows(
        lc_df,
        pos_map=pos_map,
        signal_state=sig_state,
        default_qty=max(1, int(args.default_qty)),
        trim_qty=max(1, int(args.trim_qty)),
        trade_opportunities_mode=(exec_source == "trade_opps"),
        run_mode=mode,
        session=placement_session,
    )
    _PLACE_DIAG_ROWS.extend(lifecycle_diag)
    if exec_source == "trade_opps":
        print("[EXECUTION] source: trade_opportunities.csv (final actionable set)")
    elif exec_source == "effective":
        print("[EXECUTION] source: signal_lifecycle_effective.csv (broker-aligned)")
    else:
        print("[EXECUTION] source: signal_lifecycle.csv (raw lifecycle)")
    print("ticker | stance | action | qty | decision_reason | exploration")
    for ln in lifecycle_lines:
        print(ln)

    # Load open orders and index them
    open_orders = list_open_orders(broker)
    open_idx = build_open_index(open_orders)

    planned: List[PlannedOrder] = []
    batch_notional_est = 0.0
    in_flight_sat = 0  # open order already covers intent; not a drop, no new submit
    cancelled_duplicates = 0
    illegal_sells = 0
    illegal_msgs: List[str] = []

    # If cancel-duplicates, do it FIRST globally for keys we will place,
    # so we don’t rely on a stale open_idx inside the row loop.
    if args.cancel_duplicates:
        desired_keys = {(r.symbol, r.side) for r in rows}
        for (sym, side), existing in list(open_idx.items()):
            if (sym, side) not in desired_keys:
                continue
            for o in existing:
                oid = o.get("id")
                if not oid:
                    continue
                if args.dry_run:
                    continue
                try:
                    broker.cancel_order(oid)
                    cancelled_duplicates += 1

                    # Log the cancel for audit
                    append_log_row(
                        log_path,
                        {
                            "timestamp": utc_now_iso(),
                            "session": log_session,
                            "action": "cancel",
                            "symbol": sym,
                            "side": side,
                            "qty": safe_int(o.get("qty")) or "",
                            "type": (o.get("type") or o.get("order_type") or "").strip().lower(),
                            "limit_price": o.get("limit_price", "") or "",
                            "order_id": str(oid).strip(),
                            "status": "canceled",
                            "filled_qty": safe_int(o.get("filled_qty")) or 0,
                            "filled_avg_price": o.get("filled_avg_price", "") or "",
                            "client_order_id": o.get("client_order_id", "") or "",
                            "tp_limit": "",
                            "sl_stop": "",
                        },
                    )
                except Exception as e:
                    if args.verbose:
                        print(f"[WARN] cancel duplicate failed {sym} {side} id={oid}: {e}")

        # Rebuild open_idx after cancels (important)
        if not args.dry_run and cancelled_duplicates > 0:
            time.sleep(0.25)
        open_orders = list_open_orders(broker)
        open_idx = build_open_index(open_orders)

    for r in rows:
        key = (r.symbol, r.side)
        existing = open_idx.get(key, [])

        if existing:
            if args.block_on_duplicates:
                _PLACE_DIAG_ROWS.append(
                    make_row(
                        run_mode=mode,
                        symbol=r.symbol,
                        stance="",
                        planned_qty=r.qty,
                        phase="placement_validation",
                        status="blocked",
                        reason_code="DUPLICATE_OPEN_ORDER",
                        reason_detail=f"Existing open {r.side} order(s); block_on_duplicates",
                        source="place_live_orders",
                        session=placement_session,
                    )
                )
                die(f"[DUPLICATE_OPEN_BLOCK] Existing open {r.side} order(s) for {r.symbol}.")
            in_flight_sat += 1
            _PLACE_DIAG_ROWS.append(
                make_row(
                    run_mode=mode,
                    symbol=r.symbol,
                    stance="",
                    planned_qty=r.qty,
                    phase="placement_validation",
                    status="kept",
                    reason_code="IN_FLIGHT_ORDER",
                    reason_detail="Existing open order already satisfies execution intent",
                    source="place_live_orders",
                    session=placement_session,
                )
            )
            continue

        if r.side == "sell" and not args.allow_shorts:
            avail = float(pos_map.get(r.symbol, 0.0) or 0.0)
            if avail <= 0 or float(r.qty) > float(avail) + 1e-9:
                illegal_sells += 1
                msg = f"{r.symbol} SELL qty={r.qty} available={avail}"
                illegal_msgs.append(msg)
                if args.drop_illegal_sells:
                    _PLACE_DIAG_ROWS.append(
                        make_row(
                            run_mode=mode,
                            symbol=r.symbol,
                            stance="",
                            planned_qty=r.qty,
                            phase="placement_validation",
                            status="dropped",
                            reason_code="ILLEGAL_SELL",
                            reason_detail=msg,
                            source="place_live_orders",
                            session=placement_session,
                        )
                    )
                    continue
                if not args.dry_run:
                    _PLACE_DIAG_ROWS.append(
                        make_row(
                            run_mode=mode,
                            symbol=r.symbol,
                            stance="",
                            planned_qty=r.qty,
                            phase="placement_validation",
                            status="blocked",
                            reason_code="ILLEGAL_SELL",
                            reason_detail=msg,
                            source="place_live_orders",
                            session=placement_session,
                        )
                    )
                    die(
                        f"[ILLEGAL_SELL_BLOCK] {msg}. "
                        f"Use --drop-illegal-sells to drop these rows, or --allow-shorts to override."
                    )

        order_type = "market" if args.market else "limit"
        limit_price = None if order_type == "market" else r.limit_price

        ref = get_ref_price(broker, r.symbol, r.side)
        # Capture raw quote snapshot for execution-intelligence annotations.
        # This is best-effort; failures fall back to neutral defaults later.
        try:
            _quote_snap = get_quote_snapshot(broker, r.symbol)
        except Exception:
            _quote_snap = {"bid": None, "ask": None, "last": None, "ts": None}

        if order_type == "limit":
            if limit_price is None or limit_price <= 0:
                _PLACE_DIAG_ROWS.append(
                    make_row(
                        run_mode=mode,
                        symbol=r.symbol,
                        stance="",
                        planned_qty=r.qty,
                        phase="placement_validation",
                        status="dropped",
                        reason_code="FILTERED_BY_PLACE_LIVE_ORDERS",
                        reason_detail=f"Invalid limit_price={limit_price}",
                        source="place_live_orders",
                        session=placement_session,
                    )
                )
                die(f"[BAD_LIMIT] {r.symbol} has invalid limit_price: {limit_price}")

            if ref is None:
                if args.require_marketdata:
                    _PLACE_DIAG_ROWS.append(
                        make_row(
                            run_mode=mode,
                            symbol=r.symbol,
                            stance="",
                            planned_qty=r.qty,
                            phase="placement_validation",
                            status="dropped",
                            reason_code="PRICE_UNAVAILABLE",
                            reason_detail="Could not fetch quote/trade ref (require_marketdata)",
                            source="place_live_orders",
                            session=placement_session,
                        )
                    )
                    die(f"[MARKETDATA_BLOCK] Could not fetch quote/trade ref price for {r.symbol}.")
            else:
                dev = deviation_pct(limit_price, ref)
                if dev > args.max_limit_deviation_pct:
                    _PLACE_DIAG_ROWS.append(
                        make_row(
                            run_mode=mode,
                            symbol=r.symbol,
                            stance="",
                            planned_qty=r.qty,
                            phase="placement_validation",
                            status="dropped",
                            reason_code="FILTERED_BY_PLACE_LIVE_ORDERS",
                            reason_detail=(
                                f"Limit {limit_price} deviates {dev:.1f}% from ref {ref} "
                                f"(max {args.max_limit_deviation_pct}%)"
                            ),
                            source="place_live_orders",
                            session=placement_session,
                        )
                    )
                    die(
                        f"[LIMIT_SANITY_BLOCK] {r.symbol} limit {limit_price} deviates {dev:.1f}% from ref {ref} "
                        f"(max {args.max_limit_deviation_pct}%)."
                    )
                limit_price = clamp_limit(limit_price, ref, r.side, args.open_buffer_pct)

            limit_price = quantize_to_tick(float(limit_price), r.side)
            batch_notional_est += float(r.qty) * float(limit_price)

        else:
            # market order sizing cap uses ref estimate
            if ref is None:
                if args.require_marketdata:
                    _PLACE_DIAG_ROWS.append(
                        make_row(
                            run_mode=mode,
                            symbol=r.symbol,
                            stance="",
                            planned_qty=r.qty,
                            phase="placement_validation",
                            status="dropped",
                            reason_code="PRICE_UNAVAILABLE",
                            reason_detail="Could not fetch ref for MARKET sizing (require_marketdata)",
                            source="place_live_orders",
                            session=placement_session,
                        )
                    )
                    die(
                        f"[MARKETDATA_BLOCK] Could not fetch ref price for MARKET sizing of {r.symbol}."
                    )
            else:
                batch_notional_est += float(r.qty) * float(ref)

        coid_raw = (
            f"{placement_session}|{mode}|{r.symbol}|{r.side}|{r.qty}|"
            f"{order_type}|{args.time_in_force}|{clamp_float(limit_price, 6)}"
        )
        client_order_id = f"triton-{short_hash(coid_raw, 20)}"

        planned.append(
            PlannedOrder(
                symbol=r.symbol,
                side=r.side,
                qty=r.qty,
                order_type=order_type,
                time_in_force=args.time_in_force,
                limit_price=limit_price,
                ref_price=ref,
                client_order_id=client_order_id,
                bid=_quote_snap.get("bid"),
                ask=_quote_snap.get("ask"),
                quote_ts=_quote_snap.get("ts"),
                intended_price=r.limit_price,
                force_exit_override=bool(getattr(r, "force_exit_override", False)),
            )
        )

    if not args.dry_run and planned:
        try:
            from services.order_discipline import apply_discipline_to_planned_generic

            _bk = set(open_idx.keys())
            planned, _disc_pl = apply_discipline_to_planned_generic(
                planned,
                session=placement_session,
                source_module="place_live_orders",
                mode=mode,
                context=None,
                open_side_keys=_bk,
            )
            if int(_disc_pl.get("orders_blocked") or 0) and args.verbose:
                print(
                    f"[ORDER_DISCIPLINE] blocked={_disc_pl.get('orders_blocked')} "
                    f"reasons={_disc_pl.get('block_reasons')}",
                    flush=True,
                )
        except Exception:
            pass

    # If dry-run and we found illegal sells and user didn’t opt-in to drop/short, show them all + exit cleanly.
    if (
        args.dry_run
        and illegal_sells > 0
        and (not args.drop_illegal_sells)
        and (not args.allow_shorts)
    ):
        print(
            f"[ILLEGAL_SELL_BLOCK] {illegal_sells} illegal SELL row(s) detected. "
            f"Example(s): {', '.join(illegal_msgs[:8])}"
        )
        print(
            "Fix options: (1) add --drop-illegal-sells, or (2) fix lifecycle/positions so SELL qty <= broker position, "
            "or (3) --allow-shorts (not recommended)."
        )
        _finalize_place_diag_once(blocked=False)
        return

    if not planned:
        if in_flight_sat > 0:
            print(
                f"[IN_FLIGHT_SATISFIED] {in_flight_sat} row(s) already have matching open orders; "
                f"no new submits (execution intent satisfied in-flight). placement_session={placement_session}"
            )
            _finalize_place_diag_once(blocked=False)
            return
        try:
            if orders_path.resolve() == DEFAULT_MANAGE_ORDERS_CSV.resolve():
                print(
                    "[PLACE_MANAGE_ORDERS] BATCH_EMPTY source=manage_orders.csv "
                    f"lifecycle_rows={len(rows)} planned=0 in_flight_sat={in_flight_sat} "
                    f"illegal_sells={illegal_sells} -- see manage_positions plan CSV and [MANAGE_SUMMARY]"
                )
        except Exception:
            pass
        die("[BATCH_EMPTY] Nothing to place (all skipped or invalid).")

    # Batch notional cap now applies to BOTH market + limit.
    # FORCE_EXIT exits originate from the performance-risk-overlay path in
    # manage_positions and must never be blocked by per-batch heuristics —
    # their notional is excluded from the cap comparison so a large forced
    # close cannot starve out other (already-discipline-cleared) orders.
    fx_notional_excluded = 0.0
    fx_symbols: List[str] = []
    for p in planned:
        if not bool(getattr(p, "force_exit_override", False)):
            continue
        ref_for_calc = p.limit_price if p.limit_price is not None else p.ref_price
        if ref_for_calc is None:
            continue
        fx_notional_excluded += float(p.qty) * float(ref_for_calc)
        fx_symbols.append(p.symbol)
    cap_compare_notional = batch_notional_est - fx_notional_excluded
    if fx_notional_excluded > 0:
        print(
            f"[FORCE_EXIT_NOTIONAL_EXEMPT] count={len(fx_symbols)} "
            f"symbols={sorted(set(fx_symbols))} "
            f"exempted_notional={fx_notional_excluded:.2f} "
            f"raw_batch_notional={batch_notional_est:.2f} "
            f"compare_notional={cap_compare_notional:.2f} "
            f"max_batch_notional={float(args.max_batch_notional):.2f}",
            flush=True,
        )
    if cap_compare_notional > float(args.max_batch_notional):
        _PLACE_DIAG_ROWS.append(
            make_row(
                run_mode=mode,
                symbol="",
                stance="",
                phase="placement_validation",
                status="blocked",
                reason_code="MAX_NOTIONAL",
                reason_detail=(
                    f"est_batch_notional={cap_compare_notional:.2f} (excl FORCE_EXIT "
                    f"{fx_notional_excluded:.2f}) exceeds max "
                    f"{float(args.max_batch_notional):.2f}"
                ),
                source="place_live_orders",
                session=placement_session,
            )
        )
        die(
            f"[BATCH_NOTIONAL_BLOCK] est_batch_notional={cap_compare_notional:.2f} "
            f"(excl FORCE_EXIT {fx_notional_excluded:.2f}) "
            f"exceeds max {float(args.max_batch_notional):.2f}. "
            f"(Hint: lower qty, raise cap carefully, or use --require-marketdata to make sizing deterministic.)"
        )

    fingerprint_payload = json.dumps([p.__dict__ for p in planned], sort_keys=True)
    fingerprint = sha256_hex(fingerprint_payload)

    st = load_idempotency()
    if not args.dry_run:
        idempotency_block(st, fingerprint, args.idempotency_ttl_min, args.force)

    if args.dry_run:
        print(
            f"[DRY_RUN] no broker.submit_order -- planned={len(planned)} fingerprint={fingerprint[:12]}... "
            f"est_batch_notional={batch_notional_est:.2f} in_flight_sat={in_flight_sat} "
            f"cancelled_duplicates={cancelled_duplicates} illegal_sells={illegal_sells} "
            f"placement_session={placement_session} log_session={log_session}"
        )
        for p in planned:
            ref_txt = f" ref={p.ref_price}" if p.ref_price else ""
            print(
                f"  - {p.symbol} {p.side} qty={p.qty} {p.order_type} limit={p.limit_price}{ref_txt} coid={p.client_order_id}"
            )
        _finalize_place_diag_once(blocked=False)
        return

    placed = 0
    failed = 0
    _ei_cfg = ExecutionIntelligenceConfig()

    def _emit_execution_intelligence(
        p: PlannedOrder, *, action: str, status: str, order_id: str, fill_price: Any = None
    ) -> None:
        """Write one sidecar row capturing quote/spread/style/slippage context."""
        try:
            ann = _ei_annotate_order(
                action=action,
                side=p.side,
                bid=p.bid,
                ask=p.ask,
                quote_ts=p.quote_ts,
                close=p.ref_price,
                order_qty=p.qty,
                order_notional=(float(p.qty) * float(p.limit_price)) if p.limit_price else None,
                intended_price=p.intended_price,
                submitted_limit_price=p.limit_price,
                fill_price=fill_price,
                cfg=_ei_cfg,
            )
            row: Dict[str, Any] = {
                "timestamp": utc_now_iso(),
                "session": log_session,
                "action": action,
                "symbol": p.symbol,
                "side": p.side,
                "qty": p.qty,
                "order_id": order_id or "",
                "client_order_id": p.client_order_id,
                "status": status,
            }
            for k in _EI_KEYS:
                row[k] = ann.get(k)
            append_execution_intelligence_row(row)
        except Exception:
            pass

    for p in planned:
        try:
            resp = broker.submit_order(
                symbol=p.symbol,
                qty=p.qty,
                side=p.side,
                order_type=p.order_type,
                time_in_force=p.time_in_force,
                limit_price=p.limit_price,
                client_order_id=p.client_order_id,
                extended_hours=bool(args.extended_hours),
            )

            # CRITICAL: pull broker UUID into order_id for poller compatibility
            oid = _coerce_order_id(resp)
            status = _coerce_status(resp)

            placed += 1

            append_log_row(
                log_path,
                {
                    "timestamp": utc_now_iso(),
                    "session": log_session,  # ✅ stable grouping
                    "action": "submit",
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "type": p.order_type,
                    "limit_price": p.limit_price if p.limit_price is not None else "",
                    "order_id": oid,  # ✅ MUST be Alpaca UUID
                    "status": status,
                    "filled_qty": 0,
                    "filled_avg_price": "",
                    "client_order_id": p.client_order_id,  # ✅ your deterministic triton-* id
                    "tp_limit": "",
                    "sl_stop": "",
                },
            )
            _emit_execution_intelligence(p, action="submit", status=status, order_id=oid)
            _PLACE_DIAG_ROWS.append(
                make_row(
                    run_mode=mode,
                    symbol=p.symbol,
                    stance="",
                    planned_qty=p.qty,
                    planned_notional="",
                    phase="placement_submit",
                    status="submitted",
                    reason_code="SUBMITTED",
                    reason_detail=f"status={status} order_id={oid}",
                    source="place_live_orders",
                    session=placement_session,
                    client_order_id=p.client_order_id,
                )
            )

            if args.verbose:
                if p.order_type == "limit":
                    print(
                        f"[OK] {p.symbol} {p.side} qty={p.qty} type=limit limit={p.limit_price} id={oid}"
                    )
                else:
                    print(f"[OK] {p.symbol} {p.side} qty={p.qty} type=market id={oid}")

            # If broker did not return an id, that is a hard warning (poller cannot track)
            if not oid:
                print(
                    f"[WARN] No broker order id returned for {p.symbol} {p.side}. Poller cannot track this order."
                )

        except Exception as e:
            failed += 1
            _PLACE_DIAG_ROWS.append(
                make_row(
                    run_mode=mode,
                    symbol=p.symbol,
                    stance="",
                    planned_qty=p.qty,
                    phase="placement_submit",
                    status="dropped",
                    reason_code="BROKER_SUBMIT_ERROR",
                    reason_detail=str(e)[:2000],
                    source="place_live_orders",
                    session=placement_session,
                    client_order_id=p.client_order_id,
                )
            )
            if is_auth_error(e):
                die(f"[AUTH_BLOCK] 401/403 during submit. Aborting. Details: {e}")

            append_log_row(
                log_path,
                {
                    "timestamp": utc_now_iso(),
                    "session": log_session,
                    "action": "submit",
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "type": p.order_type,
                    "limit_price": p.limit_price if p.limit_price is not None else "",
                    "order_id": "",
                    "status": "error",
                    "filled_qty": 0,
                    "filled_avg_price": "",
                    "client_order_id": p.client_order_id,
                    "tp_limit": "",
                    "sl_stop": "",
                },
            )
            _emit_execution_intelligence(p, action="submit", status="error", order_id="")
            print(f"[FAIL] {p.symbol} {p.side} qty={p.qty} err={e}")

    ok = placed > 0 and failed == 0
    record_batch(st, fingerprint, ok=ok, placed=placed, failed=failed, session=placement_session)
    write_json(IDEMPOTENCY_STATE_PATH, st)

    _dropped_reasons = Counter(
        str(r.get("reason_code") or "")
        for r in _PLACE_DIAG_ROWS
        if str(r.get("status", "")).lower() in ("dropped", "blocked")
    )
    _in_flight_n = sum(
        1
        for r in _PLACE_DIAG_ROWS
        if str(r.get("reason_code", "")).strip() == "IN_FLIGHT_ORDER"
        and str(r.get("status", "")).lower() == "kept"
    )
    print(
        f"[DONE] placement_session={placement_session} log_session={log_session} "
        f"planned={len(planned)} placed={placed} failed={failed} ok={ok} "
        f"fingerprint={fingerprint[:12]}... est_batch_notional={batch_notional_est:.2f} log={log_path}"
    )
    print(
        f"[DROP_DIAG] planned={len(planned)} placed={placed} failed={failed} in_flight={_in_flight_n} "
        f"dropped_by_reason={dict(_dropped_reasons)} diagnostics={DROP_JSON}"
    )
    _finalize_place_diag_once(blocked=False)


if __name__ == "__main__":
    main()
