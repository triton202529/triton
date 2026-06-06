"""Duplicate order protection — broker open orders, session log, local session registry (Phase 148E)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
DEFAULT_LOG_PATH = RESULTS / "live_orders_log.csv"
SESSION_INTENTS_CSV = LIVE / "placement_session_intents.csv"

INTENT_FIELDS = [
    "timestamp",
    "session",
    "symbol",
    "side",
    "qty",
    "mode",
    "source",
    "status",
]

_TERMINAL_LOG_STATUSES = frozenset(
    {
        "filled",
        "canceled",
        "cancelled",
        "done_for_day",
        "expired",
        "replaced",
        "failed",
        "rejected",
    }
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_symbol(symbol: Any) -> str:
    from services.order_discipline import normalize_symbol

    return normalize_symbol(symbol)


def _norm_side(side: Any) -> str:
    from services.order_discipline import normalize_side

    return normalize_side(side) or ""


def _key(symbol: Any, side: Any) -> Optional[Tuple[str, str]]:
    sym = _norm_symbol(symbol)
    sd = _norm_side(side)
    if not sym or not sd:
        return None
    return (sym, sd)


def load_session_intent_keys(session: str) -> Set[Tuple[str, str]]:
    """Symbol/side pairs already recorded for this placement session."""
    out: Set[Tuple[str, str]] = set()
    if not session or not SESSION_INTENTS_CSV.is_file():
        return out
    try:
        with SESSION_INTENTS_CSV.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("session") or "").strip() != session:
                    continue
                k = _key(row.get("symbol"), row.get("side"))
                if k:
                    out.add(k)
    except Exception:
        pass
    return out


def load_session_log_submit_keys(
    session: str,
    *,
    log_path: Path = DEFAULT_LOG_PATH,
) -> Set[Tuple[str, str]]:
    """Symbol/side pairs with a prior submit in live_orders_log for this session."""
    out: Set[Tuple[str, str]] = set()
    if not session:
        return out
    try:
        from services.order_discipline import load_log_events

        events = load_log_events(log_path, lookback_minutes=1e6)
    except Exception:
        return out
    for ev in events:
        if str(ev.get("session") or "").strip() != session:
            continue
        if str(ev.get("action") or "").strip().lower() != "submit":
            continue
        st = str(ev.get("status") or "").strip().lower()
        if st in _TERMINAL_LOG_STATUSES and st not in ("", "new", "accepted", "pending_new"):
            # Prior submit in session still counts as duplicate intent for re-run protection.
            pass
        k = _key(ev.get("symbol"), ev.get("side"))
        if k:
            out.add(k)
    return out


def check_duplicate_order(
    *,
    symbol: str,
    side: str,
    session: str,
    open_side_keys: Optional[Set[Tuple[str, str]]] = None,
    batch_submitted: Optional[Set[Tuple[str, str]]] = None,
    log_path: Path = DEFAULT_LOG_PATH,
    include_session_registry: bool = True,
) -> Tuple[bool, str]:
    """
    Returns (is_duplicate, reason_code).

    Checks (in order):
      1. Open broker orders (symbol, side)
      2. Current batch already submitted in this run
      3. Local session intent registry (prior run same session)
      4. live_orders_log submit rows for this session
    """
    k = _key(symbol, side)
    if k is None:
        return False, ""

    if open_side_keys and k in open_side_keys:
        return True, "OPEN_BROKER_ORDER"

    if batch_submitted and k in batch_submitted:
        return True, "SAME_BATCH_SUBMIT"

    if include_session_registry:
        registry = load_session_intent_keys(session)
        if k in registry:
            return True, "SESSION_INTENT_RECORD"

    log_keys = load_session_log_submit_keys(session, log_path=log_path)
    if k in log_keys:
        return True, "SESSION_LOG_SUBMIT"

    return False, ""


def record_session_intent(
    *,
    session: str,
    symbol: str,
    side: str,
    qty: Any,
    mode: str,
    source: str,
    status: str = "recorded",
) -> None:
    """Append one session intent row (paper + live registry; used for duplicate detection)."""
    LIVE.mkdir(parents=True, exist_ok=True)
    write_header = not SESSION_INTENTS_CSV.is_file() or SESSION_INTENTS_CSV.stat().st_size == 0
    row = {
        "timestamp": _utc_iso(),
        "session": str(session or "").strip(),
        "symbol": _norm_symbol(symbol),
        "side": _norm_side(side),
        "qty": str(qty or ""),
        "mode": str(mode or "paper").lower(),
        "source": str(source or ""),
        "status": str(status or "recorded"),
    }
    with SESSION_INTENTS_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INTENT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def record_planned_intents(
    orders: List[Any],
    *,
    session: str,
    mode: str,
    source: str,
) -> int:
    n = 0
    for order in orders:
        sym = getattr(order, "symbol", "")
        sd = getattr(order, "side", "")
        qty = getattr(order, "qty", "")
        if not sym or not sd:
            continue
        record_session_intent(
            session=session,
            symbol=sym,
            side=sd,
            qty=qty,
            mode=mode,
            source=source,
            status="planned",
        )
        n += 1
    return n


def log_duplicate_block(
    *,
    symbol: str,
    side: str,
    reason: str,
    mode: str,
    session: str,
    qty: Any = "",
    price: Any = "",
    order_type: str = "",
) -> None:
    action = _norm_side(side) or str(side or "").lower()
    print(
        f"[DUPLICATE_ORDER_BLOCK] symbol={_norm_symbol(symbol)} action={action} reason={reason}",
        flush=True,
    )
    if str(mode).lower() != "paper":
        return
    try:
        from services.paper_execution_audit import append_duplicate_block_audit

        append_duplicate_block_audit(
            mode=mode,
            session=session,
            symbol=symbol,
            side=side,
            reason=reason,
            qty=qty,
            price=price,
            order_type=order_type,
        )
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Duplicate order guard utilities")
    ap.add_argument("--session", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["buy", "sell"])
    ap.add_argument("--mode", default="paper", choices=["paper", "live"])
    args = ap.parse_args(argv)

    dup, reason = check_duplicate_order(
        symbol=args.symbol,
        side=args.side,
        session=args.session,
        open_side_keys=set(),
    )
    print(f"duplicate={dup} reason={reason or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
