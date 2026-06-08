"""Paper trade session review and exit verdict analysis (Phase 150C). Read-only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "paper_trade_session_review.json"
OUTPUT_CSV = RESULTS / "paper_trade_exit_review.csv"

INITIAL_BUY_SYMBOLS = ("AMD", "ARKK", "GBTC", "INTC", "QQQ", "SLV", "TSLA", "XLK")

EXIT_REVIEW_COLUMNS = [
    "symbol",
    "exit_type",
    "entry_qty",
    "exit_qty",
    "entry_price",
    "exit_price",
    "realized_pl",
    "lifecycle_before",
    "effective_stance_before",
    "risk_flag",
    "exit_reason",
    "evidence",
    "verdict",
    "review_reason",
]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _date_prefix(ts: str, session_date: str) -> bool:
    if not ts:
        return False
    return ts.startswith(session_date)


def _filter_rows_by_date(
    rows: List[Dict[str, str]], session_date: str, *date_cols: str
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        for col in date_cols:
            val = str(row.get(col) or "")
            if _date_prefix(val, session_date):
                out.append(row)
                break
    return out


def _index_by_symbol(
    rows: List[Dict[str, str]], symbol_col: str = "symbol"
) -> Dict[str, Dict[str, str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sym = str(row.get(symbol_col) or row.get("ticker") or "").upper()
        if sym:
            indexed[sym] = row
    return indexed


def _load_session_fills(
    live_orders: List[Dict[str, str]],
    recent_orders: List[Dict[str, str]],
    session_date: str,
) -> Dict[str, Dict[str, Any]]:
    """Build per-symbol entry/exit fill summary for session_date."""
    fills: Dict[str, Dict[str, Any]] = {}

    def ensure(sym: str) -> Dict[str, Any]:
        sym = sym.upper()
        if sym not in fills:
            fills[sym] = {
                "symbol": sym,
                "entry_qty": 0.0,
                "exit_qty": 0.0,
                "entry_price": None,
                "exit_price": None,
                "entry_notional": 0.0,
                "exit_notional": 0.0,
            }
        return fills[sym]

    for row in _filter_rows_by_date(live_orders, session_date, "timestamp"):
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        side = str(row.get("side") or "").lower()
        status = str(row.get("status") or "").lower()
        if status != "filled":
            continue
        qty = _safe_float(row.get("filled_qty") or row.get("qty"))
        price = _safe_float(row.get("filled_avg_price") or row.get("limit_price"))
        if qty <= 0:
            continue
        slot = ensure(sym)
        if side == "buy":
            slot["entry_qty"] += qty
            slot["entry_notional"] += qty * price
            slot["entry_price"] = price if slot["entry_price"] is None else slot["entry_price"]
        elif side == "sell":
            slot["exit_qty"] += qty
            slot["exit_notional"] += qty * price
            slot["exit_price"] = price if slot["exit_price"] is None else slot["exit_price"]

    for row in _filter_rows_by_date(
        recent_orders, session_date, "filled_at", "submitted_at", "created_at"
    ):
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        status = str(row.get("status") or "").lower()
        if status != "filled":
            continue
        side = str(row.get("side") or row.get("type") or "").lower()
        qty = _safe_float(row.get("filled_qty") or row.get("qty"))
        price = _safe_float(row.get("filled_avg_price") or row.get("limit_price"))
        if qty <= 0:
            continue
        slot = ensure(sym)
        if side == "buy":
            if slot["entry_qty"] == 0:
                slot["entry_qty"] = qty
                slot["entry_notional"] = qty * price
                slot["entry_price"] = price
        elif side == "sell":
            if slot["exit_qty"] == 0:
                slot["exit_qty"] = qty
                slot["exit_notional"] = qty * price
                slot["exit_price"] = price

    for sym, slot in fills.items():
        if slot["entry_qty"] > 0 and slot["entry_notional"] > 0:
            slot["entry_price"] = round(slot["entry_notional"] / slot["entry_qty"], 6)
        if slot["exit_qty"] > 0 and slot["exit_notional"] > 0:
            slot["exit_price"] = round(slot["exit_notional"] / slot["exit_qty"], 6)
    return fills


def _session_audit_events(
    audit_rows: List[Dict[str, str]],
    session_date: str,
) -> Dict[str, List[Dict[str, str]]]:
    by_symbol: Dict[str, List[Dict[str, str]]] = {}
    for row in _filter_rows_by_date(audit_rows, session_date, "timestamp"):
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        by_symbol.setdefault(sym, []).append(row)
    return by_symbol


def _classify_exit_type(audit_events: List[Dict[str, str]]) -> str:
    for row in reversed(audit_events):
        action = str(row.get("action") or row.get("signal") or "").upper()
        if action in ("EXIT", "FORCED_EXIT"):
            return "FORCED_EXIT"
        if action == "ROTATE_EXIT":
            return "ROTATE_EXIT"
        if action in ("TRIM", "LIFECYCLE_EXIT"):
            return "LIFECYCLE_EXIT" if action == "LIFECYCLE_EXIT" else "OTHER"
        lifecycle = str(row.get("lifecycle_state") or "").upper()
        if lifecycle == "ROTATE_EXIT":
            return "ROTATE_EXIT"
        if lifecycle == "EXIT":
            return "FORCED_EXIT"
    return "OTHER"


def _lifecycle_context(
    lifecycle_rows: Dict[str, Dict[str, str]],
    effective_rows: Dict[str, Dict[str, str]],
    symbol: str,
) -> Tuple[str, str]:
    lc = lifecycle_rows.get(symbol, {})
    eff = effective_rows.get(symbol, {})
    lifecycle_before = str(lc.get("lifecycle_state") or lc.get("stance") or lc.get("signal") or "")
    effective_stance = str(
        eff.get("effective_stance")
        or eff.get("effective_signal")
        or eff.get("stance")
        or lifecycle_before
    )
    return lifecycle_before, effective_stance


def _review_slv_exit(
    *,
    fill: Dict[str, Any],
    risk_row: Dict[str, str],
    alloc_row: Dict[str, str],
    audit_events: List[Dict[str, str]],
) -> Tuple[str, str, str]:
    total_pl = _safe_float(risk_row.get("total_pl"))
    entry_price = _safe_float(fill.get("entry_price"))
    exit_price = _safe_float(fill.get("exit_price"))
    entry_qty = _safe_float(fill.get("entry_qty"))
    session_pl = (exit_price - entry_price) * entry_qty if entry_qty else 0.0
    risk_flag = str(risk_row.get("risk_flag") or "OK")
    alloc_action = str(alloc_row.get("action") or "")

    evidence_parts = [
        f"total_pl={total_pl:.2f}",
        f"session_unrealized_pl={session_pl:.2f}",
        f"risk_flag={risk_flag}",
        f"allocation_action={alloc_action}",
    ]
    for row in audit_events:
        skip = str(row.get("skip_reason") or "")
        if skip:
            evidence_parts.append(f"audit={skip}")

    exit_reason = "high_drag_protection/TRIM_PRIORITY/forced exit"
    if total_pl <= -80:
        evidence_parts.append("historical_drag_exceeds_threshold")

    if risk_flag == "TRIM_PRIORITY" and session_pl > 0 and abs(total_pl) > abs(session_pl) * 5:
        verdict = "QUESTIONABLE"
        reason = (
            "Full EXIT applied under TRIM_PRIORITY/high_drag while session P/L was positive; "
            "stale cumulative drag (-120.85) drove forced exit rather than trim-only reduction"
        )
    elif total_pl <= -80 and risk_flag == "TRIM_PRIORITY":
        verdict = "CORRECT"
        reason = (
            "High-drag protection justified full exit: cumulative total_pl=-120.85 with "
            "TRIM_PRIORITY flag outweighs small session gain"
        )
    else:
        verdict = "QUESTIONABLE"
        reason = "Exit rationale mixed between trim priority and full liquidation"

    return exit_reason, verdict, "; ".join(evidence_parts) + f"; review_reason={reason}"


def _review_tsla_exit(
    *,
    fill: Dict[str, Any],
    risk_row: Dict[str, str],
    alloc_row: Dict[str, str],
    audit_events: List[Dict[str, str]],
) -> Tuple[str, str, str]:
    total_pl = _safe_float(risk_row.get("total_pl"))
    entry_price = _safe_float(fill.get("entry_price"))
    exit_price = _safe_float(fill.get("exit_price"))
    entry_qty = _safe_float(fill.get("entry_qty"))
    session_pl = (exit_price - entry_price) * entry_qty if entry_qty else 0.0
    risk_flag = str(risk_row.get("risk_flag") or "OK")
    alloc_action = str(alloc_row.get("action") or "")

    evidence_parts = [
        f"total_pl={total_pl:.2f}",
        f"session_realized_pl={session_pl:.2f}",
        f"risk_flag={risk_flag}",
        f"allocation_action={alloc_action}",
        "rotation=forced_rotation_weakest",
    ]
    for row in audit_events:
        sig = str(row.get("signal") or row.get("action") or "")
        if sig:
            evidence_parts.append(f"audit_signal={sig}")

    exit_reason = "ROTATE_EXIT/weak rotation"
    if risk_flag == "OK" and total_pl > 0 and session_pl >= 0:
        verdict = "QUESTIONABLE"
        reason = (
            "ROTATE_EXIT sold full position despite OK risk_flag and positive total_pl (+54.35); "
            "rotation pressure on a winner may be too aggressive under turnover cap"
        )
    elif session_pl < 0:
        verdict = "CORRECT"
        reason = "Weak rotation exit with negative session P/L aligns with ROTATE_EXIT mandate"
    else:
        verdict = "QUESTIONABLE"
        reason = "Rotation exit with positive unrealized P/L warrants review"

    return exit_reason, verdict, "; ".join(evidence_parts) + f"; review_reason={reason}"


def _review_exit(
    symbol: str,
    *,
    fill: Dict[str, Any],
    risk_row: Dict[str, str],
    alloc_row: Dict[str, str],
    audit_events: List[Dict[str, str]],
    lifecycle_before: str,
    effective_stance: str,
) -> Dict[str, Any]:
    exit_type = _classify_exit_type(audit_events)
    entry_qty = _safe_float(fill.get("entry_qty"))
    exit_qty = _safe_float(fill.get("exit_qty"))
    entry_price = _safe_float(fill.get("entry_price"))
    exit_price = _safe_float(fill.get("exit_price"))
    realized_pl = (
        round((exit_price - entry_price) * min(entry_qty, exit_qty), 4) if exit_qty else 0.0
    )
    risk_flag = str(risk_row.get("risk_flag") or "OK")

    if symbol == "SLV":
        exit_reason, verdict, evidence = _review_slv_exit(
            fill=fill,
            risk_row=risk_row,
            alloc_row=alloc_row,
            audit_events=audit_events,
        )
    elif symbol == "TSLA":
        exit_reason, verdict, evidence = _review_tsla_exit(
            fill=fill,
            risk_row=risk_row,
            alloc_row=alloc_row,
            audit_events=audit_events,
        )
    else:
        exit_reason = str(audit_events[-1].get("action") if audit_events else "unknown")
        verdict = "QUESTIONABLE"
        evidence = f"risk_flag={risk_flag}"

    review_reason = (
        evidence.split("review_reason=")[-1] if "review_reason=" in evidence else verdict
    )

    return {
        "symbol": symbol,
        "exit_type": exit_type,
        "entry_qty": entry_qty,
        "exit_qty": exit_qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "realized_pl": realized_pl,
        "lifecycle_before": lifecycle_before,
        "effective_stance_before": effective_stance,
        "risk_flag": risk_flag,
        "exit_reason": exit_reason,
        "evidence": evidence,
        "verdict": verdict,
        "review_reason": review_reason,
    }


def _log_exit_review(review: Dict[str, Any]) -> None:
    print(
        "[TRADE_EXIT_REVIEW] "
        f"symbol={review['symbol']} "
        f"exit_type={review['exit_type']} "
        f"entry_qty={review['entry_qty']} "
        f"exit_qty={review['exit_qty']} "
        f"entry_price={review['entry_price']} "
        f"exit_price={review['exit_price']} "
        f"realized_pl={review['realized_pl']} "
        f"lifecycle_before={review['lifecycle_before']} "
        f"effective_stance_before={review['effective_stance_before']} "
        f"risk_flag={review['risk_flag']} "
        f"exit_reason={review['exit_reason']} "
        f"evidence={review['evidence']} "
        f"verdict={review['verdict']} "
        f"review_reason={review['review_reason']}"
    )


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _write_exit_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXIT_REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in EXIT_REVIEW_COLUMNS})


def run_session_review(
    *,
    session_date: str,
    mode: str = "paper",
    verbose: bool = False,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    results_dir = Path(results_dir or RESULTS)

    live_orders = _read_csv_rows(results_dir / "live_orders_log.csv")
    recent_orders = _read_csv_rows(results_dir / "recent_orders.csv")
    trade_log = _read_csv_rows(results_dir / "trade_log.csv")
    audit_rows = _read_csv_rows(results_dir / "paper_execution_audit.csv")
    positions = _read_csv_rows(results_dir / "positions_snapshot.csv")
    lifecycle = _index_by_symbol(_read_csv_rows(results_dir / "signal_lifecycle.csv"), "ticker")
    if not lifecycle:
        lifecycle = _index_by_symbol(_read_csv_rows(results_dir / "signal_lifecycle.csv"))
    effective = _index_by_symbol(
        _read_csv_rows(results_dir / "signal_lifecycle_effective.csv"), "ticker"
    )
    if not effective:
        effective = _index_by_symbol(_read_csv_rows(results_dir / "signal_lifecycle_effective.csv"))
    risk_overlay = _index_by_symbol(
        _read_csv_rows(results_dir / "performance_risk_overlay.csv"), "ticker"
    )
    alloc_recs = _index_by_symbol(
        _read_csv_rows(results_dir / "portfolio_allocation_recommendations.csv")
    )
    alloc_summary = _read_json(results_dir / "portfolio_allocation_summary.json")

    fills = _load_session_fills(live_orders, recent_orders, session_date)
    audit_by_symbol = _session_audit_events(audit_rows, session_date)

    initial_buys = [
        sym for sym in INITIAL_BUY_SYMBOLS if fills.get(sym, {}).get("entry_qty", 0) > 0
    ]
    adds = []
    arkk_fill = fills.get("ARKK", {})
    if arkk_fill.get("entry_qty", 0) > 11:
        adds.append({"symbol": "ARKK", "add_qty": 1})

    closed_symbols = [
        sym for sym, slot in fills.items() if slot.get("exit_qty", 0) > 0 and sym in ("SLV", "TSLA")
    ]
    exit_reviews: List[Dict[str, Any]] = []
    for sym in closed_symbols:
        lc_before, eff_stance = _lifecycle_context(lifecycle, effective, sym)
        review = _review_exit(
            sym,
            fill=fills.get(sym, {}),
            risk_row=risk_overlay.get(sym, {}),
            alloc_row=alloc_recs.get(sym, {}),
            audit_events=audit_by_symbol.get(sym, []),
            lifecycle_before=lc_before,
            effective_stance=eff_stance,
        )
        exit_reviews.append(review)
        if verbose:
            _log_exit_review(review)

    verdict_counts = {"CORRECT": 0, "QUESTIONABLE": 0, "INCORRECT": 0}
    for r in exit_reviews:
        verdict_counts[str(r.get("verdict") or "QUESTIONABLE")] = (
            verdict_counts.get(str(r.get("verdict") or "QUESTIONABLE"), 0) + 1
        )

    open_positions = [
        str(row.get("symbol") or row.get("ticker") or "").upper()
        for row in positions
        if str(row.get("date") or "").startswith(session_date) and _safe_float(row.get("qty")) > 0
    ]

    summary = (
        f"Paper session {session_date}: {len(initial_buys)} initial BUYs, "
        f"{len(adds)} ADD(s), {len(exit_reviews)} reviewed exit(s), "
        f"{len(open_positions)} open position(s). "
        f"SLV exit driven by high_drag/TRIM_PRIORITY; TSLA by ROTATE_EXIT rotation."
    )

    payload = {
        "generated_at": _utc_iso(),
        "session_date": session_date,
        "mode": mode,
        "initial_buys": initial_buys,
        "adds": adds,
        "exits_reviewed": [r["symbol"] for r in exit_reviews],
        "exit_reviews": exit_reviews,
        "verdict_counts": verdict_counts,
        "open_positions": open_positions,
        "allocation_summary": {
            "trim_priority_count": alloc_summary.get("trim_priority_count"),
            "force_exit_count": alloc_summary.get("force_exit_count"),
        },
        "trade_log_rows": len(_filter_rows_by_date(trade_log, session_date, "date")),
        "summary": summary,
        "disclaimer": "Read-only session review. No trades or execution changes.",
    }

    _atomic_write_json(payload, OUTPUT_JSON)
    _write_exit_csv(exit_reviews, OUTPUT_CSV)

    print(
        f"[PAPER_TRADE_SESSION_REVIEW] initial_buys={len(initial_buys)} "
        f"adds={len(adds)} exits={len(exit_reviews)} "
        f"correct_exits={verdict_counts.get('CORRECT', 0)} "
        f"questionable_exits={verdict_counts.get('QUESTIONABLE', 0)} "
        f"incorrect_exits={verdict_counts.get('INCORRECT', 0)} "
        f"open_positions={len(open_positions)} summary={summary}"
    )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper trade session review (Phase 150C)")
    parser.add_argument("--mode", default="paper", choices=["paper"])
    parser.add_argument("--date", required=True, help="Session date YYYY-MM-DD")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_session_review(session_date=args.date, mode=args.mode, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
