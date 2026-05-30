# services/reprice_open_orders.py
"""Paper-first stale buy limit repricing (cancel + replace). Best-effort; never raises to callers."""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from services.manage_open_orders import (
    _is_buy_side,
    _is_open_manageable_status,
    _load_session_map_from_log,
    _norm_status,
    _not_filled,
    _order_ts,
    load_open_orders_snapshot_or_broker,
)
from services.place_live_orders import DEFAULT_LOG_CSV, append_log_row, utc_now_iso
from services.execution_intelligence import (
    ExecutionIntelligenceConfig,
    evaluate_quote_quality,
    evaluate_liquidity,
    recommend_execution_style,
    recommend_partial_fill_action,
    compute_slippage_diagnostics,
    compute_execution_quality_score,
    classify_execution_risk_flag,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "reprice_open_orders.json"
OUT_CSV = RESULTS / "reprice_open_orders.csv"
OUT_LOG = RESULTS / "reprice_open_orders_log.csv"

DEFAULT_STALE_MINUTES = 15.0
DEFAULT_BUY_BUFFER_BPS = 50.0


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_minutes(ts: Optional[datetime], now: datetime) -> Optional[float]:
    if not ts:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 60.0


def _round_price(p: float) -> float:
    if p >= 1.0:
        return round(p, 2)
    return round(p, 4)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        v = float(x)
        if v <= 0:
            return None
        return v
    except Exception:
        return None


def _reference_buy_price(
    broker: Any,
    symbol: str,
    old_limit: Optional[float],
) -> Tuple[float, str]:
    sym = str(symbol).upper().strip()
    try:
        q = broker.get_latest_quote(sym)
        if q:
            ap = q.get("ask")
            if ap is not None:
                v = _safe_float(ap)
                if v:
                    return v, "quote_ask"
    except Exception:
        pass
    try:
        t = broker.get_latest_trade(sym)
        if t:
            lp = t.get("last")
            v = _safe_float(lp)
            if v:
                return v, "last_trade"
    except Exception:
        pass
    try:
        lp = broker.get_latest_price(sym)
        v = _safe_float(lp)
        if v:
            return v, "latest_price"
    except Exception:
        pass
    v = _safe_float(old_limit)
    if v:
        return v, "fallback_old_limit"
    raise ValueError(f"no reference price for {sym}")


def _iter_eligible_stale_buys(
    df: pd.DataFrame,
    *,
    now: datetime,
    stale_minutes: float,
    session_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if df.empty:
        return out
    for _, r in df.iterrows():
        if not _is_buy_side(r.get("side")):
            continue
        st = _norm_status(r.get("status"))
        if not _is_open_manageable_status(st):
            continue
        if not _not_filled(dict(r)):
            continue
        ts = _order_ts(r)
        age = _age_minutes(ts, now)
        if age is None or age < float(stale_minutes):
            continue
        oid = str(r.get("order_id") or r.get("id") or "").strip()
        sym = str(r.get("symbol") or r.get("ticker") or "").strip().upper()
        try:
            qty = int(float(r.get("qty") or r.get("quantity") or 0))
        except Exception:
            qty = 0
        if qty < 1:
            continue
        old_lp = _safe_float(r.get("limit_price") or r.get("price"))
        sess = session_map.get(oid, str(r.get("session") or ""))
        # Capture fill context for execution-intelligence partial-fill annotations.
        filled_qty = _safe_float(r.get("filled_qty") or r.get("filled_quantity") or 0.0) or 0.0
        filled_avg = _safe_float(r.get("filled_avg_price") or r.get("avg_fill_price"))
        out.append(
            {
                "order_id": oid,
                "symbol": sym,
                "side": str(r.get("side") or "").strip(),
                "qty": qty,
                "old_limit_price": old_lp,
                "status": st,
                "age_minutes": round(age, 3),
                "session": sess,
                "filled_qty": float(filled_qty),
                "filled_avg_price": filled_avg,
            }
        )
    return out


def _count_stale_buy_rows(df: pd.DataFrame, *, now: datetime, stale_minutes: float) -> int:
    n = 0
    if df.empty:
        return 0
    for _, r in df.iterrows():
        if not _is_buy_side(r.get("side")):
            continue
        ts = _order_ts(r)
        age = _age_minutes(ts, now)
        if age is None or age < float(stale_minutes):
            continue
        n += 1
    return n


def _append_summary_log(summary: Dict[str, Any]) -> None:
    try:
        OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts_utc": summary.get("timestamp", _utc_iso()),
            "mode": summary.get("mode", ""),
            "dry_run": str(summary.get("dry_run", True)).lower(),
            "stale_minutes": summary.get("stale_minutes", ""),
            "buy_buffer_bps": summary.get("buy_buffer_bps", ""),
            "stale_orders_seen": summary.get("stale_orders_seen", ""),
            "eligible_to_reprice": summary.get("eligible_to_reprice", ""),
            "canceled_orders": summary.get("canceled_orders", ""),
            "replacement_orders_prepared": summary.get("replacement_orders_prepared", ""),
            "replacement_orders_submitted": summary.get("replacement_orders_submitted", ""),
            "replacement_session": summary.get("replacement_session", ""),
            "notes": ";".join(str(x) for x in summary.get("notes", []) if x)[:4000],
        }
        new_file = not OUT_LOG.is_file() or OUT_LOG.stat().st_size == 0
        with OUT_LOG.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def run(
    *,
    mode: str,
    stale_minutes: float,
    buy_buffer_bps: float,
    execute: bool,
    verbose: bool,
) -> int:
    notes: List[str] = []
    now = datetime.now(timezone.utc)
    dry_run = not execute
    replacement_session = ""
    canceled_ok = 0
    submitted_ok = 0
    prepared = 0

    df, src = load_open_orders_snapshot_or_broker(mode, False)
    if verbose and not df.empty:
        print(f"[reprice_open_orders] loaded {len(df)} open orders from {src}", flush=True)
    if df.empty:
        notes.append(f"no open orders (source={src})")
        summary = _empty_summary(
            mode,
            dry_run=True,
            stale_minutes=stale_minutes,
            buy_buffer_bps=buy_buffer_bps,
            notes=notes,
        )
        _write_all([], summary)
        _print_summary(summary, verbose)
        return 0

    session_map = _load_session_map_from_log()
    stale_seen = _count_stale_buy_rows(df, now=now, stale_minutes=stale_minutes)
    eligible = _iter_eligible_stale_buys(
        df, now=now, stale_minutes=stale_minutes, session_map=session_map
    )
    prepared = len(eligible)

    rows_out: List[Dict[str, Any]] = []
    broker = None
    if execute:
        if mode.lower() != "paper":
            notes.append("LIVE: --execute refused by policy; use mode=paper for repricing.")
            dry_run = True
            execute = False
        else:
            dry_run = False
            replacement_session = f"reprice_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            try:
                from services.broker_alpaca import AlpacaBroker

                broker = AlpacaBroker(mode="paper")
            except Exception as e:
                notes.append(f"broker init failed: {e}")
                dry_run = True
                execute = False

    # Quotes for preview and execute (reuse execute broker when present)
    ref_broker = broker
    if ref_broker is None:
        try:
            from services.broker_alpaca import AlpacaBroker

            ref_broker = AlpacaBroker(mode=mode)
        except Exception:
            ref_broker = None

    for e in eligible:
        oid = e["order_id"]
        sym = e["symbol"]
        qty = e["qty"]
        old_lp = e.get("old_limit_price")
        ref = None
        ref_src = "unavailable"
        new_lp = None
        try:
            if ref_broker:
                ref, ref_src = _reference_buy_price(ref_broker, sym, old_lp)
                new_lp = _round_price(float(ref) * (1.0 + float(buy_buffer_bps) / 10000.0))
            else:
                ol = _safe_float(old_lp)
                if ol:
                    ref = ol
                    ref_src = "fallback_old_limit_no_quote"
                    new_lp = _round_price(float(ref) * (1.0 + float(buy_buffer_bps) / 10000.0))
        except Exception as ex:
            ref_src = f"error:{ex}"

        # ── Execution-intelligence annotations (read-only; never raise) ──
        ei_quote: Dict[str, Any] = {}
        ei_partial: Dict[str, Any] = {}
        ei_slippage: Dict[str, Any] = {}
        ei_quality: Dict[str, Any] = {}
        ei_risk_flag = "UNKNOWN"
        try:
            _ei_cfg = ExecutionIntelligenceConfig()
            _bid = _ask = _qts = None
            if ref_broker is not None:
                try:
                    q = ref_broker.get_quote(sym) or {}
                    _bid = _safe_float(q.get("bid"))
                    _ask = _safe_float(q.get("ask"))
                    _qts = q.get("timestamp") or q.get("ts") or q.get("t")
                except Exception:
                    pass
            ei_quote = evaluate_quote_quality(_bid, _ask, _qts, _ei_cfg)
            ei_partial = recommend_partial_fill_action(
                filled_qty=e.get("filled_qty"),
                total_qty=qty,
                order_age_minutes=e.get("age_minutes"),
                quote=ei_quote,
                cfg=_ei_cfg,
            )
            ei_slippage = compute_slippage_diagnostics(
                side=e.get("side"),
                intended_price=old_lp,
                submitted_limit_price=new_lp if new_lp is not None else old_lp,
                fill_price=e.get("filled_avg_price"),
                decision_mid_price=ei_quote.get("mid"),
            )
            # Quality score + risk flag are diagnostics built from the same
            # quote / liquidity / style signals already used above.
            _ei_liq = evaluate_liquidity(
                close=ei_quote.get("mid") or old_lp,
                avg_volume=None,
                order_qty=qty,
                cfg=_ei_cfg,
            )
            _ei_style = recommend_execution_style(
                action="REPRICE", quote=ei_quote, liquidity=_ei_liq, cfg=_ei_cfg
            )
            ei_quality = compute_execution_quality_score(
                quote=ei_quote, liquidity=_ei_liq, style=_ei_style
            )
            ei_risk_flag = classify_execution_risk_flag(
                ei_quality.get("execution_quality_score"),
                used_fallback=bool(ei_quality.get("execution_quality_used_fallback", False)),
            )
        except Exception:
            pass

        base_row = {
            "timestamp": _utc_iso(),
            "order_id": oid,
            "symbol": sym,
            "side": e["side"],
            "old_limit_price": old_lp if old_lp is not None else "",
            "new_limit_price": new_lp if new_lp is not None else "",
            "qty": qty,
            "status": e["status"],
            "age_minutes": e["age_minutes"],
            "session": e["session"],
            "replacement_session": replacement_session if replacement_session else "",
            "reference_price": ref if ref is not None else "",
            "reference_source": ref_src,
            # ── Execution-intelligence (additive; safe defaults when missing) ──
            "fill_pct": ei_partial.get("fill_pct"),
            "partial_fill_action": ei_partial.get("partial_fill_action"),
            "partial_fill_reason": ei_partial.get("partial_fill_reason"),
            "quote_age_sec": ei_quote.get("quote_age_sec"),
            "quote_is_stale": ei_quote.get("quote_is_stale"),
            "spread_bps": ei_quote.get("spread_bps"),
            "spread_bucket": ei_quote.get("spread_bucket"),
            "decision_mid_price": ei_slippage.get("decision_mid_price"),
            "expected_slippage_bps": ei_slippage.get("expected_slippage_bps"),
            "realized_slippage_bps": ei_slippage.get("realized_slippage_bps"),
            "execution_quality_score": ei_quality.get("execution_quality_score"),
            "execution_risk_flag": ei_risk_flag,
        }

        if dry_run or not execute:
            rows_out.append(
                {
                    **base_row,
                    "action": "preview",
                    "result": "would_reprice" if new_lp is not None else "skip_no_price",
                    "reason": "" if new_lp is not None else ref_src,
                }
            )
            continue

        # execute path (paper)
        if new_lp is None:
            rows_out.append(
                {
                    **base_row,
                    "action": "replace",
                    "result": "failed",
                    "reason": f"no_new_limit:{ref_src}",
                }
            )
            continue

        cancel_ok = False
        cancel_err = ""
        try:
            broker.cancel_order(oid)
            cancel_ok = True
            canceled_ok += 1
        except Exception as ex:
            cancel_err = str(ex)
            rows_out.append(
                {
                    **base_row,
                    "action": "cancel",
                    "result": "failed",
                    "reason": cancel_err[:500],
                }
            )
            continue

        coid = f"triton-rp{uuid.uuid4().hex[:20]}"
        try:
            resp = broker.submit_order(
                symbol=sym,
                qty=qty,
                side="buy",
                order_type="limit",
                time_in_force="day",
                limit_price=float(new_lp),
                client_order_id=coid,
                extended_hours=False,
            )
            oid_new = str((resp or {}).get("id") or (resp or {}).get("order_id") or "").strip()
            st = str((resp or {}).get("status") or "").strip().lower() or "submitted"
            submitted_ok += 1
            rows_out.append(
                {
                    **base_row,
                    "action": "replace",
                    "result": "ok",
                    "reason": f"new_order_id={oid_new};status={st};coid={coid}",
                }
            )
            try:
                append_log_row(
                    Path(DEFAULT_LOG_CSV),
                    {
                        "timestamp": utc_now_iso(),
                        "session": replacement_session,
                        "action": "submit",
                        "symbol": sym,
                        "side": "buy",
                        "qty": qty,
                        "type": "limit",
                        "limit_price": new_lp,
                        "order_id": oid_new,
                        "status": st,
                        "filled_qty": 0,
                        "filled_avg_price": "",
                        "client_order_id": coid,
                        "tp_limit": "",
                        "sl_stop": "",
                    },
                )
            except Exception:
                pass
            if verbose:
                print(f"[reprice] replaced {sym} qty={qty} limit={new_lp} id={oid_new}", flush=True)
        except Exception as ex:
            rows_out.append(
                {
                    **base_row,
                    "action": "submit",
                    "result": "failed",
                    "reason": str(ex)[:800],
                }
            )
            notes.append(f"{sym}: submit after cancel failed: {ex}")

    symbols_repriced = sorted({r["symbol"] for r in rows_out if r.get("result") == "ok"})
    summary = {
        "timestamp": _utc_iso(),
        "mode": mode,
        "dry_run": bool(dry_run),
        "stale_minutes": stale_minutes,
        "buy_buffer_bps": buy_buffer_bps,
        "data_source": src,
        "stale_orders_seen": stale_seen,
        "eligible_to_reprice": prepared,
        "canceled_orders": canceled_ok,
        "replacement_orders_prepared": prepared,
        "replacement_orders_submitted": submitted_ok,
        "replacement_session": replacement_session,
        "symbols_repriced": symbols_repriced,
        "notes": notes,
    }
    _write_all(rows_out, summary)
    _append_summary_log(summary)
    _print_summary(summary, verbose)
    return 0


def _empty_summary(
    mode: str,
    *,
    dry_run: bool,
    stale_minutes: float,
    buy_buffer_bps: float,
    notes: List[str],
) -> Dict[str, Any]:
    return {
        "timestamp": _utc_iso(),
        "mode": mode,
        "dry_run": dry_run,
        "stale_minutes": stale_minutes,
        "buy_buffer_bps": buy_buffer_bps,
        "data_source": "none",
        "stale_orders_seen": 0,
        "eligible_to_reprice": 0,
        "canceled_orders": 0,
        "replacement_orders_prepared": 0,
        "replacement_orders_submitted": 0,
        "replacement_session": "",
        "symbols_repriced": [],
        "notes": notes,
    }


def _write_all(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    cols = [
        "timestamp",
        "order_id",
        "symbol",
        "side",
        "old_limit_price",
        "new_limit_price",
        "qty",
        "status",
        "age_minutes",
        "action",
        "result",
        "reason",
        "session",
        "replacement_session",
        "reference_price",
        "reference_source",
        # ── Execution intelligence (additive; safe blanks when missing) ──
        "fill_pct",
        "partial_fill_action",
        "partial_fill_reason",
        "quote_age_sec",
        "quote_is_stale",
        "spread_bps",
        "spread_bucket",
        "decision_mid_price",
        "expected_slippage_bps",
        "realized_slippage_bps",
        "execution_quality_score",
        "execution_risk_flag",
    ]
    try:
        if rows:
            df_rows = pd.DataFrame(rows)
            # Preserve existing legacy columns first, then append any new ones in `cols`.
            legacy_cols = [c for c in df_rows.columns if c not in cols]
            ordered = [c for c in cols if c in df_rows.columns] + legacy_cols
            df_rows = df_rows.reindex(columns=ordered)
            df_rows.to_csv(OUT_CSV, index=False)
        else:
            pd.DataFrame(columns=cols).to_csv(OUT_CSV, index=False)
    except Exception:
        pass


def _print_summary(summary: Dict[str, Any], verbose: bool) -> None:
    print(
        f"[reprice_open_orders] mode={summary.get('mode')} dry_run={summary.get('dry_run')} "
        f"seen={summary.get('stale_orders_seen')} eligible={summary.get('eligible_to_reprice')} "
        f"bps={summary.get('buy_buffer_bps')} stale_min={summary.get('stale_minutes')}",
        flush=True,
    )
    if summary.get("replacement_session"):
        print(
            f"[reprice_open_orders] replacement_session={summary.get('replacement_session')}",
            flush=True,
        )
    if summary.get("dry_run"):
        print(
            f"[reprice_open_orders] would_prepare={summary.get('replacement_orders_prepared')} (no broker changes)",
            flush=True,
        )
    else:
        print(
            f"[reprice_open_orders] canceled={summary.get('canceled_orders')} "
            f"submitted={summary.get('replacement_orders_submitted')}",
            flush=True,
        )
    sy = summary.get("symbols_repriced") or []
    if sy:
        print(f"[reprice_open_orders] symbols_repriced: {', '.join(sy[:50])}", flush=True)
    n = summary.get("notes") or []
    if n and verbose:
        print(f"[reprice_open_orders] notes: {'; '.join(str(x) for x in n[:8])}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON reprice stale open buy limits (paper --execute)"
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--stale-minutes", type=float, default=DEFAULT_STALE_MINUTES)
    ap.add_argument("--buy-buffer-bps", type=float, default=DEFAULT_BUY_BUFFER_BPS)
    ap.add_argument("--execute", action="store_true", help="Cancel+replace (paper only)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    return run(
        mode=args.mode,
        stale_minutes=float(args.stale_minutes),
        buy_buffer_bps=float(args.buy_buffer_bps),
        execute=bool(args.execute),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    raise SystemExit(main())
