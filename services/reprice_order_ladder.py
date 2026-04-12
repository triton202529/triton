# services/reprice_order_ladder.py
"""Adaptive multi-stage BUY limit repricing ladder (paper-first; dry-run default). Best-effort; never raises."""
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
    TERMINAL_STATUSES,
    _is_buy_side,
    _is_open_manageable_status,
    _load_session_map_from_log,
    _norm_status,
    _not_filled,
    _order_ts,
    load_open_orders_snapshot_or_broker,
)
from services.order_discipline import (
    build_event_indexes,
    load_order_discipline_config,
    read_recent_order_events,
    should_block_order,
)
from services.place_live_orders import DEFAULT_LOG_CSV, append_log_row, utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
STATE_PATH = RESULTS / "reprice_ladder_state.json"
RUN_JSON = RESULTS / "reprice_ladder_run.json"
RUN_CSV = RESULTS / "reprice_ladder_run.csv"
LOG_CSV = RESULTS / "reprice_ladder_log.csv"

# Stage -> buffer bps (BUY limit above reference)
STAGE_BPS: Dict[int, int] = {1: 50, 2: 100, 3: 150, 4: 200}
LADDER_CAP_STAGE = 4


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _round_price(p: float) -> float:
    if p >= 1.0:
        return round(p, 2)
    return round(p, 4)


def _age_minutes(ts: Optional[datetime], now: datetime) -> Optional[float]:
    if not ts:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 60.0


def _stage_threshold_minutes(stage: int, stale_base: float) -> float:
    return float(stage) * float(stale_base)


def _improves_buy_limit(old_lp: Optional[float], new_lp: Optional[float]) -> bool:
    if old_lp is None or new_lp is None:
        return False
    eps = max(1e-8, abs(old_lp) * 1e-6)
    return new_lp > old_lp + eps


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


def _load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.is_file() and STATE_PATH.stat().st_size > 0:
            o = json.loads(STATE_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(o, dict):
                return o
    except Exception:
        pass
    return {"version": 1, "updated_at": "", "entries": []}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        state["updated_at"] = _utc_iso()
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        try:
            STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass


def _entries_list(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    e = state.get("entries")
    return e if isinstance(e, list) else []


def _find_entry_by_active(entries: List[Dict[str, Any]], order_id: str) -> Optional[Dict[str, Any]]:
    oid = str(order_id or "").strip()
    if not oid:
        return None
    for e in entries:
        if str(e.get("active_order_id") or "").strip() == oid:
            return e
    return None


def _new_entry(
    *,
    order_id: str,
    symbol: str,
    qty: int,
    original_session: str,
) -> Dict[str, Any]:
    return {
        "lineage_id": f"{symbol.upper()}|buy|{qty}|{order_id}",
        "original_session": original_session,
        "latest_reprice_session": "",
        "symbol": symbol.upper(),
        "side": "buy",
        "qty": int(qty),
        "current_stage": 0,
        "last_reprice_ts": "",
        "last_limit_price": None,
        "active_order_id": str(order_id).strip(),
        "status": "open",
        "terminal": False,
    }


def _reconcile_terminal(entries: List[Dict[str, Any]], open_ids: set) -> None:
    for e in entries:
        if e.get("terminal"):
            continue
        aid = str(e.get("active_order_id") or "").strip()
        if aid and aid not in open_ids:
            e["terminal"] = True
            e["status"] = "not_open"


_LOG_FIELDS = [
    "timestamp",
    "mode",
    "dry_run",
    "eligible_orders_seen",
    "orders_advanced",
    "orders_skipped",
    "replacement_orders_submitted",
    "replacement_session",
    "stage_counts_json",
    "notes",
]


def _append_run_log_row(row: Dict[str, Any]) -> None:
    try:
        LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        new_file = not LOG_CSV.is_file() or LOG_CSV.stat().st_size == 0
        out = {k: row.get(k, "") for k in _LOG_FIELDS}
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(out)
    except Exception:
        pass


def run(
    *,
    mode: str,
    execute: bool,
    verbose: bool,
    max_stage: int,
    stale_minutes_stage1: float,
) -> int:
    notes: List[str] = []
    now = datetime.now(timezone.utc)
    mode_l = str(mode or "paper").lower().strip()
    user_requested_execute = bool(execute)
    dry_run = not execute
    replacement_session = ""

    effective_max_stage = min(int(max_stage), LADDER_CAP_STAGE)
    if effective_max_stage < 1:
        effective_max_stage = 1

    # Live + execute: refuse (paper-first policy)
    if mode_l == "live" and execute:
        notes.append(
            "LIVE: --execute refused by policy; no broker mutations. Paper-only execution for ladder."
        )
        execute = False
        dry_run = True

    if execute and mode_l != "paper":
        notes.append("execute only allowed with --mode paper; diagnostics only.")
        execute = False
        dry_run = True

    state = _load_state()
    entries = _entries_list(state)

    df, src = load_open_orders_snapshot_or_broker(mode_l, verbose)
    if verbose and not df.empty:
        print(f"[reprice_order_ladder] loaded {len(df)} open orders from {src}", flush=True)

    session_map = _load_session_map_from_log()
    open_ids: set = set()
    try:
        for _, r in df.iterrows():
            oid = str(r.get("order_id") or r.get("id") or "").strip()
            if oid:
                open_ids.add(oid)
    except Exception:
        pass

    _reconcile_terminal(entries, open_ids)

    broker = None
    if execute:
        try:
            from services.broker_alpaca import AlpacaBroker

            broker = AlpacaBroker(mode="paper")
        except Exception as e:
            notes.append(f"broker init failed: {e}")
            execute = False
            dry_run = True

    ref_broker = broker
    if ref_broker is None:
        try:
            from services.broker_alpaca import AlpacaBroker

            ref_broker = AlpacaBroker(mode=mode_l)
        except Exception:
            ref_broker = None

    run_rows: List[Dict[str, Any]] = []
    stage_counts = {"stage_1": 0, "stage_2": 0, "stage_3": 0, "stage_4": 0}

    eligible_seen = 0
    orders_advanced = 0
    orders_skipped = 0
    replacement_submitted = 0
    symbols_repriced: List[str] = []

    if df.empty:
        notes.append(f"no open orders (source={src})")
        summary = _empty_summary(
            mode_l,
            dry_run=True,
            eligible_seen=0,
            orders_advanced=0,
            orders_skipped=0,
            replacement_submitted=0,
            replacement_session="",
            stage_counts=stage_counts,
            symbols_repriced=[],
            notes=notes,
            execute_requested=user_requested_execute,
        )
        state["entries"] = entries
        _save_state(state)
        _write_artifacts(run_rows, summary)
        _print_summary(summary, verbose)
        return 0

    if execute:
        replacement_session = (
            f"reprice_ladder_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

    for _, r in df.iterrows():
        if not _is_buy_side(r.get("side")):
            continue
        st = _norm_status(r.get("status"))
        if st in TERMINAL_STATUSES or st == "filled":
            run_rows.append(
                _result_row(
                    r,
                    now,
                    session_map,
                    replacement_session,
                    old_oid=str(r.get("order_id") or ""),
                    new_oid="",
                    old_lp=_safe_float(r.get("limit_price") or r.get("price")),
                    new_lp=None,
                    cur_stage=0,
                    next_stage=0,
                    action="evaluate",
                    result="skipped",
                    reason="skipped_terminal",
                    source_session="",
                )
            )
            orders_skipped += 1
            continue
        if not _is_open_manageable_status(st):
            run_rows.append(
                _result_row(
                    r,
                    now,
                    session_map,
                    replacement_session,
                    old_oid=str(r.get("order_id") or ""),
                    new_oid="",
                    old_lp=_safe_float(r.get("limit_price") or r.get("price")),
                    new_lp=None,
                    cur_stage=0,
                    next_stage=0,
                    action="evaluate",
                    result="skipped",
                    reason="skipped_not_eligible",
                    source_session="",
                )
            )
            orders_skipped += 1
            continue
        if not _not_filled(dict(r)):
            run_rows.append(
                _result_row(
                    r,
                    now,
                    session_map,
                    replacement_session,
                    old_oid=str(r.get("order_id") or ""),
                    new_oid="",
                    old_lp=_safe_float(r.get("limit_price") or r.get("price")),
                    new_lp=None,
                    cur_stage=0,
                    next_stage=0,
                    action="evaluate",
                    result="skipped",
                    reason="skipped_terminal",
                    source_session="",
                )
            )
            orders_skipped += 1
            continue

        oid = str(r.get("order_id") or r.get("id") or "").strip()
        sym = str(r.get("symbol") or r.get("ticker") or "").strip().upper()
        try:
            qty = int(float(r.get("qty") or r.get("quantity") or 0))
        except Exception:
            qty = 0
        if qty < 1:
            run_rows.append(
                _result_row(
                    r,
                    now,
                    session_map,
                    replacement_session,
                    old_oid=oid,
                    new_oid="",
                    old_lp=_safe_float(r.get("limit_price") or r.get("price")),
                    new_lp=None,
                    cur_stage=0,
                    next_stage=0,
                    action="evaluate",
                    result="skipped",
                    reason="skipped_not_eligible",
                    source_session="",
                )
            )
            orders_skipped += 1
            continue
        old_lp = _safe_float(r.get("limit_price") or r.get("price"))
        ts = _order_ts(r)
        age = _age_minutes(ts, now)
        sess = session_map.get(oid, str(r.get("session") or ""))

        eligible_seen += 1

        entry = _find_entry_by_active(entries, oid)
        if entry is None:
            entry = _new_entry(order_id=oid, symbol=sym, qty=qty, original_session=sess or "")
            entries.append(entry)

        cur_stage = int(entry.get("current_stage") or 0)
        next_stage = cur_stage + 1

        base_row = _result_row(
            r,
            now,
            session_map,
            replacement_session,
            old_oid=oid,
            new_oid="",
            old_lp=old_lp,
            new_lp=None,
            cur_stage=cur_stage,
            next_stage=next_stage,
            action="preview",
            result="pending",
            reason="",
            source_session=str(entry.get("original_session") or sess or ""),
        )

        if entry.get("terminal"):
            base_row["result"] = "skipped"
            base_row["reason"] = "skipped_terminal"
            base_row["action"] = "evaluate"
            run_rows.append(base_row)
            orders_skipped += 1
            continue

        if next_stage > LADDER_CAP_STAGE:
            base_row["result"] = "skipped"
            base_row["reason"] = "skipped_already_at_stage"
            base_row["action"] = "evaluate"
            run_rows.append(base_row)
            orders_skipped += 1
            continue

        if next_stage > effective_max_stage:
            base_row["result"] = "skipped"
            base_row["reason"] = "skipped_max_stage_cap"
            base_row["action"] = "evaluate"
            run_rows.append(base_row)
            orders_skipped += 1
            continue

        need_age = _stage_threshold_minutes(next_stage, stale_minutes_stage1)
        if age is None or age < need_age:
            base_row["result"] = "skipped"
            base_row["reason"] = f"age_below_stage_{next_stage}_threshold"
            base_row["action"] = "evaluate"
            run_rows.append(base_row)
            orders_skipped += 1
            continue

        bps = STAGE_BPS.get(next_stage, STAGE_BPS[LADDER_CAP_STAGE])
        ref = None
        ref_src = "unavailable"
        new_lp: Optional[float] = None
        try:
            if ref_broker:
                ref, ref_src = _reference_buy_price(ref_broker, sym, old_lp)
                new_lp = _round_price(float(ref) * (1.0 + float(bps) / 10000.0))
            else:
                ol = _safe_float(old_lp)
                if ol:
                    ref = ol
                    ref_src = "fallback_old_limit_no_quote"
                    new_lp = _round_price(float(ref) * (1.0 + float(bps) / 10000.0))
        except Exception as ex:
            ref_src = f"error:{ex}"

        base_row["reference_price"] = ref if ref is not None else ""
        base_row["reference_source"] = ref_src

        if new_lp is None:
            base_row["result"] = "failed"
            base_row["reason"] = f"submit_failed:{ref_src}"
            base_row["action"] = "evaluate"
            run_rows.append(base_row)
            orders_skipped += 1
            continue

        if not _improves_buy_limit(old_lp, new_lp):
            base_row["new_limit_price"] = new_lp
            base_row["result"] = "skipped"
            base_row["reason"] = "NO_IMPROVEMENT_OVER_CURRENT_LIMIT"
            base_row["action"] = "evaluate"
            run_rows.append(base_row)
            orders_skipped += 1
            continue

        base_row["new_limit_price"] = new_lp
        sk = f"stage_{next_stage}"
        if sk in stage_counts:
            stage_counts[sk] += 1

        if not execute or dry_run:
            base_row["action"] = "preview"
            base_row["result"] = "would_advance"
            base_row["reason"] = f"next_stage={next_stage};bps={bps}"
            run_rows.append(base_row)
            orders_advanced += 1
            continue

        # execute path (paper)
        odc = load_order_discipline_config()
        if odc.get("enabled") and not odc.get("allow_reprice_replacements", True):
            lb = float(odc.get("cross_session_cooldown_minutes", 30) or 30)
            ev = read_recent_order_events(lookback_minutes=lb)
            idx = build_event_indexes(ev)
            blocked, rs = should_block_order(
                sym,
                "buy",
                replacement_session,
                {"is_reprice_replacement": False},
                cfg=odc,
                events=ev,
                event_index=idx,
                session_seen=set(),
                open_side_keys=None,
            )
            if blocked:
                base_row["action"] = "evaluate"
                base_row["result"] = "skipped"
                base_row["reason"] = f"order_discipline:{rs}"
                run_rows.append(base_row)
                orders_skipped += 1
                notes.append(f"{sym}: reprice blocked by order_discipline ({rs})")
                continue

        cancel_err = ""
        try:
            broker.cancel_order(oid)
        except Exception as ex:
            cancel_err = str(ex)
            base_row["action"] = "cancel"
            base_row["result"] = "failed"
            base_row["reason"] = f"cancel_failed:{cancel_err[:500]}"
            run_rows.append(base_row)
            orders_skipped += 1
            continue

        coid = f"triton-ld{uuid.uuid4().hex[:20]}"
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
            st_new = str((resp or {}).get("status") or "").strip().lower() or "submitted"
            replacement_submitted += 1
            orders_advanced += 1
            if sym and sym not in symbols_repriced:
                symbols_repriced.append(sym)

            entry["current_stage"] = next_stage
            entry["last_reprice_ts"] = _utc_iso()
            entry["last_limit_price"] = new_lp
            entry["latest_reprice_session"] = replacement_session
            entry["active_order_id"] = oid_new or entry["active_order_id"]
            entry["status"] = "open"
            entry["terminal"] = False

            base_row["new_order_id"] = oid_new
            base_row["action"] = "replace"
            base_row["result"] = "ok"
            base_row["reason"] = f"new_order_id={oid_new};status={st_new};coid={coid}"
            run_rows.append(base_row)

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
                        "status": st_new,
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
                print(
                    f"[reprice_order_ladder] advanced {sym} stage {cur_stage}->{next_stage} limit={new_lp} id={oid_new}",
                    flush=True,
                )
        except Exception as ex:
            base_row["action"] = "submit"
            base_row["result"] = "failed"
            base_row["reason"] = f"submit_failed:{str(ex)[:800]}"
            run_rows.append(base_row)
            orders_skipped += 1
            notes.append(f"{sym}: submit after cancel failed: {ex}")

    state["entries"] = entries
    _save_state(state)

    summary = {
        "timestamp": _utc_iso(),
        "mode": mode_l,
        "dry_run": bool(dry_run),
        "execute_requested": user_requested_execute,
        "eligible_orders_seen": eligible_seen,
        "orders_advanced": orders_advanced,
        "orders_skipped": orders_skipped,
        "replacement_orders_submitted": replacement_submitted,
        "replacement_session": replacement_session,
        "stage_counts": stage_counts,
        "symbols_repriced": sorted(symbols_repriced),
        "max_stage": effective_max_stage,
        "stale_minutes_stage1": stale_minutes_stage1,
        "data_source": src,
        "notes": notes,
    }

    _write_artifacts(run_rows, summary)
    _print_summary(summary, verbose)
    return 0


def _result_row(
    r: pd.Series,
    now: datetime,
    session_map: Dict[str, str],
    replacement_session: str,
    *,
    old_oid: str,
    new_oid: str,
    old_lp: Optional[float],
    new_lp: Optional[float],
    cur_stage: int,
    next_stage: int,
    action: str,
    result: str,
    reason: str,
    source_session: str,
    reference_price: Any = "",
    reference_source: str = "",
) -> Dict[str, Any]:
    oid = str(old_oid or r.get("order_id") or "").strip()
    sym = str(r.get("symbol") or r.get("ticker") or "").strip().upper()
    ts = _order_ts(r)
    age = _age_minutes(ts, now)
    st = _norm_status(r.get("status"))
    try:
        qty = int(float(r.get("qty") or r.get("quantity") or 0))
    except Exception:
        qty = 0
    sess = session_map.get(oid, str(r.get("session") or ""))
    return {
        "timestamp": _utc_iso(),
        "symbol": sym,
        "side": "buy",
        "old_order_id": oid,
        "new_order_id": new_oid,
        "old_limit_price": old_lp if old_lp is not None else "",
        "new_limit_price": new_lp if new_lp is not None else "",
        "qty": qty,
        "old_status": st,
        "age_minutes": round(age, 3) if age is not None else "",
        "current_stage": cur_stage,
        "next_stage": next_stage,
        "action": action,
        "result": result,
        "reason": reason,
        "source_session": source_session or sess,
        "replacement_session": replacement_session,
        "reference_price": reference_price,
        "reference_source": reference_source,
    }


def _empty_summary(
    mode: str,
    *,
    dry_run: bool,
    eligible_seen: int,
    orders_advanced: int,
    orders_skipped: int,
    replacement_submitted: int,
    replacement_session: str,
    stage_counts: Dict[str, int],
    symbols_repriced: List[str],
    notes: List[str],
    execute_requested: bool = False,
) -> Dict[str, Any]:
    return {
        "timestamp": _utc_iso(),
        "mode": mode,
        "dry_run": dry_run,
        "execute_requested": execute_requested,
        "eligible_orders_seen": eligible_seen,
        "orders_advanced": orders_advanced,
        "orders_skipped": orders_skipped,
        "replacement_orders_submitted": replacement_submitted,
        "replacement_session": replacement_session,
        "stage_counts": stage_counts,
        "symbols_repriced": symbols_repriced,
        "max_stage": LADDER_CAP_STAGE,
        "stale_minutes_stage1": 15.0,
        "data_source": "none",
        "notes": notes,
    }


def _write_artifacts(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        RUN_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    cols = [
        "timestamp",
        "symbol",
        "side",
        "old_order_id",
        "new_order_id",
        "old_limit_price",
        "new_limit_price",
        "qty",
        "old_status",
        "age_minutes",
        "current_stage",
        "next_stage",
        "action",
        "result",
        "reason",
        "source_session",
        "replacement_session",
        "reference_price",
        "reference_source",
    ]
    try:
        if rows:
            df = pd.DataFrame(rows)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[cols]
            df.to_csv(RUN_CSV, index=False)
        else:
            pd.DataFrame(columns=cols).to_csv(RUN_CSV, index=False)
    except Exception:
        pass

    try:
        _append_run_log_row(
            {
                "timestamp": summary.get("timestamp", _utc_iso()),
                "mode": summary.get("mode", ""),
                "dry_run": str(summary.get("dry_run", True)).lower(),
                "eligible_orders_seen": summary.get("eligible_orders_seen", ""),
                "orders_advanced": summary.get("orders_advanced", ""),
                "orders_skipped": summary.get("orders_skipped", ""),
                "replacement_orders_submitted": summary.get("replacement_orders_submitted", ""),
                "replacement_session": summary.get("replacement_session", ""),
                "stage_counts_json": json.dumps(summary.get("stage_counts") or {}),
                "notes": ";".join(str(x) for x in summary.get("notes", []) if x)[:4000],
            }
        )
    except Exception:
        pass


def _print_summary(summary: Dict[str, Any], verbose: bool) -> None:
    print(
        f"[reprice_order_ladder] mode={summary.get('mode')} dry_run={summary.get('dry_run')} "
        f"eligible={summary.get('eligible_orders_seen')} advanced={summary.get('orders_advanced')} "
        f"skipped={summary.get('orders_skipped')} submitted={summary.get('replacement_orders_submitted')}",
        flush=True,
    )
    if summary.get("replacement_session"):
        print(
            f"[reprice_order_ladder] replacement_session={summary.get('replacement_session')}",
            flush=True,
        )
    sy = summary.get("symbols_repriced") or []
    if sy:
        print(f"[reprice_order_ladder] symbols_repriced: {', '.join(sy[:50])}", flush=True)
    n = summary.get("notes") or []
    if n and verbose:
        print(f"[reprice_order_ladder] notes: {'; '.join(str(x) for x in n[:8])}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON adaptive BUY repricing ladder (paper --execute)"
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--execute", action="store_true", help="Cancel+replace one stage (paper only)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--max-stage", type=int, default=LADDER_CAP_STAGE, help="Max ladder stage (1-4)"
    )
    ap.add_argument(
        "--stale-minutes-stage1",
        type=float,
        default=15.0,
        help="Minutes per stage unit (stage N requires age >= N * this value)",
    )
    args = ap.parse_args(argv)
    return run(
        mode=args.mode,
        execute=bool(args.execute),
        verbose=bool(args.verbose),
        max_stage=int(args.max_stage),
        stale_minutes_stage1=float(args.stale_minutes_stage1),
    )


if __name__ == "__main__":
    raise SystemExit(main())
