# services/signal_pressure_diagnostics.py
"""Post-hoc signal funnel diagnostics (read-only). Never raises to pipeline callers."""
from __future__ import annotations

import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "signal_pressure_diagnostics.json"
OUT_CSV = RESULTS / "signal_pressure_diagnostics.csv"
OUT_LOG = RESULTS / "signal_pressure_diagnostics_log.csv"

# Match generate_signals.py
BUY_DELTA_GEN = 0.002
SELL_DELTA_GEN = -0.002

# Match services/apply_signal_lifecycle.py main()
LIFECYCLE_CFG_KWARGS = dict(
    min_hold_days=1,
    cooldown_days_after_exit=1,
    buy_delta_pct=0.0015,
    add_delta_pct=0.0020,
    exit_delta_pct=-0.0015,
    trim_delta_pct=-0.0030,
    hold_means_keep_position=True,
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_stats(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(s.min()),
        "median": float(s.median()),
        "max": float(s.max()),
    }


def _replay_lifecycle(
    signals_df: pd.DataFrame,
    state: Dict[str, Any],
    cfg,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Replay decide_action sequence (same ordering as apply_lifecycle) without writing state to disk.
    Returns per-ticker rows and aggregate filter counts.
    """
    from services.signal_lifecycle import (
        decide_action,
        _get_ticker_state,
        _set_ticker_state,
        _normalize_signal_value,
        _ensure_delta_pct,
        _get_strength,
        _parse_date_any,
        _days_between,
        _today_utc_date,
    )

    df = signals_df.copy()
    if "ticker" not in df.columns:
        return [], {}

    df = _ensure_delta_pct(df)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "as_of_date" in df.columns and df["as_of_date"].notna().any():
        df["_asof"] = df["as_of_date"]
    elif "date" in df.columns:
        df["_asof"] = df["date"]
    else:
        df["_asof"] = pd.NaT

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "signal" in df.columns:
        df["_signal_norm"] = df["signal"].apply(_normalize_signal_value)
    else:
        df["_signal_norm"] = "UNKNOWN"

    df = df.sort_values(
        ["_asof", "ticker"], ascending=[True, True], na_position="last"
    ).reset_index(drop=True)

    state = copy.deepcopy(state) if isinstance(state, dict) else {}
    filters: Dict[str, int] = {
        "low_confidence": 0,
        "low_delta": 0,
        "invalid_price": 0,
        "position_state_suppressed": 0,
        "lifecycle_hold_suppressed": 0,
        "lifecycle_wait_suppressed": 0,
        "cooldown_wait": 0,
    }
    rows_out: List[Dict[str, Any]] = []

    for i in range(len(df)):
        row = df.iloc[i]
        ticker = str(row["ticker"]).strip().upper()
        as_of = _parse_date_any(row["_asof"])

        tstate = _get_ticker_state(state, ticker)
        pos_state = str(tstate.get("position_state", "FLAT")).upper()
        pos_before = pos_state
        last_change_date = _parse_date_any(tstate.get("last_change_date"))
        cooldown_until = _parse_date_any(tstate.get("cooldown_until"))
        last_action = str(tstate.get("last_action", "NONE")).upper()

        sig = row["_signal_norm"]
        strength = _get_strength(row)
        if not np.isfinite(strength):
            strength_used = 0.0
        else:
            strength_used = float(strength)
        hold_age = _days_between(last_change_date, as_of)
        can_exit = hold_age >= max(0, cfg.min_hold_days)

        action = decide_action(
            cfg=cfg,
            signal=sig,
            strength=strength,
            pos_state=pos_state,
            last_change_date=last_change_date,
            cooldown_until=cooldown_until,
            as_of_date=as_of,
        )

        # classify filters (aligned with decide_action in signal_lifecycle.py)
        sup = ""
        if (
            cooldown_until
            and as_of
            and as_of < cooldown_until
            and sig == "BUY"
            and pos_state == "FLAT"
        ):
            filters["cooldown_wait"] += 1
            filters["lifecycle_wait_suppressed"] += 1
            sup = "cooldown_wait"
        elif (
            action == "WAIT"
            and pos_state == "FLAT"
            and sig == "BUY"
            and strength_used < cfg.buy_delta_pct
        ):
            filters["low_delta"] += 1
            filters["lifecycle_wait_suppressed"] += 1
            sup = "low_delta_wait"
        elif action == "WAIT":
            filters["lifecycle_wait_suppressed"] += 1
            sup = "wait"
        elif (
            action == "HOLD"
            and pos_state == "LONG"
            and sig == "BUY"
            and strength_used < cfg.add_delta_pct
        ):
            filters["low_confidence"] += 1
            filters["lifecycle_hold_suppressed"] += 1
            sup = "add_weak_hold"
        elif action == "HOLD" and pos_state == "LONG" and sig == "SELL" and not can_exit:
            filters["position_state_suppressed"] += 1
            filters["lifecycle_hold_suppressed"] += 1
            sup = "min_hold_blocks_exit"
        elif action == "HOLD":
            filters["lifecycle_hold_suppressed"] += 1
            sup = "hold"
        else:
            sup = ""

        if action == "BUY":
            if pos_state == "FLAT":
                pos_state = "LONG"
                last_change_date = as_of or _today_utc_date()
                last_action = "BUY"
                cooldown_until = None

        elif action == "ADD":
            if pos_state == "LONG":
                last_action = "ADD"

        elif action == "TRIM":
            if pos_state == "LONG":
                last_action = "TRIM"

        elif action == "EXIT":
            if pos_state == "LONG":
                pos_state = "FLAT"
                last_action = "EXIT"
                last_change_date = as_of or _today_utc_date()
                cd = (as_of or _today_utc_date()) + pd.Timedelta(days=cfg.cooldown_days_after_exit)
                cooldown_until = cd.date() if hasattr(cd, "date") else None

        elif action == "HOLD":
            if pos_state == "LONG":
                last_action = "HOLD"

        tstate["position_state"] = pos_state
        tstate["last_action"] = last_action
        tstate["last_change_date"] = str(last_change_date) if last_change_date else None
        tstate["cooldown_until"] = str(cooldown_until) if cooldown_until else None
        _set_ticker_state(state, ticker, tstate)

        conf = row.get("confidence")
        try:
            conf_f = float(conf) if conf is not None and pd.notna(conf) else float("nan")
        except Exception:
            conf_f = float("nan")
        dpc = row.get("delta_pct")
        try:
            dpc_f = float(dpc) if dpc is not None and pd.notna(dpc) else float("nan")
        except Exception:
            dpc_f = float("nan")

        if pos_before == "FLAT" and sig == "BUY":
            passed_delta = bool(strength_used >= cfg.buy_delta_pct)
            passed_conf = bool(np.isfinite(conf_f) and conf_f >= cfg.buy_delta_pct)
        elif pos_before == "LONG" and sig == "BUY":
            passed_delta = bool(strength_used >= cfg.add_delta_pct)
            passed_conf = bool(np.isfinite(conf_f) and conf_f >= cfg.add_delta_pct)
        else:
            passed_delta = ""
            passed_conf = ""

        asof_raw = row["_asof"]
        rows_out.append(
            {
                "timestamp": _utc_iso(),
                "ticker": ticker,
                "raw_signal": str(sig),
                "confidence": conf_f if pd.notna(conf_f) else "",
                "delta_pct": dpc_f if pd.notna(dpc_f) else "",
                "passed_confidence": passed_conf,
                "passed_delta": passed_delta,
                "lifecycle_action": action,
                "suppression_reason": sup,
                "_asof": asof_raw,
            }
        )

    return rows_out, filters


def refresh_signal_pressure_diagnostics() -> None:
    """Load artifacts from disk, compute funnel metrics, write JSON/CSV/log. Best-effort."""
    notes: List[str] = []
    try:
        from services.signal_lifecycle import LifecycleConfig, load_state

        cfg = LifecycleConfig(**LIFECYCLE_CFG_KWARGS)
        state_path = RESULTS / "signal_state.json"

        sig_path = RESULTS / "signals_with_rationale.csv"
        if not sig_path.exists() or sig_path.stat().st_size == 0:
            sig_path = RESULTS / "signals.csv"

        if not sig_path.exists() or sig_path.stat().st_size == 0:
            notes.append("No signals CSV; run generate_signals first.")
            payload = {
                "timestamp": _utc_iso(),
                "ticker_count": 0,
                "raw_signal_counts": {},
                "filters": {},
                "final_counts": {},
                "confidence_stats": {},
                "delta_pct_stats": {},
                "notes": notes,
                "rows": [],
            }
            _write_all(payload, [])
            return

        signals = pd.read_csv(sig_path)
        if signals.empty:
            notes.append("signals file empty")
            _write_minimal(notes)
            return

        signals = signals.copy()
        signals.columns = [str(c).strip() for c in signals.columns]

        n_tickers = (
            int(signals["ticker"].nunique()) if "ticker" in signals.columns else len(signals)
        )

        # Raw classification (model output band — same as generate_signals BUY/SELL/HOLD)
        if "signal" in signals.columns:
            vc = signals["signal"].astype(str).str.upper().str.strip().value_counts().to_dict()
        else:
            vc = {}

        raw_buy = int(vc.get("BUY", 0)) if vc else 0
        raw_sell = int(vc.get("SELL", 0)) if vc else 0
        raw_hold = int(vc.get("HOLD", 0)) + int(vc.get("WAIT", 0)) if vc else 0

        # Pre-threshold buy-like: delta would qualify as BUY in generate_signals
        if "delta_pct" in signals.columns:
            d = pd.to_numeric(signals["delta_pct"], errors="coerce")
            buy_before = int((d >= BUY_DELTA_GEN).sum())
            sell_before = int((d <= SELL_DELTA_GEN).sum())
            neutral = int(((d > SELL_DELTA_GEN) & (d < BUY_DELTA_GEN)).sum())
        else:
            buy_before = raw_buy
            sell_before = raw_sell
            neutral = raw_hold
            notes.append("delta_pct missing; raw_signal_counts from signal column only")

        # Replay lifecycle for accurate WAIT/HOLD reasons
        st = load_state(state_path)
        replay_rows, filt = _replay_lifecycle(signals, st, cfg)

        invalid_price = 0
        if "close" in signals.columns:
            c = pd.to_numeric(signals["close"], errors="coerce")
            invalid_price = int((c.isna() | (c <= 0)).sum())
        filt["invalid_price"] = int(filt.get("invalid_price", 0)) + invalid_price

        # Merge effective lifecycle + opportunities
        eff_path = RESULTS / "signal_lifecycle_effective.csv"
        opp_path = RESULTS / "trade_opportunities.csv"
        eff_df = None
        if eff_path.exists() and eff_path.stat().st_size > 0:
            try:
                eff_df = pd.read_csv(eff_path)
                eff_df.columns = [str(c).strip() for c in eff_df.columns]
            except Exception:
                eff_df = None
                notes.append("Could not read signal_lifecycle_effective.csv")
        else:
            notes.append(
                "signal_lifecycle_effective.csv not present yet (build_effective_lifecycle)"
            )

        opps: set = set()
        if opp_path.exists() and opp_path.stat().st_size > 0:
            try:
                od = pd.read_csv(opp_path)
                if "ticker" in od.columns:
                    opps = set(od["ticker"].astype(str).str.upper().str.strip().tolist())
            except Exception:
                notes.append("Could not read trade_opportunities.csv")

        # One row per ticker (match apply_lifecycle: latest as-of wins)
        if replay_rows:
            rr_df = pd.DataFrame(replay_rows)
            rr_df["_sort_ts"] = pd.to_datetime(rr_df["_asof"], errors="coerce")
            rr_df = rr_df.sort_values(
                ["ticker", "_sort_ts"], ascending=[True, False], na_position="last"
            )
            rr_df = rr_df.drop_duplicates("ticker", keep="first")
            replay_dedup = rr_df.to_dict("records")
        else:
            replay_dedup = []

        try:
            from services.build_trade_opportunities import classify_opportunity_from_lifecycle
        except Exception:
            classify_opportunity_from_lifecycle = None  # type: ignore

        csv_rows: List[Dict[str, Any]] = []
        for r in replay_dedup:
            t = str(r["ticker"]).upper()
            es = ""
            eps = ""
            row0 = None
            la_eff = ""
            if eff_df is not None and "ticker" in eff_df.columns:
                sub = eff_df[eff_df["ticker"].astype(str).str.upper().str.strip() == t]
                if not sub.empty:
                    row0 = sub.iloc[0]
                    es = str(row0.get("effective_stance", "") or "")
                    eps = str(row0.get("effective_position_state", "") or "").strip().upper()
                    if "lifecycle_action" in row0.index:
                        la_eff = str(row0.get("lifecycle_action") or "").strip().upper()
                    elif "stance" in row0.index:
                        la_eff = str(row0.get("stance") or "").strip().upper()
            final_a = t in opps
            sup = str(r.get("suppression_reason", "") or "")
            la_u = str(r.get("lifecycle_action", "") or "").strip().upper()
            eligible_eff = (
                bool(classify_opportunity_from_lifecycle)
                and bool(la_eff)
                and classify_opportunity_from_lifecycle(eps, la_eff) is not None
            )
            if final_a:
                sup_reason = "ACTIONABLE"
            elif sup:
                sup_reason = sup
            elif eligible_eff:
                sup_reason = "eligible_lifecycle_not_in_opportunities"
            elif la_u == "WAIT":
                sup_reason = "lifecycle_wait"
            elif la_u == "HOLD":
                sup_reason = "lifecycle_hold"
            else:
                sup_reason = la_u or ""

            csv_rows.append(
                {
                    "timestamp": r["timestamp"],
                    "ticker": t,
                    "raw_signal": r.get("raw_signal", ""),
                    "confidence": r.get("confidence", ""),
                    "delta_pct": r.get("delta_pct", ""),
                    "passed_confidence": r.get("passed_confidence", ""),
                    "passed_delta": r.get("passed_delta", ""),
                    "lifecycle_stance": r.get("lifecycle_action", ""),
                    "effective_stance": es,
                    "effective_position_state": eps,
                    "final_actionable": final_a,
                    "suppression_reason": sup_reason,
                }
            )

        # Final stance counts from effective if available
        final_counts: Dict[str, int] = {}
        if eff_df is not None and "effective_stance" in eff_df.columns:
            ec = (
                eff_df["effective_stance"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
                .value_counts()
            )
            for k in ("BUY", "ADD", "TRIM", "EXIT", "HOLD", "WAIT"):
                final_counts[k.lower()] = int(ec.get(k, 0))
        else:
            # fallback: lifecycle replay last row per ticker
            lc_path = RESULTS / "signal_lifecycle.csv"
            if lc_path.exists():
                try:
                    lc = pd.read_csv(lc_path)
                    if "lifecycle_action" in lc.columns:
                        fc = (
                            lc["lifecycle_action"].fillna("").astype(str).str.upper().value_counts()
                        )
                        for k in ("BUY", "ADD", "TRIM", "EXIT", "HOLD", "WAIT"):
                            final_counts[k.lower()] = int(fc.get(k, 0))
                except Exception:
                    pass

        actionable_count = len(opps)

        conf_stats = _safe_stats(signals["confidence"]) if "confidence" in signals.columns else {}
        d_stats = _safe_stats(signals["delta_pct"]) if "delta_pct" in signals.columns else {}

        payload: Dict[str, Any] = {
            "timestamp": _utc_iso(),
            "ticker_count": n_tickers,
            "raw_signal_counts": {
                "buy_like": raw_buy,
                "sell_like": raw_sell,
                "neutral_like": raw_hold,
            },
            "thresholds_reference": {
                "generate_signals_buy_delta": BUY_DELTA_GEN,
                "generate_signals_sell_delta": SELL_DELTA_GEN,
                "lifecycle_config": LIFECYCLE_CFG_KWARGS,
            },
            "candidates": {
                "buy_candidates_before_threshold_delta": buy_before,
                "sell_exit_candidates_before_threshold_delta": sell_before,
                "neutral_band_delta": neutral,
                "buy_candidates_after_threshold_signal_column": raw_buy,
                "sell_exit_candidates_after_threshold_signal_column": raw_sell,
            },
            "filters": filt,
            "final_counts": {**final_counts, "actionable_opportunities": actionable_count},
            "confidence_stats": conf_stats,
            "delta_pct_stats": d_stats,
            "signal_value_stats": d_stats,
            "suppression_reasons_top": [
                {"reason": k, "count": v}
                for k, v in Counter(
                    str(r.get("suppression_reason") or "")
                    for r in csv_rows
                    if str(r.get("suppression_reason") or "") not in ("", "ACTIONABLE")
                ).most_common(12)
            ],
            "notes": notes
            + [
                "Thresholds: generate_signals uses ±0.20% delta for BUY/SELL labels; lifecycle uses separate buy/add/exit/trim deltas (see lifecycle_config).",
            ],
            "rows": [],  # large per-ticker in CSV only
        }

        _write_all(payload, csv_rows)
    except Exception as e:
        try:
            payload = {
                "timestamp": _utc_iso(),
                "ticker_count": 0,
                "notes": [f"diagnostics_error: {e}"],
                "rows": [],
            }
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass


def _write_all(payload: Dict[str, Any], csv_rows: List[Dict[str, Any]]) -> None:
    try:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        if csv_rows:
            keys = list(csv_rows[0].keys())
            with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in csv_rows:
                    w.writerow(row)
    except Exception:
        pass
    try:
        log_row = {
            "ts_utc": payload.get("timestamp", _utc_iso()),
            "ticker_count": payload.get("ticker_count", ""),
            "actionable": payload.get("final_counts", {}).get("actionable_opportunities", ""),
            "wait_suppressed": payload.get("filters", {}).get("lifecycle_wait_suppressed", ""),
            "hold_suppressed": payload.get("filters", {}).get("lifecycle_hold_suppressed", ""),
        }
        newf = not OUT_LOG.exists() or OUT_LOG.stat().st_size == 0
        with OUT_LOG.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log_row.keys()))
            if newf:
                w.writeheader()
            w.writerow(log_row)
    except Exception:
        pass


def _write_minimal(notes: List[str]) -> None:
    p = {
        "timestamp": _utc_iso(),
        "ticker_count": 0,
        "notes": notes,
        "filters": {},
        "final_counts": {},
    }
    try:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(p, indent=2), encoding="utf-8")
    except Exception:
        pass
