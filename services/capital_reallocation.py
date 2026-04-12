# services/capital_reallocation.py
"""Exit → entry bridge: estimate freed capital, rank BUY opportunities, write plan; optional execute_trades hook."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from services.adaptive_position_sizing import (
    compute_size_factor_breakdown,
    max_order_notional_usd,
)
from services.portfolio_correlation import batch_correlation_scores
from services.portfolio_allocation_optimizer import (
    normalize_portfolio_notionals,
    volatilities_from_row_dicts,
)
from services.regime_portfolio_control import apply_regime_max_weight_scale, detect_market_regime

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CAPITAL_JSON = RESULTS / "capital_reallocation.json"
REALLOCATION_CSV = RESULTS / "reallocation_plan.csv"
OPPS_PATH = RESULTS / "trade_opportunities.csv"
LIVE_LOG = RESULTS / "live_orders_log.csv"
CONFIG_PATH = ROOT / "config" / "reallocation.json"
EXEC_GUARD_CONFIG = ROOT / "config" / "execution_guard.json"
POSITIONS_SNAPSHOT_PATH = RESULTS / "positions_snapshot.csv"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _sym_key(sym: str) -> str:
    """Normalize ticker for matching (BRK-B vs BRK.B)."""
    return _norm_sym(sym).replace("-", ".")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def _is_add_intent(r: pd.Series) -> bool:
    es = str(r.get("effective_stance") or "").strip().upper()
    ot = str(r.get("opportunity_type") or "").strip().upper()
    return es == "ADD" or ot == "ADD"


def load_reallocation_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "enabled": True,
        "min_freed_capital": 100.0,
        "max_new_positions_per_cycle": 5,
        "allocation_mode": "proportional",
        "prioritize_confidence": True,
        "dry_run_default": True,
        "max_portfolio_positions": None,
        "adaptive_sizing_enabled": True,
        "size_factor_min": 0.5,
        "size_factor_max": 1.5,
        "delta_pct_boost_scale": 50.0,
        "volatility_shrink_k": 0.25,
        "max_order_notional_usd": None,
        "size_factor_final_min": 0.3,
        "size_factor_final_max": 1.5,
        "volatility_impact_strength": 3.0,
        "use_quadratic_vol": True,
        "quadratic_vol_scale": 10.0,
        "vol_adjustment_floor": 0.3,
        "correlation_filter_enabled": True,
        "correlation_high_threshold": 0.7,
        "correlation_medium_threshold": 0.5,
        "correlation_penalty_high": 0.5,
        "correlation_penalty_medium": 0.75,
        "correlation_penalty_low": 1.0,
        "correlation_lookback_days": 120,
        "correlation_min_overlap_days": 20,
        "correlation_sector_proxy_fallback": True,
        "portfolio_optimizer_enabled": True,
        "max_position_weight_pct": 0.35,
        "min_position_weight_pct": 0.0,
        "min_diversification_enforce": False,
        "risk_parity_enabled": False,
        "risk_parity_strength": 0.5,
        "risk_parity_vol_floor": 0.01,
        "regime_portfolio_control_enabled": True,
        "regime_vix_risk_off": 25.0,
        "regime_vix_risk_on": 16.0,
        "regime_spy_dd_risk_off": -0.08,
        "regime_spy_dd_risk_on": -0.02,
        "regime_atr_risk_off_pct": 0.022,
        "regime_atr_risk_on_pct": 0.012,
        "regime_risk_off_exposure": 0.6,
        "regime_neutral_exposure": 0.85,
        "regime_risk_on_exposure": 1.0,
        "regime_reduce_max_weight_in_risk_off": True,
        "regime_risk_off_max_weight_scale": 0.85,
        "reallocation_relax_correlation": True,
        "reallocation_relax_confidence_boost": True,
        "reallocation_confidence_boost_mult": 1.25,
        "fallback_eligible_count": 3,
    }
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


def _load_max_positions(rcfg: Dict[str, Any]) -> int:
    o = rcfg.get("max_portfolio_positions")
    if o is not None:
        try:
            return max(1, int(o))
        except Exception:
            pass
    try:
        if EXEC_GUARD_CONFIG.is_file():
            u = json.loads(EXEC_GUARD_CONFIG.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict) and u.get("max_positions") is not None:
                return max(1, int(u.get("max_positions", 25)))
    except Exception:
        pass
    return 25


def load_portfolio_state(broker: Any, snapshot_path: Path) -> Tuple[Set[str], int, float]:
    """
    Symbols as sym_key set, position count (long qty>0), total exposure (sum market_value).
    Prefers live broker positions; falls back to positions_snapshot.csv.
    """
    symbols: Set[str] = set()
    exposure = 0.0

    if broker is not None:
        try:
            raw = broker.get_positions() or []
            for p in raw:
                sym = _norm_sym(p.get("symbol"))
                if not sym:
                    continue
                q = _safe_float(p.get("qty") or p.get("quantity"), 0.0)
                if q <= 1e-9:
                    continue
                sd = str(p.get("side") or "long").lower()
                if sd not in ("long", ""):
                    continue
                symbols.add(_sym_key(sym))
                mv = _safe_float(p.get("market_value"), 0.0)
                if mv > 0:
                    exposure += mv
                else:
                    cp = _safe_float(p.get("current_price"), 0.0)
                    if q > 0 and cp > 0:
                        exposure += q * cp
        except Exception:
            pass

    if not symbols and snapshot_path.is_file():
        try:
            df = pd.read_csv(snapshot_path)
            df.columns = [str(c).strip() for c in df.columns]
            tc = (
                "ticker"
                if "ticker" in df.columns
                else ("symbol" if "symbol" in df.columns else None)
            )
            if tc:
                for _, r in df.iterrows():
                    sym = _norm_sym(r.get(tc))
                    if not sym:
                        continue
                    q = _safe_float(r.get("qty") or r.get("qty_available"), 0.0)
                    if q <= 1e-9:
                        continue
                    sd = str(r.get("side") or "long").lower()
                    if sd not in ("long", ""):
                        continue
                    symbols.add(_sym_key(sym))
                    mv = _safe_float(r.get("market_value"), 0.0)
                    if mv <= 0:
                        mv = _safe_float(r.get("value"), 0.0)
                    exposure += mv
        except Exception:
            pass

    return symbols, len(symbols), float(exposure)


def _annotate_eligibility(
    buy_df: pd.DataFrame,
    pos_keys: set,
    pos_count: int,
    max_pos: int,
    exit_symbols: List[str],
) -> pd.DataFrame:
    """Add _eligible (bool) and _exclusion_reason (str) columns."""
    out = buy_df.copy()
    exit_keys = {_sym_key(x) for x in exit_symbols}
    elig: List[bool] = []
    reasons: List[str] = []
    for _, r in out.iterrows():
        sym = _norm_sym(r.get("ticker"))
        sk = _sym_key(sym)
        is_add = _is_add_intent(r)
        if sk in pos_keys and sk not in exit_keys and not is_add:
            elig.append(False)
            reasons.append("ALREADY_IN_PORTFOLIO")
        elif pos_count >= max_pos and sk not in exit_keys and not is_add:
            elig.append(False)
            reasons.append("MAX_POSITIONS_REACHED")
        else:
            elig.append(True)
            reasons.append("PASSED")
    out["_eligible"] = elig
    out["_exclusion_reason"] = reasons
    return out


def _log_eligibility_rows(buy_df: pd.DataFrame, *, only_rejected: bool = True) -> None:
    for _, r in buy_df.iterrows():
        sym = _norm_sym(r.get("ticker"))
        if not sym:
            continue
        ok = bool(r.get("_eligible"))
        reason = str(r.get("_exclusion_reason", ""))
        if only_rejected and ok:
            continue
        if not ok:
            print(f"[FILTER] {sym} rejected → reason={reason}", flush=True)


def _log_opportunity_filter_drops(opps: pd.DataFrame, buy_df: pd.DataFrame) -> None:
    if opps is None or opps.empty:
        return
    o = opps.copy()
    o.columns = [str(c).strip() for c in o.columns]
    if "ticker" not in o.columns:
        return
    kept = (
        {_sym_key(_norm_sym(x)) for x in buy_df["ticker"].tolist()} if not buy_df.empty else set()
    )
    for _, r in o.iterrows():
        sym = _norm_sym(r.get("ticker"))
        sk = _sym_key(sym)
        if not sk:
            continue
        if sk not in kept:
            print(f"[FILTER] {sym} rejected → reason=NOT_BUY_OPPORTUNITY_STANCE", flush=True)


def _apply_fallback_eligibility(
    buy_df: pd.DataFrame,
    cfg: Dict[str, Any],
    pos_keys: set,
    pos_count: int,
    max_pos: int,
    exit_symbols: List[str],
) -> pd.DataFrame:
    """If no eligible rows, force top N by confidence/delta_pct when portfolio rules allow."""
    if buy_df.empty:
        return buy_df
    if int(buy_df["_eligible"].sum()) > 0:
        return buy_df
    exit_keys = {_sym_key(x) for x in exit_symbols}
    ranked = _sort_opps(buy_df, bool(cfg.get("prioritize_confidence", True)))
    k = max(2, min(int(cfg.get("fallback_eligible_count", 3)), len(ranked)))
    out = buy_df.copy()
    forced = 0
    for i in range(len(ranked)):
        if forced >= k:
            break
        r = ranked.iloc[i]
        sym = _norm_sym(r.get("ticker"))
        sk = _sym_key(sym)
        is_add = _is_add_intent(r)
        if sk in pos_keys and sk not in exit_keys and not is_add:
            continue
        if pos_count >= max_pos and sk not in exit_keys and not is_add:
            continue
        mask = out["ticker"].apply(lambda x, sk=sk: _sym_key(_norm_sym(x)) == sk)
        if mask.any():
            out.loc[mask, "_eligible"] = True
            out.loc[mask, "_exclusion_reason"] = "FALLBACK_PRIORITY_SCORE"
            forced += 1
            print(f"[FILTER] {sym} promoted → reason=FALLBACK_PRIORITY_SCORE", flush=True)
    return out


def _planned_freed_and_symbols(
    planned: Sequence[Any],
) -> Tuple[float, List[str], List[str]]:
    total = 0.0
    ex: List[str] = []
    tr: List[str] = []
    for p in planned:
        try:
            n = float(getattr(p, "planned_notional", 0.0) or 0.0)
        except Exception:
            n = 0.0
        total += n
        act = str(getattr(p, "management_action", "") or "").upper()
        sym = _norm_sym(getattr(p, "symbol", ""))
        if not sym:
            continue
        if act in ("EXIT", "ROTATE_EXIT"):
            ex.append(sym)
        elif act == "TRIM":
            tr.append(sym)
    return total, ex, tr


def estimate_freed_capital_from_log(
    log_path: Path,
    session: str,
    planned_fallback: float,
) -> Tuple[float, str]:
    """
    Sum sell fill notionals for the given session (last row per order_id with status filled).
    Falls back to planned_fallback when log missing or sum is zero.
    """
    if not session or not log_path.is_file():
        return planned_fallback, "planned_notional_fallback"

    try:
        df = pd.read_csv(log_path, on_bad_lines="skip")
    except Exception:
        return planned_fallback, "planned_notional_fallback"

    if df is None or df.empty:
        return planned_fallback, "planned_notional_fallback"

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    need = {"session", "side", "order_id", "status", "filled_qty", "filled_avg_price"}
    if not need.issubset(set(df.columns)):
        return planned_fallback, "planned_notional_fallback"

    sub = df[df["session"].astype(str) == session]
    if sub.empty:
        sub = df[df["session"].astype(str).str.contains(session, na=False)]
    sub = sub[sub["side"].astype(str).str.lower() == "sell"]
    sub = sub[sub["order_id"].astype(str).str.strip() != ""]
    if sub.empty:
        return planned_fallback, "planned_notional_fallback"

    try:
        sub = sub.sort_values("timestamp" if "timestamp" in sub.columns else sub.columns[0])
    except Exception:
        pass

    last = sub.groupby("order_id", as_index=False).last()
    last = last[last["status"].astype(str).str.lower() == "filled"]
    total = 0.0
    for _, r in last.iterrows():
        fq = _safe_float(r.get("filled_qty"), 0.0)
        fp = _safe_float(r.get("filled_avg_price"), 0.0)
        if fq > 0 and fp > 0:
            total += fq * fp
        else:
            lp = _safe_float(r.get("limit_price"), 0.0)
            if fq > 0 and lp > 0:
                total += fq * lp

    if total <= 0:
        return planned_fallback, "planned_notional_fallback"
    return float(total), "live_orders_log"


def filter_buy_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "ticker" not in out.columns:
        return pd.DataFrame()

    def row_ok(r: pd.Series) -> bool:
        es = str(r.get("effective_stance") or "").strip().upper()
        ot = str(r.get("opportunity_type") or "").strip().upper()
        if es in ("BUY", "ADD"):
            return True
        if ot in ("ENTRY", "ADD", "BUY"):
            return True
        return False

    mask = out.apply(row_ok, axis=1)
    out = out[mask].copy()
    return out


def _sort_opps(df: pd.DataFrame, prioritize_confidence: bool) -> pd.DataFrame:
    out = df.copy()
    out["_conf"] = out["confidence"].map(_safe_float) if "confidence" in out.columns else 0.0
    out["_dpc"] = out["delta_pct"].map(_safe_float) if "delta_pct" in out.columns else 0.0
    if "fundamental_score" in out.columns:
        out["_fs"] = out["fundamental_score"].map(_safe_float)
    else:
        out["_fs"] = 0.0
    asc = [False, False, False]
    keys = ["_conf", "_dpc", "_fs"]
    if not prioritize_confidence:
        keys = ["_dpc", "_conf", "_fs"]
    return out.sort_values(keys, ascending=asc).reset_index(drop=True)


def _apply_relaxed_correlation_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Raise correlation bands slightly so moderate correlation is not over-penalized."""
    c = dict(cfg)
    if not c.get("reallocation_relax_correlation", True):
        return c
    hi = float(c.get("correlation_high_threshold", 0.7))
    med = float(c.get("correlation_medium_threshold", 0.5))
    c["correlation_high_threshold"] = min(0.99, hi + 0.12)
    c["correlation_medium_threshold"] = min(0.99, med + 0.12)
    try:
        c["correlation_penalty_medium"] = min(
            1.0, float(c.get("correlation_penalty_medium", 0.75)) + 0.1
        )
    except Exception:
        pass
    return c


def _boost_confidence_for_row(conf: float, cfg: Dict[str, Any]) -> float:
    if not cfg.get("reallocation_relax_confidence_boost", True):
        return conf
    try:
        m = float(cfg.get("reallocation_confidence_boost_mult", 1.25))
    except Exception:
        m = 1.25
    return min(1.0, float(conf) * m)


def _apply_correlation_penalty_to_size(
    sf_base: float,
    corr_penalty: float,
    cfg: Dict[str, Any],
) -> float:
    lo = float(cfg.get("size_factor_final_min", 0.3))
    hi = float(cfg.get("size_factor_final_max", 1.5))
    return max(lo, min(hi, float(sf_base) * float(corr_penalty)))


def build_reallocation_rows(
    opps: pd.DataFrame,
    freed_capital: float,
    cfg: Dict[str, Any],
    corr_map: Dict[str, Tuple[float, float]],
    regime_label: str,
    regime_exposure_multiplier: float,
) -> List[Dict[str, Any]]:
    mode = str(cfg.get("allocation_mode") or "proportional").lower()
    max_n = max(1, int(cfg.get("max_new_positions_per_cycle", 5)))
    pri_conf = bool(cfg.get("prioritize_confidence", True))

    ranked = _sort_opps(opps, pri_conf)
    if ranked.empty or freed_capital <= 0:
        return []

    take = ranked.head(max_n)
    n = len(take)
    if n == 0:
        return []

    max_not = max_order_notional_usd(cfg)
    rows: List[Dict[str, Any]] = []

    if mode == "equal":
        per = freed_capital / float(n)
        frac = 1.0 / float(n)
        for i, (_, r) in enumerate(take.iterrows()):
            sym = _norm_sym(r.get("ticker"))
            sk = _sym_key(sym)
            cscore, cpen = corr_map.get(sk, (0.0, 1.0))
            conf = _boost_confidence_for_row(_safe_float(r.get("confidence"), 0.0), cfg)
            dpc = _safe_float(r.get("delta_pct"), 0.0)
            rr = r.copy()
            rr["confidence"] = conf
            bd = compute_size_factor_breakdown(conf, dpc, rr, cfg=cfg)
            sf = _apply_correlation_penalty_to_size(float(bd["size_factor_final"]), cpen, cfg)
            est = per
            adj = min(est * sf, max_not)
            vu = bd["volatility_used"]
            rows.append(
                {
                    "symbol": sym,
                    "recommended_action": "BUY",
                    "priority_rank": i + 1,
                    "confidence": _safe_float(r.get("confidence"), 0.0),
                    "delta_pct": dpc,
                    "estimated_notional": round(est, 2),
                    "size_factor": round(float(bd["size_factor_confidence"]), 4),
                    "volatility_used": vu if vu is not None else "",
                    "vol_adjustment": round(float(bd["vol_adjustment"]), 4),
                    "size_factor_final": round(sf, 4),
                    "correlation_score": cscore,
                    "correlation_penalty": cpen,
                    "adjusted_notional": round(adj, 2),
                    "normalized_notional": "",
                    "portfolio_weight": "",
                    "regime_label": regime_label,
                    "regime_exposure_multiplier": round(float(regime_exposure_multiplier), 4),
                    "allocation_fraction": round(frac, 6),
                    "selected": True,
                }
            )
    else:
        weights: List[float] = []
        for _, r in take.iterrows():
            conf_raw = _safe_float(r.get("confidence"), 0.5)
            conf = _boost_confidence_for_row(conf_raw, cfg)
            w = max(0.01, conf)
            weights.append(w)
        s = sum(weights) or 1.0
        for i, ((_, r), w) in enumerate(zip(take.iterrows(), weights)):
            sym = _norm_sym(r.get("ticker"))
            sk = _sym_key(sym)
            cscore, cpen = corr_map.get(sk, (0.0, 1.0))
            conf = _boost_confidence_for_row(_safe_float(r.get("confidence"), 0.0), cfg)
            dpc = _safe_float(r.get("delta_pct"), 0.0)
            rr = r.copy()
            rr["confidence"] = conf
            bd = compute_size_factor_breakdown(conf, dpc, rr, cfg=cfg)
            sf = _apply_correlation_penalty_to_size(float(bd["size_factor_final"]), cpen, cfg)
            frac = w / s
            est = freed_capital * frac
            adj = min(est * sf, max_not)
            vu = bd["volatility_used"]
            rows.append(
                {
                    "symbol": sym,
                    "recommended_action": "BUY",
                    "priority_rank": i + 1,
                    "confidence": _safe_float(r.get("confidence"), 0.0),
                    "delta_pct": dpc,
                    "estimated_notional": round(est, 2),
                    "size_factor": round(float(bd["size_factor_confidence"]), 4),
                    "volatility_used": vu if vu is not None else "",
                    "vol_adjustment": round(float(bd["vol_adjustment"]), 4),
                    "size_factor_final": round(sf, 4),
                    "correlation_score": cscore,
                    "correlation_penalty": cpen,
                    "adjusted_notional": round(adj, 2),
                    "normalized_notional": "",
                    "portfolio_weight": "",
                    "regime_label": regime_label,
                    "regime_exposure_multiplier": round(float(regime_exposure_multiplier), 4),
                    "allocation_fraction": round(frac, 6),
                    "selected": True,
                }
            )

    total_adj = sum(float(r["adjusted_notional"]) for r in rows)
    if total_adj > freed_capital + 1e-6 and total_adj > 0:
        sc = freed_capital / total_adj
        for r in rows:
            r["adjusted_notional"] = round(float(r["adjusted_notional"]) * sc, 2)

    budget = sum(float(r["adjusted_notional"]) for r in rows)
    vols = volatilities_from_row_dicts(rows)
    nn, pwpct = normalize_portfolio_notionals(
        [float(r["adjusted_notional"]) for r in rows],
        vols,
        budget,
        cfg,
    )
    for i, r in enumerate(rows):
        r["normalized_notional"] = nn[i] if i < len(nn) else round(budget / max(len(rows), 1), 2)
        r["portfolio_weight"] = pwpct[i] if i < len(pwpct) else 0.0
    return rows


def build_full_reallocation_plan(
    buy_df: pd.DataFrame,
    freed_capital: float,
    cfg: Dict[str, Any],
    pos_keys: Set[str],
    regime_label: str,
    regime_exposure_multiplier: float,
) -> List[Dict[str, Any]]:
    """All BUY candidates with eligible/exclusion_reason; selected True for top allocations (eligible only)."""
    pri_conf = bool(cfg.get("prioritize_confidence", True))
    ranked_all = _sort_opps(buy_df, pri_conf)
    eligible_df = buy_df[buy_df["_eligible"]].copy()
    ranked_eligible = _sort_opps(eligible_df, pri_conf) if not eligible_df.empty else pd.DataFrame()

    uniq_keys = sorted({_sym_key(_norm_sym(r.get("ticker"))) for _, r in ranked_all.iterrows()})
    corr_cfg = _apply_relaxed_correlation_cfg(cfg)
    corr_map = batch_correlation_scores(uniq_keys, pos_keys, corr_cfg)

    alloc_list: List[Dict[str, Any]] = []
    if freed_capital > 0 and not ranked_eligible.empty:
        alloc_list = build_reallocation_rows(
            ranked_eligible,
            freed_capital,
            cfg,
            corr_map,
            regime_label,
            regime_exposure_multiplier,
        )

    alloc_by_key = {_sym_key(a["symbol"]): a for a in alloc_list}

    rows: List[Dict[str, Any]] = []
    for _, r in ranked_all.iterrows():
        sym = _norm_sym(r.get("ticker"))
        sk = _sym_key(sym)
        elig = bool(r["_eligible"])
        excl = str(r["_exclusion_reason"])
        cs, cp = corr_map.get(sk, (0.0, 1.0))
        base: Dict[str, Any] = {
            "symbol": sym,
            "recommended_action": "BUY",
            "priority_rank": "",
            "confidence": _safe_float(r.get("confidence"), 0.0),
            "delta_pct": _safe_float(r.get("delta_pct"), 0.0),
            "estimated_notional": "",
            "size_factor": "",
            "volatility_used": "",
            "vol_adjustment": "",
            "size_factor_final": "",
            "correlation_score": cs,
            "correlation_penalty": cp,
            "adjusted_notional": "",
            "normalized_notional": "",
            "portfolio_weight": "",
            "regime_label": regime_label,
            "regime_exposure_multiplier": round(float(regime_exposure_multiplier), 4),
            "allocation_fraction": "",
            "eligible": elig,
            "exclusion_reason": excl,
            "selected": False,
        }
        if elig and sk in alloc_by_key:
            a = alloc_by_key[sk]
            base.update(
                {
                    "priority_rank": a["priority_rank"],
                    "estimated_notional": a["estimated_notional"],
                    "size_factor": a.get("size_factor", ""),
                    "volatility_used": a.get("volatility_used", ""),
                    "vol_adjustment": a.get("vol_adjustment", ""),
                    "size_factor_final": a.get("size_factor_final", ""),
                    "correlation_score": a.get("correlation_score", cs),
                    "correlation_penalty": a.get("correlation_penalty", cp),
                    "adjusted_notional": a.get("adjusted_notional", ""),
                    "normalized_notional": a.get("normalized_notional", ""),
                    "portfolio_weight": a.get("portfolio_weight", ""),
                    "regime_label": a.get("regime_label", regime_label),
                    "regime_exposure_multiplier": a.get(
                        "regime_exposure_multiplier", regime_exposure_multiplier
                    ),
                    "allocation_fraction": a["allocation_fraction"],
                    "selected": True,
                }
            )
        rows.append(base)
    return rows


def build_annotated_plan_below_min_freed(
    buy_df: pd.DataFrame,
    cfg: Dict[str, Any],
    pos_keys: Set[str],
    regime_label: str,
    regime_exposure_multiplier: float,
) -> List[Dict[str, Any]]:
    """No notional allocation; still emit eligibility and correlation columns."""
    pri_conf = bool(cfg.get("prioritize_confidence", True))
    ranked_all = _sort_opps(buy_df, pri_conf)
    uniq_keys = sorted({_sym_key(_norm_sym(r.get("ticker"))) for _, r in ranked_all.iterrows()})
    corr_cfg = _apply_relaxed_correlation_cfg(cfg)
    corr_map = batch_correlation_scores(uniq_keys, pos_keys, corr_cfg)
    rows: List[Dict[str, Any]] = []
    for _, r in ranked_all.iterrows():
        sym = _norm_sym(r.get("ticker"))
        sk = _sym_key(sym)
        cs, cp = corr_map.get(sk, (0.0, 1.0))
        rows.append(
            {
                "symbol": sym,
                "recommended_action": "BUY",
                "priority_rank": "",
                "confidence": _safe_float(r.get("confidence"), 0.0),
                "delta_pct": _safe_float(r.get("delta_pct"), 0.0),
                "estimated_notional": "",
                "size_factor": "",
                "volatility_used": "",
                "vol_adjustment": "",
                "size_factor_final": "",
                "correlation_score": cs,
                "correlation_penalty": cp,
                "adjusted_notional": "",
                "normalized_notional": "",
                "portfolio_weight": "",
                "regime_label": regime_label,
                "regime_exposure_multiplier": round(float(regime_exposure_multiplier), 4),
                "allocation_fraction": "",
                "eligible": bool(r["_eligible"]),
                "exclusion_reason": str(r["_exclusion_reason"]),
                "selected": False,
            }
        )
    return rows


def write_reallocation_artifacts(
    payload: Dict[str, Any],
    plan_rows: List[Dict[str, Any]],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with CAPITAL_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if plan_rows:
        pd.DataFrame(plan_rows).to_csv(REALLOCATION_CSV, index=False)
    else:
        pd.DataFrame(
            columns=[
                "symbol",
                "recommended_action",
                "priority_rank",
                "confidence",
                "delta_pct",
                "estimated_notional",
                "size_factor",
                "volatility_used",
                "vol_adjustment",
                "size_factor_final",
                "correlation_score",
                "correlation_penalty",
                "adjusted_notional",
                "normalized_notional",
                "portfolio_weight",
                "regime_label",
                "regime_exposure_multiplier",
                "allocation_fraction",
                "eligible",
                "exclusion_reason",
                "selected",
            ]
        ).to_csv(REALLOCATION_CSV, index=False)


def run_execute_trades_bridge(
    mode: str,
    max_orders: int,
    verbose: bool,
    ignore_market_closed: bool,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "services.execute_trades",
        "--mode",
        mode,
        "--execute",
        "--max-orders",
        str(max_orders),
    ]
    if verbose:
        cmd.append("--verbose")
    if ignore_market_closed:
        cmd.append("--ignore-market-closed")
    try:
        return int(subprocess.call(cmd, cwd=str(ROOT)))
    except Exception as e:
        print(f"[capital_reallocation] execute_trades bridge failed: {e}")
        return 1


def run_reallocation_pipeline(
    planned: Sequence[Any],
    *,
    mode: str,
    manage_executed: bool,
    manage_session: str,
    placement_rc: int,
    broker: Any,
    verbose: bool,
    reallocate_after_exit: bool,
    ignore_market_closed: bool,
    positions_snapshot_path: Optional[Path] = None,
) -> None:
    rcfg = load_reallocation_config()
    if not rcfg.get("enabled", True):
        return

    regime_info = detect_market_regime(rcfg)
    regime_label = str(regime_info.get("regime_label", "NEUTRAL"))
    regime_mult = float(regime_info.get("regime_exposure_multiplier", 1.0))
    rcfg_regime = apply_regime_max_weight_scale(rcfg, regime_label)

    planned_total, sym_ex, sym_tr = _planned_freed_and_symbols(planned)
    method = "planned_notional_estimate"
    freed = planned_total

    if manage_executed and placement_rc == 0 and manage_session:
        fc, method = estimate_freed_capital_from_log(LIVE_LOG, manage_session, planned_total)
        freed = fc

    ps_path = positions_snapshot_path or POSITIONS_SNAPSHOT_PATH
    pos_keys, pos_count, total_exp = load_portfolio_state(broker, ps_path)
    max_pos = _load_max_positions(rcfg)

    opps = pd.DataFrame()
    if OPPS_PATH.is_file():
        try:
            opps = pd.read_csv(OPPS_PATH)
        except Exception:
            opps = pd.DataFrame()

    buy_df = filter_buy_opportunities(opps)
    n_cand = len(buy_df)
    _log_opportunity_filter_drops(opps, buy_df)

    if not buy_df.empty:
        buy_df = _annotate_eligibility(buy_df, pos_keys, pos_count, max_pos, sym_ex)
        _log_eligibility_rows(buy_df, only_rejected=True)
        buy_df = _apply_fallback_eligibility(
            buy_df, rcfg_regime, pos_keys, pos_count, max_pos, sym_ex
        )

    eligible_count = (
        int(buy_df["_eligible"].sum()) if not buy_df.empty and "_eligible" in buy_df.columns else 0
    )
    filtered_out = n_cand - eligible_count

    min_fc = float(rcfg.get("min_freed_capital", 100.0))
    plan_rows: List[Dict[str, Any]] = []
    selected_syms: List[str] = []

    preview_top: List[Dict[str, Any]] = []
    if not buy_df.empty:
        elig_df = buy_df[buy_df["_eligible"]].copy()
        if not elig_df.empty:
            pr = _sort_opps(elig_df, bool(rcfg.get("prioritize_confidence", True))).head(5)
            for _, r in pr.iterrows():
                preview_top.append(
                    {
                        "symbol": _norm_sym(r.get("ticker")),
                        "confidence": _safe_float(r.get("confidence"), 0.0),
                        "delta_pct": _safe_float(r.get("delta_pct"), 0.0),
                    }
                )

    effective_freed = max(0.0, float(freed) * regime_mult)
    if not buy_df.empty:
        if freed >= min_fc:
            plan_rows = build_full_reallocation_plan(
                buy_df,
                effective_freed,
                rcfg_regime,
                pos_keys,
                regime_label,
                regime_mult,
            )
        else:
            plan_rows = build_annotated_plan_below_min_freed(
                buy_df,
                rcfg_regime,
                pos_keys,
                regime_label,
                regime_mult,
            )
        selected_syms = [str(r["symbol"]) for r in plan_rows if r.get("selected")]

    sym_list = sorted(pos_keys)

    payload: Dict[str, Any] = {
        "timestamp": _utc_iso(),
        "freed_capital": round(float(freed), 2),
        "freed_capital_method": method,
        "symbols_exited": sym_ex,
        "symbols_trimmed": sym_tr,
        "source_session": manage_session or "",
        "n_planned_management_orders": len(planned),
        "n_exits": len(sym_ex),
        "n_trims": len(sym_tr),
        "n_buy_candidates": n_cand,
        "eligible_candidates": eligible_count,
        "filtered_out": filtered_out,
        "current_positions_count": pos_count,
        "current_total_exposure": round(float(total_exp), 2),
        "current_symbols": sym_list,
        "max_positions": max_pos,
        "selected_symbols": selected_syms,
        "manage_placement_rc": placement_rc,
        "regime_portfolio": {
            "label": regime_label,
            "exposure_multiplier": regime_mult,
            "effective_freed_capital": (
                round(float(effective_freed), 2) if buy_df is not None and not buy_df.empty else 0.0
            ),
            "vix_last": regime_info.get("vix_last"),
            "spy_drawdown_pct": regime_info.get("spy_drawdown_pct"),
            "spy_atr_pct": regime_info.get("spy_atr_pct"),
            "source": regime_info.get("source", ""),
            "max_position_weight_pct_effective": rcfg_regime.get("max_position_weight_pct"),
        },
        "reallocation_config": {
            "min_freed_capital": min_fc,
            "max_new_positions_per_cycle": rcfg.get("max_new_positions_per_cycle"),
            "allocation_mode": rcfg.get("allocation_mode"),
            "correlation_filter_enabled": rcfg.get("correlation_filter_enabled", True),
            "correlation_high_threshold": rcfg.get("correlation_high_threshold", 0.7),
            "correlation_medium_threshold": rcfg.get("correlation_medium_threshold", 0.5),
            "portfolio_optimizer_enabled": rcfg.get("portfolio_optimizer_enabled", True),
            "max_position_weight_pct": rcfg.get("max_position_weight_pct", 0.35),
            "risk_parity_enabled": rcfg.get("risk_parity_enabled", False),
        },
        "preview_top_buy_opportunities": preview_top,
    }

    write_reallocation_artifacts(payload, plan_rows)

    if verbose:
        print(
            f"[capital_reallocation] freed_capital={payload['freed_capital']} ({method}) "
            f"regime={regime_label} exposure_x={regime_mult} effective_freed={payload['regime_portfolio'].get('effective_freed_capital')} "
            f"candidates={n_cand} eligible={eligible_count} filtered_out={filtered_out} "
            f"selected={len(selected_syms)} positions={pos_count}/{max_pos}"
        )

    if not reallocate_after_exit:
        return
    if mode != "paper":
        print("[capital_reallocation] --reallocate-after-exit is paper-only; skipping bridge.")
        return
    if not manage_executed or placement_rc != 0:
        return
    if freed < min_fc:
        return

    try:
        from services.master_execution_gate import (
            MasterExecutionGate,
            append_gate_log_csv,
            write_snapshot,
        )

        gate = MasterExecutionGate(project_root=ROOT)
        dec = gate.evaluate(
            mode=mode,
            broker=broker,
            verbose=verbose,
            require_market_open=(False if ignore_market_closed else None),
        )
        write_snapshot(dec)
        append_gate_log_csv(dec)
        if not dec.ok:
            print(f"[capital_reallocation] bridge blocked (master gate): {dec.reasons}")
            return
    except Exception as e:
        print(f"[capital_reallocation] master gate error: {e}")
        return

    max_n = max(1, int(rcfg.get("max_new_positions_per_cycle", 5)))
    print(f"[capital_reallocation] launching execute_trades bridge max_orders={max_n}")
    run_execute_trades_bridge(mode, max_n, verbose, ignore_market_closed)
