# services/regime_weighting.py
# ------------------------------------------------------------
# TRITON — Regime-Gated Advisor Weighting (Phase 2 / Step 2)
#
# Goal:
#   Turn "rationale" into actionable portfolio intent:
#   - regime-aware weight scaling
#   - confidence-weighted sizing
#   - capital-mode overrides (defensive posture)
#
# Inputs (real data only):
#   - data/results/signals_with_rationale.csv   (required)
#   - data/results/stock_scores.csv            (optional)
#   - data/results/guard_snapshot.json         (optional)
#   - data/results/adaptive_risk_state.json    (optional)
#
# Outputs:
#   - data/results/target_weights.csv
#   - data/results/target_weights.json
#
# Zero impact on execution/backtest pipelines.
# ------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from services.edge_ranking import EnrichmentSpec, enrich_with_edge
from services.portfolio_intelligence import (
    PortfolioIntelligenceConfig,
    apply_portfolio_intelligence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "results"

SIG_RAT_PATH = RESULTS_DIR / "signals_with_rationale.csv"
SCORES_PATH = RESULTS_DIR / "stock_scores.csv"
GUARD_PATH = RESULTS_DIR / "guard_snapshot.json"
RISK_PATH = RESULTS_DIR / "adaptive_risk_state.json"

OUT_CSV = RESULTS_DIR / "target_weights.csv"
OUT_JSON = RESULTS_DIR / "target_weights.json"


# ─────────────────────────────────────────────────────────────
# Config (safe defaults; tune later)
# ─────────────────────────────────────────────────────────────
@dataclass
class WeightingConfig:
    max_names: int = 25  # cap names in portfolio intent
    base_weight: float = 1.0  # starting unit weight before normalization
    min_conf: float = 0.10
    max_conf: float = 0.90

    # Regime multipliers (conservative)
    regime_mult: Dict[str, float] = None

    # Capital mode multipliers (global scaling)
    capital_mode_mult: Dict[str, float] = None

    # Optional score influence (adds quality tilt)
    score_col_candidates: tuple = ("score_total", "score", "total_score")

    # Buy-only intent for now (can expand later)
    buy_signals: tuple = ("BUY", 1, "1")


def default_config() -> WeightingConfig:
    cfg = WeightingConfig()
    cfg.regime_mult = {
        "High Volatility": 0.55,  # cut exposure when vol high
        "Trend Expansion": 1.15,  # slightly increase when trend favorable
        "Neutral / Range": 0.85,
        "": 0.85,
        "UNKNOWN": 0.85,
    }
    cfg.capital_mode_mult = {
        "NORMAL": 1.00,
        "DEFENSIVE": 0.60,
        "RISK_OFF": 0.45,
        "LOCKDOWN": 0.00,  # hard stop
        "UNKNOWN": 0.85,
    }
    return cfg


# ─────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────
def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def safe_score_col(df: pd.DataFrame, candidates: tuple) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_weights(w: pd.Series) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    s = float(w.sum())
    if s <= 0:
        return w * 0.0
    return w / s


# ─────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────
def run(cfg: WeightingConfig) -> None:
    if not SIG_RAT_PATH.exists() or SIG_RAT_PATH.stat().st_size == 0:
        print(f"Missing required: {SIG_RAT_PATH}")
        return

    sig = pd.read_csv(SIG_RAT_PATH)

    if "ticker" not in sig.columns:
        print("signals_with_rationale.csv missing 'ticker'.")
        return

    # signal normalization
    if "signal" in sig.columns:
        sig_signal = sig["signal"]
    else:
        sig_signal = pd.Series(["HOLD"] * len(sig))

    # keep only BUY intent (conservative for now)
    buy_mask = sig_signal.astype(str).str.upper().isin([str(x).upper() for x in cfg.buy_signals])
    sig = sig[buy_mask].copy()

    if sig.empty:
        print("No BUY signals found. Writing empty target_weights.")
        pd.DataFrame(columns=["ticker", "raw_weight", "weight", "reason"]).to_csv(
            OUT_CSV, index=False
        )
        OUT_JSON.write_text("[]", encoding="utf-8")
        print(f"✔ Weights written: {OUT_CSV} (empty)")
        print(f"✔ Weights written: {OUT_JSON} (empty)")
        return

    sig["ticker"] = sig["ticker"].astype(str).str.upper().str.strip()

    # confidence
    if "confidence" in sig.columns:
        conf = pd.to_numeric(sig["confidence"], errors="coerce")
    else:
        conf = pd.Series([0.5] * len(sig))
    conf = conf.clip(cfg.min_conf, cfg.max_conf).fillna(0.5)
    sig["confidence"] = conf

    # regime multiplier
    regime = sig.get("regime", pd.Series(["Neutral / Range"] * len(sig))).astype(str)
    sig["regime_mult"] = regime.map(cfg.regime_mult).fillna(
        cfg.regime_mult.get("Neutral / Range", 0.85)
    )

    # capital mode (from rationale file if present; else from snapshots)
    cap_mode = None
    if "capital_mode" in sig.columns and sig["capital_mode"].notna().any():
        cap_mode = str(sig["capital_mode"].dropna().iloc[-1]).upper().strip()
    else:
        guard = read_json(GUARD_PATH) or {}
        risk = read_json(RISK_PATH) or {}
        cap_mode = (
            str(guard.get("mode") or risk.get("mode") or risk.get("capital_mode") or "NORMAL")
            .upper()
            .strip()
        )

    cap_mult = cfg.capital_mode_mult.get(cap_mode, cfg.capital_mode_mult.get("UNKNOWN", 0.85))

    # base weights: confidence * regime * base
    sig["raw_weight"] = (cfg.base_weight * sig["confidence"] * sig["regime_mult"]).astype(float)

    # optional quality tilt from scores
    if SCORES_PATH.exists() and SCORES_PATH.stat().st_size > 0:
        scores = pd.read_csv(SCORES_PATH)
        if "ticker" in scores.columns:
            scores["ticker"] = scores["ticker"].astype(str).str.upper().str.strip()
            score_col = safe_score_col(scores, cfg.score_col_candidates)
            if score_col:
                scores[score_col] = pd.to_numeric(scores[score_col], errors="coerce")
                # z-score to avoid dominating
                mu = float(scores[score_col].mean()) if scores[score_col].notna().any() else 0.0
                sd = (
                    float(scores[score_col].std(ddof=0)) if scores[score_col].notna().any() else 0.0
                )
                if sd > 0:
                    scores["score_z"] = (scores[score_col] - mu) / sd
                else:
                    scores["score_z"] = 0.0

                sig = sig.merge(scores[["ticker", "score_z"]], on="ticker", how="left")
                sig["score_z"] = sig["score_z"].fillna(0.0)

                # gentle tilt (±10% typical)
                sig["raw_weight"] *= 1.0 + 0.10 * sig["score_z"].clip(-2, 2)

    # cap number of names
    sig = sig.sort_values("raw_weight", ascending=False).head(cfg.max_names).copy()

    # apply capital mode multiplier globally
    sig["raw_weight"] *= float(cap_mult)

    # normalize to weights
    sig["weight"] = normalize_weights(sig["raw_weight"])

    # reason field (audit)
    sig["reason"] = (
        "BUY"
        + " | conf="
        + sig["confidence"].round(2).astype(str)
        + " | regime="
        + sig.get("regime", "—").astype(str)
        + " | cap_mode="
        + cap_mode
    )

    # ─────────────────────────────────────────────────────────────
    # Edge-based ranking and sizing (additive; preserves existing weights).
    # All BUY rows that survived above are sizing-eligible (treated as ENTRY).
    # The legacy `weight` column is kept untouched for backward compatibility.
    # ─────────────────────────────────────────────────────────────
    sizing_input = sig.copy()
    sizing_input["opportunity_type"] = "ENTRY"
    enriched = enrich_with_edge(
        sizing_input,
        EnrichmentSpec(opportunity_col="opportunity_type"),
    )

    sig["edge_score"] = enriched["edge_score"].values
    sig["edge_rank"] = enriched["edge_rank"].values
    sig["edge_percentile"] = enriched["edge_percentile"].values
    sig["sizing_bucket"] = enriched["sizing_bucket"].values
    sig["allocation_multiplier"] = enriched["allocation_multiplier"].values
    sig["allocation_reason"] = enriched["allocation_reason"].values

    sig["allocation_weight_raw"] = (
        pd.to_numeric(sig["raw_weight"], errors="coerce").fillna(0.0)
        * pd.to_numeric(sig["allocation_multiplier"], errors="coerce").fillna(0.0)
    ).astype(float)
    final_alloc = normalize_weights(sig["allocation_weight_raw"])
    # If the edge tilt zeroed out everything (e.g. all FILTERED_LOW_EDGE), fall back
    # to the legacy normalized weight so we never silently produce an empty book.
    if float(final_alloc.sum()) <= 0.0:
        sig["allocation_weight_final"] = sig["weight"].astype(float)
        edge_fallback_used = True
    else:
        sig["allocation_weight_final"] = final_alloc.astype(float)
        edge_fallback_used = False

    # ─────────────────────────────────────────────────────────────
    # Portfolio-aware tilt layer (concentration / sector / add / crowding).
    # All BUY rows here are treated as ENTRY for sizing semantics. Hard risk
    # caps elsewhere remain authoritative — this is only a soft tilt.
    # ─────────────────────────────────────────────────────────────
    pi_input = sig.copy()
    pi_input["opportunity_type"] = "ENTRY"
    pi_out, pi_diag = apply_portfolio_intelligence(
        pi_input,
        PortfolioIntelligenceConfig(),
    )

    # Pull the new portfolio columns back onto sig (additive, no rename/remove).
    pi_cols = [
        "sector_name",
        "existing_position_weight",
        "sector_weight_current",
        "sector_weight_proposed",
        "single_name_over_cap_flag",
        "sector_over_cap_flag",
        "add_overcrowded_flag",
        "concentration_penalty",
        "sector_penalty",
        "add_dampener",
        "crowding_penalty",
        "crowded_group_rank",
        "portfolio_adjustment_factor",
        "portfolio_weight_pre_adjustment",
        "portfolio_weight_post_adjustment",
        "portfolio_adjustment_reason",
        "portfolio_fallback_used",
    ]
    for c in pi_cols:
        if c in pi_out.columns:
            sig[c] = pi_out[c].values

    out = sig[
        [
            "ticker",
            "raw_weight",
            "weight",
            "reason",
            "edge_score",
            "edge_rank",
            "edge_percentile",
            "sizing_bucket",
            "allocation_multiplier",
            "allocation_weight_raw",
            "allocation_weight_final",
            "allocation_reason",
            # Portfolio intelligence (v1) — additive only.
            "sector_name",
            "existing_position_weight",
            "sector_weight_current",
            "sector_weight_proposed",
            "single_name_over_cap_flag",
            "sector_over_cap_flag",
            "add_overcrowded_flag",
            "concentration_penalty",
            "sector_penalty",
            "add_dampener",
            "crowding_penalty",
            "crowded_group_rank",
            "portfolio_adjustment_factor",
            "portfolio_weight_pre_adjustment",
            "portfolio_weight_post_adjustment",
            "portfolio_adjustment_reason",
            "portfolio_fallback_used",
        ]
    ].copy()
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out["capital_mode"] = cap_mode
    out["capital_mult"] = float(cap_mult)
    out["edge_fallback_used"] = bool(edge_fallback_used)
    out["portfolio_intel_diagnostics"] = json.dumps(pi_diag, default=str)

    out.to_csv(OUT_CSV, index=False)
    out.to_json(OUT_JSON, orient="records", indent=2)

    print(f"✔ Weights written: {OUT_CSV}")
    print(f"✔ Weights written: {OUT_JSON}")
    print(f"Capital mode: {cap_mode} (mult={cap_mult})")
    print(f"Names: {len(out)}")


if __name__ == "__main__":
    run(default_config())
