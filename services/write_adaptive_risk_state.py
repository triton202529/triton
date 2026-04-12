# services/write_adaptive_risk_state.py
"""
Writes data/results/adaptive_risk_state.json (authoritative risk state)

Goal:
- Single JSON file the dashboards + execution layer can trust.
- Capital-preservation defaults (conservative).
- Uses saved artifacts only (no broker calls).
- Gracefully degrades if optional artifacts missing.
- Robust JSON reading on Windows (handles UTF-8 BOM / utf-8-sig).

Inputs (best-effort):
- data/results/enhanced_portfolio_history.csv  (preferred)
- data/results/portfolio_history.csv           (fallback)
- data/results/regimes.csv                     (optional)
- data/results/guard_snapshot.json             (optional)
- data/results/risk_report.json                (optional)
- data/results/global_kill_switch.json         (optional)

Output:
- data/results/adaptive_risk_state.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path("data/results")

ENHANCED_PH = RESULTS_DIR / "enhanced_portfolio_history.csv"
PORTFOLIO_PH = RESULTS_DIR / "portfolio_history.csv"
REGIMES_CSV = RESULTS_DIR / "regimes.csv"
GUARD_JSON = RESULTS_DIR / "guard_snapshot.json"
RISK_REPORT_JSON = RESULTS_DIR / "risk_report.json"
KILL_SWITCH_JSON = RESULTS_DIR / "global_kill_switch.json"

OUT_JSON = RESULTS_DIR / "adaptive_risk_state.json"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_err(e: Exception, limit: int = 220) -> str:
    msg = " ".join((str(e) or repr(e)).split())
    return (msg[:limit].rstrip() + "…") if len(msg) > limit else msg


def _read_text_best_effort(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Returns (text, err). Handles UTF-8 BOM by trying utf-8-sig first."""
    if not path.exists() or path.stat().st_size == 0:
        return None, None
    try:
        return path.read_text(encoding="utf-8-sig"), None
    except Exception as e1:
        try:
            return path.read_text(encoding="utf-8"), None
        except Exception as e2:
            return None, _short_err(e2) or _short_err(e1)


def _load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (obj, err). err is None if OK or file missing/empty."""
    text, err = _read_text_best_effort(path)
    if text is None:
        return None, err  # None/None if missing, or (None, err) if read failed
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
        return None, f"JSON root not dict: {path.name}"
    except Exception as e:
        return None, _short_err(e)


def _load_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Returns (df, err). err is None if OK or file missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return None, None
    try:
        return pd.read_csv(path), None
    except Exception as e:
        return None, _short_err(e)


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" not in df.columns:
        for c in ["timestamp", "time", "datetime", "as_of", "Date"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    if "date" not in df.columns:
        df["date"] = pd.NaT
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _to_float(x: Any) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _drawdown_from_equity_curve(df: Optional[pd.DataFrame]) -> Tuple[float, float, float]:
    """
    Returns (latest_equity, peak_equity, drawdown_pct)
    drawdown_pct is negative when below peak.
    """
    if df is None or df.empty:
        return float("nan"), float("nan"), float("nan")

    xdf = df.copy()
    if "total_value" not in xdf.columns:
        for alt in ["equity", "portfolio_value", "value", "total"]:
            if alt in xdf.columns:
                xdf = xdf.rename(columns={alt: "total_value"})
                break

    if "total_value" not in xdf.columns:
        return float("nan"), float("nan"), float("nan")

    s = pd.to_numeric(xdf["total_value"], errors="coerce").dropna()
    if s.empty:
        return float("nan"), float("nan"), float("nan")

    latest = float(s.iloc[-1])
    peak = float(s.max())
    dd = (latest / peak - 1.0) if peak > 0 else float("nan")
    return latest, peak, float(dd)


def _infer_current_regime(ph: Optional[pd.DataFrame], regimes: Optional[pd.DataFrame]) -> str:
    """
    Regime precedence:
    1) portfolio_history has 'regime' column
    2) regimes.csv has 'regime' (or common alias)
    """
    if ph is not None and not ph.empty and "regime" in ph.columns:
        s = ph["regime"].dropna()
        if not s.empty:
            return str(s.iloc[-1])

    if regimes is not None and not regimes.empty:
        r = _ensure_date(regimes)
        if "regime" not in r.columns:
            for alt in ["state", "label", "Regime"]:
                if alt in r.columns:
                    r = r.rename(columns={alt: "regime"})
                    break
        if "regime" in r.columns:
            s = r["regime"].dropna()
            if not s.empty:
                return str(s.iloc[-1])

    return "UNKNOWN"


def _artifact_presence() -> Dict[str, bool]:
    def _has(p: Path) -> bool:
        try:
            return p.exists() and p.stat().st_size > 0
        except Exception:
            return False

    return {
        "has_enhanced": _has(ENHANCED_PH),
        "has_regimes": _has(REGIMES_CSV),
        "has_guard_snapshot": _has(GUARD_JSON),
        "has_risk_report": _has(RISK_REPORT_JSON),
        "has_kill_switch": _has(KILL_SWITCH_JSON),
    }


# ─────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────
def _compute_controls(
    *,
    kill_switch: bool,
    drawdown_pct: float,
    regime: str,
    guard_mode: str,
) -> Dict[str, Any]:
    """
    Conservative capital-preservation policy (safe starter).
    You can tune thresholds later.
    """
    # Defaults (long-only gross = 1.0)
    max_gross_exposure = 1.0
    max_position_weight = 0.10
    allow_new_orders = True
    risk_on = True
    block_reason = ""

    if kill_switch:
        return {
            "global_kill_switch": True,
            "risk_on": False,
            "allow_new_orders": False,
            "max_gross_exposure": 0.0,
            "max_position_weight": 0.0,
            "block_reason": "GLOBAL_KILL_SWITCH",
        }

    # Drawdown guardrails
    if np.isfinite(drawdown_pct) and drawdown_pct <= -0.10:
        max_gross_exposure = 0.50
        max_position_weight = 0.07

    if np.isfinite(drawdown_pct) and drawdown_pct <= -0.20:
        max_gross_exposure = 0.25
        max_position_weight = 0.05
        allow_new_orders = False
        risk_on = False
        block_reason = "DRAWDOWN_GUARD"

    # Regime tightening (safe)
    r = str(regime).strip().lower()
    if r in {"bear", "crash"}:
        max_gross_exposure = min(max_gross_exposure, 0.35)
        max_position_weight = min(max_position_weight, 0.06)
    if r in {"volatile", "high_vol", "high-vol", "highvol"}:
        max_gross_exposure = min(max_gross_exposure, 0.60)
        max_position_weight = min(max_position_weight, 0.08)

    # Guard snapshot override (soft but authoritative)
    m = str(guard_mode).strip().lower()
    if m in {"defensive", "risk_off", "risk-off", "halt"}:
        max_gross_exposure = min(max_gross_exposure, 0.25)
        max_position_weight = min(max_position_weight, 0.05)
        allow_new_orders = False
        risk_on = False
        if not block_reason:
            block_reason = "GUARD_MODE"

    return {
        "global_kill_switch": False,
        "risk_on": bool(risk_on),
        "allow_new_orders": bool(allow_new_orders),
        "max_gross_exposure": float(max_gross_exposure),
        "max_position_weight": float(max_position_weight),
        "block_reason": block_reason or "",
    }


# ─────────────────────────────────────────────────────────────
# Build state
# ─────────────────────────────────────────────────────────────
def build_state(verbose: bool = False) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    presence = _artifact_presence()

    # Portfolio history: prefer enhanced
    ph, ph_err = _load_csv(ENHANCED_PH)
    ph_source = "enhanced_portfolio_history.csv"
    if ph is None or ph.empty:
        ph, ph_err2 = _load_csv(PORTFOLIO_PH)
        if ph is not None and not ph.empty:
            ph_source = "portfolio_history.csv"
        if ph_err is None:
            ph_err = ph_err2

    # Optional artifacts
    regimes, regimes_err = _load_csv(REGIMES_CSV)
    guard, guard_err = _load_json(GUARD_JSON)
    rr, rr_err = _load_json(RISK_REPORT_JSON)
    ks, ks_err = _load_json(KILL_SWITCH_JSON)

    # Normalize portfolio history
    if ph is not None and not ph.empty:
        ph = _ensure_date(ph)
        for c in ["cash", "market_value", "total_value"]:
            if c not in ph.columns:
                ph[c] = np.nan

    latest_equity, peak_equity, drawdown_pct = _drawdown_from_equity_curve(ph)

    # Regime
    current_regime = _infer_current_regime(ph, regimes)

    # Kill switch
    kill_switch = bool(ks.get("global_kill_switch", False)) if isinstance(ks, dict) else False

    # Guard snapshot headlines
    mode = "UNKNOWN"
    reason = ""
    buying_power = float("nan")
    reserve_pct = float("nan")

    if isinstance(guard, dict):
        mode = str(guard.get("mode", mode) or mode)
        reason = str(guard.get("reason", reason) or reason)
        buying_power = _to_float(guard.get("buying_power", buying_power))
        reserve_pct = _to_float(guard.get("reserve_pct", reserve_pct))

    # Risk report headlines
    expected_vol = float("nan")
    risk_adj_return = float("nan")
    diversification = float("nan")

    if isinstance(rr, dict):
        pm = (
            rr.get("portfolio_metrics", {}) if isinstance(rr.get("portfolio_metrics"), dict) else {}
        )
        expected_vol = _to_float(pm.get("expected_volatility", expected_vol))
        risk_adj_return = _to_float(pm.get("risk_adjusted_return", risk_adj_return))
        diversification = _to_float(pm.get("diversification_ratio", diversification))

    # Controls (policy)
    controls = _compute_controls(
        kill_switch=kill_switch,
        drawdown_pct=drawdown_pct,
        regime=current_regime,
        guard_mode=mode,
    )

    # Compose state (matches current schema)
    state: Dict[str, Any] = {
        "timestamp": _now_utc_iso(),
        "source": {
            "portfolio_history": ph_source,
            "has_enhanced": presence["has_enhanced"],
            "has_regimes": presence["has_regimes"],
            "has_guard_snapshot": presence["has_guard_snapshot"],
            "has_risk_report": presence["has_risk_report"],
            "has_kill_switch": presence["has_kill_switch"],
        },
        "regime": current_regime,
        "mode": mode,
        "reason": reason,
        "portfolio": {
            "latest_equity": None if not np.isfinite(latest_equity) else latest_equity,
            "peak_equity": None if not np.isfinite(peak_equity) else peak_equity,
            "drawdown_pct": None if not np.isfinite(drawdown_pct) else drawdown_pct,
        },
        "risk_report_headlines": {
            "expected_volatility": None if not np.isfinite(expected_vol) else expected_vol,
            "risk_adjusted_return": None if not np.isfinite(risk_adj_return) else risk_adj_return,
            "diversification_ratio": None if not np.isfinite(diversification) else diversification,
        },
        "broker": {
            "buying_power": None if not np.isfinite(buying_power) else buying_power,
            "reserve_pct": None if not np.isfinite(reserve_pct) else reserve_pct,
        },
        "controls": controls,
        "diagnostics": {
            "errors": {
                "portfolio_history": ph_err,
                "regimes": regimes_err,
                "guard_snapshot": guard_err,
                "risk_report": rr_err,
                "kill_switch": ks_err,
            }
        },
    }

    if verbose:
        print("✅ adaptive_risk_state assembled:")
        print(json.dumps(state, indent=2))

    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    state = build_state(verbose=args.verbose)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")

    size = OUT_JSON.stat().st_size if OUT_JSON.exists() else 0
    print(f"✅ Wrote {OUT_JSON} ({size} bytes)")


if __name__ == "__main__":
    main()
