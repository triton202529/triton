# services/sector_exposure.py
"""Portfolio sector exposure from positions_snapshot.csv — JSON report + execute_trades concentration guard."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
DEFAULT_POSITIONS_SNAPSHOT = RESULTS / "positions_snapshot.csv"
DEFAULT_OUTPUT_JSON = RESULTS / "sector_exposure.json"

# Default thresholds (fraction of portfolio)
WARN_PCT_DEFAULT = 0.40
CRITICAL_PCT_DEFAULT = 0.60
BLOCK_PCT_DEFAULT = 0.40  # block new BUY/ADD if projected sector pct > this

# Ticker -> sector. ETFs mapped to thematic sector where applicable; unknown -> Other.
_TICKER_SECTOR: Dict[str, str] = {
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "EOG": "Energy",
    "SLB": "Energy",
    "MPC": "Energy",
    "VLO": "Energy",
    "PSX": "Energy",
    "OXY": "Energy",
    "HAL": "Energy",
    "BKR": "Energy",
    "WMB": "Energy",
    "KMI": "Energy",
    "XLE": "Energy",
    "USO": "Energy",
    "UNG": "Energy",
    "XOP": "Energy",
    "IYE": "Energy",
    # Tech
    "AAPL": "Tech",
    "MSFT": "Tech",
    "NVDA": "Tech",
    "AVGO": "Tech",
    "ORCL": "Tech",
    "CRM": "Tech",
    "ADBE": "Tech",
    "CSCO": "Tech",
    "ACN": "Tech",
    "AMD": "Tech",
    "INTC": "Tech",
    "QCOM": "Tech",
    "TXN": "Tech",
    "IBM": "Tech",
    "NOW": "Tech",
    "PANW": "Tech",
    "SNOW": "Tech",
    "CRWD": "Tech",
    "PLTR": "Tech",
    "QQQ": "Tech",
    "XLK": "Tech",
    "SMH": "Tech",
    "SOXX": "Tech",
    "IGV": "Tech",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "SCHW": "Financials",
    "BLK": "Financials",
    "AXP": "Financials",
    "USB": "Financials",
    "PNC": "Financials",
    "TFC": "Financials",
    "BK": "Financials",
    "STT": "Financials",
    "XLF": "Financials",
    "KRE": "Financials",
    "IYF": "Financials",
    # Healthcare
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "LLY": "Healthcare",
    "MRK": "Healthcare",
    "ABBV": "Healthcare",
    "PFE": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "DHR": "Healthcare",
    "BMY": "Healthcare",
    "AMGN": "Healthcare",
    "GILD": "Healthcare",
    "CVS": "Healthcare",
    "CI": "Healthcare",
    "XLV": "Healthcare",
    "IYH": "Healthcare",
    # Consumer Discretionary / Staples
    "AMZN": "Consumer",
    "TSLA": "Consumer",
    "HD": "Consumer",
    "MCD": "Consumer",
    "NKE": "Consumer",
    "LOW": "Consumer",
    "TJX": "Consumer",
    "SBUX": "Consumer",
    "BKNG": "Consumer",
    "CMG": "Consumer",
    "WMT": "Consumer",
    "COST": "Consumer",
    "PG": "Consumer",
    "KO": "Consumer",
    "PEP": "Consumer",
    "PM": "Consumer",
    "MO": "Consumer",
    "MDLZ": "Consumer",
    "CL": "Consumer",
    "KMB": "Consumer",
    "GIS": "Consumer",
    "KHC": "Consumer",
    "XLY": "Consumer",
    "XLP": "Consumer",
    "XRT": "Consumer",
    # Communication / Media
    "META": "Communication",
    "GOOGL": "Communication",
    "GOOG": "Communication",
    "NFLX": "Communication",
    "DIS": "Communication",
    "CMCSA": "Communication",
    "T": "Communication",
    "VZ": "Communication",
    "TMUS": "Communication",
    "CHTR": "Communication",
    "XLC": "Communication",
    # Industrials
    "CAT": "Industrials",
    "DE": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "UNP": "Industrials",
    "RTX": "Industrials",
    "LMT": "Industrials",
    "BA": "Industrials",
    "GE": "Industrials",
    "MMM": "Industrials",
    "XLI": "Industrials",
    "IYT": "Industrials",
    # Materials
    "LIN": "Materials",
    "APD": "Materials",
    "ECL": "Materials",
    "SHW": "Materials",
    "NEM": "Materials",
    "FCX": "Materials",
    "XLB": "Materials",
    "XME": "Materials",
    # Real Estate
    "PLD": "Real_Estate",
    "AMT": "Real_Estate",
    "EQIX": "Real_Estate",
    "CCI": "Real_Estate",
    "SPG": "Real_Estate",
    "O": "Real_Estate",
    "XLRE": "Real_Estate",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "XLU": "Utilities",
    # Commodities / diversifiers (ETFs)
    "GLD": "Commodities",
    "SLV": "Commodities",
    "IAU": "Commodities",
    "DBC": "Commodities",
    "DBA": "Commodities",
    "ARKK": "Thematic",
    "ARKG": "Thematic",
    "BITO": "Thematic",
    "GBTC": "Thematic",
    # Broad index (low thematic concentration)
    "SPY": "Diversified",
    "VOO": "Diversified",
    "VTI": "Diversified",
    "IWM": "Diversified",
    "DIA": "Diversified",
    "ITOT": "Diversified",
    "SCHB": "Diversified",
}


def get_sector(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if not s:
        return "Other"
    return _TICKER_SECTOR.get(s, "Other")


def _row_value(row: pd.Series) -> float:
    for col in ("market_value", "value", "marketValue"):
        if col in row.index:
            try:
                v = float(row.get(col) or 0.0)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    return 0.0


def _norm_ticker(row: pd.Series) -> str:
    for col in ("ticker", "symbol"):
        if col in row.index:
            t = str(row.get(col) or "").strip().upper()
            if t:
                return t.replace(".", "-")
    return ""


def compute_sector_exposure(
    positions_path: Path = DEFAULT_POSITIONS_SNAPSHOT,
    *,
    warn_pct: float = WARN_PCT_DEFAULT,
    critical_pct: float = CRITICAL_PCT_DEFAULT,
) -> Dict[str, Any]:
    """
    Read positions CSV, aggregate value by sector, attach WARNING/CRITICAL flags.
    Returns dict suitable for JSON (includes sector_values for planning).
    """
    warn_pct = float(warn_pct)
    critical_pct = float(critical_pct)
    out: Dict[str, Any] = {
        "total_value": 0.0,
        "sectors": {},
        "sector_values": {},
        "warnings": [],
        "risk_lines": [],
        "thresholds": {"warning_pct": warn_pct, "critical_pct": critical_pct},
        "positions_path": str(positions_path),
    }

    if not positions_path.is_file() or positions_path.stat().st_size == 0:
        return out

    try:
        df = pd.read_csv(positions_path)
    except Exception:
        return out

    if df is None or df.empty:
        return out

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    sector_val: Dict[str, float] = defaultdict(float)
    total = 0.0

    for _, row in df.iterrows():
        sym = _norm_ticker(row)
        if not sym:
            continue
        v = _row_value(row)
        if v <= 0:
            continue
        sec = get_sector(sym)
        sector_val[sec] += v
        total += v

    out["total_value"] = round(total, 2)
    out["sector_values"] = {
        k: round(v, 2) for k, v in sorted(sector_val.items(), key=lambda x: -x[1])
    }

    sectors_payload: Dict[str, Any] = {}
    risk_lines: List[str] = []
    warnings: List[Dict[str, Any]] = []

    if total <= 0:
        out["sectors"] = sectors_payload
        out["warnings"] = warnings
        out["risk_lines"] = risk_lines
        return out

    for sec, val in sorted(sector_val.items(), key=lambda x: -x[1]):
        pct = val / total
        entry = {"value": round(val, 2), "pct": round(pct, 4)}
        level = None
        if pct > critical_pct:
            level = "CRITICAL"
        elif pct > warn_pct:
            level = "WARNING"
        if level:
            entry["level"] = level
            warnings.append({"sector": sec, "pct": round(pct, 4), "level": level})
            pct_100 = pct * 100.0
            risk_lines.append(f"[SECTOR_RISK] {sec} = {pct_100:.0f}% ({level})")
        sectors_payload[sec] = entry

    out["sectors"] = sectors_payload
    out["warnings"] = warnings
    out["risk_lines"] = risk_lines
    return out


def write_sector_exposure_json(
    payload: Dict[str, Any],
    out_path: Path = DEFAULT_OUTPUT_JSON,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_exposure_for_planning(
    positions_path: Path = DEFAULT_POSITIONS_SNAPSHOT,
    *,
    warn_pct: float = WARN_PCT_DEFAULT,
    critical_pct: float = CRITICAL_PCT_DEFAULT,
) -> Dict[str, Any]:
    """Compute exposure and write JSON; returns same dict as compute_sector_exposure + written path."""
    payload = compute_sector_exposure(
        positions_path,
        warn_pct=warn_pct,
        critical_pct=critical_pct,
    )
    try:
        write_sector_exposure_json(payload)
    except Exception:
        pass
    return payload


def projected_sector_pct_after_buy(
    sector: str,
    add_notional: float,
    *,
    total_value: float,
    sector_values: Dict[str, float],
    pending_sector_add: Dict[str, float],
    pending_total_add: float,
) -> float:
    """Portfolio % for `sector` after adding notional (and pending in-run adds)."""
    if add_notional <= 0:
        return 0.0
    base_sec = float(sector_values.get(sector, 0.0)) + float(pending_sector_add.get(sector, 0.0))
    new_sec = base_sec + add_notional
    new_total = float(total_value) + float(pending_total_add) + add_notional
    if new_total <= 0:
        return 0.0
    return new_sec / new_total


def should_block_buy_for_sector(
    symbol: str,
    add_notional: float,
    exposure: Dict[str, Any],
    *,
    pending_sector_add: Dict[str, float],
    pending_total_add: float,
    block_pct: float = BLOCK_PCT_DEFAULT,
) -> Tuple[bool, str]:
    """
    Block new BUY/ADD if projected sector weight exceeds block_pct.
    "Other" / "Diversified" are not blocked (additive safety).
    No block when there is no portfolio baseline (denom ~ 0) — avoids first-buy deadlock.
    """
    block_pct = float(block_pct)
    if add_notional <= 0:
        return False, ""
    sector = get_sector(symbol)
    if sector in ("Other", "Diversified"):
        return False, ""

    total_value = float(exposure.get("total_value") or 0.0)
    sector_values = exposure.get("sector_values") or {}
    if isinstance(sector_values, dict):
        sv = {str(k): float(v) for k, v in sector_values.items()}
    else:
        sv = {}

    denom = total_value + float(pending_total_add)
    if denom <= 1e-9:
        return False, ""

    proj = projected_sector_pct_after_buy(
        sector,
        add_notional,
        total_value=total_value,
        sector_values=sv,
        pending_sector_add=pending_sector_add,
        pending_total_add=pending_total_add,
    )
    if proj > block_pct:
        return True, f"projected_{sector}_pct={proj:.2%}>{block_pct:.0%}"
    return False, ""
