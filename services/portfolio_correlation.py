"""Rolling return correlation vs existing book — concentration damping for reallocation (not execution guards)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SECTOR_MAP_PATH = ROOT / "data" / "config" / "sector_ticker_map.json"


def _sym_key(sym: str) -> str:
    return str(sym or "").strip().upper().replace("-", ".")


def _parquet_path(sym_key: str, processed_dir: Path) -> Optional[Path]:
    variants = [
        sym_key.replace(".", "-") + ".parquet",
        sym_key + ".parquet",
    ]
    for v in variants:
        p = processed_dir / v
        if p.is_file():
            return p
    return None


def _load_daily_returns(sym_key: str, processed_dir: Path, lookback: int) -> Optional[pd.Series]:
    path = _parquet_path(sym_key, processed_dir)
    if path is None:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df is None or df.empty or "close" not in df.columns:
        return None
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    else:
        df = df.sort_index()
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.shape[0] < 5:
        return None
    ret = close.pct_change().dropna()
    if lookback and lookback > 0 and len(ret) > lookback:
        ret = ret.iloc[-lookback:]
    return ret


def _build_returns_matrix(
    sym_keys: Sequence[str],
    processed_dir: Path,
    lookback: int,
) -> pd.DataFrame:
    cols: Dict[str, pd.Series] = {}
    for sk in sym_keys:
        s = _load_daily_returns(sk, processed_dir, lookback)
        if s is not None and len(s) >= 5:
            cols[sk] = s.rename(sk)
    if len(cols) < 1:
        return pd.DataFrame()
    out = pd.concat(cols.values(), axis=1)
    out = out.dropna(axis=0, how="any")
    return out


def _ticker_map_key(sym: str) -> str:
    return str(sym or "").strip().upper().replace(".", "-")


def _load_sector_map() -> Dict[str, str]:
    if not SECTOR_MAP_PATH.is_file():
        return {}
    try:
        u = json.loads(SECTOR_MAP_PATH.read_text(encoding="utf-8", errors="replace"))
        if isinstance(u, dict):
            return {_ticker_map_key(str(k)): str(v).strip() for k, v in u.items()}
    except Exception:
        pass
    return {}


def _sector_proxy_score(
    candidate_key: str, book_keys: Set[str], sector_map: Dict[str, str]
) -> float:
    """If price data missing: same sector id as a held name → moderate overlap proxy (0–1)."""
    if not book_keys or not sector_map:
        return 0.0
    sec = sector_map.get(_ticker_map_key(candidate_key))
    if not sec:
        return 0.0
    for h in book_keys:
        hs = sector_map.get(_ticker_map_key(h))
        if hs and hs == sec:
            return 0.55
    return 0.0


def _penalty_from_score(score: float, cfg: Dict[str, Any]) -> float:
    hi = float(cfg.get("correlation_high_threshold", 0.7))
    med = float(cfg.get("correlation_medium_threshold", 0.5))
    if score >= hi:
        return float(cfg.get("correlation_penalty_high", 0.5))
    if score >= med:
        return float(cfg.get("correlation_penalty_medium", 0.75))
    return float(cfg.get("correlation_penalty_low", 1.0))


def batch_correlation_scores(
    candidate_keys: Sequence[str],
    book_keys: Set[str],
    cfg: Dict[str, Any],
    processed_dir: Optional[Path] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    For each candidate sym_key, return (correlation_score, correlation_penalty).
    Score = max Pearson correlation of daily returns vs any held position (recent window).
    Only non-negative correlations count toward overlap. Empty book → (0, 1).
    """
    out: Dict[str, Tuple[float, float]] = {}
    if not cfg.get("correlation_filter_enabled", True):
        for k in candidate_keys:
            key = _sym_key(k)
            out[key] = (0.0, 1.0)
        return out

    pdir = processed_dir or PROCESSED_DIR
    lookback = int(cfg.get("correlation_lookback_days", 120) or 120)
    min_days = int(cfg.get("correlation_min_overlap_days", 20) or 20)

    sector_map = _load_sector_map() if cfg.get("correlation_sector_proxy_fallback", True) else {}

    book_f = {_sym_key(b) for b in book_keys}
    if not book_f:
        for k in candidate_keys:
            out[_sym_key(k)] = (0.0, 1.0)
        return out

    uniq = sorted({_sym_key(c) for c in candidate_keys} | book_f)
    rets = _build_returns_matrix(uniq, pdir, lookback)

    corr_full = None
    if not rets.empty and len(rets) >= min_days and rets.shape[1] >= 2:
        try:
            corr_full = rets.corr()
        except Exception:
            corr_full = None

    for k in candidate_keys:
        ck = _sym_key(k)
        score = 0.0
        if corr_full is not None and ck in corr_full.index:
            for h in book_f:
                if h == ck or h not in corr_full.columns:
                    continue
                try:
                    v = float(corr_full.at[ck, h])
                except Exception:
                    continue
                if v > score:
                    score = v
            score = max(0.0, score)
        if score < 1e-6 and sector_map:
            sp = _sector_proxy_score(ck, book_f, sector_map)
            if sp > score:
                score = sp

        penalty = _penalty_from_score(score, cfg)
        out[ck] = (round(score, 4), round(penalty, 4))

    return out
