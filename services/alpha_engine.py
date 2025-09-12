# services/alpha_engine.py
import math
import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Iterable, Tuple

import numpy as np
import pandas as pd

from .broker_alpaca import AlpacaBroker, ALPACA_DATA_BASE


@dataclass
class FactorSpec:
    name: str
    lookback: int
    weight: float


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class AlphaEngine:
    """
    Compute cross-sectional alphas from daily price bars, apply risk gates, and map to target weights.
    Produces a weights DataFrame compatible with services/place_live_orders.py (--from-weights).
    """

    def __init__(self, broker: AlpacaBroker, config: Dict):
        self.broker = broker
        self.config = config or {}

        # Factor specs (defaults if not supplied)
        fcfg = (self.config.get("factors") or {})
        self.factors: List[FactorSpec] = [
            FactorSpec("mom20", int((fcfg.get("mom20") or {}).get("lookback", 20)),
                       float((fcfg.get("mom20") or {}).get("weight", 0.40))),
            FactorSpec("mom60", int((fcfg.get("mom60") or {}).get("lookback", 60)),
                       float((fcfg.get("mom60") or {}).get("weight", 0.20))),
            FactorSpec("rev5",  int((fcfg.get("rev5")  or {}).get("lookback", 5)),
                       float((fcfg.get("rev5")  or {}).get("weight", 0.20))),
            FactorSpec("rsi14", int((fcfg.get("rsi14") or {}).get("lookback", 14)),
                       float((fcfg.get("rsi14") or {}).get("weight", -0.10))),
            FactorSpec("atr14", int((fcfg.get("atr14") or {}).get("lookback", 14)),
                       float((fcfg.get("atr14") or {}).get("weight", -0.10))),
        ]

        scfg = (self.config.get("scoring") or {})
        self.zclip = float(scfg.get("zclip", 3.0))
        self.min_z_var = float(scfg.get("min_z_var", 1e-6))

        self.long_gross = float(self.config.get("long_gross", 1.00))
        self.max_position_pct = float(self.config.get("max_position_pct", 0.10))
        self.min_price = float(self.config.get("min_price", 5.0))
        self.min_dollar_vol = float(self.config.get("min_dollar_vol", 2_000_000.0))

        self.earnings_calendar = self.config.get("earnings_calendar")
        self.earnings_window_days = int(self.config.get("earnings_window_days", 3))

        # Alpaca data base (reused from broker) and feed (iex for free plans; sip for paid)
        self.data_base = ALPACA_DATA_BASE
        self.data_feed = str(self.config.get("data_feed", os.getenv("ALPACA_DATA_FEED", "iex"))).lower()

        # Optional: basic debugging toggle from config
        self.debug = bool(self.config.get("debug", False))

    # ----------------------------- Data ---------------------------------

    def _fetch_symbol_bars(self, symbol: str, limit: int = 120, verbose: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch daily bars for a single symbol. Returns DataFrame with columns: t,o,h,l,c,v
        Adds 'symbol' column and sorts ascending on 't'.
        """
        end_dt = datetime.now(timezone.utc)
        # Wider start window to survive holidays
        start_dt = end_dt - timedelta(days=int(limit * 2.2))

        url = f"{self.data_base}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": "1Day",
            "start": _utc_iso(start_dt),
            "end": _utc_iso(end_dt),
            "limit": limit,
            "adjustment": "raw",
            "feed": self.data_feed,  # critical: iex for free plan
        }
        r = self.broker.session.get(url, params=params, timeout=self.broker.timeout)
        if r.status_code >= 300:
            if verbose or self.debug:
                print(f"[alpha] bars GET failed {symbol} {r.status_code}: {r.text[:200]}")
            return None

        try:
            data = r.json()
        except Exception:
            if verbose or self.debug:
                print(f"[alpha] bars JSON decode failed for {symbol}")
            return None

        bars = data.get("bars") or data.get("barset") or []
        if not bars:
            if verbose or self.debug:
                print(f"[alpha] no bars returned for {symbol}")
            return None

        df = pd.DataFrame(bars)
        # Normalize column names: ensure ['t','o','h','l','c','v']
        col_map = {}
        for k in df.columns:
            lk = k.lower()
            if lk in ("t", "timestamp", "time"):
                col_map[k] = "t"
            elif lk in ("o", "open"):
                col_map[k] = "o"
            elif lk in ("h", "high"):
                col_map[k] = "h"
            elif lk in ("l", "low"):
                col_map[k] = "l"
            elif lk in ("c", "close"):
                col_map[k] = "c"
            elif lk in ("v", "volume"):
                col_map[k] = "v"
        df = df.rename(columns=col_map)

        need = {"t", "o", "h", "l", "c", "v"}
        if not need.issubset(set(df.columns)):
            if verbose or self.debug:
                print(f"[alpha] missing cols for {symbol}: have={sorted(df.columns)} need={sorted(need)}")
            return None

        # Parse timestamps & clean
        df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
        df = df.dropna(subset=["t", "o", "h", "l", "c", "v"]).sort_values("t").reset_index(drop=True)
        df["symbol"] = symbol
        return df

    def fetch_bars_universe(self, symbols: Iterable[str], lookback: int, verbose: bool = False) -> Dict[str, pd.DataFrame]:
        res: Dict[str, pd.DataFrame] = {}
        lim = max(lookback, max([fs.lookback for fs in self.factors]) + 5)
        for sym in symbols:
            df = self._fetch_symbol_bars(sym, limit=lim, verbose=verbose)
            # tolerate some missing days; keep if we have at least ~half the requested length
            if df is not None and len(df) >= max(20, lim // 2):
                res[sym] = df
        return res

    # -------------------------- Factor calc ------------------------------

    @staticmethod
    def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
        delta = close.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=close.index).rolling(n).mean()
        roll_down = pd.Series(down, index=close.index).rolling(n).mean()
        rs = roll_up / (roll_down.replace(0.0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()
        return atr

    @staticmethod
    def _ret(close: pd.Series, n: int) -> pd.Series:
        return close.pct_change(n)

    def compute_factor_snapshot(self, bars: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        For each symbol, compute latest factor values and liquidity metrics.
        Returns (df, latest_ts) where df columns:
            ['symbol','px','dollar_vol_med20','mom20','mom60','rev5','rsi14','atr14']
        Always returns the expected columns (may be an empty frame). Timestamp is UTC-aware.
        """
        rows: List[Dict] = []
        latest_common_ts: Optional[pd.Timestamp] = None

        for sym, df in bars.items():
            if len(df) < 30:
                continue
            c = pd.to_numeric(df["c"], errors="coerce")
            h = pd.to_numeric(df["h"], errors="coerce")
            l = pd.to_numeric(df["l"], errors="coerce")
            v = pd.to_numeric(df["v"], errors="coerce")

            # Factors
            mom20 = self._ret(c, 20)
            mom60 = self._ret(c, 60)
            rev5  = -self._ret(c, 5)   # short-term reversal: negative of 5d return
            rsi14 = self._rsi(c, 14)
            atr14 = self._atr(h, l, c, 14)

            # Liquidity & price
            dollar_vol = c * v
            med_dv20 = dollar_vol.rolling(20).median()

            last_ts = df["t"].iloc[-1]
            if not isinstance(last_ts, pd.Timestamp):
                last_ts = pd.to_datetime(last_ts, utc=True, errors="coerce")
            if latest_common_ts is None:
                latest_common_ts = last_ts
            else:
                latest_common_ts = max(latest_common_ts, last_ts)

            rows.append({
                "symbol": sym,
                "px": float(c.iloc[-1]) if not math.isnan(float(c.iloc[-1])) else 0.0,
                "dollar_vol_med20": float(med_dv20.iloc[-1]) if pd.notna(med_dv20.iloc[-1]) else 0.0,
                "mom20": float(mom20.iloc[-1]) if pd.notna(mom20.iloc[-1]) else 0.0,
                "mom60": float(mom60.iloc[-1]) if pd.notna(mom60.iloc[-1]) else 0.0,
                "rev5":  float(rev5.iloc[-1])  if pd.notna(rev5.iloc[-1])  else 0.0,
                "rsi14": float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else 50.0,
                "atr14": float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else 0.0,
            })

        cols = ["symbol", "px", "dollar_vol_med20", "mom20", "mom60", "rev5", "rsi14", "atr14"]
        snap = pd.DataFrame(rows, columns=cols).dropna(how="all").reset_index(drop=True)

        if latest_common_ts is None:
            latest_common_ts = pd.Timestamp(datetime.now(timezone.utc))

        return snap, latest_common_ts

    # --------------------------- Risk gates ------------------------------

    def _load_earnings_calendar(self) -> Optional[pd.DataFrame]:
        path = self.earnings_calendar
        if not path or not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            # expected columns: symbol,date  (date ISO or YYYY-MM-DD)
            if not {"symbol", "date"}.issubset(df.columns):
                return None
            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["symbol", "date"])
            return df
        except Exception:
            return None

    def gate_universe(self, snap: pd.DataFrame, ref_ts: pd.Timestamp) -> pd.DataFrame:
        if snap is None or snap.empty:
            return snap

        df = snap.copy()

        # Price Gate
        if "px" in df.columns:
            df = df[df["px"] >= self.min_price]

        # Liquidity Gate
        if "dollar_vol_med20" in df.columns:
            df = df[df["dollar_vol_med20"] >= self.min_dollar_vol]

        # Earnings Gate (optional)
        earn = self._load_earnings_calendar()
        if earn is not None and len(earn):
            try:
                # ref_ts should already be tz-aware
                start = ref_ts - pd.Timedelta(days=self.earnings_window_days)
                end = ref_ts + pd.Timedelta(days=self.earnings_window_days)
                # Ensure UTC dtype
                start = pd.Timestamp(start).tz_convert("UTC")
                end = pd.Timestamp(end).tz_convert("UTC")
                earn = earn[(earn["date"] >= start) & (earn["date"] <= end)]
                if len(earn):
                    ex = set(earn["symbol"].unique().tolist())
                    if "symbol" in df.columns:
                        df = df[~df["symbol"].isin(ex)]
            except Exception:
                # If anything goes wrong with dates, just skip earnings gating
                pass

        return df.reset_index(drop=True)

    # --------------------------- Scoring --------------------------------

    def _zscore(self, s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd <= self.min_z_var:
            return pd.Series(np.zeros_like(s), index=s.index)
        z = (s - mu) / sd
        if self.zclip:
            z = z.clip(-self.zclip, self.zclip)
        return z

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional z-scores per factor, then weighted sum => alpha.
        """
        if df is None or df.empty:
            return df

        out = df.copy()

        # Ensure factor columns exist (fill missing with 0.0)
        for spec in self.factors:
            if spec.name not in out.columns:
                out[spec.name] = 0.0

        # z-scores (cross-sectional)
        zcols = {}
        for spec in self.factors:
            z = self._zscore(out[spec.name])
            out[f"z_{spec.name}"] = z
            zcols[spec.name] = z

        # weighted alpha
        alpha = np.zeros(len(out))
        contribs = []
        for spec in self.factors:
            w = float(spec.weight)
            z = zcols[spec.name].values
            alpha += w * z
            contribs.append((spec.name, w))
        out["alpha"] = alpha

        # contribution summary string (by factor weights)
        out["source_weights"] = out.apply(
            lambda r: "|".join([f"{name}={w:+.2f}" for name, w in contribs]),
            axis=1
        )
        return out

    # ------------------------- Weight mapping ----------------------------

    def map_alpha_to_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Positive-alpha long-only portfolio.
        - Set negatives to zero.
        - Cap each name at max_position_pct.
        - Normalize to long_gross.
        """
        if df is None or df.empty:
            return df

        out = df.copy()
        a = pd.to_numeric(out["alpha"], errors="coerce").fillna(0.0)
        a = a.clip(lower=0.0)

        if a.sum() <= 1e-12:
            # fallback: equal weight among survivors (capped)
            n = len(out)
            if n == 0:
                out["target_weight"] = 0.0
                return out
            ew = min(self.max_position_pct, self.long_gross / max(1, n))
            out["target_weight"] = float(ew)
            return out

        w = a / a.sum()
        # Cap per-name
        w = w.clip(upper=self.max_position_pct)
        # Re-normalize if we clipped
        s = float(w.sum())
        if s > 0:
            w = (w / s) * min(self.long_gross, s)
        out["target_weight"] = w.astype(float)
        return out

    # ----------------------------- Public --------------------------------

    def build_weights(self, universe: Iterable[str], lookback: int, verbose: bool = False) -> pd.DataFrame:
        bars = self.fetch_bars_universe(universe, lookback=lookback, verbose=verbose or self.debug)
        if verbose:
            print(f"[alpha] fetched bars for {len(bars)}/{len(list(universe))} symbols")

        snap, ts = self.compute_factor_snapshot(bars)
        if verbose:
            print(f"[alpha] factor snapshot @ {ts}")

        gated = self.gate_universe(snap, ts)
        if verbose and not snap.empty:
            dropped = set(snap.get("symbol", pd.Series(dtype=str))) - set(gated.get("symbol", pd.Series(dtype=str)))
            if dropped:
                print(f"[alpha] gated out: {', '.join(sorted(dropped))}")

        scored = self.score(gated)
        weighted = self.map_alpha_to_weights(scored)

        # Notes column with alpha & timestamp
        iso_ts = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if weighted is None or weighted.empty:
            # Return an empty-but-shaped frame to keep downstream happy
            out = pd.DataFrame(columns=["ticker", "target_weight", "source_weights", "notes"])
            return out

        weighted["notes"] = weighted.apply(lambda r: f"alpha={float(r.get('alpha', 0.0)):.3f} ts={iso_ts}", axis=1)

        # Final output shape for place_live_orders
        out = weighted[["symbol", "target_weight", "source_weights", "notes"]].copy()
        out = out.rename(columns={"symbol": "ticker"})
        out = out.sort_values("target_weight", ascending=False).reset_index(drop=True)

        # Ensure floats are well-formed
        out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0).round(6)
        out["ticker"] = out["ticker"].astype(str).str.upper()
        out["source_weights"] = out["source_weights"].astype(str)
        out["notes"] = out["notes"].astype(str)
        return out
