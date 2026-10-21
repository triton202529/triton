# services/fetch_raw_data.py
# --------------------------
# Fetch daily OHLCV from yfinance into data/raw/{TICKER}.csv
# Guarantees columns: date, open, high, low, close, adj_close, volume
#
# Deterministic AS_OF hardening (Jan 2026):
# - Default fetch window is driven by "last completed NYSE session" (ET),
#   same AS_OF used by preprocessing/training/signals/contracts.
# - yfinance 'end' is treated as EXCLUSIVE → fetch end = (AS_OF_DATE + 1 day)
# - A provider frame that lags the completed session, or is older than the local file,
#   is diagnostic only. It must not replace, truncate, or create a raw file.
# - A provider frame may append only validated rows strictly later than the local maximum.
#   Omitting an older prefix, or disagreeing on an older overlap, is not permission to rewrite history.
# - A same-session provider bar may replace that one local row only when it is an attributable,
#   unadjusted, complete revision that passes the shared OHLC predicate. Prices are not merged.
#   Every required XNYS session between the local maximum and the completed session is required.
# - ALLOW_DATA_LAG_DAYS remains a diagnostic tolerance. It does not authorize a write.
#
# Hardening patch:
# - Retries with exponential backoff for flaky yfinance timeouts
# - Supports a longer timeout (default 30s)
# - If download fails, reads existing data/raw/{TICKER}.csv and reports last_date
#   (so we can see whether we're truly behind AS_OF or it was already OK)
# - Writes data/results/raw_fetch_diagnostics.csv with per-ticker status + last_date

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("yfinance is required. pip install yfinance") from e


RAW_DIR = Path("data") / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("data") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DIAG_PATH = RESULTS_DIR / "raw_fetch_diagnostics.csv"
SUMMARY_JSON = RESULTS_DIR / "raw_fetch_summary.json"

DEFAULT_TICKERS_FILE = Path("data") / "config" / "tickers.txt"

# Diagnostic only. A lagging provider frame is not current evidence and is not written.
ALLOW_DATA_LAG_DAYS = 2
MIN_TRADABLE_COVERAGE = 0.90
MISSING_EXPECTED_SESSION_BAR = "MISSING_EXPECTED_SESSION_BAR"
INPUT_SKIPS_REL = "data/results/daily_refresh_input_skips.json"
FETCH_RECEIPT_REL = "data/results/daily_refresh_fetch_receipt.json"

# Accept common index tickers and BRK-B style
TICKER_RE = re.compile(r"^[A-Z0-9.\-\^]{1,15}$")

CANON_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "volume": "volume",
    "date": "date",
    "datetime": "date",
}

NEEDED = ["date", "open", "high", "low", "close", "adj_close", "volume"]

# Same-session agreement. Prices use the shared absolute tolerance. Volume is exact.
# Values that differ are kept as returned; this comparison does not round or rewrite them.
SAME_SESSION_COMPARISON_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")
_SAME_SESSION_PRICE_FIELDS = ("open", "high", "low", "close", "adj_close")
_FAIL_CLOSED_COVERAGE = False


# ─────────────────────────────────────────────────────────────
# AS_OF helpers
# ─────────────────────────────────────────────────────────────


def _compute_asof_date(now_utc: Optional[datetime] = None) -> date:
    try:
        from services.market_calendar import last_completed_nyse_session  # type: ignore

        d = last_completed_nyse_session()
        if isinstance(d, date):
            return d
        return pd.to_datetime(str(d), errors="coerce").date()
    except Exception:
        n = now_utc or datetime.now(timezone.utc)
        d = n.date()
        if d.weekday() == 5:
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d - timedelta(days=2)
        return d - timedelta(days=1)


def _asof_to_end_exclusive(asof: date) -> str:
    return (asof + timedelta(days=1)).isoformat()


def _parse_last_date(last: Optional[str]) -> Optional[date]:
    if not last:
        return None
    try:
        d = pd.to_datetime(str(last).strip(), errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


def _asof_lag_days(expected: date, latest: Optional[date]) -> Optional[int]:
    """Calendar days expected is after latest (positive = data behind). None if unknown latest."""
    if latest is None:
        return None
    return (expected - latest).days


def _asof_within_tolerance(expected: date, latest: Optional[date], allow_days: int) -> bool:
    if latest is None:
        return False
    lag = _asof_lag_days(expected, latest)
    if lag is None:
        return False
    return lag <= allow_days


# ─────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    lvl0 = {str(x).strip().lower() for x in df.columns.get_level_values(0)}
    lvl1 = {str(x).strip().lower() for x in df.columns.get_level_values(1)}
    ohlcv = {"open", "high", "low", "close", "adj close", "adj_close", "volume"}

    if len(lvl0 & ohlcv) >= 3:
        df.columns = [c[0] for c in df.columns]
    elif len(lvl1 & ohlcv) >= 3:
        df.columns = [c[1] for c in df.columns]
    else:
        df.columns = ["_".join([str(a), str(b)]) for a, b in df.columns]

    return df


def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _flatten_cols(df)

    if "date" not in {str(c).strip().lower() for c in df.columns}:
        df = df.reset_index()

    df.columns = [str(c).strip().lower() for c in df.columns]

    rename = {c: CANON_MAP[c] for c in df.columns if c in CANON_MAP}
    df = df.rename(columns=rename)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    for c in NEEDED:
        if c not in df.columns:
            df[c] = pd.NA

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df[NEEDED]


def _is_valid_token(tok: str) -> bool:
    tok = tok.strip().upper()
    if not tok:
        return False
    if tok.startswith("#"):
        return False
    if not TICKER_RE.match(tok):
        return False
    return True


def _split_tickers(items: List[str]) -> List[str]:
    out: List[str] = []
    for x in items:
        if not x:
            continue
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        out.extend(parts)

    out = [t.strip().upper() for t in out]
    out = [t for t in out if _is_valid_token(t)]

    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _load_tickers_from_file(path: Path) -> List[str]:
    if not (path.exists() and path.stat().st_size > 0):
        return []

    raw = path.read_text(encoding="utf-8-sig").splitlines()

    cleaned: List[str] = []
    for ln in raw:
        if not ln:
            continue
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        cleaned.append(ln)

    return _split_tickers(cleaned)


def _existing_last_date(ticker: str) -> Optional[str]:
    p = RAW_DIR / f"{ticker}.csv"
    if not (p.exists() and p.stat().st_size > 0):
        return None
    try:
        df = pd.read_csv(p)
        if "date" not in df.columns or df.empty:
            return None
        d = pd.to_datetime(df["date"], errors="coerce")
        if d.isna().all():
            return None
        return str(d.max().date())
    except Exception:
        return None


def _download_one(
    ticker: str, start: str, end: str, timeout_s: int, verbose: bool
) -> Tuple[bool, str, Optional[str], int, Optional[pd.DataFrame]]:
    """Download and normalize one symbol. Do not write the production raw file."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=timeout_s,
        )

        if df is None or df.empty:
            return False, "EMPTY", None, 0, None

        norm = _normalize_yf_df(df)
        if norm is None or norm.empty:
            cols = list(df.columns)[:12]
            msg = (
                f"EMPTY_AFTER_NORMALIZE cols_sample={cols}" if verbose else "EMPTY_AFTER_NORMALIZE"
            )
            return False, msg, None, 0, None

        last = str(norm["date"].max()) if "date" in norm.columns else None
        norm.attrs["provider"] = "yfinance"
        norm.attrs["adjustment"] = "unadjusted"
        norm.attrs["interval"] = "1d"
        norm.attrs["auto_adjust"] = False
        norm.attrs["symbol"] = ticker.strip().upper()
        msg = f"rows={len(norm)} max={last}" if verbose else "OK"
        return True, msg, last, int(len(norm)), norm

    except Exception as e:
        return False, f"ERROR {e}", None, 0, None


def fetch_one_with_retries(
    ticker: str,
    start: str,
    end: str,
    retries: int,
    timeout_s: int,
    verbose: bool,
) -> Tuple[bool, str, Optional[str], int, str, Optional[pd.DataFrame]]:
    """
    Returns: (ok, msg, last_date, rows, source, frame)
    source: 'download' or 'existing'
    The frame is staged in memory. Promotion is a separate, gated step.
    """
    last_err = ""
    for i in range(max(1, retries)):
        ok, msg, last_date, rows, frame = _download_one(ticker, start, end, timeout_s, verbose)
        if ok:
            return True, msg, last_date, rows, "download", frame

        last_err = msg
        if i < retries - 1:
            time.sleep(min(8.0, 1.5**i))

    existing = _existing_last_date(ticker)
    if existing:
        return (
            False,
            f"{last_err} | using existing last_date={existing}",
            existing,
            0,
            "existing",
            None,
        )

    return False, last_err, None, 0, "none", None


def _dates_of(frame: Optional[pd.DataFrame]) -> pd.Series:
    if frame is None or frame.empty or "date" not in frame.columns:
        return pd.Series(dtype="object")
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _local_frame(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _local_max(path: Path) -> Optional[str]:
    dates = _dates_of(_local_frame(path))
    if dates.empty or bool(dates.isna().all()):
        return None
    return str(dates.max())


def _date_list(frame: Optional[pd.DataFrame]) -> List[str]:
    dates = _dates_of(frame)
    if dates.empty:
        return []
    return [str(day) for day in dates.tolist() if isinstance(day, str) and day and day != "NaT"]


def _rows_by_date(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    rows: Dict[str, pd.Series] = {}
    for idx, day in enumerate(_date_list(frame)):
        rows[day] = frame.iloc[idx]
    return rows


def _same_cell(left: object, right: object) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if bool(left_missing) and bool(right_missing):
        return True
    if bool(left_missing) or bool(right_missing):
        return False
    try:
        return round(float(left), 6) == round(float(right), 6)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _overlap_conflicts(
    existing: pd.DataFrame, frame: pd.DataFrame, shared_dates: Sequence[str]
) -> List[str]:
    """Dates whose provider OHLCV differs. Older dates stay diagnostic."""
    local_rows = _rows_by_date(existing)
    provider_rows = _rows_by_date(frame)
    columns = [
        col
        for col in ("open", "high", "low", "close", "adj_close", "volume")
        if col in existing.columns and col in frame.columns
    ]
    conflicts: List[str] = []
    for day in shared_dates:
        local = local_rows.get(day)
        provider = provider_rows.get(day)
        if local is None or provider is None:
            continue
        if any(not _same_cell(local[col], provider[col]) for col in columns):
            conflicts.append(day)
    return conflicts


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_sf40_ord"] = _dates_of(out).to_numpy()
    out = out.sort_values("_sf40_ord", kind="mergesort").drop(columns=["_sf40_ord"])
    return out.reset_index(drop=True)


def _required_newer_sessions(local_max: str, asof: date) -> Optional[List[str]]:
    """XNYS sessions strictly after local_max through the completed session."""
    try:
        import exchange_calendars as xc
    except Exception:
        return None
    start = date.fromisoformat(local_max) + timedelta(days=1)
    if start > asof:
        return []
    try:
        labels = xc.get_calendar("XNYS").sessions_in_range(start.isoformat(), asof.isoformat())
    except Exception:
        return None
    return [pd.Timestamp(item).date().isoformat() for item in labels]


def _non_xnys_dates(days: Sequence[str]) -> Optional[List[str]]:
    """Dates that are not XNYS sessions. None means the calendar could not be read."""
    if not days:
        return []
    try:
        import exchange_calendars as xc
    except Exception:
        return None
    try:
        labels = xc.get_calendar("XNYS").sessions_in_range(min(days), max(days))
    except Exception:
        return None
    sessions = {pd.Timestamp(item).date().isoformat() for item in labels}
    return sorted(day for day in days if day not in sessions)


def _append_csv_bytes(existing_path: Path, merged: pd.DataFrame, newer_count: int) -> bytes:
    """Keep the local CSV bytes and append only the new rows."""
    raw = existing_path.read_bytes()
    newline = b"\r\n" if raw.endswith(b"\r\n") else b"\n"
    if raw and not raw.endswith((b"\n", b"\r")):
        raw += newline
    added = merged.iloc[-newer_count:].to_csv(
        index=False, header=False, lineterminator=newline.decode("ascii")
    )
    return raw + added.encode("utf-8")


def _merge_append(
    existing: pd.DataFrame, frame: pd.DataFrame, newer_dates: Sequence[str]
) -> Optional[pd.DataFrame]:
    """Keep every local row and append validated provider rows after the local maximum."""
    local_dates = _date_list(existing)
    if local_dates != sorted(set(local_dates)):
        return None
    provider_rows = _rows_by_date(frame)
    columns = list(existing.columns)
    added: List[Dict[str, object]] = []
    for day in newer_dates:
        source = provider_rows.get(day)
        if source is None:
            return None
        row: Dict[str, object] = {}
        for col in columns:
            if col == "date":
                row[col] = day
                continue
            if col not in source.index:
                return None
            row[col] = source[col]
        added.append(row)
    merged = pd.concat([existing.copy(), pd.DataFrame(added, columns=columns)], ignore_index=True)
    if _date_list(merged) != local_dates + list(newer_dates):
        return None
    return merged


def _ohlc_key(row: pd.Series) -> Tuple[float, float, float, float]:
    return (
        round(float(row["open"]), 6),
        round(float(row["high"]), 6),
        round(float(row["low"]), 6),
        round(float(row["close"]), 6),
    )


def _payload_problems(
    frame: Optional[pd.DataFrame], symbol: str, asof: date, existing: Optional[pd.DataFrame]
) -> List[str]:
    if frame is None or frame.empty:
        return ["empty"]
    problems: List[str] = []
    if "ticker" in frame.columns:
        found = {str(v).strip().upper() for v in frame["ticker"].dropna().tolist()}
        found.discard("")
        if found and found != {symbol.upper()}:
            problems.append("wrong_symbol")
    dates = _dates_of(frame)
    if dates.empty or bool(dates.isna().any()):
        problems.append("bad_dates")
        return problems
    if bool(dates.duplicated().any()):
        problems.append("duplicate_session")
    if bool((dates > asof.isoformat()).any()):
        problems.append("future_or_partial_bar")
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            problems.append("malformed_ohlc")
            return problems
        values = pd.to_numeric(frame[col], errors="coerce")
        if bool(values.isna().any()) or bool((values <= 0).any()):
            problems.append("malformed_ohlc")
            return problems
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    if bool((high < low).any()) or bool((close > high).any()) or bool((close < low).any()):
        problems.append("malformed_ohlc")
        return problems
    if "volume" not in frame.columns:
        problems.append("partial_bar")
    else:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        if bool(volume.isna().any()) or bool((volume < 0).any()):
            problems.append("partial_bar")
    if existing is not None and not existing.empty and MISSING_EXPECTED_SESSION_BAR not in problems:
        existing_dates = _dates_of(existing)
        staged_asof = frame.loc[dates == asof.isoformat()]
        if not staged_asof.empty:
            new_key = _ohlc_key(staged_asof.iloc[-1])
            older = existing.loc[existing_dates < asof.isoformat()]
            for _, old in older.iterrows():
                try:
                    if _ohlc_key(old) == new_key:
                        problems.append("relabeled_older_row")
                        break
                except Exception:
                    continue
    return problems


def _json_number(value: object, field: str) -> object:
    number = float(value)  # type: ignore[arg-type]
    if field == "volume" and math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def session_correction_row_sha256(values: Mapping[str, object]) -> str:
    """Hash exact correction values. The encoding does not round prices."""
    payload = {key: values[key] for key in sorted(values)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _comparison_value(row: pd.Series, field: str) -> object:
    if field not in row.index or pd.isna(row[field]):
        return None
    return _json_number(row[field], field)


def _session_values(row: pd.Series) -> Dict[str, object]:
    return {field: _comparison_value(row, field) for field in SAME_SESSION_COMPARISON_FIELDS}


def _fields_agree(local: pd.Series, provider: pd.Series) -> bool:
    from services.session_ohlc import OHLC_ABSOLUTE_TOLERANCE

    for field in SAME_SESSION_COMPARISON_FIELDS:
        left = _comparison_value(local, field)
        right = _comparison_value(provider, field)
        if left is None and right is None:
            continue
        if left is None or right is None:
            return False
        if field == "volume":
            if left != right:
                return False
            continue
        if abs(float(left) - float(right)) > OHLC_ABSOLUTE_TOLERANCE:
            return False
    return True


def _one_session_row(frame: pd.DataFrame, target: str) -> Optional[pd.Series]:
    rows = _rows_by_date(frame)
    if _date_list(frame).count(target) != 1 or target not in rows:
        return None
    return rows[target]


def _provider_response_error(frame: pd.DataFrame, symbol: str) -> Optional[str]:
    attrs = dict(getattr(frame, "attrs", {}) or {})
    identity = ("provider", "adjustment", "interval", "auto_adjust", "symbol")
    if all(attrs.get(key) is None for key in identity):
        return "missing_provider_provenance"
    if (
        attrs.get("provider") != "yfinance"
        or attrs.get("symbol") != symbol.upper()
        or attrs.get("interval") != "1d"
    ):
        return "missing_provider_provenance"
    if attrs.get("auto_adjust") is not False or attrs.get("adjustment") != "unadjusted":
        return "incompatible_provider_convention"
    return None


def _csv_cell(value: object) -> Optional[str]:
    if isinstance(value, str):
        if "," in value or "\n" in value or "\r" in value:
            return None
        return value
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) < 10**15:
        return str(int(number))
    text = format(number, ".17g")
    if float(text) != number:
        text = repr(number)
        if float(text) != number:
            return None
    return text


def _splice_session_row(
    existing_path: Path, target: str, provider_row: pd.Series
) -> Optional[bytes]:
    raw = existing_path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8-sig")
    ends_nl = text.endswith("\n")
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    matches = [
        index for index, line in enumerate(lines[1:], start=1) if line.split(",")[0] == target
    ]
    if len(matches) != 1:
        return None
    cells: List[str] = []
    for column in header:
        if column == "date":
            cells.append(target)
            continue
        if column not in provider_row.index:
            return None
        cell = _csv_cell(provider_row[column])
        if cell is None:
            return None
        cells.append(cell)
    lines[matches[0]] = ",".join(cells)
    body = newline.join(lines)
    if ends_nl:
        body += newline
    return body.encode("utf-8")


def _correction_record(
    *,
    target: str,
    local_row: pd.Series,
    provider_row: pd.Series,
    existing_path: Path,
    violation: Optional[dict],
    candidate: Optional[bytes],
    provider: Optional[str],
    adjustment: Optional[str],
) -> Dict[str, object]:
    before = _session_values(local_row)
    after = _session_values(provider_row)
    record: Dict[str, object] = {
        "session_date": target,
        "comparison_fields": list(SAME_SESSION_COMPARISON_FIELDS),
        "price_tolerance": 1e-6,
        "volume_comparison": "exact",
        "before": before,
        "after": after,
        "before_row_sha256": session_correction_row_sha256(before),
        "after_row_sha256": session_correction_row_sha256(after),
        "before_raw_sha256": hashlib.sha256(existing_path.read_bytes()).hexdigest(),
        "provider": provider,
        "adjustment": adjustment,
        "ohlc_violation": violation,
    }
    if candidate is not None:
        record["after_raw_sha256"] = hashlib.sha256(candidate).hexdigest()
    return record


def _fail_closed_revision(
    reason: str,
    *,
    target: str,
    local_row: pd.Series,
    provider_row: Optional[pd.Series],
    existing_path: Path,
    violation: Optional[dict] = None,
    provider: Optional[str] = None,
    adjustment: Optional[str] = None,
) -> Dict[str, object]:
    decision: Dict[str, object] = {
        "action": "keep",
        "reason": reason,
        "operationally_eligible": False,
        "counts_toward_coverage": _FAIL_CLOSED_COVERAGE,
    }
    if provider_row is not None:
        decision["same_session_correction"] = _correction_record(
            target=target,
            local_row=local_row,
            provider_row=provider_row,
            existing_path=existing_path,
            violation=violation,
            candidate=None,
            provider=provider,
            adjustment=adjustment,
        )
    elif violation is not None:
        decision["ohlc_violation"] = violation
    return decision


def _classify_same_session_revision(
    existing: pd.DataFrame,
    frame: pd.DataFrame,
    symbol: str,
    asof: date,
    existing_path: Path,
) -> Dict[str, object]:
    """Classify one target session. This function does not write."""
    from services.session_ohlc import session_ohlc_violation

    target = asof.isoformat()
    local_row = _one_session_row(existing, target)
    provider_row = _one_session_row(frame, target)
    if local_row is None or provider_row is None:
        return {
            "action": "keep",
            "reason": "ambiguous_provider_session",
            "operationally_eligible": False,
            "counts_toward_coverage": _FAIL_CLOSED_COVERAGE,
        }
    local_violation = session_ohlc_violation(
        open=local_row.get("open"),
        high=local_row.get("high"),
        low=local_row.get("low"),
        close=local_row.get("close"),
        symbol=symbol,
        session_date=target,
        source_layer="raw",
    )
    if _fields_agree(local_row, provider_row):
        if local_violation is not None:
            return _fail_closed_revision(
                "invalid_local_session_ohlc",
                target=target,
                local_row=local_row,
                provider_row=provider_row,
                existing_path=existing_path,
                violation=local_violation,
            )
        return {
            "action": "keep",
            "reason": "local_already_current",
            "operationally_eligible": True,
        }
    provenance_error = _provider_response_error(frame, symbol)
    attrs = dict(getattr(frame, "attrs", {}) or {})
    if provenance_error == "missing_provider_provenance":
        return _fail_closed_revision(
            "missing_provider_provenance",
            target=target,
            local_row=local_row,
            provider_row=provider_row,
            existing_path=existing_path,
            provider=attrs.get("provider"),
            adjustment=attrs.get("adjustment"),
        )
    if provenance_error == "incompatible_provider_convention":
        return _fail_closed_revision(
            "incompatible_provider_convention",
            target=target,
            local_row=local_row,
            provider_row=provider_row,
            existing_path=existing_path,
            provider=attrs.get("provider"),
            adjustment=attrs.get("adjustment"),
        )
    required = ("open", "high", "low", "close", "adj_close", "volume")
    if any(field not in frame.columns or field not in existing.columns for field in required):
        return _fail_closed_revision(
            "incompatible_provider_convention",
            target=target,
            local_row=local_row,
            provider_row=provider_row,
            existing_path=existing_path,
            provider=attrs.get("provider"),
            adjustment=attrs.get("adjustment"),
        )
    provider_violation = session_ohlc_violation(
        open=provider_row.get("open"),
        high=provider_row.get("high"),
        low=provider_row.get("low"),
        close=provider_row.get("close"),
        symbol=symbol,
        session_date=target,
        source_layer="provider",
    )
    if provider_violation is not None:
        return _fail_closed_revision(
            "invalid_provider_session_ohlc",
            target=target,
            local_row=local_row,
            provider_row=provider_row,
            existing_path=existing_path,
            violation=provider_violation,
            provider=str(attrs.get("provider")),
            adjustment=str(attrs.get("adjustment")),
        )
    candidate = _splice_session_row(existing_path, target, provider_row)
    if (
        candidate is None
        or hashlib.sha256(candidate).hexdigest()
        == hashlib.sha256(existing_path.read_bytes()).hexdigest()
    ):
        return _fail_closed_revision(
            "incomplete_provider_session",
            target=target,
            local_row=local_row,
            provider_row=provider_row,
            existing_path=existing_path,
            provider=str(attrs.get("provider")),
            adjustment=str(attrs.get("adjustment")),
        )
    record = _correction_record(
        target=target,
        local_row=local_row,
        provider_row=provider_row,
        existing_path=existing_path,
        violation=None,
        candidate=candidate,
        provider="yfinance",
        adjustment="unadjusted",
    )
    return {
        "action": "promote",
        "reason": "same_session_provider_correction",
        "operationally_eligible": True,
        "same_session_correction": record,
        "candidate_bytes": candidate,
    }


def classify_provider_payload(
    frame: Optional[pd.DataFrame],
    symbol: str,
    asof: date,
    existing_path: Path,
) -> Dict[str, object]:
    """Decide whether a staged provider frame may replace a raw file. Never writes."""
    existing = _local_frame(existing_path)
    local_max = _local_max(existing_path)
    target = asof.isoformat()
    problems = _payload_problems(frame, symbol, asof, existing)
    staged_max = None if frame is None or frame.empty else str(_dates_of(frame).max())
    decision: Dict[str, object] = {
        "symbol": symbol.upper(),
        "index": symbol.startswith("^"),
        "local_max": local_max,
        "provider_max": staged_max,
        "problems": problems,
        "action": "keep",
        "reason": "unchanged",
        "operationally_eligible": False,
    }
    local_exact = local_max == target
    if problems:
        if existing is None and ("empty" in problems or staged_max is None or staged_max < target):
            decision["action"] = "skip"
            decision["reason"] = MISSING_EXPECTED_SESSION_BAR
        else:
            decision["action"] = "keep" if existing is not None else "skip"
            decision["reason"] = problems[0]
            if decision["action"] == "skip":
                decision["reason"] = MISSING_EXPECTED_SESSION_BAR
        decision["operationally_eligible"] = bool(local_exact)
        if local_exact and frame is not None and target in _date_list(frame):
            decision["operationally_eligible"] = False
            decision["counts_toward_coverage"] = _FAIL_CLOSED_COVERAGE
        return decision
    if staged_max is None or staged_max < target:
        decision["action"] = "skip" if not local_exact else "keep"
        decision["reason"] = (
            "provider_older_than_local" if local_exact else MISSING_EXPECTED_SESSION_BAR
        )
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    if staged_max > target:
        decision["action"] = "keep" if existing is not None else "skip"
        decision["reason"] = "future_or_partial_bar"
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    target_rows = 0 if frame is None else _date_list(frame).count(target)
    if existing is None:
        if target_rows != 1:
            decision["action"] = "skip"
            decision["reason"] = MISSING_EXPECTED_SESSION_BAR
            return decision
        non_session = _non_xnys_dates(_date_list(frame))
        if non_session is None:
            decision["action"] = "skip"
            decision["reason"] = "calendar_unavailable"
            return decision
        if non_session:
            decision["action"] = "skip"
            decision["reason"] = "non_session_bar"
            decision["non_session_dates"] = non_session
            return decision
        decision["action"] = "promote"
        decision["reason"] = "exact_session"
        decision["operationally_eligible"] = True
        decision["candidate_frame"] = _ordered_frame(frame)
        return decision
    local_dates = _date_list(existing)
    staged_dates = _date_list(frame)
    if not local_dates or not staged_dates or not local_max:
        decision["reason"] = "bad_dates"
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    if staged_max < local_max:
        decision["action"] = "keep"
        decision["reason"] = "provider_older_than_local"
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    provider_min = min(staged_dates)
    staged_set = set(staged_dates)
    local_set = set(local_dates)
    decision["provider_min"] = provider_min
    decision["local_rows"] = len(local_dates)
    decision["provider_rows"] = len(staged_dates)
    decision["omitted_prefix_count"] = sum(1 for day in local_dates if day < provider_min)
    decision["local_only_internal_dates"] = sorted(
        day for day in local_dates if provider_min <= day <= local_max and day not in staged_set
    )
    decision["provider_only_internal_dates"] = sorted(
        day for day in staged_dates if day <= local_max and day not in local_set
    )
    decision["overlap_conflicts"] = _overlap_conflicts(
        existing, frame, sorted(local_set & staged_set)
    )
    required = _required_newer_sessions(local_max, asof)
    if required is None:
        decision["action"] = "keep"
        decision["reason"] = "calendar_unavailable"
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    newer = sorted(day for day in staged_dates if day > local_max)
    decision["required_newer_sessions"] = required
    decision["appended_dates"] = []
    missing = [day for day in required if day not in staged_set]
    extra = [day for day in newer if day not in set(required)]
    if not required and not newer:
        if local_exact:
            decision.update(
                _classify_same_session_revision(existing, frame, symbol, asof, existing_path)
            )
            return decision
        decision["action"] = "keep"
        decision["reason"] = "unchanged"
        decision["operationally_eligible"] = False
        return decision
    if target_rows != 1 or missing:
        decision["action"] = "keep"
        decision["reason"] = "missing_required_session" if missing else MISSING_EXPECTED_SESSION_BAR
        decision["missing_required_sessions"] = missing
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    if extra:
        decision["action"] = "keep"
        decision["reason"] = "unexpected_newer_session"
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    merged = _merge_append(existing, frame, required)
    if merged is None:
        decision["action"] = "keep"
        decision["reason"] = "partial_bar"
        decision["operationally_eligible"] = bool(local_exact)
        return decision
    decision["action"] = "promote"
    decision["reason"] = "monotonic_append"
    decision["appended_dates"] = required
    decision["operationally_eligible"] = True
    decision["candidate_frame"] = merged
    return decision


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _atomic_replace_bytes(dest: Path, blob: bytes, replace_impl) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.sf37.tmp")
    with tmp.open("wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    if _sha256_bytes(tmp.read_bytes()) != _sha256_bytes(blob) or tmp.stat().st_size != len(blob):
        tmp.unlink(missing_ok=True)
        raise OSError(f"staged hash mismatch for {dest.name}")
    try:
        replace_impl(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    written = dest.read_bytes()
    if _sha256_bytes(written) != _sha256_bytes(blob) or len(written) != len(blob):
        raise OSError(f"promoted hash mismatch for {dest.name}")
    return _sha256_bytes(written)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    blob = json.dumps(payload, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace_bytes(path, blob, os.replace)


def apply_provider_fetch(
    *,
    root: Path,
    tickers: Sequence[str],
    asof: date,
    frames: Dict[str, Optional[pd.DataFrame]],
    replace_impl=None,
    before_promote=None,
) -> Dict[str, object]:
    """Stage provider frames, then promote only an exact-session coverage-safe set."""
    root = Path(root)
    replace_impl = replace_impl or os.replace
    symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
    decisions: List[Dict[str, object]] = []
    staged: Dict[str, bytes] = {}
    for symbol in symbols:
        path = root / "data" / "raw" / f"{symbol}.csv"
        frame = frames.get(symbol)
        decision = classify_provider_payload(frame, symbol, asof, path)
        candidate = decision.pop("candidate_frame", None)
        candidate_bytes = decision.pop("candidate_bytes", None)
        decisions.append(decision)
        if decision["action"] == "promote" and isinstance(candidate_bytes, bytes):
            staged[symbol] = candidate_bytes
        elif decision["action"] == "promote" and isinstance(candidate, pd.DataFrame):
            if decision["reason"] == "monotonic_append":
                staged[symbol] = _append_csv_bytes(
                    path, candidate, len(decision.get("appended_dates") or [])
                )
            else:
                staged[symbol] = candidate.to_csv(index=False, lineterminator="\n").encode("utf-8")

    tradables = [s for s in symbols if not s.startswith("^")]
    by_symbol = {str(d["symbol"]): d for d in decisions}
    valid = []
    for symbol in tradables:
        decision = by_symbol[symbol]
        local_exact = decision.get("local_max") == asof.isoformat()
        counts = decision.get("counts_toward_coverage", True) is not False
        if decision["action"] == "promote" or (local_exact and counts):
            valid.append(symbol)
    denominator = len(tradables)
    ratio = (len(valid) / denominator) if denominator else 0.0
    skips = [
        {
            "symbol": str(d["symbol"]),
            "session": asof.isoformat(),
            "reason": MISSING_EXPECTED_SESSION_BAR,
            "operationally_eligible": False,
        }
        for d in decisions
        if d["reason"] == MISSING_EXPECTED_SESSION_BAR
    ]
    receipt: Dict[str, object] = {
        "target_session": asof.isoformat(),
        "tradable_coverage": {
            "covered": len(valid),
            "denominator": denominator,
            "ratio": ratio,
            "minimum": MIN_TRADABLE_COVERAGE,
        },
        "decisions": decisions,
        "dated_skips": skips,
        "promoted": [],
        "production_raw_unchanged": True,
        "passed": False,
        "promotion_error": None,
    }
    skip_doc = {"target_session": asof.isoformat(), "skips": skips}
    if ratio + 1e-12 < MIN_TRADABLE_COVERAGE:
        receipt["failure"] = "coverage_below_minimum"
        _write_json(root / INPUT_SKIPS_REL, skip_doc)
        _write_json(root / FETCH_RECEIPT_REL, receipt)
        return receipt
    if before_promote is not None:
        try:
            before_promote()
        except Exception as exc:
            receipt["failure"] = "staged_before_promotion"
            receipt["promotion_error"] = f"{type(exc).__name__}: {exc}"
            receipt["durable_state"] = "no_promotion_attempted"
            _write_json(root / INPUT_SKIPS_REL, skip_doc)
            _write_json(root / FETCH_RECEIPT_REL, receipt)
            return receipt
    promoted: List[str] = []
    for symbol, blob in staged.items():
        dest = root / "data" / "raw" / f"{symbol}.csv"
        try:
            _atomic_replace_bytes(dest, blob, replace_impl)
        except Exception as exc:
            receipt["promoted"] = promoted
            receipt["production_raw_unchanged"] = not promoted
            receipt["promotion_error"] = f"{type(exc).__name__}: {exc}"
            receipt["failure"] = "atomic_replacement_failed"
            receipt["durable_state"] = "promotion_stopped_no_retry"
            _write_json(root / INPUT_SKIPS_REL, skip_doc)
            _write_json(root / FETCH_RECEIPT_REL, receipt)
            return receipt
        promoted.append(symbol)
    receipt["promoted"] = promoted
    receipt["production_raw_unchanged"] = not promoted
    receipt["passed"] = True
    _write_json(root / INPUT_SKIPS_REL, skip_doc)
    _write_json(root / FETCH_RECEIPT_REL, receipt)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tickers", nargs="*", default=None, help="Tickers (space or comma separated)."
    )
    ap.add_argument("--tickers-file", default=None, help="Path to a tickers file (one per line).")
    ap.add_argument("--start", default="2020-01-01")

    ap.add_argument("--asof", default=None, help="Override AS_OF_DATE (YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (exclusive). Default: AS_OF + 1 day.")

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--min-ok", type=int, default=1)
    ap.add_argument("--min-asof-ok", type=int, default=0)
    ap.add_argument(
        "--allow-data-lag-days",
        type=int,
        default=ALLOW_DATA_LAG_DAYS,
        help=f"Max calendar days latest bar may trail AS_OF and still count as ok (default: {ALLOW_DATA_LAG_DAYS}).",
    )

    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=30, help="yfinance timeout seconds")
    ap.add_argument(
        "--repo-root", default=".", help="Repository root. Defaults to the working directory."
    )

    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    # AS_OF + end
    if args.asof:
        asof_dt = pd.to_datetime(args.asof, errors="coerce")
        if pd.isna(asof_dt):
            raise SystemExit(f"Invalid --asof: {args.asof!r}")
        asof = asof_dt.date()
    else:
        asof = _compute_asof_date()

    end = args.end or _asof_to_end_exclusive(asof)

    # tickers
    tickers: List[str] = []
    if args.tickers:
        tickers = _split_tickers(args.tickers)
    if not tickers and args.tickers_file:
        tickers = _load_tickers_from_file(Path(args.tickers_file))
    if not tickers and DEFAULT_TICKERS_FILE.exists():
        tickers = _load_tickers_from_file(DEFAULT_TICKERS_FILE)
    if not tickers:
        raise SystemExit("No tickers provided and no default tickers file found.")

    allow_lag = max(0, int(args.allow_data_lag_days))
    print(
        f"[fetch_raw] asof={asof.isoformat()} end(exclusive)={end} start={args.start} "
        f"tickers={len(tickers)} retries={args.retries} timeout={args.timeout}s "
        f"allow_data_lag_days={allow_lag}"
    )

    ok_count = 0
    asof_ok = 0
    lagging_ok_count = 0
    lagging_exceeded_count = 0
    diag_rows: List[dict] = []
    frames: Dict[str, Optional[pd.DataFrame]] = {}

    for t in tickers:
        ok, msg, last_date, rows, source, frame = fetch_one_with_retries(
            t,
            start=args.start,
            end=end,
            retries=args.retries,
            timeout_s=args.timeout,
            verbose=args.verbose,
        )
        frames[t] = frame

        # 'ok' means a fresh download occurred; but ASOF coverage is based on last_date.
        if ok:
            ok_count += 1

        latest_dt = _parse_last_date(last_date)
        lag_days = _asof_lag_days(asof, latest_dt)
        reached_strict = bool(latest_dt and latest_dt >= asof)
        reached = _asof_within_tolerance(asof, latest_dt, allow_lag)
        if reached_strict:
            asof_ok += 1
        if reached and lag_days is not None and lag_days > 0:
            lagging_ok_count += 1
        elif not reached:
            if lag_days is not None and lag_days > allow_lag:
                lagging_exceeded_count += 1
            elif latest_dt is None:
                lagging_exceeded_count += 1

        print(f"[fetch_raw] {t}: {msg}")
        if reached and lag_days is not None and lag_days > 0:
            print(f"[fetch_raw] {t}: lag={lag_days} days (diagnostic only; not written)")

        diag_rows.append(
            {
                "ticker": t,
                "download_ok": bool(ok),
                "rows": int(rows),
                "last_date": last_date or "",
                "asof": asof.isoformat(),
                "reached_asof": bool(reached),
                "reached_asof_strict": bool(reached_strict),
                "lag_days": "" if lag_days is None else int(lag_days),
                "allow_data_lag_days": allow_lag,
                "start": args.start,
                "end_exclusive": end,
                "source": source,
                "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "message": msg,
            }
        )

    print(
        f"[fetch_raw] asof_ok={asof_ok}/{len(tickers)} lagging_ok_count={lagging_ok_count} "
        f"lagging_exceeded_count={lagging_exceeded_count} download_ok={ok_count}/{len(tickers)}"
    )

    receipt = apply_provider_fetch(root=root, tickers=tickers, asof=asof, frames=frames)
    summary_payload = {
        "asof": asof.isoformat(),
        "allow_data_lag_days": allow_lag,
        "asof_ok": asof_ok,
        "tickers": len(tickers),
        "lagging_ok_count": lagging_ok_count,
        "lagging_exceeded_count": lagging_exceeded_count,
        "download_ok": ok_count,
        "promotion_passed": bool(receipt["passed"]),
        "failure": receipt.get("failure"),
    }
    try:
        pd.DataFrame(diag_rows).to_csv(DIAG_PATH, index=False)
        SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"[fetch_raw] diagnostics: {DIAG_PATH}")
    except Exception:
        pass
    print(
        f"[fetch_raw] coverage={receipt['tradable_coverage']['covered']}/"
        f"{receipt['tradable_coverage']['denominator']} "
        f"promoted={len(receipt['promoted'])} passed={receipt['passed']}"
    )
    if receipt.get("failure") == "atomic_replacement_failed":
        return 5
    if not receipt["passed"]:
        return 4
    if ok_count < args.min_ok:
        return 2
    if args.min_asof_ok and asof_ok < args.min_asof_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
