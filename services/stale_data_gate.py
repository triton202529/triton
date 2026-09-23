# services/stale_data_gate.py
"""Exact-session raw gate.

A dated input skip may explain one missing symbol. Every other raw file must
end on the completed XNYS session. One fresh index file does not bless a
truncated tradable book.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
INPUT_SKIPS_REL = "data/results/daily_refresh_input_skips.json"
MISSING_EXPECTED_SESSION_BAR = "MISSING_EXPECTED_SESSION_BAR"


def _max_date(path: Path) -> date | None:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    col = "date" if "date" in frame.columns else ("Date" if "Date" in frame.columns else None)
    if col is None or frame.empty:
        return None
    parsed = pd.to_datetime(frame[col], errors="coerce").dropna()
    if parsed.empty:
        return None
    return parsed.max().date()


def load_dated_skips(root: Path, asof: date) -> set[str]:
    path = root / INPUT_SKIPS_REL
    if not path.is_file():
        return set()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if str(doc.get("target_session") or "") != asof.isoformat():
        return set()
    found = set()
    for row in doc.get("skips") or []:
        if not isinstance(row, dict):
            continue
        if row.get("reason") != MISSING_EXPECTED_SESSION_BAR:
            continue
        if str(row.get("session") or "") != asof.isoformat():
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            found.add(symbol)
    return found


def assess_raw_sessions(root: Path, asof: date, skips: set[str] | None = None) -> dict:
    root = Path(root)
    skipped = {s.upper() for s in (skips if skips is not None else load_dated_skips(root, asof))}
    raw_dir = root / "data" / "raw"
    csvs = [p for p in raw_dir.glob("*.csv") if not p.name.startswith(".")]
    if not csvs:
        return {"passed": False, "reason": "no_raw_csvs", "problems": [], "ok_symbols": []}
    problems = []
    ok_symbols = []
    for path in sorted(csvs):
        symbol = path.stem.upper()
        if symbol in skipped:
            continue
        latest = _max_date(path)
        if latest != asof:
            problems.append(
                {"symbol": symbol, "max_date": None if latest is None else latest.isoformat()}
            )
        else:
            ok_symbols.append(symbol)
    return {
        "passed": not problems,
        "reason": None if not problems else "exact_session_missing",
        "problems": problems,
        "ok_symbols": ok_symbols,
        "skipped": sorted(skipped),
        "target_session": asof.isoformat(),
    }


def main(max_age_days: int = 3) -> None:
    """Exit non-zero unless every non-skipped raw file is on the completed session.

    max_age_days is retained for callers and is not a write or pass authorization.
    """
    del max_age_days
    from services.market_calendar import last_completed_nyse_session

    asof = last_completed_nyse_session()
    result = assess_raw_sessions(Path.cwd(), asof)
    if result["reason"] == "no_raw_csvs":
        print("[stale_gate] NO RAW CSVs FOUND -> FAIL")
        sys.exit(2)
    print(
        f"[stale_gate] target_session={asof.isoformat()} ok={len(result['ok_symbols'])} "
        f"problems={len(result['problems'])} skipped={len(result['skipped'])}"
    )
    if not result["passed"]:
        print(f"[stale_gate] EXACT SESSION MISSING -> FAIL {result['problems'][:8]}")
        sys.exit(3)
    print("[stale_gate] OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
