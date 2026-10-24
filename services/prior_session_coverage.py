"""Read-only prior completed XNYS session coverage for protected start.

Uses ``required_signal_session`` and the processed-bar inspection D1 uses.
Does not fetch bars, write signals, stage a bundle, or start a process.
A heartbeat, signals file, fetch receipt, or freshness-refresh receipt is ignored.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.lifecycle_artifact_guard import COMMITTED_ASOF_COVERAGE

REFUSAL_DECISION = "START_REFUSED_PRIOR_SESSION_COVERAGE"
REFUSAL_CODE = "PRIOR_SESSION_COVERAGE"
THRESHOLD = float(COMMITTED_ASOF_COVERAGE)


def minimum_covered_count(tradable_count: int, threshold: float = THRESHOLD) -> int:
    """Smallest covered count whose ratio meets the committed 0.90 gate."""
    count = int(tradable_count)
    if count <= 0:
        return 0
    covered = 0
    while covered < count and (covered / float(count)) < float(threshold):
        covered += 1
    return covered


def _empty_report(**overrides: Any) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "ok": False,
        "cause": "coverage_below_threshold",
        "detail": "",
        "expected_session": "",
        "covered_count": 0,
        "required_count": 0,
        "minimum_covered": 0,
        "threshold": THRESHOLD,
        "coverage": 0.0,
        "missing_symbols": [],
        "unreadable_symbols": [],
        "invalid_symbols": [],
        "index_symbols": [],
        "index_excluded_count": 0,
        "symbol_reasons": {},
    }
    report.update(overrides)
    return report


def coverage_refusal_explanation(assessment: Dict[str, Any]) -> str:
    session = assessment.get("expected_session") or "unavailable"
    threshold = float(assessment.get("threshold") or THRESHOLD)
    return (
        "Start refused: prior completed XNYS session "
        f"{session} covers {int(assessment.get('covered_count') or 0)}/"
        f"{int(assessment.get('required_count') or 0)} tradable symbols "
        f"(threshold {threshold:.2f}, minimum {int(assessment.get('minimum_covered') or 0)}). "
        f"Cause: {assessment.get('cause') or 'coverage_below_threshold'}. "
        "No process was started."
    )


def _symbol_bucket(view: Any) -> str:
    from services.generate_signals import (
        SKIP_INVALID_PROCESSED_SESSION_BAR,
        SKIP_MISSING_EXPECTED_SESSION_BAR,
    )

    if view.ok:
        return "covered"
    if (
        view.path
        and view.reason == SKIP_INVALID_PROCESSED_SESSION_BAR
        and str(view.actual) == "MISSING"
    ):
        return "processed_unreadable"
    if view.reason == SKIP_MISSING_EXPECTED_SESSION_BAR or not view.path:
        return "missing_expected_session_bar"
    return "invalid_processed_session_bar"


def assess_prior_session_coverage(
    *,
    root: Path,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return prior-session coverage for ``root``. Read-only."""
    from services.generate_signals import (
        _inspect_processed_session,
        _is_index_ticker,
        _load_optional_tickers,
        compute_asof_policy,
    )
    from services.nyse_signal_session import required_signal_session

    root = Path(root)
    try:
        session = required_signal_session(now)
    except Exception as exc:
        return _empty_report(
            cause="calendar_unavailable",
            detail=f"{type(exc).__name__}: XNYS calendar unavailable",
        )
    if not isinstance(session, date):
        return _empty_report(
            cause="calendar_unavailable",
            detail="required_signal_session did not return a date",
        )

    tickers_path = root / "data" / "config" / "tickers.txt"
    if not tickers_path.exists():
        return _empty_report(
            cause="tradable_universe_missing",
            detail=str(tickers_path),
            expected_session=session.isoformat(),
        )
    if not tickers_path.is_file():
        return _empty_report(
            cause="tradable_universe_unreadable",
            detail=str(tickers_path),
            expected_session=session.isoformat(),
        )
    try:
        governed = _load_optional_tickers(str(tickers_path))
    except OSError as exc:
        return _empty_report(
            cause="tradable_universe_unreadable",
            detail=f"{type(exc).__name__}",
            expected_session=session.isoformat(),
        )

    tradable: List[str] = [symbol for symbol in governed if not _is_index_ticker(symbol)]
    indexes: List[str] = [symbol for symbol in governed if _is_index_ticker(symbol)]
    required = len(tradable)
    minimum = minimum_covered_count(required)
    base = {
        "expected_session": session.isoformat(),
        "required_count": required,
        "minimum_covered": minimum,
        "index_symbols": list(indexes),
        "index_excluded_count": len(indexes),
    }
    if required <= 0:
        return _empty_report(cause="no_tradable_tickers", detail="tickers.txt", **base)

    processed = root / "data" / "processed"
    if not processed.exists():
        return _empty_report(
            cause="processed_store_missing",
            detail=str(processed),
            missing_symbols=list(tradable),
            **base,
        )
    if not processed.is_dir():
        return _empty_report(
            cause="processed_unreadable",
            detail=str(processed),
            unreadable_symbols=list(tradable),
            **base,
        )

    processed_dir = str(processed)
    missing: List[str] = []
    unreadable: List[str] = []
    invalid: List[str] = []
    reasons: Dict[str, str] = {}
    latest: Dict[str, date] = {}
    for symbol in tradable:
        view = _inspect_processed_session(symbol, session, processed_dir=processed_dir)
        bucket = _symbol_bucket(view)
        reasons[symbol] = str(view.reason or "covered")
        if bucket == "covered":
            latest[symbol] = session
        elif bucket == "processed_unreadable":
            unreadable.append(symbol)
        elif bucket == "missing_expected_session_bar":
            missing.append(symbol)
        else:
            invalid.append(symbol)

    _effective, diag = compute_asof_policy(
        latest,
        session,
        tradable_tickers=tradable,
        index_tickers=indexes,
    )
    covered = sum(1 for symbol in tradable if symbol in latest)
    policy_covered = int(diag.get("have_session") or 0)
    if policy_covered != covered or int(diag.get("tradable_tickers") or 0) != required:
        return _empty_report(
            cause="coverage_policy_mismatch",
            detail="compute_asof_policy disagreed with processed-bar inspection",
            covered_count=covered,
            missing_symbols=missing,
            unreadable_symbols=unreadable,
            invalid_symbols=invalid,
            symbol_reasons=reasons,
            **base,
        )

    ratio = (covered / float(required)) if required else 0.0
    ok = covered >= minimum and ratio >= THRESHOLD
    if ok:
        cause = "coverage_ok"
    elif unreadable:
        cause = "processed_unreadable"
    else:
        cause = "coverage_below_threshold"
    return _empty_report(
        ok=ok,
        cause=cause,
        detail="",
        covered_count=covered,
        coverage=round(ratio, 4),
        missing_symbols=missing,
        unreadable_symbols=unreadable,
        invalid_symbols=invalid,
        symbol_reasons=reasons,
        **base,
    )
