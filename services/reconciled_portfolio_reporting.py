"""
reconciled_portfolio_reporting.py
Phase UIR-002 — read-only reconciled portfolio reporting engine.

Presentation and evidence assembly only.
No broker calls, no writes, no runtime controls, no competing accounting authority.
Financial arithmetic uses Decimal. Missing evidence stays Unavailable (never zero-filled).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from services.market_calendar import NY_TZ

REPORTING_TZ = NY_TZ
REPORTING_TZ_NAME = "America/New_York"

# portfolio_history.csv is date-only and mixed-producer — never a certified open/close.
PORTFOLIO_HISTORY_BOUNDARY_DISCLOSURE = (
    "date-only portfolio_history row; no certified session-open timestamp; "
    "no certified exchange-close timestamp; mixed producer history"
)

PERIOD_TYPES: Tuple[str, ...] = (
    "Daily",
    "Weekly",
    "Monthly",
    "Quarterly",
    "Half-year",
    "Yearly",
    "Custom",
)
DEFAULT_PERIOD = "Daily"

AVAIL_AVAILABLE = "AVAILABLE"
AVAIL_PARTIAL = "PARTIAL"
AVAIL_UNAVAILABLE = "UNAVAILABLE"
AVAIL_CONFLICTED = "CONFLICTED"

COMPLETENESS_COMPLETE = "COMPLETE"
COMPLETENESS_IN_PROGRESS = "IN_PROGRESS"
COMPLETENESS_PARTIAL = "PARTIAL_COVERAGE"
COMPLETENESS_UNAVAILABLE = "UNAVAILABLE"

RECON_PASS = "PASS"
RECON_PARTIAL = "PARTIAL"
RECON_FAIL = "FAIL"
RECON_UNAVAILABLE = "UNAVAILABLE"

FRESH_FRESH = "FRESH"
FRESH_STALE = "STALE"
FRESH_UNKNOWN = "UNKNOWN"

FRESH_MAX_AGE_MINUTES = Decimal("90")

CURRENCY_USD = "USD"

# Order statuses treated as broker execution outcomes (not TRITON decisions).
_FILLED_STATUSES = frozenset({"filled", "partially_filled", "partial_fill"})
_CANCEL_STATUSES = frozenset({"canceled", "cancelled", "expired"})
_REJECT_STATUSES = frozenset({"rejected", "failed", "suspended"})


@dataclass(frozen=True)
class ReportRequest:
    period_type: str = DEFAULT_PERIOD
    anchor_date: Optional[date] = None
    custom_start: Optional[date] = None
    custom_end: Optional[date] = None
    as_of: Optional[datetime] = None


@dataclass
class Metric:
    value: Optional[Decimal]
    availability: str
    currency: Optional[str] = None
    unit: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    method: Optional[str] = None
    reason: Optional[str] = None
    source_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": None if self.value is None else str(self.value),
            "availability": self.availability,
            "currency": self.currency,
            "unit": self.unit,
            "sources": list(self.sources),
            "method": self.method,
            "reason": self.reason,
            "source_timestamp": self.source_timestamp,
        }


def _d(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def _unavailable(
    reason: str,
    *,
    sources: Optional[Sequence[str]] = None,
    currency: Optional[str] = None,
    unit: Optional[str] = None,
) -> Metric:
    return Metric(
        value=None,
        availability=AVAIL_UNAVAILABLE,
        currency=currency,
        unit=unit,
        sources=list(sources or []),
        reason=reason,
    )


def _available(
    value: Decimal,
    *,
    sources: Sequence[str],
    method: str,
    currency: str = CURRENCY_USD,
    source_timestamp: Optional[str] = None,
) -> Metric:
    return Metric(
        value=value,
        availability=AVAIL_AVAILABLE,
        currency=currency,
        sources=list(sources),
        method=method,
        source_timestamp=source_timestamp,
    )


def _parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _results_dir(root: Path) -> Path:
    return root / "data" / "results"


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader if any((v or "").strip() for v in row.values())]
    except Exception:
        return []


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _load_portfolio_history(root: Path) -> List[Dict[str, Any]]:
    path = _results_dir(root) / "portfolio_history.csv"
    rows: List[Dict[str, Any]] = []
    for raw in _read_csv_dicts(path):
        d = _parse_date(raw.get("date"))
        if d is None:
            continue
        rows.append(
            {
                "date": d,
                "cash": _d(raw.get("cash")),
                "market_value": _d(raw.get("market_value")),
                "total_value": _d(raw.get("total_value")),
                "source": str(path),
            }
        )
    rows.sort(key=lambda r: r["date"])
    # One row per date (last wins) — mirrors aggregate_portfolio_history_daily intent.
    by_date: Dict[date, Dict[str, Any]] = {}
    for r in rows:
        by_date[r["date"]] = r
    return [by_date[k] for k in sorted(by_date)]


def market_calendar_available() -> Tuple[bool, Optional[str]]:
    """True only when exchange_calendars can load the XNYS calendar."""
    try:
        import exchange_calendars as xc  # type: ignore

        xc.get_calendar("XNYS")
        return True, None
    except Exception as exc:  # noqa: BLE001 — truthful unavailable state
        return False, f"{type(exc).__name__}: {exc}"


def is_nyse_trading_session(day: date) -> bool:
    """True only for a verified NYSE session. Never weekday-fallback when calendar missing."""
    ok, _ = market_calendar_available()
    if not ok:
        return False
    try:
        import exchange_calendars as xc  # type: ignore

        return bool(xc.get_calendar("XNYS").is_session(day))
    except Exception:
        return False


def latest_eligible_trading_session(*, now: datetime) -> Optional[date]:
    """Latest eligible trading session from exchange_calendars only (no weekday substitute)."""
    ok, _ = market_calendar_available()
    if not ok:
        return None
    now_ny = now.astimezone(REPORTING_TZ)
    # Prefer the calendar-backed completed-session helper (no weekday fallback path used here).
    try:
        from services.market_calendar import _try_exchange_calendars_session_info  # type: ignore

        info = _try_exchange_calendars_session_info(now_ny)
        if info is not None and getattr(info, "session_date", None) is not None:
            return info.session_date
    except Exception:
        pass
    try:
        import exchange_calendars as xc  # type: ignore

        cal = xc.get_calendar("XNYS")
        d = now_ny.date()
        for i in range(0, 21):
            cand = d - timedelta(days=i)
            if bool(cal.is_session(cand)):
                return cand
    except Exception:
        return None
    return None


def latest_supported_reporting_date(
    history: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
) -> Optional[date]:
    """Latest eligible trading session date (calendar-first; ignores weekend history rows)."""
    return latest_eligible_trading_session(now=now)


def _nyse_session_close_ny(day: date) -> Optional[datetime]:
    """Market close in America/New_York from exchange_calendars only (no invented close)."""
    ok, _ = market_calendar_available()
    if not ok or not is_nyse_trading_session(day):
        return None
    try:
        import exchange_calendars as xc  # type: ignore
        import pandas as pd  # type: ignore

        cal = xc.get_calendar("XNYS")
        sched = getattr(cal, "schedule", None)
        if sched is None:
            return None
        if callable(sched):
            frame = sched(start_date=day.isoformat(), end_date=day.isoformat())
        else:
            frame = sched.loc[day.isoformat() : day.isoformat()]
        if frame is None or len(frame) == 0:
            return None
        close_col = "market_close" if "market_close" in frame.columns else "close"
        close_ts = pd.Timestamp(frame.iloc[0][close_col])
        if close_ts.tzinfo is None:
            close_ts = close_ts.tz_localize("UTC")
        return close_ts.tz_convert(REPORTING_TZ).to_pydatetime()
    except Exception:
        return None


def _session_is_genuinely_incomplete(day: date, now: datetime) -> bool:
    """IN_PROGRESS gate: valid session, current market date, before session close."""
    now_ny = now.astimezone(REPORTING_TZ)
    if not is_nyse_trading_session(day):
        return False
    if day != now_ny.date():
        return False
    close_ny = _nyse_session_close_ny(day)
    if close_ny is None:
        return False
    return now_ny < close_ny


def resolve_period_bounds(
    request: ReportRequest,
    *,
    now: datetime,
    history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Deterministic period boundaries using NYSE session calendar (not history presence)."""
    period = str(request.period_type or DEFAULT_PERIOD).strip()
    if period not in PERIOD_TYPES:
        return {
            "ok": False,
            "error": f"Unsupported period type: {request.period_type!r}",
            "period_type": period,
            "start_date": None,
            "end_date": None,
            "completeness": COMPLETENESS_UNAVAILABLE,
            "is_market_session": False,
            "calendar_available": False,
        }

    cal_ok, cal_reason = market_calendar_available()
    if not cal_ok:
        return {
            "ok": False,
            "error": (
                "Market calendar unavailable — exchange_calendars is required for verified "
                "NYSE session selection. Weekday-only holiday logic is not used."
            ),
            "period_type": period,
            "start_date": None,
            "end_date": None,
            "completeness": COMPLETENESS_UNAVAILABLE,
            "is_market_session": False,
            "calendar_available": False,
            "calendar_reason": cal_reason,
        }

    now_ny = now.astimezone(REPORTING_TZ)
    today_ny = now_ny.date()
    latest_session = latest_eligible_trading_session(now=now)
    if latest_session is None:
        return {
            "ok": False,
            "error": "Market calendar available but no eligible NYSE session could be resolved.",
            "period_type": period,
            "start_date": None,
            "end_date": None,
            "completeness": COMPLETENESS_UNAVAILABLE,
            "is_market_session": False,
            "calendar_available": True,
        }

    if period == "Custom":
        start = request.custom_start
        end = request.custom_end
        if start is None or end is None:
            return {
                "ok": False,
                "error": "Custom range requires both start and end dates.",
                "period_type": period,
                "start_date": start,
                "end_date": end,
                "completeness": COMPLETENESS_UNAVAILABLE,
                "is_market_session": False,
            }
        if end < start:
            return {
                "ok": False,
                "error": "Custom end date earlier than start date.",
                "period_type": period,
                "start_date": start,
                "end_date": end,
                "completeness": COMPLETENESS_UNAVAILABLE,
                "is_market_session": False,
            }
        return {
            "ok": True,
            "period_type": period,
            "start_date": start,
            "end_date": end,
            "label": f"{start.isoformat()} → {end.isoformat()}",
            "completeness": COMPLETENESS_PARTIAL,
            "anchor_date": end,
            "is_market_session": is_nyse_trading_session(end),
            "calendar_available": True,
        }

    anchor = request.anchor_date or latest_session or today_ny

    if period == "Daily":
        # Calendar-first: never default Daily to a weekend/holiday history row.
        if request.anchor_date is None:
            day = latest_session
        else:
            day = request.anchor_date
            if not is_nyse_trading_session(day):
                # Non-session anchor → roll back to latest eligible session on/before anchor.
                day = latest_eligible_trading_session(
                    now=datetime.combine(day, time(23, 59), tzinfo=REPORTING_TZ)
                )
        if day is None:
            return {
                "ok": False,
                "error": "No eligible NYSE session could be resolved for Daily reporting.",
                "period_type": period,
                "start_date": None,
                "end_date": None,
                "completeness": COMPLETENESS_UNAVAILABLE,
                "is_market_session": False,
                "calendar_available": True,
            }
        is_session = is_nyse_trading_session(day)
        non_session_history = [
            r["date"].isoformat() for r in history if not is_nyse_trading_session(r["date"])
        ]
        # Evidence for the selected session (history presence alone never creates IN_PROGRESS).
        has_session_evidence = any(r["date"] == day for r in history)
        if is_session and _session_is_genuinely_incomplete(day, now) and has_session_evidence:
            completeness = COMPLETENESS_IN_PROGRESS
        elif is_session:
            # Completed session without certified open/close boundaries.
            completeness = COMPLETENESS_PARTIAL
        else:
            completeness = COMPLETENESS_UNAVAILABLE
        return {
            "ok": True,
            "period_type": period,
            "start_date": day,
            "end_date": day,
            "label": day.isoformat(),
            "completeness": completeness,
            "anchor_date": day,
            "is_market_session": is_session,
            "non_session_history_observations": non_session_history,
            "calendar_available": True,
        }

    if period == "Weekly":
        # ISO week Mon–Sun window; trading sessions are those history dates inside.
        start = anchor - timedelta(days=anchor.weekday())
        end = start + timedelta(days=6)
        return {
            "ok": True,
            "period_type": period,
            "start_date": start,
            "end_date": end,
            "label": f"Week of {start.isoformat()}",
            "completeness": COMPLETENESS_PARTIAL,
            "anchor_date": anchor,
            "is_market_session": False,
        }

    if period == "Monthly":
        start = date(anchor.year, anchor.month, 1)
        if anchor.month == 12:
            end = date(anchor.year, 12, 31)
        else:
            end = date(anchor.year, anchor.month + 1, 1) - timedelta(days=1)
        return {
            "ok": True,
            "period_type": period,
            "start_date": start,
            "end_date": end,
            "label": f"{start.year}-{start.month:02d}",
            "completeness": COMPLETENESS_PARTIAL,
            "anchor_date": anchor,
            "is_market_session": False,
        }

    if period == "Quarterly":
        q = (anchor.month - 1) // 3
        start_month = q * 3 + 1
        start = date(anchor.year, start_month, 1)
        end_month = start_month + 2
        if end_month == 12:
            end = date(anchor.year, 12, 31)
        else:
            end = date(anchor.year, end_month + 1, 1) - timedelta(days=1)
        return {
            "ok": True,
            "period_type": period,
            "start_date": start,
            "end_date": end,
            "label": f"{anchor.year} Q{q + 1}",
            "completeness": COMPLETENESS_PARTIAL,
            "anchor_date": anchor,
            "is_market_session": False,
        }

    if period == "Half-year":
        if anchor.month <= 6:
            start, end, label = (
                date(anchor.year, 1, 1),
                date(anchor.year, 6, 30),
                f"{anchor.year} H1",
            )
        else:
            start, end, label = (
                date(anchor.year, 7, 1),
                date(anchor.year, 12, 31),
                f"{anchor.year} H2",
            )
        return {
            "ok": True,
            "period_type": period,
            "start_date": start,
            "end_date": end,
            "label": label,
            "completeness": COMPLETENESS_PARTIAL,
            "anchor_date": anchor,
            "is_market_session": False,
        }

    # Yearly
    start, end = date(anchor.year, 1, 1), date(anchor.year, 12, 31)
    return {
        "ok": True,
        "period_type": period,
        "start_date": start,
        "end_date": end,
        "label": str(anchor.year),
        "completeness": COMPLETENESS_PARTIAL,
        "anchor_date": anchor,
        "is_market_session": False,
    }


def _history_in_range(
    history: Sequence[Mapping[str, Any]], start: date, end: date
) -> List[Dict[str, Any]]:
    return [dict(r) for r in history if start <= r["date"] <= end]


def _row_on_or_before(history: Sequence[Mapping[str, Any]], day: date) -> Optional[Dict[str, Any]]:
    prior = [r for r in history if r["date"] <= day]
    return dict(prior[-1]) if prior else None


def _advisory_mtime(path: Path) -> Optional[str]:
    """Filesystem mtime for advisory evidence only — never establishes FRESH."""
    try:
        if path.is_file():
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None
    return None


def classify_source_freshness(
    *,
    as_of: datetime,
    source_name: str,
    source_path: str,
    content_timestamp: Optional[datetime] = None,
    date_only_value: Optional[date] = None,
    filesystem_mtime: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Per-source freshness using exact timestamp ordering (no clock-skew tolerance).

    Date-only values → UNKNOWN (no synthetic time).
    content_timestamp > as_of → UNKNOWN, excluded from as-of freshness, difference disclosed.
    Missing aware content timestamp → UNKNOWN (mtime advisory only; never FRESH).
    """
    as_of_utc = as_of.astimezone(timezone.utc)
    mtime_iso = (
        None if filesystem_mtime is None else filesystem_mtime.astimezone(timezone.utc).isoformat()
    )
    base = {
        "source": source_name,
        "path": source_path,
        "filesystem_mtime_advisory": mtime_iso,
        "excluded_from_as_of_freshness": False,
        "post_as_of": False,
        "date_only": False,
    }

    if date_only_value is not None and content_timestamp is None:
        return {
            **base,
            "status": FRESH_UNKNOWN,
            "age_minutes": None,
            "minutes_after_as_of": None,
            "source_timestamp": date_only_value.isoformat(),
            "date_only": True,
            "reason": (
                "Date-only portfolio_history value — time-aware freshness UNKNOWN "
                "(no synthetic timestamp invented)."
            ),
        }

    if content_timestamp is None:
        return {
            **base,
            "status": FRESH_UNKNOWN,
            "age_minutes": None,
            "minutes_after_as_of": None,
            "source_timestamp": None,
            "reason": (
                "No timezone-aware content timestamp — freshness UNKNOWN. "
                "Filesystem modification time is advisory only and cannot establish FRESH."
            ),
        }

    src = content_timestamp.astimezone(timezone.utc)
    if src.tzinfo is None:
        return {
            **base,
            "status": FRESH_UNKNOWN,
            "age_minutes": None,
            "minutes_after_as_of": None,
            "source_timestamp": content_timestamp.isoformat(),
            "reason": "Content timestamp lacks timezone — freshness UNKNOWN.",
        }

    if src > as_of_utc:
        ahead = Decimal(str((src - as_of_utc).total_seconds())) / Decimal("60")
        ahead_q = ahead.quantize(Decimal("0.1"))
        return {
            **base,
            "status": FRESH_UNKNOWN,
            "age_minutes": None,
            "minutes_after_as_of": str(ahead_q),
            "source_timestamp": src.isoformat(),
            "post_as_of": True,
            "excluded_from_as_of_freshness": True,
            "reason": (
                f"Post-as-of / future-dated evidence: source timestamp is {ahead_q} minutes "
                "after report as_of — excluded from as-of freshness; not FRESH."
            ),
        }

    age_min = Decimal(str((as_of_utc - src).total_seconds())) / Decimal("60")
    age_q = age_min.quantize(Decimal("0.1"))
    if age_min <= FRESH_MAX_AGE_MINUTES:
        status = FRESH_FRESH
        reason = f"{age_q} minutes old (threshold {FRESH_MAX_AGE_MINUTES} minutes)"
    else:
        status = FRESH_STALE
        reason = f"{age_q} minutes old — exceeds {FRESH_MAX_AGE_MINUTES} minute freshness threshold"
    return {
        **base,
        "status": status,
        "age_minutes": str(age_q),
        "minutes_after_as_of": None,
        "source_timestamp": src.isoformat(),
        "reason": reason,
    }


def aggregate_report_freshness(source_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Conservative report-level freshness: UNKNOWN > STALE > FRESH."""
    sources = [dict(s) for s in source_results]
    if not sources:
        return {
            "status": FRESH_UNKNOWN,
            "age_minutes": None,
            "reason": "No applicable sources for freshness evaluation",
            "source_timestamp": None,
            "sources": [],
            "aggregation": "conservative_any_unknown_or_stale",
        }
    statuses = [str(s.get("status") or FRESH_UNKNOWN) for s in sources]
    if any(s == FRESH_UNKNOWN for s in statuses):
        status = FRESH_UNKNOWN
        reason = "Report freshness UNKNOWN — at least one applicable source has UNKNOWN freshness"
    elif any(s == FRESH_STALE for s in statuses):
        status = FRESH_STALE
        reason = "Report freshness STALE — at least one applicable source is STALE"
    elif all(s == FRESH_FRESH for s in statuses):
        status = FRESH_FRESH
        reason = (
            "All applicable sources have valid timezone-aware timestamps within freshness threshold"
        )
    else:
        status = FRESH_UNKNOWN
        reason = "Report freshness UNKNOWN — unrecognized per-source freshness status"
    # Representative age: max age among non-excluded FRESH/STALE sources only (never negative clamp).
    ages: List[Decimal] = []
    for s in sources:
        if s.get("excluded_from_as_of_freshness"):
            continue
        if s.get("status") in (FRESH_FRESH, FRESH_STALE) and s.get("age_minutes") is not None:
            try:
                ages.append(Decimal(str(s["age_minutes"])))
            except (InvalidOperation, ValueError):
                pass
    return {
        "status": status,
        "age_minutes": None if not ages else str(max(ages)),
        "reason": reason,
        "source_timestamp": None,
        "sources": sources,
        "aggregation": "conservative_any_unknown_or_stale",
    }


def _classify_freshness(
    *,
    source_ts: Optional[datetime],
    now: datetime,
) -> Tuple[str, Optional[Decimal], Optional[str]]:
    """Backward-compatible wrapper around classify_source_freshness (single source)."""
    row = classify_source_freshness(
        as_of=now,
        source_name="legacy",
        source_path="",
        content_timestamp=source_ts,
    )
    age = None if row.get("age_minutes") is None else Decimal(str(row["age_minutes"]))
    return str(row["status"]), age, str(row.get("reason") or "")


def _order_event_date(row: Mapping[str, str]) -> Optional[date]:
    for key in ("filled_at", "submitted_at", "created_at", "updated_at", "canceled_at"):
        dt = _parse_dt(row.get(key))
        if dt is not None:
            return dt.astimezone(REPORTING_TZ).date()
    return _parse_date(row.get("snapshot_ts") or row.get("date"))


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_activity(
    *,
    root: Path,
    start: date,
    end: date,
) -> Dict[str, Any]:
    results = _results_dir(root)
    orders_path = results / "recent_orders.csv"
    audit_path = results / "paper_execution_audit.csv"
    orders = _read_csv_dicts(orders_path)
    audits = _read_csv_dicts(audit_path)

    linked_ids = set()
    for a in audits:
        for key in ("submitted_order_id", "order_id", "broker_order_id"):
            oid = str(a.get(key) or "").strip()
            if oid:
                linked_ids.add(oid)

    rows_out: List[Dict[str, Any]] = []
    counts = {
        "orders_in_period": 0,
        "filled": 0,
        "partially_filled": 0,
        "canceled": 0,
        "rejected": 0,
        "trades": 0,  # filled/partial only
        "unlinked_broker_activity": 0,
        "triton_linked": 0,
    }
    # recent_orders.csv is a broker list_orders(status=all, limit=N) window
    # (default N=200 via snapshot_live_orders) — never complete period history.
    coverage_complete = False
    coverage_kind = "LIMITED_RECENT_WINDOW"

    for row in orders:
        event_day = _order_event_date(row)
        if event_day is None or event_day < start or event_day > end:
            continue
        status = _normalize_status(row.get("status"))
        qty = _d(row.get("qty"))
        filled_qty = _d(row.get("filled_qty")) or Decimal("0")
        remaining = None
        if qty is not None:
            remaining = qty - filled_qty
        oid = str(row.get("id") or "").strip()
        client_id = str(row.get("client_order_id") or "")
        linked = bool(oid and oid in linked_ids) or client_id.startswith("triton-")
        attribution = "TRITON_LINKED" if linked else "UNLINKED_BROKER_ACTIVITY"
        if linked:
            counts["triton_linked"] += 1
        else:
            counts["unlinked_broker_activity"] += 1

        counts["orders_in_period"] += 1
        is_trade = False
        if status == "filled":
            counts["filled"] += 1
            is_trade = True
        elif status in ("partially_filled", "partial_fill"):
            counts["partially_filled"] += 1
            is_trade = True
        elif status in _CANCEL_STATUSES:
            counts["canceled"] += 1
        elif status in _REJECT_STATUSES:
            counts["rejected"] += 1
        if is_trade:
            counts["trades"] += 1

        rows_out.append(
            {
                "symbol": row.get("symbol") or "UNKNOWN",
                "side": row.get("side") or "UNKNOWN",
                "status": status or "UNKNOWN",
                "requested_qty": None if qty is None else str(qty),
                "filled_qty": str(filled_qty),
                "remaining_qty": None if remaining is None else str(remaining),
                "filled_avg_price": (
                    None
                    if _d(row.get("filled_avg_price")) is None
                    else str(_d(row.get("filled_avg_price")))
                ),
                "submitted_at": row.get("submitted_at"),
                "filled_at": row.get("filled_at"),
                "order_id": oid or None,
                "attribution": attribution,
                "is_trade": is_trade,
                "sources": [str(orders_path)],
            }
        )

    # Decision activity (TRITON) — separate category; not counted as trades.
    decisions: List[Dict[str, Any]] = []
    for a in audits:
        ts = _parse_dt(a.get("timestamp"))
        day = ts.astimezone(REPORTING_TZ).date() if ts else _parse_date(a.get("date"))
        if day is None or day < start or day > end:
            continue
        decisions.append(
            {
                "symbol": a.get("symbol") or "UNKNOWN",
                "action": a.get("action") or a.get("lifecycle_state") or "UNKNOWN",
                "status": a.get("status") or "UNKNOWN",
                "qty": a.get("qty"),
                "timestamp": a.get("timestamp"),
                "submitted_order_id": a.get("submitted_order_id"),
                "attribution": "TRITON_DECISION",
                "sources": [str(audit_path)],
            }
        )

    if not orders_path.is_file():
        activity_availability = AVAIL_UNAVAILABLE
        no_activity_message = "Activity unavailable or incomplete for this period."
        activity_note = "Missing recent_orders.csv"
        coverage_kind = "MISSING"
        empty_broker = (
            "No broker order rows were found in the available local evidence for this period."
        )
        empty_triton = (
            "No TRITON decision-audit rows were found in the available local evidence "
            "for this period."
        )
    else:
        # Limited recent-order window — never claim complete period totals.
        activity_availability = AVAIL_PARTIAL
        no_activity_message = (
            "Recorded activity under partial coverage — recent_orders.csv is a "
            "limited broker recent-order window (not complete period history). "
            "Zero counts below are observations inside that limited window only, "
            "not complete period totals."
        )
        activity_note = (
            f"Counts reflect rows inside the selected dates that appear in the "
            f"current recent_orders snapshot only (coverage={coverage_kind}). "
            "Missing activity is not treated as zero; limited-window zeros are not "
            "complete period totals."
        )
        empty_broker = (
            "No broker order rows were found in the available local evidence for this "
            "period (limited recent-order window; not a complete-period assertion)."
        )
        empty_triton = (
            "No TRITON decision-audit rows were found in the available local evidence "
            "for this period."
        )

    return {
        "availability": activity_availability,
        "coverage_complete": coverage_complete,
        "coverage_kind": coverage_kind,
        "no_activity_message": no_activity_message,
        "note": activity_note,
        "counts": {k: counts[k] for k in counts},
        "counts_are_complete_period_totals": False,
        "broker_orders": rows_out,
        "triton_decisions": decisions,
        "empty_state_broker": empty_broker,
        "empty_state_triton": empty_triton,
        "sources": [str(orders_path), str(audit_path)],
    }


def _build_holdings(*, root: Path, end: date, now: datetime) -> Dict[str, Any]:
    path = _results_dir(root) / "positions_snapshot.csv"
    rows = _read_csv_dicts(path)
    # positions_snapshot.csv alone never proves a certified selected-period ending boundary.
    is_period_ending_boundary = False
    if not path.is_file():
        return {
            "availability": AVAIL_UNAVAILABLE,
            "reason": "Missing positions_snapshot.csv",
            "section_title": "Latest position snapshot",
            "is_period_ending_boundary": False,
            "disclosure": (
                "No positions snapshot is available. Beginning quantities, ending holdings, "
                "and net quantity changes remain Unavailable."
            ),
            "columns": [],
            "rows": [],
            "sources": [str(path)],
        }
    if not rows:
        return {
            "availability": AVAIL_PARTIAL,
            "reason": "Positions snapshot present but empty",
            "section_title": "Latest position snapshot",
            "is_period_ending_boundary": False,
            "disclosure": (
                "Positions snapshot is empty. Beginning quantities, ending holdings, "
                "and net quantity changes remain Unavailable."
            ),
            "columns": [
                "Symbol",
                "Latest snapshot quantity",
                "Latest snapshot market value",
                "Latest snapshot cost basis",
                "Latest snapshot unrealized P/L",
                "Evidence status",
                "Notes",
            ],
            "rows": [],
            "sources": [str(path)],
        }

    snap_ts = _parse_dt(rows[0].get("snapshot_ts"))
    snap_day = _parse_date(rows[0].get("date")) or (
        snap_ts.astimezone(REPORTING_TZ).date() if snap_ts else None
    )
    historical = snap_day is not None and snap_day != end
    non_session_snap = snap_day is not None and not is_nyse_trading_session(snap_day)
    snap_ts_iso = None if snap_ts is None else snap_ts.isoformat()
    snap_day_iso = None if snap_day is None else snap_day.isoformat()

    if is_period_ending_boundary:
        qty_col = "Ending quantity"
        mv_col = "Ending market value"
        cb_col = "Authoritative cost basis"
        upl_col = "Ending unrealized P/L"
        section_title = "Period-ending holdings"
        disclosure = (
            f"Authoritative period-ending holdings for {end.isoformat()} "
            f"(snapshot_ts={snap_ts_iso or 'UNKNOWN'})."
        )
        row_note = "Beginning Unavailable · net quantity change Unavailable."
    else:
        qty_col = "Latest snapshot quantity"
        mv_col = "Latest snapshot market value"
        cb_col = "Latest snapshot cost basis"
        upl_col = "Latest snapshot unrealized P/L"
        section_title = "Latest position snapshot"
        disclosure = (
            f"Position snapshot timestamp: {snap_ts_iso or 'UNKNOWN'}. "
            f"Snapshot date: {snap_day_iso or 'UNKNOWN'}. "
            f"Selected report period end: {end.isoformat()}. "
            "This snapshot is not a certified selected-session close and cannot establish "
            "ending holdings, period quantity change, selected-session market value, "
            "selected-session unrealized P/L, or holding-level period performance. "
            "Beginning quantities are Unavailable; net quantity change is Unavailable."
        )
        if historical or non_session_snap:
            disclosure += (
                f" Snapshot date {snap_day_iso} is not the certified "
                f"{end.isoformat()} session close."
            )
        row_note = "Beginning Unavailable · net qty change Unavailable."

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        sym = r.get("symbol") or r.get("ticker") or "UNKNOWN"
        qty = _d(r.get("qty"))
        mv = _d(r.get("market_value") or r.get("value"))
        cb = _d(r.get("cost_basis"))
        upl = _d(r.get("unrealized_pl"))
        out_rows.append(
            {
                "Symbol": sym,
                "Beginning quantity": None,  # never reconstructed without opening snapshot
                "Net quantity change": None,
                qty_col: None if qty is None else str(qty),
                mv_col: None if mv is None else str(mv),
                cb_col: None if cb is None else str(cb),
                "Period realized P/L": None,
                upl_col: None if upl is None else str(upl),
                "Supported period performance": None,
                # Preserve exact signed unrealized text for negative distinction.
                "Evidence status": AVAIL_PARTIAL,
                "Notes": row_note,
            }
        )

    return {
        "availability": AVAIL_PARTIAL,
        "reason": disclosure,
        "section_title": section_title,
        "disclosure": disclosure,
        "is_period_ending_boundary": is_period_ending_boundary,
        "snapshot_date": snap_day_iso,
        "snapshot_ts": snap_ts_iso,
        "selected_period_end": end.isoformat(),
        "period_relationship": (
            "certified_period_ending_boundary"
            if is_period_ending_boundary
            else "latest_non_boundary_snapshot"
        ),
        "columns": [
            "Symbol",
            qty_col,
            mv_col,
            cb_col,
            upl_col,
            "Evidence status",
            "Notes",
        ],
        "rows": out_rows,
        "sources": [str(path)],
        "as_of_compatible": False if not is_period_ending_boundary else (snap_day == end),
    }


def _period_realized_pl(*, root: Path) -> Metric:
    """Realized P/L is withheld unless a governed accounting authority is proven.

    `trade_outcomes.csv` is produced by services.build_trade_outcomes as standalone
    local FIFO analytics from recent_orders / live_orders_log. It is derivative /
    advisory — not a formally approved period-accounting authority. PARTIAL existence
    of that artifact must not promote it to authoritative realized P/L.
    """
    path = _results_dir(root) / "trade_outcomes.csv"
    return _unavailable(
        (
            "Unavailable — no governed realized-P/L authority for this report. "
            "trade_outcomes.csv is a local FIFO derivative from "
            "services.build_trade_outcomes (recent_orders / live_orders_log), "
            "not an approved broker tax-lot or governed period-accounting source."
        ),
        sources=[str(path)],
        currency=CURRENCY_USD,
    )


def _reconcile(
    *,
    opening: Optional[Mapping[str, Any]],
    ending: Optional[Mapping[str, Any]],
    activity: Mapping[str, Any],
    holdings: Mapping[str, Any],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(
        name: str,
        *,
        status: str,
        expected: Any = None,
        observed: Any = None,
        difference: Any = None,
        affected: str = "account",
        sources: Optional[Sequence[str]] = None,
        explanation: str,
    ) -> None:
        checks.append(
            {
                "check": name,
                "status": status,
                "expected": None if expected is None else str(expected),
                "observed": None if observed is None else str(observed),
                "difference": None if difference is None else str(difference),
                "affected": affected,
                "sources": list(sources or []),
                "explanation": explanation,
            }
        )

    # Internal identity of a dated portfolio_history observation — not period-end equity.
    obs_date = ending.get("date") if ending else None
    obs_date_s = (
        obs_date.isoformat()
        if isinstance(obs_date, date)
        else (str(obs_date) if obs_date is not None else None)
    )
    if ending and ending.get("cash") is not None and ending.get("market_value") is not None:
        expected = ending["cash"] + ending["market_value"]
        observed = ending.get("total_value")
        identity_base = {
            "observation_date": obs_date_s,
            "establishes_session_close": False,
            "establishes_period_ending_equity": False,
            "establishes_complete_boundary_coverage": False,
            "establishes_period_pl": False,
            "establishes_investment_return": False,
        }
        if observed is None:
            checks.append(
                {
                    "check": "recorded_observation_equity_identity",
                    "status": RECON_UNAVAILABLE,
                    "expected": str(expected),
                    "observed": None,
                    "difference": None,
                    "affected": "account",
                    "sources": [ending.get("source")],
                    "explanation": (
                        f"Recorded-observation identity for portfolio_history date "
                        f"{obs_date_s or 'UNKNOWN'}: total_value missing; "
                        "cash+market_value identity not verified. This check does not establish "
                        "official session-close equity, selected-period ending equity, complete "
                        "boundary coverage, period P/L, or investment return."
                    ),
                    **identity_base,
                }
            )
        else:
            diff = observed - expected
            checks.append(
                {
                    "check": "recorded_observation_equity_identity",
                    "status": RECON_PASS if diff == 0 else RECON_FAIL,
                    "expected": str(expected),
                    "observed": str(observed),
                    "difference": str(diff),
                    "affected": "account",
                    "sources": [ending.get("source")],
                    "explanation": (
                        f"Internal recorded-observation identity for portfolio_history date "
                        f"{obs_date_s or 'UNKNOWN'}: recorded total_value "
                        f"{'equals' if diff == 0 else 'does not equal'} recorded cash + "
                        "recorded market value. This does not establish official session-close "
                        "equity, selected-period ending equity, complete boundary coverage, "
                        "period P/L, or investment return."
                    ),
                    **identity_base,
                }
            )
    else:
        add(
            "recorded_observation_equity_identity",
            status=RECON_UNAVAILABLE,
            explanation=(
                "Insufficient recorded cash/market_value evidence for an internal "
                "portfolio_history observation identity. This does not establish "
                "official session-close or selected-period ending equity."
            ),
            sources=[ending.get("source")] if ending else [],
        )
    # Authoritative selected-period ending equity identity remains unavailable.
    add(
        "ending_equity_identity",
        status=RECON_UNAVAILABLE,
        explanation=(
            "Authoritative selected-period ending equity identity is Unavailable — "
            "no certified session-close / period-end equity boundary exists. "
            "Do not treat recorded_observation_equity_identity as ending_equity_identity."
        ),
        sources=[ending.get("source")] if ending else [],
    )

    # Position qty chain requires opening holdings — unavailable without historical snapshots.
    add(
        "position_quantity_chain",
        status=RECON_UNAVAILABLE,
        explanation=(
            "Cannot verify ending qty = opening qty + buys − sells without authoritative "
            "opening position snapshots for the period."
        ),
        sources=list(holdings.get("sources") or []),
    )

    if activity.get("availability") == AVAIL_UNAVAILABLE:
        add(
            "order_evidence_coverage",
            status=RECON_UNAVAILABLE,
            explanation="Broker order evidence missing for the period.",
            sources=list(activity.get("sources") or []),
        )
    else:
        add(
            "order_evidence_coverage",
            status=RECON_PARTIAL,
            explanation=(
                "Broker order rows were read from local recent_orders.csv; "
                "no independent second ledger was compared."
            ),
            sources=list(activity.get("sources") or []),
        )

    statuses = {c["status"] for c in checks}
    if RECON_FAIL in statuses:
        overall = RECON_FAIL
    elif all(s == RECON_UNAVAILABLE for s in statuses):
        overall = RECON_UNAVAILABLE
    elif (
        RECON_PASS in statuses
        and RECON_UNAVAILABLE not in statuses
        and RECON_PARTIAL not in statuses
    ):
        overall = RECON_PASS
    elif RECON_PASS in statuses or RECON_PARTIAL in statuses:
        overall = RECON_PARTIAL
    else:
        overall = RECON_UNAVAILABLE

    # Missing evidence never yields PASS alone when required checks are unavailable.
    if overall == RECON_PASS and RECON_UNAVAILABLE in statuses:
        overall = RECON_PARTIAL

    return {
        "status": overall,
        "checks": checks,
        "discrepancies": [c for c in checks if c["status"] == RECON_FAIL],
    }


def build_reconciled_report(
    request: Optional[ReportRequest] = None,
    *,
    root: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a deterministic read-only reconciled report. Performs no I/O writes."""
    req = request or ReportRequest()
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    as_of = now or req.as_of or datetime.now(tz=timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    history = _load_portfolio_history(project_root)
    bounds = resolve_period_bounds(req, now=as_of, history=history)
    sources_used: List[str] = [str(_results_dir(project_root) / "portfolio_history.csv")]

    if not bounds.get("ok"):
        return {
            "ok": False,
            "error": bounds.get("error"),
            "period_type": req.period_type,
            "reporting_timezone": REPORTING_TZ_NAME,
            "as_of": as_of.astimezone(REPORTING_TZ).isoformat(),
            "availability": AVAIL_UNAVAILABLE,
            "completeness": COMPLETENESS_UNAVAILABLE,
            "reconciliation": {"status": RECON_UNAVAILABLE, "checks": [], "discrepancies": []},
            "freshness": {"status": FRESH_UNKNOWN, "reason": "Invalid report request"},
            "metrics": {},
            "activity": {},
            "holdings": {},
            "chart": {"available": False, "reason": "Invalid report request", "points": []},
            "evidence": {"sources": sources_used, "missing_requirements": [bounds.get("error")]},
            "writes_performed": False,
            "network_calls_performed": False,
        }

    start: date = bounds["start_date"]
    end: date = bounds["end_date"]
    in_range = _history_in_range(history, start, end)
    # Recorded observations from date-only portfolio_history (not certified boundaries).
    previous_recorded_row = _row_on_or_before(history, start - timedelta(days=1))
    latest_recorded_row = dict(in_range[-1]) if in_range else _row_on_or_before(history, end)
    if previous_recorded_row is None and in_range and len(in_range) >= 2:
        previous_recorded_row = dict(in_range[0])
        latest_recorded_row = dict(in_range[-1])

    # Coverage / availability — independent of freshness.
    missing: List[str] = []
    if not history:
        missing.append("portfolio_history.csv has no usable rows")
    missing.append(
        "Authoritative session-open/session-close equity boundaries unavailable "
        f"({PORTFOLIO_HISTORY_BOUNDARY_DISCLOSURE})"
    )
    missing.append(
        "Order/fill coverage limited to recent_orders broker window — not complete period history"
    )
    non_session_obs = list(bounds.get("non_session_history_observations") or [])
    if non_session_obs:
        missing.append(
            "Non-session portfolio_history observation(s) present and labeled as such: "
            + ", ".join(non_session_obs)
        )

    completeness = str(bounds.get("completeness") or COMPLETENESS_PARTIAL)
    # Never allow IN_PROGRESS on non-sessions (weekend/holiday).
    if completeness == COMPLETENESS_IN_PROGRESS and not bounds.get("is_market_session"):
        completeness = COMPLETENESS_PARTIAL
    if completeness == COMPLETENESS_IN_PROGRESS and not _session_is_genuinely_incomplete(
        end, as_of
    ):
        completeness = COMPLETENESS_PARTIAL

    if not history:
        availability = AVAIL_UNAVAILABLE
        if completeness != COMPLETENESS_IN_PROGRESS:
            completeness = COMPLETENESS_UNAVAILABLE
    elif (
        not bounds.get("is_market_session")
        and start == end
        and bounds.get("period_type") == "Daily"
    ):
        availability = AVAIL_PARTIAL
        completeness = COMPLETENESS_UNAVAILABLE
    else:
        availability = AVAIL_PARTIAL

    # Metrics — certified open/close boundaries unavailable for date-only history.
    ph_source = str(_results_dir(project_root) / "portfolio_history.csv")
    opening_equity = _unavailable(
        (
            "Unavailable — no authoritative session-open equity boundary. "
            f"{PORTFOLIO_HISTORY_BOUNDARY_DISCLOSURE}."
        ),
        sources=[ph_source],
        currency=CURRENCY_USD,
    )
    closing_equity = _unavailable(
        (
            "Unavailable — no authoritative session-close / period-end equity boundary. "
            f"{PORTFOLIO_HISTORY_BOUNDARY_DISCLOSURE}."
        ),
        sources=[ph_source],
        currency=CURRENCY_USD,
    )
    net_change = _unavailable(
        (
            "Unavailable — net equity change for the selected market session requires "
            "authoritative opening and closing boundaries (not arbitrary recorded observations)."
        ),
        sources=[ph_source],
        currency=CURRENCY_USD,
    )

    # Useful recorded observations (precisely labeled; not session P/L).
    if previous_recorded_row and previous_recorded_row.get("total_value") is not None:
        previous_recorded = _available(
            previous_recorded_row["total_value"],
            sources=[ph_source],
            method="previous_recorded_portfolio_history_total_value",
            source_timestamp=previous_recorded_row["date"].isoformat(),
        )
        previous_recorded.reason = (
            f"Observation date={previous_recorded_row['date'].isoformat()}. "
            f"{PORTFOLIO_HISTORY_BOUNDARY_DISCLOSURE}."
        )
        if not is_nyse_trading_session(previous_recorded_row["date"]):
            previous_recorded.availability = AVAIL_PARTIAL
            previous_recorded.reason += " Non-session observation."
    else:
        previous_recorded = _unavailable(
            "No previous recorded portfolio_history observation",
            sources=[ph_source],
            currency=CURRENCY_USD,
        )

    if latest_recorded_row and latest_recorded_row.get("total_value") is not None:
        latest_recorded = _available(
            latest_recorded_row["total_value"],
            sources=[ph_source],
            method="latest_recorded_portfolio_history_total_value",
            source_timestamp=latest_recorded_row["date"].isoformat(),
        )
        latest_recorded.reason = (
            f"Observation date={latest_recorded_row['date'].isoformat()}. "
            f"{PORTFOLIO_HISTORY_BOUNDARY_DISCLOSURE}."
        )
        if not is_nyse_trading_session(latest_recorded_row["date"]):
            latest_recorded.availability = AVAIL_PARTIAL
            latest_recorded.reason += " Non-session observation."
    else:
        latest_recorded = _unavailable(
            "No latest recorded portfolio_history observation",
            sources=[ph_source],
            currency=CURRENCY_USD,
        )

    if (
        previous_recorded.value is not None
        and latest_recorded.value is not None
        and previous_recorded.availability != AVAIL_UNAVAILABLE
        and latest_recorded.availability != AVAIL_UNAVAILABLE
    ):
        recorded_change = _available(
            latest_recorded.value - previous_recorded.value,
            sources=[ph_source],
            method="change_between_recorded_portfolio_history_observations",
            source_timestamp=latest_recorded.source_timestamp,
        )
        recorded_change.unit = "USD_recorded_observation_delta"
        recorded_change.reason = (
            "Change between recorded portfolio_history observations only — "
            "not Period P/L, Daily P/L, investment return, session return, "
            "or net equity change for the selected market session."
        )
    else:
        recorded_change = _unavailable(
            "Change between recorded observations requires two recorded equity values",
            sources=[ph_source],
            currency=CURRENCY_USD,
        )

    # Period P/L and return deliberately withheld — no approved cash-flow-reconciled period P/L.
    period_pl = _unavailable(
        "Period P/L withheld — no approved cash-flow-reconciled period P/L authority for this window",
        sources=[],
        currency=CURRENCY_USD,
    )
    period_return = _unavailable(
        "Return percentage withheld — no approved traceable return methodology for this period",
        sources=[],
        unit="percent",
    )
    external_cf = _unavailable(
        "Net external cash flows Unavailable — no deposit/withdrawal/fee/corporate-action artifacts",
        sources=[],
        currency=CURRENCY_USD,
    )

    realized = _period_realized_pl(root=project_root)
    sources_used.extend(realized.sources)

    # Unrealized only when positions snapshot aligns with period end / current as-of.
    pos_path = _results_dir(project_root) / "positions_snapshot.csv"
    pos_rows = _read_csv_dicts(pos_path)
    unrealized: Metric
    if not pos_rows:
        unrealized = _unavailable(
            "Ending unrealized P/L Unavailable — missing positions snapshot",
            sources=[str(pos_path)],
            currency=CURRENCY_USD,
        )
    else:
        snap_ts = _parse_dt(pos_rows[0].get("snapshot_ts"))
        snap_day = _parse_date(pos_rows[0].get("date")) or (
            snap_ts.astimezone(REPORTING_TZ).date() if snap_ts else None
        )
        total_upl = Decimal("0")
        any_upl = False
        for r in pos_rows:
            upl = _d(r.get("unrealized_pl"))
            if upl is not None:
                total_upl += upl
                any_upl = True
        if not any_upl:
            unrealized = _unavailable(
                "positions_snapshot has no unrealized_pl values",
                sources=[str(pos_path)],
                currency=CURRENCY_USD,
            )
        elif snap_day is not None and snap_day != end:
            non_session_note = ""
            if snap_day is not None and not is_nyse_trading_session(snap_day):
                non_session_note = (
                    f" Snapshot date {snap_day.isoformat()} is a non-session observation "
                    f"and is not {end.isoformat()} session close or period-end evidence."
                )
            unrealized = Metric(
                value=total_upl,
                availability=AVAIL_PARTIAL,
                currency=CURRENCY_USD,
                sources=[str(pos_path)],
                method="positions_snapshot.unrealized_pl_sum",
                reason=(
                    "Unrealized P/L reflects a later/current positions snapshot, not a historical "
                    f"period-end mark for selected session {end.isoformat()}.{non_session_note}"
                ),
                source_timestamp=None if snap_ts is None else snap_ts.isoformat(),
            )
        else:
            unrealized = _available(
                total_upl,
                sources=[str(pos_path)],
                method="positions_snapshot.unrealized_pl_sum",
                source_timestamp=None if snap_ts is None else snap_ts.isoformat(),
            )

    activity = _build_activity(root=project_root, start=start, end=end)
    holdings = _build_holdings(root=project_root, end=end, now=as_of)
    recon = _reconcile(
        opening=previous_recorded_row,
        ending=latest_recorded_row,
        activity=activity,
        holdings=holdings,
    )

    # Conservative per-source freshness (no synthetic times, no mtime→FRESH, no negative clamp).
    results_dir = _results_dir(project_root)
    ph_path = results_dir / "portfolio_history.csv"
    orders_path = results_dir / "recent_orders.csv"
    audit_path = results_dir / "paper_execution_audit.csv"
    outcomes_path = results_dir / "trade_outcomes.csv"
    source_freshness: List[Dict[str, Any]] = []

    hist_date = None
    if latest_recorded_row and latest_recorded_row.get("date"):
        hist_date = latest_recorded_row["date"]
    elif history:
        hist_date = history[-1]["date"]
    source_freshness.append(
        classify_source_freshness(
            as_of=as_of,
            source_name="portfolio_history",
            source_path=str(ph_path),
            content_timestamp=None,
            date_only_value=hist_date,
            filesystem_mtime=(
                datetime.fromtimestamp(ph_path.stat().st_mtime, tz=timezone.utc)
                if ph_path.is_file()
                else None
            ),
        )
    )

    if pos_path.is_file():
        pts = _parse_dt(pos_rows[0].get("snapshot_ts")) if pos_rows else None
        source_freshness.append(
            classify_source_freshness(
                as_of=as_of,
                source_name="positions_snapshot",
                source_path=str(pos_path),
                content_timestamp=pts,
                filesystem_mtime=(
                    datetime.fromtimestamp(pos_path.stat().st_mtime, tz=timezone.utc)
                    if pos_path.is_file()
                    else None
                ),
            )
        )

    if orders_path.is_file():
        order_rows = _read_csv_dicts(orders_path)
        ots = None
        if order_rows:
            ots = _parse_dt(
                order_rows[0].get("snapshot_ts")
                or order_rows[0].get("filled_at")
                or order_rows[0].get("submitted_at")
            )
        source_freshness.append(
            classify_source_freshness(
                as_of=as_of,
                source_name="recent_orders",
                source_path=str(orders_path),
                content_timestamp=ots,
                filesystem_mtime=datetime.fromtimestamp(
                    orders_path.stat().st_mtime, tz=timezone.utc
                ),
            )
        )

    if audit_path.is_file() and (
        activity.get("sources") and str(audit_path) in (activity.get("sources") or [])
    ):
        audit_rows = _read_csv_dicts(audit_path)
        ats = None
        if audit_rows:
            ats = _parse_dt(
                audit_rows[0].get("timestamp")
                or audit_rows[0].get("submitted_at")
                or audit_rows[0].get("created_at")
            )
        source_freshness.append(
            classify_source_freshness(
                as_of=as_of,
                source_name="paper_execution_audit",
                source_path=str(audit_path),
                content_timestamp=ats,
                filesystem_mtime=datetime.fromtimestamp(
                    audit_path.stat().st_mtime, tz=timezone.utc
                ),
            )
        )

    if outcomes_path.is_file():
        # Referenced by realized-P/L provenance even when value is withheld.
        source_freshness.append(
            classify_source_freshness(
                as_of=as_of,
                source_name="trade_outcomes",
                source_path=str(outcomes_path),
                content_timestamp=None,
                filesystem_mtime=datetime.fromtimestamp(
                    outcomes_path.stat().st_mtime, tz=timezone.utc
                ),
            )
        )

    freshness = aggregate_report_freshness(source_freshness)

    # Conflict detection: cash+MV vs total on ending row already in recon; surface CONFLICTED avail.
    if recon["status"] == RECON_FAIL:
        if availability == AVAIL_AVAILABLE:
            availability = AVAIL_CONFLICTED

    position_count = Metric(
        value=(
            Decimal(len(holdings.get("rows") or []))
            if holdings.get("rows") is not None and pos_path.is_file()
            else None
        ),
        availability=(
            AVAIL_AVAILABLE
            if pos_path.is_file()
            and holdings.get("availability") in (AVAIL_AVAILABLE, AVAIL_PARTIAL)
            else AVAIL_UNAVAILABLE
        ),
        unit="count",
        sources=[str(pos_path)],
        method="len(positions_snapshot rows)",
        reason=None if pos_path.is_file() else "Missing positions snapshot",
    )
    if not pos_path.is_file():
        position_count = _unavailable(
            "Missing positions snapshot", sources=[str(pos_path)], currency=None
        )
        position_count.unit = "count"

    if activity.get("availability") == AVAIL_UNAVAILABLE:
        order_count = _unavailable(
            activity.get("note") or "Activity unavailable",
            sources=list(activity.get("sources") or []),
        )
        order_count.unit = "count"
        trade_count = _unavailable(
            "Trade count unavailable — missing order evidence",
            sources=list(activity.get("sources") or []),
        )
        trade_count.unit = "count"
    else:
        # Values may be shown as recorded counts under PARTIAL coverage — never as complete totals.
        order_count = Metric(
            value=Decimal(int(activity.get("counts", {}).get("orders_in_period") or 0)),
            availability=AVAIL_PARTIAL,
            unit="count",
            sources=list(activity.get("sources") or []),
            method="recorded_recent_orders_in_window_and_period",
            reason=(
                "Recorded order count under partial coverage (recent_orders limited window). "
                "Not a complete period total."
            ),
        )
        trade_count = Metric(
            value=Decimal(int(activity.get("counts", {}).get("trades") or 0)),
            availability=AVAIL_PARTIAL,
            unit="count",
            sources=list(activity.get("sources") or []),
            method="recorded_fills_partials_in_window_and_period",
            reason=(
                "Recorded fill/partial count under partial coverage; canceled/rejected excluded. "
                "Not a complete period total."
            ),
        )

    # Chart: require >= 2 chronological points
    points = [
        {"date": r["date"].isoformat(), "total_value": str(r["total_value"])}
        for r in in_range
        if r.get("total_value") is not None
    ]
    if len(points) >= 2:
        chart = {
            "available": True,
            "reason": None,
            "points": points,
            "sources": [ph_source],
        }
    else:
        chart = {
            "available": False,
            "reason": "Chart unavailable — fewer than two valid chronological portfolio_history points",
            "points": points,
            "sources": [ph_source],
        }

    latest_source_date = history[-1]["date"].isoformat() if history else None

    return {
        "ok": True,
        "period_type": bounds["period_type"],
        "label": bounds.get("label"),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "requested_range": {"start": start.isoformat(), "end": end.isoformat()},
        "effective_coverage": {
            "history_dates": [r["date"].isoformat() for r in in_range],
            "previous_recorded_date": (
                None if previous_recorded_row is None else previous_recorded_row["date"].isoformat()
            ),
            "latest_recorded_date": (
                None if latest_recorded_row is None else latest_recorded_row["date"].isoformat()
            ),
            "is_market_session": bool(bounds.get("is_market_session")),
            "non_session_history_observations": non_session_obs,
            "latest_known_source_date": latest_source_date,
            "portfolio_history_disclosure": PORTFOLIO_HISTORY_BOUNDARY_DISCLOSURE,
        },
        "reporting_timezone": REPORTING_TZ_NAME,
        "as_of": as_of.astimezone(REPORTING_TZ).isoformat(),
        "availability": availability,
        "completeness": completeness,
        "reconciliation": recon,
        "freshness": freshness,
        "calendar_available": True,
        "metrics": {
            "opening_equity": opening_equity.to_dict(),
            "closing_equity": closing_equity.to_dict(),
            "net_equity_change": net_change.to_dict(),
            "previous_recorded_equity": previous_recorded.to_dict(),
            "latest_recorded_equity": latest_recorded.to_dict(),
            "recorded_observation_change": recorded_change.to_dict(),
            "period_pl": period_pl.to_dict(),
            "period_return": period_return.to_dict(),
            "realized_pl": realized.to_dict(),
            "unrealized_pl": unrealized.to_dict(),
            "net_external_cash_flows": external_cf.to_dict(),
            "position_count": position_count.to_dict(),
            "order_count": order_count.to_dict(),
            "trade_count": trade_count.to_dict(),
        },
        "activity": activity,
        "holdings": holdings,
        "chart": chart,
        "evidence": {
            "sources": sorted(
                set(
                    sources_used
                    + list(activity.get("sources") or [])
                    + list(holdings.get("sources") or [])
                )
            ),
            "missing_requirements": missing,
            "notes": [
                "Opening/closing equity withheld — portfolio_history is not a certified open/close.",
                "Recorded observation change is not Period P/L, return, or session equity change.",
                "Period P/L and return percentage are withheld without approved reconciled authorities.",
                "Realized P/L withheld — trade_outcomes local FIFO is not a governed accounting authority.",
                "Order/fill counts are recorded activity under a limited recent_orders window.",
                "Canceled and rejected orders are not counted as trades.",
                "Unlinked broker activity is not attributed to TRITON.",
                "Freshness describes source recency only — not session validity or reconciliation PASS.",
                "Report freshness is a conservative aggregate of per-source freshness (UNKNOWN masks FRESH).",
                "Date-only portfolio_history never receives a synthetic timestamp for freshness.",
                "Post-as-of / future source timestamps are excluded from as-of freshness and are not FRESH.",
                "Filesystem mtime is advisory only and cannot establish FRESH.",
            ],
            "authority_audit": {
                "realized_pl": "UNAVAILABLE_UNAPPROVED_LOCAL_FIFO",
                "order_history": "PARTIAL_LIMITED_RECENT_WINDOW",
                "opening_equity_boundary": "UNAVAILABLE_NO_CERTIFIED_SESSION_OPEN",
                "closing_equity_boundary": "UNAVAILABLE_NO_CERTIFIED_SESSION_CLOSE",
                "recorded_observations": "DATE_ONLY_MIXED_PRODUCER_PORTFOLIO_HISTORY",
            },
            "source_freshness": freshness.get("sources") or [],
        },
        "writes_performed": False,
        "network_calls_performed": False,
    }


def format_money(metric: Mapping[str, Any]) -> str:
    """Presentation helper — Unavailable stays Unavailable; never formats missing as $0.00."""
    if (
        metric.get("availability") in (AVAIL_UNAVAILABLE, AVAIL_CONFLICTED)
        and metric.get("value") is None
    ):
        return "Unavailable"
    if metric.get("value") is None:
        return "Unavailable"
    try:
        q = Decimal(str(metric["value"])).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError, TypeError):
        return "Unavailable"
    return f"${q:,.2f}"


def tone_for_report_status(value: Any) -> str:
    s = str(value or "").strip().upper()
    if s in (RECON_PASS, AVAIL_AVAILABLE, COMPLETENESS_COMPLETE, FRESH_FRESH):
        return "ok"
    if s in (
        RECON_PARTIAL,
        AVAIL_PARTIAL,
        COMPLETENESS_IN_PROGRESS,
        COMPLETENESS_PARTIAL,
        FRESH_STALE,
    ):
        return "warn"
    if s in (RECON_FAIL, AVAIL_CONFLICTED):
        return "crit"
    return "muted"
