"""Presentation-only Portfolio History view.

Certified cash, market value, and equity come from a proven SESSION_CLOSE.
Legacy charts remain available for inspection and are not certified performance.
This module does not write files, call a broker, or refresh runtime state.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from services.performance_accounting import (
    CERTIFIED_CLOSE_KINDS,
    EVENT_SESSION_CLOSE,
    EVENTS_RELATIVE,
    PerformanceAccountingError,
    event_id_for,
    validate_event,
)

NY_TZ = ZoneInfo("America/New_York")
CERTIFIED_HEADING = "Latest Certified Close"
PROVISIONAL_HEADING = "Current Provisional Paper Snapshot"
UNAVAILABLE = "Unavailable"
DRAWDOWN_UNAVAILABLE = "Unavailable — certified equity history baseline not yet available"
NON_SESSION_LABEL = "Non-session carry-forward — not a certified close"
LEGACY_SECTION_TITLE = "Legacy portfolio history — non-certified"
LEGACY_CASH_NOTE = (
    "The historical cash series contains imputed/stale cash and is not certified equity."
)
LEGACY_CHART_NOTE = (
    "Historical inspection only. This chart and these tables are not certified performance."
)
LEGACY_STATUS_COMPLETE = "Legacy completeness: complete legacy row — not certified or reconciled"
LEGACY_STATUS_INCOMPLETE = (
    "Legacy completeness: incomplete legacy row — not certified or reconciled"
)
LEGACY_STATUS_EMPTY = "Legacy completeness: no legacy rows — not certified or reconciled"


def _repo_root(root: Optional[Path]) -> Path:
    if root is not None:
        return Path(root)
    return Path(__file__).resolve().parents[1]


def _results(root: Path) -> Path:
    return root / "data" / "results"


def _money_display(value: str) -> str:
    amount = Decimal(str(value)).quantize(Decimal("0.01"))
    return f"${amount:,.2f}"


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PerformanceAccountingError("ledger_bom_present", {"path": str(path)})
    text = raw.decode("utf-8")
    if text and not text.endswith("\n"):
        raise PerformanceAccountingError("ledger_incomplete_final_line", {"path": str(path)})
    return text


def _proven_session_close(event: Mapping[str, Any]) -> Optional[str]:
    if event.get("event_type") != EVENT_SESSION_CLOSE:
        return "not_session_close"
    body = dict(event.get("body") or {})
    candidate = dict(event)
    candidate["body"] = body
    try:
        validate_event(candidate)
    except PerformanceAccountingError as exc:
        return exc.reason
    if str(candidate.get("event_id") or "") != event_id_for(candidate):
        return "event_id_mismatch"
    session_date = str(candidate.get("session_date") or "")
    if str(body.get("session_date") or "") != session_date:
        return "session_date_mismatch"
    if str(candidate.get("logical_key") or "") != f"session_close:{session_date}":
        return "logical_key_mismatch"
    if str(body.get("close_kind") or "") not in CERTIFIED_CLOSE_KINDS:
        return "session_close_kind_invalid"
    for field in ("cash", "market_value", "equity"):
        if body.get(field) in (None, ""):
            return "session_close_field_missing"
    return None


def load_latest_certified_close(
    *,
    root: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return the latest proven SESSION_CLOSE, or unavailable with evidence.

    Reads the accounting ledger only. Does not open portfolio history files.
    """
    base = _repo_root(root)
    ledger = Path(events_path) if events_path is not None else base / EVENTS_RELATIVE
    empty = {
        "available": False,
        "heading": CERTIFIED_HEADING,
        "session_date": UNAVAILABLE,
        "cash": UNAVAILABLE,
        "market_value": UNAVAILABLE,
        "equity": UNAVAILABLE,
        "cash_display": UNAVAILABLE,
        "market_value_display": UNAVAILABLE,
        "equity_display": UNAVAILABLE,
        "evidence": "",
        "source": ledger.name,
        "uses_portfolio_history": False,
        "uses_enhanced_portfolio_history": False,
    }
    if not ledger.is_file():
        empty["evidence"] = f"accounting ledger missing: {ledger}"
        return empty
    try:
        text = _read_text(ledger)
    except (OSError, UnicodeError, PerformanceAccountingError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        empty["evidence"] = f"accounting ledger unreadable: {reason}"
        return empty
    proven: List[Dict[str, Any]] = []
    rejected: List[str] = []
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        empty["evidence"] = "accounting ledger unreadable: ledger_blank_line"
        return empty
    for idx, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            empty["evidence"] = f"accounting ledger unreadable: ledger_malformed_line:{idx}"
            return empty
        if not isinstance(event, dict) or event.get("event_type") != EVENT_SESSION_CLOSE:
            continue
        reason = _proven_session_close(event)
        if reason:
            rejected.append(f"line {idx}: {reason}")
            continue
        proven.append(event)
    if not proven:
        detail = "; ".join(rejected) if rejected else "no SESSION_CLOSE event"
        empty["evidence"] = f"no proven certified close: {detail}"
        return empty
    chosen = max(
        proven,
        key=lambda ev: (str(ev.get("session_date")), str(ev.get("event_timestamp"))),
    )
    body = chosen["body"]
    session_date = str(chosen["session_date"])
    cash = str(body["cash"])
    market_value = str(body["market_value"])
    equity = str(body["equity"])
    return {
        "available": True,
        "heading": CERTIFIED_HEADING,
        "session_date": session_date,
        "cash": cash,
        "market_value": market_value,
        "equity": equity,
        "cash_display": _money_display(cash),
        "market_value_display": _money_display(market_value),
        "equity_display": _money_display(equity),
        "evidence": f"{ledger.name} session_close:{session_date}",
        "source": ledger.name,
        "close_kind": str(body.get("close_kind") or ""),
        "as_of_et": (
            format_et(parsed_close)
            if (parsed_close := parse_aware_timestamp(str(chosen.get("event_timestamp") or "")))
            else UNAVAILABLE
        ),
        "uses_portfolio_history": False,
        "uses_enhanced_portfolio_history": False,
    }


def parse_aware_timestamp(value: Any) -> Optional[datetime]:
    """Parse an explicit timezone-aware instant. Date-only values are refused."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return None
        return value.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    if "T" not in text and ":" not in text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return None
    return parsed.astimezone(timezone.utc)


def format_et(instant: datetime) -> str:
    return instant.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M ET")


def non_session_status(instant: datetime) -> Dict[str, Any]:
    """Label weekend and holiday dates. A closed clock on a session date is not a carry-forward."""
    et = instant.astimezone(NY_TZ)
    if et.weekday() >= 5:
        return {"non_session": True, "reason": "weekend", "label": NON_SESSION_LABEL}
    try:
        from services.market_calendar import regular_session_state

        state = regular_session_state(instant)
    except Exception as exc:
        return {
            "non_session": False,
            "reason": f"calendar_unavailable:{type(exc).__name__}",
            "label": None,
        }
    reason = str(state.get("reason") or "")
    if reason == "holiday":
        return {"non_session": True, "reason": "holiday", "label": NON_SESSION_LABEL}
    return {"non_session": False, "reason": reason or "session_date", "label": None}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    return frame


def _legacy_status(raw: pd.DataFrame) -> str:
    required = ("date", "cash", "market_value", "total_value")
    if raw.empty or any(column not in raw.columns for column in required):
        if raw.empty:
            return LEGACY_STATUS_EMPTY
        return LEGACY_STATUS_INCOMPLETE
    latest = raw.iloc[-1]
    if any(pd.isna(latest[column]) or str(latest[column]).strip() == "" for column in required):
        return LEGACY_STATUS_INCOMPLETE
    return LEGACY_STATUS_COMPLETE


def load_legacy_history(root: Path) -> Dict[str, Any]:
    path = _results(root) / "portfolio_history.csv"
    raw = _read_csv(path)
    from services.portfolio_history_sync import aggregate_portfolio_history_daily

    daily = aggregate_portfolio_history_daily(raw)
    return {
        "title": LEGACY_SECTION_TITLE,
        "status": _legacy_status(raw),
        "cash_note": LEGACY_CASH_NOTE,
        "chart_note": LEGACY_CHART_NOTE,
        "path": path,
        "exists": path.is_file() and path.stat().st_size > 0,
        "raw": raw,
        "daily": daily,
        "mtime_et": _mtime_et(path),
        "certified": False,
    }


def _mtime_et(path: Path) -> str:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return "—"
        instant = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return "—"
    return format_et(instant)


def load_local_snapshot(root: Path, history_dates: Optional[pd.Series] = None) -> Dict[str, Any]:
    positions_path = _results(root) / "positions_snapshot.csv"
    orders_path = _results(root) / "open_orders_snapshot.csv"
    positions = _read_csv(positions_path)
    orders = _read_csv(orders_path)
    instant = None
    if not positions.empty and "snapshot_ts" in positions.columns:
        for value in positions["snapshot_ts"]:
            instant = parse_aware_timestamp(value)
            if instant is not None:
                break
    session = (
        non_session_status(instant)
        if instant is not None
        else {"non_session": False, "reason": "snapshot_ts_missing", "label": None}
    )
    symbols: List[str] = []
    if not positions.empty:
        symbol_col = (
            "ticker"
            if "ticker" in positions.columns
            else "symbol" if "symbol" in positions.columns else None
        )
        if symbol_col:
            symbols = sorted(
                {
                    str(symbol).strip().upper()
                    for symbol in positions[symbol_col]
                    if str(symbol).strip() and str(symbol).strip().upper() != "NAN"
                }
            )
    if not orders_path.is_file():
        open_orders = 0
        open_orders_evidence = "open_orders_snapshot.csv missing"
    elif orders.empty or (
        "symbol" in orders.columns
        and int(
            (
                orders["symbol"].astype(str).str.strip().ne("")
                & orders["symbol"].astype(str).str.upper().ne("NAN")
            ).sum()
        )
        == 0
    ):
        open_orders = 0
        open_orders_evidence = "open_orders_snapshot.csv has no order rows"
    elif "symbol" in orders.columns:
        symbol = orders["symbol"].astype(str).str.strip()
        open_orders = int((symbol != "") & (symbol.str.upper() != "NAN")).sum()
        open_orders_evidence = "open_orders_snapshot.csv"
    else:
        open_orders = int(len(orders))
        open_orders_evidence = "open_orders_snapshot.csv"
    newer = False
    if instant is not None and history_dates is not None and len(history_dates):
        latest_history = pd.to_datetime(history_dates, errors="coerce").dropna()
        if not latest_history.empty:
            history_day = latest_history.max().date()
            newer = instant.astimezone(timezone.utc).date() > history_day
    timestamp_evidence = (
        "snapshot_ts"
        if instant is not None
        else "snapshot_ts missing or not a timezone-aware instant"
    )
    return {
        "position_count": len(symbols),
        "positions": symbols,
        "open_orders": open_orders,
        "open_orders_evidence": open_orders_evidence,
        "timestamp_et": format_et(instant) if instant is not None else UNAVAILABLE,
        "timestamp_evidence": timestamp_evidence,
        "non_session_label": session["label"],
        "non_session_reason": session["reason"],
        "positions_mtime_et": _mtime_et(positions_path),
        "orders_mtime_et": _mtime_et(orders_path),
        "newer_than_history": newer,
        "positions_frame": positions,
        "certified": False,
    }


def build_portfolio_history_presentation(root: Optional[Path] = None) -> Dict[str, Any]:
    """Assemble certified, snapshot, and legacy sections. Performs no writes."""
    base = _repo_root(root)
    certified = load_latest_certified_close(root=base)
    legacy = load_legacy_history(base)
    history_dates = None
    daily = legacy["daily"]
    if isinstance(daily, pd.DataFrame) and not daily.empty and "date" in daily.columns:
        history_dates = daily["date"]
    snapshot = load_local_snapshot(base, history_dates)
    return {
        "certified": certified,
        "snapshot": snapshot,
        "legacy": legacy,
        "drawdown": DRAWDOWN_UNAVAILABLE,
        "writes_performed": False,
        "broker_called": False,
    }


def render_portfolio_history(view: Mapping[str, Any], st: Any) -> None:
    """Render the presentation view. Display only."""
    certified = view["certified"]
    snapshot = view["snapshot"]
    legacy = view["legacy"]
    st.markdown("### 📈 Portfolio History")
    st.caption(
        "Certified close is the financial authority. Legacy charts below are historical "
        "inspection only and are not certified performance."
    )
    st.markdown(f"#### {certified['heading']}")
    st.caption(f"Certified close as of {certified.get('as_of_et') or UNAVAILABLE} ET")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session date", certified["session_date"])
    c2.metric("Certified cash", certified["cash_display"])
    c3.metric("Certified market value", certified["market_value_display"])
    c4.metric("Certified equity", certified["equity_display"])
    st.caption(f"Evidence: {certified['evidence']}")
    st.markdown("**Drawdown**")
    st.write(view["drawdown"])

    st.markdown(f"#### {PROVISIONAL_HEADING}")
    st.caption(
        f"Provisional snapshot as of {snapshot['timestamp_et']} ET. "
        "Current cash, market value, equity, holdings, and unrealized P/L stay provisional "
        "until this session has its own certified close. "
        "They are not combined with the certified close."
    )
    s1, s2 = st.columns(2)
    s1.metric("Positions", int(snapshot["position_count"]))
    s2.metric("Open Orders", int(snapshot["open_orders"]))
    st.caption(f"Snapshot time: **{snapshot['timestamp_et']}**")
    if snapshot.get("non_session_label"):
        st.warning(str(snapshot["non_session_label"]))
    elif snapshot["timestamp_et"] == UNAVAILABLE:
        st.caption(str(snapshot["timestamp_evidence"]))
    if snapshot.get("positions"):
        st.caption("Latest positions: **" + ", ".join(snapshot["positions"]) + "**")
    st.caption(
        f"**positions_snapshot.csv** — modified {snapshot['positions_mtime_et']} · "
        f"**open_orders_snapshot.csv** — modified {snapshot['orders_mtime_et']} · "
        f"{snapshot['open_orders_evidence']}"
    )
    if snapshot.get("newer_than_history"):
        st.warning(
            "Positions snapshot is newer than portfolio history. The legacy chart is unchanged. "
            "This snapshot is not a certified close."
        )
        frame = snapshot.get("positions_frame")
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            show_cols = [
                column
                for column in (
                    "snapshot_ts",
                    "date",
                    "ticker",
                    "symbol",
                    "qty",
                    "market_value",
                    "value",
                    "current_price",
                    "unrealized_pl",
                )
                if column in frame.columns
            ]
            st.markdown("#### Latest position snapshot summary (newer than history)")
            st.dataframe(frame[show_cols].tail(50), use_container_width=True, hide_index=True)

    st.markdown(f"#### {legacy['title']}")
    st.caption(str(legacy["status"]))
    st.caption(str(legacy["cash_note"]))
    st.caption(str(legacy["chart_note"]))
    st.caption(f"**portfolio_history.csv** — modified {legacy['mtime_et']}")

    raw = legacy.get("raw")
    daily = legacy.get("daily")
    if not isinstance(daily, pd.DataFrame):
        daily = pd.DataFrame(columns=["date", "cash", "market_value", "total_value"])
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame()
    if not legacy.get("exists"):
        st.info("No legacy portfolio history file is available.")
        st.dataframe(pd.DataFrame(), use_container_width=True, hide_index=True)
        return

    range_opt = st.selectbox(
        "Range",
        options=["All", "30D", "90D", "1Y", "2Y"],
        index=2,
        key="ph_range",
        help="Filters the chart/table only (does not modify any files).",
    )
    daily_plot = daily.copy()
    if not daily_plot.empty and "date" in daily_plot.columns:
        daily_plot["date"] = pd.to_datetime(daily_plot["date"], errors="coerce")
        daily_plot = daily_plot.dropna(subset=["date"]).sort_values("date")
        latest_dt_py = daily_plot["date"].iloc[-1].to_pydatetime()
        if range_opt != "All":
            days = {"30D": 30, "90D": 90, "1Y": 365, "2Y": 730}[range_opt]
            cutoff = latest_dt_py - timedelta(days=days)
            daily_plot = daily_plot[daily_plot["date"] >= cutoff].copy()

    chart_rows = len(daily_plot) if not daily_plot.empty else 0
    if chart_rows < 2:
        st.info(
            "Only one portfolio history row is available. Chart will become meaningful "
            "after additional cycles append history."
        )
    else:
        plotted = False
        try:
            import plotly.graph_objects as go  # type: ignore

            fig = go.Figure()
            if "total_value" in daily_plot.columns and daily_plot["total_value"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=daily_plot["date"],
                        y=daily_plot["total_value"],
                        mode="lines+markers",
                        name="Total Value",
                    )
                )
            if "market_value" in daily_plot.columns and daily_plot["market_value"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=daily_plot["date"],
                        y=daily_plot["market_value"],
                        mode="lines",
                        name="Market Value",
                    )
                )
            if "cash" in daily_plot.columns and daily_plot["cash"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=daily_plot["date"],
                        y=daily_plot["cash"],
                        mode="lines",
                        name="Cash",
                    )
                )
            fig.update_layout(
                title=LEGACY_SECTION_TITLE,
                xaxis_title="Date",
                yaxis_title="USD",
            )
            st.plotly_chart(fig, use_container_width=True)
            plotted = True
        except Exception:
            plotted = False
        if not plotted:
            st.caption("Plotly not available — showing aggregated table instead.")
            st.dataframe(daily_plot.tail(400), use_container_width=True)

    if not daily_plot.empty:
        try:
            latest_dt_py = pd.to_datetime(daily_plot["date"].iloc[-1], errors="coerce")
            if pd.notna(latest_dt_py):
                today = pd.Timestamp.today().normalize().to_pydatetime()
                age_days = (today - latest_dt_py.to_pydatetime()).days
                if age_days >= 7 and not snapshot.get("newer_than_history"):
                    st.warning(
                        f"Portfolio history data is **{age_days} days** old "
                        f"(latest row: {latest_dt_py.date()})."
                    )
        except Exception:
            pass

    st.markdown("#### Aggregated daily totals (tail)")
    if daily_plot.empty:
        st.dataframe(pd.DataFrame(), use_container_width=True, hide_index=True)
    else:
        st.dataframe(daily_plot.tail(120), use_container_width=True, hide_index=True)

    st.markdown("#### Raw history (tail)")
    if raw.empty:
        st.dataframe(pd.DataFrame(), use_container_width=True, hide_index=True)
    else:
        st.dataframe(raw.tail(200), use_container_width=True)
