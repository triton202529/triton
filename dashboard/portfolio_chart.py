"""
Read-only portfolio history chart helpers (UI-A).

Uses existing verified observations only. Does not interpolate, invent, or
download benchmark series. Missing dates remain gaps.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PERIODS: Tuple[str, ...] = ("1D", "1W", "1M", "3M", "YTD", "1Y", "All")

TONE_POSITIVE = "positive"
TONE_NEGATIVE = "negative"
TONE_NEUTRAL = "neutral"
TONE_UNAVAILABLE = "unavailable"

LINE_COLORS = {
    TONE_POSITIVE: "#39C98A",
    TONE_NEGATIVE: "#EF6461",
    TONE_NEUTRAL: "#95A2B2",
    TONE_UNAVAILABLE: "#6B7785",
}


def _parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_value(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_history_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep dated observations with a numeric portfolio value. Drop unparsable rows."""
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        dt = _parse_date(row.get("date") or row.get("as_of") or row.get("timestamp"))
        value = _parse_value(
            row.get("total_value")
            if row.get("total_value") is not None
            else row.get("total_portfolio_value")
        )
        if dt is None or value is None:
            continue
        item = {
            "date": dt,
            "total_value": value,
            "cash": _parse_value(row.get("cash")),
            "market_value": _parse_value(row.get("market_value")),
        }
        out.append(item)
    return sort_history(out)


def sort_history(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((dict(r) for r in rows), key=lambda r: r["date"])


def period_cutoff(period: str, *, end: date) -> Optional[date]:
    p = str(period or "").upper()
    if p in ("ALL", ""):
        return None
    if p == "1D":
        return end - timedelta(days=1)
    if p == "1W":
        return end - timedelta(days=6)
    if p == "1M":
        return end - timedelta(days=29)
    if p == "3M":
        return end - timedelta(days=89)
    if p == "YTD":
        return date(end.year, 1, 1)
    if p == "1Y":
        return end - timedelta(days=364)
    return None


def filter_period(
    rows: Sequence[Mapping[str, Any]],
    period: str,
    *,
    as_of: Optional[date] = None,
) -> List[Dict[str, Any]]:
    ordered = sort_history(rows)
    if not ordered:
        return []
    end = as_of or ordered[-1]["date"]
    start = period_cutoff(period, end=end)
    if start is None:
        return [dict(r) for r in ordered if r["date"] <= end]
    return [dict(r) for r in ordered if start <= r["date"] <= end]


def available_periods(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of: Optional[date] = None,
    min_points: int = 2,
) -> List[str]:
    """Only offer periods that can produce a meaningful series (2+ points, except 1D)."""
    ordered = sort_history(rows)
    if not ordered:
        return []
    out: List[str] = []
    for period in PERIODS:
        filtered = filter_period(ordered, period, as_of=as_of)
        if period == "1D" and len(filtered) >= 1:
            out.append(period)
        elif len(filtered) >= min_points:
            out.append(period)
    return out


def period_change(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sort_history(rows)
    if not ordered:
        return {
            "start_value": None,
            "end_value": None,
            "change": None,
            "change_pct": None,
            "tone": TONE_UNAVAILABLE,
            "start_date": None,
            "end_date": None,
        }
    if len(ordered) == 1:
        value = float(ordered[0]["total_value"])
        return {
            "start_value": value,
            "end_value": value,
            "change": 0.0,
            "change_pct": 0.0,
            "tone": TONE_NEUTRAL,
            "start_date": ordered[0]["date"],
            "end_date": ordered[0]["date"],
        }
    start = float(ordered[0]["total_value"])
    end = float(ordered[-1]["total_value"])
    change = end - start
    pct = (change / start * 100.0) if start else None
    if change > 0:
        tone = TONE_POSITIVE
    elif change < 0:
        tone = TONE_NEGATIVE
    else:
        tone = TONE_NEUTRAL
    return {
        "start_value": start,
        "end_value": end,
        "change": change,
        "change_pct": pct,
        "tone": tone,
        "start_date": ordered[0]["date"],
        "end_date": ordered[-1]["date"],
    }


def chart_state(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sort_history(rows)
    stats = period_change(ordered)
    if not ordered:
        stats["available"] = False
        stats["reason"] = "No verified portfolio-history observations."
        return stats
    stats["available"] = True
    stats["reason"] = None
    stats["points"] = ordered
    stats["color"] = LINE_COLORS[stats["tone"]]
    return stats


def build_plotly_figure(rows: Sequence[Mapping[str, Any]], *, title: str = "") -> Any:
    """Build a Plotly line chart. Caller must have plotly installed."""
    import plotly.graph_objects as go

    state = chart_state(rows)
    points = list(state.get("points") or [])
    fig = go.Figure()
    if not points:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E7ECF3"},
            annotations=[
                {
                    "text": state.get("reason") or "No history",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"color": "#95A2B2"},
                }
            ],
        )
        if title:
            fig.update_layout(title=title)
        else:
            fig.layout.title.text = ""
        return fig

    xs = [p["date"].isoformat() for p in points]
    ys = [p["total_value"] for p in points]
    color = state["color"]
    start = state["start_value"]
    custom = []
    for p in points:
        delta = p["total_value"] - start if start is not None else None
        custom.append(
            f"{p['date'].isoformat()}<br>Value: ${p['total_value']:,.2f}"
            + (f"<br>Change: ${delta:,.2f}" if delta is not None else "")
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": color, "width": 2.2},
            fill="tozeroy",
            fillcolor=_area_color(color),
            hovertext=custom,
            hoverinfo="text",
            connectgaps=False,
            name="Portfolio value",
        )
    )
    fig.update_layout(
        margin={"l": 16, "r": 12, "t": 28 if title else 12, "b": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E7ECF3", "family": "IBM Plex Sans, Segoe UI, sans-serif"},
        hovermode="x unified",
        showlegend=False,
        xaxis={
            "title": {"text": ""},
            "gridcolor": "rgba(38,49,61,0.65)",
            "zeroline": False,
            "type": "date",
        },
        yaxis={
            "title": {"text": ""},
            "gridcolor": "rgba(38,49,61,0.65)",
            "zeroline": False,
            "tickprefix": "$",
            "tickformat": ",.0f",
            "rangemode": "tozero" if min(ys) > 0 and min(ys) < (max(ys) * 0.15) else "normal",
        },
        height=360,
    )
    if title:
        fig.update_layout(title=title)
    else:
        fig.layout.title.text = ""
    # Honest scale: do not force zero if values are large; keep a small pad only.
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.08, 1.0)
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def _area_color(hex_color: str) -> str:
    raw = hex_color.lstrip("#")
    if len(raw) != 6:
        return "rgba(149,162,178,0.12)"
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    return f"rgba({r},{g},{b},0.14)"
