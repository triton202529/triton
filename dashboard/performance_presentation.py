"""Read-only Performance view model from committed accounting events.

Presentation only. Does not write the ledger, contact a broker, or treat
portfolio_history.csv as a performance authority.
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from dashboard.display_labels import (
    display_label,
    money_text,
    profit_factor_text,
    quantity_text,
    ratio_percent_text,
    signed_text,
)
from dashboard.preview_data import PROD_ROOT, assert_not_production_path, isolation_active
from services.performance_accounting import (
    EVENT_FILL_BUY,
    EVENT_FILL_SELL,
    EVENTS_RELATIVE,
    GROSS_PROVISIONAL,
    UNAVAILABLE,
    PerformanceAccountingError,
    load_events,
    reduce_events,
)

EXPECTED_ACCOUNTING_SHA256 = "9227fb2c802261434877860d250df4a45575c141a6ec6605a955afd5b11a73d6"
EXPECTED_EVENT_COUNT = 14
EXPECTED_SESSION_CLOSE_DATE = "2026-09-11"
EARLIEST_FIRST_RETURN_DATE = "2026-09-14"
PORTFOLIO_HISTORY_RELATIVE = "data/results/portfolio_history.csv"
PAGE_HEADING = "Performance"
PAGE_SUPPORTING_COPY = "Forward verified accounting from the September 10 opening checkpoint."
UNAVAILABLE_LABEL = "Unavailable"
FAIL_CLOSED_TITLE = "Performance data unavailable"
SECOND_CLOSE_NOTE = "Available after the second certified session close."
MORE_METRICS_NOTE = (
    "More metrics will become available as certified closes, completed trades, "
    "and fee evidence accumulate."
)
BUILDING_HISTORY_NOTE = (
    "1 of 2 certified closes collected before the first one-period return can be calculated."
)
GROSS_PROVISIONAL_LABEL = "Gross provisional"
NET_PL_NOTE = "Net realized P/L remains unavailable until attributable fees are reconciled."
CHART_CAPTION = "Only certified accounting checkpoints are shown. Missing dates are left blank."
CHART_TITLE = "Verified equity checkpoints"
FEE_PENDING_LABEL = "Pending fee reconciliation"
STATUS_BUILDING = "Building history"
STATUS_AVAILABLE = "Available"
STATUS_CONFLICT = "Evidence conflict"
KIND_OPENING = "Opening checkpoint"
KIND_CERTIFIED = "Certified close"


class PerformanceDataUnavailable(RuntimeError):
    """Fail-closed: accounting evidence is missing, invalid, or untrusted."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_unavailable(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.upper() == "UNAVAILABLE"


def metric_text(value: Any, *, kind: str = "money") -> str:
    if is_unavailable(value):
        return UNAVAILABLE_LABEL
    if kind == "money":
        return money_text(value)
    if kind == "signed":
        return signed_text(value)
    if kind == "qty":
        return quantity_text(value)
    if kind == "count":
        try:
            return str(int(Decimal(str(value))))
        except Exception:
            return str(value)
    if kind == "status":
        return display_label(value)
    if kind == "ratio_pct":
        return ratio_percent_text(value, signed=False)
    if kind == "signed_ratio_pct":
        return ratio_percent_text(value, signed=True)
    if kind == "profit_factor":
        return profit_factor_text(value)
    return str(value)


def accounting_ledger_path(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return base / EVENTS_RELATIVE


def _refuse_portfolio_history(root: Path) -> None:
    del root
    # Explicit non-authority: this module never opens portfolio_history.csv.


def _prior_qty(close_qty: Any, sold_qty: Any) -> str:
    try:
        return format(Decimal(str(close_qty)) + Decimal(str(sold_qty)), "f")
    except Exception:
        return UNAVAILABLE


def _holdings(close: Mapping[str, Any]) -> List[Dict[str, Any]]:
    positions = list(close.get("positions") or [])
    try:
        total_mv = Decimal(str(close.get("market_value") or "0"))
    except Exception:
        total_mv = Decimal("0")
    rows: List[Dict[str, Any]] = []
    for item in positions:
        try:
            mv = Decimal(str(item.get("market_value") or "0"))
        except Exception:
            mv = None
        weight = (
            f"{(mv / total_mv * Decimal('100')):.1f}%"
            if mv is not None and total_mv > 0
            else UNAVAILABLE_LABEL
        )
        rows.append(
            {
                "symbol": item.get("symbol"),
                "quantity": metric_text(item.get("qty"), kind="qty"),
                "market_value": metric_text(item.get("market_value")),
                "average_basis": metric_text(item.get("average_entry_price")),
                "unrealized": metric_text(item.get("unrealized_pl"), kind="signed"),
                "weight": weight,
                "unrealized_raw": item.get("unrealized_pl"),
                "qty_raw": item.get("qty"),
                "market_value_raw": item.get("market_value"),
            }
        )
    return rows


def _unrealized_total(close: Mapping[str, Any]) -> Any:
    total = Decimal("0")
    for item in close.get("positions") or []:
        raw = item.get("unrealized_pl")
        if is_unavailable(raw):
            return UNAVAILABLE
        total += Decimal(str(raw))
    return format(total, "f")


def _activity_rows(
    sales: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    close: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    qty_by_symbol = {
        str(row.get("symbol")): row.get("qty") for row in (close.get("positions") or [])
    }
    fill_ids = {
        str(ev.get("event_id"))
        for ev in events
        if ev.get("event_type") in {EVENT_FILL_BUY, EVENT_FILL_SELL}
    }
    rows: List[Dict[str, Any]] = []
    for sale in sales:
        if sale.get("event_id") not in fill_ids and sale.get("event_id"):
            # Reducer sales are created from fills; keep only evidenced fills.
            if str(sale.get("logical_key") or "").startswith("fill:"):
                pass
            else:
                continue
        close_qty = qty_by_symbol.get(str(sale.get("symbol")))
        prior = _prior_qty(close_qty, sale.get("qty"))
        label = sale.get("gross_realized_pl_label")
        rows.append(
            {
                "symbol": sale.get("symbol"),
                "side": "SELL",
                "quantity": metric_text(sale.get("qty"), kind="qty"),
                "fill_price": metric_text(sale.get("price")),
                "proceeds": metric_text(sale.get("proceeds")),
                "proceeds_is_not_profit": True,
                "position": (
                    f"{quantity_text(prior)} → {quantity_text(close_qty)}"
                    if not is_unavailable(prior) and not is_unavailable(close_qty)
                    else UNAVAILABLE_LABEL
                ),
                "gross_result": metric_text(sale.get("gross_realized_pl")),
                "gross_label": (
                    GROSS_PROVISIONAL_LABEL
                    if str(label) == GROSS_PROVISIONAL
                    else metric_text(label, kind="status")
                ),
                "net_realized_pl": metric_text(sale.get("realized_pl"), kind="signed"),
                "fees_status": (
                    FEE_PENDING_LABEL
                    if not sale.get("fee_reconciliation_complete")
                    else metric_text(sale.get("fees_allocated"))
                ),
                "broker_order_id": sale.get("broker_order_id"),
                "logical_key": sale.get("logical_key"),
                "event_id": sale.get("event_id"),
            }
        )
    return rows


def checkpoint_axis_label(session_date: Any, kind: str) -> str:
    """Session-date checkpoint label. Exact timestamps stay in details."""
    text = str(session_date or "").strip()[:10]
    try:
        day = date.fromisoformat(text)
    except ValueError:
        return str(kind)
    return f"{day.strftime('%b')} {day.day} · {kind}"


def _checkpoints(
    cutover: Mapping[str, Any], closes: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    from dashboard.time_presentation import format_operator_datetime

    points: List[Dict[str, Any]] = []
    if cutover:
        session_date = str(cutover.get("as_of_utc") or "")[:10] or None
        raw_open = cutover.get("as_of_utc")
        points.append(
            {
                "kind": KIND_OPENING,
                "timestamp": raw_open,
                "timestamp_et": format_operator_datetime(raw_open),
                "equity": cutover.get("equity"),
                "session_date": session_date,
                "axis_label": checkpoint_axis_label(session_date, KIND_OPENING),
                "not_a_return": True,
            }
        )
    for close in closes:
        session_date = close.get("session_date")
        raw_ts = close.get("evidence_timestamp")
        points.append(
            {
                "kind": KIND_CERTIFIED,
                "timestamp": raw_ts,
                "timestamp_et": format_operator_datetime(raw_ts),
                "equity": close.get("equity"),
                "session_date": session_date,
                "axis_label": checkpoint_axis_label(session_date, KIND_CERTIFIED),
                "not_a_return": False,
            }
        )
    return points


def _chart_tone(points: Sequence[Mapping[str, Any]]) -> str:
    equities = [p.get("equity") for p in points if not is_unavailable(p.get("equity"))]
    if len(equities) < 2:
        return "neutral"
    start = Decimal(str(equities[0]))
    end = Decimal(str(equities[-1]))
    if end > start:
        return "positive"
    if end < start:
        return "negative"
    return "neutral"


def _fee_display(reduced: Mapping[str, Any]) -> str:
    sales = list(reduced.get("sales") or [])
    pending = list(reduced.get("pending_fees") or [])
    if pending:
        return FEE_PENDING_LABEL
    if any(not sale.get("fee_reconciliation_complete") for sale in sales):
        return FEE_PENDING_LABEL
    if sales and all(sale.get("fee_reconciliation_complete") for sale in sales):
        total = Decimal("0")
        any_fee = False
        for sale in sales:
            allocated = sale.get("fees_allocated")
            if not is_unavailable(allocated):
                total += Decimal(str(allocated))
                any_fee = True
        return metric_text(format(total, "f")) if any_fee else FEE_PENDING_LABEL
    return FEE_PENDING_LABEL


def _benchmark_rows(
    benchmarks: Mapping[str, Any],
    session_date: str,
    *,
    period: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    session = dict(benchmarks.get(session_date) or {})
    period = dict(period or {})
    rows: List[Dict[str, Any]] = []
    for symbol in ("SPY", "QQQ"):
        item = dict(session.get(symbol) or {})
        if not item:
            continue
        kind = str(item.get("return_kind") or period.get(f"{symbol.lower()}_return_kind") or "")
        raw_return = period.get(f"{symbol.lower()}_return")
        rows.append(
            {
                "symbol": symbol,
                "session_date": session_date,
                "adjusted_close": metric_text(item.get("price")),
                "methodology": ("Total return" if kind == "TOTAL_RETURN" else display_label(kind)),
                "comparison": (
                    metric_text(raw_return, kind="signed_ratio_pct")
                    if raw_return is not None
                    else UNAVAILABLE_LABEL
                ),
                "comparison_period": (
                    f"{period.get('from_session')} to {period.get('to_session')}"
                    if period.get("from_session") and period.get("to_session")
                    else None
                ),
                "price_raw": item.get("price"),
                "return_kind": kind,
            }
        )
    return rows


def load_performance_view(
    *,
    root: Optional[Path] = None,
    events_path: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
    expected_event_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Load reducer-backed Performance fields. Fail closed on bad evidence."""
    root_p = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    _refuse_portfolio_history(root_p)
    ledger = Path(events_path) if events_path is not None else accounting_ledger_path(root_p)
    if isolation_active(root_p):
        try:
            assert_not_production_path(ledger, root=root_p)
        except RuntimeError as exc:
            raise PerformanceDataUnavailable("production_ledger_refused") from exc
        if ledger.resolve() == (PROD_ROOT / EVENTS_RELATIVE).resolve():
            raise PerformanceDataUnavailable("production_ledger_refused")
    if not ledger.is_file():
        raise PerformanceDataUnavailable("accounting_ledger_missing")
    try:
        digest = file_sha256(ledger)
    except OSError as exc:
        raise PerformanceDataUnavailable(f"accounting_ledger_unreadable:{exc}") from exc
    if expected_sha256 and digest != expected_sha256:
        raise PerformanceDataUnavailable(f"accounting_ledger_hash_mismatch:{digest}")
    try:
        events = load_events(ledger)
        reduced = reduce_events(events)
    except PerformanceAccountingError as exc:
        raise PerformanceDataUnavailable(f"accounting_ledger_invalid:{exc.reason}") from exc
    if expected_event_count is not None and len(events) != expected_event_count:
        raise PerformanceDataUnavailable(f"accounting_ledger_event_count_mismatch:{len(events)}")
    keys = [str(ev.get("logical_key") or "") for ev in events]
    if len(keys) != len(set(keys)):
        raise PerformanceDataUnavailable("duplicate_logical_key")
    cutover = reduced.get("cutover") or {}
    closes = list(reduced.get("session_closes") or [])
    certified = [
        c for c in closes if str(c.get("close_kind") or "") in {"regular_close", "early_close"}
    ]
    latest = certified[-1] if certified else None
    metrics = dict(reduced.get("metrics") or {})
    close_count = len(certified)
    if close_count >= 2:
        accounting_status = STATUS_AVAILABLE
    elif close_count == 1 and cutover:
        accounting_status = STATUS_BUILDING
    elif cutover:
        accounting_status = KIND_OPENING
    else:
        accounting_status = STATUS_CONFLICT
    points = _checkpoints(cutover, certified)
    latest_session = str((latest or {}).get("session_date") or "")
    sales = list(reduced.get("sales") or [])
    activity = _activity_rows(sales, events, latest or {})
    external_total = "0"
    if reduced.get("external_flows"):
        total = Decimal("0")
        for flow in reduced["external_flows"]:
            total += Decimal(str(flow.get("amount") or "0"))
        external_total = format(total, "f")
    elif latest is not None:
        external_total = str(latest.get("external_cash_flow_since_prior") or "0")
    twr = metrics.get("time_weighted_return")
    period = (metrics.get("periods") or [None])[-1] if close_count >= 2 else None
    period = dict(period or {}) if period else {}
    spy_kind = str(period.get("spy_return_kind") or "")
    qqq_kind = str(period.get("qqq_return_kind") or "")
    return {
        "ok": True,
        "fail_closed_title": FAIL_CLOSED_TITLE,
        "page_heading": PAGE_HEADING,
        "supporting_copy": PAGE_SUPPORTING_COPY,
        "ledger_path": str(ledger),
        "ledger_sha256": digest,
        "event_count": len(events),
        "event_types": [ev.get("event_type") for ev in events],
        "logical_keys": keys,
        "cutover_count": 1 if cutover else 0,
        "certified_close_count": close_count,
        "latest_session_date": latest_session or None,
        "latest_close_timestamp": (latest or {}).get("evidence_timestamp"),
        "accounting_status": accounting_status,
        "latest_equity": metric_text((latest or {}).get("equity") if latest else None),
        "latest_equity_raw": (latest or {}).get("equity"),
        "cash": metric_text((latest or {}).get("cash") if latest else None),
        "cash_raw": (latest or {}).get("cash"),
        "market_value": metric_text((latest or {}).get("market_value") if latest else None),
        "market_value_raw": (latest or {}).get("market_value"),
        "unrealized_pl": (
            metric_text(_unrealized_total(latest), kind="signed") if latest else UNAVAILABLE_LABEL
        ),
        "unrealized_pl_raw": _unrealized_total(latest) if latest else UNAVAILABLE,
        "building_note": BUILDING_HISTORY_NOTE if close_count == 1 else None,
        "shared_unavailable_note": SECOND_CLOSE_NOTE if close_count < 2 else None,
        "grouped_metrics_note": MORE_METRICS_NOTE,
        "earliest_first_return_date": EARLIEST_FIRST_RETURN_DATE if close_count < 2 else None,
        "cutover_is_not_a_return": True,
        "historical_return": UNAVAILABLE_LABEL,
        "time_weighted_return": metric_text(twr, kind="signed_ratio_pct"),
        "maximum_drawdown": metric_text(metrics.get("maximum_drawdown"), kind="ratio_pct"),
        "spy_comparison": (
            UNAVAILABLE_LABEL
            if close_count < 2
            else metric_text(period.get("spy_return"), kind="signed_ratio_pct")
        ),
        "qqq_comparison": (
            UNAVAILABLE_LABEL
            if close_count < 2
            else metric_text(period.get("qqq_return"), kind="signed_ratio_pct")
        ),
        "spy_return_kind": spy_kind or None,
        "qqq_return_kind": qqq_kind or None,
        "spy_comparison_label": (
            "SPY return · Total return" if spy_kind == "TOTAL_RETURN" else "SPY return"
        ),
        "qqq_comparison_label": (
            "QQQ return · Total return" if qqq_kind == "TOTAL_RETURN" else "QQQ return"
        ),
        "comparison_period": (
            f"{period.get('from_session')} to {period.get('to_session')}"
            if period.get("from_session") and period.get("to_session")
            else None
        ),
        "net_realized_pl": (
            UNAVAILABLE_LABEL
            if not metrics.get("proven_realized_sale_count")
            else metric_text(
                next(
                    (
                        sale.get("realized_pl")
                        for sale in sales
                        if not is_unavailable(sale.get("realized_pl"))
                    ),
                    UNAVAILABLE,
                ),
                kind="signed",
            )
        ),
        "win_rate": metric_text(metrics.get("win_rate"), kind="ratio_pct"),
        "average_winner": metric_text(metrics.get("average_winner")),
        "average_loser": metric_text(metrics.get("average_loser")),
        "profit_factor": metric_text(metrics.get("profit_factor"), kind="profit_factor"),
        "average_market_exposure": metric_text(metrics.get("average_exposure"), kind="ratio_pct"),
        "exposure_adjusted_return": metric_text(
            metrics.get("exposure_adjusted_return"), kind="signed_ratio_pct"
        ),
        "exposure_adjusted_note": (
            "Unavailable because evidence is insufficient"
            if is_unavailable(metrics.get("exposure_adjusted_return"))
            else None
        ),
        "decision_slippage": metric_text(metrics.get("decision_to_fill_slippage_notional")),
        "submit_to_fill_slippage": metric_text(metrics.get("submit_to_fill_slippage_notional")),
        "fees": _fee_display(reduced),
        "checkpoints": points,
        "checkpoint_axis_labels": [p.get("axis_label") for p in points],
        "header_badges": [
            {"text": KIND_OPENING, "tone": "info"},
            {
                "text": (
                    f"{close_count} certified close"
                    if close_count == 1
                    else f"{close_count} certified closes"
                ),
                "tone": "ok",
            },
            {
                "text": accounting_status,
                "tone": "warn" if accounting_status == STATUS_BUILDING else "ok",
            },
        ],
        "primary_metrics": [
            "time_weighted_return",
            "maximum_drawdown",
            "spy_comparison",
            "qqq_comparison",
        ],
        "more_metrics": [
            "net_realized_pl",
            "win_rate",
            "average_winner",
            "average_loser",
            "profit_factor",
            "average_market_exposure",
            "exposure_adjusted_return",
            "decision_slippage",
            "fees",
        ],
        "chart_tone": _chart_tone(points),
        "chart_title": CHART_TITLE,
        "chart_caption": CHART_CAPTION,
        "holdings": _holdings(latest) if latest else [],
        "activity": activity,
        "benchmarks": (
            _benchmark_rows(
                reduced.get("benchmarks") or {},
                latest_session,
                period=period,
            )
            if latest_session
            else []
        ),
        "external_flow": metric_text(external_total),
        "dividends": [
            {
                "session_date": row.get("session_date"),
                "amount": metric_text(row.get("amount")),
            }
            for row in reduced.get("investment_income") or []
        ]
        or [{"label": "Dividends", "amount": UNAVAILABLE_LABEL, "none": True}],
        "cash_flows": {
            "external_flow": metric_text(external_total),
            "dividends_separate": True,
            "fees_separate": True,
        },
        "margin_class": display_label("MARGIN_CAPABLE_UNBORROWED"),
        "net_pl_note": NET_PL_NOTE if activity else None,
        "uses_portfolio_history": False,
        "portfolio_history_path": str(root_p / PORTFOLIO_HISTORY_RELATIVE),
        "events": events,
        "reduced": reduced,
        "technical": {
            "schema_version": reduced.get("schema_version"),
            "ledger_sha256": digest,
            "event_count": len(events),
            "logical_keys": keys,
            "event_ids": [ev.get("event_id") for ev in events],
            "source_hashes": (latest or {}).get("source_hashes") or {},
            "ledger_path": str(ledger),
            "cutover_timestamp": cutover.get("as_of_utc"),
            "close_timestamp": (latest or {}).get("evidence_timestamp"),
            "checkpoint_timestamps": [p.get("timestamp") for p in points],
            "checkpoint_axis_labels": [p.get("axis_label") for p in points],
            "close_kind": (latest or {}).get("close_kind"),
            "identity_residual": (latest or {}).get("identity_residual"),
            "metrics": {
                k: metrics.get(k)
                for k in (
                    "time_weighted_return",
                    "maximum_drawdown",
                    "win_rate",
                    "profit_factor",
                    "decision_to_fill_slippage_notional",
                    "submit_to_fill_slippage_notional",
                    "historical_return",
                    "proven_realized_sale_count",
                )
            },
        },
    }


def overview_performance_summary(view: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "latest_certified_equity": view.get("latest_equity"),
        "certified_close_count": view.get("certified_close_count"),
        "status": view.get("accounting_status"),
        "guidance": "Open Performance in the sidebar for verified accounting.",
    }


def certified_page_totals(view: Mapping[str, Any]) -> Dict[str, Any]:
    """Overview/Portfolio totals from the latest certified close only."""
    return {
        "available": int(view.get("certified_close_count") or 0) >= 1,
        "portfolio_value": view.get("latest_equity"),
        "cash": view.get("cash"),
        "market_value": view.get("market_value"),
        "unrealized_pl": view.get("unrealized_pl"),
        "period_change": UNAVAILABLE_LABEL,
        "period_change_note": (
            SECOND_CLOSE_NOTE if int(view.get("certified_close_count") or 0) < 2 else None
        ),
        "holdings": list(view.get("holdings") or []),
        "source": "certified_accounting_close",
        "uses_portfolio_history": False,
        "latest_equity_raw": view.get("latest_equity_raw"),
        "cash_raw": view.get("cash_raw"),
        "market_value_raw": view.get("market_value_raw"),
        "unrealized_pl_raw": view.get("unrealized_pl_raw"),
        "ledger_sha256": view.get("ledger_sha256"),
        "certified_close_count": view.get("certified_close_count"),
        "accounting_status": view.get("accounting_status"),
    }


def try_certified_page_totals(
    *,
    root: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        view = load_performance_view(root=root, expected_sha256=expected_sha256)
    except PerformanceDataUnavailable:
        return None
    totals = certified_page_totals(view)
    if not totals.get("available"):
        return None
    return totals
