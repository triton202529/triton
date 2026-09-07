"""UI-A portfolio-history chart helpers. Fixtures only — no interpolation."""

from __future__ import annotations

from datetime import date

from pathlib import Path

from dashboard.portfolio_chart import (
    LINE_COLORS,
    TONE_NEGATIVE,
    TONE_NEUTRAL,
    TONE_POSITIVE,
    TONE_UNAVAILABLE,
    available_periods,
    build_plotly_figure,
    chart_state,
    filter_period,
    normalize_history_rows,
    period_change,
    sort_history,
)

ISO_ROOT = Path(__file__).resolve().parents[1]


def test_sorts_chronologically_even_when_input_reversed():
    rows = [
        {"date": "2026-07-02", "total_value": 2},
        {"date": "2026-07-01", "total_value": 1},
        {"date": "2026-07-03", "total_value": 3},
    ]
    ordered = normalize_history_rows(rows)
    assert [r["date"] for r in ordered] == [
        date(2026, 7, 1),
        date(2026, 7, 2),
        date(2026, 7, 3),
    ]


def test_date_range_filter():
    rows = normalize_history_rows(
        [
            {"date": "2026-01-01", "total_value": 100},
            {"date": "2026-06-01", "total_value": 110},
            {"date": "2026-09-01", "total_value": 120},
        ]
    )
    ytd = filter_period(rows, "YTD", as_of=date(2026, 9, 1))
    assert ytd[0]["date"] == date(2026, 1, 1)
    one_m = filter_period(rows, "1M", as_of=date(2026, 9, 1))
    assert all(r["date"] >= date(2026, 8, 3) for r in one_m)
    assert date(2026, 6, 1) not in [r["date"] for r in one_m]


def test_positive_negative_neutral_unavailable():
    pos = period_change(
        normalize_history_rows(
            [{"date": "2026-01-01", "total_value": 100}, {"date": "2026-01-02", "total_value": 110}]
        )
    )
    assert pos["tone"] == TONE_POSITIVE
    neg = period_change(
        normalize_history_rows(
            [{"date": "2026-01-01", "total_value": 100}, {"date": "2026-01-02", "total_value": 90}]
        )
    )
    assert neg["tone"] == TONE_NEGATIVE
    flat = period_change(
        normalize_history_rows(
            [{"date": "2026-01-01", "total_value": 100}, {"date": "2026-01-02", "total_value": 100}]
        )
    )
    assert flat["tone"] == TONE_NEUTRAL
    empty = chart_state([])
    assert empty["tone"] == TONE_UNAVAILABLE
    assert empty["available"] is False


def test_empty_and_missing_optional_columns():
    rows = normalize_history_rows(
        [
            {"date": "2026-01-01", "total_value": 50},
            {"as_of": "2026-01-03", "total_portfolio_value": 55, "cash": None},
            {"date": "not-a-date", "total_value": 99},
            {"date": "2026-01-02"},
        ]
    )
    assert [r["date"] for r in rows] == [date(2026, 1, 1), date(2026, 1, 3)]
    assert rows[1]["cash"] is None
    assert sort_history(rows)[0]["date"] <= sort_history(rows)[-1]["date"]


def test_available_periods_hide_unsupported():
    one = normalize_history_rows([{"date": "2026-09-01", "total_value": 10}])
    periods = available_periods(one)
    assert "1D" in periods
    assert "1Y" not in periods
    assert "All" not in periods
    many = normalize_history_rows(
        [{"date": f"2026-06-{d:02d}", "total_value": 100 + d} for d in range(1, 28)]
    )
    more = available_periods(many, as_of=date(2026, 6, 27))
    assert "1W" in more
    assert "1M" in more
    assert "YTD" in more


def test_verified_isolated_fixture_one_month_is_red():
    import csv

    path = ISO_ROOT / "data" / "results" / "portfolio_history.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    rows = normalize_history_rows(raw)
    assert rows[-1]["total_value"] == 108471.21
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates)
    one_m = filter_period(rows, "1M")
    change = period_change(one_m)
    assert round(change["change"], 2) == -19365.08
    assert round(change["change_pct"], 2) == -15.15
    assert change["tone"] == TONE_NEGATIVE
    assert chart_state(one_m)["color"] == LINE_COLORS[TONE_NEGATIVE]
    fig = build_plotly_figure(one_m)
    hover = fig.data[0].hovertext[-1]
    assert "$108,471.21" in hover
    assert (fig.layout.title.text or "") == ""
    dumped = fig.to_plotly_json()
    title_text = ((dumped.get("layout") or {}).get("title") or {}).get("text")
    assert title_text in ("", None)
    # Do not invent missing dates.
    observed = {r["date"] for r in one_m}
    span = (one_m[-1]["date"] - one_m[0]["date"]).days + 1
    assert len(observed) <= span
