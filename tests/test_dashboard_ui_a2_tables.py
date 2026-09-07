"""UI-A2 dark tables, theme tokens, and presentation helpers."""

from __future__ import annotations

from pathlib import Path

from dashboard.display_labels import (
    display_label,
    format_column_label,
    percent_text,
    quantity_text,
)
from dashboard.operator_table import build_operator_table_html, cell_tone
from dashboard.operator_ui import TOKENS

ISO_ROOT = Path(__file__).resolve().parents[1]


def test_dark_table_html_uses_theme_tokens_not_white():
    html = build_operator_table_html(
        [
            {
                "Symbol": "GLD",
                "Quantity": "2",
                "Gain / loss": "-$19.50",
                "Change": "+$1.00",
            }
        ]
    )
    assert "triton-op-table" in html
    assert "triton-op-table-wrap" in html
    assert "<th" in html and "<td" in html
    assert "GLD" in html
    assert "class='neg'" in html
    assert "class='pos'" in html
    assert ">2</td>" in html
    assert "class='pos'>2<" not in html
    assert "#FFFFFF" not in html.upper().replace(" ", "")
    src = (ISO_ROOT / "dashboard" / "operator_ui.py").read_text(encoding="utf-8")
    assert ".triton-op-table" in src
    assert TOKENS["panel"] in src
    assert TOKENS["text"] in src


def test_table_headers_are_human_when_requested():
    html = build_operator_table_html(
        [{"unrealized_pnl": "-12.5", "avg_entry_price": "10"}],
        format_headers=True,
    )
    assert "Unrealized P/L" in html
    assert "Average Cost" in html
    assert "unrealized_pnl" not in html


def test_cell_tone_and_quantity_helpers():
    assert cell_tone("+$1.00") == "pos"
    assert cell_tone("-$19,365.08") == "neg"
    assert cell_tone("0.00") == "flat"
    assert cell_tone("Unavailable") is None
    assert quantity_text(3.0) == "3"
    assert quantity_text(1.5) == "1.5"
    assert percent_text(-15.15, signed=True) == "-15.15%"
    assert format_column_label("live_execution_permitted") == "Live Execution Permitted"


def test_theme_primary_is_informational_blue():
    text = (ISO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'primaryColor = "#4EA1FF"' in text
    assert 'base = "dark"' in text
    assert 'backgroundColor = "#0B0F14"' in text
    assert "port = 8501" in text
    assert 'address = "127.0.0.1"' in text


def test_no_fragile_radio_color_overrides():
    src = (ISO_ROOT / "dashboard" / "operator_ui.py").read_text(encoding="utf-8")
    assert "st-emotion" not in src
    assert "label[data-checked" not in src
    assert 'stRadio"] label' not in src
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    reports = (ISO_ROOT / "dashboard" / "operator_reports.py").read_text(encoding="utf-8")
    assert "render_operator_table" in modern
    assert "render_operator_table" in reports


def test_primary_pages_do_not_use_dataframe_as_main_grid():
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    # Raw/interactive grids may remain under expanders via render_operator_table.
    assert "st.dataframe(display" not in modern
    assert "st.dataframe(active" not in modern
    reports = (ISO_ROOT / "dashboard" / "operator_reports.py").read_text(encoding="utf-8")
    assert "st.table(display_rows)" not in reports
    assert "st.dataframe(broker_rows" not in reports
    assert "st.dataframe(decisions" not in reports


def test_no_raw_underscore_in_known_primary_labels():
    samples = [
        "paper_entry_armed",
        "live_execution_permitted",
        "unrealized_pnl",
        "coverage_kind",
        "snapshot_ts",
    ]
    for key in samples:
        label = display_label(key)
        assert "_" not in label, key
