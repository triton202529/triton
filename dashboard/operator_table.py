"""
Dark-theme operator tables (UI-A2).

Presentation only. Builds stable HTML from already-formatted rows.
Does not call a broker, write artifacts, or change stored keys.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from dashboard.display_labels import (
    display_label,
    format_column_label,
    is_raw_identifier,
    money_text,
    quantity_text,
    signed_text,
)
from dashboard.operator_ui import TOKENS, escape

SIGNED_HEADER_HINTS = (
    "gain",
    "loss",
    "p/l",
    "pnl",
    "change",
    "return",
    "unrealized",
    "realized",
    "delta",
)


def quantity_cell(value: Any) -> str:
    return quantity_text(value)


def _is_signed_header(header: str) -> bool:
    text = str(header or "").lower()
    return any(hint in text for hint in SIGNED_HEADER_HINTS)


def cell_tone(value: Any) -> Optional[str]:
    """Color class from an already-formatted cell. Color is never the only signal."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if value > 0:
            return "pos"
        if value < 0:
            return "neg"
        return "flat"
    text = str(value).strip().replace("−", "-")
    if text in {"Unavailable", "Unknown", "n/a", "—"}:
        return None
    if text.startswith("+"):
        return "pos"
    if text.startswith("-") and text[1:2].isdigit():
        return "neg"
    if text.startswith("-$") or text.startswith("−$"):
        return "neg"
    if text.startswith("(") and text.endswith(")") and any(ch.isdigit() for ch in text):
        return "neg"
    try:
        num = float(text.replace("$", "").replace(",", "").replace("%", "").replace("+", ""))
    except ValueError:
        return None
    if num > 0:
        return "pos"
    if num < 0:
        return "neg"
    return "flat"


def _format_cell(value: Any) -> str:
    if value is None or value == "":
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if value != value:  # NaN
            return "Unavailable"
        if abs(value) >= 100 or value.is_integer():
            return f"{value:,.0f}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    text = str(value)
    if is_raw_identifier(text):
        return text
    if "_" in text and text == text.lower():
        return display_label(text)
    if text.isupper() and "_" in text:
        return display_label(text)
    return text


def prepare_display_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
    format_headers: bool = True,
    signed_columns: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Return rows with human headers and display-ready cell strings."""
    if not rows:
        return []
    raw_cols = list(columns) if columns is not None else list(rows[0].keys())
    signed = {str(c) for c in (signed_columns or ())}
    out: List[dict] = []
    for row in rows:
        item = {}
        for key in raw_cols:
            if key not in row and key not in (row or {}):
                # Still show requested columns that are missing.
                label = format_column_label(key) if format_headers else str(key)
                item[label] = "Unavailable"
                continue
            val = row.get(key)
            label = format_column_label(key) if format_headers else str(key)
            if key in signed or _is_signed_header(label):
                if isinstance(val, (int, float)) or (
                    isinstance(val, str)
                    and val[:1] in "+-"
                    or str(val).replace(".", "", 1).isdigit()
                ):
                    try:
                        item[label] = signed_text(
                            val, currency="$" in str(val) or isinstance(val, float)
                        )
                    except Exception:
                        item[label] = _format_cell(val)
                else:
                    item[label] = _format_cell(val)
            else:
                item[label] = _format_cell(val)
        out.append(item)
    return out


def build_operator_table_html(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
    format_headers: bool = False,
    signed_columns: Optional[Iterable[str]] = None,
    max_rows: int = 200,
) -> str:
    """Stable dark table markup. Callers pass already-labeled rows by default."""
    if not rows:
        return ""
    shown = list(rows)[: max(1, int(max_rows))]
    if format_headers:
        shown = prepare_display_rows(
            shown, columns=columns, format_headers=True, signed_columns=signed_columns
        )
        headers = list(shown[0].keys())
    else:
        headers = list(columns) if columns is not None else list(shown[0].keys())
    signed = {str(c) for c in (signed_columns or ())}
    t = TOKENS
    head = "".join(f"<th scope='col'>{escape(h)}</th>" for h in headers)
    body_parts: List[str] = []
    for row in shown:
        cells = []
        for header in headers:
            val = row.get(header)
            text = (
                _format_cell(val)
                if not isinstance(val, str)
                else (val if val != "" else "Unavailable")
            )
            if val is None or val == "":
                text = "Unavailable"
            elif isinstance(val, str):
                text = val
            tone = None
            if header in signed or _is_signed_header(header):
                tone = cell_tone(val if val is not None else text) or cell_tone(text)
            cls = f" class='{tone}'" if tone else ""
            cells.append(f"<td{cls}>{escape(text)}</td>")
        body_parts.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<div class='triton-op-table-wrap'>"
        "<table class='triton-op-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_parts)}</tbody>"
        "</table></div>"
    )


def render_operator_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    format_headers: bool = False,
    signed_columns: Optional[Iterable[str]] = None,
    show_raw: bool = True,
    raw_label: str = "Interactive / raw table",
    max_rows: int = 200,
) -> None:
    """Primary operator table, with the complete raw grid under an expander."""
    import streamlit as st

    html = build_operator_table_html(
        rows,
        columns=columns,
        format_headers=format_headers,
        signed_columns=signed_columns,
        max_rows=max_rows,
    )
    if html:
        st.markdown(html, unsafe_allow_html=True)
    if caption:
        st.caption(caption)
    if show_raw:
        with st.expander(raw_label, expanded=False):
            st.caption("Complete rows, including identifiers that stay out of the primary grid.")
            st.dataframe(list(rows), use_container_width=True, hide_index=True)


# Keep money/qty helpers importable from one table module for page call sites.
money_cell = money_text
