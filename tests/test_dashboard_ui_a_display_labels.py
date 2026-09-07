"""UI-A presentation-label tests. No broker, ledger, or authority writes."""

from __future__ import annotations

from dashboard.display_labels import (
    KNOWN_LABELS,
    display_label,
    display_status,
    is_raw_identifier,
    technical_payload,
)


def test_known_machine_labels():
    assert display_label("paper_entry_armed") == "Paper Entry Armed"
    assert display_label("live_execution_permitted") == "Live Execution Permitted"
    assert display_label("continuous_loop") == "Continuous Loop"
    assert display_label("SKIPPED_ZERO") == "Skipped — Zero Quantity"
    assert display_label("TRIM_PRIORITY") == "Trim Priority"
    assert display_label("BLOCK_NEW_BUY") == "Block New Buys"
    assert display_label("generated_at") == "Generated At"
    assert display_label("filled_avg_price") == "Average Fill Price"


def test_preserves_acronyms():
    assert display_label("BUY") == "BUY"
    assert display_label("SELL") == "SELL"
    assert display_label("ETF") == "ETF"
    assert display_label("UTC") == "UTC"
    assert display_label("API") == "API"
    assert "P/L" in display_label("unrealized_pnl")
    assert display_label("order_id") == "Order ID"


def test_raw_identifiers_stay_raw():
    uuid = "73dee0a3-0e86-4f2e-984e-9946a74a03b5"
    assert is_raw_identifier(uuid)
    assert display_label(uuid) == uuid
    path = "data/results/paper_order_events.jsonl"
    assert is_raw_identifier(path)
    assert display_label(path) == path
    auth = "sell-arm-c6c97b3f8d93"
    assert is_raw_identifier(auth)
    assert display_label(auth) == auth
    hx = "b4332777224ff06967dba753577718c7"
    assert display_label(hx) == hx


def test_technical_payload_keeps_keys():
    row = {"authority_id": "sell-arm-1", "symbol": "GLD", "status": "FILLED"}
    tech = technical_payload(row, ("authority_id", "dispatch_id"))
    assert tech["authority_id"] == "sell-arm-1"
    assert "dispatch_id" not in tech
    assert display_status("FILLED") == "Filled" or display_status("FILLED") == "FILLED"


def test_known_map_covers_required_examples():
    required = {
        "paper_entry_armed",
        "live_execution_permitted",
        "continuous_loop",
        "skipped_zero",
        "trim_priority",
        "block_new_buy",
        "generated_at",
        "filled_avg_price",
    }
    assert required <= set(KNOWN_LABELS)
