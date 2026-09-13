"""Shared freshness presentation for the top strip and sidebar.

Presentation only. Does not change heartbeat evaluation, locks, or gates.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dashboard.display_labels import display_status
from dashboard.preview_data import isolation_active, load_preview_capture, preview_as_of_datetime

FROZEN_PREVIEW_LABEL = "Frozen preview"
FROZEN_CAPTURE_NOTE = "Captured September 11 at 6:47 PM ET"
CAPTURED_SOURCE_STATUS_LABEL = "Captured source status: Stale at capture time"
PRODUCTION_AUTHORITY = "operator_console_status.data_freshness"
FROZEN_AUTHORITY = "frozen_preview_capture"


def freshness_evaluated_at(status: Mapping[str, Any]) -> Optional[str]:
    freshness = status.get("data_freshness") or {}
    for key in (
        "evaluated_at",
        "as_of",
        "heartbeat_timestamp",
        "timestamp",
        "as_of_utc",
    ):
        value = freshness.get(key)
        if value:
            return str(value)
    return None


def shared_freshness_view(
    status: Mapping[str, Any],
    *,
    isolated: Optional[bool] = None,
) -> Dict[str, Any]:
    """One freshness value for every operator surface.

    Isolated preview: neutral Frozen preview. Production: the existing
    data_freshness class and the same evaluation timestamp.
    """
    if isolated is None:
        isolated = isolation_active()
    evaluated = freshness_evaluated_at(status)
    raw_class = (status.get("data_freshness") or {}).get("class")
    if isolated:
        captured = load_preview_capture()
        captured_at = str(captured.get("preview_data_captured_at") or "").strip() or evaluated
        as_of = preview_as_of_datetime()
        return {
            "value": FROZEN_PREVIEW_LABEL,
            "note": FROZEN_CAPTURE_NOTE,
            "captured_source_status": CAPTURED_SOURCE_STATUS_LABEL,
            "tone": "info",
            "authority": FROZEN_AUTHORITY,
            "evaluated_at": captured_at,
            "as_of_datetime": as_of.isoformat() if as_of is not None else captured_at,
            "raw_class": raw_class,
            "isolated": True,
        }
    label = display_status(raw_class or "Unknown")
    return {
        "value": label,
        "note": None,
        "captured_source_status": None,
        "tone": _tone_for_freshness(label),
        "authority": PRODUCTION_AUTHORITY,
        "evaluated_at": evaluated,
        "as_of_datetime": evaluated,
        "raw_class": raw_class,
        "isolated": False,
    }


def isolated_status_details_freshness(raw_class: Any = None) -> Dict[str, Any]:
    """Visible Status details copy for the frozen preview.

    Current status is Frozen preview. The captured guard code is preserved as
    source evidence, not as the live preview status.
    """
    raw = str(raw_class or "").strip()
    return {
        "value": FROZEN_PREVIEW_LABEL,
        "note": FROZEN_CAPTURE_NOTE,
        "captured_source_status": CAPTURED_SOURCE_STATUS_LABEL,
        "raw_class": raw or None,
        "isolated": True,
    }


def freshness_locations_agree(top: Mapping[str, Any], sidebar: Mapping[str, Any]) -> bool:
    return (
        str(top.get("value")) == str(sidebar.get("value"))
        and str(top.get("evaluated_at") or "") == str(sidebar.get("evaluated_at") or "")
        and str(top.get("authority")) == str(sidebar.get("authority"))
    )


def _tone_for_freshness(label: str) -> str:
    upper = str(label or "").upper()
    if upper == "FRESH":
        return "ok"
    if upper == "STALE":
        return "warn"
    if upper in {"UNKNOWN", "UNAVAILABLE"}:
        return "muted"
    return "info"
