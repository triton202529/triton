"""Shared freshness presentation for the top strip, sidebar, and Status details.

Presentation only. Does not change heartbeat evaluation, locks, or gates.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from dashboard.display_labels import display_status
from dashboard.preview_data import isolation_active, load_preview_capture, preview_as_of_datetime

FROZEN_PREVIEW_LABEL = "Frozen preview"
FROZEN_CAPTURE_NOTE = "Captured September 11 at 6:47 PM ET"
CAPTURED_SOURCE_STATUS_LABEL = "Captured source status: Stale at capture time"
PRODUCTION_AUTHORITY = "operator_console_status.data_freshness"
FROZEN_AUTHORITY = "frozen_preview_capture"
AGING_AFTER_CLOSE_LABEL = "Aging after close"
TECHNICAL_RAW_LABEL = "Raw freshness evidence (not current status)"

_HUMANIZED = {
    "FRESH": ("Fresh", "ok"),
    "STALE": ("Stale", "warn"),
    "AGING_AFTER_CLOSE": (AGING_AFTER_CLOSE_LABEL, "info"),
    "UNKNOWN": ("Unknown", "muted"),
    "UNAVAILABLE": ("Unavailable", "muted"),
}

_RENDER_RESULT: Optional[Mapping[str, Any]] = None


def freshness_evaluated_at(status: Mapping[str, Any]) -> Optional[str]:
    freshness = status.get("data_freshness")
    if isinstance(freshness, Mapping):
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
    generated = status.get("generated_at")
    return str(generated) if generated else None


def _raw_freshness_class(status: Mapping[str, Any]) -> Any:
    freshness = status.get("data_freshness")
    if isinstance(freshness, Mapping):
        return freshness.get("class")
    if freshness is None or freshness == "":
        return None
    return freshness


def _humanize_production_freshness(raw_class: Any) -> tuple[str, str]:
    token = str(raw_class or "UNKNOWN").strip() or "UNKNOWN"
    mapped = _HUMANIZED.get(token.upper())
    if mapped:
        return mapped
    label = display_status(token)
    return label, _tone_for_freshness(label)


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
    raw_class = _raw_freshness_class(status)
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
    label, tone = _humanize_production_freshness(raw_class)
    return {
        "value": label,
        "note": None,
        "captured_source_status": None,
        "tone": tone,
        "authority": PRODUCTION_AUTHORITY,
        "evaluated_at": evaluated,
        "as_of_datetime": evaluated,
        "raw_class": raw_class,
        "isolated": False,
    }


def reset_freshness_presentation() -> None:
    global _RENDER_RESULT
    _RENDER_RESULT = None


def bind_freshness_presentation(
    status: Mapping[str, Any],
    *,
    isolated: Optional[bool] = None,
) -> Mapping[str, Any]:
    """Bind one immutable presentation result for this render."""
    global _RENDER_RESULT
    if _RENDER_RESULT is not None:
        return _RENDER_RESULT
    _RENDER_RESULT = MappingProxyType(dict(shared_freshness_view(status, isolated=isolated)))
    return _RENDER_RESULT


def current_freshness_presentation() -> Optional[Mapping[str, Any]]:
    return _RENDER_RESULT


def freshness_for_render(
    status: Mapping[str, Any],
    *,
    isolated: Optional[bool] = None,
) -> Mapping[str, Any]:
    """Return the bound result, or bind it from the supplied status."""
    if _RENDER_RESULT is not None:
        return _RENDER_RESULT
    return bind_freshness_presentation(status, isolated=isolated)


def technical_freshness_evidence(
    view: Mapping[str, Any],
    *,
    guard_data_freshness: Any = None,
) -> Dict[str, Any]:
    """Raw codes for Technical details — not a second current status."""
    return {
        "label": TECHNICAL_RAW_LABEL,
        "raw_class": view.get("raw_class"),
        "guard_data_freshness": guard_data_freshness,
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
    if upper.replace(" ", "_") == "AGING_AFTER_CLOSE" or label == AGING_AFTER_CLOSE_LABEL:
        return "info"
    return "info"
