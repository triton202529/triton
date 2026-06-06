"""
TRITON Loop Safety — Phase 148B stale heartbeat restart protection.

Refreshes watchdog heartbeat and grace period on loop startup so restarts
do not trigger false STALE_HEARTBEAT alerts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "data" / "results"
LIVE_DIR = ROOT / "data" / "live"
STATUS_PATH = RESULTS_DIR / "watchdog_status.json"
ALERTS_PATH = RESULTS_DIR / "watchdog_alerts.json"
STATE_PATH = LIVE_DIR / "watchdog_alert_state.json"

DEFAULT_HEARTBEAT_GRACE_MINUTES = 55.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _clear_stale_heartbeat_alerts(alerts_doc: Dict[str, Any], ts: str) -> bool:
    active: List[Dict[str, Any]] = list(alerts_doc.get("active_alerts") or [])
    resolved: List[Dict[str, Any]] = list(alerts_doc.get("resolved_alerts") or [])

    kept: List[Dict[str, Any]] = []
    cleared = False
    for alert in active:
        if not isinstance(alert, dict):
            kept.append(alert)
            continue
        if str(alert.get("alert_type") or "") == "STALE_HEARTBEAT":
            resolved_alert = dict(alert)
            resolved_alert["status"] = "RESOLVED"
            resolved_alert["resolved_at"] = ts
            resolved.append(resolved_alert)
            cleared = True
        else:
            kept.append(alert)

    if cleared:
        alerts_doc["active_alerts"] = kept
        alerts_doc["resolved_alerts"] = resolved[-200:]
        alerts_doc["generated_at"] = ts

    return cleared


def refresh_watchdog_heartbeat(
    source: str,
    *,
    grace_minutes: float = DEFAULT_HEARTBEAT_GRACE_MINUTES,
    continuous_interval_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    """Write fresh heartbeat, set grace window, clear stale-heartbeat alerts."""
    now = _utc_now()
    ts = _iso_utc(now)
    grace_until = _iso_utc(now + timedelta(minutes=grace_minutes))

    status = _read_json(STATUS_PATH)
    prev_hb = status.get("heartbeat") if isinstance(status.get("heartbeat"), dict) else {}
    heartbeat = {
        "timestamp": ts,
        "status": prev_hb.get("status") or status.get("watchdog_status") or "OK",
        "positions_count": prev_hb.get("positions_count", status.get("positions_count", 0)),
        "open_orders_count": prev_hb.get("open_orders_count", status.get("open_orders_count", 0)),
        "refresh_source": source,
    }
    status["timestamp"] = ts
    status["heartbeat"] = heartbeat
    if continuous_interval_minutes is not None:
        status["continuous_interval_minutes"] = continuous_interval_minutes
    _atomic_write_json(status, STATUS_PATH)

    state = _read_json(STATE_PATH)
    state["heartbeat_grace_until"] = grace_until
    state["heartbeat_refresh_source"] = source
    state["heartbeat_refresh_ts"] = ts
    _atomic_write_json(state, STATE_PATH)

    alerts_doc = _read_json(ALERTS_PATH)
    cleared = _clear_stale_heartbeat_alerts(alerts_doc, ts)
    if cleared or alerts_doc:
        _atomic_write_json(alerts_doc, ALERTS_PATH)

    print(f"[HEARTBEAT_REFRESHED] source={source} grace_until={grace_until}")

    return {
        "timestamp": ts,
        "grace_until": grace_until,
        "source": source,
        "grace_minutes": grace_minutes,
        "cleared_stale_heartbeat": cleared,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRITON loop safety — heartbeat refresh on loop startup"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh watchdog heartbeat and set grace period",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Startup source label (e.g. continuous_loop, watchdog_loop)",
    )
    parser.add_argument(
        "--grace-minutes",
        type=float,
        default=DEFAULT_HEARTBEAT_GRACE_MINUTES,
        help=f"Grace period before stale-heartbeat alerts (default {DEFAULT_HEARTBEAT_GRACE_MINUTES})",
    )
    parser.add_argument(
        "--continuous-interval-minutes",
        type=float,
        default=None,
        help="Continuous loop interval for status metadata",
    )
    args = parser.parse_args()

    if not args.refresh:
        parser.error("--refresh is required")

    refresh_watchdog_heartbeat(
        args.source,
        grace_minutes=args.grace_minutes,
        continuous_interval_minutes=args.continuous_interval_minutes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
