"""TEST-ONLY two certified-close ledger. Do not use in preview or screenshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from services.performance_accounting import EVENT_BENCHMARK_CLOSE, EVENT_SESSION_CLOSE

TEST_ONLY_LABEL = "TEST_ONLY_TWO_CERTIFIED_CLOSES"
SECOND_CLOSE_DATE = "2026-09-14"
SECOND_CLOSE_TS = "2026-09-14T20:15:00+00:00"
SECOND_EQUITY = "92100.00"
SECOND_CASH = "66447.22"
SECOND_MARKET_VALUE = "25652.78"


def _event_id(label: str) -> str:
    return hashlib.sha256(f"bp1-{TEST_ONLY_LABEL}-{label}".encode("utf-8")).hexdigest()


def build_two_close_events(one_close_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append one hypothetical certified close after the frozen September 11 close."""
    events = [dict(ev) for ev in one_close_events]
    close = next(ev for ev in events if ev.get("event_type") == EVENT_SESSION_CLOSE)
    second = json.loads(json.dumps(close))
    second["event_id"] = _event_id("session-close")
    second["logical_key"] = f"session_close:{SECOND_CLOSE_DATE}"
    second["event_timestamp"] = SECOND_CLOSE_TS
    second["session_date"] = SECOND_CLOSE_DATE
    body = dict(second.get("body") or {})
    body["session_date"] = SECOND_CLOSE_DATE
    body["equity"] = SECOND_EQUITY
    body["cash"] = SECOND_CASH
    body["market_value"] = SECOND_MARKET_VALUE
    body["close_kind"] = "regular_close"
    second["body"] = body
    events.append(second)
    for symbol, price in (("SPY", "765.000000"), ("QQQ", "716.000000")):
        bench = next(
            ev
            for ev in one_close_events
            if ev.get("event_type") == EVENT_BENCHMARK_CLOSE
            and (ev.get("body") or {}).get("symbol") == symbol
        )
        cloned = json.loads(json.dumps(bench))
        cloned["event_id"] = _event_id(f"benchmark-{symbol}")
        cloned["logical_key"] = f"benchmark:{symbol}:{SECOND_CLOSE_DATE}"
        cloned["event_timestamp"] = SECOND_CLOSE_TS
        cloned["session_date"] = SECOND_CLOSE_DATE
        bbody = dict(cloned.get("body") or {})
        bbody["session_date"] = SECOND_CLOSE_DATE
        bbody["adj_close"] = price
        bbody["price"] = price
        cloned["body"] = bbody
        events.append(cloned)
    return events


def write_two_close_ledger(src_ledger: Path, dest: Path) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    events = []
    for line in Path(src_ledger).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    out = build_two_close_events(events)
    dest.write_text(
        "".join(json.dumps(ev, separators=(",", ":")) + "\n" for ev in out),
        encoding="utf-8",
    )
    return {
        "label": TEST_ONLY_LABEL,
        "path": str(dest),
        "event_count": len(out),
        "second_close_date": SECOND_CLOSE_DATE,
    }
