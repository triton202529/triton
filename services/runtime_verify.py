# services/runtime_verify.py
"""
Triton - Runtime Order Verification (Verify-OpenOrders equivalent)

Key upgrades (Phase 1.5 ready):
- Correct bracket "broken parent" detection for Alpaca:
    * A bracket parent with no legs is NORMAL if parent is unfilled.
    * Only flag broken if parent is filled/partially filled and legs are missing.
- Optional remediation:
    * cancel non-GTC
    * cancel duplicates
    * cancel orphan legs
    * cancel stale (explicit flag)
- Machine-readable report dict (for automation policies)
- Optional report writer -> data/runtime/open_orders_verify.json
- Atomic writes to avoid partial file reads by dashboard.

Environment:
- APCA_API_BASE_URL (default: https://paper-api.alpaca.markets)
- APCA_API_KEY_ID
- APCA_API_SECRET_KEY

CLI examples:
  python -m services.runtime_verify
  python -m services.runtime_verify --write-report
  python -m services.runtime_verify --cancel-stale --stale-minutes 240 --write-report
  python -m services.runtime_verify --cancel-non-gtc --cancel-dupes --cancel-orphans --really-cancel --write-report
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ----------------------------
# Time helpers
# ----------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------
# Alpaca REST
# ----------------------------


def _alpaca_base_url() -> str:
    return os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def _alpaca_headers() -> Dict[str, str]:
    kid = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    if not kid or not sec:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment.")
    return {
        "APCA-API-KEY-ID": kid,
        "APCA-API-SECRET-KEY": sec,
        "Content-Type": "application/json",
    }


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Any = None,
    timeout: int = 20,
):
    url = f"{_alpaca_base_url()}{path}"
    r = requests.request(
        method, url, headers=_alpaca_headers(), params=params, json=json_body, timeout=timeout
    )
    r.raise_for_status()
    if r.text:
        return r.json()
    return None


def fetch_open_orders(nested: bool = True, limit: int = 500) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"status": "open", "limit": limit}
    if nested:
        params["nested"] = "true"
    return _request("GET", "/v2/orders", params=params)  # type: ignore


def cancel_order(order_id: str) -> None:
    _request("DELETE", f"/v2/orders/{order_id}")


# ----------------------------
# File I/O (atomic)
# ----------------------------


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


# ----------------------------
# Order helpers
# ----------------------------


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _order_age_minutes(o: Dict[str, Any], now: datetime) -> Optional[float]:
    ts = (
        _parse_ts(o.get("created_at"))
        or _parse_ts(o.get("submitted_at"))
        or _parse_ts(o.get("updated_at"))
    )
    if not ts:
        return None
    return (now - ts).total_seconds() / 60.0


def _filled_qty(o: Dict[str, Any]) -> float:
    # Alpaca fields: filled_qty is often a string
    fq = _safe_float(o.get("filled_qty"))
    if fq is None:
        return 0.0
    return fq


def _fingerprint_order(o: Dict[str, Any]) -> str:
    """
    Duplicate detection fingerprint.
    Intentionally excludes id/coid so duplicates can be detected across different IDs.
    """
    sym = (o.get("symbol") or "").upper()
    side = (o.get("side") or "").lower()
    typ = (o.get("type") or "").lower()
    tif = (o.get("time_in_force") or "").lower()
    cls = (o.get("order_class") or "").lower()
    lim = o.get("limit_price")
    stp = o.get("stop_price")
    qty = o.get("qty") or o.get("notional") or ""

    tp = None
    sl = None
    if isinstance(o.get("take_profit"), dict):
        tp = o["take_profit"].get("limit_price")
    if isinstance(o.get("stop_loss"), dict):
        sl = o["stop_loss"].get("stop_price") or o["stop_loss"].get("limit_price")

    key = f"{sym}|{side}|{typ}|{tif}|{cls}|lim={lim}|stop={stp}|qty={qty}|tp={tp}|sl={sl}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _flatten_orders(
    orders: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parents: List[Dict[str, Any]] = []
    legs: List[Dict[str, Any]] = []
    for o in orders:
        parents.append(o)
        if isinstance(o.get("legs"), list):
            for leg in o["legs"]:
                legs.append(leg)
    return parents, legs


@dataclass
class CancelAction:
    order_id: str
    reason: str
    symbol: Optional[str] = None
    side: Optional[str] = None
    tif: Optional[str] = None
    order_class: Optional[str] = None


def verify_open_orders(
    *,
    expect_tif: Optional[str] = "gtc",
    cancel_non_gtc: bool = False,
    cancel_dupes: bool = False,
    cancel_orphans: bool = False,
    cancel_stale: bool = False,
    stale_minutes: Optional[int] = 240,
    stale_only_day: bool = False,
    dry_run: bool = True,
    nested: bool = True,
    limit: int = 500,
    write_report: bool = False,
    report_path: str = "data/runtime/open_orders_verify.json",
) -> Dict[str, Any]:
    """
    Verifies open orders and optionally cancels issues (with dry_run option).

    Returns a dict:
      - status: OK | WARN | ERROR
      - summary: short string
      - counts: dict
      - issues: dict lists
      - cancel_plan: list
      - cancelled: list
      - ts: iso timestamp
    """
    now = _utcnow()
    orders = fetch_open_orders(nested=nested, limit=limit)
    parents, legs = _flatten_orders(orders)

    parent_by_id: Dict[str, Dict[str, Any]] = {str(o.get("id")): o for o in parents if o.get("id")}
    leg_parent_ids: Dict[str, str] = {}

    # Map legs -> parent id where possible
    for p in parents:
        pid = str(p.get("id"))
        if not pid:
            continue
        if isinstance(p.get("legs"), list):
            for leg in p["legs"]:
                lid = str(leg.get("id"))
                if lid:
                    leg_parent_ids[lid] = pid

    # Non-GTC (parents only)
    non_gtc: List[Dict[str, Any]] = []
    if expect_tif:
        exp = expect_tif.lower()
        for o in parents:
            tif = (o.get("time_in_force") or "").lower()
            if tif and tif != exp:
                non_gtc.append(o)

    # Duplicate detection (parents only)
    fp_map: Dict[str, List[Dict[str, Any]]] = {}
    for o in parents:
        if (o.get("status") or "").lower() != "open":
            continue
        fp = _fingerprint_order(o)
        fp_map.setdefault(fp, []).append(o)

    dup_groups = [grp for grp in fp_map.values() if len(grp) > 1]
    dup_orders: List[Dict[str, Any]] = [o for grp in dup_groups for o in grp]

    # Bracket integrity / orphan legs
    orphan_legs: List[Dict[str, Any]] = []
    broken_parents: List[Dict[str, Any]] = []
    missing_leg_counts: List[Dict[str, Any]] = []

    # IMPORTANT FIX:
    # Alpaca bracket legs are typically created/activated after the parent fills.
    # Therefore: only flag "broken" if parent filled_qty > 0 AND legs missing/insufficient.
    for p in parents:
        cls = (p.get("order_class") or "").lower()
        if cls in ("bracket", "oco"):
            legs_list = p.get("legs") if isinstance(p.get("legs"), list) else []
            fq = _filled_qty(p)

            # If unfilled, do NOT call broken just because legs are absent.
            if fq <= 0:
                continue

            # If filled/partial and legs are absent/insufficient, call broken/missing.
            if not legs_list:
                broken_parents.append(p)
            else:
                # bracket and oco generally should have 2 legs once parent filled
                if len(legs_list) < 2:
                    missing_leg_counts.append(
                        {
                            "parent_id": p.get("id"),
                            "symbol": p.get("symbol"),
                            "order_class": cls,
                            "legs_found": len(legs_list),
                        }
                    )

    # Orphan legs (best-effort)
    for l in legs:
        lid = str(l.get("id"))
        pid = leg_parent_ids.get(lid)
        if pid and pid not in parent_by_id:
            orphan_legs.append(l)

    # Stale (parents only)
    stale: List[Dict[str, Any]] = []
    if stale_minutes is not None:
        for o in parents:
            age = _order_age_minutes(o, now)
            if age is None:
                continue
            if age >= float(stale_minutes):
                if stale_only_day:
                    if (o.get("time_in_force") or "").lower() != "day":
                        continue
                stale.append(o)

    # Cancel plan
    cancel_plan: List[CancelAction] = []

    def _plan_cancel(o: Dict[str, Any], reason: str):
        oid = str(o.get("id"))
        if not oid:
            return
        cancel_plan.append(
            CancelAction(
                order_id=oid,
                reason=reason,
                symbol=o.get("symbol"),
                side=o.get("side"),
                tif=o.get("time_in_force"),
                order_class=o.get("order_class"),
            )
        )

    if cancel_non_gtc:
        for o in non_gtc:
            _plan_cancel(o, "non_gtc")

    if cancel_dupes:
        # Keep newest, cancel the rest in each group
        def _ts(o: Dict[str, Any]) -> datetime:
            return (
                _parse_ts(o.get("created_at"))
                or _parse_ts(o.get("submitted_at"))
                or datetime.min.replace(tzinfo=timezone.utc)
            )

        for grp in dup_groups:
            grp_sorted = sorted(grp, key=_ts, reverse=True)
            for o in grp_sorted[1:]:
                _plan_cancel(o, "duplicate")

    if cancel_orphans:
        for o in orphan_legs:
            _plan_cancel(o, "orphan_leg")
        for p in broken_parents:
            _plan_cancel(p, "broken_parent_missing_legs")

    if cancel_stale:
        for o in stale:
            _plan_cancel(o, "stale_order")

    # Execute cancels
    cancelled: List[str] = []
    if cancel_plan and not dry_run:
        seen = set()
        for act in cancel_plan:
            if act.order_id in seen:
                continue
            seen.add(act.order_id)
            try:
                cancel_order(act.order_id)
                cancelled.append(act.order_id)
            except Exception:
                # verifier should still return a report even if cancels fail
                pass

    # Status
    issue_count = (
        len(non_gtc)
        + len(dup_orders)
        + len(orphan_legs)
        + len(broken_parents)
        + len(stale)
        + len(missing_leg_counts)
    )
    status = "OK" if issue_count == 0 else "WARN"

    summary_parts = []
    if non_gtc:
        summary_parts.append(f"non-GTC={len(non_gtc)}")
    if dup_groups:
        summary_parts.append(f"dupe_groups={len(dup_groups)}")
    if orphan_legs:
        summary_parts.append(f"orphan_legs={len(orphan_legs)}")
    if broken_parents:
        summary_parts.append(f"broken_parents={len(broken_parents)}")
    if stale:
        summary_parts.append(f"stale={len(stale)}")

    summary = "OK" if not summary_parts else "Issues: " + ", ".join(summary_parts)

    def compact(o: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": o.get("id"),
            "symbol": o.get("symbol"),
            "side": o.get("side"),
            "type": o.get("type"),
            "time_in_force": o.get("time_in_force"),
            "order_class": o.get("order_class"),
            "status": o.get("status"),
            "limit_price": o.get("limit_price"),
            "stop_price": o.get("stop_price"),
            "filled_qty": o.get("filled_qty"),
            "created_at": o.get("created_at"),
            "client_order_id": o.get("client_order_id"),
        }

    report: Dict[str, Any] = {
        "ts": _iso(now),
        "status": status,
        "summary": summary,
        "counts": {
            "open_parents": len(parents),
            "open_legs": len(legs),
            "non_gtc": len(non_gtc),
            "dupe_groups": len(dup_groups),
            "dupe_orders": len(dup_orders),
            "orphan_legs": len(orphan_legs),
            "broken_parents": len(broken_parents),
            "stale": len(stale),
        },
        "issues": {
            "non_gtc": [compact(o) for o in non_gtc],
            "duplicates": [
                {"fingerprint": _fingerprint_order(grp[0]), "orders": [compact(o) for o in grp]}
                for grp in dup_groups
            ],
            "orphan_legs": [compact(o) for o in orphan_legs],
            "broken_parents": [compact(o) for o in broken_parents],
            "missing_leg_counts": missing_leg_counts,
            "stale": [compact(o) for o in stale],
        },
        "cancel_plan": [
            {
                "order_id": a.order_id,
                "reason": a.reason,
                "symbol": a.symbol,
                "side": a.side,
                "tif": a.tif,
                "order_class": a.order_class,
            }
            for a in cancel_plan
        ],
        "cancelled": cancelled,
        "dry_run": dry_run,
        "policy": {
            "expect_tif": expect_tif,
            "stale_minutes": stale_minutes,
            "stale_only_day": stale_only_day,
            "cancel_non_gtc": cancel_non_gtc,
            "cancel_dupes": cancel_dupes,
            "cancel_orphans": cancel_orphans,
            "cancel_stale": cancel_stale,
        },
    }

    if write_report:
        try:
            _atomic_write_json(Path(report_path), report)
            report["wrote_report"] = report_path
        except Exception as e:
            report["wrote_report"] = None
            report["write_error"] = str(e)

    return report


# ----------------------------
# CLI output
# ----------------------------


def _print_human(report: Dict[str, Any]) -> None:
    print(f"[runtime_verify] {report['ts']} status={report['status']} :: {report['summary']}")
    c = report["counts"]
    print(f"Open parents: {c['open_parents']} | Legs: {c['open_legs']}")
    if c["non_gtc"]:
        print(f"  - Non-GTC: {c['non_gtc']}")
    if c["dupe_groups"]:
        print(f"  - Duplicate groups: {c['dupe_groups']} (orders: {c['dupe_orders']})")
    if c["orphan_legs"]:
        print(f"  - Orphan legs: {c['orphan_legs']}")
    if c["broken_parents"]:
        print(f"  - Broken parents: {c['broken_parents']} (filled parents missing legs)")
    if c["stale"]:
        print(f"  - Stale orders: {c['stale']}")
    if report.get("wrote_report"):
        print(f"Report written: {report['wrote_report']}")
    if report.get("write_error"):
        print(f"Report write error: {report['write_error']}")

    if report["cancel_plan"]:
        print(
            f"Cancel plan ({'DRY-RUN' if report['dry_run'] else 'EXECUTE'}): {len(report['cancel_plan'])} orders"
        )
        for a in report["cancel_plan"][:25]:
            print(f"  - cancel {a['order_id']} reason={a['reason']} {a.get('symbol','')}")
        if len(report["cancel_plan"]) > 25:
            print("  ... (truncated)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Triton runtime verifier (Verify-OpenOrders)")
    p.add_argument("--expect-tif", default="gtc", help="Expected time_in_force. Use '' to disable.")
    p.add_argument("--cancel-non-gtc", action="store_true")
    p.add_argument("--cancel-dupes", action="store_true")
    p.add_argument("--cancel-orphans", action="store_true")
    p.add_argument("--cancel-stale", action="store_true")
    p.add_argument("--stale-minutes", type=int, default=240)
    p.add_argument(
        "--stale-only-day",
        action="store_true",
        help="Only mark/cancel stale DAY orders (ignore GTC).",
    )
    p.add_argument(
        "--really-cancel", action="store_true", help="Actually cancel orders (otherwise dry-run)"
    )
    p.add_argument("--no-nested", action="store_true", help="Do not request nested orders")
    p.add_argument(
        "--write-report",
        action="store_true",
        help="Write report JSON to data/runtime/open_orders_verify.json",
    )
    p.add_argument("--report-path", default="data/runtime/open_orders_verify.json")

    args = p.parse_args()

    report = verify_open_orders(
        expect_tif=(args.expect_tif or None),
        cancel_non_gtc=args.cancel_non_gtc,
        cancel_dupes=args.cancel_dupes,
        cancel_orphans=args.cancel_orphans,
        cancel_stale=args.cancel_stale,
        stale_minutes=args.stale_minutes,
        stale_only_day=args.stale_only_day,
        dry_run=(not args.really_cancel),
        nested=(not args.no_nested),
        write_report=args.write_report,
        report_path=args.report_path,
    )
    _print_human(report)
