"""TRITON Hard Live Execution Gate (Single Source of Truth)

Place this file at: services/live_gate.py

Both services/place_live_orders.py (executor) and view_results.py (dashboard)
should import from here to avoid drift.

This file is intentionally dependency-light and safe to import anywhere.
"""

# ─────────────────────────────────────────────────────────────
# HARD LIVE EXECUTION GATE (4B)
# Protects the *executor*, not just Streamlit UI.
# Blocks live order submission unless ALL gates are green:
# ARM + GUARD + CONFIRM + FRESH + MARKET OPEN + RISK/CPM
# ─────────────────────────────────────────────────────────────

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta


def _utcnow():
    return datetime.now(timezone.utc)


def _parse_dt(x):
    """Parse a datetime from common formats; returns aware UTC datetime or None."""
    if not x:
        return None
    if isinstance(x, (int, float)):
        # unix seconds
        try:
            return datetime.fromtimestamp(x, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip()
        # handle Z
        s = s.replace("Z", "+00:00")
        # allow "YYYY-MM-DD HH:MM:SS"
        if "T" not in s and " " in s:
            s = s.replace(" ", "T", 1)
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _load_json(path: Path):
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _truthy(v):
    return bool(v) and str(v).lower() not in ("false", "0", "no", "off", "none", "")


def _status(name, ok, detail="", meta=None):
    return {
        "name": name,
        "ok": bool(ok),
        "detail": detail or "",
        "meta": meta or {},
    }


def compute_live_gates(
    project_root: Path | None = None,
    confirm_ttl_minutes: int = 10,
    freshness_max_age_minutes: int = 180,
    require_market_open: bool = True,
    broker=None,  # optional: AlpacaBroker instance for market clock
):
    """
    Returns (live_ready: bool, gates: list[dict]).
    Conservative defaults: if we can't confirm something required, we block.
    """

    # Resolve root
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]  # /services -> project root
    data_root = project_root / "data"

    # Paths (adjust here if your files live elsewhere)
    live_dir = data_root / "live"
    results_dir = data_root / "results"

    ARM_PATH = live_dir / "live_armed.json"
    GUARD_PATH = live_dir / "guard_snapshot.json"
    CONFIRM_PATH = live_dir / "live_confirm.json"

    # Freshness sources
    HEARTBEAT_PATH = results_dir / "heartbeat.json"
    PIPELINE_STATUS_PATH = results_dir / "pipeline_status.json"

    # Optional risk/CPM artifacts (if present and red => block)
    # (These names are intentionally flexible; we check common keys.)
    CPM_PATHS = [
        results_dir / "capital_preservation.json",
        results_dir / "cpm_status.json",
        results_dir / "risk_gate.json",
        results_dir / "execution_guard.json",
        results_dir / "guard_snapshot.json",  # sometimes you store it here too
    ]

    gates = []
    now = _utcnow()

    # ── ARM
    arm = _load_json(ARM_PATH)
    arm_ok = False
    if arm:
        # accept a few schemas
        arm_ok = (
            _truthy(arm.get("armed")) or _truthy(arm.get("live_armed")) or _truthy(arm.get("ok"))
        )
    gates.append(
        _status("ARM", arm_ok, f"Missing or not armed: {ARM_PATH}" if not arm_ok else "Armed")
    )

    # ── GUARD
    guard = _load_json(GUARD_PATH)
    guard_ok = False
    if guard:
        # If your guard snapshot uses explicit fields, this covers most patterns:
        # ok / pass / cleared / blocked
        if _truthy(guard.get("blocked")):
            guard_ok = False
        elif "ok" in guard:
            guard_ok = _truthy(guard.get("ok"))
        elif "pass" in guard:
            guard_ok = _truthy(guard.get("pass"))
        elif "cleared" in guard:
            guard_ok = _truthy(guard.get("cleared"))
        else:
            # fallback: if it exists and doesn't say blocked, treat as ok
            guard_ok = True
    gates.append(
        _status("GUARD", guard_ok, f"Missing or failing: {GUARD_PATH}" if not guard_ok else "Clear")
    )

    # ── CONFIRM (typed phrase)
    confirm = _load_json(CONFIRM_PATH)
    confirm_ok = False
    confirm_age_min = None
    if confirm:
        # expected: { "confirmed": true, "confirmed_at": "...", "phrase": "...", ... }
        cflag = _truthy(confirm.get("confirmed")) or _truthy(confirm.get("ok"))
        cat = _parse_dt(confirm.get("confirmed_at") or confirm.get("ts") or confirm.get("time"))
        if cflag and cat:
            confirm_age = now - cat
            confirm_age_min = int(confirm_age.total_seconds() // 60)
            confirm_ok = confirm_age <= timedelta(minutes=confirm_ttl_minutes)
    detail = "OK" if confirm_ok else f"Missing/expired: {CONFIRM_PATH}"
    meta = {"age_min": confirm_age_min, "ttl_min": confirm_ttl_minutes}
    gates.append(_status("CONFIRM", confirm_ok, detail, meta))

    # ── FRESHNESS
    hb = _load_json(HEARTBEAT_PATH) or _load_json(PIPELINE_STATUS_PATH)
    fresh_ok = False
    as_of = None
    age_min = None
    if hb:
        # common keys: as_of, updated_at, last_success, timestamp, ts
        as_of = _parse_dt(
            hb.get("as_of")
            or hb.get("updated_at")
            or hb.get("last_success")
            or hb.get("timestamp")
            or hb.get("ts")
        )
        if as_of:
            age = now - as_of
            age_min = int(age.total_seconds() // 60)
            fresh_ok = age <= timedelta(minutes=freshness_max_age_minutes)
    detail = (
        "Fresh"
        if fresh_ok
        else f"Stale/missing heartbeat (<= {freshness_max_age_minutes} min required)"
    )
    gates.append(
        _status(
            "FRESH",
            fresh_ok,
            detail,
            {"as_of": as_of.isoformat() if as_of else None, "age_min": age_min},
        )
    )

    # ── MARKET OPEN (prefer broker clock)
    market_ok = True
    market_detail = "Not required"
    market_meta = {}
    if require_market_open:
        market_ok = False  # conservative default
        market_detail = "Cannot verify market open"
        try:
            if broker is not None and hasattr(broker, "get_clock"):
                clk = broker.get_clock()
                # expect Alpaca-like: {"is_open": bool, "next_open": "...", ...}
                is_open = _truthy(clk.get("is_open"))
                market_ok = bool(is_open)
                market_detail = "Open" if market_ok else "Closed"
                market_meta = {
                    "next_open": clk.get("next_open"),
                    "next_close": clk.get("next_close"),
                    "timestamp": clk.get("timestamp") or clk.get("ts"),
                }
        except Exception as e:
            market_ok = False
            market_detail = f"Clock error: {e}"
    gates.append(_status("MARKET", market_ok, market_detail, market_meta))

    # ── RISK/CPM (optional artifacts; if present and explicitly red => block)
    risk_ok = True
    risk_hits = []
    for p in CPM_PATHS:
        j = _load_json(p)
        if not j:
            continue
        # look for common “block” signals
        blocked = (
            _truthy(j.get("blocked")) or _truthy(j.get("halt")) or _truthy(j.get("kill_switch"))
        )
        cpm_ok = True
        if "cpm_ok" in j:
            cpm_ok = _truthy(j.get("cpm_ok"))
        if "ok" in j:
            cpm_ok = cpm_ok and _truthy(j.get("ok"))
        if "risk_ok" in j:
            cpm_ok = cpm_ok and _truthy(j.get("risk_ok"))
        if blocked or not cpm_ok:
            risk_ok = False
            risk_hits.append(
                {
                    "path": str(p),
                    "blocked": blocked,
                    "ok_fields": {
                        "ok": j.get("ok"),
                        "cpm_ok": j.get("cpm_ok"),
                        "risk_ok": j.get("risk_ok"),
                    },
                }
            )

    risk_detail = "OK (or no risk artifact present)" if risk_ok else "Blocked by risk/CPM artifact"
    gates.append(_status("RISK/CPM", risk_ok, risk_detail, {"hits": risk_hits}))

    live_ready = all(g["ok"] for g in gates)
    return live_ready, gates


class LiveExecutionBlocked(RuntimeError):
    pass


def enforce_hard_live_gate_or_raise(
    mode: str,
    broker=None,
    confirm_ttl_minutes: int = 10,
    freshness_max_age_minutes: int = 180,
    require_market_open: bool = True,
):
    """
    Call this IMMEDIATELY before placing any live orders.
    Blocks if mode is live-like and gates are not all green.
    """
    mode_norm = (mode or "").lower().strip()
    is_live = mode_norm in ("live", "real", "prod", "production")

    if not is_live:
        return  # paper/dry-run: do not gate

    live_ready, gates = compute_live_gates(
        confirm_ttl_minutes=confirm_ttl_minutes,
        freshness_max_age_minutes=freshness_max_age_minutes,
        require_market_open=require_market_open,
        broker=broker,
    )

    if not live_ready:
        # build a readable refusal message
        lines = ["HARD LIVE GATE: BLOCKED. One or more gates are RED:"]
        for g in gates:
            flag = "✅" if g["ok"] else "⛔"
            meta = f" | {g['meta']}" if g.get("meta") else ""
            lines.append(f"  {flag} {g['name']}: {g['detail']}{meta}")
        raise LiveExecutionBlocked("\n".join(lines))
