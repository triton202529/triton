# services/guard_auto.py
# ----------------------
# Auto-manages guard_snapshot.json
# Rules:
#   - Weekend (Sat/Sun UTC) => WEEKEND_FREEZE
#   - Weekday + market open (NYSE regular session ET) => CLEAR (ONLY if ACTIVE WEEKEND_FREEZE)
#   - Otherwise => NOOP
#
# Safety:
#   - Only auto-clears ACTIVE WEEKEND_FREEZE (never clears other codes)
#   - ACTIVE means: code==WEEKEND_FREEZE AND blocked==True AND kill_switch==True

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_now_utc(now_utc_str: str) -> datetime:
    """
    Parse an override like '2026-01-12T15:00:00Z' into an aware UTC datetime.
    If empty/unparseable, return current UTC.
    """
    if not now_utc_str:
        return _utc_now()

    s = now_utc_str.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return _utc_now()


def is_weekend_utc(now_utc: datetime) -> bool:
    return now_utc.weekday() >= 5  # 5=Sat, 6=Sun


def is_market_open_et(now_utc: datetime) -> bool:
    """
    NYSE regular session approximation: 09:30–16:00 ET, Mon–Fri.
    (No holiday calendar here; safe enough for auto-clear gating.)
    """
    if _ET is None:
        return False
    now_et = now_utc.astimezone(_ET)
    if now_et.weekday() >= 5:
        return False
    h, m = now_et.hour, now_et.minute
    after_open = (h > 9) or (h == 9 and m >= 30)
    before_close = h < 16
    return after_open and before_close


def read_guard_snapshot(path: Path) -> dict:
    if not path.exists():
        return {
            "updated_at": "",
            "blocked": False,
            "kill_switch": False,
            "code": "",
            "message": "",
            "reason": "",
            "extra": {},
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # If corrupted, fail safe by keeping blocked state (do not auto-clear)
        return {
            "updated_at": "",
            "blocked": True,
            "kill_switch": True,
            "code": "CORRUPT_GUARD",
            "message": "guard_snapshot.json unreadable",
            "reason": "guard_snapshot.json unreadable",
            "extra": {"source": "guard_auto"},
        }


def write_guard_snapshot(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="paper", choices=["paper", "live"])
    ap.add_argument("--path", default="data/results/guard_snapshot.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now-utc", default="", help="Override UTC time, e.g. 2026-01-12T15:00:00Z")
    args = ap.parse_args()

    now_utc = parse_now_utc(args.now_utc)
    weekday = now_utc.weekday()
    weekend = is_weekend_utc(now_utc)

    path = Path(args.path).resolve()
    snap = read_guard_snapshot(path)

    blocked = bool(snap.get("blocked", False))
    kill_switch = bool(snap.get("kill_switch", False))
    code = str(snap.get("code", "") or "")
    msg = str(snap.get("message", "") or "")

    if args.verbose:
        print(f"[guard_auto] now_utc={now_utc.isoformat()} weekday={weekday} weekend={weekend}")
        print(f"[guard_auto] path={path}")
        print(
            f"[guard_auto] current blocked={blocked} kill_switch={kill_switch} code='{code}' msg='{msg}'"
        )

    market_open = is_market_open_et(now_utc)
    if args.verbose:
        print(f"[guard_auto] market_open_et={market_open}")

    # 1) Weekend => set freeze (idempotent)
    if weekend:
        if code == "WEEKEND_FREEZE" and blocked and kill_switch:
            print("[guard_auto] NOOP (already WEEKEND_FREEZE)")
            return

        payload = {
            "updated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "blocked": True,
            "kill_switch": True,
            "code": "WEEKEND_FREEZE",
            "message": "Weekend freeze until market open",
            "reason": "Weekend freeze until market open",
            "extra": {"source": "guard_auto", "mode": args.mode},
        }

        if args.dry_run:
            print("[guard_auto] SET -> WEEKEND_FREEZE (dry-run)")
            print(json.dumps(payload, indent=2))
            return

        write_guard_snapshot(path, payload)
        print("[guard_auto] SET -> WEEKEND_FREEZE")
        return

    # 2) Weekday + market open => clear ONLY if ACTIVE WEEKEND_FREEZE
    if market_open:
        if not (code == "WEEKEND_FREEZE" and blocked and kill_switch):
            # Safety: don't auto-clear other codes, and don't rewrite stale/partial states.
            print("[guard_auto] NOOP (market open, but no ACTIVE WEEKEND_FREEZE)")
            return

        payload = {
            "updated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "blocked": False,
            "kill_switch": False,
            "code": "CLEAR",
            "message": "cleared (market open)",
            "reason": "cleared (market open)",
            "extra": {"source": "guard_auto", "mode": args.mode},
        }

        if args.dry_run:
            print("[guard_auto] CLEAR -> market open (dry-run)")
            print(json.dumps(payload, indent=2))
            return

        write_guard_snapshot(path, payload)
        print("[guard_auto] CLEAR -> market open")
        return

    # 3) Otherwise => noop
    print("[guard_auto] NOOP (no rule triggered)")


if __name__ == "__main__":
    main()
