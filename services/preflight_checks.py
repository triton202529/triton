# services/preflight_checks.py
# ----------------------------
# Preflight checks before market open / order placement.
# Produces data/results/preflight_report.json
#
# Checks:
#  - raw_freshness: newest raw date within limit
#  - signals_freshness: signals max date within limit
#  - signals_vs_raw: signals date must not lag raw date by more than max-signal-lag-days
#  - signals_generated_age: generated_at_utc freshness inside signals_with_rationale.csv (optional)
#  - snapshot_generated_age: generated_at_utc freshness inside signals_snapshot.csv (optional)
#  - guard_state: guard_snapshot is not blocked / kill_switch
#  - orders_csv: orders_today.csv exists and has rows
#  - live_arming: shows arming status (does not require active)
#  - broker_account: account is active, buying_power >= min
#  - open_orders: open order count <= max_open (strict optional)

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional broker imports (keep lazy-ish to avoid breaking offline)
try:
    from services.broker_alpaca import AlpacaBroker
except Exception:
    AlpacaBroker = None  # type: ignore


RESULTS_DIR = Path("data") / "results"
ORDERS_DIR = Path("data") / "orders"
RAW_DIR = Path("data") / "raw"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"
SNAPSHOT_CSV = RESULTS_DIR / "signals_snapshot.csv"

GUARD_JSON = RESULTS_DIR / "guard_snapshot.json"
ARM_JSON = RESULTS_DIR / "live_armed.json"
ORDERS_CSV = ORDERS_DIR / "orders_today.csv"
REPORT_JSON = RESULTS_DIR / "preflight_report.json"


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    extra: Dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_z() -> str:
    """UTC timestamp formatted as ISO8601 with Z suffix (no microseconds)."""
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _newest_date_in_raw_dir(raw_dir: Path) -> Tuple[Optional[date], Optional[str]]:
    """
    Scan all data/raw/*.csv and return newest date found in any file.
    Returns (newest_date, filename).
    """
    newest: Optional[date] = None
    newest_file: Optional[str] = None

    if not raw_dir.exists():
        return None, None

    for f in raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            if df is None or df.empty or "date" not in df.columns:
                continue
            d = pd.to_datetime(df["date"], errors="coerce").dropna()
            if d.empty:
                continue
            m = d.max().date()
            if newest is None or m > newest:
                newest = m
                newest_file = str(f)
        except Exception:
            continue

    return newest, newest_file


def check_raw_freshness(max_age_days: int) -> CheckResult:
    newest_raw, newest_file = _newest_date_in_raw_dir(RAW_DIR)
    today_utc = _utc_now().date()

    if newest_raw is None:
        return CheckResult(
            name="raw_freshness",
            ok=False,
            message="no raw CSV files with valid date found",
            extra={"raw_dir": str(RAW_DIR)},
        )

    age_days = (today_utc - newest_raw).days
    ok = age_days <= max_age_days

    return CheckResult(
        name="raw_freshness",
        ok=ok,
        message=f"newest_raw_date={newest_raw} age_days={age_days} (limit={max_age_days})",
        extra={
            "newest_raw_date": str(newest_raw),
            "age_days": age_days,
            "limit": max_age_days,
            "file": newest_file,
            "today_utc": str(today_utc),
        },
    )


def _signals_max_date() -> Tuple[Optional[date], Optional[str]]:
    if not SIGNALS_CSV.exists():
        return None, None
    try:
        df = pd.read_csv(SIGNALS_CSV)
        if df.empty or "date" not in df.columns:
            return None, None
        max_d = pd.to_datetime(df["date"], errors="coerce").dropna().max()
        if pd.isna(max_d):
            return None, None
        return max_d.date(), str(SIGNALS_CSV)
    except Exception:
        return None, None


def check_signals_freshness(max_age_days: int) -> CheckResult:
    if not SIGNALS_CSV.exists():
        return CheckResult(
            name="signals_freshness",
            ok=False,
            message=f"signals file missing: {SIGNALS_CSV}",
            extra={"path": str(SIGNALS_CSV)},
        )

    try:
        df = pd.read_csv(SIGNALS_CSV)
        if df.empty or "date" not in df.columns:
            return CheckResult(
                name="signals_freshness",
                ok=False,
                message="signals CSV empty or missing 'date' column",
                extra={"path": str(SIGNALS_CSV), "cols": list(df.columns)},
            )

        max_d = pd.to_datetime(df["date"], errors="coerce").dropna().max()
        if pd.isna(max_d):
            return CheckResult(
                name="signals_freshness",
                ok=False,
                message="signals date column could not be parsed",
                extra={"path": str(SIGNALS_CSV)},
            )

        max_date = max_d.date()
        today_utc = _utc_now().date()
        age_days = (today_utc - max_date).days
        ok = age_days <= max_age_days

        return CheckResult(
            name="signals_freshness",
            ok=ok,
            message=f"signals max_date={max_date} age_days={age_days} (limit={max_age_days})",
            extra={"signals_max_date": str(max_date), "age_days": age_days, "limit": max_age_days},
        )

    except Exception as e:
        return CheckResult(
            name="signals_freshness",
            ok=False,
            message=f"error reading signals: {e}",
            extra={"path": str(SIGNALS_CSV)},
        )


def check_signals_vs_raw(max_signal_lag_days: int) -> CheckResult:
    """
    Enforces: signals_max_date >= newest_raw_date - max_signal_lag_days

    If max_signal_lag_days=0:
      signals_max_date must match newest_raw_date (best practice).
    """
    newest_raw, newest_file = _newest_date_in_raw_dir(RAW_DIR)
    if newest_raw is None:
        return CheckResult(
            name="signals_vs_raw",
            ok=False,
            message="cannot compare; newest raw date not found",
            extra={"raw_dir": str(RAW_DIR)},
        )

    if not SIGNALS_CSV.exists():
        return CheckResult(
            name="signals_vs_raw",
            ok=False,
            message=f"cannot compare; signals file missing: {SIGNALS_CSV}",
            extra={"signals_path": str(SIGNALS_CSV)},
        )

    try:
        sdf = pd.read_csv(SIGNALS_CSV)
        if sdf.empty or "date" not in sdf.columns:
            return CheckResult(
                name="signals_vs_raw",
                ok=False,
                message="signals CSV empty or missing 'date' column",
                extra={"signals_path": str(SIGNALS_CSV), "cols": list(sdf.columns)},
            )

        smax = pd.to_datetime(sdf["date"], errors="coerce").dropna().max()
        if pd.isna(smax):
            return CheckResult(
                name="signals_vs_raw",
                ok=False,
                message="signals date could not be parsed",
                extra={"signals_path": str(SIGNALS_CSV)},
            )

        signals_max = smax.date()
        lag_days = (newest_raw - signals_max).days  # positive if signals behind raw
        ok = lag_days <= max_signal_lag_days

        return CheckResult(
            name="signals_vs_raw",
            ok=ok,
            message=f"signals_max={signals_max} raw_max={newest_raw} lag_days={lag_days} (max_lag={max_signal_lag_days})",
            extra={
                "signals_max_date": str(signals_max),
                "raw_max_date": str(newest_raw),
                "raw_file": newest_file,
                "lag_days": lag_days,
                "max_signal_lag_days": max_signal_lag_days,
            },
        )

    except Exception as e:
        return CheckResult(
            name="signals_vs_raw",
            ok=False,
            message=f"error comparing signals vs raw: {e}",
            extra={},
        )


def _read_generated_at_utc_from_csv(path: Path) -> Optional[datetime]:
    """
    Reads generated_at_utc from a CSV (expects column 'generated_at_utc').
    Returns parsed UTC datetime or None.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df is None or df.empty or "generated_at_utc" not in df.columns:
            return None
        v = df["generated_at_utc"].dropna().astype(str).iloc[0]
        dt = pd.to_datetime(v, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        # pandas Timestamp -> python datetime
        return dt.to_pydatetime()
    except Exception:
        return None


def check_generated_age(name: str, path: Path, max_age_minutes: Optional[int]) -> CheckResult:
    """
    Validates that generated_at_utc inside a CSV is not older than max_age_minutes.
    If max_age_minutes is None, this becomes a PASS informational check.
    """
    gen = _read_generated_at_utc_from_csv(path)
    if max_age_minutes is None:
        return CheckResult(
            name=name,
            ok=True,
            message=f"skipped (no limit) path={path}",
            extra={"path": str(path), "limit_minutes": None},
        )

    if gen is None:
        return CheckResult(
            name=name,
            ok=False,
            message=f"missing/invalid generated_at_utc in {path.name}",
            extra={"path": str(path), "limit_minutes": max_age_minutes},
        )

    now = _utc_now()
    age_minutes = int((now - gen).total_seconds() // 60)
    ok = age_minutes <= int(max_age_minutes)

    return CheckResult(
        name=name,
        ok=ok,
        message=f"generated_at_utc={gen.replace(microsecond=0).isoformat().replace('+00:00','Z')} age_minutes={age_minutes} (limit={max_age_minutes})",
        extra={
            "path": str(path),
            "generated_at_utc": gen.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "age_minutes": age_minutes,
            "limit_minutes": max_age_minutes,
            "now_utc": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
    )


def check_guard_state() -> CheckResult:
    g = _read_json(GUARD_JSON)

    blocked = bool(g.get("blocked", False))
    kill = bool(g.get("kill_switch", False))
    code = str(g.get("code", "")) if g else ""
    msg = str(g.get("message", "guard snapshot missing")) if g else "guard snapshot missing"
    mode = str(g.get("mode", "")) if g else ""
    updated_at = g.get("updated_at", None)

    ok = (not blocked) and (not kill)

    return CheckResult(
        name="guard_state",
        ok=ok,
        message="guard OK" if ok else f"guard BLOCKED: {code} {msg}",
        extra={
            "blocked": blocked,
            "kill_switch": kill,
            "code": code,
            "message": msg,
            "updated_at": updated_at,
            "mode": mode,
        },
    )


def check_orders_csv() -> CheckResult:
    if not ORDERS_CSV.exists():
        return CheckResult(
            name="orders_csv",
            ok=False,
            message=f"missing orders csv: {ORDERS_CSV}",
            extra={"path": str(ORDERS_CSV)},
        )
    try:
        df = pd.read_csv(ORDERS_CSV)
        ok = not df.empty
        return CheckResult(
            name="orders_csv",
            ok=ok,
            message=(
                f"orders_today.csv present rows={len(df)}"
                if ok
                else "orders_today.csv exists but empty"
            ),
            extra={"path": str(ORDERS_CSV), "rows": int(len(df)), "cols": list(df.columns)},
        )
    except Exception as e:
        return CheckResult(
            name="orders_csv",
            ok=False,
            message=f"error reading orders csv: {e}",
            extra={"path": str(ORDERS_CSV)},
        )


def check_live_arming() -> CheckResult:
    a = _read_json(ARM_JSON)
    armed = bool(a.get("armed", False))
    expires_at = a.get("expires_at", None)
    session = a.get("session", None)

    active = False
    try:
        if armed and expires_at:
            exp = pd.to_datetime(expires_at, utc=True)
            active = _utc_now() < exp.to_pydatetime()
    except Exception:
        active = False

    return CheckResult(
        name="live_arming",
        ok=True,  # informational
        message=f"armed={armed} active={active} expires_at={expires_at}",
        extra={"armed": armed, "active": active, "expires_at": expires_at, "session": session},
    )


def check_broker_account(mode: str, min_buying_power: float) -> CheckResult:
    if AlpacaBroker is None:
        return CheckResult(
            name="broker_account",
            ok=False,
            message="AlpacaBroker not available (import failed)",
            extra={"mode": mode},
        )

    try:
        b = AlpacaBroker(mode=mode)
        a = b.get_account()

        status = str(a.get("status", ""))
        buying_power = float(a.get("buying_power") or 0.0)
        ok = (status.upper() == "ACTIVE") and (buying_power >= min_buying_power)

        return CheckResult(
            name="broker_account",
            ok=ok,
            message=f"status={status} buying_power={buying_power:.2f}",
            extra={
                "mode": mode,
                "status": status,
                "buying_power": buying_power,
                "min_buying_power": min_buying_power,
                "account_id": a.get("id"),
                "account_number": a.get("account_number"),
                "pattern_day_trader": a.get("pattern_day_trader"),
            },
        )
    except Exception as e:
        return CheckResult(
            name="broker_account",
            ok=False,
            message=f"error fetching broker account: {e}",
            extra={"mode": mode},
        )


def check_open_orders(mode: str, max_open: int, strict: bool) -> CheckResult:
    if AlpacaBroker is None:
        return CheckResult(
            name="open_orders",
            ok=False,
            message="AlpacaBroker not available (import failed)",
            extra={"mode": mode},
        )

    try:
        b = AlpacaBroker(mode=mode)
        oo = b.list_orders(status="open", nested=True, limit=500) or []
        count = len(oo)
        within = count <= max_open

        if not within and strict:
            msg = f"open_orders={count} exceeds max_open={max_open} (strict=True)"
        else:
            msg = f"open_orders={count} (max_open={max_open}, strict={strict})"

        sample = []
        for o in oo[:10]:
            sample.append(
                {
                    "id": o.get("id"),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "type": o.get("type"),
                    "status": o.get("status"),
                    "tif": o.get("time_in_force"),
                }
            )

        ok = within or (not strict)

        return CheckResult(
            name="open_orders",
            ok=ok,
            message=msg,
            extra={"count": count, "max_open": max_open, "strict": strict, "sample": sample},
        )

    except Exception as e:
        return CheckResult(
            name="open_orders",
            ok=False,
            message=f"error fetching open orders: {e}",
            extra={"mode": mode},
        )


def run_all(args) -> Dict[str, Any]:
    checks: List[CheckResult] = []

    # Stamp once per run
    generated_at_utc = utc_now_z()

    checks.append(check_raw_freshness(max_age_days=args.raw_max_age_days))
    checks.append(check_signals_freshness(max_age_days=args.signals_max_age_days))
    checks.append(check_signals_vs_raw(max_signal_lag_days=args.max_signal_lag_days))

    checks.append(
        check_generated_age(
            name="signals_generated_age",
            path=SIGNALS_CSV,
            max_age_minutes=args.max_signals_generated_age_minutes,
        )
    )
    checks.append(
        check_generated_age(
            name="snapshot_generated_age",
            path=SNAPSHOT_CSV,
            max_age_minutes=args.max_snapshot_generated_age_minutes,
        )
    )

    checks.append(check_guard_state())
    checks.append(check_orders_csv())
    checks.append(check_live_arming())
    checks.append(check_broker_account(mode=args.mode, min_buying_power=args.min_buying_power))
    checks.append(
        check_open_orders(
            mode=args.mode, max_open=args.max_open_orders, strict=args.strict_open_orders
        )
    )

    # Attach generated_at_utc to each check.extra (for dashboards/logs)
    for c in checks:
        try:
            c.extra["generated_at_utc"] = generated_at_utc
        except Exception:
            pass

    ok = all(c.ok for c in checks)

    report = {
        "timestamp_utc": _utc_now().isoformat(),  # back-compat
        "generated_at_utc": generated_at_utc,  # preferred
        "mode": args.mode,
        "ok": ok,
        "checks": [asdict(c) for c in checks],
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="TRITON Preflight Checks")
    ap.add_argument("--mode", default="paper", choices=["paper", "live"])
    ap.add_argument("--min-buying-power", type=float, default=50.0)

    # Freshness rules (AS_OF dates)
    ap.add_argument(
        "--raw-max-age-days", type=int, default=1, help="Max allowed age of newest raw date."
    )
    ap.add_argument(
        "--signals-max-age-days", type=int, default=1, help="Max allowed age of signals max date."
    )
    ap.add_argument(
        "--max-signal-lag-days",
        type=int,
        default=0,
        help="Max allowed lag between signals_max_date and newest_raw_date. 0 means must match raw.",
    )

    # Generated-at freshness (artifact freshness)
    ap.add_argument(
        "--max-signals-generated-age-minutes",
        type=int,
        default=None,
        help="Optional: max minutes since signals_with_rationale.csv generated_at_utc.",
    )
    ap.add_argument(
        "--max-snapshot-generated-age-minutes",
        type=int,
        default=None,
        help="Optional: max minutes since signals_snapshot.csv generated_at_utc.",
    )

    # Open order rules
    ap.add_argument(
        "--strict-open-orders",
        action="store_true",
        help="Fail if open orders exceed max-open-orders.",
    )
    ap.add_argument("--max-open-orders", type=int, default=0)

    args = ap.parse_args()

    report = run_all(args)

    print(f"[preflight] ok={report['ok']} mode={report['mode']} time={report['timestamp_utc']}")
    for c in report["checks"]:
        status = "OK " if c["ok"] else "FAIL"
        print(f"  - {status}  {c['name']}: {c['message']}")
    print(f"[preflight] report_saved={REPORT_JSON}")


if __name__ == "__main__":
    main()
