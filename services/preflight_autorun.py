# services/preflight_autorun.py
# -----------------------------
# "Self-healing" preflight runner.
#
# Behavior:
#  - Run preflight_checks
#  - If FAIL due to data freshness (raw/signals/signals_vs_raw), optionally auto-refresh pipeline then re-run.
#  - If FAIL due to guard_state/open_orders/broker_account/orders_csv, do NOT auto-fix; exit non-zero.
#
# Market-closed guard:
#  - By default, auto-refresh will NOT run if the market is closed.
#  - You can override with --refresh-when-closed.
#  - Uses Alpaca clock when available (via services.broker_alpaca.AlpacaBroker).
#
# Exit codes:
#   0 = PASS
#   2 = FAIL (non-refreshable reasons OR market closed and refresh not allowed)
#   3 = FAIL (refresh attempted but still failing)

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from services import preflight_checks

RESULTS_DIR = Path("data") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUTORUN_REPORT = RESULTS_DIR / "preflight_autorun_report.json"

REFRESHABLE_CHECKS = {"raw_freshness", "signals_freshness", "signals_vs_raw"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _py() -> str:
    return sys.executable


def _run_cmd(cmd: List[str]) -> Tuple[int, str]:
    """
    Runs a command and returns (returncode, combined_output).
    """
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def _is_refreshable_failure(preflight_report: Dict[str, Any]) -> bool:
    """
    Only auto-refresh if ALL failing checks are within REFRESHABLE_CHECKS.
    """
    fails = [c for c in preflight_report.get("checks", []) if not c.get("ok", False)]
    if not fails:
        return False
    return all(c.get("name") in REFRESHABLE_CHECKS for c in fails)


def _summarize_failures(preflight_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    fails = [c for c in preflight_report.get("checks", []) if not c.get("ok", False)]
    return [
        {"name": c.get("name"), "message": c.get("message"), "extra": c.get("extra", {})}
        for c in fails
    ]


def _get_market_clock(mode: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Best-effort market clock via Alpaca.
    Returns dict:
      {
        "ok": bool,
        "is_open": bool | None,
        "timestamp": str | None,
        "next_open": str | None,
        "next_close": str | None,
        "raw": dict | None,
        "error": str | None
      }
    """
    try:
        from services.broker_alpaca import AlpacaBroker  # type: ignore

        b = AlpacaBroker(mode=mode)
        clk = b.get_clock()  # expected Alpaca-like dict
        # Alpaca clock usually includes: timestamp, is_open, next_open, next_close
        is_open = clk.get("is_open")
        res = {
            "ok": True,
            "is_open": bool(is_open) if is_open is not None else None,
            "timestamp": clk.get("timestamp"),
            "next_open": clk.get("next_open"),
            "next_close": clk.get("next_close"),
            "raw": clk,
            "error": None,
        }
        if verbose:
            print(
                f"[preflight_autorun] market_clock ok is_open={res['is_open']} ts={res['timestamp']}"
            )
        return res
    except Exception as e:
        if verbose:
            print(f"[preflight_autorun] market_clock unavailable: {e}")
        return {
            "ok": False,
            "is_open": None,
            "timestamp": None,
            "next_open": None,
            "next_close": None,
            "raw": None,
            "error": str(e),
        }


def _refresh_pipeline(verbose: bool) -> Dict[str, Any]:
    """
    Refresh steps:
      1) fetch raw
      2) stale gate
      3) full pipeline
    """
    steps = [
        {
            "name": "fetch_raw_data",
            "cmd": [_py(), "-m", "services.fetch_raw_data"] + (["--verbose"] if verbose else []),
        },
        {"name": "stale_data_gate", "cmd": [_py(), "-m", "services.stale_data_gate"]},
        {
            "name": "run_full_pipeline",
            "cmd": (
                [_py(), "run_full_pipeline.py", "--verbose"]
                if verbose
                else [_py(), "run_full_pipeline.py"]
            ),
        },
    ]

    results: List[Dict[str, Any]] = []
    ok = True

    for s in steps:
        rc, out = _run_cmd(s["cmd"])
        step_ok = rc == 0
        ok = ok and step_ok
        results.append(
            {
                "name": s["name"],
                "ok": step_ok,
                "returncode": rc,
                "cmd": " ".join(s["cmd"]),
                "output_tail": out[-4000:],  # keep log reasonable
            }
        )
        if not step_ok:
            break

    return {"ok": ok, "steps": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="paper", choices=["paper", "live"])
    ap.add_argument("--min-buying-power", type=float, default=50.0)

    ap.add_argument("--raw-max-age-days", type=int, default=1)
    ap.add_argument("--signals-max-age-days", type=int, default=1)
    ap.add_argument("--max-signal-lag-days", type=int, default=0)

    ap.add_argument("--strict-open-orders", action="store_true")
    ap.add_argument("--max-open-orders", type=int, default=0)

    ap.add_argument(
        "--auto-refresh", action="store_true", help="Attempt refresh if failing freshness checks."
    )
    ap.add_argument(
        "--refresh-when-closed",
        action="store_true",
        help="Override: allow auto-refresh even if market is closed (default: do NOT refresh when closed).",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # 1) run preflight
    pre = preflight_checks.run_all(args)

    report: Dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "mode": args.mode,
        "auto_refresh": bool(args.auto_refresh),
        "refresh_when_closed": bool(args.refresh_when_closed),
        "market_clock": None,
        "first_pass_ok": bool(pre.get("ok", False)),
        "first_pass_failures": _summarize_failures(pre),
        "refresh_attempted": False,
        "refresh_skipped_reason": None,
        "refresh_result": None,
        "second_pass_ok": None,
        "second_pass_failures": None,
        "final_ok": None,
    }

    if pre.get("ok", False):
        report["final_ok"] = True
        AUTORUN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[preflight_autorun] PASS (first pass)")
        sys.exit(0)

    # If not ok:
    refreshable = _is_refreshable_failure(pre)
    if (not args.auto_refresh) or (not refreshable):
        report["final_ok"] = False
        AUTORUN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[preflight_autorun] FAIL (not refreshable or auto-refresh disabled)")
        for f in report["first_pass_failures"]:
            print(f"  - FAIL {f['name']}: {f['message']}")
        sys.exit(2)

    # 2) market closed guard
    clk = _get_market_clock(args.mode, verbose=args.verbose)
    report["market_clock"] = clk

    # If we can tell and market is closed, skip refresh unless override
    if clk.get("ok") and clk.get("is_open") is False and not args.refresh_when_closed:
        report["refresh_skipped_reason"] = "MARKET_CLOSED"
        report["final_ok"] = False
        AUTORUN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(
            "[preflight_autorun] FAIL (freshness issue) but SKIPPING auto-refresh because market is closed."
        )
        if clk.get("next_open"):
            print(f"[preflight_autorun] next_open={clk.get('next_open')}")
        for f in report["first_pass_failures"]:
            print(f"  - FAIL {f['name']}: {f['message']}")
        sys.exit(2)

    # 3) refresh attempt
    report["refresh_attempted"] = True
    refresh_res = _refresh_pipeline(verbose=args.verbose)
    report["refresh_result"] = refresh_res

    if not refresh_res.get("ok", False):
        report["final_ok"] = False
        AUTORUN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[preflight_autorun] FAIL (refresh attempt failed)")
        sys.exit(3)

    # 4) rerun preflight
    post = preflight_checks.run_all(args)
    report["second_pass_ok"] = bool(post.get("ok", False))
    report["second_pass_failures"] = _summarize_failures(post)
    report["final_ok"] = bool(post.get("ok", False))

    AUTORUN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if post.get("ok", False):
        print("[preflight_autorun] PASS (after refresh)")
        sys.exit(0)

    print("[preflight_autorun] FAIL (still failing after refresh)")
    for f in report["second_pass_failures"] or []:
        print(f"  - FAIL {f['name']}: {f['message']}")
    sys.exit(3)


if __name__ == "__main__":
    main()
