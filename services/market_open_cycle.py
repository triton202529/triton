# services/market_open_cycle.py
"""
TRITON — Market Open Cycle Runner (Repo-aligned)
------------------------------------------------
- Uses AlpacaBroker directly to list/cancel/check pending (since helper modules don't exist)
- Calls existing modules:
    - services.execute_cycle
    - services.poll_order_status
- Computes placement_session exactly as execute_cycle does when --refresh-orders is enabled
- Adds a robust runner-side session summary that doesn't depend on poll_order_status internals
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


RESULTS_DIR = Path("data") / "results"


@dataclass
class StepResult:
    name: str
    ok: bool
    returncode: int
    detail: str = ""


def _now_utc_tag_full() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _now_utc_tag_short_r() -> str:
    return "R" + datetime.now(timezone.utc).strftime("%H%M%SZ")


def _python_mod(module: str, args: List[str]) -> List[str]:
    return [sys.executable, "-m", module, *args]


def _run_module(module: str, args: List[str], name: str) -> StepResult:
    cmd = _python_mod(module, args)
    print(f"\n=== {name} ===")
    print("CMD:", " ".join(cmd))
    p = subprocess.run(cmd, text=True)
    ok = p.returncode == 0
    return StepResult(name=name, ok=ok, returncode=p.returncode)


def _fmt_order(o: dict) -> str:
    sym = o.get("symbol", "?")
    side = o.get("side", "?")
    typ = o.get("type", "?")
    st = o.get("status", "?")
    oid = o.get("id", "?")
    lp = o.get("limit_price", o.get("limitPrice", None))
    tif = o.get("time_in_force", o.get("timeInForce", None))
    return f"{sym} {side} {typ} {st} id={oid} limit={lp} tif={tif}"


def _list_open_orders(mode: str, limit: int = 200) -> StepResult:
    print("\n=== 1) List Open Orders (broker) ===")
    try:
        from services.broker_alpaca import AlpacaBroker

        b = AlpacaBroker(mode=mode)
        orders = b.list_orders(status="open", nested=True, limit=limit) or []
        print(f"open={len(orders)} (mode={mode})")
        for o in orders[:25]:
            print(" ", _fmt_order(o))
        return StepResult(
            name="1) List Open Orders (broker)", ok=True, returncode=0, detail=f"open={len(orders)}"
        )
    except Exception as e:
        print("ERROR:", repr(e))
        return StepResult(
            name="1) List Open Orders (broker)", ok=False, returncode=1, detail=repr(e)
        )


def _cancel_open_orders(mode: str, limit: int = 200) -> StepResult:
    print("\n=== 2) Cancel Open Orders (broker) ===")
    try:
        from services.broker_alpaca import AlpacaBroker

        b = AlpacaBroker(mode=mode)
        orders = b.list_orders(status="open", nested=True, limit=limit) or []
        print(f"found open={len(orders)}")
        cancelled = 0
        failed = 0
        for o in orders:
            oid = o.get("id")
            if not oid:
                failed += 1
                continue
            try:
                b.cancel_order(oid)
                cancelled += 1
            except Exception as ce:
                failed += 1
                print("  cancel failed:", oid, repr(ce))
        print(f"cancelled={cancelled} failed={failed}")
        # Treat partial failures as non-fatal here (we can harden later if needed)
        return StepResult(
            name="2) Cancel Open Orders (broker)",
            ok=True,
            returncode=0,
            detail=f"cancelled={cancelled} failed={failed}",
        )
    except Exception as e:
        print("ERROR:", repr(e))
        return StepResult(
            name="2) Cancel Open Orders (broker)", ok=False, returncode=1, detail=repr(e)
        )


def _check_pending(mode: str, limit: int = 500) -> StepResult:
    print("\n=== 3) Check Pending Cancel/Replace (broker) ===")
    try:
        from services.broker_alpaca import AlpacaBroker

        b = AlpacaBroker(mode=mode)
        orders = b.list_orders(status="open", nested=True, limit=limit) or []
        pending = [o for o in orders if str(o.get("status", "")).startswith("pending_")]

        if not pending:
            print("pending=0 (good)")
            return StepResult(
                name="3) Check Pending (broker)", ok=True, returncode=0, detail="pending=0"
            )

        print(f"pending={len(pending)} (review):")
        for o in pending[:50]:
            print(" ", _fmt_order(o))

        # Mark as FAIL to highlight it, but runner can proceed.
        return StepResult(
            name="3) Check Pending (broker)",
            ok=False,
            returncode=1,
            detail=f"pending={len(pending)}",
        )
    except Exception as e:
        print("ERROR:", repr(e))
        return StepResult(name="3) Check Pending (broker)", ok=False, returncode=1, detail=repr(e))


def _recent_csvs(results_dir: Path, max_age_seconds: int = 600) -> List[Path]:
    if not results_dir.exists():
        return []
    now = time.time()
    csvs = []
    for p in results_dir.glob("*.csv"):
        try:
            age = now - p.stat().st_mtime
            if age <= max_age_seconds:
                csvs.append(p)
        except Exception:
            pass
    # newest first
    csvs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return csvs


def _find_session_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if "session" in lc:
            cols.append(c)
    return cols


def _string_contains_any(series: pd.Series, needles: List[str]) -> pd.Series:
    s = series.astype(str)
    mask = pd.Series(False, index=s.index)
    for n in needles:
        if n:
            mask = mask | s.str.contains(n, na=False)
    return mask


def _runner_session_summary(base_session: str, placement_session: str) -> None:
    """
    Best-effort summary using whatever poll_order_status wrote.
    We search recent CSVs, find session-like columns, filter, and print a compact summary.
    """
    print("\n=== Runner Session Summary (best-effort) ===")
    print("Looking in:", RESULTS_DIR.resolve())
    needles = [placement_session, base_session]

    csvs = _recent_csvs(RESULTS_DIR, max_age_seconds=900)
    if not csvs:
        print("No recent CSVs found to summarize.")
        return

    summarized_any = False
    for p in csvs[:12]:  # don't spam
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        sess_cols = _find_session_columns(df)
        if not sess_cols:
            continue

        # Build mask: any session-like column contains either session tag
        mask = pd.Series(False, index=df.index)
        for c in sess_cols:
            mask = mask | _string_contains_any(df[c], needles)

        hits = int(mask.sum())
        if hits == 0:
            continue

        summarized_any = True
        sub = df.loc[mask].copy()

        print(f"\nFile: {p.name} | rows={len(df)} | matched_rows={hits} | session_cols={sess_cols}")

        # If there's a status-like column, show counts
        status_cols = [
            c for c in sub.columns if str(c).lower() in ("status", "order_status", "state")
        ]
        if status_cols:
            sc = status_cols[0]
            try:
                vc = sub[sc].astype(str).value_counts().head(12)
                print(f"Top {sc} counts:")
                for k, v in vc.items():
                    print(f"  {k}: {v}")
            except Exception:
                pass

        # Show latest few rows
        tail_cols = []
        for c in [
            "timestamp",
            "ts",
            "updated_at",
            "filled_at",
            "submitted_at",
            "symbol",
            "side",
            "qty",
            "status",
            "order_status",
            "id",
            "order_id",
        ]:
            if c in sub.columns:
                tail_cols.append(c)
        tail_cols = tail_cols[:10] if tail_cols else list(sub.columns[:10])

        print("Latest rows:")
        print(sub[tail_cols].tail(8).to_string(index=False))

    if not summarized_any:
        print("No session-tagged rows found in recent CSVs.")
        print(
            "That strongly suggests poll_order_status is not writing session tags to its output file(s),"
        )
        print(
            "or it writes to a non-CSV artifact. Next step: we patch poll_order_status to persist session."
        )
        print(f"Needles tried: {needles}")


def main() -> None:
    ap = argparse.ArgumentParser(description="TRITON Market Open Cycle Runner (repo-aligned)")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--session", default="", help="Optional base session tag (auto if blank)")
    ap.add_argument(
        "--cancel-open", action="store_true", help="Cancel open orders before placement"
    )
    ap.add_argument("--poll-every", type=int, default=30)
    ap.add_argument("--poll-rounds", type=int, default=6)
    ap.add_argument("--skip-execute", action="store_true")
    ap.add_argument("--skip-poll", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--ignore-pending-cancel", action="store_true")
    ap.add_argument("--drop-illegal-sells", action="store_true", default=True)
    ap.add_argument("--allow-shorts", action="store_true")

    args = ap.parse_args()

    base_session = args.session.strip() or f"mktopen_{args.mode}_{_now_utc_tag_full()}"

    print("\nTRITON — Market Open Cycle")
    print("Mode          :", args.mode)
    print("Base session  :", base_session)

    results: List[StepResult] = []

    results.append(_list_open_orders(args.mode, limit=200))

    if args.cancel_open:
        results.append(_cancel_open_orders(args.mode, limit=200))
        results.append(_list_open_orders(args.mode, limit=200))

    results.append(_check_pending(args.mode, limit=500))

    placement_session = f"{base_session}_{_now_utc_tag_short_r()}"
    print("Placement session (expected):", placement_session)

    if not args.skip_execute:
        exec_args = ["--mode", args.mode, "--session", base_session, "--refresh-orders"]
        if args.verbose:
            exec_args.append("--verbose")
        if args.ignore_pending_cancel:
            exec_args.append("--ignore-pending-cancel")
        if args.drop_illegal_sells and not args.allow_shorts:
            exec_args.append("--drop-illegal-sells")
        if args.allow_shorts:
            exec_args.append("--allow-shorts")

        results.append(_run_module("services.execute_cycle", exec_args, "4) Execute Cycle"))

    if not args.skip_poll:
        for i in range(args.poll_rounds):
            poll_args = ["--mode", args.mode, "--session", placement_session, "--refresh"]
            if args.verbose:
                poll_args.append("--verbose")
            results.append(
                _run_module(
                    "services.poll_order_status",
                    poll_args,
                    f"5) Poll Orders ({i+1}/{args.poll_rounds})",
                )
            )
            if i < args.poll_rounds - 1:
                time.sleep(max(1, args.poll_every))

    # Runner-side summary (this is the important new part)
    _runner_session_summary(base_session=base_session, placement_session=placement_session)

    print("\n========================")
    print("Market Open Cycle Summary")
    print("========================")
    for r in results:
        status = "OK " if r.ok else "FAIL"
        extra = f" | {r.detail}" if r.detail else ""
        print(f"[{status}] {r.name} (rc={r.returncode}){extra}")

    # Hard-fail only if execute/poll failed
    exec_poll_failed = any(
        (not r.ok) and (r.name.startswith("4)") or r.name.startswith("5)")) for r in results
    )
    if exec_poll_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
