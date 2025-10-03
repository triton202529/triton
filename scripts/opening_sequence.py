# scripts/opening_sequence.py
from __future__ import annotations

import json
import sys
import time
import subprocess
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of /scripts)


def load_client() -> TradingClient:
    conf = json.loads((REPO_ROOT / "config" / "alpaca.json").read_text(encoding="utf-8-sig"))
    return (
        TradingClient(conf["key_id"], conf["secret_key"], paper=conf.get("paper", True)),
        conf,
    )


def open_orders_count(c: TradingClient) -> int:
    return len(list(c.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))))


def positions_count(c: TradingClient) -> int:
    return len(list(c.get_all_positions()))


def run_py(*args: str) -> int:
    """Run a Python module/script with the current interpreter, in repo root."""
    cmd = [sys.executable, *args]
    print(">", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=True).returncode


def wait_until_flat(c: TradingClient, timeout_s: int = 90, poll_s: int = 2) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if open_orders_count(c) == 0 and positions_count(c) == 0:
            return True
        time.sleep(poll_s)
    return False


def main() -> None:
    c, conf = load_client()
    paper_flag = ["--paper"] if conf.get("paper", True) else []

    clock = c.get_clock()
    if not getattr(clock, "is_open", False):
        print("⏸️ Market is closed — exiting without placing/closing orders.")
        return

    acct = c.get_account()
    print(f"🕒 Market OPEN. Buying power: {acct.buying_power}")

    # 1) Safety: cancel any straggler orders
    c.cancel_orders()
    time.sleep(1)  # brief settle
    print(f"🧹 Open orders after cancel: {open_orders_count(c)}")

    # 2) Flatten everything (no new buys)
    run_py(
        "scripts/auto_execute_signals.py",
        *paper_flag,
        "--force-close-all",
        "--cancel-all-open",
        "--max-buys",
        "0",
    )

    # 3) Verify we’re flat
    ok = wait_until_flat(c, timeout_s=120, poll_s=2)
    print(f"🔎 Open orders: {open_orders_count(c)} | Positions: {positions_count(c)}")
    if not ok:
        print("⚠️ Not fully flat after timeout — review open orders/positions and rerun.")
        return

    # 4) Run buys pass
    run_py(
        "scripts/auto_execute_signals.py",
        *paper_flag,
        "--dollars-per-trade",
        "50",
        "--autoscale",
        "--min-alloc",
        "25",
        "--max-buys",
        "25",
    )

    print("✅ Opening sequence complete.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"❌ Subprocess failed (exit {e.returncode}). See output above.")
    except Exception as e:
        print(f"❌ Unexpected error: {e.__class__.__name__}: {e}")
