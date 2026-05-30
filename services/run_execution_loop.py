# services/run_execution_loop.py
"""Continuous paper execution loop: execute_trades → manage_positions → manage_open_orders → poll_order_status."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

OPEN_SLEEP_SECONDS = 60
CLOSED_SLEEP_SECONDS = 300
# Back-compat alias
LOOP_INTERVAL_SECONDS = OPEN_SLEEP_SECONDS


# -----------------------------------------------------------
# Lifecycle self-heal helpers
# Keep signal_lifecycle_effective.csv fresh *before* each
# execute_trades cycle so the child's internal lifecycle gate
# never sees a stale effective file (STALE_EFFECTIVE).
# -----------------------------------------------------------
def _run_module_checked(
    module_name: str, label: str, extra_args: Optional[list[str]] = None
) -> bool:
    """Run `python -m <module_name>`; return True on success, False on failure."""
    cmd = [sys.executable, "-m", module_name] + list(extra_args or [])
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[LOOP] {label} FAILED")
        print(result.stdout)
        print(result.stderr)
        return False

    return True


def _self_heal_effective_lifecycle() -> bool:
    """Rebuild `signal_lifecycle_effective.csv` to clear STALE_EFFECTIVE conditions."""
    print("[LIFECYCLE_GATE] self-heal triggered...")
    ok = _run_module_checked("services.build_effective_lifecycle", "build_effective_lifecycle")
    if ok:
        print("[LIFECYCLE_GATE] self-heal complete")
    return ok


def _python_exe() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _market_clock_state(mode: str = "paper") -> Tuple[Optional[bool], str]:
    """
    Returns (is_open, next_open_str).
    is_open None => clock unavailable; caller should use open-market fallback (full cycle).
    """
    try:
        from services.broker_alpaca import AlpacaBroker

        br = AlpacaBroker(mode=mode)
        ck = br.get_clock()
        if not isinstance(ck, dict):
            return None, ""
        io = ck.get("is_open")
        if io is None:
            return None, ""
        no = ck.get("next_open")
        next_s = "" if no is None else str(no).strip()
        return bool(io), next_s
    except Exception:
        return None, ""


_DEGRADED_REASON_RE = re.compile(r"DEGRADED\s+reason=([a-z0-9_]+)", re.IGNORECASE)


def _parse_degraded_reason_from_subprocess_out(blob: str) -> str:
    """
    Match lines like: [POLL] DEGRADED reason=dns, [MOO] DEGRADED reason=network
    (maps to network | broker_timeout | dns | connection_reset as emitted by child tools).
    """
    m = _DEGRADED_REASON_RE.search(blob or "")
    if m:
        return m.group(1).lower()
    if "BROKER_RETRY" in (blob or ""):
        return "network"
    return "network"


def _run_step(cmd: list[str], step_name: str) -> int:
    """
    Run a subprocess; return 0/1/2+ (1 = DEGRADED: transient/operator-safe from poll/MOO, etc.)
    See each module for exit contract; rc=1 does not mean strategy failure.
    """
    try:
        r = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        if r.stdout:
            sys.stdout.write(r.stdout)
            if not r.stdout.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        if r.stderr:
            sys.stderr.write(r.stderr)
            if not r.stderr.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()

        rc = int(r.returncode)
        if rc == 0:
            print(f"[LOOP] step={step_name} DONE", flush=True)
        elif rc == 1:
            why = _parse_degraded_reason_from_subprocess_out(out)
            print(
                f"[LOOP] step={step_name} DEGRADED reason={why}",
                flush=True,
            )
        else:
            print(
                f"[LOOP] step={step_name} FAILED rc={r.returncode}",
                flush=True,
            )
        return rc
    except Exception as e:
        print(f"[LOOP] step={step_name} ERROR {e}", flush=True)
        return 3


def _print_loop_health(rcs: List[int], *, step_names: List[str]) -> None:
    """Operator-facing; rc 1 = degraded, >=2 = failed, 0 = ok."""
    if any(x >= 2 for x in rcs):
        cyc = "FAILED"
    elif any(x == 1 for x in rcs):
        cyc = "DEGRADED"
    else:
        cyc = "OK"
    bsum = 0
    bmul = 0
    for sn, c in zip(step_names, rcs):
        if sn in ("poll_order_status", "manage_open_orders"):
            bsum += 1
            bmul += 0 if c == 0 else 1
    if bsum == 0:
        brk = "OK"
    elif bmul == 0:
        brk = "OK"
    elif bmul < bsum:
        brk = "UNSTABLE"
    else:
        brk = "DOWN" if cyc == "FAILED" else "UNSTABLE"
    strat = "OK"
    if cyc == "FAILED":
        strat = "BLOCKED"
    elif cyc == "DEGRADED" and bmul:
        strat = "IDLE"
    print(
        f"[LOOP_HEALTH] cycle_status={cyc} broker_connectivity={brk} strategy_state={strat}",
        flush=True,
    )


def main() -> None:
    py = _python_exe()
    try:
        while True:
            is_open, next_open = _market_clock_state("paper")

            # Closed: no execute_trades / manage_open_orders (or manage_positions); long sleep
            if is_open is False:
                print("[LOOP] market_closed", flush=True)
                print(f"[LOOP] next_open={next_open or 'unknown'}", flush=True)
                print(f"[LOOP] sleeping_seconds={CLOSED_SLEEP_SECONDS}", flush=True)
                prc = _run_step(
                    [py, "-m", "services.poll_order_status", "--mode", "paper", "--refresh"],
                    "poll_order_status",
                )
                _print_loop_health([prc], step_names=["poll_order_status"])
                time.sleep(CLOSED_SLEEP_SECONDS)
                continue

            # Open or clock unknown: full cycle + short sleep (fallback matches prior behavior)
            print(f"[LOOP] cycle start: {_ts()}", flush=True)

            # Preemptively rebuild signal_lifecycle_effective.csv so the
            # internal lifecycle gate inside execute_trades never blocks
            # on STALE_EFFECTIVE at the start of a fresh cycle.
            _self_heal_effective_lifecycle()

            nms = [
                "execute_trades",
                "manage_positions",
                "manage_open_orders",
                "poll_order_status",
            ]
            rcs: List[int] = []
            rcs.append(
                _run_step(
                    [py, "-m", "services.execute_trades", "--mode", "paper", "--execute"],
                    nms[0],
                )
            )
            rcs.append(
                _run_step(
                    [py, "-m", "services.manage_positions", "--mode", "paper"],
                    nms[1],
                )
            )
            rcs.append(
                _run_step(
                    [py, "-m", "services.manage_open_orders", "--mode", "paper"],
                    nms[2],
                )
            )
            rcs.append(
                _run_step(
                    [py, "-m", "services.poll_order_status", "--mode", "paper", "--refresh"],
                    nms[3],
                )
            )
            _print_loop_health(rcs, step_names=nms)

            time.sleep(OPEN_SLEEP_SECONDS)
    except KeyboardInterrupt:
        print("[LOOP] stopped (KeyboardInterrupt)", flush=True)


if __name__ == "__main__":
    main()
