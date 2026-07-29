# DO-004 — Safe Boundary Contract

## Continuous loop (`run_continuous_loop.ps1`)

**Safe unit of work:** one full cycle of `run_market_open.ps1` plus (on success) `paper_entry_loop_stage`, including bounded broker/order behavior already defined by those stages.

**Checkpoints:**

1. **PRE_CYCLE / BETWEEN_CYCLES** — before starting a new cycle. If accepted, no cycle begins.
2. **POST_CYCLE_PRE_SLEEP** — after the cycle’s work finishes; before sleep. If accepted, skip further cycles and exit.
3. **INTERRUPTIBLE_WAIT** — during the normal interval sleep (default 45 minutes), polled every **10 seconds**.

Shutdown never interrupts mid-cycle critical sections. After acceptance, no new cycle starts. Exit uses normal `finally` → `Remove-OwnLoopLock` (only if lock PID matches this process).

## Watchdog loop (`run_watchdog_loop.ps1`)

**Safe unit of work:** one `risk_watchdog` invocation and its bounded alert/status persistence.

**Checkpoints:** same three patterns. Wait poll interval: **5 seconds** (normal cadence remains 1 minute).

Watchdog never modifies the continuous lock, broker, order, approval, or authorization state.

## Child processes

Cycle children (e.g. market-open / paper-entry / risk_watchdog) are allowed to finish as part of the current safe unit. Shutdown is acknowledged only at documented boundaries after that unit completes (or before the unit starts).

## Interruptible wait rationale

Chunked sleep preserves operational cadence when no request exists, avoids busy polling, and meets recognition targets (~10s continuous / ~5s watchdog) without OS signals.
