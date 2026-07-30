# DO-004 / DO-005 Defect Repair — Interruptible Cooperative Sleep and Stop-Timeout Alignment

## Observed incident

Protected Stop was requested against the authentic continuous loop:

- PID: `24008`
- Created: `2026-07-20 09:31:02`
- Command: `powershell.exe … run_continuous_loop.ps1 -IntervalMinutes 45`

Evidence showed the loop was healthy: lock owner matched, heartbeat advanced through cycle 161, latest cycle completed `[CYCLE_OK]` at `2026-07-29 20:38:16`, then the process entered its normal 45-minute inter-cycle wait. Protected Stop timed out after roughly 18 seconds with no acknowledgement. The shutdown request later became inactive. Watchdog was already stopped; paper authorization was expired; live execution remained blocked.

This was **not** a hung-loop incident. The earlier dashboard overdue reading was stale/outdated relative to live heartbeat evidence.

## Why PID 24008 was proven healthy

- Process remained responsive.
- Lock ownership matched PID 24008.
- Heartbeat/cycle metadata advanced through cycle 161.
- Cycles completed with `[CYCLE_OK]`.
- Behavior after cycle completion matched the configured 45-minute sleep cadence.

## Why the original Stop timed out

Two coordination defects combined:

1. **Legacy in-memory script (primary for PID 24008)**
   PID 24008 started on `2026-07-20`. Cooperative interruptible wait was introduced in commit `79509b43` (`TRITON: add cooperative shutdown protocol`) on `2026-07-29`. The running process therefore continued executing the pre-DO-004 path:

   `Start-Sleep -Seconds ($IntervalMinutes * 60)`

   That is one uninterruptible ~2700-second wait with **no** shutdown-request inspection during sleep.

2. **Controller acknowledgement wait vs observation interval**
   Protected Stop / DO-004 acknowledgement wait was on the order of tens of seconds (`ACK_TIMEOUT_SEC`, previously 30). A loop that only inspects shutdown requests at pre/post-cycle boundaries cannot acknowledge during a 45-minute blocking sleep. The controller correctly reported an explicit acknowledgement timeout and did not force-terminate.

Additionally, marking a timed-out request as terminal (`TIMED_OUT`) could clear unresolved-shutdown Start blocking while the process was still running, and PowerShell piping of sleep helper output through `ForEach-Object` could clobber `$LASTEXITCODE` for supported (post-DO-004) scripts.

## Exact root cause

**Coordination mismatch + legacy runtime:**

- Supported continuous cadence: 45 minutes between cycles.
- Protected Stop acknowledgement timeout: seconds.
- Pre-DO-004 continuous loop does not observe shutdown requests during sleep.
- Therefore Stop expires / times out before the next safe boundary for PID 24008.

For **supported** post-repair runtimes, the remaining defects addressed here are:

- preserve interruptible cooperative wait (10s continuous / 5s watchdog checks),
- capture cooperative-sleep exit codes reliably,
- align acknowledgement timeout to the observation interval + bounded margin (45s),
- keep the typed request observable (non-terminal) after controller acknowledgement timeout so Start/Restart remain blocked and a supported loop can still acknowledge until `expires_at`.

## Repaired observation interval

| Component | Inter-cycle cadence | Cooperative check interval |
|-----------|---------------------|----------------------------|
| Continuous | 45 minutes (unchanged) | every **10 seconds** during wait |
| Watchdog | 1 minute (unchanged) | every **5 seconds** during wait |

Controller acknowledgement timeout:

`recommended_ack_timeout_sec() = CONTINUOUS_WAIT_CHECK_SEC + ACK_OBSERVATION_MARGIN_SEC = 10 + 35 = 45`

Exit timeout remains separate (`EXIT_TIMEOUT_SEC = 60`).

## Preservation of the 45-minute cycle cadence

Interruptible wait sleeps the full configured interval in bounded chunks. It does **not** start a trading cycle early, call `run_market_open`, or invoke paper-entry stages during sleep. Cadence remains `IntervalMinutes * 60` total wait when no matching shutdown request is accepted.

## Acknowledgement-versus-completion distinction

| Stage | Meaning |
|-------|---------|
| Request accepted / delivered | Typed shutdown request written and active |
| Acknowledgement observed | Loop wrote ACCEPTED/EXITING ack at a safe boundary |
| Cooperative exit initiated | Loop chose EXIT_NOW and proceeds through normal cleanup |
| Process disappearance verified | Classifier shows targeted PID absent |
| Owner-lock release verified | Component lock no longer present |
| Stop fully completed | Ack + exit + lock release all verified |
| Explicit timeout | Controller observation/completion deadline elapsed |
| Partial completion | Some stages succeeded; request remains unresolved |

**Acknowledgement alone is never full completion.**

## Timeout behavior

- Timeout remains an explicit failure state (`SHUTDOWN_TIMED_OUT` / `STOP_TIMED_OUT`).
- Timeout never force-terminates a process.
- Timeout never deletes or repairs a lock.
- Timeout never automatically retries.
- Timeout never authorizes Start or Restart.
- After acknowledgement timeout, request state remains `DELIVERED` with `controller_ack_timed_out=true` (non-terminal) until TTL, completion, or later resolution so Start/Restart stay blocked and a supported loop may still acknowledge.
- After ack-with-incomplete-exit, request remains `EXIT_PENDING` (non-terminal).

## Safety boundaries

Safe boundaries for acknowledgement/exit:

- `PRE_CYCLE` / `BETWEEN_CYCLES`
- `POST_CYCLE_PRE_SLEEP`
- `INTERRUPTIBLE_WAIT` (inter-cycle sleep only — no cycle work in flight)

`INTERRUPTIBLE_WAIT` is safe because the continuous loop has finished the current cycle’s market-open/paper-entry work and is only waiting before the next cycle. Checking the existing typed shutdown request and exiting through normal `finally` lock release does not cut mid-cycle work and does not start a new cycle.

## Helper failure vs request refusal

Typed request refusal (`exit 11`) is distinct from helper failure. Refusal at PRE_CYCLE continues normal cadence without accepting shutdown. Unexpected helper exit codes (crash, missing interpreter, `EXIT_ERROR`, unknown) enter a bounded degraded wait (`HELPER_DEGRADED_WAIT_SEC = 30`) and **refuse to start a new cycle** until the helper returns a known PRE_CYCLE code (`0`, `10`, or `11`). Unexpected codes never impersonate acceptance (`10`).

Sleep-mode CLI emits only `0` or `10`. After interruptible sleep, any non-zero code other than `10` (including a hypothetical `11`) is treated as degraded so an early helper return cannot shorten cadence or open a new cycle.

## Legacy-runtime limitation

**This repair does not apply to the currently running PID 24008.**

That process still executes the July 20 in-memory PowerShell script with blocking `Start-Sleep`. It cannot load the repaired script without a separately governed process replacement. Do not claim PID 24008 has received the repair.

## Validation results

Repair suite: `tests/test_cooperative_sleep_stop_timeout_repair.py`
Plus DO-002 through DO-007 regressions (see operator validation run output for exact counts).

## Operator procedure after the repair is eventually committed

1. Confirm live execution remains blocked; do not rearm paper solely for this repair.
2. Refresh the Operator Console after deploy.
3. Treat PID 24008 as a **legacy runtime** until it is replaced under a separately governed one-time shutdown proposal (see disposition below). Do **not** blindly repeat Stop expecting the repair to take effect in that process.
4. For any **new** continuous loop started after this repair is deployed: Protected Stop should be observable within one cooperative check interval (~10s) plus bounded processing, with acknowledgement timeout aligned to 45s, and completion still requiring process absence + owner-lock release.
5. If Stop times out with `controller_ack_timed_out` while the request remains `DELIVERED`, Start/Restart remain blocked; investigate Diagnostics; do not delete locks; do not force-terminate as ordinary workflow.
