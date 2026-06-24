# TRITON Paper ARM Operator Runbook

Read-only reference for the paper ARM operator latch. This does **not** arm paper mode or submit orders.

## What is `paper_arm_confirm`?

`data/live/paper_arm_confirm.json` is an **operator-controlled time-limited latch** used when ARM mode is **ASSISTED** (`config/arm_mode.json`).

It authorizes **paper-mode mutation steps** (for example controlled paper entry paths) for a limited window. It is:

- **Not** live execution
- **Not** a broker order
- **Not** the same as `live_confirm.json`
- **Not** capital preservation certification

## Why ASSISTED mode requires it

ASSISTED mode is designed so automated loops cannot place or mutate paper positions without a fresh operator acknowledgment. The continuous loop may run, readiness may be READY, and paper policy may authorize execution — but **paper entry stays blocked** until `paper_arm_confirm` is valid.

## Schema

```json
{
  "timestamp": "2026-06-23T13:10:40Z",
  "mode": "ASSISTED",
  "expires_at": "2026-06-23T21:10:40Z",
  "allow_execute": true
}
```

## TTL

- Config default: `confirmation_ttl_minutes` in `config/arm_mode.json` (currently 30 minutes).
- Writer sets explicit `expires_at` when created via `services.set_paper_arm_confirm --ttl <minutes>`.

## Check status (read-only)

PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_triton_paper_arm_status.ps1
```

Python:

```powershell
.\.venv\Scripts\python.exe -m services.paper_arm_status
.\.venv\Scripts\python.exe -m services.paper_arm_status --json
```

Dashboard sidebar and **Execution Dashboard → ARM Mode Status** also show paper arm vs live confirm separately.

## What would arm paper mode (operator only)

Only after explicit operator approval:

```powershell
.\.venv\Scripts\python.exe -m services.set_paper_arm_confirm --ttl 480
```

This writes `paper_arm_confirm.json` only. It does **not**:

- enable `live_execution`
- write `live_confirm.json`
- call `execute_trades`
- submit broker orders

## Related gates (do not conflate)

| Gate | Artifact | Purpose |
|------|----------|---------|
| Paper arm confirm | `data/live/paper_arm_confirm.json` | ASSISTED paper mutation latch |
| Live confirm | `data/live/live_confirm.json` | Live execution typed confirmation |
| Certification | `data/results/capital_preservation_certification.json` | Governance capital preservation |
| Execution readiness | `data/results/execution_readiness.json` | Operational checks (can be READY while cert is NOT_CERTIFIED) |

## Doctrine

- Do **not** run `python -m services.execute_trades` manually.
- Do **not** weaken or bypass paper arm, certification, or live confirm gates.
- Arming paper is a deliberate operator action after review — not automatic at market open.
