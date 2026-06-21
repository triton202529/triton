# Monday Paper Trading Readiness Runbook

Phase 196A — operator steps for Sunday evening through Monday market open.

## Sunday evening — readiness check

```powershell
cd C:\Users\akimw\triton
python -m services.phase_196a_monday_paper_readiness_hardening
```

Review `data/results/monday_paper_readiness_hardening.json`:

- `monday_readiness_status` should be `READY_FOR_MONDAY` or `READY_FOR_MONDAY_PENDING_APPROVAL`
- `market_gate_status` should be `SAFE_BLOCK` (weekend/market closed is expected)
- Watchdog count = 1, continuous count = 1
- `live_execution_permitted` = false

## Sunday — repair stale lifecycle/signals (if needed)

If blockers mention stale lifecycle or signals, or `execution_readiness` is `NOT_READY`:

```powershell
python -m services.phase_196a_monday_paper_readiness_hardening --repair
python -m services.phase_196a_monday_paper_readiness_hardening
python -m services.execution_readiness
python -m services.phase_195h_operational_readiness_freeze
```

Repair runs the official chain only (no orders, no governance fabrication):

- `fetch_raw_data` → `preprocess` → `train_model` → `generate_signals`
- `apply_signal_lifecycle` → `build_effective_lifecycle` → `build_trade_opportunities`
- `execution_readiness` refresh
- ARM paper confirm refresh if ASSISTED mode and confirm expired

## Monday — 15 minutes before market open (09:15 ET)

1. Refresh ARM confirmation (ASSISTED mode):

```powershell
python -m services.set_paper_arm_confirm --ttl 480
python -m services.arm_mode --status
```

2. Verify broker snapshot (read-only):

```powershell
python -m services.snapshot_live_orders --mode paper
```

3. Verify readiness:

```powershell
python -m services.execution_readiness
python -m services.phase_195h_operational_readiness_freeze
python -m services.phase_196a_monday_paper_readiness_hardening
```

4. Set environment and restart **continuous loop only** for immediate retry at open:

```powershell
$env:TRITON_ENABLE_PAPER_EXECUTION='1'
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES='1'

# Stop existing continuous loop process first (note PID from 196A output), then:
Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\Users\akimw\triton\scripts\run_continuous_loop.ps1','-IntervalMinutes','45' -WorkingDirectory 'C:\Users\akimw\triton'
```

Watchdog loop can keep running; do not restart unless unhealthy.

## Monday — after first paper-entry cycle generates governance request

List pending paper-only approvals:

```powershell
python -m services.execution_approval_workflow list-pending --paper-only
```

Approve latest paper capital deployment (only when symbols match planned entries):

```powershell
python -m services.execution_approval_workflow approve-latest-paper `
  --authority "Akim Wallace" `
  --rationale "Paper capital deployment approved after readiness, ARM, broker snapshot, market-gate, and loop verification"
```

Reject unsafe requests:

```powershell
python -m services.execution_approval_workflow reject --approval-id eg-apr-XXXXXXXXXXXX `
  --authority "Akim Wallace" `
  --rationale "Reason for rejection"
```

## Monday — after approval

Restart continuous loop again so the next cycle picks up the approved governance record:

```powershell
# stop prior continuous loop, then:
Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\Users\akimw\triton\scripts\run_continuous_loop.ps1','-IntervalMinutes','45' -WorkingDirectory 'C:\Users\akimw\triton'
```

## Confirm paper execution

1. `data/results/execution_plan.json` — `dry_run=false`, `orders_executed` > 0, `blocked=false`
2. `data/results/paper_execution_audit.csv` — submission records
3. `python -m services.snapshot_live_orders --mode paper` — open orders / positions updated
4. Continuous loop log — `[PAPER_ENTRY_GATE] status=PASS` then `[PAPER_ENTRY_EXECUTION] completed exit_code=0`

## What must never be approved

- `liveMode=true` or `execution_path=live_execution`
- Expired requests
- Empty symbol lists
- `execution_gate_decision != ALLOW`
- Live mixed context

Do not manually edit `data/governance/execution_approval_requests.json`.

## Continuous loop weekend behavior

While market is closed the loop logs:

`[PAPER_ENTRY_GATE] status=SAFE_BLOCK reason=market is closed`

No orders are submitted. Cycle failures do not stop the loop; it sleeps and retries.

## Final safety checks

```powershell
python -m services.execution_readiness
python -m services.phase_195z_credential_log_masking_validation
git status --short
```

Confirm `live_execution_permitted` remains false.
