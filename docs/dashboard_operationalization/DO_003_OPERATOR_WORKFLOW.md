# DO-003 — Operator Workflow

## Start TRITON (local only)

1. Launch console: `scripts/run_operator_console.ps1` (binds `127.0.0.1`)
2. Stay in **Operator Console** → **Operations**
3. Confirm banner shows **LIVE EXECUTION BLOCKED**
4. Read Start eligibility on the Runtime controls panel
5. If enabled, click **Start TRITON**
6. Read confirmation: paper-only runtime; live stays blocked; paper arm is not created/extended
7. Click **Confirm Start**
8. Wait for validation result

### Result meanings

| Result | Meaning |
|--------|---------|
| STARTED | Verified one continuous + one watchdog; live blocked |
| ALREADY_RUNNING | Already healthy; nothing launched |
| PARTIAL_START | One loop healthy, another failed; no auto-kill |
| START_REFUSED_* | Nothing started; review Diagnostics |
| START_IN_PROGRESS | Another Start is active; wait |

## Do not

- Use Stop/Restart (unavailable)
- Clear locks from the dashboard
- Expect Start to arm paper authorization
- Run PowerShell for routine Start when the button is eligible

## Backup

Task Scheduler `TRITON_Runtime_931AM` remains the market-open backup.
