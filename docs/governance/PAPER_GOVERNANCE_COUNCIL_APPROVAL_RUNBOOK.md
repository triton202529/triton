# Paper Governance Council Approval Runbook

Phase 195Z — operator guide for approving controlled paper capital deployment requests.

## When approval is required

`execute_trades --mode paper --execute` reaches planning with `dry_run=false` but broker submission is blocked when the governed execution path requires **GOVERNANCE_COUNCIL_APPROVAL** (`mandatory_council_path`).

Typical artifact: `data/results/execution_plan.json` shows `blocked=true` with `GOVERNANCE_COUNCIL_APPROVAL required`.

## Prerequisites before approving

Confirm all of the following:

- `execution_readiness` is **READY**
- ARM mode is **ASSISTED** or **AUTO** with valid `paper_arm_confirm.json` (if ASSISTED)
- `TRITON_ENABLE_PAPER_EXECUTION=1` and `TRITON_ENABLE_AUTO_PAPER_ENTRIES=1` on the continuous loop
- Request is **paper-only** (`paperMode=true`, `liveMode=false`)
- Request path is `capital_deployment` with `execution_gate_decision=ALLOW`
- Request has a non-empty symbol list matching planned entries
- Request is **PENDING** and not expired

## List pending approvals

All pending:

```powershell
python -m services.execution_approval_workflow list-pending
```

Paper capital deployment only (safe filter):

```powershell
python -m services.execution_approval_workflow list-pending --paper-only
```

## Approve latest pending paper capital deployment

Recommended for the current planned paper entry batch:

```powershell
python -m services.execution_approval_workflow approve-latest-paper `
  --authority "YOUR_NAME" `
  --rationale "Paper capital deployment approved for planned ENTRY batch after readiness/ARM verification"
```

## Approve a specific request

```powershell
python -m services.execution_approval_workflow approve `
  --id eg-apr-XXXXXXXXXXXX `
  --authority "YOUR_NAME" `
  --rationale "Specific approval rationale"
```

## Reject a request

```powershell
python -m services.execution_approval_workflow reject `
  --id eg-apr-XXXXXXXXXXXX `
  --authority "YOUR_NAME" `
  --rationale "Rejection reason"
```

## What must NEVER be approved

The CLI refuses (by design):

- `live_execution` or any request with `liveMode=true`
- Expired requests (`expires_at` in the past)
- Non-`PENDING` requests (already approved/rejected/cancelled/expired)
- Requests with empty symbol lists
- Requests where `execution_gate_decision != ALLOW`
- Non-`capital_deployment` paths via `approve-latest-paper` (use explicit review for other paths)

Do **not** manually edit `data/governance/execution_approval_requests.json`.

## After approval — rerun continuous loop cycle

Approval is consumed on the next governed gate evaluation. Either wait for the next 45-minute continuous loop cycle, or restart loops with env vars set:

```powershell
$env:TRITON_ENABLE_PAPER_EXECUTION='1'
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES='1'

# restart watchdog + continuous loop if needed (see Phase 195Y operator commands)
```

## Verify broker submission after next cycle

1. Check continuous loop log for `[PAPER_ENTRY_EXECUTION] completed exit_code=0`
2. Inspect `data/results/execution_plan.json` — `orders_executed` should increase, `blocked=false`
3. Run read-only snapshot:

```powershell
python -m services.snapshot_live_orders --mode paper
```

4. Confirm `data/results/open_orders_snapshot.csv` reflects new paper orders (or fills)
5. Re-run readiness:

```powershell
python -m services.execution_readiness
python -m services.phase_195h_operational_readiness_freeze
```

## Audit trail

Approvals append to:

- `data/governance/execution_approval_history.csv`
- `data/governance/execution_approval_requests.json` (atomic update)

## Validation

```powershell
python -m services.phase_195z_governance_council_paper_approval_wiring
python -m services.phase_195z_credential_log_masking_validation
```
