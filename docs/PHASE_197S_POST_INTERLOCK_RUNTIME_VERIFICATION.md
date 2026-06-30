# Phase 197S — Post-Interlock Runtime Verification / One-Cycle Observation Soak

## Problem

Phase 197R added the defensive-exit dispatch interlock in code and validated it against
historical logs, but the live continuous-loop had not yet run a cycle emitting
`[DISPATCH_INTERLOCK] status=DISPATCH_BLOCKED_APPROVAL_REQUIRED`.

## Purpose

Verify one post-197R continuous-loop cycle obeys the interlock before broker placement.

### Runtime states

| State | Meaning |
|-------|---------|
| `POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET` | No post-197R cycle with interlock in logs yet |
| `POST_INTERLOCK_RUNTIME_VERIFIED` | Interlock blocked approval-required exits; no broker orders |
| `POST_INTERLOCK_RUNTIME_FAIL_NOTIONAL_CAP_LAST_LINE` | Placement reached notional cap after interlock |
| `POST_INTERLOCK_RUNTIME_FAIL_BROKER_ORDERS_SUBMITTED` | Broker orders submitted |

### Observation command (text only — not executed by TRITON)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_triton_runtime_supervisor.ps1
```

After one cycle completes, re-run:

```powershell
python -m services.phase_197s_post_interlock_runtime_verification
```

## Expected outcomes

**Before restart:** `status=PASS_SAFE_WAIT`, `restart_required=true`,
`POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET`

**After one cycle:** `status=PASS`, `POST_INTERLOCK_RUNTIME_VERIFIED`,
`dispatch_interlock_state=DISPATCH_BLOCKED_APPROVAL_REQUIRED`

## Validation

```powershell
python -m services.phase_197s_post_interlock_runtime_verification
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197s_post_interlock_runtime_verification.ps1
python -m services.phase_197r_post_restart_dispatch_interlock_audit
```

## Preserved

No approval artifact; no broker orders; live blocked; observation only.
