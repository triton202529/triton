# Phase 197S-FIX2 — Dispatch Interlock Runtime Import / Crash Regression Repair

## Problem

Post-197R continuous-loop cycle crashed at `F. POSITION MANAGEMENT` with:

```
NameError: name 'is_defensive_exit_manage_handoff' is not defined
```

`manage_positions.maybe_execute_management` called `is_defensive_exit_manage_handoff(allowed)`
before importing it from `defensive_exit_dispatch_interlock`.

## Fix

1. Import `is_defensive_exit_manage_handoff` inside the paper-mode block **before** use.
2. Add `stage=` to stable `[DISPATCH_INTERLOCK]` markers:
   - `stage=manage_positions`
   - `stage=place_live_orders`
3. Regression validation simulates missing approval → `DISPATCH_BLOCKED_APPROVAL_REQUIRED`.
4. 197S verifier classifies crash logs as `POST_INTERLOCK_RUNTIME_FAIL_INTERLOCK_IMPORT_ERROR` (FAIL).

## Validation

```powershell
python -m services.phase_197s_fix_dispatch_interlock_runtime_import_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197s_fix_dispatch_interlock_runtime_import_validation.ps1
python -m services.phase_197s_post_interlock_runtime_verification
```

After fix, restart continuous for one observation cycle and re-run 197S to reach `POST_INTERLOCK_RUNTIME_VERIFIED`.

## Preserved

No approval artifact; no broker orders; live blocked.
