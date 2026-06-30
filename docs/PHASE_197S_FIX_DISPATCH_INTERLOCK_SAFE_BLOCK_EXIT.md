# Phase 197S-FIX3 — Dispatch Interlock Safe-Block Exit Semantics

## Problem

After 197S-FIX2, the dispatch interlock correctly blocked defensive exit dispatch when the real
operator approval artifact was absent:

```
[DISPATCH_INTERLOCK] status=DISPATCH_BLOCKED_APPROVAL_REQUIRED stage=manage_positions ...
```

No broker orders were submitted. However, `manage_positions` returned exit code **2**, which caused:

```
[FAIL] Step 'F. POSITION MANAGEMENT' failed (exit 2)
[ABORT] Market-open sequence aborted at step: F. POSITION MANAGEMENT
[CYCLE_FAIL] cycle=1 exit_code=2
```

The safety block was correct; the runtime outcome classification was not.

## Fix

1. Define `SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED` and helpers in `defensive_exit_dispatch_interlock.py`.
2. When `DISPATCH_BLOCKED_APPROVAL_REQUIRED`, `manage_positions` exits **0** after auditing and emits:
   - `[DISPATCH_INTERLOCK] status=DISPATCH_BLOCKED_APPROVAL_REQUIRED stage=manage_positions ...`
   - `[SAFE_BLOCK] state=SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED reason=DISPATCH_APPROVAL_REQUIRED`
   - `broker_orders_submitted=false`
   - `eligible_for_execution=false`
   - `[MANAGE_SUMMARY] ... executed=0 blocked=1`
3. `place_live_orders` uses the same safe-block exit **0** if reached on manage handoff.
4. 197S verifier adds `POST_INTERLOCK_RUNTIME_VERIFIED_SAFE_BLOCK` and treats pre-fix abort logs
   (interlock present, no `[SAFE_BLOCK]`, sequence aborted) as `POST_INTERLOCK_RUNTIME_FAIL_SEQUENCE_ABORT_WITHOUT_SAFE_BLOCK`.

Exit **0** does **not** mean execution is eligible. All governance gates and interlock state remain enforced.

## Validation

```powershell
python -m services.phase_197s_fix_dispatch_interlock_safe_block_exit_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197s_fix_dispatch_interlock_safe_block_exit_validation.ps1
python -m services.phase_197s_post_interlock_runtime_verification
```

After the next continuous-loop restart, expect:

- `[SAFE_BLOCK]` in market-open log
- No `[ABORT]` for step F on expected missing-approval block
- `[CYCLE_OK]` instead of `[CYCLE_FAIL]`
- 197S status `PASS_SAFE_BLOCK`

## Preserved

No approval artifact; no broker orders; live blocked; no loop start/stop from this phase.
