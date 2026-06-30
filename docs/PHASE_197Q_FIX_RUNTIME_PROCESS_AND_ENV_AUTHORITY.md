# Phase 197Q-FIX — Runtime Process Detection + CLI/Supervisor Env Authority

## Problem

After Phase 197Q, two consistency issues appeared:

1. **False continuous-loop detection** — `continuous_loop_running=True` when only the
   watchdog was running and `continuous_loop.lock` was absent. Root cause: process scan
   PowerShell subprocess matched its own `-Command` text containing `run_continuous_loop.ps1`.

2. **CLI env mismatch** — Phase 197M failed in standalone CLI because
   `TRITON_ENABLE_PAPER_EXECUTION` is set by supervisor/continuous scripts, not plain Python.

## Fix

### Process detection (`services/runtime_process_detector.py`)

- Excludes detection subprocess (`$PID -ne $PID` in scan query)
- Requires `-File ... run_continuous_loop.ps1` style invocation (not bare substring)
- Excludes `Get-CimInstance` / validation command lines
- Reports PIDs, detection method, confidence, warnings
- Reconciles process + lock into states:
  - `RUNTIME_PROCESS_STOPPED_LOCK_ABSENT`
  - `RUNTIME_PROCESS_RUNNING_LOCK_PRESENT`
  - etc.

Also updated `scripts/loop_lock_utils.ps1` with the same self-exclusion logic.

### Paper env authority (`services/triton_env_loader.py`)

States:

| State | Meaning |
|-------|---------|
| `PAPER_ENV_AUTHORITY_SUPERVISOR_ONLY` | Paper flags set by runtime supervisor/continuous loop |
| `PAPER_ENV_AUTHORITY_CLI_LOADED` | Validator loaded local env via `load_cli_validation_env()` |
| `PAPER_ENV_AUTHORITY_SHARED_BOOTSTRAP` | Bootstrap files loaded |
| `PAPER_ENV_AUTHORITY_MISSING` | Paper env not armed |
| `PAPER_ENV_AUTHORITY_CONFLICT` | Live flags enabled or partial paper flags |

Missing paper env classifies as `SAFE_BLOCK_PAPER_ENV_NOT_ARMED` (not execution-ready).

### Validator updates

- `live_block_classifier.validator_paper_gate_acceptable` accepts `SAFE_BLOCK_PAPER_ENV_NOT_ARMED`
- `phase_197m_fix_live_block_detection_validation` loads CLI env before gate checks
- `runtime_reentry_observation_gate` uses corrected process detector

## Validation

```powershell
python -m services.phase_197q_fix_runtime_process_and_env_authority_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197q_fix_runtime_process_and_env_authority_validation.ps1
python -m services.phase_197q_runtime_reentry_observation_validation
python -m services.phase_197m_fix_live_block_detection_validation
```

## Preserved

No loop start/stop; no approval artifact; no execution; live remains blocked.
