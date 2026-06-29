# Phase 197E — Runtime Quarantine Registry Write Discipline

## Problem

Phase 197D quarantine worked correctly (`QUARANTINED_GAPS`, SPCX quarantined), but every
runtime preflight cycle rewrote the tracked governance file:

`data/governance/symbol_hydration_quarantine.json`

The only diff was `generated_at` timestamp churn — no symbol classification, reason, or
evidence changed. This dirtied git on every supervisor cycle.

## Root cause

`persist_quarantine_registry()` always wrote a new document with a fresh `generated_at`,
even when substantive quarantine content was identical to the existing registry.

## Fix

`services/symbol_hydration_quarantine.py`:

1. Load existing registry before write
2. Compare substantive content (`framework` + `symbols`) — **ignore `generated_at`**
3. Skip file write when unchanged (`skip_reason = unchanged_substance`)
4. Update `generated_at` only on substantive quarantine changes
5. Preserve trailing newline formatting on writes

Runtime observations and timestamps remain in:

`data/results/runtime_universe_hydration_preflight.json`

Audit field: `quarantine_registry_write` records whether the governance file was written or
skipped.

## Substantive changes (registry write allowed)

- New symbol added to quarantine
- Symbol removed from quarantine
- Classification change
- Reason change
- Evidence change (row counts, skipped-ticker reason, etc.)
- Framework/schema change

## Validation

```powershell
python -m services.runtime_universe_hydration_preflight --verbose
python -m services.runtime_universe_hydration_preflight --verbose
python -m services.phase_197e_quarantine_registry_write_discipline_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197e_quarantine_registry_write_discipline_validation.ps1
python -m services.phase_197d_spcx_unsupported_symbol_quarantine_validation
python -m services.phase_197c_runtime_hydration_integration_validation
python -m services.paper_entry_loop_stage --verbose --dry-run
python -m compileall services scripts
git status --short
```

## Preserved

Phases 196X–196Z, 197A–197D behavior; SPCX quarantine; 45-minute cadence; runtime authority;
no live execution; no broker orders; all execution/safety gates intact.

## Restart

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_triton_runtime_supervisor.ps1
```
