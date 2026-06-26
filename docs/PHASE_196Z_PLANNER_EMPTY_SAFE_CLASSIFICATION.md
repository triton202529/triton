# Phase 196Z — Planner-Empty Safe Classification

## Problem

After Phase 196Y, when allocation produced eligible candidates but the governed planner
rejected every symbol for valid execution reasons (sector caps, score thresholds, cooldowns),
`paper_entry_loop_stage` classified the outcome as:

```
CONTRACT_MISMATCH:allocation_eligible_but_planner_empty
```

That was inaccurate. The allocation-to-planner contract was honored; there were simply no
executable candidates after planner filters.

## Fix

`evaluate_allocation_planner_contract()` in `paper_entry_candidate_contract.py` now
distinguishes:

| Outcome | `contract_status` | `gate_reason` |
|---------|-------------------|---------------|
| Valid planner-empty (all considered symbols rejected with reasons) | `PLANNER_EMPTY_VALID` | `NO_EXECUTABLE_CANDIDATES_AFTER_PLANNER_FILTERS` |
| Planner selected outside allocation | `PLANNER_OUTSIDE_ALLOCATION` | `CONTRACT_MISMATCH:...` |
| Planner considered outside allocation | `PLANNER_CONSIDERED_OUTSIDE_ALLOCATION` | `CONTRACT_MISMATCH:...` |
| Unexplained planner-empty | `PLANNER_EMPTY` | `CONTRACT_MISMATCH:...` |

## Audit fields

`paper_entry_loop_stage_audit.json` now includes:

- `contract_status`, `planner_empty_valid`, `no_executable_candidates`
- `planner_rejection_buckets` (e.g. `ADD_BLOCKED_BY_SECTOR_SOFT`, `BELOW_MIN_SCORE`)
- `planner_empty_rejection_coverage`, `final_gate_classification`

## Validation

```powershell
python -m services.phase_196z_planner_empty_safe_classification_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_196z_planner_empty_safe_classification_validation.ps1
python -m services.paper_entry_loop_stage --verbose --dry-run
```

## Preserved

- Phase 196X dispatch contract protection
- Phase 196Y approval sync (subset/exact/fresh)
- All risk, sector, cooldown, and duplicate-order gates
