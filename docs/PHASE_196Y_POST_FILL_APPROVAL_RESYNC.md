# Phase 196Y — Post-Fill Approval Resync / Idempotent Paper Approval

## Problem

After Phase 196X controlled paper execution, TSLA and XLK filled. The next paper-entry cycle selected only BITO, but the gate blocked with `FINAL_SELECTION_NOT_APPROVED` even though an existing approval covered BITO, TSLA, and XLK.

Root cause: `ensure_governance_approval_for_symbols` required **exact** symbol-set match. `find_approved_request` already supports superset lookup, but the final gate rejected valid subset coverage.

## Repair

### Subset approval coverage

`auto_paper_approval.py` now accepts:

| Mode | Condition |
|------|-----------|
| `EXACT` | approved symbols == final selection |
| `SUBSET` | final selection ⊆ approved symbols (post-fill shrink) |
| `FRESH_CREATED` | auto-approval created a new exact record |

### Diagnostics in audit

`paper_entry_loop_stage_audit.json` includes:

- `approval_match_mode`
- `approval_covers_final_selection`
- `approved_symbol_sets`
- `governance_approval_present_after_auto` (fixed propagation from auto-paper path)

## Validation (safe)

```powershell
python -m services.phase_196y_post_fill_approval_resync_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_196y_post_fill_approval_resync_validation.ps1
python -m services.paper_entry_loop_stage --verbose --dry-run
```

Do not restart continuous loop until validation PASS.

## Safety preserved

- Phase 196X dispatch contract unchanged
- Duplicate protection for held TSLA/XLK
- No live execution, no gate weakening
