# Phase 197M-FIX — Stale Live Confirm Detection / Live Block Regression Repair

## Problem

After Phase 197M, validators reported `live_blocked=False` when the paper entry gate failed
early (e.g. market closed) before populating `live_execution_permitted` in gate checks.
The default `checks.get("live_execution_permitted", True)` inverted live block status.

Stale April 2026 `live_confirm.json` files must never imply live authorization.

## Fix

`services/live_block_classifier.py`:

- Classifies each live confirm path: `LIVE_CONFIRM_ABSENT`, `EXPIRED`, `STALE`, `INVALID`, `VALID_CURRENT`
- `evaluate_live_block_status()` — conservative paper-mode live block (default blocked)
- `validator_paper_gate_acceptable()` — treats market closed and readiness blocks as safe

Updated validators 197I–197M use the centralized classifier instead of gate-check defaults.

## Expected current state

- `live_blocked=True`
- `live_block_reason=LIVE_CONFIRM_STALE_OR_EXPIRED` (or `PAPER_MODE_LIVE_BLOCKED`)
- `stale_live_confirm_count >= 1`
- `valid_live_confirm_count=0`
- Paper entry: `SAFE_BLOCK_MARKET_CLOSED` after hours (acceptable)

## Validation

```powershell
python -m services.phase_197m_fix_live_block_detection_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197m_fix_live_block_detection_validation.ps1
python -m services.phase_197m_real_approval_runtime_isolation_validation
python -m services.phase_197l_real_approval_template_validation
```

## Preserved

No live_confirm refresh; no file deletion; no broker orders; live remains blocked.
