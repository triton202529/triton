# Phase 197L — Real Approval Artifact Template / No-Execution Activation Gate

## Problem

197K proved the approval contract validator, but TRITON lacked a guarded operator workflow
to produce an approval template without auto-creating runtime authority at
`data/governance/operator_paper_defensive_exit_approval.json`.

## Fix

`services/operator_paper_defensive_exit_approval_builder.py`:

- **Default:** dry-run only → writes `data/results/operator_paper_defensive_exit_approval_template.json`
- **Template** includes `template_only=true` and `not_runtime_authority=true`
- **Real write** requires explicit `--write-real-approval` and `--approved-by`
- Refuses real write when preflight checks fail (live not blocked, open orders, expired packet, etc.)

## CLI

```powershell
# Dry-run template (default)
python -m services.operator_paper_defensive_exit_approval_builder --dry-run

# Optional subset and expiration
python -m services.operator_paper_defensive_exit_approval_builder --dry-run --symbols SLV,CVX --expires-minutes 60

# Real approval (operator decision only; not used in 197L validation)
python -m services.operator_paper_defensive_exit_approval_builder --write-real-approval --approved-by "operator_name"
```

## Safety guarantees

- Template is rejected by validator (`template_only_not_runtime_authority`)
- Simulated real contract (without template flag) validates as `OPERATOR_APPROVAL_VALID_PAPER_ONLY`
- Real runtime remains `OPERATOR_APPROVAL_REQUIRED` during validation
- No broker orders; no governance/readiness mutation; live remains blocked

## Validation

```powershell
python -m services.phase_197l_real_approval_template_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197l_real_approval_template_validation.ps1
python -m services.operator_paper_defensive_exit_approval_builder --dry-run
python -m services.phase_197k_approval_contract_dry_run_validation
python -m services.phase_197j_operator_paper_approval_packet_validation
python -m services.paper_entry_loop_stage --verbose --dry-run
```

## Preserved

Paper-only; live blocked; no auto-approval; no cap increase; no forced execution; all gates intact.
