# THP-002 — TRITON Execution Gate Authorization Domains

## Overview

TRITON routes every execution-sensitive operation through a centralized **Execution Gate** before any order placement, portfolio mutation, trade authorization, or capital deployment action may proceed.

This phase converts the THP-001 gate from wallet/payment domains into TRITON-native trade authorization and capital deployment protection. Governance only — no broker API, allocation logic, lifecycle model, or schema changes.

## Protected Execution Chain

```
Signal
  ↓
Lifecycle Decision
  ↓
Portfolio Allocation
  ↓
Risk Validation
  ↓
Execution Gate
  ↓
Broker Authorization
  ↓
Paper / Live Execution
  ↓
Position Management
  ↓
Governance Audit
```

No capital can be deployed without passing a centralized authorization checkpoint.

## Modules

| Module | Purpose |
|--------|---------|
| `lib/moneyPathProtection.js` | TRITON execution path registry and default policies |
| `lib/executionGate.js` | Gate evaluation, assertions, and decision explanations |

## TRITON Execution Paths

All paths are registered in `EXECUTION_PATH_POLICIES`:

| Path | Default Decision | Live Blocked | Notes |
|------|------------------|--------------|-------|
| `signal_generation` | ALLOW | No | Signal generation without broker submission |
| `lifecycle_authorization` | ALLOW | No | Lifecycle transition authorization |
| `portfolio_allocation` | REVIEW_REQUIRED | No | Allocation target approval |
| `position_entry` | BLOCK | Yes | New position entry |
| `position_add` | BLOCK | Yes | Scale-in to existing position |
| `position_trim` | ALLOW | Yes | Partial position reduction |
| `position_exit` | ALLOW | Yes | Full or material exit |
| `forced_rotation` | REVIEW_REQUIRED | Yes | Forced concentration rotation |
| `capital_deployment` | BLOCK | Yes | Capital deployment across targets |
| `broker_submission` | BLOCK | Yes | Order submission to broker |
| `paper_execution` | ALLOW | No | Paper broker environment |
| `live_execution` | BLOCK | Yes | Live broker with real capital |
| `governance_override` | REVIEW_REQUIRED | Yes | Governance override of controls |
| `risk_emergency_action` | ALLOW | No | FORCE_EXIT / FORCE_TRIM / RISK_LOCKDOWN |
| `manual_operator_override` | REVIEW_REQUIRED | Yes | Manual operator override |

## Gate Response Structure

Every `evaluateExecutionGate()` call returns:

```json
{
  "allowed": true,
  "decision": "ALLOW",
  "reason": "",
  "environment": "development",
  "user_id": "",
  "path": "paper_execution",
  "risk_level": "medium",
  "requires_risk_validation": true,
  "requires_governance": false,
  "requires_operator": false,
  "timestamp": "2026-06-13T21:00:00.000Z"
}
```

### Decision Types

- **ALLOW** — Operation may proceed.
- **BLOCK** — Operation must not execute.
- **REVIEW_REQUIRED** — Queued for governance review; not auto-executed.
- **SANDBOX_ONLY** — Permitted in non-live environments only.
- **DISABLED** — Feature permanently off.

## Required Functions

```javascript
import {
  evaluateExecutionGate,
  assertExecutionAllowed,
  getExecutionPathStatus,
  explainExecutionDecision,
  EXECUTION_PATHS,
} from '../lib/executionGate.js';

// Evaluate before any execution operation
const result = evaluateExecutionGate(EXECUTION_PATHS.BROKER_SUBMISSION, {
  operatorId: 'op_123',
  lifecycleApproved: true,
  executionAuthorized: true,
  brokerConnected: true,
  executionGuardApproved: true,
  duplicateOrderClear: true,
});

if (!result.allowed) {
  return res.status(403).json(result);
}

// Or throw on blocked paths
assertExecutionAllowed(EXECUTION_PATHS.PAPER_EXECUTION, {
  paperMode: true,
  lifecycleApproved: true,
  riskConstraintsSatisfied: true,
});

// Static path metadata
const status = getExecutionPathStatus(EXECUTION_PATHS.LIVE_EXECUTION);

// Human-readable explanation
console.log(explainExecutionDecision(result));
```

## Protection Rules

### 1. Live Trading Protection

- Default: **BLOCK**
- Live capital deployment disabled by default
- Requires `liveModeAuthorized` and `governanceApproved`
- Production requires explicit `productionReadinessOverride`

### 2. Paper Trading Protection

- Default: **ALLOW** when `paperMode` is set
- Paper and live broker environments must remain separated
- Risk constraints must be satisfied when validation is required

### 3. Broker Submission Protection

Requires:

- `executionAuthorized` or `lifecycleApproved`
- `brokerConnected`
- `executionGuardApproved`
- `duplicateOrderClear`

### 4. Capital Deployment Protection

Requires:

- `lifecycleApproved`
- `allocationApproved` (when policy requires)
- `riskConstraintsSatisfied`

Applies to: `capital_deployment`, `position_entry`, `position_add`, and related deploy paths.

### 5. Governance Override Protection

Default: **REVIEW_REQUIRED**

Requires:

- `operatorAuthorized`
- `overrideReason`
- `operationalContext`
- Full audit traceability via `auditTrailId`

Applies to: `governance_override`, `manual_operator_override`

### 6. Emergency Risk Actions

Permitted when `riskEmergencyTriggered` is true and `emergencyAction` is one of:

- `FORCE_EXIT`
- `FORCE_TRIM`
- `RISK_LOCKDOWN`

## Integration Points

Wire `assertExecutionAllowed()` or `gateApiRequest()` at these surfaces before existing logic runs:

| Surface | Execution Path |
|---------|----------------|
| `services/enhanced_signal_generator.py` | `signal_generation` |
| `services/lifecycle_truth.py` | `lifecycle_authorization` |
| `services/portfolio_allocation_engine.py` | `portfolio_allocation` |
| `services/execute_trades.py` | `position_entry` |
| `services/manage_positions.py` | `position_add`, `position_trim`, `position_exit` |
| `services/capital_reallocation.py` | `forced_rotation` |
| `services/capital_deployment_engine.py` | `capital_deployment` |
| `services/broker_alpaca.py` | `broker_submission` |
| `services/run_trade_cycle.py` | `paper_execution` |
| `place_orders_from_csv.py` | `live_execution` |
| `services/governance_runtime_injection_engine.py` | `governance_override` |
| `services/pretrade_guard.py` | `risk_emergency_action` |
| `ui/pages/Order_Desk.py` | `manual_operator_override` |

## Environment Resolution

The gate reads (in order):

1. `TRITON_ENV`
2. `VERCEL_ENV`
3. `NODE_ENV`

Defaults to `development` when unset.

## Admin Dashboard

Optional status page: `pages/admin/execution-gates.jsx`

## THP-003 Pipeline Integration (Enforced)

The execution gate is actively enforced in the TRITON pipeline via `services/triton_execution_gate.py`, which delegates to `scripts/evaluate-execution-gate.mjs` (single source of truth in `lib/executionGate.js`).

### Enforcement Points

| Layer | File | Function | Gate Action |
|-------|------|----------|-------------|
| Entry planner | `services/execute_trades.py` | `maybe_execute_plan` | `capital_deployment` + paper/live |
| Management planner | `services/manage_positions.py` | `maybe_execute_management` | position exit/rotation/emergency + paper/live |
| Order placement | `services/place_live_orders.py` | `main` | `broker_submission` + paper/live |
| CSV executor | `place_orders_from_csv.py` | `main`, `place_order` | pipeline + per-order broker gate |
| Broker choke point | `services/broker_alpaca.py` | `submit_order`, `submit_order_notional` | position path + `broker_submission` |

### Audit Trail

Every gate evaluation appends to `data/results/execution_gate_audit.csv` with:

- `execution_path`, `decision`, `allowed`, `reason`, `environment`
- `operator_source`, `broker_mode`, `symbols`, `execution_identifier`, `timestamp`

### Authorization Environment Variables

| Variable | Purpose |
|----------|---------|
| `TRITON_ENABLE_PAPER_EXECUTION` | Required for paper execution authorization |
| `TRITON_ENABLE_LIVE_EXECUTION` | Explicit live execution authorization |
| `TRITON_LIVE_GOVERNANCE_APPROVED` | Governance approval for live capital deployment |
| `TRITON_DISABLE_EXECUTION_GATE` | Emergency bypass (audit only; not for production) |

### Python Interface

```python
from services.triton_execution_gate import (
    evaluate_execution_gate,
    assert_execution_allowed,
    enforce_broker_submission_gate,
    enforce_pipeline_gate,
    ExecutionGateBlocked,
)
```

## Validation

```bash
npm run lint
npm run build
```

## Out of Scope (This Phase)

- Broker API changes
- Order placement algorithm changes
- Portfolio allocation logic changes
- Lifecycle decision model changes
- Risk scoring model changes
- Governance system changes
- Database schema changes
- Production environment configuration changes

## THP-004 Governance Evidence & Decision Registry

Every `evaluate_execution_gate()` and `assert_execution_allowed()` call produces:

1. Operational row in `data/results/execution_gate_audit.csv` (preserved)
2. Institutional decision in `data/governance/execution_gate_decisions.json`
3. Linked evidence in `data/governance/execution_gate_evidence.json`
4. CSV mirrors and `execution_gate_authority_matrix.json`

### Decision Record Fields

`decision_id`, `timestamp`, `execution_identifier`, `execution_path`, `symbol(s)`,
`action_requested`, `quantity`, `capital_exposure`, `environment`, `paper_or_live`,
`requesting_service`, `operator_identity`, `execution_gate_decision`, `decision_reason`,
`risk_state`, `lifecycle_state`, `allocation_state`, `governance_status`,
`authorization_chain`, `evidence_links`, `authority_classification`, `allowed`

### Authorization Chain

Signal → Lifecycle → Allocation → Risk Protection → Execution Gate →
Broker Authorization → Execution Outcome

### Dashboard

`dashboard/execution_gate_governance_dashboard.py` — read-only Streamlit governance view.

### Registry Module

`services/execution_gate_evidence_registry.py` — deduplicated decision and evidence persistence.

## THP-005 Simulation, Replay & Failure Investigation

Read-only validation laboratory — no broker calls or capital deployment.

| Module | Purpose |
|--------|---------|
| `services/execution_gate_replay_engine.py` | Replay `execution_gate_decisions.json` and detect drift |
| `services/execution_gate_simulator.py` | Scenario catalog via `evaluate_execution_gate(record_evidence=False)` |
| `services/execution_gate_failure_investigator.py` | Human-readable failure analysis |
| `dashboard/execution_gate_simulation_dashboard.py` | Streamlit lab UI |

### Artifacts

- `execution_gate_replay_results.json` / `.csv`
- `execution_gate_simulation_results.json` / `.csv`
- `execution_gate_failure_analysis.json` / `.csv`
- `execution_gate_drift_report.json`

### Replay Classifications

- **CONSISTENT** — Current gate reproduces original outcome
- **DRIFT_DETECTED** — Authorization behavior changed under current conditions
- **REQUIRES_REVIEW** — Historical replay mismatch or ambiguous outcome

## THP-006 Stress Testing, Chaos Engineering & Resilience Certification

Read-only resilience validation — no broker calls, order placement, or portfolio mutations.

| Module | Purpose |
|--------|---------|
| `services/execution_gate_stress_suite.py` | Broker, governance, risk, environment, concurrency stress catalog |
| `services/execution_gate_chaos_engine.py` | Controlled fault injection via gate context |
| `services/execution_gate_resilience_certification.py` | Aggregate scores and institutional certification |
| `dashboard/execution_gate_resilience_dashboard.py` | Streamlit resilience lab UI |

All tests use `evaluate_execution_gate(record_evidence=False)` to avoid governance pollution.

### Artifacts

- `execution_gate_stress_results.json` / `.csv`
- `execution_gate_chaos_results.json` / `.csv`
- `execution_gate_resilience_report.json` / `.csv`
- `execution_gate_certification.json`

### Resilience Categories

Broker Safety, Governance Integrity, Risk Protection, Environment Isolation, Authorization Determinism, Audit Reliability, Failure Recovery.

### Certification Levels

| Score | Level |
|-------|-------|
| 95–100 | INSTITUTIONAL_GRADE |
| 85–94 | PRODUCTION_READY |
| 70–84 | REQUIRES_IMPROVEMENT |
| &lt;70 | FAILED |

### Run Certification

```bash
python -m services.execution_gate_resilience_certification
```

## THP-007 Continuous Monitoring, Health Scoring & Degradation Detection

Read-only institutional supervision — monitors gate health during real operation without
modifying authorization logic or trading behavior.

| Module | Purpose |
|--------|---------|
| `services/execution_gate_health_monitor.py` | Continuous 0–100 health score across 10 components |
| `services/execution_gate_anomaly_detector.py` | Execution, governance, and operational anomaly detection |
| `services/execution_gate_degradation_manager.py` | Autonomous MONITOR/WARNING/RESTRICT/LOCKDOWN classification |
| `dashboard/execution_gate_health_dashboard.py` | Streamlit health supervision UI |

### Artifacts

- `execution_gate_health_report.json`
- `execution_gate_health_history.csv`
- `execution_gate_anomalies.json`
- `execution_gate_degradation_events.json`
- `execution_gate_operational_status.json`

### Health Classifications

| Score | Level |
|-------|-------|
| 95–100 | GREEN — Institutional Grade |
| 85–94 | YELLOW — Warning |
| 70–84 | ORANGE — Degraded |
| &lt;70 | RED — Critical |

### Degradation States

| State | Behavior |
|-------|----------|
| MONITOR | Normal supervision cadence |
| WARNING | Enhanced monitoring; stale evidence alerts |
| RESTRICT | New capital deployment restricted; drift or missing artifacts |
| LOCKDOWN | Capital deployment blocked; emergency risk actions preserved |

### Run Supervision

```bash
python -m services.execution_gate_degradation_manager
```

## THP-008 Governance Council, Escalation & Human Approval

Investment committee-grade human governance layer — wraps execution gate evaluation with
authority levels, approval workflows, and escalation without modifying core gate logic.

| Module | Purpose |
|--------|---------|
| `services/execution_governance_council.py` | Authority levels 0–3 and council status |
| `services/execution_escalation_engine.py` | Automatic escalation on execution/operational/risk conditions |
| `services/execution_approval_workflow.py` | Approval requests and `evaluate_governed_execution()` wrapper |
| `dashboard/execution_governance_council_dashboard.py` | Council UI with approval queue and escalation center |

### Authority Levels

| Level | Name | Authority |
|-------|------|-----------|
| 0 | FULL_AUTOMATION | Gate may authorize routine paper/lifecycle actions |
| 1 | OPERATOR_REVIEW_REQUIRED | Pause for operator review |
| 2 | GOVERNANCE_COUNCIL_APPROVAL | Named institutional approval required |
| 3 | EMERGENCY_AUTHORITY | Emergency risk actions during lockdown |

### Artifacts

- `execution_authority_levels.json`
- `execution_approval_requests.json`
- `execution_approval_history.csv`
- `execution_escalations.json`
- `execution_governance_council_status.json`

### Governed Evaluation

```python
from services.execution_approval_workflow import evaluate_governed_execution

evaluation = evaluate_governed_execution(path, context, record_evidence=True)
```

Live deployment and high-risk actions downgrade ALLOW → REVIEW_REQUIRED until approved.
LOCKDOWN blocks non-emergency capital deployment; emergency paths remain available.

## THP-009 Time Travel, Decision Reconstruction & Regulatory Audit

Forensic-grade execution reconstruction — read-only on historical evidence; new snapshot
and audit artifacts only.

| Module | Purpose |
|--------|---------|
| `services/execution_time_machine.py` | Immutable fingerprinted execution snapshots |
| `services/execution_decision_reconstructor.py` | Historical vs present-day decision comparison |
| `services/execution_regulatory_audit_pack.py` | Exportable regulatory audit packages |
| `dashboard/execution_time_travel_dashboard.py` | Time travel search, replay, and audit export UI |

### Artifacts

- `execution_snapshots.json` / `execution_snapshot_index.csv`
- `execution_reconstruction_reports.json` / `.csv`
- `execution_audit_packages.json` / `.csv`
- `execution_policy_drift_analysis.json`

### Comparison Classifications

SAME_DECISION, POLICY_CHANGED, RISK_CHANGED, GOVERNANCE_CHANGED, ENVIRONMENT_CHANGED

### Run Forensic Pipeline

```bash
python -m services.execution_regulatory_audit_pack
```

## THP-H001 Full System Integration & Source-of-Truth Audit

Investigation-only institutional audit — no architecture or trading changes.

| Module | Purpose |
|--------|---------|
| `services/system_integration_audit.py` | Dependency map, source-of-truth registry, integration risk matrix |

### Artifacts

- `system_integration_audit.json` / `.csv`
- `source_of_truth_registry.json`
- `dependency_map.json`
- `unused_components_report.json`
- `integration_risk_matrix.json`

### Run Audit

```bash
python -m services.system_integration_audit
```

## THP-H002 Broker Execution Gate Hardening

Seals CRITICAL bypass in `place_bracket_market` and enforces governed authorization on all broker order paths.

| Change | Purpose |
|--------|---------|
| `enforce_broker_submission_gate` → `evaluate_governed_execution` | Governance overlay on every broker submit |
| `_submit_order_payload` + `_post` final check | No POST `/v2/orders` without auth context |
| `place_bracket_market` → `submit_order` | Bracket orders use full gate chain |
| `submit_bracket_order_notional` | Fixes `position_protector` notional bypass |
| `services/broker_execution_audit.py` | Protection matrix and bypass inventory |

### Artifacts

- `broker_execution_protection_matrix.json` / `.csv`
- `execution_bypass_report.json`

### Run Audit

```bash
python -m services.broker_execution_audit
```

## THP-H003 Execution Authority Consolidation

Ensures a single institutional capital deployment chain. Legacy execution scripts are classified, protected, or retired.

| Module | Purpose |
|--------|---------|
| `services/execution_path_registry.py` | Authoritative execution path catalog |
| `services/execution_path_guard.py` | Safe exit for DEPRECATED / ARCHIVED paths |

### Official Capital Chain

```
run_execution_loop / run_scheduled_paper_cycle
  → execute_trades → place_live_orders / manage_positions
  → triton_execution_gate → broker_alpaca → Alpaca
```

Approved operator workflows: `place_orders_from_csv.py`, `pages/manual_order_desk.py`, `pages_order_desk.py` (gated broker only).

### Artifacts

- `execution_path_registry.json` / `.csv`
- `execution_path_cleanup_report.json`

### Run Registry

```bash
python -m services.execution_path_registry
python -m services.broker_execution_audit
```

## THP-H004 Governance Council Enforcement

Ensures every ACTIVE capital deployment path uses `evaluate_governed_execution()` — not execution gate alone.

| Change | Purpose |
|--------|---------|
| `enforce_pipeline_gate` → `evaluate_governed_execution` | Pipeline stages require Governance Council |
| `enforce_broker_submission_gate` | Already governed (THP-H002) |
| Level 3 emergency authority | FORCE_EXIT / FORCE_TRIM preserved with evidence |
| `governance_execution_compliance_audit.py` | Compliance inventory |

### Required authorization chain

```
evaluate_governed_execution → Gate → Council → Approval → Broker → Alpaca
```

### Artifacts

- `governance_execution_compliance.json` / `.csv`

### Run Audit

```bash
python -m services.governance_execution_compliance_audit
```

## THP-H005 Source of Truth Authority

Establishes institutional clarity: one decision, one owner, one artifact, many read-only consumers.

| Module | Purpose |
|--------|---------|
| `services/source_of_truth_enforcement.py` | Registry, conflict detection, dependency map |

### Principle

```
ONE DECISION → ONE OWNER → ONE ARTIFACT → MANY READ-ONLY CONSUMERS
```

### Official authorities

| Domain | Authoritative artifact |
|--------|---------------------|
| Lifecycle | `signal_lifecycle_effective.csv` |
| Allocation | `portfolio_allocation_recommendations.csv` |
| Risk | `master_execution_gate.json` |
| Execution gate | `execution_gate_decisions.json` |
| Governance | `execution_approval_requests.json` |
| Broker state | Alpaca API via `broker_alpaca.py` |

### Artifacts

- `source_of_truth_registry.json` / `.csv`
- `source_of_truth_conflicts.json`
- `artifact_dependency_map.json`
- `artifact_staleness_report.json`
- `artifact_remediation_plan.json`

### Run Audit

```bash
python -m services.source_of_truth_enforcement
```

## THP-H006 Lifecycle & Risk Authority Consolidation

Enforces named authorities from THP-H005 on execution consumers.

| Authority | Artifact | Module |
|-----------|----------|--------|
| Lifecycle | `signal_lifecycle_effective.csv` | `services/decision_authority.py` |
| Execution risk | `master_execution_gate.json` | `load_risk_execution_controls()` |

### Changes

- `place_live_orders`: default effective lifecycle; no raw fallback or runtime merge
- `execute_trades`: risk flags from master gate snapshot (not adaptive_risk_state)
- `manage_positions`: effective lifecycle only (no raw fallback)
- Readiness/diagnostics: validate effective lifecycle artifact

### Artifacts

- `lifecycle_risk_authority_compliance.json` / `.csv`

### Run Audit

```bash
python -m services.lifecycle_risk_authority_audit
python -m services.source_of_truth_enforcement
```
