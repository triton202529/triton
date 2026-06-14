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
