# THP-001 — Execution Gate Unification & Money Path Protection

## Overview

Tropicash now routes every money-moving operation through a centralized **Execution Gate** before any balance mutation, payout, or wallet API action may proceed. This phase adds governance and protection only — it does not change Supabase schema, balance mutation logic, PayPal credentials, OAuth token implementation, KYC policy tables, or production environment settings.

## Modules

| Module | Purpose |
|--------|---------|
| `lib/moneyPathProtection.js` | Path registry, default policies, entry-point classification |
| `lib/executionGate.js` | Gate evaluation, assertions, and decision explanations |

## Protected Money Path Categories

All paths are registered in `MONEY_PATH_POLICIES`:

| Path | Default Decision | Production | Notes |
|------|------------------|------------|-------|
| `wallet_funding` | BLOCK | Blocked | Requires KYC when enabled |
| `wallet_send` | BLOCK | Blocked | Requires KYC when enabled |
| `wallet_withdrawal` | BLOCK | Blocked | KYC + withdrawal limits enforced |
| `paypal_payout` | BLOCK | Blocked | PayPal payout execution |
| `merchant_payment` | BLOCK | Blocked | Merchant checkout |
| `tap_to_pay` | DISABLED | Disabled | Until network + compliance certs |
| `developer_wallet_api` | SANDBOX_ONLY | Blocked | Testing environments only |
| `oauth_wallet_access` | ALLOW (read) | Read scopes | Balance mutation forbidden |
| `admin_manual_release` | REVIEW_REQUIRED | Blocked | Admin auth + audit trail |

## Gate Response Structure

Every `evaluateExecutionGate()` call returns:

```json
{
  "allowed": true,
  "decision": "ALLOW",
  "reason": "",
  "environment": "development",
  "user_id": "",
  "path": "wallet_send",
  "risk_level": "high",
  "requires_kyc": true,
  "requires_admin": false,
  "timestamp": "2026-06-13T21:00:00.000Z"
}
```

### Decision Types

- **ALLOW** — Operation may proceed.
- **BLOCK** — Operation must not execute.
- **REVIEW_REQUIRED** — Queued for manual or admin review; not auto-executed.
- **SANDBOX_ONLY** — Permitted in sandbox/dev; blocked in production.
- **DISABLED** — Feature permanently off (tap-to-pay).

## Required Functions

```javascript
import {
  evaluateExecutionGate,
  assertMoneyPathAllowed,
  getMoneyPathStatus,
  explainGateDecision,
  MONEY_PATHS,
} from '../lib/executionGate.js';

// Evaluate before any money operation
const result = evaluateExecutionGate(MONEY_PATHS.WALLET_SEND, {
  userId: 'user_123',
  kycVerified: true,
});

if (!result.allowed) {
  return res.status(403).json(result);
}

// Or throw on blocked paths
assertMoneyPathAllowed(MONEY_PATHS.WALLET_WITHDRAWAL, {
  userId: 'user_123',
  kycVerified: true,
  withdrawalAmount: 100,
  withdrawalLimitRemaining: 500,
});

// Static path metadata
const status = getMoneyPathStatus(MONEY_PATHS.TAP_TO_PAY);

// Human-readable explanation
console.log(explainGateDecision(result));
```

## Protection Rules

### 1. Production Money Movement

All production wallet mutations default to **BLOCK**. Future production readiness controls may set `productionReadinessOverride: true` in gate context — not wired in this phase.

### 2. Tap-to-Pay

Always **DISABLED** until:

- Payment network integration
- Security certification
- Device compatibility validation
- Compliance approval

### 3. Developer Wallet APIs

**SANDBOX_ONLY** in `sandbox` / `development` environments. Production wallet mutations are always blocked.

### 4. Wallet Withdrawals

Gate enforces:

- `kycVerified` must be true
- `withdrawalAmount` must not exceed `withdrawalLimitRemaining`

Existing compliance policies remain authoritative; the gate reads context flags only.

### 5. Admin Manual Release

Requires:

- `isAdmin` and `adminAuthorized`
- `operationalContext` for audit trail
- Returns `REVIEW_REQUIRED` even when admin-authorized (human approval step)

### 6. OAuth Wallet Access

- Allowed: `wallet:read`, `wallet:balance:read`, `wallet:transactions:read`
- Forbidden: `wallet:write`, `wallet:transfer`, `wallet:withdraw`
- Direct balance mutation via OAuth is always blocked

## Integration Points

Wire `assertMoneyPathAllowed()` or `gateApiRequest()` at these surfaces before existing logic runs:

| Surface | Money Path |
|---------|------------|
| `pages/fund*` | `wallet_funding` |
| `pages/send*` | `wallet_send` |
| `pages/withdraw*` | `wallet_withdrawal` |
| `pages/dev-console*` | `developer_wallet_api` |
| `pages/api/wallet/fund` | `wallet_funding` |
| `pages/api/wallet/send` | `wallet_send` |
| `pages/api/wallet/withdraw` | `wallet_withdrawal` |
| `pages/api/paypal/payout` | `paypal_payout` |
| `pages/api/merchant/pay` | `merchant_payment` |
| `pages/api/tap-to-pay` | `tap_to_pay` |
| `pages/api/oauth/wallet` | `oauth_wallet_access` |
| `pages/api/admin/release` | `admin_manual_release` |

**Important:** Do not bypass the gate in API routes, server actions, or client-side submit handlers.

## Environment Resolution

The gate reads (in order):

1. `TROPICASH_ENV`
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

Build verifies all gate functions export correctly and runs representative evaluations.

## Out of Scope (This Phase)

- Supabase schema changes
- Balance mutation logic changes
- PayPal credential handling
- OAuth token implementation
- KYC policy table changes
- Production environment configuration changes
