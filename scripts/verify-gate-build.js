/**
 * verify-gate-build.js
 * THP-001 build verification — ensures execution gate exports and rules work.
 */

import {
  evaluateExecutionGate,
  assertMoneyPathAllowed,
  getMoneyPathStatus,
  explainGateDecision,
  listAllMoneyPathStatuses,
  MONEY_PATHS,
} from '../lib/executionGate.js';

const REQUIRED_PATHS = Object.values(MONEY_PATHS);
const checks = [];

function assert(name, condition, detail = '') {
  checks.push({ name, ok: Boolean(condition), detail });
}

// Export surface
assert('evaluateExecutionGate is function', typeof evaluateExecutionGate === 'function');
assert('assertMoneyPathAllowed is function', typeof assertMoneyPathAllowed === 'function');
assert('getMoneyPathStatus is function', typeof getMoneyPathStatus === 'function');
assert('explainGateDecision is function', typeof explainGateDecision === 'function');

// All paths classified
const statuses = listAllMoneyPathStatuses();
assert('all money paths registered', statuses.length === REQUIRED_PATHS.length);
assert(
  'every path is known',
  statuses.every((s) => s.known),
);

// Production money movement blocked
const prodSend = evaluateExecutionGate(MONEY_PATHS.WALLET_SEND, {
  environment: 'production',
  kycVerified: true,
});
assert('production send blocked', !prodSend.allowed && prodSend.decision === 'BLOCK');

// Tap-to-pay disabled
const tap = evaluateExecutionGate(MONEY_PATHS.TAP_TO_PAY, { environment: 'sandbox' });
assert('tap-to-pay disabled', !tap.allowed && tap.decision === 'DISABLED');

// Developer sandbox preserved
const devApi = evaluateExecutionGate(MONEY_PATHS.DEVELOPER_WALLET_API, {
  environment: 'sandbox',
});
assert('developer sandbox allowed', devApi.allowed && devApi.decision === 'SANDBOX_ONLY');

// Developer production blocked
const devProd = evaluateExecutionGate(MONEY_PATHS.DEVELOPER_WALLET_API, {
  environment: 'production',
});
assert('developer production blocked', !devProd.allowed);

// OAuth read allowed
const oauthRead = evaluateExecutionGate(MONEY_PATHS.OAUTH_WALLET_ACCESS, {
  oauthScope: 'wallet:read',
});
assert('oauth read allowed', oauthRead.allowed && oauthRead.decision === 'ALLOW');

// OAuth mutation blocked
const oauthMut = evaluateExecutionGate(MONEY_PATHS.OAUTH_WALLET_ACCESS, {
  oauthMutationRequested: true,
});
assert('oauth mutation blocked', !oauthMut.allowed);

// Withdrawal KYC enforcement
const withdrawNoKyc = evaluateExecutionGate(MONEY_PATHS.WALLET_WITHDRAWAL, {
  environment: 'development',
  kycVerified: false,
});
assert('withdrawal requires kyc', !withdrawNoKyc.allowed);

// assertMoneyPathAllowed throws when blocked
let threw = false;
try {
  assertMoneyPathAllowed(MONEY_PATHS.TAP_TO_PAY);
} catch {
  threw = true;
}
assert('assertMoneyPathAllowed throws on block', threw);

// explainGateDecision produces text
const explanation = explainGateDecision(prodSend);
assert('explainGateDecision returns string', typeof explanation === 'string' && explanation.length > 0);

// getMoneyPathStatus returns metadata
const pathStatus = getMoneyPathStatus(MONEY_PATHS.WALLET_FUNDING);
assert('getMoneyPathStatus returns path metadata', pathStatus.path === MONEY_PATHS.WALLET_FUNDING);

const failed = checks.filter((c) => !c.ok);

if (failed.length > 0) {
  console.error('Execution gate build verification FAILED:');
  for (const check of failed) {
    console.error(`  ✗ ${check.name}${check.detail ? `: ${check.detail}` : ''}`);
  }
  process.exit(1);
}

console.log(`Execution gate build verification passed (${checks.length} checks).`);
