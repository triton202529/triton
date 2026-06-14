/**
 * verify-gate-build.js
 * THP-002/THP-003 build verification — ensures TRITON execution gate exports, rules, and CLI bridge work.
 */

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  evaluateExecutionGate,
  assertExecutionAllowed,
  getExecutionPathStatus,
  explainExecutionDecision,
  listAllExecutionPathStatuses,
  EXECUTION_PATHS,
} from '../lib/executionGate.js';

const REQUIRED_PATHS = Object.values(EXECUTION_PATHS);
const checks = [];

function assert(name, condition, detail = '') {
  checks.push({ name, ok: Boolean(condition), detail });
}

assert('evaluateExecutionGate is function', typeof evaluateExecutionGate === 'function');
assert('assertExecutionAllowed is function', typeof assertExecutionAllowed === 'function');
assert('getExecutionPathStatus is function', typeof getExecutionPathStatus === 'function');
assert('explainExecutionDecision is function', typeof explainExecutionDecision === 'function');

const statuses = listAllExecutionPathStatuses();
assert('all execution paths registered', statuses.length === REQUIRED_PATHS.length);
assert(
  'every path is known',
  statuses.every((s) => s.known),
);

const liveBlocked = evaluateExecutionGate(EXECUTION_PATHS.LIVE_EXECUTION, {
  environment: 'production',
  liveMode: true,
  liveModeAuthorized: false,
  governanceApproved: false,
});
assert('live execution blocked by default', !liveBlocked.allowed && liveBlocked.decision === 'BLOCK');

const liveAuthorized = evaluateExecutionGate(EXECUTION_PATHS.LIVE_EXECUTION, {
  environment: 'development',
  liveModeAuthorized: true,
  governanceApproved: true,
  lifecycleApproved: true,
  allocationApproved: true,
  riskConstraintsSatisfied: true,
});
assert('live execution review when authorized', !liveAuthorized.allowed && liveAuthorized.decision === 'REVIEW_REQUIRED');

const paperAllowed = evaluateExecutionGate(EXECUTION_PATHS.PAPER_EXECUTION, {
  environment: 'development',
  paperMode: true,
  lifecycleApproved: true,
  riskConstraintsSatisfied: true,
});
assert('paper execution allowed', paperAllowed.allowed && paperAllowed.decision === 'ALLOW');

const paperLiveSeparation = evaluateExecutionGate(EXECUTION_PATHS.PAPER_EXECUTION, {
  environment: 'development',
  liveMode: true,
  paperMode: false,
});
assert('paper/live separation enforced', !paperLiveSeparation.allowed);

const brokerBlocked = evaluateExecutionGate(EXECUTION_PATHS.BROKER_SUBMISSION, {
  environment: 'development',
  lifecycleApproved: false,
  brokerConnected: false,
});
assert('broker submission blocked without auth', !brokerBlocked.allowed);

const brokerAllowed = evaluateExecutionGate(EXECUTION_PATHS.BROKER_SUBMISSION, {
  environment: 'development',
  lifecycleApproved: true,
  executionAuthorized: true,
  brokerConnected: true,
  executionGuardApproved: true,
  duplicateOrderClear: true,
  riskConstraintsSatisfied: true,
});
assert('broker submission allowed with guards', brokerAllowed.allowed);

const capitalBlocked = evaluateExecutionGate(EXECUTION_PATHS.CAPITAL_DEPLOYMENT, {
  environment: 'development',
  lifecycleApproved: false,
  allocationApproved: false,
});
assert('capital deployment requires lifecycle and allocation', !capitalBlocked.allowed);

const emergencyAllowed = evaluateExecutionGate(EXECUTION_PATHS.RISK_EMERGENCY_ACTION, {
  riskEmergencyTriggered: true,
  emergencyAction: 'FORCE_EXIT',
});
assert('emergency risk action allowed', emergencyAllowed.allowed && emergencyAllowed.decision === 'ALLOW');

const emergencyBlocked = evaluateExecutionGate(EXECUTION_PATHS.RISK_EMERGENCY_ACTION, {
  riskEmergencyTriggered: false,
});
assert('emergency blocked without trigger', !emergencyBlocked.allowed);

const governanceReview = evaluateExecutionGate(EXECUTION_PATHS.GOVERNANCE_OVERRIDE, {
  operatorAuthorized: false,
});
assert('governance override requires operator', !governanceReview.allowed && governanceReview.decision === 'REVIEW_REQUIRED');

let threw = false;
try {
  assertExecutionAllowed(EXECUTION_PATHS.LIVE_EXECUTION, { liveModeAuthorized: false });
} catch {
  threw = true;
}
assert('assertExecutionAllowed throws on block', threw);

const explanation = explainExecutionDecision(liveBlocked);
assert('explainExecutionDecision returns string', typeof explanation === 'string' && explanation.length > 0);

const pathStatus = getExecutionPathStatus(EXECUTION_PATHS.SIGNAL_GENERATION);
assert('getExecutionPathStatus returns path metadata', pathStatus.path === EXECUTION_PATHS.SIGNAL_GENERATION);

const cliScript = fileURLToPath(new URL('evaluate-execution-gate.mjs', import.meta.url));
const cliPayload = JSON.stringify({
  path: EXECUTION_PATHS.PAPER_EXECUTION,
  context: { paperMode: true, lifecycleApproved: true, riskConstraintsSatisfied: true },
});
const cli = spawnSync('node', [cliScript, cliPayload], { encoding: 'utf8' });
assert('THP-003 CLI bridge runs', cli.status === 0, cli.stderr || cli.stdout);
if (cli.status === 0) {
  const cliResult = JSON.parse(cli.stdout);
  assert('THP-003 CLI returns evaluation', Boolean(cliResult.evaluation?.path));
}

const failed = checks.filter((c) => !c.ok);

if (failed.length > 0) {
  console.error('Execution gate build verification FAILED:');
  for (const check of failed) {
    console.error(`  ✗ ${check.name}${check.detail ? `: ${check.detail}` : ''}`);
  }
  process.exit(1);
}

console.log(`Execution gate build verification passed (${checks.length} checks).`);
