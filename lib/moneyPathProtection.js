/**
 * moneyPathProtection.js
 * THP-002 — Registry and default policies for all TRITON execution paths.
 *
 * Single source of truth for execution path classification, risk metadata, and
 * baseline protection rules. Does not place orders or call broker APIs.
 */

/** @typedef {'ALLOW' | 'BLOCK' | 'REVIEW_REQUIRED' | 'SANDBOX_ONLY' | 'DISABLED'} GateDecisionType */

/** @typedef {'low' | 'medium' | 'high' | 'critical'} RiskLevel */

/** @typedef {'FORCE_EXIT' | 'FORCE_TRIM' | 'RISK_LOCKDOWN'} EmergencyActionType */

/**
 * Canonical TRITON execution path identifiers. Every capital deployment or
 * broker submission entry point must register under one of these categories.
 */
export const EXECUTION_PATHS = Object.freeze({
  SIGNAL_GENERATION: 'signal_generation',
  LIFECYCLE_AUTHORIZATION: 'lifecycle_authorization',
  PORTFOLIO_ALLOCATION: 'portfolio_allocation',
  POSITION_ENTRY: 'position_entry',
  POSITION_ADD: 'position_add',
  POSITION_TRIM: 'position_trim',
  POSITION_EXIT: 'position_exit',
  FORCED_ROTATION: 'forced_rotation',
  CAPITAL_DEPLOYMENT: 'capital_deployment',
  BROKER_SUBMISSION: 'broker_submission',
  PAPER_EXECUTION: 'paper_execution',
  LIVE_EXECUTION: 'live_execution',
  GOVERNANCE_OVERRIDE: 'governance_override',
  RISK_EMERGENCY_ACTION: 'risk_emergency_action',
  MANUAL_OPERATOR_OVERRIDE: 'manual_operator_override',
});

/** @type {ReadonlySet<string>} */
export const ALL_EXECUTION_PATHS = new Set(Object.values(EXECUTION_PATHS));

/**
 * Environments recognized by the execution gate.
 * Live capital deployment defaults to BLOCK unless explicitly authorized.
 */
export const ENVIRONMENTS = Object.freeze({
  PRODUCTION: 'production',
  STAGING: 'staging',
  SANDBOX: 'sandbox',
  DEVELOPMENT: 'development',
});

/** @type {ReadonlySet<string>} */
const NON_LIVE_ENVIRONMENTS = new Set([
  ENVIRONMENTS.SANDBOX,
  ENVIRONMENTS.DEVELOPMENT,
  ENVIRONMENTS.STAGING,
]);

/** @type {ReadonlySet<EmergencyActionType>} */
export const EMERGENCY_ACTIONS = new Set(['FORCE_EXIT', 'FORCE_TRIM', 'RISK_LOCKDOWN']);

/**
 * Default policy per execution path. Applied before request-specific context
 * overrides in evaluateExecutionGate().
 *
 * @type {Record<string, {
 *   defaultDecision: GateDecisionType,
 *   riskLevel: RiskLevel,
 *   description: string,
 *   requiresRiskValidation: boolean,
 *   requiresGovernance: boolean,
 *   requiresOperator: boolean,
 *   requiresLifecycleApproval: boolean,
 *   requiresAllocationApproval: boolean,
 *   liveBlocked: boolean,
 *   paperPreferred: boolean,
 *   permanentlyDisabled: boolean,
 *   deploysCapital: boolean,
 *   emergencyPermitted: boolean,
 * }>}
 */
export const EXECUTION_PATH_POLICIES = Object.freeze({
  [EXECUTION_PATHS.SIGNAL_GENERATION]: {
    defaultDecision: 'ALLOW',
    riskLevel: 'low',
    description: 'Generate or refresh trading signals without broker submission.',
    requiresRiskValidation: false,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: false,
    requiresAllocationApproval: false,
    liveBlocked: false,
    paperPreferred: false,
    permanentlyDisabled: false,
    deploysCapital: false,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.LIFECYCLE_AUTHORIZATION]: {
    defaultDecision: 'ALLOW',
    riskLevel: 'medium',
    description: 'Authorize signal lifecycle transitions (WATCH → DEPLOY, etc.).',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: false,
    requiresAllocationApproval: false,
    liveBlocked: false,
    paperPreferred: false,
    permanentlyDisabled: false,
    deploysCapital: false,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.PORTFOLIO_ALLOCATION]: {
    defaultDecision: 'REVIEW_REQUIRED',
    riskLevel: 'high',
    description: 'Compute or approve portfolio allocation targets.',
    requiresRiskValidation: true,
    requiresGovernance: true,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: false,
    liveBlocked: false,
    paperPreferred: false,
    permanentlyDisabled: false,
    deploysCapital: false,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.POSITION_ENTRY]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'high',
    description: 'Open a new position via broker order submission.',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: true,
    liveBlocked: true,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.POSITION_ADD]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'high',
    description: 'Add to an existing position (scale-in).',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: true,
    liveBlocked: true,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.POSITION_TRIM]: {
    defaultDecision: 'ALLOW',
    riskLevel: 'medium',
    description: 'Reduce position size without full exit.',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: false,
    liveBlocked: true,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: false,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.POSITION_EXIT]: {
    defaultDecision: 'ALLOW',
    riskLevel: 'medium',
    description: 'Close or materially reduce a position.',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: false,
    liveBlocked: true,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: false,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.FORCED_ROTATION]: {
    defaultDecision: 'REVIEW_REQUIRED',
    riskLevel: 'high',
    description: 'Forced sector or concentration rotation requiring review.',
    requiresRiskValidation: true,
    requiresGovernance: true,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: true,
    liveBlocked: true,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.CAPITAL_DEPLOYMENT]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'critical',
    description: 'Deploy capital across approved allocation targets.',
    requiresRiskValidation: true,
    requiresGovernance: true,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: true,
    liveBlocked: true,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.BROKER_SUBMISSION]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'critical',
    description: 'Submit orders to the broker after execution authorization.',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: false,
    liveBlocked: true,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.PAPER_EXECUTION]: {
    defaultDecision: 'ALLOW',
    riskLevel: 'medium',
    description: 'Execute orders against the paper broker environment.',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: false,
    liveBlocked: false,
    paperPreferred: true,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.LIVE_EXECUTION]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'critical',
    description: 'Execute orders against the live broker with real capital.',
    requiresRiskValidation: true,
    requiresGovernance: true,
    requiresOperator: false,
    requiresLifecycleApproval: true,
    requiresAllocationApproval: true,
    liveBlocked: true,
    paperPreferred: false,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.GOVERNANCE_OVERRIDE]: {
    defaultDecision: 'REVIEW_REQUIRED',
    riskLevel: 'critical',
    description: 'Governance override of normal execution authorization.',
    requiresRiskValidation: false,
    requiresGovernance: true,
    requiresOperator: true,
    requiresLifecycleApproval: false,
    requiresAllocationApproval: false,
    liveBlocked: true,
    paperPreferred: false,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
  [EXECUTION_PATHS.RISK_EMERGENCY_ACTION]: {
    defaultDecision: 'ALLOW',
    riskLevel: 'critical',
    description: 'Emergency risk intervention (FORCE_EXIT, FORCE_TRIM, RISK_LOCKDOWN).',
    requiresRiskValidation: true,
    requiresGovernance: false,
    requiresOperator: false,
    requiresLifecycleApproval: false,
    requiresAllocationApproval: false,
    liveBlocked: false,
    paperPreferred: false,
    permanentlyDisabled: false,
    deploysCapital: false,
    emergencyPermitted: true,
  },
  [EXECUTION_PATHS.MANUAL_OPERATOR_OVERRIDE]: {
    defaultDecision: 'REVIEW_REQUIRED',
    riskLevel: 'critical',
    description: 'Manual operator override of automated execution controls.',
    requiresRiskValidation: false,
    requiresGovernance: true,
    requiresOperator: true,
    requiresLifecycleApproval: false,
    requiresAllocationApproval: false,
    liveBlocked: true,
    paperPreferred: false,
    permanentlyDisabled: false,
    deploysCapital: true,
    emergencyPermitted: false,
  },
});

/**
 * Known TRITON integration entry points mapped to execution paths.
 *
 * @type {ReadonlyArray<{ id: string, path: string, surface: string, notes?: string }>}
 */
export const EXECUTION_PATH_ENTRY_POINTS = Object.freeze([
  { id: 'signal-gen', path: EXECUTION_PATHS.SIGNAL_GENERATION, surface: 'services/enhanced_signal_generator.py', notes: 'Signal generation' },
  { id: 'lifecycle', path: EXECUTION_PATHS.LIFECYCLE_AUTHORIZATION, surface: 'services/lifecycle_truth.py', notes: 'Lifecycle authorization' },
  { id: 'allocation', path: EXECUTION_PATHS.PORTFOLIO_ALLOCATION, surface: 'services/portfolio_allocation_engine.py', notes: 'Portfolio allocation' },
  { id: 'position-entry', path: EXECUTION_PATHS.POSITION_ENTRY, surface: 'services/execute_trades.py', notes: 'New position entry' },
  { id: 'position-add', path: EXECUTION_PATHS.POSITION_ADD, surface: 'services/manage_positions.py', notes: 'Position add/scale-in' },
  { id: 'position-trim', path: EXECUTION_PATHS.POSITION_TRIM, surface: 'services/manage_positions.py', notes: 'Position trim' },
  { id: 'position-exit', path: EXECUTION_PATHS.POSITION_EXIT, surface: 'services/manage_positions.py', notes: 'Position exit' },
  { id: 'forced-rotation', path: EXECUTION_PATHS.FORCED_ROTATION, surface: 'services/capital_reallocation.py', notes: 'Forced rotation' },
  { id: 'capital-deploy', path: EXECUTION_PATHS.CAPITAL_DEPLOYMENT, surface: 'services/capital_deployment_engine.py', notes: 'Capital deployment' },
  { id: 'broker-submit', path: EXECUTION_PATHS.BROKER_SUBMISSION, surface: 'services/broker_alpaca.py', notes: 'Broker order submission' },
  { id: 'paper-exec', path: EXECUTION_PATHS.PAPER_EXECUTION, surface: 'services/run_trade_cycle.py', notes: 'Paper execution cycle' },
  { id: 'live-exec', path: EXECUTION_PATHS.LIVE_EXECUTION, surface: 'place_orders_from_csv.py', notes: 'Live execution entry' },
  { id: 'governance-override', path: EXECUTION_PATHS.GOVERNANCE_OVERRIDE, surface: 'services/governance_runtime_injection_engine.py', notes: 'Governance override' },
  { id: 'risk-emergency', path: EXECUTION_PATHS.RISK_EMERGENCY_ACTION, surface: 'services/pretrade_guard.py', notes: 'Emergency risk action' },
  { id: 'operator-override', path: EXECUTION_PATHS.MANUAL_OPERATOR_OVERRIDE, surface: 'ui/pages/Order_Desk.py', notes: 'Manual operator override' },
]);

/**
 * Resolve the active TRITON environment from process env vars.
 *
 * @returns {string}
 */
export function resolveEnvironment() {
  const raw =
    process.env.TRITON_ENV ||
    process.env.VERCEL_ENV ||
    process.env.NODE_ENV ||
    'development';

  const normalized = String(raw).trim().toLowerCase();

  if (normalized === 'prod' || normalized === 'production') {
    return ENVIRONMENTS.PRODUCTION;
  }
  if (normalized === 'stage' || normalized === 'staging' || normalized === 'preview') {
    return ENVIRONMENTS.STAGING;
  }
  if (normalized === 'sandbox' || normalized === 'test') {
    return ENVIRONMENTS.SANDBOX;
  }
  return ENVIRONMENTS.DEVELOPMENT;
}

/**
 * @param {string} path
 * @returns {boolean}
 */
export function isKnownExecutionPath(path) {
  return ALL_EXECUTION_PATHS.has(path);
}

/**
 * @param {string} path
 * @returns {typeof EXECUTION_PATH_POLICIES[string] | null}
 */
export function getExecutionPathPolicy(path) {
  if (!isKnownExecutionPath(path)) {
    return null;
  }
  return EXECUTION_PATH_POLICIES[path];
}

/**
 * @param {string} path
 * @returns {ReadonlyArray<{ id: string, path: string, surface: string, notes?: string }>}
 */
export function getEntryPointsForPath(path) {
  return EXECUTION_PATH_ENTRY_POINTS.filter((entry) => entry.path === path);
}

/**
 * @param {string} environment
 * @returns {boolean}
 */
export function isNonLiveEnvironment(environment) {
  return NON_LIVE_ENVIRONMENTS.has(environment);
}

/**
 * @param {string} environment
 * @returns {boolean}
 */
export function isProductionEnvironment(environment) {
  return environment === ENVIRONMENTS.PRODUCTION;
}

/**
 * @param {string} path
 * @returns {{
 *   path: string,
 *   known: boolean,
 *   defaultDecision: GateDecisionType | null,
 *   riskLevel: RiskLevel | null,
 *   requiresRiskValidation: boolean,
 *   requiresGovernance: boolean,
 *   requiresOperator: boolean,
 *   liveBlocked: boolean,
 *   paperPreferred: boolean,
 *   permanentlyDisabled: boolean,
 *   deploysCapital: boolean,
 *   entryPoints: ReadonlyArray<{ id: string, path: string, surface: string, notes?: string }>,
 *   description: string | null,
 * }}
 */
export function getExecutionPathStatus(path) {
  const policy = getExecutionPathPolicy(path);
  if (!policy) {
    return {
      path,
      known: false,
      defaultDecision: null,
      riskLevel: null,
      requiresRiskValidation: true,
      requiresGovernance: false,
      requiresOperator: false,
      liveBlocked: true,
      paperPreferred: false,
      permanentlyDisabled: false,
      deploysCapital: false,
      entryPoints: [],
      description: null,
    };
  }

  return {
    path,
    known: true,
    defaultDecision: policy.defaultDecision,
    riskLevel: policy.riskLevel,
    requiresRiskValidation: policy.requiresRiskValidation,
    requiresGovernance: policy.requiresGovernance,
    requiresOperator: policy.requiresOperator,
    liveBlocked: policy.liveBlocked,
    paperPreferred: policy.paperPreferred,
    permanentlyDisabled: policy.permanentlyDisabled,
    deploysCapital: policy.deploysCapital,
    entryPoints: getEntryPointsForPath(path),
    description: policy.description,
  };
}

/**
 * @returns {Array<ReturnType<typeof getExecutionPathStatus>>}
 */
export function listAllExecutionPathStatuses() {
  return [...ALL_EXECUTION_PATHS].map((path) => getExecutionPathStatus(path));
}
