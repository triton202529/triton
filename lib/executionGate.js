/**
 * executionGate.js
 * THP-002 — Centralized execution gate for all TRITON execution paths.
 *
 * Every signal authorization, capital deployment, broker submission, paper/live
 * execution, governance override, and emergency risk action must pass through
 * evaluateExecutionGate() before proceeding.
 *
 * Governance only — does not place orders, mutate portfolios, or call broker APIs.
 */

import {
  EMERGENCY_ACTIONS,
  getExecutionPathPolicy,
  isKnownExecutionPath,
  isNonLiveEnvironment,
  isProductionEnvironment,
  resolveEnvironment,
} from './moneyPathProtection.js';

export {
  getExecutionPathStatus,
  listAllExecutionPathStatuses,
  EXECUTION_PATHS,
  ALL_EXECUTION_PATHS,
  EMERGENCY_ACTIONS,
} from './moneyPathProtection.js';

/**
 * @typedef {import('./moneyPathProtection.js').GateDecisionType} GateDecisionType
 * @typedef {import('./moneyPathProtection.js').EmergencyActionType} EmergencyActionType
 */

/**
 * @typedef {{
 *   allowed: boolean,
 *   decision: GateDecisionType,
 *   reason: string,
 *   environment: string,
 *   user_id: string,
 *   path: string,
 *   risk_level: string,
 *   requires_risk_validation: boolean,
 *   requires_governance: boolean,
 *   requires_operator: boolean,
 *   timestamp: string,
 *   audit?: Record<string, unknown>,
 * }} GateEvaluationResult
 */

/**
 * @typedef {{
 *   userId?: string,
 *   operatorId?: string,
 *   environment?: string,
 *   paperMode?: boolean,
 *   liveMode?: boolean,
 *   liveModeAuthorized?: boolean,
 *   lifecycleApproved?: boolean,
 *   allocationApproved?: boolean,
 *   riskConstraintsSatisfied?: boolean,
 *   executionAuthorized?: boolean,
 *   executionGuardApproved?: boolean,
 *   brokerConnected?: boolean,
 *   duplicateOrderClear?: boolean,
 *   governanceApproved?: boolean,
 *   operatorAuthorized?: boolean,
 *   overrideReason?: string,
 *   operationalContext?: string,
 *   auditTrailId?: string,
 *   riskEmergencyTriggered?: boolean,
 *   emergencyAction?: EmergencyActionType,
 *   productionReadinessOverride?: boolean,
 * }} ExecutionGateContext
 */

export class ExecutionGateError extends Error {
  /**
   * @param {GateEvaluationResult} evaluation
   */
  constructor(evaluation) {
    super(evaluation.reason || `Execution blocked for path: ${evaluation.path}`);
    this.name = 'ExecutionGateError';
    this.evaluation = evaluation;
  }
}

/**
 * @returns {string}
 */
function utcTimestamp() {
  return new Date().toISOString();
}

/**
 * @param {ReturnType<typeof getExecutionPathPolicy>} policy
 * @param {object} base
 * @returns {object}
 */
function policyFields(policy, base) {
  return {
    ...base,
    requiresRiskValidation: policy.requiresRiskValidation,
    requiresGovernance: policy.requiresGovernance,
    requiresOperator: policy.requiresOperator,
  };
}

/**
 * @param {GateDecisionType} decision
 * @param {string} reason
 * @param {object} fields
 * @returns {GateEvaluationResult}
 */
function buildResult(decision, reason, fields) {
  const allowed = decision === 'ALLOW' || decision === 'SANDBOX_ONLY';
  return {
    allowed,
    decision,
    reason,
    environment: fields.environment,
    user_id: fields.userId || fields.operatorId || '',
    path: fields.path,
    risk_level: fields.riskLevel || 'medium',
    requires_risk_validation: Boolean(fields.requiresRiskValidation),
    requires_governance: Boolean(fields.requiresGovernance),
    requires_operator: Boolean(fields.requiresOperator),
    timestamp: utcTimestamp(),
    ...(fields.audit ? { audit: fields.audit } : {}),
  };
}

/**
 * @param {ReturnType<typeof getExecutionPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult}
 */
function evaluateLiveExecution(policy, context, base) {
  if (!context.liveModeAuthorized || !context.governanceApproved) {
    return buildResult(
      'BLOCK',
      'Live execution blocked: production authorization and governance approval are required.',
      policyFields(policy, { ...base, requiresGovernance: true }),
    );
  }

  if (isProductionEnvironment(base.environment) && !context.productionReadinessOverride) {
    return buildResult(
      'BLOCK',
      'Live capital deployment remains disabled until explicit production authorization is granted.',
      policyFields(policy, base),
    );
  }

  const capitalCheck = evaluateCapitalDeploymentRequirements(policy, context, base);
  if (capitalCheck) {
    return capitalCheck;
  }

  return buildResult(
    'REVIEW_REQUIRED',
    'Live execution authorized pending final governance checkpoint.',
    policyFields(policy, {
      ...base,
      audit: {
        auditTrailId: context.auditTrailId || null,
        governanceApproved: true,
      },
    }),
  );
}

/**
 * @param {ReturnType<typeof getExecutionPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult}
 */
function evaluatePaperExecution(policy, context, base) {
  if (context.liveMode && !context.paperMode) {
    return buildResult(
      'BLOCK',
      'Paper and live broker environments must remain separated.',
      policyFields(policy, base),
    );
  }

  if (!context.paperMode && isProductionEnvironment(base.environment)) {
    return buildResult(
      'BLOCK',
      'Paper execution requires paper environment authorization.',
      policyFields(policy, base),
    );
  }

  if (policy.requiresRiskValidation && context.riskConstraintsSatisfied === false) {
    return buildResult(
      'BLOCK',
      'Paper execution blocked: risk constraints are not satisfied.',
      policyFields(policy, { ...base, requiresRiskValidation: true }),
    );
  }

  return buildResult(
    'ALLOW',
    'Paper execution permitted with environment authorization satisfied.',
    policyFields(policy, base),
  );
}

/**
 * @param {ReturnType<typeof getExecutionPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult | null}
 */
function evaluateBrokerSubmission(policy, context, base) {
  if (!context.executionAuthorized && !context.lifecycleApproved) {
    return buildResult(
      'BLOCK',
      'Broker submission blocked: valid execution authorization is required.',
      policyFields(policy, base),
    );
  }

  if (!context.brokerConnected) {
    return buildResult(
      'BLOCK',
      'Broker submission blocked: active broker connectivity is required.',
      policyFields(policy, base),
    );
  }

  if (!context.executionGuardApproved) {
    return buildResult(
      'BLOCK',
      'Broker submission blocked: execution guard approval is required.',
      policyFields(policy, base),
    );
  }

  if (context.duplicateOrderClear === false) {
    return buildResult(
      'BLOCK',
      'Broker submission blocked: duplicate order protection triggered.',
      policyFields(policy, base),
    );
  }

  return null;
}

/**
 * @param {ReturnType<typeof getExecutionPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult | null}
 */
function evaluateCapitalDeploymentRequirements(policy, context, base) {
  if (policy.requiresLifecycleApproval && !context.lifecycleApproved) {
    return buildResult(
      'BLOCK',
      'Capital deployment blocked: approved lifecycle decision is required.',
      policyFields(policy, base),
    );
  }

  if (policy.requiresAllocationApproval && !context.allocationApproved) {
    return buildResult(
      'BLOCK',
      'Capital deployment blocked: portfolio allocation approval is required.',
      policyFields(policy, base),
    );
  }

  if (policy.requiresRiskValidation && context.riskConstraintsSatisfied === false) {
    return buildResult(
      'BLOCK',
      'Capital deployment blocked: risk constraints, sizing, or concentration limits not satisfied.',
      policyFields(policy, { ...base, requiresRiskValidation: true }),
    );
  }

  return null;
}

/**
 * @param {ReturnType<typeof getExecutionPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult}
 */
function evaluateGovernanceOverride(policy, context, base) {
  if (!context.operatorAuthorized) {
    return buildResult(
      'REVIEW_REQUIRED',
      'Governance override requires verified operator identity.',
      policyFields(policy, { ...base, requiresOperator: true, requiresGovernance: true }),
    );
  }

  if (!context.overrideReason || !String(context.overrideReason).trim()) {
    return buildResult(
      'REVIEW_REQUIRED',
      'Governance override requires documented override reason for audit trail.',
      policyFields(policy, {
        ...base,
        requiresOperator: true,
        requiresGovernance: true,
        audit: { auditTrailId: context.auditTrailId || null },
      }),
    );
  }

  if (!context.operationalContext || !String(context.operationalContext).trim()) {
    return buildResult(
      'REVIEW_REQUIRED',
      'Governance override requires operational context for full traceability.',
      policyFields(policy, {
        ...base,
        requiresOperator: true,
        requiresGovernance: true,
        audit: {
          auditTrailId: context.auditTrailId || null,
          overrideReason: context.overrideReason,
        },
      }),
    );
  }

  return buildResult(
    'REVIEW_REQUIRED',
    'Governance override queued for review with audit evidence recorded.',
    policyFields(policy, {
      ...base,
      requiresOperator: true,
      requiresGovernance: true,
      audit: {
        auditTrailId: context.auditTrailId || null,
        overrideReason: context.overrideReason,
        operationalContext: context.operationalContext,
        operatorId: context.operatorId || context.userId || null,
      },
    }),
  );
}

/**
 * @param {ReturnType<typeof getExecutionPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult}
 */
function evaluateRiskEmergency(policy, context, base) {
  if (!context.riskEmergencyTriggered) {
    return buildResult(
      'BLOCK',
      'Risk emergency action blocked: no active risk emergency detected.',
      policyFields(policy, base),
    );
  }

  const action = context.emergencyAction;
  if (!action || !EMERGENCY_ACTIONS.has(action)) {
    return buildResult(
      'BLOCK',
      'Risk emergency action blocked: valid emergency action (FORCE_EXIT, FORCE_TRIM, RISK_LOCKDOWN) is required.',
      policyFields(policy, base),
    );
  }

  return buildResult(
    'ALLOW',
    `Emergency risk authority granted for ${action}.`,
    policyFields(policy, {
      ...base,
      audit: {
        emergencyAction: action,
        auditTrailId: context.auditTrailId || null,
      },
    }),
  );
}

/**
 * @param {string} path
 * @param {ExecutionGateContext} [context={}]
 * @returns {GateEvaluationResult}
 */
export function evaluateExecutionGate(path, context = {}) {
  const environment = context.environment || resolveEnvironment();
  const userId = context.userId || context.operatorId || '';
  const isProduction = isProductionEnvironment(environment);
  const isNonLive = isNonLiveEnvironment(environment);

  if (!isKnownExecutionPath(path)) {
    return buildResult(
      'BLOCK',
      `Unknown execution path "${path}" — all unrecognized paths are blocked.`,
      {
        environment,
        userId,
        path,
        riskLevel: 'critical',
        requiresRiskValidation: true,
        requiresGovernance: false,
        requiresOperator: false,
      },
    );
  }

  const policy = getExecutionPathPolicy(path);
  const base = {
    environment,
    userId,
    operatorId: context.operatorId,
    path,
    riskLevel: policy.riskLevel,
    isProduction,
    isNonLive,
  };

  if (policy.permanentlyDisabled || policy.defaultDecision === 'DISABLED') {
    return buildResult(
      'DISABLED',
      `Execution path "${path}" is permanently disabled.`,
      policyFields(policy, base),
    );
  }

  if (path === 'live_execution') {
    return evaluateLiveExecution(policy, context, base);
  }

  if (path === 'paper_execution') {
    return evaluatePaperExecution(policy, context, base);
  }

  if (path === 'broker_submission') {
    const brokerResult = evaluateBrokerSubmission(policy, context, base);
    if (brokerResult) {
      return brokerResult;
    }
    return buildResult(
      'ALLOW',
      'Broker submission authorized with execution guard and connectivity checks satisfied.',
      policyFields(policy, base),
    );
  }

  if (path === 'governance_override' || path === 'manual_operator_override') {
    return evaluateGovernanceOverride(policy, context, base);
  }

  if (path === 'risk_emergency_action') {
    return evaluateRiskEmergency(policy, context, base);
  }

  if (policy.deploysCapital || policy.requiresLifecycleApproval || policy.requiresAllocationApproval) {
    const capitalResult = evaluateCapitalDeploymentRequirements(policy, context, base);
    if (capitalResult) {
      return capitalResult;
    }
    if (policy.deploysCapital && isNonLive && !context.liveMode) {
      return buildResult(
        'ALLOW',
        `Execution path "${path}" permitted in non-live environment with deployment checks satisfied.`,
        policyFields(policy, base),
      );
    }
  }

  if (policy.liveBlocked && isProduction && context.liveMode && !context.liveModeAuthorized) {
    return buildResult(
      'BLOCK',
      'Live capital deployment is disabled by default; explicit production authorization is required.',
      policyFields(policy, base),
    );
  }

  if (policy.defaultDecision === 'ALLOW' && !policy.deploysCapital) {
    return buildResult(
      'ALLOW',
      `Execution path "${path}" permitted.`,
      policyFields(policy, base),
    );
  }

  if (policy.defaultDecision === 'ALLOW' && policy.deploysCapital && isNonLive) {
    return buildResult(
      'ALLOW',
      `Execution path "${path}" permitted in non-live environment.`,
      policyFields(policy, base),
    );
  }

  if (policy.defaultDecision === 'REVIEW_REQUIRED') {
    return buildResult(
      'REVIEW_REQUIRED',
      `Execution path "${path}" requires governance review before proceeding.`,
      policyFields(policy, base),
    );
  }

  return buildResult(
    policy.defaultDecision,
    `Default policy applied for execution path "${path}".`,
    policyFields(policy, base),
  );
}

/**
 * @param {string} path
 * @param {ExecutionGateContext} [context={}]
 * @returns {GateEvaluationResult}
 */
export function assertExecutionAllowed(path, context = {}) {
  const evaluation = evaluateExecutionGate(path, context);
  if (!evaluation.allowed) {
    throw new ExecutionGateError(evaluation);
  }
  return evaluation;
}

/**
 * @param {GateEvaluationResult} evaluation
 * @returns {string}
 */
export function explainExecutionDecision(evaluation) {
  if (!evaluation) {
    return 'No gate evaluation provided.';
  }

  const lines = [
    `Decision: ${evaluation.decision}`,
    `Allowed: ${evaluation.allowed ? 'yes' : 'no'}`,
    `Path: ${evaluation.path}`,
    `Environment: ${evaluation.environment}`,
    `Risk level: ${evaluation.risk_level}`,
    `Reason: ${evaluation.reason}`,
  ];

  if (evaluation.user_id) {
    lines.push(`Operator: ${evaluation.user_id}`);
  }
  if (evaluation.requires_risk_validation) {
    lines.push('Risk validation: required');
  }
  if (evaluation.requires_governance) {
    lines.push('Governance approval: required');
  }
  if (evaluation.requires_operator) {
    lines.push('Operator authorization: required');
  }
  if (evaluation.audit) {
    lines.push(`Audit: ${JSON.stringify(evaluation.audit)}`);
  }
  lines.push(`Evaluated at: ${evaluation.timestamp}`);

  return lines.join('\n');
}

/**
 * @param {string} path
 * @param {ExecutionGateContext} [context={}]
 * @returns {{ proceed: boolean, evaluation: GateEvaluationResult, httpStatus: number }}
 */
export function gateApiRequest(path, context = {}) {
  const evaluation = evaluateExecutionGate(path, context);
  return {
    proceed: evaluation.allowed,
    evaluation,
    httpStatus: evaluation.allowed ? 200 : evaluation.decision === 'REVIEW_REQUIRED' ? 202 : 403,
  };
}
