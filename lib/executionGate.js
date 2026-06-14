/**
 * executionGate.js
 * THP-001 — Centralized execution gate for all Tropicash money paths.
 *
 * Every wallet funding, transfer, withdrawal, payout, merchant payment,
 * tap-to-pay, developer wallet, OAuth, and admin release operation must
 * pass through evaluateExecutionGate() before proceeding.
 *
 * This module adds governance only — it does not mutate balances, credentials,
 * OAuth tokens, or KYC policy tables.
 */

import {
  getMoneyPathPolicy,
  isKnownMoneyPath,
  isProductionEnvironment,
  isSandboxEnvironment,
  resolveEnvironment,
} from './moneyPathProtection.js';

export {
  getMoneyPathStatus,
  listAllMoneyPathStatuses,
  MONEY_PATHS,
  ALL_MONEY_PATHS,
} from './moneyPathProtection.js';

/**
 * @typedef {import('./moneyPathProtection.js').GateDecisionType} GateDecisionType
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
 *   requires_kyc: boolean,
 *   requires_admin: boolean,
 *   timestamp: string,
 *   audit?: Record<string, unknown>,
 * }} GateEvaluationResult
 */

/**
 * @typedef {{
 *   userId?: string,
 *   environment?: string,
 *   kycVerified?: boolean,
 *   kycLevel?: string,
 *   isAdmin?: boolean,
 *   adminAuthorized?: boolean,
 *   operationalContext?: string,
 *   oauthScope?: string,
 *   oauthMutationRequested?: boolean,
 *   withdrawalAmount?: number,
 *   withdrawalLimitRemaining?: number,
 *   sandboxMode?: boolean,
 *   productionReadinessOverride?: boolean,
 *   auditTrailId?: string,
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
    user_id: fields.userId || '',
    path: fields.path,
    risk_level: fields.riskLevel || 'medium',
    requires_kyc: Boolean(fields.requiresKyc),
    requires_admin: Boolean(fields.requiresAdmin),
    timestamp: utcTimestamp(),
    ...(fields.audit ? { audit: fields.audit } : {}),
  };
}

/**
 * Evaluate OAuth wallet access — read scopes allowed, balance mutation forbidden.
 *
 * @param {ReturnType<typeof getMoneyPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult}
 */
function evaluateOAuthPath(policy, context, base) {
  const scope = String(context.oauthScope || '').trim().toLowerCase();
  const mutationRequested = Boolean(context.oauthMutationRequested);

  if (mutationRequested) {
    return buildResult(
      'BLOCK',
      'OAuth wallet access cannot mutate balances directly.',
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  if (scope && policy.forbiddenOAuthScopes?.includes(scope)) {
    return buildResult(
      'BLOCK',
      `OAuth scope "${scope}" is forbidden for wallet balance mutation.`,
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  if (scope && policy.allowedOAuthScopes?.includes(scope)) {
    return buildResult(
      'ALLOW',
      `OAuth read scope "${scope}" is authorized.`,
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  if (!scope) {
    return buildResult(
      'REVIEW_REQUIRED',
      'OAuth wallet access requires an explicit approved read scope.',
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  return buildResult(
    'REVIEW_REQUIRED',
    `OAuth scope "${scope}" requires manual review before access.`,
    { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
  );
}

/**
 * Evaluate wallet withdrawal — enforce KYC, limits, and compliance placeholders.
 *
 * @param {ReturnType<typeof getMoneyPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult | null} null to fall through to default rules
 */
function evaluateWithdrawalPath(policy, context, base) {
  if (!context.kycVerified) {
    return buildResult(
      'BLOCK',
      'Withdrawal blocked: KYC verification is required.',
      { ...base, requiresKyc: true, requiresAdmin: policy.requiresAdmin },
    );
  }

  const amount = context.withdrawalAmount;
  const limitRemaining = context.withdrawalLimitRemaining;

  if (typeof amount === 'number' && typeof limitRemaining === 'number' && amount > limitRemaining) {
    return buildResult(
      'BLOCK',
      'Withdrawal blocked: amount exceeds remaining withdrawal limit.',
      { ...base, requiresKyc: true, requiresAdmin: policy.requiresAdmin },
    );
  }

  return null;
}

/**
 * Evaluate admin manual release — requires admin auth, audit trail, and context.
 *
 * @param {ReturnType<typeof getMoneyPathPolicy>} policy
 * @param {ExecutionGateContext} context
 * @param {object} base
 * @returns {GateEvaluationResult | null}
 */
function evaluateAdminReleasePath(policy, context, base) {
  if (!context.isAdmin || !context.adminAuthorized) {
    return buildResult(
      'REVIEW_REQUIRED',
      'Admin manual release requires administrative authorization.',
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: true },
    );
  }

  if (!context.operationalContext || !String(context.operationalContext).trim()) {
    return buildResult(
      'REVIEW_REQUIRED',
      'Admin manual release requires operational context for audit trail.',
      {
        ...base,
        requiresKyc: policy.requiresKyc,
        requiresAdmin: true,
        audit: { auditTrailId: context.auditTrailId || null },
      },
    );
  }

  if (base.isProduction && !context.productionReadinessOverride) {
    return buildResult(
      'BLOCK',
      'Admin manual release is blocked in production until production readiness controls enable it.',
      {
        ...base,
        requiresKyc: policy.requiresKyc,
        requiresAdmin: true,
        audit: {
          auditTrailId: context.auditTrailId || null,
          operationalContext: context.operationalContext,
        },
      },
    );
  }

  return buildResult(
    'REVIEW_REQUIRED',
    'Admin manual release queued for review with audit trail recorded.',
    {
      ...base,
      requiresKyc: policy.requiresKyc,
      requiresAdmin: true,
      audit: {
        auditTrailId: context.auditTrailId || null,
        operationalContext: context.operationalContext,
      },
    },
  );
}

/**
 * Core gate evaluation. Single source of truth for whether a financial operation
 * may proceed.
 *
 * @param {string} path — money path category (see MONEY_PATHS)
 * @param {ExecutionGateContext} [context={}]
 * @returns {GateEvaluationResult}
 */
export function evaluateExecutionGate(path, context = {}) {
  const environment = context.environment || resolveEnvironment();
  const userId = context.userId || '';
  const isProduction = isProductionEnvironment(environment);
  const isSandbox = isSandboxEnvironment(environment) || Boolean(context.sandboxMode);

  if (!isKnownMoneyPath(path)) {
    return buildResult(
      'BLOCK',
      `Unknown money path "${path}" — all unrecognized paths are blocked.`,
      {
        environment,
        userId,
        path,
        riskLevel: 'critical',
        requiresKyc: true,
        requiresAdmin: false,
      },
    );
  }

  const policy = getMoneyPathPolicy(path);
  const base = {
    environment,
    userId,
    path,
    riskLevel: policy.riskLevel,
    isProduction,
    isSandbox,
  };

  // Rule 2: Tap-to-pay permanently disabled until certifications complete.
  if (policy.permanentlyDisabled || policy.defaultDecision === 'DISABLED') {
    return buildResult(
      'DISABLED',
      'Tap-to-pay is disabled until payment network integration, security certification, device compatibility validation, and compliance approval are complete.',
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  // Rule 6: OAuth wallet access — scoped reads only, no balance mutation.
  if (path === 'oauth_wallet_access') {
    return evaluateOAuthPath(policy, context, base);
  }

  // Rule 4: Wallet withdrawals — KYC and limit enforcement.
  if (path === 'wallet_withdrawal') {
    const withdrawalResult = evaluateWithdrawalPath(policy, context, base);
    if (withdrawalResult) {
      return withdrawalResult;
    }
  }

  // Rule 5: Admin manual release — admin auth + audit + context.
  if (path === 'admin_manual_release') {
    return evaluateAdminReleasePath(policy, context, base);
  }

  // Rule 3: Developer wallet APIs — sandbox only, no production mutations.
  if (policy.sandboxOnly) {
    if (isSandbox) {
      return buildResult(
        'SANDBOX_ONLY',
        'Developer wallet API permitted in sandbox/testing environment only.',
        { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
      );
    }
    return buildResult(
      'BLOCK',
      'Developer wallet API blocked outside sandbox — no production wallet mutations allowed.',
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  // Rule 1: Production money movement — default BLOCK unless future override.
  if (isProduction && policy.productionBlocked && !context.productionReadinessOverride) {
    return buildResult(
      'BLOCK',
      'Production money movement is disabled until explicitly enabled through production readiness controls.',
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  // Non-production paths that require KYC when policy says so.
  if (policy.requiresKyc && policy.allowsBalanceMutation && !context.kycVerified) {
    return buildResult(
      'BLOCK',
      `KYC verification required before ${path.replace(/_/g, ' ')}.`,
      { ...base, requiresKyc: true, requiresAdmin: policy.requiresAdmin },
    );
  }

  // Staging/sandbox: allow simulated flows when KYC satisfied (if required).
  if (!isProduction && policy.allowsBalanceMutation) {
    return buildResult(
      'ALLOW',
      `Money path "${path}" permitted in non-production environment.`,
      { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
    );
  }

  return buildResult(
    policy.defaultDecision,
    `Default policy applied for money path "${path}".`,
    { ...base, requiresKyc: policy.requiresKyc, requiresAdmin: policy.requiresAdmin },
  );
}

/**
 * Assert that a money path is allowed. Throws ExecutionGateError when blocked.
 *
 * @param {string} path
 * @param {ExecutionGateContext} [context={}]
 * @returns {GateEvaluationResult}
 */
export function assertMoneyPathAllowed(path, context = {}) {
  const evaluation = evaluateExecutionGate(path, context);
  if (!evaluation.allowed) {
    throw new ExecutionGateError(evaluation);
  }
  return evaluation;
}

/**
 * Human-readable explanation of a gate evaluation result.
 *
 * @param {GateEvaluationResult} evaluation
 * @returns {string}
 */
export function explainGateDecision(evaluation) {
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
    lines.push(`User: ${evaluation.user_id}`);
  }
  if (evaluation.requires_kyc) {
    lines.push('KYC: required');
  }
  if (evaluation.requires_admin) {
    lines.push('Admin authorization: required');
  }
  if (evaluation.audit) {
    lines.push(`Audit: ${JSON.stringify(evaluation.audit)}`);
  }
  lines.push(`Evaluated at: ${evaluation.timestamp}`);

  return lines.join('\n');
}

/**
 * Convenience wrapper for API route handlers.
 * Returns a 403-style payload when blocked; does not throw.
 *
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
