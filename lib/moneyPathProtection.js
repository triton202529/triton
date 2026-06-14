/**
 * moneyPathProtection.js
 * THP-001 — Registry and default policies for all TRITON money paths.
 *
 * Single source of truth for path classification, risk metadata, and
 * baseline protection rules. Does not mutate balances or call external services.
 */

/** @typedef {'ALLOW' | 'BLOCK' | 'REVIEW_REQUIRED' | 'SANDBOX_ONLY' | 'DISABLED'} GateDecisionType */

/** @typedef {'low' | 'medium' | 'high' | 'critical'} RiskLevel */

/**
 * Canonical money path identifiers. Every money-moving entry point must
 * register under one of these categories before execution.
 */
export const MONEY_PATHS = Object.freeze({
  WALLET_FUNDING: 'wallet_funding',
  WALLET_SEND: 'wallet_send',
  WALLET_WITHDRAWAL: 'wallet_withdrawal',
  PAYPAL_PAYOUT: 'paypal_payout',
  MERCHANT_PAYMENT: 'merchant_payment',
  TAP_TO_PAY: 'tap_to_pay',
  DEVELOPER_WALLET_API: 'developer_wallet_api',
  OAUTH_WALLET_ACCESS: 'oauth_wallet_access',
  ADMIN_MANUAL_RELEASE: 'admin_manual_release',
});

/** @type {ReadonlySet<string>} */
export const ALL_MONEY_PATHS = new Set(Object.values(MONEY_PATHS));

/**
 * Environments recognized by the execution gate.
 * Production money movement defaults to BLOCK unless explicitly overridden
 * by future production readiness controls (not implemented in this phase).
 */
export const ENVIRONMENTS = Object.freeze({
  PRODUCTION: 'production',
  STAGING: 'staging',
  SANDBOX: 'sandbox',
  DEVELOPMENT: 'development',
});

/** @type {ReadonlySet<string>} */
const SANDBOX_ENVIRONMENTS = new Set([
  ENVIRONMENTS.SANDBOX,
  ENVIRONMENTS.DEVELOPMENT,
]);

/**
 * Default policy per money path. These baselines are applied before any
 * request-specific context overrides in evaluateExecutionGate().
 *
 * @type {Record<string, {
 *   defaultDecision: GateDecisionType,
 *   riskLevel: RiskLevel,
 *   description: string,
 *   requiresKyc: boolean,
 *   requiresAdmin: boolean,
 *   productionBlocked: boolean,
 *   sandboxOnly: boolean,
 *   permanentlyDisabled: boolean,
 *   allowsBalanceMutation: boolean,
 *   allowedOAuthScopes?: string[],
 *   forbiddenOAuthScopes?: string[],
 * }>}
 */
export const MONEY_PATH_POLICIES = Object.freeze({
  [MONEY_PATHS.WALLET_FUNDING]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'high',
    description: 'Add funds to a user wallet via card, bank, or external source.',
    requiresKyc: true,
    requiresAdmin: false,
    productionBlocked: true,
    sandboxOnly: false,
    permanentlyDisabled: false,
    allowsBalanceMutation: true,
  },
  [MONEY_PATHS.WALLET_SEND]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'high',
    description: 'Peer-to-peer or internal wallet transfer between accounts.',
    requiresKyc: true,
    requiresAdmin: false,
    productionBlocked: true,
    sandboxOnly: false,
    permanentlyDisabled: false,
    allowsBalanceMutation: true,
  },
  [MONEY_PATHS.WALLET_WITHDRAWAL]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'critical',
    description: 'Withdraw wallet balance to external bank or payout destination.',
    requiresKyc: true,
    requiresAdmin: false,
    productionBlocked: true,
    sandboxOnly: false,
    permanentlyDisabled: false,
    allowsBalanceMutation: true,
  },
  [MONEY_PATHS.PAYPAL_PAYOUT]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'critical',
    description: 'Execute PayPal payout to external recipient.',
    requiresKyc: true,
    requiresAdmin: false,
    productionBlocked: true,
    sandboxOnly: false,
    permanentlyDisabled: false,
    allowsBalanceMutation: true,
  },
  [MONEY_PATHS.MERCHANT_PAYMENT]: {
    defaultDecision: 'BLOCK',
    riskLevel: 'high',
    description: 'Merchant checkout or payment capture against wallet balance.',
    requiresKyc: true,
    requiresAdmin: false,
    productionBlocked: true,
    sandboxOnly: false,
    permanentlyDisabled: false,
    allowsBalanceMutation: true,
  },
  [MONEY_PATHS.TAP_TO_PAY]: {
    defaultDecision: 'DISABLED',
    riskLevel: 'critical',
    description: 'In-person tap-to-pay via payment network hardware.',
    requiresKyc: true,
    requiresAdmin: false,
    productionBlocked: true,
    sandboxOnly: false,
    permanentlyDisabled: true,
    allowsBalanceMutation: true,
  },
  [MONEY_PATHS.DEVELOPER_WALLET_API]: {
    defaultDecision: 'SANDBOX_ONLY',
    riskLevel: 'medium',
    description: 'Developer console wallet API for testing integrations.',
    requiresKyc: false,
    requiresAdmin: false,
    productionBlocked: true,
    sandboxOnly: true,
    permanentlyDisabled: false,
    allowsBalanceMutation: true,
  },
  [MONEY_PATHS.OAUTH_WALLET_ACCESS]: {
    defaultDecision: 'ALLOW',
    riskLevel: 'medium',
    description: 'Third-party OAuth access to wallet data and scoped operations.',
    requiresKyc: false,
    requiresAdmin: false,
    productionBlocked: false,
    sandboxOnly: false,
    permanentlyDisabled: false,
    allowsBalanceMutation: false,
    allowedOAuthScopes: ['wallet:read', 'wallet:balance:read', 'wallet:transactions:read'],
    forbiddenOAuthScopes: ['wallet:write', 'wallet:transfer', 'wallet:withdraw'],
  },
  [MONEY_PATHS.ADMIN_MANUAL_RELEASE]: {
    defaultDecision: 'REVIEW_REQUIRED',
    riskLevel: 'critical',
    description: 'Administrative manual release or approval of held funds.',
    requiresKyc: false,
    requiresAdmin: true,
    productionBlocked: true,
    sandboxOnly: false,
    permanentlyDisabled: false,
    allowsBalanceMutation: true,
  },
});

/**
 * Known integration entry points (API routes, pages, components) mapped to
 * money paths. Used for documentation, auditing, and future middleware wiring.
 * No balance mutation occurs here — classification only.
 *
 * @type {ReadonlyArray<{ id: string, path: string, surface: string, notes?: string }>}
 */
export const MONEY_PATH_ENTRY_POINTS = Object.freeze([
  { id: 'pages-fund', path: MONEY_PATHS.WALLET_FUNDING, surface: 'pages/fund*', notes: 'Wallet funding UI' },
  { id: 'pages-send', path: MONEY_PATHS.WALLET_SEND, surface: 'pages/send*', notes: 'Send money UI' },
  { id: 'pages-withdraw', path: MONEY_PATHS.WALLET_WITHDRAWAL, surface: 'pages/withdraw*', notes: 'Withdrawal UI' },
  { id: 'api-wallet-fund', path: MONEY_PATHS.WALLET_FUNDING, surface: 'pages/api/wallet/fund', notes: 'Funding API' },
  { id: 'api-wallet-send', path: MONEY_PATHS.WALLET_SEND, surface: 'pages/api/wallet/send', notes: 'Send API' },
  { id: 'api-wallet-withdraw', path: MONEY_PATHS.WALLET_WITHDRAWAL, surface: 'pages/api/wallet/withdraw', notes: 'Withdrawal API' },
  { id: 'api-paypal-payout', path: MONEY_PATHS.PAYPAL_PAYOUT, surface: 'pages/api/paypal/payout', notes: 'PayPal payout API' },
  { id: 'api-merchant-pay', path: MONEY_PATHS.MERCHANT_PAYMENT, surface: 'pages/api/merchant/pay', notes: 'Merchant payment API' },
  { id: 'api-tap-to-pay', path: MONEY_PATHS.TAP_TO_PAY, surface: 'pages/api/tap-to-pay', notes: 'Tap-to-pay API' },
  { id: 'dev-console', path: MONEY_PATHS.DEVELOPER_WALLET_API, surface: 'pages/dev-console*', notes: 'Developer wallet console' },
  { id: 'oauth-wallet', path: MONEY_PATHS.OAUTH_WALLET_ACCESS, surface: 'pages/api/oauth/wallet', notes: 'OAuth wallet access' },
  { id: 'admin-release', path: MONEY_PATHS.ADMIN_MANUAL_RELEASE, surface: 'pages/api/admin/release', notes: 'Admin manual release' },
]);

/**
 * Resolve the active TRITON environment from process env vars.
 * Does not read or modify production environment settings files.
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
export function isKnownMoneyPath(path) {
  return ALL_MONEY_PATHS.has(path);
}

/**
 * @param {string} path
 * @returns {typeof MONEY_PATH_POLICIES[string] | null}
 */
export function getMoneyPathPolicy(path) {
  if (!isKnownMoneyPath(path)) {
    return null;
  }
  return MONEY_PATH_POLICIES[path];
}

/**
 * @param {string} path
 * @returns {ReadonlyArray<{ id: string, path: string, surface: string, notes?: string }>}
 */
export function getEntryPointsForPath(path) {
  return MONEY_PATH_ENTRY_POINTS.filter((entry) => entry.path === path);
}

/**
 * @param {string} environment
 * @returns {boolean}
 */
export function isSandboxEnvironment(environment) {
  return SANDBOX_ENVIRONMENTS.has(environment);
}

/**
 * @param {string} environment
 * @returns {boolean}
 */
export function isProductionEnvironment(environment) {
  return environment === ENVIRONMENTS.PRODUCTION;
}

/**
 * Returns static status metadata for a money path (policy defaults + entry points).
 * Does not evaluate a specific user request.
 *
 * @param {string} path
 * @returns {{
 *   path: string,
 *   known: boolean,
 *   defaultDecision: GateDecisionType | null,
 *   riskLevel: RiskLevel | null,
 *   requiresKyc: boolean,
 *   requiresAdmin: boolean,
 *   productionBlocked: boolean,
 *   sandboxOnly: boolean,
 *   permanentlyDisabled: boolean,
 *   entryPoints: ReadonlyArray<{ id: string, path: string, surface: string, notes?: string }>,
 *   description: string | null,
 * } | null}
 */
export function getMoneyPathStatus(path) {
  const policy = getMoneyPathPolicy(path);
  if (!policy) {
    return {
      path,
      known: false,
      defaultDecision: null,
      riskLevel: null,
      requiresKyc: false,
      requiresAdmin: false,
      productionBlocked: true,
      sandboxOnly: false,
      permanentlyDisabled: false,
      entryPoints: [],
      description: null,
    };
  }

  return {
    path,
    known: true,
    defaultDecision: policy.defaultDecision,
    riskLevel: policy.riskLevel,
    requiresKyc: policy.requiresKyc,
    requiresAdmin: policy.requiresAdmin,
    productionBlocked: policy.productionBlocked,
    sandboxOnly: policy.sandboxOnly,
    permanentlyDisabled: policy.permanentlyDisabled,
    entryPoints: getEntryPointsForPath(path),
    description: policy.description,
  };
}

/**
 * List all registered money paths with their static status.
 *
 * @returns {Array<NonNullable<ReturnType<typeof getMoneyPathStatus>>>}
 */
export function listAllMoneyPathStatuses() {
  return [...ALL_MONEY_PATHS].map((path) => getMoneyPathStatus(path));
}
