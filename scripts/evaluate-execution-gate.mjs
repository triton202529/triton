#!/usr/bin/env node
/**
 * evaluate-execution-gate.mjs
 * THP-003 — CLI bridge for Python pipeline integration with the JS execution gate.
 *
 * Usage:
 *   node scripts/evaluate-execution-gate.mjs '{"path":"broker_submission","context":{...}}'
 *   node scripts/evaluate-execution-gate.mjs '{"path":"...","context":{...},"assert":true}'
 */

import {
  evaluateExecutionGate,
  assertExecutionAllowed,
  explainExecutionDecision,
} from '../lib/executionGate.js';

function readPayload() {
  const raw = process.argv[2];
  if (!raw) {
    throw new Error('Missing JSON payload argument');
  }
  return JSON.parse(raw);
}

function main() {
  const payload = readPayload();
  const path = payload.path;
  const context = payload.context || {};
  const doAssert = Boolean(payload.assert);

  if (!path) {
    throw new Error('Missing "path" in payload');
  }

  if (doAssert) {
    const evaluation = assertExecutionAllowed(path, context);
    process.stdout.write(JSON.stringify({
      ok: true,
      evaluation,
      explanation: explainExecutionDecision(evaluation),
    }));
    return;
  }

  const evaluation = evaluateExecutionGate(path, context);
  process.stdout.write(JSON.stringify({
    ok: evaluation.allowed,
    evaluation,
    explanation: explainExecutionDecision(evaluation),
  }));
}

try {
  main();
} catch (error) {
  const evaluation = error.evaluation || null;
  process.stdout.write(JSON.stringify({
    ok: false,
    error: error.message || String(error),
    evaluation,
    explanation: evaluation ? explainExecutionDecision(evaluation) : null,
  }));
  process.exit(1);
}
