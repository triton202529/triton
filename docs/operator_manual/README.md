# TRITON Operator Manual

Index for operator-facing documentation. These pages explain posture, evidence, and safe work
boundaries. They do **not** create approval, broker authorization, or execution authority.

---

## Manual index

| Topic | Document |
|-------|----------|
| Partial GLD Defensive Path | [PARTIAL_GLD_DEFENSIVE_PATH.md](./PARTIAL_GLD_DEFENSIVE_PATH.md) |
| Weekend Safe Work Map | [WEEKEND_SAFE_WORK_MAP.md](./WEEKEND_SAFE_WORK_MAP.md) |
| Monday Market-Open Restart Checklist | [MONDAY_MARKET_OPEN_RESTART_CHECKLIST.md](./MONDAY_MARKET_OPEN_RESTART_CHECKLIST.md) |
| Paper Execution Safety Rules | [PAPER_EXECUTION_SAFETY_RULES.md](./PAPER_EXECUTION_SAFETY_RULES.md) |
| Approval Artifact Rules | [§ Approval Artifact Rules](#approval-artifact-rules) |
| Broker Authorization Rules | [§ Broker Authorization Rules](#broker-authorization-rules) |
| Market-Closed Safe Work | [§ Market-Closed Safe Work](#market-closed-safe-work) |
| Monday Market-Open Review | [§ Monday Market-Open Review](#monday-market-open-review) |
| Runtime Cleanup Rules | [§ Runtime Cleanup Rules](#runtime-cleanup-rules) |
| What Not To Touch While Market Is Closed | [§ What Not To Touch While Market Is Closed](#what-not-to-touch-while-market-is-closed) |

---

## Partial GLD Defensive Path

Certified no-execution path for partial GLD reduction (qty **48** of **195**; remaining **147**
deferred). Raw handoff may show **195**; authoritative dispatch qty is **48** only.

Full detail: [PARTIAL_GLD_DEFENSIVE_PATH.md](./PARTIAL_GLD_DEFENSIVE_PATH.md)

---

## Paper Execution Safety Rules

Full manual (four layers, short-lived artifacts, GLD partial qty, stop rules, authorization phrase):

→ [PAPER_EXECUTION_SAFETY_RULES.md](./PAPER_EXECUTION_SAFETY_RULES.md)

Summary:

- Paper execution policy being enabled does **not** mean orders may be placed.
- Paper execution is not implied by documentation, dry-run PASS, or no-execution certification.
- Never treat `data/live/manage_orders.csv` alone as broker dispatch authority.
- Use governed projected dispatch quantity from 197X / 198J / 198K evidence only.
- Live execution remains blocked unless separately and explicitly authorized by live policy.
- Do not place paper orders as part of weekend or market-closed documentation work.

---

## Approval Artifact Rules

- Path: `data/governance/operator_paper_defensive_exit_approval.json`
- Absent means no operator paper approval authority.
- Do **not** create this artifact during market-closed / weekend documentation work.
- Short-lived approval, when intentionally created later, must match the governed partial scope
  (e.g. GLD qty **48** only)—never expand to full raw handoff qty.
- Remove or expire approval after the intended paper window; do not leave standing authority overnight
  into the next session unless that is an explicit operator decision.

---

## Broker Authorization Rules

- Path: `data/governance/operator_paper_broker_execution_authorization.json`
- Absent means no paper broker execution authority.
- Do **not** create this artifact during market-closed / weekend documentation work.
- Authorization must match the projected partial dispatch scope; do not authorize raw full qty.
- Broker authorization without matching approval, or with mismatched quantity, is invalid.

---

## Market-Closed Safe Work

Preferred work while the market is closed:

- Documentation and operator manuals
- Checklists and explanations
- Read-only audits and report summaries

Classify every task using [WEEKEND_SAFE_WORK_MAP.md](./WEEKEND_SAFE_WORK_MAP.md).

### Explicit safety warning

> **Market-closed work must not create execution authority for the next market session.**

---

## Monday Market-Open Review

Full operator checklist (commands, stop points, paper preconditions):

→ [MONDAY_MARKET_OPEN_RESTART_CHECKLIST.md](./MONDAY_MARKET_OPEN_RESTART_CHECKLIST.md)

Before any market-open action, rerun:

1. Broker snapshot
2. Open-order check
3. Readiness check
4. Approval artifact check (`Test-Path` / absence of
   `data/governance/operator_paper_defensive_exit_approval.json` unless intentionally created)
5. Broker authorization artifact check (`Test-Path` / absence of
   `data/governance/operator_paper_broker_execution_authorization.json` unless intentionally created)

Do not assume Friday or weekend state still applies Monday morning.

> Market-open restart does not itself authorize paper execution.

---

## Runtime Cleanup Rules

- Prefer not committing noisy runtime `data/governance` / `data/results` churn from observation loops.
- Do not leave real approval or broker-authorization artifacts after dry-run or documentation work.
- Restore unintended tracked runtime edits before commits when the phase is documentation-only.
- Generated certification JSON under `data/results/` is evidence, not execution authority.

---

## What Not To Touch While Market Is Closed

Do **not** touch:

- Broker dispatch / order submission
- `manage_positions` execution
- Approval artifact creation
- Broker authorization creation
- Scheduled tasks
- Live execution gates
- Strategy threshold changes
- BUY/ADD gates
- Runtime governance state under `data/governance/` (except read-only inspection)
- `data/live/` mutation for execution handoff

See also: [WEEKEND_SAFE_WORK_MAP.md](./WEEKEND_SAFE_WORK_MAP.md) — **DO NOT TOUCH** list.
