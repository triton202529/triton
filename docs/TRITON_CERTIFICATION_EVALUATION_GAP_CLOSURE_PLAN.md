# TRITON Certification Evaluation Gap Closure Plan (Phase 196U)

Read-only operator plan. **This document does not certify TRITON, edit scores, or submit orders.**

Re-check status:

```powershell
.\.venv\Scripts\python.exe -m services.phase_196u_certification_evaluation_gap_closure_plan_validation
```

## Current certification posture

| Signal | Value |
|--------|-------|
| Certification status | **NOT_CERTIFIED** |
| Overall score | **80.2** (7/8 areas certified) |
| Pre-veto status | PARTIALLY_CERTIFIED |
| Failed requirements | **Evaluation**, **veto:mandatory_preservation_gate_failed** |
| Mandatory veto | `mandatory_preservation_gate_failed` (failed_gates=Evaluation) |
| Readiness | **READY** (8/8 checks) |
| Governance authorization | **AUTHORIZED** (paper) |
| Live execution | **BLOCKED** |
| Paper arm | **EXPIRED** (separate from certification) |

**Important:** `NOT_CERTIFIED` is a **governance/certification display signal**. It does **not** directly block `paper_entry_loop_stage` today — paper entry is blocked by **ASSISTED `paper_arm_confirm`**. Certification remains a required governance concern before treating the system as fully certified for operational posture.

## Why Evaluation fails

The **Evaluation** area score is **34** (average trial effectiveness **34.8**).

| Trial | Score | Outcome |
|-------|-------|---------|
| Paper Concentration Reduction | 67 | BENEFICIAL |
| Paper Risk-Off Transition | 42 | NEUTRAL |
| Paper Drawdown Reduction | 25 | NEGATIVE |
| Paper Exposure Reduction | 5 | NEGATIVE |

**Threshold in effect:** `standard>=50` (because `high_cps_adjustment_applied=false` under `HIGH_CPS_PRESERVATION` baseline).

**Gap:** **16.0 points** below threshold (50 − 34.8).

**Root cause:** Two negative trials dominate the average despite one strong beneficial concentration trial. This is a **real evaluation evidence gap**, not stale artifacts — all certification sources refreshed on the latest cycle (~2026-06-25T02:19Z).

## Concentration and governance context (secondary)

| Signal | Value |
|--------|-------|
| Largest position weight | ~**51.7%** (WFC) |
| CPI concentration component | **40** |
| Escalation | **YELLOW** — EXCESS_CONCENTRATION |
| Governor posture | **ORANGE** — drivers: Concentration Risk, Weak Trial Outcomes |
| REDUCE_CONCENTRATION policy | **Disabled** for JPM/WFC candidates |
| Beneficial concentration simulation | Exists (score 67) — not sufficient alone |

Concentration is a **true risk driver** but **not the direct Evaluation math failure** — weak trial averages are.

## What must improve before certification can pass

1. **Raise average trial effectiveness to ≥50** without lowering thresholds or editing JSON by hand.
2. **Reduce negative trial count** (currently 2) or improve drawdown/exposure trial outcomes through legitimate portfolio/simulation evolution.
3. **Resolve mandatory preservation veto** automatically when Evaluation area certifies (do not suppress the veto).
4. **Address concentration** via governance-approved protective paths — not forced trades in this plan.

Optional high-CPS adjustment path (`>=40` with constraints) is **not active** (`high_cps_adjustment_applied=false`) and would still fail today because `negative_count=2` exceeds allowed limits.

## What must NOT be bypassed

- Do **not** lower Evaluation threshold (50 / 40 / 45 rules).
- Do **not** hand-edit `capital_preservation_certification.json` or `protective_action_evaluation.json`.
- Do **not** suppress `mandatory_preservation_gate_failed` veto.
- Do **not** weaken final risk overlay, allocation, or lifecycle rules.
- Do **not** run `execute_trades` manually to inflate trials.
- Do **not** arm paper mode to “fix” certification — **paper arm is not required** for certification closure.

## Safe closure actions (recommended order)

### Phase 196V — Evaluation observation cycle (recommended next)

- Continue healthy runtime / continuous loop.
- Allow natural protective evaluation refresh across multiple cycles.
- Track whether drawdown/exposure trials improve as portfolio state evolves.
- Re-run certification validation read-only after each cycle batch.
- **No trades required** if trials improve from observation alone.

### Phase 197A — Concentration governance review (after 196V)

- Operator reviews concentration reduction policy enablement for JPM/WFC.
- Review expired concentration approval requests in accountability registry.
- Consider approved protective simulations only — **no forced rebalancing**.
- Requires **manual governance review**; does not require paper arm.

## Operator checklist

| Question | Answer |
|----------|--------|
| Is paper arm needed to close certification? | **No** |
| Are market hours needed? | Helpful for cycle refresh, not strictly required for plan authoring |
| Is more observation needed? | **Yes** — primary path for 196V |
| Are code changes needed now? | **No** for 196U; reassess after observation if trials remain weak |
| Does certification block paper entry today? | **No directly** — `paper_arm_confirm` EXPIRED blocks entry |
| Does certification block live execution? | **Yes** — live remains blocked; NOT_CERTIFIED reinforces governance posture |

## Artifacts

- `data/results/phase_196u_certification_evaluation_gap_closure_plan.json`
- `data/results/phase_196u_certification_gap_map.csv`

## Doctrine

Observation and governance-first closure only. Certification passes only when **evidence supports it**, not when labels are forced.
