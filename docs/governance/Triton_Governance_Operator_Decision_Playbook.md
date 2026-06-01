# Triton Governance Operator Decision Tree & Escalation Playbook

**Document type:** Governance Manual — Operator Decision Tree & Escalation
**System:** Triton Institutional Trading Platform
**Classification:** Internal — Operator / Developer / Audit / Training
**Version:** 1.0
**Status:** Manual-ready SOP
**Companion document:** [Triton Governance Incident & Escalation Framework](./Triton_Governance_Incident_Escalation_Framework.md) (Step 90)

---

## Purpose

This playbook converts **Governance Command Center (GCC) posture** into **clear operator action discipline**.

After GCC renders the Operator Decision Brief, an operator must be able to answer in under **5 seconds**:

| Question | Answer source |
|----------|---------------|
| What posture are we in? | Final Operator Brief |
| What should I do? | Immediate Instruction |
| What should I NOT do? | Blocked Condition |
| Should I escalate? | This playbook — Section 1 + 3 |
| Who do I notify? | Escalation chain — Section 3 |
| Do we halt? | Halt playbook — Section 4 |
| Do we wait? | Monitoring cadence — Section 1 |
| When do we reassess? | Reassessment trigger — Section 1 |

**Capital Preservation Doctrine:** When uncertain, observe, contain, and escalate before acting. Default posture is **no runtime enablement** unless explicitly authorized through governance escalation.

This document is **procedural only**. It does not modify runtime, governance engines, execution paths, or broker integration.

---

## How to use this playbook

1. Open GCC → **Governance Command Center** → **Operator Decision Brief** (top card).
2. Read **Final Operator Brief**, **Immediate Instruction**, **Watch Condition**, **Blocked Condition**.
3. Jump to the matching posture playbook (Section 1).
4. If escalation or halt is indicated, use Sections 3–4.
5. Log actions per Section 5 and weekend rules (Section 6) as applicable.

---

# Section 1 — Governance Posture Decision Matrix

GCC assigns one of five operator briefs. Each brief maps to a governance mode and a single immediate instruction. **Follow the instruction. Do not improvise.**

| Final Operator Brief | Governance Mode | Operator Command |
|---------------------|-----------------|------------------|
| LOCKED_OBSERVE_ONLY | Constitutional Lock Mode | **MAINTAIN_LOCK_AND_OBSERVE** |
| LOCKED_HEIGHTENED_MONITORING | Constitutional Lock Mode | **MONITOR_AND_PREPARE_ESCALATION** |
| GOVERNANCE_REPAIR_REQUIRED | Repair Mode | **ESCALATE_AND_CONTAIN** |
| TRANSITION_WATCH | Transition Mode | **OBSERVE_VALIDATE_AND_REASSESS** |
| STABLE_CONTINUE_MONITORING | Observation Mode | **ROUTINE_MONITORING_ONLY** |

---

## 1. LOCKED_OBSERVE_ONLY

### Meaning

- Constitutional lock is active
- No material governance deterioration detected
- Directionally stable institutional posture
- Runtime mutation blocked

This is the **default disciplined posture** for a locked system. It is not an emergency.

### Operator objective

Preserve constitutional lock, maintain observability, and avoid unnecessary escalation or intervention.

### Immediate actions

1. **MAINTAIN_LOCK_AND_OBSERVE**
2. Confirm **Blocked Condition** = Runtime enablement (or equivalent).
3. Note **Watch Condition** (e.g., trustworthiness drift, contradiction persistence).
4. Refresh GCC at next scheduled cadence.
5. Log posture confirmation in operator log (timestamp UTC).

### Prohibited actions

- Runtime enablement
- Policy override or governance relaxation
- Execution pressure or “just one trade” exceptions
- Premature escalation without trigger breach
- Manual edits to governance JSON or memory artifacts
- Broker actions driven by dashboard anxiety

### Escalation requirement

**None** unless Watch Condition deteriorates or reassessment trigger fires.

Escalate to Senior Operator only if:

- Posture changes to LOCKED_HEIGHTENED_MONITORING or worse, or
- Same Watch Condition worsens for **two consecutive** refresh cycles.

### Monitoring cadence

- **Routine:** next GCC refresh (scheduled governance run or dashboard reload)
- **Passive:** note cockpit strips (Severity, Trend, Confidence) — no action if stable

### Reassessment trigger

Reassess immediately if any of the following occur:

- Final Operator Brief changes
- Watch Condition escalates (e.g., trust BROKEN/WEAK, rising contradiction intensity)
- Blocked Condition bypass attempted or reported
- Reconciliation or lifecycle mismatch detected
- Unexpected execution or fill notification

### Example operator response

> “Posture: LOCKED_OBSERVE_ONLY. Instruction: MAINTAIN_LOCK_AND_OBSERVE. Lock holds, drift minimal. No escalation. Reassess at next refresh.”

---

## 2. LOCKED_HEIGHTENED_MONITORING

### Meaning

- Governance remains constitutionally locked
- Elevated stress signals present
- Contradictions or trust/coherence pressure rising
- Deterioration **possible** but not confirmed

This is **heightened attention**, not a crisis. Avoid panic framing.

### Operator objective

**MONITOR_AND_PREPARE_ESCALATION** — increase observation discipline without breaking lock or overreacting.

### Immediate actions

1. **MONITOR_AND_PREPARE_ESCALATION**
2. Document Watch Condition and current cockpit severity/trend strips.
3. Confirm runtime remains blocked.
4. Shorten passive monitoring interval (e.g., check GCC twice per session vs once).
5. Pre-stage escalation contacts (do not notify until threshold met).
6. Review Scenario Outlook / deterioration triggers in GCC expanders.

### Escalation threshold

Escalate to **Senior Operator** when:

- Brief persists for **3+ refresh cycles** without improvement, or
- Watch Condition matches an active deterioration trigger, or
- Trend strip moves from stable to material drift

Escalate to **Risk / Governance Lead** when:

- Brief upgrades to GOVERNANCE_REPAIR_REQUIRED or TRANSITION_WATCH with adverse trend, or
- Reconciliation uncertainty appears

### Watch conditions

Typical GCC watch items:

- Trustworthiness drift
- Contradiction persistence
- Alignment fragmentation
- Governance coherence weakening
- Regime transition signals

### Monitoring cadence

- **Active:** every refresh until posture improves or escalates
- **Log:** each refresh — brief, watch condition, one-line trend note

### Soft halt considerations

- **Default:** no halt required if execution already blocked by constitutional lock
- **Soft Halt** if: scheduled execution window approaches AND stale-data or lifecycle inconsistency appears
- Soft Halt is containment, not alarm — document reason and notify Senior Operator

### Example operator response

> “Posture: LOCKED_HEIGHTENED_MONITORING. MONITOR_AND_PREPARE_ESCALATION. Stress elevated, lock intact. Watching contradiction persistence. Senior Operator notified if trigger holds next cycle.”

---

## 3. GOVERNANCE_REPAIR_REQUIRED

### Meaning

- Governance integrity degrading
- Material contradictions or weak trust/coherence
- Institutional repair needed before any runtime consideration
- Runtime enablement inappropriate

### Operator objective

**ESCALATE_AND_CONTAIN** — stop forward motion, preserve evidence, invoke governance review.

### Immediate containment

1. **ESCALATE_AND_CONTAIN**
2. Enforce **Soft Halt** if any execution path could activate (Section 4).
3. Do not attempt repair via runtime toggles or overrides.
4. Preserve GCC snapshot, logs, and affected artifacts.
5. Run reconciliation if authorized and trained (`lifecycle_reconciliation` review).

### Escalation path

```
Operator → Senior Operator (within 30 min)
         → Risk / Governance Lead (within 30 min)
         → Triton System Administrator (if pipeline/artifact failure suspected)
```

Classify as **Level 3 — Trading Risk** per Step 90 if lifecycle or execution eligibility is implicated.

### Required documentation

- Incident record (Step 90 template) — minimum preliminary entry
- GCC Decision Brief screenshot or export
- Watch Condition + Blocked Condition at time of detection
- Timeline of posture change (when brief shifted from locked-observe to repair)

### Halt considerations

| Condition | Halt |
|-----------|------|
| Execution already blocked | Maintain lock; Soft Halt as formal posture |
| Scheduled run imminent | **Soft Halt** mandatory |
| Reconciliation mismatch | Soft Halt → escalate Level 3 |
| Integrity compromise suspected | **Hard Halt** — Section 4 |

### Required approvals

- **No execution resume** without Risk / Governance Lead written approval
- **No override** without dual approval per Step 90 Section 5
- **Hard Halt lift** requires Governance Committee if Level 4 criteria met

### Review expectations

- Preliminary RCA within **4 hours**
- Governance review memo within **3 business days** (Level 3)
- Operator remains in contain-and-observe until explicit stand-down

### Example operator response

> “Posture: GOVERNANCE_REPAIR_REQUIRED. ESCALATE_AND_CONTAIN. Soft Halt confirmed. Risk Lead notified 14:32 UTC. Evidence preserved. No runtime action.”

---

## 4. TRANSITION_WATCH

### Meaning

- Governance materially shifting
- Stabilization **or** deterioration plausible
- Regime transition signals present
- Ambiguity is expected — discipline is required

### Operator objective

**OBSERVE_VALIDATE_AND_REASSESS** — gather evidence, avoid premature escalation, maintain readiness.

### Decision ambiguity handling

1. Do not force a binary “good/bad” judgment on a single refresh.
2. Compare **two consecutive** refresh cycles before escalating.
3. Weight **direction of change** (improving vs deteriorating) over absolute labels.
4. If ambiguous after two cycles → Senior Operator consult (not full incident unless triggers fire).

### Evidence gathering

Record each refresh:

- Final Operator Brief
- Governance Mode
- Trend / Delta / Forward Forecast strips (if visible)
- Scenario deterioration triggers active
- Any change in Blocked or Watch Condition

### Escalation posture

| Signal | Action |
|--------|--------|
| Transition toward STABLE or LOCKED_OBSERVE_ONLY | Continue observe; log improvement |
| Transition toward GOVERNANCE_REPAIR_REQUIRED | **ESCALATE_AND_CONTAIN** playbook |
| Transition toward LOCKED_HEIGHTENED_MONITORING | **MONITOR_AND_PREPARE_ESCALATION** playbook |
| Execution or reconciliation anomaly | Escalate per Section 3 regardless of brief |

### Observation requirements

- No runtime enablement
- No policy override to “test” the transition
- No broker action based on forecast strips alone

### Reassessment cadence

- **Minimum:** every GCC refresh until brief stabilizes for **3 consecutive cycles**
- **Maximum interval:** 4 hours during active market hours if refresh unavailable — escalate operational Level 2

### Example operator response

> “Posture: TRANSITION_WATCH. OBSERVE_VALIDATE_AND_REASSESS. Regime shift possible; evidence logged cycle 2/3. No escalation unless repair brief on next refresh.”

---

## 5. STABLE_CONTINUE_MONITORING

### Meaning

- Governance stable
- Institutional coherence acceptable
- Minimal stress signals
- Routine operations observability sufficient

### Operator objective

**ROUTINE_MONITORING_ONLY** — standard cadence, minimal operator burden.

### Standard monitoring procedure

1. Confirm Final Operator Brief at scheduled check.
2. Verify Blocked Condition still reflects policy (runtime enablement blocked if lock active).
3. Scan Severity strip for unexpected elevation.
4. Proceed with normal operator duties.

### Routine reassessment

- **Scheduled:** next governance refresh or daily stand-up
- **Ad hoc:** only if external alert (broker, scheduler, preflight failure)

### Logging expectations

- **Daily log entry:** one line — brief + “no material change” acceptable
- **Skip verbose logging** if posture unchanged for 5+ consecutive checks

### No-action guidance

Explicit permission to **take no action**:

- Do not escalate stable posture
- Do not tighten halts without trigger
- Do not request overrides “for efficiency”
- Do not refresh GCC more frequently than needed (avoid over-management)

### Example operator response

> “Posture: STABLE_CONTINUE_MONITORING. ROUTINE_MONITORING_ONLY. No action. Next check at scheduled refresh.”

---

# Section 2 — Operator Decision Tree

## Master tree (5-second path)

```
GCC Operator Decision Brief loaded
              │
              ▼
    Read Final Operator Brief
              │
    ┌─────────┼─────────┬─────────────┬──────────────┐
    ▼         ▼         ▼             ▼              ▼
 LOCKED    LOCKED    REPAIR      TRANSITION      STABLE
 OBSERVE   HEIGHT.   REQUIRED      WATCH        CONTINUE
    │         │         │             │              │
    ▼         ▼         ▼             ▼              ▼
 MAINTAIN   MONITOR   ESCALATE    OBSERVE         ROUTINE
 LOCK &     & PREP    & CONTAIN   VALIDATE        MONITOR
 OBSERVE    ESCALATE              REASSESS        ONLY
    │         │         │             │              │
    └─────────┴─────────┴─────────────┴──────────────┘
              │
              ▼
    Any execution anomaly or integrity signal?
              │
         YES ─┴─ NO
              │      │
              ▼      ▼
         Section 4   Log + reassess
         (Halt)      at next trigger
```

---

## Tree A — Stable posture

```
Brief = STABLE_CONTINUE_MONITORING or LOCKED_OBSERVE_ONLY
              │
              ▼
    Runtime blocked? ──NO──► Escalate Senior Operator (unexpected)
              │
             YES
              │
              ▼
    Watch Condition worsening? ──YES──► Go to Tree B
              │
             NO
              │
              ▼
    MAINTAIN_LOCK_AND_OBSERVE or ROUTINE_MONITORING_ONLY
              │
              ▼
    Reassess at next refresh
```

---

## Tree B — Deteriorating posture

```
Brief = LOCKED_HEIGHTENED_MONITORING
   OR Watch Condition worsened 2 cycles
              │
              ▼
    Material contradiction / trust break?
              │
         YES ─┴─ NO
              │      │
              ▼      ▼
    GOVERNANCE_     MONITOR_AND_
    REPAIR path     PREPARE_ESCALATION
    (Tree D)              │
                          ▼
              Trigger persists 3 cycles?
                          │
                     YES ─┴─ NO
                          │      │
                          ▼      ▼
              Notify Senior    Continue monitor
              Operator
```

---

## Tree C — Transition posture

```
Brief = TRANSITION_WATCH
              │
              ▼
    Log evidence (refresh 1)
              │
              ▼
    Log evidence (refresh 2)
              │
              ▼
    Direction clear?
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 Stabilizing  Worsening  Still ambiguous
    │         │         │
    ▼         ▼         ▼
 Tree A    Tree D    Senior Operator
                       consult
              │
              ▼
    3 stable cycles → return to Tree A
```

---

## Tree D — Repair posture

```
Brief = GOVERNANCE_REPAIR_REQUIRED
              │
              ▼
    ESCALATE_AND_CONTAIN
              │
              ▼
    Soft Halt (if execution window exists)
              │
              ▼
    Notify Risk / Governance Lead
              │
              ▼
    Open incident record (Level 3)
              │
              ▼
    No resume until written approval
```

---

## Tree E — Critical escalation posture

```
ANY of:
  • unauthorized override
  • unexpected fill / execution while halted
  • governance bypass
  • duplicate execution risk
  • broker position unexplained delta
              │
              ▼
    HARD HALT immediately
              │
              ▼
    Level 4 Critical Incident (Step 90)
              │
              ▼
    Operator → Senior → Risk Lead → Sys Admin
              → Governance Committee → Executive
              │
              ▼
    Preserve evidence. No remediation without approval chain.
```

---

# Section 3 — Escalation Playbook

Aligns with [Step 90 — Escalation Chain](./Triton_Governance_Incident_Escalation_Framework.md#section-2--governance-escalation-chain).

## Escalation order

```
Operator
    ↓
Senior Operator
    ↓
Risk / Governance Lead
    ↓
Triton System Administrator
    ↓
Governance Committee
    ↓
Executive Oversight (Critical only)
```

---

## Operator

| Field | Detail |
|-------|--------|
| **Trigger** | Any posture change; Watch Condition breach; operator uncertainty on severity |
| **Responsibility** | Detect, classify, contain (Soft Halt), log, execute Section 1 playbook |
| **Authority limit** | No runtime enablement; no overrides; no Hard Halt lift |
| **Response time** | Immediate (self) |
| **Required evidence** | GCC brief screenshot, UTC timestamp, one-line summary |

---

## Senior Operator

| Field | Detail |
|-------|--------|
| **Trigger** | LOCKED_HEIGHTENED_MONITORING persistence; TRANSITION_WATCH ambiguity; Level 2 operational failure |
| **Responsibility** | Triage, confirm classification, approve Soft Halt lift (Level 2 only) |
| **Authority limit** | Cannot approve Hard Halt lift, overrides, or execution resume (Level 3+) |
| **Response time** | **30 minutes** (trading hours); **4 hours** (off-hours non-critical) |
| **Required evidence** | Operator log, 2+ refresh snapshots, trend note |

---

## Risk / Governance Lead

| Field | Detail |
|-------|--------|
| **Trigger** | GOVERNANCE_REPAIR_REQUIRED; Level 3 trading risk; reconciliation mismatch |
| **Responsibility** | Trading risk decision, escalation dossier review, execution resume approval |
| **Authority limit** | No unilateral runtime enablement; dual approval for overrides |
| **Response time** | **30 minutes** (Level 3); **15 minutes** (Level 4 notification) |
| **Required evidence** | Incident record, reconciliation output, lifecycle state, dossier summary |

---

## Triton System Administrator

| Field | Detail |
|-------|--------|
| **Trigger** | Pipeline failure; missing artifacts; scheduler down > 4 hours; GCC unavailable |
| **Responsibility** | System recovery, log access, technical RCA |
| **Authority limit** | Change control for code/config; no silent governance JSON edits |
| **Response time** | **4 hours** (Level 2) |
| **Required evidence** | Preflight logs, artifact timestamps, error output |

---

## Governance Committee

| Field | Detail |
|-------|--------|
| **Trigger** | Constitutional violation; unauthorized override; Level 4; Hard Halt lift with integrity event |
| **Responsibility** | Formal review, policy exception, autonomy boundary decisions |
| **Authority limit** | Documented votes; extraordinary actions logged |
| **Response time** | **24 hours** convene target; **immediate** for active Level 4 |
| **Required evidence** | Full incident package (Step 90 template) |

---

## Executive Oversight

| Field | Detail |
|-------|--------|
| **Trigger** | Level 4 Critical only; capital at risk; regulatory or reputational exposure |
| **Responsibility** | Hard Halt lift ratification; exceptional override approval |
| **Authority limit** | Extraordinary; fully audited |
| **Response time** | **15 minutes** notification; **1 hour** decision target for active critical |
| **Required evidence** | Executive summary, containment status, recommended actions |

---

# Section 4 — Halt Decision Playbook

**Capital Preservation Doctrine:** When halt criteria are met, halt first, investigate second.

## Decision summary

| Posture | Default halt state | Upgrade to Soft | Upgrade to Hard |
|---------|-------------------|-----------------|-----------------|
| LOCKED_OBSERVE_ONLY | None (lock sufficient) | Stale-data + run window | Integrity event |
| LOCKED_HEIGHTENED_MONITORING | None | Trigger + run window | Integrity event |
| GOVERNANCE_REPAIR_REQUIRED | Soft Halt | Already recommended | Integrity event |
| TRANSITION_WATCH | None | Reconciliation uncertainty | Integrity event |
| STABLE_CONTINUE_MONITORING | None | Stale-data + run window | Integrity event |

---

## Continue monitoring

**When:**

- LOCKED_OBSERVE_ONLY or STABLE_CONTINUE_MONITORING
- No reconciliation mismatch
- No execution anomaly
- Runtime blocked as expected

**Authority:** Operator discretion
**Action:** Log and reassess at next refresh
**Restart requirements:** N/A

---

## Soft halt

**Meaning:** Execution paused; observability continues; investigation permitted.

### Trigger conditions

- Lifecycle inconsistency under review
- Stale-data concern before scheduled run
- Temporary governance mismatch (Level 3 unconfirmed)
- GOVERNANCE_REPAIR_REQUIRED with imminent execution window
- LOCKED_HEIGHTENED_MONITORING + reconciliation uncertainty
- Scheduler delay with run pending

### Required authority

| Action | Authority |
|--------|-----------|
| **Initiate** | Operator or Senior Operator |
| **Lift** | Senior Operator (Level 2) or Risk / Governance Lead (Level 3) |

### Restart requirements

1. Root cause documented or downgraded
2. GCC confirms posture stable or improved
3. Reconciliation pass or documented exception
4. Written sign-off in operator / incident log

### Escalation path

Operator → Senior Operator → Risk / Governance Lead (if not resolved in 4 hours)

### Evidence expectations

- Halt timestamp UTC
- Trigger reason (one sentence)
- GCC brief at halt and at lift
- Reconciliation result if applicable

---

## Hard halt

**Meaning:** Full trading stop; execution disabled; emergency governance posture.

### Trigger conditions

- Execution integrity compromise
- Unauthorized override
- Governance bypass
- Duplicate execution risk
- Corrupted idempotency or execution state
- Broker inconsistency with unexplained delta
- Risk-control failure

### Required authority

| Action | Authority |
|--------|-----------|
| **Initiate** | Any operator (mandatory); notify chain immediately |
| **Lift** | Governance Committee + Executive Oversight (Level 4) |

### Restart requirements

1. Full Step 90 Phase 6 validation checklist
2. Independent review (Risk Lead + Committee)
3. Post-incident review scheduled within 2 business days
4. No override without dual approval + expiration

### Escalation path

Full Level 4 chain — Section 3, Tree E

### Evidence expectations

- Complete forensic package: logs, trade IDs, session IDs, broker statements
- Incident record complete before lift consideration

---

# Section 5 — Operator Do / Don't Guide

## Quick reference

| DO | DON'T |
|----|-------|
| Read Operator Decision Brief first | Act on cockpit strips alone without brief |
| **Observe** governance signals at scheduled cadence | **Bypass** safeguards or constitutional lock |
| **Log** anomalies with UTC timestamp | **Enable runtime** without authorization chain |
| **Escalate** per Section 3 when triggers fire | **Ignore** contradictions or reconciliation mismatches |
| **Reassess** on schedule and on trigger | **Escalate emotionally** or for stable locked posture |
| **Preserve** constitutional lock by default | **Override** policy casually or without dual approval |
| **Contain** (Soft Halt) when Step 4 triggers | **Modify** governance JSON or memory artifacts |
| **Preserve evidence** before remediation | **Trade** or broker-adjust to “fix” governance anxiety |
| **Use** Step 90 incident template for Level 2+ | **Assume** STABLE means execution authorized |
| **Wait** through TRANSITION_WATCH ambiguity | **Refresh** GCC obsessively (over-management) |
| **Notify** Senior Operator when uncertain | **Self-approve** halt lift or override |
| **Confirm** Blocked Condition every session | **Treat** LOCKED_OBSERVE_ONLY as failure state |

## Posture → primary DO

| Brief | Primary DO |
|-------|------------|
| LOCKED_OBSERVE_ONLY | Maintain lock; observe |
| LOCKED_HEIGHTENED_MONITORING | Monitor triggers; prepare escalation |
| GOVERNANCE_REPAIR_REQUIRED | Escalate and contain |
| TRANSITION_WATCH | Observe; validate; reassess |
| STABLE_CONTINUE_MONITORING | Routine monitoring only |

## Posture → primary DON'T

| Brief | Primary DON'T |
|-------|---------------|
| LOCKED_OBSERVE_ONLY | Enable runtime or relax lock |
| LOCKED_HEIGHTENED_MONITORING | Panic-escalate on single refresh |
| GOVERNANCE_REPAIR_REQUIRED | Attempt self-repair via overrides |
| TRANSITION_WATCH | Force early execution or override |
| STABLE_CONTINUE_MONITORING | Over-monitor or unnecessary escalation |

---

# Section 6 — Weekend / Closed-Market Governance Rules

## Principle

> **Observation discipline over intervention**

Markets closed does not mean governance is inactive. Artifacts age, schedulers run, and posture can shift. Operators monitor; they do not “manage” a locked system into action.

---

## Market closed / weekend freeze

| Rule | Operator behavior |
|------|-------------------|
| Constitutional lock active | **MAINTAIN_LOCK_AND_OBSERVE** — no change required |
| GCC stale vs market closed | Expected — log as Level 1 unless execution window exists |
| No scheduled execution | No Soft Halt needed solely for weekend |
| Posture change on stale refresh | Log; reassess at first post-open refresh before escalating |

---

## Stale quotes / stale data

- Stale quotes during closed market are **normal**
- Do not infer trading risk from quote staleness alone
- Escalate only if: stale **processed/signal** data intersects a **scheduled execution window**
- Action: Soft Halt + Senior Operator if run is imminent

---

## Scheduler interruption

- Weekend scheduler pause = Level 2 if artifacts miss SLA
- Confirm execution blocked before troubleshooting
- Notify System Administrator if outage > 4 hours
- Do not manually trigger execution to “catch up”

---

## Blocked execution

- Blocked execution during closed market is **expected**
- Do not treat as incident unless unauthorized fill reported
- Document: “Execution blocked — market closed — no action”

---

## Paper-mode monitoring

- Paper mode does not reduce governance discipline
- Same posture playbooks apply
- Overrides and runtime rules unchanged
- Log paper-specific anomalies under Level 1–2 unless integrity event

---

## Weekend escalation matrix

| Condition | Action |
|-----------|--------|
| Stable LOCKED_OBSERVE_ONLY | None; log at Monday open |
| HEIGHTENED persists through weekend | Email Senior Operator; no overnight paging unless trigger worsens |
| REPAIR_REQUIRED | Risk / Governance Lead notified; Soft Halt if run scheduled |
| Level 4 integrity event | Full chain immediately regardless of market hours |

---

# Section 7 — Quick Reference Operator Cards

*Print or pin — target comprehension: **under 10 seconds***

---

### Card 1 — LOCKED_OBSERVE_ONLY

```
SITUATION   Constitutional lock; stable; runtime blocked
DO          MAINTAIN_LOCK_AND_OBSERVE — log, refresh next cadence
DON'T       Enable runtime; override; panic-escalate
ESCALATE?   Only if Watch worsens 2 cycles or brief changes
REASSESS    Next scheduled GCC refresh
```

---

### Card 2 — LOCKED_HEIGHTENED_MONITORING

```
SITUATION   Locked; stress up; deterioration possible
DO          MONITOR_AND_PREPARE_ESCALATION — log each refresh
DON'T       Break lock; treat as emergency; ignore triggers
ESCALATE?   Senior Operator if trigger persists 3 cycles
REASSESS    Every refresh until stable or upgrade
```

---

### Card 3 — GOVERNANCE_REPAIR_REQUIRED

```
SITUATION   Integrity degrading; repair needed
DO          ESCALATE_AND_CONTAIN — Soft Halt, notify Risk Lead
DON'T       Self-repair; runtime enable; override
ESCALATE?   YES — Risk / Governance Lead within 30 min
REASSESS    After governance review approval only
```

---

### Card 4 — TRANSITION_WATCH

```
SITUATION   Regime shifting; outcome unclear
DO          OBSERVE_VALIDATE_AND_REASSESS — 2+ cycle evidence
DON'T       Overreact; force trades; override to test
ESCALATE?   Only if direction → REPAIR or anomaly appears
REASSESS    Every refresh; 3 stable cycles to stand down
```

---

### Card 5 — STABLE_CONTINUE_MONITORING

```
SITUATION   Governance stable; minimal stress
DO          ROUTINE_MONITORING_ONLY — daily log OK
DON'T       Over-monitor; unnecessary escalation
ESCALATE?   No — unless external alert or brief change
REASSESS    Next scheduled check
```

---

### Card 6 — CRITICAL (any brief)

```
SITUATION   Integrity compromise; unauthorized execution
DO          HARD HALT — preserve evidence — Level 4 chain
DON'T       Remediate before containment; hide incident
ESCALATE?   YES — immediate — Executive for Level 4
REASSESS    After Committee + Executive approval only
```

---

# Appendix — GCC field mapping

| GCC Decision Brief field | Playbook use |
|--------------------------|--------------|
| Final Operator Brief | Section 1 posture playbook |
| Immediate Instruction | Primary operator command |
| Governance Mode | Context (lock / repair / transition / observation) |
| Watch Condition | Reassessment trigger |
| Blocked Condition | Primary DON'T |
| Decision memo | Narrative confirmation — read once, then playbook |

---

# Document control

| Field | Value |
|-------|-------|
| Owner | Risk / Governance Lead |
| Review cycle | Quarterly; after any Level 3+ incident |
| Distribution | Operator Manual, Developer Manual, Training, Audit |
| Related docs | Step 90 Incident & Escalation Framework |
| Change process | Governance Committee for material revisions |

---

*This playbook is documentation only. It does not modify Triton runtime behavior, governance engines, execution logic, or broker integration.*
