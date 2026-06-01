# Triton Governance Roles, Authority Matrix & Approval Hierarchy

**Document type:** Governance Manual — Roles, Authority & Approval
**System:** Triton Institutional Trading Platform
**Classification:** Internal — Operator / Developer / Audit / Executive
**Version:** 1.0
**Status:** Manual-ready SOP
**Companion documents:**
- [Triton Governance Incident & Escalation Framework](./Triton_Governance_Incident_Escalation_Framework.md) (Step 90)
- [Triton Governance Operator Decision Playbook](./Triton_Governance_Operator_Decision_Playbook.md) (Step 91)
- [Triton Governance Metrics, KPI & Institutional Health Framework](./Triton_Governance_Metrics_KPI_Framework.md) (Step 92)

---

## Purpose

This framework answers:

> **Who is allowed to do what in Triton governance?**

It formalizes:

- governance role clarity
- approval hierarchy discipline
- authority boundaries
- segregation of duties
- override authorization control
- institutional accountability
- audit-grade governance permissions

This document is **procedural and definitional**. It does not implement RBAC, alter runtime permissions, enable execution, mutate governance engines, or modify broker integration. Human actions remain subject to GCC posture, constitutional lock, and Capital Preservation Doctrine.

**Capital Preservation Doctrine:** When authority is unclear, **default to containment, observation, and escalation**. No role may infer permission from convenience or urgency.

---

## Scope

**Applies to:**

- All governance, operator, and oversight actions related to GCC, incidents, halts, overrides, and reviews
- Escalation chain participants (Operator through Executive Oversight)
- Audit and compliance attestation of who approved what

**Does not:**

- grant technical system permissions by itself
- replace legal, regulatory, or broker agreements
- authorize runtime enablement without explicit approved path and documentation

---

# Card 1 — Role Definitions

Institutional hierarchy (escalation order):

```
Operator
  → Senior Operator
    → Risk / Governance Lead
      → Triton System Administrator
        → Governance Committee
          → Executive Oversight (Founder / Executive Authority)
```

---

## Operator

### Purpose

Primary governance observer and first responder. Converts GCC posture into disciplined action per Step 91 without improvising policy.

### Responsibilities

- Monitor GCC Operator Decision Brief at scheduled cadence
- Classify incidents (initial severity per Step 90)
- Initiate **Soft Halt** when triggers met
- Initiate **Hard Halt** when mandatory (notify chain immediately)
- Log all actions with UTC timestamp and evidence pointers
- Execute posture playbooks (observe, contain, escalate)
- Preserve constitutional lock and evidence before remediation
- Escalate when uncertain, when Watch conditions persist, or when KPI/risk flags fire (Step 92)

### Authority boundaries

| Authorized | Not authorized |
|------------|----------------|
| Observe, log, classify, contain (Soft Halt) | Runtime enablement |
| Request escalation | Override approval |
| Recommend containment upgrades | Hard Halt lift |
| Initiate Hard Halt (mandatory when triggered) | Governance JSON / memory mutation |
| Close Level 1 with documentation | Broker discretionary action outside SOP |
| | Self-approve halt lift or override |
| | Incident closure Level 2+ without sign-off |

### Prohibited actions

- Enabling runtime or relaxing execution blocks without authorization chain
- Approving or executing overrides
- Lifting Hard Halt
- Modifying governance artifacts to “fix” observability
- Self-approving any halt lift, override, or incident closure above Level 1
- Acting on cockpit strips without Operator Decision Brief confirmation

### Escalation expectations

- **To Senior Operator:** uncertainty on severity; recurring Level 1; LOCKED_HEIGHTENED_MONITORING persistence; TRANSITION_WATCH ambiguity
- **To Risk / Governance Lead:** GOVERNANCE_REPAIR_REQUIRED; Level 3 signals; reconciliation mismatch
- **Immediate Critical path:** execution while halted; unauthorized override; integrity compromise

### Approval scope

- **Level 1** incident closure (operator discretion, documented)
- **Soft Halt initiate** (shared with Senior Operator)
- **Hard Halt initiate** (any operator — mandatory notify, not lift)
- All other approvals: **escalate**

---

## Senior Operator

### Purpose

Second-line operator governance: triage, classification confirmation, and limited containment authority for operational (Level 2) events.

### Responsibilities

- Triage operator escalations within SLA
- Confirm or revise incident severity
- Coordinate initial RCA for Level 2
- Approve **Soft Halt lift** for Level 2 only
- Sign off Level 2 incident closure
- Supervise operator compliance sampling
- Prepare escalation packages for Risk / Governance Lead

### Authority boundaries

| Authorized | Not authorized |
|------------|----------------|
| Soft Halt initiate and lift (Level 2) | Hard Halt lift |
| Level 2 incident closure sign-off | Override approval (any type) |
| Operational remediation coordination (no execution change) | Runtime enablement |
| Recommend Hard Halt upgrade | Constitutional lock relaxation |
| | Level 3+ execution resume |
| | Governance policy changes |
| | Self-approve overrides or Hard Halt lift |

### Prohibited actions

- Approving Hard Halt lift or overrides
- Unilateral execution resume for Level 3+
- Silent edits to governance JSON or memory
- Closing Level 3+ incidents without Risk / Governance Lead
- Self-approval as sole authority on dual-approval actions

### Escalation expectations

- **To Risk / Governance Lead:** Level 3 trading risk; reconciliation mismatch; GOVERNANCE_REPAIR_REQUIRED not resolving in 4 hours
- **To System Administrator:** pipeline/dashboard down > 4 hours
- **To Governance Committee:** constitutional violation; unauthorized override

### Approval scope

- **Soft Halt lift** — Level 2, with documented validation
- **Level 2** incident closure and operational remediation
- **Recommend** Soft → Hard upgrade, overrides, runtime paths (no approval)

---

## Risk / Governance Lead

### Purpose

Institutional trading-risk and governance authority for Level 3–4 events, escalation dossiers, execution resume (where authorized), and override dual-approval participant.

### Responsibilities

- Own Level 3–4 incidents from classification through validation
- Review human escalation dossiers and ARM governance outputs
- Approve execution resume after Level 3 validation checklist
- Participate in **dual approval** for overrides (never sole approver)
- Approve **Soft Halt lift** for Level 3
- Authorize documented remediation with rollback plan
- Own KPI official scoring and governance metrics (Step 92)
- Convene Governance Committee when triggers require

### Authority boundaries

| Authorized | Not authorized |
|------------|----------------|
| Level 3 execution resume (documented) | Unilateral runtime enablement |
| Soft Halt lift (Level 3) | Sole override approval |
| Dual-approval participant for overrides | Hard Halt lift alone |
| Trading-risk containment decisions | Governance policy change without Committee |
| Escalation dossier acceptance/rejection | Self-approval on dual-approval actions |

### Prohibited actions

- Enabling runtime without documented approval chain and GCC confirmation
- Single-signer overrides
- Lifting Hard Halt without Committee (and Executive when required)
- Bypassing post-incident review
- Approving incident closure Level 4 without Committee + Executive path

### Escalation expectations

- **To Governance Committee:** constitutional violation; autonomy boundary; Hard Halt lift; policy exception; unauthorized override
- **To Executive Oversight:** Level 4; capital at risk; Hard Halt active; override ratification

### Approval scope

- **Level 3** remediation, execution resume, Soft Halt lift (Level 3)
- **Dual approval** — overrides (with second role per matrix)
- **Recommend** Committee/Executive actions
- **Hard Halt lift** — **not** without Committee + Executive per severity

---

## Triton System Administrator

### Purpose

Technical recovery of pipelines, schedulers, GCC availability, and logs—under change control—without governance policy authority.

### Responsibilities

- Restore pipeline outputs, schedulers, dashboards
- Provide logs and technical RCA
- Execute approved configuration/code changes per change control
- Support forensic evidence collection
- Document artifact timestamps and preflight results

### Authority boundaries

| Authorized | Not authorized |
|------------|----------------|
| System recovery (Level 2 technical) | Governance policy decisions |
| Log access and export | Override approval |
| Change control execution (approved) | Silent governance JSON edits |
| Recommend operational fixes | Halt lift or execution resume |
| | Runtime enablement for trading |

### Prohibited actions

- Modifying governance memory or JSON without explicit change authorization
- Using technical access to bypass halts or locks
- Self-approving trading or governance risk decisions
- Closing incidents without governance sign-off

### Escalation expectations

- **To Risk / Governance Lead:** recovery actions may affect trading windows or reconciliation
- **To Senior Operator:** operational status for operator visibility

### Approval scope

- **Technical remediation** only, with Senior Operator (L2) or Risk Lead (L3+) awareness
- **No** halt, override, restart, or incident closure authority

---

## Governance Committee

### Purpose

Formal institutional body for constitutional incidents, policy exceptions, autonomy boundaries, Hard Halt lift (with Executive when required), and extraordinary governance decisions.

### Responsibilities

- Review Level 4 and constitutional events
- Vote on policy exceptions and governance repair authorization
- Ratify Hard Halt lift (minimum participant with Risk Lead)
- Review override dependency and safeguard violations
- Approve material governance manual revisions
- Document votes and dissent

### Authority boundaries

| Authorized | Not authorized |
|------------|----------------|
| Policy exceptions (documented vote) | Routine Level 1–2 operator actions |
| Hard Halt lift (with Executive when required) | Unilateral runtime enablement without Executive when required |
| Constitutional / autonomy decisions | Informal or undocumented overrides |
| Governance repair authorization | Bypassing audit requirements |
| Override (governance/constitutional path) with Executive | Single-member Committee action without quorum |

### Prohibited actions

- Standing informal exceptions without documentation
- Ratifying actions without incident package
- Self-approval by a single member claiming Committee role

### Escalation expectations

- **To Executive Oversight:** Level 4; capital at risk; regulatory/reputational exposure; override ratification

### Approval scope

- **Hard Halt lift** (with Risk Lead + Executive when Step 90 requires)
- **Governance / constitutional overrides** (with Executive)
- **Governance policy changes** (material)
- **Level 4** remediation and closure (with Executive)

---

## Executive Oversight (Founder / Executive Authority)

### Purpose

Extraordinary institutional authority for Level 4 Critical events, Hard Halt lift ratification, and exceptional override approval—fully audited.

### Responsibilities

- Decide on active Level 4 containment and communication
- Ratify Hard Halt lift when Committee + Risk Lead have validated
- Approve exceptional overrides (with Committee for constitutional paths)
- Resource remediation and regulatory interface when required
- Receive executive governance scorecard (Step 92)

### Authority boundaries

| Authorized | Not authorized |
|------------|----------------|
| Level 4 decisions | Routine operator monitoring |
| Hard Halt lift ratification (with Committee) | Bypassing dual approval |
| Exceptional override ratification | Undocumented or permanent overrides |
| Institutional attestation | Technical system changes without Admin change control |

### Prohibited actions

- Silent approval without audit record
- Permanent override without Committee ratification
- Direct technical governance JSON edits

### Escalation expectations

- **From:** Risk Lead, Committee (Level 4 only)
- **To:** external counsel/regulatory as required (outside this document)

### Approval scope

- **Hard Halt lift** (with Committee + Risk Lead validation)
- **Level 4** closure and exceptional overrides
- **Executive notification** acknowledgment — not a substitute for operator containment

---

# Card 2 — Authority Matrix

**Legend:**

| Code | Meaning |
|------|---------|
| **A** | Allowed (may execute within role limits) |
| **E** | Escalate (initiate/recommend; higher role approves) |
| **R** | Recommend only |
| **Ap** | Approve (sole approver within scope) |
| **J** | Joint / dual approval required |
| **P** | Prohibited |

| Action | Operator | Senior Operator | Risk / Governance Lead | System Admin | Governance Committee | Executive |
|--------|:--------:|:---------------:|:----------------------:|:------------:|:--------------------:|:---------:|
| **Soft Halt — initiate** | A | A | A | P | P | P |
| **Soft Halt — lift (L2)** | E | Ap | E | P | P | P |
| **Soft Halt — lift (L3)** | E | E | Ap | P | P | P |
| **Hard Halt — initiate** | A | A | A | P | P | P |
| **Hard Halt — lift** | P | P | E | P | J | J |
| **Override — request** | E | E | R | P | E | E |
| **Override — approve** | P | P | J | P | J | J |
| **Governance escalation — initiate** | A | A | A | E | E | E |
| **Governance escalation — accept** | P | E | Ap | P | Ap | Ap |
| **Incident closure — L1** | Ap | E | E | P | P | P |
| **Incident closure — L2** | E | Ap | E | P | P | P |
| **Incident closure — L3** | P | E | Ap | P | P | P |
| **Incident closure — L4** | P | P | E | P | J | J |
| **Constitutional lock preservation** | A | A | A | A | A | A |
| **Constitutional lock relaxation** | P | P | E | P | J | J |
| **Runtime enablement — recommend** | R | R | R | P | R | R |
| **Runtime enablement — approve** | P | P | E | P | J | J |
| **Emergency containment — initiate** | A | A | A | E | E | E |
| **Emergency containment — authorize** | P | E | Ap | P | J | J |
| **KPI review — operational input** | A | A | A | P | E | E |
| **KPI review — official sign-off** | P | E | Ap | P | Ap | Ap |
| **Post-incident review — participate** | A | A | A | A | A | A |
| **Post-incident review — sign-off L3+** | P | E | Ap | P | J | J |

**Notes:**

- **J** requires two distinct roles; no self-approval.
- Hard Halt lift always **J** (Committee + Executive minimum; Risk Lead validates).
- Overrides always **J** per override type (Step 90 Section 5).
- Runtime enablement while GCC shows constitutional lock requires full chain per Step 91.

---

# Card 3 — Approval Hierarchy

## Approval types

| Type | Definition | When used |
|------|------------|-----------|
| **Single approval** | One authorized role documents and signs | Level 1 closure; Soft Halt lift L2 |
| **Dual approval** | Two distinct roles; no self-approval | Overrides; many Level 3 resumes |
| **Executive approval** | Executive + Committee and/or Risk Lead per matrix | Level 4; Hard Halt lift; constitutional override |
| **Emergency exception process** | Containment first; retroactive full approval within SLA | Active capital risk; documented immediately |

---

## Soft Halt

| Step | Authority |
|------|-----------|
| Initiate | Operator **or** Senior Operator **or** Risk Lead |
| Lift (Level 2) | **Single:** Senior Operator |
| Lift (Level 3) | **Single:** Risk / Governance Lead |
| Escalate to Hard | Operator → Senior Operator → Risk Lead |

**Required before lift:** root cause documented or downgraded; GCC stable; reconciliation pass or exception; written sign-off.

---

## Hard Halt

| Step | Authority |
|------|-----------|
| Initiate | **Any operator** (mandatory); all roles may initiate when triggered |
| Notify | Risk Lead **15 min**; Executive **15 min** (Level 4); Committee **immediate** active L4 |
| Lift | **Dual + Executive:** Governance Committee **+** Executive Oversight; Risk Lead validation mandatory |
| Override during Hard Halt | **Prohibited** without Committee + Executive |

**Required before lift:** Step 90 Phase 6 checklist complete; forensic package; post-incident review scheduled.

---

## Restart (execution resume)

| Context | Minimum approval |
|---------|------------------|
| After Soft Halt (L2) | Senior Operator |
| After Soft Halt (L3) | Risk / Governance Lead |
| After Hard Halt | Committee + Executive + Risk Lead validation |
| After override expiration | Original approvers or higher; post-override review complete |

**No restart** without GCC Blocked Condition verified and documented.

---

## Override

| Control | Minimum approvers |
|---------|-------------------|
| Execution block / Soft Halt temporary resume | Risk Lead + Senior Operator |
| Stale-data gate windowed exception | Risk Lead + System Administrator |
| Lifecycle freeze ticker exception | Risk Lead + Governance Committee |
| Governance / constitutional lock path | Governance Committee + Executive |
| Hard Halt any lift | Governance Committee + Executive |

**All overrides:** documented justification, dual approval, audit log, expiration, rollback plan, post-override review.

---

## Emergency containment

| Step | Authority |
|------|-----------|
| Invoke (observe/soft/hard per threat) | Operator initiate; Senior Operator / Risk Lead authorize upgrade |
| Full stop | Hard Halt — any operator initiate; chain notify immediately |
| Retroactive review | Risk Lead **24h**; Committee if constitutional; Executive if L4 |

---

## Governance repair

| Step | Authority |
|------|-----------|
| Detect / contain | Operator per GOVERNANCE_REPAIR_REQUIRED playbook |
| Authorize repair path | Risk / Governance Lead |
| Policy or constitutional repair | Governance Committee |
| Runtime-affecting repair | Committee + Executive |

**No** repair via unauthorized JSON mutation.

---

## Incident closure

| Level | Approver |
|-------|----------|
| 1 | Operator |
| 2 | Senior Operator |
| 3 | Risk / Governance Lead |
| 4 | Governance Committee + Executive |

---

## Governance policy changes

| Materiality | Approver |
|-------------|----------|
| Editorial / clarifying | Governance Lead + Committee acknowledgment |
| Material authority or safeguard change | Governance Committee vote |
| Constitutional / runtime posture policy | Committee + Executive |

---

# Card 4 — Segregation of Duties

## Principles

1. **No self-approval** on halts lift, overrides, execution resume, or Level 2+ closure.
2. **No single-person governance** for Critical actions (Hard Halt lift, constitutional override, Level 4 closure).
3. **Technical administration ≠ governance approval** — Admin recovers systems; Risk Lead owns trading risk.
4. **Executive ≠ operator** — Executive ratifies; operators contain and document.

## Who cannot approve themselves

| Action | Rule |
|--------|------|
| Override | Requestor cannot be approver; two distinct roles |
| Soft Halt lift | Initiator cannot be sole approver at L3; second review if same person on shift |
| Hard Halt lift | Initiator cannot sign as Committee or Executive |
| Incident closure L3+ | RCA author cannot be sole approver |
| KPI official sign-off | Scorer cannot be sole Executive attestation |

## Who cannot bypass review

| Role | Cannot bypass |
|------|----------------|
| Operator | Senior Operator / Risk Lead for L2+ closure and overrides |
| Senior Operator | Risk Lead for L3+ and overrides |
| Risk Lead | Committee for constitutional / Hard Halt lift |
| System Administrator | All governance approvals |
| Any role | Post-incident review (Step 90 Section 7) |

## What always requires dual approval

- Any **override** (all types)
- **Hard Halt lift** (Committee + Executive = institutional dual layer; Risk Lead validates)
- **Governance / constitutional lock** relaxation
- **Runtime enablement** when lock active
- **Level 4** incident closure

## Conflict of interest

- Approvers must disclose same-day trading or position interest in affected tickers; recuse and substitute approver.
- Developers who deployed affected code cannot be sole approver for related incident closure — second governance reviewer required.

---

# Card 5 — Escalation Authority Rules

Mandatory escalations. **Who escalates** = detecting role. **To whom** = minimum target.

| Trigger | Who escalates | To whom | Required evidence | SLA |
|---------|---------------|---------|---------------------|-----|
| **Hard Halt invoked** | Initiating operator | Risk Lead + Executive + Committee | Halt UTC, trigger reason, GCC brief, incident ID | Risk Lead **15 min**; Executive **15 min** (L4) |
| **Governance bypass suspected** | Any operator | Governance Committee (Critical) | Logs, artifact paths, brief delta | **Immediate** |
| **Duplicate execution risk** | Operator | Risk Lead → Critical if confirmed | Session IDs, idempotency state, trade IDs | **30 min** to Risk Lead |
| **Override request** | Requesting role | Risk Lead + second approver per type | Justification memo, scope, expiration | Review **4h**; no standing override |
| **GHS CRITICAL or DEGRADED** | Operator / Governance Lead | Risk Lead; Executive if CRITICAL | Step 92 scorecard, KPI flags | Risk Lead **4h**; Executive same day if CRITICAL |
| **Constitutional safeguard weakening** | Any operator | Committee + Executive | CLPR breach log, override audit | **Immediate** |
| **Unauthorized override detected** | Any operator | Committee (Critical) | Audit log, approver fields | **Immediate** |
| **Reconciliation unexplained delta** | Operator | Risk Lead → Critical if unresolved | `lifecycle_reconciliation.csv`, broker statement | **30 min** |
| **Execution while halted** | Any operator | Critical — full chain | Halt record, execution log | **Immediate** |
| **Operator uncertainty on severity** | Operator | Senior Operator | GCC snapshots, one-line summary | **30 min** trading hours |
| **Pipeline down > 4h** | Operator | System Administrator | Preflight logs, artifact timestamps | **4h** |
| **Material contradiction > 4h** | Operator | Risk Lead | Lifecycle, rationale, signal, brief | **4h** |
| **Level 3 trading risk** | Operator | Risk Lead | Incident record, dossier | **30 min** |
| **Level 4 Critical** | Operator | Full chain to Executive | Full Step 90 package | **15 min** Executive notify |

---

# Card 6 — Emergency Authority Rules

Containment-first. Emergency power **stops harm**; it does not grant permanent privilege.

## Emergency Soft Halt

| Field | Rule |
|-------|------|
| **Who may invoke** | Operator, Senior Operator, Risk Lead |
| **Documentation required** | UTC timestamp, trigger, GCC brief, incident ID (or provisional ID) |
| **Retroactive review** | Senior Operator within **4h**; Risk Lead if L3 criteria |
| **Expiration window** | Until lift criteria met or upgrade to Hard Halt |
| **Restart authority** | Senior Operator (L2) or Risk Lead (L3) per Card 3 |

## Emergency Hard Halt

| Field | Rule |
|-------|------|
| **Who may invoke** | **Any operator** (mandatory when triggers met); Senior Operator / Risk Lead may invoke directly |
| **Documentation required** | Immediate incident record start; forensic preservation; full chain notification |
| **Retroactive review** | Committee within **24h**; Executive acknowledgment same day |
| **Expiration window** | Until formal lift — **no** timed auto-lift |
| **Restart authority** | Committee + Executive + Risk Lead validation (Card 3) |

## Temporary governance containment

| Field | Rule |
|-------|------|
| **Who may invoke** | Operator (observe/soft); Risk Lead (authorize upgrade); Committee (constitutional containment) |
| **Documentation required** | Containment type, systems affected, tickers, rollback expectation |
| **Retroactive review** | Risk Lead **24h**; Committee if constitutional |
| **Expiration window** | Match override expiration if applicable; max **72h** without Committee extension |
| **Restart authority** | Per containment type — never operator alone for runtime paths |

**Emergency exception process:** If immediate action prevents capital loss, act to contain first, notify chain within **15 minutes**, complete dual approval and audit fields within **4 hours** or downgrade to observation-only until approved.

---

# Card 7 — Governance Approval Protocols

## Override request

```
Operator detects need / receives request
        ↓
Log issue + provisional incident ID (UTC, GCC state, tickers)
        ↓
Escalate to Risk / Governance Lead — no override until reviewed
        ↓
Governance Lead: risk assessment + scope + expiration draft
        ↓
Dual approval (second role per Step 90 matrix)
        ↓
Temporary authorization — audit log entry (approvers, scope, expiry)
        ↓
Operator executes ONLY within approved scope
        ↓
Automatic revert at expiration where feasible
        ↓
Post-override review (24h L3+ / 72h L2)
        ↓
Incident closure or linkage to parent incident
```

## Hard Halt

```
Trigger detected (any operator)
        ↓
HARD HALT — stop execution paths; preserve evidence
        ↓
Notify: Risk Lead (15m) → Executive (15m L4) → Committee (active L4)
        ↓
Incident record + forensic package (Step 90)
        ↓
RCA — no governance JSON edits unauthorized
        ↓
Phase 6 validation checklist
        ↓
Risk Lead validation memo
        ↓
Committee review + vote
        ↓
Executive ratification
        ↓
Controlled restart + post-incident review (2 business days L4)
```

## Restart approval

```
Halt active — confirm Blocked Condition
        ↓
Root cause documented / downgraded
        ↓
Reconciliation pass or documented exception
        ↓
GCC confirms stable or improved posture
        ↓
Approver per level (Senior Operator L2 / Risk Lead L3 / Committee+Executive Hard)
        ↓
Written sign-off in incident record
        ↓
Monitor next 2 cycles — log any regression
```

## Governance repair

```
GCC: GOVERNANCE_REPAIR_REQUIRED
        ↓
Operator: ESCALATE_AND_CONTAIN (Soft Halt if window)
        ↓
Risk Lead: owns dossier + repair plan
        ↓
No runtime toggles / unauthorized JSON edits
        ↓
Committee if constitutional / policy exception required
        ↓
Validate coherence (lifecycle, rationale, signal, brief)
        ↓
Downgrade posture only after documented validation
```

## Incident closure

```
Incident open — severity assigned
        ↓
Containment → RCA → remediation (approvals per level)
        ↓
Phase 6 validation checklist complete
        ↓
Documentation template complete (Card 8)
        ↓
Approver sign-off per level (Card 3)
        ↓
Post-incident review scheduled (Step 90 Section 7)
        ↓
Archive evidence — notify chain of closure
```

---

# Card 8 — Auditability Requirements

**All authority actions** (halt, lift, override, escalation acceptance, closure, policy exception, runtime recommendation approval) must record:

| Field | Requirement |
|-------|-------------|
| **Timestamp** | UTC, start and end for timed authorizations |
| **Incident ID** | `INC-YYYY-MM-DD-###` or linked parent |
| **Session ID** | Trading / pipeline session if applicable |
| **Approval IDs** | Role + name + UTC for each approver |
| **Rationale** | Business and risk justification (2+ sentences) |
| **Affected systems** | Pipeline, lifecycle, execution, broker, GCC, governance engines |
| **Tickers impacted** | List or `NONE` |
| **Rollback expectations** | Steps to restore prior posture |
| **Reviewer signoff** | Second-party validation for dual-approval actions |

### Audit-grade standards

- **Immutable narrative:** append corrections; do not delete prior entries.
- **Evidence pointers:** file paths, log names, screenshot retention.
- **Expiration:** overrides without expiration are **invalid**.
- **Retention:** per institutional record policy; minimum through post-incident review + 1 year for Level 3+.
- **Sampling:** Audit may request 10% of Level 2+ actions quarterly; 100% for Level 4 and overrides.

### Prohibited audit practices

- Backdating approvals
- Oral-only authorization without log within **1 hour**
- Shared credentials or generic “admin” sign-off without named role

---

# Card 9 — Quick Reference Authority Cards

*Under 10-second comprehension.*

---

**Soft Halt**
Who approves lift: **Senior Operator (L2)** / **Risk Lead (L3)**
Escalation? Yes if not resolved **4h**
Dual approval? No (single approver per level)
Evidence: halt/lift UTC, GCC brief, reconciliation

---

**Hard Halt**
Who approves lift: **Committee + Executive** (+ Risk Lead validation)
Escalation? **Immediate** full chain on invoke
Dual approval? **Yes** (institutional layers)
Evidence: forensic package, Phase 6 checklist, incident record

---

**Override**
Who approves: **Dual** per type (Risk Lead + partner role)
Escalation? Always — no operator self-approve
Dual approval? **Always**
Evidence: justification, scope, expiry, rollback, post-review

---

**Restart**
Who approves: **Per halt level** (see Card 3)
Escalation? Hard Halt → Committee path
Dual approval? Hard Halt **yes**
Evidence: RCA, reconciliation, GCC sign-off

---

**Governance Repair**
Who approves: **Risk Lead**; Committee if constitutional
Escalation? Operator → Risk Lead immediately
Dual approval? If runtime/policy exception
Evidence: dossier, coherence validation, no JSON mutation

---

**Incident Closure**
Who approves: **L1 Op / L2 Sr / L3 Risk / L4 Committee+Exec**
Escalation? Per severity
Dual approval? **L4 yes**
Evidence: Step 90 template complete

---

**Emergency Containment**
Who approves: **Contain first** — Risk Lead **4h** retro; Committee if constitutional
Escalation? **15 min** notify on capital risk
Dual approval? Overrides and Hard lift **yes**
Evidence: provisional ID within **15 min**; full package **4h**

---

## Document control

| Field | Value |
|-------|-------|
| Owner | Governance Lead |
| Review cycle | Quarterly |
| Change authority | Governance Committee |
| Distribution | Operator Manual, Developer Manual, Governance SOPs, Audit/Compliance, Executive oversight |

---

## Verification checklist (Step 93 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Cards / sections created (1–9) | Complete |
| 2 | Role definitions completed (6 roles) | Complete |
| 3 | Authority matrix completed | Complete |
| 4 | Approval hierarchy documented | Complete |
| 5 | Segregation of duties completed | Complete |
| 6 | Emergency authority documented | Complete |
| 7 | Approval protocols documented | Complete |
| 8 | Auditability requirements documented | Complete |
| 9 | Quick-reference authority cards completed | Complete |
| 10 | No runtime, code, UI, broker, or governance mutation | **Confirmed — documentation only** |
| 11 | Enterprise-grade SOP quality | **Confirmed** |

---

*End of document — Triton Governance Roles, Authority Matrix & Approval Hierarchy (Step 93)*
