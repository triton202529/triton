# Triton Developer Governance Handbook & Engineering Operating Guide

**Document type:** Governance Manual — Developer / Engineering Handbook
**System:** Triton Institutional Trading Platform — Governance Command Center (GCC)
**Classification:** Internal — Developer / Engineer / System Administrator
**Version:** 1.0
**Status:** Manual-ready — Engineering SOP
**Navigation hub:** [Governance README (Step 101)](./README.md)
**Companion:** [Operator Handbook (Step 102)](./Triton_Operator_Handbook.md)
**Authority manuals:** [Steps 90–100](./README.md#card-2--governance-manual-master-index) — this handbook **summarizes**; linked steps **govern** on conflict

---

## Card 1 — Developer Handbook Philosophy

### Purpose of the Developer Governance Handbook

This handbook defines how **engineering work respects Triton governance** without redefining it in code. It translates Steps 90–102 into **implementation boundaries**, safe change discipline, anti-patterns, and escalation paths for developers who build or maintain systems adjacent to GCC, pipelines, execution, and observability.

Use this handbook **before** proposing or shipping changes that touch governance assumptions. Use linked Steps for **full authority**.

### Why developers exist inside Triton governance

Engineers build **systems that operators and oversight rely on**. Poor boundaries—silent JSON edits, hidden enablement paths, or weakened audit trails—become **constitutional failures** (Step 100), not mere bugs.

> **Engineers implement systems that respect governance — they do not redefine governance through code.**

### Core principles

| Principle | Engineering meaning |
|-----------|---------------------|
| **Governance before implementation** | Read applicable Step docs before coding policy-adjacent behavior |
| **Capital Preservation Doctrine supremacy** | Default deny on enablement; fail closed |
| **Constitutional safeguards dominate** | Lock, halts, dual approval, evidence are not “UX friction” |
| **Containment-first engineering** | Prefer block/observe over auto-resume |
| **Evidence-first engineering** | Logs, IDs, immutable narratives for human decisions |
| **Escalation-aware development** | If behavior changes who escalates or when—governance review |
| **Engineering discipline over improvisation** | No shadow policy in config without Step 98 |

### What developers are responsible for

- Understanding layer boundaries (Card 2) before changes
- Preserving audit trails, timestamps, and artifact paths operators report (Step 96)
- Technical change control for code/config (with Admin role per Step 93)
- Supporting forensic evidence collection (logs, exports)—not altering governance truth
- Flagging governance uncertainty to **Risk / Governance Lead** before shipping ambiguous behavior
- Participating in governance test scenarios when systems touch escalation, halts, or GCC (Step 95)
- Documenting implementation boundaries in PRs/tickets when policy-adjacent

### What developers are NOT responsible for

- Approving overrides, lifting halts, or enabling runtime for operators
- Editing `data/results/` governance JSON or memory artifacts to “fix” displays without authorized change (Step 98)
- Setting KPI thresholds or maturity gates (Steps 92, 94)—documentation + Committee process
- Replacing operator judgment or GCC Operator Decision Brief
- Implementing RBAC or live policy engines **in this handbook’s scope** (separate authorized projects)
- Broker or execution strategy decisions

---

## Card 2 — Governance vs Engineering Boundaries

**Constitutional-first:** When layers conflict, **governance layer interpretation wins** for human action; **runtime must fail closed** for enablement.

### Governance layer

| Field | Detail |
|-------|--------|
| **Purpose** | Institutional rules: who decides, when to halt, what to document |
| **Engineering responsibility** | Do not encode policy exceptions without approved spec + Step 98 |
| **May do** | Read-only integrations; surface GCC fields; export artifacts for reports |
| **May NOT do** | Auto-approve overrides; auto-lift halts; mutate governance memory silently |
| **Escalation** | Policy ambiguity → Governance Lead |

### Runtime layer

| Field | Detail |
|-------|--------|
| **Purpose** | Application execution state: schedulers, pipelines, guards, preflight |
| **Engineering responsibility** | Change control; fail-safe defaults; no hidden enablement |
| **May do** | Fix bugs, restore pipelines, improve preflight with Lead awareness if trading window |
| **May NOT do** | Bypass constitutional lock because “tests need it” without chain |
| **Escalation** | Trading-window impact → Risk Lead (Step 93 Admin path) |

### Execution layer

| Field | Detail |
|-------|--------|
| **Purpose** | Orders, broker interface, idempotency, reconciliation outputs |
| **Engineering responsibility** | Integrity, duplicate prevention, audit-friendly trade IDs |
| **May do** | Execution code fixes under normal CC |
| **May NOT do** | Trade through known halt; disable idempotency to “unstick” |
| **Escalation** | Unexplained delta → operator incident path (Step 90), not dev solo fix |

### Observability layer

| Field | Detail |
|-------|--------|
| **Purpose** | Metrics, dashboards, GCC display, early-warning inputs (Step 99) |
| **Engineering responsibility** | Accurate timestamps; distinguish leading vs lagging feeds |
| **May do** | Improve freshness, add read-only strips mirroring artifacts |
| **May NOT do** | Hide Critical KPIs; greenwash unhealthy GHS |
| **Escalation** | False stability risk → Governance Lead |

### Reporting layer

| Field | Detail |
|-------|--------|
| **Purpose** | Human reports and audit packs (Step 96)—not auto-trading decisions |
| **Engineering responsibility** | Stable field names; export paths; retention |
| **May do** | Assist export formatting per spec |
| **May NOT do** | Drop approval fields; backdate records |
| **Escalation** | ACR-impacting change → Lead + Committee ack |

### Change layer

| Field | Detail |
|-------|--------|
| **Purpose** | How manuals and policy evolve (Step 98) |
| **Engineering responsibility** | Separate **code CC** from **governance manual CC** |
| **May do** | Implement after governance approval and version effective date |
| **May NOT do** | Ship behavior change same day as undocumented policy shift |
| **Escalation** | Constitutional tier → Committee + Executive |

### What code must never bypass

| Control | Reference |
|---------|-----------|
| Constitutional lock / Blocked Condition | Steps 91, 100 |
| Hard Halt / Soft Halt gates | Steps 90, 91 |
| Dual approval for overrides | Steps 90, 93 |
| Incident evidence before closure | Step 90 |
| Prohibition on silent governance JSON mutation | Steps 90, 98, 100 |

### What requires governance approval (before policy-adjacent code ships)

- New runtime enablement path or default-on execution
- Changing escalation SLAs or approver roles in product behavior
- KPI threshold or GHS formula changes (Step 92 + 98)
- Weakening audit fields or retention
- Auto-override or auto-resume logic

---

## Card 3 — Engineering Safeguards

| Safeguard | Why it matters | Failure risk | Implementation boundary | Escalation |
|-----------|----------------|--------------|-------------------------|------------|
| **Capital Preservation Doctrine** | Capital over convenience | Auto-resume losses | Default block; explicit human gate | Risk Lead |
| **Constitutional safeguards** | Institutional control core | CLPR violation; Critical incident | No silent lock relaxation | Committee + Executive |
| **Escalation pathways** | Right human at right time | ESCALATION_CHAOS | Surface triggers; don’t hide errors | Governance Lead |
| **Halt discipline** | Stop harm first | Hard Halt after preventable run | Honor halt flags in all paths | Operator → L4 chain |
| **Approval chains** | Segregation of duties | Self-approval in software | Software requests; humans approve | Step 93 |
| **Evidence integrity** | Audit defensibility | RCA impossible | Immutable logs; pointer paths | Lead on ACR impact |
| **Auditability** | Regulatory/institutional trust | AUDIT_DISCIPLINE_BREAKDOWN | Required fields in exports | Committee on material gap |
| **Governance observability** | Early warning (Step 99) | FALSE_STABILITY | Expose leading indicators honestly | Governance Lead |

---

## Card 4 — Developer Anti-Patterns

| Anti-pattern | Why dangerous | Failure consequence | Escalation | Recovery |
|--------------|---------------|---------------------|------------|----------|
| **Bypass safeguards** | Direct constitutional risk | Level 4; CLPR breach | Immediate Committee path | Rollback deploy; incident |
| **Weaken escalation logic** | Wrong human or late response | EF Critical | Lead + 98 review | Revert + drill |
| **Silently change governance assumptions** | GOVERNANCE_DRIFT | Invalid operator actions | Committee **10bd** | Versioned manual + bulletin |
| **Bypass evidence requirements** | Empty audit pack | ACR Critical | Lead **4h** | Restore fields + backfill |
| **Remove auditability** | Non-attestable institution | Adverse audit | Executive on pack | Restore retention |
| **Blur runtime/governance separation** | Operators confuse code with policy | Wrong halts / enablement | Lead architecture review | Layer refactor spec |
| **Undocumented governance logic** | Shadow policy in code | Certification fail | 98 emergency if live | Document or remove |
| **Change approval behavior without review** | Matrix violation (Step 93) | Unauthorized override appearance | Committee | Revert + tabletop |
| **“Fix” governance JSON in prod** | Forensic corruption | CONSTITUTIONAL_WEAKENING | Immediate Critical | Restore from backup; incident |
| **Feature flag to skip preflight** | Trading without visibility | Level 3 incident | Risk Lead before ship | Remove flag |

**Containment-first:** If anti-pattern may be live, **assume halt posture** and notify Lead before further deploys.

---

## Card 5 — Governance-Aware Development Workflow

```
Understand governance rule
   ↓
Identify implementation boundary
   ↓
Check Step references
   ↓
Escalate if uncertain
   ↓
Implement safely
   ↓
Validate
   ↓
Document
```

---

### Understand governance rule

| Field | Detail |
|-------|--------|
| **Purpose** | Know institutional intent before design |
| **Developer actions** | Read Step 100 rule + domain Step (90–99); Operator Handbook for UX context |
| **What NOT to do** | Invent policy in ticket comments only |
| **Escalation** | Ambiguous rule → Governance Lead |
| **Evidence** | Link `GOVCHG` or manual version in ticket |

---

### Identify implementation boundary

| Field | Detail |
|-------|--------|
| **Purpose** | Map feature to layer (Card 2) |
| **Developer actions** | Document: governance vs runtime vs execution |
| **What NOT to do** | Mix approval UI with auto-approval backend |
| **Escalation** | Boundary dispute → Lead + Senior Operator input |
| **Evidence** | Boundary diagram in PR |

---

### Check Step references

| Field | Detail |
|-------|--------|
| **Purpose** | Traceability for audit |
| **Developer actions** | PR lists Steps 90–102 sections affected |
| **What NOT to do** | “Misc governance fix” without reference |
| **Escalation** | Missing Step for policy change → stop merge |
| **Evidence** | PR template checklist (Card 9) |

---

### Escalate if uncertain

| Field | Detail |
|-------|--------|
| **Purpose** | Prevent shipping ambiguity |
| **Developer actions** | Ask Lead before merge on policy-adjacent work |
| **What NOT to do** | Ship and hope operators adapt |
| **Escalation** | Lead → Committee if constitutional |
| **Evidence** | Written answer attached to ticket |

---

### Implement safely

| Field | Detail |
|-------|--------|
| **Purpose** | Fail closed; preserve artifacts |
| **Developer actions** | Feature flags default off for enablement; log decisions |
| **What NOT to do** | Hard-code override approval |
| **Escalation** | Blocker on safeguard → do not merge |
| **Evidence** | Code review sign-off |

---

### Validate

| Field | Detail |
|-------|--------|
| **Purpose** | Prove behavior matches governance |
| **Developer actions** | Step 95 scenarios relevant to change; regression tests |
| **What NOT to do** | Only unit test happy path |
| **Escalation** | Critical drill fail → Card 7 Step 95 path |
| **Evidence** | `GOVTEST-*` or test log link |

---

### Document

| Field | Detail |
|-------|--------|
| **Purpose** | Operators and audit can follow |
| **Developer actions** | Release note; operator bulletin if behavior visible; update dev docs |
| **What NOT to do** | Secret deploy |
| **Escalation** | Material visibility → 98 effective date |
| **Evidence** | Changelog + version register if manual changes |

---

## Card 6 — Safe Change Management for Engineers

Mapped to [Step 98](./Triton_Governance_Change_Management_Framework.md). **Code change** and **governance manual change** are linked but not identical.

| Change type | When to escalate | Who approves | Evidence required | Rollback |
|-------------|------------------|--------------|-------------------|----------|
| **KPI / GHS display logic** | Always before merge | Governance Lead proposes; Committee if thresholds | 90d data analysis (Step 92) | Revert metric version |
| **Escalation behavior assumptions** | UI/workflow changes notify chain | Lead + Senior Operator review | Tabletop (Step 95) | Revert + chain memo |
| **Halt behavior dependencies** | Any new path to execution | Risk Lead | Phase 6 checklist impact memo | Feature off; halt test |
| **Reporting field changes** | Add/remove audit fields | Lead + Committee ack if remove | Step 96 field map | Parallel export one cycle |
| **Monitoring / watch inputs** | Leading indicator definition change | Lead | Step 99 domain review | Rollback if false stability |
| **Audit trail / retention** | Any reduction | Committee | Legal/compliance if material | Restore from backup policy |
| **Preflight / pipeline only** | No policy change | System Administrator CC | Logs, timestamps | Standard deploy rollback |
| **GCC read-only display** | Wording mirrors brief | Lead acknowledgment | Screenshot parity test | UI revert |

**Rule:** If operators would need **new training** (Step 97), governance change process (Step 98) likely applies—even if only code changed.

---

## Card 7 — Governance Testing for Engineers

Mapped to [Step 95](./Triton_Governance_Testing_Simulation_Framework.md). Developers **support** drills; **Governance Lead** owns cadence.

| Activity | Purpose | Developer responsibility | Failure signal | Escalation | Evidence |
|----------|---------|-------------------------|----------------|------------|----------|
| **Scenario validation** | Playbooks match product | Provide test env; fixture data | Wrong brief → wrong UI | Lead | Test script log |
| **Governance simulation** | Tabletop realism | No prod mutation tools | Live JSON edit in test | Stop test | `GOVTEST-*` |
| **Escalation testing** | Chain reachable | Mock notifications; log routing | SLA miss in test | Lead | Route diagram |
| **Audit validation** | Exports complete | Sample `INC-*` export | Missing fields | Lead **4h** | Export file |
| **Observability verification** | KPI feeds honest | Stale data labeled | False green | Lead | Timestamp proof |
| **Regression testing** | No safeguard regression | Automated + manual halt path | Enablement when blocked | Block release | CI log |

**Developers do not** declare governance “validated” for maturity/readiness—that is institutional attestation (Steps 94, 96).

---

## Card 8 — Developer Quick Start

*Under 1-minute read.*

| Stage | Document / action |
|-------|-------------------|
| **Read first** | [Step 100](./Triton_Governance_Constitution_Operating_Charter.md) rules 8, 10 + this handbook Card 2 |
| **Read second** | [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) scope + [Step 93](./Triton_Governance_Roles_Authority_Framework.md) Admin row |
| **Daily references** | Card 9 checklist; [Operator Handbook](./Triton_Operator_Handbook.md) (what operators see) |
| **Escalation references** | Card 6 Step 98; Risk Lead for policy-adjacent |
| **Advanced references** | [Step 95](./Triton_Governance_Testing_Simulation_Framework.md), [Step 98](./Triton_Governance_Change_Management_Framework.md), [README](./README.md) |

**First PR mantra:** *Which layer? Which Step? Fail closed? Evidence?*

---

## Card 9 — Engineering Checklist

Use for **policy-adjacent** PRs and releases.

### Design

- [ ] Governance rule identified (Step # cited)
- [ ] Layer boundary documented (Card 2)
- [ ] Approval assumptions verified against Step 93—not implemented in code
- [ ] Anti-patterns reviewed (Card 4)—none introduced

### Build

- [ ] Fail-closed default for enablement / resume paths
- [ ] No silent governance JSON / memory mutation
- [ ] Audit fields preserved or additive only (removal → 98)
- [ ] Escalation implications reviewed (operators notified how?)

### Verify

- [ ] Governance test scenarios run or scheduled (Step 95)
- [ ] Observability impact checked (Step 99)—no false stability
- [ ] Rollback path understood (deploy + feature flag)

### Ship

- [ ] Release note; operator bulletin if UX changes
- [ ] `GOVCHG` linked if manual changed (Step 98)
- [ ] Training delta flagged if operators affected (Step 97)

**If any item blocked:** escalate to Governance Lead—do not merge on assumption.

---

## Card 10 — Quick Reference Engineering Cards

*Under 10-second comprehension.*

| Situation | What to do | Escalate? | Evidence | Step |
|-----------|------------|-----------|----------|------|
| **Governance uncertainty** | Stop; read 100; ask Lead | Yes | Written answer in ticket | 100, 98 |
| **Escalation logic concern** | No auto-approve; surface to human | Lead | PR + 95 tabletop | 90, 93, 95 |
| **KPI logic change** | Manual 98 first | Committee if thresholds | 90d data | 92, 98 |
| **Monitoring logic concern** | Label stale; no greenwash | Lead | Feed timestamps | 99 |
| **Reporting change** | Preserve fields | Committee if remove | Field map | 96, 98 |
| **Approval chain question** | Software requests only | Lead | Matrix cite | 93 |
| **Safeguard concern** | Fail closed; halt path | Immediate if live risk | Incident ID | 90, 100 |
| **Change management question** | `GOVCHG` + version | Per tier | Register entry | 98 |
| **Operator-visible change** | Read 102; bulletin | Lead | Handoff note | 102, 97 |
| **Prod “quick fix” JSON** | **Do not** | Critical path | Preserve state | 90, 100 |

---

## Card 11 — Developer Handbook Appendix

| Term | Engineering definition |
|------|------------------------|
| **Auditability** | Systems produce complete, immutable records for Step 96 |
| **Constitutional safeguard** | Control engineering must not weaken (lock, dual approval, halts) |
| **Containment** | Engineering defaults that block harm (halt, deny enablement) |
| **Evidence integrity** | Logs/exports support human decisions with pointers |
| **Escalation event** | Human chain invoked—software notifies, does not replace |
| **Governance boundary** | Line where code ends and Step manuals govern humans |
| **Governance drift** | Behavior diverges from documented Steps without 98 |
| **Governance layer** | Policy, roles, halts, documentation (Steps 90–100) |
| **Governance mutation** | Unauthorized change to governance artifacts or policy in prod |
| **Halt discipline** | All execution paths respect Soft/Hard halt state |
| **Runtime layer** | Schedulers, guards, pipelines—change control, fail closed |
| **Execution layer** | Trading/broker/idempotency—integrity, no halt bypass |

**Related roles:** Technical changes often flow through **Triton System Administrator** (Step 93)—developers coordinate; do not self-approve trading risk.

**Full glossary:** [Step 100 — Card 10](./Triton_Governance_Constitution_Operating_Charter.md).

---

## Document control

| Field | Value |
|-------|-------|
| Owner | Governance Lead (policy); Engineering lead (custody of checklist) |
| Review cycle | Quarterly or material Step 98 change |
| Change authority | [Step 98](./Triton_Governance_Change_Management_Framework.md) |
| Distribution | Developer Manual, onboarding, PR templates |

---

## Verification checklist (Step 103 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Handbook philosophy completed | Complete |
| 2 | Governance boundaries completed | Complete |
| 3 | Engineering safeguards completed | Complete |
| 4 | Anti-patterns completed | Complete |
| 5 | Development workflow completed | Complete |
| 6 | Safe change management completed | Complete |
| 7 | Governance testing completed | Complete |
| 8 | Quick start completed | Complete |
| 9 | Engineering checklist completed | Complete |
| 10 | Quick-reference cards completed | Complete |
| 11 | Appendix completed | Complete |
| 12 | No runtime, code, UI, broker, or governance mutation | **Confirmed — documentation only** |
| 13 | Enterprise-grade developer usability | **Confirmed** |

---

*End of document — Triton Developer Governance Handbook & Engineering Operating Guide (Step 103). Policy-adjacent work: Card 9 + Step 100 + Step 98.*
