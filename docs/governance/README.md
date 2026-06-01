# Triton Governance Operating Library

**Step 101 — Canonical Navigation Index & Governance README**
**System:** Triton Institutional Trading Platform — Governance Command Center (GCC)
**Classification:** Internal — Operator / Developer / Audit / Executive
**Version:** 1.0
**Status:** Navigation layer — Manual-ready

> **This README is a navigation layer, not a substitute for governance manuals.**
> For authoritative procedure, open the linked Step document. When in doubt: **contain, observe, escalate** (Capital Preservation Doctrine).

---

## Card 1 — Governance README Philosophy

### Why the governance system exists

Triton governance is an **institutional control layer** that protects capital, integrity, and auditability before execution decisions. It converts GCC posture, incidents, metrics, roles, and evidence into **repeatable institutional behavior**—not individual judgment under pressure.

Governance does **not** promise market outcomes. It promises **disciplined observation, escalation, containment, review, recovery, and improvement** (see [Step 100 — Constitution](./Triton_Governance_Constitution_Operating_Charter.md)).

### Core principles (navigation context)

| Principle | Meaning for readers |
|-----------|---------------------|
| **Governance before execution** | Open GCC and Step 91 before any trading action |
| **Evidence-first discipline** | Every claim links to timestamps and artifacts |
| **Constitutional safeguards dominate** | Lock, halts, dual approval override convenience |
| **Escalation before intervention** | Uncertain → escalate; do not edit JSON or enable runtime |
| **Institutional consistency** | Same situation → same manual path |
| **Containment-first governance** | Halt and lock before optimization |
| **Operator usability** | This README routes you in seconds |

### What this README helps people do

- Find the **correct manual** for a situation in under 10 seconds
- Onboard by **role** and **workflow** without reading all 11 documents first
- See **dependencies** between manuals for audits and training
- Maintain the library long-term (index + Step 98 change process)

### What this README does not replace

- **Step documents 90–100** — authoritative procedure and thresholds
- Technical runbooks, broker agreements, legal/regulatory advice
- Runtime permissions, RBAC, or automated policy enforcement
- GCC itself — always read the **Operator Decision Brief** on shift

---

## Card 2 — Governance Manual Master Index

| Step | Document | Purpose | When to use | Primary users | Escalation relevance | Dependencies |
|------|----------|---------|-------------|---------------|----------------------|--------------|
| **90** | [Incident & Escalation](./Triton_Governance_Incident_Escalation_Framework.md) | Severity, workflow, halts, overrides, incident template | Anomaly, incident, halt, override | All roles | **Definitive** — L1–4 SLAs and chain | Foundation for 91, 93, 96 |
| **91** | [Operator Decision Playbook](./Triton_Governance_Operator_Decision_Playbook.md) | GCC brief → immediate action; posture playbooks | **Every operator session** | Operator, Senior Operator | Maps posture to escalate/contain | Requires 90; feeds 96 daily |
| **92** | [Metrics & KPI](./Triton_Governance_Metrics_KPI_Framework.md) | GHS, 15 KPIs, thresholds, risk flags | Weekly+ health review; deterioration | Governance Lead, Executive | Critical KPI → Executive | Feeds 94, 96, 99 |
| **93** | [Roles & Authority](./Triton_Governance_Roles_Authority_Framework.md) | Who approves what; dual approval; matrix | Halt lift, override, closure, resume | All roles | **Definitive** for authority | Implements 90 chain; 95, 97 |
| **94** | [Maturity & Lifecycle](./Triton_Governance_Lifecycle_Maturity_Framework.md) | Maturity levels, gates, readiness, regression | Quarterly attestation; promotion | Lead, Committee, Executive | Regression → Committee | Uses 92; feeds 96 readiness |
| **95** | [Testing & Validation](./Triton_Governance_Testing_Simulation_Framework.md) | Drills, stress, tabletops, test failures | Scheduled; post-incident; pre-promotion | All roles | Critical drill fail → Committee | Validates 90–93 |
| **96** | [Reporting & Audit Packs](./Triton_Governance_Reporting_Audit_Framework.md) | Reports, audit pack, executive comms, evidence | Daily–quarterly; post-incident | Operator → Executive | Never hide Critical from exec line | Packages 90, 92, 94, 95 |
| **97** | [Training & Certification](./Triton_Governance_Training_Certification_Framework.md) | Tracks, certs, drills, continuity | Onboarding; annual re-cert | All roles; Lead owns roster | Critical competency → Committee | Teaches 90–96 |
| **98** | [Change Management](./Triton_Governance_Change_Management_Framework.md) | Versioning, approval, rollback, constitutional evolution | Any manual/policy change | Lead, Committee, Executive | Constitutional → Exec + Committee | Governs 90–100 updates |
| **99** | [Observability & Early Warning](./Triton_Governance_Observability_Monitoring_Framework.md) | Domains, watch states, early warnings | Continuous monitoring | Operator daily; Lead weekly | GWS CRITICAL → Executive | Consumes 92; prevents 90 incidents |
| **100** | [Constitution & Charter](./Triton_Governance_Constitution_Operating_Charter.md) | Supreme principles, index, rules, glossary | Orientation; conflict resolution; attestation | Executive, Committee, all | Constitutional breach → immediate | Supersedes informal guidance |
| **102** | [Operator Handbook](./Triton_Operator_Handbook.md) | Daily operator loop, watch states, playbooks, checklist | **Every operator shift** | Operator, Senior Operator | Same as 90–93 via summaries | Distills 90–101 for console use |
| **103** | [Developer Governance Handbook](./Triton_Developer_Governance_Handbook.md) | Engineering boundaries, safeguards, anti-patterns, PR checklist | Policy-adjacent design/PR | Developer, Engineer, Admin | Lead before safeguard changes | Distills 90–102 for engineering |

**Library count:** 13 governance manuals (Steps 90–100, 102–103) + this README (Step 101).

---

## Card 3 — Situation Navigator

*Under 10-second comprehension.*

| Situation | Go to | Why |
|-----------|-------|-----|
| Something went wrong | **Step 90** | Severity, containment, template |
| What should I do right now? | **Step 102** or **91** | Handbook daily loop; GCC brief → command |
| Governance health worsening? | **Step 92 + 99** | KPIs + watch state |
| Who approves this? | **Step 93** | Authority matrix |
| Are we institutional-grade yet? | **Step 94** | Maturity / readiness gates |
| How do we test governance? | **Step 95** | Drills and validation |
| How do we report governance? | **Step 96** | Reports and audit pack |
| How do we train operators? | **Step 97** | Certification tracks |
| How do we safely change governance? | **Step 98** | Versioning and approval |
| How do we monitor governance? | **Step 99** | Early warning |
| What are the constitutional rules? | **Step 100** | Charter + glossary |
| Where am I in the library? | **This README** | Navigation only |

---

## Card 4 — Governance Dependency Map

### Layered architecture

```text
                    ┌─────────────────────────────────────┐
                    │  Step 100 — CONSTITUTION / CHARTER   │
                    └─────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ FOUNDATIONAL  │           │ OPERATIONAL   │           │  OVERSIGHT    │
│ Step 90       │──────────▶│ Step 91       │           │ Step 92       │
│ Incidents     │           │ Operator      │           │ KPI / GHS     │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        └───────────────┬───────────┴───────────┬───────────────┘
                        ▼                       ▼
                ┌───────────────┐       ┌───────────────┐
                │ Step 93       │       │ Step 99       │
                │ Authority     │       │ Observability │
                └───────┬───────┘       └───────┬───────┘
                        │                       │
        ┌───────────────┼───────────────────────┤
        ▼               ▼                       ▼
┌───────────────┐ ┌───────────────┐       ┌───────────────┐
│ Step 94       │ │ Step 95       │       │ Step 96       │
│ Maturity      │ │ Testing       │       │ Reporting     │
└───────────────┘ └───────────────┘       └───────────────┘
        │               │                       │
        └───────────────┴───────────┬───────────┘
                                    ▼
                            ┌───────────────┐
                            │ Step 97       │
                            │ Training      │
                            └───────┬───────┘
                                    │
                            ┌───────┴───────┐
                            │ Step 98       │
                            │ Change Mgmt   │
                            └───────────────┘
```

### Foundational manuals

- **Step 90** — Incident taxonomy and response (everything else assumes this)
- **Step 100** — Constitutional rules and operating loop

### Operational manuals

- **Step 91** — Daily operator actions
- **Step 93** — Approvals and halts
- **Step 99** — Continuous watch before incidents

### Oversight-oriented manuals

- **Step 92** — Metrics and health
- **Step 94** — Maturity and readiness
- **Step 96** — Reports and audit packs
- **Step 95** — Proof via drills
- **Step 97** — Competency evidence
- **Step 98** — Safe evolution of the library

### Typical read order (new governance lead)

`100 → 90 → 93 → 92 → 99 → 96 → 94 → 95 → 97 → 98 → 91`

---

## Card 5 — Role-Based Navigation

### Operator

| Track | Manuals |
|-------|---------|
| **Read first** | Step 91 → Step 90 (L1–2) |
| **Core** | 91, 90, 93 (Operator row), 96 (daily), 99 (daily) |
| **Advanced** | 92 (watch bands), 95 (drills) |
| **Escalation** | 90, 93 |
| **Reference** | 100 (rules 1, 4, 7, 8); this README |

### Senior Operator

| Track | Manuals |
|-------|---------|
| **Read first** | 91 → 90 → 93 |
| **Core** | 91, 90, 93, 92, 96 (weekly), 99 |
| **Advanced** | 95 (weekly drill), 97 (mentoring) |
| **Escalation** | 90, 93 |
| **Reference** | 100; README |

### Governance Lead

| Track | Manuals |
|-------|---------|
| **Read first** | 100 → 90 → 93 → 92 |
| **Core** | 90–96, 99 (all domains) |
| **Advanced** | 94, 95, 97, 98 |
| **Escalation** | 90, 93, 100 |
| **Reference** | README (maintain index) |

### System Administrator

| Track | Manuals |
|-------|---------|
| **Read first** | 90 (systems) → 93 (Admin row) |
| **Core** | 90 (L2 technical), 96 (evidence paths), 98 (change control) |
| **Advanced** | 99 (GCC outage) |
| **Escalation** | 93 → Risk Lead if trading impact |
| **Reference** | 100 (rule 10); README |

### Governance Committee

| Track | Manuals |
|-------|---------|
| **Read first** | 100 → 90 (L4) → 93 |
| **Core** | 90, 93, 94, 96 (quarterly pack), 98 (constitutional) |
| **Advanced** | 92, 95, 99 |
| **Escalation** | 100, 90 |
| **Reference** | README |

### Executive Oversight

| Track | Manuals |
|-------|---------|
| **Read first** | 100 → 92 (scorecard) → 96 (exec comms) |
| **Core** | 100, 92, 96, 99 (GWS), 94 (readiness disclaimer) |
| **Advanced** | 90 (L4), 93, 98 |
| **Escalation** | 90, 100 |
| **Reference** | README Card 7 Executive Quick Start |

### Developer / Engineer

| Track | Manuals |
|-------|---------|
| **Read first** | [103](./Triton_Developer_Governance_Handbook.md) → 100 (rules 8, 10) |
| **Core** | 103, 100, 90 (scope), 98 (change control), 102 (operator context) |
| **Advanced** | 95 (testing support), 99 (observability feeds) |
| **Escalation** | 93 → Admin → Risk Lead (not dev unilateral) |
| **Reference** | README; Card 9 engineering checklist |

---

## Card 6 — Governance Workflow Paths

### Incident workflow

```text
Step 90  →  classify, contain, template
   ↓
Step 91  →  posture / halt discipline
   ↓
Step 93  →  approvals, lift, override rules
   ↓
Step 96  →  report, evidence, closure
   ↓
Step 100 →  constitutional compliance check
```

| Field | Detail |
|-------|--------|
| **Purpose** | Resolve events without safeguard erosion |
| **Why sequence matters** | Severity before action; authority before lift; evidence before close |
| **Escalation relevance** | 90 defines SLAs; 93 defines who; 100 blocks shortcuts |

---

### Governance deterioration workflow

```text
Step 92  →  KPI / GHS thresholds
   ↓
Step 99  →  watch state, early warnings
   ↓
Step 94  →  maturity / readiness hold or regression
   ↓
Step 95  →  targeted drill / retest
```

| Field | Detail |
|-------|--------|
| **Purpose** | Act on leading indicators before Hard Halts |
| **Why sequence matters** | Metrics define “bad”; observability defines “now”; maturity defines institutional response |
| **Escalation relevance** | 99 GWS CRITICAL → Executive; 94 regression → Committee |

---

### Governance change workflow

```text
Step 98  →  propose, approve, version, effective date
   ↓
Step 93  →  authority impact / SoD check
   ↓
Step 97  →  training delta, re-cert if needed
   ↓
Step 95  →  drill pass before effective (High/Critical)
   ↓
Step 96  →  bulletin, report version in pack
   ↓
Step 100 →  constitutional tier compliance
```

| Field | Detail |
|-------|--------|
| **Purpose** | Evolve manuals without drift or surprise |
| **Why sequence matters** | Approval before teach before prove before report |
| **Escalation relevance** | Constitutional change → Committee + Executive |

---

### Onboarding workflow (summary)

```text
Step 100 (orientation) → Step 97 (track) → Step 91 + 90 → Step 95 (assess) → GOVCERT
```

---

## Card 7 — Quick Start Guides

### New operator quick start

| Stage | Action |
|-------|--------|
| **Read first** | [Step 102](./Triton_Operator_Handbook.md) + [Step 91](./Triton_Governance_Operator_Decision_Playbook.md) (posture matrix) |
| **Read second** | [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) (L1–2 + Soft Halt) |
| **Daily reference** | GCC brief → 91; [Step 96](./Triton_Governance_Reporting_Audit_Framework.md) daily summary |
| **Escalation** | [Step 93](./Triton_Governance_Roles_Authority_Framework.md) chain card |
| **Advanced** | [Step 99](./Triton_Governance_Observability_Monitoring_Framework.md), [Step 97](./Triton_Governance_Training_Certification_Framework.md) cert |

---

### Executive quick start

| Stage | Action |
|-------|--------|
| **Read first** | [Step 100](./Triton_Governance_Constitution_Operating_Charter.md) charter + Card 8 scorecard |
| **Read second** | [Step 92](./Triton_Governance_Metrics_KPI_Framework.md) GHS; [Step 96](./Triton_Governance_Reporting_Audit_Framework.md) exec comms |
| **Daily reference** | [Step 99](./Triton_Governance_Observability_Monitoring_Framework.md) GWS (when CRITICAL) |
| **Escalation** | [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) L4 notify rules |
| **Advanced** | [Step 94](./Triton_Governance_Lifecycle_Maturity_Framework.md) readiness (not runtime auth) |

---

### Governance lead quick start

| Stage | Action |
|-------|--------|
| **Read first** | [Step 100](./Triton_Governance_Constitution_Operating_Charter.md) → [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) → [Step 93](./Triton_Governance_Roles_Authority_Framework.md) |
| **Read second** | [Step 92](./Triton_Governance_Metrics_KPI_Framework.md) + [Step 99](./Triton_Governance_Observability_Monitoring_Framework.md) |
| **Daily reference** | 99 domains; 96 weekly/monthly |
| **Escalation** | 90, 93, 100 |
| **Advanced** | 94, 95, 97, 98 |

---

### Developer quick start

| Stage | Action |
|-------|--------|
| **Read first** | [Step 103](./Triton_Developer_Governance_Handbook.md) Cards 2, 4, 9 |
| **Read second** | [Step 100](./Triton_Governance_Constitution_Operating_Charter.md) + [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) scope |
| **Daily reference** | 103 Card 9 on policy-adjacent PRs; 102 for operator-visible UX |
| **Escalation** | 93 Admin row → Risk Lead |
| **Advanced** | 95, 98 |

---

## Card 8 — Governance Canonical Structure

| Layer | Purpose | Manuals |
|-------|---------|---------|
| **Constitution layer** | Supreme rules, operating loop, glossary | **100** |
| **Operations layer** | Shift actions, incidents, authority | **90, 91, 93, 102** |
| **Monitoring layer** | Continuous watch, early warning | **99** (consumes **92**) |
| **Oversight layer** | Metrics, maturity, reporting, audit | **92, 94, 96** |
| **Assurance layer** | Proof drills and validation | **95** |
| **Training layer** | Competency and continuity | **97** |
| **Change layer** | Safe manual evolution | **98** |
| **Navigation layer** | Discoverability (this file) | **101 (README)** |

**Cross-cutting:** Capital Preservation Doctrine appears in all layers; **Step 100** resolves conflicts.

---

## Card 9 — README Quick Reference Cards

| Situation | Go to | Why | Escalate? |
|-----------|-------|-----|-----------|
| **Incident** | 90 → 91 → 93 | Severity, action, approval | Per L1–4 |
| **Escalation** | 93, 90 | Chain + SLA | Next level on miss |
| **KPI concern** | 92, 99 | Thresholds + watch state | Elevated → Lead 4h |
| **Override request** | 93, 90 §5 | Dual approval | Committee if constitutional |
| **Governance deterioration** | 92 → 99 → 94 | Metrics → watch → maturity | Committee if regression |
| **Training question** | 97 | Certs, drills | Lead |
| **Reporting question** | 96 | Templates, audit pack | Lead / Executive |
| **Governance change** | 98 | Version, approve | Committee + Exec if constitutional |
| **Constitutional clarification** | 100 | Rules + glossary | Immediate if breach |

---

## Record ID quick reference

| Prefix | Step | Meaning |
|--------|------|---------|
| `INC-` | 90 | Incident |
| `GOVRPT-` | 96 | Governance report |
| `GOVCHG-` | 98 | Governance change |
| `GOVTEST-` / `GOVTRAIN-` | 95, 97 | Test / training |
| `GOVCERT-` | 97 | Certification |
| `GOVOBS-` | 99 | Observation log |

Full glossary: [Step 100 — Card 10](./Triton_Governance_Constitution_Operating_Charter.md).

---

## Maintenance

| Task | Owner | Process |
|------|-------|---------|
| Update this README when adding Step 102+ | Governance Lead | [Step 98](./Triton_Governance_Change_Management_Framework.md) |
| Quarterly link check | Governance Lead | Audit pack checklist (Step 96) |
| Version register | Governance Lead | Step 98 `GOVVER-*` |

---

## Document control

| Field | Value |
|-------|-------|
| Document | Step 101 — Governance README |
| Owner | Governance Lead |
| Custodian | Governance Committee (charter); Lead (index) |
| Review cycle | Quarterly or when any Step 90–100 manual has minor+ version bump |
| Change authority | Governance Lead (navigation); Committee for structural reorganization |

---

## Verification checklist (Step 101 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | README philosophy completed | Complete |
| 2 | Master index completed (Steps 90–100) | Complete |
| 3 | Situation navigator completed | Complete |
| 4 | Dependency map completed | Complete |
| 5 | Role-based navigation completed | Complete |
| 6 | Workflow paths completed | Complete |
| 7 | Quick starts completed | Complete |
| 8 | Canonical structure completed | Complete |
| 9 | Quick-reference cards completed | Complete |
| 10 | No runtime/code/UI/broker changes | **Confirmed — documentation only** |
| 11 | Enterprise-grade documentation quality | **Confirmed** |

---

*Triton Governance Operating Library — Steps 90–101. Start here; execute in the linked manual.*
