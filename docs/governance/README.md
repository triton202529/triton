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
| **104** | [Executive Governance Handbook](./Triton_Executive_Governance_Handbook.md) | Strategic oversight, watch states, ratification, scorecards | Weekly; daily if CRITICAL/Hard Halt | Founder, Committee, Executive | L4 / constitutional / Hard lift | Distills 90–103 for executives |
| **106** | [Governance Committee Charter](./Triton_Governance_Committee_Charter.md) | Committee operating loop, votes, Hard lift, constitutional approval | Scheduled + L4/Hard Halt sessions | Governance Committee | Quorum votes; Executive constitutional tier | Steps 93, 94, 98, 100 |
| **107** | [Audit & Regulatory Readiness](./Triton_Governance_Audit_Regulatory_Readiness_Handbook.md) | Evidence preservation, diligence Q&A, retention, audit pack | Quarterly; diligence events | Audit, compliance, Committee, LPs | Adverse pack → Committee **5bd** | Steps 90–106 evidence index |
| **108** | [Crisis Management & Recovery](./Triton_Governance_Crisis_Recovery_Handbook.md) | Crisis classification, emergency response, Hard Halt, normalization | Active crisis; post-systemic review | All crisis chain roles | Constitutional/systemic → Committee+Exec | Steps 90–107 under stress |
| **109** | [War Games & Stress Testing](./Triton_Governance_Wargaming_Stress_Testing_Handbook.md) | Rehearsal tiers, resilience scoring, extreme scenarios | Quarterly war game+; annual systemic | Lead, Committee, all roles in exercises | Critical exercise fail → remediation | Steps 95 drills + 108 live crisis |
| **110** | [Readiness Scoring & Certification](./Triton_Governance_Readiness_Certification_Framework.md) | IRS bands, institutional `GOVCERT-INST`, authority gates | Quarterly IRS; annual certification | Lead, Committee, Executive | Revoke on R1–R8 / safeguard breach | Steps 92, 94, 95, 107–109 inputs |
| **111** | [Institutional Memory & Succession](./Triton_Governance_Institutional_Memory_Succession_Framework.md) | Knowledge continuity, `GOVSUCC` handoffs, key-person risk | Per transition; annual drill | Lead, Committee | MULTI_ROLE / Founder absence | Steps 97, 98, 107–110 |
| **112** | [Meta-Governance & Constitutional Evolution](./Triton_Governance_Meta_Governance_Framework.md) | Governance-of-governance, `GOVMETA`/`GOVCHG`, drift control | Proposed change; annual review | Lead, Committee, Executive | Constitutional instability | Steps 98, 100, 106, 110–111 |
| **113** | [Governance Codex](./Triton_Governance_Codex.md) | Unified constitutional system map, priority order, interoperability | Orientation; conflict routing; audit “show the system” | All roles | Interpretation disputes → 100 + 98 | Steps 90–112 synthesis |
| **114** | [Maturity Roadmap & Milestones](./Triton_Governance_Maturity_Roadmap.md) | Multi-year stages, capability gating, `GOVMAT` advancement | Stage review; advancement/regression | Lead, Committee, Executive | Stage regress / false maturity | Steps 94, 110, 113 |
| **115** | [Strategic Foresight & Scenario Planning](./Triton_Governance_Strategic_Foresight_Framework.md) | `GOVFORE` scenarios, horizons, black swan preparedness | Annual + quarterly NEAR signals | Lead, Committee, Executive | Convergence / BLACK_SWAN live | Steps 99, 109–114 |
| **116** | [Ethics & Decision Integrity](./Triton_Governance_Ethics_Integrity_Framework.md) | Institutional values, `GOVETH`, integrity under pressure | Quarterly review; pressure events | All roles; Committee | GOVERNANCE_INTEGRITY_RISK / trust decay | Steps 93, 100, 106–107, 110 |
| **117** | [Stakeholder Trust & Legitimacy](./Triton_Governance_Stakeholder_Trust_Framework.md) | External trust, `GOVTRUST`, reputation & communication discipline | Quarterly; LP/audit events | Lead, Committee, Executive | Legitimacy / crisis trust events | Steps 96–97, 107–108, 116 |
| **118** | [Capital Stewardship & Fiduciary Discipline](./Triton_Governance_Capital_Stewardship_Framework.md) | Capital domains, `GOVCAP`, preservation & fiduciary duty | Quarterly; preservation events | All roles; Committee | CAPITAL_PRESERVATION_BREACH | Steps 90, 100, 116–117 |
| **119** | [Postmortems & Institutional Learning](./Triton_Governance_Postmortem_Learning_Framework.md) | `GOVPM`, near-miss intelligence, anti-repeat learning | Per incident; quarterly learning review | All roles; Committee | REPEAT_FAILURE_RISK / LEARNING_DEFICIT | Steps 90, 107, 111–112 |
| **120** | [Decision Quality & Cognitive Risk](./Triton_Governance_Decision_Quality_Framework.md) | Judgment calibration, `GOVDQ`, bias awareness under uncertainty | Material decisions; quarterly review | All roles; Committee | Repeated cognitive blind spot | Steps 90–91, 116, 119 |
| **121** | [Precedent & Constitutional Case Law](./Triton_Governance_Precedent_Case_Law_Framework.md) | `GOVPREC` index, interpretations, escalation consistency | Material decisions; ambiguity lookup | Lead, Committee | Conflicting ACTIVE precedent | Steps 100, 111–113, 119–120 |

**Library count:** 30 governance manuals (Steps 90–100, 102–104, 106–121) + this README (Step 101).

---

## Card 3 — Situation Navigator

*Under 10-second comprehension.*

| Situation | Go to | Why |
|-----------|-------|-----|
| Something went wrong | **Step 90** | Severity, containment, template |
| Postmortem / near-miss / repeat failure? | **Step 119** | `GOVPM` after contain; learn before blame |
| Capital / fiduciary / preservation concern? | **Step 118** + **100** | `GOVCAP` → halt → escalate |
| What should I do right now? | **Step 102** or **91** | Handbook daily loop; GCC brief → command |
| Uncertain judgment / bias / escalation hesitation? | **Step 120** | Classify → evidence → escalate |
| Governance health worsening? | **Step 92 + 99** | KPIs + watch state |
| Who approves this? | **Step 93** | Authority matrix |
| Are we institutional-grade yet? | **Step 94** + **110** + **114** | Current level + IRS + roadmap stage |
| How do we test governance? | **Step 95** + **109** | Drills; war games & resilience |
| Future risk or scenario planning? | **Step 115** | `GOVFORE` → war-game → readiness |
| How do we report governance? | **Step 96** | Reports and audit pack |
| How do we train operators? | **Step 97** | Certification tracks |
| How do we safely change governance? | **Step 98** + **112** | `GOVCHG` execution; meta-governance loop |
| How do we monitor governance? | **Step 99** | Early warning |
| What are the constitutional rules? | **Step 100** + **113** Codex |
| How does the whole system fit together? | **Step 113** | Unified map — manuals stay authoritative |
| Committee vote / quorum? | **Step 106** | Charter + glossary |
| Audit / LP / regulator diligence? | **Step 107** + **96** + **117** | Audit pack; trust & legitimacy discipline |
| Active governance crisis? | **Step 108** | Classify → contain → recover |
| Rehearse governance crisis? | **Step 109** | War game before reality |
| Is governance objectively ready? | **Step 110** | IRS + institutional cert |
| Leadership or role transition? | **Step 111** | `GOVSUCC` succession playbook |
| How does governance improve itself? | **Step 112** | `GOVMETA` → classify → `GOVCHG` |
| Governance advancement or stage concern? | **Step 114** | `GOVMAT` gates + timeline |
| Governance conflict / unclear ownership? | **Step 113** Card 8 | Stricter containment → 98 |
| Same question again / precedent lookup? | **Step 121** | `GOVPREC` — cite, distinguish, or retire |
| Ethics, pressure, or integrity concern? | **Step 116** | `GOVETH` → escalate → contain |
| Stakeholder trust or reputation concern? | **Step 117** | `GOVTRUST` → evidence-aligned communication |
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
| **Read first** | [106](./Triton_Governance_Committee_Charter.md) → 100 → 93 |
| **Core** | 106, 90 (L4), 93, 94, 96, 98 |
| **Advanced** | 92, 95, 99, 104 (Executive interface) |
| **Escalation** | Executive constitutional ratification |
| **Reference** | README; 106 Card 9 session checklist |

### Executive Oversight

| Track | Manuals |
|-------|---------|
| **Read first** | [104](./Triton_Executive_Governance_Handbook.md) → 100 → 92 |
| **Core** | 104, 100, 92, 96, 99, 94 |
| **Advanced** | 90 (L4), 93, 98, 95 (validation attestation) |
| **Escalation** | Card 4 (104); 90, 100 |
| **Reference** | 102/103 for context only—do not micromanage |

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
| **Read first** | [Step 104](./Triton_Executive_Governance_Handbook.md) Cards 3, 4, 8 |
| **Read second** | [Step 100](./Triton_Governance_Constitution_Operating_Charter.md) + [Step 92](./Triton_Governance_Metrics_KPI_Framework.md) |
| **Daily reference** | 104 Card 9 (CRITICAL/Hard Halt); else weekly scorecard |
| **Escalation** | 104 Card 4; [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) L4 |
| **Advanced** | [Step 94](./Triton_Governance_Lifecycle_Maturity_Framework.md), [98](./Triton_Governance_Change_Management_Framework.md) |

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
| **Oversight layer** | Metrics, maturity, reporting, audit | **92, 94, 96, 104** |
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
