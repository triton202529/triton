# Triton Governance Institutional Memory of Decisions, Precedent & Constitutional Case Law Framework

**Document type:** Governance Manual — Precedent, Constitutional Case Law & Decision Memory
**System:** Triton Institutional Trading Platform — Governance Command Center (GCC)
**Classification:** Internal — All governance roles / Committee / Executive / Audit
**Version:** 1.0
**Status:** Manual-ready SOP
**Navigation hub:** [Governance README (Step 101)](./README.md)
**Related manuals:** [Step 100 Constitution](./Triton_Governance_Constitution_Operating_Charter.md) · [Step 111 Continuity & Succession](./Triton_Governance_Institutional_Memory_Succession_Framework.md) · [Step 113 Codex](./Triton_Governance_Codex.md) · [Step 119 Postmortems](./Triton_Governance_Postmortem_Learning_Framework.md)

---

## Scope disclaimer

This framework governs **how Triton records, classifies, and reuses governance decisions and interpretations** as institutional precedent—not ad hoc rediscovery each time ambiguity returns. It is **not** a runtime precedent engine, court system, or automatic policy applicator.

> **Governance precedent improves institutional consistency — not guaranteed outcomes.**

**Precedent record ID:** `GOVPREC-YYYY-MM-DD-###` — indexed decision, interpretation, or exception; links to `GOVCOMM-*`, `GOVCHG-*`, `GOVPM-*`, `GOVDQ-*`, `GOVETH-*`.

**Step 111 vs 121:** **111** = people, handoffs, succession (`GOVSUCC`); **121** = **decision memory**, constitutional case law, escalation consistency (`GOVPREC`).

**Override rule:** Precedent **guides**; Step 100 supremacy and **stricter containment** (113) always win. Outdated precedent is **retired**, not blindly followed (Card 5).

**Documentation only.** No code, UI, broker, execution, or governance JSON mutation.

---

# Card 1 — Precedent & Constitutional Memory Philosophy

### Purpose of governance precedent

Precedent ensures Triton **answers the same governance question the same way** when facts are comparable—reducing improvisation, political reinterpretation, and founder-only memory while preserving flexibility when evidence or law changes.

### Why Triton governance must remember decisions institutionally

| Without precedent | With precedent discipline |
|-------------------|---------------------------|
| “How did we lift last time?” unknown | ESCALATION_PRECEDENT indexed |
| Committee re-debates settled safeguard | CONSTITUTIONAL_INTERPRETATION cited |
| Ethics one-off | ETHICS_PRECEDENT consistency |
| Exception becomes shadow policy | EXCEPTION_HANDLING documented |
| New Lead reinvents | `GOVPREC` register + 111 handoff |
| Audit asks “why twice different?” | Comparable reasoning shown |

### Core principles

| Principle | Precedent meaning |
|-----------|-------------------|
| **Capital Preservation Doctrine supremacy** | Precedent never weakens containment retroactively |
| **Precedent before repeated ambiguity** | Search register before improvising |
| **Consistency before improvisation** | Same facts → same tier/authority path |
| **Constitutional safeguards dominate** | Case law cannot authorize safeguard bypass |
| **Evidence-first interpretation** | Precedent cites artifacts, not oral history |
| **Escalation continuity** | Tier decisions indexed (90) |
| **Institutional memory over individual memory** | `GOVPREC` outranks tenure |
| **Flexibility without inconsistency** | Retire or distinguish—do not ignore |

### What governance precedent proves

- Material decisions are **indexed** with rationale (Card 3)
- Interpretations are **discoverable** for future ambiguity (Card 4)
- Exceptions are **bounded** and reviewed (EXCEPTION_HANDLING)
- Inconsistency triggers **reassessment** (Card 7)
- Precedent corpus stays **lean**—anti-inflation (Card 5)

### What governance precedent cannot guarantee

- Identical outcomes in non-identical facts
- Zero debate on hard cases
- Replacement of Committee judgment
- Legal precedent status in courts
- Automatic application without human review
- Immunity from needed doctrine evolution (98)

---

# Card 2 — Precedent Classification Framework

Ten precedent types map to Steps 90–120. Each `GOVPREC` entry tags one primary type; secondary tags allowed.

---

### CONSTITUTIONAL_INTERPRETATION

| Field | Detail |
|-------|--------|
| **Definition** | Binding reading of Step 100 safeguards or non-negotiables |
| **Observed trigger** | 113 conflict; safeguard ambiguity |
| **Escalation implication** | Committee+Exec; `GOVCHG` if manual update |
| **Consistency expectation** | Cited until retired via 98 |
| **Failure implication** | Conflicting lifts |
| **Review expectation** | Annual constitutional review |

---

### ESCALATION_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | Severity tier, SLA, or chain decision for fact pattern |
| **Observed trigger** | Repeat ambiguity on L-tier |
| **Escalation implication** | Lead maintains index |
| **Consistency expectation** | Same pattern → same tier unless distinguished |
| **Failure implication** | ESCALATION_BREAKDOWN (119) |
| **Review expectation** | Quarterly with 90 metrics |

---

### CRISIS_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | Crisis tier, cell composition, normalization path (108) |
| **Observed trigger** | Second similar systemic event |
| **Escalation implication** | Committee+Exec on systemic |
| **Consistency expectation** | Reuse playbooks; update after `GOVPM` |
| **Failure implication** | Crisis reinvention |
| **Review expectation** | Post-crisis within 30d |

---

### ETHICS_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | `GOVETH` outcome for pressure class (116) |
| **Observed trigger** | INCENTIVE / EXECUTIVE pressure repeat |
| **Escalation implication** | Committee if material |
| **Consistency expectation** | Same pressure → same containment |
| **Failure implication** | Ethics inconsistency |
| **Review expectation** | Annual ethics agenda |

---

### CAPITAL_STEWARDSHIP_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | `GOVCAP` / halt discipline for capital domain (118) |
| **Observed trigger** | Lift debate on client capital |
| **Escalation implication** | Committee+Exec for client domain |
| **Consistency expectation** | Stricter domain = stricter precedent |
| **Failure implication** | Preservation breach |
| **Review expectation** | Linked to stewardship quarterly |

---

### TRUST_LEGITIMACY_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | Stakeholder communication boundary (117) |
| **Observed trigger** | LP diligence question repeat |
| **Escalation implication** | Committee on external message |
| **Consistency expectation** | Evidence-aligned disclosure pattern |
| **Failure implication** | Trust decay |
| **Review expectation** | With audit pack updates |

---

### DECISION_QUALITY_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | `GOVDQ` calibration for decision class (120) |
| **Observed trigger** | UNCERTAIN vs HIGH dispute |
| **Escalation implication** | Lead |
| **Consistency expectation** | Second reviewer rules cited |
| **Failure implication** | Judgment inconsistency |
| **Review expectation** | Quarterly Card 7 (120) |

---

### META_GOVERNANCE_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | `GOVMETA` / `GOVCHG` classification outcome (112, 98) |
| **Observed trigger** | MATERIAL vs CLARIFICATION dispute |
| **Escalation implication** | Committee per tier |
| **Consistency expectation** | Change tier matches prior similar change |
| **Failure implication** | Drift |
| **Review expectation** | Annual meta-governance review |

---

### FAILURE_LEARNING_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | Root tag + prevention from `GOVPM` (119) |
| **Observed trigger** | REPEAT_FAILURE_RISK |
| **Escalation implication** | Committee on third tag |
| **Consistency expectation** | Prevention owner pattern |
| **Failure implication** | LEARNING_DEFICIT |
| **Review expectation** | 12m root tag scan |

---

### EXCEPTION_HANDLING_PRECEDENT

| Field | Detail |
|-------|--------|
| **Definition** | Time-bound, evidenced deviation with sunset |
| **Observed trigger** | One-off waiver request |
| **Escalation implication** | Committee+Exec if constitutional touch |
| **Consistency expectation** | Never oral; sunset ≤72h or formalize 98 |
| **Failure implication** | Shadow policy |
| **Review expectation** | Mandatory expiry review |

---

# Card 3 — Precedent Operating Model

```
Identify governance decision → Preserve evidence → Classify constitutional relevance
→ Document rationale → Committee review if material → Store precedent
→ Reference in future ambiguity → Review for continued validity
```

---

### Identify governance decision

| Field | Detail |
|-------|--------|
| **Purpose** | Capture precedent-worthy matter |
| **Required actions** | Material decision list (120 appendix); open `GOVPREC` candidate |
| **What NOT to do** | Index trivial typos |
| **Escalation expectation** | Lead triage |
| **Evidence expectation** | Decision summary UTC |

---

### Preserve evidence

| Field | Detail |
|-------|--------|
| **Purpose** | Case file |
| **Required actions** | Link `INC-*`, `GOVCOMM`, logs, 107 index |
| **What NOT to do** | Precedent without artifacts |
| **Escalation expectation** | Integrity risk if missing (116) |
| **Evidence expectation** | Evidence packet ID |

---

### Classify constitutional relevance

| Field | Detail |
|-------|--------|
| **Purpose** | Tag Card 2 type |
| **Required actions** | Primary + optional secondary tag |
| **What NOT to do** | Mis-tag to avoid Committee |
| **Escalation expectation** | CONSTITUTIONAL → Exec path |
| **Evidence expectation** | Classification memo |

---

### Document rationale

| Field | Detail |
|-------|--------|
| **Purpose** | Reusable reasoning |
| **Required actions** | Facts, holding, distinguishing factors |
| **What NOT to do** | “Because we decided” without why |
| **Escalation expectation** | N/A |
| **Evidence expectation** | Rationale block in `GOVPREC` |

---

### Committee review if material

| Field | Detail |
|-------|--------|
| **Purpose** | Institutional ratification |
| **Required actions** | Constitutional, Hard lift, EXCEPTION → minutes |
| **What NOT to do** | Index oral consensus |
| **Escalation expectation** | Quorum 106 |
| **Evidence expectation** | `GOVCOMM-*` |

---

### Store precedent

| Field | Detail |
|-------|--------|
| **Purpose** | Discoverability |
| **Required actions** | Assign `GOVPREC-*`; register keywords; 107 retention |
| **What NOT to do** | Duplicate ten entries for same holding |
| **Escalation expectation** | Lead owns register |
| **Evidence expectation** | Register entry |

---

### Reference in future ambiguity

| Field | Detail |
|-------|--------|
| **Purpose** | Apply consistency |
| **Required actions** | Search register; cite or distinguish in `GOVDQ`/`INC` |
| **What NOT to do** | Ignore contrary precedent |
| **Escalation expectation** | If conflict → Card 4 playbook |
| **Evidence expectation** | Citation ID in decision record |

---

### Review for continued validity

| Field | Detail |
|-------|--------|
| **Purpose** | Anti-rigidity |
| **Required actions** | Card 7 cycle; retire if superseded by 98 |
| **What NOT to do** | Permanent “forever” exceptions |
| **Escalation expectation** | Committee on retirement |
| **Evidence expectation** | Status: ACTIVE / DISTINGUISHED / RETIRED |

---

# Card 4 — Constitutional Case Law Playbooks

| Scenario | What happened | Immediate containment | Escalation | Evidence | Consistency | Recovery |
|----------|---------------|----------------------|------------|----------|-------------|----------|
| **Repeated governance ambiguity** | Same question 3× | Search `GOVPREC`; interim stricter rule | Lead → Committee | Prior entries | Publish holding or `GOVCHG` | 98 clarify |
| **Conflicting constitutional interpretation** | Two ACTIVE precedents clash | Stricter containment (113) | Committee+Exec | Both `GOVPREC` | Retire or distinguish one | Vote + bulletin |
| **Escalation inconsistency** | Different tiers, same facts | Apply ESCALATION_PRECEDENT or escalate | Lead **5bd** | ESC logs | Align to index | 97 training |
| **Repeated committee debate** | Re-litigate settled vote | Cite `GOVCOMM` + `GOVPREC` | Chair rules scope | Minutes | Defer unless new facts | Agenda discipline |
| **Ethics inconsistency** | Same pressure, different outcome | Contain per stricter prior | Committee | `GOVETH` | Align ETHICS_PRECEDENT | 116 review |
| **Fiduciary interpretation conflict** | Client vs firm precedent clash | Stricter domain | Committee+Exec | `GOVCAP` | CAPITAL_STEWARDSHIP wins | Document hierarchy |
| **Crisis precedent reuse** | Second systemic event | 108 playbook + index | Crisis cell | `GOVCRISIS`, `GOVPREC` | Update if drill improved | `GOVPM` |
| **Governance exception handling** | Waiver request | No oral exception | Committee+Exec if needed | EXCEPTION record | Sunset enforced | Formalize or expire |
| **Precedent drift concern** | Practice ≠ indexed holdings | Freeze shadow policy | 112 drift review | Register vs practice | OCR audit | `GOVCHG` or retire |

---

# Card 5 — Precedent Quality & Anti-Rigidity Model

| Risk | Why dangerous | Failure consequence | Escalation | Corrective expectation |
|------|---------------|---------------------|------------|------------------------|
| **Precedent rigidity** | Obsolete rule blocks improvement | Wrong containment | Committee | RETIRED status via 98 |
| **Outdated assumptions** | Market/org changed | Bad lift | Annual Card 7 | Distinguish or retire |
| **Institutional forgetfulness** | No index | Reinvention | 111 + 121 | Mandatory `GOVPREC` on material |
| **Case-law inflation** | Thousands of entries | Unusable register | Lead | One holding per issue; merge dupes |
| **Blind precedent following** | Facts differ | Error | 120 distinguish step | “Distinguish” memo required |
| **Founder-memory dependence** | Oral precedent | Inconsistency | 111 delegate | Written `GOVPREC` only |
| **Inconsistent reinterpretation** | Political rereading | Trust loss | Committee | Cite or retire—no silent change |

**Quality rule:** Maximum **one ACTIVE holding** per issue key (e.g., `hard-lift-client-capital-2025`); amendments supersede with link chain.

---

# Card 6 — Escalation Consistency & Case-Law Model

| Dimension | Why important | Failure consequence | Escalation | Continuity expectation |
|-----------|---------------|---------------------|------------|------------------------|
| **Escalation consistency** | Fair, predictable chain | SLA miss; morale | Lead | ESCALATION_PRECEDENT indexed |
| **Constitutional interpretation continuity** | Safeguard meaning stable | CLPR breach | Committee+Exec | CONSTITUTIONAL_INTERPRETATION ACTIVE set |
| **Governance comparability** | Audit/LP trust | Adverse finding | 107 | Comparable fact sheets |
| **Repeatable committee reasoning** | Votes not arbitrary | Legitimacy risk | 106 | Minutes + `GOVPREC` link |
| **Institutional judgment durability** | Survives turnover | 111 gap | `GOVSUCC` includes register tour | 97 precedent module |
| **Exception-handling discipline** | No shadow law | Drift | Committee | EXCEPTION sunset 100% |

---

# Card 7 — Precedent Review & Constitutional Reassessment Model

```
Review precedent quality → Assess continued constitutional fit → Committee review
→ Escalate inconsistency concern → Document lessons → Preserve valid interpretations → Retire outdated assumptions
```

---

### Review precedent quality

| Field | Detail |
|-------|--------|
| **Purpose** | Register health |
| **Required actions** | Quarterly: dupes, orphans, missing evidence links |
| **What NOT to do** | Grow register without pruning |
| **Escalation expectation** | Lead |
| **Evidence expectation** | Quality score |

---

### Assess continued constitutional fit

| Field | Detail |
|-------|--------|
| **Purpose** | Align with Step 100 / effective 98 |
| **Required actions** | Flag ACTIVE entries contradicted by new `GOVCHG` |
| **What NOT to do** | Leave conflicting ACTIVE |
| **Escalation expectation** | Auto-retire superseded |
| **Evidence expectation** | Supersede chain |

---

### Committee review

| Field | Detail |
|-------|--------|
| **Purpose** | Ratify retirements and holdings |
| **Required actions** | Annual precedent agenda (106) |
| **What NOT to do** | Skip EXCEPTION expiry review |
| **Escalation expectation** | Quorum |
| **Evidence expectation** | `GOVCOMM-*` |

---

### Escalate inconsistency concern

| Field | Detail |
|-------|--------|
| **Purpose** | Stop divergent practice |
| **Required actions** | Open inconsistency packet |
| **What NOT to do** | Pick favorite precedent politically |
| **Escalation expectation** | Committee **10bd** |
| **Evidence expectation** | Side-by-side comparison |

---

### Document lessons

| Field | Detail |
|-------|--------|
| **Purpose** | Learning link |
| **Required actions** | `GOVPM` if failure; update register |
| **What NOT to do** | Lesson only oral |
| **Escalation expectation** | 119 path |
| **Evidence expectation** | Lesson memo |

---

### Preserve valid interpretations

| Field | Detail |
|-------|--------|
| **Purpose** | Stability |
| **Required actions** | Keep ACTIVE with clear facts |
| **What NOT to do** | Retire useful holdings without replacement |
| **Escalation expectation** | N/A |
| **Evidence expectation** | ACTIVE list published internally |

---

### Retire outdated assumptions

| Field | Detail |
|-------|--------|
| **Purpose** | Anti-rigidity |
| **Required actions** | RETIRED + reason; bulletin if operators affected |
| **What NOT to do** | Silent ignore |
| **Escalation expectation** | 98 if manual change needed |
| **Evidence expectation** | Retirement memo |

---

# Card 8 — Humility, Memory & Constitutional Discipline Model

| Risk | Why dangerous | Failure consequence | Escalation | Corrective expectation |
|------|---------------|---------------------|------------|------------------------|
| **Founder memory dependency** | Oral “precedent” | Inconsistency | 111 | `GOVPREC` required |
| **Governance reinvention** | Ignore register | Repeated debate | Lead | Search before decide |
| **Interpretive arrogance** | New reading without vote | Trust loss | Committee+Exec | CONSTITUTIONAL path |
| **Certainty through precedent illusion** | False comfort | Wrong application | 120 distinguish | Facts comparison mandatory |
| **Institutional forgetfulness** | No index | Drift | 107 + 121 | Material decisions indexed |
| **Historical cherry-picking** | Select favorable case | Bad faith | 116 | Full register review |
| **Governance inconsistency** | Active conflicts | Audit adverse | Card 4 playbook | Stricter rule interim |

---

# Card 9 — Precedent Quick Start

*Under 1-minute comprehension.*

| Stage | Document / action |
|-------|-------------------|
| **Read first** | [Step 113](./Triton_Governance_Codex.md) Card 8 — conflict → stricter containment |
| **Read second** | Search `GOVPREC` register — cite, distinguish, or escalate conflict |
| **Precedent references** | Card 2 type; Card 3 store loop |
| **Escalation references** | 90 + 106; Card 4 if inconsistency |
| **Constitution references** | [Step 100](./Triton_Governance_Constitution_Operating_Charter.md); retire via 98 |

**Precedent mantra:** *Search → compare facts → cite or distinguish → document—never oral-only precedent.*

---

# Card 10 — Precedent Checklist

**Create / update `GOVPREC` (material decisions)**

- [ ] Decision classified (Card 2 type)
- [ ] Evidence preserved and linked
- [ ] Constitutional implications reviewed
- [ ] Escalation consistency reviewed (tier vs priors)
- [ ] Governance rationale documented (facts, holding, distinguishers)
- [ ] Precedent relevance reviewed (cite or distinguish prior)
- [ ] Committee escalation completed if constitutional / exception / Hard lift
- [ ] Institutional implications (110, 117, 118) noted
- [ ] Status set: ACTIVE / DISTINGUISHED / RETIRED
- [ ] Register keywords for search

**Use precedent (future decision)**

- [ ] Register searched for comparable facts
- [ ] ACTIVE holdings cited or distinguished in writing
- [ ] Conflicts escalated—not silently picked
- [ ] EXCEPTION sunset checked if applicable

---

# Card 11 — Quick Reference Precedent Cards

*Under 10-second comprehension.*

| Situation | What to assess | Escalate? | Evidence | Step |
|-----------|----------------|-----------|----------|------|
| **Repeated governance ambiguity** | `GOVPREC` index | Lead → Committee | Prior holdings | 121, 113 |
| **Conflicting interpretation concern** | ACTIVE clash | Committee+Exec | Both `GOVPREC` | 121, 100 |
| **Escalation inconsistency** | ESCALATION type | Lead **5bd** | ESC + index | 90, 121 |
| **Crisis precedent concern** | CRISIS type | 108 cell | `GOVCRISIS` | 108, 121 |
| **Ethics precedent concern** | ETHICS type | Committee | `GOVETH` | 116, 121 |
| **Fiduciary precedent concern** | CAPITAL type | Committee+Exec | `GOVCAP` | 118, 121 |
| **Governance exception concern** | EXCEPTION sunset | Committee if material | `GOVPREC` | 121, 98 |
| **Outdated precedent concern** | RETIRE candidate | Committee annual | 98 supersede | 121, 112 |

---

# Card 12 — Precedent & Case Law Appendix

### Precedent entry minimum fields

| Field | Required |
|-------|----------|
| `GOVPREC` ID | Yes |
| Primary type (Card 2) | Yes |
| Facts summary | Yes |
| Holding (one paragraph) | Yes |
| Distinguishing factors | If any |
| Evidence links | Yes |
| `GOVCOMM` / `GOVCHG` link | If applicable |
| Status ACTIVE / DISTINGUISHED / RETIRED | Yes |
| Review date | Yes |
| Owner (Governance Lead) | Yes |

### Glossary

| Term | Definition |
|------|------------|
| **Constitutional case law** | Indexed CONSTITUTIONAL_INTERPRETATION holdings under Step 100 |
| **Constitutional continuity** | Stable safeguard meaning until formally changed (98) |
| **Escalation consistency** | Same fact pattern → same tier/chain unless distinguished |
| **Exception handling** | Time-bound evidenced deviation—never oral |
| **Governance interpretation** | Documented reading of manual/constitutional text |
| **Governance precedent** | Indexed `GOVPREC` decision usable for future matters |
| **Governance reassessment** | Card 7 review retiring or updating holdings |
| **Institutional memory** | Written register + 107 retention—not individual recall |
| **Institutional reasoning** | Evidence-first rationale reusable across roles |
| **Precedent drift** | Practice diverges from ACTIVE holdings |

**Record IDs:** `GOVPREC-*` · `GOVSUCC-*` (111 handoff includes register) · `GOVPM-*` (119 may create precedent)

**Extended references:** [Step 111](./Triton_Governance_Institutional_Memory_Succession_Framework.md) · [Step 107](./Triton_Governance_Audit_Regulatory_Readiness_Handbook.md)

---

## Document control

| Field | Value |
|-------|-------|
| Owner | Governance Lead |
| Review cycle | Quarterly register quality; annual Committee precedent agenda |
| Change authority | Step 98 (`GOVCHG`); holdings RETIRED when superseded |
| Distribution | All governance roles; Committee; Audit |

---

## Verification checklist (Step 121 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Precedent philosophy completed | Complete |
| 2 | Precedent classifications completed (10) | Complete |
| 3 | Precedent operating model completed | Complete |
| 4 | Case-law playbooks completed (9) | Complete |
| 5 | Anti-rigidity model completed (7) | Complete |
| 6 | Escalation consistency model completed (6) | Complete |
| 7 | Constitutional reassessment model completed | Complete |
| 8 | Humility/memory discipline model completed (7) | Complete |
| 9 | Quick start completed | Complete |
| 10 | Checklist completed | Complete |
| 11 | Quick-reference cards completed | Complete |
| 12 | Appendix completed | Complete |
| 13 | No runtime, code, UI, broker, or governance mutation | **Confirmed — documentation only** |
| 14 | Enterprise-grade precedent framework | **Confirmed** |

---

*End of document — Triton Governance Institutional Memory of Decisions, Precedent & Constitutional Case Law Framework (Step 121)*
