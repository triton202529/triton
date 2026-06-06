# Triton Governance Delegation, Authority Distribution & Decision Rights Framework

**Document type:** Governance Manual — Delegation, Authority Distribution & Decision Rights
**System:** Triton Institutional Trading Platform — Governance Command Center (GCC)
**Classification:** Internal — All governance roles / Committee / Executive / Audit
**Version:** 1.0
**Status:** Manual-ready SOP
**Navigation hub:** [Governance README (Step 101)](./README.md)
**Related manuals:** [Step 93 Roles & Authority Matrix](./Triton_Governance_Roles_Authority_Framework.md) · [Step 106 Committee Charter](./Triton_Governance_Committee_Charter.md) · [Step 111 Continuity & Succession](./Triton_Governance_Institutional_Memory_Succession_Framework.md) · [Step 126 Scalability](./Triton_Governance_Scalability_Evolution_Framework.md)

---

## Scope disclaimer

This framework governs **how authority is delegated, distributed, constrained, reviewed, and reclaimed** as Triton scales—decision-rights architecture and accountability chains. It **complements** Step 93 (who may approve what); Step 127 covers **delegation lifecycle**, non-delegable boundaries, and authority drift.

> **Delegation improves institutional scalability and accountability — not guaranteed outcomes.**

**Delegation record ID:** `GOVDELEG-YYYY-MM-DD-###` — assignment, temporary delegate, emergency authority, or revocation; links to `GOVSUCC-*`, `GOVPREC-*`, `GOVCOMM-*`.

**Not HR/org chart:** Titles and reporting lines are out of scope; **governance decision rights** are in scope.

**Documentation only.** No code, UI, broker, execution, or governance JSON mutation.

---

# Card 1 — Authority & Delegation Philosophy

### Purpose of authority governance

Authority governance ensures power is **explicit, bounded, accountable, and reversible**—neither concentrated in one person nor fragmented into ambiguous committees—so escalation works and audits reconstruct who decided what.

### Why institutions fail when authority is overly concentrated or dangerously fragmented

| Failure mode | Delegation discipline |
|--------------|----------------------|
| Founder-only ratification | Executive delegate + COMMITTEE_REQUIRED |
| “Everyone decides” | Card 3 decision-right class |
| Delegate acts beyond scope | `GOVDELEG` scope + sunset |
| Emergency authority sticks | EMERGENCY sunset ≤72h |
| Oral delegation | Written assignment only |
| No revocation on transition | 111 reclaims on `GOVSUCC` |
| Matrix ignored at scale | 126 scale-state review |

### Core principles

| Principle | Authority meaning |
|-----------|-------------------|
| **Capital Preservation Doctrine supremacy** | Delegation never includes safeguard waiver |
| **Accountability before authority** | Named owner before grant |
| **Authority with responsibility** | Delegate inherits duty, not immunity |
| **Constitutional safeguards dominate** | NON_DELEGABLE set fixed |
| **Clarity before delegation** | 93 matrix updated before grant |
| **Escalation before ambiguity** | Unclear → up-tier per 90 |
| **Stewardship of authority** | Power serves mission (124) |
| **Authority exists to serve mission** | Not personal discretion |

### What delegation proves

- **Twelve authority domains** have holders and boundaries (Card 2)
- Decisions map to **decision-right class** (Card 3)
- Assignments follow **documented loop** (Card 4)
- Failures trigger **playbooks** (Card 5)
- **Annual authority review** occurs (Card 7)

### What delegation cannot guarantee

- Perfect judgment by delegates
- Zero delays from proper escalation
- Elimination of all authority disputes
- Legal agency or employment authority
- Runtime RBAC (technical systems separate)

---

# Card 2 — Authority Domain Framework

Twelve domains align with Steps 90–126. Step **93** remains definitive for role-level approvals within each domain.

---

### Constitutional Authority

| Field | Detail |
|-------|--------|
| **Definition** | Safeguard interpretation, Hard lift ratification tier |
| **Who may hold it** | Committee + Executive (106, 104); not delegable core |
| **Why protected** | Step 100 supremacy |
| **Observed misuse signal** | Oral constitutional “OK” |
| **Escalation implication** | Immediate Committee+Exec |
| **Review expectation** | Annual + per CONSTITUTIONAL `GOVCHG` |

---

### Governance Committee Authority

| Field | Detail |
|-------|--------|
| **Definition** | Votes, quorum, MATERIAL+ policy |
| **Who may hold it** | Committee members; Chair executes agenda |
| **Why protected** | Institutional ratification |
| **Observed misuse signal** | Vote without quorum |
| **Escalation implication** | Interim chair 48h |
| **Review expectation** | Per session |

---

### Executive Authority

| Field | Detail |
|-------|--------|
| **Definition** | Strategic oversight, attestation, delegate ratification |
| **Who may hold it** | Executive Oversight; secondary delegate named |
| **Why protected** | L4 and external representation |
| **Observed misuse signal** | Lift without signature |
| **Escalation implication** | Committee emergency |
| **Review expectation** | Quarterly delegate list |

---

### Operational Authority

| Field | Detail |
|-------|--------|
| **Definition** | Daily GCC, Soft Halt, L1–L2, operator playbook |
| **Who may hold it** | Operator, Senior Operator, Lead (delegable subset) |
| **Why protected** | First-line containment |
| **Observed misuse signal** | Self-approve override |
| **Escalation implication** | Lead **48h** |
| **Review expectation** | Per cert (97) |

---

### Crisis Authority

| Field | Detail |
|-------|--------|
| **Definition** | 108 crisis cell, systemic tier |
| **Who may hold it** | Pre-named crisis roles; Lead chairs |
| **Why protected** | Compressed clock |
| **Observed misuse signal** | Ad hoc crisis dictator |
| **Escalation implication** | Committee+Exec systemic |
| **Review expectation** | Post-crisis + annual drill |

---

### Audit Authority

| Field | Detail |
|-------|--------|
| **Definition** | Read/evaluate packs; adverse finding escalation |
| **Who may hold it** | Audit/compliance function; not halt authority |
| **Why protected** | Independence |
| **Observed misuse signal** | Audit approves lift |
| **Escalation implication** | Committee **5bd** |
| **Review expectation** | Quarterly |

---

### Capital Stewardship Authority

| Field | Detail |
|-------|--------|
| **Definition** | Halt discipline, client-capital tier decisions |
| **Who may hold it** | Lead + Committee+Exec for client domain |
| **Why protected** | 118 fiduciary |
| **Observed misuse signal** | Delegate lifts Hard Halt |
| **Escalation implication** | `GOVCAP` |
| **Review expectation** | Weekly |

---

### Ethics Authority

| Field | Detail |
|-------|--------|
| **Definition** | Integrity review, pressure class response |
| **Who may hold it** | Lead triage; Committee material |
| **Why protected** | 116 |
| **Observed misuse signal** | Pressure ignored |
| **Escalation implication** | `GOVETH` |
| **Review expectation** | Quarterly |

---

### Mission Alignment Authority

| Field | Detail |
|-------|--------|
| **Definition** | Purpose drift classification |
| **Who may hold it** | Committee + Executive |
| **Why protected** | 124 |
| **Observed misuse signal** | Growth overrides purpose |
| **Escalation implication** | `GOVALIGN` |
| **Review expectation** | Per scale transition |

---

### Exception Authority

| Field | Detail |
|-------|--------|
| **Definition** | Time-bound waiver (121 EXCEPTION_HANDLING) |
| **Who may hold it** | Committee+Exec if constitutional touch |
| **Why protected** | Anti-shadow policy |
| **Observed misuse signal** | Permanent “exception” |
| **Escalation implication** | Mandatory sunset |
| **Review expectation** | Quarterly expiry sweep |

---

### Scalability Authority

| Field | Detail |
|-------|--------|
| **Definition** | Entity/jurisdiction/scale state approval |
| **Who may hold it** | Committee+Exec MULTI_ENTITY+ |
| **Why protected** | 126 gates |
| **Observed misuse signal** | Entity without `GOVSCALE` |
| **Escalation implication** | Block expansion |
| **Review expectation** | Per event |

---

### Meta-Governance Authority

| Field | Detail |
|-------|--------|
| **Definition** | `GOVMETA`/`GOVCHG` classification |
| **Who may hold it** | Lead proposes; Committee MATERIAL+ |
| **Why protected** | 112 |
| **Observed misuse signal** | Shadow manual edit |
| **Escalation implication** | Committee |
| **Review expectation** | Annual meta-review |

---

# Card 3 — Decision Rights Classification Model

Six categories. Every material decision tags one class in `GOVDELEG` or decision record.

---

### NON_DELEGABLE

| Field | Detail |
|-------|--------|
| **Definition** | Cannot be assigned away from statutory holder |
| **Scope** | Hard lift ratification; constitutional `GOVCHG`; dual approval on overrides; CLPR waiver |
| **Escalation expectation** | Committee+Exec always |
| **Review expectation** | Annual non-delegable list ack |
| **Failure implication** | Constitutional failure |
| **Revocation expectation** | N/A—void ab initio if attempted |

---

### COMMITTEE_REQUIRED

| Field | Detail |
|-------|--------|
| **Definition** | Quorum vote mandatory |
| **Scope** | MATERIAL policy; institutional cert; Hard lift vote; MULTI_ENTITY |
| **Escalation expectation** | Chair convenes |
| **Review expectation** | Minutes per 106 |
| **Failure implication** | Oral governance |
| **Revocation expectation** | Vote to rescind delegate action if ultra vires |

---

### EXECUTIVE_DELEGABLE

| Field | Detail |
|-------|--------|
| **Definition** | Executive may delegate with written scope |
| **Scope** | Attestation prep; external comms prep; L4 notification ack |
| **Escalation expectation** | Executive retains accountability |
| **Review expectation** | Quarterly delegate roster |
| **Failure implication** | Founder dependency |
| **Revocation expectation** | Executive notice; immediate |

---

### OPERATIONAL_DELEGABLE

| Field | Detail |
|-------|--------|
| **Definition** | Lead/Senior Op may delegate within 93 matrix |
| **Scope** | Shift coverage; L1–L2 triage; GCC brief ack |
| **Escalation expectation** | Lead accountable |
| **Review expectation** | Per shift / `GOVSUCC` |
| **Failure implication** | Uncertified acting |
| **Revocation expectation** | Lead same day |

---

### TEMPORARY_DELEGATION

| Field | Detail |
|-------|--------|
| **Definition** | Time-bound `GOVDELEG` for absence |
| **Scope** | Explicit list—never blanket “all authority” |
| **Escalation expectation** | 111 tier rules |
| **Review expectation** | Sunset date mandatory |
| **Failure implication** | Permanent temp delegate |
| **Revocation expectation** | Auto-expire or principal revoke |

---

### EMERGENCY_AUTHORITY

| Field | Detail |
|-------|--------|
| **Definition** | Crisis-only expanded scope (108, 112 META emergency) |
| **Scope** | Contain-first; no runtime enablement |
| **Escalation expectation** | Committee **24h** ratify or revoke |
| **Review expectation** | Sunset **≤72h** |
| **Failure implication** | Emergency becomes normal |
| **Revocation expectation** | **Mandatory** at sunset |

---

# Card 4 — Authority Distribution Operating Model

```
Identify decision domain → Assess authority requirements → Review constitutional implications
→ Determine delegation eligibility → Escalate if required → Document authority assignment
→ Monitor authority usage → Reassess delegation validity
```

---

### Identify decision domain

| Field | Detail |
|-------|--------|
| **Purpose** | Map to Card 2 |
| **Required actions** | Tag domain; link decision type |
| **What NOT to do** | Decide before domain tagged |
| **Escalation expectation** | Ambiguity → 113 |
| **Evidence expectation** | Domain tag |

---

### Assess authority requirements

| Field | Detail |
|-------|--------|
| **Purpose** | 93 matrix lookup |
| **Required actions** | Dual approval? Committee? Executive? |
| **What NOT to do** | Infer from urgency |
| **Escalation expectation** | Up-tier if matrix gap |
| **Evidence expectation** | Matrix row citation |

---

### Review constitutional implications

| Field | Detail |
|-------|--------|
| **Purpose** | NON_DELEGABLE check |
| **Required actions** | If touch → COMMITTEE_REQUIRED minimum |
| **What NOT to do** | Delegate constitutional tier |
| **Escalation expectation** | Committee+Exec |
| **Evidence expectation** | Checklist |

---

### Determine delegation eligibility

| Field | Detail |
|-------|--------|
| **Purpose** | Assign Card 3 class |
| **Required actions** | If delegating → scope, sunset, cert |
| **What NOT to do** | Blanket delegation |
| **Escalation expectation** | Lead approves OPERATIONAL only |
| **Evidence expectation** | `GOVDELEG` draft |

---

### Escalate if required

| Field | Detail |
|-------|--------|
| **Purpose** | No solo ultra vires act |
| **Required actions** | 90 chain when class requires |
| **What NOT to do** | Delegate then hide escalation |
| **Escalation expectation** | Per class |
| **Evidence expectation** | ESC record |

---

### Document authority assignment

| Field | Detail |
|-------|--------|
| **Purpose** | Audit trail |
| **Required actions** | Sign `GOVDELEG`; update delegate roster (111) |
| **What NOT to do** | Email-only delegation |
| **Escalation expectation** | Committee ack TEMPORARY+ if Lead+ |
| **Evidence expectation** | Signed assignment |

---

### Monitor authority usage

| Field | Detail |
|-------|--------|
| **Purpose** | Detect drift |
| **Required actions** | Sample decisions vs scope; override log |
| **What NOT to do** | Ignore out-of-scope acts |
| **Escalation expectation** | Ultra vires → Card 5 |
| **Evidence expectation** | Usage audit quarterly |

---

### Reassess delegation validity

| Field | Detail |
|-------|--------|
| **Purpose** | Revoke or renew |
| **Required actions** | Card 7; expire TEMPORARY/EMERGENCY |
| **What NOT to do** | Auto-renew without review |
| **Escalation expectation** | Committee annual |
| **Evidence expectation** | Reassessment memo |

---

# Card 5 — Delegation Failure Playbooks

| Scenario | What happened | Immediate containment | Escalation | Evidence | Recovery | Authority implication |
|----------|---------------|----------------------|------------|----------|----------|----------------------|
| **Authority ambiguity** | Two roles claim decision | Stricter containment; freeze act | Lead **24h** | Conflict log | 93 matrix clarify via 98 | Matrix gap |
| **Authority concentration** | All paths to one person | Name mandatory delegate | Committee | Roster | 111 drill | Scalability block |
| **Authority fragmentation** | Nobody owns decision | Lead triage owner | Committee **5bd** | SoD gap | 93 update | Ownership confusion |
| **Delegation without accountability** | Delegate, no principal | Suspend delegation | Principal + Lead | `GOVDELEG` | Re-assign with owner | Void acts ultra vires |
| **Escalation bypass** | Skip tier | Restore chain | Per 90 | ESC audit | 97 retrain | NON_DELEGABLE breach if L4 hidden |
| **Unauthorized decision making** | No matrix/DELEG | Block; revert if lift | Committee **48h** | Decision log | Discipline path external | Integrity 116 |
| **Conflicting authority claims** | dueling instructions | Stricter rule 113 | Committee | Both citations | `GOVPREC` | Precedent fix |
| **Emergency authority abuse** | Emergency past 72h | Revoke emergency | Committee+Exec | Sunset miss | Rollback acts | EMERGENCY class failure |
| **Governance ownership confusion** | Incident/decision unowned | Assign named owner | Lead | `INC-*` | RACI in 96 | Operational gap |

---

# Card 6 — Authority Safeguards & Anti-Drift Model

| Risk | Why dangerous | Failure consequence | Escalation | Corrective expectation |
|------|---------------|---------------------|------------|------------------------|
| **Authority creep** | Scope grows silently | Ultra vires | Committee | Revoke + narrow `GOVDELEG` |
| **Power concentration** | Single ratifier | Key-person | 111 | Mandatory secondary |
| **Accountability gaps** | Delegate only | No principal | Lead | Principal named always |
| **Role confusion** | Title ≠ rights | Wrong halt | 93 refresh | Cert 97 |
| **Governance capture** | One faction owns votes | Trust loss | Committee+Exec | Quorum rotation |
| **Delegation inflation** | Everyone delegate | Noise | 125 | Tier guide |
| **Escalation avoidance** | Delegation to avoid up-tier | SLA miss | 120 | Escalate mandatory |

---

# Card 7 — Authority Review & Reassessment Model

```
Review authority assignments → Assess accountability quality → Committee review
→ Escalate authority concern → Document observations → Adjust delegation where justified → Reassess governance clarity
```

---

### Review authority assignments

| Field | Detail |
|-------|--------|
| **Purpose** | Inventory |
| **Required actions** | Annual: all open `GOVDELEG`; delegate roster; crisis names |
| **What NOT to do** | Stale roster post-`GOVSUCC` |
| **Escalation expectation** | Expired → auto-revoke |
| **Evidence expectation** | Assignment register |

---

### Assess accountability quality

| Field | Detail |
|-------|--------|
| **Purpose** | Principal-delegate health |
| **Required actions** | Ultra vires sample; override SoD |
| **What NOT to do** | Ignore near-miss bypass |
| **Escalation expectation** | Pattern → Card 5 |
| **Evidence expectation** | Accountability scorecard |

---

### Committee review

| Field | Detail |
|-------|--------|
| **Purpose** | Institutional ack |
| **Required actions** | Annual authority agenda (106) |
| **What NOT to do** | Skip NON_DELEGABLE re-read |
| **Escalation expectation** | Quorum |
| **Evidence expectation** | `GOVCOMM-*` |

---

### Escalate authority concern

| Field | Detail |
|-------|--------|
| **Purpose** | Fix before breach |
| **Required actions** | Card 5 playbook |
| **What NOT to do** | Informal fix |
| **Escalation expectation** | Per scenario |
| **Evidence expectation** | `GOVDELEG` revoke record |

---

### Document observations

| Field | Detail |
|-------|--------|
| **Purpose** | Memory |
| **Required actions** | 107 if external; `GOVPREC` if novel |
| **What NOT to do** | Oral-only revoke |
| **Evidence expectation** | Observation memo |

---

### Adjust delegation where justified

| Field | Detail |
|-------|--------|
| **Purpose** | Right-size |
| **Required actions** | 98 if matrix change; new `GOVDELEG` |
| **What NOT to do** | Expand without sunset |
| **Escalation expectation** | Committee MATERIAL+ |
| **Evidence expectation** | Effective date |

---

### Reassess governance clarity

| Field | Detail |
|-------|--------|
| **Purpose** | Close loop |
| **Required actions** | 126 scale fit; operator survey |
| **What NOT to do** | Clarity assumed |
| **Escalation expectation** | 122 if OCR drops |
| **Evidence expectation** | Clarity memo |

---

# Card 8 — Humility, Power & Stewardship Model

| Risk | Why dangerous | Failure consequence | Escalation | Corrective expectation |
|------|---------------|---------------------|------------|------------------------|
| **Authority arrogance** | “My call” culture | Bypass | 116 | Second reviewer |
| **Power accumulation bias** | Collect approvals | Bottleneck + capture | Committee | SoD audit |
| **Founder authority dependency** | Undocumented exec path | 111 failure | `GOVSUCC` | Written delegate |
| **Governance empire building** | Authority as status | Sprawl | 124/125 | Revoke unnecessary |
| **Unchecked discretion** | No scope limits | Ultra vires | Revoke `GOVDELEG` | Scope templates |
| **Delegation complacency** | Never review roster | Expired emergency | Calendar | Card 7 mandatory |
| **Institutional overreach** | Authority beyond mission | Drift | `GOVALIGN` | Narrow scope |

---

# Card 9 — Authority Quick Start

*Under 1-minute comprehension.*

| Stage | Document / action |
|-------|-------------------|
| **Read first** | [Step 93](./Triton_Governance_Roles_Authority_Framework.md) — who approves |
| **Read second** | Card 3 — classify decision-right; check NON_DELEGABLE |
| **Authority references** | Card 2 domain; Card 4 loop |
| **Escalation references** | 90 when class requires up-tier |
| **Delegation references** | Open `GOVDELEG` before acting as delegate |

**Authority mantra:** *Domain → class → matrix → document → monitor—never oral delegation of safeguards.*

---

# Card 10 — Authority Checklist

**Grant or renew delegation**

- [ ] Authority domain identified (Card 2)
- [ ] Decision-right class assigned (Card 3)
- [ ] 93 matrix row cited
- [ ] Constitutional / NON_DELEGABLE check passed
- [ ] Scope, sunset, and principal named
- [ ] Cert/competency verified (97) if operational
- [ ] Committee ack if TEMPORARY Lead+ or COMMITTEE_REQUIRED
- [ ] `GOVDELEG` signed and roster updated
- [ ] Monitoring/reassessment date set

**Revoke delegation**

- [ ] Reason documented
- [ ] Ultra vires acts identified and remediated
- [ ] Successor delegate named if gap remains
- [ ] `GOVSUCC` link if transition-related

---

# Card 11 — Quick Reference Authority Cards

*Under 10-second comprehension.*

| Situation | What to assess | Escalate? | Evidence | Step |
|-----------|----------------|-----------|----------|------|
| **Authority ambiguity concern** | 93 matrix gap | Lead **24h** | Conflict log | 93, **127** |
| **Delegation concern** | `GOVDELEG` scope | Principal | Assignment | **127**, 111 |
| **Escalation bypass concern** | Class vs act | **Yes** — restore tier | ESC log | 90, **127** |
| **Emergency authority concern** | Sunset ≤72h | Committee+Exec | `GOVDELEG` | 108, **127** |
| **Accountability gap concern** | Principal named? | Lead | Roster | **127** |
| **Conflicting authority concern** | `GOVPREC` | Committee | Citations | 121, **127** |
| **Authority concentration concern** | Delegate roster | Committee | 111 list | 111, **127** |
| **Governance ownership concern** | Domain owner | Lead **5bd** | `INC-*` | **127**, 113 |

---

# Card 12 — Authority & Delegation Appendix

### Non-delegable summary (minimum)

- Hard Halt **lift ratification** (Committee+Executive)
- **Constitutional** `GOVCHG` approval path
- **Dual approval** on overrides (cannot collapse to one delegate)
- **Self-approval** prohibition on halts/overrides
- **CLPR** / safeguard waiver
- **Institutional cert grant/revoke** (Committee vote)

### Glossary

| Term | Definition |
|------|------------|
| **Accountability chain** | Principal → delegate → escalation path |
| **Authority domain** | Card 2 bounded area of governance power |
| **Authority drift** | Scope/practice diverges from assignment |
| **Decision rights** | Card 3 classification of who may decide |
| **Delegation** | Written transfer of subset of authority via `GOVDELEG` |
| **Delegation review** | Card 7 periodic validity assessment |
| **Emergency authority** | Time-bound EMERGENCY class only |
| **Escalation boundary** | Tier above which delegate cannot act |
| **Governance ownership** | Named accountable role for domain/decision |
| **Non-delegable authority** | Must remain with statutory holder |

**Record IDs:** `GOVDELEG-*` · `GOVSUCC-*` · `GOVPREC-*`

**Step 93 vs 127:** **93** = role matrix and approvals; **127** = delegation lifecycle, distribution, and drift control.

---

## Document control

| Field | Value |
|-------|-------|
| Owner | Governance Lead |
| Review cycle | Annual Committee authority agenda; per `GOVSUCC` |
| Change authority | Step 98 (`GOVCHG`) for matrix updates |
| Distribution | All governance roles; Committee; Audit |

---

## Verification checklist (Step 127 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Authority philosophy completed | Complete |
| 2 | Authority domains completed (12) | Complete |
| 3 | Decision-rights classifications completed (6) | Complete |
| 4 | Authority operating model completed | Complete |
| 5 | Delegation failure playbooks completed (9) | Complete |
| 6 | Anti-drift safeguards completed (7) | Complete |
| 7 | Authority reassessment model completed | Complete |
| 8 | Humility/power stewardship model completed (7) | Complete |
| 9 | Quick start completed | Complete |
| 10 | Checklist completed | Complete |
| 11 | Quick-reference cards completed | Complete |
| 12 | Appendix completed | Complete |
| 13 | No runtime, code, UI, broker, or governance mutation | **Confirmed — documentation only** |
| 14 | Enterprise-grade delegation framework | **Confirmed** |

---

*End of document — Triton Governance Delegation, Authority Distribution & Decision Rights Framework (Step 127)*
