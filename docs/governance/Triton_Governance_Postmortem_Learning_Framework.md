# Triton Governance Failure Postmortems, Institutional Learning & Near-Miss Intelligence Framework

**Document type:** Governance Manual — Postmortems, Institutional Learning & Near-Miss Intelligence
**System:** Triton Institutional Trading Platform — Governance Command Center (GCC)
**Classification:** Internal — All governance roles / Committee / Executive / Audit
**Version:** 1.0
**Status:** Manual-ready SOP
**Navigation hub:** [Governance README (Step 101)](./README.md)
**Related manuals:** [Step 90 Incident & Escalation](./Triton_Governance_Incident_Escalation_Framework.md) · [Step 112 Meta-Governance](./Triton_Governance_Meta_Governance_Framework.md) · [Step 111 Continuity](./Triton_Governance_Institutional_Memory_Succession_Framework.md)

---

## Scope disclaimer

This framework governs **how Triton learns from failures, governance errors, and near-misses**—systematic postmortems, root-cause discipline, and anti-repeat safeguards—without blame theater, denial, or HR performance enforcement.

> **Institutional learning improves resilience — not guaranteed future outcomes.**

**Postmortem record ID:** `GOVPM-YYYY-MM-DD-###` — links to `INC-*` (90), `GOVWAR-*` (109), `GOVMETA-*` (112), prevention owners in 96/107 index.

**Not blame assignment:** Postmortems identify **system and process gaps**; personnel accountability paths are outside this manual unless tied to documented SoD breach (93).

**Documentation only.** No code, UI, broker, execution, or governance JSON mutation.

---

# Card 1 — Failure Learning Philosophy

### Purpose of governance postmortems

Postmortems convert **painful events into durable institutional knowledge**—so the same governance failure does not require the same crisis to teach the same lesson twice.

### Why Triton treats failures and near-misses as institutional learning opportunities

| Denial pattern | Learning outcome |
|----------------|------------------|
| “Operator error only” | Root cause includes SOP, training, tooling gaps |
| Close incident without memo | `GOVPM` + 107 index |
| Hide near-miss | Card 6 intelligence lost |
| Repeat root cause 2× | REPEAT_FAILURE_RISK → Committee |
| Success = no drills | Near-miss from war game valued (109) |
| Blame replaces fix | Card 5 anti-blame; `GOVMETA` process fix |

### Core principles

| Principle | Learning meaning |
|-----------|------------------|
| **Capital Preservation Doctrine supremacy** | Contain first; learn second—never reverse order |
| **Learning before blame** | System diagnosis before personnel narrative |
| **Truth before comfort** | Full timeline; no severity cosmetics |
| **Containment before denial** | Acknowledge failure class early |
| **Evidence-first diagnosis** | Artifacts drive root cause |
| **Institutional humility** | Near-misses are gifts (Card 8) |
| **Antifragility through learning** | Each failure strengthens safeguards if closed properly |
| **Transparency with discipline** | Share lessons internally; prudent external disclosure (117) |

### What institutional learning proves

- Events are **classified** (Card 2) and processed through **Card 3 loop**
- Root causes address **governance**, not only individuals
- Lessons feed **98/112** and **111** memory
- Repeat risks are **escalated** before third occurrence
- Near-misses are **logged** without waiting for loss (Card 6)

### What institutional learning cannot guarantee

- Zero future failures or near-misses
- Perfect root cause on first pass
- Immediate culture change
- Forgiveness of intentional misconduct (escalate 116)
- That learning alone prevents market losses

---

# Card 2 — Failure & Near-Miss Classification Framework

Seven classes. Near-misses **always** merit `GOVPM` lite or full per severity of what almost happened.

---

### NEAR_MISS_EVENT

| Field | Detail |
|-------|--------|
| **Definition** | Harm avoided by containment, timing, or luck—governance stress present |
| **Observed signal** | Almost lifted halt; SLA almost missed; drill “almost failed” |
| **Escalation implication** | Lead review **5bd**; Committee if constitutional brush |
| **Learning expectation** | Mandatory near-miss memo; optional full `GOVPM` |
| **Failure implication** | Unlogged near-miss → repeat real failure |
| **Containment expectation** | Treat as warning; no ridicule of reporter |

---

### GOVERNANCE_FAILURE_EVENT

| Field | Detail |
|-------|--------|
| **Definition** | Documented governance process failed (wrong tier, missing evidence) |
| **Observed signal** | Playbook deviation; audit finding on process |
| **Escalation implication** | Committee if material |
| **Learning expectation** | Full `GOVPM` within post-incident SLA (90) |
| **Failure implication** | Drift (112) |
| **Containment expectation** | Revert to effective manual version |

---

### CONTAINED_FAILURE_EVENT

| Field | Detail |
|-------|--------|
| **Definition** | Loss or harm occurred but halts/escalation limited impact |
| **Observed signal** | L2–L3 with successful Hard/Soft Halt |
| **Escalation implication** | Per 90 tier + post-mortem |
| **Learning expectation** | Full `GOVPM`; celebrate containment, fix cause |
| **Failure implication** | Complacency if only “we halted” |
| **Containment expectation** | Do not skip learning because damage was small |

---

### ESCALATION_BREAKDOWN_EVENT

| Field | Detail |
|-------|--------|
| **Definition** | Wrong owner, SLA miss, or skipped level |
| **Observed signal** | Timeout; Executive surprised by Critical |
| **Escalation implication** | Lead **48h**; Committee if pattern |
| **Learning expectation** | Chain audit; 97 retrain |
| **Failure implication** | Crisis mismanagement |
| **Containment expectation** | Restore correct tier immediately |

---

### CONSTITUTIONAL_FAILURE_EVENT

| Field | Detail |
|-------|--------|
| **Definition** | Safeguard breach, unauthorized lift, CLPR failure |
| **Observed signal** | 118 CAPITAL_PRESERVATION_BREACH; 116 integrity |
| **Escalation implication** | Committee+Exec **immediate** |
| **Learning expectation** | Full `GOVPM` + mandatory `GOVCHG` or rollback |
| **Failure implication** | Cert revoke (110) |
| **Containment expectation** | Hard Halt until ratified path |

---

### REPEAT_FAILURE_RISK

| Field | Detail |
|-------|--------|
| **Definition** | Same root cause category within **12 months** |
| **Observed signal** | Second `GOVPM` matching prior root tag |
| **Escalation implication** | Committee **5bd**; REPEAT_FAILURE mandatory |
| **Learning expectation** | Systemic fix; war game (109) |
| **Failure implication** | LEARNING_DEFICIT_EVENT |
| **Containment expectation** | Hold maturity/readiness promotion |

---

### LEARNING_DEFICIT_EVENT

| Field | Detail |
|-------|--------|
| **Definition** | Postmortem skipped, hollow, or lessons not implemented |
| **Observed signal** | Open prevention owner overdue; same drill fail twice |
| **Escalation implication** | Committee |
| **Learning expectation** | Re-open `GOVPM`; audit 107 index |
| **Failure implication** | Institutional forgetfulness |
| **Containment expectation** | Moratorium on related `GOVCHG` until closed |

---

# Card 3 — Postmortem Operating Model

```
Identify failure or near-miss → Contain immediate risk → Preserve evidence
→ Classify governance implications → Committee review (if tier requires)
→ Root-cause analysis → Lessons documented → Safeguards strengthened → Readiness reassessed
```

---

### Identify failure or near-miss

| Field | Detail |
|-------|--------|
| **Purpose** | Start learning clock |
| **Required actions** | Open `GOVPM-*`; link `INC-*` if exists; classify Card 2 |
| **What NOT to do** | Wait for “big enough” failure |
| **Escalation expectation** | Reporter protected; Lead triage **24h** |
| **Evidence expectation** | Initial narrative + UTC |

---

### Contain immediate risk

| Field | Detail |
|-------|--------|
| **Purpose** | Stop ongoing harm |
| **Required actions** | 90/108 containment; halt if capital at risk |
| **What NOT to do** | Debate root cause before contain |
| **Escalation expectation** | Live crisis per 108 |
| **Evidence expectation** | Halt/containment log |

---

### Preserve evidence

| Field | Detail |
|-------|--------|
| **Purpose** | Enable honest diagnosis |
| **Required actions** | Snapshot GCC, logs, comms; 107 retention rules |
| **What NOT to do** | Delete or edit artifacts |
| **Escalation expectation** | GOVERNANCE_INTEGRITY_RISK if tampering (116) |
| **Evidence expectation** | Evidence index in `GOVPM` |

---

### Classify governance implications

| Field | Detail |
|-------|--------|
| **Purpose** | Route severity of learning |
| **Required actions** | Map 90 tier; 116/117/118 if applicable |
| **What NOT to do** | Under-classify to avoid Committee |
| **Escalation expectation** | CONSTITUTIONAL → Exec+Committee |
| **Evidence expectation** | Classification table |

---

### Committee review

| Field | Detail |
|-------|--------|
| **Purpose** | Institutional acceptance of findings |
| **Required actions** | L3+ or CONSTITUTIONAL: Committee within SLA; quorum 106 |
| **What NOT to do** | Oral postmortem only |
| **Escalation expectation** | REPEAT_FAILURE_RISK always Committee |
| **Evidence expectation** | `GOVCOMM-*` minutes |

---

### Root-cause analysis

| Field | Detail |
|-------|--------|
| **Purpose** | Systemic truth |
| **Required actions** | Five-whys or equivalent; tag root category |
| **What NOT to do** | Stop at “human error” without system layer |
| **Escalation expectation** | Lead facilitates; independent reviewer if L4 |
| **Evidence expectation** | RCA document in `GOVPM` |

---

### Lessons documented

| Field | Detail |
|-------|--------|
| **Purpose** | Institutional memory |
| **Required actions** | Prevention owners; due dates; 111/107 index |
| **What NOT to do** | Vague “be more careful” |
| **Escalation expectation** | `GOVMETA` if manual gap |
| **Evidence expectation** | Lesson table with owners |

---

### Safeguards strengthened

| Field | Detail |
|-------|--------|
| **Purpose** | Anti-repeat |
| **Required actions** | `GOVCHG`, drill 95/109, training 97 as needed |
| **What NOT to do** | Loosen safeguards as “lesson learned” |
| **Escalation expectation** | 98 path for changes |
| **Evidence expectation** | `GOVCHG` or drill pass record |

---

### Readiness reassessed

| Field | Detail |
|-------|--------|
| **Purpose** | Honest institutional posture |
| **Required actions** | 110/114 hold or regress if warranted |
| **What NOT to do** | Certify through failure window |
| **Escalation expectation** | Committee on institutional impact |
| **Evidence expectation** | Readiness memo |

---

# Card 4 — Failure Postmortem Playbooks

| Scenario | What happened | Immediate containment | Escalation | Evidence | Learning expectation | Recovery |
|----------|---------------|----------------------|------------|----------|----------------------|----------|
| **Governance breakdown** | Process collapse | Halt; observe | Committee **5bd** | Timeline | Full `GOVPM`; 112 drift check | 60d monitoring |
| **Escalation failure** | SLA/chain break | Correct tier now | Lead **48h** | ESC log | Chain audit; 97 | Drill pass |
| **Crisis mismanagement** | 108 path not followed | Crisis cell | Committee+Exec | `GOVCRISIS-*` | Full `GOVPM`; 109 scenario | Normalization per 108 |
| **Fiduciary error** | Capital duty breach | Hard Halt | 118 `GOVCAP` | Halt, recon | Stewardship lesson | 90d KPI stable |
| **Trust erosion event** | LP/audit shock | Transparent plan | 117 `GOVTRUST` | 107 pack | Communication discipline | Remediation closed |
| **Constitutional safeguard failure** | CLPR/bypass | Hard Halt | Committee+Exec | CLPR log | Mandatory rollback/tighten | War game |
| **Governance drift event** | Practice ≠ manuals | Effective version | 112 | Version register | `GOVCHG` clarify | OCR recovery |
| **Repeated operator mistake** | Same ops error 2× | Supervised mode | REPEAT_FAILURE | Training records | SOP fix not only retrain | Third → Committee |
| **Successful near-miss containment** | Harm avoided | Acknowledge reporter | Lead **5bd** | Near-miss memo | Card 6 intelligence feed | Optional drill inject |

---

# Card 5 — Root Cause & Anti-Repeat Failure Model

| Risk | Why dangerous | Failure consequence | Escalation | Corrective expectation |
|------|---------------|---------------------|------------|------------------------|
| **Blame culture** | Honesty suppressed | Hidden near-misses | Lead culture review | Learning-before-blame enforced |
| **Repeated governance mistakes** | No system fix | REPEAT_FAILURE | Committee | Root tag tracking 12m |
| **Surface-level diagnosis** | Symptom fixes only | Third occurrence | Independent reviewer L4 | Five-whys to system |
| **Denial behavior** | “Not governance” | Trust loss | 116/117 | Classify honestly Card 2 |
| **Evidence suppression** | RCA impossible | Integrity risk | Committee **48h** | Forensic preserve 107 |
| **False closure** | Checkbox postmortem | LEARNING_DEFICIT | Committee | Prevention owner verified |
| **Institutional forgetfulness** | Lessons not indexed | Same crisis twice | 111 archive | 107 + quarterly review |

**Anti-repeat rule:** Same **root tag** twice in 12 months triggers **REPEAT_FAILURE_RISK** Committee session before any readiness or maturity promotion.

---

# Card 6 — Near-Miss Intelligence & Early Warning Model

| Signal type | Why monitored | Failure consequence | Escalation | Learning expectation |
|-------------|---------------|---------------------|------------|----------------------|
| **Almost-failures** | Cheapest teacher | Real failure next | Lead log | `GOVPM` lite within 5bd |
| **Escalation delays** | Chain fragility | SLA breach | 90 ESC audit | Trend in quarterly review |
| **Near constitutional breaches** | Safeguard almost lost | Next time real | Committee if brush | 109 inject scenario |
| **Preservation doctrine pressure** | Capital almost compromised | Breach | `GOVCAP` | 118 lesson link |
| **Trust deterioration warnings** | External confidence | Redemption | `GOVTRUST` | 117 corrective narrative |
| **Governance ambiguity** | Wrong action under stress | Incident | 113 interpretation | `GOVCHG` clarify |
| **Crisis preparation weaknesses** | Drill near-fail | Live crisis | 109 remediation | Update `GOVFORE` (115) |

**Near-miss register:** Lead maintains rolling **90-day** near-miss log linked to `GOVPM` lite entries—fed to 99 GWS review and Card 7 quarterly.

---

# Card 7 — Postmortem Review & Learning Reassessment Model

```
Review evidence → Assess institutional learning → Committee review
→ Escalate repeat-risk concern → Document lessons → Strengthen safeguards → Reassess readiness & maturity
```

---

### Review evidence

| Field | Detail |
|-------|--------|
| **Purpose** | Verify postmortem quality |
| **Required actions** | Quarterly: all open `GOVPM`; closed within SLA |
| **What NOT to do** | Accept RCA without artifacts |
| **Escalation expectation** | LEARNING_DEFICIT → Committee |
| **Evidence expectation** | Postmortem completeness score |

---

### Assess institutional learning

| Field | Detail |
|-------|--------|
| **Purpose** | Measure anti-repeat |
| **Required actions** | Root tag trend; prevention owner closure rate |
| **What NOT to do** | Count postmortems without implementation |
| **Escalation expectation** | REPEAT_FAILURE → Committee |
| **Evidence expectation** | Learning dashboard |

---

### Committee review

| Field | Detail |
|-------|--------|
| **Purpose** | Oversight of learning system |
| **Required actions** | Annual learning agenda; L3+ postmortems sampled |
| **What NOT to do** | Skip near-miss aggregate review |
| **Escalation expectation** | Quorum for REPEAT_FAILURE actions |
| **Evidence expectation** | `GOVCOMM-*` |

---

### Escalate repeat-risk concern

| Field | Detail |
|-------|--------|
| **Purpose** | Stop third strike |
| **Required actions** | Open REPEAT_FAILURE packet |
| **What NOT to do** | Hope training alone suffices |
| **Escalation expectation** | Committee **5bd** |
| **Evidence expectation** | Prior `GOVPM` links |

---

### Document lessons

| Field | Detail |
|-------|--------|
| **Purpose** | Memory |
| **Required actions** | 107 index; 111 if succession-relevant |
| **What NOT to do** | Lessons only in email |
| **Escalation expectation** | `GOVMETA` for process |
| **Evidence expectation** | Published lesson memo |

---

### Strengthen safeguards

| Field | Detail |
|-------|--------|
| **Purpose** | Antifragile response |
| **Required actions** | 98/95/109 as per RCA |
| **What NOT to do** | Weaken to “move on” |
| **Escalation expectation** | Constitutional tighten freely |
| **Evidence expectation** | Closure proof |

---

### Reassess readiness & maturity

| Field | Detail |
|-------|--------|
| **Purpose** | Honest posture |
| **Required actions** | 110/114 per impact |
| **What NOT to do** | Promote through open REPEAT_FAILURE |
| **Escalation expectation** | Committee vote |
| **Evidence expectation** | Hold/regress memo |

---

# Card 8 — Humility, Failure & Antifragility Model

| Risk | Why dangerous | Failure consequence | Escalation | Corrective expectation |
|------|---------------|---------------------|------------|------------------------|
| **Perfectionism culture** | Near-misses hidden | Surprise failure | Lead | Celebrate reporting |
| **Blame politics** | Scapegoating | Repeat | 116 if bad faith | System RCA required |
| **Founder denial** | “Not our governance” | Trust loss | Committee+Exec | Documented attestation |
| **Governance ego** | Failures dismissed | Drift | 112 review | Public internal lessons |
| **Institutional complacency** | Skip postmortems | LEARNING_DEFICIT | Committee | Calendar SLA enforcement |
| **Repeated fragility** | Same tag 3× | Systemic | Committee emergency | Structural `GOVCHG` |
| **Success arrogance** | No near-miss logging | Tail event | 109 mandatory | Quiet quarter ≠ healthy |

**Antifragility rule:** A **failed war game** with documented fixes is **more valuable** than a quiet quarter with no learning records.

---

# Card 9 — Postmortem Quick Start

*Under 1-minute comprehension.*

| Stage | Document / action |
|-------|-------------------|
| **Read first** | [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) — contain + `INC-*` then this Card 3 |
| **Read second** | Card 2 — classify event (near-miss vs failure) |
| **Postmortem references** | Open `GOVPM`; Card 4 playbook for scenario |
| **Escalation references** | Tier per 90; Committee for L3+/CONSTITUTIONAL |
| **Learning references** | Card 5 anti-repeat; `GOVMETA` (112) for manual fixes |

**Postmortem mantra:** *Contain → Preserve → Classify → Learn → Fix system—never blame before evidence.*

---

# Card 10 — Postmortem Checklist

**Open `GOVPM` (all material failures; near-misses per Card 2)**

- [ ] Failure classified (Card 2)
- [ ] Evidence preserved (107 rules)
- [ ] Root cause documented (system + human layers)
- [ ] Escalation path reviewed (was chain correct?)
- [ ] Learning captured (owners, dates)
- [ ] Safeguards reassessed (`GOVCHG` / drill if needed)
- [ ] Repeat-risk evaluated (12m root tag)
- [ ] Institutional implications (110, 114, 117, 118) documented
- [ ] Committee review completed if required
- [ ] Near-miss register updated if applicable

**Close `GOVPM`**

- [ ] Prevention owners verified complete
- [ ] Lesson indexed (107 / 111)
- [ ] Readiness/maturity impact recorded
- [ ] No false closure (Card 5)

---

# Card 11 — Quick Reference Postmortem Cards

*Under 10-second comprehension.*

| Situation | What to assess | Escalate? | Evidence | Step |
|-----------|----------------|-----------|----------|------|
| **Near miss concern** | What almost happened | Lead **5bd** | Near-miss memo | 119 |
| **Governance failure concern** | Process break | Committee if material | `GOVPM` | 119, 90 |
| **Repeat mistake concern** | Root tag match | Committee **5bd** | Prior `GOVPM` | 119 |
| **Escalation breakdown concern** | SLA/chain | Lead **48h** | ESC log | 90, 119 |
| **Fiduciary failure concern** | Capital duty | 118 `GOVCAP` | Halt log | 118, 119 |
| **Trust erosion event** | Stakeholder impact | 117 | 107/96 | 117, 119 |
| **Constitutional failure concern** | Safeguard breach | Committee+Exec | CLPR | 100, 119 |
| **Learning deficit concern** | Hollow PM | Committee | Open owners | 119 |

---

# Card 12 — Postmortem & Learning Appendix

### Postmortem SLA (governance)

| Event class | `GOVPM` open | Committee | Close target |
|-------------|--------------|-----------|--------------|
| NEAR_MISS (material) | 5bd | If constitutional brush | 15bd |
| GOVERNANCE_FAILURE | 3bd | If material | Per 90 post-incident |
| CONTAINED_FAILURE | With `INC-*` | L3+ | Per 90 |
| CONSTITUTIONAL_FAILURE | 1bd | Immediate | 30bd + safeguards |
| REPEAT_FAILURE_RISK | 2bd | **5bd** | Before promotion |

### Glossary

| Term | Definition |
|------|------------|
| **Antifragility** | Institution strengthens through disciplined learning from stress |
| **Contained failure** | Harm limited by halts/escalation—still requires full learning |
| **Governance postmortem** | Structured `GOVPM` review after failure or material near-miss |
| **Governance recovery** | Return to stable posture after containment + lessons implemented |
| **Institutional humility** | Honest accounting of limits and mistakes |
| **Institutional learning** | Durable knowledge from failures and near-misses |
| **Learning deficit** | Postmortem skipped or lessons not implemented |
| **Near-miss intelligence** | Systematic capture of almost-failures (Card 6) |
| **Repeat failure risk** | Same root tag twice in 12 months |
| **Root cause analysis** | Evidence-based systemic diagnosis in `GOVPM` |

**Record IDs:** `GOVPM-*` · `INC-*` · `GOVMETA-*` (process fixes)

**Extended references:** [Step 113 Codex](./Triton_Governance_Codex.md) · [Step 111 Memory](./Triton_Governance_Institutional_Memory_Succession_Framework.md)

---

## Document control

| Field | Value |
|-------|-------|
| Owner | Governance Lead |
| Review cycle | Quarterly learning review; annual Committee learning agenda |
| Change authority | Step 98 (`GOVCHG`) |
| Distribution | All governance roles; Audit |

---

## Verification checklist (Step 119 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Learning philosophy completed | Complete |
| 2 | Failure classifications completed (7) | Complete |
| 3 | Postmortem operating model completed | Complete |
| 4 | Failure playbooks completed (9) | Complete |
| 5 | Anti-repeat model completed (7) | Complete |
| 6 | Near-miss intelligence completed (7) | Complete |
| 7 | Learning reassessment model completed | Complete |
| 8 | Humility/antifragility model completed (7) | Complete |
| 9 | Quick start completed | Complete |
| 10 | Checklist completed | Complete |
| 11 | Quick-reference cards completed | Complete |
| 12 | Appendix completed | Complete |
| 13 | No runtime, code, UI, broker, or governance mutation | **Confirmed — documentation only** |
| 14 | Enterprise-grade learning framework | **Confirmed** |

---

*End of document — Triton Governance Failure Postmortems, Institutional Learning & Near-Miss Intelligence Framework (Step 119)*
