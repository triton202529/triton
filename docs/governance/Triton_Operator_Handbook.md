# Triton Operator Handbook & Institutional Operating Guide

**Document type:** Governance Manual — Operator Handbook
**System:** Triton Institutional Trading Platform — Governance Command Center (GCC)
**Classification:** Internal — Operator / Senior Operator
**Version:** 1.0
**Status:** Manual-ready — Daily-use SOP
**Navigation hub:** [Governance README (Step 101)](./README.md)
**Authority manuals:** [Steps 90–100](./README.md#card-2--governance-manual-master-index) — this handbook **summarizes**; linked steps **govern** on conflict

---

## Card 1 — Handbook Philosophy

### Purpose of the Operator Handbook

This handbook is the **daily operating surface** for Triton operators. It turns the governance library (Steps 90–101) into **repeatable shift discipline**: what to do each session, when to escalate, how to contain, and how to document—without improvising policy.

Use this handbook **at the console**. Use linked Steps for **full authority and templates**.

### Why operators exist inside Triton governance

Operators are the **institutional sensors and first responders** for governance posture. You translate GCC signals into **documented action** so capital, integrity, and audit trails are protected when models, pipelines, or markets stress the system.

> **Operators execute governance discipline, not personal trading judgment.**

### Core principles

| Principle | Operator meaning |
|-----------|------------------|
| **Governance before execution** | Read Operator Decision Brief before any execution-related action |
| **Capital Preservation Doctrine supremacy** | When uncertain: contain, observe, escalate—do not “push through” |
| **Escalation before intervention** | Ask the chain; do not edit governance files or enable runtime |
| **Containment-first operations** | Soft/Hard halt and lock preservation over convenience |
| **Evidence-first discipline** | UTC, brief state, paths, incident IDs—every time |
| **Procedural consistency** | Same brief → same playbook command |
| **Calm, non-emotional operations** | Severity and SOP, not urgency narratives |

### What operators are responsible for

- Opening GCC and reading **Final Operator Brief**, **Immediate Instruction**, **Blocked Condition**, **Watch Condition**
- Executing the matching Step 91 posture command
- Daily governance summary and watch-state logging (Step 96)
- Initiating **Soft Halt** when triggers met; **Hard Halt** when mandatory (then notify chain)
- Classifying and documenting incidents Level 1–2; escalating Level 3+ signals
- Preserving evidence before remediation
- Escalating when uncertain within SLA (**30 minutes** trading hours to Senior Operator)
- Participating in drills and maintaining certification (Step 97)

### What operators are NOT responsible for

- Approving overrides or lifting Hard Halt
- Enabling runtime or relaxing constitutional lock without authorized chain
- Modifying governance JSON, memory artifacts, or manuals locally
- Broker discretionary trades to “fix” governance anxiety
- Declaring institutional maturity, readiness, or “safe to trade”
- Replacing Risk Lead, Committee, or Executive decisions

---

## Card 2 — Operator Daily Operating Model

Repeat every shift. Full charter loop: [Step 100](./Triton_Governance_Constitution_Operating_Charter.md).

```
Observe → Interpret → Escalate if needed → Contain → Document → Review → Continue monitoring
```

---

### Observe

| Field | Detail |
|-------|--------|
| **Purpose** | Establish current institutional posture |
| **Operator actions** | Open GCC → Operator Decision Brief; note UTC; confirm refresh acceptable |
| **What NOT to do** | Act on cockpit strips alone; skip Blocked Condition |
| **Escalation expectation** | Stale GCC / dashboard down **>4h** → System Administrator path (notify Senior Operator) |
| **Evidence expectation** | Brief label + UTC in shift log |

---

### Interpret

| Field | Detail |
|-------|--------|
| **Purpose** | Map brief to playbook and watch state |
| **Operator actions** | Step 91 posture matrix; assign GWS (Card 3); note contradictions |
| **What NOT to do** | Reclassify severity down without Senior Operator |
| **Escalation expectation** | Material contradiction **>4h** → Risk Lead |
| **Evidence expectation** | One-line interpretation in daily summary |

---

### Escalate if needed

| Field | Detail |
|-------|--------|
| **Purpose** | Involve correct role before harm |
| **Operator actions** | Use Card 6 chain; open escalation report when trigger fires |
| **What NOT to do** | Skip levels; oral-only handoff |
| **Escalation expectation** | Per Card 6 SLAs |
| **Evidence expectation** | `GOVRPT-ESC-*` or incident link; GCC snapshot |

---

### Contain

| Field | Detail |
|-------|--------|
| **Purpose** | Limit exposure while truth is established |
| **Operator actions** | Soft Halt per Step 91/90; Hard Halt if integrity trigger—notify immediately |
| **What NOT to do** | “Test” runtime; override policy |
| **Escalation expectation** | Hard Halt → full Level 4 notify (Card 6) |
| **Evidence expectation** | Halt UTC, trigger one-liner |

---

### Document

| Field | Detail |
|-------|--------|
| **Purpose** | Institutional memory and audit |
| **Operator actions** | Daily summary (Step 96); incident template L2+ (Step 90) |
| **What NOT to do** | Close incidents without approver |
| **Escalation expectation** | Cannot complete evidence → escalate, do not guess |
| **Evidence expectation** | Paths, IDs, timelines |

---

### Review

| Field | Detail |
|-------|--------|
| **Purpose** | Confirm nothing material changed unnoticed |
| **Operator actions** | End-of-shift: open escalations, watch conditions, handoff note |
| **What NOT to do** | Assume next shift will notice |
| **Escalation expectation** | Open L3+ → handoff to Risk Lead in writing |
| **Evidence expectation** | Handoff UTC + state |

---

### Continue monitoring

| Field | Detail |
|-------|--------|
| **Purpose** | Maintain observability until next cadence |
| **Operator actions** | Scheduled GCC refresh; Watch logging |
| **What NOT to do** | Obsessive refresh (over-management per Step 91) |
| **Escalation expectation** | Watch → Elevated triggers (Card 3) |
| **Evidence expectation** | Card 9 daily checklist complete |

---

## Card 3 — Governance Watch States for Operators

*Under 15-second comprehension. Full detail: [Step 99](./Triton_Governance_Observability_Monitoring_Framework.md).*

| State | What it means | What you do | What you avoid | Escalate? | Cadence |
|-------|---------------|-------------|----------------|-----------|---------|
| **NORMAL** | KPIs healthy; no material watch | Routine monitoring; daily summary | Complacency; skipping log | Only standard triggers | Scheduled refresh |
| **WATCH** | Leading stress or GCC Watch Condition | Explicit watch log; prep escalation packet | Ignoring Watch Condition | Uncertainty → Senior Op **30m** | Increased refresh |
| **ELEVATED** | Multiple warnings or Elevated KPI | Heightened playbook; Soft Halt bias near windows | Policy improvisation | Risk Lead **4h** on entry | Lead + you daily |
| **DEGRADED** | GHS weak or GOVERNANCE_REPAIR_REQUIRED | **ESCALATE_AND_CONTAIN**; full logging | Runtime enablement | Risk Lead owns | Continuous until down |
| **CRITICAL** | GHS critical, HHF, safeguard breach | Lock posture; preserve evidence; no lift | Any override without chain | Executive same day | Continuous |

**Rule:** If GCC brief and watch state disagree, apply **stricter** containment and notify Senior Operator.

---

## Card 4 — Incident Response Playbook

Master flow when **something went wrong**:

```
Observe → Step 90 (classify) → Step 91 (command) → Escalate if required → Contain → Document
```

---

### General — something went wrong

| Field | Detail |
|-------|--------|
| **When to use** | Any anomaly affecting governance, data, or execution eligibility |
| **Operator actions** | UTC timestamp; preserve state; read brief; classify severity (Step 90) |
| **Escalation trigger** | Level 3+ or uncertainty → Senior Operator **30m** |
| **Documentation** | L1 log minimum; L2+ full `INC-*` template |

---

### Governance contradiction

| Field | Detail |
|-------|--------|
| **When to use** | Lifecycle vs rationale vs signal vs GCC misalignment |
| **Operator actions** | Log material vs informational; consider Soft Halt if run window |
| **Escalation trigger** | Material **>4h** → Risk Lead |
| **Documentation** | Contradiction note + brief screenshots |

---

### Escalation event

| Field | Detail |
|-------|--------|
| **When to use** | You need higher authority or SLA exceeded |
| **Operator actions** | Card 6 chain; one-page summary + evidence |
| **Escalation trigger** | No response in SLA → next level |
| **Documentation** | `GOVRPT-ESC-*` linked to incident if any |

---

### Soft halt consideration

| Field | Detail |
|-------|--------|
| **When to use** | Stale data, lifecycle mismatch, repair posture + imminent window |
| **Operator actions** | Initiate Soft Halt; notify Senior Operator |
| **Escalation trigger** | Not resolved **4h** → Risk Lead |
| **Documentation** | Halt/lift UTC; four restart conditions before lift |

---

### Hard halt escalation

| Field | Detail |
|-------|--------|
| **When to use** | Integrity, unauthorized override, duplicate execution, broker unexplained delta |
| **Operator actions** | **Hard Halt**; notify chain **15 min**; do not lift |
| **Escalation trigger** | **Immediate** Level 4 path |
| **Documentation** | Start `INC-*`; forensic preservation list |

---

### Override request

| Field | Detail |
|-------|--------|
| **When to use** | Someone asks to bypass a control |
| **Operator actions** | Log request; **do not approve**; route Risk Lead |
| **Escalation trigger** | Constitutional path → Committee awareness |
| **Documentation** | Override exception report draft (Step 96) |

---

### Governance deterioration

| Field | Detail |
|-------|--------|
| **When to use** | GHS down, KPI Watch+, false stability (quiet halts, bad leading indicators) |
| **Operator actions** | Card 3 Elevated/DEGRADED; no readiness narrative |
| **Escalation trigger** | CRITICAL watch → Executive visibility via Lead |
| **Documentation** | Daily summary + KPI flags referenced |

---

## Card 5 — Operator Decision Guide

*Under 10-second comprehension. Full manuals linked.*

| Question | Go to |
|----------|-------|
| **What do I do right now?** | [Step 91](./Triton_Governance_Operator_Decision_Playbook.md) — match Final Operator Brief |
| **Something feels wrong?** | [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) + [Step 99](./Triton_Governance_Observability_Monitoring_Framework.md) |
| **Who approves this?** | [Step 93](./Triton_Governance_Roles_Authority_Framework.md) |
| **How bad is governance health?** | [Step 92](./Triton_Governance_Metrics_KPI_Framework.md) |
| **Can I lift this halt?** | Step 93 — probably **no** (Operator); escalate |
| **Can governance be changed?** | [Step 98](./Triton_Governance_Change_Management_Framework.md) — not you alone |
| **What do I report today?** | [Step 96](./Triton_Governance_Reporting_Audit_Framework.md) daily summary |
| **Am I certified for this shift?** | [Step 97](./Triton_Governance_Training_Certification_Framework.md) |
| **What are the supreme rules?** | [Step 100](./Triton_Governance_Constitution_Operating_Charter.md) |
| **Where is everything?** | [README](./README.md) |

---

## Card 6 — Operator Escalation Handbook

Chain (full authority: [Step 93](./Triton_Governance_Roles_Authority_Framework.md)):

```
Operator → Senior Operator → Risk / Governance Lead → Triton System Administrator
         → Governance Committee → Executive Oversight (Critical)
```

*System Administrator is parallel for technical recovery—not a skip of Risk Lead for trading risk.*

---

### To Senior Operator

| Field | Detail |
|-------|--------|
| **When to escalate** | Uncertainty on severity; recurring Level 1; WATCH persistence; TRANSITION_WATCH ambiguity |
| **Required evidence** | GCC snapshot, UTC, one-line summary |
| **Expected response** | **30 min** trading hours; triage and classify |
| **What NOT to do** | Self-approve halt lift; hide recurrence |

---

### To Risk / Governance Lead

| Field | Detail |
|-------|--------|
| **When to escalate** | GOVERNANCE_REPAIR_REQUIRED; Level 3 signals; reconciliation mismatch; material contradiction **>4h** |
| **Required evidence** | Incident record start, dossier summary, reconciliation if applicable |
| **Expected response** | **30 min** Level 3; **15 min** Level 4 notify assist |
| **What NOT to do** | Request overrides verbally without record |

---

### To System Administrator

| Field | Detail |
|-------|--------|
| **When to escalate** | Pipeline/dashboard down **>4h**; missing artifacts (with Senior Operator awareness) |
| **Required evidence** | Preflight logs, error output, artifact paths |
| **Expected response** | **4h** Level 2 target |
| **What NOT to do** | Ask Admin to bypass governance controls |

---

### To Governance Committee

| Field | Detail |
|-------|--------|
| **When to escalate** | Constitutional violation; unauthorized override; active Level 4 (with Lead) |
| **Required evidence** | Full Step 90 package |
| **Expected response** | **24h** convene; immediate if active Critical |
| **What NOT to do** | Operator direct unless Lead directs |

---

### To Executive Oversight

| Field | Detail |
|-------|--------|
| **When to escalate** | Level 4; Hard Halt active; CLPR breach—not routine operator path |
| **Required evidence** | Executive summary from Lead; containment status |
| **Expected response** | **15 min** notification on L4 |
| **What NOT to do** | Pressure Executive to lift without Committee package |

---

## Card 7 — Operator Do / Don't Playbook

| DO | DON'T |
|----|-------|
| Read Operator Decision Brief **first** | Act on strips alone |
| **Observe** at scheduled cadence | Enable runtime without authorization |
| **Log** UTC + brief state every shift | Improvise governance policy |
| **Escalate** when uncertain (**30m**) | Self-approve halt lift or override |
| **Contain** (Soft Halt) when Step 91/90 triggers | Edit governance JSON or memory |
| **Preserve** evidence before fixes | Trade or broker-adjust from anxiety |
| **Follow** Step 91 immediate instruction | Treat LOCKED_OBSERVE_ONLY as failure |
| **Use** `INC-*` template for Level 2+ | Assume STABLE means execution authorized |
| **Confirm** Blocked Condition each session | Refresh GCC obsessively |
| **Notify** Senior Operator on WATCH persistence | Escalate emotionally |
| **Complete** daily checklist (Card 9) | Hide contradictions or halts from log |
| **Maintain** current certification | Substitute personal judgment for SOP |

---

## Card 8 — Operator Quick Start

*Under 1-minute read. Full onboarding: [Step 97](./Triton_Governance_Training_Certification_Framework.md).*

| Stage | Document / action |
|-------|-------------------|
| **Read first** | This handbook Card 2 + [Step 91](./Triton_Governance_Operator_Decision_Playbook.md) posture matrix |
| **Read second** | [Step 90](./Triton_Governance_Incident_Escalation_Framework.md) L1–2 + Soft Halt |
| **Daily references** | GCC brief → 91; Card 9 checklist; [Step 96](./Triton_Governance_Reporting_Audit_Framework.md) daily summary |
| **Escalation references** | Card 6; [Step 93](./Triton_Governance_Roles_Authority_Framework.md) quick cards |
| **Advanced references** | [Step 99](./Triton_Governance_Observability_Monitoring_Framework.md), [Step 92](./Triton_Governance_Metrics_KPI_Framework.md), [README](./README.md) |

**First shift mantra:** *Brief → Command → Log → Escalate if unsure.*

---

## Card 9 — Operator Daily Checklist

Complete before handoff. Check `[ ]` or note N/A with reason.

### Session open

- [ ] GCC opened; Operator Decision Brief read
- [ ] **Final Operator Brief** recorded (UTC)
- [ ] **Immediate Instruction** understood
- [ ] **Blocked Condition** confirmed (e.g., runtime enablement blocked)
- [ ] **Watch Condition** noted; GWS assigned (Card 3)
- [ ] Certification current (`GOVCERT`)

### During shift

- [ ] Scheduled GCC refresh completed
- [ ] Contradictions reviewed; material ones logged
- [ ] Escalation queue: none open or all have owner + UTC
- [ ] Observability warnings (Step 99) reviewed if ELEVATED+
- [ ] Any Soft/Hard halt logged with trigger
- [ ] No unauthorized override requests approved

### Session close

- [ ] **Daily governance summary** filed (Step 96)
- [ ] Open incidents: status updated
- [ ] Handoff note to next operator (if applicable)
- [ ] Governance health / GHS snapshot if weekly duty
- [ ] Drill or training item if scheduled (Step 97/95)

**If any item cannot be completed:** escalate per Card 6—do not mark complete.

---

## Card 10 — Quick Reference Operator Cards

*Under 10-second comprehension.*

| Situation | What to do | Escalate? | Evidence | Step |
|-----------|------------|-----------|----------|------|
| **Incident** | Classify; contain; template L2+ | Per L1–4 | `INC-*`, UTC | 90, 91 |
| **Contradiction** | Log; Soft Halt if window | Material **>4h** → Lead | Brief snap | 91, 99 |
| **Override request** | Log; route Lead | Yes | Request memo | 93, 96 |
| **Governance deterioration** | Elevated watch; contain | Lead **4h** | KPI flags | 92, 99 |
| **Soft halt concern** | Initiate; document | Sr Op; 4h Lead | Halt UTC | 90, 91 |
| **Hard halt concern** | Hard Halt; notify | **Immediate** L4 | Forensic list | 90, 93 |
| **Escalation confusion** | Senior Op **30m** | Yes | One-line + brief | 93, 6 |
| **Reporting question** | Daily summary fields | Lead if template gap | Step 96 paths | 96 |

---

## Card 11 — Operator Handbook Appendix

Standard terms (full glossary: [Step 100](./Triton_Governance_Constitution_Operating_Charter.md)).

| Term | Operator definition |
|------|---------------------|
| **Blocked Condition** | What you must **not** do (check every session) |
| **Capital Preservation Doctrine** | Uncertain → contain, observe, escalate |
| **Constitutional safeguard** | Lock, dual approval, halt discipline—non-negotiable |
| **Containment** | Observe, Soft Halt, or Hard Halt to limit harm |
| **Contradiction** | Governance signals disagree—log and escalate material |
| **Escalation event** | You invoked chain above routine monitoring |
| **Final Operator Brief** | GCC label that picks Step 91 playbook |
| **Governance drift** | Informal exceptions—refuse; use Step 98 path |
| **Governance Health Score (GHS)** | 0–100 health index—not permission to trade |
| **Governance Watch State (GWS)** | NORMAL → CRITICAL—your session risk lens (Card 3) |
| **Hard Halt** | Full stop; you may initiate; you **do not** lift |
| **Immediate Instruction** | Single command from Step 91—follow it |
| **Override** | Exception path—you **do not** approve |
| **Soft Halt** | Pause execution; keep watching; document lift criteria |
| **Watch Condition** | GCC line telling you what to monitor until next refresh |

### Posture → command (Step 91 summary)

| Final Operator Brief | Command |
|---------------------|---------|
| LOCKED_OBSERVE_ONLY | MAINTAIN_LOCK_AND_OBSERVE |
| LOCKED_HEIGHTENED_MONITORING | MONITOR_AND_PREPARE_ESCALATION |
| GOVERNANCE_REPAIR_REQUIRED | ESCALATE_AND_CONTAIN |
| TRANSITION_WATCH | OBSERVE_VALIDATE_AND_REASSESS |
| STABLE_CONTINUE_MONITORING | ROUTINE_MONITORING_ONLY |

---

## Document control

| Field | Value |
|-------|-------|
| Owner | Governance Lead |
| Review cycle | Quarterly or when Step 91 material change |
| Change authority | [Step 98](./Triton_Governance_Change_Management_Framework.md) |
| Distribution | Operator Manual, training packets (Step 97), shift handoff |

---

## Verification checklist (Step 102 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Handbook philosophy completed | Complete |
| 2 | Daily operating model completed | Complete |
| 3 | Watch states completed | Complete |
| 4 | Incident playbooks completed | Complete |
| 5 | Decision guide completed | Complete |
| 6 | Escalation handbook completed | Complete |
| 7 | Do/Don't playbook completed | Complete |
| 8 | Quick start completed | Complete |
| 9 | Daily checklist completed | Complete |
| 10 | Quick-reference cards completed | Complete |
| 11 | Appendix completed | Complete |
| 12 | No runtime, code, UI, broker, or governance mutation | **Confirmed — documentation only** |
| 13 | Enterprise-grade operator usability | **Confirmed** |

---

*End of document — Triton Operator Handbook & Institutional Operating Guide (Step 102). Daily use: Card 2 + Card 9 + GCC + Step 91.*
