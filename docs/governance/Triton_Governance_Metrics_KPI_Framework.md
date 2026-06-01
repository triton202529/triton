# Triton Governance Metrics, KPI & Institutional Health Framework

**Document type:** Governance Manual — Metrics, KPI & Institutional Health
**System:** Triton Institutional Trading Platform
**Classification:** Internal — Operator / Developer / Audit / Executive
**Version:** 1.0
**Status:** Manual-ready SOP
**Companion documents:**
- [Triton Governance Incident & Escalation Framework](./Triton_Governance_Incident_Escalation_Framework.md) (Step 90)
- [Triton Governance Operator Decision Playbook](./Triton_Governance_Operator_Decision_Playbook.md) (Step 91)

---

## Purpose

This framework answers:

> **How healthy is Triton governance over time?**

It formalizes measurement standards for:

- governance health scoring
- institutional reliability tracking
- operator discipline measurement
- escalation effectiveness tracking
- auditability KPIs
- governance deterioration detection
- institutional oversight metrics
- constitutional safeguard effectiveness

This document is **observational and procedural**. It defines how to measure, interpret, and report governance health. It does **not** enable execution, mutate governance, modify runtime policy, or automate any control path.

**Capital Preservation Doctrine** dominates all thresholds and interpretations: when metrics conflict with capital safety, **contain, observe, and escalate** before resuming normal operations.

---

## Scope

**Applies to:**

- Governance Command Center (GCC) posture and artifacts
- Operator logs, incident records, and escalation dossiers
- ARM governance pipeline observability (readiness through human escalation)
- Lifecycle, rationale, and signal consistency reviews
- Soft Halt, Hard Halt, and override discipline
- Audit and post-incident review completeness

**Does not:**

- implement metrics in code
- alter runtime, broker, or execution paths
- replace formal incident classification (Step 90)
- substitute executive or regulatory reporting obligations

---

# Section 1 — Governance Health Score

## Definition

The **Governance Health Score (GHS)** is a composite institutional index on a **0–100** scale representing the overall reliability, discipline, and auditability of Triton governance over a defined measurement window (default: rolling 30 calendar days unless otherwise specified for executive reporting).

GHS is a **management and oversight instrument**, not a trading signal and not an authorization to enable runtime.

## Conceptual components

GHS is derived conceptually from weighted assessment of:

| Component | What it reflects |
|-----------|------------------|
| Trustworthiness | GCC trust band stability and absence of trust breaks |
| Auditability | Incident documentation, timeline integrity, evidence preservation |
| Governance coherence | Alignment between lifecycle, rationale, signals, and GCC brief |
| Contradiction severity | Frequency and intensity of internal governance contradictions |
| Escalation stability | Predictable, justified escalation; low false-escalation burden |
| Evidence integrity | Confidence claims supported by preserved artifacts |
| Operator discipline | SOP adherence, logging completeness, halt discipline |
| Intervention quality | Containment proportionate to severity; no improvisation |
| Incident frequency | Count and severity-weighted rate of governance incidents |
| Override behavior | Rate, justification quality, dual approval, expiration compliance |

Component weights are assigned by **Governance Lead** in quarterly calibration; default posture is **equal weighting** with **override and Hard Halt** components capped so a single severe event cannot be masked by stable ancillary metrics.

## Scoring methodology (conceptual)

1. **Normalize** each component to 0–100 for the measurement window using KPI thresholds (Section 3).
2. **Apply** severity caps: any **Critical** KPI forces component floor ≤ 25 for that component.
3. **Aggregate** using agreed weights; round to nearest integer.
4. **Map** to Health State (below).
5. **Document** score, window, data sources, and reviewer in governance metrics log.

Scores are **point-in-time** and **trended**; a single healthy reading does not erase recent Critical events without documented recovery (see Governance Recovery Rate, Section 2).

## Health states

| State | GHS range | Definition |
|-------|-----------|------------|
| **CRITICAL** | 0–39 | Governance reliability is materially compromised. Multiple severe KPI breaches or a Level 4-equivalent event in-window. |
| **DEGRADED** | 40–59 | Sustained weakness in discipline, coherence, or escalation. Elevated incident or override pressure. |
| **GUARDED** | 60–74 | Functional but fragile. Watch conditions dominate; recovery not yet proven. |
| **HEALTHY** | 75–89 | Stable discipline, acceptable incident load, evidence-backed confidence. |
| **INSTITUTIONAL_GRADE** | 90–100 | Sustained excellence across components; no Critical KPI in-window; audit-ready posture. |

### CRITICAL

**Definition:** Governance cannot be relied upon for institutional confidence without immediate executive awareness and containment review.

**Implications:**

- Assume **elevated capital and control risk** until proven otherwise.
- Default GCC posture treated as **minimum** containment; no relaxation without committee path.
- All discretionary runtime paths treated as **prohibited** unless explicitly authorized per Step 90/91.

**Operator expectations:**

- **MAINTAIN_LOCK_AND_OBSERVE** or stricter per Operator Decision Brief.
- Full incident logging; preserve artifacts; no governance JSON edits.
- Daily metrics review until state improves.

**Escalation expectations:**

- Notify **Risk / Governance Lead** within 30 minutes of classification.
- **Executive Authority** notified same business day.
- Governance Committee briefing within 24 hours if GHS remains CRITICAL.

### DEGRADED

**Definition:** Material weakness in one or more governance dimensions; deterioration trend likely without intervention.

**Implications:**

- Increased scrutiny on contradictions, overrides, and audit gaps.
- Soft Halt posture favored when execution windows approach.

**Operator expectations:**

- Execute Step 91 repair or heightened monitoring playbooks as briefed.
- Increase GCC refresh cadence per Section 8.
- Document all escalations with evidence pointers.

**Escalation expectations:**

- Risk / Governance Lead within 4 hours on first entry to DEGRADED.
- Weekly executive summary until recovery to GUARDED or better.

### GUARDED

**Definition:** Governance is operating but under stress; leading indicators may be worsening while lagging indicators remain quiet.

**Implications:**

- **False stability** risk: do not confuse absence of halts with health.
- Watch KPIs (contradiction rate, alignment stability, false escalation) closely.

**Operator expectations:**

- Routine monitoring with **explicit watch logging**.
- Prepare escalation materials before triggers fire.

**Escalation expectations:**

- Senior Operator awareness continuous; Governance Lead on weekly review.
- Escalate on Watch → Elevated Risk KPI transitions (Section 3).

### HEALTHY

**Definition:** Governance discipline and coherence meet institutional minimums; incidents are rare, contained, and well-documented.

**Implications:**

- Normal operator cadence; constitutional lock may still apply (health ≠ runtime enablement).
- Continue trending analysis; do not reduce audit discipline.

**Operator expectations:**

- **ROUTINE_MONITORING_ONLY** or equivalent per GCC brief.
- Maintain logging and SLA adherence.

**Escalation expectations:**

- Standard incident path only; no standing executive briefing required.

### INSTITUTIONAL_GRADE

**Definition:** Sustained demonstration of governance excellence suitable for audit sampling and executive attestation.

**Implications:**

- Suitable for institutional oversight packages and external audit narrative support.
- Does **not** authorize policy relaxation or standing overrides.

**Operator expectations:**

- Preserve discipline that produced the score; no efficiency shortcuts.
- Mentor and cross-check operator compliance metrics.

**Escalation expectations:**

- Quarterly executive attestation; immediate escalation on any Critical KPI breach (score cap applies).

---

# Section 2 — Core Governance KPIs

All KPIs use a **measurement window** (default: rolling 30 days) unless noted. Numerators and denominators must be **documented** in the metrics log. Severity weighting for incidents follows Step 90 (Level 1 = 1, Level 2 = 2, Level 3 = 4, Level 4 = 8).

---

### 1. Governance Stability Rate (GSR)

**Meaning:** Percentage of time governance posture remained stable without unplanned state transition attributable to deterioration (excluding scheduled reviews and documented maintenance).

**Formula (conceptual):**
`GSR = (hours in stable posture / total measured hours) × 100`

**Stable posture** includes GCC briefs equivalent to `LOCKED_OBSERVE_ONLY`, `STABLE_CONTINUE_MONITORING`, and other states explicitly classified as non-deteriorating in the metrics log.

**Data sources:** GCC posture history, operator logs, incident timeline.

---

### 2. Escalation Frequency (EF)

**Meaning:** Count of human escalations per period (operator-initiated escalations above routine monitoring).

**Includes:** Escalations to Senior Operator, Risk / Governance Lead, Governance Committee, Executive Authority per Step 90 chain.

**Excludes:** Routine scheduled governance refresh with no escalation trigger.

---

### 3. Governance Incident Rate (GIR)

**Meaning:** Incident frequency per period, **severity-weighted**.

**Formula (conceptual):**
`GIR = Σ (incident count × severity weight) / period length in weeks`

---

### 4. Governance Contradiction Rate (GCR)

**Meaning:** Frequency of internal governance contradictions (lifecycle vs rationale vs signal vs GCC brief misalignment) per review cycle or per 100 GCC refreshes.

**Document:** Contradiction intensity (informational vs material) per operator playbook.

---

### 5. Evidence Integrity Reliability (EIR)

**Meaning:** Percentage of governance confidence statements or operator conclusions that were **evidence-supported** (artifact path, timestamp, reproducible observation) at time of decision.

**Formula (conceptual):**
`EIR = (supported decisions / total sampled decisions) × 100`

**Sampling:** Minimum 10 decisions per window or 100% if fewer than 10 escalations/incidents.

---

### 6. Override Frequency (OF)

**Meaning:** Count of emergency or exceptional overrides per period (any control listed in Step 90 Section 5).

**Warning:** Rising OF is a **governance integrity** indicator, not an efficiency metric.

---

### 7. Soft Halt Frequency (SHF)

**Meaning:** Count of Soft Halt activations per period.

**Context:** Soft Halt is containment; elevated SHF may indicate rising operational or governance stress.

---

### 8. Hard Halt Frequency (HHF)

**Meaning:** Count of Hard Halt activations per period.

**Context:** Any HHF > 0 in-window requires executive visibility and post-incident review per Step 90.

---

### 9. Operator Compliance Rate (OCR)

**Meaning:** Percentage of operator actions audited against SOP that fully adhered to Step 91 playbooks, logging requirements, and halt discipline.

**Formula (conceptual):**
`OCR = (compliant actions / audited actions) × 100`

---

### 10. Governance Recovery Rate (GRR)

**Meaning:** Percentage of governance deterioration events that achieved **documented recovery** to GUARDED or better within agreed recovery SLA without recurrence in 14 days.

**Recovery** requires: root cause addressed or downgraded, GCC posture confirmed, validation checklist complete.

---

### 11. False Escalation Rate (FER)

**Meaning:** Percentage of escalations later deemed **unnecessary** upon review (no material risk, no SOP trigger, retrospective Level 1 equivalent).

**Formula (conceptual):**
`FER = (unnecessary escalations / total escalations) × 100`

**Note:** Low FER is not pursued at expense of Capital Preservation; err toward escalation when uncertain.

---

### 12. Governance Alignment Stability (GAS)

**Meaning:** Durability of consensus across lifecycle, rationale, signals, and GCC brief without material realignment events.

**Formula (conceptual):**
`GAS = (cycles without material realignment / total cycles) × 100`

---

### 13. Review Timeliness (RT)

**Meaning:** Percentage of required reviews (incident, post-override, scheduled governance, executive summary) completed within SLA.

**Formula (conceptual):**
`RT = (on-time reviews / due reviews) × 100`

---

### 14. Audit Coverage Rate (ACR)

**Meaning:** Percentage of incidents and material events with **full documentation** per Step 90 template (timeline, impact, approvals, validation, lessons learned).

**Formula (conceptual):**
`ACR = (fully documented incidents / total incidents) × 100`

---

### 15. Constitutional Lock Preservation Rate (CLPR)

**Meaning:** Percentage of measurement window during which constitutional safeguards were **respected** (no unauthorized runtime enablement, no prohibited governance JSON mutation, no override without dual approval).

**Formula (conceptual):**
`CLPR = (hours without safeguard violation / total hours) × 100`

**Violation:** Any confirmed breach forces KPI to **Critical** regardless of percentage.

---

# Section 3 — KPI Thresholds

Thresholds are **conservative**. When a KPI spans two bands, apply the **more severe** band. Capital Preservation Doctrine: at **Critical**, default action is **contain and escalate**, not interpret optimistically.

### 1. Governance Stability Rate

| Band | Threshold |
|------|-----------|
| Healthy | ≥ 98% |
| Watch | 95–97.9% |
| Elevated Risk | 90–94.9% |
| Critical | < 90% |

### 2. Escalation Frequency

| Band | Threshold (per 30 days) |
|------|---------------------------|
| Healthy | 0–2 |
| Watch | 3–5 |
| Elevated Risk | 6–10 |
| Critical | > 10 |

### 3. Governance Incident Rate (severity-weighted per week)

| Band | Threshold |
|------|-----------|
| Healthy | < 2 |
| Watch | 2–4 |
| Elevated Risk | 5–8 |
| Critical | > 8 **or** any Level 4 in-window |

### 4. Governance Contradiction Rate (per 100 GCC cycles)

| Band | Threshold |
|------|-----------|
| Healthy | 0–1 material |
| Watch | 2–3 material |
| Elevated Risk | 4–6 material |
| Critical | > 6 **or** any unresolved material > 24h |

### 5. Evidence Integrity Reliability

| Band | Threshold |
|------|-----------|
| Healthy | ≥ 98% |
| Watch | 95–97.9% |
| Elevated Risk | 90–94.9% |
| Critical | < 90% |

### 6. Override Frequency

| Band | Threshold (per 30 days) |
|------|---------------------------|
| Healthy | 0 |
| Watch | 1 |
| Elevated Risk | 2–3 |
| Critical | > 3 **or** any override without dual approval |

### 7. Soft Halt Frequency

| Band | Threshold (per 30 days) |
|------|---------------------------|
| Healthy | 0–1 |
| Watch | 2–3 |
| Elevated Risk | 4–6 |
| Critical | > 6 |

### 8. Hard Halt Frequency

| Band | Threshold (per 30 days) |
|------|---------------------------|
| Healthy | 0 |
| Watch | 0 (N/A — any HHF triggers Watch minimum on GHS) |
| Elevated Risk | 1 |
| Critical | ≥ 2 **or** 1 with open post-incident review |

*Any Hard Halt in-window: minimum GHS cap **GUARDED** until executive review complete.*

### 9. Operator Compliance Rate

| Band | Threshold |
|------|-----------|
| Healthy | ≥ 99% |
| Watch | 97–98.9% |
| Elevated Risk | 95–96.9% |
| Critical | < 95% |

### 10. Governance Recovery Rate

| Band | Threshold |
|------|-----------|
| Healthy | ≥ 90% |
| Watch | 80–89.9% |
| Elevated Risk | 70–79.9% |
| Critical | < 70% |

### 11. False Escalation Rate

| Band | Threshold |
|------|-----------|
| Healthy | < 10% |
| Watch | 10–20% |
| Elevated Risk | 21–35% |
| Critical | > 35% *(investigate escalation instability; do not discourage prudent escalation)* |

### 12. Governance Alignment Stability

| Band | Threshold |
|------|-----------|
| Healthy | ≥ 97% |
| Watch | 94–96.9% |
| Elevated Risk | 90–93.9% |
| Critical | < 90% |

### 13. Review Timeliness

| Band | Threshold |
|------|-----------|
| Healthy | ≥ 98% on-time |
| Watch | 95–97.9% |
| Elevated Risk | 90–94.9% |
| Critical | < 90% |

### 14. Audit Coverage Rate

| Band | Threshold |
|------|-----------|
| Healthy | 100% |
| Watch | 95–99.9% |
| Elevated Risk | 90–94.9% |
| Critical | < 90% |

### 15. Constitutional Lock Preservation Rate

| Band | Threshold |
|------|-----------|
| Healthy | 100% |
| Watch | 99.9% *(document near-miss)* |
| Elevated Risk | Any confirmed minor procedural deviation without capital impact |
| Critical | Any unauthorized enablement, mutation, or unapproved override |

---

# Section 4 — Operator Interpretation Guide

Use this guide when KPIs **deteriorate** or **trend adversely**. Always cross-check GCC Operator Decision Brief before action.

| Signal | What it means | What to investigate | When to escalate |
|--------|---------------|----------------------|------------------|
| **Escalation frequency increasing** | Governance instability or operator uncertainty rising | Recent contradictions, lifecycle drift, stale data, reconciliation warnings | Senior Operator if Watch band; Risk Lead if Elevated or Critical |
| **Override frequency rising** | Governance integrity concern; exception path becoming normalized | Justification quality, dual approval, expiration, post-override reviews | Risk Lead on first Elevated; Committee + Executive on Critical |
| **Audit coverage weakening** | Governance discipline degradation; forensic gap risk | Open incidents, missing timelines, unsigned validations | Governance Lead within 4h if < 95%; Executive if Critical |
| **Contradiction rate rising** | Internal model misalignment; false stability risk | Lifecycle vs rationale vs signal; GCC watch conditions | Senior Operator at Watch; Risk Lead if material unresolved > 4h |
| **Evidence integrity falling** | Decisions not reproducible; audit defensibility eroding | Artifact paths, screenshot retention, dossier completeness | Governance Lead at Elevated; immediate training/containment at Critical |
| **Soft Halt frequency rising** | Containment used more often; stress before Hard Halt | Scheduled windows, stale outputs, pipeline SLA | Risk Lead if Elevated; prepare Hard Halt materials |
| **Hard Halt any occurrence** | Capital or control boundary breach | Root cause, broker, reconciliation, idempotency | Level 4 path per Step 90 immediately |
| **Operator compliance falling** | SOP drift; playbook bypass risk | Sampled actions vs Step 91; weekend rules | Governance Lead at Watch; remedial stand-down at Critical |
| **Recovery rate falling** | Deterioration not closing; repeat events | Prior incident IDs, recurrence within 14d | Committee if < 80% with Level 3+ history |
| **False escalation rate high** | Escalation instability or unclear triggers | Playbook clarity, GCC brief ambiguity, training | Governance Lead (process fix); **do not** reduce prudence without Lead approval |
| **Alignment stability falling** | Consensus not durable | Signal churn, lifecycle edits, rationale conflicts | Risk Lead at Elevated |
| **Review timeliness slipping** | Oversight backlog | Open post-override reviews, incident closures | Governance Lead at Watch; Executive summary delay flag at Critical |
| **CLPR below 100%** | Safeguard stress or violation | Override logs, runtime paths, JSON edit audit | **Immediate** Executive path on any violation |
| **GSR declining with flat incident rate** | Possible **false stability** | Leading indicators (contradictions, coherence) | Increase monitoring cadence; Senior Operator briefing |

### Default operator posture on ambiguous deterioration

1. **Observe** — refresh GCC; do not mutate runtime.
2. **Contain** — Soft Halt if execution window exists per Step 91.
3. **Document** — log KPI signal, evidence, brief state.
4. **Escalate** — per table above; prefer early escalation for trading-risk signals.

---

# Section 5 — Governance Trend Analysis

## Objectives

Determine whether governance is **improving**, **deteriorating**, **stable**, exhibiting **false stability**, or responding to **interventions**.

## Analysis method

1. Plot KPIs over **7d, 30d, 90d** windows (minimum).
2. Compare **leading** vs **lagging** indicators (below).
3. Correlate with **GCC posture timeline** and **incident IDs**.
4. Annotate **interventions** (training, process change, halt, override) on timeline.
5. Classify trend: improving / stable / deteriorating / false stable.
6. Record conclusion in governance metrics log with reviewer signature.

## Trend classifications

| Classification | Pattern | Interpretation |
|----------------|---------|----------------|
| **Improving** | Leading and lagging KPIs bettering; GHS up ≥ 5 pts over 30d; no Critical open | Recovery credible; maintain discipline |
| **Deteriorating** | Leading KPIs worsening 2+ periods; GHS down ≥ 5 pts; or lagging KPIs rising | Pre-incident phase likely; escalate preparedness |
| **Stable** | KPIs within Healthy/Watch; GHS variance < 3 pts | Normal operations; continue cadence |
| **False stability** | Lagging flat while leading worsen (GCR, GAS, EIR) | **High risk** — halts may be delayed; intensify monitoring |
| **Intervention effective** | Post-intervention KPI improvement within SLA; GRR met | Document what worked; do not relax safeguards early |
| **Intervention ineffective** | No improvement 14d post-intervention; recurrence | Escalate to Committee; root-cause review mandatory |

## Leading indicators

Signals that often **precede** incidents and halts:

- Governance Contradiction Rate increase
- Governance Alignment Stability decrease
- Evidence Integrity Reliability decrease
- Escalation Frequency increase without incident closure
- Governance Stability Rate decline
- Watch-condition persistence in GCC brief

## Lagging indicators

Signals that **confirm** outcomes after stress:

- Hard Halt Frequency
- Governance Incident Rate (severity-weighted)
- Override Frequency
- Audit Coverage Rate (often drops after fast response)
- Governance Recovery Rate (post-event)
- GHS state transitions to DEGRADED or CRITICAL

## Rule

**Never** declare institutional health from lagging indicators alone when leading indicators are Elevated or Critical.

---

# Section 6 — Executive Governance Scorecard

**Audience:** Executive Authority, Governance Committee
**Read time:** Under 1 minute
**Cadence:** Weekly (minimum); immediate append on Level 4 or Hard Halt

```
TRITON EXECUTIVE GOVERNANCE SCORECARD
Period: [YYYY-MM-DD] to [YYYY-MM-DD]     Prepared by: [Role]     UTC: [timestamp]

GOVERNANCE HEALTH SCORE: [0-100] — [CRITICAL | DEGRADED | GUARDED | HEALTHY | INSTITUTIONAL_GRADE]

ONE-LINE POSTURE: [e.g., Constitutional lock maintained; elevated contradiction watch]

INCIDENT TREND:     [↓ | → | ↑]  Severity-weighted GIR: [value]  Open Level 3+: [N]
ESCALATION TREND:   [↓ | → | ↑]  EF: [N]  FER: [%]
OVERRIDE TREND:     [↓ | → | ↑]  OF: [N]  Last override: [date or NONE]
HALT POSTURE:       Soft [N] / Hard [N] in-window
AUDIT DISCIPLINE:   ACR [%]  RT [%]  Open doc gaps: [N]
OPERATOR RELIABILITY: OCR [%]  Compliance actions due: [N]
CONSTITUTIONAL SAFEGUARDS: CLPR [%]  Violations: [NONE | describe]

TOP RISK FLAGS (if any): [flag codes from Section 7]

EXECUTIVE ACTION REQUIRED: [ YES — describe | NO ]

Next review: [date]     Escalation owner: [name/role]
```

**Executive decision prompts (yes/no):**

- Is GHS acceptable for current capital deployment posture?
- Are any Critical KPIs unaddressed > 24h?
- Is override trend institutional or episodic?
- Is audit coverage sufficient for attestation?

Default when **no** to safeguard questions: **maintain containment** per Capital Preservation Doctrine.

---

# Section 7 — Governance Risk Flags

Automatic **concern categories** for metrics review. Flags do not change runtime; they mandate **documented response**.

| Flag code | Definition | Trigger condition (any) | Severity | Response expectation |
|-----------|------------|---------------------------|----------|------------------------|
| **OVERRIDE_DEPENDENCY_RISK** | Normal operations increasingly rely on exceptions | OF Elevated/Critical 2 periods; or >1 override same control | High | Risk Lead review; Committee if repeats; freeze new overrides pending review |
| **ESCALATION_INSTABILITY** | Escalation volume or quality inconsistent with risk | EF Critical; or FER Elevated with EF rising | Medium–High | Playbook/training review; Senior Operator triage audit |
| **AUDIT_DISCIPLINE_BREAKDOWN** | Incidents not fully documented | ACR < 95% or RT < 95% | High | Stop closure of open incidents until docs complete; Governance Lead ownership |
| **CONSTITUTIONAL_GUARD_WEAKENING** | Safeguards stressed or breached | CLPR < 100% or any violation | Critical | Executive + Committee; Hard Halt posture until cleared |
| **FALSE_STABILITY_PATTERN** | Lagging calm, leading stress | GCR or GAS Elevated while GIR/HHF flat 14d | High | Heightened monitoring; mandatory Senior Operator sign-off each shift |
| **RECOVERY_FAILURE** | Deterioration not closing | GRR Critical; or same root cause twice in 30d | High | Committee root-cause session within 5 business days |
| **EVIDENCE_DEFICIT** | Decisions lack defensible artifacts | EIR < 95% | Medium–High | Remedial logging; sample review expanded |
| **HALT_ESCALATION_LADDER** | Containment trending toward Hard Halt | SHF Elevated 2 periods; or HHF ≥ 1 | High | Risk Lead daily stand-up until trend reverses |
| **OPERATOR_SOP_DRIFT** | Playbook discipline slipping | OCR < 97% | Medium | Supervised shifts; retraining before override authority |
| **ALIGNMENT_FRACTURE** | Governance consensus breaking | GAS Critical; or material contradiction > 24h | High | Risk Lead; lifecycle/rationale freeze recommendation to Committee |

**Severity response times:**

| Severity | Governance Lead | Executive |
|----------|-----------------|-----------|
| Critical | Immediate | Same business day |
| High | 4 hours | Next executive scorecard |
| Medium | Next daily review | Weekly summary |

---

# Section 8 — Review Cadence Framework

| Cadence | Metrics reviewed | Primary reviewer | Secondary / oversight | SLA |
|---------|------------------|------------------|------------------------|-----|
| **Daily** | GHS snapshot, open flags, SHF/HHF, critical contradictions, CLPR | Operator | Senior Operator on Elevated+ | Complete within 30 min of shift start |
| **Weekly** | Full KPI set (30d window), trends, EF, OF, ACR, RT, executive scorecard draft | Governance Lead | Risk / Governance Lead sign-off | Complete by Monday 12:00 UTC |
| **Monthly** | GHS official, all KPI thresholds, GRR, OCR sample, intervention effectiveness | Governance Lead | Governance Committee summary | By 3rd business day of month |
| **Quarterly** | Weight calibration, threshold review, institutional attestation package | Governance Committee | Executive Authority | Within 10 business days of quarter end |

### Role responsibilities

| Role | Responsibility |
|------|----------------|
| **Operator** | Daily capture, GCC correlation, incident KPI inputs, escalate on trigger |
| **Senior Operator** | Validate operator logs; confirm halt/override counts; Watch band triage |
| **Governance Lead** | Official KPI scoring, trend analysis, flag assignment, monthly report |
| **Executive** | Scorecard decisions; Critical/Level 4 visibility; resource for remediation |

### SLAs (review timeliness inputs)

| Review type | SLA |
|-------------|-----|
| Daily metrics log | Same shift |
| Weekly governance metrics report | 5 business days max lateness: **Critical KPI** |
| Post-override review | 24h (Level 3+) / 72h (Level 2) per Step 90 |
| Post-incident closure with full audit | 5 business days Level 3+; 10 Level 2 |
| Quarterly attestation | Committee meeting + archived scorecard |

---

# Section 9 — KPI Quick Reference Cards

*Format: under 10-second comprehension. Bands use default 30-day window.*

---

**Governance Stability Rate**
Healthy: ≥98% | Watch: 95–98% | Critical: <90%
**Action:** If Critical → Senior Operator + trend log; check false stability.

---

**Escalation Frequency**
Healthy: 0–2 | Watch: 3–5 | Critical: >10
**Action:** If Watch+ → map to incident types; Risk Lead if Elevated.

---

**Governance Incident Rate**
Healthy: <2/wk weighted | Watch: 2–4 | Critical: >8 or Level 4
**Action:** Critical → Step 90 Level 4 path; preserve evidence.

---

**Governance Contradiction Rate**
Healthy: 0–1 material / 100 cycles | Watch: 2–3 | Critical: >6
**Action:** Material unresolved >4h → Risk Lead.

---

**Evidence Integrity Reliability**
Healthy: ≥98% | Watch: 95–98% | Critical: <90%
**Action:** Critical → no escalation closure without artifacts.

---

**Override Frequency**
Healthy: 0 | Watch: 1 | Critical: >3
**Action:** Any unapproved → CONSTITUTIONAL_GUARD flag + Executive.

---

**Soft Halt Frequency**
Healthy: 0–1 | Watch: 2–3 | Critical: >6
**Action:** Elevated → review execution windows; prepare Hard Halt.

---

**Hard Halt Frequency**
Healthy: 0 | Elevated: 1 | Critical: ≥2
**Action:** Any HHF → executive notification + post-incident review.

---

**Operator Compliance Rate**
Healthy: ≥99% | Watch: 97–99% | Critical: <95%
**Action:** Critical → supervised operations; training before overrides.

---

**Governance Recovery Rate**
Healthy: ≥90% | Watch: 80–90% | Critical: <70%
**Action:** Critical → Committee root-cause within 5 business days.

---

**False Escalation Rate**
Healthy: <10% | Watch: 10–20% | Critical: >35%
**Action:** High FER → clarify triggers; **do not** suppress prudent escalation.

---

**Governance Alignment Stability**
Healthy: ≥97% | Watch: 94–97% | Critical: <90%
**Action:** Elevated → lifecycle/rationale coherence review.

---

**Review Timeliness**
Healthy: ≥98% | Watch: 95–98% | Critical: <90%
**Action:** Critical → backlog owners assigned within 24h.

---

**Audit Coverage Rate**
Healthy: 100% | Watch: 95–99% | Critical: <90%
**Action:** Critical → freeze incident closure; Governance Lead owns backlog.

---

**Constitutional Lock Preservation Rate**
Healthy: 100% | Critical: any violation
**Action:** Violation → Hard Halt posture review + Executive same day.

---

**Governance Health Score (composite)**
Healthy: 75–89 | Guarded: 60–74 | Critical: 0–39
**Action:** State drives Section 1 escalation expectations; scorecard Section 6.

---

## Document control

| Field | Value |
|-------|-------|
| Owner | Governance Lead |
| Review cycle | Quarterly |
| Next review | Per quarterly cadence |
| Change authority | Governance Committee |
| Distribution | Operator Manual, Developer Manual, Governance SOPs, Audit/Compliance, Executive oversight |

---

## Verification checklist (Step 92 completion)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Sections 1–9 created | Complete |
| 2 | Governance Health Score documented (0–100, components, methodology) | Complete |
| 3 | KPI framework completed (15 core KPIs) | Complete |
| 4 | Thresholds documented (Healthy / Watch / Elevated Risk / Critical) | Complete |
| 5 | Operator interpretation guide completed | Complete |
| 6 | Executive scorecard completed | Complete |
| 7 | Governance risk flags completed | Complete |
| 8 | Review cadence framework completed | Complete |
| 9 | KPI quick reference cards completed | Complete |
| 10 | No runtime, code, UI, broker, or governance mutation | **Confirmed — documentation only** |
| 11 | Enterprise-grade SOP quality (institutional tone, audit-ready structure) | **Confirmed** |

---

*End of document — Triton Governance Metrics, KPI & Institutional Health Framework (Step 92)*
