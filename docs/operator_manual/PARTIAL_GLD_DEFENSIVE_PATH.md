# Partial GLD Defensive Path

Operator reference for the **certified partial GLD defensive workflow** (Phases 198C–198K).
This document does **not** authorize paper or live execution.

---

## Certified quantities (Phase 198K)

| Field | Value | Authority |
|-------|-------|-----------|
| Symbol | GLD | Certified partial path |
| Full position quantity | **195** | Position / full defensive candidate |
| Certified partial quantity | **48** | Governed projected dispatch |
| Remaining after partial | **147** | Deferred full exit |
| Raw handoff quantity | May show **195** | Non-authoritative |
| Authoritative dispatch quantity | **48** | 197X / 198I / 198J / 198K |
| Full GLD exit | Deferred | Cap / partial path |
| Execution allowed now | **false** | No-execution certification |
| Approval artifact | Absent | Required before any future paper run |
| Broker authorization artifact | Absent | Required before any future paper run |
| Live execution | **Blocked** | Must remain blocked |

Certification marker: `PARTIAL_GLD_NO_EXECUTION_CERTIFIED`
Evidence: `data/results/phase_198k_partial_gld_no_execution_certification.json`

---

## Why GLD full exit exceeds the cap

A full GLD defensive exit (qty **195**) is part of a larger defensive batch whose notional exceeds the governed placement caps used by `place_live_orders` / manage handoff. Full-batch dispatch is therefore blocked by sizing-cap governance (Phase 197H and related controls). Caps are **not** raised to force a full exit.

---

## Why partial reduction exists

When the full GLD exit would breach the batch notional cap, TRITON builds a **partial defensive exit preview** (Phase 198D): reduce GLD by the governed partial quantity (**48**) and leave the remainder (**147**) deferred. Partial reduction is a controlled, capped alternative—not an expansion of BUY/ADD rights and not authorization to liquidate the full position in one dispatch.

---

## Why raw handoff may show 195

`data/live/manage_orders.csv` may still list GLD at qty **195**. That row is the **full defensive handoff candidate**—the raw management intent before governed sizing, approval, and dispatch-scope narrowing.

Raw handoff quantity is **not** broker dispatch authority.

### Operator warning

> **Raw handoff quantity may represent the full defensive candidate. Broker dispatch must use the governed projected dispatch quantity only.**

---

## Why broker dispatch authority is 48 only

Downstream evidence (197X projected paper orders, 198I dispatch-scope dry-run, 198J quantity reconciliation, 198K certification) locks the **authoritative broker dispatch quantity** at **48**.

- Projected / approved / authoritative partial qty = **48**
- Expanding scope from 48 → 195 is a certification failure and must not be dispatched
- Classification may be: `RAW_HANDOFF_FULL_BUT_DISPATCH_PARTIAL`

### Safe operating rule

> **Never dispatch from raw `manage_orders.csv` alone. Always use the certified 197X / 198J / 198K governed dispatch quantity.**

---

## What remains deferred

- Full GLD exit of the remaining **147** shares
- Any dispatch that would treat raw qty **195** as executable broker scope
- Paper execution until short-lived approval and broker-authorization artifacts exist for qty **48** only
- Live execution under all circumstances for this path

---

## What is approved vs not approved

### Certified (analysis → dispatch-scope evidence only)

- Partial GLD sizing and preview path
- Approval-packet / broker-auth / loop-gate / readiness-bridge **compatibility** (dry-run)
- Dispatch scope locked to qty **48**
- Reconciliation that raw **195** is non-authoritative vs dispatch **48**
- No-execution certification (`execution_allowed_now=false`)

### Not approved / not authorized

- Operator paper approval artifact (`data/governance/operator_paper_defensive_exit_approval.json`)
- Paper broker execution authorization (`data/governance/operator_paper_broker_execution_authorization.json`)
- Broker order submission (paper or live)
- Live execution
- Dispatching full GLD qty **195**
- Loosening BUY/ADD gates

---

## Approvals required before any future paper execution

Before any future **paper-only** attempt on this partial path (separate explicit operator action—not covered by 198K):

1. Short-lived operator paper approval artifact for **partial GLD qty=48 only**
2. Short-lived paper broker execution authorization matching the **projected partial scope (48)**
3. 197Y loop gate `eligible_for_execution=true` for that partial scope only
4. Readiness bridge active for the paper loop-mediated path only
5. Confirm authoritative dispatch qty remains **48** (do not expand to 195)
6. Keep **live execution blocked**

Until those steps are completed deliberately, treat the path as certified for understanding and dry-run evidence only.

---

## Live execution remains blocked

Phase 198K and this operator note **do not** enable live execution. Live remains blocked. Do not create live confirm artifacts, do not run live dispatch, and do not treat certification as permission to trade.

---

## Related evidence (read-only)

| Artifact | Role |
|----------|------|
| `data/results/gld_partial_exit_sizing_analysis.json` | Partial sizing |
| `data/results/phase_198d_partial_defensive_exit_preview_validation.json` | Partial preview |
| `data/results/phase_198i_partial_dispatch_scope_dry_run_validation.json` | Dispatch scope = 48 |
| `data/results/phase_198j_partial_dispatch_evidence_reconciliation.json` | Raw 195 vs authoritative 48 |
| `data/results/phase_198k_partial_gld_no_execution_certification.json` | No-execution certification |
| `data/results/partial_gld_operator_certification_summary.md` | Operator certification summary |

Display (read-only):

```powershell
python scripts/show_partial_gld_no_execution_certification.py
```
