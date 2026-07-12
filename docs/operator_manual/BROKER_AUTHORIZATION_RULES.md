# Broker Authorization Rules

Operator rules for TRITON **paper broker execution authorization artifacts**.
This page does **not** create broker authorization, approval files, or orders.

Related:

- [Operator manual index](./README.md)
- [Paper Execution Safety Rules](./PAPER_EXECUTION_SAFETY_RULES.md)
- [Approval Artifact Rules](./APPROVAL_ARTIFACT_RULES.md)
- [Partial GLD Defensive Path](./PARTIAL_GLD_DEFENSIVE_PATH.md)
- [Monday Market-Open Restart Checklist](./MONDAY_MARKET_OPEN_RESTART_CHECKLIST.md)

---

## What the broker authorization artifact is

**Path:** `data/governance/operator_paper_broker_execution_authorization.json`

Short-lived governance record that authorizes a **paper broker execution phase** only after a
valid operator approval artifact exists. Created intentionally via Phase 197Z (or equivalent)
with the exact paper-only operator confirmation phrase.

**Absent file = no paper broker execution authority.**

Verify at session start:

```powershell
Test-Path data/governance/operator_paper_broker_execution_authorization.json
```

Expected at restart unless an intentional paper window is in progress: **False**.

---

## What it does and does not do

### Does

- Authorize a **paper broker execution phase** after operator approval exists
- Record paper-only scope tied to the approved capped defensive batch
- Enable downstream loop / readiness gates to consider paper dispatch *when all other layers align*

### Does not

- Authorize live trading
- Override live execution blocks
- Expand quantity beyond the approved / projected scope
- Authorize BUY/ADD
- Submit broker orders by itself (dispatch is a separate layer)
- Replace operator approval — approval must exist first and remain valid

The artifact must match the **approved capped defensive batch** and current 197X projected
quantity evidence. Stale or mismatched authorization is invalid.

---

## Relationship to operator approval

| Layer | Artifact | Role |
|-------|----------|------|
| Operator approval | `operator_paper_defensive_exit_approval.json` | Paper-only batch approval for review |
| Broker authorization | `operator_paper_broker_execution_authorization.json` | Paper broker execution phase authority |

Broker authorization is **separate** from operator approval:

- Approval alone does not authorize broker execution.
- Broker authorization without matching approval is invalid.
- Both are short-lived and must be removed after use or closeout.

See [APPROVAL_ARTIFACT_RULES.md](./APPROVAL_ARTIFACT_RULES.md) for approval lifecycle.

---

## Required preconditions

All must be satisfied **before** intentional broker authorization creation:

| # | Precondition |
|---|--------------|
| 1 | Market open (or explicit market-hours policy satisfied for the phase) |
| 2 | Fresh broker snapshot |
| 3 | Open orders = **0** |
| 4 | Current readiness review |
| 5 | Valid operator approval artifact present and unexpired |
| 6 | Final dry-run showing approved / projected quantity (197X) |
| 7 | Exact paper-only authorization phrase supplied |
| 8 | Live execution blocked |

Typical creation path (intentional only):

```powershell
python -m services.phase_197z_paper_broker_execution_authorization_creator
```

Use dry-run / validation modes when checking compatibility without writing the artifact.

---

## Exact paper-only authorization phrase

Required **before real broker authorization creation**:

> I authorize the paper broker execution phase for the approved capped defensive batch. Paper only. No live trading.

- Phrase must match exactly.
- Supplying the phrase in documentation does **not** create the artifact.
- Only the intentional 197Z creator with operator confirmation writes the file.

---

## GLD partial broker authorization rule

For the certified GLD partial defensive path (198C–198K):

| Field | Value |
|-------|-------|
| Authorized symbol | **GLD** |
| Authorized quantity | **48** |
| Full quantity | **195** |
| Remaining after partial | **147** |
| Full exit | **Deferred** |

Rules:

- Raw `data/live/manage_orders.csv` qty **195** is **not** broker authority.
- **197X / 198J / 198K** evidence controls broker authorization scope.
- Authorized quantity must match 197X projected quantity (**48**).
- Authorization must **not** be interpreted as full GLD exit authority.
- Expanding scope from **48** → **195** is forbidden.

Read-only certification display:

```powershell
python scripts/show_partial_gld_no_execution_certification.py
```

---

## Lifecycle

| Stage | Operator action |
|-------|-----------------|
| **Pre-check** | Confirm all preconditions (§ Required preconditions) |
| **Validate** | Confirm approval exists, 197X dry-run PASS, qty scope matches |
| **Create** | Intentional 197Z only with exact confirmation phrase |
| **Use** | Downstream loop gate / readiness / dispatch paths reference this artifact |
| **Expire** | Honor TTL / `expires_at` (e.g. 10 minutes); expired auth is not authority |
| **Cleanup** | Remove after use, rejection, or closeout (see § Cleanup rules) |

Broker authorization alone does not place orders. Dispatch is layer 4 — see
[PAPER_EXECUTION_SAFETY_RULES.md](./PAPER_EXECUTION_SAFETY_RULES.md).

---

## Cleanup rules

1. **Remove broker authorization artifact after use, expiry, rejection, or closeout.**
2. **Never commit the broker authorization artifact** — runtime authority only.
3. **If found unexpectedly, stop and remove after backup** — do not dispatch on surprise authority.
4. **Recreate from current evidence; never reuse stale broker authorization** — refresh approval,
   197X dry-run, and projected quantity before a new artifact.

Backup before removal (optional audit trail):

```powershell
Copy-Item data/governance/operator_paper_broker_execution_authorization.json `
  data/results/operator_paper_broker_execution_authorization.backup.json
Remove-Item data/governance/operator_paper_broker_execution_authorization.json
```

Confirm removal:

```powershell
Test-Path data/governance/operator_paper_broker_execution_authorization.json
```

Remove operator approval as well when the paper window is complete — see
[APPROVAL_ARTIFACT_RULES.md](./APPROVAL_ARTIFACT_RULES.md).

---

## Stop rules

**Stop immediately if:**

| Condition | Why |
|-----------|-----|
| Operator approval artifact is missing | Broker auth requires prior approval |
| Broker authorization artifact exists unexpectedly | Unintended standing authority |
| Authorization references stale approval or stale preview | Scope no longer matches current evidence |
| Authorized quantity differs from 197X projected quantity | Dispatch scope mismatch |
| Authorization attempts full GLD exit instead of partial qty **48** | Full exit remains deferred |
| Open orders exist | Do not add new authority on unsettled broker state |
| Live execution is not blocked | Live must remain blocked |
| Git status shows unexpected source changes | Unknown code/config drift before session |

Do not run dispatch to “use up” bad authorization. Remove the artifact and rebuild from current
approval and 197X evidence.

---

## Danger boundaries

Do **not**:

- Create broker authorization during weekend / market-closed documentation work
- Create broker authorization without valid operator approval
- Treat broker authorization as live trading permission
- Commit `data/governance/operator_paper_broker_execution_authorization.json` to git
- Authorize qty **195** for GLD when governed scope is **48**
- Use raw `manage_orders.csv` to set authorization quantity
- Authorize BUY/ADD through the defensive-exit broker auth path

---

## Related evidence (read-only)

| Artifact | Role |
|----------|------|
| `data/governance/operator_paper_defensive_exit_approval.json` | Required prior approval |
| `data/results/phase_197x_paper_broker_execution_final_dry_run.json` | Projected qty dry-run |
| `data/results/phase_197x_projected_paper_orders.csv` | Projected order scope |
| `data/results/phase_197z_paper_broker_execution_authorization_creator.json` | 197Z creator state |
| `data/results/phase_198j_partial_dispatch_evidence_reconciliation.json` | Raw vs authoritative qty |
| `data/results/phase_198k_partial_gld_no_execution_certification.json` | No-execution certification |

---

## Live trading remains blocked

Broker authorization for paper defensive exits does **not** authorize live trading and does
**not** override live execution blocks. Live requires a separate authorization path that must
not be assumed from paper work.
