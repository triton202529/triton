# Approval Artifact Rules

Operator rules for TRITON **paper defensive exit approval artifacts**.
This page does **not** create approval files, broker authorization, or orders.

Related:

- [Operator manual index](./README.md)
- [Paper Execution Safety Rules](./PAPER_EXECUTION_SAFETY_RULES.md)
- [Partial GLD Defensive Path](./PARTIAL_GLD_DEFENSIVE_PATH.md)
- [Monday Market-Open Restart Checklist](./MONDAY_MARKET_OPEN_RESTART_CHECKLIST.md)

---

## What the approval artifact is

**Path:** `data/governance/operator_paper_defensive_exit_approval.json`

Short-lived operator governance record that records **paper-only** approval of a capped
defensive batch (or partial symbol scope) after operator review. Created intentionally via
Phase 197W (or equivalent) with the required operator authorization phrase.

**Absent file = no operator paper approval authority.**

Verify at session start:

```powershell
Test-Path data/governance/operator_paper_defensive_exit_approval.json
```

Expected at restart unless an intentional paper window is in progress: **False**.

---

## What it does and does not do

### Does

- Approves a **capped paper defensive batch** for operator review and downstream gates
- Records paper-only scope tied to the current governed preview
- Enables the *next* layer (broker authorization) only when scope, hash, and expiry are valid

### Does not

- Authorize broker execution by itself
- Authorize live trading
- Submit broker orders
- Replace 197X / 198J / 198K governed dispatch quantity evidence
- Approve full GLD exit when only partial qty **48** is governed

The artifact must match the **current governed preview hash and scope**. Stale or mismatched
approval is invalid and must not be carried forward.

---

## Required properties

A valid approval artifact should include or imply:

| Property | Requirement |
|----------|-------------|
| Paper-only approval | `approval_scope`: `PAPER_ONLY`; no live path |
| Approved symbols | Explicit list matching governed preview (e.g. GLD only for partial path) |
| Approved quantities | Where applicable — must match governed projected qty (GLD **48**, not **195**) |
| Source preview hash | Must match current `defensive_exit_operator_approval_packet` / 197V review hash |
| Max notional | Capped to governed batch limit; not an expansion of placement caps |
| Live-blocked acknowledgement | `live_authorized`: **false** |
| Broker auth is separate | `broker_execution_authorized`: **false**; broker auth is a distinct artifact |
| Expiry / short-lived validity | `expires_at` or TTL (e.g. 30 minutes); not standing overnight authority by default |

Typical creation path (intentional only; dry-run is default):

```powershell
# Dry-run / syntax check (no governance write)
python -m services.phase_197w_create_paper_operator_approval_artifact --help
python -m services.phase_197w_create_paper_operator_approval_artifact --dry-run

# Real create (writes data/governance/operator_paper_defensive_exit_approval.json)
python -m services.phase_197w_create_paper_operator_approval_artifact --create --operator-confirmation "I authorize Phase 197W to create the paper-only operator approval artifact for the current capped defensive batch. No broker orders yet."
```

197W operator phrase (approval creation only — still no broker orders):

> I authorize Phase 197W to create the paper-only operator approval artifact
> for the current capped defensive batch. No broker orders yet.

`--create` is required for a real write. Without `--create`, 197W remains dry-run and
does **not** create `data/governance/operator_paper_defensive_exit_approval.json`.

Use `--dry-run` (or plain invocation with no `--create`) when validating compatibility
without writing the artifact.

---

## Lifecycle

| Stage | Operator action |
|-------|-----------------|
| **Review** | Read `defensive_exit_operator_approval_packet.json` / 197V review output |
| **Validate** | Confirm preview hash, symbols, quantities, and live-blocked posture |
| **Create** | Intentional 197W only when proceeding to a time-boxed paper window |
| **Use** | Downstream dry-runs / broker-auth creation reference this artifact |
| **Expire** | Honor `expires_at`; expired approval is not authority |
| **Cleanup** | Remove after use, rejection, or closeout (see § Cleanup rules) |

Approval alone does not complete the paper execution chain. Broker authorization and dispatch
are separate layers — see [PAPER_EXECUTION_SAFETY_RULES.md](./PAPER_EXECUTION_SAFETY_RULES.md).

---

## GLD partial approval rule

For the certified GLD partial defensive path (198C–198K):

| Field | Value |
|-------|-------|
| Approved symbol | **GLD** |
| Approved quantity | **48** |
| Full quantity | **195** |
| Remaining after partial | **147** |
| Full exit | **Deferred** |

Rules:

- Approval must **not** be interpreted as full GLD exit approval.
- Raw handoff may show **195**; approval scope must still be partial qty **48**.
- Approved quantity must match **197X** projected quantity and **198J / 198K** evidence.
- Approving qty **195** when governed scope is **48** is a stop condition.

---

## Cleanup rules

1. **Remove approval artifact after use, expiry, rejection, or closeout.**
2. **Never commit the approval artifact** — it is runtime authority, not source documentation.
3. **If found unexpectedly, stop and remove after backup** — do not dispatch on surprise authority.
4. **Rebuild the review packet instead of reusing stale approval** — refresh 197V / packet
   and create a new artifact only when scope and hash match current preview.

Backup before removal (optional audit trail):

```powershell
Copy-Item data/governance/operator_paper_defensive_exit_approval.json `
  data/results/operator_paper_defensive_exit_approval.backup.json
Remove-Item data/governance/operator_paper_defensive_exit_approval.json
```

Confirm removal:

```powershell
Test-Path data/governance/operator_paper_defensive_exit_approval.json
```

---

## Stop rules

**Stop immediately if:**

| Condition | Why |
|-----------|-----|
| Approval artifact exists unexpectedly | Unintended standing authority |
| Approval references stale preview hash | Scope no longer matches governed preview |
| Approval quantity differs from 197X projected quantity | Dispatch scope mismatch |
| Approval tries to approve full GLD exit instead of partial qty **48** | Full exit remains deferred |
| Broker authorization artifact is missing when dispatch is attempted | Approval alone does not authorize broker execution |
| Live execution is not blocked | Live must remain blocked for paper defensive path |

Do not create broker authorization or run dispatch to “use up” a bad approval. Remove or
expire the artifact and rebuild from current packet review.

---

## Danger boundaries

Do **not**:

- Create approval during weekend / market-closed documentation work
- Leave approval in place into the next session without explicit operator intent
- Treat approval as broker authorization or as permission to place orders
- Commit `data/governance/operator_paper_defensive_exit_approval.json` to git
- Reuse approval after preview hash, symbol list, or quantity scope changes
- Expand GLD approval from **48** to **195** based on raw `manage_orders.csv`

---

## Related evidence (read-only)

| Artifact | Role |
|----------|------|
| `data/results/defensive_exit_operator_approval_packet.json` | Review packet source |
| `data/results/phase_197v_operator_approval_packet_review.json` | 197V review |
| `data/results/phase_197x_paper_broker_execution_final_dry_run.json` | Projected qty authority |
| `data/results/phase_198k_partial_gld_no_execution_certification.json` | No-execution certification |

Display partial GLD certification (read-only):

```powershell
python scripts/show_partial_gld_no_execution_certification.py
```
