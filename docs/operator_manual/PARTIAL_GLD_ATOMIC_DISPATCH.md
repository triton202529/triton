# Partial GLD Atomic Paper Dispatch

Operator guide for the **atomic paper-only partial GLD defensive runner**.

This page documents the command. It does **not** by itself create approval, broker
authorization, or orders.

Related:

- [Partial GLD Defensive Path](./PARTIAL_GLD_DEFENSIVE_PATH.md)
- [Paper Execution Safety Rules](./PAPER_EXECUTION_SAFETY_RULES.md)
- [Approval Artifact Rules](./APPROVAL_ARTIFACT_RULES.md)
- [Broker Authorization Rules](./BROKER_AUTHORIZATION_RULES.md)

---

## Why this runner exists

The multi-step PowerShell sequence can fail safely when the short-lived **197W** approval
artifact expires before **197Z** broker authorization is created.

The atomic runner keeps the same gates, but runs them in **one uninterrupted Python process**
so approval → 197X → evidence → broker auth → 197Y/bridge happen without a paste delay.

---

## Scope locked by this runner

| Field | Value |
|-------|-------|
| Symbol | **GLD** |
| Authorized / projected qty | **48** |
| Full position qty | **195** (deferred) |
| Mode | Paper only |
| Live execution | Blocked / forced off |
| Other symbols | Not expanded beyond approved paper scope |

---

## Exact operator confirmation

Required for every run:

> I authorize the paper broker execution phase for the approved capped defensive batch. Paper only. No live trading.

Phrase must match exactly.

Internally, the runner still creates the 197W approval artifact with the 197W-specific
approval phrase. That does **not** bypass 197W; it keeps approval creation intentional and
tied to this atomic operator action.

---

## Commands

### Dry-run (default / validation; no manage_positions)

```powershell
python -m services.phase_197wz_atomic_partial_gld_paper_dispatch --dry-run --operator-confirmation "I authorize the paper broker execution phase for the approved capped defensive batch. Paper only. No live trading."
```

Expected:

- status `PASS`
- final_state `ATOMIC_PARTIAL_GLD_DRY_RUN_GATES_READY_NO_DISPATCH`
- proves 197X PASS with `projected_quantities.GLD == 48`
- proves 197Y `eligible_for_execution=true`
- proves 197Z `bridge_active=true` and bridge qty **48**
- `broker_orders_submitted=false`
- `open_orders_count=0`
- live remains blocked
- short-lived approval/auth artifacts cleaned after the dry-run proof

### Real paper dispatch (explicit)

```powershell
python -m services.phase_197wz_atomic_partial_gld_paper_dispatch --dispatch --operator-confirmation "I authorize the paper broker execution phase for the approved capped defensive batch. Paper only. No live trading."
```

Expected:

- all dry-run gates pass first
- one `manage_positions --mode paper` invocation (controlled path; scope clamps GLD to qty **48**)
- post-dispatch snapshot / poll / manage open orders
- **PASS / `ATOMIC_PARTIAL_GLD_DISPATCH_COMPLETED` only when** broker order evidence exists for GLD sell qty **48** (open or recent)
- if no broker order is submitted: status `FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED`, artifacts cleaned, never `DISPATCH_COMPLETED`
- final JSON report under
  `data/results/phase_197wz_atomic_partial_gld_paper_dispatch.json`

---

## What the runner does

In one process:

1. Force paper-only env flags; force live execution flags to `0`
2. Refuse if open orders exist, live is enabled, or valid leftover artifacts remain
3. Clean expired/invalid leftover artifacts when safe
4. Refresh GLD partial sizing evidence
5. Refresh governance packet (197U) and operator review (197V)
6. Validate 197W dry-run contract (**no** governance write)
7. Run 197X via the dry-run contract path (require PASS + GLD **48**)
8. Run **198I / 198J / 198K** while approval/auth are still absent (those certifications require artifact absence by design; the runner does not loosen them)
9. Create fresh 197W approval (GLD **48**, full **195** deferred)
10. Immediately re-run 197X dry-run against the fresh approval and require PASS + `projected_quantities.GLD == 48`
11. Create 197Z broker authorization (qty **48**)
12. Verify 197Y eligible and 197Z bridge active at qty **48**
13. Dry-run: stop and clean artifacts
14. `--dispatch`: invoke `manage_positions` once; require open/recent GLD sell qty **48** evidence for PASS; on no-submit set `FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED` and clean artifacts

### Why 198I/J/K run before 197W create

198I / 198J / 198K are **no-execution** certifications. They hard-fail when a real approval or broker-auth artifact is present. The atomic runner therefore certifies the partial GLD qty-**48** evidence path **before** creating short-lived 197W/197Z artifacts, then creates those artifacts in the same process so they cannot expire between paste steps.

### No false dispatch success

`ATOMIC_PARTIAL_GLD_DISPATCH_COMPLETED` / status `PASS` is allowed only when post-dispatch proof shows a paper GLD **sell** qty **48** in open or recent orders. A `manage_positions` exit code of 0 alone is **not** enough (for example batch-notional blocks previously returned 0 with `exits_executed=0`).

---

## Hard stops

The runner stops immediately and does **not** dispatch if:

- operator confirmation mismatches
- open orders are present
- live execution is enabled / not blocked
- valid approval or broker-auth artifacts already exist
- GLD authorized/projected qty is not **48**
- full GLD exit is not deferred
- any of 197U / 197V / 198I / 198J / 198K / 197W / 197X / 197Z / 197Y / bridge fails

On failure before broker auth, the runner prefers cleanup of the approval artifact it created.

---

## Report artifact

Path:

`data/results/phase_197wz_atomic_partial_gld_paper_dispatch.json`

Key fields:

- `status`
- `final_state`
- `dry_run`
- `dispatch_attempted`
- `broker_orders_submitted`
- `open_orders_count`
- `live_blocked`
- `authorized_qty` (48)
- `full_qty` (195)
- `full_exit_deferred` (true)
- `steps`

---

## Safety reminders

- Do not use this runner for live trading.
- Do not expand GLD from 48 to 195.
- Do not bypass missing gates by re-running with weaker flags.
- Dry-run first. Only use `--dispatch` when intentionally opening a paper window.
