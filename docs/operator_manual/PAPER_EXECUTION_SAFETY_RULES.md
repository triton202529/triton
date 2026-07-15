# Paper Execution Safety Rules

Operator rules for TRITON **paper** defensive-exit execution authority.
This page does **not** authorize orders, create artifacts, or enable live trading.

Related:

- [Operator manual index](./README.md)
- [Monday Market-Open Restart Checklist](./MONDAY_MARKET_OPEN_RESTART_CHECKLIST.md)
- [Partial GLD Defensive Path](./PARTIAL_GLD_DEFENSIVE_PATH.md)
- [Weekend Safe Work Map](./WEEKEND_SAFE_WORK_MAP.md)

---

## Core rule

> **Paper execution policy being enabled does not mean orders may be placed.**

Policy, approval, broker authorization, and dispatch are **four separate layers**. All must
align for paper dispatch. Any missing or mismatched layer means **stop**.

Live execution remains blocked under these rules unless a separate live-authorization path
exists (it must not be assumed from paper work).

---

## Four separate layers

| Layer | What it is | What it is not |
|-------|------------|----------------|
| **1. Paper execution policy enabled** | Config / env / readiness posture that *allows paper mode to be considered* | Permission to place orders; not approval; not broker auth |
| **2. Operator approval artifact** | Short-lived `operator_paper_defensive_exit_approval.json` matching the approved scope | Broker authorization; not dispatch |
| **3. Broker authorization artifact** | Short-lived `operator_paper_broker_execution_authorization.json` for the approved paper scope | Automatic order send; not live authority |
| **4. Actual broker dispatch** | Explicit placement / manage handoff that submits paper broker orders | Implied by docs, dry-run PASS, or policy flags alone |

### Layer detail

**Paper execution policy enabled**
Examples include paper ARM, paper env flags, or readiness stating paper may proceed *in
principle*. None of these place orders. None create governance approval files.

**Operator approval artifact**
Created only by intentional 197W (or equivalent) with the correct operator phrase and
scope. Absent file = no operator paper approval authority.

**Broker authorization artifact**
Created only by intentional 197Z (or equivalent) after approval exists and the exact
paper-only phrase is supplied. Absent file = no paper broker execution authority.

**Actual broker dispatch**
Occurs only when gates allow and an operator-run path (e.g. manage / place) submits
orders. Dry-runs, previews, and certifications are not dispatch.

---

## Short-lived artifact rules

| Artifact | Path |
|----------|------|
| Operator approval | `data/governance/operator_paper_defensive_exit_approval.json` |
| Broker authorization | `data/governance/operator_paper_broker_execution_authorization.json` |

Rules:

1. **`operator_paper_defensive_exit_approval.json` must be intentional** — never create from
   documentation, weekend work, or accidental validation side effects.
2. **`operator_paper_broker_execution_authorization.json` must be intentional** — never create
   without matching approval and the exact paper-only phrase (§ Exact paper-only authorization
   phrase).
3. **Both must be removed after use, expiry, or closeout** — do not leave standing authority
   into the next session unless that is an explicit operator decision.
4. **Never commit either artifact** — keep them out of git; they are runtime authority, not
   source documentation.

Verify absence when starting a session:

```powershell
Test-Path data/governance/operator_paper_defensive_exit_approval.json
Test-Path data/governance/operator_paper_broker_execution_authorization.json
```

---

## GLD partial rule (certified path)

For the certified GLD partial defensive path (198C–198K):

| Fact | Value |
|------|-------|
| Raw handoff qty may show | **195** |
| Governed partial dispatch qty | **48** |
| Remaining after partial | **147** |
| Full GLD exit | **Deferred** |
| Quantity authority | **197X / 198J / 198K** evidence |

- Do not treat raw `data/live/manage_orders.csv` as broker dispatch quantity.
- Broker dispatch must use governed projected quantity **48** only.
- Expanding scope from 48 → 195 is forbidden without a new governed path.

---

## Stop rules

**Stop immediately if:**

- Open orders exist
- Approval artifact exists unexpectedly
- Broker authorization artifact exists unexpectedly
- Live execution is not blocked
- Projected dispatch quantity differs from approved quantity
- Git status shows unexpected source changes

Do not create the next-layer artifact to “fix” a stop condition. Resolve the condition first.

---

## Completed defensive batches and stale previews

After a paper defensive batch has already filled (for example GLD reduced from 147 to 16):

- Broker position is the authority.
- Stale preview / projection artifacts must be invalidated (`STALE_DEFENSIVE_PREVIEW_INVALIDATED`
  or `TARGET_ALREADY_COMPLETED`).
- Do **not** recreate approval/broker-auth artifacts to revive a completed stale preview.
- 197Y must not advertise the old sell qty; 197Z must not advertise the old bridge qty.

---

## Exact paper-only authorization phrase

Required **before real broker authorization creation** (197Z path):

> I authorize the paper broker execution phase for the approved capped defensive batch. Paper only. No live trading.

- Phrase must match exactly (operator confirmation text).
- Paper only. No live trading.
- Supplying the phrase for documentation or dry-run discussion does **not** create the
  artifact; only the intentional creator module with confirmation does.

Related earlier gate (197W approval creation) uses its own phrase and still does **not**
authorize broker orders:

```powershell
python -m services.phase_197w_create_paper_operator_approval_artifact --create --operator-confirmation "I authorize Phase 197W to create the paper-only operator approval artifact for the current capped defensive batch. No broker orders yet."
```

197Z create (only after valid 197W approval exists):

```powershell
python -m services.phase_197z_paper_broker_execution_authorization_creator --create --operator-confirmation "I authorize the paper broker execution phase for the approved capped defensive batch. Paper only. No live trading."
```

---

## Hard-stop operator command block

Use `$ErrorActionPreference = "Stop"` so later commands do not continue after a failed gate.

```powershell
$ErrorActionPreference = "Stop"

function Assert-JsonField {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)]$Expected
  )
  if (-not (Test-Path $Path)) { throw "Missing result file: $Path" }
  $doc = Get-Content -Raw -Path $Path | ConvertFrom-Json
  $actual = $doc.$Name
  if ("$actual" -ne "$Expected") {
    throw "Hard stop: $Path.$Name expected=$Expected actual=$actual"
  }
}

# 1) Create/refresh short-lived approval (197W) — paper only, no broker orders
python -m services.phase_197w_create_paper_operator_approval_artifact --create --operator-confirmation "I authorize Phase 197W to create the paper-only operator approval artifact for the current capped defensive batch. No broker orders yet."
if ($LASTEXITCODE -ne 0) { throw "Hard stop: 197W create failed" }
if (-not (Test-Path .\data\governance\operator_paper_defensive_exit_approval.json)) {
  throw "Hard stop: 197W approval artifact absent"
}

# 2) Final dry-run must PASS with GLD qty 48
python -m services.phase_197x_paper_broker_execution_final_dry_run
if ($LASTEXITCODE -ne 0) { throw "Hard stop: 197X failed" }
Assert-JsonField .\data\results\phase_197x_paper_broker_execution_final_dry_run.json status PASS
$x197 = Get-Content -Raw .\data\results\phase_197x_paper_broker_execution_final_dry_run.json | ConvertFrom-Json
if ($x197.projected_quantities.GLD -ne 48) {
  throw "Hard stop: GLD projected qty is not 48"
}

# 3) Evidence gates
python -m services.phase_198j_partial_dispatch_evidence_reconciliation
if ($LASTEXITCODE -ne 0) { throw "Hard stop: 198J failed" }
Assert-JsonField .\data\results\phase_198j_partial_dispatch_evidence_reconciliation.json status PASS
python -m services.phase_198k_partial_gld_no_execution_certification
if ($LASTEXITCODE -ne 0) { throw "Hard stop: 198K failed" }
Assert-JsonField .\data\results\phase_198k_partial_gld_no_execution_certification.json status PASS

# 4) Broker auth create (197Z) — still no manage_positions dispatch in this block
python -m services.phase_197z_paper_broker_execution_authorization_creator --create --operator-confirmation "I authorize the paper broker execution phase for the approved capped defensive batch. Paper only. No live trading."
if ($LASTEXITCODE -ne 0) { throw "Hard stop: 197Z create failed" }
if (-not (Test-Path .\data\governance\operator_paper_broker_execution_authorization.json)) {
  throw "Hard stop: 197Z broker auth artifact absent"
}

# 5) Eligibility / bridge
python -m services.phase_197y_loop_mediated_paper_execution_gate
if ($LASTEXITCODE -ne 0) { throw "Hard stop: 197Y failed" }
$y197 = Get-Content -Raw .\data\results\phase_197y_loop_mediated_paper_execution_gate.json | ConvertFrom-Json
if (-not $y197.eligible_for_execution) { throw "Hard stop: 197Y eligible_for_execution is not true" }

python -m services.phase_197z_readiness_bridge_validation
if ($LASTEXITCODE -ne 0) { throw "Hard stop: 197Z bridge validation failed" }
$zBridge = Get-Content -Raw .\data\results\phase_197z_readiness_bridge_validation.json | ConvertFrom-Json
if (-not $zBridge.bridge_active) { throw "Hard stop: 197Z bridge_active is not true" }

Write-Host "Hard-stop gate block passed. Dispatch remains a separate intentional step."
```

Do **not** continue into `manage_positions` from this block unless the operator separately and intentionally starts a paper dispatch window after all hard stops above have passed.

---

## What does not authorize paper orders

- Operator manuals and checklists
- Weekend / market-closed documentation
- No-execution certification (198K)
- Dry-run PASS on 197X / 197Y / 197Z / 198*
- Paper policy or ARM alone
- Raw manage_orders rows
- Readiness or allocation rebuilds

---

## Safe closeout reminder

After any intentional paper window:

1. Confirm orders settled or open-order count understood
2. Remove or expire approval and broker-authorization artifacts
3. Confirm live still blocked
4. Confirm neither artifact is staged for commit
