# TRITON Paper ARM Decision Packet (Phase 196R)

Read-only operator decision reference. **This document does not arm paper mode or submit orders.**

Generated from live artifacts at validation time. Re-check with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_triton_paper_arm_status.ps1
.\.venv\Scripts\python.exe -m services.phase_196r_controlled_paper_arm_decision_packet_validation
```

## Current system state

| Signal | Value |
|--------|-------|
| Runtime pipeline | HEALTHY (watchdog ON, continuous loop ON) |
| Data freshness | FRESH (~6 min artifact ages) |
| Market | OPEN |
| ARM mode | ASSISTED |
| Paper arm confirm | **EXPIRED** |
| Paper entry armed | **NO** |
| Live confirm | EXPIRED (separate gate) |
| Live execution | **BLOCKED** (`live_execution: false`) |
| Execution readiness | READY (8/8 checks) |
| Paper execution policy | authorized |
| Certification | **NOT_CERTIFIED** (Evaluation area failed; veto active) |
| Governance authorization | true (paper) |
| Open orders | 0 |
| Positions | JPM, WFC |

Latest paper entry loop audit (`paper_entry_loop_stage_audit.json`):

- **Result:** BLOCKED at `arm_permission`
- **Reason:** `ARM_EXPIRED_OR_MISSING paper_arm_confirm`
- **Candidates evaluated:** none (blocked before candidate selection)
- **Final selection / risk overlay / order submission:** not reached
- **Broker order submitted:** no

## What arming paper mode does

Arming writes a time-limited operator latch to `data/live/paper_arm_confirm.json` via:

```powershell
.\.venv\Scripts\python.exe -m services.set_paper_arm_confirm --ttl 30
```

or for a longer window:

```powershell
.\.venv\Scripts\python.exe -m services.set_paper_arm_confirm --ttl 480
```

**Do not run either command unless you explicitly approve arming.**

When valid, arming:

- Satisfies the ASSISTED-mode operator latch required by `paper_entry_loop_stage`
- Grants ARM `execute_trades` permission for the continuous loop's paper entry stage
- Allows the **next natural** continuous-loop cycle to proceed past the arm gate

## What arming paper mode does NOT do

- Does **not** enable live execution (`live_execution` stays false)
- Does **not** write `live_confirm.json`
- Does **not** bypass certification (`NOT_CERTIFIED` remains a governance signal)
- Does **not** bypass risk overlay (`BLOCK_NEW_BUY`, `FORCE_EXIT`, etc.)
- Does **not** bypass governance approval / final selection sync
- Does **not** submit a broker order by itself — only unblocks the automated paper entry path

## Could arming lead to a paper order on the next loop?

**Unlikely in the current posture — recommended to keep observing.**

Current allocation summary:

- `open_new_candidates`: **0**
- `add_candidates`: **0**
- `exit_candidates`: **1** (GE — not held)
- `blocked_candidates`: **5**

Entry pipeline preview (read-only, no `execute_trades`):

- ENTRY rows in `trade_opportunities.csv`: BAC, SCHW, UNG
- **Eligible after risk/allocation gates:** BAC only (SCHW/UNG blocked by `BLOCK_NEW_BUY`)
- Planner preview top buy: **UNG**
- **Post-arm risk overlay on planner selection:** would block UNG (`FINAL_SELECTION_RISK_OVERLAY_BLOCK`)

Held positions (JPM, WFC): risk flags **OK**; recommendations are HOLD (no actionable add).

**Conclusion:** Even with a fresh `paper_arm_confirm`, the next loop would likely still **not** submit a paper buy because the planner's top symbol is risk-blocked and there are no open-new/add allocation candidates. Arming mainly changes *permission to attempt* the pipeline, not the current candidate quality.

## Remaining blocks after arming

1. `FINAL_SELECTION_RISK_OVERLAY_BLOCK:UNG` (planner vs eligibility mismatch)
2. `NOT_CERTIFIED` — governance/certification display (Evaluation veto; separate from arm latch)
3. `LIVE_EXECUTION_BLOCKED` — always
4. `live_confirm` EXPIRED — affects live only, not paper arm

## Recommended operator option

**`DO_NOT_ARM_KEEP_OBSERVING`**

Rationale:

- Paper arm is expired but **no actionable open-new/add allocation** exists
- Eligible ENTRY (BAC) does not match planner preview (UNG), which is risk-blocked
- Certification remains NOT_CERTIFIED — arming does not resolve governance evaluation gap
- Runtime is healthy; observation continues safely with zero open orders

If you still choose to arm for pipeline exercise:

- **`ARM_SHORT_WINDOW_30_MIN`** — minimal exposure window aligned with config TTL
- **`ARM_STANDARD_WINDOW_480_MIN`** — longer window; only if you accept sustained automated paper mutation permission

## Operator commands (reference only — not executed in 196R)

```powershell
# Short window (30 minutes)
.\.venv\Scripts\python.exe -m services.set_paper_arm_confirm --ttl 30

# Standard window (480 minutes / 8 hours)
.\.venv\Scripts\python.exe -m services.set_paper_arm_confirm --ttl 480
```

After arming (if approved):

- **Do not** run `python -m services.execute_trades` manually
- Let the **continuous loop** act naturally on its 45-minute cycle
- Monitor `data/results/paper_entry_loop_stage_audit.json` after the next cycle

## Doctrine

- No manual `execute_trades`
- No manual broker orders
- No live execution enablement
- Arming paper ≠ live trading
