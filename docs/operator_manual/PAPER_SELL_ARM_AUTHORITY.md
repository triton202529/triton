# Paper Sell-Arm Authority (Phase 198H / 198I / 198I-FIX)

## Why buying and selling use separate authority

Paper **BUY / ADD** entries use the normal daily paper arm (`data/live/paper_arm_confirm.json`).

That buy arm is operated today through the **established PowerShell / environment workflow**.

**There is no Operator Console buy-arm button yet.**  
**There is no Operator Console sell-arm or confirmation button yet.**

Paper **TRIM / EXIT / ROTATE_EXIT / FORCE_EXIT** management sells — and every other paper sell submission, replacement, repricing, retry, or recovery attempt — require the hardened batch-specific paper sell arm. The daily buy arm never authorizes sells, and a sell arm never authorizes buys.

## Canonical storage (Phase 198I-FIX)

| Artifact | Path | Role |
|----------|------|------|
| Hardened sell authority | `data/governance/paper_sell_arm_authority.json` | **Sole** runtime paper-sell authority |
| Confirmation challenge | `data/governance/paper_sell_confirmation_challenge.json` | Non-authoritative |
| Consumption / intent | `data/governance/paper_sell_arm_consumption.json`, `paper_sell_arm_dispatch_intent.json` | Irreversible attempt evidence |
| Legacy 197W evidence | `data/governance/operator_paper_defensive_exit_approval_legacy_197w.json` | Historical only — **never** runtime authority |
| Deprecated shared name | `operator_paper_defensive_exit_approval.json` | Must stay unused; presence fails closed |

Phase **197W** writers must never overwrite, replace, shadow, reset, or erase hardened authority or consumption state.

## Runtime enforcement

- Schema: `authority_type=PAPER_SELL_ARM`, `198H-PAPER-SELL-ARM-1` only.
- Legacy **197W** cannot authorize dispatch and cannot write the hardened path.
- Every broker-facing paper sell (including `AlpacaBroker.submit_order` / notional / bracket, CSV, reprice, replace, position-protector) requires a hardened dispatch permit issued only after durable consumption.
- Immediate pre-dispatch checks use **independent** current evidence (live-block classifier, paper arm, plan, readiness, challenge, positions/orders) — not values copied from the authority alone.
- Caller-supplied `live_blocked=True` is non-authoritative and cannot override enabled, missing, contradictory, or ambiguous live state.
- Authority is irreversibly **consumed before** the first possible broker-submission attempt (durable intent + consumption + flush/fsync where supported).
- Success, rejection, timeout, exception, partial, cancelled, duplicate client-order-id, and indeterminate outcomes **cannot** restore reusable ARMED authority.
- Recovery **reconciles** stable identifiers and never blindly resubmits; a new operator-reviewed plan and authority are required.
- `FORCE_EXIT` urgency is not execution authority — it still requires the hardened sell arm.
- The **$5,000** approval CLI cap governs authority until formally reconciled; the **$50,000** runtime manage cap does not bypass it.
- Live execution remains blocked for paper sell dispatch.

## Future Operator Console buttons (not implemented yet)

Operator Console integration remains a later phase and requires a passing Phase **198I-V** re-verification first.

### Arm Paper Buys / Arm Reviewed Sells / Confirm & Arm Paper Sells

- Not present today.
- Future controls may create authorization-related state only.
- They must not start a cycle or dispatch an order.

Acknowledgement text for Confirm & Arm:

> I reviewed these paper sell actions and authorize TRITON to dispatch only this exact batch, subject to final safety revalidation.

## Exact authority scope

Authority is bound to PAPER mode, sanitized account fingerprint, trading date, cycle/run/batch/plan IDs, plan hash, readiness-packet hash, confirmation challenge id/hash, exact symbols/actions/max quantities/notionals, applied **$5,000** cap, single-use, live execution BLOCKED, and no symbol/action/quantity/batch expansion.

## Consumption / durability

Consume on the **first dispatch attempt**. Dispatch-critical writes require temp-file + flush + file fsync + atomic replace + **proven directory durability for the required crash class**. On POSIX this is directory `fsync`. On Windows, `CreateFileW(FILE_FLAG_BACKUP_SEMANTICS)` + `FlushFileBuffers(GENERIC_WRITE)` may succeed and is recorded for diagnostics, but that API success alone is **not** accepted as proof of durable directory-entry replacement across OS crash or power loss. Until a primary-documented Windows metadata durability contract is established, Windows paper-sell dispatch remains **fail-closed** (no broker-usable permit). Directory durability failure or unsupported APIs are also fail-closed. Stale ARMED content cannot overwrite CONSUMED/ATTEMPTED/terminal evidence.

## Caps

- Approval workflow / authority-creation cap: **$5,000**
- Runtime management batch cap: **$50,000**
- Lower applicable cap governs; actions above $5,000 must be deferred, not silently split.

## FORCE_EXIT

FORCE_EXIT remains a defensive sell action and **requires the same reviewed paper sell arm**. No autonomous emergency sell route is introduced.

## Service interface (no dashboard wiring yet)

```text
python -m services.paper_sell_arm_cli inspect --store-root <tmp>
python -m services.paper_sell_arm_cli prepare-challenge --store-root <tmp> --readiness-json <fixture>
python -m services.paper_sell_arm_cli confirm-and-arm --store-root <tmp> --readiness-json <fixture> --challenge-json <fixture> --acknowledgement-file <ack.txt>
python -m services.paper_sell_arm_cli validate-authority --authority-json <fixture>
```

## Recovery

| State | Operator action |
|-------|-----------------|
| STALE / EXPIRED / REJECTED / CONSUMED / ATTEMPTED | Rebuild readiness from a fresh completed management cycle; do not reuse old challenges/arms |
| Indeterminate dispatch | Authority already consumed; reconcile broker state before any new arm |
| BLOCKED | Resolve mismatch before retrying |

## Non-negotiable sentences

1. The first click creates **no** execution authority.
2. The second click creates authority but **does not** dispatch.
3. Only the runtime may dispatch after final independent revalidation and durable consumption.
4. Paper BUY/ADD authority remains separate and PowerShell/environment operated.
5. Live execution remains blocked.
