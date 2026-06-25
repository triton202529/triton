# TRITON Paper Session ARM Policy

Bounded auto-renewal keeps ASSISTED-mode `paper_arm_confirm.json` fresh during market sessions so paper trading is not blocked by a daily manual arming step.

## Paper arm purpose

`paper_arm_confirm.json` is an **operator permission latch** for ASSISTED mode. While valid, it allows the continuous loop's **paper entry stage** to proceed past the arm gate — it does **not** place orders by itself.

## Paper arm vs live confirm

| Artifact | Purpose |
|----------|---------|
| `paper_arm_confirm.json` | Paper-mode mutation permission (ASSISTED latch) |
| `live_confirm.json` | Live execution typed confirmation (separate) |

Session policy **only** writes `paper_arm_confirm.json`. It **never** writes `live_confirm.json`.

## Why auto-renewal

TRITON should be paper-trade ready on market days when:

- Paper execution env flags are enabled
- Broker mode is paper
- `live_execution` remains false

Manual `set_paper_arm_confirm` before every session is an operational bottleneck. Session policy renews with **bounded TTL** (default 600 min, max 600 min).

## How auto-renewal works

Config: `config/paper_arm_session_policy.json`

| Setting | Default | Meaning |
|---------|---------|---------|
| `pre_market_arm_minutes` | 45 | Arm window starts 45 min before 09:30 ET |
| `post_market_expiry_minutes` | 60 | Renew allowed until 60 min after 16:00 ET |
| `renew_if_minutes_remaining_below` | 90 | Refresh when <90 min left |
| `default_ttl_minutes` | 600 | Bounded renewal TTL |

**Integration points:**

1. `scripts/run_market_open.ps1` — step 0 before market-open pipeline
2. `scripts/run_data_freshness_refresh.ps1` — non-fatal renew during session window

Audit: `data/results/paper_session_arm_policy_audit.json`

## What armed paper mode can do

- Allow paper entry stage to evaluate gates on the natural continuous loop
- Permit `execute_trades` **only** when all downstream gates pass

## What still blocks trades

- Market closed / outside session arm window
- No OPEN_NEW / ADD allocation candidates
- Risk overlay (`BLOCK_NEW_BUY`, `FORCE_EXIT`)
- Paper candidate contract blocks
- Duplicate-order protection
- Governance / final selection approval sync
- Broker snapshot / readiness failures
- Certification / live gates for **live** trading (live remains blocked)

## Commands

```powershell
.\.venv\Scripts\python.exe -m services.paper_session_arm_policy --json
.\.venv\Scripts\python.exe -m services.paper_session_arm_policy --dry-run --json
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_triton_paper_arm_status.ps1
```

## Doctrine

- Do **not** run `execute_trades` manually to "fix" arming
- Do **not** enable live execution via this policy
- Do **not** create unbounded / never-expiring arm confirms
- Arming is permission only — allocation, risk, lifecycle, and candidate contract remain authoritative
