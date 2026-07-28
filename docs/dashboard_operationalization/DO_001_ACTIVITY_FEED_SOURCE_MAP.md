# DO-001 — Activity Feed Source Map

**Phase:** DO-001 (design)
**Invariant:** Feed events must be derived from real artifacts/logs. Do not invent successes.

---

## 1. Gap today

There is **no unified operator activity feed**. Closest surfaces:

| Surface | What it is |
|---------|------------|
| Signals → Ticker Decision Timeline | Per-ticker narrative — not runtime ops |
| Governance timelines / registries | Governance CSV activity |
| Continuous Loop / Hung Loop pages | Health metrics, not chronologic ops feed |
| Live Orders / Execution Intelligence | Tables, not narrated events |

---

## 2. Source inventory

| Domain | Path pattern | Cadence | Key fields |
|--------|--------------|---------|------------|
| Continuous-loop logs | `data/results/continuous_loop/continuous_loop_*.log` | ~45 min/cycle | `[CYCLE_START]`, `[CYCLE_OK/FAIL]`, paper-entry stage lines |
| Market-open transcripts | `data/results/scheduled_runs/market_open_*.log` | Per continuous cycle | D1 refresh, signal universe |
| Watchdog-loop logs | `data/results/watchdog_logs/watchdog_loop_*.log`, `watchdog_*.log` | ~1 min | `[CYCLE_OK/FAIL]`, `active_alerts` |
| Runtime locks | `data/live/continuous_loop.lock`, `watchdog_loop.lock` | Per cycle | `pid`, `cycle`, `refreshed_at` |
| Startup audit | `data/results/runtime_autostart_931.json`, `runtime_supervisor_autostart.json` | On start | `postconditions_passed`, confirmed flags, errors |
| Paper arm | `data/live/paper_arm_confirm.json` | Session TTL | validity / expiry |
| Broker snapshots | `positions_snapshot.csv`, `open_orders_snapshot.csv`, `recent_orders.csv` | Freshness / cycles | qty, status, ages |
| Paper/live orders | `paper_execution_audit.csv`, `live_orders.csv`, `live_orders_log.csv`, dispatch audits | On place/poll | order id, status, skip_reason |
| Allocation | `portfolio_allocation_recommendations.csv`, summary JSON | Market-open | recommended_action |
| Lifecycle | `signal_lifecycle_effective.csv` (authority) | Market-open | stance / action |
| Execution gates | `data/governance/execution_gate_decisions.json`, `execution_gate_audit.csv` | On eval | allowed, reason, paper_or_live |
| Risk overlay | `performance_risk_overlay.csv` | Market-open | `BLOCK_NEW_BUY` / etc. |
| Governance | `governance_authorization.json`, `execution_readiness.json`, `human_approval_queue.json` | Watchdog cadence | live_execution_permitted, authorization, queue |
| Watchdog alerts | `watchdog_alerts.json`, `watchdog_status.json` | ~1 min | alert_type, severity, subject |
| Freshness | `data_freshness_refresh_audit.json`, freshness logs, `heartbeat.json` | ~35 min | steps_run, errors |
| Paper entry narrative | `paper_entry_loop_stage_audit.json`, last market decision, `paper_entry_dispatch_contract.json` | After entry stage | gate_pass, eligible/rejected, submitted |

---

## 3. Message → source mapping

| Plain-language message | Best source(s) | Confidence | Rule |
|------------------------|----------------|------------|------|
| TRITON started successfully | Startup audit `postconditions_passed=true`; else START banner + confirmed locks | **High** | Emit only with audit/log proof |
| Market data refresh completed | `data_freshness_refresh_audit.json` or market-open D1 complete line | **High** | Prefer freshness audit timestamp |
| Evaluated N symbols | `[SIGNAL_UNIVERSE] tickers_processed=N` or post-cycle CSV count | **Medium** | Prefer explicit log N |
| No candidate passed all entry gates | `paper_entry_loop_stage_audit.json` empty eligible/final + `gate_pass=false` | **High** | Prefer stage audit over allocation-only |
| Paper order submitted | `paper_execution_audit.csv` with `submitted_order_id` | **High** | Require order id |
| Order filled | `recent_orders` / logs with fill status | **Medium–High** | Do not infer fill from submitted |
| Entry blocked by risk controls | Stage audit reject `RISK_OVERLAY_*` / `BLOCK_NEW_BUY` | **High** | Same-cycle evidence |
| Live execution remains blocked | `execution_readiness` / governance `live_execution_permitted=false` | **High** | Status line; avoid fake “new block event” without delta |
| Cycle N completed successfully | Continuous log `[CYCLE_OK] cycle=N` | **High** | Lock cycle alone ≠ success |
| Watchdog detected stale data | Map carefully: `STALE_HEARTBEAT` ≠ market-data stale; use freshness class / readiness | **Medium** | Branch wording by source |
| Operator action is required | Pending `human_approval_queue`; HIGH alerts; hung/health recommended action | **High** | Only when field demands action |

---

## 4. Feed design constraints

1. Prefer append-only / timestamped sources for **events** (logs, audit CSVs, alert transitions).
2. Snapshot JSON is excellent for **posture** lines; require mtime/content delta for “something happened.”
3. Never invent “Evaluated N” or “no candidates” without log/stage audit evidence.
4. Distinguish heartbeat stale vs broker snapshot age vs `data_freshness=STALE`.
5. Deduplicate: one event id per `(source, timestamp, message_key)` for idempotent UI refresh.

---

## 5. First-feed source priority

1. Latest continuous + watchdog loop logs
2. `paper_entry_loop_stage_audit.json`
3. `paper_execution_audit.csv` + order snapshots
4. `watchdog_alerts.json` (active + newly resolved)
5. Startup + freshness audits
6. `governance_authorization.json` + `execution_readiness.json` for live-blocked / block posture

---

## 6. Proposed Activity page behavior (later phase)

- Chronological list, newest first
- Severity tags: info / success / warn / block / action_required
- Deep-link to source artifact path (read-only)
- Manual refresh + optional auto-refresh interval
- Empty state: “No evidenced events yet” — never fabricated filler
