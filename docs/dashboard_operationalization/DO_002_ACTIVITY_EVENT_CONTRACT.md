# DO-002 — Activity Event Contract

**Phase:** DO-002
**Producer:** `services/operator_activity_feed.py`
**Consumer:** Operator Console → Activity page

---

## Event schema

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `event_id` | string | yes | Stable hash of source+time+title+message (+ cycle/order) |
| `occurred_at` | ISO-8601 string | yes | UTC when possible |
| `source` | string | yes | e.g. `continuous_loop_log`, `paper_execution_audit` |
| `category` | enum | yes | See categories |
| `severity` | enum | yes | See severities |
| `title` | string | yes | Short operator title |
| `message` | string | yes | Plain-language explanation |
| `evidence_path` | string | yes | Allowlisted artifact path |
| `cycle_number` | int \| null | no | When evidenced |
| `symbol` | string \| null | no | When evidenced |
| `order_id` | string \| null | no | When evidenced |
| `raw_reference` | string | no | Truncated technical snippet |

### Categories

`RUNTIME`, `CYCLE`, `DATA`, `SIGNAL`, `ENTRY_GATE`, `ORDER`, `RISK`, `ALERT`, `GOVERNANCE`, `BROKER`, `SAFETY`

### Severities

`INFO`, `SUCCESS`, `WARN`, `BLOCK`, `ACTION_REQUIRED`, `CRITICAL`

---

## Evidence rules (non-negotiable)

1. **`exit_code=0`** ⇒ stage completed successfully only. **Never** an ORDER event by itself.
2. **Cycle start** does not prove cycle completion.
3. **Planned / authorized** does not prove submitted.
4. **Submitted** does not prove filled.
5. **Zero open orders** does not prove no prior submission.
6. **Missing evidence** ⇒ unknown/unavailable — never fabricate rejection reasons.
7. Events are **deduplicated** by `event_id`.
8. Feed is **read-only** — never writes source logs/artifacts.

---

## Preferred sources

| Source | Typical events |
|--------|----------------|
| `data/results/continuous_loop/continuous_loop_*.log` | Cycle start/OK/fail, stage success, signal universe |
| `data/results/watchdog_logs/watchdog_*.log` | Watchdog cycle health |
| `data/results/runtime_autostart_931.json` / `runtime_supervisor_autostart.json` | Startup validation, live blocked |
| `data/results/paper_entry_loop_stage_audit.json` | Gates evaluated, no candidates, risk blocks, explicit submits |
| `data/results/paper_execution_audit.csv` | Submitted vs filled (status-gated) |
| `data/results/watchdog_alerts.json` | Active alerts |
| `data/results/data_freshness_refresh_audit.json` | Data refresh completed |
| `data/results/execution_readiness.json` | Live remains blocked |
| `data/results/governance_authorization.json` | Governance not authorized |

---

## Example titles

- TRITON startup validation passed.
- Continuous cycle 114 started.
- Cycle 114 completed successfully.
- Market-data processing completed.
- Paper-entry gates were evaluated.
- No candidate passed all entry gates.
- Entry was blocked by risk controls.
- Paper order was submitted.
- Paper order was filled.
- Watchdog status is healthy.
- Live execution remains blocked.

---

## UI behavior

- Newest first
- Manual Refresh + optional auto-refresh if `streamlit-autorefresh` is installed
- Filters: category, severity, source, text/symbol, time range
- Expandable raw evidence
- Stale-source warnings when continuous log mtime is old
- Empty state when no evidenced events match filters
