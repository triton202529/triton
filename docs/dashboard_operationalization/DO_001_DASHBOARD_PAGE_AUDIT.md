# DO-001 — Dashboard & Page Audit

**Phase:** DO-001 (read-only analysis and design)
**Generated:** 2026-07-28
**Canonical entrypoint:** `streamlit run view_results.py`

---

## 1. Dashboard technology

| Item | Finding |
|------|---------|
| Framework | **Streamlit** |
| Canonical Command Center | `view_results.py` (~24k lines) |
| Entry command | `streamlit run view_results.py` |
| Alternate launcher | `python main.py` → subprocess launches Streamlit on `view_results.py` |
| Multipage mechanism | **Custom** `PAGE_REGISTRY` + `SECTIONS` sidebar radios — **not** `st.navigation` / `st.Page`, and **not** classic Streamlit `pages/` auto-routing |
| Shared layout | Inline in `view_results.py` (CSS, freshness bar, execution gate bar, header stack, `sidebar_controls()`, `main()`) |
| State / loading | `@st.cache_data` CSV/JSON loaders; session keys for contracts/pipeline health |
| Refresh | Sidebar **⟳ Refresh data** clears caches and `st.rerun()` |
| Authentication | **None** found (no login, no `.streamlit/secrets.toml`, no `.streamlit/config.toml`) |
| Access model today | Local process exposure only by operator practice; **no enforced localhost binding in-repo** |

### Application entry / launch locations

| Path | Role |
|------|------|
| `view_results.py` | Primary Command Center |
| `main.py` | Pipeline then launches CC |
| `control_center.py` | Legacy signals UI |
| `dashboard.py` | Early signals dashboard |
| `app.py` | Parallel Alpaca ops UI (exec-capable) |
| `view_live_trades.py` | Legacy live trades |
| `backtest_dashboard.py` / `services/backtest_dashboard.py` | Backtest UIs |
| `services/risk_dashboard.py` | Adaptive risk standalone |
| `ui/pages/Order_Desk.py` | Retired order desk |
| `pages/manual_order_desk.py` | Canonical order desk (lazy-imported) |
| `ui/pages/run_csv_orders_page.py` | CSV orders (lazy-imported) |
| Cert/docs references | Phase scripts / operator docs expect `streamlit run view_results.py` |

---

## 2. Navigation implementation

- `PAGE_REGISTRY` at ~L23785: `(section, page_label) → page_* callable`
- `SECTIONS` at ~L24147: sidebar section → radio labels
- Optional inserts when files exist: Live Trading, Run CSV Orders, Manual Order Desk
- Selecting a SECTIONS page missing from registry shows `❌ Page not wired` (currently **0** unwired)
- **12** registry pages are **not** in SECTIONS → **unreachable via sidebar**

### Unreachable (registry ⊃ sidebar)

1. Governance Authorization Intelligence & Alignment
2. Governance Recovery Roadmap & Evidence Tracker
3. Governance Recovery Daily Evidence Monitor
4. Governance Recovery Acceleration & Bottleneck Intelligence
5. Governance Recovery Action Readiness & Execution Feasibility Intelligence
6. Governance Recovery Execution Simulation & Certification Rehearsal
7. GREEN Authorization Strategy & Long-Term Portfolio Certification Intelligence
8. Post-GREEN Institutional Stability & Continuous Certification Intelligence
9. Institutional Memory & Governance Learning Engine
10. TRITON Institutional Self-Evaluation & Strategic Evolution Engine
11. Execution Readiness Stability & Certification Integrity
12. Recovery Stabilization Intelligence (System)

---

## 3. Page counts (verified)

| Metric | Count |
|--------|------:|
| Command Center `PAGE_REGISTRY` pages | **135** |
| Sidebar-reachable (`SECTIONS` + optionals) | **123** |
| Unreachable registry pages | **12** |
| Standalone Streamlit / parallel surfaces inventoried | **16** |
| **Total inventory rows (CSV)** | **151** |
| Portfolio / Signals / Risk / Research / System | 5 / 17 / 81 / 3 / 29 |

Classification counts (CSV, including standalones) — must match JSON `classification_counts`:

| Classification | Count |
|----------------|------:|
| `KEEP_OPERATIONS` | 15 |
| `KEEP_PORTFOLIO` | 7 |
| `KEEP_ORDERS` | 4 |
| `KEEP_RISK_ALERTS` | 19 |
| `KEEP_SETTINGS_DIAGNOSTICS` | 12 |
| `MERGE_INTO_OPERATIONS` | 1 |
| `MOVE_TO_ADVANCED_DIAGNOSTICS` | 57 |
| `RESEARCH_ONLY` | 13 |
| `SIMULATION_RETIRE_CANDIDATE` | 14 |
| `BROKEN_OR_UNUSED` | 6 |
| `REQUIRES_MANUAL_REVIEW` | 3 |
| **Total inventory rows** | **151** |

Note: `data/results/**` is gitignored; audit CSV/JSON exist on disk at the required paths.

---

## 4. Classification rules used

Every page received **exactly one** of:

| Class | Meaning |
|-------|---------|
| `KEEP_OPERATIONS` | Daily runtime / signals / execution posture |
| `KEEP_PORTFOLIO` | Positions, PnL, trade history, performance |
| `KEEP_ORDERS` | Order visibility / gated paper order controls |
| `KEEP_RISK_ALERTS` | Risk, alerts, approvals, preservation, authorization |
| `KEEP_SETTINGS_DIAGNOSTICS` | Health, contracts, loop diagnostics, gov registries |
| `MERGE_INTO_OPERATIONS` | Parallel app content to fold into CC |
| `MOVE_TO_ADVANCED_DIAGNOSTICS` | Useful but not daily operator nav |
| `RESEARCH_ONLY` | Research/intel; out of operator nav |
| `SIMULATION_RETIRE_CANDIDATE` | Sim/rehearsal UI; page removable after dependency check |
| `BROKEN_OR_UNUSED` | Stub, backup, retired, or superseded |
| `REQUIRES_MANUAL_REVIEW` | Exec/write risk or unclear ownership |

**Retirement candidates were not marked safe to delete** without dependency evidence (imports, links, registry, docs, tests, runtime). Continuous execution does **not** import `dashboard.*` page modules.

---

## 5. Inventory location

Full per-page rows (visible name, path, route, purpose, writes, exec impact, classification, dependency evidence):

- `data/results/dashboard_page_inventory.csv`

Command Center pages are also enumerated in `tmp/do001_registry_extract.json` (working extract; not a phase deliverable).

---

## 6. Exec-capable / write surfaces (operator-relevant)

| Surface | Writes | Affects execution | Notes |
|---------|--------|-------------------|-------|
| Manual Order Desk | yes | yes (paper/broker under gates) | Keep under Orders; protect |
| Run CSV Orders | yes | yes | Keep under Orders; protect |
| Live Trading panel | unknown | gated | Keep under Orders; no live activation in operator nav |
| Human Approval Center | unknown | no direct place | Risk & Alerts |
| `app.py` | yes | yes | Manual review / merge |
| Live Run (Phase 1.5) | no | no | Stub → `BROKEN_OR_UNUSED` |

---

## 7. Auth / safety gap for later phases

Any dashboard runtime controller (Start/Stop/Restart) **must** enforce local-only access. Today Streamlit has **no** in-repo auth or bind restriction. This is a DO-002/DO-003 prerequisite, not fixed in DO-001.

---

## 8. What this phase did **not** do

- No navigation changes
- No page deletes/renames
- No execution logic changes
- No process start/stop
- No lock removal
- No orders / approvals / authorizations
- No 9:31 schedule changes
- No paper/live policy changes
