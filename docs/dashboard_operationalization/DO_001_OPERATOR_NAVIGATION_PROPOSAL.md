# DO-001 — Operator Navigation Proposal

**Phase:** DO-001 (design only — do not alter navigation yet)

---

## 1. Target six-page operator structure

| # | Nav label | Primary job |
|---|-----------|-------------|
| 1 | **Operations** | Is TRITON running/stopped/degraded/stale? Loop + watchdog health; paper armed; live blocked; Start/Stop/Restart (future); required actions |
| 2 | **Activity** | Plain-language chronological feed from real artifacts/logs |
| 3 | **Portfolio** | Positions, exposure, PnL, trade log |
| 4 | **Orders** | Open/recent orders; gated paper order tools |
| 5 | **Risk & Alerts** | Risk office, alerts, approvals, blocks, preservation posture |
| 6 | **Diagnostics** | Contracts, registries, loop forensics, advanced tools (collapsed) |

There must be **no live-execution activation button** in this operator interface.

---

## 2. Paper Arm placement

**Recommendation: Operations — protected control area** (not a seventh top-level page).

Rationale:

- Arm state is part of “is paper execution armed?” — an Operations posture question
- Belongs next to Start/Stop and live-blocked status
- Requires confirmation / TTL visibility; not a browsing destination
- Separate nav page would dilute daily focus

Do **not** place Paper Arm under Orders (orders consume arm state; Operations owns it).

---

## 3. Mapping from current pages → target nav

### Operations (merge / keep)

| Source (today) | Role in Operations |
|----------------|--------------------|
| Pipeline Health / Heartbeat | Status strip |
| Continuous Loop Health Monitor | Loop health |
| Hung Loop Recovery Intelligence | Degraded/hung signals |
| Execution Dashboard | Paper cycle / arm context |
| Execution / Health | Exec posture |
| Governance Command Center | Compact governance posture |
| Execution Readiness Center | Readiness / live-permitted=false |
| Signal Lifecycle + AI Signals (summary) | “What the system is evaluating” summary cards |
| *(new)* Runtime control panel | Start/Stop/Restart — **not in DO-001** |

`MERGE_INTO_OPERATIONS`: fold useful bits of `services/risk_dashboard.py` into Risk Office / Operations status as needed later.

### Activity (new page)

- Not present today as a unified feed
- Sources: see `DO_001_ACTIVITY_FEED_SOURCE_MAP.md`
- Closest existing: Ticker Decision Timeline (per-ticker, not ops) + governance timelines

### Portfolio

| Keep |
|------|
| Portfolio History |
| Positions & Exposure |
| Profit & Loss Ledger |
| Ledger Reconciliation Center |
| Trade Log |
| Target Weights / Performance Intelligence (optional subsections) |

### Orders

| Keep |
|------|
| Live Orders Panel |
| Manual Order Desk (protected) |
| Run CSV Orders (protected / advanced fold) |
| Live Trading panel (read + gated; **no live enable**) |

### Risk & Alerts

| Keep |
|------|
| Risk Report / Risk Office / Executive Risk Command Center |
| Smart Alerts / Late Exit Detector |
| Human Approval Center |
| Governance Authorization Center + Trace |
| Protective Action Policy / Trials / Evaluation |
| Capital Preservation Governor / Audit / Certification |
| Risk Committee / Accountability / Preservation Board |

### Diagnostics (advanced, collapsed)

| Move here |
|-----------|
| Data Contracts |
| Governance Library + Evidence/Audit/Decision/Escalation registries |
| Traceability / Investigation / Intelligence Lab / Timeline |
| Recovery Restart / Stabilization diagnostics |
| Feedback / Adaptation intelligence |
| Entire concentration/recovery/forecast institutional stack (`MOVE_TO_ADVANCED_DIAGNOSTICS`) |
| Simulation labs (`SIMULATION_RETIRE_CANDIDATE`) — behind “Research / Simulation” subfolder or hidden |
| Unreachable registry pages (wire here or leave hidden until retired) |

### Out of operator nav

| Class | Examples |
|-------|----------|
| `RESEARCH_ONLY` | Feature Importance, Model Comparison, Top Picks, News Sentiment, Economic Calendar, causal/wisdom/scenario research centers |
| `BROKEN_OR_UNUSED` | Live Run stub; legacy `control_center.py` / `dashboard.py` / retired Order Desk |
| Standalone orphans | Prefer not linked from operator nav |

---

## 4. Sidebar information architecture (proposed)

```
TRITON Operator Console
├── Operations          ← default landing
├── Activity
├── Portfolio
├── Orders
├── Risk & Alerts
└── Diagnostics
    ├── Health & Contracts
    ├── Governance Registries
    ├── Advanced Recovery (collapsed)
    └── Research & Simulation (collapsed / restricted)
```

Single composition per page; status strip (running / paper-armed / live-blocked / freshness) sticky across all six.

---

## 5. Explicit exclusions from normal operator UI

- Live execution activation / live arm enable
- Arbitrary shell command runner
- Generic process killer with user-supplied PID
- Simulation “run experiment” buttons on Operations
- Duplicate start authorities presented as equal defaults (prefer one primary Start path; keep Task Scheduler as backup note)

---

## 6. Transition rule (later phases)

1. Add new nav scaffold **without deleting** old SECTIONS (feature flag or dual mode)
2. Move pages by registry remapping
3. Only after Activity + Operations status are proven, hide research/sim from default sidebar
4. Retire pages only with dependency evidence from this audit

DO-001 does **not** change `PAGE_REGISTRY` or `SECTIONS`.
