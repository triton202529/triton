# DO-001 — Simulation Retirement Matrix

**Phase:** DO-001 (read-only)
**Rule:** Do not remove anything in this phase. Retirement candidates require dependency evidence.

---

## 1. Runtime coupling verdict

| Runtime surface | Imports `dashboard.*` sim modules? | Notes |
|-----------------|--------------------------------------|-------|
| Continuous loop | **No** | Page removal does not break continuous execution |
| Watchdog | **No** dashboard sims | Still **generates** defensive/stress/scenario artifacts via services |
| Execution gates | **No** | |
| Broker / place live | **No** | |
| Allocation | Soft | May read concentration recovery artifacts/constants |
| `execute_trades.py` | **Yes** | `adaptation_simulation.apply_runtime_score_influence` — **MUST_RETAIN service** |

---

## 2. Matrix (Command Center simulation-named pages)

| Page / module | Type | Executes sim? | Runtime dep? | Proposed disposition | Evidence |
|---------------|------|---------------|--------------|----------------------|----------|
| Lifecycle Rule Simulation Lab | Artifact viewer | No (UI) | No continuous | Advanced diagnostics / research | `dashboard/lifecycle_rule_simulation_lab.py`; CLI service writes JSON |
| Defensive Simulation Lab | Artifact viewer | No | Artifact gen via watchdog | Keep generation; page → advanced | `defensive_simulation_results.json` |
| Preservation Stress Lab | Artifact viewer | No | Gen via watchdog | Page → advanced/research | `stress_test_results.json` |
| Scenario Planning Center | Artifact viewer | No | Gen via watchdog | Research | `scenario_planning.json` |
| Concentration Recovery Simulation | Artifact viewer | No | Service shared CPS helpers | **Page** removable later; **service MUST_RETAIN** until extract | `concentration_recovery_simulator` imported widely |
| Recovery Basket Exception Simulation | Artifact viewer | No | No | Research | CLI + advisory unused list |
| Final Recovery Authorization Simulation | Artifact viewer | No | No | Research | Dress-rehearsal only |
| Governance Recovery Execution Simulation… | Artifact viewer | No | No | Advanced (unreachable in sidebar) | Certification rehearsal |
| Recovery Path Scenario Forecasting | Artifact viewer | No | Soft (CPS constants) | Advanced | Forecast CSVs/JSON |
| Adaptation Simulation (System) | Artifact viewer | No | Page no | Page removable later | Reads `adaptation_simulation.csv` |
| Protective Action Trials | Viewer / eval | No | Soft (preservation stack) | **Retain under Risk** | Not pure retire |
| Defensive Automation Sandbox | Approval UX (sim-labeled) | No trades | Soft | **Retain under Risk** | Activation queue |

---

## 3. Standalone simulation / forensic labs

| Module | Executes? | Runtime dep? | Disposition |
|--------|-----------|--------------|-------------|
| `dashboard/execution_gate_simulation_dashboard.py` | Yes (button → `run_simulations`) | No continuous | Advanced diagnostics |
| `dashboard/execution_gate_resilience_dashboard.py` | Lab | Soft (cert) | Advanced |
| `dashboard/execution_time_travel_dashboard.py` | Yes (forensic buttons) | No continuous | Advanced diagnostics |
| `services/execution_gate_simulator.py` | CLI/lab | Soft (resilience/investigator) | Retain diagnostics |
| `services/execution_gate_stress_suite.py` | On demand | Soft (cert) | Retain diagnostics |
| `backtest_dashboard.py` (+ services copy) | Backtest UI | No | Orphan retire candidate |
| `services/live_simulator.py` | Pseudo print | No | Orphan retire candidate |
| `services/adaptation_simulation.py` | CLI + runtime hook | **YES** | **MUST_RETAIN** |
| `services/concentration_recovery_simulator.py` | CLI + shared helpers | **YES** | **MUST_RETAIN** until CPS extract |
| ARM / autonomous sim engines | Offline | Soft | Research until ARM doctrine path cleared |

---

## 4. Which pages are merely visual?

Almost all Command Center *Simulation Lab / Stress Lab / Scenario* pages are **read-only artifact viewers**. They do not drive continuous execution.

## 5. Which pages execute simulations?

- Standalone execution-gate simulation dashboard (button-driven)
- Execution time-travel forensic UI
- Offline CLI services (`lifecycle_rule_simulation_lab`, concentration/recovery simulators, gate stress suite)
- Watchdog-path **generation** of defensive/stress/scenario JSON (not dashboard pages)

## 6. Historical artifact display

Defensive / stress / scenario / adaptation / concentration recovery JSON+CSV under `data/results/` and gate results under `data/governance/`.

## 7. Safe eventual page removals (continuous exec unaffected)

After nav redesign + docs/tests scrub:

- Lifecycle Rule Simulation Lab
- Defensive Simulation Lab (page only)
- Preservation Stress Lab (page only)
- Scenario Planning Center
- Concentration Recovery Simulation (**page only**)
- Recovery Basket Exception Simulation
- Final Recovery Authorization Simulation
- Governance Recovery Execution Simulation
- Recovery Path Scenario Forecasting (if forecasts unused elsewhere)
- Adaptation Simulation **page**
- Orphan backtest / live_simulator UIs

**Do not delete** without confirming: no imports, no registry docs requiring UI, no tests launching the page, no operator runbooks.

## 8. Must retain under advanced diagnostics / research

- Execution gate simulation / resilience / time travel tooling
- Protective Action Trials (risk eval)
- Restart diagnostics (ops, not research sim)
- Watchdog artifact generation while certification consumers read those files

## 9. Counts (aligned with inventory)

See CSV `classification == SIMULATION_RETIRE_CANDIDATE` and JSON `simulation_pages` / `simulation_page_names`.

Simulation **page** removal does **not** imply deleting runtime-coupled services.
