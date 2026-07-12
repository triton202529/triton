# Weekend Safe Work Map

Classify TRITON work while the market is closed so documentation and audits can continue
**without** creating execution authority for the next session.

Parent index: [README.md](./README.md)

---

## Explicit safety warning

> **Market-closed work must not create execution authority for the next market session.**

That means: no approval artifacts, no broker authorization artifacts, no broker orders,
no live-gate changes, and no schedule or strategy changes that would arm Monday dispatch.

---

## SAFE

Work that is appropriate over the weekend / after hours:

| Category | Examples |
|----------|----------|
| Documentation | Operator manual pages, phase notes, runbook clarifications |
| Manuals | Index updates, glossaries, cross-links |
| Checklists | Monday open checklist text, pre-flight lists (text only) |
| Read-only audits | Inspecting existing JSON/CSV evidence without rewriting authority |
| Report summaries | Operator-facing markdown summaries of already-certified facts |
| Operator explanations | Why raw handoff qty ≠ governed dispatch qty; what is deferred |

These activities explain and organize. They do **not** authorize paper or live trading.

---

## REQUIRES CAUTION

Allowed only when the task is explicitly dry-run / read-only and leaves no standing authority:

| Category | Guardrails |
|----------|------------|
| Validation scripts | Prefer modules that write `data/results/` evidence only; confirm no approval/auth files created |
| Read-only previews | Do not promote preview rows into dispatch |
| Dry-run compatibility checks | Use `--dry-run` where available; verify artifacts absent after |
| Generated reports | Do not commit unrelated runtime governance churn; restore if dirty |

After any caution work, verify:

```powershell
Test-Path data/governance/operator_paper_defensive_exit_approval.json
Test-Path data/governance/operator_paper_broker_execution_authorization.json
```

Both should be `False` unless an explicit, time-boxed paper session was approved separately
(not part of weekend documentation).

---

## DO NOT TOUCH

Stop. Do not perform these while using this map as the work boundary:

| Area | Why |
|------|-----|
| Broker dispatch | Creates real paper/live order path |
| `manage_positions` execution | Can write handoff / trigger placement paths |
| Approval artifact creation | Creates execution authority for a later session |
| Broker authorization creation | Creates broker execution authority |
| Scheduled tasks | Can arm automated loops overnight into Monday |
| Live execution gates | Live must remain blocked |
| Strategy threshold changes | Alters trading behavior without market-open review |
| BUY/ADD gates | Must not be loosened as side-effect of defensive work |
| Runtime governance state | Do not mutate `data/governance/` authority files |
| `data/live` mutation for execution | Handoff / arm / confirm files are session-sensitive |

---

## Monday restart note

Before any market-open action, rerun:

1. Broker snapshot
2. Open-order check
3. Readiness check
4. Approval / auth artifact checks

Do not carry weekend documentation or dry-run PASS forward as Monday execution permission.

---

## Related certified path (read-only reference)

Partial GLD defensive path (no-execution certified):
[PARTIAL_GLD_DEFENSIVE_PATH.md](./PARTIAL_GLD_DEFENSIVE_PATH.md)

- Full qty **195** · partial **48** · remaining **147**
- Authoritative dispatch qty **48** only
- Raw handoff may show **195** — not dispatch authority
- `execution_allowed_now=false` · live blocked
