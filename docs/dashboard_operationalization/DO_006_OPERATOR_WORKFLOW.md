# DO-006 — Operator Workflow

1. Open Operations on localhost.
2. Review Start, Stop, and Restart eligibility.
3. Click **Restart TRITON Safely** when enabled.
4. Read the two-transaction warning.
5. Check the acknowledgement box.
6. Type exactly: `RESTART TRITON`
7. Confirm Restart.
8. Observe Stop stages (if Stop required), then fresh Start evaluation and Start stages.
9. Treat timeout/partial as unresolved — investigate Diagnostics; do not force-kill or delete locks.
10. Standalone Start and Stop remain unavailable while Restart is active.
