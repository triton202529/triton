# DO-006 — Restart Confirmation Contract

## Required confirmation

1. Acknowledgement checkbox checked
2. Exact case-sensitive phrase: `RESTART TRITON`

## Rejected inputs

- lowercase / mixed case
- leading, trailing, or extra internal whitespace
- partial phrases
- empty input
- unchecked acknowledgement
- `STOP TRITON` (Stop phrase cannot authorize Restart)

## Operator explanation

Restart TRITON performs a verified cooperative Stop followed by a fresh protected Start. Active bounded work may first reach its documented safe boundary. Restart will not continue unless Stop fully completes and a new Start evaluation passes.

## Must not imply

- immediate termination
- guaranteed completion
- forced termination
- automatic lock repair
- automatic paper authorization
- live-execution enablement
