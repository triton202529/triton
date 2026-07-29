# DO-004 — Shutdown Request Contract

## Action

`REQUEST_COOPERATIVE_SHUTDOWN` (written only by the backend controller).

## Schema (request)

| Field | Description |
|-------|-------------|
| `schema_version` | Integer (1) |
| `request_id` | UUID |
| `action` | `REQUEST_COOPERATIVE_SHUTDOWN` |
| `requested_at` / `expires_at` | UTC ISO timestamps |
| `target_components` | `continuous` and/or `watchdog` |
| `target` | `CONTINUOUS` \| `WATCHDOG` \| `BOTH` |
| `expected_instances` | Per-component PID, lock identity, command fingerprint |
| `reason_code` | e.g. `OPERATOR_PREPARE` |
| `source` | Controller source label |
| `controller_version` | `DO-004` |
| `state` | Lifecycle state |

## Forbidden in request

Executable commands, scripts, arbitrary paths, shell args, environment dicts, credentials, unvalidated PIDs, free-form command instructions.

## Acknowledgement schema

`request_id`, `component`, `acknowledged_at`, `state`, `actual_pid`, `command_fingerprint`, `lock_identity`, `safe_boundary`, `current_cycle_state`, `next_action`, `safe_message`.

### Ack states

`ACCEPTED`, `REFUSED_WRONG_INSTANCE`, `REFUSED_EXPIRED`, `REFUSED_MALFORMED`, `REFUSED_UNSUPPORTED_ACTION`, `REFUSED_UNSAFE_BOUNDARY`, `COMPLETING_CURRENT_SAFE_UNIT`, `EXITING`, `FAILED`.

## Atomic write

Temp file in same directory → flush/fsync → `os.replace` onto fixed path.

## Replay / expiry

Expired requests refused. Expected PID must match the live instance. Old PID cannot stop a new instance.
