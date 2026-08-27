"""
paper_sell_arm_authority.py
Phase 198H — Batch-specific paper sell-arm authority (two-step).

Operator semantics (dashboard wiring is a later phase):
  1) Arm Reviewed Sells  -> confirmation challenge only (NO authority)
  2) Confirm & Arm Paper Sells -> short-lived single-use PAPER authority (NO dispatch)

Defaults are fail-closed and non-executing. Writers require an explicit store root.
Writing into the live repository governance path additionally requires
explicit_runtime_write=True (forbidden for Phase 198H validation runs).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
LIVE_GOVERNANCE = ROOT / "data" / "governance"
# Phase 198I-FIX — hardened authority has a reserved canonical filename.
HARDENED_AUTHORITY_FILENAME = "paper_sell_arm_authority.json"
LIVE_APPROVAL_PATH = LIVE_GOVERNANCE / HARDENED_AUTHORITY_FILENAME
# Deprecated shared path (must never be written by legacy 197W; collision fails closed).
DEPRECATED_SHARED_APPROVAL_FILENAME = "operator_paper_defensive_exit_approval.json"
DEPRECATED_SHARED_APPROVAL_PATH = LIVE_GOVERNANCE / DEPRECATED_SHARED_APPROVAL_FILENAME
# Phase 197W historical evidence only — never runtime-authoritative.
LEGACY_197W_APPROVAL_FILENAME = "operator_paper_defensive_exit_approval_legacy_197w.json"
LEGACY_197W_APPROVAL_PATH = LIVE_GOVERNANCE / LEGACY_197W_APPROVAL_FILENAME
LIVE_BUY_ARM_PATH = ROOT / "data" / "live" / "paper_arm_confirm.json"

SCHEMA_VERSION = "198H-PAPER-SELL-ARM-1"
SCHEMA_VERSION_DUAL = "198N-E-DUAL-IDENTITY-SELL-1"
AUTHORITY_TYPE = "PAPER_SELL_ARM"
CHALLENGE_ARTIFACT_TYPE = "PAPER_SELL_CONFIRMATION_CHALLENGE"
READINESS_STATUS_OK = "UNAPPROVED_APPROVAL_READINESS_ONLY"

OPERATOR_ACKNOWLEDGEMENT_VERSION = "198H-ACK-1"
OPERATOR_ACKNOWLEDGEMENT_TEXT = (
    "I reviewed these paper sell actions and authorize TRITON to dispatch only "
    "this exact batch, subject to final safety revalidation."
)

# Lifetimes (minutes) — one freshness contract shared by F2 writer, console,
# and runtime enforcement. Packet must remain eligible at the next 45-minute
# loop (44m and 46m) and expire before AUTHORITY_TTL with a safety margin.
CONTINUOUS_LOOP_INTERVAL_MINUTES = 45
READINESS_REFERENCE_TTL_MINUTES = 50
CHALLENGE_TTL_MINUTES = 10
# Phase 198Q-B: 45-minute continuous-loop interval + 15-minute execution margin.
# New authorities only; stored expiry on existing rows is never rewritten.
AUTHORITY_TTL_MINUTES = 60

# Caps — lower applicable cap governs authority creation.
APPROVAL_CLI_CAP = 5000.0
RUNTIME_MANAGE_CAP = 50000.0

SELL_ACTIONS = frozenset({"TRIM", "EXIT", "ROTATE_EXIT", "FORCE_EXIT"})
BUY_ACTIONS = frozenset({"BUY", "ADD", "ENTRY", "ROTATE_ENTRY"})


class SellArmState(str, Enum):
    NO_SELLS_WAITING = "NO_SELLS_WAITING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    READY_FOR_CONFIRMATION = "READY_FOR_CONFIRMATION"
    CONFIRMATION_CHALLENGE_ISSUED = "CONFIRMATION_CHALLENGE_ISSUED"
    ARMED = "ARMED"
    STALE = "STALE"
    EXPIRED = "EXPIRED"
    BLOCKED = "BLOCKED"
    DISPATCH_IN_PROGRESS = "DISPATCH_IN_PROGRESS"
    CONSUMED = "CONSUMED"
    REJECTED = "REJECTED"


REQUIRED_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "authority_type",
        "status",
        "mode",
        "account_fingerprint",
        "trading_date",
        "cycle_id",
        "run_id",
        "batch_id",
        "plan_id",
        "plan_hash",
        "readiness_packet_hash",
        "confirmation_challenge_id",
        "confirmation_challenge_hash",
        "issued_at",
        "expires_at",
        "issued_by",
        "operator_acknowledgement_version",
        "operator_acknowledgement_text",
        "authorized_actions",
        "per_symbol_max_quantity",
        "total_batch_max_notional",
        "applied_cap",
        "applied_cap_source",
        "market_session_requirement",
        "single_use",
        "consumption_status",
        "live_execution_required_state",
        "source_provenance",
        "audit_id",
        "idempotency_key",
        "no_symbol_substitution",
        "no_action_substitution",
        "no_quantity_expansion",
        "no_batch_expansion",
        "creates_dispatch_authority",
        "orders_submitted",
    }
)

REQUIRED_CHALLENGE_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "challenge_id",
        "challenge_hash",
        "nonce",
        "issued_at",
        "expires_at",
        "mode",
        "account_fingerprint",
        "trading_date",
        "cycle_id",
        "batch_id",
        "plan_hash",
        "readiness_packet_hash",
        "authorized_scope",
        "creates_no_authority",
        "status",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Naive timestamps fail closed for authority decisions.
            return None
        return dt
    except Exception:
        return None


def readiness_packet_valid_until(
    now: datetime, *, session_close: Optional[datetime] = None
) -> datetime:
    """Cap packet lifetime at TTL and never past the authorized session close."""
    if now.tzinfo is None:
        raise ValueError("timestamp_must_be_timezone_aware")
    cap = now.astimezone(timezone.utc) + timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES)
    if session_close is None:
        return cap
    close = session_close if session_close.tzinfo else None
    if close is None:
        return cap
    close_utc = close.astimezone(timezone.utc)
    return close_utc if close_utc < cap else cap


def evaluate_readiness_packet_freshness(
    packet: Mapping[str, Any],
    *,
    now: datetime,
    session_payload: Optional[Mapping[str, Any]] = None,
    check_session_boundary: bool = True,
) -> Tuple[bool, str, Optional[datetime]]:
    """Shared freshness contract for writer, controller, and runtime enforcement."""
    if now.tzinfo is None:
        return False, "naive_now", None
    now_utc = now.astimezone(timezone.utc)
    if "generated_at" not in packet or packet.get("generated_at") in (None, ""):
        return False, "missing_generated_at", None
    generated = _parse_iso(packet.get("generated_at"))
    if generated is None:
        raw = packet.get("generated_at")
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return False, "unparseable_generated_at", None
        if dt.tzinfo is None:
            return False, "naive_generated_at", None
        generated = dt
    generated_utc = generated.astimezone(timezone.utc)
    if generated_utc > now_utc:
        return False, "future_generated_at", generated_utc
    age = now_utc - generated_utc
    if age > timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES):
        return False, "stale_readiness_packet", generated_utc
    valid_until = _parse_iso(packet.get("valid_until"))
    if valid_until is not None and now_utc >= valid_until.astimezone(timezone.utc):
        return False, "stale_readiness_packet", generated_utc

    if check_session_boundary:
        payload: Mapping[str, Any]
        if session_payload is not None:
            payload = session_payload
        else:
            from services.market_calendar import regular_session_state

            payload = regular_session_state(now=now_utc)
        state = str(payload.get("session_state") or "").upper()
        if state == "UNKNOWN":
            return False, "session_derivation_unknown", generated_utc
        if state == "CLOSED":
            return False, "session_boundary_closed", generated_utc
        if state != "OPEN":
            return False, "session_boundary_closed", generated_utc
        current_session_date = str(payload.get("session_date") or "")
        packet_session = str(packet.get("session_date") or "").strip()
        if not packet_session:
            try:
                from zoneinfo import ZoneInfo

                packet_session = (
                    generated_utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()
                )
            except Exception:
                packet_session = generated_utc.date().isoformat()
        if current_session_date and packet_session and packet_session != current_session_date:
            return False, "prior_session_packet", generated_utc
    return True, "ok", generated_utc


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


_TERMINAL_AUTHORITY_STATUSES = frozenset(
    {
        "CONSUMED",
        "DISPATCH_IN_PROGRESS",
        "REJECTED",
        "ATTEMPTED",
        "RESERVED",
    }
)


# Last atomic-write durability observation (tests/ops). Not a security control.
# Dispatch permit issuance requires file_and_dir_fsync for intent + consumption.
# atomic_replace_only / directory-fsync failure / uncertain writes are insufficient
# for broker-usable permits on every platform (including Windows).
LAST_ATOMIC_WRITE_DURABILITY: Dict[str, Any] = {
    "guaranteed": "unset",
    "file_fsync_ok": None,
    "dir_fsync_ok": None,
    "path": None,
}
REQUIRED_DISPATCH_DURABILITY_POSIX = "file_and_dir_fsync"
REQUIRED_DISPATCH_DURABILITY_WINDOWS = "file_fsync_atomic_replace_windows"  # legacy label only
REQUIRED_DISPATCH_DURABILITY = "file_and_dir_fsync"


class DispatchDurabilityError(RuntimeError):
    """Raised when a dispatch-critical write cannot establish required durability."""

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _platform_dispatch_durability_standard() -> str:
    """
    Declared durability standard for broker-usable paper-sell permits.

    All platforms: temporary-file create/write/flush + file fsync + atomic replace
    + successful directory durability flush that is proven for the crash class.

    POSIX: directory descriptor fsync may satisfy process-crash metadata needs.
    Windows: CreateFileW(FILE_FLAG_BACKUP_SEMANTICS)+FlushFileBuffers(GENERIC_WRITE)
    may return success and is recorded for diagnostics, but primary Microsoft
    documentation does not establish that this alone proves durable directory-entry
    replacement metadata across OS crash or power loss. Until that contract is
    proven, Windows broker-usable paper-sell permits remain fail-closed.
    """
    return REQUIRED_DISPATCH_DURABILITY


def durability_meets_dispatch_standard(meta: Dict[str, Any]) -> bool:
    if meta.get("flush_ok") is False:
        return False
    if not meta.get("file_fsync_ok"):
        return False
    # Windows: FlushFileBuffers API success is not accepted as proof of
    # OS-crash/power-loss directory-entry durability for rename metadata.
    if sys.platform == "win32":
        return False
    return bool(meta.get("dir_fsync_ok"))


def _fsync_dir_windows(dir_path: Path) -> bool:
    """
    Windows directory metadata flush via CreateFileW(FILE_FLAG_BACKUP_SEMANTICS)
    + FlushFileBuffers with GENERIC_WRITE.

    Authoritative API docs:
    - CreateFileW FILE_FLAG_BACKUP_SEMANTICS to obtain a directory handle
      (https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)
    - FlushFileBuffers to flush buffered metadata for a handle
      (https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers)

    Returns False on any failure/unsupported/ambiguous result.
    A True return means only that FlushFileBuffers succeeded for the directory
    handle. It is diagnostic evidence only and must not be treated as proven
    OS-crash/power-loss directory-entry durability for broker-usable permits.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False

    generic_write = 0x40000000
    file_share_read = 0x1
    file_share_write = 0x2
    file_share_delete = 0x4
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    invalid_handle = wintypes.HANDLE(-1).value

    create_file_w = ctypes.windll.kernel32.CreateFileW
    create_file_w.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file_w.restype = wintypes.HANDLE
    flush_file_buffers = ctypes.windll.kernel32.FlushFileBuffers
    close_handle = ctypes.windll.kernel32.CloseHandle

    handle = create_file_w(
        str(dir_path),
        generic_write,
        file_share_read | file_share_write | file_share_delete,
        None,
        open_existing,
        file_flag_backup_semantics,
        None,
    )
    if int(handle) in {invalid_handle, -1}:
        return False
    try:
        return bool(flush_file_buffers(handle))
    except Exception:
        return False
    finally:
        try:
            close_handle(handle)
        except Exception:
            pass


def _fsync_dir(dir_path: Path) -> bool:
    """Directory durability. Returns False when unsupported/failed/uncertain."""
    if sys.platform == "win32":
        if _fsync_dir_windows(dir_path):
            return True
        # Fall through to POSIX-style attempt; still fail closed if both fail.
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
    except (AttributeError, OSError, NotImplementedError):
        return False
    try:
        os.fsync(fd)
        return True
    except (OSError, AttributeError, NotImplementedError):
        return False
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def _atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
    *,
    refuse_authority_downgrade: bool = False,
    require_full_durability: bool = False,
) -> Path:
    """
    Durable atomic JSON write: temp in same dir → flush/fsync file → replace → fsync dir.

    When require_full_durability=True (dispatch intent/consumption/permit), both file
    fsync and directory fsync must succeed on every platform. Otherwise
    DispatchDurabilityError is raised after the write attempt and callers must not
    issue a broker-usable permit. Atomic replacement alone is not sufficient.

    When refuse_authority_downgrade=True, never let ARMED/UNUSED overwrite terminal states.
    """
    global LAST_ATOMIC_WRITE_DURABILITY
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_authority_downgrade and path.is_file():
        existing = _read_json(path)
        ex_status = str(existing.get("status") or "")
        ex_cons = str(existing.get("consumption_status") or "")
        new_status = str(payload.get("status") or "")
        new_cons = str(payload.get("consumption_status") or "")
        if ex_cons == "CONSUMED" or ex_status in _TERMINAL_AUTHORITY_STATUSES:
            if new_status == "ARMED" or new_cons == "UNUSED":
                raise PermissionError(
                    "refuse_stale_armed_overwrite_of_consumed_or_terminal_authority"
                )
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, default=str)
    file_fsync_ok = True
    flush_ok = True
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(data)
            try:
                fh.flush()
            except (OSError, AttributeError, NotImplementedError) as exc:
                flush_ok = False
                LAST_ATOMIC_WRITE_DURABILITY = {
                    "path": str(path),
                    "file_fsync_ok": False,
                    "dir_fsync_ok": False,
                    "flush_ok": False,
                    "guaranteed": "none",
                    "uncertain": True,
                    "error": f"flush:{exc}",
                }
                raise DispatchDurabilityError(
                    "temp_flush_failed", dict(LAST_ATOMIC_WRITE_DURABILITY)
                ) from exc
            try:
                os.fsync(fh.fileno())
            except (OSError, AttributeError, NotImplementedError):
                file_fsync_ok = False
    except DispatchDurabilityError:
        raise
    try:
        os.replace(tmp, path)
    except OSError as exc:
        LAST_ATOMIC_WRITE_DURABILITY = {
            "path": str(path),
            "file_fsync_ok": file_fsync_ok,
            "dir_fsync_ok": False,
            "flush_ok": flush_ok,
            "guaranteed": "none",
            "uncertain": True,
            "error": f"replace:{exc}",
        }
        raise DispatchDurabilityError(
            "atomic_replace_failed", dict(LAST_ATOMIC_WRITE_DURABILITY)
        ) from exc
    dir_fsync_ok = _fsync_dir(path.parent)
    standard = _platform_dispatch_durability_standard()
    if file_fsync_ok and dir_fsync_ok:
        guaranteed = REQUIRED_DISPATCH_DURABILITY
    elif file_fsync_ok:
        guaranteed = "atomic_replace_only"
    else:
        guaranteed = "none"
    meta = {
        "path": str(path),
        "file_fsync_ok": file_fsync_ok,
        "dir_fsync_ok": dir_fsync_ok,
        "flush_ok": flush_ok,
        "guaranteed": guaranteed,
        "platform_standard": standard,
        "uncertain": not durability_meets_dispatch_standard(
            {
                "flush_ok": flush_ok,
                "file_fsync_ok": file_fsync_ok,
                "dir_fsync_ok": dir_fsync_ok,
            }
        ),
    }
    LAST_ATOMIC_WRITE_DURABILITY = meta
    if require_full_durability and meta["uncertain"]:
        raise DispatchDurabilityError("dispatch_durability_uncertain", dict(meta))
    return path


def detect_hardened_authority_path_collision(
    store_root: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Fail closed when default path selection is ambiguous:
    - both hardened and deprecated shared paths present
    - hardened schema written onto the deprecated shared filename
    Presence of legacy residue alone on the deprecated filename is handled by
    refusing to use that path as authority (runtime reads hardened only).
    """
    root = store_root or LIVE_GOVERNANCE
    hardened = root / HARDENED_AUTHORITY_FILENAME
    deprecated = root / DEPRECATED_SHARED_APPROVAL_FILENAME
    if deprecated.is_file() and hardened.is_file():
        return True, "hardened_and_deprecated_approval_paths_both_present"
    if deprecated.is_file():
        doc = _read_json(deprecated)
        if (
            doc.get("authority_type") == AUTHORITY_TYPE
            or str(doc.get("schema_version") or "") == SCHEMA_VERSION
        ):
            return True, "hardened_schema_found_on_deprecated_shared_path"
    return False, "ok"


def canonical_json_hash(doc: Dict[str, Any]) -> str:
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sanitize_account_fingerprint(raw: Optional[str]) -> str:
    text = str(raw or "").strip()
    if not text:
        return "unknown_account"
    # Never store raw account numbers — fingerprint only.
    return "acctfp_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def applied_authority_cap() -> Tuple[float, str]:
    """Lower applicable cap governs authority creation (do not raise)."""
    return float(APPROVAL_CLI_CAP), "APPROVAL_CLI_CAP"


@dataclass(frozen=True)
class PaperSellArmPaths:
    root: Path

    @property
    def challenge(self) -> Path:
        return self.root / "paper_sell_confirmation_challenge.json"

    @property
    def authority(self) -> Path:
        return self.root / HARDENED_AUTHORITY_FILENAME

    @property
    def consumption(self) -> Path:
        return self.root / "paper_sell_arm_consumption.json"

    @property
    def audit(self) -> Path:
        return self.root / "paper_sell_arm_audit.jsonl"

    @property
    def state(self) -> Path:
        return self.root / "paper_sell_arm_state.json"


def _assert_writable_root(root: Path, *, explicit_runtime_write: bool) -> PaperSellArmPaths:
    root = root.resolve()
    live_gov = LIVE_GOVERNANCE.resolve()
    if root == live_gov or str(root).startswith(str(live_gov) + os.sep):
        if not explicit_runtime_write:
            raise PermissionError(
                "Refusing write into live governance without explicit_runtime_write=True"
            )
    return PaperSellArmPaths(root=root)


def append_audit(paths: PaperSellArmPaths, event: Dict[str, Any]) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _utc_now().isoformat(),
        **event,
        "orders_submitted": False,
    }
    with paths.audit.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def write_state(paths: PaperSellArmPaths, state: SellArmState, detail: Dict[str, Any]) -> None:
    payload = {
        "state": state.value,
        "updated_at": _utc_now().isoformat(),
        "detail": detail,
    }
    _atomic_write_json(paths.state, payload)


def classify_readiness_packet(doc: Dict[str, Any]) -> Tuple[bool, str]:
    if not doc:
        return False, "readiness_missing"
    if str(doc.get("status") or "") != READINESS_STATUS_OK:
        return False, "readiness_status_not_unapproved_readiness"
    if doc.get("creates_no_authority") is not True:
        return False, "readiness_missing_creates_no_authority"
    if str(doc.get("mode") or "").upper() != "PAPER":
        return False, "readiness_mode_not_paper"
    if str(doc.get("live_execution_state") or "").upper() != "BLOCKED":
        return False, "readiness_live_not_blocked"
    if not doc.get("plan_hash") or not doc.get("batch_id") or not doc.get("cycle_id"):
        return False, "readiness_missing_binding_ids"
    ready = list(doc.get("approval_ready_symbols") or [])
    actions = list(doc.get("approval_ready_actions") or [])
    if not ready or not actions:
        return False, "readiness_has_no_approval_ready_actions"
    return True, "ok"


def readiness_packet_hash(doc: Dict[str, Any]) -> str:
    # Bind to material scope, not volatile generation timestamps alone.
    material = {
        "status": doc.get("status"),
        "mode": doc.get("mode"),
        "cycle_id": doc.get("cycle_id"),
        "run_id": doc.get("run_id"),
        "batch_id": doc.get("batch_id"),
        "plan_id": doc.get("plan_id"),
        "plan_hash": doc.get("plan_hash"),
        "approval_ready_actions": doc.get("approval_ready_actions"),
        "approval_ready_symbols": doc.get("approval_ready_symbols"),
        "total_proposed_notional": doc.get("total_proposed_notional"),
        "schema_version": doc.get("schema_version"),
    }
    return canonical_json_hash(material)


def source_readiness_packet_hash(doc: Mapping[str, Any]) -> str:
    from services.paper_sell_arm_cap_slice import source_readiness_packet_hash as _src

    return _src(doc)


def is_dual_identity_record(doc: Optional[Mapping[str, Any]]) -> bool:
    if not doc:
        return False
    return str(doc.get("identity_schema") or "") == SCHEMA_VERSION_DUAL


def legacy_ambiguous_slice_authority(
    authority: Mapping[str, Any],
    *,
    readiness_packet: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Fail closed on 198H authorities that exceed the $5,000 CLI cap.

    Hash mismatch versus a later F2 packet is handled by existing
    readiness_packet_hash_changed revalidation so under-cap 198H/198I
    callers remain unchanged even if an unrelated over-cap packet exists
    on the default results path.
    """
    if is_dual_identity_record(authority):
        return None
    try:
        total = float(authority.get("total_batch_max_notional") or 0.0)
    except (TypeError, ValueError):
        return "legacy_over_cap_authority_ambiguous"
    if total > APPROVAL_CLI_CAP + 1e-9:
        return "legacy_over_cap_authority_ambiguous"
    return None


def _scope_from_readiness(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in doc.get("approval_ready_actions") or []:
        sym = str(row.get("symbol") or "").upper()
        action = str(row.get("action") or "").upper()
        qty = float(row.get("qty") or 0)
        notional = float(row.get("est_notional") or 0)
        if not sym or action not in SELL_ACTIONS or qty <= 0:
            continue
        out.append(
            {
                "symbol": sym,
                "action": action,
                "max_qty": qty,
                "max_notional": notional if notional > 0 else None,
            }
        )
    return out


def prepare_confirmation_challenge(
    *,
    store_root: Path,
    readiness_packet: Dict[str, Any],
    account_raw: str = "paper-test-account",
    trading_date: Optional[str] = None,
    now: Optional[datetime] = None,
    explicit_runtime_write: bool = False,
    db_path: Optional[Path] = None,
    authorized_slice: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    BUTTON 1 semantics: issue confirmation challenge only.
    Creates no execution authority. Never writes live governance by default.
    authorized_slice=None retains 198H/198I full-packet behavior.
    """
    paths = _assert_writable_root(store_root, explicit_runtime_write=explicit_runtime_write)
    ok, reason = classify_readiness_packet(readiness_packet)
    if not ok:
        write_state(paths, SellArmState.BLOCKED, {"reason": reason})
        append_audit(paths, {"event": "challenge_blocked", "reason": reason})
        raise ValueError(f"challenge_blocked:{reason}")

    now = now or _utc_now()
    cap_value, cap_source = applied_authority_cap()
    dual = authorized_slice is not None
    if dual:
        from services.paper_sell_arm_cap_slice import (
            authorized_slice_hash as _slice_hash,
            validate_slice_against_packet,
        )

        ok_slice, slice_reason, scope = validate_slice_against_packet(
            readiness_packet,
            authorized_slice,
            cap=cap_value,
            cap_source=cap_source,
        )
        if not ok_slice:
            mapped = slice_reason
            if slice_reason in {"scope_exceeds_approval_cli_cap"} or str(slice_reason).startswith(
                "symbol_exceeds_cap:"
            ):
                mapped = slice_reason
            write_state(paths, SellArmState.BLOCKED, {"reason": mapped})
            append_audit(paths, {"event": "challenge_blocked", "reason": mapped})
            raise ValueError(f"challenge_blocked:{mapped}")
    else:
        scope = _scope_from_readiness(readiness_packet)
    if not scope:
        raise ValueError("challenge_blocked:empty_scope")

    total_notional = round(sum(float(s.get("max_notional") or 0.0) for s in scope), 2)
    if total_notional > cap_value + 1e-9:
        write_state(paths, SellArmState.BLOCKED, {"reason": "scope_exceeds_approval_cli_cap"})
        append_audit(
            paths,
            {
                "event": "challenge_blocked",
                "reason": "scope_exceeds_approval_cli_cap",
                "total_notional": total_notional,
                "cap": cap_value,
            },
        )
        raise ValueError("challenge_blocked:scope_exceeds_approval_cli_cap")

    for item in scope:
        n = item.get("max_notional")
        if n is not None and float(n) > cap_value + 1e-9:
            raise ValueError(f"challenge_blocked:symbol_exceeds_cap:{item['symbol']}")

    challenge_id = f"sell-chal-{uuid.uuid4().hex[:16]}"
    nonce = uuid.uuid4().hex
    td = trading_date or now.date().isoformat()
    acct = sanitize_account_fingerprint(account_raw)
    pkt_hash = readiness_packet_hash(readiness_packet)
    expires = now + timedelta(minutes=CHALLENGE_TTL_MINUTES)
    src_hash = source_readiness_packet_hash(readiness_packet) if dual else None
    slice_h = None
    if dual:
        from services.paper_sell_arm_cap_slice import (
            authorized_slice_hash as _slice_hash,
        )

        slice_h = _slice_hash(
            scope=scope,
            slice_total=total_notional,
            applied_cap=cap_value,
            cap_source=cap_source,
        )

    material = {
        "challenge_id": challenge_id,
        "nonce": nonce,
        "mode": "PAPER",
        "account_fingerprint": acct,
        "trading_date": td,
        "cycle_id": readiness_packet.get("cycle_id"),
        "batch_id": readiness_packet.get("batch_id"),
        "plan_hash": readiness_packet.get("plan_hash"),
        "readiness_packet_hash": pkt_hash,
        "authorized_scope": scope,
        "applied_cap": cap_value,
        "applied_cap_source": cap_source,
    }
    if dual:
        material["identity_schema"] = SCHEMA_VERSION_DUAL
        material["source_readiness_packet_hash"] = src_hash
        material["authorized_slice_hash"] = slice_h
        material["source_plan_sha256"] = readiness_packet.get("source_plan_sha256")
        material["selected_row_hash"] = readiness_packet.get("selected_row_hash")
        material["generated_at"] = readiness_packet.get("generated_at")
        material["market_open_invocation_id"] = readiness_packet.get(
            "market_open_invocation_id"
        ) or readiness_packet.get("invocation_id")
    challenge_hash = canonical_json_hash(material)
    challenge = {
        **material,
        "artifact_type": CHALLENGE_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "challenge_hash": challenge_hash,
        "issued_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "run_id": readiness_packet.get("run_id"),
        "plan_id": readiness_packet.get("plan_id"),
        "status": SellArmState.CONFIRMATION_CHALLENGE_ISSUED.value,
        "creates_no_authority": True,
        "orders_submitted": False,
        "approval_created": False,
        "live_execution_state": "BLOCKED",
        "operator_button": "ARM_REVIEWED_SELLS",
    }
    missing = REQUIRED_CHALLENGE_FIELDS - set(challenge.keys())
    if missing:
        raise ValueError(f"challenge_schema_incomplete:{sorted(missing)}")

    from services.paper_sell_arm_store import (
        PaperSellArmStoreError,
        persist_open_challenge,
        resolve_runtime_sqlite_path,
        sqlite_permit_grade_enabled,
    )

    if sqlite_permit_grade_enabled():
        try:
            persist_open_challenge(
                resolve_runtime_sqlite_path(db_path=db_path),
                challenge=challenge,
                now=now,
            )
        except PaperSellArmStoreError as exc:
            raise ValueError(f"challenge_blocked:{exc.reason}") from exc
    # JSON is AUDIT_EXPORT_ONLY after SQLite commit (or fail-closed JSON rollback).
    _atomic_write_json(paths.challenge, challenge)
    # Ensure no authority file is created by step 1.
    write_state(
        paths,
        SellArmState.CONFIRMATION_CHALLENGE_ISSUED,
        {"challenge_id": challenge_id, "creates_no_authority": True},
    )
    append_audit(
        paths,
        {
            "event": "confirmation_challenge_issued",
            "challenge_id": challenge_id,
            "challenge_hash": challenge_hash,
            "plan_hash": challenge["plan_hash"],
            "batch_id": challenge["batch_id"],
            "creates_no_authority": True,
            "authority_exists": paths.authority.is_file(),
        },
    )
    return challenge


def _validate_challenge_doc(
    challenge: Dict[str, Any],
    *,
    readiness_packet: Dict[str, Any],
    now: datetime,
) -> Tuple[bool, str]:
    missing = REQUIRED_CHALLENGE_FIELDS - set(challenge.keys())
    if missing:
        return False, f"challenge_missing_fields:{sorted(missing)}"
    if challenge.get("artifact_type") != CHALLENGE_ARTIFACT_TYPE:
        return False, "challenge_wrong_artifact_type"
    if challenge.get("creates_no_authority") is not True:
        return False, "challenge_must_create_no_authority"
    if str(challenge.get("mode") or "").upper() != "PAPER":
        return False, "challenge_mode_not_paper"
    exp = _parse_iso(challenge.get("expires_at"))
    issued = _parse_iso(challenge.get("issued_at"))
    if not exp or not issued:
        return False, "challenge_timestamp_unparseable"
    if exp <= now:
        return False, "challenge_expired"
    if str(challenge.get("plan_hash") or "") != str(readiness_packet.get("plan_hash") or ""):
        return False, "challenge_plan_hash_mismatch"
    if str(challenge.get("batch_id") or "") != str(readiness_packet.get("batch_id") or ""):
        return False, "challenge_batch_id_mismatch"
    if str(challenge.get("cycle_id") or "") != str(readiness_packet.get("cycle_id") or ""):
        return False, "challenge_cycle_id_mismatch"
    expected_pkt = readiness_packet_hash(readiness_packet)
    if str(challenge.get("readiness_packet_hash") or "") != expected_pkt:
        return False, "challenge_readiness_hash_mismatch"
    if is_dual_identity_record(challenge):
        expected_src = source_readiness_packet_hash(readiness_packet)
        if str(challenge.get("source_readiness_packet_hash") or "") != expected_src:
            return False, "challenge_source_readiness_hash_mismatch"
        if str(challenge.get("generated_at") or "") != str(
            readiness_packet.get("generated_at") or ""
        ):
            return False, "challenge_generated_at_mismatch"
        inv = readiness_packet.get("market_open_invocation_id") or readiness_packet.get(
            "invocation_id"
        )
        if str(challenge.get("market_open_invocation_id") or "") != str(inv or ""):
            return False, "challenge_invocation_mismatch"
        if str(challenge.get("source_plan_sha256") or "") != str(
            readiness_packet.get("source_plan_sha256") or ""
        ):
            return False, "challenge_source_plan_hash_mismatch"
        if str(challenge.get("selected_row_hash") or "") != str(
            readiness_packet.get("selected_row_hash") or ""
        ):
            return False, "challenge_selected_row_hash_mismatch"
    material = {
        "challenge_id": challenge.get("challenge_id"),
        "nonce": challenge.get("nonce"),
        "mode": challenge.get("mode"),
        "account_fingerprint": challenge.get("account_fingerprint"),
        "trading_date": challenge.get("trading_date"),
        "cycle_id": challenge.get("cycle_id"),
        "batch_id": challenge.get("batch_id"),
        "plan_hash": challenge.get("plan_hash"),
        "readiness_packet_hash": challenge.get("readiness_packet_hash"),
        "authorized_scope": challenge.get("authorized_scope"),
        "applied_cap": challenge.get("applied_cap"),
        "applied_cap_source": challenge.get("applied_cap_source"),
    }
    if is_dual_identity_record(challenge):
        material["identity_schema"] = challenge.get("identity_schema")
        material["source_readiness_packet_hash"] = challenge.get("source_readiness_packet_hash")
        material["authorized_slice_hash"] = challenge.get("authorized_slice_hash")
        material["source_plan_sha256"] = challenge.get("source_plan_sha256")
        material["selected_row_hash"] = challenge.get("selected_row_hash")
        material["generated_at"] = challenge.get("generated_at")
        material["market_open_invocation_id"] = challenge.get("market_open_invocation_id")
    if canonical_json_hash(material) != str(challenge.get("challenge_hash") or ""):
        return False, "challenge_hash_integrity_failed"
    return True, "ok"


def confirm_and_create_paper_sell_arm(
    *,
    store_root: Path,
    readiness_packet: Dict[str, Any],
    challenge: Dict[str, Any],
    operator_acknowledgement_text: str,
    issued_by: str = "operator",
    current_plan_hash: Optional[str] = None,
    newer_plan_exists: bool = False,
    now: Optional[datetime] = None,
    explicit_runtime_write: bool = False,
    db_path: Optional[Path] = None,
    authorized_slice: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    BUTTON 2 semantics: create batch-specific paper sell authority.
    Does not dispatch orders. Refuses live governance writes by default.
    """
    paths = _assert_writable_root(store_root, explicit_runtime_write=explicit_runtime_write)
    now = now or _utc_now()

    if operator_acknowledgement_text.strip() != OPERATOR_ACKNOWLEDGEMENT_TEXT:
        write_state(paths, SellArmState.REJECTED, {"reason": "acknowledgement_mismatch"})
        append_audit(paths, {"event": "arm_rejected", "reason": "acknowledgement_mismatch"})
        raise ValueError("arm_rejected:acknowledgement_mismatch")

    ok, reason = classify_readiness_packet(readiness_packet)
    if not ok:
        raise ValueError(f"arm_rejected:{reason}")

    cok, creason = _validate_challenge_doc(challenge, readiness_packet=readiness_packet, now=now)
    if not cok:
        write_state(paths, SellArmState.REJECTED, {"reason": creason})
        append_audit(paths, {"event": "arm_rejected", "reason": creason})
        raise ValueError(f"arm_rejected:{creason}")

    cap_value, cap_source = applied_authority_cap()
    dual = is_dual_identity_record(challenge) or authorized_slice is not None
    if dual:
        from services.paper_sell_arm_cap_slice import (
            authorized_slice_hash as _slice_hash,
            validate_slice_against_packet,
        )

        slice_doc = authorized_slice or {
            "slice_actions": challenge.get("authorized_scope") or [],
            "authorized_slice_hash": challenge.get("authorized_slice_hash"),
            "slice_hash": challenge.get("authorized_slice_hash"),
        }
        ok_slice, slice_reason, slice_scope = validate_slice_against_packet(
            readiness_packet,
            slice_doc,
            cap=cap_value,
            cap_source=cap_source,
        )
        if not ok_slice:
            raise ValueError(f"arm_rejected:{slice_reason}")
        chal_slice = str(challenge.get("authorized_slice_hash") or "")
        recomputed_slice = _slice_hash(
            scope=slice_scope,
            slice_total=round(sum(float(s.get("max_notional") or 0.0) for s in slice_scope), 2),
            applied_cap=cap_value,
            cap_source=cap_source,
        )
        if chal_slice and chal_slice != recomputed_slice:
            raise ValueError("arm_rejected:authorized_slice_hash_changed")
        if str(challenge.get("source_readiness_packet_hash") or "") != source_readiness_packet_hash(
            readiness_packet
        ):
            raise ValueError("arm_rejected:source_readiness_packet_hash_changed")

    # Replay protection: challenge already used to create authority.
    existing = _read_json(paths.authority)
    if existing:
        if str(existing.get("confirmation_challenge_id") or "") == str(
            challenge.get("challenge_id") or ""
        ):
            write_state(paths, SellArmState.REJECTED, {"reason": "challenge_replay"})
            append_audit(paths, {"event": "arm_rejected", "reason": "challenge_replay"})
            raise ValueError("arm_rejected:challenge_replay")
        # Any existing unused authority for different challenge still blocks silent replace.
        if (
            str(existing.get("status") or "") == SellArmState.ARMED.value
            and str(existing.get("consumption_status") or "") == "UNUSED"
        ):
            exp = _parse_iso(existing.get("expires_at"))
            if exp and exp > now:
                raise ValueError("arm_rejected:existing_armed_authority_present")

    if newer_plan_exists:
        raise ValueError("arm_rejected:newer_plan_supersedes")
    if current_plan_hash and str(current_plan_hash) != str(readiness_packet.get("plan_hash") or ""):
        raise ValueError("arm_rejected:plan_hash_superseded")

    scope = list(challenge.get("authorized_scope") or [])
    if not scope:
        raise ValueError("arm_rejected:empty_scope")
    for item in scope:
        action = str(item.get("action") or "").upper()
        if action in BUY_ACTIONS:
            raise ValueError("arm_rejected:buy_action_in_sell_arm")
        if action not in SELL_ACTIONS:
            raise ValueError(f"arm_rejected:invalid_sell_action:{action}")

    cap_value, cap_source = applied_authority_cap()
    total_notional = round(sum(float(s.get("max_notional") or 0.0) for s in scope), 2)
    if total_notional > cap_value + 1e-9:
        raise ValueError("arm_rejected:total_notional_exceeds_approval_cli_cap")

    per_qty = {str(s["symbol"]).upper(): float(s["max_qty"]) for s in scope}
    per_notional = {
        str(s["symbol"]).upper(): float(s["max_notional"])
        for s in scope
        if s.get("max_notional") is not None
    }
    expires = now + timedelta(minutes=AUTHORITY_TTL_MINUTES)
    # Must not survive trading date (end of trading_date UTC day as conservative bound).
    td = str(challenge.get("trading_date") or now.date().isoformat())
    try:
        day = date.fromisoformat(td)
        day_end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
        if expires > day_end:
            expires = day_end
    except Exception as exc:
        raise ValueError("arm_rejected:trading_date_unparseable") from exc

    audit_id = f"sell-arm-audit-{uuid.uuid4().hex[:12]}"
    idem = (
        f"sell-arm|{challenge.get('plan_hash')}|{challenge.get('batch_id')}|"
        f"{challenge.get('challenge_id')}"
    )
    authority = {
        "schema_version": SCHEMA_VERSION,
        "authority_type": AUTHORITY_TYPE,
        "status": SellArmState.ARMED.value,
        "mode": "PAPER",
        "approval_mode": "PAPER_ONLY",
        "account_fingerprint": challenge.get("account_fingerprint"),
        "trading_date": td,
        "cycle_id": challenge.get("cycle_id"),
        "run_id": challenge.get("run_id") or readiness_packet.get("run_id"),
        "batch_id": challenge.get("batch_id"),
        "plan_id": challenge.get("plan_id") or readiness_packet.get("plan_id"),
        "plan_hash": challenge.get("plan_hash"),
        "readiness_packet_hash": challenge.get("readiness_packet_hash"),
        "confirmation_challenge_id": challenge.get("challenge_id"),
        "confirmation_challenge_hash": challenge.get("challenge_hash"),
        "issued_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "issued_by": issued_by,
        "operator_acknowledgement_version": OPERATOR_ACKNOWLEDGEMENT_VERSION,
        "operator_acknowledgement_text": OPERATOR_ACKNOWLEDGEMENT_TEXT,
        "authorized_actions": scope,
        "approved_symbols": sorted(per_qty.keys()),
        "per_symbol_max_quantity": per_qty,
        "per_symbol_max_notional": per_notional,
        "total_batch_max_notional": min(total_notional, cap_value),
        "max_notional": min(total_notional, cap_value),
        "applied_cap": cap_value,
        "applied_cap_source": cap_source,
        "runtime_manage_cap_observed": RUNTIME_MANAGE_CAP,
        "market_session_requirement": "MARKET_OPEN",
        "single_use": True,
        "consumption_status": "UNUSED",
        "live_execution_required_state": "BLOCKED",
        "live_authorized": False,
        "broker_execution_authorized": False,
        "source_provenance": {
            "readiness_schema": readiness_packet.get("schema_version"),
            "challenge_id": challenge.get("challenge_id"),
            "phase": "198N-E" if dual else "198H",
        },
        "audit_id": audit_id,
        "idempotency_key": idem,
        "no_symbol_substitution": True,
        "no_action_substitution": True,
        "no_quantity_expansion": True,
        "no_batch_expansion": True,
        "creates_dispatch_authority": False,
        "orders_submitted": False,
        # Legacy interlock compatibility fields (still PAPER_ONLY / blocked).
        "decision": "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
        "approval_id": audit_id,
        "approved_at": now.isoformat(),
        "approved_by": issued_by,
        "source_packet_id": f"readiness:{challenge.get('readiness_packet_hash')}",
        "source_preview_hash": str(challenge.get("readiness_packet_hash")),
        "acknowledgement_live_blocked": True,
        "acknowledgement_no_live_execution": True,
        "acknowledgement_paper_only": True,
        "invalidation_policy": "WHOLE_BATCH_DEFAULT",
        "force_exit_policy": "REQUIRES_SAME_REVIEWED_SELL_ARM_FAIL_CLOSED",
    }
    if dual:
        authority["identity_schema"] = SCHEMA_VERSION_DUAL
        authority["source_readiness_packet_hash"] = challenge.get("source_readiness_packet_hash")
        authority["authorized_slice_hash"] = challenge.get("authorized_slice_hash")
        authority["source_plan_sha256"] = challenge.get("source_plan_sha256")
        authority["selected_row_hash"] = challenge.get("selected_row_hash")
        authority["source_generated_at"] = challenge.get("generated_at")
        authority["market_open_invocation_id"] = challenge.get("market_open_invocation_id")
        authority["authorized_scope"] = list(scope)
    missing = REQUIRED_AUTHORITY_FIELDS - set(authority.keys())
    if missing:
        raise ValueError(f"arm_schema_incomplete:{sorted(missing)}")

    from services.paper_sell_arm_store import (
        PaperSellArmStoreError,
        batch_hash_from_scope,
        consume_challenge_and_create_authority,
        resolve_runtime_sqlite_path,
        sqlite_permit_grade_enabled,
    )

    if sqlite_permit_grade_enabled():
        try:
            stored = consume_challenge_and_create_authority(
                resolve_runtime_sqlite_path(db_path=db_path),
                challenge_id=str(challenge.get("challenge_id") or ""),
                readiness_plan_hash=str(readiness_packet.get("plan_hash") or ""),
                readiness_batch_id=str(readiness_packet.get("batch_id") or ""),
                readiness_batch_hash=batch_hash_from_scope(scope),
                authority=authority,
                now=now,
            )
            authority["authority_id"] = stored["authority_id"]
            authority["sqlite_state"] = "UNUSED"
            authority["json_role"] = "AUDIT_EXPORT_ONLY"
        except PaperSellArmStoreError as exc:
            raise ValueError(f"arm_rejected:{exc.reason}") from exc
    else:
        authority["json_role"] = "AUDIT_EXPORT_ONLY"
        authority["sqlite_state"] = "rollback_json_fail_closed"

    _atomic_write_json(paths.authority, authority)
    write_state(paths, SellArmState.ARMED, {"audit_id": audit_id, "orders_submitted": False})
    append_audit(
        paths,
        {
            "event": "authority_created",
            "audit_id": audit_id,
            "plan_hash": authority["plan_hash"],
            "batch_id": authority["batch_id"],
            "symbols": authority["approved_symbols"],
            "status": SellArmState.ARMED.value,
            "orders_submitted": False,
            "dispatch_performed": False,
        },
    )
    return authority


def inspect_paper_sell_arm(
    *,
    store_root: Path,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Read-only inspection suitable for future dashboard use."""
    paths = PaperSellArmPaths(root=store_root.resolve())
    now = now or _utc_now()
    challenge = _read_json(paths.challenge)
    authority = _read_json(paths.authority)
    consumption = _read_json(paths.consumption)
    state_doc = _read_json(paths.state)

    status = str(state_doc.get("state") or SellArmState.NO_SELLS_WAITING.value)
    detail: Dict[str, Any] = {"state_file": status}

    if authority:
        exp = _parse_iso(authority.get("expires_at"))
        if not exp:
            status = SellArmState.BLOCKED.value
            detail["reason"] = "authority_timestamp_unparseable"
        elif exp <= now:
            status = SellArmState.EXPIRED.value
        elif str(authority.get("consumption_status") or "") != "UNUSED":
            status = SellArmState.CONSUMED.value
        elif str(authority.get("status") or "") == SellArmState.ARMED.value:
            status = SellArmState.ARMED.value
    elif challenge:
        exp = _parse_iso(challenge.get("expires_at"))
        if not exp:
            status = SellArmState.BLOCKED.value
        elif exp <= now:
            status = SellArmState.EXPIRED.value
        else:
            status = SellArmState.CONFIRMATION_CHALLENGE_ISSUED.value

    return {
        "state": status,
        "challenge_present": bool(challenge),
        "authority_present": bool(authority),
        "consumption_present": bool(consumption),
        "challenge_creates_no_authority": challenge.get("creates_no_authority"),
        "authority_status": authority.get("status"),
        "authority_consumption_status": authority.get("consumption_status"),
        "orders_submitted": False,
        "read_only": True,
        "inspected_at": now.isoformat(),
        "detail": detail,
    }


def revoke_paper_sell_arm(
    *,
    store_root: Path,
    reason: str = "operator_revoke",
    explicit_runtime_write: bool = False,
) -> Dict[str, Any]:
    paths = _assert_writable_root(store_root, explicit_runtime_write=explicit_runtime_write)
    authority = _read_json(paths.authority)
    if not authority:
        return {"revoked": False, "reason": "no_authority"}
    authority["status"] = SellArmState.REJECTED.value
    authority["consumption_status"] = "REVOKED"
    authority["revoked_at"] = _utc_now().isoformat()
    authority["revoke_reason"] = reason
    _atomic_write_json(paths.authority, authority)
    write_state(paths, SellArmState.REJECTED, {"reason": reason})
    append_audit(paths, {"event": "authority_revoked", "reason": reason})
    return {"revoked": True, "reason": reason}


def mark_authority_dispatch_attempt(
    *,
    store_root: Path,
    authority: Dict[str, Any],
    result: str,
    explicit_runtime_write: bool = False,
    now: Optional[datetime] = None,
    require_full_durability: bool = False,
) -> Dict[str, Any]:
    """
    Consumption semantics (safe default for existing architecture):
    Consume on first dispatch attempt (including indeterminate/timeout/partial),
    so authority cannot be reused to duplicate sells.
    Does not submit orders.
    """
    paths = _assert_writable_root(store_root, explicit_runtime_write=explicit_runtime_write)
    now = now or _utc_now()
    if str(authority.get("idempotency_key") or "") == "":
        raise ValueError("consume_rejected:missing_idempotency_key")

    consumption = {
        "schema_version": SCHEMA_VERSION,
        "consumed_at": now.isoformat(),
        "audit_id": authority.get("audit_id"),
        "idempotency_key": authority.get("idempotency_key"),
        "plan_hash": authority.get("plan_hash"),
        "batch_id": authority.get("batch_id"),
        "result": result,
        "consumption_rule": "CONSUME_ON_FIRST_DISPATCH_ATTEMPT",
        "orders_submitted_unknown": result in {"INDETERMINATE", "TIMEOUT", "PARTIAL"},
    }
    authority = dict(authority)
    authority["status"] = (
        SellArmState.DISPATCH_IN_PROGRESS.value
        if result == "STARTED"
        else SellArmState.CONSUMED.value
    )
    authority["consumption_status"] = "CONSUMED"
    authority["consumed_at"] = now.isoformat()
    _atomic_write_json(
        paths.authority,
        authority,
        refuse_authority_downgrade=True,
        require_full_durability=require_full_durability,
    )
    _atomic_write_json(
        paths.consumption,
        consumption,
        require_full_durability=require_full_durability,
    )
    write_state(paths, SellArmState.CONSUMED, {"result": result})
    append_audit(
        paths,
        {
            "event": "authority_consumed",
            "result": result,
            "idempotency_key": authority.get("idempotency_key"),
            "plan_hash": authority.get("plan_hash"),
        },
    )
    return consumption


@dataclass
class BrokerSnapshotView:
    """Independent current-runtime evidence for pre-dispatch revalidation."""

    positions: Dict[str, Dict[str, float]]
    open_order_symbols: Set[str]
    mode: str = "paper"
    live_blocked: bool = True
    trading_date: Optional[str] = None
    account_fingerprint: Optional[str] = None
    current_plan_hash: Optional[str] = None
    newer_completed_plan: bool = False
    market_session: str = "MARKET_OPEN"
    paper_buy_arm_valid: bool = False
    # Phase 198I-FIX independent bindings (must not be copied from authority alone).
    cycle_id: Optional[str] = None
    run_id: Optional[str] = None
    batch_id: Optional[str] = None
    plan_id: Optional[str] = None
    readiness_packet_hash: Optional[str] = None
    confirmation_challenge_id: Optional[str] = None
    confirmation_challenge_hash: Optional[str] = None
    newer_batch_supersedes: bool = False
    lifecycle_actions: Optional[Dict[str, str]] = None
    newer_order_symbols: Optional[Set[str]] = None
    newer_fill_symbols: Optional[Set[str]] = None
    paper_execution_authorized: bool = False
    live_state_source: str = "unset"
    live_state_ambiguous: bool = False
    live_state_missing: bool = False
    evidence_independent: bool = False


def _nonempty_identity_values(*values: Any) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text:
            out.add(text)
    return out


def classify_consumption_record_for_authority(
    authority: Mapping[str, Any],
    consumption: Optional[Mapping[str, Any]],
) -> Tuple[str, str]:
    """Scope a consumption JSON record against one authority.

    Returns (classification, reason):
      - ("ignore", "") historical / other-authority evidence
      - ("block", "consumption_record_present") matching current authority
      - ("fail", explicit_reason) malformed or ambiguous current-authority record
    """
    if consumption is None:
        return "ignore", ""
    if not isinstance(consumption, Mapping) or not dict(consumption):
        return "fail", "consumption_record_malformed"

    auth_ids = _nonempty_identity_values(
        authority.get("authority_id"),
        authority.get("audit_id"),
        authority.get("approval_id"),
    )
    cons_ids = _nonempty_identity_values(
        consumption.get("authority_id"),
        consumption.get("audit_id"),
        consumption.get("approval_id"),
    )
    auth_key = str(authority.get("idempotency_key") or "").strip()
    cons_key = str(consumption.get("idempotency_key") or "").strip()
    auth_plan = str(authority.get("plan_hash") or "").strip()
    cons_plan = str(consumption.get("plan_hash") or "").strip()
    auth_batch = str(authority.get("batch_id") or "").strip()
    cons_batch = str(consumption.get("batch_id") or "").strip()
    auth_chal = str(
        authority.get("confirmation_challenge_id") or authority.get("challenge_id") or ""
    ).strip()

    if not cons_ids and not cons_key:
        return "fail", "consumption_identity_ambiguous"

    id_overlap = bool(cons_ids & auth_ids)
    id_foreign = bool(cons_ids and auth_ids and cons_ids.isdisjoint(auth_ids))
    key_match = bool(cons_key and auth_key and cons_key == auth_key)
    key_conflict = bool(cons_key and auth_key and cons_key != auth_key)
    mixed_ids = bool(cons_ids and auth_ids and cons_ids - auth_ids and cons_ids & auth_ids)

    appears_current = id_overlap or key_match
    if mixed_ids or (id_overlap and key_conflict) or (key_match and id_foreign):
        return "fail", "consumption_identity_ambiguous"
    if appears_current and cons_plan and auth_plan and cons_plan != auth_plan:
        return "fail", "consumption_identity_ambiguous"
    if appears_current and cons_batch and auth_batch and cons_batch != auth_batch:
        return "fail", "consumption_identity_ambiguous"
    if appears_current and auth_chal and cons_key and auth_chal not in cons_key and not key_match:
        if id_overlap:
            return "fail", "consumption_identity_ambiguous"
    if appears_current:
        return "block", "consumption_record_present"
    return "ignore", ""


def validate_paper_sell_arm_authority(
    authority: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    expected_challenge_id: Optional[str] = None,
    expected_plan_hash: Optional[str] = None,
    expected_batch_id: Optional[str] = None,
    expected_cycle_id: Optional[str] = None,
    expected_account_fingerprint: Optional[str] = None,
    expected_trading_date: Optional[str] = None,
    consumption: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate authoritative sell-arm document (no broker I/O)."""
    now = now or _utc_now()
    details: Dict[str, Any] = {"schema_version": authority.get("schema_version")}

    if not authority:
        return False, "authority_missing", details

    # Reject non-authority artifacts explicitly.
    if authority.get("artifact_type") == CHALLENGE_ARTIFACT_TYPE:
        return False, "challenge_is_not_authority", details
    if str(authority.get("status") or "") == READINESS_STATUS_OK:
        return False, "readiness_is_not_authority", details
    if authority.get("preview_status") in {
        "UNAPPROVED_PREVIEW_ONLY",
        "UNAPPROVED_APPROVAL_READINESS_ONLY",
    }:
        return False, "preview_or_readiness_is_not_authority", details
    if (
        authority.get("creates_no_authority") is True
        and authority.get("authority_type") != AUTHORITY_TYPE
    ):
        return False, "non_authoritative_artifact", details

    if str(authority.get("authority_type") or "") != AUTHORITY_TYPE:
        return False, "authority_type_mismatch", details
    if str(authority.get("schema_version") or "") != SCHEMA_VERSION:
        return False, "unsupported_or_historical_schema", details

    missing = REQUIRED_AUTHORITY_FIELDS - set(authority.keys())
    if missing:
        return False, f"missing_required_fields:{sorted(missing)}", details

    if str(authority.get("mode") or "").upper() != "PAPER":
        return False, "mode_not_paper", details
    if str(authority.get("live_execution_required_state") or "").upper() != "BLOCKED":
        return False, "live_execution_not_required_blocked", details
    if authority.get("live_authorized") is True:
        return False, "live_authorized_true", details
    if authority.get("single_use") is not True:
        return False, "not_single_use", details
    if str(authority.get("status") or "") != SellArmState.ARMED.value:
        return False, f"status_not_armed:{authority.get('status')}", details
    if str(authority.get("consumption_status") or "") != "UNUSED":
        return False, "authority_consumed_or_revoked", details
    classification, cons_reason = classify_consumption_record_for_authority(authority, consumption)
    if classification == "fail":
        return False, cons_reason, details
    if classification == "block":
        return False, cons_reason, details

    issued = _parse_iso(authority.get("issued_at"))
    expires = _parse_iso(authority.get("expires_at"))
    if not issued or not expires:
        return False, "timestamp_unparseable_fail_closed", details
    if expires <= now:
        return False, "authority_expired", details

    if expected_challenge_id and str(authority.get("confirmation_challenge_id")) != str(
        expected_challenge_id
    ):
        return False, "challenge_id_mismatch", details
    if expected_plan_hash and str(authority.get("plan_hash")) != str(expected_plan_hash):
        return False, "plan_hash_mismatch", details
    if expected_batch_id and str(authority.get("batch_id")) != str(expected_batch_id):
        return False, "batch_id_mismatch", details
    if expected_cycle_id and str(authority.get("cycle_id")) != str(expected_cycle_id):
        return False, "cycle_id_mismatch", details
    if expected_account_fingerprint and str(authority.get("account_fingerprint")) != str(
        expected_account_fingerprint
    ):
        return False, "account_fingerprint_mismatch", details
    if expected_trading_date and str(authority.get("trading_date")) != str(expected_trading_date):
        return False, "trading_date_mismatch", details

    # Sell-only scope.
    for item in authority.get("authorized_actions") or []:
        action = str(item.get("action") or "").upper()
        if action in BUY_ACTIONS:
            return False, "buy_or_add_in_sell_arm", details
        if action not in SELL_ACTIONS:
            return False, f"invalid_action:{action}", details

    try:
        applied = float(authority.get("applied_cap"))
    except Exception:
        return False, "applied_cap_unparseable", details
    if applied > APPROVAL_CLI_CAP + 1e-9:
        return False, "applied_cap_raises_cli_limit", details
    try:
        total = float(authority.get("total_batch_max_notional"))
    except Exception:
        return False, "total_notional_unparseable", details
    if total > applied + 1e-9:
        return False, "total_notional_exceeds_applied_cap", details

    return True, "ok", details


def recompute_allowed_runtime_qty(
    authorized_qty: Any, stored_max_notional: Any, price: Any
) -> Tuple[int, str]:
    """Independent Decimal downward qty used by clip-audit revalidation."""
    from decimal import Decimal, InvalidOperation, ROUND_FLOOR

    from services.paper_sell_arm_cap_slice import money_round

    try:
        auth = int(Decimal(str(authorized_qty)).to_integral_value(rounding=ROUND_FLOOR))
    except (InvalidOperation, ValueError, TypeError, ArithmeticError):
        return 0, "invalid_authorized_qty"
    if auth <= 0:
        return 0, "invalid_authorized_qty"
    if stored_max_notional is None:
        return auth, "unchanged_no_stored_max"
    try:
        max_n = Decimal(str(money_round(stored_max_notional)))
        px = Decimal(str(price))
    except (InvalidOperation, ValueError, TypeError, ArithmeticError):
        return 0, "invalid_price_or_max"
    if (not px.is_finite()) or px <= 0:
        return 0, "invalid_price"
    runtime = min(Decimal(auth), (max_n / px).to_integral_value(rounding=ROUND_FLOOR))
    if runtime < 0:
        runtime = Decimal("0")
    while runtime > 0 and Decimal(str(money_round(float(runtime * px)))) > max_n:
        runtime -= 1
    return int(runtime), "ok"


def _clip_price_unavailable(value: Any) -> bool:
    import math as _math
    from decimal import Decimal, InvalidOperation

    if value is None:
        return True
    try:
        if isinstance(value, float) and (_math.isnan(value) or _math.isinf(value)):
            return True
        price = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError, ArithmeticError):
        return True
    return (not price.is_finite()) or price <= 0


def _integral_qty(value: Any) -> Tuple[bool, int]:
    from decimal import Decimal, InvalidOperation

    try:
        qty = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError, ArithmeticError):
        return False, 0
    if not qty.is_finite() or qty != qty.to_integral_value():
        return False, 0
    return True, int(qty)


def validate_runtime_clip_audit(
    authority: Mapping[str, Any],
    clip_audit: Sequence[Mapping[str, Any]],
    requested_actions: Sequence[Mapping[str, Any]],
    runtime_ref_prices: Mapping[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Independently recompute a supplied downward-clipping audit. Fail closed."""
    from services.paper_sell_arm_cap_slice import money_round

    scope = list(authority.get("authorized_actions") or authority.get("authorized_scope") or [])
    details: Dict[str, Any] = {"clip_audit_validated": False}
    if not isinstance(clip_audit, (list, tuple)):
        return False, "clip_audit_malformed", details
    if len(clip_audit) != len(scope):
        return False, "clip_audit_row_count_mismatch", details
    auth_syms = [str(item.get("symbol") or "").upper() for item in scope]
    if len(set(auth_syms)) != len(auth_syms):
        return False, "authorized_duplicate_symbol", details
    seen_audit: Set[str] = set()
    audit_by_sym: Dict[str, Mapping[str, Any]] = {}
    for row in clip_audit:
        if not isinstance(row, Mapping):
            return False, "clip_audit_malformed_row", details
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            return False, "clip_audit_missing_symbol", details
        if sym in seen_audit:
            return False, f"clip_audit_duplicate_symbol:{sym}", details
        seen_audit.add(sym)
        audit_by_sym[sym] = row
    extra = sorted(seen_audit - set(auth_syms))
    if extra:
        return False, f"clip_audit_extra_symbol:{extra[0]}", details
    missing = [s for s in auth_syms if s not in seen_audit]
    if missing:
        return False, f"clip_audit_missing_row:{missing[0]}", details

    req_by_sym: Dict[str, Mapping[str, Any]] = {}
    for req in requested_actions:
        sym = str(req.get("symbol") or "").upper()
        if not sym:
            return False, "requested_missing_symbol", details
        if sym in req_by_sym:
            return False, f"duplicate_requested_symbol:{sym}", details
        qty_ok, req_qty = _integral_qty(req.get("qty"))
        if not qty_ok:
            return False, f"requested_qty_not_integer:{sym}", details
        if req_qty <= 0:
            return False, f"zero_quantity_broker_request:{sym}", details
        req_by_sym[sym] = req

    derived_total = 0.0
    for item in scope:
        sym = str(item.get("symbol") or "").upper()
        action = str(item.get("action") or "").upper()
        auth_qty = float(item.get("max_qty") or 0)
        max_n = item.get("max_notional")
        row = audit_by_sym[sym]
        if str(row.get("action") or "").upper() != action:
            return False, f"clip_audit_action_mismatch:{sym}", details
        try:
            row_auth_qty = float(row.get("authorized_quantity"))
        except (TypeError, ValueError):
            return False, f"clip_audit_authorized_quantity_mismatch:{sym}", details
        if abs(row_auth_qty - auth_qty) > 1e-9:
            return False, f"clip_audit_authorized_quantity_mismatch:{sym}", details
        try:
            row_max_n = (
                float(row.get("authorized_max_notional"))
                if row.get("authorized_max_notional") is not None
                else None
            )
        except (TypeError, ValueError):
            return False, f"clip_audit_max_notional_mismatch:{sym}", details
        if max_n is None or row_max_n is None or abs(row_max_n - float(max_n)) > 1e-9:
            return False, f"clip_audit_max_notional_mismatch:{sym}", details
        price_raw = row.get("current_reference_price")
        if _clip_price_unavailable(price_raw):
            return False, f"clip_audit_invalid_price:{sym}", details
        price = float(price_raw)
        snap = runtime_ref_prices.get(sym) if runtime_ref_prices is not None else None
        if snap is None or _clip_price_unavailable(snap):
            return False, f"clip_audit_missing_runtime_price:{sym}", details
        if abs(float(snap) - price) > 1e-9:
            return False, f"clip_audit_forged_current_reference_price:{sym}", details
        qty_ok, runtime_qty = _integral_qty(row.get("runtime_quantity"))
        if not qty_ok:
            return False, f"clip_audit_runtime_quantity_not_integer:{sym}", details
        if runtime_qty < 0 or runtime_qty - auth_qty > 1e-9:
            return False, f"clip_audit_runtime_quantity_out_of_range:{sym}", details
        if runtime_qty > int(auth_qty) + 1e-9:
            return False, f"quantity_expansion:{sym}", details
        allowed_qty, _why = recompute_allowed_runtime_qty(auth_qty, max_n, price)
        if runtime_qty != allowed_qty:
            return False, f"clip_audit_runtime_quantity_mismatch:{sym}", details
        try:
            deferred = float(row.get("deferred_quantity"))
        except (TypeError, ValueError):
            return False, f"clip_audit_deferred_mismatch:{sym}", details
        if abs(deferred - (auth_qty - float(runtime_qty))) > 1e-9:
            return False, f"clip_audit_deferred_mismatch:{sym}", details
        expected_notional = money_round(runtime_qty * price) if runtime_qty else 0.0
        try:
            got_notional = float(row.get("runtime_estimated_notional") or 0)
        except (TypeError, ValueError):
            return False, f"clip_audit_runtime_notional_mismatch:{sym}", details
        if abs(got_notional - expected_notional) > 1e-9:
            return False, f"clip_audit_runtime_notional_mismatch:{sym}", details
        if expected_notional > money_round(max_n) + 1e-9:
            return False, f"notional_expansion_vs_authorized_slice:{sym}", details
        if runtime_qty == 0:
            expected_outcome = "SKIPPED_ZERO"
        elif runtime_qty + 1e-9 < auth_qty:
            expected_outcome = "CLIPPED_DOWN"
        else:
            expected_outcome = "UNCHANGED"
        if str(row.get("outcome") or "") != expected_outcome:
            return False, f"clip_audit_outcome_mismatch:{sym}", details
        if runtime_qty > 0:
            req = req_by_sym.pop(sym, None)
            if req is None:
                return False, f"incomplete_authorized_qty:{sym}", details
            if str(req.get("action") or "").upper() != action:
                return False, f"action_substitution:{sym}", details
            req_ok, req_qty = _integral_qty(req.get("qty"))
            if not req_ok or req_qty != runtime_qty:
                return False, f"requested_qty_mismatch:{sym}", details
            req_px = float(req.get("price") or 0)
            req_n = (
                money_round(req_qty * req_px)
                if req_qty and req_px
                else float(req.get("notional") or 0)
            )
            if req_n > money_round(max_n) + 1e-9:
                return False, f"notional_expansion_vs_authorized_slice:{sym}", details
            derived_total = money_round(derived_total + req_n)
        else:
            if sym in req_by_sym:
                return False, f"zero_derived_symbol_requested:{sym}", details
    if req_by_sym:
        extra_req = sorted(req_by_sym)[0]
        return False, f"unauthorized_symbol_vs_authorized_slice:{extra_req}", details
    slice_total = money_round(sum(float(i.get("max_notional") or 0) for i in scope))
    stored_total = authority.get("total_batch_max_notional")
    cap_total = money_round(stored_total) if stored_total is not None else slice_total
    if derived_total > cap_total + 1e-9:
        return False, "notional_expansion_vs_authorized_slice_total", details
    if cap_total > APPROVAL_CLI_CAP + 1e-9 or slice_total > APPROVAL_CLI_CAP + 1e-9:
        return False, "batch_notional_exceeds_approval_cli_cap", details
    if derived_total > APPROVAL_CLI_CAP + 1e-9:
        return False, "batch_notional_exceeds_approval_cli_cap", details
    details.update(
        {
            "clip_audit_validated": True,
            "derived_requested_total": derived_total,
            "authorized_slice_total": slice_total,
        }
    )
    return True, "ok", details


def revalidate_before_dispatch(
    authority: Dict[str, Any],
    *,
    broker: BrokerSnapshotView,
    requested_actions: Sequence[Dict[str, Any]],
    now: Optional[datetime] = None,
    consumption: Optional[Dict[str, Any]] = None,
    readiness_packet_hash_current: Optional[str] = None,
    clip_audit: Optional[Sequence[Mapping[str, Any]]] = None,
    runtime_ref_prices: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Strict pre-dispatch revalidation. Fail closed on ambiguity.
    Default invalidation unit: WHOLE BATCH.
    """
    now = now or _utc_now()
    ok, reason, details = validate_paper_sell_arm_authority(
        authority,
        now=now,
        expected_account_fingerprint=broker.account_fingerprint,
        expected_trading_date=broker.trading_date,
        expected_plan_hash=broker.current_plan_hash,
        expected_cycle_id=broker.cycle_id,
        expected_batch_id=broker.batch_id,
        expected_challenge_id=broker.confirmation_challenge_id,
        consumption=consumption,
    )
    if not ok:
        return False, reason, details

    if str(broker.mode or "").lower() != "paper":
        return False, "mode_not_paper_runtime", details
    if broker.live_state_missing:
        return False, "live_execution_state_missing", details
    if broker.live_state_ambiguous:
        return False, "live_execution_state_ambiguous", details
    if not broker.live_blocked:
        return False, "live_execution_not_blocked", details
    if not broker.paper_execution_authorized and not broker.paper_buy_arm_valid:
        # Paper sells require a valid paper execution posture (buy-arm ASSISTED/allow
        # or explicit paper authorization evidence). Fail closed when absent.
        return False, "paper_execution_authorization_invalid", details
    if broker.newer_completed_plan:
        return False, "newer_management_plan_supersedes", details
    if broker.newer_batch_supersedes:
        return False, "newer_batch_supersedes", details
    if not broker.current_plan_hash:
        return False, "independent_plan_hash_missing", details
    if str(broker.current_plan_hash) != str(authority.get("plan_hash")):
        return False, "plan_hash_superseded", details
    if broker.run_id is not None and str(broker.run_id) != str(authority.get("run_id") or ""):
        return False, "run_id_mismatch", details
    if broker.plan_id is not None and str(broker.plan_id) != str(authority.get("plan_id") or ""):
        return False, "plan_id_mismatch", details
    readiness_current = readiness_packet_hash_current or broker.readiness_packet_hash
    if not readiness_current:
        return False, "readiness_packet_hash_missing_independent", details
    if str(readiness_current) != str(authority.get("readiness_packet_hash")):
        return False, "readiness_packet_hash_changed", details
    if broker.confirmation_challenge_hash and str(broker.confirmation_challenge_hash) != str(
        authority.get("confirmation_challenge_hash") or ""
    ):
        return False, "confirmation_challenge_hash_mismatch", details
    if not broker.confirmation_challenge_id:
        return False, "confirmation_challenge_binding_missing", details
    if str(authority.get("market_session_requirement") or "") == "MARKET_OPEN":
        if str(broker.market_session or "").upper() not in {"MARKET_OPEN", "OPEN"}:
            return False, "market_session_not_open", details

    auth_map = {
        str(a.get("symbol")).upper(): a for a in (authority.get("authorized_actions") or [])
    }
    req_syms = {str(r.get("symbol") or "").upper() for r in requested_actions}
    auth_syms = set(auth_map.keys())
    if clip_audit is None:
        # Exact batch match required unless an explicit clip audit is supplied.
        if req_syms != auth_syms:
            return False, "batch_symbol_set_mismatch_whole_batch_required", details
    else:
        if runtime_ref_prices is None:
            return False, "clip_audit_runtime_prices_missing", details
        ok_clip, clip_reason, clip_details = validate_runtime_clip_audit(
            authority,
            clip_audit,
            requested_actions,
            runtime_ref_prices,
        )
        details.update(clip_details)
        if not ok_clip:
            return False, clip_reason, details

    total_req_notional = 0.0
    newer_orders = broker.newer_order_symbols or set()
    newer_fills = broker.newer_fill_symbols or set()
    lifecycle = broker.lifecycle_actions or {}
    for req in requested_actions:
        sym = str(req.get("symbol") or "").upper()
        action = str(req.get("action") or "").upper()
        qty = float(req.get("qty") or 0)
        px = float(req.get("price") or 0)
        notional = qty * px if qty and px else float(req.get("notional") or 0)
        total_req_notional += notional

        allowed = auth_map.get(sym) or {}
        if action != str(allowed.get("action") or "").upper():
            return False, f"action_substitution:{sym}", details
        if sym in lifecycle and str(lifecycle[sym]).upper() not in {
            action,
            "EXIT",
            "TRIM",
            "ROTATE_EXIT",
            "FORCE_EXIT",
            "SELL",
        }:
            # Lifecycle no longer supports a sell for this symbol.
            if str(lifecycle[sym]).upper() in {"HOLD", "WAIT", "BUY", "ADD", "FLAT"}:
                return False, f"lifecycle_action_mismatch:{sym}", details
        max_qty = float(allowed.get("max_qty") or 0)
        if qty > max_qty + 1e-9:
            return False, f"quantity_expansion:{sym}", details
        if qty <= 0:
            return False, f"non_positive_qty:{sym}", details

        pos = broker.positions.get(sym) or {}
        if "qty" not in pos:
            return False, f"held_qty_evidence_missing:{sym}", details
        held = float(pos.get("qty") or 0)
        if pos.get("qty_available") is None:
            return False, f"available_qty_evidence_missing:{sym}", details
        available = float(pos.get("qty_available"))
        reserved = held - available
        if held + 1e-9 < qty:
            return False, f"held_qty_insufficient:{sym}", details
        if available + 1e-9 < qty:
            return False, f"available_qty_insufficient:{sym}", details
        if reserved > 1e-6:
            return False, f"shares_reserved:{sym}", details
        if sym in broker.open_order_symbols:
            return False, f"open_order_conflict:{sym}", details
        if sym in newer_orders:
            return False, f"newer_order_invalidates:{sym}", details
        if sym in newer_fills:
            return False, f"newer_fill_invalidates:{sym}", details

        max_n = allowed.get("max_notional")
        if max_n is not None and notional > float(max_n) + 1e-6:
            reason = (
                f"notional_expansion_vs_authorized_slice:{sym}"
                if is_dual_identity_record(authority)
                else f"symbol_notional_cap:{sym}"
            )
            return False, reason, details
        if notional > APPROVAL_CLI_CAP + 1e-6:
            return False, f"symbol_notional_cap:{sym}", details

    max_total = float(authority.get("total_batch_max_notional") or 0)
    if total_req_notional > max_total + 1e-6:
        reason = (
            "notional_expansion_vs_authorized_slice_total"
            if is_dual_identity_record(authority)
            else "batch_notional_exceeds_authority"
        )
        return False, reason, details
    if total_req_notional > float(authority.get("applied_cap") or 0) + 1e-6:
        return False, "batch_notional_exceeds_approval_cli_cap", details
    if total_req_notional > APPROVAL_CLI_CAP + 1e-6:
        return False, "batch_notional_exceeds_approval_cli_cap", details

    details.update(
        {
            "requested_symbols": sorted(req_syms),
            "total_req_notional": round(total_req_notional, 2),
            "invalidation_unit": "WHOLE_BATCH",
            "evidence_independent": bool(broker.evidence_independent),
            "live_state_source": broker.live_state_source,
        }
    )
    return True, "ok", details


def challenge_rejected_by_approval_validator(challenge: Dict[str, Any]) -> Tuple[bool, str]:
    """Challenges must never validate as approval authority."""
    ok, reason, _ = validate_paper_sell_arm_authority(challenge)
    return (not ok), reason


def paper_buy_arm_rejected_for_sells(buy_arm_doc: Dict[str, Any]) -> Tuple[bool, str]:
    """Normal daily paper-buy arm must never satisfy sell-arm validator."""
    ok, reason, _ = validate_paper_sell_arm_authority(buy_arm_doc)
    return (not ok), reason


def is_hardened_sell_arm_contract(doc: Dict[str, Any]) -> bool:
    return (
        str(doc.get("authority_type") or "") == AUTHORITY_TYPE
        and str(doc.get("schema_version") or "") == SCHEMA_VERSION
    )


def allowed_transition(src: SellArmState, dst: SellArmState) -> bool:
    allowed = {
        SellArmState.NO_SELLS_WAITING: {
            SellArmState.REVIEW_REQUIRED,
            SellArmState.READY_FOR_CONFIRMATION,
        },
        SellArmState.REVIEW_REQUIRED: {
            SellArmState.READY_FOR_CONFIRMATION,
            SellArmState.BLOCKED,
        },
        SellArmState.READY_FOR_CONFIRMATION: {
            SellArmState.CONFIRMATION_CHALLENGE_ISSUED,
            SellArmState.STALE,
            SellArmState.BLOCKED,
            SellArmState.EXPIRED,
        },
        SellArmState.CONFIRMATION_CHALLENGE_ISSUED: {
            SellArmState.ARMED,
            SellArmState.EXPIRED,
            SellArmState.STALE,
            SellArmState.BLOCKED,
            SellArmState.REJECTED,
        },
        SellArmState.ARMED: {
            SellArmState.DISPATCH_IN_PROGRESS,
            SellArmState.EXPIRED,
            SellArmState.STALE,
            SellArmState.BLOCKED,
            SellArmState.REJECTED,
            SellArmState.CONSUMED,
        },
        SellArmState.DISPATCH_IN_PROGRESS: {
            SellArmState.CONSUMED,
            SellArmState.BLOCKED,
        },
        SellArmState.STALE: {
            SellArmState.REVIEW_REQUIRED,
            SellArmState.READY_FOR_CONFIRMATION,
        },
        SellArmState.EXPIRED: {
            SellArmState.REVIEW_REQUIRED,
            SellArmState.READY_FOR_CONFIRMATION,
        },
        SellArmState.BLOCKED: {SellArmState.REVIEW_REQUIRED},
        SellArmState.REJECTED: {SellArmState.REVIEW_REQUIRED},
        SellArmState.CONSUMED: {SellArmState.REVIEW_REQUIRED},
    }
    # No direct jump from review/ready to ARMED.
    if (
        src in {SellArmState.REVIEW_REQUIRED, SellArmState.READY_FOR_CONFIRMATION}
        and dst == SellArmState.ARMED
    ):
        return False
    return dst in allowed.get(src, set())
