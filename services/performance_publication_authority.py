"""In-process production publication authority for Phase 198T-BN2F2/BN2H1.

Minted only after the collector's complete gate passes. Never written to disk.
This is an in-process capability that protects accidental or unauthorized
direct publisher calls inside this application process. It is not a
cross-process or cryptographic unforgeability guarantee.

Serialized, copied, reconstructed, dummy, and subclassed objects fail closed.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import socket
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PRODUCTION_ROOT = Path(r"C:\Users\akimw\triton").resolve()
EXPECTED_PRODUCTION_HEAD = "18c790469c2f6f46d549fd0c42eb5df33df47ab9"
BN2E_LEDGER_SHA256 = "42868be678c4ed0bb75a3b527fa412d7e7c96b2061eb69b51f410bf8e061ec95"
BN2E_CUTOVER_TIMESTAMP = "2026-09-10T23:22:50+00:00"
PRODUCTION_PORTS = (8501, 8502, 8765)
FRESHNESS_LOCK_RELATIVE = Path("data/live/data_freshness_refresh.lock")
DEFAULT_AUTHORITY_TTL_SECONDS = 120.0
REQUIRED_BRANCH = "main"
COMMAND_CUTOVER = "publish-cutover"
COMMAND_SESSION_CLOSE = "publish-session-close"
EMPTY_LEDGER_SHA256 = hashlib.sha256(b"").hexdigest()

_ISSUED: Dict[str, "PublicationAuthority"] = {}
_CONSUMED: set[str] = set()
_REGISTRY_LOCK = threading.Lock()


class PublicationAuthorityError(Exception):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = _public_details(details or {})
        super().__init__(reason)

    def __repr__(self) -> str:
        return f"PublicationAuthorityError({self.reason!r})"


def _public_details(details: Mapping[str, Any]) -> Dict[str, Any]:
    blocked = {"nonce", "token", "_nonce", "account_id", "account_number", "secret"}
    out: Dict[str, Any] = {}
    for key, value in details.items():
        if str(key).lower() in blocked:
            continue
        out[key] = value
    return out


class PublicationAuthority:
    """Process-local single-use capability. Direct construction is forged."""

    __slots__ = (
        "_nonce",
        "resolved_root",
        "events_path",
        "lock_path",
        "expected_git_head",
        "command",
        "publication_mode",
        "confirm_phrase",
        "payload_sha256",
        "existing_ledger_sha256",
        "existing_ledger_nbytes",
        "required_prefix",
        "issued_at",
        "expires_at",
        "account_mode",
        "live_execution_permitted",
        "git_identity",
        "runtime_state",
        "session",
        "evidence_hashes",
        "git_inspector",
        "runtime_inspector",
        "session_inspector",
        "evidence_inspector",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise PublicationAuthorityError("forged_publication_authority")

    def __repr__(self) -> str:
        return (
            "<PublicationAuthority "
            f"command={getattr(self, 'command', None)!r} "
            f"mode={getattr(self, 'publication_mode', None)!r}>"
        )

    def __copy__(self) -> "PublicationAuthority":
        raise PublicationAuthorityError("authority_not_serializable")

    def __deepcopy__(self, memo: Any) -> "PublicationAuthority":
        raise PublicationAuthorityError("authority_not_serializable")

    def __getstate__(self) -> Dict[str, Any]:
        raise PublicationAuthorityError("authority_not_serializable")

    def __setstate__(self, state: Any) -> None:
        raise PublicationAuthorityError("authority_not_serializable")

    def __reduce__(self) -> Any:
        raise PublicationAuthorityError("authority_not_serializable")

    def __reduce_ex__(self, protocol: int) -> Any:
        raise PublicationAuthorityError("authority_not_serializable")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_path(path: Path) -> Path:
    raw = Path(path)
    try:
        real = Path(os.path.realpath(raw))
    except OSError:
        real = raw
    try:
        resolved = real.resolve()
    except OSError:
        resolved = real
    return resolved


def same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(canonical_path(left))) == os.path.normcase(
        str(canonical_path(right))
    )


def _under(path: Path, root: Path) -> bool:
    text = os.path.normcase(str(canonical_path(path)))
    prefix = os.path.normcase(str(canonical_path(root))).rstrip("\\/") + os.sep
    return text.startswith(prefix) or same_path(path, root)


def is_temp_path(path: Path) -> bool:
    resolved = os.path.normcase(str(canonical_path(path)))
    roots = [
        os.environ.get("TEMP", ""),
        os.environ.get("TMP", ""),
        r"c:\users\akimw\appdata\local\temp",
        r"c:\users\akimw\appdata\local\pytest-of-akimw",
    ]
    return any(
        resolved.startswith(os.path.normcase(str(root)).rstrip("\\/") + os.sep)
        for root in roots
        if root
    )


def is_real_production_path(path: Path) -> bool:
    resolved = canonical_path(path)
    return _under(resolved, PRODUCTION_ROOT) and not is_temp_path(resolved)


def destination_requires_authority(
    events_path: Path,
    lock_path: Path,
    production_root: Optional[Path] = None,
) -> bool:
    if is_real_production_path(events_path) or is_real_production_path(lock_path):
        return True
    if production_root is None:
        return False
    root = canonical_path(production_root)
    return _under(events_path, root) or _under(lock_path, root)


def _stable_inspector(snapshot: Mapping[str, Any]) -> Callable[..., Mapping[str, Any]]:
    frozen = dict(snapshot)

    def _inner(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        return dict(frozen)

    return _inner


def mint_publication_authority(
    *,
    resolved_root: Path,
    expected_git_head: str,
    events_path: Path,
    lock_path: Path,
    command: str,
    publication_mode: str,
    confirm_phrase: str,
    payload_sha256: str,
    existing_ledger_sha256: str,
    existing_ledger_nbytes: int,
    required_prefix: bytes = b"",
    account_mode: str = "PAPER_ONLY",
    live_execution_permitted: bool = False,
    git_identity: Optional[Mapping[str, Any]] = None,
    runtime_state: Optional[Mapping[str, Any]] = None,
    session: Optional[Mapping[str, Any]] = None,
    evidence_hashes: Optional[Mapping[str, str]] = None,
    git_inspector: Optional[Callable[..., Mapping[str, Any]]] = None,
    runtime_inspector: Optional[Callable[..., Mapping[str, Any]]] = None,
    session_inspector: Optional[Callable[..., Mapping[str, Any]]] = None,
    evidence_inspector: Optional[Callable[..., Mapping[str, Any]]] = None,
    ttl_seconds: float = DEFAULT_AUTHORITY_TTL_SECONDS,
    now: Optional[datetime] = None,
) -> PublicationAuthority:
    if publication_mode not in {"create", "append"}:
        raise PublicationAuthorityError("publication_mode_invalid", {"mode": publication_mode})
    if command not in {COMMAND_CUTOVER, COMMAND_SESSION_CLOSE}:
        raise PublicationAuthorityError("publication_command_invalid", {"command": command})
    issued = now or datetime.now(timezone.utc)
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)
    identity = dict(git_identity or passing_git_identity(str(expected_git_head)))
    runtime = dict(runtime_state or passing_runtime_state())
    sess = dict(session or {})
    hashes = dict(evidence_hashes or {})
    auth = object.__new__(PublicationAuthority)
    auth._nonce = secrets.token_hex(32)
    auth.resolved_root = canonical_path(resolved_root)
    auth.events_path = canonical_path(events_path)
    auth.lock_path = canonical_path(lock_path)
    auth.expected_git_head = str(expected_git_head or "")
    auth.command = str(command)
    auth.publication_mode = str(publication_mode)
    auth.confirm_phrase = str(confirm_phrase)
    auth.payload_sha256 = str(payload_sha256)
    auth.existing_ledger_sha256 = str(existing_ledger_sha256)
    auth.existing_ledger_nbytes = int(existing_ledger_nbytes)
    auth.required_prefix = bytes(required_prefix)
    auth.issued_at = issued
    auth.expires_at = issued + timedelta(seconds=float(ttl_seconds))
    auth.account_mode = str(account_mode or "PAPER_ONLY")
    auth.live_execution_permitted = bool(live_execution_permitted)
    auth.git_identity = identity
    auth.runtime_state = runtime
    auth.session = sess
    auth.evidence_hashes = hashes
    auth.git_inspector = git_inspector or _stable_inspector(identity)
    auth.runtime_inspector = runtime_inspector or _stable_inspector(runtime)
    auth.session_inspector = session_inspector or _stable_inspector(sess)
    auth.evidence_inspector = evidence_inspector or _stable_inspector(hashes)
    with _REGISTRY_LOCK:
        _ISSUED[auth._nonce] = auth
    return auth


def _clock(now: Optional[datetime]) -> datetime:
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    return clock


def _reconfirm_mapping(
    bound: Mapping[str, Any],
    inspector: Callable[..., Mapping[str, Any]],
    reason: str,
    root: Path,
) -> None:
    current = dict(inspector(root))
    if dict(bound) != current:
        raise PublicationAuthorityError(reason)


def preflight_publication_authority(
    authority: Any,
    *,
    events_path: Path,
    lock_path: Path,
    production_root: Path,
    expected_git_head: str,
    publication_mode: str,
    command: str,
    confirm_phrase: str,
    now: Optional[datetime] = None,
) -> PublicationAuthority:
    if authority is None:
        raise PublicationAuthorityError("missing_publication_authority")
    if type(authority) is not PublicationAuthority:
        raise PublicationAuthorityError("forged_publication_authority")
    nonce = getattr(authority, "_nonce", None)
    with _REGISTRY_LOCK:
        if not nonce or nonce not in _ISSUED or _ISSUED[nonce] is not authority:
            if nonce in _CONSUMED:
                raise PublicationAuthorityError("reused_publication_authority")
            raise PublicationAuthorityError("forged_publication_authority")
        if _clock(now) >= authority.expires_at:
            raise PublicationAuthorityError("expired_publication_authority")
    root = canonical_path(production_root)
    dest = canonical_path(events_path)
    lock = canonical_path(lock_path)
    if not same_path(authority.resolved_root, root):
        raise PublicationAuthorityError("publication_authority_wrong_root")
    if not same_path(authority.events_path, dest) or not _under(dest, root):
        raise PublicationAuthorityError("publication_authority_wrong_ledger")
    if not same_path(authority.lock_path, lock) or not _under(lock, root):
        raise PublicationAuthorityError("publication_authority_wrong_lock")
    if str(authority.expected_git_head) != str(expected_git_head or ""):
        raise PublicationAuthorityError("publication_authority_wrong_head")
    if str(authority.publication_mode) != str(publication_mode):
        raise PublicationAuthorityError("publication_authority_wrong_mode")
    if str(authority.command) != str(command):
        raise PublicationAuthorityError("publication_authority_wrong_command")
    if str(authority.confirm_phrase) != str(confirm_phrase):
        raise PublicationAuthorityError("publication_authority_wrong_phrase")
    return authority


def require_publication_authority(
    authority: Any,
    *,
    events_path: Path,
    lock_path: Path,
    production_root: Path,
    expected_git_head: str,
    publication_mode: str,
    command: str,
    confirm_phrase: str,
    planned_payload: bytes,
    prior: bytes,
    now: Optional[datetime] = None,
) -> PublicationAuthority:
    if authority is None:
        raise PublicationAuthorityError("missing_publication_authority")
    if type(authority) is not PublicationAuthority:
        raise PublicationAuthorityError("forged_publication_authority")
    nonce = getattr(authority, "_nonce", None)
    with _REGISTRY_LOCK:
        if not nonce or nonce not in _ISSUED or _ISSUED[nonce] is not authority:
            if nonce in _CONSUMED:
                raise PublicationAuthorityError("reused_publication_authority")
            raise PublicationAuthorityError("forged_publication_authority")
        clock = _clock(now)
        if clock >= authority.expires_at:
            raise PublicationAuthorityError("expired_publication_authority")
        root = canonical_path(production_root)
        dest = canonical_path(events_path)
        lock = canonical_path(lock_path)
        if not same_path(authority.resolved_root, root):
            raise PublicationAuthorityError("publication_authority_wrong_root")
        if not same_path(authority.events_path, dest) or not _under(dest, root):
            raise PublicationAuthorityError("publication_authority_wrong_ledger")
        if not same_path(authority.lock_path, lock) or not _under(lock, root):
            raise PublicationAuthorityError("publication_authority_wrong_lock")
        if str(authority.expected_git_head) != str(expected_git_head or ""):
            raise PublicationAuthorityError("publication_authority_wrong_head")
        if str(authority.publication_mode) != str(publication_mode):
            raise PublicationAuthorityError("publication_authority_wrong_mode")
        if str(authority.command) != str(command):
            raise PublicationAuthorityError("publication_authority_wrong_command")
        if str(authority.confirm_phrase) != str(confirm_phrase):
            raise PublicationAuthorityError("publication_authority_wrong_phrase")
        if sha256_bytes(planned_payload) != authority.payload_sha256:
            raise PublicationAuthorityError("publication_authority_wrong_payload")
        if sha256_bytes(prior) != authority.existing_ledger_sha256:
            raise PublicationAuthorityError("publication_authority_wrong_existing_hash")
        if len(prior) != int(authority.existing_ledger_nbytes):
            raise PublicationAuthorityError("publication_authority_wrong_existing_length")
        if publication_mode == "append" and prior != authority.required_prefix:
            raise PublicationAuthorityError("publication_authority_wrong_prefix")
        if publication_mode == "create" and authority.required_prefix not in (b"", prior):
            raise PublicationAuthorityError("publication_authority_wrong_prefix")
        if authority.account_mode != "PAPER_ONLY" or authority.live_execution_permitted:
            raise PublicationAuthorityError("account_mode_changed")
        _reconfirm_mapping(
            authority.git_identity, authority.git_inspector, "git_identity_changed", root
        )
        fresh_git = dict(authority.git_inspector(root))
        if str(fresh_git.get("head") or "") != authority.expected_git_head:
            raise PublicationAuthorityError("git_head_changed")
        if str(fresh_git.get("branch") or "") != REQUIRED_BRANCH:
            raise PublicationAuthorityError("git_branch_changed")
        if fresh_git.get("index_empty") is not True:
            raise PublicationAuthorityError("git_index_changed")
        if "git_identity_valid" in fresh_git and fresh_git.get("git_identity_valid") is not True:
            raise PublicationAuthorityError("git_identity_changed")
        _reconfirm_mapping(
            authority.runtime_state,
            authority.runtime_inspector,
            "runtime_state_changed",
            root,
        )
        _reconfirm_mapping(
            authority.session, authority.session_inspector, "session_state_changed", root
        )
        _reconfirm_mapping(
            authority.evidence_hashes,
            authority.evidence_inspector,
            "evidence_changed",
            root,
        )
        fresh_runtime = dict(authority.runtime_inspector(root))
        if not fresh_runtime.get("runtime_stopped"):
            raise PublicationAuthorityError("runtime_state_changed")
        if not fresh_runtime.get("watchdog_stopped"):
            raise PublicationAuthorityError("runtime_state_changed")
        if not fresh_runtime.get("streamlit_stopped"):
            raise PublicationAuthorityError("runtime_state_changed")
        if not fresh_runtime.get("ports_closed"):
            raise PublicationAuthorityError("runtime_state_changed")
        if not fresh_runtime.get("freshness_writer_idle"):
            raise PublicationAuthorityError("runtime_state_changed")
        if fresh_runtime.get("freshness_lock_present"):
            raise PublicationAuthorityError("runtime_state_changed")
        fresh_session = dict(authority.session_inspector(root))
        if str(fresh_session.get("session_state") or "") != "CLOSED":
            raise PublicationAuthorityError("session_state_changed")
        if str(fresh_session.get("calendar") or "").upper() != "XNYS":
            raise PublicationAuthorityError("session_state_changed")
        _ISSUED.pop(nonce, None)
        _CONSUMED.add(nonce)
    return authority


CLASS_KNOWN_MUTABLE_OPERATIONAL_DATA = "KNOWN_MUTABLE_OPERATIONAL_DATA"
CLASS_KNOWN_ACCOUNTING_LEDGER_OR_LOCK = "KNOWN_ACCOUNTING_LEDGER_OR_LOCK"
CLASS_KNOWN_PAPER_ORDER_LEDGER_OR_LOCK = "KNOWN_PAPER_ORDER_LEDGER_OR_LOCK"
CLASS_KNOWN_TEMP_EVIDENCE = "KNOWN_TEMP_EVIDENCE"
CLASS_TRACKED_CODE_OR_CONFIG_CHANGE = "TRACKED_CODE_OR_CONFIG_CHANGE"
CLASS_UNTRACKED_CODE_OR_CONFIG = "UNTRACKED_CODE_OR_CONFIG"
CLASS_UNTRACKED_TEST = "UNTRACKED_TEST"
CLASS_UNEXPLAINED = "UNEXPLAINED"
CLASS_OTHER_NONEXECUTABLE_ARTIFACT = "OTHER_NONEXECUTABLE_ARTIFACT"

REASON_INDEX_NOT_EMPTY = "git_index_not_empty"
REASON_TRACKED_CODE_DIRTY = "git_tracked_code_dirty"
REASON_UNTRACKED_CODE = "git_untracked_code_present"
REASON_UNTRACKED_TEST = "git_untracked_test_present"
REASON_STATUS_CONFLICT = "git_status_conflict"
REASON_STATUS_UNPARSEABLE = "git_status_unparseable"
REASON_ARTIFACT_UNEXPLAINED = "git_worktree_artifact_unexplained"

_PROTECTED_CODE_PREFIXES = (
    "services/",
    "scripts/",
    "dashboard/",
    "pages/",
    "plugins/",
    "hooks/",
)
_PROTECTED_ROOT_FILES = frozenset(
    {
        "view_results.py",
        "conftest.py",
        "pytest.ini",
        ".pre-commit-config.yaml",
        ".gitattributes",
        "pyproject.toml",
        "setup.cfg",
    }
)
_PUBLICATION_CODE_RELATIVE = (
    "services/performance_publication_authority.py",
    "services/performance_evidence_collector.py",
    "services/performance_accounting.py",
)
_PUBLICATION_BASENAMES = frozenset(Path(p).name for p in _PUBLICATION_CODE_RELATIVE)
_ACCOUNTING_RELATIVE = frozenset(
    {
        "data/results/performance_accounting_events.jsonl",
        "data/results/performance_accounting_events.lock",
    }
)
_PAPER_LEDGER_RELATIVE = frozenset(
    {
        "data/results/paper_order_events.jsonl",
        "data/results/paper_order_events.lock",
    }
)
_EXPLICIT_OPERATIONAL_FILES = frozenset(
    {
        "data/governance/execution_gate_authority_matrix.json",
        "data/governance/execution_gate_decisions.json",
        "data/governance/execution_gate_evidence.json",
        "data/governance/symbol_hydration_quarantine.json",
        "data/governance/paper_sell_arm_authority.json",
        "data/governance/independent_dispatch_bindings.json",
        "data/governance/paper_sell_arm_audit.jsonl",
        "data/governance/paper_sell_arm_consumption.json",
        "data/governance/paper_sell_arm_dispatch.lock",
        "data/governance/paper_sell_arm_dispatch_intent.json",
        "data/governance/paper_sell_arm_dispatch_permit.json",
        "data/governance/paper_sell_arm_state.json",
        "data/governance/paper_sell_confirmation_challenge.json",
        "data/results/exit_logic_refinement_validation.json",
        "data/results/execution_readiness.json",
        "data/results/defensive_exit_approval_readiness.json",
        "data/results/positions_snapshot.csv",
        "data/results/open_orders_snapshot.csv",
        "data/results/open_orders_snapshot.json",
        "data/results/broker_snapshot_paper.json",
        "data/results/portfolio_history.csv",
        "data/results/guard_snapshot.json",
        "data/raw/SPY.csv",
        "data/raw/QQQ.csv",
        "data/live/paper_arm_confirm.json",
        "data/live/live_armed.json",
        "data/live/paper_sell_arm.sqlite3",
        "data/live/paper_sell_arm.sqlite3-wal",
        "data/live/paper_sell_arm.sqlite3-shm",
        "data/live/operator_restart_action.guard.json",
        "data/live/operator_restart_idempotency.json",
        "data/live/operator_restart_preflight_tokens.json",
        "data/live/operator_restart_transaction.json",
        "data/live/operator_shutdown_ack_continuous.json",
        "data/live/operator_shutdown_ack_watchdog.json",
        "data/live/operator_shutdown_request.json",
        "data/live/operator_stop_idempotency.json",
        "data/live/operator_stop_preflight_tokens.json",
        "data/live/paper_entry_dispatch_contract.json",
    }
)
_CODE_SHADOW_PREFIXES = (
    "services/",
    "scripts/",
    "dashboard/",
    "pages/",
    "tests/",
)
_EXECUTABLE_SUFFIXES = frozenset(
    {".py", ".pyw", ".pyi", ".ps1", ".psm1", ".psd1", ".cmd", ".bat", ".exe", ".pyd", ".dll"}
)
_TEMP_EVIDENCE_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".txt", ".md", ".log", ".lock", ""})
_CONFLICT_INDEX_CODES = frozenset("AUDTCRU")
_UNMERGED = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _posix_rel(path: str) -> str:
    return str(path or "").replace("\\", "/").strip()


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        import ctypes

        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False
        return bool(attrs & _FILE_ATTRIBUTE_REPARSE_POINT)
    except (AttributeError, OSError):
        return False


def _path_escapes_root(root: Path, rel: str) -> bool:
    norm = _posix_rel(rel)
    if not norm or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        return True
    parts = Path(norm).parts
    if any(part in {"..", "."} and part == ".." for part in parts) or ".." in parts:
        return True
    current = Path(root)
    for part in parts:
        current = current / part
        if _is_reparse_point(current):
            try:
                resolved = Path(os.path.realpath(current))
            except OSError:
                return True
            root_real = Path(os.path.realpath(root))
            try:
                resolved.relative_to(root_real)
            except ValueError:
                return True
    try:
        real = Path(os.path.realpath(Path(root) / norm))
        Path(os.path.realpath(root))
        real.relative_to(Path(os.path.realpath(root)))
    except (OSError, ValueError):
        return True
    return False


def _under_prefix(rel: str, prefixes: Sequence[str]) -> bool:
    text = _posix_rel(rel)
    return any(text == prefix.rstrip("/") or text.startswith(prefix) for prefix in prefixes)


def classify_publication_worktree_path(
    rel: str,
    *,
    tracked: bool,
    x_status: str,
    y_status: str,
) -> str:
    norm = _posix_rel(rel).rstrip("/")
    suffix = Path(norm).suffix.lower()
    name = Path(norm).name
    if norm in _ACCOUNTING_RELATIVE:
        return CLASS_KNOWN_ACCOUNTING_LEDGER_OR_LOCK
    if norm in _PAPER_LEDGER_RELATIVE:
        return CLASS_KNOWN_PAPER_ORDER_LEDGER_OR_LOCK
    if not tracked and _under_prefix(norm, ("tests/",)) and suffix == ".py":
        return CLASS_UNTRACKED_TEST
    if tracked and (_under_prefix(norm, _PROTECTED_CODE_PREFIXES) or name in _PROTECTED_ROOT_FILES):
        return CLASS_TRACKED_CODE_OR_CONFIG_CHANGE
    if not tracked and (
        _under_prefix(norm, _PROTECTED_CODE_PREFIXES)
        or _under_prefix(norm, (".githooks/", ".github/", "hooks/"))
        or (norm in _PROTECTED_ROOT_FILES)
        or (suffix in _EXECUTABLE_SUFFIXES and not _under_prefix(norm, ("tmp/", "data/")))
    ):
        return CLASS_UNTRACKED_CODE_OR_CONFIG
    if suffix in _EXECUTABLE_SUFFIXES and _under_prefix(norm, ("data/", "tmp/")):
        return CLASS_UNTRACKED_CODE_OR_CONFIG
    if name in _PUBLICATION_BASENAMES:
        return (
            CLASS_UNTRACKED_CODE_OR_CONFIG if not tracked else CLASS_TRACKED_CODE_OR_CONFIG_CHANGE
        )
    if norm in _EXPLICIT_OPERATIONAL_FILES:
        return CLASS_KNOWN_MUTABLE_OPERATIONAL_DATA
    if _under_prefix(norm, ("tmp/",)):
        nested = norm[4:] if norm.startswith("tmp/") else norm
        if any(
            nested == prefix.rstrip("/") or nested.startswith(prefix)
            for prefix in _CODE_SHADOW_PREFIXES
        ):
            return CLASS_UNTRACKED_CODE_OR_CONFIG
        if Path(nested).name in _PUBLICATION_BASENAMES:
            return CLASS_UNTRACKED_CODE_OR_CONFIG
        if suffix in _EXECUTABLE_SUFFIXES:
            return CLASS_UNTRACKED_CODE_OR_CONFIG
        if suffix in _TEMP_EVIDENCE_SUFFIXES:
            return CLASS_KNOWN_TEMP_EVIDENCE
        return CLASS_UNEXPLAINED
    if suffix in {".zip", ".diff", ".md", ".txt"} or _under_prefix(norm, ("docs/",)):
        return CLASS_OTHER_NONEXECUTABLE_ARTIFACT
    return CLASS_UNEXPLAINED


def parse_porcelain_line(line: str) -> Optional[Dict[str, str]]:
    raw = line.rstrip("\n")
    if not raw:
        return None
    if len(raw) < 4 or raw[2] != " ":
        return {"error": REASON_STATUS_UNPARSEABLE, "line": raw}
    x_status, y_status = raw[0], raw[1]
    remainder = raw[3:]
    if " -> " in remainder:
        path = remainder.split(" -> ", 1)[1]
        original = remainder.split(" -> ", 1)[0]
    else:
        path = remainder
        original = ""
    return {
        "x": x_status,
        "y": y_status,
        "path": _posix_rel(path),
        "original": _posix_rel(original),
        "xy": x_status + y_status,
    }


def _conflict_or_structural(entry: Mapping[str, str]) -> Optional[str]:
    xy = str(entry.get("xy") or "")
    x_status = str(entry.get("x") or "")
    y_status = str(entry.get("y") or "")
    if xy in _UNMERGED or x_status == "U" or y_status == "U":
        return REASON_STATUS_CONFLICT
    if x_status in {"R", "C", "T", "S"} or y_status in {"R", "C", "T", "S"}:
        return REASON_STATUS_CONFLICT
    if x_status == "D" or y_status == "D":
        return REASON_STATUS_CONFLICT
    return None


def evaluate_git_publication_identity(
    *,
    root: Path,
    head: str,
    branch: str,
    index_empty: bool,
    porcelain_lines: Sequence[str],
    staged_paths: Sequence[str],
    verify_filesystem: bool = True,
) -> Dict[str, Any]:
    tolerated: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    failure_reason: Optional[str] = None
    tracked_safe = True
    untracked_safe = True
    publication_matches = True
    parsed: List[Dict[str, str]] = []

    if any(str(path) for path in staged_paths):
        index_empty = False
        failure_reason = REASON_INDEX_NOT_EMPTY

    for line in porcelain_lines:
        entry = parse_porcelain_line(str(line))
        if entry is None:
            continue
        if entry.get("error"):
            failure_reason = REASON_STATUS_UNPARSEABLE
            rejected.append(
                {"path": str(line), "classification": CLASS_UNEXPLAINED, "reason": failure_reason}
            )
            tracked_safe = False
            untracked_safe = False
            continue
        parsed.append(entry)

    for entry in parsed:
        rel = entry["path"]
        tracked = entry["xy"] != "??"
        structural = _conflict_or_structural(entry)
        classification = classify_publication_worktree_path(
            rel, tracked=tracked, x_status=entry["x"], y_status=entry["y"]
        )
        row = {
            "path": rel,
            "classification": classification,
            "x": entry["x"],
            "y": entry["y"],
        }
        if verify_filesystem and _path_escapes_root(Path(root), rel):
            untracked_safe = False
            tracked_safe = False
            rejected.append(
                {**row, "reason": REASON_ARTIFACT_UNEXPLAINED, "detail": "path_escape_or_reparse"}
            )
            if failure_reason is None:
                failure_reason = REASON_ARTIFACT_UNEXPLAINED
            continue
        if structural:
            tracked_safe = False
            rejected.append({**row, "reason": structural})
            if failure_reason is None:
                failure_reason = structural
            continue
        if rel in _PUBLICATION_CODE_RELATIVE:
            publication_matches = False
        if classification in {
            CLASS_KNOWN_MUTABLE_OPERATIONAL_DATA,
            CLASS_KNOWN_ACCOUNTING_LEDGER_OR_LOCK,
            CLASS_KNOWN_PAPER_ORDER_LEDGER_OR_LOCK,
            CLASS_KNOWN_TEMP_EVIDENCE,
        }:
            tolerated.append(row)
            continue
        if classification == CLASS_TRACKED_CODE_OR_CONFIG_CHANGE:
            tracked_safe = False
            rejected.append({**row, "reason": REASON_TRACKED_CODE_DIRTY})
            if failure_reason in {None, REASON_INDEX_NOT_EMPTY}:
                if failure_reason != REASON_INDEX_NOT_EMPTY:
                    failure_reason = REASON_TRACKED_CODE_DIRTY
            if failure_reason is None:
                failure_reason = REASON_TRACKED_CODE_DIRTY
            continue
        if classification == CLASS_UNTRACKED_CODE_OR_CONFIG:
            untracked_safe = False
            rejected.append({**row, "reason": REASON_UNTRACKED_CODE})
            if failure_reason is None:
                failure_reason = REASON_UNTRACKED_CODE
            continue
        if classification == CLASS_UNTRACKED_TEST:
            untracked_safe = False
            rejected.append({**row, "reason": REASON_UNTRACKED_TEST})
            if failure_reason is None:
                failure_reason = REASON_UNTRACKED_TEST
            continue
        untracked_safe = False
        rejected.append({**row, "reason": REASON_ARTIFACT_UNEXPLAINED})
        if failure_reason is None:
            failure_reason = REASON_ARTIFACT_UNEXPLAINED

    if not index_empty:
        failure_reason = REASON_INDEX_NOT_EMPTY
        tracked_safe = tracked_safe and True

    valid = (
        index_empty
        and tracked_safe
        and untracked_safe
        and publication_matches
        and failure_reason is None
    )
    return {
        "head": head,
        "branch": branch,
        "index_empty": bool(index_empty),
        "tracked_worktree_safe": bool(tracked_safe),
        "untracked_worktree_safe": bool(untracked_safe),
        "publication_code_matches_head": bool(publication_matches),
        "git_identity_valid": bool(valid),
        "failure_reason": failure_reason,
        "tolerated_paths": tolerated,
        "rejected_paths": rejected,
    }


def inspect_git_identity(root: Path) -> Dict[str, Any]:
    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    try:
        head_proc = run_git("rev-parse", "HEAD")
        branch_proc = run_git("rev-parse", "--abbrev-ref", "HEAD")
        porcelain_proc = run_git("status", "--porcelain")
        cached_quiet = run_git("diff", "--cached", "--quiet", "--exit-code")
        cached_names = run_git("diff", "--cached", "--name-only")
    except OSError as exc:
        raise PublicationAuthorityError(
            "git_identity_unavailable", {"error": type(exc).__name__}
        ) from exc
    if head_proc.returncode != 0 or branch_proc.returncode != 0 or porcelain_proc.returncode != 0:
        raise PublicationAuthorityError("git_identity_unavailable", {"error": "git_status_failed"})
    if cached_quiet.returncode not in {0, 1}:
        raise PublicationAuthorityError(
            "git_identity_unavailable", {"error": "git_diff_cached_failed"}
        )
    staged = [line for line in cached_names.stdout.splitlines() if line.strip()]
    porcelain_lines = porcelain_proc.stdout.splitlines()
    return evaluate_git_publication_identity(
        root=Path(root),
        head=head_proc.stdout.strip(),
        branch=branch_proc.stdout.strip(),
        index_empty=cached_quiet.returncode == 0 and not staged,
        porcelain_lines=porcelain_lines,
        staged_paths=staged,
        verify_filesystem=True,
    )


def _command_blob() -> str:
    try:
        return subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }",
            ],
            text=True,
            timeout=20,
            stderr=subprocess.DEVNULL,
        ).lower()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _port_in_use(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError:
        return True
    finally:
        sock.close()


def inspect_runtime_state(root: Path) -> Dict[str, Any]:
    blob = _command_blob()
    inspect_failed = blob == ""
    lock = canonical_path(Path(root) / FRESHNESS_LOCK_RELATIVE)
    runtime_markers = (
        "run_continuous_loop.ps1",
        "start_triton_runtime",
        "run_triton_runtime",
    )
    return {
        "runtime_stopped": inspect_failed is False
        and not any(marker in blob for marker in runtime_markers),
        "watchdog_stopped": inspect_failed is False and "run_watchdog_loop.ps1" not in blob,
        "streamlit_stopped": inspect_failed is False and "streamlit" not in blob,
        "ports_closed": inspect_failed is False
        and not any(_port_in_use(port) for port in PRODUCTION_PORTS),
        "freshness_writer_idle": inspect_failed is False
        and "run_data_freshness_refresh.ps1" not in blob,
        "freshness_lock_present": lock.is_file(),
        "inspect_failed": inspect_failed,
    }


def passing_runtime_state() -> Dict[str, Any]:
    return {
        "runtime_stopped": True,
        "watchdog_stopped": True,
        "streamlit_stopped": True,
        "ports_closed": True,
        "freshness_writer_idle": True,
        "freshness_lock_present": False,
        "inspect_failed": False,
    }


def passing_git_identity(head: str = EXPECTED_PRODUCTION_HEAD) -> Dict[str, Any]:
    return {
        "head": head,
        "branch": REQUIRED_BRANCH,
        "index_empty": True,
        "tracked_worktree_safe": True,
        "untracked_worktree_safe": True,
        "publication_code_matches_head": True,
        "git_identity_valid": True,
        "failure_reason": None,
        "tolerated_paths": [],
        "rejected_paths": [],
    }


def passing_session(
    kind: str = "regular_close", session_date: str = "2026-09-10"
) -> Dict[str, Any]:
    return {
        "session_state": "CLOSED",
        "reason": kind,
        "session_date": session_date,
        "calendar": "XNYS",
    }
