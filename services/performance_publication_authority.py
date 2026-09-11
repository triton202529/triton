"""In-process production publication authority for Phase 198T-BN2F2.

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
from typing import Any, Callable, Dict, Mapping, Optional

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


def inspect_git_identity(root: Path) -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    try:
        head = git("rev-parse", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        porcelain = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicationAuthorityError(
            "git_identity_unavailable", {"error": type(exc).__name__}
        ) from exc
    return {
        "head": head,
        "branch": branch,
        "index_empty": porcelain == "",
    }


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
    return {"head": head, "branch": REQUIRED_BRANCH, "index_empty": True}


def passing_session(
    kind: str = "regular_close", session_date: str = "2026-09-10"
) -> Dict[str, Any]:
    return {
        "session_state": "CLOSED",
        "reason": kind,
        "session_date": session_date,
        "calendar": "XNYS",
    }
