"""Transactional SQLite store for paper SELL-arm challenges, authorities, and dispatches.

Permit-grade source of truth on Windows (and when selected). JSON files are
AUDIT_EXPORT_ONLY and cannot grant a broker-usable permit.

Contract:
- PRAGMA journal_mode=WAL (must report wal)
- PRAGMA synchronous=FULL (must report 2)
- PRAGMA foreign_keys=ON
- PRAGMA busy_timeout=5000
- BEGIN IMMEDIATE for state changes
- Never import legacy JSON as broker-usable authority
- SQLite durability remains subject to SQLite's documented OS/storage assumptions
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_NAME = "paper_sell_arm.sqlite3"
DEFAULT_RUNTIME_DB = ROOT / "data" / "live" / DEFAULT_DB_NAME
SCHEMA_USER_VERSION = 2
STORE_SCHEMA_VERSION = "198I-YR-SQLITE-2"
LEGACY_SCHEMA_USER_VERSION = 1

ROLLBACK_ENV = "TRITON_PAPER_SELL_ARM_STORE"
ROLLBACK_JSON_FAIL_CLOSED = "legacy_json_fail_closed"
PRODUCTION_PATH_OVERRIDE_ENVS = (
    "TRITON_PAPER_SELL_ARM_STORE_ROOT",
    "TRITON_PAPER_SELL_ARM_RESULTS_ROOT",
    "TRITON_PAPER_SELL_ARM_DB",
    "RESULTS_ROOT",
)

CHALLENGE_STATES = frozenset({"OPEN", "CONSUMED", "EXPIRED", "REJECTED"})
AUTHORITY_STATES = frozenset({"UNUSED", "CONSUMED", "EXPIRED", "RESTRICTED", "REVOKED"})
DISPATCH_STATES = frozenset(
    {"INTENT_COMMITTED", "SUBMITTED", "REJECTED", "OUTCOME_UNKNOWN", "NOT_SUBMITTED"}
)
TERMINAL_DISPATCH_STATES = frozenset({"SUBMITTED", "REJECTED", "NOT_SUBMITTED"})
IN_PROGRESS_DISPATCH_STATES = frozenset({"INTENT_COMMITTED"})

INTENT_CLASS_NO_INTENT = "no_intent"
INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL = "foreign_historical_terminal"
INTENT_CLASS_MATCHING_CURRENT_TERMINAL = "matching_current_terminal"
INTENT_CLASS_MATCHING_CURRENT_IN_PROGRESS = "matching_current_in_progress"
INTENT_CLASS_AMBIGUOUS = "ambiguous"
INTENT_CLASS_FOREIGN_NONTERMINAL = "foreign_nonterminal"

REASON_MATCHING_TERMINAL = "matching_dispatch_intent_already_terminal"
REASON_MATCHING_IN_PROGRESS = "matching_dispatch_intent_recovery_required"
REASON_INTENT_AMBIGUOUS = "dispatch_intent_identity_ambiguous"
REASON_FOREIGN_NONTERMINAL = "foreign_dispatch_intent_not_terminal"


class PaperSellArmStoreError(RuntimeError):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch_hash_from_scope(scope: Any) -> str:
    return sha256_text(canonical_json(scope))


def sqlite_permit_grade_enabled() -> bool:
    """SQLite is permit-grade unless an explicit fail-closed JSON rollback is selected."""
    mode = str(os.environ.get(ROLLBACK_ENV) or "").strip().lower()
    if mode == ROLLBACK_JSON_FAIL_CLOSED:
        return False
    return True


def canonical_runtime_db_path() -> Path:
    """Sole default permit-grade SQLite path: data/live/paper_sell_arm.sqlite3."""
    return Path(DEFAULT_RUNTIME_DB)


def resolve_runtime_sqlite_path(*, db_path: Optional[Path] = None) -> Path:
    """Production: canonical live DB. Tests: pass an explicit isolated Path."""
    if db_path is not None:
        return Path(db_path)
    return canonical_runtime_db_path()


def db_path_for_store_root(store_root: Path) -> Path:
    """Isolated/test colocation helper. Production must use resolve_runtime_sqlite_path()."""
    return Path(store_root).resolve() / DEFAULT_DB_NAME


def production_path_overrides_present() -> Tuple[bool, str]:
    for key in PRODUCTION_PATH_OVERRIDE_ENVS:
        if str(os.environ.get(key) or "").strip():
            return True, "configuration_blocked"
    return False, "ok"


def refuse_unsupported_db_path(path: Path) -> None:
    raw = str(path)
    probe = raw.replace("/", "\\")
    if probe.startswith("\\\\") or raw.startswith("//") or raw.startswith("\\\\"):
        raise PaperSellArmStoreError(
            "sqlite_unc_or_network_path_refused",
            {"path_kind": "unc"},
        )
    try:
        exists = path.exists() or path.parent.exists()
    except OSError as exc:
        raise PaperSellArmStoreError(
            "sqlite_unc_or_network_path_refused",
            {"path_kind": "os_error"},
        ) from exc
    resolved = str(path.resolve()) if exists else raw
    probe = resolved.replace("/", "\\")
    if probe.startswith("\\\\") or raw.startswith("//") or raw.startswith("\\\\"):
        raise PaperSellArmStoreError(
            "sqlite_unc_or_network_path_refused",
            {"path_kind": "unc"},
        )
    if os.name == "nt":
        drive = os.path.splitdrive(probe)[0]
        if drive:
            try:
                import ctypes

                GetDriveTypeW = ctypes.windll.kernel32.GetDriveTypeW
                GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
                GetDriveTypeW.restype = ctypes.c_uint
                dtype = int(GetDriveTypeW(drive + "\\"))
                # DRIVE_REMOTE = 4, DRIVE_NO_ROOT_DIR = 1
                if dtype in {4}:
                    raise PaperSellArmStoreError(
                        "sqlite_network_drive_refused",
                        {"drive": drive, "drive_type": dtype},
                    )
            except PaperSellArmStoreError:
                raise
            except Exception:
                pass


def _pragma_scalar(conn: sqlite3.Connection, pragma: str) -> Any:
    row = conn.execute(f"PRAGMA {pragma}").fetchone()
    if row is None:
        return None
    return row[0]


def apply_and_verify_pragmas(conn: sqlite3.Connection) -> Dict[str, Any]:
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    journal = str(_pragma_scalar(conn, "journal_mode=WAL") or "").lower()
    if journal != "wal":
        raise PaperSellArmStoreError(
            "sqlite_journal_mode_not_wal",
            {"journal_mode": journal},
        )
    conn.execute("PRAGMA synchronous=FULL")
    syn = _pragma_scalar(conn, "synchronous")
    try:
        syn_i = int(syn)
    except (TypeError, ValueError) as exc:
        raise PaperSellArmStoreError(
            "sqlite_synchronous_not_full",
            {"synchronous": syn},
        ) from exc
    if syn_i != 2:
        raise PaperSellArmStoreError(
            "sqlite_synchronous_not_full",
            {"synchronous": syn_i},
        )
    fk = _pragma_scalar(conn, "foreign_keys")
    if int(fk or 0) != 1:
        raise PaperSellArmStoreError(
            "sqlite_foreign_keys_not_on",
            {"foreign_keys": fk},
        )
    timeout = _pragma_scalar(conn, "busy_timeout")
    return {
        "journal_mode": journal,
        "synchronous": syn_i,
        "foreign_keys": int(fk),
        "busy_timeout": timeout,
        "sqlite_version": sqlite3.sqlite_version,
        "durability_note": (
            "WAL+FULL is the declared SQLite contract. Durability remains subject "
            "to SQLite's documented operating-system and storage assumptions."
        ),
    }


def verify_pragmas_or_fail(conn: sqlite3.Connection) -> Dict[str, Any]:
    journal = str(_pragma_scalar(conn, "journal_mode") or "").lower()
    syn = _pragma_scalar(conn, "synchronous")
    fk = _pragma_scalar(conn, "foreign_keys")
    if journal != "wal":
        raise PaperSellArmStoreError(
            "sqlite_journal_mode_not_wal",
            {"journal_mode": journal},
        )
    if int(syn or 0) != 2:
        raise PaperSellArmStoreError(
            "sqlite_synchronous_not_full",
            {"synchronous": syn},
        )
    if int(fk or 0) != 1:
        raise PaperSellArmStoreError(
            "sqlite_foreign_keys_not_on",
            {"foreign_keys": fk},
        )
    return {
        "journal_mode": journal,
        "synchronous": int(syn),
        "foreign_keys": int(fk),
        "busy_timeout": _pragma_scalar(conn, "busy_timeout"),
        "user_version": int(_pragma_scalar(conn, "user_version") or 0),
        "sqlite_version": sqlite3.sqlite_version,
    }


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sell_challenges (
  challenge_id TEXT PRIMARY KEY,
  plan_hash TEXT NOT NULL,
  batch_hash TEXT NOT NULL,
  batch_id TEXT NOT NULL,
  canonical_batch_json TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('OPEN','CONSUMED','EXPIRED','REJECTED')),
  consumed_at TEXT,
  consuming_authority_id TEXT,
  operator_action_label TEXT NOT NULL,
  schema_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sell_authorities (
  authority_id TEXT PRIMARY KEY,
  challenge_id TEXT NOT NULL UNIQUE,
  plan_hash TEXT NOT NULL,
  batch_hash TEXT NOT NULL,
  canonical_batch_json TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  single_use INTEGER NOT NULL CHECK (single_use = 1),
  state TEXT NOT NULL CHECK (state IN ('UNUSED','CONSUMED','EXPIRED','RESTRICTED','REVOKED')),
  consumption_time TEXT,
  consuming_dispatch_id TEXT,
  final_result TEXT,
  schema_version TEXT NOT NULL,
  FOREIGN KEY (challenge_id) REFERENCES sell_challenges(challenge_id)
);
CREATE TABLE IF NOT EXISTS sell_dispatches (
  dispatch_id TEXT PRIMARY KEY,
  authority_id TEXT NOT NULL UNIQUE,
  plan_hash TEXT NOT NULL,
  batch_hash TEXT NOT NULL,
  canonical_batch_json TEXT NOT NULL,
  intent_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('INTENT_COMMITTED','SUBMITTED','REJECTED','OUTCOME_UNKNOWN','NOT_SUBMITTED')),
  outcome_at TEXT,
  broker_ref TEXT,
  result_class TEXT,
  error_class TEXT,
  reconciliation_required INTEGER NOT NULL DEFAULT 0 CHECK (reconciliation_required IN (0,1)),
  symbols_submitted_json TEXT NOT NULL DEFAULT '[]',
  schema_version TEXT NOT NULL,
  FOREIGN KEY (authority_id) REFERENCES sell_authorities(authority_id)
);
CREATE TABLE IF NOT EXISTS sell_audit_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  event_type TEXT NOT NULL,
  challenge_id TEXT,
  authority_id TEXT,
  dispatch_id TEXT,
  payload_json TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sell_authorities_challenge
  ON sell_authorities(challenge_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sell_dispatches_authority
  ON sell_dispatches(authority_id);
CREATE INDEX IF NOT EXISTS idx_sell_challenges_state ON sell_challenges(state);
CREATE INDEX IF NOT EXISTS idx_sell_authorities_state ON sell_authorities(state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sell_challenges_open_exact_batch
  ON sell_challenges(plan_hash, batch_hash, operator_action_label)
  WHERE state = 'OPEN';
"""


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    dups = conn.execute(
        """
        SELECT plan_hash, batch_hash, operator_action_label, COUNT(*) AS c
        FROM sell_challenges
        WHERE state='OPEN'
        GROUP BY plan_hash, batch_hash, operator_action_label
        HAVING c > 1
        """
    ).fetchall()
    if dups:
        raise PaperSellArmStoreError(
            "sqlite_duplicate_open_requires_reconciliation",
            {"duplicate_open_groups": len(dups)},
        )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sell_challenges_open_exact_batch
          ON sell_challenges(plan_hash, batch_hash, operator_action_label)
          WHERE state = 'OPEN'
        """
    )
    conn.execute(f"PRAGMA user_version={SCHEMA_USER_VERSION}")
    conn.execute(
        """
        INSERT INTO sell_audit_events(created_at, event_type, payload_json)
        VALUES (?,?,?)
        """,
        (
            _utc_now().isoformat(),
            "schema_migrated_v1_to_v2",
            canonical_json(
                {
                    "user_version": SCHEMA_USER_VERSION,
                    "store_schema": STORE_SCHEMA_VERSION,
                }
            ),
        ),
    )


def _init_schema(conn: sqlite3.Connection) -> None:
    version = int(_pragma_scalar(conn, "user_version") or 0)
    if version not in {0, LEGACY_SCHEMA_USER_VERSION, SCHEMA_USER_VERSION}:
        raise PaperSellArmStoreError(
            "sqlite_incompatible_schema_version",
            {"user_version": version, "expected": SCHEMA_USER_VERSION},
        )
    if version == 0:
        # executescript auto-commits; do not wrap it in BEGIN/COMMIT.
        conn.executescript(_SCHEMA_SQL)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"PRAGMA user_version={SCHEMA_USER_VERSION}")
            conn.execute(
                "INSERT INTO sell_audit_events(created_at, event_type, payload_json) VALUES (?,?,?)",
                (
                    _utc_now().isoformat(),
                    "schema_initialized",
                    canonical_json(
                        {
                            "user_version": SCHEMA_USER_VERSION,
                            "store_schema": STORE_SCHEMA_VERSION,
                            "imported_json_authority": False,
                            "json_role": "AUDIT_EXPORT_ONLY",
                        }
                    ),
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
    elif version == LEGACY_SCHEMA_USER_VERSION:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _migrate_v1_to_v2(conn)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
    _require_tables(conn)


def _require_tables(conn: sqlite3.Connection) -> None:
    needed = {
        "sell_challenges",
        "sell_authorities",
        "sell_dispatches",
        "sell_audit_events",
    }
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    have = {str(r[0]) for r in rows}
    missing = sorted(needed - have)
    if missing:
        raise PaperSellArmStoreError(
            "sqlite_schema_tables_missing",
            {"missing": missing},
        )


def integrity_ok(conn: sqlite3.Connection) -> Tuple[bool, str]:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    msg = str(row[0] if row else "missing")
    return msg.lower() == "ok", msg


_SCHEMA_INIT_LOCK = threading.Lock()


def open_connection(db_path: Path, *, initialize: bool = True) -> sqlite3.Connection:
    refuse_unsupported_db_path(db_path)
    db_path = Path(db_path)
    if db_path.exists() and db_path.stat().st_size > 0:
        head = db_path.read_bytes()[:16]
        if not head.startswith(b"SQLite format 3"):
            raise PaperSellArmStoreError(
                "sqlite_database_corrupt_or_incompatible",
                {"hint": "refusing_silent_recreate"},
            )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        with _SCHEMA_INIT_LOCK:
            apply_and_verify_pragmas(conn)
            if initialize:
                _init_schema(conn)
                ok, msg = integrity_ok(conn)
                if not ok:
                    raise PaperSellArmStoreError(
                        "sqlite_integrity_check_failed",
                        {"integrity_check": msg},
                    )
                verify_pragmas_or_fail(conn)
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def write_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = open_connection(db_path, initialize=True)
    try:
        yield conn
    finally:
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        conn.close()


def health_check(db_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    details: Dict[str, Any] = {"db_path_kind": "local_fixed_disk"}
    try:
        refuse_unsupported_db_path(db_path)
        conn = open_connection(db_path, initialize=True)
        try:
            pragmas = verify_pragmas_or_fail(conn)
            ok, msg = integrity_ok(conn)
            details.update(pragmas)
            details["integrity_check"] = msg
            if not ok:
                return False, "sqlite_integrity_check_failed", details
            return True, "ok", details
        finally:
            conn.close()
    except PaperSellArmStoreError as exc:
        details.update(exc.details)
        return False, exc.reason, details


def _audit(
    conn: sqlite3.Connection,
    event_type: str,
    *,
    challenge_id: Optional[str] = None,
    authority_id: Optional[str] = None,
    dispatch_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sell_audit_events(
            created_at, event_type, challenge_id, authority_id, dispatch_id, payload_json
        ) VALUES (?,?,?,?,?,?)
        """,
        (
            _utc_now().isoformat(),
            event_type,
            challenge_id,
            authority_id,
            dispatch_id,
            canonical_json(payload or {}),
        ),
    )


def persist_open_challenge(
    db_path: Path,
    *,
    challenge: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    cid = str(challenge.get("challenge_id") or "")
    if not cid:
        raise PaperSellArmStoreError("sqlite_challenge_missing_id")
    scope = challenge.get("authorized_scope") or []
    bhash = batch_hash_from_scope(scope)
    created = str(challenge.get("issued_at") or now.isoformat())
    expires = str(challenge.get("expires_at") or "")
    plan_hash = str(challenge.get("plan_hash") or "")
    label = str(challenge.get("operator_button") or "ARM_REVIEWED_SELLS")
    now_iso = now.isoformat()
    try:
        with write_connection(db_path) as conn:
            verify_pragmas_or_fail(conn)
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    UPDATE sell_challenges
                    SET state='EXPIRED'
                    WHERE state='OPEN' AND expires_at <= ?
                    """,
                    (now_iso,),
                )
                existing = conn.execute(
                    """
                    SELECT challenge_id FROM sell_challenges
                    WHERE state='OPEN'
                      AND plan_hash=?
                      AND batch_hash=?
                      AND operator_action_label=?
                    """,
                    (plan_hash, bhash, label),
                ).fetchone()
                if existing:
                    raise PaperSellArmStoreError(
                        "sqlite_duplicate_open_challenge",
                        {"existing_challenge_id": existing["challenge_id"]},
                    )
                conn.execute(
                    """
                    INSERT INTO sell_challenges(
                        challenge_id, plan_hash, batch_hash, batch_id, canonical_batch_json,
                        payload_json, created_at, expires_at, state, consumed_at,
                        consuming_authority_id, operator_action_label, schema_version
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        cid,
                        plan_hash,
                        bhash,
                        str(challenge.get("batch_id") or ""),
                        canonical_json(scope),
                        canonical_json(challenge),
                        created,
                        expires,
                        "OPEN",
                        None,
                        None,
                        label,
                        str(challenge.get("schema_version") or STORE_SCHEMA_VERSION),
                    ),
                )
                _audit(
                    conn,
                    "challenge_persisted",
                    challenge_id=cid,
                    payload={"state": "OPEN"},
                )
                conn.execute("COMMIT")
            except PaperSellArmStoreError:
                conn.execute("ROLLBACK")
                raise
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise PaperSellArmStoreError(
                    "sqlite_duplicate_open_challenge",
                    {"error": str(exc)},
                ) from exc
            except sqlite3.OperationalError as exc:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise PaperSellArmStoreError(
                    "sqlite_database_locked",
                    {"error": str(exc)},
                ) from exc
            except Exception:
                conn.execute("ROLLBACK")
                raise
    except sqlite3.OperationalError as exc:
        raise PaperSellArmStoreError(
            "sqlite_database_locked",
            {"error": str(exc)},
        ) from exc
    except PaperSellArmStoreError as exc:
        if exc.reason == "sqlite_duplicate_open_challenge":
            raise
        snap = inspect_sqlite_arm_snapshot(db_path, now=now)
        for rec in snap.get("open_challenges") or []:
            if (
                str(rec.get("plan_hash") or "") == plan_hash
                and str(rec.get("batch_hash") or "") == bhash
                and not rec.get("expired")
            ):
                raise PaperSellArmStoreError(
                    "sqlite_duplicate_open_challenge",
                    {"existing_challenge_id": rec.get("challenge_id")},
                ) from exc
        raise
    except Exception as exc:
        snap = inspect_sqlite_arm_snapshot(db_path, now=now)
        for rec in snap.get("open_challenges") or []:
            if (
                str(rec.get("plan_hash") or "") == plan_hash
                and str(rec.get("batch_hash") or "") == bhash
                and not rec.get("expired")
            ):
                raise PaperSellArmStoreError(
                    "sqlite_duplicate_open_challenge",
                    {"existing_challenge_id": rec.get("challenge_id")},
                ) from exc
        raise PaperSellArmStoreError("sqlite_database_locked", {"error": str(exc)}) from exc
    return {"challenge_id": cid, "batch_hash": bhash, "state": "OPEN"}


def consume_challenge_and_create_authority(
    db_path: Path,
    *,
    challenge_id: str,
    readiness_plan_hash: str,
    readiness_batch_id: str,
    readiness_batch_hash: str,
    authority: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    """One BEGIN IMMEDIATE: validate challenge, mark consumed, insert unused authority."""
    aid = str(authority.get("audit_id") or "") or f"sell-auth-{uuid.uuid4().hex[:16]}"
    authority = dict(authority)
    authority["authority_id"] = aid
    now_iso = now.isoformat()
    with write_connection(db_path) as conn:
        verify_pragmas_or_fail(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM sell_challenges WHERE challenge_id=?",
                (challenge_id,),
            ).fetchone()
            if row is None:
                raise PaperSellArmStoreError(
                    "sqlite_challenge_not_found", {"challenge_id": challenge_id}
                )
            if str(row["state"]) != "OPEN":
                raise PaperSellArmStoreError(
                    "sqlite_challenge_not_open",
                    {"state": row["state"]},
                )
            exp = str(row["expires_at"] or "")
            if exp <= now_iso:
                conn.execute(
                    "UPDATE sell_challenges SET state='EXPIRED' WHERE challenge_id=? AND state='OPEN'",
                    (challenge_id,),
                )
                raise PaperSellArmStoreError("sqlite_challenge_expired")
            if str(row["plan_hash"]) != str(readiness_plan_hash):
                raise PaperSellArmStoreError("sqlite_challenge_plan_hash_mismatch")
            if str(row["batch_id"]) != str(readiness_batch_id):
                raise PaperSellArmStoreError("sqlite_challenge_batch_id_mismatch")
            if str(row["batch_hash"]) != str(readiness_batch_hash):
                raise PaperSellArmStoreError("sqlite_challenge_batch_hash_mismatch")
            existing = conn.execute(
                "SELECT authority_id FROM sell_authorities WHERE challenge_id=?",
                (challenge_id,),
            ).fetchone()
            if existing:
                raise PaperSellArmStoreError(
                    "sqlite_challenge_replay",
                    {"authority_id": existing["authority_id"]},
                )
            conn.execute(
                """
                UPDATE sell_challenges
                SET state='CONSUMED', consumed_at=?, consuming_authority_id=?
                WHERE challenge_id=? AND state='OPEN'
                """,
                (now_iso, aid, challenge_id),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise PaperSellArmStoreError("sqlite_challenge_consume_race")
            scope = authority.get("authorized_actions") or []
            conn.execute(
                """
                INSERT INTO sell_authorities(
                    authority_id, challenge_id, plan_hash, batch_hash, canonical_batch_json,
                    payload_json, created_at, expires_at, single_use, state,
                    consumption_time, consuming_dispatch_id, final_result, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,1,'UNUSED',NULL,NULL,NULL,?)
                """,
                (
                    aid,
                    challenge_id,
                    str(authority.get("plan_hash") or ""),
                    batch_hash_from_scope(scope),
                    canonical_json(scope),
                    canonical_json(authority),
                    str(authority.get("issued_at") or now_iso),
                    str(authority.get("expires_at") or ""),
                    str(authority.get("schema_version") or STORE_SCHEMA_VERSION),
                ),
            )
            _audit(
                conn,
                "authority_created",
                challenge_id=challenge_id,
                authority_id=aid,
                payload={"state": "UNUSED"},
            )
            conn.execute("COMMIT")
        except PaperSellArmStoreError:
            conn.execute("ROLLBACK")
            raise
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise PaperSellArmStoreError(
                "sqlite_challenge_replay",
                {"error": str(exc)},
            ) from exc
        except sqlite3.OperationalError as exc:
            conn.execute("ROLLBACK")
            raise PaperSellArmStoreError(
                "sqlite_database_locked",
                {"error": str(exc)},
            ) from exc
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {"authority_id": aid, "challenge_id": challenge_id, "state": "UNUSED"}


def load_sqlite_authority_payload(db_path: Path, *, now: datetime) -> Optional[Dict[str, Any]]:
    status = inspect_sqlite_authority(db_path, now=now)
    if status["state"] == "UNUSED":
        return status.get("payload")
    return None


def _readonly_sqlite_uri(db_path: Path) -> str:
    resolved = Path(db_path).resolve()
    posix = resolved.as_posix()
    if os.name == "nt" and len(posix) >= 2 and posix[1] == ":":
        posix = "/" + posix
    return f"file:{posix}?mode=ro&immutable=1"


def readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Open existing DB read-only. Never creates files, schema, WAL, or SHM."""
    refuse_unsupported_db_path(db_path)
    path = Path(db_path)
    if not path.is_file():
        raise PaperSellArmStoreError("sqlite_absent")
    conn = sqlite3.connect(_readonly_sqlite_uri(path), uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def inspect_sqlite_authority(db_path: Path, *, now: datetime) -> Dict[str, Any]:
    """Inspect permit-grade authority without treating JSON as authoritative."""
    if not Path(db_path).is_file():
        return {"state": "ABSENT"}
    try:
        conn = readonly_connection(db_path)
    except PaperSellArmStoreError as exc:
        return {"state": "ABSENT", "reason": exc.reason, "ok": False}
    try:
        row = conn.execute(
            """
            SELECT * FROM sell_authorities
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return {"state": "ABSENT"}
        payload = json.loads(row["payload_json"])
        st = str(row["state"])
        if st == "UNUSED" and str(row["expires_at"] or "") <= now.isoformat():
            return {
                "state": "EXPIRED",
                "payload": payload,
                "reason": "sqlite_authority_expired",
            }
        return {"state": st, "payload": payload, "expires_at": row["expires_at"]}
    except sqlite3.Error as exc:
        return {
            "state": "ABSENT",
            "ok": False,
            "reason": "sqlite_readonly_inspect_failed",
        }
    finally:
        conn.close()


def inspect_sqlite_arm_snapshot(db_path: Path, *, now: datetime) -> Dict[str, Any]:
    """Read-only challenge/authority/dispatch counts. Does not create or mutate the DB."""
    empty: Dict[str, Any] = {
        "db_present": False,
        "ok": True,
        "reason": "sqlite_absent",
        "challenge_count": 0,
        "authority_count": 0,
        "dispatch_count": 0,
        "open_challenges": [],
        "latest_challenge": None,
        "latest_authority": {"state": "ABSENT"},
    }
    path = Path(db_path)
    if not path.is_file():
        return empty
    wal = Path(str(path) + "-wal")
    if wal.is_file() and wal.stat().st_size > 0:
        empty["db_present"] = True
        empty["ok"] = False
        empty["reason"] = "sqlite_readonly_wal_unreadable"
        return empty
    try:
        conn = readonly_connection(path)
    except PaperSellArmStoreError as exc:
        empty["db_present"] = True
        empty["ok"] = False
        empty["reason"] = exc.reason
        return empty
    try:
        challenges = conn.execute(
            "SELECT * FROM sell_challenges ORDER BY created_at DESC"
        ).fetchall()
        authorities = conn.execute(
            "SELECT * FROM sell_authorities ORDER BY created_at DESC"
        ).fetchall()
        dispatch_count = int(conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0])
        open_challenges: List[Dict[str, Any]] = []
        latest_challenge = None
        for row in challenges:
            payload = json.loads(row["payload_json"] or "{}")
            rec = {
                "challenge_id": row["challenge_id"],
                "state": row["state"],
                "plan_hash": row["plan_hash"],
                "batch_hash": row["batch_hash"],
                "batch_id": row["batch_id"],
                "expires_at": row["expires_at"],
                "created_at": row["created_at"],
                "payload": payload,
            }
            if latest_challenge is None:
                latest_challenge = rec
            if str(row["state"]) == "OPEN":
                expired = str(row["expires_at"] or "") <= now.isoformat()
                rec["expired"] = expired
                open_challenges.append(rec)
        latest_authority: Dict[str, Any] = {"state": "ABSENT"}
        if authorities:
            arow = authorities[0]
            payload = json.loads(arow["payload_json"] or "{}")
            st = str(arow["state"])
            if st == "UNUSED" and str(arow["expires_at"] or "") <= now.isoformat():
                latest_authority = {
                    "state": "EXPIRED",
                    "authority_id": arow["authority_id"],
                    "payload": payload,
                    "expires_at": arow["expires_at"],
                    "single_use": bool(arow["single_use"]),
                    "reason": "sqlite_authority_expired",
                }
            else:
                latest_authority = {
                    "state": st,
                    "authority_id": arow["authority_id"],
                    "payload": payload,
                    "expires_at": arow["expires_at"],
                    "single_use": bool(arow["single_use"]),
                    "plan_hash": arow["plan_hash"],
                    "batch_hash": arow["batch_hash"],
                }
        return {
            "db_present": True,
            "ok": True,
            "reason": "ok",
            "challenge_count": len(challenges),
            "authority_count": len(authorities),
            "dispatch_count": dispatch_count,
            "open_challenges": open_challenges,
            "latest_challenge": latest_challenge,
            "latest_authority": latest_authority,
        }
    except sqlite3.Error:
        empty["db_present"] = True
        empty["ok"] = False
        empty["reason"] = "sqlite_readonly_inspect_failed"
        return empty
    finally:
        conn.close()


def count_dispatches_for_authority(db_path: Path, authority_id: str) -> int:
    """Count sell_dispatches rows bound to one authority_id. History for others is ignored."""
    aid = str(authority_id or "").strip()
    if not aid or not Path(db_path).is_file():
        return 0
    conn = readonly_connection(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM sell_dispatches WHERE authority_id = ?",
            (aid,),
        ).fetchone()
        return int(row[0] if row is not None else 0)
    finally:
        conn.close()


def inspect_authority_row(db_path: Path, authority_id: str) -> Optional[Dict[str, Any]]:
    """Read-only row for one authority_id. Never mutates."""
    aid = str(authority_id or "").strip()
    if not aid or not Path(db_path).is_file():
        return None
    conn = readonly_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM sell_authorities WHERE authority_id=?",
            (aid,),
        ).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}
    finally:
        conn.close()


def inspect_dispatches_for_authority(db_path: Path, authority_id: str) -> List[Dict[str, Any]]:
    """Read-only dispatch rows for one authority_id."""
    aid = str(authority_id or "").strip()
    if not aid or not Path(db_path).is_file():
        return []
    conn = readonly_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM sell_dispatches WHERE authority_id=? ORDER BY created_at",
            (aid,),
        ).fetchall()
        return [{k: row[k] for k in row.keys()} for row in rows]
    finally:
        conn.close()


def _nonempty_identity_values(*values: Any) -> set:
    out = set()
    for value in values:
        text = str(value or "").strip()
        if text:
            out.add(text)
    return out


def _dispatch_terminal_proof(dispatches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in dispatches:
        state = str(row.get("state") or "")
        result = str(row.get("result_class") or row.get("final_result") or "")
        if state in TERMINAL_DISPATCH_STATES or result.upper() in {"SUCCESS", "SUBMITTED"}:
            if state == "OUTCOME_UNKNOWN":
                continue
            return row
    return None


def classify_dispatch_intent_for_authority(
    intent: Optional[Mapping[str, Any]],
    authority: Mapping[str, Any],
    *,
    db_path: Optional[Path] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify a durable JSON dispatch-intent relative to one current authority.

    SQLite is the only source that can prove a historical dispatch terminal.
    JSON intent identity never grants a broker permit.
    """
    details: Dict[str, Any] = {"intent_present": bool(intent)}
    if intent is None:
        return INTENT_CLASS_NO_INTENT, "", details
    if not isinstance(intent, Mapping) or not dict(intent):
        return (
            INTENT_CLASS_AMBIGUOUS,
            REASON_INTENT_AMBIGUOUS,
            {
                **details,
                "error": "intent_not_object",
            },
        )

    auth_ids = _nonempty_identity_values(
        authority.get("authority_id"),
        authority.get("audit_id"),
        authority.get("approval_id"),
    )
    intent_ids = _nonempty_identity_values(
        intent.get("authority_id"),
        intent.get("audit_id"),
        intent.get("approval_id"),
    )
    auth_key = str(authority.get("idempotency_key") or "").strip()
    intent_key = str(intent.get("idempotency_key") or "").strip()
    auth_plan = str(authority.get("plan_hash") or "").strip()
    intent_plan = str(intent.get("plan_hash") or "").strip()
    auth_batch = str(authority.get("batch_id") or "").strip()
    intent_batch = str(intent.get("batch_id") or "").strip()
    auth_date = str(authority.get("trading_date") or "").strip()
    intent_date = str(intent.get("trading_date") or "").strip()
    auth_packet = str(authority.get("readiness_packet_hash") or "").strip()
    intent_packet = str(
        intent.get("readiness_packet_hash") or intent.get("packet_hash") or ""
    ).strip()
    dispatch_id = str(intent.get("dispatch_intent_id") or intent.get("dispatch_id") or "").strip()
    details.update(
        {
            "dispatch_intent_id": dispatch_id,
            "intent_ids": sorted(intent_ids),
            "authority_ids": sorted(auth_ids),
        }
    )

    if not dispatch_id and not intent_ids and not intent_key:
        return INTENT_CLASS_AMBIGUOUS, REASON_INTENT_AMBIGUOUS, details

    id_overlap = bool(intent_ids and auth_ids and intent_ids & auth_ids)
    id_foreign = bool(intent_ids and auth_ids and intent_ids.isdisjoint(auth_ids))
    mixed_ids = bool(intent_ids and auth_ids and intent_ids - auth_ids and intent_ids & auth_ids)
    key_match = bool(intent_key and auth_key and intent_key == auth_key)
    key_conflict = bool(intent_key and auth_key and intent_key != auth_key)

    if mixed_ids or (id_overlap and key_conflict) or (key_match and id_foreign):
        return INTENT_CLASS_AMBIGUOUS, REASON_INTENT_AMBIGUOUS, details

    appears_current = id_overlap or key_match
    if appears_current and intent_plan and auth_plan and intent_plan != auth_plan:
        return INTENT_CLASS_AMBIGUOUS, REASON_INTENT_AMBIGUOUS, details
    if appears_current and intent_batch and auth_batch and intent_batch != auth_batch:
        return INTENT_CLASS_AMBIGUOUS, REASON_INTENT_AMBIGUOUS, details
    if appears_current and intent_date and auth_date and intent_date != auth_date:
        return INTENT_CLASS_AMBIGUOUS, REASON_INTENT_AMBIGUOUS, details
    if appears_current and intent_packet and auth_packet and intent_packet != auth_packet:
        return INTENT_CLASS_AMBIGUOUS, REASON_INTENT_AMBIGUOUS, details

    path = Path(db_path) if db_path is not None else None
    current_aid = str(authority.get("authority_id") or authority.get("audit_id") or "").strip()
    intent_aid = str(intent.get("authority_id") or intent.get("audit_id") or "").strip()

    if appears_current:
        dispatches = inspect_dispatches_for_authority(path, current_aid) if path else []
        terminal = _dispatch_terminal_proof(dispatches)
        details["sqlite_dispatch_count"] = len(dispatches)
        if terminal is not None:
            details["sqlite_dispatch_id"] = terminal.get("dispatch_id")
            details["sqlite_dispatch_state"] = terminal.get("state")
            return (
                INTENT_CLASS_MATCHING_CURRENT_TERMINAL,
                REASON_MATCHING_TERMINAL,
                details,
            )
        return INTENT_CLASS_MATCHING_CURRENT_IN_PROGRESS, REASON_MATCHING_IN_PROGRESS, details

    if id_foreign or (key_conflict and intent_ids):
        lookup_id = intent_aid
        dispatches = inspect_dispatches_for_authority(path, lookup_id) if path else []
        auth_row = inspect_authority_row(path, lookup_id) if path else None
        terminal = _dispatch_terminal_proof(dispatches)
        details["foreign_authority_id"] = lookup_id
        details["foreign_sqlite_dispatch_count"] = len(dispatches)
        details["foreign_sqlite_authority_state"] = (
            None if auth_row is None else auth_row.get("state")
        )
        if terminal is not None:
            details["sqlite_dispatch_id"] = terminal.get("dispatch_id")
            details["sqlite_dispatch_state"] = terminal.get("state")
            return INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL, "", details
        return INTENT_CLASS_FOREIGN_NONTERMINAL, REASON_FOREIGN_NONTERMINAL, details

    return INTENT_CLASS_AMBIGUOUS, REASON_INTENT_AMBIGUOUS, details


def count_current_unused_exact_batch_authorities(
    db_path: Path,
    *,
    now: datetime,
    plan_hash: str,
    batch_hash: str,
) -> int:
    """Count UNUSED, unexpired authorities for one exact plan/batch. Ignores history."""
    if not Path(db_path).is_file():
        return 0
    conn = readonly_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM sell_authorities").fetchall()
        n = 0
        now_iso = now.isoformat()
        for row in rows:
            if str(row["state"]) != "UNUSED":
                continue
            if str(row["expires_at"] or "") <= now_iso:
                continue
            if str(row["plan_hash"] or "") != str(plan_hash):
                continue
            if str(row["batch_hash"] or "") != str(batch_hash):
                continue
            n += 1
        return n
    finally:
        conn.close()


def consume_authority_insert_dispatch(
    db_path: Path,
    *,
    authority_id: str,
    plan_hash: str,
    batch_hash: str,
    expected_scope: Any,
    mode: str,
    live_blocked: bool,
    requested: List[Dict[str, Any]],
    now: datetime,
) -> Dict[str, Any]:
    """Atomic consume + dispatch intent. Permit only after COMMIT."""
    if str(mode).lower() != "paper":
        raise PaperSellArmStoreError("sqlite_mode_not_paper", {"mode": mode})
    if not live_blocked:
        raise PaperSellArmStoreError("sqlite_live_execution_not_blocked")
    dispatch_id = f"sell-dispatch-{uuid.uuid4().hex[:16]}"
    now_iso = now.isoformat()
    with write_connection(db_path) as conn:
        verify_pragmas_or_fail(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM sell_authorities WHERE authority_id=?",
                (authority_id,),
            ).fetchone()
            if row is None:
                raise PaperSellArmStoreError("sqlite_authority_not_found")
            if str(row["state"]) != "UNUSED":
                raise PaperSellArmStoreError(
                    "sqlite_authority_not_unused",
                    {"state": row["state"]},
                )
            if int(row["single_use"] or 0) != 1:
                raise PaperSellArmStoreError("sqlite_authority_not_single_use")
            if str(row["expires_at"] or "") <= now_iso:
                conn.execute(
                    "UPDATE sell_authorities SET state='EXPIRED' WHERE authority_id=? AND state='UNUSED'",
                    (authority_id,),
                )
                raise PaperSellArmStoreError("sqlite_authority_expired")
            if str(row["plan_hash"]) != str(plan_hash):
                raise PaperSellArmStoreError("sqlite_authority_plan_hash_mismatch")
            if str(row["batch_hash"]) != str(batch_hash):
                raise PaperSellArmStoreError("sqlite_authority_batch_hash_mismatch")
            stored_scope = json.loads(row["canonical_batch_json"])
            if expected_scope and batch_hash_from_scope(expected_scope) != str(row["batch_hash"]):
                raise PaperSellArmStoreError("sqlite_expected_scope_mismatch")
            _assert_requested_within_scope(stored_scope, requested)
            intent = {
                "dispatch_id": dispatch_id,
                "authority_id": authority_id,
                "requested": requested,
                "recorded_at": now_iso,
                "status": "INTENT_COMMITTED",
            }
            conn.execute(
                """
                UPDATE sell_authorities
                SET state='CONSUMED', consumption_time=?, consuming_dispatch_id=?, final_result='DISPATCH_INTENT'
                WHERE authority_id=? AND state='UNUSED'
                """,
                (now_iso, dispatch_id, authority_id),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise PaperSellArmStoreError("sqlite_authority_consume_race")
            conn.execute(
                """
                INSERT INTO sell_dispatches(
                    dispatch_id, authority_id, plan_hash, batch_hash, canonical_batch_json,
                    intent_json, created_at, state, reconciliation_required,
                    symbols_submitted_json, schema_version
                ) VALUES (?,?,?,?,?,?,?,'INTENT_COMMITTED',0,'[]',?)
                """,
                (
                    dispatch_id,
                    authority_id,
                    plan_hash,
                    batch_hash,
                    canonical_json(stored_scope),
                    canonical_json(intent),
                    now_iso,
                    STORE_SCHEMA_VERSION,
                ),
            )
            _audit(
                conn,
                "dispatch_intent_committed",
                authority_id=authority_id,
                dispatch_id=dispatch_id,
                payload={"state": "INTENT_COMMITTED"},
            )
            conn.execute("COMMIT")
        except PaperSellArmStoreError:
            conn.execute("ROLLBACK")
            raise
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise PaperSellArmStoreError(
                "sqlite_duplicate_dispatch",
                {"error": str(exc)},
            ) from exc
        except sqlite3.OperationalError as exc:
            conn.execute("ROLLBACK")
            raise PaperSellArmStoreError(
                "sqlite_database_locked",
                {"error": str(exc)},
            ) from exc
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {
        "dispatch_id": dispatch_id,
        "authority_id": authority_id,
        "state": "INTENT_COMMITTED",
        "permit_eligible": True,
    }


def _assert_requested_within_scope(
    stored_scope: List[Dict[str, Any]], requested: List[Dict[str, Any]]
) -> None:
    by_sym = {str(s.get("symbol") or "").upper(): s for s in stored_scope}
    for row in requested:
        sym = str(row.get("symbol") or "").upper()
        action = str(row.get("action") or "").upper()
        qty = float(row.get("qty") or 0)
        if str(row.get("side") or "sell").lower() != "sell":
            raise PaperSellArmStoreError("sqlite_buy_add_not_authorized_on_sell_store")
        if sym not in by_sym:
            raise PaperSellArmStoreError("sqlite_symbol_mismatch", {"symbol": sym})
        allowed = by_sym[sym]
        if str(allowed.get("action") or "").upper() != action:
            raise PaperSellArmStoreError(
                "sqlite_action_mismatch",
                {"symbol": sym, "action": action},
            )
        if qty > float(allowed.get("max_qty") or 0) + 1e-9:
            raise PaperSellArmStoreError(
                "sqlite_quantity_mismatch",
                {"symbol": sym, "qty": qty, "max_qty": allowed.get("max_qty")},
            )


def load_committed_dispatch(db_path: Path, dispatch_id: str) -> Optional[sqlite3.Row]:
    if not Path(db_path).is_file():
        return None
    with write_connection(db_path) as conn:
        return conn.execute(
            "SELECT * FROM sell_dispatches WHERE dispatch_id=?",
            (dispatch_id,),
        ).fetchone()


def require_permit_sqlite_state(
    db_path: Path,
    *,
    dispatch_id: str,
    permit_token: str,
    symbol: str,
    qty: Optional[float],
) -> Dict[str, Any]:
    with write_connection(db_path) as conn:
        verify_pragmas_or_fail(conn)
        drow = conn.execute(
            "SELECT * FROM sell_dispatches WHERE dispatch_id=?",
            (dispatch_id,),
        ).fetchone()
        if drow is None:
            raise PaperSellArmStoreError("sqlite_dispatch_not_committed")
        arow = conn.execute(
            "SELECT * FROM sell_authorities WHERE authority_id=?",
            (drow["authority_id"],),
        ).fetchone()
        if arow is None or str(arow["state"]) != "CONSUMED":
            raise PaperSellArmStoreError("sqlite_authority_not_consumed_at_submit")
        if str(drow["state"]) not in {"INTENT_COMMITTED", "SUBMITTED"}:
            if str(drow["state"]) in {"REJECTED", "NOT_SUBMITTED"}:
                raise PaperSellArmStoreError("sqlite_dispatch_not_submittable")
        submitted = json.loads(drow["symbols_submitted_json"] or "[]")
        submitted_u = [str(s).upper() for s in submitted]
        sym = str(symbol or "").upper()
        if not sym:
            raise PaperSellArmStoreError("sqlite_permit_symbol_required")
        if sym in submitted_u:
            raise PaperSellArmStoreError("sqlite_permit_symbol_already_submitted")
        scope = json.loads(drow["canonical_batch_json"])
        by_sym = {str(s.get("symbol") or "").upper(): s for s in scope}
        if sym not in by_sym:
            raise PaperSellArmStoreError("sqlite_permit_symbol_not_authorized")
        if qty is not None and float(qty) > float(by_sym[sym].get("max_qty") or 0) + 1e-9:
            raise PaperSellArmStoreError("sqlite_permit_quantity_exceeds_authorized")
        conn.execute("BEGIN IMMEDIATE")
        try:
            submitted_u.append(sym)
            conn.execute(
                """
                UPDATE sell_dispatches
                SET symbols_submitted_json=?, state='SUBMITTED'
                WHERE dispatch_id=?
                """,
                (canonical_json(submitted_u), dispatch_id),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return {"dispatch_id": dispatch_id, "symbols_submitted": submitted_u}


def record_dispatch_outcome(
    db_path: Path,
    *,
    dispatch_id: str,
    outcome_state: str,
    broker_ref: Optional[str] = None,
    result_class: Optional[str] = None,
    error_class: Optional[str] = None,
    reconciliation_required: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    if outcome_state not in DISPATCH_STATES:
        raise PaperSellArmStoreError("sqlite_invalid_outcome_state", {"state": outcome_state})
    now_iso = (now or _utc_now()).isoformat()
    with write_connection(db_path) as conn:
        verify_pragmas_or_fail(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM sell_dispatches WHERE dispatch_id=?",
                (dispatch_id,),
            ).fetchone()
            if row is None:
                raise PaperSellArmStoreError("sqlite_dispatch_not_found")
            # Never restore authority to UNUSED.
            conn.execute(
                """
                UPDATE sell_dispatches
                SET state=?, outcome_at=?, broker_ref=?, result_class=?, error_class=?,
                    reconciliation_required=?
                WHERE dispatch_id=?
                """,
                (
                    outcome_state,
                    now_iso,
                    broker_ref,
                    result_class,
                    error_class,
                    1 if reconciliation_required else 0,
                    dispatch_id,
                ),
            )
            conn.execute(
                """
                UPDATE sell_authorities
                SET state='CONSUMED', final_result=?
                WHERE authority_id=?
                """,
                (outcome_state, row["authority_id"]),
            )
            _audit(
                conn,
                "dispatch_outcome",
                authority_id=row["authority_id"],
                dispatch_id=dispatch_id,
                payload={
                    "state": outcome_state,
                    "reconciliation_required": reconciliation_required,
                },
            )
            conn.execute("COMMIT")
        except PaperSellArmStoreError:
            conn.execute("ROLLBACK")
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {"dispatch_id": dispatch_id, "state": outcome_state}


def dump_schema_sql() -> str:
    return _SCHEMA_SQL
