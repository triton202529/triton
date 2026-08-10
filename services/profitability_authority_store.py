"""UIR-003B-HC durable profitability-authority evidence store (SQLite).

Import has no side effects. The default production path is never opened unless
an explicit store instance is constructed with that path by a caller.
UIR-003B validation must use temporary directories only.

Schema version 1 was corrected in-place before the first committed release:
UIR-003B remained uncommitted, no production database was created, and no
migration artifact is required.

The store is the final authority-enforcement boundary. Certified boundaries and
verified epochs require exact membership in the immutable production capability
registries defined in this module. Both registries are empty in UIR-003B-HC.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

SCHEMA_VERSION = "1"

# Documented future production location. Never auto-created by import.
DEFAULT_PRODUCTION_STORE_PATH = Path("data/reporting_authority/profitability_authority.sqlite3")

# ---------------------------------------------------------------------------
# Store-owned immutable production authority capability registries (empty).
# Keys are exact authority identifiers. Values are frozensets of the
# classifications / statuses each authority may grant.
# No constructor, parameter, env var, file, DB row, payload, adapter, or
# coordinator option may extend or replace these registries.
# ---------------------------------------------------------------------------
PRODUCTION_BOUNDARY_AUTHORITY_REGISTRY: Mapping[str, frozenset[str]] = MappingProxyType({})
PRODUCTION_EPOCH_AUTHORITY_REGISTRY: Mapping[str, frozenset[str]] = MappingProxyType({})

# Stable sanitized authority failure codes (never embed rejected IDs/payloads).
AUTHORITY_BOUNDARY_REQUIRED = "AUTHORITY_BOUNDARY_REQUIRED"
AUTHORITY_BOUNDARY_UNTRUSTED = "AUTHORITY_BOUNDARY_UNTRUSTED"
AUTHORITY_BOUNDARY_PROVENANCE_INVALID = "AUTHORITY_BOUNDARY_PROVENANCE_INVALID"
AUTHORITY_EPOCH_REQUIRED = "AUTHORITY_EPOCH_REQUIRED"
AUTHORITY_EPOCH_UNTRUSTED = "AUTHORITY_EPOCH_UNTRUSTED"
AUTHORITY_EPOCH_PROVENANCE_INVALID = "AUTHORITY_EPOCH_PROVENANCE_INVALID"
PAGE_INDEX_INVALID = "PAGE_INDEX_INVALID"

CAPTURE_STATUSES = frozenset(
    {
        "CAPTURE_STARTED",
        "CAPTURE_INCOMPLETE",
        "CAPTURE_COMPLETE_UNVERIFIED",
        "CAPTURE_FAILED",
    }
)

EPOCH_STATUSES = frozenset(
    {
        "EPOCH_UNVERIFIED",
        "EPOCH_VERIFIED",
    }
)

BOUNDARY_CLASSIFICATIONS = frozenset(
    {
        "INSTANTANEOUS_ACCOUNT_OBSERVATION",
        "BROKER_PORTFOLIO_HISTORY_OBSERVATION",
        "RECORDED_DATE_ONLY_OBSERVATION",
        "CERTIFIED_SESSION_OPEN",
        "CERTIFIED_SESSION_CLOSE",
        "BOUNDARY_UNVERIFIED",
    }
)

CERTIFIED_BOUNDARIES = frozenset({"CERTIFIED_SESSION_OPEN", "CERTIFIED_SESSION_CLOSE"})

ACTIVITY_CATEGORIES = frozenset(
    {
        "EXTERNAL_CONTRIBUTION",
        "EXTERNAL_WITHDRAWAL",
        "INVESTMENT_INCOME",
        "INVESTMENT_EXPENSE",
        "INTERNAL_TRADE_CASH_MOVEMENT",
        "ASSET_TRANSFER",
        "PAPER_ACCOUNT_RESET",
        "UNKNOWN",
    }
)

WARNING_CODES = frozenset(
    {
        "UNTRUSTED_CERTIFIED_BOUNDARY_CLAIM",
        "UNTRUSTED_EPOCH_VERIFICATION_CLAIM",
        "UNKNOWN_ACTIVITY_MAPPING",
        "FUTURE_SOURCE_TIMESTAMP",
        "INCOMPLETE_PAGINATION",
        "PAGE_SEQUENCE_GAP",
        "CREDENTIAL_MATERIAL_REJECTED",
    }
)

EVIDENCE_STREAMS = frozenset({"activities", "equity_observations"})


class ProfitabilityAuthorityStoreError(Exception):
    """Base store error."""


class StoreConflictError(ProfitabilityAuthorityStoreError):
    """Same stable ID with incompatible payload."""


class StoreValidationError(ProfitabilityAuthorityStoreError):
    """Invalid record for persistence."""


class StoreAuthorityError(ProfitabilityAuthorityStoreError):
    """Governed authority required but missing or invalid."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def enforce_boundary_authority(classification: str, authority_id: Optional[str]) -> None:
    """Fail closed unless certified boundary has exact trusted capability."""
    if classification in CERTIFIED_BOUNDARIES:
        if not isinstance(authority_id, str) or authority_id == "":
            raise StoreAuthorityError(AUTHORITY_BOUNDARY_REQUIRED)
        allowed = PRODUCTION_BOUNDARY_AUTHORITY_REGISTRY.get(authority_id)
        if allowed is None or classification not in allowed:
            raise StoreAuthorityError(AUTHORITY_BOUNDARY_UNTRUSTED)
        return
    if authority_id is not None:
        raise StoreAuthorityError(AUTHORITY_BOUNDARY_PROVENANCE_INVALID)


def enforce_epoch_authority(status: str, authority_id: Optional[str]) -> None:
    """Fail closed unless EPOCH_VERIFIED has exact trusted capability."""
    if status == "EPOCH_VERIFIED":
        if not isinstance(authority_id, str) or authority_id == "":
            raise StoreAuthorityError(AUTHORITY_EPOCH_REQUIRED)
        allowed = PRODUCTION_EPOCH_AUTHORITY_REGISTRY.get(authority_id)
        if allowed is None or "EPOCH_VERIFIED" not in allowed:
            raise StoreAuthorityError(AUTHORITY_EPOCH_UNTRUSTED)
        return
    if authority_id is not None:
        raise StoreAuthorityError(AUTHORITY_EPOCH_PROVENANCE_INVALID)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic JSON encoding for payload hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload_sha256(payload: Any) -> str:
    return sha256_hex(canonical_json_bytes(payload))


def decimal_to_text(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, Decimal):
        raise StoreValidationError("monetary values must be Decimal instances for persistence")
    text = format(value, "f")
    if text == "-0" or text == "-0.0":
        return "0"
    return text


def text_to_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise StoreValidationError(f"invalid decimal text: {value!r}") from exc


@dataclass(frozen=True)
class CaptureRunRecord:
    run_id: str
    schema_version: str
    source: str
    environment: str
    account_fingerprint: str
    requested_coverage_start: Optional[str]
    requested_coverage_end: Optional[str]
    capture_start_timestamp: str
    capture_completion_timestamp: Optional[str]
    pagination_complete: bool
    source_page_count: int
    received_row_count: int
    inserted_row_count: int
    duplicate_row_count: int
    unknown_activity_count: int
    error_count: int
    warning_count: int
    warning_codes: str
    final_capture_status: str
    source_payload_hash: Optional[str]
    source_semantic_policy_id: Optional[str]
    future_timestamp_flag_count: int = 0
    notes: str = ""


@dataclass(frozen=True)
class AccountEpochRecord:
    epoch_id: str
    environment: str
    account_fingerprint: str
    broker_account_created_timestamp: Optional[str]
    first_observed_timestamp: Optional[str]
    latest_observed_timestamp: Optional[str]
    explicit_reset_marker: Optional[str]
    epoch_verification_status: str
    source_capture_run_id: str
    source_provenance: str
    source_epoch_claim: Optional[str] = None
    epoch_authority_id: Optional[str] = None


@dataclass(frozen=True)
class PageManifestRecord:
    capture_run_id: str
    evidence_stream: str
    page_index: int
    coordinator_receive_sequence: int
    page_token_hash: Optional[str]
    canonical_page_hash: str
    row_count: int
    has_more: bool
    inserted_timestamp: str


@dataclass(frozen=True)
class CaptureWarningRecord:
    capture_run_id: str
    warning_code: str
    evidence_stream: Optional[str]
    page_index: Optional[int]
    row_index: Optional[int]
    detail_code: str


@dataclass(frozen=True)
class BrokerActivityRecord:
    environment: str
    account_epoch_id: str
    broker_activity_id: Optional[str]
    evidence_key: str
    activity_type: str
    governed_activity_category: str
    amount: Optional[Decimal]
    net_amount: Optional[Decimal]
    currency: Optional[str]
    symbol: Optional[str]
    side: Optional[str]
    quantity: Optional[Decimal]
    price: Optional[Decimal]
    per_share_amount: Optional[Decimal]
    effective_timestamp: Optional[str]
    transaction_timestamp: Optional[str]
    description: Optional[str]
    capture_run_id: str
    evidence_stream: str
    source_page_index: int
    source_row_index: int
    source_page_token_hash: Optional[str]
    canonical_source_page_hash: str
    raw_source_payload: Mapping[str, Any]
    raw_source_payload_sha256: str
    evidence_key_is_derived: bool
    future_timestamp_flag: bool = False


@dataclass(frozen=True)
class EquityObservationRecord:
    environment: str
    account_epoch_id: str
    source_observation_id: Optional[str]
    evidence_key: str
    observation_timestamp: Optional[str]
    timezone: Optional[str]
    session_date: Optional[str]
    cash: Optional[Decimal]
    market_value: Optional[Decimal]
    equity: Optional[Decimal]
    currency: Optional[str]
    source_observation_type: str
    boundary_classification: str
    capture_run_id: str
    evidence_stream: str
    source_page_index: int
    source_row_index: int
    source_page_token_hash: Optional[str]
    canonical_source_page_hash: str
    raw_source_payload: Mapping[str, Any]
    raw_source_payload_sha256: str
    evidence_key_is_derived: bool
    boundary_authority_id: Optional[str] = None
    future_timestamp_flag: bool = False


class ProfitabilityAuthorityStore:
    """Transaction-safe SQLite evidence store for profitability authority capture."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._opened = False

    def open(self) -> None:
        if self._opened and self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        self._conn = conn
        self._opened = True
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._opened = False

    def __enter__(self) -> "ProfitabilityAuthorityStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ProfitabilityAuthorityStoreError("store is not open")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self.connection
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self._create_tables()
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            return
        if str(row["value"]) != SCHEMA_VERSION:
            raise ProfitabilityAuthorityStoreError(
                f"unsupported schema_version={row['value']!r}; " f"expected {SCHEMA_VERSION!r}"
            )
        # Ensure hardened tables exist for in-place pre-release schema.
        self._create_tables()

    def _create_tables(self) -> None:
        conn = self.connection
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS capture_runs (
                run_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                source TEXT NOT NULL,
                environment TEXT NOT NULL,
                account_fingerprint TEXT NOT NULL,
                requested_coverage_start TEXT,
                requested_coverage_end TEXT,
                capture_start_timestamp TEXT NOT NULL,
                capture_completion_timestamp TEXT,
                pagination_complete INTEGER NOT NULL,
                source_page_count INTEGER NOT NULL,
                received_row_count INTEGER NOT NULL,
                inserted_row_count INTEGER NOT NULL,
                duplicate_row_count INTEGER NOT NULL,
                unknown_activity_count INTEGER NOT NULL,
                error_count INTEGER NOT NULL,
                warning_count INTEGER NOT NULL DEFAULT 0,
                warning_codes TEXT NOT NULL DEFAULT '',
                final_capture_status TEXT NOT NULL,
                source_payload_hash TEXT,
                source_semantic_policy_id TEXT,
                future_timestamp_flag_count INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS account_epochs (
                epoch_id TEXT PRIMARY KEY,
                environment TEXT NOT NULL,
                account_fingerprint TEXT NOT NULL,
                broker_account_created_timestamp TEXT,
                first_observed_timestamp TEXT,
                latest_observed_timestamp TEXT,
                explicit_reset_marker TEXT,
                epoch_verification_status TEXT NOT NULL,
                source_capture_run_id TEXT NOT NULL,
                source_provenance TEXT NOT NULL,
                source_epoch_claim TEXT,
                epoch_authority_id TEXT,
                UNIQUE(environment, account_fingerprint, epoch_id)
            );

            CREATE TABLE IF NOT EXISTS page_manifests (
                capture_run_id TEXT NOT NULL,
                evidence_stream TEXT NOT NULL,
                page_index INTEGER NOT NULL,
                coordinator_receive_sequence INTEGER NOT NULL,
                page_token_hash TEXT,
                canonical_page_hash TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                has_more INTEGER NOT NULL,
                inserted_timestamp TEXT NOT NULL,
                PRIMARY KEY (capture_run_id, evidence_stream, page_index),
                FOREIGN KEY(capture_run_id) REFERENCES capture_runs(run_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_page_manifest_token
                ON page_manifests(capture_run_id, evidence_stream, page_token_hash)
                WHERE page_token_hash IS NOT NULL;

            CREATE TABLE IF NOT EXISTS capture_warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_run_id TEXT NOT NULL,
                warning_code TEXT NOT NULL,
                evidence_stream TEXT,
                page_index INTEGER,
                row_index INTEGER,
                detail_code TEXT NOT NULL,
                FOREIGN KEY(capture_run_id) REFERENCES capture_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS broker_activities (
                evidence_key TEXT PRIMARY KEY,
                environment TEXT NOT NULL,
                account_epoch_id TEXT NOT NULL,
                broker_activity_id TEXT,
                activity_type TEXT NOT NULL,
                governed_activity_category TEXT NOT NULL,
                amount TEXT,
                net_amount TEXT,
                currency TEXT,
                symbol TEXT,
                side TEXT,
                quantity TEXT,
                price TEXT,
                per_share_amount TEXT,
                effective_timestamp TEXT,
                transaction_timestamp TEXT,
                description TEXT,
                capture_run_id TEXT NOT NULL,
                evidence_stream TEXT NOT NULL,
                source_page_index INTEGER NOT NULL,
                source_row_index INTEGER NOT NULL,
                source_page_token_hash TEXT,
                canonical_source_page_hash TEXT NOT NULL,
                raw_source_payload TEXT NOT NULL,
                raw_source_payload_sha256 TEXT NOT NULL,
                evidence_key_is_derived INTEGER NOT NULL,
                future_timestamp_flag INTEGER NOT NULL DEFAULT 0,
                inserted_timestamp TEXT NOT NULL,
                FOREIGN KEY(account_epoch_id) REFERENCES account_epochs(epoch_id),
                FOREIGN KEY(capture_run_id) REFERENCES capture_runs(run_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_broker_activities_stable_id
                ON broker_activities(environment, account_epoch_id, broker_activity_id)
                WHERE broker_activity_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS equity_observations (
                evidence_key TEXT PRIMARY KEY,
                environment TEXT NOT NULL,
                account_epoch_id TEXT NOT NULL,
                source_observation_id TEXT,
                observation_timestamp TEXT,
                timezone TEXT,
                session_date TEXT,
                cash TEXT,
                market_value TEXT,
                equity TEXT,
                currency TEXT,
                source_observation_type TEXT NOT NULL,
                boundary_classification TEXT NOT NULL,
                boundary_authority_id TEXT,
                capture_run_id TEXT NOT NULL,
                evidence_stream TEXT NOT NULL,
                source_page_index INTEGER NOT NULL,
                source_row_index INTEGER NOT NULL,
                source_page_token_hash TEXT,
                canonical_source_page_hash TEXT NOT NULL,
                raw_source_payload TEXT NOT NULL,
                raw_source_payload_sha256 TEXT NOT NULL,
                evidence_key_is_derived INTEGER NOT NULL,
                future_timestamp_flag INTEGER NOT NULL DEFAULT 0,
                inserted_timestamp TEXT NOT NULL,
                FOREIGN KEY(account_epoch_id) REFERENCES account_epochs(epoch_id),
                FOREIGN KEY(capture_run_id) REFERENCES capture_runs(run_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_equity_obs_stable_id
                ON equity_observations(
                    environment, account_epoch_id, source_observation_id
                )
                WHERE source_observation_id IS NOT NULL;
            """
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def insert_capture_run(self, run: CaptureRunRecord) -> None:
        if run.final_capture_status not in CAPTURE_STATUSES:
            raise StoreValidationError(f"invalid capture status: {run.final_capture_status}")
        self.connection.execute(
            """
            INSERT INTO capture_runs(
                run_id, schema_version, source, environment, account_fingerprint,
                requested_coverage_start, requested_coverage_end,
                capture_start_timestamp, capture_completion_timestamp,
                pagination_complete, source_page_count, received_row_count,
                inserted_row_count, duplicate_row_count, unknown_activity_count,
                error_count, warning_count, warning_codes, final_capture_status,
                source_payload_hash, source_semantic_policy_id,
                future_timestamp_flag_count, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run.run_id,
                run.schema_version,
                run.source,
                run.environment,
                run.account_fingerprint,
                run.requested_coverage_start,
                run.requested_coverage_end,
                run.capture_start_timestamp,
                run.capture_completion_timestamp,
                1 if run.pagination_complete else 0,
                run.source_page_count,
                run.received_row_count,
                run.inserted_row_count,
                run.duplicate_row_count,
                run.unknown_activity_count,
                run.error_count,
                run.warning_count,
                run.warning_codes,
                run.final_capture_status,
                run.source_payload_hash,
                run.source_semantic_policy_id,
                run.future_timestamp_flag_count,
                run.notes,
            ),
        )

    def update_capture_run(self, run: CaptureRunRecord) -> None:
        if run.final_capture_status not in CAPTURE_STATUSES:
            raise StoreValidationError(f"invalid capture status: {run.final_capture_status}")
        cur = self.connection.execute(
            """
            UPDATE capture_runs SET
                schema_version=?, source=?, environment=?, account_fingerprint=?,
                requested_coverage_start=?, requested_coverage_end=?,
                capture_start_timestamp=?, capture_completion_timestamp=?,
                pagination_complete=?, source_page_count=?, received_row_count=?,
                inserted_row_count=?, duplicate_row_count=?, unknown_activity_count=?,
                error_count=?, warning_count=?, warning_codes=?,
                final_capture_status=?, source_payload_hash=?,
                source_semantic_policy_id=?, future_timestamp_flag_count=?, notes=?
            WHERE run_id=?
            """,
            (
                run.schema_version,
                run.source,
                run.environment,
                run.account_fingerprint,
                run.requested_coverage_start,
                run.requested_coverage_end,
                run.capture_start_timestamp,
                run.capture_completion_timestamp,
                1 if run.pagination_complete else 0,
                run.source_page_count,
                run.received_row_count,
                run.inserted_row_count,
                run.duplicate_row_count,
                run.unknown_activity_count,
                run.error_count,
                run.warning_count,
                run.warning_codes,
                run.final_capture_status,
                run.source_payload_hash,
                run.source_semantic_policy_id,
                run.future_timestamp_flag_count,
                run.notes,
                run.run_id,
            ),
        )
        if cur.rowcount != 1:
            raise StoreValidationError(f"capture run not found: {run.run_id}")

    def upsert_account_epoch(self, epoch: AccountEpochRecord) -> None:
        if epoch.epoch_verification_status not in EPOCH_STATUSES:
            raise StoreValidationError(f"invalid epoch status: {epoch.epoch_verification_status}")
        enforce_epoch_authority(epoch.epoch_verification_status, epoch.epoch_authority_id)
        existing = self.connection.execute(
            "SELECT * FROM account_epochs WHERE epoch_id = ?",
            (epoch.epoch_id,),
        ).fetchone()
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO account_epochs(
                    epoch_id, environment, account_fingerprint,
                    broker_account_created_timestamp, first_observed_timestamp,
                    latest_observed_timestamp, explicit_reset_marker,
                    epoch_verification_status, source_capture_run_id,
                    source_provenance, source_epoch_claim, epoch_authority_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    epoch.epoch_id,
                    epoch.environment,
                    epoch.account_fingerprint,
                    epoch.broker_account_created_timestamp,
                    epoch.first_observed_timestamp,
                    epoch.latest_observed_timestamp,
                    epoch.explicit_reset_marker,
                    epoch.epoch_verification_status,
                    epoch.source_capture_run_id,
                    epoch.source_provenance,
                    epoch.source_epoch_claim,
                    epoch.epoch_authority_id,
                ),
            )
            return
        if (
            existing["environment"] != epoch.environment
            or existing["account_fingerprint"] != epoch.account_fingerprint
        ):
            raise StoreConflictError("account epoch identity conflict for epoch_id")
        if (
            existing["explicit_reset_marker"]
            and epoch.explicit_reset_marker
            and existing["explicit_reset_marker"] != epoch.explicit_reset_marker
        ):
            raise StoreConflictError("conflicting explicit_reset_marker for account epoch")
        if (
            existing["broker_account_created_timestamp"]
            and epoch.broker_account_created_timestamp
            and existing["broker_account_created_timestamp"]
            != epoch.broker_account_created_timestamp
        ):
            raise StoreConflictError(
                "conflicting broker_account_created_timestamp for account epoch"
            )
        self.connection.execute(
            """
            UPDATE account_epochs SET
                broker_account_created_timestamp=COALESCE(
                    ?, broker_account_created_timestamp
                ),
                first_observed_timestamp=COALESCE(?, first_observed_timestamp),
                latest_observed_timestamp=?,
                explicit_reset_marker=COALESCE(?, explicit_reset_marker),
                epoch_verification_status=?,
                source_capture_run_id=?,
                source_provenance=?,
                source_epoch_claim=COALESCE(?, source_epoch_claim),
                epoch_authority_id=COALESCE(?, epoch_authority_id)
            WHERE epoch_id=?
            """,
            (
                epoch.broker_account_created_timestamp,
                epoch.first_observed_timestamp,
                epoch.latest_observed_timestamp,
                epoch.explicit_reset_marker,
                epoch.epoch_verification_status,
                epoch.source_capture_run_id,
                epoch.source_provenance,
                epoch.source_epoch_claim,
                epoch.epoch_authority_id,
                epoch.epoch_id,
            ),
        )

    def insert_page_manifest(self, page: PageManifestRecord) -> str:
        if page.evidence_stream not in EVIDENCE_STREAMS:
            raise StoreValidationError(f"invalid evidence_stream: {page.evidence_stream}")
        if not isinstance(page.page_index, int) or isinstance(page.page_index, bool):
            raise StoreValidationError(PAGE_INDEX_INVALID)
        if page.page_index < 0:
            raise StoreValidationError(PAGE_INDEX_INVALID)
        existing = self.connection.execute(
            """
            SELECT * FROM page_manifests
            WHERE capture_run_id=? AND evidence_stream=? AND page_index=?
            """,
            (page.capture_run_id, page.evidence_stream, page.page_index),
        ).fetchone()
        if existing is not None:
            if existing["canonical_page_hash"] != page.canonical_page_hash:
                raise StoreConflictError(
                    "page identity conflict: same page_index different content"
                )
            return "duplicate"
        if page.page_token_hash:
            tok = self.connection.execute(
                """
                SELECT * FROM page_manifests
                WHERE capture_run_id=? AND evidence_stream=? AND page_token_hash=?
                """,
                (
                    page.capture_run_id,
                    page.evidence_stream,
                    page.page_token_hash,
                ),
            ).fetchone()
            if tok is not None:
                if tok["canonical_page_hash"] != page.canonical_page_hash:
                    raise StoreConflictError("page token conflict: same token different content")
                return "duplicate"
        self.connection.execute(
            """
            INSERT INTO page_manifests(
                capture_run_id, evidence_stream, page_index,
                coordinator_receive_sequence, page_token_hash,
                canonical_page_hash, row_count, has_more, inserted_timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                page.capture_run_id,
                page.evidence_stream,
                page.page_index,
                page.coordinator_receive_sequence,
                page.page_token_hash,
                page.canonical_page_hash,
                page.row_count,
                1 if page.has_more else 0,
                page.inserted_timestamp,
            ),
        )
        return "inserted"

    def insert_warning(self, warning: CaptureWarningRecord) -> None:
        if warning.warning_code not in WARNING_CODES:
            raise StoreValidationError(f"invalid warning_code: {warning.warning_code}")
        self.connection.execute(
            """
            INSERT INTO capture_warnings(
                capture_run_id, warning_code, evidence_stream,
                page_index, row_index, detail_code
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                warning.capture_run_id,
                warning.warning_code,
                warning.evidence_stream,
                warning.page_index,
                warning.row_index,
                warning.detail_code,
            ),
        )

    def insert_broker_activity(self, activity: BrokerActivityRecord) -> str:
        if activity.governed_activity_category not in ACTIVITY_CATEGORIES:
            raise StoreValidationError(
                f"invalid governed category: {activity.governed_activity_category}"
            )
        if activity.evidence_stream != "activities":
            raise StoreValidationError("activity evidence_stream must be activities")
        if (
            not isinstance(activity.source_page_index, int)
            or isinstance(activity.source_page_index, bool)
            or activity.source_page_index < 0
        ):
            raise StoreValidationError(PAGE_INDEX_INVALID)
        existing = self._find_existing_activity(activity)
        raw_json = canonical_json_bytes(dict(activity.raw_source_payload)).decode("utf-8")
        if existing is not None:
            if existing["raw_source_payload_sha256"] != activity.raw_source_payload_sha256:
                raise StoreConflictError(
                    "broker activity conflict: same identity with different payload hash"
                )
            return "duplicate"
        self.connection.execute(
            """
            INSERT INTO broker_activities(
                evidence_key, environment, account_epoch_id, broker_activity_id,
                activity_type, governed_activity_category,
                amount, net_amount, currency, symbol, side, quantity, price,
                per_share_amount, effective_timestamp, transaction_timestamp,
                description, capture_run_id, evidence_stream,
                source_page_index, source_row_index, source_page_token_hash,
                canonical_source_page_hash, raw_source_payload,
                raw_source_payload_sha256, evidence_key_is_derived,
                future_timestamp_flag, inserted_timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                activity.evidence_key,
                activity.environment,
                activity.account_epoch_id,
                activity.broker_activity_id,
                activity.activity_type,
                activity.governed_activity_category,
                decimal_to_text(activity.amount),
                decimal_to_text(activity.net_amount),
                activity.currency,
                activity.symbol,
                activity.side,
                decimal_to_text(activity.quantity),
                decimal_to_text(activity.price),
                decimal_to_text(activity.per_share_amount),
                activity.effective_timestamp,
                activity.transaction_timestamp,
                activity.description,
                activity.capture_run_id,
                activity.evidence_stream,
                activity.source_page_index,
                activity.source_row_index,
                activity.source_page_token_hash,
                activity.canonical_source_page_hash,
                raw_json,
                activity.raw_source_payload_sha256,
                1 if activity.evidence_key_is_derived else 0,
                1 if activity.future_timestamp_flag else 0,
                _utc_now_iso(),
            ),
        )
        return "inserted"

    def _find_existing_activity(self, activity: BrokerActivityRecord) -> Optional[sqlite3.Row]:
        if activity.broker_activity_id:
            row = self.connection.execute(
                """
                SELECT * FROM broker_activities
                WHERE environment=? AND account_epoch_id=? AND broker_activity_id=?
                """,
                (
                    activity.environment,
                    activity.account_epoch_id,
                    activity.broker_activity_id,
                ),
            ).fetchone()
            if row is not None:
                return row
        return self.connection.execute(
            "SELECT * FROM broker_activities WHERE evidence_key=?",
            (activity.evidence_key,),
        ).fetchone()

    def insert_equity_observation(self, obs: EquityObservationRecord) -> str:
        if obs.boundary_classification not in BOUNDARY_CLASSIFICATIONS:
            raise StoreValidationError(
                f"invalid boundary classification: {obs.boundary_classification}"
            )
        enforce_boundary_authority(obs.boundary_classification, obs.boundary_authority_id)
        if obs.evidence_stream != "equity_observations":
            raise StoreValidationError("equity evidence_stream must be equity_observations")
        if (
            not isinstance(obs.source_page_index, int)
            or isinstance(obs.source_page_index, bool)
            or obs.source_page_index < 0
        ):
            raise StoreValidationError(PAGE_INDEX_INVALID)
        existing = self._find_existing_equity(obs)
        raw_json = canonical_json_bytes(dict(obs.raw_source_payload)).decode("utf-8")
        if existing is not None:
            if existing["raw_source_payload_sha256"] != obs.raw_source_payload_sha256:
                raise StoreConflictError(
                    "equity observation conflict: same identity with different payload hash"
                )
            return "duplicate"
        self.connection.execute(
            """
            INSERT INTO equity_observations(
                evidence_key, environment, account_epoch_id, source_observation_id,
                observation_timestamp, timezone, session_date,
                cash, market_value, equity, currency,
                source_observation_type, boundary_classification,
                boundary_authority_id, capture_run_id, evidence_stream,
                source_page_index, source_row_index, source_page_token_hash,
                canonical_source_page_hash, raw_source_payload,
                raw_source_payload_sha256, evidence_key_is_derived,
                future_timestamp_flag, inserted_timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                obs.evidence_key,
                obs.environment,
                obs.account_epoch_id,
                obs.source_observation_id,
                obs.observation_timestamp,
                obs.timezone,
                obs.session_date,
                decimal_to_text(obs.cash),
                decimal_to_text(obs.market_value),
                decimal_to_text(obs.equity),
                obs.currency,
                obs.source_observation_type,
                obs.boundary_classification,
                obs.boundary_authority_id,
                obs.capture_run_id,
                obs.evidence_stream,
                obs.source_page_index,
                obs.source_row_index,
                obs.source_page_token_hash,
                obs.canonical_source_page_hash,
                raw_json,
                obs.raw_source_payload_sha256,
                1 if obs.evidence_key_is_derived else 0,
                1 if obs.future_timestamp_flag else 0,
                _utc_now_iso(),
            ),
        )
        return "inserted"

    def _find_existing_equity(self, obs: EquityObservationRecord) -> Optional[sqlite3.Row]:
        if obs.source_observation_id:
            row = self.connection.execute(
                """
                SELECT * FROM equity_observations
                WHERE environment=? AND account_epoch_id=? AND source_observation_id=?
                """,
                (
                    obs.environment,
                    obs.account_epoch_id,
                    obs.source_observation_id,
                ),
            ).fetchone()
            if row is not None:
                return row
        return self.connection.execute(
            "SELECT * FROM equity_observations WHERE evidence_key=?",
            (obs.evidence_key,),
        ).fetchone()

    def get_capture_run(self, run_id: str) -> Optional[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM capture_runs WHERE run_id=?", (run_id,)
        ).fetchone()

    def list_broker_activities(
        self, *, account_epoch_id: Optional[str] = None
    ) -> List[sqlite3.Row]:
        if account_epoch_id is None:
            return list(
                self.connection.execute("SELECT * FROM broker_activities ORDER BY evidence_key")
            )
        return list(
            self.connection.execute(
                "SELECT * FROM broker_activities WHERE account_epoch_id=? " "ORDER BY evidence_key",
                (account_epoch_id,),
            )
        )

    def list_equity_observations(
        self, *, account_epoch_id: Optional[str] = None
    ) -> List[sqlite3.Row]:
        if account_epoch_id is None:
            return list(
                self.connection.execute("SELECT * FROM equity_observations ORDER BY evidence_key")
            )
        return list(
            self.connection.execute(
                "SELECT * FROM equity_observations WHERE account_epoch_id=? "
                "ORDER BY evidence_key",
                (account_epoch_id,),
            )
        )

    def list_page_manifests(self, run_id: str) -> List[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM page_manifests WHERE capture_run_id=? "
                "ORDER BY evidence_stream, page_index",
                (run_id,),
            )
        )

    def list_warnings(self, run_id: str) -> List[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM capture_warnings WHERE capture_run_id=? ORDER BY id",
                (run_id,),
            )
        )

    def list_account_epochs(self) -> List[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM account_epochs ORDER BY epoch_id"))

    def count_rows(self) -> Dict[str, int]:
        return {
            "capture_runs": int(
                self.connection.execute("SELECT COUNT(*) AS c FROM capture_runs").fetchone()["c"]
            ),
            "account_epochs": int(
                self.connection.execute("SELECT COUNT(*) AS c FROM account_epochs").fetchone()["c"]
            ),
            "broker_activities": int(
                self.connection.execute("SELECT COUNT(*) AS c FROM broker_activities").fetchone()[
                    "c"
                ]
            ),
            "equity_observations": int(
                self.connection.execute("SELECT COUNT(*) AS c FROM equity_observations").fetchone()[
                    "c"
                ]
            ),
            "page_manifests": int(
                self.connection.execute("SELECT COUNT(*) AS c FROM page_manifests").fetchone()["c"]
            ),
            "capture_warnings": int(
                self.connection.execute("SELECT COUNT(*) AS c FROM capture_warnings").fetchone()[
                    "c"
                ]
            ),
        }
