"""UIR-003B-HC offline profitability-authority capture coordinator.

Import has no side effects: no network, no file creation, no broker SDK init.
Accepts a SourceAdapter via dependency injection. Does not calculate profitability.

Authority capability registries live only in profitability_authority_store.
This coordinator never certifies boundaries or verifies epochs.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from services.profitability_authority_store import (
    PAGE_INDEX_INVALID,
    SCHEMA_VERSION,
    AccountEpochRecord,
    BrokerActivityRecord,
    CaptureRunRecord,
    CaptureWarningRecord,
    EquityObservationRecord,
    PageManifestRecord,
    ProfitabilityAuthorityStore,
    StoreConflictError,
    StoreValidationError,
    payload_sha256,
)

logger = logging.getLogger(__name__)

PERMITTED_ENVIRONMENTS = frozenset({"paper"})
REFUSED_ENVIRONMENTS = frozenset({"live"})
CERTIFIED_BOUNDARIES = frozenset({"CERTIFIED_SESSION_OPEN", "CERTIFIED_SESSION_CLOSE"})

# Page-index convention: zero-based contiguous integers starting at 0.
PAGE_INDEX_CONVENTION = "zero_based_contiguous"

CREDENTIAL_KEY_NAMES = frozenset(
    {
        "authorization",
        "api_key",
        "api_secret",
        "secret_key",
        "client_secret",
        "access_token",
        "refresh_token",
        "password",
        "private_key",
    }
)

# ---------------------------------------------------------------------------
# Source-semantic policy (trusted application configuration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSemanticPolicy:
    """Code-controlled activity classification policy.

    Not selectable from raw payloads. Adapter source must match policy.source.
    """

    policy_id: str
    source: str
    version: str
    activity_type_map: Mapping[str, str]

    def classify(self, activity_type: str) -> str:
        key = str(activity_type or "").strip().upper()
        if not key:
            return "UNKNOWN"
        return self.activity_type_map.get(key, "UNKNOWN")


# Fixture-only policy — not approved Alpaca production mappings.
FIXTURE_UIR003B_V1 = SourceSemanticPolicy(
    policy_id="FIXTURE_UIR003B_V1",
    source="uir003b_fixtures",
    version="1",
    activity_type_map={
        "FILL": "INTERNAL_TRADE_CASH_MOVEMENT",
        "PARTIAL_FILL": "INTERNAL_TRADE_CASH_MOVEMENT",
        "DIV": "INVESTMENT_INCOME",
        "DIVIDEND": "INVESTMENT_INCOME",
        "INT": "INVESTMENT_INCOME",
        "INTEREST": "INVESTMENT_INCOME",
        "FEE": "INVESTMENT_EXPENSE",
        "REG_FEE": "INVESTMENT_EXPENSE",
        "COMMISSION": "INVESTMENT_EXPENSE",
        "CSD": "EXTERNAL_CONTRIBUTION",
        "CSW": "EXTERNAL_WITHDRAWAL",
        "ACATS": "ASSET_TRANSFER",
        "PAPER_RESET": "PAPER_ACCOUNT_RESET",
        "ACCOUNT_RESET": "PAPER_ACCOUNT_RESET",
        # TRANSFER, DEPOSIT, WITHDRAWAL, TRADE, BUY, SELL intentionally absent.
    },
)

KNOWN_POLICIES: Mapping[str, SourceSemanticPolicy] = {
    FIXTURE_UIR003B_V1.policy_id: FIXTURE_UIR003B_V1,
}


class CaptureError(Exception):
    """Fatal capture failure (fail closed)."""

    def __init__(self, code: str, *, coordinate: str = "") -> None:
        self.code = code
        self.coordinate = coordinate
        msg = code if not coordinate else f"{code}@{coordinate}"
        super().__init__(msg)


class CaptureEnvironmentError(CaptureError):
    """Invalid or refused environment."""


class CaptureValidationError(CaptureError):
    """Invalid source evidence."""


class CaptureCredentialError(CaptureError):
    """Credential-like material detected in payload."""


def redact_secrets(text: str) -> str:
    """Redact account identifiers and credential-like tokens from text."""
    if not text:
        return text
    redacted = text
    patterns = [
        (r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]"),
        (r"(?i)(secret\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]"),
        (r"(?i)(password\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]"),
        (r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]"),
        (r"(?i)(account[_-]?id\s*[:=]\s*)([A-Za-z0-9_-]+)", r"\1[REDACTED]"),
        (r"(?i)(account[_-]?number\s*[:=]\s*)([A-Za-z0-9_-]+)", r"\1[REDACTED]"),
        (r"(?i)\b(acct_|account_)([A-Za-z0-9]{6,})\b", r"\1[REDACTED]"),
    ]
    for pat, repl in patterns:
        redacted = re.sub(pat, repl, redacted)
    return redacted


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def assert_no_credential_material(payload: Any, *, coordinate: str) -> None:
    """Fail closed if credential-like keys appear anywhere in the payload tree."""

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                nk = _normalize_key(k)
                if nk in {_normalize_key(x) for x in CREDENTIAL_KEY_NAMES}:
                    raise CaptureCredentialError(
                        "CREDENTIAL_MATERIAL_REJECTED", coordinate=coordinate
                    )
                walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)


# ---------------------------------------------------------------------------
# Source adapter contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountMetadata:
    environment: str
    account_fingerprint: str
    broker_account_created_timestamp: Optional[str] = None
    explicit_reset_marker: Optional[str] = None
    source_epoch_claim: Optional[str] = None
    source_epoch_verification_claim: Optional[str] = None
    source_provenance: str = "fixture_or_adapter"


@dataclass(frozen=True)
class ActivityPage:
    """Page-index convention: zero-based contiguous integers starting at 0."""

    page_index: int
    activities: Sequence[Mapping[str, Any]]
    has_more: bool
    page_token: Optional[str] = None


@dataclass(frozen=True)
class EquityObservationPage:
    """Page-index convention: zero-based contiguous integers starting at 0."""

    page_index: int
    observations: Sequence[Mapping[str, Any]]
    has_more: bool
    page_token: Optional[str] = None


@dataclass(frozen=True)
class CaptureProvenance:
    source: str
    adapter_name: str
    notes: str = ""


@runtime_checkable
class SourceAdapter(Protocol):
    """Offline-capable source of profitability evidence pages."""

    def get_account_metadata(self) -> AccountMetadata: ...

    def iter_activity_pages(self) -> Iterator[ActivityPage]: ...

    def iter_equity_observation_pages(self) -> Iterator[EquityObservationPage]: ...

    def get_capture_provenance(self) -> CaptureProvenance: ...


# ---------------------------------------------------------------------------
# Numeric / timestamp truth
# ---------------------------------------------------------------------------


def require_decimal(value: Any, *, field_name: str) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    if isinstance(value, float):
        raise CaptureValidationError(
            "FLOAT_REFUSED",
            coordinate=field_name,
        )
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise CaptureValidationError("INVALID_DECIMAL", coordinate=field_name) from exc
    raise CaptureValidationError("UNSUPPORTED_NUMERIC_TYPE", coordinate=field_name)


def require_currency_if_amounts(
    *,
    currency: Any,
    amounts: Sequence[Optional[Decimal]],
    field_name: str = "currency",
) -> Optional[str]:
    supplied = [a for a in amounts if a is not None]
    if not supplied:
        return None if currency in (None, "") else str(currency).upper()
    if currency is None or currency == "":
        raise CaptureValidationError("CURRENCY_REQUIRED", coordinate=field_name)
    return str(currency).upper()


def normalize_aware_timestamp_to_utc(
    value: Any,
    *,
    field_name: str,
    capture_now: Optional[datetime] = None,
) -> Tuple[Optional[str], bool]:
    if value is None or value == "":
        return None, False
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            raise CaptureValidationError("DATE_ONLY_NOT_TIMESTAMP", coordinate=field_name)
        try:
            if text.endswith("Z"):
                dt = datetime.fromisoformat(text[:-1] + "+00:00")
            else:
                dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise CaptureValidationError("UNPARSABLE_TIMESTAMP", coordinate=field_name) from exc
    else:
        raise CaptureValidationError("UNSUPPORTED_TIMESTAMP_TYPE", coordinate=field_name)

    if dt.tzinfo is None:
        raise CaptureValidationError("NAIVE_TIMESTAMP_REFUSED", coordinate=field_name)
    utc_dt = dt.astimezone(timezone.utc)
    iso = utc_dt.isoformat().replace("+00:00", "Z")
    now = capture_now or datetime.now(timezone.utc)
    future = utc_dt > now.astimezone(timezone.utc)
    return iso, future


def parse_session_date_only(value: Any, *, field_name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            date.fromisoformat(text)
            return text
        raise CaptureValidationError("INVALID_DATE_ONLY", coordinate=field_name)
    raise CaptureValidationError("UNSUPPORTED_DATE_TYPE", coordinate=field_name)


def derive_evidence_key(*, kind: str, payload: Mapping[str, Any]) -> str:
    digest = payload_sha256(dict(payload))
    return f"derived:{kind}:{digest}"


def fingerprint_account_token(raw_account_id: Optional[str], *, context: str) -> str:
    """Context-bound one-way digest; never use raw account ID as fingerprint."""
    material = f"{context}|{raw_account_id or ''}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:32]


def hash_page_token(token: Optional[str]) -> Optional[str]:
    if token in (None, ""):
        return None
    return sha256_hex_str(str(token))


def sha256_hex_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Capture result
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult:
    run_id: str
    final_capture_status: str
    environment: str
    account_fingerprint: str
    account_epoch_id: str
    epoch_verification_status: str
    source_epoch_claim: Optional[str]
    pagination_complete: bool
    source_page_count: int
    received_row_count: int
    inserted_row_count: int
    duplicate_row_count: int
    unknown_activity_count: int
    error_count: int
    warning_count: int
    warning_codes: List[str]
    future_timestamp_flag_count: int
    source_payload_hash: Optional[str]
    source_semantic_policy_id: Optional[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "final_capture_status": self.final_capture_status,
            "environment": self.environment,
            "account_fingerprint": self.account_fingerprint,
            "account_epoch_id": self.account_epoch_id,
            "epoch_verification_status": self.epoch_verification_status,
            "source_epoch_claim": self.source_epoch_claim,
            "pagination_complete": self.pagination_complete,
            "source_page_count": self.source_page_count,
            "received_row_count": self.received_row_count,
            "inserted_row_count": self.inserted_row_count,
            "duplicate_row_count": self.duplicate_row_count,
            "unknown_activity_count": self.unknown_activity_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "warning_codes": list(self.warning_codes),
            "future_timestamp_flag_count": self.future_timestamp_flag_count,
            "source_payload_hash": self.source_payload_hash,
            "source_semantic_policy_id": self.source_semantic_policy_id,
            "notes": list(self.notes),
            "authority_claim": None,
            "profitability_available": False,
        }


# ---------------------------------------------------------------------------
# Capture coordinator
# ---------------------------------------------------------------------------


class ProfitabilityAuthorityCapture:
    """Offline capture coordinator. Inject SourceAdapter; never opens broker."""

    def __init__(
        self,
        store: ProfitabilityAuthorityStore,
        adapter: SourceAdapter,
        *,
        source_semantic_policy: Optional[SourceSemanticPolicy] = None,
        run_id: Optional[str] = None,
        capture_now: Optional[datetime] = None,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.policy = source_semantic_policy
        self.run_id = run_id or str(uuid.uuid4())
        self.capture_now = capture_now or datetime.now(timezone.utc)

    def run(
        self,
        *,
        requested_coverage_start: Optional[str] = None,
        requested_coverage_end: Optional[str] = None,
    ) -> CaptureResult:
        self.store.open()
        provenance = self.adapter.get_capture_provenance()
        metadata = self.adapter.get_account_metadata()
        environment = self._validate_environment(metadata.environment)
        policy = self._resolve_policy(provenance.source)

        capture_start = self.capture_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        notes: List[str] = []
        warning_codes: List[str] = []

        # Per-capture internal epoch — never fingerprint-collapsed, never
        # adapter-supplied ID as authoritative PK.
        epoch_id = f"epoch_run:{self.run_id}"
        source_epoch_claim = metadata.source_epoch_claim
        if metadata.source_epoch_verification_claim == "EPOCH_VERIFIED":
            warning_codes.append("UNTRUSTED_EPOCH_VERIFICATION_CLAIM")
        epoch_status = "EPOCH_UNVERIFIED"

        started = CaptureRunRecord(
            run_id=self.run_id,
            schema_version=SCHEMA_VERSION,
            source=provenance.source,
            environment=environment,
            account_fingerprint=metadata.account_fingerprint,
            requested_coverage_start=requested_coverage_start,
            requested_coverage_end=requested_coverage_end,
            capture_start_timestamp=capture_start,
            capture_completion_timestamp=None,
            pagination_complete=False,
            source_page_count=0,
            received_row_count=0,
            inserted_row_count=0,
            duplicate_row_count=0,
            unknown_activity_count=0,
            error_count=0,
            warning_count=0,
            warning_codes="",
            final_capture_status="CAPTURE_STARTED",
            source_payload_hash=None,
            source_semantic_policy_id=policy.policy_id if policy else None,
            future_timestamp_flag_count=0,
            notes="",
        )

        page_count = 0
        received = 0
        inserted = 0
        duplicates = 0
        unknown_count = 0
        errors = 0
        future_flags = 0
        pagination_complete = True
        observe_times: List[str] = []
        manifest_chunks: List[Dict[str, Any]] = []
        receive_seq = 0

        try:
            with self.store.transaction():
                self.store.insert_capture_run(started)
                # Epoch row must exist before activity/equity FK inserts.
                # first/latest observed timestamps are updated after ingestion.
                self.store.upsert_account_epoch(
                    AccountEpochRecord(
                        epoch_id=epoch_id,
                        environment=environment,
                        account_fingerprint=metadata.account_fingerprint,
                        broker_account_created_timestamp=(
                            metadata.broker_account_created_timestamp
                        ),
                        first_observed_timestamp=capture_start,
                        latest_observed_timestamp=capture_start,
                        explicit_reset_marker=metadata.explicit_reset_marker,
                        epoch_verification_status=epoch_status,
                        source_capture_run_id=self.run_id,
                        source_provenance=metadata.source_provenance,
                        source_epoch_claim=source_epoch_claim,
                        epoch_authority_id=None,
                    )
                )
                for code in warning_codes:
                    self.store.insert_warning(
                        CaptureWarningRecord(
                            capture_run_id=self.run_id,
                            warning_code=code,
                            evidence_stream=None,
                            page_index=None,
                            row_index=None,
                            detail_code=code,
                        )
                    )

                # Activities stream
                act_stats = self._ingest_activity_stream(
                    policy=policy,
                    environment=environment,
                    epoch_id=epoch_id,
                    warning_codes=warning_codes,
                    observe_times=observe_times,
                    manifest_chunks=manifest_chunks,
                    receive_seq_start=receive_seq,
                )
                receive_seq = act_stats["receive_seq"]
                page_count += act_stats["unique_pages"]
                received += act_stats["received"]
                inserted += act_stats["inserted"]
                duplicates += act_stats["duplicates"]
                unknown_count += act_stats["unknown"]
                future_flags += act_stats["future"]
                if not act_stats["stream_complete"]:
                    pagination_complete = False

                # Equity stream
                eq_stats = self._ingest_equity_stream(
                    environment=environment,
                    epoch_id=epoch_id,
                    warning_codes=warning_codes,
                    observe_times=observe_times,
                    manifest_chunks=manifest_chunks,
                    receive_seq_start=receive_seq,
                )
                receive_seq = eq_stats["receive_seq"]
                page_count += eq_stats["unique_pages"]
                received += eq_stats["received"]
                inserted += eq_stats["inserted"]
                duplicates += eq_stats["duplicates"]
                future_flags += eq_stats["future"]
                if not eq_stats["stream_complete"]:
                    pagination_complete = False

                if page_count == 0:
                    pagination_complete = False
                    if "INCOMPLETE_PAGINATION" not in warning_codes:
                        warning_codes.append("INCOMPLETE_PAGINATION")
                        self.store.insert_warning(
                            CaptureWarningRecord(
                                capture_run_id=self.run_id,
                                warning_code="INCOMPLETE_PAGINATION",
                                evidence_stream=None,
                                page_index=None,
                                row_index=None,
                                detail_code="NO_SOURCE_PAGES",
                            )
                        )

                first_obs = min(observe_times) if observe_times else capture_start
                latest_obs = max(observe_times) if observe_times else capture_start
                epoch = AccountEpochRecord(
                    epoch_id=epoch_id,
                    environment=environment,
                    account_fingerprint=metadata.account_fingerprint,
                    broker_account_created_timestamp=(metadata.broker_account_created_timestamp),
                    first_observed_timestamp=first_obs,
                    latest_observed_timestamp=latest_obs,
                    explicit_reset_marker=metadata.explicit_reset_marker,
                    epoch_verification_status=epoch_status,
                    source_capture_run_id=self.run_id,
                    source_provenance=metadata.source_provenance,
                    source_epoch_claim=source_epoch_claim,
                    epoch_authority_id=None,
                )
                self.store.upsert_account_epoch(epoch)

                completion = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                manifest_hash = payload_sha256(
                    {
                        "run_id": self.run_id,
                        "source": provenance.source,
                        "adapter": provenance.adapter_name,
                        "policy_id": policy.policy_id if policy else None,
                        "pages": manifest_chunks,
                    }
                )
                if pagination_complete and errors == 0:
                    status = "CAPTURE_COMPLETE_UNVERIFIED"
                elif errors > 0:
                    status = "CAPTURE_FAILED"
                else:
                    status = "CAPTURE_INCOMPLETE"

                codes_csv = ",".join(sorted(set(warning_codes)))
                final = CaptureRunRecord(
                    run_id=self.run_id,
                    schema_version=SCHEMA_VERSION,
                    source=provenance.source,
                    environment=environment,
                    account_fingerprint=metadata.account_fingerprint,
                    requested_coverage_start=requested_coverage_start,
                    requested_coverage_end=requested_coverage_end,
                    capture_start_timestamp=capture_start,
                    capture_completion_timestamp=completion,
                    pagination_complete=pagination_complete,
                    source_page_count=page_count,
                    received_row_count=received,
                    inserted_row_count=inserted,
                    duplicate_row_count=duplicates,
                    unknown_activity_count=unknown_count,
                    error_count=errors,
                    warning_count=len(set(warning_codes)),
                    warning_codes=codes_csv,
                    final_capture_status=status,
                    source_payload_hash=manifest_hash,
                    source_semantic_policy_id=policy.policy_id if policy else None,
                    future_timestamp_flag_count=future_flags,
                    notes="; ".join(notes),
                )
                self.store.update_capture_run(final)

            return CaptureResult(
                run_id=self.run_id,
                final_capture_status=status,
                environment=environment,
                account_fingerprint=metadata.account_fingerprint,
                account_epoch_id=epoch_id,
                epoch_verification_status=epoch_status,
                source_epoch_claim=source_epoch_claim,
                pagination_complete=pagination_complete,
                source_page_count=page_count,
                received_row_count=received,
                inserted_row_count=inserted,
                duplicate_row_count=duplicates,
                unknown_activity_count=unknown_count,
                error_count=errors,
                warning_count=len(set(warning_codes)),
                warning_codes=sorted(set(warning_codes)),
                future_timestamp_flag_count=future_flags,
                source_payload_hash=manifest_hash,
                source_semantic_policy_id=policy.policy_id if policy else None,
                notes=notes,
            )
        except StoreConflictError:
            self._record_failed_manifest(
                provenance=provenance,
                environment=environment,
                account_fingerprint=metadata.account_fingerprint,
                policy=policy,
                capture_start=capture_start,
                requested_coverage_start=requested_coverage_start,
                requested_coverage_end=requested_coverage_end,
                page_count=page_count,
                received=received,
                unknown_count=unknown_count,
                future_flags=future_flags,
                warning_codes=warning_codes,
                fail_code="STORE_CONFLICT",
            )
            raise CaptureError("STORE_CONFLICT", coordinate=f"run:{self.run_id}")
        except CaptureCredentialError as exc:
            self._record_failed_manifest(
                provenance=provenance,
                environment=environment,
                account_fingerprint=metadata.account_fingerprint,
                policy=policy,
                capture_start=capture_start,
                requested_coverage_start=requested_coverage_start,
                requested_coverage_end=requested_coverage_end,
                page_count=page_count,
                received=received,
                unknown_count=unknown_count,
                future_flags=future_flags,
                warning_codes=warning_codes + ["CREDENTIAL_MATERIAL_REJECTED"],
                fail_code="CREDENTIAL_MATERIAL_REJECTED",
            )
            raise CaptureError(exc.code, coordinate=exc.coordinate) from None
        except CaptureValidationError as exc:
            self._record_failed_manifest(
                provenance=provenance,
                environment=environment,
                account_fingerprint=metadata.account_fingerprint,
                policy=policy,
                capture_start=capture_start,
                requested_coverage_start=requested_coverage_start,
                requested_coverage_end=requested_coverage_end,
                page_count=page_count,
                received=received,
                unknown_count=unknown_count,
                future_flags=future_flags,
                warning_codes=warning_codes,
                fail_code=exc.code,
            )
            raise CaptureError(exc.code, coordinate=exc.coordinate) from None
        except StoreValidationError as exc:
            fail_code = str(exc) if str(exc) == PAGE_INDEX_INVALID else "STORE_VALIDATION"
            self._record_failed_manifest(
                provenance=provenance,
                environment=environment,
                account_fingerprint=metadata.account_fingerprint,
                policy=policy,
                capture_start=capture_start,
                requested_coverage_start=requested_coverage_start,
                requested_coverage_end=requested_coverage_end,
                page_count=page_count,
                received=received,
                unknown_count=unknown_count,
                future_flags=future_flags,
                warning_codes=warning_codes,
                fail_code=fail_code,
            )
            raise CaptureError(fail_code, coordinate=f"run:{self.run_id}") from None
        except CaptureError:
            raise
        except Exception:
            logger.error(
                "CAPTURE_UNEXPECTED_FAILURE code=%s run=%s",
                "UNEXPECTED",
                self.run_id,
                exc_info=False,
            )
            raise CaptureError("UNEXPECTED_CAPTURE_FAILURE", coordinate=self.run_id)

    def _record_failed_manifest(
        self,
        *,
        provenance: CaptureProvenance,
        environment: str,
        account_fingerprint: str,
        policy: Optional[SourceSemanticPolicy],
        capture_start: str,
        requested_coverage_start: Optional[str],
        requested_coverage_end: Optional[str],
        page_count: int,
        received: int,
        unknown_count: int,
        future_flags: int,
        warning_codes: List[str],
        fail_code: str,
    ) -> None:
        completion = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        codes = sorted(set(warning_codes + [fail_code]))
        try:
            with self.store.transaction():
                existing = self.store.get_capture_run(self.run_id)
                failed = CaptureRunRecord(
                    run_id=self.run_id,
                    schema_version=SCHEMA_VERSION,
                    source=provenance.source,
                    environment=environment,
                    account_fingerprint=account_fingerprint,
                    requested_coverage_start=requested_coverage_start,
                    requested_coverage_end=requested_coverage_end,
                    capture_start_timestamp=capture_start,
                    capture_completion_timestamp=completion,
                    pagination_complete=False,
                    source_page_count=page_count,
                    received_row_count=received,
                    inserted_row_count=0,
                    duplicate_row_count=0,
                    unknown_activity_count=unknown_count,
                    error_count=1,
                    warning_count=len(codes),
                    warning_codes=",".join(codes),
                    final_capture_status="CAPTURE_FAILED",
                    source_payload_hash=None,
                    source_semantic_policy_id=(policy.policy_id if policy else None),
                    future_timestamp_flag_count=future_flags,
                    notes=fail_code,
                )
                if existing is None:
                    self.store.insert_capture_run(failed)
                else:
                    self.store.update_capture_run(failed)
        except Exception:
            logger.error(
                "CAPTURE_FAILED_MANIFEST_WRITE_ERROR code=%s run=%s",
                "FAILED_MANIFEST_WRITE",
                self.run_id,
                exc_info=False,
            )

    def _resolve_policy(self, adapter_source: str) -> Optional[SourceSemanticPolicy]:
        if self.policy is None:
            return None
        known = KNOWN_POLICIES.get(self.policy.policy_id)
        if known is None:
            raise CaptureValidationError(
                "UNKNOWN_POLICY",
                coordinate=self.policy.policy_id,
            )
        if known.source != adapter_source or self.policy.source != adapter_source:
            raise CaptureValidationError(
                "POLICY_SOURCE_MISMATCH",
                coordinate=f"policy:{self.policy.policy_id}",
            )
        # Always use the code-controlled policy instance (ignore forged maps).
        return known

    def _validate_environment(self, environment: Any) -> str:
        if environment is None or str(environment).strip() == "":
            raise CaptureEnvironmentError("ENVIRONMENT_REQUIRED", coordinate="environment")
        env = str(environment).strip().lower()
        if env in REFUSED_ENVIRONMENTS:
            raise CaptureEnvironmentError("LIVE_REFUSED", coordinate="environment")
        if env not in PERMITTED_ENVIRONMENTS:
            raise CaptureEnvironmentError("UNSUPPORTED_ENVIRONMENT", coordinate="environment")
        return env

    def _ingest_activity_stream(
        self,
        *,
        policy: Optional[SourceSemanticPolicy],
        environment: str,
        epoch_id: str,
        warning_codes: List[str],
        observe_times: List[str],
        manifest_chunks: List[Dict[str, Any]],
        receive_seq_start: int,
    ) -> Dict[str, Any]:
        unique_pages = 0
        received = 0
        inserted = 0
        duplicates = 0
        unknown = 0
        future = 0
        receive_seq = receive_seq_start
        expected_index = 0
        stream_complete = True
        seen_any = False
        last_has_more = False

        for page in self.adapter.iter_activity_pages():
            seen_any = True
            receive_seq += 1
            if (
                not isinstance(page.page_index, int)
                or isinstance(page.page_index, bool)
                or page.page_index < 0
            ):
                raise CaptureValidationError(PAGE_INDEX_INVALID, coordinate="activities:page")
            coord = f"activities:page:{page.page_index}"
            assert_no_credential_material(page.activities, coordinate=coord)
            if page.page_index != expected_index:
                stream_complete = False
                if "PAGE_SEQUENCE_GAP" not in warning_codes:
                    warning_codes.append("PAGE_SEQUENCE_GAP")
                    self.store.insert_warning(
                        CaptureWarningRecord(
                            capture_run_id=self.run_id,
                            warning_code="PAGE_SEQUENCE_GAP",
                            evidence_stream="activities",
                            page_index=page.page_index,
                            row_index=None,
                            detail_code=f"EXPECTED_{expected_index}",
                        )
                    )
                # Continue fail-closed only if conflict; gap => incomplete.
            page_token_hash = hash_page_token(page.page_token)
            page_body = {
                "page_index": page.page_index,
                "has_more": page.has_more,
                "page_token_hash": page_token_hash,
                "rows": [dict(r) for r in page.activities],
            }
            canonical_page_hash = payload_sha256(page_body)
            page_outcome = self.store.insert_page_manifest(
                PageManifestRecord(
                    capture_run_id=self.run_id,
                    evidence_stream="activities",
                    page_index=page.page_index,
                    coordinator_receive_sequence=receive_seq,
                    page_token_hash=page_token_hash,
                    canonical_page_hash=canonical_page_hash,
                    row_count=len(page.activities),
                    has_more=page.has_more,
                    inserted_timestamp=(
                        self.capture_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                    ),
                )
            )
            if page_outcome == "duplicate":
                # Exact duplicate page: do not inflate counts or re-ingest.
                expected_index = page.page_index + 1
                last_has_more = page.has_more
                continue

            unique_pages += 1
            manifest_chunks.append(
                {
                    "kind": "activity_page",
                    "page_index": page.page_index,
                    "row_count": len(page.activities),
                    "has_more": page.has_more,
                    "page_token_hash": page_token_hash,
                    "canonical_page_hash": canonical_page_hash,
                }
            )
            for row_index, raw in enumerate(page.activities):
                row_coord = f"{coord}:row:{row_index}"
                assert_no_credential_material(raw, coordinate=row_coord)
                received += 1
                record, is_unknown, fut = self._normalize_activity(
                    raw,
                    environment=environment,
                    account_epoch_id=epoch_id,
                    policy=policy,
                    page_index=page.page_index,
                    row_index=row_index,
                    page_token_hash=page_token_hash,
                    canonical_page_hash=canonical_page_hash,
                )
                if is_unknown:
                    unknown += 1
                    if "UNKNOWN_ACTIVITY_MAPPING" not in warning_codes:
                        warning_codes.append("UNKNOWN_ACTIVITY_MAPPING")
                    self.store.insert_warning(
                        CaptureWarningRecord(
                            capture_run_id=self.run_id,
                            warning_code="UNKNOWN_ACTIVITY_MAPPING",
                            evidence_stream="activities",
                            page_index=page.page_index,
                            row_index=row_index,
                            detail_code=record.activity_type,
                        )
                    )
                if fut:
                    future += 1
                    if "FUTURE_SOURCE_TIMESTAMP" not in warning_codes:
                        warning_codes.append("FUTURE_SOURCE_TIMESTAMP")
                        self.store.insert_warning(
                            CaptureWarningRecord(
                                capture_run_id=self.run_id,
                                warning_code="FUTURE_SOURCE_TIMESTAMP",
                                evidence_stream="activities",
                                page_index=page.page_index,
                                row_index=row_index,
                                detail_code="FUTURE_TS",
                            )
                        )
                observe_times.append(
                    self.capture_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                )
                outcome = self.store.insert_broker_activity(record)
                if outcome == "inserted":
                    inserted += 1
                else:
                    duplicates += 1
            expected_index = page.page_index + 1
            last_has_more = page.has_more

        if seen_any and last_has_more:
            stream_complete = False
            if "INCOMPLETE_PAGINATION" not in warning_codes:
                warning_codes.append("INCOMPLETE_PAGINATION")
                self.store.insert_warning(
                    CaptureWarningRecord(
                        capture_run_id=self.run_id,
                        warning_code="INCOMPLETE_PAGINATION",
                        evidence_stream="activities",
                        page_index=expected_index - 1 if expected_index else 0,
                        row_index=None,
                        detail_code="HAS_MORE_TRUE",
                    )
                )
        # Empty activity stream is allowed if equity pages exist.
        return {
            "unique_pages": unique_pages,
            "received": received,
            "inserted": inserted,
            "duplicates": duplicates,
            "unknown": unknown,
            "future": future,
            "receive_seq": receive_seq,
            "stream_complete": stream_complete,
        }

    def _ingest_equity_stream(
        self,
        *,
        environment: str,
        epoch_id: str,
        warning_codes: List[str],
        observe_times: List[str],
        manifest_chunks: List[Dict[str, Any]],
        receive_seq_start: int,
    ) -> Dict[str, Any]:
        unique_pages = 0
        received = 0
        inserted = 0
        duplicates = 0
        future = 0
        receive_seq = receive_seq_start
        expected_index = 0
        stream_complete = True
        seen_any = False
        last_has_more = False

        for page in self.adapter.iter_equity_observation_pages():
            seen_any = True
            receive_seq += 1
            if (
                not isinstance(page.page_index, int)
                or isinstance(page.page_index, bool)
                or page.page_index < 0
            ):
                raise CaptureValidationError(PAGE_INDEX_INVALID, coordinate="equity:page")
            coord = f"equity:page:{page.page_index}"
            assert_no_credential_material(page.observations, coordinate=coord)
            if page.page_index != expected_index:
                stream_complete = False
                if "PAGE_SEQUENCE_GAP" not in warning_codes:
                    warning_codes.append("PAGE_SEQUENCE_GAP")
                    self.store.insert_warning(
                        CaptureWarningRecord(
                            capture_run_id=self.run_id,
                            warning_code="PAGE_SEQUENCE_GAP",
                            evidence_stream="equity_observations",
                            page_index=page.page_index,
                            row_index=None,
                            detail_code=f"EXPECTED_{expected_index}",
                        )
                    )
            page_token_hash = hash_page_token(page.page_token)
            page_body = {
                "page_index": page.page_index,
                "has_more": page.has_more,
                "page_token_hash": page_token_hash,
                "rows": [dict(r) for r in page.observations],
            }
            canonical_page_hash = payload_sha256(page_body)
            page_outcome = self.store.insert_page_manifest(
                PageManifestRecord(
                    capture_run_id=self.run_id,
                    evidence_stream="equity_observations",
                    page_index=page.page_index,
                    coordinator_receive_sequence=receive_seq,
                    page_token_hash=page_token_hash,
                    canonical_page_hash=canonical_page_hash,
                    row_count=len(page.observations),
                    has_more=page.has_more,
                    inserted_timestamp=(
                        self.capture_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                    ),
                )
            )
            if page_outcome == "duplicate":
                expected_index = page.page_index + 1
                last_has_more = page.has_more
                continue

            unique_pages += 1
            manifest_chunks.append(
                {
                    "kind": "equity_page",
                    "page_index": page.page_index,
                    "row_count": len(page.observations),
                    "has_more": page.has_more,
                    "page_token_hash": page_token_hash,
                    "canonical_page_hash": canonical_page_hash,
                }
            )
            for row_index, raw in enumerate(page.observations):
                row_coord = f"{coord}:row:{row_index}"
                assert_no_credential_material(raw, coordinate=row_coord)
                received += 1
                record, fut, downgraded = self._normalize_equity(
                    raw,
                    environment=environment,
                    account_epoch_id=epoch_id,
                    page_index=page.page_index,
                    row_index=row_index,
                    page_token_hash=page_token_hash,
                    canonical_page_hash=canonical_page_hash,
                )
                if downgraded:
                    if "UNTRUSTED_CERTIFIED_BOUNDARY_CLAIM" not in warning_codes:
                        warning_codes.append("UNTRUSTED_CERTIFIED_BOUNDARY_CLAIM")
                    self.store.insert_warning(
                        CaptureWarningRecord(
                            capture_run_id=self.run_id,
                            warning_code="UNTRUSTED_CERTIFIED_BOUNDARY_CLAIM",
                            evidence_stream="equity_observations",
                            page_index=page.page_index,
                            row_index=row_index,
                            detail_code="DOWNGRADED_TO_BOUNDARY_UNVERIFIED",
                        )
                    )
                if fut:
                    future += 1
                    if "FUTURE_SOURCE_TIMESTAMP" not in warning_codes:
                        warning_codes.append("FUTURE_SOURCE_TIMESTAMP")
                        self.store.insert_warning(
                            CaptureWarningRecord(
                                capture_run_id=self.run_id,
                                warning_code="FUTURE_SOURCE_TIMESTAMP",
                                evidence_stream="equity_observations",
                                page_index=page.page_index,
                                row_index=row_index,
                                detail_code="FUTURE_TS",
                            )
                        )
                observe_times.append(
                    self.capture_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                )
                outcome = self.store.insert_equity_observation(record)
                if outcome == "inserted":
                    inserted += 1
                else:
                    duplicates += 1
            expected_index = page.page_index + 1
            last_has_more = page.has_more

        if seen_any and last_has_more:
            stream_complete = False
            if "INCOMPLETE_PAGINATION" not in warning_codes:
                warning_codes.append("INCOMPLETE_PAGINATION")
                self.store.insert_warning(
                    CaptureWarningRecord(
                        capture_run_id=self.run_id,
                        warning_code="INCOMPLETE_PAGINATION",
                        evidence_stream="equity_observations",
                        page_index=expected_index - 1 if expected_index else 0,
                        row_index=None,
                        detail_code="HAS_MORE_TRUE",
                    )
                )
        return {
            "unique_pages": unique_pages,
            "received": received,
            "inserted": inserted,
            "duplicates": duplicates,
            "future": future,
            "receive_seq": receive_seq,
            "stream_complete": stream_complete,
        }

    def _normalize_activity(
        self,
        raw: Mapping[str, Any],
        *,
        environment: str,
        account_epoch_id: str,
        policy: Optional[SourceSemanticPolicy],
        page_index: int,
        row_index: int,
        page_token_hash: Optional[str],
        canonical_page_hash: str,
    ) -> Tuple[BrokerActivityRecord, bool, bool]:
        # Payloads must not select classification policy.
        if any(
            k in raw
            for k in (
                "source_semantic_policy",
                "policy_id",
                "classification_policy",
                "governed_activity_category",
            )
        ):
            raise CaptureValidationError(
                "PAYLOAD_POLICY_ASSERTION_FORBIDDEN",
                coordinate=f"activities:page:{page_index}:row:{row_index}",
            )
        payload = dict(raw)
        activity_type = str(payload.get("activity_type") or payload.get("type") or "").strip()
        if not activity_type:
            raise CaptureValidationError(
                "ACTIVITY_TYPE_REQUIRED",
                coordinate=f"activities:page:{page_index}:row:{row_index}",
            )
        if policy is None:
            category = "UNKNOWN"
        else:
            category = policy.classify(activity_type)
        is_unknown = category == "UNKNOWN"

        amount = require_decimal(payload.get("amount"), field_name="amount")
        net_amount = require_decimal(payload.get("net_amount"), field_name="net_amount")
        quantity = require_decimal(
            payload.get("qty") or payload.get("quantity"), field_name="quantity"
        )
        price = require_decimal(payload.get("price"), field_name="price")
        per_share = require_decimal(payload.get("per_share_amount"), field_name="per_share_amount")
        currency = require_currency_if_amounts(
            currency=payload.get("currency"),
            amounts=[amount, net_amount, price, per_share],
        )

        effective = None
        fut_e = False
        if payload.get("effective_timestamp") not in (None, ""):
            effective, fut_e = normalize_aware_timestamp_to_utc(
                payload.get("effective_timestamp"),
                field_name="effective_timestamp",
                capture_now=self.capture_now,
            )
        elif payload.get("date") not in (None, ""):
            date_val = payload.get("date")
            if isinstance(date_val, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_val.strip()):
                parse_session_date_only(date_val, field_name="date")
                effective = None
                fut_e = False
            else:
                effective, fut_e = normalize_aware_timestamp_to_utc(
                    date_val,
                    field_name="effective_timestamp",
                    capture_now=self.capture_now,
                )

        transaction, fut_t = normalize_aware_timestamp_to_utc(
            payload.get("transaction_timestamp") or payload.get("created_at"),
            field_name="transaction_timestamp",
            capture_now=self.capture_now,
        )
        future_flag = bool(fut_e or fut_t)

        broker_id = (
            payload.get("id") or payload.get("activity_id") or payload.get("broker_activity_id")
        )
        broker_activity_id = str(broker_id).strip() if broker_id not in (None, "") else None
        derived = False
        if broker_activity_id:
            evidence_key = (
                f"broker_activity:{environment}:{account_epoch_id}:" f"{broker_activity_id}"
            )
        else:
            evidence_key = derive_evidence_key(kind="activity", payload=payload)
            derived = True

        record = BrokerActivityRecord(
            environment=environment,
            account_epoch_id=account_epoch_id,
            broker_activity_id=broker_activity_id,
            evidence_key=evidence_key,
            activity_type=activity_type.upper(),
            governed_activity_category=category,
            amount=amount,
            net_amount=net_amount,
            currency=currency,
            symbol=(str(payload["symbol"]).upper() if payload.get("symbol") else None),
            side=(str(payload["side"]).lower() if payload.get("side") else None),
            quantity=quantity,
            price=price,
            per_share_amount=per_share,
            effective_timestamp=effective,
            transaction_timestamp=transaction,
            description=(
                str(payload["description"]) if payload.get("description") is not None else None
            ),
            capture_run_id=self.run_id,
            evidence_stream="activities",
            source_page_index=page_index,
            source_row_index=row_index,
            source_page_token_hash=page_token_hash,
            canonical_source_page_hash=canonical_page_hash,
            raw_source_payload=payload,
            raw_source_payload_sha256=payload_sha256(payload),
            evidence_key_is_derived=derived,
            future_timestamp_flag=future_flag,
        )
        return record, is_unknown, future_flag

    def _normalize_equity(
        self,
        raw: Mapping[str, Any],
        *,
        environment: str,
        account_epoch_id: str,
        page_index: int,
        row_index: int,
        page_token_hash: Optional[str],
        canonical_page_hash: str,
    ) -> Tuple[EquityObservationRecord, bool, bool]:
        payload = dict(raw)
        source_type = str(
            payload.get("source_observation_type")
            or payload.get("observation_type")
            or "UNSPECIFIED"
        ).strip()

        requested = str(payload.get("boundary_classification") or "").strip().upper()
        downgraded = False
        boundary_authority_id = None
        if requested in CERTIFIED_BOUNDARIES:
            # Raw/adapter cannot certify. Preserve claim in raw payload only.
            boundary = "BOUNDARY_UNVERIFIED"
            downgraded = True
        elif requested in {
            "INSTANTANEOUS_ACCOUNT_OBSERVATION",
            "BROKER_PORTFOLIO_HISTORY_OBSERVATION",
            "RECORDED_DATE_ONLY_OBSERVATION",
            "BOUNDARY_UNVERIFIED",
        }:
            boundary = requested
        else:
            if payload.get("session_date") and not payload.get("observation_timestamp"):
                boundary = "RECORDED_DATE_ONLY_OBSERVATION"
            elif source_type.upper() in {"ACCOUNT_SNAPSHOT", "CURRENT_ACCOUNT"}:
                boundary = "INSTANTANEOUS_ACCOUNT_OBSERVATION"
            elif source_type.upper() in {
                "PORTFOLIO_HISTORY",
                "BROKER_PORTFOLIO_HISTORY",
            }:
                boundary = "BROKER_PORTFOLIO_HISTORY_OBSERVATION"
            else:
                boundary = "BOUNDARY_UNVERIFIED"

        session_date = None
        if payload.get("session_date") or payload.get("date"):
            session_date = parse_session_date_only(
                payload.get("session_date") or payload.get("date"),
                field_name="session_date",
            )

        obs_ts = None
        tz_name = None
        future_flag = False
        if payload.get("observation_timestamp"):
            obs_ts, future_flag = normalize_aware_timestamp_to_utc(
                payload.get("observation_timestamp"),
                field_name="observation_timestamp",
                capture_now=self.capture_now,
            )
            tz_name = "UTC"

        cash = require_decimal(payload.get("cash"), field_name="cash")
        market_value = require_decimal(payload.get("market_value"), field_name="market_value")
        equity = require_decimal(
            (
                payload.get("equity")
                if payload.get("equity") is not None
                else payload.get("total_value")
            ),
            field_name="equity",
        )
        currency = require_currency_if_amounts(
            currency=payload.get("currency"),
            amounts=[cash, market_value, equity],
        )

        source_id_raw = payload.get("id") or payload.get("source_observation_id")
        source_observation_id = (
            str(source_id_raw).strip() if source_id_raw not in (None, "") else None
        )
        derived = False
        if source_observation_id:
            evidence_key = f"equity:{environment}:{account_epoch_id}:{source_observation_id}"
        else:
            evidence_key = derive_evidence_key(kind="equity", payload=payload)
            derived = True

        record = EquityObservationRecord(
            environment=environment,
            account_epoch_id=account_epoch_id,
            source_observation_id=source_observation_id,
            evidence_key=evidence_key,
            observation_timestamp=obs_ts,
            timezone=tz_name,
            session_date=session_date,
            cash=cash,
            market_value=market_value,
            equity=equity,
            currency=currency,
            source_observation_type=source_type,
            boundary_classification=boundary,
            capture_run_id=self.run_id,
            evidence_stream="equity_observations",
            source_page_index=page_index,
            source_row_index=row_index,
            source_page_token_hash=page_token_hash,
            canonical_source_page_hash=canonical_page_hash,
            raw_source_payload=payload,
            raw_source_payload_sha256=payload_sha256(payload),
            evidence_key_is_derived=derived,
            boundary_authority_id=boundary_authority_id,
            future_timestamp_flag=future_flag,
        )
        return record, future_flag, downgraded


@dataclass
class FixtureSourceAdapter:
    metadata: AccountMetadata
    activity_pages: Sequence[ActivityPage]
    equity_pages: Sequence[EquityObservationPage]
    provenance: CaptureProvenance

    def get_account_metadata(self) -> AccountMetadata:
        return self.metadata

    def iter_activity_pages(self) -> Iterator[ActivityPage]:
        yield from self.activity_pages

    def iter_equity_observation_pages(self) -> Iterator[EquityObservationPage]:
        yield from self.equity_pages

    def get_capture_provenance(self) -> CaptureProvenance:
        return self.provenance
