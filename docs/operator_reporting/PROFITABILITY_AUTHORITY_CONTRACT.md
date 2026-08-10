# Profitability Authority Contract (UIR-003B-HC)

Status: **trusted-authority registry correction on the offline capture foundation**
Highest capture status in this phase: `CAPTURE_COMPLETE_UNVERIFIED`

**No profitability metric becomes available merely because UIR-003B is implemented.**

---

## 1. UIR-003A findings (preserved)

- Latest open-position unrealized P/L was `+1696.591686 USD` at `2026-08-09T04:17:08Z` (snapshot only).
- `portfolio_history.csv` date-only observations are not certified session boundaries.
- The `+27836.29` observation delta is not proven profit.
- Local FIFO realized ≈ `-1451.268703` is not governed authority.
- True period profit, cumulative profit, realized P/L, and returns remain unavailable until later verified phases.

---

## 2. UIR-003B-R / UIR-003B-HR defects corrected

Hardening (UIR-003B-H) corrected global alias mappings, untrusted self-certification via payload fields, adapter-claimed epoch verification, fingerprint-collapsed epochs, missing provenance, and weak pagination.

UIR-003B-HC additionally corrects:

1. Store acceptance of any nonempty `boundary_authority_id` / `epoch_authority_id`.
2. Unused duplicate authority registries in the capture module.
3. Missing runtime proof against attacker-supplied authority identifiers.
4. Missing negative page-index fail-closed behavior.
5. Missing proof that distinct source IDs with identical financial fields remain separate evidence.

---

## 3. Source-semantic policy selection

Classification is performed only by a **code-controlled** `SourceSemanticPolicy` supplied to the coordinator.

- Raw payloads must not select, modify, or assert the policy.
- Adapter `CaptureProvenance.source` must match `policy.source`.
- Unknown / missing policies fail closed to category `UNKNOWN`.
- Policies do not leak across sources.
- The coordinator always applies the known registry instance for a policy ID (forged maps are ignored).

### Fixture-only registry: `FIXTURE_UIR003B_V1`

Source: `uir003b_fixtures` · Version: `1`
**Not** approved Alpaca production mappings. Real broker mappings require separate UIR-003C review.

| Broker activity_type | Governed category |
|---|---|
| FILL, PARTIAL_FILL | INTERNAL_TRADE_CASH_MOVEMENT |
| DIV, DIVIDEND, INT, INTEREST | INVESTMENT_INCOME |
| FEE, REG_FEE, COMMISSION | INVESTMENT_EXPENSE |
| CSD | EXTERNAL_CONTRIBUTION |
| CSW | EXTERNAL_WITHDRAWAL |
| ACATS | ASSET_TRANSFER |
| PAPER_RESET, ACCOUNT_RESET | PAPER_ACCOUNT_RESET |

Without a matching policy, and even under the fixture policy, these remain **`UNKNOWN`**:

- `TRANSFER`
- `DEPOSIT`
- `WITHDRAWAL`
- `TRADE`
- `BUY`
- `SELL`

Original activity type is always preserved separately from the governed category.

---

## 4. Store-owned boundary and epoch authority

The store is the **final authority-enforcement boundary**.

Authority requires **exact membership** in an immutable, code-controlled capability registry owned by `services/profitability_authority_store.py`:

- `PRODUCTION_BOUNDARY_AUTHORITY_REGISTRY`
- `PRODUCTION_EPOCH_AUTHORITY_REGISTRY`

Both production registries are **empty** in UIR-003B-HC. Therefore every certified boundary and every verified epoch is **unreachable** in this phase.

Rules:

- A nonempty authority ID is **not** sufficient.
- Exact identifier matching is required (no trim, normalize, partial match, or alias).
- Boundary capability entries (when introduced by a future reviewed phase) must authorize specific certified classifications.
- Epoch capability entries (when introduced by a future reviewed phase) must explicitly authorize `EPOCH_VERIFIED`.
- No caller, payload, adapter, coordinator option, environment variable, file, database row, or constructor argument may extend or replace the registries.
- Raw claims remain evidence, not authority.
- Future authority activation requires a separately reviewed governed change.
- UIR-003C does **not** automatically establish boundary or epoch authority.

### Boundary behavior in this phase

Certified classifications remain in the vocabulary for future governed use:

- `CERTIFIED_SESSION_OPEN`
- `CERTIFIED_SESSION_CLOSE`

- Raw fields such as `governed_boundary_semantic` have no authority.
- Untrusted certified claims are preserved only inside the raw payload, normalized to `BOUNDARY_UNVERIFIED`, and recorded as warning `UNTRUSTED_CERTIFIED_BOUNDARY_CLAIM`.
- Stored `boundary_authority_id` remains null for ordinary capture rows.
- Direct store insertion of a certified boundary fails closed unless the authority ID is an exact registry member with capability for that classification.
- Authority IDs supplied on non-certified classifications are rejected (no misleading provenance).
- Date-only observations receive no synthetic time.
- A timestamp alone cannot establish a certified exchange-session boundary.

### Epoch behavior in this phase

- Adapter epoch verification claims are evidence only (`source_epoch_verification_claim`).
- Adapter epoch references are stored as `source_epoch_claim`, never as the internal PK.
- Every current capture stores `EPOCH_UNVERIFIED` with null `epoch_authority_id`.
- Direct store `EPOCH_VERIFIED` fails closed unless the authority ID is an exact registry member with `EPOCH_VERIFIED` capability.
- Authority IDs supplied on `EPOCH_UNVERIFIED` are rejected.
- Internal epoch ID is per capture run: `epoch_run:{run_id}`.
- Same account fingerprint across captures does **not** merge epochs.
- Conflicting reset markers or account-created timestamps for the same governed epoch fail closed.

### Capture-time epoch observation semantics

`first_observed_timestamp` / `latest_observed_timestamp` mean when TRITON observed the epoch during capture (UTC capture observation times). They do **not** certify account continuity or market-session boundaries, and they do not replace financial effective/transaction timestamps.

---

## 5. Page-manifest schema

Table `page_manifests`:

- `capture_run_id`
- `evidence_stream` (`activities` | `equity_observations`)
- `page_index` (zero-based contiguous; negative indexes fail closed)
- `coordinator_receive_sequence`
- `page_token_hash` (raw tokens never stored)
- `canonical_page_hash`
- `row_count`
- `has_more`
- `inserted_timestamp`

Page-index convention: **zero-based contiguous integers starting at 0**.
Negative page indexes are invalid: no manifest, no rows, capture fails closed, ingestion transaction rolls back.

---

## 6. Row-level provenance schema

Every activity and equity row stores:

- `capture_run_id`
- `evidence_stream`
- `source_page_index`
- `source_row_index`
- `source_page_token_hash`
- `canonical_source_page_hash`
- `raw_source_payload`
- `raw_source_payload_sha256`

Distinct source identifiers remain separate evidence even when non-identity financial fields match.

---

## 7. Pagination validation

1. Contiguous page indexes required for completeness.
2. Negative page index → fail closed (`PAGE_INDEX_INVALID`); rollback; sanitized failed manifest only.
3. Gap → `CAPTURE_INCOMPLETE` + `PAGE_SEQUENCE_GAP` (unless fatal conflict → `CAPTURE_FAILED`).
4. Trailing `has_more=true` → `CAPTURE_INCOMPLETE` + `INCOMPLETE_PAGINATION`.
5. Exact duplicate page (same index/token + same canonical hash): no count inflation; no row re-ingest.
6. Same page identity with different content → fail closed.
7. Unique source pages determine manifest counts.

---

## 8. Capture warning codes

Machine-readable, sanitized (no payloads, tokens, credentials, account IDs, exception text):

- `UNTRUSTED_CERTIFIED_BOUNDARY_CLAIM`
- `UNTRUSTED_EPOCH_VERIFICATION_CLAIM`
- `UNKNOWN_ACTIVITY_MAPPING`
- `FUTURE_SOURCE_TIMESTAMP`
- `INCOMPLETE_PAGINATION`
- `PAGE_SEQUENCE_GAP`
- `CREDENTIAL_MATERIAL_REJECTED`

Authority rejection uses stable store codes such as `AUTHORITY_BOUNDARY_UNTRUSTED` and `AUTHORITY_EPOCH_UNTRUSTED` without embedding rejected identifiers.

---

## 9. Credential and redaction protections

Before hashing/persistence, payloads are scanned recursively for credential-like keys (case-insensitive normalized), including:

`authorization`, `api_key`, `api_secret`, `secret_key`, `client_secret`, `access_token`, `refresh_token`, `password`, `private_key`

On detection: fail closed; no persist; no secret hashing into exposed manifests; sanitized error code + page/row coordinate only; rollback. Logging uses stable codes with `exc_info=False`.

Account fingerprints are context-bound digests — never raw account IDs.

---

## 10. Transaction and rollback

A single transaction covers capture-run start, epoch, page manifests, rows, warnings, and counters. Fatal failure rolls all of these back. A separate sanitized `CAPTURE_FAILED` manifest may be written afterward. Earlier committed evidence is preserved. Conflicts never overwrite.

---

## 11. Schema version decision

Schema version remains **1**, corrected in place before first committed release:

- UIR-003B remained uncommitted.
- No production database was created.
- No migration artifact is required.

Monetary columns are TEXT (exact decimal strings), never SQLite `REAL`.
`PRAGMA foreign_keys=ON` on every connection.

Documented future path (not created by this phase):

`data/reporting_authority/profitability_authority.sqlite3`

---

## 12. Future UIR-003C responsibility

- Controlled paper broker adapter behind `SourceAdapter`
- Provider-scoped production activity mappings (reviewed)
- Optional governed boundary / epoch authorities only via separately reviewed store-registry activation (not automatic)
- Still no automatic profitability calculation

## 13. Reporting-layer read-only contract

Reporting may later consume verified records read-only. It must never call the broker for profitability, mutate the authority DB, invent activities, certify boundaries, or become the accounting authority.
