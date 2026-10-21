"""Resolve one DailyRefresh run's stage receipts.

Completed stages are published as immutable generations named
``daily_prediction_refresh_stage_receipts_<run_id>.gen.<n>.json``. The legacy
aggregate is used only when that run has no generation file. A malformed,
unreadable, or ambiguous newest generation fails closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

RECEIPT_PREFIX = "daily_prediction_refresh_stage_receipts_"
DETAIL_AMBIGUOUS = "stage_receipt_generation_ambiguous"
DETAIL_MALFORMED = "stage_receipt_generation_malformed"
DETAIL_MISSING = "stage_receipt_generation_missing"
DETAIL_ABSENT = "stage_receipt_missing"


@dataclass(frozen=True)
class StageReceiptResolution:
    status: str
    detail: str
    path: Optional[Path] = None
    document: Optional[list] = None
    generation: Optional[int] = None


def legacy_receipt_name(run_id: str) -> str:
    return f"{RECEIPT_PREFIX}{run_id}.json"


def generation_receipt_name(run_id: str, generation: int) -> str:
    return f"{RECEIPT_PREFIX}{run_id}.gen.{generation}.json"


def run_id_from_stage_receipt_filename(name: str) -> str:
    """Return the run id encoded in a legacy or generation receipt name."""
    if not name.startswith(RECEIPT_PREFIX) or not name.endswith(".json"):
        return ""
    body = name[len(RECEIPT_PREFIX) : -len(".json")]
    marker = ".gen."
    if marker in body:
        run_id, _, token = body.rpartition(marker)
        if not _canonical_generation(token):
            return ""
        body = run_id
    if not _safe_run_token(body) or ".gen." in body:
        return ""
    return body


def resolve_stage_receipts(results_dir: Path, run_id: str) -> StageReceiptResolution:
    """Select the receipt a completion check must use for ``run_id``."""
    if not _safe_run_token(run_id) or ".gen." in run_id:
        return StageReceiptResolution(status="ambiguous", detail=DETAIL_AMBIGUOUS)
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return StageReceiptResolution(status="missing", detail=DETAIL_ABSENT)

    prefix = f"{RECEIPT_PREFIX}{run_id}.gen."
    numbered: list[tuple[int, Path]] = []
    ambiguous = False
    try:
        entries = list(results_dir.iterdir())
    except OSError:
        return StageReceiptResolution(status="missing", detail=DETAIL_MISSING)
    for path in entries:
        name = path.name
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        token = name[len(prefix) : -len(".json")]
        generation = _canonical_generation(token)
        if generation is None:
            ambiguous = True
            continue
        numbered.append((generation, path))
    if ambiguous:
        return StageReceiptResolution(status="ambiguous", detail=DETAIL_AMBIGUOUS)
    numbers = [generation for generation, _path in numbered]
    if len(numbers) != len(set(numbers)):
        return StageReceiptResolution(status="ambiguous", detail=DETAIL_AMBIGUOUS)
    if numbered:
        generation, path = max(numbered, key=lambda item: item[0])
        return _read_receipt(path, generation=generation, found_status="generation")

    legacy = results_dir / legacy_receipt_name(run_id)
    if not legacy.exists():
        return StageReceiptResolution(status="missing", detail=DETAIL_ABSENT)
    return _read_receipt(legacy, generation=None, found_status="legacy")


def _safe_run_token(run_id: str) -> bool:
    text = str(run_id or "")
    return bool(text) and text == Path(text).name and "/" not in text and "\\" not in text


def _canonical_generation(token: str) -> Optional[int]:
    if not token.isdecimal():
        return None
    if str(int(token)) != token:
        return None
    value = int(token)
    if value < 1:
        return None
    return value


def _read_receipt(
    path: Path, *, generation: Optional[int], found_status: str
) -> StageReceiptResolution:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError:
        detail = DETAIL_MISSING if generation is not None else DETAIL_ABSENT
        return StageReceiptResolution(
            status="missing",
            detail=detail,
            path=path,
            generation=generation,
        )
    except (UnicodeError, json.JSONDecodeError):
        return StageReceiptResolution(
            status="malformed",
            detail=DETAIL_MALFORMED,
            path=path,
            generation=generation,
        )
    if not isinstance(payload, list):
        return StageReceiptResolution(
            status="malformed",
            detail=DETAIL_MALFORMED,
            path=path,
            generation=generation,
        )
    return StageReceiptResolution(
        status=found_status,
        detail="",
        path=path,
        document=payload,
        generation=generation,
    )
