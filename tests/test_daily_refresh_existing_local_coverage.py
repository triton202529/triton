"""Existing local raw coverage. Temporary fixtures only."""

from __future__ import annotations

import importlib.util
import json
from datetime import timedelta
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "sf50_already_current_helpers",
    Path(__file__).with_name("test_phase_198t_sf50_already_current_raw.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
sf50 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sf50)

SESSION = sf50.SESSION
RUN = sf50.RUN


def _older_decision(symbol: str, **overrides) -> dict:
    decision = {
        "symbol": symbol,
        "action": "keep",
        "reason": "provider_older_than_local",
        "local_max": SESSION.isoformat(),
        "provider_max": (SESSION - timedelta(days=1)).isoformat(),
        "operationally_eligible": True,
        "problems": [],
    }
    decision.update(overrides)
    return decision


def _write_fetch(root: Path, symbols: list[str], decisions: list[dict], **overrides) -> None:
    tradables = [symbol for symbol in symbols if not symbol.startswith("^")]
    body = {
        "target_session": SESSION.isoformat(),
        "run_id": RUN,
        "passed": True,
        "promoted": [],
        "tradable_coverage": {
            "covered": len(tradables),
            "denominator": len(tradables),
            "ratio": 1.0,
            "minimum": 0.9,
        },
        "decisions": decisions,
    }
    body.update(overrides)
    (root / "data/results/daily_refresh_fetch_receipt.json").write_text(
        json.dumps(body), encoding="utf-8"
    )


def _run(root: Path):
    from services.daily_prediction_refresh_validate import validate_refresh

    started, baseline = sf50._prepare(root)
    return validate_refresh(
        repo_root=root,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=sf50._binding(root, baseline),
    )


def _validate_without_writes(root: Path):
    from services.daily_prediction_refresh_validate import validate_refresh

    started, baseline = sf50._prepare(root)
    watched = [root / "data/results/daily_refresh_fetch_receipt.json"]
    watched.extend(sorted((root / "data/raw").glob("*.csv")))
    before = {path: path.read_bytes() for path in watched if path.is_file()}
    result = validate_refresh(
        repo_root=root,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=sf50._binding(root, baseline),
    )
    after = {path: path.read_bytes() for path in before}
    assert after == before
    return result


def test_provider_older_than_local_counts_without_rewrite(tmp_path: Path) -> None:
    symbols = ["PLTR", "^VIX"]
    root = sf50._layout(tmp_path, symbols)
    _write_fetch(
        root,
        symbols,
        [
            _older_decision("PLTR"),
            {
                "symbol": "^VIX",
                "action": "keep",
                "reason": "local_already_current",
                "local_max": SESSION.isoformat(),
                "operationally_eligible": True,
            },
        ],
    )
    result = _validate_without_writes(root)
    assert result["passed"] is True
    assert result["tradable_coverage"]["covered"] == 1
    assert result["tradable_coverage"]["denominator"] == 1
    assert "PLTR" in result["raw"]["ok_symbols"]
    assert result["raw"]["accepted_existing_local_evidence"] == ["PLTR"]
    assert "PLTR" not in result["raw"]["already_current"]["accepted"]
    assert "^VIX" in result["raw"]["already_current"]["accepted"]
    evidence = result["raw"]["advancement"]["PLTR"]
    assert evidence["advanced"] is False
    assert evidence["accepted_existing_local_evidence"] is True
    assert "current_without_rewrite" not in evidence
    assert result["raw"]["advancement"]["^VIX"]["current_without_rewrite"] is True
    assert result["raw"]["advancement"]["^VIX"]["advanced"] is False
    assert "^VIX" not in result["raw"]["accepted_existing_local_evidence"]
    assert result.get("stage_receipts", [{}])[0].get("promoted") in (None, [])


def test_legacy_local_already_current_is_unchanged(tmp_path: Path) -> None:
    root = sf50._layout(tmp_path, ["PLTR"])
    sf50._fetch(root, ["PLTR"])
    result = _validate_without_writes(root)
    assert result["passed"] is True
    assert result["raw"]["already_current"]["accepted"] == ["PLTR"]
    assert result["raw"]["accepted_existing_local_evidence"] == []
    assert result["raw"]["advancement"]["PLTR"]["current_without_rewrite"] is True
    assert result["raw"]["advancement"]["PLTR"]["advanced"] is False


def test_stale_newer_conflict_and_hash_fail_closed(tmp_path: Path) -> None:
    root = sf50._layout(tmp_path, ["PLTR"])
    _write_fetch(
        root,
        ["PLTR"],
        [_older_decision("PLTR", local_max=(SESSION - timedelta(days=1)).isoformat())],
    )
    sf50._raw(root / "data/raw/PLTR.csv", "PLTR", session_rows=0)
    stale = _run(root)
    assert stale["passed"] is False
    assert "PLTR" in stale["raw"]["stale_symbols"]
    assert stale["raw"]["accepted_existing_local_evidence"] == []
    assert stale["tradable_coverage"]["covered"] == 0

    root = sf50._layout(tmp_path / "newer", ["PLTR"])
    _write_fetch(
        root,
        ["PLTR"],
        [_older_decision("PLTR", provider_max=(SESSION + timedelta(days=1)).isoformat())],
    )
    newer = _run(root)
    assert newer["raw"]["already_current"]["rejected"]["PLTR"] == "provider_newer_than_local"
    assert "PLTR" not in newer["raw"]["ok_symbols"]

    root = sf50._layout(tmp_path / "conflict", ["PLTR"])
    _write_fetch(
        root,
        ["PLTR"],
        [_older_decision("PLTR", overlap_conflicts=[{"date": SESSION.isoformat()}])],
    )
    conflict = _run(root)
    assert conflict["raw"]["already_current"]["rejected"]["PLTR"] == "overlap_conflict"

    root = sf50._layout(tmp_path / "problems", ["PLTR"])
    _write_fetch(root, ["PLTR"], [_older_decision("PLTR", problems=["partial_bar"])])
    problems = _run(root)
    assert problems["raw"]["already_current"]["rejected"]["PLTR"] == "fetch_problems"

    root = sf50._layout(tmp_path / "ineligible", ["PLTR"])
    _write_fetch(root, ["PLTR"], [_older_decision("PLTR", operationally_eligible=False)])
    ineligible = _run(root)
    assert ineligible["raw"]["already_current"]["rejected"]["PLTR"] == "not_operationally_eligible"

    root = sf50._layout(tmp_path / "missing", ["PLTR"])
    _write_fetch(root, ["PLTR"], [])
    missing = _run(root)
    assert missing["raw"]["already_current"]["rejected"]["PLTR"] == "missing_fetch_decision"

    root = sf50._layout(tmp_path / "hash", ["PLTR"])
    _write_fetch(root, ["PLTR"], [_older_decision("PLTR", raw_sha256="ab" * 32)])
    forged = _run(root)
    assert forged["raw"]["already_current"]["rejected"]["PLTR"] == "forged_raw_hash"
    assert (root / "data/raw/PLTR.csv").is_file()


def test_partial_coverage_fails(tmp_path: Path) -> None:
    symbols = ["PLTR", "NVDA"]
    root = sf50._layout(tmp_path, symbols)
    _write_fetch(
        root,
        symbols,
        [
            _older_decision("PLTR"),
            _older_decision("NVDA", local_max=(SESSION - timedelta(days=1)).isoformat()),
        ],
    )
    sf50._raw(root / "data/raw/NVDA.csv", "NVDA", session_rows=0)
    result = _run(root)
    assert result["passed"] is False
    assert result["first_failure"] == "coverage_below_minimum"
    assert result["tradable_coverage"]["covered"] == 1
    assert result["tradable_coverage"]["denominator"] == 2
    assert result["raw"]["accepted_existing_local_evidence"] == ["PLTR"]
    assert "NVDA" in result["raw"]["stale_symbols"]


def test_validation_does_not_rewrite_raw_or_receipt(tmp_path: Path) -> None:
    root = sf50._layout(tmp_path, ["PLTR", "NVDA"])
    _write_fetch(root, ["PLTR", "NVDA"], [_older_decision("PLTR"), _older_decision("NVDA")])
    result = _validate_without_writes(root)
    assert result["passed"] is True
    assert set(result["raw"]["accepted_existing_local_evidence"]) == {"NVDA", "PLTR"}
