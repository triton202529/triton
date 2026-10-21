"""Immutable DailyRefresh stage receipts. Temporary paths only."""

from __future__ import annotations

import ctypes
import json
import subprocess
import threading
from ctypes import wintypes
from pathlib import Path

from services.daily_refresh_completion import (
    CLASS_COMPLETE,
    CLASS_FETCH_ONLY,
    REQUIRED_STAGES,
    evaluate_daily_refresh_completion,
)
from services.daily_refresh_stage_receipts import (
    generation_receipt_name,
    legacy_receipt_name,
    resolve_stage_receipts,
)
from services.operator_closeout_control import _daily_refresh_status

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_daily_refresh.ps1"
RUN_ID = "daily_prediction_refresh_20261021_120000"
SESSION = "2026-10-21"


def _ps(tmp: Path, body: str) -> subprocess.CompletedProcess[str]:
    driver = tmp / "driver.ps1"
    driver.write_text(body, encoding="utf-8")
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(driver)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _publish(tmp: Path, results: Path, stages: list[dict]) -> subprocess.CompletedProcess[str]:
    results.mkdir(parents=True, exist_ok=True)
    spec = tmp / "stages.json"
    spec.write_text(json.dumps(stages), encoding="utf-8")
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}'
$loaded = @(Get-Content -LiteralPath '{spec}' -Raw | ConvertFrom-Json)
$script:StageReceipts = @($loaded | Where-Object {{ $null -ne $_ }})
Write-StageReceipts -ResultsDirectory '{results}' -RunId '{RUN_ID}'
if ($script:LatestStageReceiptPath) {{ Write-Output $script:LatestStageReceiptPath }}
"""
    return _ps(tmp, body)


def _generation(results: Path, generation: int) -> Path:
    return results / generation_receipt_name(RUN_ID, generation)


def _stage(name: str, *, ok: bool = True) -> dict:
    return {
        "name": name,
        "run_id": RUN_ID,
        "target_session": SESSION,
        "exit_code": 0 if ok else 1,
        "outcome": "ok" if ok else "failed",
        "status": "COMPLETE" if ok else "FAILED",
    }


def _names(path: Path) -> list[str]:
    return [row["name"] for row in json.loads(path.read_text(encoding="utf-8"))]


class _Handle:
    def __init__(self, path: Path, share: int) -> None:
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._kernel.CreateFileW.restype = wintypes.HANDLE
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        handle = self._kernel.CreateFileW(str(path), 0x80000000, share, None, 3, 0x80, None)
        invalid = ctypes.c_void_p(-1).value
        if not handle or int(handle) == int(invalid or 0):
            raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
        self._handle = handle

    def close(self) -> None:
        if self._handle is not None:
            self._kernel.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "_Handle":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _install(
    root: Path,
    stages: list[dict],
    *,
    success_log: bool = True,
    fetch_passed: bool = False,
) -> Path:
    results = root / "data" / "results"
    results.mkdir(parents=True)
    (results / f"daily_prediction_refresh_baseline_{RUN_ID}.json").write_text(
        json.dumps({"run_id": RUN_ID}),
        encoding="utf-8",
    )
    (results / "daily_prediction_refresh_summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "target_session": SESSION}),
        encoding="utf-8",
    )
    if fetch_passed:
        (results / "daily_refresh_fetch_receipt.json").write_text(
            json.dumps({"target_session": SESSION, "passed": True}),
            encoding="utf-8",
        )
    log = root / "logs" / "daily_refresh_20261021_120000.log"
    log.parent.mkdir(parents=True)
    lines = [f"run_id={RUN_ID}"]
    if success_log:
        lines.append("=== TRITON Daily Prediction Refresh OK ===")
        lines.append("FinishedAt=2026-10-21T12:00:00 exit=0")
    else:
        lines.append("FinishedAt=2026-10-21T12:00:00 exit=1")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def _assert_not_complete(root: Path) -> dict:
    verdict = evaluate_daily_refresh_completion(root, SESSION)
    closeout = _daily_refresh_status(root, SESSION)
    assert verdict["ok"] is False
    assert verdict["classification"] != CLASS_COMPLETE
    assert closeout["ok"] is False
    assert closeout["classification"] != CLASS_COMPLETE
    return verdict


def test_writer_does_not_replace_the_aggregate_receipt() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "[System.IO.File]::Replace" not in text
    body = text.split("function Run-PyStep", 1)[1].split("\nif ($MyInvocation", 1)[0]
    assert body.index("STEP_FAILED:") < body.index("Write-StageReceipts")


def test_locked_legacy_destination_without_delete_sharing(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    legacy = results / legacy_receipt_name(RUN_ID)
    legacy.write_text('[{"name":"legacy"}]\n', encoding="utf-8")
    before = legacy.read_bytes()
    with _Handle(legacy, 0x1 | 0x2):
        proc = _publish(tmp_path, results, [_stage("fetch_raw_data")])
    assert proc.returncode == 0, proc.stdout
    assert legacy.read_bytes() == before
    assert _names(_generation(results, 1)) == ["fetch_raw_data"]
    assert list(results.glob("*.tmp")) == []


def test_concurrent_dashboard_style_reader(tmp_path: Path) -> None:
    results = tmp_path / "results"
    legacy = results / legacy_receipt_name(RUN_ID)
    results.mkdir()
    legacy.write_text("legacy\n", encoding="utf-8")
    before = legacy.read_bytes()
    errors: list[str] = []
    stop = threading.Event()

    def _read() -> None:
        while not stop.is_set():
            try:
                legacy.read_bytes()
                for path in results.glob(f"{legacy_receipt_name(RUN_ID)[:-5]}.gen.*.json"):
                    path.read_bytes()
            except Exception as exc:
                errors.append(repr(exc))
                return

    worker = threading.Thread(target=_read)
    worker.start()
    try:
        for count in range(1, 4):
            proc = _publish(tmp_path, results, [_stage(name) for name in REQUIRED_STAGES[:count]])
            assert proc.returncode == 0, proc.stdout
    finally:
        stop.set()
        worker.join(timeout=5)
    assert errors == []
    assert legacy.read_bytes() == before
    assert _names(_generation(results, 3)) == list(REQUIRED_STAGES[:3])


def test_first_generation_without_legacy_destination(tmp_path: Path) -> None:
    results = tmp_path / "results"
    proc = _publish(tmp_path, results, [_stage("fetch_raw_data")])
    assert proc.returncode == 0, proc.stdout
    assert not (results / legacy_receipt_name(RUN_ID)).exists()
    assert _names(_generation(results, 1)) == ["fetch_raw_data"]
    resolved = resolve_stage_receipts(results, RUN_ID)
    assert resolved.status == "generation"
    assert resolved.generation == 1


def test_repeated_cumulative_publication(tmp_path: Path) -> None:
    results = tmp_path / "results"
    first = _publish(tmp_path, results, [_stage("fetch_raw_data")])
    assert first.returncode == 0, first.stdout
    original = _generation(results, 1).read_bytes()
    occupied = _generation(results, 2)
    occupied.write_text('["occupied"]\n', encoding="utf-8")
    occupied_bytes = occupied.read_bytes()
    second = _publish(
        tmp_path,
        results,
        [_stage("fetch_raw_data"), _stage("stale_data_gate")],
    )
    assert second.returncode == 0, second.stdout
    assert _generation(results, 1).read_bytes() == original
    assert occupied.read_bytes() == occupied_bytes
    assert _names(_generation(results, 3)) == ["fetch_raw_data", "stale_data_gate"]
    assert not (results / legacy_receipt_name(RUN_ID)).exists()


def test_failed_stage_leaves_prior_generation(tmp_path: Path) -> None:
    results = tmp_path / "results"
    proc = _publish(tmp_path, results, [_stage("fetch_raw_data"), _stage("stale_data_gate")])
    assert proc.returncode == 0, proc.stdout
    before = _generation(results, 1).read_bytes()
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}'
$script:StageReceipts = @([pscustomobject]@{{ name = 'preprocess'; outcome = 'failed' }})
try {{
    throw 'STEP_FAILED:preprocess:1'
    Write-StageReceipts -ResultsDirectory '{results}' -RunId '{RUN_ID}'
}} catch {{
    if ($_.Exception.Message -notlike 'STEP_FAILED:*') {{ throw }}
}}
"""
    failed = _ps(tmp_path, body)
    assert failed.returncode == 0, failed.stdout
    assert _generation(results, 1).read_bytes() == before
    assert not _generation(results, 2).exists()


def test_missing_corrupt_and_ambiguous_generations_fail_closed(tmp_path: Path) -> None:
    complete = [_stage(name) for name in REQUIRED_STAGES]
    root = tmp_path / "corrupt"
    results = _install(root, complete, success_log=True)
    _generation(results, 1).write_text(json.dumps(complete), encoding="utf-8")
    (results / legacy_receipt_name(RUN_ID)).write_text(json.dumps(complete), encoding="utf-8")
    _generation(results, 2).write_text("{", encoding="utf-8")
    corrupt = _assert_not_complete(root)
    assert corrupt["detail"] == "stage_receipt_generation_malformed"
    assert resolve_stage_receipts(results, RUN_ID).status == "malformed"

    root = tmp_path / "ambiguous"
    results = _install(root, complete, success_log=True)
    _generation(results, 1).write_text(json.dumps(complete), encoding="utf-8")
    (results / legacy_receipt_name(RUN_ID)).write_text(json.dumps(complete), encoding="utf-8")
    padded = results / generation_receipt_name(RUN_ID, 1).replace(".gen.1.", ".gen.01.")
    padded.write_text(json.dumps(complete), encoding="utf-8")
    ambiguous = _assert_not_complete(root)
    assert ambiguous["detail"] == "stage_receipt_generation_ambiguous"

    root = tmp_path / "missing"
    results = _install(root, complete, success_log=True)
    _generation(results, 1).write_text(json.dumps(complete), encoding="utf-8")
    (results / legacy_receipt_name(RUN_ID)).write_text(json.dumps(complete), encoding="utf-8")
    newest = _generation(results, 2)
    newest.write_text(json.dumps(complete), encoding="utf-8")
    with _Handle(newest, 0):
        missing = resolve_stage_receipts(results, RUN_ID)
        verdict = _assert_not_complete(root)
    assert missing.status == "missing"
    assert missing.generation == 2
    assert verdict["detail"] == "stage_receipt_generation_missing"


def test_incomplete_run_remains_incomplete(tmp_path: Path) -> None:
    partial = [_stage(name) for name in REQUIRED_STAGES[:3]]
    root = tmp_path / "partial"
    results = _install(root, partial, fetch_passed=True)
    _generation(results, 1).write_text(
        json.dumps([_stage(name) for name in REQUIRED_STAGES]),
        encoding="utf-8",
    )
    _generation(results, 2).write_text(json.dumps(partial), encoding="utf-8")
    verdict = _assert_not_complete(root)
    assert verdict["classification"] == CLASS_FETCH_ONLY
    assert resolve_stage_receipts(results, RUN_ID).generation == 2


def test_complete_run_with_all_required_stage_evidence_passes(tmp_path: Path) -> None:
    stages = [_stage(name) for name in REQUIRED_STAGES]
    root = tmp_path / "complete"
    results = _install(root, stages)
    _generation(results, 1).write_text(json.dumps(stages[:2]), encoding="utf-8")
    _generation(results, 2).write_text(json.dumps(stages), encoding="utf-8")
    verdict = evaluate_daily_refresh_completion(root, SESSION)
    closeout = _daily_refresh_status(root, SESSION)
    assert verdict["ok"] is True
    assert verdict["classification"] == CLASS_COMPLETE
    assert verdict["run_id"] == RUN_ID
    assert closeout == verdict


def test_legacy_only_receipt_compatibility(tmp_path: Path) -> None:
    stages = [_stage(name) for name in REQUIRED_STAGES]
    root = tmp_path / "legacy"
    results = _install(root, stages)
    legacy = results / legacy_receipt_name(RUN_ID)
    legacy.write_text(json.dumps(stages), encoding="utf-8")
    resolved = resolve_stage_receipts(results, RUN_ID)
    assert resolved.status == "legacy"
    assert resolved.generation is None
    assert resolved.path == legacy
    verdict = evaluate_daily_refresh_completion(root, SESSION)
    assert verdict["classification"] == CLASS_COMPLETE
    assert verdict["ok"] is True
