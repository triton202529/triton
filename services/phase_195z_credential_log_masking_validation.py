"""
phase_195z_credential_log_masking_validation.py
Phase 195Z — Validate Alpaca API keys are masked in ENV CHECK logs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
SERVICES = ROOT / "services"

OUTPUT_JSON = RESULTS / "credential_log_masking_validation.json"
BROKER_ALPACA = SERVICES / "broker_alpaca.py"
CREDENTIAL_MASK = SERVICES / "credential_mask.py"

# Alpaca paper keys often look like PKxxxxxxxxxxxxxxxxxxxxxxxxxxxx
FULL_KEY_PATTERN = re.compile(r"PK[A-Z0-9]{20,}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _scan_source_for_unsafe_prints(path: Path) -> List[str]:
    issues: List[str] = []
    if not path.is_file():
        return [f"missing:{path}"]
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        if "ENV CHECK" in line and "getenv" in line and "mask" not in line.lower():
            if "format_env_key_status" not in line and "mask_api_key" not in line:
                issues.append(f"{path.name}:{i}:{line.strip()[:120]}")
    return issues


def _capture_broker_env_check_output() -> Tuple[str, List[str]]:
    code = (
        "import os; "
        "os.environ.setdefault('ALPACA_KEY_ID','PKD3ZAABNRCTKKV2V2BSQGX4DH'); "
        "os.environ.setdefault('APCA_API_KEY_ID','PKQOKTQT5S4ROIBFW2WFRJY7OU'); "
        "import importlib; "
        "import services.broker_alpaca as b; "
        "importlib.reload(b)"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=45,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        leaks = FULL_KEY_PATTERN.findall(out)
        return out, leaks
    except Exception as e:
        return "", [str(e)]


def _scan_recent_logs(max_files: int = 30) -> Tuple[List[str], List[str]]:
    log_dirs = [ROOT / "logs", RESULTS / "continuous_loop", RESULTS / "watchdog_logs"]
    files: List[Path] = []
    for d in log_dirs:
        if d.is_dir():
            files.extend(sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True))
    leaks: List[str] = []
    scanned: List[str] = []
    for path in files[:max_files]:
        scanned.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "ENV CHECK" not in line:
                continue
            for match in FULL_KEY_PATTERN.findall(line):
                if "****" not in line:
                    leaks.append(f"{path.name}:ENV_CHECK:{match[:8]}...")
    return scanned, leaks


def run_phase_195z_credential_validation(*, verbose: bool = True) -> Dict[str, Any]:
    from services.credential_mask import mask_api_key

    sample = "PKD3ZAABNRCTKKV2V2BSQGX4DH"
    masked = mask_api_key(sample)
    mask_ok = "****" in masked and sample not in masked and masked.endswith(sample[-4:])

    source_issues = _scan_source_for_unsafe_prints(BROKER_ALPACA)
    runtime_out, runtime_leaks = _capture_broker_env_check_output()
    log_scanned, log_leaks = _scan_recent_logs()

    checks = {
        "credential_mask_module_exists": CREDENTIAL_MASK.is_file(),
        "mask_api_key_redacts_full_key": mask_ok,
        "broker_alpaca_uses_masked_env_check": not source_issues,
        "runtime_import_no_full_key_leak": len(runtime_leaks) == 0,
    }
    if log_leaks:
        checks["recent_env_check_lines_historical_leak"] = False
    else:
        checks["recent_env_check_lines_historical_leak"] = True
    phase_pass = (
        checks["credential_mask_module_exists"]
        and checks["mask_api_key_redacts_full_key"]
        and checks["broker_alpaca_uses_masked_env_check"]
        and checks["runtime_import_no_full_key_leak"]
    )

    payload: Dict[str, Any] = {
        "phase": "195Z",
        "generated_at": _utc_now_iso(),
        "phase_pass": phase_pass,
        "validation_status": "PASS" if phase_pass else "FAIL",
        "checks": checks,
        "masked_sample": masked,
        "broker_alpaca_source_issues": source_issues,
        "runtime_env_check_output": runtime_out.strip()[:500],
        "runtime_leaks_found": runtime_leaks,
        "recent_logs_scanned": log_scanned,
        "recent_log_leaks": log_leaks,
    }
    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(f"[195Z] credential validation_status={payload['validation_status']}")
        print(f"[195Z] masked_sample={masked}")
        if runtime_out.strip():
            for line in runtime_out.strip().splitlines():
                if "ENV CHECK" in line:
                    print(line)
        if not phase_pass:
            failed = [k for k, v in checks.items() if not v]
            print(f"[195Z] failed checks: {failed}")

    return payload


def main() -> None:
    result = run_phase_195z_credential_validation(verbose=True)
    sys.exit(0 if result.get("validation_status") == "PASS" else 1)


if __name__ == "__main__":
    main()
