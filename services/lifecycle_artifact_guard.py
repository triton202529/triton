"""
lifecycle_artifact_guard.py
Phase 195I — Protect lifecycle authority CSVs from partial pipeline overwrite.

Atomic write + minimum row validation + timestamped backups + recovery.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
BACKUPS_DIR = RESULTS_DIR / "backups"
TICKERS_FILE = PROJECT_ROOT / "data" / "config" / "tickers.txt"

DEFAULT_MIN_ROWS = 50
DIAGNOSTICS_JSON = RESULTS_DIR / "lifecycle_artifact_write_diagnostics.json"

RAW_LIFECYCLE_PATH = RESULTS_DIR / "signal_lifecycle.csv"
EFFECTIVE_LIFECYCLE_PATH = RESULTS_DIR / "signal_lifecycle_effective.csv"

RAW_REQUIRED_COLUMNS = ("ticker", "position_state")
EFFECTIVE_REQUIRED_COLUMNS = ("ticker", "effective_position_state", "effective_stance")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def allow_partial_lifecycle_writes() -> bool:
    return str(os.environ.get("TRITON_LIFECYCLE_ALLOW_PARTIAL", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )


def load_universe_ticker_count() -> int:
    if not TICKERS_FILE.is_file():
        return 0
    count = 0
    for line in TICKERS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        count += 1
    return count


def _operational_universe_count() -> int:
    """Prefer signals_asof diagnostics (excludes intentionally skipped tickers)."""
    diag = RESULTS_DIR / "signals_asof_diagnostics.json"
    if diag.is_file():
        try:
            doc = json.loads(diag.read_text(encoding="utf-8"))
            for key in ("tickers", "tickers_processed", "have_session"):
                val = doc.get(key)
                if val is not None:
                    return int(val)
        except Exception:
            pass
    return 0


def minimum_required_rows(*, override: Optional[int] = None) -> int:
    if override is not None:
        return max(1, int(override))
    env = os.environ.get("TRITON_LIFECYCLE_MIN_ROWS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    operational = _operational_universe_count()
    configured = load_universe_ticker_count()
    if operational >= DEFAULT_MIN_ROWS:
        return operational
    if configured >= DEFAULT_MIN_ROWS:
        return configured
    return DEFAULT_MIN_ROWS


def preferred_expected_rows() -> int:
    operational = _operational_universe_count()
    if operational > 0:
        return operational
    universe = load_universe_ticker_count()
    return universe if universe > 0 else DEFAULT_MIN_ROWS


def count_lifecycle_rows(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    try:
        df = pd.read_csv(path)
        if df.empty or "ticker" not in df.columns:
            return 0
        return int(df["ticker"].astype(str).str.strip().str.upper().nunique())
    except Exception:
        return 0


def count_dataframe_rows(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if "ticker" not in df.columns:
        return int(len(df))
    return int(df["ticker"].astype(str).str.strip().str.upper().nunique())


def validate_lifecycle_columns(df: pd.DataFrame, *, artifact: str) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if df is None or df.empty:
        return False, ["empty_dataframe"]
    cols = {str(c).strip().lower() for c in df.columns}
    if artifact == "effective":
        required = EFFECTIVE_REQUIRED_COLUMNS
        if "effective_position_state" not in cols:
            missing.append("effective_position_state")
        if "effective_stance" not in cols:
            missing.append("effective_stance")
    else:
        required = RAW_REQUIRED_COLUMNS
        if "position_state" not in cols:
            missing.append("position_state")
        has_action = "lifecycle_action" in cols or "stance" in cols
        if not has_action:
            missing.append("lifecycle_action_or_stance")
    if "ticker" not in cols:
        missing.append("ticker")
    for col in required:
        if col not in cols and col not in missing:
            missing.append(col)
    return len(missing) == 0, missing


def is_lifecycle_authoritative(path: Path, *, min_rows: Optional[int] = None) -> bool:
    if allow_partial_lifecycle_writes():
        return path.is_file() and path.stat().st_size > 0
    threshold = min_rows if min_rows is not None else minimum_required_rows()
    return count_lifecycle_rows(path) >= threshold


def _artifact_label(path: Path) -> str:
    name = path.name.lower()
    if "effective" in name:
        return "effective"
    return "raw"


def _backup_path_for(production: Path) -> Path:
    stem = production.stem
    return BACKUPS_DIR / f"{stem}_{_utc_stamp()}.csv"


def backup_lifecycle_artifact(production: Path) -> Optional[Path]:
    if not production.is_file() or production.stat().st_size == 0:
        return None
    if not is_lifecycle_authoritative(production):
        return None
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _backup_path_for(production)
    shutil.copy2(production, dest)
    return dest


def list_backups(production: Path) -> List[Path]:
    if not BACKUPS_DIR.is_dir():
        return []
    stem = production.stem
    prefix = f"{stem}_"
    return sorted(
        (
            p
            for p in BACKUPS_DIR.glob("*.csv")
            if p.name.startswith(prefix) and p.name[len(prefix) : len(prefix) + 1].isdigit()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def restore_latest_valid_backup(
    production: Path, *, min_rows: Optional[int] = None
) -> Optional[Path]:
    threshold = min_rows if min_rows is not None else minimum_required_rows()
    for backup in list_backups(production):
        if count_lifecycle_rows(backup) >= threshold:
            production.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, production)
            return backup
    return None


@dataclass
class LifecycleWriteResult:
    ok: bool
    path: Path
    rows: int
    min_rows: int
    action: str
    backup_path: Optional[Path] = None
    restored_from: Optional[Path] = None
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "path": str(self.path),
            "rows": self.rows,
            "min_rows": self.min_rows,
            "action": self.action,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "restored_from": str(self.restored_from) if self.restored_from else None,
            "error": self.error,
            "details": self.details,
        }


def _write_diagnostics(entry: Dict[str, Any]) -> None:
    try:
        existing: Dict[str, Any] = {}
        if DIAGNOSTICS_JSON.is_file():
            existing = json.loads(DIAGNOSTICS_JSON.read_text(encoding="utf-8"))
        history = existing.get("history") or []
        history.append(entry)
        existing["history"] = history[-50:]
        existing["last"] = entry
        DIAGNOSTICS_JSON.parent.mkdir(parents=True, exist_ok=True)
        tmp = DIAGNOSTICS_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, DIAGNOSTICS_JSON)
    except Exception:
        pass


def write_lifecycle_csv_protected(
    df: pd.DataFrame,
    production: Path,
    *,
    artifact: Optional[str] = None,
    min_rows: Optional[int] = None,
    allow_partial: Optional[bool] = None,
) -> LifecycleWriteResult:
    """
    Write lifecycle CSV atomically only if row count meets minimum threshold.
    On failure, preserve existing production file; attempt backup restore if current is partial.
    """
    production = Path(production)
    artifact_kind = artifact or _artifact_label(production)
    threshold = min_rows if min_rows is not None else minimum_required_rows()
    partial_ok = allow_partial if allow_partial is not None else allow_partial_lifecycle_writes()
    rows = count_dataframe_rows(df)
    cols_ok, missing_cols = validate_lifecycle_columns(df, artifact=artifact_kind)

    result = LifecycleWriteResult(
        ok=False,
        path=production,
        rows=rows,
        min_rows=threshold,
        action="rejected",
    )

    if not cols_ok:
        result.error = f"missing_columns:{','.join(missing_cols)}"
        result.action = "rejected_invalid_columns"
        _handle_failed_write(production, result, threshold)
        return result

    if rows < threshold and not partial_ok:
        result.error = f"insufficient_rows:{rows}<{threshold}"
        result.action = "rejected_insufficient_rows"
        print(
            f"[LIFECYCLE_ARTIFACT_GUARD] action=rejected path={production.name} "
            f"rows={rows} min_rows={threshold} reason=insufficient_rows",
            flush=True,
        )
        _handle_failed_write(production, result, threshold)
        return result

    production.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = production.with_suffix(production.suffix + ".tmp")
    backup_path: Optional[Path] = None

    try:
        df.to_csv(tmp_path, index=False)
        tmp_rows = count_lifecycle_rows(tmp_path)
        if tmp_rows < threshold and not partial_ok:
            tmp_path.unlink(missing_ok=True)
            result.error = f"tmp_validation_failed:{tmp_rows}<{threshold}"
            result.action = "rejected_tmp_validation"
            _handle_failed_write(production, result, threshold)
            return result

        if is_lifecycle_authoritative(production, min_rows=threshold):
            backup_path = backup_lifecycle_artifact(production)

        os.replace(tmp_path, production)
        result.ok = True
        result.action = "replaced"
        result.backup_path = backup_path
        result.rows = tmp_rows
        print(
            f"[LIFECYCLE_ARTIFACT_GUARD] action=replaced path={production.name} "
            f"rows={tmp_rows} min_rows={threshold}"
            + (f" backup={backup_path.name}" if backup_path else ""),
            flush=True,
        )
    except Exception as e:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        result.error = f"{type(e).__name__}:{e}"
        result.action = "write_error"
        _handle_failed_write(production, result, threshold)
        return result
    finally:
        _write_diagnostics(
            {
                "timestamp": _utc_iso(),
                **result.to_dict(),
            }
        )

    return result


def _handle_failed_write(production: Path, result: LifecycleWriteResult, threshold: int) -> None:
    current_rows = count_lifecycle_rows(production)
    if current_rows > 0 and current_rows < threshold:
        restored = restore_latest_valid_backup(production, min_rows=threshold)
        if restored:
            result.restored_from = restored
            result.action = "restored_from_backup"
            print(
                f"[LIFECYCLE_ARTIFACT_GUARD] action=restored_from_backup "
                f"path={production.name} backup={restored.name} "
                f"rows={count_lifecycle_rows(production)}",
                flush=True,
            )
        else:
            result.action = "fail_closed_no_backup"
            print(
                f"[LIFECYCLE_ARTIFACT_GUARD] action=fail_closed path={production.name} "
                f"rows={current_rows} min_rows={threshold} no_valid_backup",
                flush=True,
            )
    else:
        print(
            f"[LIFECYCLE_ARTIFACT_GUARD] action=preserved_existing path={production.name} "
            f"existing_rows={current_rows}",
            flush=True,
        )
    _write_diagnostics({"timestamp": _utc_iso(), **result.to_dict()})


def audit_lifecycle_authority_preservation() -> Dict[str, Any]:
    min_rows = minimum_required_rows()
    raw_rows = count_lifecycle_rows(RAW_LIFECYCLE_PATH)
    eff_rows = count_lifecycle_rows(EFFECTIVE_LIFECYCLE_PATH)
    raw_ok = raw_rows >= min_rows
    eff_ok = eff_rows >= min_rows
    partial = (raw_rows > 0 and not raw_ok) or (eff_rows > 0 and not eff_ok)
    return {
        "timestamp": _utc_iso(),
        "minimum_required_rows": min_rows,
        "preferred_expected_rows": preferred_expected_rows(),
        "signal_lifecycle_rows": raw_rows,
        "effective_lifecycle_rows": eff_rows,
        "raw_authoritative": raw_ok,
        "effective_authoritative": eff_ok,
        "partial_detected": partial,
        "latest_raw_backup": (
            str(list_backups(RAW_LIFECYCLE_PATH)[0]) if list_backups(RAW_LIFECYCLE_PATH) else None
        ),
        "latest_effective_backup": (
            str(list_backups(EFFECTIVE_LIFECYCLE_PATH)[0])
            if list_backups(EFFECTIVE_LIFECYCLE_PATH)
            else None
        ),
    }


def insufficient_rows_reason(path: Path) -> str:
    if "effective" in path.name.lower():
        return "INSUFFICIENT_EFFECTIVE_ROWS"
    return "INSUFFICIENT_RAW_ROWS"
