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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
BACKUPS_DIR = RESULTS_DIR / "backups"
TICKERS_FILE = PROJECT_ROOT / "data" / "config" / "tickers.txt"

DEFAULT_MIN_ROWS = 50
DIAGNOSTICS_JSON = RESULTS_DIR / "lifecycle_artifact_write_diagnostics.json"

RAW_LIFECYCLE_PATH = RESULTS_DIR / "signal_lifecycle.csv"
EFFECTIVE_LIFECYCLE_PATH = RESULTS_DIR / "signal_lifecycle_effective.csv"
ASOF_DIAGNOSTICS_JSON = RESULTS_DIR / "signals_asof_diagnostics.json"
POSITIONS_SNAPSHOT_PATH = RESULTS_DIR / "positions_snapshot.csv"

RAW_REQUIRED_COLUMNS = ("ticker", "position_state")
EFFECTIVE_REQUIRED_COLUMNS = ("ticker", "effective_position_state", "effective_stance")

# Committed tradable-coverage floor (must match generate_signals ASOF_COVERAGE default).
COMMITTED_ASOF_COVERAGE = 0.90
SKIP_INVALID_PROCESSED_SESSION_BAR = "INVALID_PROCESSED_SESSION_BAR"
SKIP_INDEX_ONLY_EXCLUDED = "INDEX_ONLY_EXCLUDED"
_SHA256_HEX_LEN = 64


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
    doc = load_asof_diagnostics()
    if doc:
        for key in ("tickers", "tickers_processed", "have_session"):
            val = doc.get(key)
            if val is not None:
                return int(val)
    return 0


def load_asof_diagnostics(*, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    target = Path(path) if path is not None else ASOF_DIAGNOSTICS_JSON
    if not target.is_file() or target.stat().st_size == 0:
        return None
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def _norm_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _is_sha256_hex(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if len(text) != _SHA256_HEX_LEN:
        return False
    return all(ch in "0123456789abcdef" for ch in text)


def _parse_session_date(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text or text.upper() in {"UNKNOWN", "NONE", "NULL", "N/A"}:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _source_tickers(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty or "ticker" not in df.columns:
        return []
    return [_norm_symbol(v) for v in df["ticker"].tolist() if _norm_symbol(v)]


def _source_session_dates(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    col = None
    for name in ("as_of_date", "signal_date", "date"):
        if name in df.columns:
            col = name
            break
    if col is None:
        return []
    out: List[str] = []
    for value in df[col].tolist():
        parsed = _parse_session_date(value)
        if parsed:
            out.append(parsed)
    return out


def _held_tickers_from_snapshot(path: Optional[Path] = None) -> List[str]:
    target = Path(path) if path is not None else POSITIONS_SNAPSHOT_PATH
    if not target.is_file() or target.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(target)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    sym_col = next((c for c in ("ticker", "symbol", "Symbol") if c in df.columns), None)
    if sym_col is None:
        return []
    qty_col = next((c for c in ("qty", "qty_available", "quantity") if c in df.columns), None)
    held: List[str] = []
    for _, row in df.iterrows():
        ticker = _norm_symbol(row.get(sym_col))
        if not ticker:
            continue
        qty = 0.0
        if qty_col is not None:
            try:
                qty = float(row.get(qty_col) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
        else:
            qty = 1.0
        if qty > 0:
            held.append(ticker)
    return sorted(set(held))


def verified_invalid_bar_skips(
    diag: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Independently verify dated INVALID_PROCESSED_SESSION_BAR skip evidence."""
    reasons: List[str] = []
    session = _parse_session_date(diag.get("session_asof_date"))
    if not session:
        return [], ["unknown_session"]
    raw_skips = diag.get("per_symbol_skips")
    if raw_skips is None:
        return [], ["missing_skip_evidence"]
    if not isinstance(raw_skips, list):
        return [], ["forged_skip_evidence"]

    verified: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rec in raw_skips:
        if not isinstance(rec, dict):
            reasons.append("forged_skip_evidence")
            continue
        reason = str(rec.get("reason") or "").strip()
        if reason != SKIP_INVALID_PROCESSED_SESSION_BAR:
            continue
        symbol = _norm_symbol(rec.get("symbol"))
        expected = _parse_session_date(rec.get("expected_session_date"))
        rec_class = str(rec.get("class") or "").strip().lower()
        sha = rec.get("source_sha256")
        if not symbol:
            reasons.append("forged_skip_evidence")
            continue
        if symbol in seen:
            reasons.append("duplicate_skip_symbol")
            continue
        seen.add(symbol)
        if expected != session:
            reasons.append("dated_skip_session_mismatch")
            continue
        if rec_class != "tradable":
            reasons.append("forged_skip_evidence")
            continue
        if not _is_sha256_hex(sha):
            reasons.append("forged_skip_evidence")
            continue
        verified.append(
            {
                "symbol": symbol,
                "expected_session_date": expected,
                "reason": reason,
                "source_sha256": str(sha).strip().lower(),
                "class": rec_class,
            }
        )
    return verified, reasons


def evaluate_coverage_skip_acceptance(
    *,
    diagnostics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Accept current-session row count only when 0.90 coverage + dated skips verify."""
    result: Dict[str, Any] = {
        "ok": False,
        "accepted_rows": 0,
        "tradable_tickers": 0,
        "have_session": 0,
        "coverage": 0.0,
        "threshold": COMMITTED_ASOF_COVERAGE,
        "session_asof_date": "",
        "verified_invalid_bar_skips": [],
        "index_exclusions": 0,
        "reason": "missing_skip_evidence",
    }
    diag = load_asof_diagnostics(path=diagnostics_path)
    if not diag:
        result["reason"] = "missing_skip_evidence"
        return result

    session = _parse_session_date(diag.get("session_asof_date"))
    if not session:
        result["reason"] = "unknown_session"
        return result
    result["session_asof_date"] = session

    try:
        tradable = int(diag.get("tradable_tickers") or diag.get("tickers") or 0)
        have_session = int(diag.get("have_session") or 0)
        reported_threshold = float(diag.get("asof_coverage_threshold"))
    except (TypeError, ValueError):
        result["reason"] = "forged_skip_evidence"
        return result

    result["tradable_tickers"] = tradable
    result["have_session"] = have_session
    if tradable <= 0:
        result["reason"] = "unknown_session"
        return result
    if abs(reported_threshold - COMMITTED_ASOF_COVERAGE) > 1e-9:
        result["reason"] = "coverage_threshold_mismatch"
        return result
    if diag.get("fail") is True:
        result["reason"] = "coverage_below_threshold"
        return result

    verified, skip_reasons = verified_invalid_bar_skips(diag)
    if skip_reasons:
        result["reason"] = skip_reasons[0]
        result["verified_invalid_bar_skips"] = verified
        return result
    result["verified_invalid_bar_skips"] = verified

    index_skips = [
        rec
        for rec in (diag.get("per_symbol_skips") or [])
        if isinstance(rec, dict)
        and str(rec.get("reason") or "").strip() == SKIP_INDEX_ONLY_EXCLUDED
    ]
    result["index_exclusions"] = len(index_skips)

    accepted = tradable - len(verified)
    independent_coverage = accepted / float(tradable)
    result["coverage"] = round(independent_coverage, 4)
    result["accepted_rows"] = accepted
    if independent_coverage < COMMITTED_ASOF_COVERAGE:
        result["reason"] = "coverage_below_threshold"
        return result
    if have_session != accepted:
        result["reason"] = "forged_skip_evidence"
        return result
    if diag.get("batch_actionable") is not True:
        result["reason"] = "coverage_below_threshold"
        return result
    result["ok"] = True
    result["reason"] = "coverage_ok_verified_skips"
    return result


def deferred_skipped_held_tickers(
    *,
    source_tickers: Sequence[str],
    held_tickers: Sequence[str],
    verified_skip_symbols: Sequence[str],
) -> List[str]:
    """Held names skipped for invalid bars — never silent HOLD or unauthorized SELL."""
    source_set = {_norm_symbol(s) for s in source_tickers if _norm_symbol(s)}
    held_set = {_norm_symbol(s) for s in held_tickers if _norm_symbol(s)}
    skip_set = {_norm_symbol(s) for s in verified_skip_symbols if _norm_symbol(s)}
    return sorted((held_set & skip_set) - source_set)


def evaluate_lifecycle_source_gate(
    df: pd.DataFrame,
    *,
    diagnostics_path: Optional[Path] = None,
    positions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """D1b source gate: coverage-verified current-session rows, else configured min."""
    src_rows = count_dataframe_rows(df)
    configured_min = _configured_minimum_rows()
    details: Dict[str, Any] = {
        "src_rows": src_rows,
        "duplicate_tickers": [],
        "deferred_skipped_held": [],
        "coverage": None,
    }
    tickers = _source_tickers(df)
    dupes = sorted({t for t in tickers if tickers.count(t) > 1})
    details["duplicate_tickers"] = dupes
    if dupes:
        return {
            "ok": False,
            "min_rows": configured_min,
            "accepted_rows": src_rows,
            "reason": "duplicate_symbols",
            "details": details,
        }

    coverage = evaluate_coverage_skip_acceptance(diagnostics_path=diagnostics_path)
    details["coverage"] = coverage
    session = coverage.get("session_asof_date") or ""
    source_dates = set(_source_session_dates(df))
    if coverage.get("reason") == "unknown_session":
        return {
            "ok": False,
            "min_rows": configured_min,
            "accepted_rows": src_rows,
            "reason": "unknown_session",
            "details": details,
        }
    if session and source_dates and source_dates != {session}:
        return {
            "ok": False,
            "min_rows": configured_min,
            "accepted_rows": src_rows,
            "reason": "stale_lifecycle_inputs",
            "details": details,
        }

    verified_syms = [
        str(rec.get("symbol")) for rec in (coverage.get("verified_invalid_bar_skips") or [])
    ]
    overlap = sorted(set(tickers) & set(verified_syms))
    if overlap:
        return {
            "ok": False,
            "min_rows": configured_min,
            "accepted_rows": src_rows,
            "reason": "forged_skip_evidence",
            "details": details,
        }

    held = _held_tickers_from_snapshot(positions_path)
    details["deferred_skipped_held"] = deferred_skipped_held_tickers(
        source_tickers=tickers,
        held_tickers=held,
        verified_skip_symbols=verified_syms,
    )

    if coverage.get("ok"):
        accepted = int(coverage["accepted_rows"])
        if src_rows != accepted:
            return {
                "ok": False,
                "min_rows": accepted,
                "accepted_rows": src_rows,
                "reason": "signal_row_count_mismatch",
                "details": details,
            }
        return {
            "ok": True,
            "min_rows": accepted,
            "accepted_rows": src_rows,
            "reason": coverage.get("reason") or "coverage_ok_verified_skips",
            "details": details,
        }

    if src_rows < configured_min:
        return {
            "ok": False,
            "min_rows": configured_min,
            "accepted_rows": src_rows,
            "reason": coverage.get("reason") or "insufficient_signal_rows",
            "details": details,
        }
    return {
        "ok": True,
        "min_rows": configured_min,
        "accepted_rows": src_rows,
        "reason": "full_universe_rows",
        "details": details,
    }


def _configured_minimum_rows(*, override: Optional[int] = None) -> int:
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


def minimum_required_rows(*, override: Optional[int] = None) -> int:
    if override is not None:
        return max(1, int(override))
    env = os.environ.get("TRITON_LIFECYCLE_MIN_ROWS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    coverage = evaluate_coverage_skip_acceptance()
    if coverage.get("ok") and int(coverage.get("accepted_rows") or 0) >= 1:
        return int(coverage["accepted_rows"])
    return _configured_minimum_rows()


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
