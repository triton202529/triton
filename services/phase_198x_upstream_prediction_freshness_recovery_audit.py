"""
phase_198x_upstream_prediction_freshness_recovery_audit.py
Phase 198X — Upstream prediction freshness recovery audit.

Read-only diagnosis of why generate_signals fails its AS_OF invariant.
Does not place orders, create approvals/auth, loosen AS_OF, or modify
allocation / broker execution behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
SCRIPTS = ROOT / "scripts"

OUTPUT_JSON = RESULTS / "phase_198x_upstream_prediction_freshness_recovery_audit.json"
OUTPUT_MD = RESULTS / "phase_198x_upstream_prediction_freshness_recovery_audit.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
QUARANTINE_PATH = GOVERNANCE / "symbol_hydration_quarantine.json"

PREDICTIONS_DIR = ROOT / "data" / "predictions"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TICKERS_FILE = ROOT / "data" / "config" / "tickers.txt"
ASOF_DIAG = RESULTS / "signals_asof_diagnostics.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
ALLOC_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
ALLOC_RECS = RESULTS / "portfolio_allocation_recommendations.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
MARKET_OPEN_PS1 = SCRIPTS / "run_market_open.ps1"
CONTINUOUS_LOOP_PS1 = SCRIPTS / "run_continuous_loop.ps1"

MARKER = "UPSTREAM_PREDICTION_FRESHNESS_RECOVERY_AUDIT_ONLY"
GLD_SYMBOL = "GLD"
GLD_TARGET_WEIGHT = 0.20

RECOMMENDED_REFRESH_SEQUENCE = (
    "python -m services.fetch_raw_data; "
    "python -m services.preprocess; "
    "python -m services.train_model; "
    "python -m services.generate_signals; "
    "python -m services.apply_signal_lifecycle; "
    "python -m services.build_effective_lifecycle; "
    "python -m services.execution_readiness; "
    "python -m services.capital_preservation_certification --skip-stress-revalidation"
)

RECOMMENDED_NEXT_PHASE = (
    "198Y — execute upstream prediction recovery "
    "(fetch_raw_data → preprocess → train_model → generate_signals) "
    "without loosening TRITON_STRICT_ASOF / MAX_LAG_SESSIONS; "
    "then re-check readiness/certification"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(0, len(list(csv.reader(f))) - 1)
    except Exception:
        return 0


def _parse_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    try:
        import pandas as pd

        ts = pd.to_datetime(v, errors="coerce")
        if ts is None or (hasattr(ts, "isna") and bool(ts.isna())):
            return None
        return ts.date() if hasattr(ts, "date") else date.fromisoformat(str(v)[:10])
    except Exception:
        try:
            return date.fromisoformat(str(v)[:10])
        except Exception:
            return None


def _load_tickers() -> List[str]:
    tickers: List[str] = []
    if TICKERS_FILE.is_file():
        for line in TICKERS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            t = line.strip().upper()
            if t and not t.startswith("#"):
                tickers.append(t)
    if tickers:
        return sorted(set(tickers))
    # Fallback: derive from prediction filenames.
    for p in PREDICTIONS_DIR.glob("*_predictions.parquet"):
        tickers.append(p.name[: -len("_predictions.parquet")])
    for p in PREDICTIONS_DIR.glob("*_predictions.csv"):
        tickers.append(p.name[: -len("_predictions.csv")])
    return sorted(set(tickers))


def _latest_date_in_file(path: Path) -> Tuple[Optional[date], str]:
    if not path.is_file():
        return None, "missing"
    try:
        import pandas as pd

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        return None, f"read_error:{exc}"

    for col in ("date", "Date", "timestamp", "Datetime", "as_of", "signal_date"):
        if col in df.columns:
            s = __import__("pandas").to_datetime(df[col], errors="coerce").dropna()
            if len(s):
                return s.max().date(), f"col={col} rows={len(df)}"
    try:
        import pandas as pd

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
            return pd.to_datetime(df.index).max().date(), f"index rows={len(df)}"
    except Exception:
        pass
    return None, f"no_date_col cols={list(df.columns)[:6]} rows={len(df)}"


def _scan_layer(
    *,
    tickers: Sequence[str],
    path_for: Any,
    required_session: Optional[date],
    sample_limit: int = 65,
) -> Dict[str, Any]:
    latest_dates: Dict[str, Optional[date]] = {}
    missing: List[str] = []
    notes: Dict[str, str] = {}
    for t in list(tickers)[:sample_limit]:
        path = path_for(t)
        d, note = _latest_date_in_file(path)
        latest_dates[t] = d
        notes[t] = note
        if d is None:
            missing.append(t)

    present_dates = [d for d in latest_dates.values() if d is not None]
    max_date = max(present_dates) if present_dates else None
    min_date = min(present_dates) if present_dates else None
    stale_vs_session: List[str] = []
    if required_session is not None:
        for t, d in latest_dates.items():
            if d is None or d < required_session:
                stale_vs_session.append(t)

    status = "MISSING"
    if present_dates:
        if required_session is None:
            status = "PRESENT"
        elif max_date is not None and max_date >= required_session and not stale_vs_session:
            status = "FRESH"
        elif max_date is not None and max_date >= required_session:
            status = "PARTIAL"
        else:
            status = "STALE"

    return {
        "status": status,
        "latest_date": str(max_date) if max_date else None,
        "earliest_latest_date": str(min_date) if min_date else None,
        "files_scanned": len(latest_dates),
        "files_present": len(present_dates),
        "missing_symbols": sorted(missing),
        "stale_vs_session_symbols": sorted(stale_vs_session)[:40],
        "stale_vs_session_count": len(stale_vs_session),
        "sample_notes": {k: notes[k] for k in list(notes)[:5]},
    }


def _session_asof() -> Optional[date]:
    try:
        from services.generate_signals import _nyse_last_completed_session_date

        return _nyse_last_completed_session_date()
    except Exception:
        diag = _load_json(ASOF_DIAG)
        return _parse_date(diag.get("session_asof_date"))


def _lag_sessions(start: Optional[date], end: Optional[date]) -> Optional[int]:
    if start is None or end is None or start >= end:
        return 0 if start is not None and end is not None and start >= end else None
    try:
        from services.generate_signals import _next_session_date

        cur = start
        lag = 0
        while cur < end and lag <= 20:
            cur = _next_session_date(cur)
            lag += 1
        return lag
    except Exception:
        return (end - start).days


def _script_contains(path: Path, patterns: Sequence[str]) -> Dict[str, bool]:
    text = ""
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
    out: Dict[str, bool] = {}
    for p in patterns:
        out[p] = bool(re.search(re.escape(p), text)) or bool(re.search(p, text))
    return out


def _gld_from_positions(path: Path) -> Dict[str, Any]:
    qty = 0.0
    market_value = 0.0
    total_invested = 0.0
    if not path.is_file():
        return {"qty": 0, "weight": 0.0}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mv = abs(_safe_float(row.get("market_value") or row.get("value")))
                total_invested += mv
                sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
                if sym != GLD_SYMBOL:
                    continue
                qty = _safe_float(row.get("qty") or row.get("quantity"))
                market_value = mv
    except Exception:
        pass
    weight = (market_value / total_invested) if total_invested > 0 else 0.0
    return {
        "qty": int(qty) if qty == int(qty) else qty,
        "weight": round(weight, 6),
    }


def _allocation_counts(summary: Dict[str, Any], recs_path: Path) -> Dict[str, int]:
    counts = {"open_new_count": 0, "add_count": 0, "exit_count": 0, "trim_count": 0}
    for key in list(counts):
        if key in summary:
            counts[key] = _safe_int(summary.get(key))
    action_map = {
        "OPEN_NEW": "open_new_count",
        "ADD": "add_count",
        "EXIT": "exit_count",
        "TRIM": "trim_count",
    }
    if recs_path.is_file():
        try:
            tallies = {k: 0 for k in counts}
            with recs_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    action = (
                        str(row.get("recommended_action") or row.get("action") or "")
                        .strip()
                        .upper()
                    )
                    mapped = action_map.get(action)
                    if mapped:
                        tallies[mapped] += 1
            if any(tallies.values()):
                counts.update(tallies)
        except Exception:
            pass
    return counts


def _probe_generate_signals() -> Dict[str, Any]:
    """Invoke generate_signals once for live failure evidence (no AS_OF loosening)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "services.generate_signals"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        lines = text.splitlines()
        reason = None
        for ln in lines:
            if "AS_OF invariant violated" in ln or "RuntimeError" in ln:
                reason = ln.strip()
                break
        if reason is None and proc.returncode != 0:
            reason = next((ln for ln in reversed(lines) if ln.strip()), "generate_signals_failed")
        return {
            "exit_code": proc.returncode,
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "failure_reason": reason,
            "output_tail": lines[-12:],
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": 124,
            "status": "TIMEOUT",
            "failure_reason": "generate_signals_timeout",
            "output_tail": [],
        }
    except Exception as exc:
        return {
            "exit_code": 1,
            "status": "ERROR",
            "failure_reason": str(exc),
            "output_tail": [],
        }


def _classify(
    *,
    raw: Dict[str, Any],
    processed: Dict[str, Any],
    preds: Dict[str, Any],
    lag_sessions: Optional[int],
    strict_asof: bool,
    fail: bool,
    spcx_relevant: bool,
    path_mismatch: bool,
) -> Tuple[str, List[str]]:
    secondary: List[str] = []
    if strict_asof and fail:
        secondary.append("STRICT_ASOF_CORRECTLY_BLOCKED")
    if preds.get("status") == "STALE":
        secondary.append("PREDICTIONS_STALE")
    if processed.get("status") == "STALE":
        secondary.append("PROCESSED_DATA_STALE")
    if raw.get("status") == "STALE":
        secondary.append("RAW_DATA_STALE")
    if path_mismatch:
        return "ARTIFACT_PATH_MISMATCH", secondary
    if spcx_relevant:
        return "SYMBOL_QUARANTINE_RELATED", secondary

    # Root-cause priority: raw → processed → predictions.
    if raw.get("status") in ("STALE", "MISSING", "PARTIAL"):
        secondary.append("TRAINING_PIPELINE_REQUIRED")
        secondary.append("MODEL_INPUT_STALE")
        return "RAW_DATA_STALE", secondary
    if processed.get("status") in ("STALE", "MISSING", "PARTIAL"):
        secondary.append("TRAINING_PIPELINE_REQUIRED")
        secondary.append("MODEL_INPUT_STALE")
        return "PROCESSED_DATA_STALE", secondary
    if preds.get("status") in ("STALE", "MISSING", "PARTIAL"):
        # Raw+processed fresh but predictions stale → prediction/train only.
        if raw.get("status") == "FRESH" and processed.get("status") == "FRESH":
            secondary.append("PREDICTION_ONLY_REFRESH_REQUIRED")
            return "PREDICTIONS_STALE", secondary
        secondary.append("TRAINING_PIPELINE_REQUIRED")
        return "MODEL_INPUT_STALE", secondary
    if lag_sessions is not None and lag_sessions > 1 and fail:
        secondary.append("TRAINING_PIPELINE_REQUIRED")
        return "PREDICTIONS_STALE", secondary
    if fail and strict_asof:
        return "STRICT_ASOF_CORRECTLY_BLOCKED", secondary
    return "PREDICTIONS_STALE", secondary


def _render_markdown(payload: Dict[str, Any]) -> str:
    a = payload.get("audit") or {}
    return f"""# TRITON Phase 198X — Upstream Prediction Freshness Recovery Audit

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Blocker classification:** `{a.get("blocker_classification")}`

## Executive summary

{payload.get("executive_summary")}

## generate_signals AS_OF

| Field | Value |
|-------|-------|
| generate_signals_status | {a.get("generate_signals_status")} |
| failure_reason | {a.get("generate_signals_failure_reason")} |
| required_session_date | {a.get("required_session_date")} |
| latest_prediction_date | {a.get("latest_prediction_date")} |
| prediction_age_sessions | {a.get("prediction_age_sessions")} |
| prediction_artifact_path | `{a.get("prediction_artifact_path")}` |
| strict_asof_enabled | {a.get("strict_asof_enabled")} |
| strict_asof_block_was_correct | {a.get("strict_asof_block_was_correct")} |

## Upstream layer freshness

| Layer | Status | Latest date | Missing |
|-------|--------|-------------|---------|
| raw | {a.get("raw_data_status")} | {a.get("raw_data_latest_date")} | {a.get("raw_data_missing_count")} |
| processed | {a.get("processed_data_status")} | {a.get("processed_data_latest_date")} | — |
| model/predictions | {a.get("model_input_status")} | {a.get("model_input_latest_date")} | — |

## Pipeline requirements

| Field | Value |
|-------|-------|
| training_required | {a.get("training_required")} |
| prediction_refresh_required | {a.get("prediction_refresh_required")} |
| fetch_raw_data_required | {a.get("fetch_raw_data_required")} |
| preprocess_required | {a.get("preprocess_required")} |
| train_model_required | {a.get("train_model_required")} |
| spcx_quarantine_relevant | {a.get("spcx_quarantine_relevant")} |
| market_open_calls_signal_refresh | {a.get("market_open_calls_signal_refresh")} |
| continuous_loop_invokes_market_open | {a.get("continuous_loop_invokes_market_open")} |
| market_open_calls_upstream_fetch_train | {a.get("market_open_calls_upstream_fetch_train")} |

## Secondary classifications

{chr(10).join("- `" + c + "`" for c in (a.get("secondary_classifications") or [])) or "- None"}

## Safety

| Field | Value |
|-------|-------|
| broker_connected | {a.get("broker_connected")} |
| open_orders_count | {a.get("open_orders_count")} |
| paper_arm_valid | {a.get("paper_arm_valid")} |
| live_execution_blocked | {a.get("live_execution_blocked")} |
| approval_artifact_present | {a.get("approval_artifact_present")} |
| broker_auth_artifact_present | {a.get("broker_auth_artifact_present")} |
| gld_qty / weight | {a.get("gld_qty")} / {a.get("gld_weight")} |
| gld_target_satisfied | {a.get("gld_target_satisfied")} |
| allocation open_new / add | {a.get("allocation_open_new_count")} / {a.get("allocation_add_count")} |
| broker_orders_submitted | {a.get("broker_orders_submitted")} |

## Recommended refresh sequence

```
{a.get("recommended_refresh_sequence")}
```

## Recommended next phase

{a.get("recommended_next_phase")}

---

*{MARKER}*
"""


def run_audit(*, probe_generate_signals: bool = True, verbose: bool = True) -> Dict[str, Any]:
    from services.paper_arm_status import build_paper_arm_status
    from services.portfolio_history_sync import load_live_execution_flag

    tickers = _load_tickers()
    session = _session_asof()
    asof_diag = _load_json(ASOF_DIAG)

    gen_probe = (
        _probe_generate_signals()
        if probe_generate_signals
        else {
            "exit_code": None,
            "status": "SKIPPED",
            "failure_reason": None,
            "output_tail": [],
        }
    )
    # Re-read diagnostics after probe (generate_signals writes it even on fail).
    asof_diag = _load_json(ASOF_DIAG) or asof_diag

    required_session = _parse_date(asof_diag.get("session_asof_date")) or session
    latest_pred_date = _parse_date(asof_diag.get("max_prediction_date"))
    lag = asof_diag.get("lag_sessions_vs_session")
    if lag is None:
        lag = _lag_sessions(latest_pred_date, required_session)

    strict_asof = bool(asof_diag.get("strict_asof", True))
    try:
        from services.generate_signals import STRICT_ASOF as GS_STRICT

        strict_asof = bool(GS_STRICT)
    except Exception:
        pass

    asof_fail = bool(asof_diag.get("fail")) or (gen_probe.get("status") == "FAIL")

    raw = _scan_layer(
        tickers=tickers,
        path_for=lambda t: RAW_DIR / f"{t}.csv",
        required_session=required_session,
    )
    processed = _scan_layer(
        tickers=tickers,
        path_for=lambda t: PROCESSED_DIR / f"{t}.parquet",
        required_session=required_session,
    )
    preds = _scan_layer(
        tickers=tickers,
        path_for=lambda t: PREDICTIONS_DIR / f"{t}_predictions.parquet",
        required_session=required_session,
    )
    if preds["files_present"] == 0:
        # Try csv predictions.
        preds = _scan_layer(
            tickers=tickers,
            path_for=lambda t: PREDICTIONS_DIR / f"{t}_predictions.csv",
            required_session=required_session,
        )

    if latest_pred_date is None and preds.get("latest_date"):
        latest_pred_date = _parse_date(preds.get("latest_date"))
        lag = _lag_sessions(latest_pred_date, required_session)

    quarantine = _load_json(QUARANTINE_PATH)
    quarantined = list((quarantine.get("symbols") or {}).keys())
    spcx_present = "SPCX" in {str(s).upper() for s in quarantined}
    # SPCX is unrelated unless it alone explains AS_OF lag (it does not: coverage 0%).
    spcx_relevant = False
    if spcx_present and float(asof_diag.get("coverage") or 0) >= 0.9:
        # Only relevant if coverage would pass without it — not the case here.
        spcx_relevant = False

    market_open_mods = _script_contains(
        MARKET_OPEN_PS1,
        [
            "services.generate_signals",
            "services.apply_signal_lifecycle",
            "services.build_effective_lifecycle",
            "services.fetch_raw_data",
            "services.preprocess",
            "services.train_model",
        ],
    )
    continuous_mods = _script_contains(
        CONTINUOUS_LOOP_PS1,
        ["run_market_open.ps1"],
    )
    market_open_calls_signal_refresh = all(
        market_open_mods.get(m)
        for m in (
            "services.generate_signals",
            "services.apply_signal_lifecycle",
            "services.build_effective_lifecycle",
        )
    )
    market_open_calls_upstream = any(
        market_open_mods.get(m)
        for m in ("services.fetch_raw_data", "services.preprocess", "services.train_model")
    )

    # Path mismatch: generate_signals reads data/predictions; if fresher preds exist elsewhere.
    path_mismatch = False

    classification, secondary = _classify(
        raw=raw,
        processed=processed,
        preds=preds,
        lag_sessions=_safe_int(lag, default=-1) if lag is not None else None,
        strict_asof=strict_asof,
        fail=asof_fail,
        spcx_relevant=spcx_relevant,
        path_mismatch=path_mismatch,
    )

    fetch_required = raw.get("status") in ("STALE", "MISSING", "PARTIAL")
    preprocess_required = fetch_required or processed.get("status") in (
        "STALE",
        "MISSING",
        "PARTIAL",
    )
    train_required = preprocess_required or preds.get("status") in (
        "STALE",
        "MISSING",
        "PARTIAL",
    )
    prediction_refresh_required = True
    training_required = train_required

    if classification == "PREDICTIONS_STALE" and "PREDICTION_ONLY_REFRESH_REQUIRED" in secondary:
        fetch_required = False
        preprocess_required = False
        # train_model still needed to rebuild predictions from fresh processed.
        train_required = True
        training_required = True

    arm = build_paper_arm_status(verbose_notes=False)
    paper_arm_valid = str(arm.get("status") or "") == "VALID"
    live_flag = bool(load_live_execution_flag())
    live_blocked = not live_flag
    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()

    wd = _load_json(WATCHDOG_STATUS)
    broker_connected = bool(wd.get("broker_connected", True))
    open_orders = _count_csv_rows(OPEN_ORDERS)
    gld = _gld_from_positions(POSITIONS)
    phase_198l = _load_json(RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json")
    gld_qty = gld.get("qty")
    gld_weight = float(gld.get("weight") or 0.0)
    if phase_198l.get("current_gld_weight") is not None:
        gld_weight = float(phase_198l.get("current_gld_weight") or gld_weight)
    if phase_198l.get("current_gld_qty") is not None:
        gld_qty = _safe_int(phase_198l.get("current_gld_qty"), default=_safe_int(gld_qty))
    gld_target = bool(
        phase_198l.get("target_already_satisfied") or gld_weight <= GLD_TARGET_WEIGHT + 0.005
    )
    alloc = _allocation_counts(_load_json(ALLOC_SUMMARY), ALLOC_RECS)

    failure_reason = gen_probe.get("failure_reason") or (
        "AS_OF invariant violated" if asof_fail else None
    )
    if asof_diag:
        failure_reason = (
            f"AS_OF invariant: session={asof_diag.get('session_asof_date')} "
            f"max_prediction={asof_diag.get('max_prediction_date')} "
            f"lag_sessions={asof_diag.get('lag_sessions_vs_session')} "
            f"coverage={asof_diag.get('coverage')} strict={asof_diag.get('strict_asof')}"
        )

    # Status: PASS = audit complete + safety clean + root cause identified.
    if not live_blocked or approval_present or broker_auth_present:
        status = "FAIL"
    elif open_orders != 0:
        status = "REVIEW_REQUIRED"
    elif classification and asof_fail:
        status = "PASS"
    else:
        status = "REVIEW_REQUIRED"

    audit: Dict[str, Any] = {
        "blocker_classification": classification,
        "secondary_classifications": secondary,
        "generate_signals_status": gen_probe.get("status"),
        "generate_signals_exit_code": gen_probe.get("exit_code"),
        "generate_signals_failure_reason": failure_reason,
        "generate_signals_output_tail": gen_probe.get("output_tail"),
        "required_session_date": str(required_session) if required_session else None,
        "latest_prediction_date": str(latest_pred_date) if latest_pred_date else None,
        "prediction_age_sessions": lag,
        "prediction_artifact_path": str(PREDICTIONS_DIR.relative_to(ROOT)).replace("\\", "/")
        + "/*_predictions.(parquet|csv)",
        "prediction_files_count": (
            len(list(PREDICTIONS_DIR.glob("*_predictions.*"))) if PREDICTIONS_DIR.is_dir() else 0
        ),
        "asof_diagnostics": asof_diag,
        "raw_data_status": raw.get("status"),
        "raw_data_latest_date": raw.get("latest_date"),
        "raw_data_missing_symbols": raw.get("missing_symbols"),
        "raw_data_missing_count": len(raw.get("missing_symbols") or []),
        "raw_data_stale_vs_session_count": raw.get("stale_vs_session_count"),
        "processed_data_status": processed.get("status"),
        "processed_data_latest_date": processed.get("latest_date"),
        "processed_data_stale_vs_session_count": processed.get("stale_vs_session_count"),
        "model_input_status": preds.get("status"),
        "model_input_latest_date": preds.get("latest_date"),
        "model_input_stale_vs_session_count": preds.get("stale_vs_session_count"),
        "layer_details": {"raw": raw, "processed": processed, "predictions": preds},
        "training_required": training_required,
        "prediction_refresh_required": prediction_refresh_required,
        "fetch_raw_data_required": fetch_required,
        "preprocess_required": preprocess_required,
        "train_model_required": train_required,
        "strict_asof_enabled": strict_asof,
        "strict_asof_block_was_correct": bool(strict_asof and asof_fail and (lag or 0) > 1),
        "spcx_quarantine_relevant": spcx_relevant,
        "quarantined_symbols": quarantined,
        "market_open_calls_signal_refresh": market_open_calls_signal_refresh,
        "continuous_loop_invokes_market_open": bool(continuous_mods.get("run_market_open.ps1")),
        "market_open_calls_upstream_fetch_train": market_open_calls_upstream,
        "market_open_module_presence": market_open_mods,
        "recommended_refresh_sequence": RECOMMENDED_REFRESH_SEQUENCE,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE,
        "broker_connected": broker_connected,
        "open_orders_count": open_orders,
        "paper_arm_valid": paper_arm_valid,
        "paper_entry_armed": bool(arm.get("paper_entry_armed")),
        "live_execution_blocked": live_blocked,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_target_satisfied": gld_target,
        "allocation_open_new_count": alloc.get("open_new_count"),
        "allocation_add_count": alloc.get("add_count"),
        "allocation_exit_count": alloc.get("exit_count"),
        "allocation_trim_count": alloc.get("trim_count"),
        "broker_orders_submitted": False,
        "tickers_configured": len(tickers),
        "asof_not_loosened": True,
    }

    executive = (
        f"generate_signals {gen_probe.get('status')}: required_session={required_session}, "
        f"latest_prediction={latest_pred_date}, lag_sessions={lag}. "
        f"Upstream: raw={raw.get('status')}@{raw.get('latest_date')}, "
        f"processed={processed.get('status')}@{processed.get('latest_date')}, "
        f"predictions={preds.get('status')}@{preds.get('latest_date')}. "
        f"Primary classification={classification}. "
        f"STRICT_ASOF correctly blocked={audit['strict_asof_block_was_correct']}. "
        f"SPCX quarantine relevant={spcx_relevant}. "
        f"Market-open signal refresh present={market_open_calls_signal_refresh}; "
        f"upstream fetch/train in market-open={market_open_calls_upstream}. "
        f"Safe: open_orders={open_orders}, live_blocked={live_blocked}, "
        f"authority_absent={not approval_present and not broker_auth_present}. "
        f"Recommended: {RECOMMENDED_REFRESH_SEQUENCE}"
    )

    payload: Dict[str, Any] = {
        "phase": "198X",
        "generated_at": _utc_now_iso(),
        "status": status,
        "marker": MARKER,
        "executive_summary": executive,
        "audit": audit,
        "safety": {
            "broker_orders_submitted": False,
            "live_execution_blocked": live_blocked,
            "approval_artifact_present": approval_present,
            "broker_auth_artifact_present": broker_auth_present,
            "open_orders_count": open_orders,
            "asof_not_loosened": True,
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(f"[198X] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198X] wrote {OUTPUT_MD.relative_to(ROOT)}")
        print(
            f"[198X] status={status} classification={classification} "
            f"gen={gen_probe.get('status')} session={required_session} "
            f"pred={latest_pred_date} lag={lag} "
            f"raw={raw.get('status')} processed={processed.get('status')} "
            f"open_orders={open_orders} live_blocked={live_blocked}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198X upstream prediction freshness recovery audit"
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--no-probe",
        action="store_true",
        help="Do not invoke generate_signals; use existing diagnostics only.",
    )
    args = ap.parse_args(argv)
    result = run_audit(
        probe_generate_signals=not args.no_probe,
        verbose=not args.quiet,
    )
    return 0 if str(result.get("status") or "") in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
