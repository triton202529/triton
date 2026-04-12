# run_full_pipeline.py — TRITON Phase 1.5 Runner
# ---------------------------------------------------
# Goals:
#   - Run full pipeline from ONE command
#   - Always use the active venv (sys.executable)
#   - Robust path/module discovery (root vs services/)
#   - Write heartbeat.json at each stage (no need to manually create)
#   - Windows-safe UTF-8 console for emoji/status printing
#
# Notes (Phase 1.5):
#   - Optional stages are skipped safely if missing
#
# Patch (2026-01-16):
#   ✅ Fix "scores" stage to use services.score_stocks (NOT services.build_stock_scores)
#      because build_stock_scores does not exist in this repo.
#
# Patch (2026-01-16b):
#   ✅ Add AS_OF Contract Check (internal stage):
#      - Verifies the pipeline artifacts agree on AS_OF (signals/freshness/portfolio/predictions)
#      - Writes data/results/asof_contract.json for dashboards + audits
#      - Fails the run if mismatch is detected (capital preservation / freshness discipline)
#
# Patch (2026-01-21):
#   ✅ End-of-run "Dashboard Ready" block (required if present):
#      - Snapshot live orders/positions -> data/results/{live_orders,recent_orders,positions_snapshot}.csv
#      - Normalize results contracts -> ensures required columns exist (e.g., positions_snapshot.date/value/symbol/ticker)
#      - Validate contracts -> FAILS pipeline if contracts fail
#
# Patch (2026-01-22):
#   ✅ Add "fetch_raw" stage FIRST (required if present):
#      - Auto-recovers from yfinance timeouts via retries/timeout
#      - Enforces min-ok and min-asof-ok BEFORE preprocessing/training
#      - Makes daily pipeline “hands-free” again
#
# Patch (2026-04-01):
#   ✅ Add "lifecycle" stage after rationale:
#      - Automatically regenerates data/results/signal_lifecycle.csv
#      - Keeps lifecycle fresh for dashboards + execution
#      - Removes the last manual gap between rationale and execution
#
# Patch (2026-04-04):
#   ✅ Improve failure visibility:
#      - Print stdout/stderr more aggressively on subprocess failure
#      - Capture full Python traceback into heartbeat error field
#      - Print traceback to console so stage failures are no longer opaque

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd


# ──────────────────────────────
# Console-safe printing (Windows scheduled tasks often run cp1252)
# ──────────────────────────────
def safe_print(msg: str, *, file=None) -> None:
    """
    Print safely on consoles that can't encode unicode (e.g., cp1252).
    Replaces unencodable characters instead of crashing the pipeline.
    """
    file = file or sys.stdout
    try:
        print(msg, file=file)
    except UnicodeEncodeError:
        enc = getattr(file, "encoding", None) or getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = msg.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe, file=file)


# ──────────────────────────────
# Paths
# ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEARTBEAT_PATH = RESULTS_DIR / "heartbeat.json"
PIPELINE_STATUS_PATH = RESULTS_DIR / "pipeline_status.json"  # fallback compatibility

ASOF_CONTRACT_PATH = RESULTS_DIR / "asof_contract.json"


# ──────────────────────────────
# Heartbeat writer (standalone, no imports)
# ──────────────────────────────
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_heartbeat(
    status: str,
    stage: str,
    message: str = "",
    error: str = "",
    last_success_stage: str = "",
    run_id: str = "",
) -> None:
    """
    Writes BOTH heartbeat.json and pipeline_status.json (compat).
    Avoids dependency on services imports during early boot.
    """
    payload: Dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "status": status,  # ok|warn|fail
        "stage": stage,
        "last_success_stage": last_success_stage or "",
        "message": message or "",
        "error": error or "",
        "run_id": run_id or "",
        "host": os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or "",
        "python": sys.executable,
        "cwd": str(PROJECT_ROOT),
    }

    for p in (HEARTBEAT_PATH, PIPELINE_STATUS_PATH):
        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception:
            try:
                p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                pass


# ──────────────────────────────
# Discovery
# ──────────────────────────────
def find_script(name: str) -> Optional[Path]:
    """
    Find a python script in common locations.
    Returns absolute Path or None.
    """
    candidates = [
        PROJECT_ROOT / name,
        PROJECT_ROOT / "services" / name,
        PROJECT_ROOT / "pipelines" / name,
        PROJECT_ROOT / "scripts" / name,
        PROJECT_ROOT / "jobs" / name,
        PROJECT_ROOT / "tools" / name,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    return None


def module_for_script(path: Path) -> Optional[str]:
    """
    If script lives under a package-like folder (e.g., services/train_model.py),
    prefer running it as a module: python -m services.train_model

    Only returns a module path if that folder has __init__.py.
    """
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except Exception:
        return None

    parts = list(rel.parts)
    if len(parts) >= 2 and parts[0] in ("services", "pipelines", "scripts", "tools"):
        pkg = parts[0]
        mod = Path(*parts).with_suffix("").as_posix().replace("/", ".")
        init_ok = (PROJECT_ROOT / pkg / "__init__.py").exists()
        return mod if init_ok else None

    return None


def best_target(target: str) -> Tuple[str, Optional[Path], Optional[str]]:
    """
    Resolve a target that may be:
      - a script file name (e.g., train_model.py)
      - a module path (e.g., services.train_model)

    Returns:
      ("module", None, "services.train_model") OR ("script", path, None)
    """
    t = (target or "").strip()

    # If user passed an explicit module name
    if "." in t and not t.lower().endswith(".py"):
        return "module", None, t

    # Otherwise treat as script
    if not t.lower().endswith(".py"):
        t = t + ".py"

    script_path = find_script(t)
    if script_path:
        mod = module_for_script(script_path)
        if mod:
            return "module", None, mod
        return "script", script_path, None

    return "script", None, None


# ──────────────────────────────
# Internal: AS_OF Contract Check
# ──────────────────────────────
def _to_date_safe(x: Any) -> Optional[date]:
    if x is None:
        return None
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text())
        except Exception:
            return None


def _max_date_from_df(df: pd.DataFrame, cols: List[str]) -> Optional[date]:
    for c in cols:
        if c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce")
            d = d.dropna()
            if not d.empty:
                return d.max().date()
    return None


def internal_asof_contract_check(verbose: bool = False) -> None:
    """
    Verifies AS_OF consistency across key artifacts:
      - signals_with_rationale.csv (as_of_date OR max(date))
      - portfolio_history.csv (max(date))
      - heartbeat.json timestamp (informational)
      - predictions parquet files (max(date) or max(ds) if present)

    Writes: data/results/asof_contract.json

    Fails if:
      - required artifacts are missing OR
      - portfolio/signals/predictions disagree on AS_OF
    """
    signals_path = RESULTS_DIR / "signals_with_rationale.csv"
    portfolio_path = RESULTS_DIR / "portfolio_history.csv"

    signals_df = _read_csv_safe(signals_path)
    portfolio_df = _read_csv_safe(portfolio_path)

    missing_required: List[str] = []
    if signals_df is None:
        missing_required.append(str(signals_path))
    if portfolio_df is None:
        missing_required.append(str(portfolio_path))

    if missing_required:
        payload = {
            "timestamp_utc": _utc_now_iso(),
            "ok": False,
            "error": "Missing required artifacts for AS_OF contract check.",
            "missing": missing_required,
        }
        try:
            ASOF_CONTRACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        raise RuntimeError(f"AS_OF contract check failed: missing {', '.join(missing_required)}")

    # --- Determine signals AS_OF ---
    signals_asof: Optional[date] = None
    if "as_of_date" in signals_df.columns:
        s = signals_df["as_of_date"].dropna()
        if not s.empty:
            signals_asof = _to_date_safe(s.iloc[0])

    if signals_asof is None:
        signals_asof = _max_date_from_df(signals_df, ["date", "ds", "timestamp", "datetime"])

    # --- Determine portfolio AS_OF ---
    portfolio_asof = _max_date_from_df(portfolio_df, ["date", "ds", "timestamp", "datetime"])

    # --- Predictions AS_OF (should not be behind) ---
    pred_dir = DATA_ROOT / "predictions"
    pred_files: List[Path] = (
        sorted(pred_dir.glob("*_predictions.parquet")) if pred_dir.exists() else []
    )

    stale_preds: List[Dict[str, Any]] = []
    pred_asof_max: Optional[date] = None

    for p in pred_files:
        ticker = p.name.replace("_predictions.parquet", "")
        try:
            dfp = pd.read_parquet(p)
        except Exception:
            continue

        pmax = _max_date_from_df(dfp, ["date", "ds", "timestamp", "datetime"])
        if pmax is None:
            continue

        if pred_asof_max is None or pmax > pred_asof_max:
            pred_asof_max = pmax

        if signals_asof is not None and pmax < signals_asof:
            stale_preds.append(
                {
                    "ticker": ticker,
                    "latest": str(pmax),
                    "expected": str(signals_asof),
                    "file": str(p),
                }
            )

    expected_asof = signals_asof or portfolio_asof or pred_asof_max

    problems: List[str] = []
    if expected_asof is None:
        problems.append("Could not determine expected AS_OF from signals/portfolio/predictions.")
    if signals_asof is None:
        problems.append(
            "signals_with_rationale.csv missing 'as_of_date' and no parsable date column found."
        )
    if portfolio_asof is None:
        problems.append("portfolio_history.csv has no parsable date column.")

    if expected_asof is not None:
        if signals_asof is not None and signals_asof != expected_asof:
            problems.append(
                f"Signals AS_OF mismatch: signals={signals_asof} expected={expected_asof}"
            )
        if portfolio_asof is not None and portfolio_asof != expected_asof:
            problems.append(
                f"Portfolio AS_OF mismatch: portfolio={portfolio_asof} expected={expected_asof}"
            )

    ok = (len(problems) == 0) and (len(stale_preds) == 0)

    heartbeat = _read_json_safe(HEARTBEAT_PATH) or {}
    payload = {
        "timestamp_utc": _utc_now_iso(),
        "ok": ok,
        "expected_as_of": str(expected_asof) if expected_asof else None,
        "signals_as_of": str(signals_asof) if signals_asof else None,
        "portfolio_as_of": str(portfolio_asof) if portfolio_asof else None,
        "predictions_max_as_of": str(pred_asof_max) if pred_asof_max else None,
        "stale_prediction_files": stale_preds[:200],
        "problems": problems,
        "heartbeat_stage": heartbeat.get("stage"),
        "heartbeat_status": heartbeat.get("status"),
        "heartbeat_timestamp": heartbeat.get("timestamp"),
        "artifacts": {
            "signals_with_rationale": str(signals_path),
            "portfolio_history": str(portfolio_path),
            "heartbeat": str(HEARTBEAT_PATH),
        },
    }

    try:
        ASOF_CONTRACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    if not ok:
        msg = "AS_OF contract check failed."
        if problems:
            msg += " " + "; ".join(problems[:6])
        if stale_preds:
            msg += f" Stale predictions: {len(stale_preds)} file(s) behind expected AS_OF."
        raise RuntimeError(msg)

    if verbose:
        safe_print(f"✅ AS_OF contract OK: expected_as_of={expected_asof}")


# ──────────────────────────────
# Subprocess runner
# ──────────────────────────────
def run_step(
    label: str,
    target: str,
    verbose: bool = False,
    extra_args: Optional[List[str]] = None,
) -> None:
    """
    Run a pipeline step by script name/module path, OR internal stage.

    Windows fix:
      - Decode captured output as UTF-8 (errors=replace)
      - Use safe_print for any unicode status output.
    """
    extra_args = extra_args or []

    # Internal stage routing
    if isinstance(target, str) and target.startswith("__internal__:"):
        name = target.split(":", 1)[1].strip().lower()
        if name == "asof_contract":
            internal_asof_contract_check(verbose=verbose)
            return
        raise FileNotFoundError(f"Unknown internal stage target: {target}")

    kind, script_path, mod = best_target(target)

    if kind == "module":
        if not mod:
            raise FileNotFoundError(f"Could not resolve module target: {target}")
        cmd: List[str] = [sys.executable, "-m", mod] + extra_args
    else:
        if not script_path:
            raise FileNotFoundError(
                f"Could not find {target} in project (root/services/pipelines/scripts/tools)."
            )
        cmd = [sys.executable, str(script_path)] + extra_args

    env = os.environ.copy()

    # Ensure project root importable
    sep = ";" if os.name == "nt" else ":"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + sep + env.get("PYTHONPATH", "")

    # Encourage children to emit UTF-8
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if verbose:
        safe_print(f"▶ {label}: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    # Print child output in verbose mode even on success, so pipeline logs are more transparent
    if verbose:
        if out:
            safe_print(out)
        if err:
            safe_print(err, file=sys.stderr)

    if proc.returncode != 0:
        if out:
            safe_print(out)
        if err:
            safe_print(err, file=sys.stderr)
        raise RuntimeError(f"{label} failed (return code {proc.returncode})")


# ──────────────────────────────
# Main
# ──────────────────────────────
Stage = Tuple[str, str, bool, List[str]]  # (stage_name, target, required, extra_args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="Print commands.")
    ap.add_argument("--run-id", default="", help="Optional run id to stamp into heartbeat.")
    ap.add_argument(
        "--min-rows",
        type=int,
        default=int(os.getenv("TRITON_MIN_ROWS", "30")),
        help="Passed to train step as --min-rows (default: TRITON_MIN_ROWS or 30).",
    )
    ap.add_argument(
        "--broker-mode",
        choices=["paper", "live"],
        default=(os.getenv("TRITON_BROKER_MODE", "paper") or "paper").strip().lower(),
        help="Broker mode for snapshot_live_orders (default: TRITON_BROKER_MODE or 'paper').",
    )

    # NEW: fetch_raw controls (so “one command” also hardens yfinance timeouts)
    ap.add_argument(
        "--fetch-start",
        default=os.getenv("TRITON_FETCH_START", "2020-01-01"),
        help="Start date for fetch_raw_data (default: TRITON_FETCH_START or 2020-01-01).",
    )
    ap.add_argument(
        "--fetch-end",
        default=os.getenv("TRITON_FETCH_END", ""),
        help="End date (exclusive) for fetch_raw_data. Blank => module default (tomorrow UTC).",
    )
    ap.add_argument(
        "--fetch-retries",
        type=int,
        default=int(os.getenv("TRITON_FETCH_RETRIES", "8")),
        help="Retries for fetch_raw_data (default: TRITON_FETCH_RETRIES or 8).",
    )
    ap.add_argument(
        "--fetch-timeout",
        type=int,
        default=int(os.getenv("TRITON_FETCH_TIMEOUT", "60")),
        help="Timeout seconds per yfinance call (default: TRITON_FETCH_TIMEOUT or 60).",
    )
    ap.add_argument(
        "--min-ok",
        type=int,
        default=int(os.getenv("TRITON_FETCH_MIN_OK", "50")),
        help="Min ok downloads for fetch_raw_data (default: TRITON_FETCH_MIN_OK or 50).",
    )
    ap.add_argument(
        "--min-asof-ok",
        type=int,
        default=int(os.getenv("TRITON_FETCH_MIN_ASOF_OK", "50")),
        help="Min tickers reaching AS_OF for fetch_raw_data (default: TRITON_FETCH_MIN_ASOF_OK or 50).",
    )
    ap.add_argument(
        "--fetch-allow-data-lag-days",
        type=int,
        default=int(os.getenv("TRITON_FETCH_ALLOW_DATA_LAG_DAYS", "2")),
        help="Max calendar days raw bars may trail AS_OF and still pass (default: TRITON_FETCH_ALLOW_DATA_LAG_DAYS or 2).",
    )
    ap.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetch_raw stage even if services.fetch_raw_data exists.",
    )
    args = ap.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Detect optional scripts
    fetch_raw_exists = bool(
        (PROJECT_ROOT / "services" / "fetch_raw_data.py").exists()
        or (PROJECT_ROOT / "fetch_raw_data.py").exists()
    )

    adaptive_state_script_exists = bool(
        (PROJECT_ROOT / "services" / "write_adaptive_risk_state.py").exists()
        or (PROJECT_ROOT / "write_adaptive_risk_state.py").exists()
    )

    snapshot_live_orders_exists = bool(
        (PROJECT_ROOT / "services" / "snapshot_live_orders.py").exists()
        or (PROJECT_ROOT / "snapshot_live_orders.py").exists()
    )
    normalize_contracts_exists = bool(
        (PROJECT_ROOT / "services" / "normalize_results_contracts.py").exists()
        or (PROJECT_ROOT / "normalize_results_contracts.py").exists()
    )
    validate_contracts_exists = bool(
        (PROJECT_ROOT / "tools" / "validate_contracts.py").exists()
        or (PROJECT_ROOT / "validate_contracts.py").exists()
    )

    # Build fetch args (only if stage runs)
    fetch_args: List[str] = []
    if fetch_raw_exists and (not args.skip_fetch):
        fetch_args = [
            "--start",
            str(args.fetch_start),
            "--retries",
            str(args.fetch_retries),
            "--timeout",
            str(args.fetch_timeout),
            "--min-ok",
            str(args.min_ok),
            "--min-asof-ok",
            str(args.min_asof_ok),
            "--allow-data-lag-days",
            str(args.fetch_allow_data_lag_days),
        ]
        if args.verbose:
            fetch_args.append("--verbose")
        if args.fetch_end and str(args.fetch_end).strip():
            fetch_args += ["--end", str(args.fetch_end).strip()]

    # Core pipeline (required)
    # NOTE: fetch_raw is REQUIRED IF PRESENT (so “green” implies data freshness discipline)
    stages: List[Stage] = [
        (
            "fetch_raw",
            "services.fetch_raw_data",
            bool(fetch_raw_exists and (not args.skip_fetch)),
            fetch_args,
        ),
        ("preprocess", "run_preprocessing.py", True, []),
        (
            "fundamentals",
            "services.fetch_fundamentals",
            True,
            ["--verbose"] if args.verbose else [],
        ),
        ("scores", "services.score_stocks", True, []),
        ("train", "services.train_model", True, ["--min-rows", str(args.min_rows)]),
        ("signals", "services.generate_signals", True, []),
        ("backtest", "services.backtest_signals", True, []),
        # Optional (safe skips)
        ("rationale", "services.trade_rationale", False, []),
        # ✅ NEW: lifecycle state generation (authoritative state layer)
        ("lifecycle", "services.apply_signal_lifecycle", False, []),
        ("snapshot", "services.write_guard_snapshot", False, []),
        # Optional Phase 1.5+ risk artifacts
        ("regime", "services.regime_detector", False, ["--verbose"] if args.verbose else []),
        ("risk_report", "services.generate_risk_report", False, []),
        # Adaptive risk state: strongly recommended (required if present)
        (
            "adaptive_state",
            "services.write_adaptive_risk_state",
            bool(adaptive_state_script_exists),
            ["--verbose"] if args.verbose else [],
        ),
        # Optional smoke check: read risk state (no broker calls)
        ("risk_gate_smoke", "services.risk_gate", False, []),
        # ✅ AS_OF Contract Check (required)
        ("asof_contract", "__internal__:asof_contract", True, []),
        # ✅ Dashboard-ready tail (required if present)
        (
            "snapshot_live_orders",
            "services.snapshot_live_orders",
            bool(snapshot_live_orders_exists),
            ["--mode", args.broker_mode],
        ),
        ("reconcile_lifecycle", "services.reconcile_lifecycle_vs_positions", False, []),
        ("build_effective_lifecycle", "services.build_effective_lifecycle", False, []),
        ("build_trade_opportunities", "services.build_trade_opportunities", False, []),
        (
            "normalize_results_contracts",
            "services.normalize_results_contracts",
            bool(normalize_contracts_exists),
            [],
        ),
        ("validate_contracts", "validate_contracts.py", bool(validate_contracts_exists), []),
    ]

    last_success = ""

    for stage, target, required, extra_args in stages:
        try:
            write_heartbeat(
                "ok",
                stage=stage,
                message=f"Starting {stage}...",
                last_success_stage=last_success,
                run_id=run_id,
            )

            run_step(stage, target, verbose=args.verbose, extra_args=extra_args)

            last_success = stage
            write_heartbeat(
                "ok",
                stage=stage,
                message=f"{stage} complete.",
                last_success_stage=last_success,
                run_id=run_id,
            )

        except FileNotFoundError as e:
            if not required:
                write_heartbeat(
                    "warn",
                    stage=stage,
                    message=f"Optional stage missing: {target}",
                    error=str(e),
                    last_success_stage=last_success,
                    run_id=run_id,
                )
                if args.verbose:
                    safe_print(f"⚠️  Skipping optional stage '{stage}' (missing {target})")
                continue

            write_heartbeat(
                "fail",
                stage=stage,
                message=f"Missing required stage target: {target}",
                error=str(e),
                last_success_stage=last_success,
                run_id=run_id,
            )
            safe_print(f"❌ Failed at {stage}: {e}")
            return 1

        except Exception as e:
            tb = traceback.format_exc()

            # Extra hint if fetch fails (most common “hands-free” breaker)
            hint = ""
            if stage == "fetch_raw":
                hint = (
                    "Hint: fetch_raw failed (usually transient yfinance timeout). "
                    "Try re-run pipeline, or run only:\n"
                    "  python -m services.fetch_raw_data --retries 8 --timeout 60 --verbose --min-ok 1 --min-asof-ok 1\n"
                )

            write_heartbeat(
                "fail",
                stage=stage,
                message=f"Failed at {stage}",
                error=f"{str(e)}\n\nTRACEBACK:\n{tb}",
                last_success_stage=last_success,
                run_id=run_id,
            )
            safe_print(f"❌ Failed at {stage}: {e}")
            safe_print(tb)
            if hint:
                safe_print(hint)
            return 1

    write_heartbeat(
        "ok",
        stage="done",
        message="Full pipeline completed (dashboard-ready).",
        last_success_stage=last_success,
        run_id=run_id,
    )
    safe_print("✅ Full pipeline completed (dashboard-ready).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
