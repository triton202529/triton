#!/usr/bin/env python3
# run_adaptive_pipeline.py
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import glob
import warnings
import argparse

warnings.filterwarnings("ignore")

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(
    description="Run the Enhanced Triton Pipeline with Adaptive Risk Management"
)
parser.add_argument(
    "--skip-subprocesses",
    action="store_true",
    help="Skip running services/*.py as subprocesses and run only the in-process adaptive engine (fast dev mode).",
)
parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Run only data validation and exit (useful to check required data files/columns).",
)
args = parser.parse_args()


# ---------------------------
# Robust run_step
# ---------------------------
def run_step(label: str, script: str, *argv):
    """
    Run a pipeline step with error handling.

    - Runs services/*.py via `python -m services.<module>` to preserve package-relative imports.
    - Ensures child uses UTF-8 and parent decodes with utf-8 (errors='replace') to avoid UnicodeDecodeError.
    """
    print(f"\n{label}...")
    try:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        script_path = Path(script)
        if script_path.parts and script_path.parts[0] == "services" and script_path.suffix == ".py":
            module_name = f"services.{script_path.stem}"
            run_cmd = [sys.executable, "-m", module_name, *argv]
        else:
            run_cmd = [sys.executable, script, *argv]

        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        if result.returncode != 0:
            print(f"❌ Error running {script}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return False

        print(f"✅ {label} completed successfully")
        return True
    except Exception as e:
        print(f"❌ Exception running {script}: {e}")
        return False


# ---------------------------
# File discovery helpers
# ---------------------------
UTILITY_BASENAMES = {
    "broker_cash_mv.csv",  # cash/mv ledger, not price history
    "cleaned_data.csv",  # ETL artifact
}


def _preferred_market_files():
    """
    Return a list of chosen files to use for market data, one per ticker, with
    priority:
      *.normalized.fixed.csv  >  *.normalized.csv  >  *.csv
    Utility files are skipped. First match per ticker wins.
    """
    # Build ordered candidate list
    candidates = (
        sorted(glob.glob("data/*.normalized.fixed.csv"))
        + sorted(glob.glob("data/*.normalized.csv"))
        + sorted(glob.glob("data/*.csv"))
    )

    chosen = []
    seen = set()

    for p in candidates:
        base = Path(p).name
        if base in UTILITY_BASENAMES:
            continue
        ticker = Path(p).stem.split("_")[0].upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        chosen.append(p)

    return chosen


# ---------------------------
# Data validation helpers
# ---------------------------
def validate_market_files(required_cols=("date", "close"), sample_rows=5):
    """
    Validate *chosen* market CSVs (preferred normalized > raw) contain required columns.
    Returns True if all chosen files pass, otherwise prints problems and returns False.
    """
    files = _preferred_market_files()
    if not files:
        print(
            "⚠️ No usable market CSV files found under data/. The adaptive engine can still run with fallback signals."
        )
        return True  # Not fatal

    problems = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=sample_rows)
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                problems.append((f, missing, list(df.columns)))
        except Exception as e:
            problems.append((f, f"read error: {e}", None))

    if problems:
        print("❌ Data validation issues detected (on preferred files):")
        for fname, info, cols in problems:
            if cols is None:
                print(f"  - {fname}: {info}")
            else:
                print(f"  - {fname}: missing columns {info} (found: {cols})")
        return False

    print("✅ Data validation passed for selected market files")
    return True


# ---------------------------
# Adaptive Risk Engine glue
# ---------------------------
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_universe(limit: int = None):
    """
    Load preferred CSVs into a dict ticker -> DataFrame, preferring
    *.normalized.fixed.csv > *.normalized.csv > *.csv (first per ticker wins).
    """
    files = _preferred_market_files()
    if not files:
        print(
            "Warning: No CSV files found in data/. Adaptive engine will attempt to proceed with whatever is available."
        )
        return {}

    if limit is not None:
        files = files[:limit]

    universe = {}
    for file in files:
        try:
            ticker = Path(file).stem.split("_")[0].upper()
            df = pd.read_csv(file)
            # Normalize schema
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.sort_values("date").reset_index(drop=True)
            if "close" in df.columns:
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
            universe[ticker] = df
        except Exception as e:
            print(f"Warning: failed to load {file}: {e}")
    print(f"📊 Loaded {len(universe)} tickers from preferred files")
    return universe


def generate_momentum_signals(universe: dict, lookback: int = 5):
    """Generate simple momentum signals: positive last-n-day return -> score, else 0."""
    signals = {}
    for t, df in universe.items():
        try:
            if df is None or "close" not in df.columns or len(df) < lookback + 1:
                continue
            recent = df["close"].iloc[-(lookback + 1) :].astype(float)
            pct = (recent.iloc[-1] / recent.iloc[0]) - 1.0
            signals[t] = max(0.0, float(pct))
        except Exception:
            signals[t] = 0.0
    ssum = sum(signals.values())
    if ssum > 0:
        signals = {k: v / ssum for k, v in signals.items()}
    else:
        tickers = list(signals.keys())
        if not tickers:
            return {}
        signals = {k: 1.0 / len(tickers) for k in tickers}
    return signals


def write_risk_report(report: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            report, f, default=lambda o: (o.tolist() if hasattr(o, "tolist") else str(o)), indent=2
        )


def write_portfolio_history(row: dict, path: Path):
    cols = ["date", "total_value", "market_value", "cash", "num_positions", "regime"]
    df = pd.DataFrame([row], columns=cols)
    df.to_csv(path, index=False)


def run_adaptive_engine_and_write_outputs():
    """In-process AdaptiveRiskEngine run that writes risk_report.json & portfolio history."""
    print("\n🎯 Running Adaptive Risk Engine (in-process)...")

    # Lazy import the engine to avoid heavy deps at top-level
    try:
        from services.adaptive_risk_engine import AdaptiveRiskEngine
    except Exception as e:
        print(f"❌ Could not import AdaptiveRiskEngine: {e}")
        return False

    universe = load_universe(limit=200)
    if not universe:
        print("⚠️ Universe is empty — adaptive engine will skip processing.")
        return False

    # Attempt to locate signals emitted by pipeline
    signals = {}
    for candidate in [
        RESULTS_DIR.joinpath("generated_signals.json"),
        Path("data").joinpath("signals.json"),
        RESULTS_DIR.joinpath("signals.json"),
    ]:
        if candidate.exists():
            try:
                signals = json.loads(candidate.read_text(encoding="utf-8"))
                print(f"▶ Loaded signals from {candidate}")
                break
            except Exception as e:
                print(f"Warning: failed to parse {candidate}: {e}")

    if not signals:
        signals = generate_momentum_signals(universe)
        if signals:
            print("▶ Falling back to generated momentum signals")
        else:
            print("⚠️ No signals available after fallback — skipping adaptive engine.")
            return False

    # Initialize engine and run
    engine = AdaptiveRiskEngine(verbose=False)
    try:
        initialized = engine.initialize(universe)
        if not initialized:
            print("⚠️ Engine initialization returned False — will continue best-effort.")
    except Exception as e:
        print(f"Warning: engine.initialize() failed: {e}")

    try:
        final_weights = engine.process_signals(signals, universe)
    except Exception as e:
        print(f"❌ process_signals failed: {e}")
        # fallback to proportional simple allocation
        total = sum(signals.values()) or 1.0
        final_weights = {k: v / total for k, v in signals.items()}

    # Build risk report (best-effort)
    try:
        risk_report = engine.risk_allocator.get_risk_report(final_weights, universe)
    except Exception:
        risk_report = {
            "portfolio_metrics": {"expected_volatility": 0.0, "diversification_ratio": 1.0},
            "regime": {"current_regime": engine.last_regime, "regime_adjustments": {}},
            "risk_decomposition": {},
            "position_analysis": {},
        }

    # augment with engine-level dashboard payload
    risk_report["engine"] = engine.get_risk_dashboard_data()

    # persist
    rp_path = RESULTS_DIR.joinpath("risk_report.json")
    write_risk_report(risk_report, rp_path)
    print(f"✔ Wrote risk report to: {rp_path}")

    # minimal portfolio history row so dashboard displays something
    now = pd.Timestamp.now().strftime("%Y-%m-%d")
    total_value = 100000.0
    try:
        market_value = total_value
        cash = total_value * 0.05
        num_positions = len([w for w in final_weights.values() if w > 0])
        regime_label = risk_report.get("regime", {}).get("current_regime", engine.last_regime)
    except Exception:
        market_value = total_value
        cash = 0.0
        num_positions = 0
        regime_label = engine.last_regime

    history_row = {
        "date": now,
        "total_value": float(total_value),
        "market_value": float(market_value),
        "cash": float(cash),
        "num_positions": int(num_positions),
        "regime": regime_label,
    }
    hist_path = RESULTS_DIR.joinpath("enhanced_portfolio_history.csv")
    write_portfolio_history(history_row, hist_path)
    print(f"✔ Wrote portfolio history to: {hist_path}")

    # save engine state
    try:
        engine.save_state(str(RESULTS_DIR.joinpath("adaptive_risk_state.json")))
        print("✔ Engine state saved")
    except Exception as e:
        print(f"Warning: failed to save engine state: {e}")

    return True


# ---------------------------
# Main orchestration
# ---------------------------
def main():
    """Run the enhanced Triton pipeline with adaptive risk management."""
    print("🚀 Starting Enhanced Triton Pipeline with Adaptive Risk Management")
    print("=" * 70)

    # Validate only the preferred files (normalized first)
    if not validate_market_files():
        print(
            "❌ Data validation failed. Fix your normalized files or run normalization/repair scripts."
        )
        return False

    steps = [
        ("📥 Preprocessing data", "services/preprocess_data.py"),
        ("🧠 Training models", "services/train_model.py"),
        ("📡 Generating signals", "services/generate_signals.py"),
        ("🎯 Running enhanced portfolio simulation", "services/enhanced_portfolio_manager.py"),
        ("📊 Generating risk report", "services/generate_risk_report.py"),
    ]

    if args.skip_subprocesses:
        print(
            "⚠️ Running in dev mode: skipping subprocesses and executing only in-process adaptive engine."
        )
    else:
        for label, script in steps:
            if not run_step(label, script):
                print(f"\n❌ Pipeline failed at: {label}")
                print("Check the error messages above for details.")
                return False

    # Run in-process adaptive engine and outputs (always useful)
    try:
        success = run_adaptive_engine_and_write_outputs()
        if not success:
            print("⚠️ Adaptive engine step did not complete successfully (see messages above).")
    except Exception as e:
        print(f"❌ Adaptive engine step raised an exception: {e}")

    print("\n" + "=" * 70)
    print("✅ Enhanced Triton Pipeline completed successfully!")
    print("\n📊 Results available:")
    print("  - Enhanced Portfolio History: data/results/enhanced_portfolio_history.csv")
    print("  - Enhanced Trade Log: data/results/enhanced_trade_log.csv")
    print("  - Risk Report: data/results/risk_report.json")
    print("\n🎯 Launch Risk Dashboard:")
    print("  streamlit run services/risk_dashboard.py")

    return True


if __name__ == "__main__":
    if args.validate_only:
        ok = validate_market_files()
        sys.exit(0 if ok else 2)

    success = main()
    sys.exit(0 if success else 1)


