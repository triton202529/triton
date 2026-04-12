from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from services.signal_lifecycle import (
    LifecycleConfig,
    apply_lifecycle,
    lifecycle_logic_from_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

SIGNALS_RATIONALE = RESULTS_DIR / "signals_with_rationale.csv"
SIGNALS_FALLBACK = RESULTS_DIR / "signals.csv"
OUT_LIFECYCLE = RESULTS_DIR / "signal_lifecycle.csv"
STATE_PATH = RESULTS_DIR / "signal_state.json"


def load_lifecycle_config() -> dict:
    """Load config/lifecycle_logic.json merged onto defaults (project-root path)."""
    defaults = {
        "enabled": True,
        "add_confidence_min": 0.62,
        "add_delta_pct_min": 0.01,
        "hold_delta_floor": -0.002,
        "hold_delta_ceiling": 0.008,
        "trim_delta_pct_threshold": -0.002,
        "exit_delta_pct_threshold": -0.006,
        "exit_confidence_min": 0.58,
    }
    path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "config", "lifecycle_logic.json"
    )
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if isinstance(user_cfg, dict):
            defaults.update(user_cfg)
    except Exception as e:
        print(f"[CONFIG] Failed to load lifecycle config: {e}")
    return defaults


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    src = None
    if SIGNALS_RATIONALE.exists() and SIGNALS_RATIONALE.stat().st_size > 0:
        src = SIGNALS_RATIONALE
    elif SIGNALS_FALLBACK.exists() and SIGNALS_FALLBACK.stat().st_size > 0:
        src = SIGNALS_FALLBACK

    if src is None:
        print("No signals file found (signals_with_rationale.csv or signals.csv).")
        return 2

    df = pd.read_csv(src)

    lc_cfg = load_lifecycle_config()
    print("[LIFECYCLE CONFIG]", lc_cfg)
    lifecycle_logic = lifecycle_logic_from_dict(lc_cfg)

    engine_cfg = LifecycleConfig(
        min_hold_days=1,
        cooldown_days_after_exit=1,
        buy_delta_pct=0.0015,  # 0.15%
        add_delta_pct=0.0020,  # 0.20%
        exit_delta_pct=-0.0015,  # -0.15%
        trim_delta_pct=-0.0030,  # -0.30%
        hold_means_keep_position=True,
    )

    out_df, state = apply_lifecycle(
        df,
        state_path=STATE_PATH,
        cfg=engine_cfg,
        lifecycle_logic=lifecycle_logic,
    )
    out_df.to_csv(OUT_LIFECYCLE, index=False)

    try:
        from services.signal_pressure_diagnostics import refresh_signal_pressure_diagnostics

        refresh_signal_pressure_diagnostics()
    except Exception:
        pass

    print(f"Lifecycle applied: {src.name} -> {OUT_LIFECYCLE.name}")
    print(f"State persisted: {STATE_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
