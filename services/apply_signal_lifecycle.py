from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
POSITIONS_SNAPSHOT = RESULTS_DIR / "positions_snapshot.csv"


# ──────────────────────────────────────────────────────────────────────
# Lifecycle row normalization
#
# Mirrors services.lifecycle_truth.VALID_EFFECTIVE_PAIRS so the
# signal_lifecycle.csv self-check (which reads `position_state` plus
# `lifecycle_action`/`stance`) cannot fail on values produced here.
#
# Rules (per spec):
#   - position_state must be one of {LONG, FLAT}.
#   - lifecycle_action / stance must be one of
#     {BUY, ADD, HOLD, TRIM, EXIT, WAIT}.
#   - Pair must be one of VALID_PAIRS below.
#
# Normalization policy ("safely where possible, otherwise WAIT"):
#   - Unknown / empty / NaN position_state → FLAT  (no-position default;
#     never causes a sell-side action downstream).
#   - Unknown / empty / NaN action       → WAIT  (explicit per spec).
#   - LONG + BUY                         → HOLD  (BUY on a LONG position
#     would otherwise be ambiguous — we do NOT silently promote to ADD,
#     since ADD has stricter confidence/delta thresholds upstream).
#   - FLAT + ADD/TRIM/EXIT               → WAIT  (no position to act on).
#   - Any remaining invalid pair         → WAIT  (catch-all, never crashes).
#
# Important safety addition:
#   Before lifecycle is rebuilt, signal_state.json is synchronized to the
#   broker position snapshot. This prevents stale LONG memory from converting
#   fresh BUY signals into HOLD after the account has been manually flattened.
#   This is still read-only with respect to broker/execution; it only updates
#   the lifecycle memory file to match the latest local positions_snapshot.csv.
# ──────────────────────────────────────────────────────────────────────
_VALID_POSITION_STATES = frozenset({"LONG", "FLAT"})
_VALID_ACTIONS = frozenset({"BUY", "ADD", "HOLD", "TRIM", "EXIT", "WAIT"})
_VALID_PAIRS: frozenset = frozenset(
    {
        ("FLAT", "BUY"),
        ("FLAT", "WAIT"),
        ("FLAT", "HOLD"),
        ("LONG", "HOLD"),
        ("LONG", "ADD"),
        ("LONG", "EXIT"),
        ("LONG", "TRIM"),
        ("LONG", "WAIT"),
    }
)


def _coerce_str(x: Any) -> str:
    """Defensive str() that survives NaN, None, ints, etc."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().upper()


def _norm_state_symbol(x: Any) -> str:
    """Normalize symbols for lifecycle/broker-state comparisons."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    s = str(x).strip().upper()
    # Broker/lifecycle may disagree on BRK.B vs BRK-B.
    if s == "BRK.B":
        s = "BRK-B"
    return s


def _norm_ticker(x: Any) -> str:
    """Normalize ticker keys used by lifecycle state and broker snapshots."""
    s = _coerce_str(x)
    if not s:
        return ""
    return s.replace(".", "-")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _load_state_file(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state_file(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _load_broker_long_tickers(path: Path = POSITIONS_SNAPSHOT) -> Set[str]:
    """
    Read local positions_snapshot.csv and return symbols that are truly LONG.

    This does not call the broker. It only uses the latest snapshot already
    written by services.snapshot_live_orders. Empty/missing/malformed snapshot
    safely returns an empty set, which is the correct representation after a
    manual account flatten if snapshot_live_orders wrote positions_written=0.
    """
    out: Set[str] = set()
    try:
        if not path.exists() or path.stat().st_size == 0:
            return out
        df = pd.read_csv(path)
    except Exception:
        return out

    if df is None or df.empty:
        return out

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    sym_col = next((c for c in ("ticker", "symbol", "Symbol") if c in df.columns), None)
    if sym_col is None:
        return out

    qty_col = next((c for c in ("qty", "qty_available", "quantity") if c in df.columns), None)
    mv_col = next((c for c in ("market_value", "value") if c in df.columns), None)

    for _, row in df.iterrows():
        ticker = _norm_ticker(row.get(sym_col))
        if not ticker:
            continue

        qty = _safe_float(row.get(qty_col), default=1.0) if qty_col else 1.0
        mv = _safe_float(row.get(mv_col), default=1.0) if mv_col else 1.0

        if qty > 0 and mv > 0:
            out.add(ticker)

    return out


def _sync_signal_state_with_broker_snapshot(
    *,
    state_path: Path,
    signals_df: pd.DataFrame,
    positions_snapshot: Path = POSITIONS_SNAPSHOT,
) -> Dict[str, int]:
    """
    Synchronize signal_state.json position_state with broker truth snapshot.

    Why this exists:
      If Triton previously held positions, signal_state.json may still say LONG.
      After the operator manually exits all positions, fresh BUY signals can be
      incorrectly treated as LONG+BUY and normalized to HOLD. This sync fixes
      that by forcing tickers absent from positions_snapshot.csv to FLAT before
      apply_lifecycle() runs.

    Safety:
      - No broker calls.
      - No orders.
      - No execution mutation.
      - Only lifecycle memory is corrected to match local broker snapshot.
    """
    if signals_df is None or signals_df.empty:
        return {
            "signals_tickers": 0,
            "broker_long": 0,
            "forced_flat": 0,
            "forced_long": 0,
            "unchanged": 0,
        }

    ticker_col = next((c for c in ("ticker", "symbol", "sym") if c in signals_df.columns), None)
    if ticker_col is None:
        print(
            "[LIFECYCLE_BROKER_SYNC] skipped reason=no_ticker_column_in_signals",
            flush=True,
        )
        return {
            "signals_tickers": 0,
            "broker_long": 0,
            "forced_flat": 0,
            "forced_long": 0,
            "unchanged": 0,
        }

    signal_tickers = sorted(
        {_norm_ticker(v) for v in signals_df[ticker_col].tolist() if _norm_ticker(v)}
    )
    broker_long = _load_broker_long_tickers(positions_snapshot)
    state = _load_state_file(state_path)

    forced_flat = 0
    forced_long = 0
    unchanged = 0

    for ticker in signal_tickers:
        current = state.get(ticker)
        if not isinstance(current, dict):
            current = {}

        old_pos = _coerce_str(current.get("position_state")) or "FLAT"
        target_pos = "LONG" if ticker in broker_long else "FLAT"

        if old_pos != target_pos:
            current["position_state"] = target_pos
            if target_pos == "FLAT":
                # Clear stale long-side memory. This lets fresh BUY signals become
                # BUY instead of being interpreted as already-held HOLDs.
                current["last_action"] = "NONE"
                current["last_change_date"] = None
                current["cooldown_until"] = None
                forced_flat += 1
            else:
                # Broker says long; keep state conservative and aligned.
                current["last_action"] = current.get("last_action") or "HOLD"
                forced_long += 1
            state[ticker] = current
        else:
            unchanged += 1

    try:
        _save_state_file(state_path, state)
    except Exception as e:
        print(
            f"[LIFECYCLE_BROKER_SYNC] save_failed error={type(e).__name__}: {e}",
            flush=True,
        )

    print(
        "[LIFECYCLE_BROKER_SYNC] "
        f"signals_tickers={len(signal_tickers)} "
        f"broker_long={len(broker_long)} "
        f"forced_flat={forced_flat} "
        f"forced_long={forced_long} "
        f"unchanged={unchanged} "
        f"positions_snapshot={positions_snapshot}",
        flush=True,
    )

    return {
        "signals_tickers": len(signal_tickers),
        "broker_long": len(broker_long),
        "forced_flat": forced_flat,
        "forced_long": forced_long,
        "unchanged": unchanged,
    }


def _normalize_pair(pos_in: str, act_in: str) -> Tuple[str, str, List[Tuple[str, str, str, str]]]:
    """
    Normalize a single (position_state, action) pair to a valid pair.

    Returns (new_pos, new_act, changes) where `changes` is a list of
    (field, old, new, reason) tuples — one per modified field. The empty
    list means the input was already valid.
    """
    pos = pos_in if pos_in in _VALID_POSITION_STATES else ""
    act = act_in if act_in in _VALID_ACTIONS else ""
    changes: List[Tuple[str, str, str, str]] = []

    if not pos:
        changes.append(("position_state", pos_in, "FLAT", "unknown_position_state"))
        pos = "FLAT"

    if not act:
        changes.append(("lifecycle_action", act_in, "WAIT", "unknown_action"))
        act = "WAIT"

    if (pos, act) in _VALID_PAIRS:
        return pos, act, changes

    if pos == "LONG" and act == "BUY":
        changes.append(("lifecycle_action", act, "HOLD", "long_buy_invalid_use_hold"))
        act = "HOLD"
    elif pos == "FLAT" and act in ("ADD", "TRIM", "EXIT"):
        changes.append(("lifecycle_action", act, "WAIT", f"flat_{act.lower()}_no_position"))
        act = "WAIT"
    else:
        changes.append(("lifecycle_action", act, "WAIT", "invalid_pair_fallback"))
        act = "WAIT"

    return pos, act, changes


def _normalize_lifecycle_dataframe(df: pd.DataFrame) -> Tuple[int, int, int]:
    """
    In-place row-by-row normalization of `df` so every row carries a valid
    (position_state, lifecycle_action) pair. Mirrors `lifecycle_action`
    into `stance` (the backwards-compat column) so downstream readers
    that consult either column observe the same normalized value.

    Returns (rows_normalized, fields_changed, remaining_invalid_after).
    """
    if df is None or df.empty:
        return 0, 0, 0
    if "position_state" not in df.columns:
        df["position_state"] = "FLAT"
    action_col = (
        "lifecycle_action"
        if "lifecycle_action" in df.columns
        else ("stance" if "stance" in df.columns else None)
    )
    if action_col is None:
        df["lifecycle_action"] = "WAIT"
        action_col = "lifecycle_action"

    rows_normalized = 0
    fields_changed = 0
    remaining_invalid = 0
    has_ticker = "ticker" in df.columns

    for idx in df.index:
        ticker = str(df.at[idx, "ticker"]).strip().upper() if has_ticker else f"row_{idx}"
        pos_in = _coerce_str(df.at[idx, "position_state"])
        act_in = _coerce_str(df.at[idx, action_col])

        # Entry-preservation fix:
        # apply_lifecycle() records position_state *after* a BUY transition, so
        # a valid FLAT -> BUY entry can appear here as LONG + BUY. The older
        # normalizer treated every LONG + BUY as invalid and converted it to
        # HOLD, which erased new-entry opportunities before broker reconciliation
        # could build trade_opportunities.csv. When the row itself shows this was
        # a fresh BUY transition, normalize the pair to FLAT + BUY instead. That
        # represents the pre-trade broker state needed by downstream opportunity
        # classification and remains a valid self-check pair.
        state_changed_raw = (
            _coerce_str(df.at[idx, "state_changed"]) if "state_changed" in df.columns else ""
        )
        last_action_raw = (
            _coerce_str(df.at[idx, "last_action"]) if "last_action" in df.columns else ""
        )
        decision_reason_raw = (
            str(df.at[idx, "lifecycle_decision_reason"]).strip().lower()
            if "lifecycle_decision_reason" in df.columns
            else ""
        )
        is_fresh_entry_buy = (
            pos_in == "LONG"
            and act_in == "BUY"
            and (
                state_changed_raw in {"TRUE", "1", "YES"}
                or last_action_raw == "BUY"
                or "flat_buy_to_buy" in decision_reason_raw
            )
        )
        pre_changes: List[Tuple[str, str, str, str]] = []
        if is_fresh_entry_buy:
            pre_changes.append(
                ("position_state", pos_in, "FLAT", "entry_buy_preserve_pre_trade_flat")
            )
            pos_in = "FLAT"

        new_pos, new_act, changes = _normalize_pair(pos_in, act_in)
        if pre_changes:
            changes = pre_changes + changes

        if changes:
            rows_normalized += 1
            fields_changed += len(changes)
            for field, old, new, reason in changes:
                old_disp = old if old != "" else "<empty>"
                print(
                    f"[LIFECYCLE_ROW_NORMALIZED] ticker={ticker} field={field} "
                    f"old={old_disp} new={new} reason={reason}",
                    flush=True,
                )

        df.at[idx, "position_state"] = new_pos
        df.at[idx, action_col] = new_act
        if "stance" in df.columns and action_col != "stance":
            df.at[idx, "stance"] = new_act

        if (new_pos, new_act) not in _VALID_PAIRS:
            remaining_invalid += 1

    return rows_normalized, fields_changed, remaining_invalid


def load_lifecycle_config() -> dict:
    """Load config/lifecycle_logic.json merged onto defaults (project-root path)."""
    defaults = {
        "enabled": True,
        "add_confidence_min": 0.52,
        "add_delta_pct_min": 0.008,
        "hold_delta_floor": -0.003,
        "hold_delta_ceiling": 0.006,
        "trim_delta_pct_threshold": -0.001,
        "exit_delta_pct_threshold": -0.0025,
        "exit_confidence_min": 0.50,
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


# Universe sanity threshold: a "healthy" signals file should carry at least
# this many unique valid tickers. Below this, we still proceed (the file may
# legitimately be small in dev / test setups) but emit
# `[LIFECYCLE_SOURCE_WARN]` so an operator can correlate a tiny lifecycle
# rebuild with its upstream cause instead of chasing the symptom downstream.
_MIN_VALID_TICKER_WARN_THRESHOLD = 10


def _count_valid_tickers(path: Path) -> Tuple[int, int]:
    """
    Defensive `(row_count, unique_valid_ticker_count)` for a candidate
    signals CSV.

    "Valid" means a non-empty, non-NaN, non-"NONE" string in whichever of
    {ticker, symbol, sym} the file uses. Files that are missing, empty,
    unreadable, or lack a recognizable ticker column collapse to (0, 0)
    so `_resolve_signals_source` can treat them as ineligible without
    branching on filesystem errors.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return 0, 0
    except OSError:
        return 0, 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0, 0

    rows = int(len(df))
    if rows == 0:
        return 0, 0

    col = next((c for c in ("ticker", "symbol", "sym") if c in df.columns), None)
    if col is None:
        return rows, 0

    s = df[col].astype(str).str.strip().str.upper()
    valid = s[(s != "") & (s != "NAN") & (s != "NONE")]
    return rows, int(valid.nunique())


def _resolve_signals_source() -> Optional[Path]:
    """
    Pick the signals file to use, preferring whichever candidate has the
    most unique valid tickers. This guards against the failure mode where
    an upstream stage half-populates ``signals_with_rationale.csv`` (e.g.
    only 5 mega-caps) while ``signals.csv`` already carries the full
    universe — historical "rationale first" logic would have rebuilt the
    lifecycle from the tiny file and silently shrunk the universe.

    Selection:
      - Skip candidates that are missing, empty, or unreadable.
      - Among the remainder, pick the file with the highest valid ticker
        count.
      - On a tie, fall back to the historical rationale-first ordering
        (Python's ``max`` is stable on ties; ``signals_with_rationale.csv``
        is the first item in the candidate list, so it wins ties).

    Emits:
      - ``[LIFECYCLE_SOURCE_SELECTED]`` for the chosen source — always,
        so callers can audit why the lifecycle has the size it has.
      - ``[LIFECYCLE_SOURCE_WARN]`` when the chosen source has fewer than
        ``_MIN_VALID_TICKER_WARN_THRESHOLD`` valid tickers.

    Returns ``None`` when neither candidate yields any rows; the caller
    must NOT overwrite ``signal_lifecycle.csv`` in that case (safety).
    """
    candidates: List[Tuple[Path, int, int]] = []
    for cand in (SIGNALS_RATIONALE, SIGNALS_FALLBACK):
        rows, tickers = _count_valid_tickers(cand)
        if rows == 0:
            continue
        candidates.append((cand, rows, tickers))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c[2])
    best_path, best_rows, best_tickers = best

    if len(candidates) == 1:
        reason = "only_available_source"
    else:
        other = next(c for c in candidates if c[0] != best_path)
        if best_tickers > other[2]:
            reason = f"more_valid_tickers_than_alt(alt={other[0].name}," f"alt_tickers={other[2]})"
        else:
            reason = (
                f"tie_preferred_rationale_first(alt={other[0].name}," f"alt_tickers={other[2]})"
            )

    print(
        f"[LIFECYCLE_SOURCE_SELECTED] source={best_path.name} rows={best_rows} "
        f"valid_tickers={best_tickers} reason={reason}",
        flush=True,
    )

    if best_tickers < _MIN_VALID_TICKER_WARN_THRESHOLD:
        print(
            f"[LIFECYCLE_SOURCE_WARN] source={best_path.name} "
            f"valid_tickers={best_tickers} "
            f"threshold={_MIN_VALID_TICKER_WARN_THRESHOLD} "
            f"reason=low_ticker_count",
            flush=True,
        )

    return best_path


def _lifecycle_is_stale(signals_path: Path) -> Tuple[bool, str]:
    """
    Returns (is_stale, reason).

    Stale iff:
      - signal_lifecycle.csv is missing OR empty, OR
      - signal_lifecycle.csv mtime is strictly older than signals_path mtime.

    `reason` is "" when fresh, otherwise a short tag suitable for logging.
    """
    try:
        if not OUT_LIFECYCLE.exists() or OUT_LIFECYCLE.stat().st_size == 0:
            return True, "lifecycle_missing_or_empty"
        sig_mtime = signals_path.stat().st_mtime
        lc_mtime = OUT_LIFECYCLE.stat().st_mtime
    except OSError as e:
        return True, f"stat_error:{e.__class__.__name__}"
    if sig_mtime > lc_mtime:
        return True, "signals_newer_than_lifecycle"
    return False, ""


def _preserve_entry_buys_against_broker_snapshot(
    df: pd.DataFrame,
    *,
    positions_snapshot: Path = POSITIONS_SNAPSHOT,
) -> int:
    """
    Preserve BUY intent for broker-flat symbols before row-pair normalization.

    apply_lifecycle() stores position_state after applying a BUY transition, so a
    valid new entry can appear as LONG + BUY inside signal_lifecycle.csv. The
    normalizer then treats LONG + BUY as invalid and converts it to HOLD.

    The broker snapshot is the authority for pre-trade position state. If the
    broker does not currently hold the symbol and lifecycle_action/stance is BUY,
    force the row pair back to FLAT + BUY so downstream opportunity generation
    can classify it as ENTRY. This is read-only with respect to broker/execution.
    """
    if df is None or df.empty or "ticker" not in df.columns:
        return 0

    broker_long = _load_broker_long_tickers(positions_snapshot)
    action_col = (
        "lifecycle_action"
        if "lifecycle_action" in df.columns
        else ("stance" if "stance" in df.columns else None)
    )
    if action_col is None:
        return 0

    changed = 0
    for idx in df.index:
        ticker = _norm_state_symbol(df.at[idx, "ticker"])
        if not ticker or ticker in broker_long:
            continue
        act = _coerce_str(df.at[idx, action_col])
        sig = _coerce_str(df.at[idx, "signal"]) if "signal" in df.columns else ""
        if act == "BUY" or sig == "BUY" and act == "BUY":
            old_pos = (
                _coerce_str(df.at[idx, "position_state"]) if "position_state" in df.columns else ""
            )
            if old_pos != "FLAT":
                df.at[idx, "position_state"] = "FLAT"
                changed += 1
                print(
                    f"[LIFECYCLE_ENTRY_BUY_PRESERVED] ticker={ticker} "
                    f"old_position_state={old_pos or '<empty>'} new_position_state=FLAT "
                    f"reason=broker_flat_buy_entry",
                    flush=True,
                )
            if "stance" in df.columns:
                df.at[idx, "stance"] = "BUY"
            if "lifecycle_action" in df.columns:
                df.at[idx, "lifecycle_action"] = "BUY"

    print(
        f"[LIFECYCLE_ENTRY_PRESERVE_SUMMARY] broker_long={len(broker_long)} "
        f"entry_buys_preserved={changed} positions_snapshot={positions_snapshot}",
        flush=True,
    )
    return changed


def _rebuild_lifecycle(src: Path) -> int:
    """
    Core rebuild path shared between main() and ensure_lifecycle_fresh().
    Reads `src`, applies lifecycle, writes signal_lifecycle.csv. Returns 0
    on success and any non-zero code from underlying machinery on failure.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)

    # Critical broker-state sync: stale LONG state was causing FLAT account
    # BUY signals to become HOLD. Sync before apply_lifecycle reads STATE_PATH.
    _sync_signal_state_with_broker_snapshot(
        state_path=STATE_PATH,
        signals_df=df,
        positions_snapshot=POSITIONS_SNAPSHOT,
    )

    lc_cfg = load_lifecycle_config()
    print("[LIFECYCLE CONFIG]", lc_cfg)
    tuned_keys = (
        "add_confidence_min",
        "exit_confidence_min",
        "exit_delta_pct_threshold",
        "hold_delta_floor",
        "hold_delta_ceiling",
        "add_delta_pct_min",
        "trim_delta_pct_threshold",
    )
    new_thresholds = {k: lc_cfg.get(k) for k in tuned_keys if k in lc_cfg}
    print(
        f"[LIFECYCLE_TUNING_APPLIED] new_thresholds={json.dumps(new_thresholds, sort_keys=True)}",
        flush=True,
    )
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

    out_df, _state = apply_lifecycle(
        df,
        state_path=STATE_PATH,
        cfg=engine_cfg,
        lifecycle_logic=lifecycle_logic,
    )

    # Important: preserve new-entry BUY rows before pair normalization.
    # Without this, apply_lifecycle() may output LONG + BUY for a fresh
    # entry transition, and the normalizer converts it to HOLD. Broker
    # snapshot truth determines whether the symbol is actually pre-trade FLAT.
    _preserve_entry_buys_against_broker_snapshot(
        out_df,
        positions_snapshot=POSITIONS_SNAPSHOT,
    )

    rows_norm, fields_changed, remaining_invalid = _normalize_lifecycle_dataframe(out_df)
    if rows_norm or fields_changed:
        print(
            f"[LIFECYCLE_NORMALIZE_SUMMARY] rows_normalized={rows_norm} "
            f"fields_changed={fields_changed} remaining_invalid={remaining_invalid}",
            flush=True,
        )

    out_df.to_csv(OUT_LIFECYCLE, index=False)

    try:
        from services.lifecycle_truth import print_self_check_result

        print_self_check_result(OUT_LIFECYCLE)
    except Exception as e:
        print(f"[LIFECYCLE_SELF_CHECK] skipped: {e}")

    status = "OK" if remaining_invalid == 0 else "FAIL"
    print(
        f"[LIFECYCLE_SELF_CHECK] status={status} invalid_rows={remaining_invalid}",
        flush=True,
    )

    try:
        from services.signal_pressure_diagnostics import refresh_signal_pressure_diagnostics

        refresh_signal_pressure_diagnostics()
    except Exception:
        pass

    print(f"Lifecycle applied: {src.name} -> {OUT_LIFECYCLE.name}")
    print(f"State persisted: {STATE_PATH.name}")
    return 0


def ensure_lifecycle_fresh(*, verbose: bool = False) -> bool:
    """
    Public freshness API for downstream callers (dashboards, execution,
    manage_positions). Rebuilds signal_lifecycle.csv ONLY if it is stale
    relative to the current signals file. Returns True iff a rebuild was
    actually performed.

    Safety:
      - If no signals file exists, this is a no-op (returns False) and
        does NOT touch signal_lifecycle.csv. The spec is explicit:
        "do NOT overwrite lifecycle if signals missing; only run when
        signals exist."
      - On any rebuild, emits the audit log
        ``[LIFECYCLE_AUTO_REFRESH] reason=signals_newer_than_lifecycle``
        (or the more specific reason tag from _lifecycle_is_stale, e.g.
        ``lifecycle_missing_or_empty``).
    """
    src = _resolve_signals_source()
    if src is None:
        if verbose:
            print(
                "[LIFECYCLE_FRESHNESS] signals file missing — skip "
                "(safety: do not overwrite existing lifecycle)."
            )
        return False

    stale, reason = _lifecycle_is_stale(src)
    if not stale:
        if verbose:
            print(
                f"[LIFECYCLE_FRESH] lifecycle is current relative to "
                f"{src.name} — no rebuild needed."
            )
        return False

    print(
        f"[LIFECYCLE_AUTO_REFRESH] reason={reason} signals={src.name} "
        f"lifecycle={OUT_LIFECYCLE.name}",
        flush=True,
    )
    rc = _rebuild_lifecycle(src)
    return rc == 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Apply signal lifecycle: read signals_with_rationale.csv (or "
            "signals.csv fallback) and write signal_lifecycle.csv."
        )
    )
    ap.add_argument(
        "--only-if-stale",
        action="store_true",
        help=(
            "Skip the rebuild when signal_lifecycle.csv is already current "
            "relative to the signals file. Equivalent to calling "
            "ensure_lifecycle_fresh() — useful for downstream callers that "
            "want a freshness guarantee without forcing a rebuild."
        ),
    )
    args = ap.parse_args(argv)

    src = _resolve_signals_source()
    if src is None:
        # Safety: do NOT overwrite lifecycle if signals missing.
        # Returning rc=2 preserves the prior contract (the pipeline runner
        # treats any non-zero rc as a fail — so a required-stage skip is
        # surfaced rather than silently passed).
        print(
            "No signals file found (signals_with_rationale.csv or signals.csv); "
            "lifecycle NOT modified."
        )
        return 2

    stale, reason = _lifecycle_is_stale(src)
    if args.only_if_stale and not stale:
        print(
            f"[LIFECYCLE_FRESH] only_if_stale=True and lifecycle is current "
            f"relative to {src.name} — skip rebuild."
        )
        return 0

    if stale:
        print(
            f"[LIFECYCLE_AUTO_REFRESH] reason={reason} signals={src.name} "
            f"lifecycle={OUT_LIFECYCLE.name}",
            flush=True,
        )

    return _rebuild_lifecycle(src)


if __name__ == "__main__":
    raise SystemExit(main())
