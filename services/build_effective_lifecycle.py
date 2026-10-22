"""
build_effective_lifecycle.py
----------------------------
Overlay broker reconciliation truth onto signal_lifecycle rows.

Reads:
  - data/results/signal_lifecycle.csv  (unchanged on disk)
  - data/results/lifecycle_reconciliation.csv
  - data/results/open_orders_snapshot.csv (optional; in-flight BUY preserves effective BUY when broker flat)

Writes:
  - data/results/signal_lifecycle_effective.csv

Read-only with respect to signal_lifecycle.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Set

import pandas as pd

from services.lifecycle_artifact_guard import write_lifecycle_csv_protected
from services.signal_lifecycle import (
    LifecycleLogicConfig,
    lifecycle_logic_from_dict,
    long_buy_qualifies_for_add,
    long_qualifies_for_exit_delta,
    long_qualifies_for_trim,
)
from services.symbol_identity import (
    SymbolIdentityCollision,
    assert_no_identity_collisions,
    candidate_display_map,
    collect_identity_collisions,
    identity_key,
    restore_candidate_spellings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_LIFECYCLE = RESULTS_DIR / "signal_lifecycle.csv"
DEFAULT_RECON = RESULTS_DIR / "lifecycle_reconciliation.csv"
DEFAULT_OUT = RESULTS_DIR / "signal_lifecycle_effective.csv"
DEFAULT_OPEN_ORDERS = RESULTS_DIR / "open_orders_snapshot.csv"
EFFECTIVE_BINDING_SCHEMA = "SIGNAL-BUNDLE-EFFECTIVE-LIFECYCLE-1"

# Status values meaning the order is still working (not fully filled / terminal).
_TERMINAL_ORDER_STATUSES = frozenset(
    {
        "filled",
        "done",
        "canceled",
        "cancelled",
        "expired",
        "failed",
        "replaced",
        "rejected",
        "closed",
    }
)


def load_lifecycle_config() -> dict:
    """Load config/lifecycle_logic.json merged onto defaults (project-root path)."""
    defaults = {
        "enabled": True,
        "add_confidence_min": 0.60,
        "add_delta_pct_min": 0.008,
        "hold_delta_floor": -0.002,
        "hold_delta_ceiling": 0.006,
        "trim_delta_pct_threshold": -0.001,
        "exit_delta_pct_threshold": -0.004,
        "exit_confidence_min": 0.55,
    }
    path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "..",
        "config",
        "lifecycle_logic.json",
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


def _norm_symbol(s: str) -> str:
    """Canonical identity for lifecycle vs broker snapshot joins (BRK.B == BRK-B)."""
    return identity_key(s)


def _delta_pct_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float, int)) and pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def load_reconciliation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing reconciliation file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "mismatch_reason"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("lifecycle_reconciliation.csv must have ticker")
    df = df.copy()
    raw = df["ticker"].astype(str).tolist()
    collisions = collect_identity_collisions(raw)
    if collisions:
        raise SymbolIdentityCollision(collisions, source="reconciliation")
    df["ticker"] = df["ticker"].map(_norm_symbol)
    if "mismatch_reason" not in df.columns:
        df["mismatch_reason"] = ""
    df = df[["ticker", "mismatch_reason"]].drop_duplicates(subset=["ticker"], keep="last")
    return df


def load_open_buy_tickers(path: Path) -> Set[str]:
    """
    Symbols with an in-flight BUY (open/accepted/pending/etc., not fully terminal).
    Uses same snapshot as snapshot_live_orders / manage_positions.
    """
    out: Set[str] = set()
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return out
        df = pd.read_csv(path)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if not sym_col:
        return out
    side_col = "side" if "side" in df.columns else None
    st_col = "status" if "status" in df.columns else None
    fq_col = "filled_qty" if "filled_qty" in df.columns else None
    qcol = "qty" if "qty" in df.columns else None

    for _, row in df.iterrows():
        side = str(row.get(side_col) or "").strip().lower() if side_col else ""
        if "buy" not in side:
            continue
        st = str(row.get(st_col) or "").strip().lower() if st_col else ""
        if st in _TERMINAL_ORDER_STATUSES:
            continue
        if fq_col and qcol:
            try:
                fq = float(row.get(fq_col) or 0)
                qq = float(row.get(qcol) or 0)
                if qq > 0 and fq >= qq:
                    continue
            except Exception:
                pass
        sym = _norm_symbol(str(row.get(sym_col) or ""))
        if sym:
            out.add(sym)
    return out


def load_lifecycle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing lifecycle file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "ticker" not in df.columns:
        raise ValueError("signal_lifecycle.csv must have ticker")
    df = df.copy()
    raw = df["ticker"].astype(str).tolist()
    collisions = collect_identity_collisions(raw)
    if collisions:
        raise SymbolIdentityCollision(collisions, source="lifecycle")
    df["ticker"] = df["ticker"].map(_norm_symbol)
    df = df[df["ticker"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["ticker"], keep="last")
    return df


def apply_heal(
    stance: str,
    position_state: str,
    mismatch_reason: str,
    *,
    ticker: str = "",
    open_buy_tickers: Optional[Set[str]] = None,
) -> tuple[str, str, bool, str, str, bool]:
    """
    Broker vs persisted lifecycle reconciliation only.

    Returns:
      effective_position, effective_stance, healed, heal_reason,
      lifecycle_authoritative_source, reconciled_with_broker

    `healed` is True only for rare broker/persisted mismatch resolution (not stance normalization).
    Open-BUY in-flight keeps BUY without counting as a broker "heal" for metrics.
    """
    stance_u = str(stance or "").strip().upper()
    reason = str(mismatch_reason or "").strip()
    sym_u = _norm_symbol(str(ticker or ""))
    ob = open_buy_tickers or set()

    if reason == "lifecycle_long_broker_flat":
        ep = "FLAT"
        if stance_u == "BUY" and sym_u and sym_u in ob:
            return ep, "BUY", False, "", "open_buy_in_flight", False
        if stance_u in ("HOLD", "ADD", "BUY", "EXIT", "TRIM"):
            es = "WAIT"
        else:
            es = stance_u
        return (
            ep,
            es,
            True,
            "broker_flat_overrides_lifecycle_long",
            "broker_reconciled",
            True,
        )

    if reason == "lifecycle_flat_broker_long":
        ep = "LONG"
        if stance_u == "WAIT":
            es = "HOLD"
        else:
            es = stance_u
        return (
            ep,
            es,
            True,
            "broker_long_overrides_lifecycle_flat",
            "broker_reconciled",
            True,
        )

    ps = str(position_state or "").strip()
    st = str(stance or "").strip()
    return ps, st, False, "", "persisted_lifecycle", False


def _deterministic_effective_stance(
    effective_position_state: str,
    effective_stance: str,
    *,
    delta_pct: Any = None,
    confidence: Any = None,
    logic: Optional[LifecycleLogicConfig] = None,
) -> tuple[str, str, str]:
    """
    Deterministic stance from persisted signal + thresholds (not broker healing).

    Returns (effective_position, effective_stance, stance_adjustment_tag_or_empty).

    - FLAT cannot EXIT/TRIM → WAIT.
    - LONG+BUY should not appear from apply_lifecycle; if present, map to EXIT/TRIM/ADD/HOLD.
    """
    _logic = logic or lifecycle_logic_from_dict(load_lifecycle_config())
    ep = str(effective_position_state or "").strip()
    ep_u = ep.upper()
    es_u = str(effective_stance or "").strip().upper()
    st_out = str(effective_stance or "").strip()

    if ep_u == "FLAT" and es_u in ("EXIT", "TRIM"):
        return ep, "WAIT", "effective_flat_suppresses_exit_trim"

    if ep_u == "LONG" and es_u == "ADD":
        return ep, st_out, ""

    if ep_u == "LONG" and es_u == "BUY":
        # RF residual must not remap LONG+BUY to EXIT/TRIM/ADD.
        if long_qualifies_for_exit_delta(delta_pct, logic=_logic):
            return ep, "EXIT", "effective_long_buy_resolved_to_exit_delta"
        if long_qualifies_for_trim(delta_pct, logic=_logic):
            return ep, "TRIM", "effective_long_buy_resolved_to_trim_delta"
        if long_buy_qualifies_for_add(delta_pct, confidence, logic=_logic):
            return ep, "ADD", "effective_long_buy_resolved_to_add_thresholds"
        return ep, "HOLD", "effective_long_buy_resolved_to_hold_weak"

    return ep, st_out, ""


def build_effective(
    lc: pd.DataFrame,
    recon: pd.DataFrame,
    *,
    open_orders_path: Path = DEFAULT_OPEN_ORDERS,
    lifecycle_logic: Any = None,
) -> pd.DataFrame:
    merged = lc.merge(recon, on="ticker", how="left")
    merged["mismatch_reason"] = merged["mismatch_reason"].fillna("").astype(str)

    open_buy_tickers = load_open_buy_tickers(open_orders_path)
    logic = lifecycle_logic or lifecycle_logic_from_dict(load_lifecycle_config())

    eps: list[str] = []
    es: list[str] = []
    healed: list[bool] = []
    reasons: list[str] = []
    inflight: list[bool] = []
    auth_src: list[str] = []
    recon_broker: list[bool] = []
    stance_adj: list[str] = []
    lc_consistency: list[str] = []

    for _, row in merged.iterrows():
        ep, est, h, hr, asrc, rbrok = apply_heal(
            str(row.get("stance", "")),
            str(row.get("position_state", "")),
            str(row.get("mismatch_reason", "")),
            ticker=str(row.get("ticker", "")),
            open_buy_tickers=open_buy_tickers,
        )
        eps.append(ep)
        es.append(est)
        healed.append(h)
        reasons.append(hr)
        inflight.append(asrc == "open_buy_in_flight")
        auth_src.append(asrc)
        recon_broker.append(rbrok)

    for i in range(len(eps)):
        row = merged.iloc[i]
        ep0, es0, adj_tag = _deterministic_effective_stance(
            eps[i],
            es[i],
            delta_pct=row.get("delta_pct"),
            confidence=row.get("confidence"),
            logic=logic,
        )
        eps[i] = ep0
        es[i] = es0
        stance_adj.append(adj_tag)
        if adj_tag:
            lc_consistency.append("ADJUSTED_STANCE")
        elif recon_broker[i]:
            lc_consistency.append("RECONCILED_BROKER")
        else:
            lc_consistency.append("OK")

    out = merged.copy()
    out["effective_position_state"] = eps
    out["effective_stance"] = es
    out["healed"] = healed
    out["heal_reason"] = reasons
    out["effective_in_flight"] = inflight
    out["lifecycle_authoritative_source"] = auth_src
    out["reconciled_with_broker"] = recon_broker
    out["reconciled_reason"] = [
        (reasons[i] if recon_broker[i] else "") for i in range(len(reasons))
    ]
    out["stance_adjustment"] = stance_adj
    out["lifecycle_consistency"] = lc_consistency

    # Reconciliation field was join-only; output = lifecycle columns + effective fields
    if "mismatch_reason" in out.columns:
        out = out.drop(columns=["mismatch_reason"])

    n_recon = int(sum(recon_broker)) if recon_broker else 0
    print(f"[RECONCILIATION] broker_reconciled_rows={n_recon} (snapshot+reconciliation.csv)")

    try:
        from services.lifecycle_truth import (
            enrich_execution_flags,
            print_effective_summary_stats,
            print_lifecycle_summary_from_effective,
        )

        out = enrich_execution_flags(out)
        print_lifecycle_summary_from_effective(out)
        print_effective_summary_stats(out)
    except Exception:
        pass

    return out


def _snapshot_symbol_column(frame: pd.DataFrame) -> str:
    for name in ("ticker", "symbol"):
        if name in frame.columns:
            return name
    raise ValueError("positions_snapshot_malformed")


def read_positions_snapshot_evidence(path: Path) -> dict:
    """Hash and timestamp of a local positions snapshot. Does not call the broker."""
    from services.lifecycle_artifact_guard import file_sha256

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("positions_snapshot_missing")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError("positions_snapshot_malformed") from exc
    if (
        frame is None
        or frame.empty
        or "snapshot_ts" not in frame.columns
        or "qty" not in frame.columns
    ):
        raise ValueError("positions_snapshot_malformed")
    try:
        _snapshot_symbol_column(frame)
    except ValueError as exc:
        raise ValueError("positions_snapshot_malformed") from exc
    stamps = {
        str(value).strip()
        for value in frame["snapshot_ts"].tolist()
        if str(value).strip() and str(value).strip().lower() != "nan"
    }
    if len(stamps) != 1:
        raise ValueError("positions_snapshot_malformed")
    return {
        "sha256": file_sha256(path),
        "snapshot_ts": next(iter(stamps)),
        "rows": int(len(frame.index)),
    }


def _require_snapshot_evidence(path: Path, evidence: Mapping[str, Any]) -> dict:
    if not isinstance(evidence, Mapping):
        raise ValueError("positions_snapshot_malformed")
    current = read_positions_snapshot_evidence(path)
    expected_hash = str(evidence.get("sha256") or "").strip().lower()
    expected_ts = str(evidence.get("snapshot_ts") or "").strip()
    if not expected_hash or expected_hash != current["sha256"]:
        raise ValueError("positions_snapshot_hash_mismatch")
    if not expected_ts or expected_ts != current["snapshot_ts"]:
        raise ValueError("positions_snapshot_timestamp_mismatch")
    return current


def _snapshot_held_symbols(path: Path) -> Set[str]:
    frame = pd.read_csv(path)
    symbol_col = _snapshot_symbol_column(frame)
    raw_held: list[str] = []
    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_col) or "").strip().upper()
        try:
            qty = float(row.get("qty") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if symbol and symbol not in {"NAN", "NONE", "NULL"} and qty > 0:
            raw_held.append(symbol)
    assert_no_identity_collisions(raw_held, source="positions_snapshot")
    return {_norm_symbol(symbol) for symbol in raw_held}


def _candidate_action(row: Mapping[str, Any]) -> str:
    for key in ("lifecycle_action", "stance"):
        value = str(row.get(key) or "").strip().upper()
        if value and value != "NAN":
            return value
    return ""


def _preserve_snapshot_held_state(
    frame: pd.DataFrame,
    *,
    candidate_actions: Mapping[str, str],
    held: Set[str],
    basis_blocked: Set[str],
    basis_classification: str,
) -> pd.DataFrame:
    """Keep snapshot-held names LONG. A candidate HOLD is not rewritten to EXIT."""
    out = frame.copy()
    if "basis_classification" not in out.columns:
        out["basis_classification"] = ""
    for index, row in out.iterrows():
        ticker = _norm_symbol(str(row.get("ticker") or ""))
        if ticker not in held:
            continue
        out.at[index, "effective_position_state"] = "LONG"
        if "position_state" in out.columns:
            out.at[index, "position_state"] = "LONG"
        action = candidate_actions.get(ticker, "")
        if action == "HOLD" or ticker in basis_blocked:
            out.at[index, "effective_stance"] = "HOLD"
        if ticker in basis_blocked:
            out.at[index, "basis_classification"] = basis_classification
    return out


def write_effective_lifecycle(
    *,
    lifecycle_path: Path,
    reconciliation_path: Path,
    open_orders_path: Path,
    output_path: Path,
    frame: Optional[pd.DataFrame] = None,
    min_rows: Optional[int] = None,
) -> int:
    """Write effective lifecycle to an explicit path. Callers do not use sys.argv."""
    if frame is None:
        try:
            lc = load_lifecycle(lifecycle_path)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        try:
            recon = load_reconciliation(reconciliation_path)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        lc_cfg = load_lifecycle_config()
        print("[LIFECYCLE CONFIG]", lc_cfg)
        lifecycle_logic = lifecycle_logic_from_dict(lc_cfg)
        if lc.empty:
            print("[WARN] signal_lifecycle.csv is empty; skipping effective overwrite.")
            frame = lc.copy()
            frame["effective_position_state"] = pd.Series(dtype=object)
            frame["effective_stance"] = pd.Series(dtype=object)
            frame["healed"] = pd.Series(dtype=bool)
            frame["heal_reason"] = pd.Series(dtype=object)
            frame["effective_in_flight"] = pd.Series(dtype=bool)
            frame["lifecycle_authoritative_source"] = pd.Series(dtype=object)
            frame["reconciled_with_broker"] = pd.Series(dtype=bool)
            frame["reconciled_reason"] = pd.Series(dtype=object)
            frame["stance_adjustment"] = pd.Series(dtype=object)
            frame["lifecycle_consistency"] = pd.Series(dtype=object)
        else:
            frame = build_effective(
                lc,
                recon,
                open_orders_path=open_orders_path,
                lifecycle_logic=lifecycle_logic,
            )
    write_result = write_lifecycle_csv_protected(
        frame,
        output_path,
        artifact="effective",
        min_rows=min_rows,
    )
    if not write_result.ok:
        print(
            f"[build_effective_lifecycle] effective write blocked: rows={write_result.rows} "
            f"min_rows={write_result.min_rows} action={write_result.action}",
            file=sys.stderr,
        )
        return 3
    healed = int(frame["healed"].sum()) if "healed" in frame.columns else 0
    in_flight = (
        int(frame["effective_in_flight"].sum()) if "effective_in_flight" in frame.columns else 0
    )
    print(f"[build_effective_lifecycle] wrote {output_path}")
    print(
        f"[build_effective_lifecycle] total_rows={len(frame)} healed_count={healed} "
        f"open_buy_in_flight_preserved={in_flight}"
    )
    return 0


def build_staged_effective_lifecycle(
    *,
    lifecycle_path: Path,
    reconciliation_path: Path,
    open_orders_path: Path,
    output_path: Path,
    positions_snapshot: Path,
    snapshot_evidence: Mapping[str, Any],
    bundle_id: str,
    session_as_of_date: str,
    lifecycle_sha256: str,
    expected_symbols: Sequence[str],
    min_rows: Optional[int] = None,
) -> dict:
    """Build one candidate's effective lifecycle from explicit staged paths."""
    from services.basis_unproven_hold import (
        CLASSIFICATION,
        evaluate_basis_unproven_hold_for_packet,
    )
    from services.lifecycle_artifact_guard import file_sha256

    lifecycle_path = Path(lifecycle_path)
    positions_snapshot = Path(positions_snapshot)
    evidence = _require_snapshot_evidence(positions_snapshot, snapshot_evidence)
    actual_hash = file_sha256(lifecycle_path)
    if actual_hash != str(lifecycle_sha256 or "").strip().lower():
        raise ValueError("lifecycle_hash_mismatch")
    display = candidate_display_map(expected_symbols)
    expected = sorted(display)
    if (
        not expected
        or not str(bundle_id or "").strip()
        or not str(session_as_of_date or "").strip()
    ):
        raise ValueError("effective_lifecycle_unbound")
    try:
        lifecycle = load_lifecycle(lifecycle_path)
        reconciliation = load_reconciliation(reconciliation_path)
    except SymbolIdentityCollision:
        raise
    except Exception as exc:
        raise ValueError("effective_lifecycle_input_unreadable") from exc
    if "stance" not in lifecycle.columns:
        source = lifecycle["lifecycle_action"] if "lifecycle_action" in lifecycle.columns else ""
        lifecycle = lifecycle.copy()
        lifecycle["stance"] = source
    candidate_actions = {
        _norm_symbol(str(row.get("ticker") or "")): _candidate_action(row)
        for _, row in lifecycle.iterrows()
    }
    built = build_effective(
        lifecycle,
        reconciliation,
        open_orders_path=Path(open_orders_path),
    )
    basis = evaluate_basis_unproven_hold_for_packet()
    blocked = {_norm_symbol(symbol) for symbol in basis.get("blocked_symbols") or []}
    built = _preserve_snapshot_held_state(
        built,
        candidate_actions=candidate_actions,
        held=_snapshot_held_symbols(positions_snapshot),
        basis_blocked=blocked,
        basis_classification=CLASSIFICATION,
    )
    actual = sorted({_norm_symbol(str(symbol)) for symbol in built["ticker"].tolist()})
    if actual != expected:
        raise ValueError("effective_lifecycle_symbol_mismatch")
    built = built.copy()
    built["ticker"] = restore_candidate_spellings(built["ticker"].tolist(), expected_symbols)
    if set(built["ticker"].astype(str)) != set(display.values()):
        raise ValueError("candidate_spelling_not_preserved")
    built["candidate_effective_schema"] = EFFECTIVE_BINDING_SCHEMA
    built["candidate_bundle_id"] = str(bundle_id)
    built["candidate_session_as_of_date"] = str(session_as_of_date)
    built["candidate_lifecycle_sha256"] = actual_hash
    built["positions_snapshot_sha256"] = evidence["sha256"]
    built["positions_snapshot_ts"] = evidence["snapshot_ts"]
    threshold = len(expected) if min_rows is None else int(min_rows)
    rc = write_effective_lifecycle(
        lifecycle_path=lifecycle_path,
        reconciliation_path=Path(reconciliation_path),
        open_orders_path=Path(open_orders_path),
        output_path=Path(output_path),
        frame=built,
        min_rows=threshold,
    )
    if rc != 0:
        raise RuntimeError(f"effective_lifecycle_write_failed:{rc}")
    return {
        "ok": True,
        "rows": int(len(built.index)),
        "bundle_id": str(bundle_id),
        "session_as_of_date": str(session_as_of_date),
        "lifecycle_sha256": actual_hash,
        "positions_snapshot_sha256": evidence["sha256"],
        "positions_snapshot_ts": evidence["snapshot_ts"],
        "output_path": str(output_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build signal_lifecycle_effective.csv from lifecycle + reconciliation"
    )
    ap.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    ap.add_argument("--recon", type=Path, default=DEFAULT_RECON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--open-orders",
        type=Path,
        default=DEFAULT_OPEN_ORDERS,
        help="Snapshot CSV for open orders (default data/results/open_orders_snapshot.csv)",
    )
    args = ap.parse_args()
    return write_effective_lifecycle(
        lifecycle_path=args.lifecycle,
        reconciliation_path=args.recon,
        open_orders_path=args.open_orders,
        output_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
