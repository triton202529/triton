"""Publish one signal generation, or leave the prior bundle untouched.

Success replaces signals, rationale, lifecycle, effective lifecycle, allocation,
and readiness together. A failed candidate writes a diagnostic receipt and
does not replace those artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
STAGING_DIR = RESULTS_DIR / "signal_bundle_stage"

BUNDLE_FILES = (
    "signals.csv",
    "signals_with_rationale.csv",
    "signal_lifecycle.csv",
    "signal_lifecycle_effective.csv",
    "portfolio_allocation_recommendations.csv",
    "execution_readiness.json",
)
FAILURE_RECEIPT = "signal_bundle_failure_receipt.json"
STAGE_MARKER = "signal_bundle_stage.json"
SCHEMA_FAILURE = "SIGNAL-BUNDLE-FAILURE-1"
SCHEMA_STAGE = "SIGNAL-BUNDLE-STAGE-1"


def utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_failure_receipt(results_dir: Path | str, payload: Mapping[str, object]) -> Path:
    """Write the only production artifact a failed candidate may create."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": SCHEMA_FAILURE,
        "status": "FAILED",
        "published_signal_artifacts": False,
        "generated_at_utc": utc_now_z(),
    }
    body.update(dict(payload))
    body["published_signal_artifacts"] = False
    target = results_dir / FAILURE_RECEIPT
    _atomic_bytes(target, (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return target


def held_symbols_from_snapshot(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if frame.empty or ("ticker" not in frame.columns and "symbol" not in frame.columns):
        return set()
    symbol_col = "ticker" if "ticker" in frame.columns else "symbol"
    qty_col = "qty" if "qty" in frame.columns else None
    held: Set[str] = set()
    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_col) or "").strip().upper()
        if not symbol:
            continue
        qty = 0.0
        if qty_col is not None:
            try:
                qty = float(row.get(qty_col) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
        if qty > 0:
            held.add(symbol)
    return held


def assert_held_positions_stay_long(lifecycle: pd.DataFrame, held: Iterable[str]) -> None:
    """Broker-held names stay LONG. EXIT is preserved; WAIT is refused."""
    if lifecycle is None or lifecycle.empty:
        raise RuntimeError("lifecycle candidate is empty")
    symbol_col = "ticker" if "ticker" in lifecycle.columns else "symbol"
    action_col = "lifecycle_action" if "lifecycle_action" in lifecycle.columns else "stance"
    for symbol in sorted({str(s).strip().upper() for s in held if str(s).strip()}):
        rows = lifecycle[lifecycle[symbol_col].astype(str).str.upper() == symbol]
        if rows.empty:
            raise RuntimeError(f"held symbol {symbol} missing from lifecycle candidate")
        position = str(rows.iloc[0].get("position_state") or "").strip().upper()
        action = str(rows.iloc[0].get(action_col) or "").strip().upper()
        if position != "LONG":
            raise RuntimeError(f"held symbol {symbol} position_state={position}")
        if action in {"", "WAIT"}:
            raise RuntimeError(f"held symbol {symbol} action={action}")


def _atomic_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _stamp_csv(path: Path, generated_at: str, session: str) -> None:
    frame = pd.read_csv(path)
    frame["bundle_generated_at_utc"] = generated_at
    frame["bundle_session_as_of_date"] = session
    path.write_text(frame.to_csv(index=False), encoding="utf-8", newline="")


def _validate_staged(staged: Path, generated_at: str, session: str) -> None:
    missing = [name for name in BUNDLE_FILES if not (staged / name).is_file()]
    if missing:
        raise RuntimeError(f"candidate bundle incomplete: {missing}")
    signals = pd.read_csv(staged / "signals.csv")
    if signals.empty or "ticker" not in signals.columns or signals.dropna(how="all").empty:
        raise RuntimeError("refusing to publish a header-only signal file")
    rationale = pd.read_csv(staged / "signals_with_rationale.csv")
    if set(signals["ticker"].astype(str)) != set(rationale["ticker"].astype(str)):
        raise RuntimeError("signals and rationale ticker sets differ")
    for name in (
        "signals.csv",
        "signals_with_rationale.csv",
        "signal_lifecycle.csv",
        "signal_lifecycle_effective.csv",
        "portfolio_allocation_recommendations.csv",
    ):
        frame = pd.read_csv(staged / name)
        if "bundle_generated_at_utc" not in frame.columns:
            raise RuntimeError(f"{name} missing bundle timestamp")
        stamps = set(frame["bundle_generated_at_utc"].astype(str))
        sessions = set(frame["bundle_session_as_of_date"].astype(str))
        if stamps != {generated_at} or sessions != {session}:
            raise RuntimeError(f"{name} is not on the candidate generation")
    readiness = json.loads((staged / "execution_readiness.json").read_text(encoding="utf-8"))
    if readiness.get("bundle_generated_at_utc") != generated_at:
        raise RuntimeError("readiness is not on the candidate generation")
    if str(readiness.get("bundle_session_as_of_date") or "") != session:
        raise RuntimeError("readiness session does not match the candidate")
    if readiness.get("live_execution_permitted") is True:
        raise RuntimeError("readiness candidate permits live execution")


def publish_coherent_bundle(
    production: Path, staged: Path, *, generated_at: str, session: str
) -> None:
    """Replace the six production artifacts, or restore them if a replace fails."""
    _validate_staged(staged, generated_at, session)
    prior: Dict[str, Optional[bytes]] = {}
    for name in BUNDLE_FILES:
        path = production / name
        prior[name] = path.read_bytes() if path.is_file() else None
    replaced: list[str] = []
    try:
        for name in BUNDLE_FILES:
            _atomic_bytes(production / name, (staged / name).read_bytes())
            replaced.append(name)
    except Exception:
        for name in replaced:
            path = production / name
            previous = prior[name]
            if previous is None:
                if path.exists():
                    path.unlink()
            else:
                _atomic_bytes(path, previous)
        raise


def _production_results() -> Path:
    return RESULTS_DIR


def stage_signal_bundle() -> int:
    """Build the candidate in staging. Production bundle files stay in place."""
    import services.apply_signal_lifecycle as lifecycle
    import services.build_effective_lifecycle as effective
    import services.generate_signals as generate

    production = Path(generate.RESULTS_DIR)
    if not production.is_absolute():
        production = PROJECT_ROOT / production
    staging = STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    generate.RESULTS_DIR = str(staging)
    generate.OUT_WITH_RATIONALE = str(staging / "signals_with_rationale.csv")
    generate.OUT_SIGNALS = str(staging / "signals.csv")
    generate.ASOF_DIAG_PATH = str(staging / "signals_asof_diagnostics.json")

    try:
        generate.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        _fail_stage(production, staging, reason="generate_signals_exit", exit_code=code)
        return code or 1
    except Exception as exc:
        _fail_stage(production, staging, reason="generate_signals_failed", detail=str(exc))
        return 1

    signals_path = staging / "signals.csv"
    if not signals_path.is_file() or pd.read_csv(signals_path).empty:
        _fail_stage(production, staging, reason="empty_signal_candidate")
        return 1

    lifecycle.RESULTS_DIR = staging
    lifecycle.SIGNALS_RATIONALE = staging / "signals_with_rationale.csv"
    lifecycle.SIGNALS_FALLBACK = staging / "signals.csv"
    lifecycle.OUT_LIFECYCLE = staging / "signal_lifecycle.csv"
    lifecycle.STATE_PATH = staging / "signal_state.json"
    lifecycle.POSITIONS_SNAPSHOT = production / "positions_snapshot.csv"
    rc = lifecycle.main([])
    if rc != 0:
        _fail_stage(production, staging, reason="apply_signal_lifecycle_failed", exit_code=rc)
        return int(rc or 1)

    recon = production / "lifecycle_reconciliation.csv"
    orders = production / "open_orders_snapshot.csv"
    eff_rc = effective.main(
        [
            "--lifecycle",
            str(staging / "signal_lifecycle.csv"),
            "--recon",
            str(recon),
            "--open-orders",
            str(orders),
            "--out",
            str(staging / "signal_lifecycle_effective.csv"),
        ]
    )
    if eff_rc != 0:
        _fail_stage(
            production, staging, reason="build_effective_lifecycle_failed", exit_code=eff_rc
        )
        return int(eff_rc or 1)

    try:
        held = held_symbols_from_snapshot(production / "positions_snapshot.csv")
        assert_held_positions_stay_long(
            pd.read_csv(staging / "signal_lifecycle.csv"),
            held,
        )
    except Exception as exc:
        _fail_stage(production, staging, reason="held_position_guard", detail=str(exc))
        return 1

    diag_path = staging / "signals_asof_diagnostics.json"
    session = ""
    generated_at = utc_now_z()
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        session = str(diag.get("session_as_of_date") or diag.get("effective_as_of_date") or "")
    frame = pd.read_csv(signals_path)
    if "generated_at_utc" in frame.columns and not frame.empty:
        generated_at = str(frame["generated_at_utc"].iloc[0])
    if "session_as_of_date" in frame.columns and not frame.empty:
        session = str(frame["session_as_of_date"].iloc[0])[:10]
    marker = {
        "schema": SCHEMA_STAGE,
        "status": "STAGED",
        "session_as_of_date": session,
        "generated_at_utc": generated_at,
        "published_signal_artifacts": False,
    }
    (staging / STAGE_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[SIGNAL_BUNDLE_STAGED] session={session} generated_at={generated_at}")
    return 0


def _fail_stage(production: Path, staging: Path, **payload: object) -> None:
    diag = staging / "signals_asof_diagnostics.json"
    if diag.is_file():
        _atomic_bytes(production / diag.name, diag.read_bytes())
    receipt = staging / FAILURE_RECEIPT
    if receipt.is_file():
        _atomic_bytes(production / FAILURE_RECEIPT, receipt.read_bytes())
    else:
        write_failure_receipt(production, payload)
    print("[SIGNAL_BUNDLE_STAGE_FAILED] production bundle preserved")


def commit_signal_bundle() -> int:
    """Stamp allocation and readiness, then publish the staged generation."""
    staging = STAGING_DIR
    marker_path = staging / STAGE_MARKER
    if not marker_path.is_file():
        write_failure_receipt(RESULTS_DIR, {"reason": "missing_stage"})
        print("[SIGNAL_BUNDLE_COMMIT_FAILED] no staged candidate")
        return 1
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "STAGED":
        write_failure_receipt(RESULTS_DIR, {"reason": "stage_not_ready"})
        return 1
    session = str(marker.get("session_as_of_date") or "")
    generated_at = str(marker.get("generated_at_utc") or "")
    if not session or not generated_at:
        write_failure_receipt(RESULTS_DIR, {"reason": "stage_missing_generation"})
        return 1

    import services.portfolio_allocation_engine as allocation

    alloc_rc = allocation.main(
        [
            "--signals",
            str(staging / "signals_with_rationale.csv"),
            "--lifecycle-effective",
            str(staging / "signal_lifecycle_effective.csv"),
            "--positions",
            str(RESULTS_DIR / "positions_snapshot.csv"),
            "--edge-sizing",
            str(RESULTS_DIR / "edge_sizing_recommendations.csv"),
            "--risk-overlay",
            str(RESULTS_DIR / "performance_risk_overlay.csv"),
            "--perf-by-symbol",
            str(RESULTS_DIR / "performance_intelligence_by_symbol.csv"),
            "--opportunities",
            str(RESULTS_DIR / "trade_opportunities.csv"),
            "--out-csv",
            str(staging / "portfolio_allocation_recommendations.csv"),
            "--out-json",
            str(staging / "portfolio_allocation_summary.json"),
        ]
    )
    if alloc_rc != 0:
        write_failure_receipt(
            RESULTS_DIR, {"reason": "allocation_failed", "exit_code": int(alloc_rc)}
        )
        print("[SIGNAL_BUNDLE_COMMIT_FAILED] allocation candidate refused")
        return int(alloc_rc or 1)

    readiness_path = RESULTS_DIR / "execution_readiness.json"
    if not readiness_path.is_file():
        write_failure_receipt(RESULTS_DIR, {"reason": "readiness_missing"})
        return 1
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["bundle_generated_at_utc"] = generated_at
    readiness["bundle_session_as_of_date"] = session
    readiness["live_execution_permitted"] = False
    (staging / "execution_readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        for name in (
            "signals.csv",
            "signals_with_rationale.csv",
            "signal_lifecycle.csv",
            "signal_lifecycle_effective.csv",
            "portfolio_allocation_recommendations.csv",
        ):
            _stamp_csv(staging / name, generated_at, session)
        held = held_symbols_from_snapshot(RESULTS_DIR / "positions_snapshot.csv")
        assert_held_positions_stay_long(pd.read_csv(staging / "signal_lifecycle.csv"), held)
        publish_coherent_bundle(RESULTS_DIR, staging, generated_at=generated_at, session=session)
    except Exception as exc:
        write_failure_receipt(RESULTS_DIR, {"reason": "publish_refused", "detail": str(exc)})
        print(f"[SIGNAL_BUNDLE_COMMIT_FAILED] {exc}")
        return 1

    print(
        f"[SIGNAL_BUNDLE_PUBLISHED] session={session} generated_at={generated_at} "
        "artifacts=signals,rationale,lifecycle,effective,allocation,readiness"
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in {"stage", "commit"}:
        print("usage: python -m services.coherent_signal_bundle stage|commit", file=sys.stderr)
        return 2
    if args[0] == "stage":
        return stage_signal_bundle()
    return commit_signal_bundle()


if __name__ == "__main__":
    raise SystemExit(main())
