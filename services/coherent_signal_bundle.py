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
PUBLIC_DIAGNOSTIC = "signals_asof_diagnostics.json"
PUBLISHED_ARTIFACTS = BUNDLE_FILES + (PUBLIC_DIAGNOSTIC,)
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
    from services.symbol_identity import SymbolIdentityCollision, assert_no_identity_collisions

    try:
        assert_no_identity_collisions(held, source="positions_snapshot")
    except SymbolIdentityCollision as exc:
        raise RuntimeError("symbol_identity_collision") from exc
    return held


_RISK_REDUCING_ACTIONS = frozenset({"EXIT", "TRIM"})


def _identity_indexes(frame: pd.DataFrame, column: str, symbol: str) -> list:
    from services.symbol_identity import identity_key

    target = identity_key(symbol)
    if not target:
        return []
    return [idx for idx, value in frame[column].items() if identity_key(value) == target]


def _reject_symbol_collisions(symbols: Iterable[object], *, source: str) -> None:
    from services.symbol_identity import SymbolIdentityCollision, assert_unique_identities

    try:
        assert_unique_identities(symbols, source=source)
    except SymbolIdentityCollision as exc:
        raise RuntimeError("symbol_identity_collision") from exc


def assert_held_positions_stay_long(lifecycle: pd.DataFrame, held: Iterable[str]) -> None:
    """Broker-held names stay LONG. EXIT is preserved; WAIT is refused."""
    if lifecycle is None or lifecycle.empty:
        raise RuntimeError("lifecycle candidate is empty")
    symbol_col = "ticker" if "ticker" in lifecycle.columns else "symbol"
    action_col = "lifecycle_action" if "lifecycle_action" in lifecycle.columns else "stance"
    _reject_symbol_collisions(lifecycle[symbol_col].tolist(), source="lifecycle")
    for symbol in sorted({str(s).strip().upper() for s in held if str(s).strip()}):
        indexes = _identity_indexes(lifecycle, symbol_col, symbol)
        if not indexes:
            raise RuntimeError(f"held symbol {symbol} missing from lifecycle candidate")
        if len(indexes) > 1:
            raise RuntimeError(f"duplicate symbol {symbol}")
        row = lifecycle.loc[indexes[0]]
        position = str(row.get("position_state") or "").strip().upper()
        action = str(row.get(action_col) or "").strip().upper()
        if position != "LONG":
            raise RuntimeError(f"held symbol {symbol} position_state={position}")
        if action in {"", "WAIT"}:
            raise RuntimeError(f"held symbol {symbol} action={action}")


def _frame_symbol_column(frame: pd.DataFrame) -> str:
    if "ticker" in frame.columns:
        return "ticker"
    if "symbol" in frame.columns:
        return "symbol"
    raise RuntimeError("effective lifecycle is missing ticker")


def _raw_action_map(raw: pd.DataFrame) -> dict[str, str]:
    from services.symbol_identity import identity_key

    if raw is None or raw.empty:
        return {}
    if "ticker" not in raw.columns and "symbol" not in raw.columns:
        return {}
    symbol_col = _frame_symbol_column(raw)
    action_col = "lifecycle_action" if "lifecycle_action" in raw.columns else "stance"
    _reject_symbol_collisions(raw[symbol_col].tolist(), source="lifecycle")
    actions: dict[str, str] = {}
    for _, row in raw.iterrows():
        symbol = identity_key(row.get(symbol_col))
        if symbol:
            actions[symbol] = str(row.get(action_col) or "").strip().upper()
    return actions


def apply_snapshot_held_effective_state(
    effective: pd.DataFrame,
    raw: pd.DataFrame,
    held: Iterable[str],
) -> pd.DataFrame:
    """Operational held state for the effective lifecycle. The raw frame is not written."""
    from services.symbol_identity import identity_key

    out = effective.copy()
    if out.empty:
        return out
    symbol_col = _frame_symbol_column(out)
    original_tickers = out[symbol_col].astype(str).tolist()
    _reject_symbol_collisions(original_tickers, source="effective_lifecycle")
    actions = _raw_action_map(raw)
    for symbol in {str(item).strip().upper() for item in held if str(item).strip()}:
        indexes = _identity_indexes(out, symbol_col, symbol)
        if not indexes:
            continue
        if len(indexes) > 1:
            raise RuntimeError(f"duplicate symbol {symbol}")
        index = indexes[0]
        action = actions.get(identity_key(symbol), "")
        out.at[index, "effective_position_state"] = "LONG"
        if action in _RISK_REDUCING_ACTIONS:
            out.at[index, "effective_stance"] = action
        elif action in {"", "WAIT", "FLAT", "HOLD"}:
            out.at[index, "effective_stance"] = "HOLD"
    if out[symbol_col].astype(str).tolist() != original_tickers:
        raise RuntimeError("candidate_spelling_not_preserved")
    return out


def assert_effective_held_positions(
    effective: pd.DataFrame,
    held: Iterable[str],
    snapshot_evidence: Mapping[str, object],
) -> None:
    """Fail closed unless snapshot-held names are LONG on the bound effective lifecycle."""
    if effective is None or effective.empty:
        raise RuntimeError("effective lifecycle candidate is empty")
    for column in ("positions_snapshot_sha256", "positions_snapshot_ts"):
        if column not in effective.columns:
            raise RuntimeError("effective_lifecycle_unbound")
    expected_hash = str(snapshot_evidence.get("sha256") or "")
    expected_ts = str(snapshot_evidence.get("snapshot_ts") or "")
    if set(effective["positions_snapshot_sha256"].astype(str)) != {expected_hash}:
        raise RuntimeError("positions_snapshot_hash_mismatch")
    if set(effective["positions_snapshot_ts"].astype(str)) != {expected_ts}:
        raise RuntimeError("positions_snapshot_timestamp_mismatch")
    symbol_col = _frame_symbol_column(effective)
    _reject_symbol_collisions(effective[symbol_col].tolist(), source="effective_lifecycle")
    for symbol in sorted({str(item).strip().upper() for item in held if str(item).strip()}):
        indexes = _identity_indexes(effective, symbol_col, symbol)
        if not indexes:
            raise RuntimeError(f"held symbol {symbol} missing from effective lifecycle")
        if len(indexes) > 1:
            raise RuntimeError(f"duplicate symbol {symbol}")
        row = effective.loc[indexes[0]]
        position = str(row.get("effective_position_state") or "").strip().upper()
        action = str(row.get("effective_stance") or "").strip().upper()
        if position != "LONG":
            raise RuntimeError(
                f"held symbol {symbol} effective_position_state={position or 'FLAT'} "
                "terminal_flat_unproven"
            )
        if action in {"", "WAIT", "FLAT"}:
            raise RuntimeError(f"held symbol {symbol} effective_stance={action or 'WAIT'}")


def guard_staged_held_positions(production: Path, staging: Path) -> None:
    """Apply snapshot-held operational state, then validate the staged effective lifecycle."""
    from services.build_effective_lifecycle import read_positions_snapshot_evidence

    raw_path = staging / "signal_lifecycle.csv"
    effective_path = staging / "signal_lifecycle_effective.csv"
    if not effective_path.is_file():
        raise RuntimeError("effective lifecycle missing")
    raw_bytes = raw_path.read_bytes() if raw_path.is_file() else b""
    evidence = read_positions_snapshot_evidence(production / "positions_snapshot.csv")
    effective = pd.read_csv(effective_path)
    raw = pd.read_csv(raw_path) if raw_path.is_file() else pd.DataFrame()
    for column, key in (
        ("positions_snapshot_sha256", "sha256"),
        ("positions_snapshot_ts", "snapshot_ts"),
    ):
        if column not in effective.columns or set(effective[column].astype(str)) != {
            str(evidence.get(key) or "")
        }:
            raise RuntimeError(
                "positions_snapshot_hash_mismatch"
                if key == "sha256"
                else "positions_snapshot_timestamp_mismatch"
            )
    held = held_symbols_from_snapshot(production / "positions_snapshot.csv")
    updated = apply_snapshot_held_effective_state(effective, raw, held)
    if not updated.equals(effective):
        _atomic_bytes(
            effective_path,
            updated.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )
    if raw_path.is_file() and raw_path.read_bytes() != raw_bytes:
        raise RuntimeError("raw_lifecycle_mutated")
    assert_effective_held_positions(updated, held, evidence)


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


def _frame_identity(frame: pd.DataFrame) -> tuple[str, str]:
    session = ""
    generated_at = ""
    if "session_as_of_date" in frame.columns and not frame.empty:
        session = str(frame["session_as_of_date"].iloc[0])[:10]
    elif "bundle_session_as_of_date" in frame.columns and not frame.empty:
        session = str(frame["bundle_session_as_of_date"].iloc[0])[:10]
    if "generated_at_utc" in frame.columns and not frame.empty:
        generated_at = str(frame["generated_at_utc"].iloc[0])
    elif "bundle_generated_at_utc" in frame.columns and not frame.empty:
        generated_at = str(frame["bundle_generated_at_utc"].iloc[0])
    return session, generated_at


def bind_staged_diagnostic(staging: Path, signals_path: Path) -> dict:
    """Bind the staged diagnostic to this candidate. Production files are not written."""
    from services.lifecycle_artifact_guard import (
        CANDIDATE_PROVENANCE_SCHEMA,
        candidate_bundle_id,
        file_sha256,
    )

    diagnostic_path = staging / PUBLIC_DIAGNOSTIC
    if not diagnostic_path.is_file() or diagnostic_path.stat().st_size == 0:
        raise ValueError("staged_diagnostic_missing")
    try:
        document = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("staged_diagnostic_malformed") from exc
    if not isinstance(document, dict):
        raise ValueError("staged_diagnostic_malformed")
    frame = pd.read_csv(signals_path)
    if frame.empty or "ticker" not in frame.columns:
        raise ValueError("staged_diagnostic_malformed")
    session, generated_at = _frame_identity(frame)
    diagnostic_session = str(
        document.get("session_asof_date") or document.get("effective_asof_date") or ""
    )[:10]
    if not session or session != diagnostic_session:
        raise ValueError("staged_diagnostic_session_mismatch")
    if not generated_at or str(document.get("generated_at_utc") or "") != generated_at:
        raise ValueError("staged_diagnostic_timestamp_mismatch")
    raw_symbols = document.get("current_tradable_symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise ValueError("staged_diagnostic_malformed")
    expected = sorted(
        {str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()}
    )
    from services.symbol_identity import SymbolIdentityCollision, candidate_display_map

    try:
        candidate_display_map(expected)
    except SymbolIdentityCollision as exc:
        raise ValueError("symbol_identity_collision") from exc
    actual = sorted(
        {str(symbol).strip().upper() for symbol in frame["ticker"].tolist() if str(symbol).strip()}
    )
    if expected != actual:
        raise ValueError("staged_diagnostic_universe_mismatch")
    provenance = {
        "schema": CANDIDATE_PROVENANCE_SCHEMA,
        "bundle_id": candidate_bundle_id(session, generated_at, expected),
        "session_as_of_date": session,
        "signals_sha256": file_sha256(signals_path),
        "expected_symbols": expected,
        "generated_at_utc": generated_at,
    }
    document["candidate_provenance"] = provenance
    _atomic_bytes(
        diagnostic_path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return provenance


def refresh_staged_signals_hash(staging: Path) -> str:
    """Record the post-stamp signals bytes without changing the candidate identity."""
    from services.lifecycle_artifact_guard import file_sha256

    diagnostic_path = staging / PUBLIC_DIAGNOSTIC
    document = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    provenance = document.get("candidate_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("staged_diagnostic_unbound")
    digest = file_sha256(staging / "signals.csv")
    provenance["signals_sha256"] = digest
    document["candidate_provenance"] = provenance
    _atomic_bytes(
        diagnostic_path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return digest


def _require_bound_diagnostic(staged: Path, generated_at: str, session: str) -> None:
    from services.lifecycle_artifact_guard import CANDIDATE_PROVENANCE_SCHEMA, file_sha256

    document = json.loads((staged / PUBLIC_DIAGNOSTIC).read_text(encoding="utf-8"))
    provenance = document.get("candidate_provenance") if isinstance(document, dict) else None
    if not isinstance(provenance, dict) or provenance.get("schema") != CANDIDATE_PROVENANCE_SCHEMA:
        raise RuntimeError("staged diagnostic is not bound to this candidate")
    if str(provenance.get("session_as_of_date") or "") != session:
        raise RuntimeError("staged diagnostic session does not match the candidate")
    if str(provenance.get("generated_at_utc") or "") != generated_at:
        raise RuntimeError("staged diagnostic timestamp does not match the candidate")
    if str(provenance.get("signals_sha256") or "") != file_sha256(staged / "signals.csv"):
        raise RuntimeError("staged diagnostic signals hash does not match")
    if not str(provenance.get("bundle_id") or "").strip():
        raise RuntimeError("staged diagnostic is missing bundle identity")


def _require_staged_provenance(staging: Path, marker: Mapping[str, object]) -> None:
    from services.lifecycle_artifact_guard import file_sha256

    document = json.loads((staging / PUBLIC_DIAGNOSTIC).read_text(encoding="utf-8"))
    provenance = document.get("candidate_provenance") if isinstance(document, dict) else None
    if not isinstance(provenance, dict):
        raise RuntimeError("staged_diagnostic_unbound")
    if str(provenance.get("bundle_id") or "") != str(marker.get("bundle_id") or ""):
        raise RuntimeError("staged_diagnostic_bundle_mismatch")
    if str(provenance.get("session_as_of_date") or "") != str(
        marker.get("session_as_of_date") or ""
    ):
        raise RuntimeError("staged_diagnostic_session_mismatch")
    if str(provenance.get("generated_at_utc") or "") != str(marker.get("generated_at_utc") or ""):
        raise RuntimeError("staged_diagnostic_timestamp_mismatch")
    if str(provenance.get("signals_sha256") or "") != file_sha256(staging / "signals.csv"):
        raise RuntimeError("staged_diagnostic_signals_hash_mismatch")


def _candidate_diagnostic_excerpt(document: Mapping[str, object]) -> dict:
    skips = document.get("per_symbol_skips")
    return {
        "generated_at_utc": document.get("generated_at_utc"),
        "session_asof_date": document.get("session_asof_date"),
        "have_session": document.get("have_session"),
        "tradable_tickers": document.get("tradable_tickers"),
        "coverage": document.get("coverage"),
        "candidate_provenance": document.get("candidate_provenance"),
        "per_symbol_skips": skips if isinstance(skips, list) else [],
        "candidate_diagnostic_path": f"data/results/signal_bundle_stage/{PUBLIC_DIAGNOSTIC}",
    }


def _validate_staged(staged: Path, generated_at: str, session: str) -> None:
    missing = [name for name in PUBLISHED_ARTIFACTS if not (staged / name).is_file()]
    if missing:
        raise RuntimeError(f"candidate bundle incomplete: {missing}")
    signals = pd.read_csv(staged / "signals.csv")
    if signals.empty or "ticker" not in signals.columns or signals.dropna(how="all").empty:
        raise RuntimeError("refusing to publish a header-only signal file")
    _require_bound_diagnostic(staged, generated_at, session)
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
    """Replace the seven production artifacts, or restore them if a replace fails."""
    _validate_staged(staged, generated_at, session)
    prior: Dict[str, Optional[bytes]] = {}
    for name in PUBLISHED_ARTIFACTS:
        path = production / name
        prior[name] = path.read_bytes() if path.is_file() else None
    replaced: list[str] = []
    try:
        for name in PUBLISHED_ARTIFACTS:
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


def tradable_invalid_ohlc_skips(diag: Mapping[str, object]) -> list:
    """Required tradable bars rejected by the shared OHLC contract."""
    found = []
    for rec in diag.get("per_symbol_skips") or []:
        if not isinstance(rec, Mapping):
            continue
        if str(rec.get("reason") or "") != "INVALID_PROCESSED_SESSION_BAR":
            continue
        if str(rec.get("class") or "tradable") == "index":
            continue
        found.append(dict(rec))
    return found


def _staged_ohlc_block(staging: Path) -> list:
    path = staging / PUBLIC_DIAGNOSTIC
    if not path.is_file():
        return []
    try:
        diag = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(diag, dict):
        return []
    return tradable_invalid_ohlc_skips(diag)


def stage_effective_from_candidate(
    production: Path, staging: Path, provenance: Mapping[str, object]
) -> int:
    """Write the candidate effective lifecycle. Does not call the CLI main()."""
    from services.build_effective_lifecycle import (
        build_staged_effective_lifecycle,
        read_positions_snapshot_evidence,
    )
    from services.lifecycle_artifact_guard import file_sha256

    snapshot = production / "positions_snapshot.csv"
    lifecycle_path = staging / "signal_lifecycle.csv"
    expected = list(provenance.get("expected_symbols") or [])
    try:
        evidence = read_positions_snapshot_evidence(snapshot)
        build_staged_effective_lifecycle(
            lifecycle_path=lifecycle_path,
            reconciliation_path=production / "lifecycle_reconciliation.csv",
            open_orders_path=production / "open_orders_snapshot.csv",
            output_path=staging / "signal_lifecycle_effective.csv",
            positions_snapshot=snapshot,
            snapshot_evidence=evidence,
            bundle_id=str(provenance.get("bundle_id") or ""),
            session_as_of_date=str(provenance.get("session_as_of_date") or ""),
            lifecycle_sha256=file_sha256(lifecycle_path),
            expected_symbols=expected,
            min_rows=len(expected),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _fail_stage(
            production,
            staging,
            reason="build_effective_lifecycle_failed",
            detail=str(exc),
        )
        return 1
    return 0


def _candidate_symbol_spellings(staging: Path) -> list[str]:
    for name in (
        "signals.csv",
        "signals_with_rationale.csv",
        "signal_lifecycle.csv",
        "signal_lifecycle_effective.csv",
    ):
        path = staging / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        column = "ticker" if "ticker" in frame.columns else "symbol"
        if column not in frame.columns:
            continue
        symbols = _symbol_list(frame)
        if symbols:
            return symbols
    return []


def _restore_allocation_candidate_spellings(staging: Path) -> None:
    from services.symbol_identity import SymbolIdentityCollision, restore_candidate_spellings

    path = staging / "portfolio_allocation_recommendations.csv"
    frame = pd.read_csv(path)
    if frame.empty:
        return
    column = "ticker" if "ticker" in frame.columns else "symbol"
    if column not in frame.columns:
        raise RuntimeError("allocation_symbol_mismatch")
    candidates = _candidate_symbol_spellings(staging)
    if not candidates:
        raise RuntimeError("candidate_symbols_missing")
    try:
        frame[column] = restore_candidate_spellings(frame[column].tolist(), candidates)
    except SymbolIdentityCollision as exc:
        raise RuntimeError("symbol_identity_collision") from exc
    path.write_text(frame.to_csv(index=False), encoding="utf-8", newline="")


def run_staged_allocation(results_dir: Path, staging: Path) -> int:
    """Allocate from the staged effective lifecycle into the staging directory."""
    import services.portfolio_allocation_engine as allocation
    from services.symbol_identity import SymbolIdentityCollision

    try:
        rc = allocation.main(
            [
                "--signals",
                str(staging / "signals_with_rationale.csv"),
                "--lifecycle-effective",
                str(staging / "signal_lifecycle_effective.csv"),
                "--positions",
                str(results_dir / "positions_snapshot.csv"),
                "--edge-sizing",
                str(results_dir / "edge_sizing_recommendations.csv"),
                "--risk-overlay",
                str(results_dir / "performance_risk_overlay.csv"),
                "--perf-by-symbol",
                str(results_dir / "performance_intelligence_by_symbol.csv"),
                "--opportunities",
                str(results_dir / "trade_opportunities.csv"),
                "--out-csv",
                str(staging / "portfolio_allocation_recommendations.csv"),
                "--out-json",
                str(staging / "portfolio_allocation_summary.json"),
            ]
        )
    except SymbolIdentityCollision as exc:
        print(f"[PORTFOLIO_ALLOCATION] symbol_identity_collision {exc}", flush=True)
        return 1
    if rc != 0:
        return int(rc)
    try:
        _restore_allocation_candidate_spellings(staging)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[PORTFOLIO_ALLOCATION] candidate spelling refused: {exc}", flush=True)
        return 1
    return 0


def write_staged_readiness(
    results_dir: Path, staging: Path, *, generated_at: str, session: str
) -> None:
    readiness_path = results_dir / "execution_readiness.json"
    if not readiness_path.is_file():
        raise RuntimeError("readiness_missing")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["bundle_generated_at_utc"] = generated_at
    readiness["bundle_session_as_of_date"] = session
    readiness["live_execution_permitted"] = False
    (staging / "execution_readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _symbol_list(frame: pd.DataFrame) -> list[str]:
    column = "ticker" if "ticker" in frame.columns else "symbol"
    return [str(value).strip().upper() for value in frame[column].tolist() if str(value).strip()]


def _symbol_set(frame: pd.DataFrame) -> set[str]:
    return set(_symbol_list(frame))


def _assert_candidate_symbol_set(
    actual: Iterable[str],
    expected: Iterable[str],
    *,
    mismatch: str,
) -> None:
    """Identity sets must match, and artifact rows must keep the candidate spelling."""
    from services.symbol_identity import (
        SymbolIdentityCollision,
        candidate_display_map,
        identity_key,
    )

    try:
        display = candidate_display_map(expected)
        _reject_symbol_collisions(actual, source="artifact")
    except SymbolIdentityCollision as exc:
        raise RuntimeError("symbol_identity_collision") from exc
    actual_raw = {str(symbol).strip().upper() for symbol in actual if str(symbol).strip()}
    if {identity_key(symbol) for symbol in actual_raw} != set(display):
        raise RuntimeError(mismatch)
    if actual_raw != set(display.values()):
        raise RuntimeError("candidate_spelling_not_preserved")


def _require_effective_snapshot(staging: Path, marker: Mapping[str, object]) -> None:
    """Reject a staged candidate whose local snapshot or effective binding moved."""
    if "positions_snapshot_sha256" not in marker:
        return
    from services.build_effective_lifecycle import read_positions_snapshot_evidence
    from services.lifecycle_artifact_guard import file_sha256

    try:
        current = read_positions_snapshot_evidence(RESULTS_DIR / "positions_snapshot.csv")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if current["sha256"] != str(marker.get("positions_snapshot_sha256") or ""):
        raise RuntimeError("positions_snapshot_changed")
    if current["snapshot_ts"] != str(marker.get("positions_snapshot_ts") or ""):
        raise RuntimeError("positions_snapshot_timestamp_mismatch")
    frame = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    binding = {
        "candidate_bundle_id": "bundle_id",
        "candidate_session_as_of_date": "session_as_of_date",
        "candidate_lifecycle_sha256": "lifecycle_sha256",
        "positions_snapshot_sha256": "positions_snapshot_sha256",
        "positions_snapshot_ts": "positions_snapshot_ts",
    }
    for column, key in binding.items():
        if column not in frame.columns:
            raise RuntimeError("effective_lifecycle_unbound")
        if set(frame[column].astype(str)) != {str(marker.get(key) or "")}:
            raise RuntimeError("effective_lifecycle_binding_mismatch")
    if file_sha256(staging / "signal_lifecycle.csv") != str(marker.get("lifecycle_sha256") or ""):
        raise RuntimeError("lifecycle_hash_mismatch")
    expected = marker.get("expected_symbols")
    if isinstance(expected, list) and expected:
        _assert_candidate_symbol_set(
            _symbol_list(frame),
            expected,
            mismatch="effective_lifecycle_symbol_mismatch",
        )


def _require_allocation_symbols(staging: Path, marker: Mapping[str, object]) -> None:
    expected = marker.get("expected_symbols")
    if not isinstance(expected, list) or not expected:
        return
    frame = pd.read_csv(staging / "portfolio_allocation_recommendations.csv")
    _assert_candidate_symbol_set(
        _symbol_list(frame),
        expected,
        mismatch="allocation_symbol_mismatch",
    )


def stage_signal_bundle() -> int:
    """Build the candidate in staging. Production bundle files stay in place."""
    import services.apply_signal_lifecycle as lifecycle
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
    generate.ASOF_DIAG_PATH = str(staging / PUBLIC_DIAGNOSTIC)

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
    blocked = _staged_ohlc_block(staging)
    if blocked:
        _fail_stage(
            production,
            staging,
            reason="invalid_session_ohlc",
            ohlc_violations=[row.get("ohlc_violation") or row for row in blocked],
        )
        return 1

    try:
        provenance = bind_staged_diagnostic(staging, signals_path)
    except ValueError as exc:
        _fail_stage(production, staging, reason=str(exc))
        return 1

    lifecycle.RESULTS_DIR = staging
    lifecycle.SIGNALS_RATIONALE = staging / "signals_with_rationale.csv"
    lifecycle.SIGNALS_FALLBACK = staging / "signals.csv"
    lifecycle.OUT_LIFECYCLE = staging / "signal_lifecycle.csv"
    lifecycle.STATE_PATH = staging / "signal_state.json"
    lifecycle.POSITIONS_SNAPSHOT = production / "positions_snapshot.csv"
    rc = lifecycle.main(
        [
            "--diagnostics",
            str(staging / PUBLIC_DIAGNOSTIC),
            "--signals-artifact",
            str(signals_path),
            "--bundle-id",
            str(provenance["bundle_id"]),
            "--candidate-session",
            str(provenance["session_as_of_date"]),
            "--generated-at",
            str(provenance["generated_at_utc"]),
            "--signals-sha256",
            str(provenance["signals_sha256"]),
            "--expected-symbols",
            ",".join(provenance["expected_symbols"]),
        ]
    )
    if rc != 0:
        _fail_stage(production, staging, reason="apply_signal_lifecycle_failed", exit_code=rc)
        return int(rc or 1)

    eff_rc = stage_effective_from_candidate(production, staging, provenance)
    if eff_rc != 0:
        return int(eff_rc)

    try:
        guard_staged_held_positions(production, staging)
    except Exception as exc:
        _fail_stage(production, staging, reason="held_position_guard", detail=str(exc))
        return 1

    session = str(provenance["session_as_of_date"])
    generated_at = str(provenance["generated_at_utc"])
    try:
        effective_frame = pd.read_csv(staging / "signal_lifecycle_effective.csv")
        marker = {
            "schema": SCHEMA_STAGE,
            "status": "STAGED",
            "bundle_id": provenance["bundle_id"],
            "session_as_of_date": session,
            "generated_at_utc": generated_at,
            "signals_sha256": provenance["signals_sha256"],
            "lifecycle_sha256": str(effective_frame["candidate_lifecycle_sha256"].iloc[0]),
            "positions_snapshot_sha256": str(effective_frame["positions_snapshot_sha256"].iloc[0]),
            "positions_snapshot_ts": str(effective_frame["positions_snapshot_ts"].iloc[0]),
            "expected_symbols": list(provenance["expected_symbols"]),
            "published_signal_artifacts": False,
        }
    except (OSError, IndexError, KeyError, ValueError) as exc:
        _fail_stage(production, staging, reason="effective_lifecycle_unbound", detail=str(exc))
        return 1
    (staging / STAGE_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[SIGNAL_BUNDLE_STAGED] session={session} generated_at={generated_at}")
    return 0


def _fail_stage(production: Path, staging: Path, **payload: object) -> None:
    """Record the candidate failure without replacing any published artifact."""
    body = dict(payload)
    diagnostic = staging / PUBLIC_DIAGNOSTIC
    if diagnostic.is_file():
        try:
            document = json.loads(diagnostic.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            document = None
        if isinstance(document, dict):
            body["candidate_diagnostic"] = _candidate_diagnostic_excerpt(document)
    write_failure_receipt(production, body)
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
    blocked = _staged_ohlc_block(staging)
    if blocked:
        write_failure_receipt(
            RESULTS_DIR,
            {
                "reason": "invalid_session_ohlc",
                "ohlc_violations": [row.get("ohlc_violation") or row for row in blocked],
            },
        )
        print("[SIGNAL_BUNDLE_COMMIT_FAILED] invalid session OHLC")
        return 1
    try:
        _require_staged_provenance(staging, marker)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        write_failure_receipt(
            RESULTS_DIR,
            {"reason": "staged_diagnostic_rejected", "detail": str(exc)},
        )
        print(f"[SIGNAL_BUNDLE_COMMIT_FAILED] {exc}")
        return 1
    try:
        _require_effective_snapshot(staging, marker)
    except (OSError, RuntimeError, ValueError) as exc:
        write_failure_receipt(
            RESULTS_DIR,
            {"reason": "effective_lifecycle_rejected", "detail": str(exc)},
        )
        print(f"[SIGNAL_BUNDLE_COMMIT_FAILED] {exc}")
        return 1

    snapshot_bound = "positions_snapshot_sha256" in marker
    if snapshot_bound:
        try:
            guard_staged_held_positions(RESULTS_DIR, staging)
        except (OSError, RuntimeError, ValueError) as exc:
            write_failure_receipt(
                RESULTS_DIR,
                {"reason": "held_position_guard", "detail": str(exc)},
            )
            print(f"[SIGNAL_BUNDLE_COMMIT_FAILED] {exc}")
            return 1

    alloc_rc = run_staged_allocation(RESULTS_DIR, staging)
    if alloc_rc != 0:
        write_failure_receipt(
            RESULTS_DIR, {"reason": "allocation_failed", "exit_code": int(alloc_rc)}
        )
        print("[SIGNAL_BUNDLE_COMMIT_FAILED] allocation candidate refused")
        return int(alloc_rc or 1)
    try:
        _require_allocation_symbols(staging, marker)
        write_staged_readiness(RESULTS_DIR, staging, generated_at=generated_at, session=session)
    except (OSError, RuntimeError, ValueError) as exc:
        reason = str(exc) if str(exc) == "readiness_missing" else "allocation_symbol_mismatch"
        if str(exc) not in {"readiness_missing", "allocation_symbol_mismatch"}:
            reason = "staged_downstream_failed"
        write_failure_receipt(RESULTS_DIR, {"reason": reason, "detail": str(exc)})
        print(f"[SIGNAL_BUNDLE_COMMIT_FAILED] {exc}")
        return 1

    try:
        for name in (
            "signals.csv",
            "signals_with_rationale.csv",
            "signal_lifecycle.csv",
            "signal_lifecycle_effective.csv",
            "portfolio_allocation_recommendations.csv",
        ):
            _stamp_csv(staging / name, generated_at, session)
        refresh_staged_signals_hash(staging)
        if snapshot_bound:
            evidence_path = RESULTS_DIR / "positions_snapshot.csv"
            from services.build_effective_lifecycle import read_positions_snapshot_evidence

            assert_effective_held_positions(
                pd.read_csv(staging / "signal_lifecycle_effective.csv"),
                held_symbols_from_snapshot(evidence_path),
                read_positions_snapshot_evidence(evidence_path),
            )
        else:
            held = held_symbols_from_snapshot(RESULTS_DIR / "positions_snapshot.csv")
            assert_held_positions_stay_long(pd.read_csv(staging / "signal_lifecycle.csv"), held)
        publish_coherent_bundle(RESULTS_DIR, staging, generated_at=generated_at, session=session)
    except Exception as exc:
        write_failure_receipt(RESULTS_DIR, {"reason": "publish_refused", "detail": str(exc)})
        print(f"[SIGNAL_BUNDLE_COMMIT_FAILED] {exc}")
        return 1

    print(
        f"[SIGNAL_BUNDLE_PUBLISHED] session={session} generated_at={generated_at} "
        "artifacts=signals,rationale,lifecycle,effective,allocation,readiness,diagnostic"
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
