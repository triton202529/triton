"""
phase_198w_market_open_refresh_pipeline_validation.py
Phase 198W — Validate restored market-open signal/lifecycle refresh pipeline.

Runs the refresh sequence (generate_signals -> apply_signal_lifecycle ->
build_effective_lifecycle), recomputes readiness, and proves lifecycle/signals
are fresh afterwards. Read-only w.r.t. broker: no orders, no approvals, no
authorization artifacts, live execution stays blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"

OUTPUT_JSON = RESULTS / "phase_198w_market_open_refresh_pipeline_validation.json"
OUTPUT_MD = RESULTS / "phase_198w_market_open_refresh_pipeline_validation.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

LIFECYCLE_RAW = RESULTS / "signal_lifecycle.csv"
LIFECYCLE_EFFECTIVE = RESULTS / "signal_lifecycle_effective.csv"
SIGNALS_CSV = RESULTS / "signals.csv"
EXEC_READINESS = RESULTS / "execution_readiness.json"
CERT_PATH = RESULTS / "capital_preservation_certification.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
ALLOC_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
ALLOC_RECS = RESULTS / "portfolio_allocation_recommendations.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"

MARKER = "MARKET_OPEN_REFRESH_PIPELINE_VALIDATION_ONLY"
GLD_SYMBOL = "GLD"
GLD_TARGET_WEIGHT = 0.20

# Mirrors governance_execution_readiness thresholds (48h).
LIFECYCLE_FRESHNESS_HOURS = 48
SIGNALS_FRESHNESS_HOURS = 48
LIFECYCLE_FRESHNESS_SECONDS = LIFECYCLE_FRESHNESS_HOURS * 3600
SIGNALS_FRESHNESS_SECONDS = SIGNALS_FRESHNESS_HOURS * 3600

REFRESH_STEPS = (
    ("generate_signals", "services.generate_signals"),
    ("apply_signal_lifecycle", "services.apply_signal_lifecycle"),
    ("build_effective_lifecycle", "services.build_effective_lifecycle"),
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


def _file_age_seconds(path: Path) -> Optional[float]:
    try:
        if path.is_file():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return round((_utc_now() - mtime).total_seconds(), 3)
    except OSError:
        pass
    return None


def _file_mtime_iso(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            return (
                datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
    except OSError:
        pass
    return None


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(0, len(list(csv.reader(f))) - 1)
    except Exception:
        return 0


def _run_module(module: str, extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    args = [sys.executable, "-m", module]
    if extra_args:
        args.extend(extra_args)
    try:
        proc = subprocess.run(
            args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip().splitlines()
        return {
            "module": module,
            "exit_code": proc.returncode,
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "output_tail": tail[-8:],
        }
    except subprocess.TimeoutExpired:
        return {"module": module, "exit_code": 124, "status": "TIMEOUT", "output_tail": []}
    except Exception as exc:  # pragma: no cover
        return {"module": module, "exit_code": 1, "status": "ERROR", "output_tail": [str(exc)]}


def _gld_from_positions(path: Path) -> Dict[str, Any]:
    """GLD weight = GLD market_value / sum(|position market_value|), matching 198L."""
    qty = 0.0
    market_value = 0.0
    total_invested = 0.0
    if not path.is_file():
        return {"qty": 0, "weight": 0.0, "market_value": 0.0}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mv = abs(_safe_float(row.get("market_value") or row.get("value")))
                total_invested += mv
                sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
                if sym != GLD_SYMBOL:
                    continue
                qty = _safe_float(row.get("qty") or row.get("quantity") or row.get("position_qty"))
                market_value = mv
    except Exception:
        pass
    weight = (market_value / total_invested) if total_invested > 0 else 0.0
    return {
        "qty": int(qty) if qty == int(qty) else qty,
        "weight": round(weight, 6),
        "market_value": market_value,
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


def _render_markdown(payload: Dict[str, Any]) -> str:
    a = payload.get("audit") or {}
    steps = a.get("refresh_steps") or []
    step_lines = (
        "\n".join(
            f"- `{s.get('module')}`: {s.get('status')} (exit={s.get('exit_code')})" for s in steps
        )
        or "- None"
    )
    return f"""# TRITON Phase 198W — Market-Open Refresh Pipeline Validation

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Refresh pipeline status:** `{a.get("refresh_pipeline_status")}`

## Executive summary

{payload.get("executive_summary")}

## Refresh steps

{step_lines}

## Artifact freshness after refresh

| Artifact | Age (s) | Threshold (s) | Stale |
|----------|---------|---------------|-------|
| signals.csv | {a.get("signals_file_age_seconds")} | {SIGNALS_FRESHNESS_SECONDS} | {a.get("signals_stale_after_refresh")} |
| signal_lifecycle.csv | {a.get("lifecycle_file_age_seconds")} | {LIFECYCLE_FRESHNESS_SECONDS} | {a.get("lifecycle_stale_after_refresh")} |
| signal_lifecycle_effective.csv | {a.get("effective_lifecycle_file_age_seconds")} | {LIFECYCLE_FRESHNESS_SECONDS} | {a.get("effective_lifecycle_stale_after_refresh")} |

## Readiness / certification

| Field | Value |
|-------|-------|
| execution_readiness_status | {a.get("execution_readiness_status")} |
| execution_readiness_failed_checks | {a.get("execution_readiness_failed_checks")} |
| certification_status | {a.get("certification_status")} |
| certification_failed_requirements | {a.get("certification_failed_requirements")} |

## Safety / posture

| Field | Value |
|-------|-------|
| broker_orders_submitted | {a.get("broker_orders_submitted")} |
| open_orders_count | {a.get("open_orders_count")} |
| paper_arm_valid | {a.get("paper_arm_valid")} |
| live_execution_blocked | {a.get("live_execution_blocked")} |
| approval_artifact_present | {a.get("approval_artifact_present")} |
| broker_auth_artifact_present | {a.get("broker_auth_artifact_present")} |
| gld_qty / gld_weight | {a.get("gld_qty")} / {a.get("gld_weight")} |
| gld_target_satisfied | {a.get("gld_target_satisfied")} |
| allocation_open_new / add | {a.get("allocation_open_new_count")} / {a.get("allocation_add_count")} |

## Recommended runtime posture

{a.get("recommended_runtime_posture")}

---

*{MARKER}*
"""


def run_validation(*, run_refresh: bool = True, verbose: bool = True) -> Dict[str, Any]:
    from services.governance_execution_readiness import (
        LIFECYCLE_FRESHNESS_HOURS as GOV_LC_H,
        SIGNALS_FRESHNESS_HOURS as GOV_SIG_H,
        _check_lifecycle_integrity,
        _check_signal_freshness,
    )
    from services.paper_arm_status import build_paper_arm_status
    from services.portfolio_history_sync import load_live_execution_flag

    refresh_steps: List[Dict[str, Any]] = []
    if run_refresh:
        # Run all three refresh steps and report each independently for
        # diagnostics. Runtime safe-block semantics (stop on first failure,
        # no orders) live in run_market_open.ps1; here we want full visibility
        # into which stage failed and whether the downstream stages recover.
        for _, module in REFRESH_STEPS:
            refresh_steps.append(_run_module(module))
        # Recompute readiness from refreshed inputs (read-only).
        _run_module("services.execution_readiness")
    else:
        refresh_steps = [
            {"module": m, "exit_code": None, "status": "SKIPPED", "output_tail": []}
            for _, m in REFRESH_STEPS
        ]

    step_status = {s["module"].split(".")[-1]: s for s in refresh_steps}
    gen_status = (step_status.get("generate_signals") or {}).get("status", "SKIPPED")
    apply_status = (step_status.get("apply_signal_lifecycle") or {}).get("status", "SKIPPED")
    build_status = (step_status.get("build_effective_lifecycle") or {}).get("status", "SKIPPED")

    all_refresh_ok = all(
        (step_status.get(name) or {}).get("exit_code") == 0
        for name in ("generate_signals", "apply_signal_lifecycle", "build_effective_lifecycle")
    )
    refresh_pipeline_status = "PASS" if all_refresh_ok else "SAFE_BLOCKED"

    refresh_failure_reason = None
    if not all_refresh_ok:
        for name in ("generate_signals", "apply_signal_lifecycle", "build_effective_lifecycle"):
            step = step_status.get(name) or {}
            if step.get("exit_code") not in (0, None):
                tail = step.get("output_tail") or []
                detail = next(
                    (ln for ln in reversed(tail) if ln.strip()),
                    "unknown_error",
                )
                refresh_failure_reason = f"{name}_failed: {detail}"
                break

    # Freshness after refresh.
    signals_age = _file_age_seconds(SIGNALS_CSV)
    lifecycle_age = _file_age_seconds(LIFECYCLE_RAW)
    effective_age = _file_age_seconds(LIFECYCLE_EFFECTIVE)

    lc_ok, lc_reason = _check_lifecycle_integrity(RESULTS)
    sig_ok, sig_reason = _check_signal_freshness(RESULTS)

    signals_stale = not sig_ok
    lifecycle_stale = not lc_ok
    effective_stale = effective_age is None or effective_age > (float(GOV_LC_H) * 3600.0)

    readiness = _load_json(EXEC_READINESS)
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    readiness_failed = list(readiness.get("failed_checks") or [])

    cert = _load_json(CERT_PATH)
    cert_status = str(cert.get("certification_status") or "UNKNOWN")
    cert_failed = list(cert.get("failed_requirements") or [])

    arm = build_paper_arm_status(verbose_notes=False)
    paper_arm_valid = str(arm.get("status") or "") == "VALID"

    live_flag = bool(load_live_execution_flag())
    live_readiness = bool(readiness.get("live_execution_permitted"))
    live_blocked = not (live_flag or live_readiness)

    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()

    wd_status = _load_json(WATCHDOG_STATUS)
    broker_connected = bool(wd_status.get("broker_connected")) or bool(
        readiness.get("checks", {}).get("broker")
    )

    open_orders = _count_csv_rows(OPEN_ORDERS)
    positions_count = _count_csv_rows(POSITIONS)
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

    lifecycle_signals_fresh = not signals_stale and not lifecycle_stale and not effective_stale
    readiness_blocked_by_freshness = ("lifecycle" in readiness_failed) or (
        "signals" in readiness_failed
    )

    # Recommended posture.
    if not all_refresh_ok:
        posture = (
            "SAFE_BLOCK — refresh step failed ("
            + str(refresh_failure_reason or "refresh_step_failed")
            + "); readiness safe-blocked; investigate upstream predictions/model "
            "(fetch_raw_data/preprocess/train_model) before retrying. "
            "No broker orders, no authority artifacts; live blocked."
        )
    elif lifecycle_signals_fresh and not readiness_blocked_by_freshness:
        posture = (
            "PAPER_MONITOR — lifecycle/signals fresh; readiness no longer blocked by stale "
            "inputs. Allocation remains sole authority for normal paper entries; live blocked."
        )
    else:
        posture = (
            "REVIEW — refresh succeeded but readiness still reports lifecycle/signals; "
            "inspect residual freshness/gate logic (do not loosen thresholds)."
        )

    # Status determination.
    if not live_blocked or approval_present or broker_auth_present:
        status = "FAIL"
    elif not all_refresh_ok:
        status = "REVIEW_REQUIRED"
    elif not lifecycle_signals_fresh:
        status = "REVIEW_REQUIRED"
    elif open_orders != 0:
        status = "REVIEW_REQUIRED"
    else:
        status = "PASS"

    audit: Dict[str, Any] = {
        "refresh_pipeline_status": refresh_pipeline_status,
        "refresh_failure_reason": refresh_failure_reason,
        "refresh_steps": refresh_steps,
        "generate_signals_status": gen_status,
        "apply_signal_lifecycle_status": apply_status,
        "build_effective_lifecycle_status": build_status,
        "signals_file_age_seconds": signals_age,
        "signals_file_mtime": _file_mtime_iso(SIGNALS_CSV),
        "lifecycle_file_age_seconds": lifecycle_age,
        "lifecycle_file_mtime": _file_mtime_iso(LIFECYCLE_RAW),
        "effective_lifecycle_file_age_seconds": effective_age,
        "effective_lifecycle_file_mtime": _file_mtime_iso(LIFECYCLE_EFFECTIVE),
        "lifecycle_stale_after_refresh": lifecycle_stale,
        "signals_stale_after_refresh": signals_stale,
        "effective_lifecycle_stale_after_refresh": effective_stale,
        "lifecycle_reason": lc_reason,
        "signals_reason": sig_reason,
        "lifecycle_freshness_threshold_seconds": LIFECYCLE_FRESHNESS_SECONDS,
        "signals_freshness_threshold_seconds": SIGNALS_FRESHNESS_SECONDS,
        "execution_readiness_status": readiness_status,
        "execution_readiness_failed_checks": readiness_failed,
        "readiness_blocked_by_freshness": readiness_blocked_by_freshness,
        "certification_status": cert_status,
        "certification_failed_requirements": cert_failed,
        "paper_arm_valid": paper_arm_valid,
        "paper_entry_armed": bool(arm.get("paper_entry_armed")),
        "live_execution_blocked": live_blocked,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "broker_connected": broker_connected,
        "open_orders_count": open_orders,
        "positions_count": positions_count,
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_target_satisfied": gld_target,
        "allocation_open_new_count": alloc.get("open_new_count"),
        "allocation_add_count": alloc.get("add_count"),
        "allocation_exit_count": alloc.get("exit_count"),
        "allocation_trim_count": alloc.get("trim_count"),
        "broker_orders_submitted": False,
        "recommended_runtime_posture": posture,
    }

    executive = (
        f"Refresh pipeline {refresh_pipeline_status}: "
        f"generate_signals={gen_status}, apply_signal_lifecycle={apply_status}, "
        f"build_effective_lifecycle={build_status}. "
        f"After refresh: signals_stale={signals_stale}, lifecycle_stale={lifecycle_stale}, "
        f"effective_stale={effective_stale}. "
        f"execution_readiness={readiness_status} (failed={readiness_failed}); "
        f"certification={cert_status}. "
        f"Broker safe: open_orders={open_orders}, live_blocked={live_blocked}, "
        f"approval_absent={not approval_present}, broker_auth_absent={not broker_auth_present}. "
        f"GLD target satisfied={gld_target}. allocation open_new={alloc.get('open_new_count')} "
        f"add={alloc.get('add_count')}. No broker orders submitted by this phase."
    )

    payload: Dict[str, Any] = {
        "phase": "198W",
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
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(f"[198W] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198W] wrote {OUTPUT_MD.relative_to(ROOT)}")
        print(
            f"[198W] status={status} refresh={refresh_pipeline_status} "
            f"signals_stale={signals_stale} lifecycle_stale={lifecycle_stale} "
            f"readiness={readiness_status} cert={cert_status} "
            f"open_orders={open_orders} live_blocked={live_blocked}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 198W market-open refresh pipeline validation")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not run refresh modules in-script; only inspect current artifacts.",
    )
    args = ap.parse_args(argv)
    result = run_validation(run_refresh=not args.no_refresh, verbose=not args.quiet)
    return 0 if str(result.get("status") or "") in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
