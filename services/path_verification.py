"""Full paper-trading operational path verification before Monday (Phase 148F)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
DEFAULT_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"
DEFAULT_ALLOCATION_CSV = RESULTS / "portfolio_allocation_recommendations.csv"
DEFAULT_ALLOCATION_JSON = RESULTS / "portfolio_allocation_summary.json"
DEFAULT_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
GOVERNANCE_AUTH_PATH = RESULTS / "governance_authorization.json"
PERFORMANCE_RISK_OVERLAY = RESULTS / "performance_risk_overlay.csv"
OUTPUT_JSON = RESULTS / "path_verification.json"

EXIT_STANCES = frozenset({"EXIT", "TRIM", "ROTATE_EXIT"})


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


@dataclass
class PathCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class PathVerificationResult:
    timestamp: str
    lifecycle_loaded: bool
    allocation_loaded: bool
    governance_authorized: bool
    execution_authorized: bool
    opportunities: int
    duplicate_protection: bool
    audit_trail: bool
    broker_reachable: bool
    management_path: bool
    exit_path: bool
    ready_for_monday: bool
    block_reasons: List[str] = field(default_factory=list)
    checks: List[PathCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "lifecycle_loaded": self.lifecycle_loaded,
            "allocation_loaded": self.allocation_loaded,
            "governance_authorized": self.governance_authorized,
            "execution_authorized": self.execution_authorized,
            "opportunities": self.opportunities,
            "duplicate_protection": self.duplicate_protection,
            "audit_trail": self.audit_trail,
            "broker_reachable": self.broker_reachable,
            "management_path": self.management_path,
            "exit_path": self.exit_path,
            "ready_for_monday": self.ready_for_monday,
            "block_reasons": list(self.block_reasons),
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


def _check_lifecycle_loaded() -> PathCheck:
    try:
        from services.lifecycle_truth import evaluate_lifecycle_gate

        gate = evaluate_lifecycle_gate(path=DEFAULT_LIFECYCLE)
        ok = gate.status == "OK" and DEFAULT_LIFECYCLE.is_file()
        return PathCheck(
            "lifecycle_loaded",
            ok,
            f"status={gate.status} reason={gate.reason} rows={len(gate.tickers or [])}",
        )
    except Exception as e:
        return PathCheck("lifecycle_loaded", False, str(e))


def _check_allocation_loaded() -> PathCheck:
    loaded = False
    detail_parts: List[str] = []
    for path in (DEFAULT_ALLOCATION_CSV, DEFAULT_ALLOCATION_JSON):
        if not path.is_file() or path.stat().st_size == 0:
            detail_parts.append(f"{path.name}=missing")
            continue
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                n = len(df) if df is not None else 0
                if n > 0:
                    loaded = True
                detail_parts.append(f"{path.name}=rows:{n}")
            else:
                doc = _read_json(path)
                if doc:
                    loaded = True
                detail_parts.append(f"{path.name}=ok")
        except Exception as e:
            detail_parts.append(f"{path.name}=error:{e}")
    return PathCheck(
        "allocation_loaded",
        loaded,
        "; ".join(detail_parts) or "no allocation artifacts",
    )


def _check_governance_authorized() -> PathCheck:
    gov = _read_json(GOVERNANCE_AUTH_PATH)
    ok = gov.get("governance_authorized") is True
    return PathCheck(
        "governance_authorized",
        ok,
        f"governance_authorized={gov.get('governance_authorized')} "
        f"overall={gov.get('overall_authorization')}",
    )


def _check_execution_authorized() -> PathCheck:
    gov = _read_json(GOVERNANCE_AUTH_PATH)
    ok = gov.get("execution_authorized") is True
    env_ok = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() == "1"
    return PathCheck(
        "execution_authorized",
        ok,
        f"execution_authorized={gov.get('execution_authorized')} "
        f"paper_execution_permitted={gov.get('paper_execution_permitted')} "
        f"TRITON_ENABLE_PAPER_EXECUTION={'1' if env_ok else '0'}",
    )


def _check_opportunities() -> tuple[PathCheck, int]:
    if not DEFAULT_OPPORTUNITIES.is_file() or DEFAULT_OPPORTUNITIES.stat().st_size == 0:
        return PathCheck("opportunities", False, "trade_opportunities.csv missing or empty"), 0
    try:
        df = pd.read_csv(DEFAULT_OPPORTUNITIES)
        n = len(df) if df is not None and not df.empty else 0
        ok = n > 0 and "ticker" in [str(c).strip() for c in df.columns]
        return PathCheck("opportunities", ok, f"rows={n}"), int(n)
    except Exception as e:
        return PathCheck("opportunities", False, str(e)), 0


def _check_duplicate_protection() -> PathCheck:
    try:
        from services.order_discipline import load_order_discipline_config

        cfg = load_order_discipline_config()
        enabled = bool(cfg.get("enabled", True))
        block_open = bool(cfg.get("block_if_open_same_side_exists", True))
        session_lock = bool(cfg.get("same_session_symbol_lock", True))
        ok = enabled and (block_open or session_lock)
        return PathCheck(
            "duplicate_protection",
            ok,
            f"enabled={enabled} block_open={block_open} session_lock={session_lock}",
        )
    except Exception as e:
        return PathCheck("duplicate_protection", False, str(e))


def _check_audit_trail() -> PathCheck:
    try:
        from services.paper_execution_audit import AUDIT_COLUMNS, AUDIT_CSV

        RESULTS.mkdir(parents=True, exist_ok=True)
        probe = RESULTS / ".path_verification_audit_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)

        write_header = not AUDIT_CSV.is_file() or AUDIT_CSV.stat().st_size == 0
        with AUDIT_CSV.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
        return PathCheck(
            "audit_trail",
            True,
            f"writable path={AUDIT_CSV.name} columns={len(AUDIT_COLUMNS)}",
        )
    except Exception as e:
        return PathCheck("audit_trail", False, str(e))


def _check_broker_reachable() -> PathCheck:
    try:
        from services.place_live_orders import make_broker

        broker = make_broker(mode="paper")
        base = str(getattr(broker, "base", "") or "")
        if "paper-api" not in base.lower():
            return PathCheck("broker_reachable", False, f"not_paper_api base={base}")
        acct = broker.get_account() or {}
        positions = broker.get_positions() or []
        acct_id = str(acct.get("id") or acct.get("account_number") or "")[:12]
        return PathCheck(
            "broker_reachable",
            bool(acct),
            f"account={acct_id} positions={len(positions)}",
        )
    except Exception as e:
        return PathCheck("broker_reachable", False, str(e))


def _check_management_path() -> PathCheck:
    try:
        import services.manage_positions as mp

        cfg = mp.load_manage_config()
        lc_path = mp.resolve_lifecycle_path(None)
        lc = mp.load_lifecycle_df(lc_path)
        if lc.empty:
            return PathCheck(
                "management_path",
                False,
                f"lifecycle_empty path={lc_path.name}",
            )
        has_plan_fn = callable(getattr(mp, "build_management_plan", None))
        ok = bool(cfg) and has_plan_fn and not lc.empty
        return PathCheck(
            "management_path",
            ok,
            f"config_ok lifecycle={lc_path.name} rows={len(lc)} build_management_plan={has_plan_fn}",
        )
    except Exception as e:
        return PathCheck("management_path", False, str(e))


def _check_exit_path() -> PathCheck:
    try:
        import services.manage_positions as mp

        cfg = mp.load_manage_config()
        lc = mp.load_lifecycle_df(mp.resolve_lifecycle_path(None))
        exit_rows = 0
        if not lc.empty:
            for _, row in lc.iterrows():
                stance = mp.resolve_management_stance(row)
                if stance in EXIT_STANCES:
                    exit_rows += 1

        overlay_rows = 0
        if PERFORMANCE_RISK_OVERLAY.is_file():
            try:
                odf = pd.read_csv(PERFORMANCE_RISK_OVERLAY)
                overlay_rows = len(odf) if odf is not None else 0
            except Exception:
                overlay_rows = 0

        forced_rotation = bool(cfg.get("forced_rotation_no_signal_enabled", True))
        overlay_fn = callable(getattr(mp, "_load_performance_risk_overlay_map", None))
        ok = (
            overlay_fn
            and callable(getattr(mp, "build_management_plan", None))
            and (exit_rows > 0 or overlay_rows > 0 or forced_rotation)
        )
        return PathCheck(
            "exit_path",
            ok,
            f"exit_stances={exit_rows} overlay_rows={overlay_rows} "
            f"forced_rotation={forced_rotation} overlay_fn={overlay_fn}",
        )
    except Exception as e:
        return PathCheck("exit_path", False, str(e))


def run_path_verification() -> PathVerificationResult:
    ts = _utc_iso()
    checks: List[PathCheck] = [
        _check_lifecycle_loaded(),
        _check_allocation_loaded(),
        _check_governance_authorized(),
        _check_execution_authorized(),
        _check_duplicate_protection(),
        _check_audit_trail(),
        _check_broker_reachable(),
        _check_management_path(),
        _check_exit_path(),
    ]
    opp_check, opp_count = _check_opportunities()
    checks.insert(4, opp_check)

    block_reasons = [c.name for c in checks if not c.ok]
    ready = len(block_reasons) == 0

    by_name = {c.name: c for c in checks}
    return PathVerificationResult(
        timestamp=ts,
        lifecycle_loaded=by_name["lifecycle_loaded"].ok,
        allocation_loaded=by_name["allocation_loaded"].ok,
        governance_authorized=by_name["governance_authorized"].ok,
        execution_authorized=by_name["execution_authorized"].ok,
        opportunities=opp_count,
        duplicate_protection=by_name["duplicate_protection"].ok,
        audit_trail=by_name["audit_trail"].ok,
        broker_reachable=by_name["broker_reachable"].ok,
        management_path=by_name["management_path"].ok,
        exit_path=by_name["exit_path"].ok,
        ready_for_monday=ready,
        block_reasons=block_reasons,
        checks=checks,
    )


def print_summary(result: PathVerificationResult, *, verbose: bool = False) -> None:
    print("[PATH_TEST_SUMMARY]")
    print(f"lifecycle_loaded={_bool_str(result.lifecycle_loaded)}")
    print(f"allocation_loaded={_bool_str(result.allocation_loaded)}")
    print(f"governance_authorized={_bool_str(result.governance_authorized)}")
    print(f"execution_authorized={_bool_str(result.execution_authorized)}")
    print(f"opportunities={result.opportunities}")
    print(f"duplicate_protection={_bool_str(result.duplicate_protection)}")
    print(f"audit_trail={_bool_str(result.audit_trail)}")
    print(f"broker_reachable={_bool_str(result.broker_reachable)}")
    print(f"management_path={_bool_str(result.management_path)}")
    print(f"exit_path={_bool_str(result.exit_path)}")
    print(f"ready_for_monday={_bool_str(result.ready_for_monday)}")
    if verbose:
        print("[PATH_TEST_DETAIL]")
        for c in result.checks:
            mark = "OK" if c.ok else "FAIL"
            print(f"  {mark} {c.name}: {c.detail}")
    if not result.ready_for_monday and result.block_reasons:
        print(f"[PATH_TEST_BLOCK] reason={';'.join(result.block_reasons)}")


def write_result_json(result: PathVerificationResult) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON full paper-trading operational path verification"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    result = run_path_verification()
    print_summary(result, verbose=bool(args.verbose))
    write_result_json(result)
    return 0 if result.ready_for_monday else 2


if __name__ == "__main__":
    raise SystemExit(main())
