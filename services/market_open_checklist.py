"""Monday pre-execution readiness checklist before paper order submission (Phase 148D)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "data" / "results"
CONFIG = ROOT / "config"
DEFAULT_ALLOCATION_CSV = RESULTS / "portfolio_allocation_recommendations.csv"
DEFAULT_ALLOCATION_JSON = RESULTS / "portfolio_allocation_summary.json"
DEFAULT_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"
DEFAULT_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
GOVERNANCE_AUTH_PATH = RESULTS / "governance_authorization.json"
POLICY_PATH = RESULTS / "protective_action_policy.json"
CHECKLIST_JSON = RESULTS / "market_open_checklist.json"

DEFAULT_ALLOCATION_MAX_AGE_MINUTES = 24 * 60


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


def _file_age_minutes(path: Path) -> Optional[float]:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0
    except Exception:
        return None


def _json_generated_age_minutes(doc: Dict[str, Any]) -> Optional[float]:
    ts = doc.get("generated_at") or doc.get("timestamp")
    if not ts:
        return None
    try:
        s = str(ts).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60.0
    except Exception:
        return None


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str
    critical: bool = True


@dataclass
class ChecklistResult:
    timestamp: str
    market_open: bool
    paper_mode: bool
    paper_execution_env: bool
    live_blocked: bool
    open_orders: int
    ready: bool
    checks: List[CheckItem] = field(default_factory=list)
    block_reasons: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "market_open": self.market_open,
            "paper_mode": self.paper_mode,
            "paper_execution_env": self.paper_execution_env,
            "live_blocked": self.live_blocked,
            "open_orders": self.open_orders,
            "ready": self.ready,
            "block_reasons": list(self.block_reasons),
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "detail": c.detail,
                    "critical": c.critical,
                }
                for c in self.checks
            ],
        }


def _check_market_open(broker: Any) -> CheckItem:
    try:
        clk = broker.get_clock() or {}
        is_open = bool(clk.get("is_open"))
        nxt = clk.get("next_open") or clk.get("next_close") or ""
        return CheckItem(
            "market_open",
            is_open,
            f"is_open={is_open} next_event={nxt}".strip(),
        )
    except Exception as e:
        return CheckItem("market_open", False, f"clock_unavailable: {e}")


def _check_paper_mode(broker: Any, mode: str) -> CheckItem:
    if str(mode).lower() != "paper":
        return CheckItem("paper_mode", False, f"requested_mode={mode}")
    try:
        base = str(getattr(broker, "base", "") or "")
        if "paper-api" not in base.lower():
            return CheckItem("paper_mode", False, f"broker_base={base}")
        acct = broker.get_account() or {}
        acct_id = str(acct.get("id") or acct.get("account_number") or "")
        return CheckItem(
            "paper_mode",
            True,
            (
                f"broker_base=paper-api account={acct_id[:8]}..."
                if acct_id
                else "broker_base=paper-api"
            ),
        )
    except Exception as e:
        return CheckItem("paper_mode", False, f"account_unavailable: {e}")


def _check_paper_execution_env() -> CheckItem:
    ok = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() == "1"
    return CheckItem(
        "paper_execution_env",
        ok,
        "TRITON_ENABLE_PAPER_EXECUTION=1" if ok else "TRITON_ENABLE_PAPER_EXECUTION not set to 1",
    )


def _check_live_blocked() -> CheckItem:
    policy = _read_json(POLICY_PATH)
    gov = _read_json(GOVERNANCE_AUTH_PATH)
    live_policy = policy.get("live_execution_enabled") is True
    live_gov = gov.get("live_execution_permitted") is True
    mode_live = str(policy.get("mode") or "").lower() == "live"
    blocked = not live_policy and not live_gov and not mode_live
    detail = (
        f"live_execution_enabled={live_policy} "
        f"live_execution_permitted={live_gov} policy_mode={policy.get('mode', '')}"
    )
    return CheckItem("live_blocked", blocked, detail)


def _check_open_orders(broker: Any) -> Tuple[CheckItem, int]:
    try:
        from services.place_live_orders import list_open_orders

        orders = list_open_orders(broker) or []
        count = len(orders)
        return CheckItem("open_orders_known", True, f"count={count}"), count
    except Exception as e:
        return CheckItem("open_orders_known", False, f"open_orders_unavailable: {e}"), -1


def _check_duplicate_protection() -> CheckItem:
    try:
        from services.order_discipline import load_order_discipline_config

        cfg = load_order_discipline_config()
        enabled = bool(cfg.get("enabled", True))
        block_open = bool(cfg.get("block_if_open_same_side_exists", True))
        session_lock = bool(cfg.get("same_session_symbol_lock", True))
        ok = enabled and (block_open or session_lock)
        return CheckItem(
            "duplicate_protection",
            ok,
            f"enabled={enabled} block_if_open_same_side={block_open} same_session_lock={session_lock}",
        )
    except Exception as e:
        return CheckItem("duplicate_protection", False, f"config_unavailable: {e}")


def _check_lifecycle_fresh() -> CheckItem:
    try:
        from services.lifecycle_truth import evaluate_lifecycle_gate

        gate = evaluate_lifecycle_gate(path=DEFAULT_LIFECYCLE)
        ok = gate.status == "OK"
        return CheckItem(
            "lifecycle_fresh",
            ok,
            f"status={gate.status} reason={gate.reason} details={gate.details}",
        )
    except Exception as e:
        return CheckItem("lifecycle_fresh", False, f"lifecycle_gate_error: {e}")


def _check_allocation_fresh(max_age_minutes: float) -> CheckItem:
    paths = [DEFAULT_ALLOCATION_JSON, DEFAULT_ALLOCATION_CSV]
    existing = [p for p in paths if p.is_file() and p.stat().st_size > 0]
    if not existing:
        return CheckItem(
            "allocation_fresh",
            False,
            "portfolio_allocation artifacts missing",
        )

    ages: List[float] = []
    for p in existing:
        age = _file_age_minutes(p)
        if age is not None:
            ages.append(age)
    if DEFAULT_ALLOCATION_JSON in existing:
        doc_age = _json_generated_age_minutes(_read_json(DEFAULT_ALLOCATION_JSON))
        if doc_age is not None:
            ages.append(doc_age)

    if not ages:
        return CheckItem("allocation_fresh", False, "could not determine allocation age")

    best_age = min(ages)
    ok = best_age <= max_age_minutes
    return CheckItem(
        "allocation_fresh",
        ok,
        f"age_minutes={best_age:.1f} max={max_age_minutes:.0f} files={[p.name for p in existing]}",
    )


def _check_execution_authorized() -> CheckItem:
    gov = _read_json(GOVERNANCE_AUTH_PATH)
    ok = gov.get("execution_authorized") is True
    return CheckItem(
        "execution_authorized",
        ok,
        f"execution_authorized={gov.get('execution_authorized')} "
        f"paper_execution_permitted={gov.get('paper_execution_permitted')}",
    )


def _check_governance_authorized() -> CheckItem:
    gov = _read_json(GOVERNANCE_AUTH_PATH)
    ok = gov.get("governance_authorized") is True
    return CheckItem(
        "governance_authorized",
        ok,
        f"governance_authorized={gov.get('governance_authorized')}",
    )


def _check_top_opportunities() -> CheckItem:
    if not DEFAULT_OPPORTUNITIES.is_file() or DEFAULT_OPPORTUNITIES.stat().st_size == 0:
        return CheckItem("top_opportunities", False, "trade_opportunities.csv missing or empty")
    try:
        df = pd.read_csv(DEFAULT_OPPORTUNITIES)
        if df is None or df.empty:
            return CheckItem("top_opportunities", False, "trade_opportunities.csv has zero rows")
        if "ticker" not in [str(c).strip() for c in df.columns]:
            return CheckItem(
                "top_opportunities", False, "trade_opportunities.csv missing ticker column"
            )
        n = len(df)
        top = str(df.iloc[0].get("ticker") or df.iloc[0].get("symbol") or "")
        return CheckItem(
            "top_opportunities",
            n > 0,
            f"rows={n} first={top}",
        )
    except Exception as e:
        return CheckItem("top_opportunities", False, f"read_error: {e}")


def _check_positions_readable(broker: Any) -> CheckItem:
    try:
        positions = broker.get_positions() or []
        n = len(positions)
        return CheckItem("positions_readable", True, f"positions={n}")
    except Exception as e:
        return CheckItem("positions_readable", False, f"positions_unavailable: {e}")


def run_market_open_checklist(
    *,
    mode: str = "paper",
    allocation_max_age_minutes: float = DEFAULT_ALLOCATION_MAX_AGE_MINUTES,
) -> ChecklistResult:
    ts = _utc_iso()
    checks: List[CheckItem] = []
    open_orders = -1
    broker: Any = None

    if str(mode).lower() != "paper":
        checks.append(CheckItem("paper_mode", False, f"only paper mode supported (got {mode})"))
        return ChecklistResult(
            timestamp=ts,
            market_open=False,
            paper_mode=False,
            paper_execution_env=False,
            live_blocked=True,
            open_orders=-1,
            ready=False,
            checks=checks,
            block_reasons=["non_paper_mode"],
        )

    try:
        from services.place_live_orders import make_broker

        broker = make_broker(mode="paper")
    except Exception as e:
        checks.append(CheckItem("broker_connectivity", False, str(e)))
        return ChecklistResult(
            timestamp=ts,
            market_open=False,
            paper_mode=False,
            paper_execution_env=os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() == "1",
            live_blocked=True,
            open_orders=-1,
            ready=False,
            checks=checks,
            block_reasons=["broker_connectivity"],
        )

    market_chk = _check_market_open(broker)
    paper_chk = _check_paper_mode(broker, mode)
    env_chk = _check_paper_execution_env()
    live_chk = _check_live_blocked()
    open_chk, open_orders = _check_open_orders(broker)
    dup_chk = _check_duplicate_protection()
    lifecycle_chk = _check_lifecycle_fresh()
    alloc_chk = _check_allocation_fresh(allocation_max_age_minutes)
    exec_chk = _check_execution_authorized()
    gov_chk = _check_governance_authorized()
    opp_chk = _check_top_opportunities()
    pos_chk = _check_positions_readable(broker)

    checks.extend(
        [
            market_chk,
            paper_chk,
            env_chk,
            live_chk,
            open_chk,
            dup_chk,
            lifecycle_chk,
            alloc_chk,
            exec_chk,
            gov_chk,
            opp_chk,
            pos_chk,
        ]
    )

    block_reasons = [c.name for c in checks if c.critical and not c.ok]
    ready = len(block_reasons) == 0

    return ChecklistResult(
        timestamp=ts,
        market_open=market_chk.ok,
        paper_mode=paper_chk.ok,
        paper_execution_env=env_chk.ok,
        live_blocked=live_chk.ok,
        open_orders=max(open_orders, 0) if open_orders >= 0 else 0,
        ready=ready,
        checks=checks,
        block_reasons=block_reasons,
    )


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def print_checklist(result: ChecklistResult, *, verbose: bool = False) -> None:
    print("[MARKET_OPEN_CHECKLIST]")
    print(f"market_open={_bool_str(result.market_open)}")
    print(f"paper_mode={_bool_str(result.paper_mode)}")
    print(f"paper_execution_env={_bool_str(result.paper_execution_env)}")
    print(f"live_blocked={_bool_str(result.live_blocked)}")
    print(f"open_orders={result.open_orders}")
    print(f"ready={_bool_str(result.ready)}")
    if verbose:
        print("[MARKET_OPEN_CHECKLIST_DETAIL]")
        for c in result.checks:
            mark = "OK" if c.ok else "FAIL"
            print(f"  {mark} {c.name}: {c.detail}")
    if not result.ready and result.block_reasons:
        print(f"[MARKET_OPEN_BLOCK] reason={';'.join(result.block_reasons)}")


def write_checklist_json(result: ChecklistResult) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKLIST_JSON.write_text(
        json.dumps(result.to_summary_dict(), indent=2),
        encoding="utf-8",
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON Monday pre-execution paper readiness checklist"
    )
    ap.add_argument("--mode", choices=["paper"], default="paper")
    ap.add_argument(
        "--allocation-max-age-minutes",
        type=float,
        default=DEFAULT_ALLOCATION_MAX_AGE_MINUTES,
        help="Max age for portfolio allocation artifacts (default 24h)",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--no-json", action="store_true", help="Skip writing market_open_checklist.json"
    )
    args = ap.parse_args(argv)

    result = run_market_open_checklist(
        mode=args.mode,
        allocation_max_age_minutes=float(args.allocation_max_age_minutes),
    )
    print_checklist(result, verbose=bool(args.verbose))
    if not args.no_json:
        write_checklist_json(result)
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
