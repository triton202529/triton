"""
ARM execution-control layer: MANUAL / ASSISTED / AUTO (paper-first; live auto stays off).

Single source of truth: config/arm_mode.json
Confirmation (ASSISTED): data/live/paper_arm_confirm.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "arm_mode.json"
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
STATUS_JSON = RESULTS / "arm_mode_status.json"
LOG_CSV = RESULTS / "arm_mode_log.csv"
CONFIRM_PATH = LIVE / "paper_arm_confirm.json"

VALID_MODES = ("MANUAL", "ASSISTED", "AUTO")

_DEFAULT_ARM: Dict[str, Any] = {
    "enabled": True,
    "mode": "MANUAL",
    "paper_auto_allowed": True,
    "live_auto_allowed": False,
    "require_master_gate": True,
    "require_fresh_snapshots": True,
    "require_no_stale_open_orders_for_auto": False,
    "require_regime_check": True,
    "arm_max_snapshot_age_minutes": 60,
    "allow_reprice_ladder_in_auto": True,
    "allow_manage_positions_in_auto": True,
    "allow_manage_open_orders_in_auto": True,
    "allow_execute_trades_in_auto": True,
    "allow_reallocation_bridge_in_auto": False,
    "allow_reallocation_bridge_in_assisted": True,
    "confirmation_ttl_minutes": 30,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        dt = x
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(x, str):
        s = x.strip().replace("Z", "+00:00")
        if "T" not in s and " " in s:
            s = s.replace(" ", "T", 1)
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _file_age_minutes(path: Path) -> Optional[float]:
    try:
        if not path.is_file():
            return None
        m = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (_utc_now() - m).total_seconds() / 60.0
    except Exception:
        return None


def load_arm_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULT_ARM)
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


def validate_mode(mode: Any) -> str:
    m = str(mode or "MANUAL").strip().upper()
    return m if m in VALID_MODES else "MANUAL"


def get_arm_mode() -> str:
    return validate_mode(load_arm_config().get("mode"))


def is_manual_mode() -> bool:
    return get_arm_mode() == "MANUAL"


def is_assisted_mode() -> bool:
    return get_arm_mode() == "ASSISTED"


def is_auto_mode() -> bool:
    return get_arm_mode() == "AUTO"


def read_paper_arm_confirm() -> Optional[Dict[str, Any]]:
    try:
        if not CONFIRM_PATH.is_file():
            return None
        d = json.loads(CONFIRM_PATH.read_text(encoding="utf-8", errors="replace"))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def is_paper_arm_confirm_valid(arm_cfg: Optional[Dict[str, Any]] = None) -> bool:
    arm_cfg = arm_cfg or load_arm_config()
    doc = read_paper_arm_confirm()
    if not doc:
        return False
    if not bool(doc.get("allow_execute", True)):
        return False
    exp = _parse_dt(doc.get("expires_at"))
    if exp is not None and _utc_now() >= exp:
        return False
    ttl = int(arm_cfg.get("confirmation_ttl_minutes", 30) or 30)
    ts = _parse_dt(doc.get("timestamp"))
    if ts is not None and exp is None:
        if (_utc_now() - ts).total_seconds() / 60.0 > ttl:
            return False
    return True


def evaluate_arm_mutation_gates(arm_cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Safety gates for ASSISTED/AUTO paper mutations. Best-effort; never raises."""
    reasons: List[str] = []
    try:
        if arm_cfg.get("require_master_gate", True):
            from services.master_execution_gate import MasterExecutionGate

            gate = MasterExecutionGate(project_root=ROOT)
            dec = gate.evaluate(mode="paper", broker=None, verbose=False)
            if not dec.ok:
                reasons.append("MASTER_GATE")
                reasons.extend(dec.reasons)
    except Exception as e:
        reasons.append(f"MASTER_GATE_ERROR:{e!s}")

    try:
        if arm_cfg.get("require_fresh_snapshots", True):
            max_age = float(arm_cfg.get("arm_max_snapshot_age_minutes", 60) or 60)
            pos = RESULTS / "positions_snapshot.csv"
            age = _file_age_minutes(pos)
            if age is None:
                reasons.append("SNAPSHOT_MISSING")
            elif age > max_age:
                reasons.append(f"SNAPSHOT_STALE:{age:.0f}m")
    except Exception as e:
        reasons.append(f"SNAPSHOT_CHECK_ERROR:{e!s}")

    try:
        if arm_cfg.get("require_regime_check", True):
            from services.capital_reallocation import load_reallocation_config
            from services.regime_portfolio_control import detect_market_regime

            r = detect_market_regime(load_reallocation_config())
            if str(r.get("source") or "") == "no_data":
                reasons.append("REGIME_NO_DATA")
    except Exception as e:
        reasons.append(f"REGIME_CHECK_ERROR:{e!s}")

    try:
        if arm_cfg.get("require_no_stale_open_orders_for_auto"):
            lo = RESULTS / "live_orders.csv"
            age = _file_age_minutes(lo)
            if age is not None and age > 120:
                reasons.append(f"STALE_LIVE_ORDERS:{age:.0f}m")
    except Exception:
        pass

    return (len(reasons) == 0), list(dict.fromkeys(reasons))


def get_arm_permissions(
    mode: Optional[str] = None, arm_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """Logical permissions for mode (before gates)."""
    arm_cfg = arm_cfg or load_arm_config()
    if not arm_cfg.get("enabled", True):
        mode = "MANUAL"
    else:
        mode = validate_mode(mode or arm_cfg.get("mode"))

    z = {
        "execute_trades": False,
        "manage_positions_execute": False,
        "manage_open_orders_execute_cancel": False,
        "reprice_order_ladder_execute": False,
        "reallocate_after_exit": False,
    }
    if mode == "MANUAL":
        return z

    allow_ex = bool(arm_cfg.get("allow_execute_trades_in_auto", True))
    allow_mg = bool(arm_cfg.get("allow_manage_positions_in_auto", True))
    allow_moo = bool(arm_cfg.get("allow_manage_open_orders_in_auto", True))
    allow_rpl = bool(arm_cfg.get("allow_reprice_ladder_in_auto", True))
    allow_ra = bool(arm_cfg.get("allow_reallocation_bridge_in_auto", False))
    allow_ra_assisted = bool(arm_cfg.get("allow_reallocation_bridge_in_assisted", True))

    if mode == "ASSISTED":
        if not is_paper_arm_confirm_valid(arm_cfg):
            return z
        return {
            "execute_trades": allow_ex,
            "manage_positions_execute": allow_mg,
            "manage_open_orders_execute_cancel": allow_moo,
            "reprice_order_ladder_execute": allow_rpl,
            "reallocate_after_exit": allow_ra_assisted,
        }

    if mode == "AUTO":
        if not bool(arm_cfg.get("paper_auto_allowed", True)):
            return z
        if bool(arm_cfg.get("live_auto_allowed", False)):
            # Policy: never allow live automation via this patch
            pass
        return {
            "execute_trades": allow_ex,
            "manage_positions_execute": allow_mg,
            "manage_open_orders_execute_cancel": allow_moo,
            "reprice_order_ladder_execute": allow_rpl,
            "reallocate_after_exit": allow_ra,
        }

    return z


def resolve_arm_mutation_permissions(
    arm_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, bool], List[str]]:
    """
    Final permissions for scheduled paper cycle mutations (AND with gates for ASSISTED/AUTO).
    """
    arm_cfg = arm_cfg or load_arm_config()
    mode = validate_mode(arm_cfg.get("mode")) if arm_cfg.get("enabled", True) else "MANUAL"
    block: List[str] = []

    if not arm_cfg.get("enabled", True):
        block.append("ARM_DISABLED")
        return {
            "execute_trades": False,
            "manage_positions_execute": False,
            "manage_open_orders_execute_cancel": False,
            "reprice_order_ladder_execute": False,
            "reallocate_after_exit": False,
        }, block

    if mode == "MANUAL":
        block.append("MANUAL_MODE")
        return get_arm_permissions("MANUAL", arm_cfg), block

    base = get_arm_permissions(mode, arm_cfg)
    if mode == "ASSISTED":
        if not is_paper_arm_confirm_valid(arm_cfg):
            block.append("ASSISTED_CONFIRMATION_MISSING_OR_EXPIRED")
            return {
                "execute_trades": False,
                "manage_positions_execute": False,
                "manage_open_orders_execute_cancel": False,
                "reprice_order_ladder_execute": False,
                "reallocate_after_exit": False,
            }, block

    if mode == "AUTO":
        if not bool(arm_cfg.get("paper_auto_allowed", True)):
            block.append("PAPER_AUTO_NOT_ALLOWED")
            return {
                "execute_trades": False,
                "manage_positions_execute": False,
                "manage_open_orders_execute_cancel": False,
                "reprice_order_ladder_execute": False,
                "reallocate_after_exit": False,
            }, block

    ok, gr = evaluate_arm_mutation_gates(arm_cfg)
    if not ok:
        block.extend(gr)
        return {
            "execute_trades": False,
            "manage_positions_execute": False,
            "manage_open_orders_execute_cancel": False,
            "reprice_order_ladder_execute": False,
            "reallocate_after_exit": False,
        }, block

    return base, block


def write_arm_mode_status_snapshot(
    permissions: Dict[str, bool],
    arm_cfg: Dict[str, Any],
    block_reasons: List[str],
    notes: Optional[List[str]] = None,
) -> None:
    doc = {
        "timestamp": _utc_iso(),
        "mode": validate_mode(arm_cfg.get("mode")),
        "paper_auto_allowed": bool(arm_cfg.get("paper_auto_allowed", True)),
        "live_auto_allowed": bool(arm_cfg.get("live_auto_allowed", False)),
        "permissions": {
            "execute_trades": bool(permissions.get("execute_trades")),
            "manage_positions_execute": bool(permissions.get("manage_positions_execute")),
            "manage_open_orders_execute_cancel": bool(
                permissions.get("manage_open_orders_execute_cancel")
            ),
            "reprice_order_ladder_execute": bool(permissions.get("reprice_order_ladder_execute")),
            "reallocate_after_exit": bool(permissions.get("reallocate_after_exit")),
        },
        "block_reasons": list(block_reasons),
        "notes": list(notes or []),
    }
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        STATUS_JSON.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except Exception:
        pass


def append_arm_mode_log(row: Dict[str, Any]) -> None:
    fields = [
        "ts_utc",
        "mode",
        "blocked",
        "block_reasons",
        "perm_execute",
        "perm_manage",
        "perm_moo",
        "perm_rpl",
    ]
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        new_file = not LOG_CSV.is_file() or LOG_CSV.stat().st_size == 0
        out = {k: row.get(k, "") for k in fields}
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(out)
    except Exception:
        pass


def print_status() -> None:
    cfg = load_arm_config()
    mode = validate_mode(cfg.get("mode"))
    perms, blocks = resolve_arm_mutation_permissions(cfg)
    print("ARM_MODE:", mode)
    print("enabled:", cfg.get("enabled"))
    print("paper_auto_allowed:", cfg.get("paper_auto_allowed"))
    print("live_auto_allowed:", cfg.get("live_auto_allowed"))
    print("effective_permissions:", json.dumps(perms, indent=2))
    print("block_reasons:", blocks)
    print("paper_arm_confirm_valid:", is_paper_arm_confirm_valid(cfg))
    print("config:", CONFIG_PATH)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TRITON ARM mode status")
    ap.add_argument("--status", action="store_true", help="Print current ARM status (default)")
    args = ap.parse_args(argv)
    print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
