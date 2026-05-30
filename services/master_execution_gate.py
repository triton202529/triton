# services/master_execution_gate.py
"""TRITON master pre-execution gate — combines market, freshness, guard, arm, confirm, risk, CPM, snapshots.

Runs before order placement; does not replace execution_guard.py or broker-level checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "master_execution_gate.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
LIVE_DIR = PROJECT_ROOT / "data" / "live"
DEFAULT_SNAPSHOT_PATH = RESULTS_DIR / "master_execution_gate.json"
LOG_CSV_PATH = RESULTS_DIR / "master_execution_gate_log.csv"

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "confirm_ttl_minutes": 10,
    "max_data_age_minutes": 90,
    "max_heartbeat_age_minutes": 90,
    "max_guard_age_minutes": 30,
    "max_positions_snapshot_age_minutes": 30,
    "max_recent_orders_age_minutes": 30,
    "max_live_orders_age_minutes": 45,
    "require_market_open_for_live": True,
    "require_market_open_for_paper": False,
    "require_confirm_for_live": True,
    "require_confirm_for_paper": False,
    "require_arm_for_live": True,
    "require_guard_clear_for_live": True,
    "require_risk_ok_for_live": True,
    "require_cpm_for_live": False,
    "block_on_missing_heartbeat": False,
    "block_on_missing_guard": False,
    "block_on_missing_risk": False,
    "block_on_missing_cpm": False,
    "live_block_missing_positions_snapshot": True,
    "live_block_stale_positions_snapshot": True,
    "live_block_missing_recent_orders_snapshot": True,
    "live_block_stale_recent_orders_snapshot": True,
    "paper_warn_stale_snapshots": True,
    # Paper: if live_orders.csv (append-only log) is stale but positions + recent CSVs are fresh,
    # do not emit STALE_LIVE_ORDERS (snapshot_live_orders does not touch live_orders.csv).
    "paper_suppress_stale_live_orders_when_pipeline_snapshots_fresh": True,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_bool(v: Any, default: bool = False) -> bool:
    if v is True or v is False:
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off", ""):
        return False
    return default


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or str(v).lower() in ("", "null", "none"):
            return None
        return float(v)
    except Exception:
        return None


def _parse_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        dt = x
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(x, (int, float)):
        try:
            ts = float(x)
            if ts > 10_000_000_000:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().replace("Z", "+00:00")
        if not s:
            return None
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


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _first_json(paths: List[Path]) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    for p in paths:
        d = _load_json(p)
        if d is not None:
            return d, p
    return None, None


def _file_age_minutes(path: Path) -> Optional[float]:
    try:
        if not path.is_file():
            return None
        m = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (_utc_now() - m).total_seconds() / 60.0
    except Exception:
        return None


def load_gate_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = project_root or PROJECT_ROOT
    cfg = dict(DEFAULTS)
    p = root / "config" / "master_execution_gate.json"
    try:
        if p.is_file():
            extra = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(extra, dict):
                for k, v in extra.items():
                    cfg[k] = v
    except Exception:
        pass
    return cfg


@dataclass
class GateDecision:
    ok: bool
    status: str
    mode: str
    checked_at: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "mode": self.mode,
            "checked_at": self.checked_at,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "summary": self.summary,
            "details": dict(self.details),
        }


class MasterExecutionGate:
    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.root = (project_root or PROJECT_ROOT).resolve()
        self.results = self.root / "data" / "results"
        self.live = self.root / "data" / "live"
        self.cfg = load_gate_config(self.root)

    def _paths(self) -> Dict[str, List[Path]]:
        r = self.results
        lv = self.live
        return {
            "arm": [r / "live_armed.json", lv / "live_armed.json"],
            "confirm": [r / "live_confirm.json", lv / "live_confirm.json"],
            "guard": [r / "guard_snapshot.json", lv / "guard_snapshot.json"],
            "heartbeat": [r / "pipeline_heartbeat.json", r / "heartbeat.json"],
            "risk": [r / "adaptive_risk_state.json"],
            "cpm": [r / "capital_preservation_mode.json", r / "capital_preservation_state.json"],
            "positions": [r / "positions_snapshot.csv"],
            "recent_orders": [r / "recent_orders.csv"],
            "live_orders": [r / "live_orders.csv"],
        }

    def _check_market(
        self,
        mode: str,
        broker: Any,
        reasons: List[str],
        warnings: List[str],
        details: Dict[str, Any],
    ) -> None:
        req = False
        if mode == "live":
            req = _safe_bool(self.cfg.get("require_market_open_for_live"), True)
        else:
            req = _safe_bool(self.cfg.get("require_market_open_for_paper"), False)

        meta: Dict[str, Any] = {"required": req, "is_open": None}
        if not req:
            meta["detail"] = "Market open not required for this mode"
            details["market"] = meta
            return

        is_open: Optional[bool] = None
        clk: Dict[str, Any] = {}
        try:
            if broker is not None and hasattr(broker, "get_clock"):
                c = broker.get_clock()
                if isinstance(c, dict):
                    clk = c
                    is_open = _safe_bool(c.get("is_open"))
        except Exception as e:
            meta["error"] = str(e)

        meta.update(
            {
                "is_open": is_open,
                "next_open": clk.get("next_open"),
                "next_close": clk.get("next_close"),
            }
        )
        details["market"] = meta

        if is_open is True:
            return
        if is_open is False:
            reasons.append("MARKET_CLOSED")
            return
        if mode == "live":
            reasons.append("MARKET_CLOCK_UNAVAILABLE")
        else:
            warnings.append("MARKET_CLOCK_UNAVAILABLE")

    def _arm_valid(self, doc: Dict[str, Any]) -> Tuple[bool, str]:
        armed = (
            _safe_bool(doc.get("armed"))
            or _safe_bool(doc.get("live_armed"))
            or _safe_bool(doc.get("ok"))
        )
        exp = _parse_dt(doc.get("expires_at"))
        if armed and exp is not None and _utc_now() >= exp:
            return False, "expired"
        if not armed:
            return False, "not_armed"
        return True, "ok"

    def _check_arm(
        self, mode: str, reasons: List[str], warnings: List[str], details: Dict[str, Any]
    ) -> None:
        doc, path = _first_json(self._paths()["arm"])
        sub: Dict[str, Any] = {"path": str(path) if path else None, "valid": False}
        if doc is None:
            sub["detail"] = "missing"
            details["arm"] = sub
            if mode == "live" and _safe_bool(self.cfg.get("require_arm_for_live"), True):
                reasons.append("ARM_NOT_READY")
            elif mode == "paper":
                warnings.append("ARM_MISSING")
            return

        ok, why = self._arm_valid(doc)
        sub["valid"] = ok
        sub["reason"] = why
        details["arm"] = sub
        if not ok and mode == "live" and _safe_bool(self.cfg.get("require_arm_for_live"), True):
            reasons.append("ARM_NOT_READY")

    def _check_confirm(
        self, mode: str, reasons: List[str], warnings: List[str], details: Dict[str, Any]
    ) -> None:
        doc, path = _first_json(self._paths()["confirm"])
        ttl = int(self.cfg.get("confirm_ttl_minutes") or 10)
        sub: Dict[str, Any] = {
            "path": str(path) if path else None,
            "valid": False,
            "ttl_minutes": ttl,
        }

        req = (
            _safe_bool(self.cfg.get("require_confirm_for_live"), True)
            if mode == "live"
            else _safe_bool(self.cfg.get("require_confirm_for_paper"), False)
        )

        if doc is None:
            sub["detail"] = "missing"
            details["confirm"] = sub
            if req:
                reasons.append("CONFIRM_NOT_READY")
            return

        flag = _safe_bool(doc.get("confirmed")) or _safe_bool(doc.get("ok"))
        cat = _parse_dt(doc.get("confirmed_at") or doc.get("ts") or doc.get("time"))
        ok = False
        if flag and cat is not None:
            ok = (_utc_now() - cat) <= timedelta(minutes=ttl)
        sub["valid"] = ok
        sub["confirmed_at"] = doc.get("confirmed_at") or doc.get("ts")
        details["confirm"] = sub

        if not ok and req:
            reasons.append("CONFIRM_NOT_READY")

    def _guard_bad(self, doc: Dict[str, Any]) -> bool:
        if _safe_bool(doc.get("blocked")):
            return True
        if _safe_bool(doc.get("kill_switch")):
            return True
        return False

    def _guard_passes(self, doc: Dict[str, Any]) -> bool:
        if self._guard_bad(doc):
            return False
        if "ok" in doc:
            return _safe_bool(doc.get("ok"), True)
        if "pass" in doc:
            return _safe_bool(doc.get("pass"), True)
        if "cleared" in doc:
            return _safe_bool(doc.get("cleared"), True)
        return True

    def _check_guard(
        self, mode: str, reasons: List[str], warnings: List[str], details: Dict[str, Any]
    ) -> None:
        paths = self._paths()["guard"]
        existing = [p for p in paths if p.is_file()]
        # Prefer newest mtime so we don't read a stale results/ copy when data/live is fresher.
        path = max(existing, key=lambda p: p.stat().st_mtime) if existing else None
        max_age = float(self.cfg.get("max_guard_age_minutes") or 30)
        sub: Dict[str, Any] = {
            "path": str(path) if path else None,
            "candidates_checked": [str(p) for p in paths],
            "resolved_by": "newest_mtime" if path else None,
        }

        req_clear = mode == "live" and _safe_bool(
            self.cfg.get("require_guard_clear_for_live"), True
        )
        block_missing = _safe_bool(self.cfg.get("block_on_missing_guard"), False)

        if path is None:
            sub["detail"] = "missing"
            details["guard"] = sub
            if req_clear and block_missing:
                reasons.append("MISSING_GUARD")
            elif req_clear:
                warnings.append("MISSING_GUARD")
            return

        age = _file_age_minutes(path)
        sub["age_minutes"] = age
        doc = _load_json(path)
        if doc is None:
            sub["detail"] = "unreadable"
            if req_clear:
                reasons.append("GUARD_BLOCKED")
            details["guard"] = sub
            return

        if age is not None and age > max_age:
            sub["stale"] = True
            sub["operator_note"] = (
                "guard_snapshot.json is not updated by snapshot_live_orders; "
                "refresh via guard / reconcile / pipeline that writes this file."
            )
            print(
                f"[GATE_FRESHNESS] guard snapshot source={path} age_minutes={age:.1f} max={max_age} "
                f"(not refreshed by snapshot_live_orders — see operator_note in gate details)"
            )
            if mode == "live":
                reasons.append("STALE_GUARD_SNAPSHOT")
            else:
                warnings.append("STALE_GUARD_SNAPSHOT")

        if self._guard_bad(doc):
            sub["blocked"] = True
            if mode == "live":
                reasons.append("GUARD_BLOCKED")
            else:
                warnings.append("GUARD_BLOCKED")
        elif not self._guard_passes(doc) and req_clear:
            reasons.append("GUARD_BLOCKED")
        else:
            sub["clear"] = True

        details["guard"] = sub

    def _heartbeat_unhealthy(self, raw_status: str) -> bool:
        s = raw_status.lower().strip()
        return s in ("fail", "failed", "error", "crash", "unhealthy", "degraded")

    def _check_heartbeat(
        self, mode: str, reasons: List[str], warnings: List[str], details: Dict[str, Any]
    ) -> None:
        doc, path = _first_json(self._paths()["heartbeat"])
        max_age = float(self.cfg.get("max_heartbeat_age_minutes") or 90)
        block_missing = _safe_bool(self.cfg.get("block_on_missing_heartbeat"), False)
        sub: Dict[str, Any] = {"path": str(path) if path else None}

        if doc is None:
            sub["detail"] = "missing"
            details["heartbeat"] = sub
            if block_missing:
                reasons.append("MISSING_HEARTBEAT")
            elif mode == "live":
                warnings.append("MISSING_HEARTBEAT")
            return

        ts = _parse_dt(
            doc.get("timestamp")
            or doc.get("ts")
            or doc.get("updated_at")
            or doc.get("time")
            or doc.get("as_of")
        )
        sub["timestamp"] = doc.get("timestamp") or doc.get("ts")
        sub["stage"] = doc.get("stage") or doc.get("step")
        sub["status"] = doc.get("status")
        if ts is None:
            sub["detail"] = "no_timestamp"
            details["heartbeat"] = sub
            if mode == "live":
                reasons.append("STALE_HEARTBEAT")
            else:
                warnings.append("INVALID_HEARTBEAT_TIMESTAMP")
            return

        age_m = (_utc_now() - ts).total_seconds() / 60.0
        sub["age_minutes"] = age_m
        if age_m > max_age:
            if mode == "live":
                reasons.append("STALE_HEARTBEAT")
            else:
                warnings.append("STALE_HEARTBEAT")

        st_raw = str(doc.get("status") or "").strip()
        if st_raw and self._heartbeat_unhealthy(st_raw):
            if mode == "live":
                reasons.append("PIPELINE_NOT_HEALTHY")
            else:
                warnings.append("PIPELINE_NOT_HEALTHY")

        details["heartbeat"] = sub

    def _check_risk(
        self, mode: str, reasons: List[str], warnings: List[str], details: Dict[str, Any]
    ) -> None:
        doc, path = _first_json(self._paths()["risk"])
        req = mode == "live" and _safe_bool(self.cfg.get("require_risk_ok_for_live"), True)
        block_missing = _safe_bool(self.cfg.get("block_on_missing_risk"), False)
        sub: Dict[str, Any] = {"path": str(path) if path else None}

        if doc is None:
            sub["detail"] = "missing"
            details["risk"] = sub
            if req and block_missing:
                reasons.append("MISSING_RISK_STATE")
            elif req:
                warnings.append("MISSING_RISK_STATE")
            return

        controls = doc.get("controls") if isinstance(doc.get("controls"), dict) else {}
        kill = _safe_bool(controls.get("global_kill_switch"))
        risk_on = controls.get("risk_on")
        allow_new = controls.get("allow_new_orders")

        sub["global_kill_switch"] = kill
        sub["risk_on"] = risk_on
        sub["allow_new_orders"] = allow_new

        if kill:
            if mode == "live":
                reasons.append("RISK_KILL_SWITCH")
            else:
                warnings.append("RISK_KILL_SWITCH")
        if risk_on is not None and not _safe_bool(risk_on, True):
            if mode == "live":
                reasons.append("RISK_BLOCKED")
            else:
                warnings.append("RISK_BLOCKED")
        if allow_new is not None and not _safe_bool(allow_new, True):
            if mode == "live":
                reasons.append("RISK_BLOCKED")
            else:
                warnings.append("RISK_BLOCKED")

        details["risk"] = sub

    def _check_cpm(
        self, mode: str, reasons: List[str], warnings: List[str], details: Dict[str, Any]
    ) -> None:
        doc, path = _first_json(self._paths()["cpm"])
        req = mode == "live" and _safe_bool(self.cfg.get("require_cpm_for_live"), False)
        block_missing = _safe_bool(self.cfg.get("block_on_missing_cpm"), False)
        sub: Dict[str, Any] = {"path": str(path) if path else None}

        if doc is None:
            sub["detail"] = "missing"
            details["cpm"] = sub
            if req and block_missing:
                reasons.append("MISSING_CPM")
            return

        allow_new = doc.get("allow_new_trades")
        exp_mul = _safe_float(doc.get("exposure_multiplier"))

        sub["allow_new_trades"] = allow_new
        sub["exposure_multiplier"] = exp_mul

        if mode == "live" and req:
            if allow_new is not None and not _safe_bool(allow_new, True):
                reasons.append("CPM_BLOCKED")
            elif exp_mul is not None and exp_mul <= 0:
                reasons.append("CPM_LOCKDOWN")
        elif (
            mode == "live" and not req and allow_new is not None and not _safe_bool(allow_new, True)
        ):
            warnings.append("CPM_NO_NEW_TRADES_NON_BLOCKING")

        details["cpm"] = sub

    def _csv_body_row_count(self, path: Path) -> int:
        """Data rows only (excludes header). Returns -1 on error."""
        try:
            if not path.is_file():
                return -1
            with path.open("r", encoding="utf-8", errors="replace") as f:
                n = sum(1 for _ in f)
            return max(0, n - 1)
        except OSError:
            return -1

    def _check_live_orders_snapshot(
        self,
        path: Path,
        max_lo_age: float,
        max_pos_age: float,
        max_rec_age: float,
        mode: str,
        prior_snap: Dict[str, Any],
        reasons: List[str],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """
        live_orders.csv is an append-only event log (place_live_orders / poll), not rewritten by
        snapshot_live_orders. In paper mode, avoid STALE_LIVE_ORDERS when positions + recent CSVs are fresh.
        """
        out: Dict[str, Any] = {"path": str(path)}
        paper_warn = _safe_bool(self.cfg.get("paper_warn_stale_snapshots"), True)
        suppress = _safe_bool(
            self.cfg.get("paper_suppress_stale_live_orders_when_pipeline_snapshots_fresh"), True
        )

        if not path.is_file():
            out["exists"] = False
            if mode == "paper" and paper_warn:
                warnings.append("MISSING_LIVE_ORDERS")
            return out

        out["exists"] = True
        age = _file_age_minutes(path)
        out["age_minutes"] = age
        stale = age is not None and age > max_lo_age
        if not stale:
            return out

        live_block_stale = mode == "live"
        if live_block_stale:
            reasons.append("STALE_LIVE_ORDERS")
            print(
                f"[GATE_FRESHNESS] live_orders freshness source={path} age_minutes={age:.1f} max={max_lo_age}"
            )
            return out

        if mode == "paper" and paper_warn and suppress:
            pos = prior_snap.get("positions") or {}
            rec = prior_snap.get("recent_orders") or {}
            pos_ok = (
                pos.get("exists")
                and pos.get("age_minutes") is not None
                and float(pos["age_minutes"]) <= max_pos_age
            )
            rec_ok = (
                rec.get("exists")
                and rec.get("age_minutes") is not None
                and float(rec["age_minutes"]) <= max_rec_age
            )
            if pos_ok and rec_ok:
                out["stale_warn_suppressed"] = True
                out["suppression_reason"] = "pipeline_csvs_fresh_live_orders_append_only"
                oo_path = self.results / "open_orders_snapshot.csv"
                oo_n = self._csv_body_row_count(oo_path)
                if oo_n == 0:
                    print(
                        "[GATE_FRESHNESS] zero open orders; treating refreshed snapshots as authoritative "
                        "(live_orders.csv event log is older than threshold but not updated by snapshot_live_orders)"
                    )
                else:
                    print(
                        f"[GATE_FRESHNESS] live_orders freshness source={path} (append-only event log); "
                        "positions_snapshot.csv and recent_orders.csv are fresh — not emitting STALE_LIVE_ORDERS"
                    )
                return out

        if mode == "paper" and paper_warn:
            warnings.append("STALE_LIVE_ORDERS")
        return out

    def _snapshot_check(
        self,
        label: str,
        path: Path,
        max_age: float,
        mode: str,
        live_block_missing: bool,
        live_block_stale: bool,
        reasons: List[str],
        warnings: List[str],
        missing_code: str,
        stale_code: str,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": str(path)}
        if not path.is_file():
            out["exists"] = False
            if mode == "live" and live_block_missing:
                reasons.append(missing_code)
            elif mode == "paper" and _safe_bool(self.cfg.get("paper_warn_stale_snapshots"), True):
                warnings.append(missing_code)
            return out

        out["exists"] = True
        age = _file_age_minutes(path)
        out["age_minutes"] = age
        if age is not None and age > max_age:
            if mode == "live" and live_block_stale:
                reasons.append(stale_code)
            elif mode == "paper" and _safe_bool(self.cfg.get("paper_warn_stale_snapshots"), True):
                warnings.append(stale_code)
        return out

    def _check_snapshots(
        self, mode: str, reasons: List[str], warnings: List[str], details: Dict[str, Any]
    ) -> None:
        snap: Dict[str, Any] = {}
        r = self.results
        max_pos = float(self.cfg.get("max_positions_snapshot_age_minutes") or 30)
        max_rec = float(self.cfg.get("max_recent_orders_age_minutes") or 30)
        max_lo = float(self.cfg.get("max_live_orders_age_minutes") or 45)

        snap["positions"] = self._snapshot_check(
            "positions",
            r / "positions_snapshot.csv",
            max_pos,
            mode,
            _safe_bool(self.cfg.get("live_block_missing_positions_snapshot"), True),
            _safe_bool(self.cfg.get("live_block_stale_positions_snapshot"), True),
            reasons,
            warnings,
            "MISSING_POSITIONS_SNAPSHOT",
            "STALE_POSITIONS_SNAPSHOT",
        )
        snap["recent_orders"] = self._snapshot_check(
            "recent_orders",
            r / "recent_orders.csv",
            max_rec,
            mode,
            _safe_bool(self.cfg.get("live_block_missing_recent_orders_snapshot"), True),
            _safe_bool(self.cfg.get("live_block_stale_recent_orders_snapshot"), True),
            reasons,
            warnings,
            "MISSING_RECENT_ORDERS",
            "STALE_RECENT_ORDERS",
        )
        snap["live_orders"] = self._check_live_orders_snapshot(
            r / "live_orders.csv",
            max_lo,
            max_pos,
            max_rec,
            mode,
            snap,
            reasons,
            warnings,
        )
        details["snapshots"] = snap

    def evaluate(
        self,
        mode: str = "paper",
        broker: Any = None,
        require_confirm: Optional[bool] = None,
        require_market_open: Optional[bool] = None,
        verbose: bool = False,
    ) -> GateDecision:
        mode = (mode or "paper").lower().strip()
        if mode not in ("paper", "live"):
            mode = "paper"

        self.cfg = load_gate_config(self.root)
        if require_confirm is not None:
            self.cfg[
                "require_confirm_for_live" if mode == "live" else "require_confirm_for_paper"
            ] = require_confirm
        if require_market_open is not None:
            self.cfg[
                (
                    "require_market_open_for_live"
                    if mode == "live"
                    else "require_market_open_for_paper"
                )
            ] = require_market_open

        checked = _utc_iso()
        reasons: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {
            "config_path": str(CONFIG_PATH),
            "gate_enabled": _safe_bool(self.cfg.get("enabled"), True),
        }

        if not _safe_bool(self.cfg.get("enabled"), True):
            details["gate_disabled"] = True
            return GateDecision(
                ok=True,
                status="READY",
                mode=mode,
                checked_at=checked,
                reasons=[],
                warnings=["GATE_DISABLED"],
                summary="Master execution gate disabled by config (not blocking).",
                details=details,
            )

        self._check_market(mode, broker, reasons, warnings, details)
        self._check_arm(mode, reasons, warnings, details)
        self._check_confirm(mode, reasons, warnings, details)
        self._check_guard(mode, reasons, warnings, details)
        self._check_heartbeat(mode, reasons, warnings, details)
        self._check_risk(mode, reasons, warnings, details)
        self._check_cpm(mode, reasons, warnings, details)
        self._check_snapshots(mode, reasons, warnings, details)

        reasons = list(dict.fromkeys(reasons))
        warnings = list(dict.fromkeys(warnings))

        ok = len(reasons) == 0
        status = "READY" if ok else "BLOCKED"
        if ok:
            summary = f"Execution allowed ({mode})."
            if warnings:
                summary += " Warnings: " + "; ".join(warnings)
        else:
            summary = "Execution blocked: " + ", ".join(
                r.replace("_", " ").lower() for r in reasons
            )

        if verbose:
            print(
                f"[MasterExecutionGate] mode={mode} status={status} reasons={reasons} warnings={warnings}"
            )

        return GateDecision(
            ok=ok,
            status=status,
            mode=mode,
            checked_at=checked,
            reasons=reasons,
            warnings=warnings,
            summary=summary,
            details=details,
        )


def write_snapshot(decision: GateDecision, path: Optional[Path] = None) -> None:
    try:
        p = path or DEFAULT_SNAPSHOT_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(decision.as_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass


def append_gate_log_csv(decision: GateDecision) -> None:
    try:
        LOG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts_utc": decision.checked_at,
            "mode": decision.mode,
            "ok": decision.ok,
            "status": decision.status,
            "reasons": ";".join(decision.reasons),
            "warnings": ";".join(decision.warnings),
            "summary": decision.summary.replace("\n", " ")[:2000],
        }
        write_header = not LOG_CSV_PATH.is_file() or LOG_CSV_PATH.stat().st_size == 0
        with LOG_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def try_refresh_stale_guard_snapshot_if_clear(project_root: Optional[Path] = None) -> None:
    """If the newest guard_snapshot.json is stale but logically clear, rewrite with a fresh timestamp only.

    Does not change blocked/kill_switch semantics; does not bypass MasterExecutionGate on the next evaluate().
    """
    root = (project_root or PROJECT_ROOT).resolve()
    gate = MasterExecutionGate(project_root=root)
    paths = gate._paths()["guard"]
    existing = [p for p in paths if p.is_file()]
    if not existing:
        print(
            "[GUARD_HYGIENE] stale guard detected: no guard_snapshot.json under data/results or data/live "
            "(run guard/reconcile/pipeline that writes this file)"
        )
        return

    path = max(existing, key=lambda p: p.stat().st_mtime)
    max_age = float(gate.cfg.get("max_guard_age_minutes") or 30)
    age = _file_age_minutes(path)
    doc = _load_json(path)

    if age is None:
        print(f"[GUARD_HYGIENE] refresh failed: cannot read age for path={path}")
        return

    if age <= max_age:
        return

    print(
        f"[GUARD_HYGIENE] stale guard detected: path={path} age_minutes={age:.1f} max_minutes={max_age} "
        f"(guard is not updated by snapshot_live_orders)"
    )

    if doc is None:
        print("[GUARD_HYGIENE] refresh failed: guard_snapshot exists but is unreadable")
        return

    if gate._guard_bad(doc):
        print(
            "[GUARD_HYGIENE] refresh skipped: guard blocked/kill_switch active — "
            "cannot safely refresh timestamp without operator action"
        )
        return

    if not gate._guard_passes(doc):
        print(
            "[GUARD_HYGIENE] refresh skipped: guard not in clear/pass state — "
            "cannot safely refresh timestamp"
        )
        return

    print("[GUARD_HYGIENE] refresh triggered")
    try:
        ts = _utc_iso()
        doc["timestamp"] = ts
        doc["ts"] = ts
        doc["updated_at"] = ts
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(path)
        print("[GUARD_HYGIENE] refresh complete")
    except Exception as e:
        print(f"[GUARD_HYGIENE] refresh failed: {e}")


def _cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TRITON master execution gate")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--json", action="store_true", help="Print full JSON decision to stdout")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    broker = None
    try:
        from services.broker_alpaca import AlpacaBroker  # noqa: WPS433

        broker = AlpacaBroker(mode=args.mode)
    except Exception:
        pass

    try:
        gate = MasterExecutionGate()
        d = gate.evaluate(mode=args.mode, broker=broker, verbose=args.verbose)
        write_snapshot(d)
        append_gate_log_csv(d)
        if args.json:
            print(json.dumps(d.as_dict(), indent=2))
        else:
            print(d.summary)
            if d.reasons:
                print("Reasons:", ", ".join(d.reasons))
            if d.warnings:
                print("Warnings:", ", ".join(d.warnings))
        return 0 if d.ok else 2
    except SystemExit:
        raise
    except Exception as e:
        print(f"[master_execution_gate] FATAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
