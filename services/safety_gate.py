# services/safety_gate.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class SafetyGateError(Exception):
    """Raised when SafetyGate blocks execution."""

    pass


@dataclass
class SafetyGateDecision:
    allow: bool
    code: str
    message: str
    details: Dict[str, Any]


class SafetyGate:
    """
    Live Order Safety Gate — final execution firewall.

    Key guarantees:
    - FAIL CLOSED for live submissions.
    - Optional "ARM" token required to place live orders.
    - Weekend + market-open checks (configurable).
    """

    def __init__(
        self,
        *,
        root_dir: str,
        results_dir: str,
        guard_snapshot_path: str,
        arm_file: str,
        default_arm_ttl_mins: int = 120,
        block_weekends_utc: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.results_dir = results_dir
        self.guard_snapshot_path = guard_snapshot_path
        self.arm_file = arm_file
        self.default_arm_ttl_mins = int(default_arm_ttl_mins)
        self.block_weekends_utc = bool(block_weekends_utc)

    # ─────────────────────────────────────────────────────────────────────────
    # Time helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ─────────────────────────────────────────────────────────────────────────
    # Guard snapshot (fail-closed for live)
    # ─────────────────────────────────────────────────────────────────────────
    def read_guard_snapshot(self) -> Dict[str, Any]:
        p = Path(self.guard_snapshot_path)
        if not p.exists() or p.stat().st_size <= 0:
            return {}
        # BOM-safe read for Windows PowerShell Set-Content UTF8 (BOM)
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            txt = p.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").strip()
            if not txt:
                return {}
            try:
                return json.loads(txt)
            except Exception:
                return {}

    @staticmethod
    def guard_is_active(snapshot: Dict[str, Any]) -> Tuple[bool, str]:
        blocked = bool(snapshot.get("blocked", False))
        kill = bool(snapshot.get("kill_switch", False))
        if blocked or kill:
            code = str(snapshot.get("code", "GUARD_BLOCKED"))
            msg = snapshot.get("message") or snapshot.get("reason") or "Guard snapshot active"
            return True, f"{code}: {msg}"
        return False, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Arm token
    # ─────────────────────────────────────────────────────────────────────────
    def arm_live(self, *, session: str, ttl_mins: Optional[int] = None) -> Dict[str, Any]:
        os.makedirs(self.results_dir, exist_ok=True)
        ttl = int(ttl_mins if ttl_mins is not None else self.default_arm_ttl_mins)
        now = self._utc_now()
        exp = now + timedelta(minutes=max(1, ttl))
        payload = {
            "armed": True,
            "session": session,
            "armed_at": self._iso(now),
            "expires_at": self._iso(exp),
            "ttl_mins": ttl,
        }
        Path(self.arm_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def read_arm_token(self) -> Dict[str, Any]:
        p = Path(self.arm_file)
        if not p.exists() or p.stat().st_size <= 0:
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def arm_is_valid(self) -> Tuple[bool, str, Dict[str, Any]]:
        token = self.read_arm_token()
        if not token or not bool(token.get("armed", False)):
            return False, "ARM_MISSING", token

        try:
            exp_s = str(token.get("expires_at", "")).strip()
            exp = datetime.strptime(exp_s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            return False, "ARM_MALFORMED", token

        now = self._utc_now()
        if now >= exp:
            return False, "ARM_EXPIRED", token
        return True, "ARM_OK", token

    # ─────────────────────────────────────────────────────────────────────────
    # Session checks
    # ─────────────────────────────────────────────────────────────────────────
    def is_weekend_utc(self) -> bool:
        # Monday=0 ... Sunday=6
        wd = self._utc_now().weekday()
        return wd >= 5

    # ─────────────────────────────────────────────────────────────────────────
    # Primary live preflight
    # ─────────────────────────────────────────────────────────────────────────
    def preflight_live(
        self,
        *,
        session: str,
        is_market_open: Optional[bool],
        require_market_open: bool,
        ignore_arm: bool,
        allow_weekend: bool,
        verbose: bool = False,
    ) -> SafetyGateDecision:
        # 1) Guard snapshot — supreme
        snap = self.read_guard_snapshot()
        active, reason = self.guard_is_active(snap)
        if verbose:
            keys = sorted(list(snap.keys())) if isinstance(snap, dict) else []
            print(f"[SafetyGate] guard_path={self.guard_snapshot_path}")
            print(f"[SafetyGate] guard_keys={keys}")
            print(
                f"[SafetyGate] guard_blocked={bool(snap.get('blocked', False))} guard_kill={bool(snap.get('kill_switch', False))}"
            )
        if active:
            return SafetyGateDecision(
                allow=False,
                code="GUARD_BLOCK",
                message=f"Live blocked by guard_snapshot.json -> {reason}",
                details={"snapshot": snap},
            )

        # 2) Weekend block (UTC)
        if self.block_weekends_utc and (not allow_weekend) and self.is_weekend_utc():
            return SafetyGateDecision(
                allow=False,
                code="WEEKEND_BLOCK",
                message="Live blocked on weekend (UTC). Use --allow-weekend to override.",
                details={"utc_now": self._iso(self._utc_now())},
            )

        # 3) Market-open requirement
        if require_market_open:
            if is_market_open is False:
                return SafetyGateDecision(
                    allow=False,
                    code="MARKET_CLOSED",
                    message="Market is CLOSED. Live blocked (use --allow-market-closed to override).",
                    details={},
                )
            if is_market_open is None:
                # fail-closed when caller requested market-open enforcement
                return SafetyGateDecision(
                    allow=False,
                    code="CLOCK_UNKNOWN",
                    message="Market clock unknown; fail-closed because require_market_open=True.",
                    details={},
                )

        # 4) Arm token requirement (unless ignore_arm)
        if not ignore_arm:
            ok, code, tok = self.arm_is_valid()
            if not ok:
                return SafetyGateDecision(
                    allow=False,
                    code=code,
                    message="Live blocked: trading is not ARMED (or arm expired). Use --arm-live to arm.",
                    details={"arm": tok},
                )

        return SafetyGateDecision(
            allow=True,
            code="ALLOW",
            message="SafetyGate passed.",
            details={"session": session},
        )
