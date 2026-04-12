# dashboard/live_trading_panel.py
# ------------------------------------------------------------
# TRITON Command Center — Live Trading Panel (Streamlit)
#
# Safety:
# - No direct order placement here.
# - ARM / DISARM writes live_armed.json
# - GUARD kill switch writes guard_snapshot.json
# - Typed confirmation writes live_confirm.json
# - Broker snapshot hidden unless:
#       ARM=ON AND GUARD=CLEAR AND CONFIRMED (not expired)
# - Snapshot Now writes:
#     data/results/live_orders.csv (open)
#     data/results/recent_orders.csv (all)
#     data/results/positions_snapshot.csv (positions)
#
# UX:
# - Safety Files Status strip with last-modified + size
# - Snapshot Freshness tiles (FRESH/AGING/STALE) based on CSV mtimes
# ------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import streamlit as st


# -------------------------
# Paths (absolute, cwd-safe)
# -------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

LIVE_ARMED_PATH = RESULTS_DIR / "live_armed.json"
GUARD_SNAPSHOT_PATH = RESULTS_DIR / "guard_snapshot.json"
LIVE_CONFIRM_PATH = RESULTS_DIR / "live_confirm.json"

CPM_STATE_PATH = RESULTS_DIR / "capital_preservation_state.json"
RISK_STATE_PATH = RESULTS_DIR / "adaptive_risk_state.json"

LIVE_ORDERS_LOG = RESULTS_DIR / "live_orders.csv"  # OPEN orders snapshot
RECENT_ORDERS_LOG = RESULTS_DIR / "recent_orders.csv"  # ALL recent orders
POSITIONS_SNAPSHOT = RESULTS_DIR / "positions_snapshot.csv"

CONFIRM_PHRASE = "TRITON LIVE CONFIRM"


# -------------------------
# File / JSON helpers
# -------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        txt = path.read_text(encoding="utf-8-sig")
        txt = txt.lstrip("\ufeff").strip()
        if not txt:
            return {}
        d = json.loads(txt)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        if isinstance(s, str) and s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(str(s))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _pick_cols(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    return [c for c in preferred if c in df.columns]


def _file_meta(path: Path) -> Dict[str, Any]:
    """Return {exists, size_bytes, mtime_iso, age_mins}."""
    try:
        if not path.exists():
            return {"exists": False, "size_bytes": 0, "mtime_iso": None, "age_mins": None}
        stt = path.stat()
        mtime = datetime.fromtimestamp(stt.st_mtime, tz=timezone.utc)
        age_mins = int(max(0, (_utc_now() - mtime).total_seconds() // 60))
        return {
            "exists": True,
            "size_bytes": int(stt.st_size),
            "mtime_iso": _iso(mtime),
            "age_mins": age_mins,
        }
    except Exception:
        return {"exists": False, "size_bytes": 0, "mtime_iso": None, "age_mins": None}


def _fmt_bytes(n: int) -> str:
    if n is None:
        return ""
    n = int(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _status_badge(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        st.success(label)
        if detail:
            st.caption(detail)
    else:
        st.warning(label)
        if detail:
            st.caption(detail)


# -------------------------
# Snapshot freshness helpers
# -------------------------
def _fresh_label(age_mins: Optional[int]) -> tuple[str, str]:
    """Returns (status_text, level) where level in {'fresh','aging','stale','missing'}."""
    if age_mins is None:
        return ("MISSING", "missing")
    if age_mins <= 5:
        return (f"FRESH • {age_mins}m", "fresh")
    if age_mins <= 30:
        return (f"AGING • {age_mins}m", "aging")
    return (f"STALE • {age_mins}m", "stale")


def _render_freshness_tile(title: str, meta: Dict[str, Any]) -> None:
    if not meta.get("exists", False):
        st.error(f"{title}: MISSING")
        return

    age = meta.get("age_mins")
    status_text, level = _fresh_label(age)
    detail = f"updated={meta.get('mtime_iso')} • size={_fmt_bytes(meta.get('size_bytes', 0))}"

    if level == "fresh":
        st.success(f"{title}: {status_text}")
        st.caption(detail)
    elif level == "aging":
        st.warning(f"{title}: {status_text}")
        st.caption(detail)
    else:
        st.error(f"{title}: {status_text}")
        st.caption(detail)


# -------------------------
# ARM
# -------------------------
@dataclass
class ArmStatus:
    armed: bool
    reason: str
    session: str = ""
    armed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    ttl_mins: Optional[int] = None


def get_arm_status() -> ArmStatus:
    d = _read_json(LIVE_ARMED_PATH)
    armed = bool(d.get("armed", False))
    exp = _parse_iso(d.get("expires_at"))
    now = _utc_now()

    if not armed:
        return ArmStatus(False, "DISARMED")

    if exp and now >= exp:
        return ArmStatus(False, "EXPIRED", session=str(d.get("session", "")), expires_at=exp)

    ttl = d.get("ttl_mins", None)
    ttl_val = int(ttl) if ttl is not None and str(ttl).strip() != "" else None

    return ArmStatus(
        True,
        "ARMED",
        session=str(d.get("session", "")),
        armed_at=_parse_iso(d.get("armed_at")),
        expires_at=exp,
        ttl_mins=ttl_val,
    )


def arm_live(ttl_mins: int) -> Dict[str, Any]:
    now = _utc_now()
    exp = now + timedelta(minutes=int(ttl_mins))
    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "armed": True,
        "session": session,
        "armed_at": _iso(now),
        "expires_at": _iso(exp),
        "ttl_mins": int(ttl_mins),
    }
    _atomic_write_json(LIVE_ARMED_PATH, payload)
    return payload


def disarm_live() -> None:
    d = _read_json(LIVE_ARMED_PATH) or {}
    d["armed"] = False
    d["disarmed_at"] = _iso(_utc_now())
    _atomic_write_json(LIVE_ARMED_PATH, d)


# -------------------------
# CONFIRM (Typed)
# -------------------------
@dataclass
class ConfirmStatus:
    confirmed: bool
    reason: str
    confirmed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    arm_session: str = ""


def get_confirm_status(current_arm_session: str = "") -> ConfirmStatus:
    d = _read_json(LIVE_CONFIRM_PATH)
    confirmed = bool(d.get("confirmed", False))
    exp = _parse_iso(d.get("expires_at"))
    now = _utc_now()

    if not confirmed:
        return ConfirmStatus(False, "NOT_CONFIRMED")

    if exp and now >= exp:
        return ConfirmStatus(
            False, "EXPIRED", expires_at=exp, arm_session=str(d.get("arm_session", ""))
        )

    stored_sess = str(d.get("arm_session", "") or "")
    if stored_sess and current_arm_session and stored_sess != current_arm_session:
        return ConfirmStatus(False, "ARM_SESSION_MISMATCH", arm_session=stored_sess)

    return ConfirmStatus(
        True,
        "CONFIRMED",
        confirmed_at=_parse_iso(d.get("confirmed_at")),
        expires_at=exp,
        arm_session=stored_sess,
    )


def set_confirm(ttl_mins: int, arm_session: str = "") -> Dict[str, Any]:
    now = _utc_now()
    exp = now + timedelta(minutes=int(ttl_mins))
    payload = {
        "confirmed": True,
        "confirmed_at": _iso(now),
        "expires_at": _iso(exp),
        "ttl_mins": int(ttl_mins),
        "arm_session": arm_session or "",
    }
    _atomic_write_json(LIVE_CONFIRM_PATH, payload)
    return payload


def clear_confirm() -> Dict[str, Any]:
    payload = {
        "confirmed": False,
        "cleared_at": _iso(_utc_now()),
    }
    _atomic_write_json(LIVE_CONFIRM_PATH, payload)
    return payload


# -------------------------
# GUARD
# -------------------------
def set_guard_kill(code: str, message: str) -> Dict[str, Any]:
    now = _utc_now()
    payload = {
        "blocked": True,
        "kill_switch": True,
        "code": code,
        "message": message,
        "updated_at": _iso(now),
        "reason": message,
        "extra": {"source": "dashboard/live_trading_panel.py"},
    }
    _atomic_write_json(GUARD_SNAPSHOT_PATH, payload)
    return payload


def clear_guard() -> Dict[str, Any]:
    now = _utc_now()
    payload = {
        "blocked": False,
        "kill_switch": False,
        "code": "CLEAR",
        "message": "Cleared from dashboard",
        "updated_at": _iso(now),
        "reason": "Cleared from dashboard",
        "extra": {"source": "dashboard/live_trading_panel.py"},
    }
    _atomic_write_json(GUARD_SNAPSHOT_PATH, payload)
    return payload


def _guard_status() -> Dict[str, Any]:
    snap = _read_json(GUARD_SNAPSHOT_PATH)
    return {
        "blocked": bool(snap.get("blocked", False)),
        "kill": bool(snap.get("kill_switch", False)),
        "code": str(snap.get("code", "") or ""),
        "message": snap.get("message") or snap.get("reason") or "",
        "raw": snap,
    }


# -------------------------
# Broker (optional)
# -------------------------
def _try_get_broker(mode: str):
    try:
        from services.broker_alpaca import AlpacaBroker  # type: ignore

        return AlpacaBroker(mode=mode)
    except Exception:
        return None


def _get_clock(broker) -> Optional[Dict[str, Any]]:
    if broker is None:
        return None
    for meth in ("get_clock", "clock", "get_market_clock"):
        fn = getattr(broker, meth, None)
        if callable(fn):
            try:
                c = fn()
                return c if isinstance(c, dict) else None
            except Exception:
                return None
    return None


def _clock_is_open(clock: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not clock:
        return None
    v = clock.get("is_open", None)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y")
    return None


def _get_account(broker) -> Dict[str, Any]:
    try:
        return broker.get_account() if broker else {}
    except Exception:
        return {}


def _get_positions(broker) -> List[Dict[str, Any]]:
    try:
        return broker.get_positions() if broker else []
    except Exception:
        return []


def _get_open_orders(broker) -> List[Dict[str, Any]]:
    try:
        return broker.list_orders(status="open", nested=True, limit=500) if broker else []
    except Exception:
        return []


def _get_recent_orders(broker, limit: int = 200) -> List[Dict[str, Any]]:
    try:
        return broker.list_orders(status="all", nested=True, limit=limit) if broker else []
    except Exception:
        return []


def _snapshot_to_csvs(broker, *, recent_limit: int = 200) -> Dict[str, Any]:
    ts = _iso(_utc_now())

    open_orders = _get_open_orders(broker)
    recent_orders = _get_recent_orders(broker, limit=recent_limit)
    positions = _get_positions(broker)

    df_open = pd.DataFrame(open_orders) if open_orders else pd.DataFrame()
    df_recent = pd.DataFrame(recent_orders) if recent_orders else pd.DataFrame()
    df_pos = pd.DataFrame(positions) if positions else pd.DataFrame()

    for df in (df_open, df_recent, df_pos):
        if not df.empty and "snapshot_ts" not in df.columns:
            df.insert(0, "snapshot_ts", ts)

    _safe_write_csv(
        LIVE_ORDERS_LOG, df_open if not df_open.empty else pd.DataFrame(columns=["snapshot_ts"])
    )
    _safe_write_csv(
        RECENT_ORDERS_LOG,
        df_recent if not df_recent.empty else pd.DataFrame(columns=["snapshot_ts"]),
    )
    _safe_write_csv(
        POSITIONS_SNAPSHOT, df_pos if not df_pos.empty else pd.DataFrame(columns=["snapshot_ts"])
    )

    return {
        "ok": True,
        "ts": ts,
        "open_orders_written": int(len(df_open)) if not df_open.empty else 0,
        "recent_orders_written": int(len(df_recent)) if not df_recent.empty else 0,
        "positions_written": int(len(df_pos)) if not df_pos.empty else 0,
        "paths": {
            "open_orders": str(LIVE_ORDERS_LOG),
            "recent_orders": str(RECENT_ORDERS_LOG),
            "positions": str(POSITIONS_SNAPSHOT),
        },
    }


# -------------------------
# Safety Files Status strip
# -------------------------
def _render_safety_files_status() -> None:
    st.subheader("Safety Files Status")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    meta_cpm = _file_meta(CPM_STATE_PATH)
    with c1:
        _status_badge(
            "CPM",
            ok=meta_cpm["exists"],
            detail=(
                "optional • missing"
                if not meta_cpm["exists"]
                else f"{_fmt_bytes(meta_cpm['size_bytes'])} • age ~{meta_cpm['age_mins']}m"
            ),
        )

    meta_risk = _file_meta(RISK_STATE_PATH)
    with c2:
        _status_badge(
            "RISK",
            ok=meta_risk["exists"],
            detail=(
                "optional • missing"
                if not meta_risk["exists"]
                else f"{_fmt_bytes(meta_risk['size_bytes'])} • age ~{meta_risk['age_mins']}m"
            ),
        )

    meta_guard = _file_meta(GUARD_SNAPSHOT_PATH)
    with c3:
        _status_badge(
            "GUARD",
            ok=meta_guard["exists"],
            detail=(
                "missing"
                if not meta_guard["exists"]
                else f"{_fmt_bytes(meta_guard['size_bytes'])} • age ~{meta_guard['age_mins']}m"
            ),
        )

    meta_arm = _file_meta(LIVE_ARMED_PATH)
    with c4:
        _status_badge(
            "ARM FILE",
            ok=meta_arm["exists"],
            detail=(
                "missing"
                if not meta_arm["exists"]
                else f"{_fmt_bytes(meta_arm['size_bytes'])} • age ~{meta_arm['age_mins']}m"
            ),
        )

    meta_confirm = _file_meta(LIVE_CONFIRM_PATH)
    with c5:
        _status_badge(
            "CONFIRM",
            ok=meta_confirm["exists"],
            detail=(
                "missing"
                if not meta_confirm["exists"]
                else f"{_fmt_bytes(meta_confirm['size_bytes'])} • age ~{meta_confirm['age_mins']}m"
            ),
        )

    meta_recent = _file_meta(RECENT_ORDERS_LOG)
    with c6:
        _status_badge(
            "RECENT CSV",
            ok=meta_recent["exists"] and meta_recent["size_bytes"] > 20,
            detail=(
                "missing/empty"
                if not meta_recent["exists"]
                else f"{_fmt_bytes(meta_recent['size_bytes'])} • age ~{meta_recent['age_mins']}m"
            ),
        )

    meta_pos = _file_meta(POSITIONS_SNAPSHOT)
    with c7:
        _status_badge(
            "POSITIONS CSV",
            ok=meta_pos["exists"] and meta_pos["size_bytes"] > 20,
            detail=(
                "missing/empty"
                if not meta_pos["exists"]
                else f"{_fmt_bytes(meta_pos['size_bytes'])} • age ~{meta_pos['age_mins']}m"
            ),
        )

    st.caption(
        "Note: CPM + Risk are optional readouts. Guard + Arm + Confirm are required for broker visibility in this panel."
    )


# -------------------------
# UI
# -------------------------
def render_live_trading_panel() -> None:
    st.title("🔴 Live Trading")

    with st.sidebar:
        st.subheader("Broker Mode")
        mode = st.selectbox("Mode", ["paper", "live"], index=0)

        st.subheader("Refresh")
        auto = st.toggle("Auto-refresh", value=False)
        interval = st.slider("Refresh interval (seconds)", 5, 60, 15, disabled=not auto)

        if auto:
            st.autorefresh(interval=interval * 1000, key="live_trading_autorefresh")

    # Safety files status strip + freshness
    _render_safety_files_status()

    st.subheader("Snapshot Freshness")
    m_open = _file_meta(LIVE_ORDERS_LOG)
    m_recent = _file_meta(RECENT_ORDERS_LOG)
    m_pos = _file_meta(POSITIONS_SNAPSHOT)

    f1, f2, f3 = st.columns(3)
    with f1:
        _render_freshness_tile("OPEN ORDERS CSV", m_open)
    with f2:
        _render_freshness_tile("RECENT ORDERS CSV", m_recent)
    with f3:
        _render_freshness_tile("POSITIONS CSV", m_pos)

    st.caption(
        "These timestamps come from the CSV files on disk. Click 📸 Snapshot Now to refresh."
    )
    st.divider()

    # Status
    arm = get_arm_status()
    guard = _guard_status()
    confirm = get_confirm_status(current_arm_session=arm.session)

    broker_allowed = (
        bool(arm.armed) and not bool(guard["blocked"] or guard["kill"]) and bool(confirm.confirmed)
    )
    broker = _try_get_broker(mode) if broker_allowed else None

    clock = _get_clock(broker) if broker_allowed else None
    is_open = _clock_is_open(clock) if broker_allowed else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if arm.armed:
            st.success("ARM: ARMED")
            if arm.expires_at:
                remaining = arm.expires_at - _utc_now()
                mins = int(max(0, remaining.total_seconds() // 60))
                st.caption(f"expires ~{mins}m • {_iso(arm.expires_at)}")
            if arm.session:
                st.caption(f"session={arm.session}")
        else:
            st.error(f"ARM: {arm.reason}")

    with col2:
        if guard["blocked"] or guard["kill"]:
            st.error("GUARD: BLOCKED")
            st.caption(f"{guard.get('code','')}: {guard.get('message','')}".strip())
        else:
            st.success("GUARD: CLEAR")
            st.caption("blocked=false • kill_switch=false")

    with col3:
        if confirm.confirmed:
            st.success("CONFIRM: OK")
            if confirm.expires_at:
                remaining = confirm.expires_at - _utc_now()
                mins = int(max(0, remaining.total_seconds() // 60))
                st.caption(f"expires ~{mins}m • {_iso(confirm.expires_at)}")
            if confirm.arm_session:
                st.caption(f"bound_session={confirm.arm_session}")
        else:
            st.error(f"CONFIRM: {confirm.reason}")

    with col4:
        if not broker_allowed:
            st.warning("BROKER: LOCKED")
            st.caption("Requires ARM + GUARD CLEAR + CONFIRM.")
        else:
            if is_open is True:
                st.success("MARKET: OPEN")
            elif is_open is False:
                st.warning("MARKET: CLOSED")
            else:
                st.info("MARKET: UNKNOWN")
            if clock:
                for k in ("timestamp", "next_open", "next_close"):
                    if k in clock and clock.get(k) is not None:
                        st.caption(f"{k}={clock.get(k)}")

    st.divider()

    # Controls
    st.subheader("Controls (Safe Gates)")
    cA, cB, cC = st.columns([1, 1, 2], vertical_alignment="top")

    with cA:
        ttl = st.selectbox("ARM TTL (minutes)", [60, 120, 240, 600], index=1)
        if st.button("🔐 ARM LIVE", use_container_width=True):
            payload = arm_live(int(ttl))
            st.success("LIVE ARMED")
            st.json(payload)
            st.rerun()

    with cB:
        if st.button("🧯 DISARM", use_container_width=True):
            disarm_live()
            st.warning("LIVE DISARMED")
            st.rerun()

    with cC:
        st.caption("Emergency Kill Switch blocks ALL live submissions immediately.")
        kill_msg = st.text_input("Kill message", value="Manual emergency stop from dashboard")
        k1, k2 = st.columns(2)
        with k1:
            if st.button("🚨 KILL SWITCH", use_container_width=True):
                payload = set_guard_kill(code="MANUAL_KILL", message=kill_msg)
                st.error("KILL SWITCH ACTIVATED")
                st.json(payload)
                st.rerun()
        with k2:
            if st.button("✅ CLEAR GUARD", use_container_width=True):
                payload = clear_guard()
                st.success("GUARD CLEARED")
                st.json(payload)
                st.rerun()

    st.divider()

    # Typed confirmation
    st.subheader("Typed Confirmation (Required)")
    cx, cy, cz = st.columns([1, 1, 2], vertical_alignment="top")

    with cx:
        confirm_ttl = st.selectbox("Confirm TTL (minutes)", [15, 30, 60, 120], index=2)

    with cy:
        typed = st.text_input("Type phrase to confirm", value="", placeholder=CONFIRM_PHRASE)

    with cz:
        st.caption(f"Required phrase: `{CONFIRM_PHRASE}`")
        b1, b2 = st.columns(2)
        with b1:
            if st.button(
                "✅ CONFIRM LIVE",
                use_container_width=True,
                disabled=(typed.strip() != CONFIRM_PHRASE or not arm.armed),
                help="Requires ARM to be ON. Confirmation expires automatically.",
            ):
                payload = set_confirm(int(confirm_ttl), arm_session=arm.session or "")
                st.success("LIVE CONFIRMED")
                st.json(payload)
                st.rerun()
        with b2:
            if st.button("🧹 CLEAR CONFIRM", use_container_width=True):
                payload = clear_confirm()
                st.warning("CONFIRM CLEARED")
                st.json(payload)
                st.rerun()

    st.divider()

    # Snapshot Now
    st.subheader("Snapshots (Read-only)")
    st.caption("Writes CSV snapshots to data/results/. No broker actions, no cancellations.")
    snap_col1, snap_col2 = st.columns([1, 2], vertical_alignment="top")
    with snap_col1:
        recent_limit = st.selectbox("Recent orders limit", [50, 100, 200, 500], index=2)
        do_snap = st.button(
            "📸 Snapshot Now", use_container_width=True, disabled=(not broker_allowed)
        )
    with snap_col2:
        if not broker_allowed:
            st.warning("Snapshot disabled until ARM=ON, GUARD=CLEAR, CONFIRM=OK.")
        else:
            st.info("Snapshot will refresh: open orders, recent orders, positions.")

    if do_snap and broker_allowed and broker is not None:
        try:
            res = _snapshot_to_csvs(broker, recent_limit=int(recent_limit))
            st.success("Snapshot written")
            st.json(res)
            st.rerun()
        except Exception as e:
            st.error(f"Snapshot failed: {e}")

    st.divider()

    # Safety Stack (read-only JSONs)
    st.subheader("Safety Stack (Read Only)")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Capital Preservation Mode (CPM)**")
        cpm = _read_json(CPM_STATE_PATH)
        if cpm:
            st.json(
                {
                    "mode": cpm.get("mode"),
                    "cpi": cpm.get("cpi"),
                    "allow_new_trades": cpm.get("allow_new_trades"),
                    "allow_increase": cpm.get("allow_increase"),
                    "exposure_multiplier": cpm.get("exposure_multiplier"),
                    "limit_only": cpm.get("limit_only"),
                    "cancel_open_orders": cpm.get("cancel_open_orders"),
                    "updated_at": cpm.get("updated_at") or cpm.get("as_of"),
                }
            )
        else:
            st.info(f"CPM state missing (optional): {CPM_STATE_PATH}")

    with s2:
        st.markdown("**RiskGate / Adaptive Risk State**")
        risk = _read_json(RISK_STATE_PATH)
        if risk:
            st.json(
                {
                    "ok": risk.get("ok"),
                    "regime": risk.get("regime"),
                    "mode": risk.get("mode"),
                    "allow_new_orders": risk.get("allow_new_orders"),
                    "global_kill_switch": risk.get("global_kill_switch"),
                    "max_position_weight": risk.get("max_position_weight"),
                    "max_gross_exposure": risk.get("max_gross_exposure"),
                    "reason": risk.get("reason"),
                    "updated_at": risk.get("updated_at") or risk.get("as_of"),
                }
            )
        else:
            st.info(f"Risk state missing (optional): {RISK_STATE_PATH}")

    st.divider()

    # Broker Snapshot
    st.subheader("Broker Snapshot")

    if not broker_allowed:
        st.warning(
            "Broker data is hidden. Requires **ARM LIVE**, **GUARD CLEAR**, and **CONFIRM OK**."
        )
        st.caption(f"live_armed={LIVE_ARMED_PATH}")
        st.caption(f"live_confirm={LIVE_CONFIRM_PATH}")
        st.caption(f"guard_snapshot={GUARD_SNAPSHOT_PATH}")
    else:
        acct = _get_account(broker)
        pos = _get_positions(broker)
        oo = _get_open_orders(broker)
        ro = _get_recent_orders(broker, limit=200)

        if acct:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Equity", str(acct.get("equity", "")))
            m2.metric("Buying Power", str(acct.get("buying_power", "")))
            m3.metric("Cash", str(acct.get("cash", "")))
            m4.metric("Status", str(acct.get("status", "")))
        else:
            st.warning("Broker account not available (check env vars / connectivity).")

        if pos:
            dfp = pd.DataFrame(pos)
            keep = _pick_cols(
                dfp,
                [
                    "symbol",
                    "side",
                    "qty",
                    "market_value",
                    "avg_entry_price",
                    "current_price",
                    "unrealized_pl",
                ],
            )
            st.markdown("**Positions**")
            st.dataframe(dfp[keep] if keep else dfp, use_container_width=True, height=240)

        st.markdown("**Open Orders**")
        if oo:
            dfo = pd.DataFrame(oo)
            show_cols = _pick_cols(
                dfo,
                [
                    "id",
                    "symbol",
                    "side",
                    "type",
                    "limit_price",
                    "stop_price",
                    "status",
                    "time_in_force",
                    "submitted_at",
                ],
            )
            st.dataframe(dfo[show_cols] if show_cols else dfo, use_container_width=True, height=260)
        else:
            st.info("No open orders.")

        st.markdown("**Recent Orders (status=all)**")
        if ro:
            dfr = pd.DataFrame(ro)
            show_cols = _pick_cols(
                dfr,
                [
                    "submitted_at",
                    "symbol",
                    "side",
                    "type",
                    "status",
                    "qty",
                    "filled_qty",
                    "filled_avg_price",
                    "limit_price",
                ],
            )
            st.dataframe(dfr[show_cols] if show_cols else dfr, use_container_width=True, height=320)
        else:
            st.info("No recent orders returned (unexpected).")

    st.divider()

    # Local CSV tails
    st.subheader("Local CSVs (tails)")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.caption("data/results/live_orders.csv (OPEN)")
        st.dataframe(_safe_read_csv(LIVE_ORDERS_LOG).tail(50), use_container_width=True, height=260)
    with t2:
        st.caption("data/results/recent_orders.csv (ALL)")
        st.dataframe(
            _safe_read_csv(RECENT_ORDERS_LOG).tail(50), use_container_width=True, height=260
        )
    with t3:
        st.caption("data/results/positions_snapshot.csv")
        st.dataframe(
            _safe_read_csv(POSITIONS_SNAPSHOT).tail(50), use_container_width=True, height=260
        )

    st.caption(f"ROOT={ROOT}")
