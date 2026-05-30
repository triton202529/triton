"""
Shared lifecycle truth validation, gating, and desk-level summaries.

Used by build_effective_lifecycle, build_trade_opportunities, execute_trades,
manage_positions, and apply_signal_lifecycle (self-check).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

SIGNALS_RATIONALE = RESULTS_DIR / "signals_with_rationale.csv"
SIGNALS_FALLBACK = RESULTS_DIR / "signals.csv"
DEFAULT_LIFECYCLE = RESULTS_DIR / "signal_lifecycle.csv"
DEFAULT_EFFECTIVE = RESULTS_DIR / "signal_lifecycle_effective.csv"
DEFAULT_POSITIONS_SNAPSHOT = RESULTS_DIR / "positions_snapshot.csv"

KNOWN_STANCES = frozenset({"BUY", "WAIT", "HOLD", "ADD", "EXIT", "TRIM"})
KNOWN_POSITION = frozenset({"FLAT", "LONG"})

# Authoritative effective-layer pairs (LONG+BUY is invalid — upstream must emit ADD).
VALID_EFFECTIVE_PAIRS: frozenset[Tuple[str, str]] = frozenset(
    {
        ("FLAT", "BUY"),
        ("FLAT", "WAIT"),
        ("FLAT", "HOLD"),
        ("LONG", "HOLD"),
        ("LONG", "ADD"),
        ("LONG", "EXIT"),
        ("LONG", "TRIM"),
        ("LONG", "WAIT"),
    }
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper().replace(".", "-")


def validate_effective_pair(position_state: str, stance: str) -> Tuple[bool, str]:
    p = str(position_state or "").strip().upper()
    s = str(stance or "").strip().upper()
    if p not in KNOWN_POSITION:
        return False, f"unknown_position_state:{p}"
    if s not in KNOWN_STANCES:
        return False, f"unknown_stance:{s}"
    if (p, s) in VALID_EFFECTIVE_PAIRS:
        return True, ""
    if p == "LONG" and s == "BUY":
        return False, "long_buy_invalid_use_add_or_hold"
    return False, f"invalid_pair:{p}+{s}"


def validate_raw_lifecycle_pair(position_state: str, lifecycle_action: str) -> Tuple[bool, str]:
    """Validate signal_lifecycle.csv-style rows (position_state + lifecycle_action)."""
    return validate_effective_pair(position_state, lifecycle_action)


def _read_csv_raw(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def find_duplicate_tickers(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty or "ticker" not in df.columns:
        return []
    t = df["ticker"].astype(str).str.strip().str.upper()
    dup = t[t.duplicated(keep=False)]
    return sorted(dup.unique().tolist())


def _file_mtime(path: Path) -> Optional[float]:
    try:
        if path.is_file() and path.stat().st_size > 0:
            return path.stat().st_mtime
    except Exception:
        pass
    return None


def check_pipeline_file_staleness(
    *,
    effective_path: Path = DEFAULT_EFFECTIVE,
    base_lifecycle_path: Path = DEFAULT_LIFECYCLE,
    signals_paths: Optional[Sequence[Path]] = None,
) -> Tuple[bool, str, str]:
    """
    Returns (is_stale, reason_code, detail).
    Stale = inputs newer than derived artifacts (must rebuild lifecycle / effective).
    """
    sp = list(signals_paths or (SIGNALS_RATIONALE, SIGNALS_FALLBACK))
    sig_mtime: Optional[float] = None
    sig_used: Optional[Path] = None
    for p in sp:
        m = _file_mtime(p)
        if m is not None and (sig_mtime is None or m > sig_mtime):
            sig_mtime = m
            sig_used = p

    lc_m = _file_mtime(base_lifecycle_path)
    eff_m = _file_mtime(effective_path)

    if sig_mtime is not None and lc_m is not None and sig_mtime > lc_m + 0.5:
        return (
            True,
            "STALE_LIFECYCLE",
            f"signals_newer_than_signal_lifecycle; signals={sig_used.name if sig_used else ''}",
        )
    if lc_m is not None and eff_m is not None and lc_m > eff_m + 0.5:
        return (
            True,
            "STALE_EFFECTIVE",
            "signal_lifecycle_newer_than_signal_lifecycle_effective",
        )
    return False, "", ""


def check_row_date_staleness(df: pd.DataFrame) -> Tuple[bool, str, str]:
    """If signal_date / as_of_date exist, block when any signal_date > as_of_date."""
    if df is None or df.empty:
        return False, "", ""
    sig_col = None
    for c in ("signal_date", "date", "signal_as_of"):
        if c in df.columns:
            sig_col = c
            break
    asof_col = None
    for c in ("as_of_date", "lifecycle_as_of_date", "asof_date"):
        if c in df.columns:
            asof_col = c
            break
    if not sig_col or not asof_col:
        return False, "", ""
    sig = pd.to_datetime(df[sig_col], errors="coerce")
    asof = pd.to_datetime(df[asof_col], errors="coerce")
    bad = sig.notna() & asof.notna() & (sig.dt.normalize() > asof.dt.normalize())
    if bad.any():
        n = int(bad.sum())
        return (
            True,
            "STALE_ROW_DATES",
            f"signal_date>as_of_date on {n} row(s); columns {sig_col}>{asof_col}",
        )
    return False, "", ""


def enrich_execution_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row execution_blocked / execution_block_reason from (effective_position_state, effective_stance).

    Preserves prior lifecycle_consistency from build_effective_lifecycle when the pair is valid
    (RECONCILED_BROKER / ADJUSTED_STANCE); on invalid pairs sets INCONSISTENT and blocks execution.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "effective_position_state" not in out.columns or "effective_stance" not in out.columns:
        return out

    lc_cons: List[str] = []
    blocked: List[bool] = []
    reasons: List[str] = []
    for i in range(len(out)):
        row = out.iloc[i]
        ok, reason = validate_effective_pair(
            str(row.get("effective_position_state", "")),
            str(row.get("effective_stance", "")),
        )
        if not ok:
            lc_cons.append("INCONSISTENT")
            blocked.append(True)
            reasons.append(reason)
            continue
        blocked.append(False)
        reasons.append("")
        prior = str(row.get("lifecycle_consistency", "") or "").strip()
        lc_cons.append(prior if prior else "OK")
    out["lifecycle_consistency"] = lc_cons
    out["execution_blocked"] = blocked
    out["execution_block_reason"] = reasons
    return out


@dataclass
class LifecycleGateResult:
    status: str  # OK | BLOCKED
    reason: str
    details: str
    tickers: List[str] = field(default_factory=list)
    checked_at: str = field(default_factory=_utc_iso)
    row_issue_count: int = 0
    duplicate_tickers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_block(self) -> str:
        lines = [
            "[LIFECYCLE_GATE]",
            f"status={self.status}",
            f"reason={self.reason or 'n/a'}",
            f"details={self.details or 'n/a'}",
            f"checked_at={self.checked_at}",
        ]
        if self.tickers:
            lines.append(f"tickers={','.join(self.tickers[:200])}")
        if self.duplicate_tickers:
            lines.append(f"duplicate_tickers={','.join(self.duplicate_tickers[:50])}")
        if self.row_issue_count:
            lines.append(f"row_issue_count={self.row_issue_count}")
        return "\n".join(lines)


def evaluate_lifecycle_gate(
    *,
    path: Optional[Path] = None,
    effective_path: Optional[Path] = None,
    base_lifecycle_path: Path = DEFAULT_LIFECYCLE,
    signals_paths: Optional[Sequence[Path]] = None,
    require_effective_file: bool = True,
) -> LifecycleGateResult:
    """
    Hard gate for execution / opportunity builders. BLOCKED = do not trade from lifecycle artifacts.

    Pass either `path` (preferred) or `effective_path` (alias for backward compatibility).
    Validates signal_lifecycle_effective-style rows when effective_* columns exist; otherwise
    validates raw signal_lifecycle.csv (position_state + lifecycle_action/stance).
    """
    checked_at = _utc_iso()
    target = path or effective_path or DEFAULT_EFFECTIVE

    if require_effective_file and (not target.is_file() or target.stat().st_size == 0):
        return LifecycleGateResult(
            status="BLOCKED",
            reason="MISSING_LIFECYCLE_FILE",
            details=f"file missing or empty: {target}",
            checked_at=checked_at,
        )

    eff_for_stale = DEFAULT_EFFECTIVE if DEFAULT_EFFECTIVE.is_file() else target
    stale, st_reason, st_detail = check_pipeline_file_staleness(
        effective_path=eff_for_stale,
        base_lifecycle_path=base_lifecycle_path,
        signals_paths=signals_paths,
    )
    if stale:
        return LifecycleGateResult(
            status="BLOCKED",
            reason=st_reason,
            details=st_detail,
            checked_at=checked_at,
        )

    raw = _read_csv_raw(target)
    dups = find_duplicate_tickers(raw)
    if dups:
        return LifecycleGateResult(
            status="BLOCKED",
            reason="DUPLICATE_TICKERS",
            details="duplicate ticker rows in lifecycle file",
            checked_at=checked_at,
            duplicate_tickers=dups,
        )

    if raw.empty:
        return LifecycleGateResult(
            status="OK",
            reason="EMPTY",
            details="lifecycle file has zero rows",
            tickers=[],
            checked_at=checked_at,
        )

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "ticker" not in df.columns:
        return LifecycleGateResult(
            status="BLOCKED",
            reason="MISSING_COLUMNS",
            details="required column ticker missing",
            checked_at=checked_at,
        )

    df = df.drop_duplicates(subset=["ticker"], keep="last")

    effective_mode = "effective_position_state" in df.columns and "effective_stance" in df.columns
    row_bad = 0
    if effective_mode:
        for _, row in df.iterrows():
            ok, _ = validate_effective_pair(
                str(row.get("effective_position_state", "")),
                str(row.get("effective_stance", "")),
            )
            if not ok:
                row_bad += 1
    else:
        if "position_state" not in df.columns:
            return LifecycleGateResult(
                status="BLOCKED",
                reason="MISSING_COLUMNS",
                details="position_state missing (raw lifecycle)",
                checked_at=checked_at,
            )
        action_col = (
            "lifecycle_action"
            if "lifecycle_action" in df.columns
            else ("stance" if "stance" in df.columns else None)
        )
        if not action_col:
            return LifecycleGateResult(
                status="BLOCKED",
                reason="MISSING_COLUMNS",
                details="lifecycle_action or stance missing (raw lifecycle)",
                checked_at=checked_at,
            )
        for _, row in df.iterrows():
            ok, _ = validate_raw_lifecycle_pair(
                str(row.get("position_state", "")),
                str(row.get(action_col, "")),
            )
            if not ok:
                row_bad += 1

    if row_bad:
        return LifecycleGateResult(
            status="BLOCKED",
            reason="INVALID_STATE_ACTION",
            details=f"{row_bad} row(s) fail state/action validation",
            tickers=sorted(df["ticker"].astype(str).str.upper().unique().tolist()),
            checked_at=checked_at,
            row_issue_count=row_bad,
        )

    r_stale, r_reason, r_detail = check_row_date_staleness(df)
    if r_stale:
        return LifecycleGateResult(
            status="BLOCKED",
            reason=r_reason,
            details=r_detail,
            tickers=sorted(df["ticker"].astype(str).str.upper().unique().tolist()),
            checked_at=checked_at,
        )

    tickers = sorted(df["ticker"].astype(str).str.upper().unique().tolist())
    return LifecycleGateResult(
        status="OK",
        reason="OK",
        details="lifecycle passed validation",
        tickers=tickers,
        checked_at=checked_at,
    )


def print_lifecycle_summary_from_effective(
    df: pd.DataFrame, *, prefix: str = "[LIFECYCLE_SUMMARY]"
) -> None:
    """Desk-level counts from an effective lifecycle dataframe."""
    if df is None or df.empty:
        print(f"{prefix} rows=0")
        return
    pos = (
        df["effective_position_state"].fillna("").astype(str).str.strip().str.upper()
        if "effective_position_state" in df.columns
        else pd.Series([], dtype=object)
    )
    st = (
        df["effective_stance"].fillna("").astype(str).str.strip().str.upper()
        if "effective_stance" in df.columns
        else pd.Series([], dtype=object)
    )
    inv = 0
    if "effective_position_state" in df.columns and "effective_stance" in df.columns:
        for i in range(len(df)):
            ok, _ = validate_effective_pair(str(pos.iloc[i]), str(st.iloc[i]))
            if not ok:
                inv += 1
    healed = int(df["healed"].sum()) if "healed" in df.columns else 0
    recon = int(df["reconciled_with_broker"].sum()) if "reconciled_with_broker" in df.columns else 0
    print(
        f"{prefix} tickers={len(df)} long={(pos == 'LONG').sum()} flat={(pos == 'FLAT').sum()} "
        f"buy={(st == 'BUY').sum()} add={(st == 'ADD').sum()} hold={(st == 'HOLD').sum()} "
        f"exit={(st == 'EXIT').sum()} wait={(st == 'WAIT').sum()} trim={(st == 'TRIM').sum()} "
        f"invalid={inv} reconciled={recon} healed={healed}"
    )


def print_effective_summary_stats(df: pd.DataFrame) -> None:
    """[EFFECTIVE_LIFECYCLE_SUMMARY] — healed = broker reconciliation only (see build_effective_lifecycle)."""
    if df is None or df.empty:
        print(
            "[EFFECTIVE_LIFECYCLE_SUMMARY] rows=0 healed=0 stance_adjustments=0 rare_exceptions_only=true"
        )
        return
    n = len(df)
    healed = int(df["healed"].sum()) if "healed" in df.columns else 0
    adj = 0
    if "stance_adjustment" in df.columns:
        adj = int(df["stance_adjustment"].fillna("").astype(str).str.strip().ne("").sum())
    rare = healed <= max(1, n // 50) if n else True
    print(
        f"[EFFECTIVE_LIFECYCLE_SUMMARY] rows={n} healed_broker_reconciliation={healed} "
        f"stance_adjustments={adj} rare_exceptions_only={str(rare).lower()}"
    )


def self_check_signal_lifecycle_csv(path: Path) -> Tuple[int, List[str]]:
    """
    PASS/FAIL for raw signal_lifecycle.csv (one row per ticker expected).
    Returns (exit_code, messages).
    """
    msgs: List[str] = []
    if not path.is_file() or path.stat().st_size == 0:
        return 2, [f"FAIL: missing or empty {path}"]
    raw = _read_csv_raw(path)
    dups = find_duplicate_tickers(raw)
    if dups:
        msgs.append(f"FAIL: duplicate tickers: {dups[:20]}")
    req = {"ticker", "position_state"}
    cols = {str(c).strip() for c in raw.columns}
    if not req.issubset(cols):
        msgs.append(f"FAIL: missing columns need {req} have {cols}")
    if raw.empty:
        msgs.append("WARN: empty dataframe")
        return (0 if not msgs else 1), msgs
    action_col = "lifecycle_action" if "lifecycle_action" in raw.columns else "stance"
    if action_col not in raw.columns:
        msgs.append("FAIL: missing lifecycle_action or stance")
        return 1, msgs
    bad = 0
    for _, row in raw.iterrows():
        ok, _ = validate_raw_lifecycle_pair(
            str(row.get("position_state", "")), str(row.get(action_col, ""))
        )
        if not ok:
            bad += 1
    if bad:
        msgs.append(f"FAIL: {bad} invalid position_state/action rows")
    if msgs:
        return 1, msgs
    return 0, ["PASS: signal_lifecycle self-check"]


def print_self_check_result(path: Path, *, label: str = "[LIFECYCLE_SELF_CHECK]") -> int:
    code, msgs = self_check_signal_lifecycle_csv(path)
    status = "PASS" if code == 0 else "FAIL"
    print(f"{label} status={status}")
    for m in msgs:
        print(f"{label} {m}")
    return code


def summarize_opportunity_build(
    *,
    entry: int,
    add: int,
    exit_n: int,
    trim: int,
    suppressed: int,
    blocked_due_to_lifecycle: int,
) -> None:
    print(
        f"[OPPORTUNITY_SUMMARY] entry={entry} add={add} exit={exit_n} trim={trim} "
        f"suppressed={suppressed} blocked_due_to_lifecycle={blocked_due_to_lifecycle}"
    )
