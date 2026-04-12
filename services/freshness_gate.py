# services/freshness_gate.py
# ------------------------------------------------------------
# TRITON — Freshness Clock + Contract Gate (Phase 1.5+)
#
# What it does:
#   - Runs in-process contracts validation (no shell calls)
#   - Computes file freshness (last modified age)
#   - Renders a top-of-dashboard "Freshness Clock" with colors
#   - Returns a gate verdict you can use to block Auto / execution UI
#
# Dependencies:
#   - services/contracts_registry.py  (get_contracts)
#   - services/data_contracts.py      (validate_all)
#
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from services.contracts_registry import get_contracts
from services.data_contracts import validate_all


@dataclass
class GateVerdict:
    ok: bool
    level: str  # "GREEN" | "YELLOW" | "RED"
    reason: str
    age_seconds: Optional[int]
    contracts_passed: bool
    contracts_summary: Dict[str, Any]
    file_status: List[Dict[str, Any]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _age_seconds(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        ts = path.stat().st_mtime
        age = int((_utc_now().timestamp() - ts))
        return max(age, 0)
    except Exception:
        return None


def _fmt_age(age_s: Optional[int]) -> str:
    if age_s is None:
        return "missing"
    if age_s < 60:
        return f"{age_s}s"
    if age_s < 3600:
        return f"{age_s // 60}m"
    return f"{age_s // 3600}h"


def _pick_level(age_s: Optional[int], *, green_s: int, yellow_s: int) -> str:
    if age_s is None:
        return "RED"
    if age_s <= green_s:
        return "GREEN"
    if age_s <= yellow_s:
        return "YELLOW"
    return "RED"


def run_contracts(project_root: Path) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Runs validate_all() using the registry contracts.

    Returns:
      - passed_all (bool)
      - summary dict
      - normalized results list (dicts)
    """
    contracts = get_contracts(project_root)
    results = validate_all(project_root, contracts)

    # validate_all() in your repo returns dict-like results (you confirmed via --debug),
    # but we still normalize defensively.
    norm: List[Dict[str, Any]] = []
    passed = True

    for r in results:
        if isinstance(r, dict):
            d = dict(r)
        else:
            d = {}
            for k in (
                "ok",
                "name",
                "path",
                "fmt",
                "rows",
                "missing_columns",
                "date_range",
                "errors",
            ):
                if hasattr(r, k):
                    d[k] = getattr(r, k)
        norm.append(d)
        if not bool(d.get("ok", False)):
            passed = False

    summary = {
        "total": len(norm),
        "passed": sum(1 for x in norm if x.get("ok") is True),
        "failed": sum(1 for x in norm if x.get("ok") is False),
    }
    return passed, summary, norm


def compute_gate(
    project_root: Path,
    *,
    critical_files: Optional[List[Tuple[str, Path]]] = None,
    green_seconds: int = 15 * 60,  # 15 minutes
    yellow_seconds: int = 90 * 60,  # 90 minutes
    require_contracts: bool = True,
) -> GateVerdict:
    """
    Gate = contracts must pass (if require_contracts) AND data must be fresh enough.

    Freshness uses the *max age* among critical files (worst/oldest determines the clock).
    """
    if critical_files is None:
        critical_files = [
            ("Portfolio History", Path("data/results/portfolio_history.csv")),
            ("Positions Snapshot", Path("data/results/positions_snapshot.csv")),
            ("Signals (preferred)", Path("data/results/signals_with_rationale.csv")),
            ("Signals (fallback)", Path("data/results/signals.csv")),
        ]

    # Contracts
    contracts_passed, contracts_summary, contract_results = run_contracts(project_root)

    # Files
    status: List[Dict[str, Any]] = []
    ages: List[int] = []

    for label, rel in critical_files:
        p = (project_root / rel).resolve()
        a = _age_seconds(p)
        lvl = _pick_level(a, green_s=green_seconds, yellow_s=yellow_seconds)
        status.append(
            {
                "label": label,
                "path": str(p),
                "exists": p.exists(),
                "age_seconds": a,
                "age": _fmt_age(a),
                "level": lvl,
                "mtime_utc": (
                    datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    if p.exists()
                    else None
                ),
            }
        )
        if a is not None:
            ages.append(a)

    worst_age = max(ages) if ages else None
    freshness_level = _pick_level(worst_age, green_s=green_seconds, yellow_s=yellow_seconds)

    # Final decision
    if require_contracts and not contracts_passed:
        return GateVerdict(
            ok=False,
            level="RED",
            reason="Contracts FAILED",
            age_seconds=worst_age,
            contracts_passed=False,
            contracts_summary={"contracts": contracts_summary, "results": contract_results},
            file_status=status,
        )

    if freshness_level == "RED":
        return GateVerdict(
            ok=False,
            level="RED",
            reason="Critical data is STALE or missing",
            age_seconds=worst_age,
            contracts_passed=contracts_passed,
            contracts_summary={"contracts": contracts_summary, "results": contract_results},
            file_status=status,
        )

    if freshness_level == "YELLOW":
        return GateVerdict(
            ok=True,
            level="YELLOW",
            reason="Data is aging (watch freshness)",
            age_seconds=worst_age,
            contracts_passed=contracts_passed,
            contracts_summary={"contracts": contracts_summary, "results": contract_results},
            file_status=status,
        )

    return GateVerdict(
        ok=True,
        level="GREEN",
        reason="Fresh & contracts OK",
        age_seconds=worst_age,
        contracts_passed=contracts_passed,
        contracts_summary={"contracts": contracts_summary, "results": contract_results},
        file_status=status,
    )


def render_freshness_clock(verdict: GateVerdict) -> None:
    """
    Renders a compact header clock + details expander.
    """
    level = verdict.level
    badge = "🟢" if level == "GREEN" else ("🟡" if level == "YELLOW" else "🔴")
    age = _fmt_age(verdict.age_seconds)

    left, mid, right = st.columns([1.2, 2.2, 1.6])

    with left:
        st.markdown(f"### {badge} Freshness: **{level}**")
        st.caption(f"Worst age: **{age}**")

    with mid:
        st.markdown("### ✅ Contracts")
        if verdict.contracts_passed:
            st.success("PASS")
        else:
            st.error("FAIL")
        st.caption(verdict.reason)

    with right:
        st.markdown("### ⏱️ Gate")
        if verdict.ok and verdict.level in ("GREEN", "YELLOW"):
            st.success("EXECUTION OK (UI gated by level)")
        else:
            st.error("EXECUTION BLOCKED")

    with st.expander("Freshness details (critical files + contracts)"):
        # Files
        st.subheader("Critical files")
        st.dataframe(
            [
                {
                    "label": x["label"],
                    "level": x["level"],
                    "age": x["age"],
                    "exists": x["exists"],
                    "mtime_utc": x["mtime_utc"],
                    "path": x["path"],
                }
                for x in verdict.file_status
            ],
            use_container_width=True,
        )

        # Contracts summary
        st.subheader("Contracts summary")
        st.json(verdict.contracts_summary.get("contracts", {}))

        # Show failures if any
        results = verdict.contracts_summary.get("results", [])
        bad = [r for r in results if isinstance(r, dict) and not r.get("ok", True)]
        if bad:
            st.subheader("Contract failures")
            st.json(bad)
        else:
            st.caption("No contract failures.")
