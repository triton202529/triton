"""
TRITON Phase 165B — P&L Reconciliation Audit (read-only).

Combines the dashboard P&L ledger reconciliation with the broker history
recovery integrity report into persistent audit artifacts:

    data/results/pnl_reconciliation_audit.json
    data/results/cash_flow_reconciliation.json

Audit only — no trading, execution, or lifecycle behavior is touched.

Run:
    python -m services.pnl_reconciliation_audit
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.pnl_ledger_builder import build_pnl_ledger

RESULTS = ROOT / "data" / "results"
OUT_AUDIT = RESULTS / "pnl_reconciliation_audit.json"
OUT_CASH = RESULTS / "cash_flow_reconciliation.json"
IN_INTEGRITY = RESULTS / "ledger_integrity_report.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "to_dict") and not isinstance(obj, (str, bytes)):
        try:
            return None  # DataFrames are intentionally excluded from JSON output
        except Exception:
            return None
    return obj


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    bundle = build_pnl_ledger(force=True)

    integrity: Dict[str, Any] = {}
    if IN_INTEGRITY.is_file():
        try:
            integrity = json.loads(IN_INTEGRITY.read_text(encoding="utf-8"))
        except Exception:
            integrity = {}

    recon = {k: v for k, v in bundle.reconciliation.items() if k != "realized_sources"}
    recon["realized_sources"] = [
        {"source": s.get("source"), "realized_pl": s.get("realized_pl")}
        for s in bundle.reconciliation.get("realized_sources", [])
    ]

    audit = _jsonable(
        {
            "generated_at": _now_iso(),
            "local_reconciliation": recon,
            "broker_integrity": (
                {
                    k: integrity.get(k)
                    for k in (
                        "ledger_integrity_score",
                        "status",
                        "identity_ok",
                        "residual_difference",
                        "previous_epoch_net_flow",
                        "fill_match",
                        "position_match",
                        "cash_match",
                        "pnl_match",
                        "trade_match",
                        "account_created",
                        "reasons",
                    )
                }
                if integrity
                else {}
            ),
            "notes": (
                "Local residual is measured against the local fill ledger; "
                "broker integrity is measured against recovered broker history. "
                "Run services.broker_history_recovery first for broker truth."
            ),
        }
    )
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    cash = _jsonable(
        {
            "generated_at": _now_iso(),
            "cash_flow_summary": bundle.cash_flow_summary,
            "broker_residual_difference": integrity.get("residual_difference"),
            "broker_identity_ok": integrity.get("identity_ok"),
        }
    )
    OUT_CASH.write_text(json.dumps(cash, indent=2, default=str), encoding="utf-8")

    print(f"[pnl_reconciliation_audit] wrote {OUT_AUDIT.name} and {OUT_CASH.name}")
    return audit


if __name__ == "__main__":
    result = run()
    print(json.dumps(result.get("broker_integrity", {}), indent=2, default=str))
