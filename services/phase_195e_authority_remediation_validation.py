"""
phase_195e_authority_remediation_validation.py
Phase 195E — Validate authority conflict remediation for build_trade_opportunities.py.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "phase_195e_authority_remediation_validation.json"
OUTPUT_CSV = RESULTS / "phase_195e_authority_remediation_scorecard.csv"

REMEDIATED_FILE = "services/build_trade_opportunities.py"
REMEDIATED_FUNCTION = "_parse_hard_risk_flags"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _inspect_remediated_module() -> Dict[str, Any]:
    path = ROOT / REMEDIATED_FILE.replace("/", os.sep)
    text = path.read_text(encoding="utf-8", errors="ignore")
    fn_block = ""
    m = re.search(
        rf"def {REMEDIATED_FUNCTION}\([\s\S]*?(?=\ndef |\Z)",
        text,
    )
    if m:
        fn_block = m.group(0)

    uses_loader = "load_risk_execution_controls" in fn_block
    adaptive_in_fn = bool(re.search(r"adaptive_risk_state\.json", fn_block))
    adaptive_authoritative = (
        adaptive_in_fn and "load_risk_execution_controls" not in fn_block
    )

    return {
        "uses_loader": uses_loader,
        "adaptive_in_function": adaptive_in_fn,
        "adaptive_risk_state_authoritative": adaptive_authoritative,
        "master_gate_used": uses_loader,
        "new_authority_source": "data/results/master_execution_gate.json via decision_authority.load_risk_execution_controls()",
        "previous_authority_source": "data/results/adaptive_risk_state.json (direct read in _parse_hard_risk_flags)",
    }


def _run_cmd(args: List[str]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "cmd": " ".join(args),
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stderr_tail": (proc.stderr or "")[-300:],
        }
    except Exception as e:
        return {"cmd": " ".join(args), "exit_code": -1, "ok": False, "error": str(e)}


def run_phase_195e_validation(*, verbose: bool = True) -> Dict[str, Any]:
    before_doc = _load_json(
        RESULTS / "phase_195d_authority_consolidation_validation.json"
    )
    high_before = int(before_doc.get("high_severity_count") or 0)

    inspection = _inspect_remediated_module()

    steps = [
        [sys.executable, "-m", "services.build_trade_opportunities"],
        [
            sys.executable,
            "-m",
            "services.phase_195d_authority_consolidation_validation",
        ],
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        [sys.executable, "-m", "services.portfolio_allocation_engine"],
        [sys.executable, "-m", "services.path_verification"],
        [sys.executable, "-m", "services.phase_195c_evaluation_contract_validation"],
    ]
    step_results = [_run_cmd(c) for c in steps]

    after_doc = _load_json(
        RESULTS / "phase_195d_authority_consolidation_validation.json"
    )
    high_after = int(after_doc.get("high_severity_count") or 0)
    p195d_pass = bool(after_doc.get("phase_pass"))
    p195c = _load_json(RESULTS / "phase_195c_evaluation_contract_validation.json")
    auth = _load_json(RESULTS / "governance_authorization.json")

    cert_status = _load_json(RESULTS / "capital_preservation_certification.json").get(
        "certification_status"
    )
    cert_ok = cert_status == "CERTIFIED_FOR_PAPER_PROTECTION"
    broker_ok = auth.get("live_execution_permitted") is False
    execution_ok = all(s.get("ok") for s in step_results)

    phase_pass = (
        inspection["uses_loader"]
        and not inspection["adaptive_risk_state_authoritative"]
        and high_after == 0
        and p195d_pass
        and cert_ok
        and broker_ok
        and execution_ok
    )

    checks = [
        {
            "check": "loader_wired",
            "status": "PASS" if inspection["uses_loader"] else "FAIL",
            "detail": inspection["new_authority_source"],
        },
        {
            "check": "adaptive_not_authoritative_in_function",
            "status": (
                "PASS"
                if not inspection["adaptive_risk_state_authoritative"]
                else "FAIL"
            ),
            "detail": f"adaptive_in_fn={inspection['adaptive_in_function']}",
        },
        {
            "check": "high_conflicts_after",
            "status": "PASS" if high_after == 0 else "FAIL",
            "detail": f"before={high_before} after={high_after}",
        },
        {
            "check": "phase_195d_pass",
            "status": "PASS" if p195d_pass else "FAIL",
            "detail": str(after_doc.get("phase_pass")),
        },
        {
            "check": "phase_195c_certification",
            "status": "PASS" if cert_ok else "FAIL",
            "detail": str(cert_status),
        },
        {
            "check": "broker_state",
            "status": "PASS" if broker_ok else "FAIL",
            "detail": json.dumps(
                {
                    "live": auth.get("live_execution_permitted"),
                    "paper": auth.get("paper_execution_permitted"),
                }
            ),
        },
        {
            "check": "safety_chain",
            "status": "PASS" if execution_ok else "FAIL",
            "detail": str([s.get("ok") for s in step_results]),
        },
    ]

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "Phase-195E",
        "remediated_file": REMEDIATED_FILE,
        "remediated_function": REMEDIATED_FUNCTION,
        "previous_authority_source": inspection["previous_authority_source"],
        "new_authority_source": inspection["new_authority_source"],
        "master_gate_used": inspection["master_gate_used"],
        "adaptive_risk_state_authoritative": inspection[
            "adaptive_risk_state_authoritative"
        ],
        "high_conflicts_before": high_before,
        "high_conflicts_after": high_after,
        "phase_195d_pass_after_remediation": p195d_pass,
        "broker_state_unchanged": broker_ok,
        "execution_safety_unchanged": broker_ok and cert_ok,
        "phase_195c_certification_unchanged": cert_ok and bool(p195c.get("phase_pass")),
        "phase_pass": phase_pass,
        "inspection": inspection,
        "validation_steps": step_results,
        "checks": checks,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(OUTPUT_CSV, checks, ("check", "status", "detail"))

    if verbose:
        print(f"Phase 195E validation: {OUTPUT_JSON}")
        print(
            f"HIGH {high_before}->{high_after} phase_195d_pass={p195d_pass} phase_pass={phase_pass}"
        )
        for c in checks:
            print(f"  {c['check']}: {c['status']}")

    return payload


if __name__ == "__main__":
    run_phase_195e_validation()
