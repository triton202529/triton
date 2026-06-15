"""
risk_committee_oversight.py
CLI entrypoint for Phase 25 risk committee oversight artifact generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from services.institutional_protection import compute_risk_committee_oversight

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_risk_committee_oversight(*, results_dir: Path) -> Dict[str, Any]:
    rd = Path(results_dir)
    cert_doc = _read_json(rd / "capital_preservation_certification.json")
    doc = compute_risk_committee_oversight(
        cpi_doc=_read_json(rd / "capital_preservation_intelligence.json"),
        gov_doc=_read_json(rd / "governance_risk_summary.json"),
        auth_doc=_read_json(rd / "governance_authorization.json"),
        governor_doc=_read_json(rd / "capital_preservation_governor.json"),
        cpe_doc=_read_json(rd / "capital_preservation_escalation.json"),
        readiness_doc=_read_json(rd / "execution_readiness.json"),
        cert_doc=cert_doc,
    )
    out = rd / "risk_committee_oversight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(out)
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate risk committee oversight artifact")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    doc = run_risk_committee_oversight(results_dir=Path(args.results_dir))
    if args.verbose:
        print(
            f"committee_status={doc.get('committee_status')} "
            f"certification={doc.get('certification_status')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
