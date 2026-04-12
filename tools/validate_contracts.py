# tools/validate_contracts.py
# ------------------------------------------------------------
# TRITON — Validate all data contracts (CLI runner)
#
# Robust output:
#   - Works whether validate_all() returns dicts or objects
#   - Prints FAIL blocks even when `issues` doesn't exist
#
# Exit codes:
#   0 = all passed
#   2 = one or more failed
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.contracts_registry import get_contracts  # noqa: E402
from services.data_contracts import validate_all  # noqa: E402


def _as_dict(r: Any) -> Dict[str, Any]:
    """Normalize a result (dict or object) into a dict for printing."""
    if isinstance(r, dict):
        return r

    out: Dict[str, Any] = {}
    for k in (
        "ok",
        "name",
        "path",
        "fmt",
        "rows",
        "issues",
        "missing_columns",
        "date_range",
        "errors",
    ):
        if hasattr(r, k):
            out[k] = getattr(r, k)
    # Some versions use different names
    if "name" not in out and hasattr(r, "contract_name"):
        out["name"] = getattr(r, "contract_name")
    if "path" not in out and hasattr(r, "file"):
        out["path"] = getattr(r, "file")
    return out


def _get_ok(d: Dict[str, Any]) -> bool:
    v = d.get("ok")
    if isinstance(v, bool):
        return v
    # Some validators might use "passed"
    v2 = d.get("passed")
    if isinstance(v2, bool):
        return v2
    return False  # default pessimistic


def _summarize(results: List[Any]) -> Dict[str, int]:
    total = len(results)
    failed = 0
    warn_issues = 0
    error_issues = 0

    for r in results:
        d = _as_dict(r)
        if not _get_ok(d):
            failed += 1

        issues = d.get("issues") or []
        if isinstance(issues, list):
            for i in issues:
                # issue can be dict or object
                lvl = ""
                if isinstance(i, dict):
                    lvl = str(i.get("level", "")).upper()
                else:
                    lvl = str(getattr(i, "level", "")).upper()

                if lvl == "WARN":
                    warn_issues += 1
                elif lvl == "ERROR":
                    error_issues += 1

    return {
        "total": total,
        "passed": total - failed,
        "failed": failed,
        "warn_issues": warn_issues,
        "error_issues": error_issues,
    }


def _print_fail(d: Dict[str, Any]) -> None:
    name = d.get("name") or "Unknown"
    path = d.get("path") or ""

    print(f"\nFAIL: {name}")
    if path:
        print(f"  {path}")

    # If your validator uses the newer style keys, print them
    missing = d.get("missing_columns") or d.get("missing") or []
    if missing:
        print(f"  ERROR [MISSING_COLUMNS] Missing required columns: {missing}")
        print("    hint: Fix upstream generator or update contract intentionally.")

    dr = d.get("date_range")
    if isinstance(dr, dict) and (dr.get("min_date") or dr.get("max_date")):
        print(f"  INFO  [DATE_RANGE] {dr.get('min_date')} -> {dr.get('max_date')}")

    # Print generic errors list if present
    errs = d.get("errors") or []
    if isinstance(errs, list) and errs:
        for e in errs:
            print(f"  ERROR [ERROR] {e}")

    # Print issues if present
    issues = d.get("issues") or []
    if isinstance(issues, list) and issues:
        for i in issues:
            if isinstance(i, dict):
                level = str(i.get("level", "")).upper()
                code = i.get("code", "")
                msg = i.get("message", "")
                hint = i.get("hint", "")
            else:
                level = str(getattr(i, "level", "")).upper()
                code = getattr(i, "code", "")
                msg = getattr(i, "message", "")
                hint = getattr(i, "hint", "")

            if level in ("ERROR", "WARN"):
                print(f"  {level} [{code}] {msg}")
                if hint:
                    print(f"    hint: {hint}")

    # If nothing else printed, show raw keys to help us patch fast
    printed_any = (
        bool(missing) or bool(dr) or bool(errs) or (isinstance(issues, list) and len(issues) > 0)
    )
    if not printed_any:
        keys = sorted(list(d.keys()))
        print(f"  NOTE: No detailed issues provided. Result keys: {keys}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="Print raw result dicts (first 2).")
    args = ap.parse_args()

    contracts = get_contracts(PROJECT_ROOT)
    results = validate_all(PROJECT_ROOT, contracts)

    s = _summarize(results)
    print("---- TRITON DATA CONTRACTS ----")
    print(s)

    if args.debug:
        print("\n[debug] first result objects (normalized):")
        for r in results[:2]:
            print(json.dumps(_as_dict(r), indent=2, default=str))

    for r in results:
        d = _as_dict(r)
        if not _get_ok(d):
            _print_fail(d)

    return 0 if s["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
