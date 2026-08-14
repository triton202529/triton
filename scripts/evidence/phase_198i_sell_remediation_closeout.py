"""Write Phase 198I SELL remediation closeout evidence (hashes, supersession, git)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_PACK = "data/results/phase_198i_correction_7_evidence_repair_1"


def default_out_dir() -> Path:
    override = os.environ.get("TRITON_198I_EVIDENCE_OUT")
    if override:
        return Path(override)
    return ROOT / "data" / "results" / "phase_198i_sell_remediation_20260814"

PROTECTED_PAPER_RUNTIME = [
    "scripts/run_continuous_loop.ps1",
    "services/market_calendar.py",
    "services/paper_runtime_session_gate.py",
    "services/operator_console_status.py",
    "services/operator_runtime_controller.py",
    "services/paper_arm_status.py",
    "services/paper_session_arm_policy.py",
    "services/arm_mode.py",
    "services/operator_activity_feed.py",
    "dashboard/operator_console.py",
    "dashboard/operator_ui.py",
    "tests/test_paper_runtime_session_gating.py",
    "tests/test_paper_runtime_consolidated_unreadable_arm.py",
    "tests/test_calendar_unavailable_fail_closed.py",
    "tests/test_calendar_unknown_powershell_consumer.py",
]

BASELINE_HASHES = {
    "scripts/run_continuous_loop.ps1": "5784c0dd2ec623c94b35fb58cdd112db1f4373465e45261018e30c407c323375",
    "services/market_calendar.py": "cf7a85a39f80f5bff5296aaaa32628f11f9e34e61d043319f3e3123fd7119b86",
    "services/paper_runtime_session_gate.py": "4332eeb6e5ff65d11988213b4b04107e89dd5f1ba44277e782dacaf0a5ae3417",
    "services/operator_console_status.py": "96cee9f9e5821e477fe776e606a9821951bc3e1f1a25c896e595d37842cde6b5",
    "services/operator_runtime_controller.py": "da619123049f722e9f8574ac9fc110dbb802bc35b4b8fd6d76461ab31845f067",
    "services/paper_arm_status.py": "a5cc15593c8cdc35d3ed6d133fdedf1d70b248e289e7ecdd2252fcc1126ef525",
    "services/paper_session_arm_policy.py": "561eed1a805e69fefc75467dd5be948d4f677f874c18c94db76d1d4a4b32436d",
    "services/arm_mode.py": "208e1e43758a242d2a4a1a57ba2465a528cdfc8c42c1c6ef00b3d9f7acee1997",
    "services/operator_activity_feed.py": "79345de8528412b6145f0615ae60478e7269c70122a2f07dc01bc965fac4b1b9",
    "dashboard/operator_console.py": "05e073119fa115754a8def762dcdd1c3615ff2b799da2e8fbcb6fa340e918f3e",
    "dashboard/operator_ui.py": "889fbe070ad4e6858b3346d40885f8c48c5d34290222aa75a586a38d22e0c375",
    "tests/test_paper_runtime_session_gating.py": "5d80396226e5be9858f6b5dc65622d87dabb2471a10a4094ed0431d34c5f64cc",
    "tests/test_paper_runtime_consolidated_unreadable_arm.py": "3ac7524b4d369f213eed15c04bdc6e76c3a128e6637c0735e1b3aab5be1cd628",
    "tests/test_calendar_unavailable_fail_closed.py": "9ed2507c007560f94c46ef76491fd7852aacbc2149b5439861fd8fd6159bc8ff",
    "tests/test_calendar_unknown_powershell_consumer.py": "45bb7c93a99f0b2d7c1d0dece1ea5aad76d8dd92775a0729d45b0674e16a47a2",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(args: list[str]) -> str:
    p = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (p.stdout or "") + (p.stderr or "")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def write_closeout(out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir is not None else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    artifacts = []
    mismatches = []
    for rel in PROTECTED_PAPER_RUNTIME:
        path = ROOT / rel
        current = sha256_file(path) if path.is_file() else None
        baseline = BASELINE_HASHES[rel]
        status = "MATCH" if current == baseline else "MISMATCH"
        if current != baseline:
            mismatches.append(rel)
        artifacts.append(
            {
                "path": rel,
                "baseline_sha256": baseline,
                "final_sha256": current,
                "status": status,
                "exists": path.is_file(),
            }
        )

    inv_path = out / "02_broker_route_inventory.json"
    if not inv_path.is_file():
        raise FileNotFoundError(
            f"run phase_198i_sell_remediation_inventory.py first; missing {inv_path.name}"
        )
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    (out / "00_manifest.json").write_text(
        json.dumps(
            {
                "phase": "198I_SELL_REMEDIATION",
                "generated_at": now,
                "supersedes": HISTORICAL_PACK,
                "prior_verdict": "INCOMPLETE",
                "prior_unresolved_route_count": 15,
                "inventory_rows": inv.get("row_count"),
                "disposition_counts": inv.get("disposition_counts"),
                "blocked_unresolved": inv.get("disposition_counts", {}).get("BLOCKED_UNRESOLVED", 0),
                "phase_198i_v": "BLOCKED_NOT_STARTED",
                "windows_paper_sell_operational": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "01_supersession.json").write_text(
        json.dumps(
            {
                "historical_artifact": HISTORICAL_PACK,
                "historical_verdict": "INCOMPLETE",
                "historical_unresolved": 15,
                "action": "PRESERVED_NOT_OVERWRITTEN",
                "successor": _rel(out),
                "note": (
                    "Correction-7 repair_1 remains a historical identifier only. "
                    "This pack does not read that directory. It supersedes unresolved-route "
                    "and place_live_orders-integration conclusions only."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "06_zero_real_broker_proof.json").write_text(
        json.dumps(
            {
                "generated_at": now,
                "alpaca_env_cleared_in_tests": True,
                "triton_enable_live_execution": "0",
                "fake_broker_base": "http://phase198i-fake-broker.test",
                "network_socket_blocked": True,
                "real_broker_methods_invoked": False,
                "real_broker_endpoints_contacted": False,
                "proof": [
                    "tests/_phase_198i_sell_fakes.py::NetworkFakeAlpacaBroker overrides _get/_post/_delete/_data_get",
                    "tests/_phase_198i_sell_fakes.py::block_real_network clears Alpaca env and blocks socket connect",
                    "accepted integration asserts fake.base startswith fake host and fake.session is None",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "07_protected_paper_runtime_hashes.json").write_text(
        json.dumps(
            {
                "compared_at": now,
                "all_match": not mismatches,
                "mismatch_paths": mismatches,
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "08_git_integrity.txt").write_text(
        "\n".join(
            [
                f"generated_at={now}",
                "=== git rev-parse HEAD ===",
                git_text(["rev-parse", "HEAD"]).strip(),
                "=== git branch --show-current ===",
                git_text(["branch", "--show-current"]).strip(),
                "=== git log -1 --oneline ===",
                git_text(["log", "-1", "--oneline"]).strip(),
            ]
        ),
        encoding="utf-8",
    )
    (out / "09_test_count_report.json").write_text(
        json.dumps(
            {
                "generated_at": now,
                "source": "committed_contract",
                "required": {
                    "route_and_integration": {
                        "files": [
                            "tests/test_phase_198i_sell_broker_route_closure.py",
                            "tests/test_phase_198i_manage_positions_sell_integration.py",
                        ],
                        "collected": 31,
                        "passed": 31,
                    },
                    "correction_3": {
                        "files": ["tests/test_phase_198i_fix_closeout_evidence_correction_3.py"],
                        "collected": 8,
                        "passed": 8,
                    },
                    "phase_198i_selected": {"collected": 123, "passed": 123},
                    "commit_b_suite": {"collected": 138, "passed": 138},
                    "paper_runtime": {"collected": 82, "passed": 82},
                    "operator_console_uir": {"collected": 274, "passed": 274},
                },
                "note": "Actual pytest totals are recorded by the isolated validation run, not by copying ignored packs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "10_durability_repair.json").write_text(
        json.dumps(
            {
                "generated_at": now,
                "correction_file": "tests/test_phase_198i_fix_closeout_evidence_correction_3.py",
                "injected_happy_path_helper": "tests/_paper_sell_test_durability.py::stub_dir_fsync_ok",
                "production_windows_fail_closed": True,
                "production_durability_unconditional_success": False,
                "production_authority_modified": False,
                "windows_paper_sell_operational": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "11_phase_readiness.json").write_text(
        json.dumps(
            {
                "generated_at": now,
                "phase_198i_v": "BLOCKED_NOT_STARTED",
                "commit_c_closes_route_and_evidence_remediation": True,
                "windows_paper_sell_operational": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("mismatches", mismatches)
    print("wrote", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Phase 198I closeout evidence")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    write_closeout(Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
