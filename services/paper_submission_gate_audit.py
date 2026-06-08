"""Read-only paper order submission gate audit (Phase 150A). Logs decisions; does not change behavior."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE_AUTH_PATH = RESULTS / "governance_authorization.json"
EXECUTION_READINESS_PATH = RESULTS / "execution_readiness.json"
POLICY_PATH = RESULTS / "protective_action_policy.json"
AUDIT_JSON = RESULTS / "paper_submission_gate_audit.json"
REPORT_MD = RESULTS / "paper_submission_gate_audit_report.md"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


@dataclass
class DryRunDecision:
    dry_run: bool
    source: str
    reason: str
    execute_flag: bool
    config_dry_run_default: Optional[bool] = None
    forcing_conditions: List[str] = field(default_factory=list)


@dataclass
class PaperExecutionGate:
    env_enabled: bool
    execution_authorized: bool
    governance_authorized: bool
    readiness: str
    overall_authorization: bool
    paper_execution_permitted: bool
    live_execution_permitted: bool
    final_submit_allowed: bool
    block_reasons: List[str] = field(default_factory=list)


def load_governance_context() -> Dict[str, Any]:
    gov = _read_json(GOVERNANCE_AUTH_PATH)
    ready = _read_json(EXECUTION_READINESS_PATH)
    policy = _read_json(POLICY_PATH)
    env_enabled = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() == "1"
    return {
        "env_enabled": env_enabled,
        "governance_authorized": gov.get("governance_authorized"),
        "execution_authorized": gov.get("execution_authorized"),
        "overall_authorization": gov.get("overall_authorization"),
        "paper_execution_permitted": gov.get("paper_execution_permitted"),
        "live_execution_permitted": gov.get("live_execution_permitted"),
        "readiness": str(ready.get("readiness_status") or ""),
        "readiness_checks_passing": ready.get("checks_passing_count"),
        "readiness_checks_total": ready.get("checks_total"),
        "policy_paper_execution_enabled": policy.get("paper_execution_enabled"),
        "policy_global_execution_enabled": policy.get("global_execution_enabled"),
        "policy_live_execution_enabled": policy.get("live_execution_enabled"),
        "gate_reasons": gov.get("gate_reasons") or {},
    }


def analyze_execute_trades_dry_run(
    *,
    execute_flag: bool,
    config: Optional[Dict[str, Any]] = None,
    argv: Optional[List[str]] = None,
) -> DryRunDecision:
    """Trace how execute_trades sets dry_run (report-only semantics)."""
    cfg = config or {}
    config_default = cfg.get("dry_run_default")
    forcing: List[str] = []

    if not execute_flag:
        forcing.append("cli_missing_execute_flag")
        reason = (
            "--execute not passed; execute_trades sets dry_run = not bool(args.execute) "
            "(default plan-only)"
        )
        source = "execute_trades.main:cli_args"
    else:
        reason = "--execute passed; dry_run=False at execute_trades layer"
        source = "execute_trades.main:cli_args"

    if config_default is True and not execute_flag:
        forcing.append("config_dry_run_default_true_unwired")
    if argv is not None and "--execute" not in argv:
        forcing.append("argv_has_no_execute")

    return DryRunDecision(
        dry_run=not bool(execute_flag),
        source=source,
        reason=reason,
        execute_flag=bool(execute_flag),
        config_dry_run_default=(bool(config_default) if config_default is not None else None),
        forcing_conditions=forcing or (["none"] if execute_flag else forcing),
    )


def evaluate_paper_execution_gate(
    *,
    dry_run: bool,
    mode: str = "paper",
    execute_flag: bool = False,
) -> PaperExecutionGate:
    ctx = load_governance_context()
    block_reasons: List[str] = []

    if mode != "paper":
        block_reasons.append("mode_not_paper")
    if dry_run:
        block_reasons.append("execute_trades_dry_run_true")
    if not execute_flag:
        block_reasons.append("cli_execute_flag_false")
    if not ctx["env_enabled"]:
        block_reasons.append("TRITON_ENABLE_PAPER_EXECUTION_not_1")
    if ctx.get("execution_authorized") is not True:
        block_reasons.append("execution_authorized_false")
    if ctx.get("governance_authorized") is not True:
        block_reasons.append("governance_authorized_false")
    if ctx.get("paper_execution_permitted") is not True:
        block_reasons.append("paper_execution_permitted_false")
    if ctx.get("live_execution_permitted") is True:
        block_reasons.append("live_execution_permitted_true")

    readiness = str(ctx.get("readiness") or "")
    if readiness and readiness not in ("READY", "SIMULATION_ONLY"):
        block_reasons.append(f"readiness_status_{readiness}")

    final_submit_allowed = len(block_reasons) == 0

    return PaperExecutionGate(
        env_enabled=bool(ctx["env_enabled"]),
        execution_authorized=ctx.get("execution_authorized") is True,
        governance_authorized=ctx.get("governance_authorized") is True,
        readiness=readiness,
        overall_authorization=ctx.get("overall_authorization") is True,
        paper_execution_permitted=ctx.get("paper_execution_permitted") is True,
        live_execution_permitted=ctx.get("live_execution_permitted") is True,
        final_submit_allowed=final_submit_allowed,
        block_reasons=block_reasons,
    )


def log_dry_run_decision(decision: DryRunDecision) -> None:
    print("[DRY_RUN_DECISION]")
    print(f"dry_run={_bool_str(decision.dry_run)}")
    print(f"source={decision.source}")
    print(f"reason={decision.reason}")
    if decision.config_dry_run_default is not None:
        print(f"config_dry_run_default={_bool_str(bool(decision.config_dry_run_default))}")
    if decision.forcing_conditions:
        print(f"forcing_conditions={','.join(decision.forcing_conditions)}")


def log_paper_execution_gate(gate: PaperExecutionGate) -> None:
    print("[PAPER_EXECUTION_GATE]")
    print(f"env_enabled={_bool_str(gate.env_enabled)}")
    print(f"execution_authorized={_bool_str(gate.execution_authorized)}")
    print(f"governance_authorized={_bool_str(gate.governance_authorized)}")
    print(f"readiness={gate.readiness or 'unknown'}")
    print(f"final_submit_allowed={_bool_str(gate.final_submit_allowed)}")
    if gate.block_reasons:
        print(f"block_reasons={';'.join(gate.block_reasons)}")


def all_dry_run_forcing_conditions() -> List[Dict[str, str]]:
    """Catalog every condition in TRITON that can force plan-only / no broker submit."""
    return [
        {
            "layer": "execute_trades",
            "condition": "cli_missing_execute_flag",
            "effect": "dry_run=True via dry_run = not bool(args.execute)",
            "notes": "Default CLI invocation has no --execute; plan-only is intentional default.",
        },
        {
            "layer": "execute_trades",
            "condition": "config_dry_run_default_true",
            "effect": "Documented default in load_execute_trades_config()",
            "notes": "Config key exists but is NOT read in main(); only --execute controls dry_run today.",
        },
        {
            "layer": "execute_trades",
            "condition": "maybe_execute_plan_dry_run",
            "effect": "Skips master gate + place_live_orders subprocess when dry_run=True",
            "notes": "Early return at maybe_execute_plan() line: if dry_run or not planned: return 0,0,[].",
        },
        {
            "layer": "execute_trades",
            "condition": "main_early_return_dry_run",
            "effect": "Returns before maybe_execute_plan when dry_run after planning",
            "notes": "if dry_run: emit drop payload and return 0 — no placement handoff.",
        },
        {
            "layer": "place_live_orders",
            "condition": "argparse_default_dry_run",
            "effect": "ap.set_defaults(dry_run=True); requires --no-dry-run to submit",
            "notes": "When called standalone without --no-dry-run, never calls broker.submit_order.",
        },
        {
            "layer": "place_live_orders",
            "condition": "execute_trades_subprocess_uses_no_dry_run",
            "effect": "Only when execute_trades dry_run=False; subprocess passes --no-dry-run",
            "notes": "Handoff in maybe_execute_plan cmd list.",
        },
        {
            "layer": "run_scheduled_paper_cycle",
            "condition": "ex_skip_or_arm_block",
            "effect": "execute_trades stage skipped or run without effective mutation",
            "notes": "ex_skip when ARM execute_trades permission false or run_execute_trades false.",
        },
        {
            "layer": "run_scheduled_paper_cycle",
            "condition": "paper_execution_env_blocked",
            "effect": "Sets paper_exec_env_ok=False; notes blocked; may skip execute intent",
            "notes": "TRITON_ENABLE_PAPER_EXECUTION must be 1.",
        },
        {
            "layer": "governance",
            "condition": "execution_authorized_false",
            "effect": "Master gate / readiness block at submit time (not dry_run flag)",
            "notes": "Does not set dry_run=True but blocks final_submit_allowed.",
        },
        {
            "layer": "governance",
            "condition": "master_execution_gate_block",
            "effect": "maybe_execute_plan / place_live_orders return rc=2",
            "notes": "When dry_run=False but gate fails, placement blocked after planning.",
        },
    ]


def write_audit_artifacts(
    *,
    decision: DryRunDecision,
    gate: PaperExecutionGate,
    mode: str,
    module: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "timestamp": _utc_iso(),
        "module": module,
        "mode": mode,
        "dry_run_decision": {
            "dry_run": decision.dry_run,
            "source": decision.source,
            "reason": decision.reason,
            "execute_flag": decision.execute_flag,
            "config_dry_run_default": decision.config_dry_run_default,
            "forcing_conditions": decision.forcing_conditions,
        },
        "paper_execution_gate": {
            "env_enabled": gate.env_enabled,
            "execution_authorized": gate.execution_authorized,
            "governance_authorized": gate.governance_authorized,
            "readiness": gate.readiness,
            "overall_authorization": gate.overall_authorization,
            "paper_execution_permitted": gate.paper_execution_permitted,
            "final_submit_allowed": gate.final_submit_allowed,
            "block_reasons": gate.block_reasons,
        },
        "dry_run_forcing_catalog": all_dry_run_forcing_conditions(),
        "governance_context": load_governance_context(),
    }
    if extra:
        payload["extra"] = extra
    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# TRITON Phase 150A — Paper Order Submission Gate Audit",
        "",
        f"Generated: {payload['timestamp']}",
        "",
        "## DRY_RUN_DECISION",
        "",
        f"- **dry_run**: `{decision.dry_run}`",
        f"- **source**: `{decision.source}`",
        f"- **reason**: {decision.reason}",
        f"- **--execute flag**: `{decision.execute_flag}`",
        f"- **config dry_run_default** (unwired): `{decision.config_dry_run_default}`",
        "",
        "## PAPER_EXECUTION_GATE",
        "",
        f"- **env_enabled**: `{gate.env_enabled}`",
        f"- **execution_authorized**: `{gate.execution_authorized}`",
        f"- **governance_authorized**: `{gate.governance_authorized}`",
        f"- **readiness**: `{gate.readiness}`",
        f"- **final_submit_allowed**: `{gate.final_submit_allowed}`",
        "",
        "### Block reasons",
        "",
    ]
    if gate.block_reasons:
        for r in gate.block_reasons:
            lines.append(f"- `{r}`")
    else:
        lines.append("- _(none)_")
    lines.extend(
        [
            "",
            "## Why dry_run=True when running without --execute",
            "",
            "1. `execute_trades.main()` sets `dry_run = not bool(args.execute)` immediately after parsing CLI args.",
            "2. The `--execute` flag is **opt-in** (`action='store_true'`); default invocation is plan-only.",
            "3. `dry_run_default: True` in `load_execute_trades_config()` is **not** applied in `main()` — only the CLI flag matters.",
            "4. When `dry_run=True`, `main()` returns after planning and **never** calls `maybe_execute_plan()` for placement.",
            "5. Paper execution authorization (`TRITON_ENABLE_PAPER_EXECUTION`, governance JSON) affects **submit gates**, not the `dry_run` boolean.",
            "",
            "## All conditions capable of forcing dry_run=True / no submit",
            "",
        ]
    )
    for item in all_dry_run_forcing_conditions():
        lines.append(f"### {item['layer']} — `{item['condition']}`")
        lines.append(f"- **effect**: {item['effect']}")
        lines.append(f"- **notes**: {item['notes']}")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def audit_execute_trades_entry(
    *,
    execute_flag: bool,
    mode: str,
    config: Optional[Dict[str, Any]] = None,
    argv: Optional[List[str]] = None,
) -> tuple[DryRunDecision, PaperExecutionGate]:
    decision = analyze_execute_trades_dry_run(
        execute_flag=execute_flag,
        config=config,
        argv=argv,
    )
    gate = evaluate_paper_execution_gate(
        dry_run=decision.dry_run,
        mode=mode,
        execute_flag=execute_flag,
    )
    log_dry_run_decision(decision)
    log_paper_execution_gate(gate)
    write_audit_artifacts(
        decision=decision,
        gate=gate,
        mode=mode,
        module="execute_trades",
        extra={"argv": list(argv) if argv is not None else None},
    )
    return decision, gate
