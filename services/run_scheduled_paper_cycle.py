# services/run_scheduled_paper_cycle.py
"""Safe scheduled PAPER trading cycle: pipeline → execute_trades → manage_positions → snapshots."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CONFIG_PATH = ROOT / "config" / "run_scheduled_paper_cycle.json"
SUMMARY_JSON = RESULTS / "paper_trade_cycle_summary.json"
LOG_CSV = RESULTS / "paper_trade_cycle_log.csv"
EXEC_DROP_JSON = RESULTS / "execution_drop_diagnostics.json"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_config() -> Dict[str, Any]:
    base = {
        "enabled": True,
        "mode": "paper",
        "run_pipeline": True,
        "run_execute_trades": True,
        "run_manage_positions": True,
        "manage_positions_execute": False,
        "refresh_snapshots_after_cycle": True,
        # Optional: snapshot broker open orders before pipeline (usually off).
        "refresh_snapshots_at_cycle_start": False,
        # Refresh open_orders_snapshot before manage_open_orders / reprice ladder (recommended when maintenance runs).
        "refresh_snapshots_before_open_order_maintenance": True,
        "run_manage_open_orders": True,
        "manage_open_orders_execute_cancel": False,
        "manage_open_orders_stale_minutes": 30.0,
        "run_reprice_order_ladder": True,
        "reprice_order_ladder_execute": False,
        "reprice_ladder_max_stage": 4,
        "reprice_ladder_stage1_minutes": 15.0,
        "stop_on_pipeline_failure": True,
        "stop_on_execute_failure": False,
        "verbose_pipeline": True,
        "verbose_subprocess": True,
        "reallocate_after_exit": False,
        # New: if entries are blocked by MAX_POSITIONS, allow manage_positions
        # to auto-switch into --execute in ASSISTED/AUTO (still ARM-gated).
        "auto_manage_execute_on_entry_block": True,
    }
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                base.update(u)
    except Exception:
        pass
    return base


def _python_exe() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


@dataclass
class CycleStageResult:
    ok: bool
    exit_code: int
    duration_sec: float
    blocked: bool
    message: str


@dataclass
class PaperTradeCycleSummary:
    timestamp: str
    mode: str
    ok: bool
    blocked: bool
    had_warnings: bool
    stages: Dict[str, Any]
    cycle_notes: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    arm_mode: str = "MANUAL"
    arm_permissions: Dict[str, Any] = field(default_factory=dict)
    arm_block_reasons: List[str] = field(default_factory=list)


def _stage_dict(res: CycleStageResult) -> Dict[str, Any]:
    return {
        "ok": res.ok,
        "exit_code": res.exit_code,
        "duration_sec": round(res.duration_sec, 3),
        "blocked": res.blocked,
        "message": res.message[:4000] if res.message else "",
    }


def run_cmd(
    cmd: List[str],
    *,
    verbose: bool,
    capture: bool = True,
) -> Tuple[int, str, float]:
    t0 = time.perf_counter()
    if verbose:
        print(f"\n{'='*60}\nCMD: {' '.join(cmd)}\n{'='*60}", flush=True)
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=capture,
            text=True,
        )
        dur = time.perf_counter() - t0
        out = ""
        if capture and p.stdout:
            out += p.stdout
        if capture and p.stderr:
            out += "\n" + p.stderr
        msg = (out or "").strip()
        if len(msg) > 8000:
            msg = msg[:4000] + "\n...[truncated]...\n" + msg[-4000:]
        return int(p.returncode), msg, dur
    except Exception as e:
        dur = time.perf_counter() - t0
        return 1, str(e), dur


def write_cycle_summary(summary: PaperTradeCycleSummary) -> None:
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        SUMMARY_JSON.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not write {SUMMARY_JSON}: {e}", flush=True)


_CYCLE_LOG_FIELDS = [
    "ts_utc",
    "ok",
    "blocked",
    "pipeline_ok",
    "execute_ok",
    "manage_ok",
    "manage_open_ok",
    "reprice_ladder_ok",
    "snapshot_ok",
    "manage_execute",
    "notes",
]


def append_cycle_log(row: Dict[str, Any]) -> None:
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        new_file = not LOG_CSV.is_file() or LOG_CSV.stat().st_size == 0
        out = {k: row.get(k, "") for k in _CYCLE_LOG_FIELDS}
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_CYCLE_LOG_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(out)
    except Exception as e:
        print(f"[WARN] Could not append {LOG_CSV}: {e}", flush=True)


def _empty_stage() -> CycleStageResult:
    return CycleStageResult(
        ok=True, exit_code=0, duration_sec=0.0, blocked=False, message="skipped"
    )


def _refresh_execution_pressure_best_effort() -> None:
    try:
        from services.execution_pressure_diagnostics import refresh_execution_pressure_diagnostics

        refresh_execution_pressure_diagnostics()
    except Exception:
        pass
    try:
        from services.session_fill_pressure import refresh_session_fill_pressure

        refresh_session_fill_pressure()
    except Exception:
        pass


def _should_auto_execute_manage_positions(
    *,
    cfg: Dict[str, Any],
    arm_perms: Dict[str, Any],
    ex_skip: bool,
    execute_result: CycleStageResult,
    manage_skip: bool,
    manage_exec_already: bool,
) -> bool:
    """Auto-enable manage_positions --execute when entries are blocked by a full book."""
    if manage_skip:
        return False
    if manage_exec_already:
        return False
    if ex_skip:
        return False
    if not bool(cfg.get("auto_manage_execute_on_entry_block", True)):
        return False
    if not bool(arm_perms.get("manage_positions_execute", False)):
        return False
    if execute_result.exit_code != 2:
        return False
    msg = (execute_result.message or "").upper()
    return "MAX_POSITIONS" in msg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON scheduled PAPER trade cycle (orchestrator only)"
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-execute-trades", action="store_true")
    ap.add_argument("--skip-manage-positions", action="store_true")
    ap.add_argument(
        "--skip-manage-open-orders", action="store_true", help="Skip manage_open_orders stage."
    )
    ap.add_argument(
        "--skip-reprice-ladder", action="store_true", help="Skip reprice_order_ladder stage."
    )
    ap.add_argument(
        "--manage-execute",
        action="store_true",
        help="Pass --execute to manage_positions (default plan-only).",
    )
    ap.add_argument("--no-snapshot-refresh", action="store_true")
    args = ap.parse_args(argv)

    cfg = _load_config()
    py = _python_exe()
    verbose = bool(args.verbose) or bool(cfg.get("verbose_subprocess", True))
    vp = bool(cfg.get("verbose_pipeline", True))

    try:
        from services.arm_mode import (
            append_arm_mode_log,
            load_arm_config,
            resolve_arm_mutation_permissions,
            validate_mode,
            write_arm_mode_status_snapshot,
        )

        arm_cfg = load_arm_config()
        arm_perms, arm_block_reasons = resolve_arm_mutation_permissions(arm_cfg)
        arm_mode_label = validate_mode(arm_cfg.get("mode"))
    except Exception as e:
        arm_cfg = {"enabled": False, "mode": "MANUAL"}
        arm_perms = {
            "execute_trades": False,
            "manage_positions_execute": False,
            "manage_open_orders_execute_cancel": False,
            "reprice_order_ladder_execute": False,
            "reallocate_after_exit": False,
        }
        arm_block_reasons = [f"ARM_INIT_ERROR:{e!s}"]
        arm_mode_label = "MANUAL"

    want_mg_exec = bool(args.manage_execute) or bool(cfg.get("manage_positions_execute", False))
    manage_exec = bool(want_mg_exec and arm_perms.get("manage_positions_execute", False))
    auto_manage_exec_triggered = False

    moo_skip = bool(args.skip_manage_open_orders) or (
        not bool(cfg.get("run_manage_open_orders", True))
    )
    rpl_skip = bool(args.skip_reprice_ladder) or (
        not bool(cfg.get("run_reprice_order_ladder", True))
    )
    want_moo_cancel = bool(cfg.get("manage_open_orders_execute_cancel", False))
    moo_exec_cancel = bool(
        want_moo_cancel and arm_perms.get("manage_open_orders_execute_cancel", False)
    )
    want_rpl_exec = bool(cfg.get("reprice_order_ladder_execute", False))
    rpl_exec = bool(want_rpl_exec and arm_perms.get("reprice_order_ladder_execute", False))
    rpl_max_stage = int(cfg.get("reprice_ladder_max_stage", 4) or 4)
    rpl_stage1_min = float(cfg.get("reprice_ladder_stage1_minutes", 15.0) or 15.0)
    moo_stale_min = float(cfg.get("manage_open_orders_stale_minutes", 30.0) or 30.0)

    maint_needed = (not moo_skip) or (not rpl_skip)
    start_snap_skip = (
        not bool(cfg.get("refresh_snapshots_at_cycle_start", False))
    ) or args.no_snapshot_refresh
    snap_maint_skip = (
        args.no_snapshot_refresh
        or (not bool(cfg.get("refresh_snapshots_before_open_order_maintenance", True)))
        or (not maint_needed)
    )

    notes: List[str] = []
    stages_out: Dict[str, Any] = {}
    blocked_any = False
    had_warnings = False

    # PAPER ONLY — this module never invokes live
    mode = "paper"
    if str(cfg.get("mode", "paper")).lower() != "paper":
        notes.append("config mode was not paper; forced paper")
    want_ex = bool(cfg.get("run_execute_trades", True)) and not args.skip_execute_trades
    ex_skip = (not want_ex) or (not arm_perms.get("execute_trades", False))
    mg_skip = args.skip_manage_positions or (not bool(cfg.get("run_manage_positions", True)))

    cfg_snapshot = {
        "manage_positions_execute": manage_exec,
        "want_manage_positions_execute": want_mg_exec,
        "auto_manage_execute_on_entry_block": bool(
            cfg.get("auto_manage_execute_on_entry_block", True)
        ),
        "auto_manage_execute_triggered": False,
        "run_manage_open_orders": not moo_skip,
        "manage_open_orders_execute_cancel": moo_exec_cancel,
        "want_manage_open_orders_execute_cancel": want_moo_cancel,
        "run_reprice_order_ladder": not rpl_skip,
        "reprice_order_ladder_execute": rpl_exec,
        "want_reprice_order_ladder_execute": want_rpl_exec,
        "reprice_ladder_max_stage": rpl_max_stage,
        "reprice_ladder_stage1_minutes": rpl_stage1_min,
        "reallocate_after_exit": bool(cfg.get("reallocate_after_exit", False)),
        "refresh_snapshots_at_cycle_start": bool(
            cfg.get("refresh_snapshots_at_cycle_start", False)
        ),
        "refresh_snapshots_before_open_order_maintenance": bool(
            cfg.get("refresh_snapshots_before_open_order_maintenance", True)
        ),
        "run_scheduled_paper_cycle_config": str(CONFIG_PATH),
        "arm_mode": arm_mode_label,
        "arm_block_reasons": list(arm_block_reasons),
        "arm_effective_permissions": dict(arm_perms),
    }

    if arm_block_reasons:
        notes.append("blocked_by_arm_mode:" + ",".join(arm_block_reasons))
    if want_ex and not arm_perms.get("execute_trades"):
        notes.append("execute_trades: mutation blocked by ARM policy")
    if want_mg_exec and not arm_perms.get("manage_positions_execute"):
        notes.append("manage_positions --execute blocked by ARM policy")
    if want_moo_cancel and not arm_perms.get("manage_open_orders_execute_cancel"):
        notes.append("manage_open_orders --execute-cancel blocked by ARM policy")
    if want_rpl_exec and not arm_perms.get("reprice_order_ladder_execute"):
        notes.append("reprice_order_ladder --execute blocked by ARM policy")

    def run_stage(
        cmd: List[str],
        *,
        skip: bool,
        treat_blocked: bool = True,
    ) -> CycleStageResult:
        nonlocal blocked_any
        if skip:
            r = _empty_stage()
            r.message = "skipped by flag"
            return r
        code, msg, dur = run_cmd(cmd, verbose=verbose)
        is_blocked = treat_blocked and code == 2
        is_ok = code == 0
        if is_blocked:
            blocked_any = True
        return CycleStageResult(
            ok=is_ok,
            exit_code=code,
            duration_sec=dur,
            blocked=is_blocked,
            message=msg or f"exit={code}",
        )

    snap_cmd = [py, "-m", "services.snapshot_live_orders", "--mode", "paper"]

    # --- Optional: snapshot at cycle start (broker open orders before pipeline) ---
    ss0 = run_stage(snap_cmd, skip=start_snap_skip, treat_blocked=False)
    stages_out["snapshot_start"] = _stage_dict(ss0)
    if not start_snap_skip and ss0.exit_code != 0:
        had_warnings = True
        notes.append(f"snapshot_start rc={ss0.exit_code}")

    # --- Stage 1: Pipeline ---
    pipeline_skip = args.skip_pipeline or (not bool(cfg.get("run_pipeline", True)))
    pipeline_cmd = [py, str(ROOT / "run_full_pipeline.py")]
    if vp:
        pipeline_cmd.append("--verbose")
    pr = run_stage(pipeline_cmd, skip=pipeline_skip, treat_blocked=False)
    stages_out["pipeline"] = _stage_dict(pr)
    if not pipeline_skip and cfg.get("stop_on_pipeline_failure", True) and pr.exit_code != 0:
        notes.append(f"pipeline failed rc={pr.exit_code}; stopping cycle")
        summary = PaperTradeCycleSummary(
            timestamp=_utc_iso(),
            mode=mode,
            ok=False,
            blocked=False,
            had_warnings=False,
            stages=stages_out,
            cycle_notes=notes,
            config=cfg_snapshot,
            arm_mode=arm_mode_label,
            arm_permissions=dict(arm_perms),
            arm_block_reasons=list(arm_block_reasons),
        )
        write_cycle_summary(summary)
        try:
            write_arm_mode_status_snapshot(arm_perms, arm_cfg, arm_block_reasons, notes)
            append_arm_mode_log(
                {
                    "ts_utc": summary.timestamp,
                    "mode": arm_mode_label,
                    "blocked": "true" if arm_block_reasons else "false",
                    "block_reasons": ";".join(arm_block_reasons),
                    "perm_execute": str(arm_perms.get("execute_trades")),
                    "perm_manage": str(arm_perms.get("manage_positions_execute")),
                    "perm_moo": str(arm_perms.get("manage_open_orders_execute_cancel")),
                    "perm_rpl": str(arm_perms.get("reprice_order_ladder_execute")),
                }
            )
        except Exception:
            pass
        append_cycle_log(
            {
                "ts_utc": summary.timestamp,
                "ok": "false",
                "blocked": "false",
                "pipeline_ok": "false",
                "execute_ok": "",
                "manage_ok": "",
                "manage_open_ok": "",
                "reprice_ladder_ok": "",
                "snapshot_ok": "",
                "manage_execute": str(manage_exec).lower(),
                "notes": ";".join(notes),
            }
        )
        print(f"\n[paper_cycle] ABORT: pipeline exit_code={pr.exit_code}", flush=True)
        _refresh_execution_pressure_best_effort()
        return 1

    # --- Stage 2: execute_trades ---
    ex_cmd = [py, "-m", "services.execute_trades", "--mode", "paper", "--execute"]
    if verbose:
        ex_cmd.append("--verbose")
    er = run_stage(ex_cmd, skip=ex_skip, treat_blocked=True)
    stages_out["execute_trades"] = _stage_dict(er)

    if _should_auto_execute_manage_positions(
        cfg=cfg,
        arm_perms=arm_perms,
        ex_skip=ex_skip,
        execute_result=er,
        manage_skip=mg_skip,
        manage_exec_already=manage_exec,
    ):
        manage_exec = True
        auto_manage_exec_triggered = True
        cfg_snapshot["manage_positions_execute"] = True
        cfg_snapshot["auto_manage_execute_triggered"] = True
        notes.append(
            "manage_positions --execute auto-enabled after execute_trades MAX_POSITIONS block"
        )

    if not ex_skip and er.exit_code == 1 and bool(cfg.get("stop_on_execute_failure", False)):
        notes.append("execute_trades hard failure; stopping per config")
        summary = PaperTradeCycleSummary(
            timestamp=_utc_iso(),
            mode=mode,
            ok=False,
            blocked=blocked_any,
            had_warnings=True,
            stages=stages_out,
            cycle_notes=notes,
            config=cfg_snapshot,
            arm_mode=arm_mode_label,
            arm_permissions=dict(arm_perms),
            arm_block_reasons=list(arm_block_reasons),
        )
        write_cycle_summary(summary)
        try:
            write_arm_mode_status_snapshot(arm_perms, arm_cfg, arm_block_reasons, notes)
            append_arm_mode_log(
                {
                    "ts_utc": summary.timestamp,
                    "mode": arm_mode_label,
                    "blocked": "true" if arm_block_reasons else "false",
                    "block_reasons": ";".join(arm_block_reasons),
                    "perm_execute": str(arm_perms.get("execute_trades")),
                    "perm_manage": str(arm_perms.get("manage_positions_execute")),
                    "perm_moo": str(arm_perms.get("manage_open_orders_execute_cancel")),
                    "perm_rpl": str(arm_perms.get("reprice_order_ladder_execute")),
                }
            )
        except Exception:
            pass
        append_cycle_log(
            {
                "ts_utc": summary.timestamp,
                "ok": "false",
                "blocked": str(blocked_any).lower(),
                "pipeline_ok": str(pr.ok if not pipeline_skip else "skipped"),
                "execute_ok": "false",
                "manage_ok": "",
                "manage_open_ok": "",
                "reprice_ladder_ok": "",
                "snapshot_ok": "",
                "manage_execute": str(manage_exec).lower(),
                "notes": ";".join(notes),
            }
        )
        print("\n[paper_cycle] ABORT: execute_trades stop_on_execute_failure", flush=True)
        _refresh_execution_pressure_best_effort()
        return 1

    if not ex_skip and er.exit_code in (0, 2):
        try:
            if EXEC_DROP_JSON.is_file():
                dj = json.loads(EXEC_DROP_JSON.read_text(encoding="utf-8", errors="replace"))
                if isinstance(dj, dict):
                    sub = int(dj.get("submitted_orders") or 0)
                    pl = int(dj.get("planned_orders") or 0)
                    dr = int(dj.get("dropped_orders") or 0)
                    inf = int(dj.get("in_flight_orders") or 0)
                    if er.exit_code == 0 and pl > 0 and sub == 0:
                        if inf > 0 and dr == 0:
                            notes.append(
                                "execute_trades: placement intent satisfied by existing open orders (in-flight); no new submits."
                            )
                        elif dr > 0:
                            notes.append(
                                "execute_trades: some orders dropped before submit; see execution_drop_diagnostics.json"
                            )
        except Exception:
            pass

    # --- Stage 3: manage_positions (plan-only unless enabled / auto-promoted) ---
    mg_cmd = [py, "-m", "services.manage_positions", "--mode", "paper"]
    if manage_exec:
        mg_cmd.append("--execute")
        if bool(cfg.get("reallocate_after_exit", False)) and bool(
            arm_perms.get("reallocate_after_exit")
        ):
            mg_cmd.append("--reallocate-after-exit")
    if verbose:
        mg_cmd.append("--verbose")
    mr = run_stage(mg_cmd, skip=mg_skip, treat_blocked=True)
    stages_out["manage_positions"] = _stage_dict(mr)
    if (not mg_skip) and mr.exit_code == 2:
        notes.append(
            "manage_positions: execution blocked — see [MANAGE_BLOCK] / [MANAGE_SUMMARY] / [ROTATION_RESULT] in stage output"
        )

    # --- Snapshot before open-order maintenance (fresh snapshot for ladder / diagnostics) ---
    snap_maint = run_stage(snap_cmd, skip=snap_maint_skip, treat_blocked=False)
    stages_out["snapshot_before_maintenance"] = _stage_dict(snap_maint)
    if not snap_maint_skip and snap_maint.exit_code != 0:
        had_warnings = True
        notes.append(f"snapshot_before_maintenance rc={snap_maint.exit_code}")

    # --- manage_open_orders (paper; optional cancel) ---
    moo_cmd = [
        py,
        "-m",
        "services.manage_open_orders",
        "--mode",
        "paper",
        "--stale-minutes",
        str(moo_stale_min),
    ]
    if moo_exec_cancel:
        moo_cmd.append("--execute-cancel")
    if verbose:
        moo_cmd.append("--verbose")
    moo = run_stage(moo_cmd, skip=moo_skip, treat_blocked=True)
    stages_out["manage_open_orders"] = _stage_dict(moo)

    # --- reprice_order_ladder (paper; optional execute) ---
    rpl_cmd = [
        py,
        "-m",
        "services.reprice_order_ladder",
        "--mode",
        "paper",
        "--max-stage",
        str(rpl_max_stage),
        "--stale-minutes-stage1",
        str(rpl_stage1_min),
    ]
    if rpl_exec:
        rpl_cmd.append("--execute")
    if verbose:
        rpl_cmd.append("--verbose")
    rpl = run_stage(rpl_cmd, skip=rpl_skip, treat_blocked=True)
    stages_out["reprice_order_ladder"] = _stage_dict(rpl)

    # --- Final snapshot refresh ---
    snap_final_skip = args.no_snapshot_refresh or (
        not bool(cfg.get("refresh_snapshots_after_cycle", True))
    )
    sr = run_stage(snap_cmd, skip=snap_final_skip, treat_blocked=False)
    stages_out["snapshot_refresh"] = _stage_dict(sr)
    if not snap_final_skip and sr.exit_code != 0:
        had_warnings = True
        notes.append(f"snapshot_refresh rc={sr.exit_code}")

    # Overall ok: hard failure (exit 1) on trading stages; snapshot failures fail the cycle (same as prior behavior)
    exec_bad = (not ex_skip) and er.exit_code not in (0, 2)
    mg_bad = (not mg_skip) and mr.exit_code not in (0, 2)
    moo_bad = (not moo_skip) and moo.exit_code not in (0, 2)
    rpl_bad = (not rpl_skip) and rpl.exit_code not in (0, 2)
    overall_ok = (
        not exec_bad
        and not mg_bad
        and not moo_bad
        and not rpl_bad
        and (snap_final_skip or sr.exit_code == 0)
        and (start_snap_skip or ss0.exit_code == 0)
        and (snap_maint_skip or snap_maint.exit_code == 0)
    )

    if exec_bad:
        notes.append(f"execute_trades failed rc={er.exit_code}")
    if mg_bad:
        notes.append(f"manage_positions failed rc={mr.exit_code}")
    if moo_bad:
        notes.append(f"manage_open_orders failed rc={moo.exit_code}")
    if rpl_bad:
        notes.append(f"reprice_order_ladder failed rc={rpl.exit_code}")

    summary = PaperTradeCycleSummary(
        timestamp=_utc_iso(),
        mode=mode,
        ok=overall_ok,
        blocked=blocked_any,
        had_warnings=had_warnings or blocked_any,
        stages=stages_out,
        cycle_notes=notes,
        config=cfg_snapshot,
        arm_mode=arm_mode_label,
        arm_permissions=dict(arm_perms),
        arm_block_reasons=list(arm_block_reasons),
    )
    write_cycle_summary(summary)
    try:
        write_arm_mode_status_snapshot(arm_perms, arm_cfg, arm_block_reasons, notes)
        append_arm_mode_log(
            {
                "ts_utc": summary.timestamp,
                "mode": arm_mode_label,
                "blocked": "true" if arm_block_reasons else "false",
                "block_reasons": ";".join(arm_block_reasons),
                "perm_execute": str(arm_perms.get("execute_trades")),
                "perm_manage": str(arm_perms.get("manage_positions_execute")),
                "perm_moo": str(arm_perms.get("manage_open_orders_execute_cancel")),
                "perm_rpl": str(arm_perms.get("reprice_order_ladder_execute")),
            }
        )
    except Exception:
        pass
    append_cycle_log(
        {
            "ts_utc": summary.timestamp,
            "ok": str(overall_ok).lower(),
            "blocked": str(blocked_any).lower(),
            "pipeline_ok": str(pr.ok if not pipeline_skip else "skipped"),
            "execute_ok": str(er.ok if not ex_skip else "skipped"),
            "manage_ok": str(mr.ok if not mg_skip else "skipped"),
            "manage_open_ok": str(moo.ok if not moo_skip else "skipped"),
            "reprice_ladder_ok": str(rpl.ok if not rpl_skip else "skipped"),
            "snapshot_ok": str(sr.ok if not snap_final_skip else "skipped"),
            "manage_execute": str(manage_exec).lower(),
            "notes": ";".join(notes),
        }
    )

    print(
        f"\n[paper_cycle] summary ok={summary.ok} blocked={summary.blocked} warnings={summary.had_warnings}",
        flush=True,
    )

    _refresh_execution_pressure_best_effort()

    if exec_bad or mg_bad or moo_bad or rpl_bad:
        return 1
    if blocked_any:
        return 2
    if not snap_final_skip and sr.exit_code != 0:
        return 1
    if not start_snap_skip and ss0.exit_code != 0:
        return 1
    if not snap_maint_skip and snap_maint.exit_code != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
