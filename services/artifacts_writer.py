# services/artifacts_writer.py
from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


def _project_root() -> Path:
    """
    Resolve repo root. We assume this file lives at <root>/services/artifacts_writer.py
    """
    return Path(__file__).resolve().parents[1]


def _results_dir() -> Path:
    return _project_root() / "data" / "results"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_heartbeat(
    *,
    status: str,
    stage: str,
    last_success_stage: Optional[str] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Write data/results/heartbeat.json (preferred) in the schema your UI expects.

    status: "ok" | "warn" | "fail"
    stage:  "fetch" | "train" | "signals" | "backtest" | "orders" | "snapshot" | etc.
    """
    status = (status or "").strip().lower()
    stage = (stage or "").strip().lower()

    if status not in {"ok", "warn", "fail"}:
        raise ValueError("heartbeat.status must be one of: ok, warn, fail")
    if not stage:
        raise ValueError("heartbeat.stage is required")

    payload: Dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "status": status,
        "stage": stage,
        "last_success_stage": last_success_stage or stage,
        "message": message or "",
        "error": error or "",
        "run_id": run_id or "",
        "host": socket.gethostname(),
    }

    if extra:
        # keep it safe/flat and JSONable
        payload["extra"] = extra

    results = _results_dir()
    results.mkdir(parents=True, exist_ok=True)

    target = out_path or (results / "heartbeat.json")
    tmp = target.with_suffix(".json.tmp")

    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)  # atomic on Windows

    return target


def write_pipeline_status_fallback(
    *,
    status: str,
    stage: str,
    message: Optional[str] = None,
    error: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Optional: Some older panels fall back to pipeline_status.json.
    We'll write it too if you want, but heartbeat.json is preferred.
    """
    payload = {
        "timestamp": _utc_now_iso(),
        "status": (status or "").strip().lower(),
        "stage": (stage or "").strip().lower(),
        "message": message or "",
        "error": error or "",
        "host": socket.gethostname(),
    }

    results = _results_dir()
    results.mkdir(parents=True, exist_ok=True)

    target = out_path or (results / "pipeline_status.json")
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
    return target
