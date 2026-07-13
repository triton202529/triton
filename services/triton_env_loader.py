"""
triton_env_loader.py
Phase 197F — Safe local env loading and Alpaca credential alias normalization.

Never logs raw secrets. Local secret files are git-ignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.credential_mask import format_env_key_status, mask_api_key

ROOT = Path(__file__).resolve().parents[1]

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE = "https://api.alpaca.markets"

KEY_ALIASES = (
    "APCA_API_KEY_ID",
    "ALPACA_API_KEY_ID",
    "ALPACA_KEY_ID",
)
SECRET_ALIASES = (
    "APCA_API_SECRET_KEY",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_API_SECRET",
)
BASE_URL_ALIASES = (
    "APCA_API_BASE_URL",
    "ALPACA_BASE_URL",
    "ALPACA_ENDPOINT",
)

LOCAL_ENV_FILES = (
    ROOT / ".env.local",
    ROOT / ".env.paper.local",
    ROOT / "config" / "local" / "runtime_env.local",
    ROOT / "config" / ".env",
    ROOT / ".env",
)


def _set_if_missing(key: str, value: str) -> None:
    if value and os.getenv(key) is None:
        os.environ[key] = value


def _load_plain_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                _set_if_missing(k, v)
    except Exception:
        pass


def _load_json_credentials(path: Path) -> None:
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in KEY_ALIASES + SECRET_ALIASES + BASE_URL_ALIASES:
            if key in data and data[key]:
                _set_if_missing(key, str(data[key]))
    except Exception:
        pass


def load_triton_local_env(*, verbose: bool = False) -> List[str]:
    """Load ignored local env files (first wins for unset vars only). Returns loaded paths."""
    loaded: List[str] = []
    for path in LOCAL_ENV_FILES:
        if path.is_file():
            _load_plain_env_file(path)
            loaded.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    _load_json_credentials(ROOT / "config" / "alpaca.json")
    normalize_alpaca_env_aliases()
    if verbose and loaded:
        print(f"[ENV_LOADER] loaded_local_files={loaded}", flush=True)
    return loaded


def _first_env(names: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    for name in names:
        val = os.getenv(name)
        if val is not None and str(val).strip():
            return name, str(val).strip()
    return None, None


def normalize_alpaca_env_aliases() -> Dict[str, str]:
    """
    Canonicalize Alpaca env aliases without overwriting explicitly set canonical keys.
    Returns mapping of canonical_name -> source_alias_used.
    """
    mapping: Dict[str, str] = {}

    key_alias, key_val = _first_env(KEY_ALIASES)
    if key_val:
        for name in KEY_ALIASES:
            _set_if_missing(name, key_val)
        if key_alias:
            mapping["APCA_API_KEY_ID"] = key_alias

    secret_alias, secret_val = _first_env(SECRET_ALIASES)
    if secret_val:
        for name in SECRET_ALIASES:
            _set_if_missing(name, secret_val)
        if secret_alias:
            mapping["APCA_API_SECRET_KEY"] = secret_alias

    base_alias, base_val = _first_env(BASE_URL_ALIASES)
    if base_val:
        for name in BASE_URL_ALIASES:
            _set_if_missing(name, base_val.rstrip("/"))
        if base_alias:
            mapping["APCA_API_BASE_URL"] = base_alias

    return mapping


def alpaca_env_diagnostics(*, mode: str = "paper") -> Dict[str, Any]:
    load_triton_local_env()
    alias_map = normalize_alpaca_env_aliases()
    key_alias, key_val = _first_env(KEY_ALIASES)
    secret_alias, secret_val = _first_env(SECRET_ALIASES)
    base_alias, base_val = _first_env(BASE_URL_ALIASES)

    mode_l = str(mode or "paper").lower()
    expected_base = ALPACA_PAPER_BASE if mode_l == "paper" else ALPACA_LIVE_BASE
    resolved_base = base_val or expected_base
    paper_endpoint = mode_l == "paper" and ALPACA_PAPER_BASE in resolved_base.lower()

    aliases_present = {
        name: bool(os.getenv(name))
        for group in (KEY_ALIASES, SECRET_ALIASES, BASE_URL_ALIASES)
        for name in group
    }

    missing: List[str] = []
    if not key_val:
        missing.append("api_key")
    if not secret_val:
        missing.append("api_secret")

    return {
        "mode": mode_l,
        "aliases_present": aliases_present,
        "key_alias_used": key_alias,
        "secret_alias_used": secret_alias,
        "base_url_alias_used": base_alias,
        "key_masked": mask_api_key(key_val),
        "key_length": len(key_val) if key_val else 0,
        "secret_length": len(secret_val) if secret_val else 0,
        "base_url": resolved_base,
        "expected_base_url": expected_base,
        "paper_endpoint_match": paper_endpoint,
        "credentials_complete": not missing,
        "missing_fields": missing,
        "alias_normalization": alias_map,
        "env_check_lines": [
            format_env_key_status(name, os.getenv(name))
            for name in (
                "ALPACA_KEY_ID",
                "APCA_API_KEY_ID",
                "ALPACA_SECRET_KEY",
                "APCA_API_SECRET_KEY",
                "ALPACA_API_SECRET",
                "ALPACA_BASE_URL",
                "APCA_API_BASE_URL",
            )
        ],
    }


def resolve_alpaca_credentials(*, mode: str = "paper") -> Dict[str, Any]:
    diag = alpaca_env_diagnostics(mode=mode)
    if not diag["credentials_complete"]:
        return {
            "ok": False,
            "status": "SAFE_AUTH_BLOCK",
            "reason": "missing_credentials",
            "diagnostics": diag,
        }
    mode_l = str(mode or "paper").lower()
    expected = ALPACA_PAPER_BASE if mode_l == "paper" else ALPACA_LIVE_BASE
    base = str(diag["base_url"] or expected).rstrip("/")
    if mode_l == "paper" and ALPACA_PAPER_BASE not in base.lower():
        return {
            "ok": False,
            "status": "SAFE_AUTH_BLOCK",
            "reason": "paper_mode_live_base_url_mismatch",
            "diagnostics": diag,
        }
    key_alias, key_val = _first_env(KEY_ALIASES)
    secret_alias, secret_val = _first_env(SECRET_ALIASES)
    return {
        "ok": True,
        "status": "READY",
        "mode": mode_l,
        "key": key_val,
        "secret": secret_val,
        "base_url": base,
        "key_alias_used": key_alias,
        "secret_alias_used": secret_alias,
        "diagnostics": diag,
    }


def format_auth_failure_diagnostic(
    *,
    mode: str,
    status_code: int,
    diagnostics: Dict[str, Any],
    response_text: str = "",
) -> str:
    d = diagnostics
    lines = [
        f"[BROKER_AUTH] status=SAFE_AUTH_BLOCK http={status_code}",
        f"[BROKER_AUTH] mode={mode}",
        f"[BROKER_AUTH] base_url={d.get('base_url')}",
        f"[BROKER_AUTH] key_alias_used={d.get('key_alias_used')}",
        f"[BROKER_AUTH] secret_alias_used={d.get('secret_alias_used')}",
        f"[BROKER_AUTH] key_masked={d.get('key_masked')} key_length={d.get('key_length')}",
        f"[BROKER_AUTH] secret_length={d.get('secret_length')}",
    ]
    if mode == "paper" and not d.get("paper_endpoint_match"):
        lines.append(
            "[BROKER_AUTH] warning=paper_mode_base_url_not_paper_endpoint "
            f"expected={d.get('expected_base_url')}"
        )
    if response_text:
        lines.append(f"[BROKER_AUTH] broker_message={response_text[:200]}")
    lines.append(
        "[BROKER_AUTH] hint=verify local ignored env file "
        "(.env.local / .env.paper.local / config/local/runtime_env.local)"
    )
    return "\n".join(lines)


PAPER_ENV_AUTHORITY_SUPERVISOR_ONLY = "PAPER_ENV_AUTHORITY_SUPERVISOR_ONLY"
PAPER_ENV_AUTHORITY_CLI_LOADED = "PAPER_ENV_AUTHORITY_CLI_LOADED"
PAPER_ENV_AUTHORITY_SHARED_BOOTSTRAP = "PAPER_ENV_AUTHORITY_SHARED_BOOTSTRAP"
PAPER_ENV_AUTHORITY_MISSING = "PAPER_ENV_AUTHORITY_MISSING"
PAPER_ENV_AUTHORITY_CONFLICT = "PAPER_ENV_AUTHORITY_CONFLICT"

PAPER_EXECUTION_ENV = "TRITON_ENABLE_PAPER_EXECUTION"
AUTO_PAPER_ENTRIES_ENV = "TRITON_ENABLE_AUTO_PAPER_ENTRIES"
AUTO_PAPER_APPROVAL_ENV = "TRITON_ENABLE_AUTO_PAPER_APPROVAL"
LIVE_EXECUTION_ENV = "TRITON_ENABLE_LIVE_EXECUTION"
LIVE_FORCE_ENV = "TRITON_FORCE_LIVE"


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip() == "1"


def _broker_paper_env_present() -> bool:
    diag = alpaca_env_diagnostics(mode="paper")
    return bool(diag.get("credentials_complete"))


def _broker_live_env_present_or_unknown() -> bool:
    key_alias, key_val = _first_env(KEY_ALIASES)
    secret_alias, secret_val = _first_env(SECRET_ALIASES)
    base_alias, base_val = _first_env(BASE_URL_ALIASES)
    if not key_val or not secret_val:
        return False
    if base_val and ALPACA_LIVE_BASE in base_val.lower():
        return True
    if base_val and ALPACA_PAPER_BASE in base_val.lower():
        return False
    return bool(key_val and secret_val)


def evaluate_paper_env_authority(
    *,
    load_local_env: bool = False,
    cli_validation: bool = False,
) -> Dict[str, Any]:
    """
    Report paper/live env authority for validators. Never enables live execution.
    Optionally loads git-ignored local env files (credentials only unless present in files).
    """
    env_source = "process_environment"
    loaded_files: List[str] = []
    if load_local_env or cli_validation:
        loaded_files = load_triton_local_env()
        env_source = "triton_env_loader"

    paper_enabled = _env_truthy(PAPER_EXECUTION_ENV)
    auto_entries = _env_truthy(AUTO_PAPER_ENTRIES_ENV)
    auto_approval = _env_truthy(AUTO_PAPER_APPROVAL_ENV)
    live_enabled = _env_truthy(LIVE_EXECUTION_ENV)
    live_force = _env_truthy(LIVE_FORCE_ENV)

    if live_enabled or live_force:
        state = PAPER_ENV_AUTHORITY_CONFLICT
        reason = "live_execution_env_enabled"
    elif paper_enabled and auto_entries and auto_approval:
        if loaded_files and cli_validation:
            state = PAPER_ENV_AUTHORITY_CLI_LOADED
            reason = "cli_loaded_paper_flags_present"
        elif loaded_files:
            state = PAPER_ENV_AUTHORITY_SHARED_BOOTSTRAP
            reason = "bootstrap_files_loaded_paper_flags_present"
        else:
            state = PAPER_ENV_AUTHORITY_SUPERVISOR_ONLY
            reason = "supervisor_runtime_paper_flags_present"
    elif paper_enabled or auto_entries or auto_approval:
        state = PAPER_ENV_AUTHORITY_CONFLICT
        reason = "partial_paper_env_flags"
    elif loaded_files and cli_validation:
        state = PAPER_ENV_AUTHORITY_CLI_LOADED
        reason = "cli_loaded_credentials_only_paper_flags_missing"
    elif loaded_files:
        state = PAPER_ENV_AUTHORITY_SHARED_BOOTSTRAP
        reason = "bootstrap_loaded_paper_flags_missing"
    else:
        state = PAPER_ENV_AUTHORITY_MISSING
        reason = "paper_env_not_armed"

    broker_diag = alpaca_env_diagnostics(mode="paper")
    return {
        "env_authority_state": state,
        "env_authority_reason": reason,
        "env_source": env_source,
        "loaded_local_env_files": loaded_files,
        "cli_validation_env_loaded": bool(loaded_files) if cli_validation else False,
        "paper_execution_enabled": paper_enabled,
        "auto_paper_entries_enabled": auto_entries,
        "auto_paper_approval_enabled": auto_approval,
        "live_execution_enabled": live_enabled,
        "live_force_enabled": live_force,
        "broker_paper_env_present": bool(broker_diag.get("credentials_complete")),
        "broker_live_env_present_or_unknown": _broker_live_env_present_or_unknown(),
        "secrets_masked": True,
        "broker_env_check_lines": broker_diag.get("env_check_lines") or [],
        "key_masked": broker_diag.get("key_masked"),
    }


def load_cli_validation_env(*, verbose: bool = False) -> Dict[str, Any]:
    """Read-only CLI validation env load: local credentials, never arms live execution."""
    if _env_truthy(LIVE_EXECUTION_ENV) or _env_truthy(LIVE_FORCE_ENV):
        return evaluate_paper_env_authority(load_local_env=True, cli_validation=True)
    return evaluate_paper_env_authority(load_local_env=True, cli_validation=True)


def _readiness_paper_supervisor_active() -> Tuple[bool, str]:
    """True when continuous loop is running and readiness permits paper (not live)."""
    try:
        from services.runtime_process_detector import detect_runtime_loop_processes

        procs = detect_runtime_loop_processes()
        if not procs.get("continuous_loop_running"):
            return False, "continuous_loop_not_running"
        readiness_path = ROOT / "data" / "results" / "execution_readiness.json"
        if not readiness_path.is_file():
            return False, "execution_readiness_missing"
        doc = json.loads(readiness_path.read_text(encoding="utf-8"))
        if not bool(doc.get("paper_execution_permitted", False)):
            return False, "paper_execution_not_permitted"
        if bool(doc.get("live_execution_permitted", False)):
            return False, "live_execution_permitted"
        return True, "continuous_loop_paper_runtime"
    except Exception as exc:
        return False, f"supervisor_check_error:{exc}"


def resolve_paper_runtime_env_flags() -> Dict[str, Any]:
    """
    Single source of truth for paper execution env flags across CLI, loop stage,
    session arm policy, and dashboard posture.
    """
    paper = _env_truthy(PAPER_EXECUTION_ENV)
    auto_entries = _env_truthy(AUTO_PAPER_ENTRIES_ENV)
    auto_approval = _env_truthy(AUTO_PAPER_APPROVAL_ENV)
    source = "process_environment" if paper else "missing"
    supervisor_active = False
    supervisor_reason = ""

    if not paper:
        supervisor_active, supervisor_reason = _readiness_paper_supervisor_active()
        if supervisor_active:
            paper = True
            auto_entries = True
            auto_approval = True
            source = "supervisor_runtime_inherited"

    all_required = paper and auto_entries
    return {
        "paper_execution_enabled": paper,
        "auto_paper_entries_enabled": auto_entries,
        "auto_paper_approval_enabled": auto_approval,
        "all_required_enabled": all_required,
        "env_source": source,
        "supervisor_runtime_active": supervisor_active,
        "supervisor_runtime_reason": supervisor_reason,
        "process_env_paper_execution": _env_truthy(PAPER_EXECUTION_ENV),
        "process_env_auto_paper_entries": _env_truthy(AUTO_PAPER_ENTRIES_ENV),
        "process_env_auto_paper_approval": _env_truthy(AUTO_PAPER_APPROVAL_ENV),
    }


def paper_runtime_env_ready() -> Tuple[bool, str, Dict[str, Any]]:
    """Return (ok, reason, flags) for paper execution env preconditions."""
    flags = resolve_paper_runtime_env_flags()
    if flags["paper_execution_enabled"] and flags["auto_paper_entries_enabled"]:
        return True, "ok", flags
    if not flags["paper_execution_enabled"]:
        if flags.get("supervisor_runtime_active"):
            return (
                False,
                "TRITON_ENABLE_PAPER_EXECUTION not set in shell (supervisor active)",
                flags,
            )
        return False, "TRITON_ENABLE_PAPER_EXECUTION not set to 1", flags
    return False, "TRITON_ENABLE_AUTO_PAPER_ENTRIES not set to 1", flags
