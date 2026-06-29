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
