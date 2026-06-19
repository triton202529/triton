"""Safe masking for API keys and secrets in logs."""

from __future__ import annotations

from typing import Optional


def mask_api_key(value: Optional[str]) -> str:
    """Return a log-safe representation of an API key (never the full value)."""
    if value is None or not str(value).strip():
        return "NOT_SET"
    s = str(value).strip()
    if len(s) <= 4:
        return "****"
    if len(s) <= 8:
        return f"{s[:2]}****{s[-2:]}"
    return f"{s[:2]}****{s[-4:]}"


def format_env_key_status(env_name: str, value: Optional[str]) -> str:
    """Format [ENV CHECK] line without exposing secrets."""
    return f"[ENV CHECK] {env_name} = {mask_api_key(value)}"
