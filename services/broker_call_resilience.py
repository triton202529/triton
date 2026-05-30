"""
Bounded retry/backoff for Alpaca / HTTP client transient faults.

Used from poll_order_status, manage_open_orders, and similar (not broker API semantics).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar("T")

_DEFAULT_ATTEMPTS = 3
_DEFAULT_BACKOFF = (2.0, 5.0, 10.0)


def load_resilience_config() -> Dict[str, Any]:
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    p = root / "config" / "execution_resilience.json"
    out: Dict[str, Any] = {
        "transient_max_attempts": _DEFAULT_ATTEMPTS,
        "transient_backoff_seconds": list(_DEFAULT_BACKOFF),
    }
    try:
        if p.is_file():
            u = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(u, dict):
                out.update({k: u[k] for k in u if k in out or k == "transient_backoff_seconds"})
    except Exception:
        pass
    return out


def is_transient_broker_error(exc: BaseException) -> bool:
    """Connection/DNS/timeout/reset class errors — not HTTP 4xx/5xx from Alpaca."""
    if exc is None:
        return False
    try:
        import requests

        if isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.Timeout,
            ),
        ):
            return True
    except Exception:
        pass
    try:
        from urllib3.exceptions import NewConnectionError, MaxRetryError, ReadTimeoutError

        if isinstance(exc, (NewConnectionError, MaxRetryError, ReadTimeoutError)):
            return True
    except Exception:
        pass
    if isinstance(exc, (ConnectionResetError, BrokenPipeError, OSError)):
        en = type(exc).__name__
        if "gaierror" in en or "Timeout" in en:
            return True
        if isinstance(exc, OSError) and exc.errno in (10051, 10050, 101, 110, 11):
            return True
    return False


def _err_type(exc: BaseException) -> str:
    return type(exc).__name__


def call_with_transient_retry(
    step: str,
    fn: Callable[..., T],
    *args: Any,
    out_counts: Optional[Dict[str, int]] = None,
    max_attempts: Optional[int] = None,
    backoff: Optional[Union[List[float], tuple]] = None,
    **fn_kwargs: Any,
) -> T:
    """
    Call fn; on transient errors, retry with backoff, logging [BROKER_RETRY] lines.
    Re-raises the last exception if all attempts fail.
    """
    cfg = load_resilience_config()
    n = int(max_attempts or cfg.get("transient_max_attempts") or _DEFAULT_ATTEMPTS)
    raw = (
        (list(backoff) if backoff is not None else None)
        or cfg.get("transient_backoff_seconds")
        or list(_DEFAULT_BACKOFF)
    )
    backoff: List[float] = []
    for i, v in enumerate(raw):
        try:
            backoff.append(float(v))
        except Exception:
            backoff.append(2.0)
    if len(backoff) < n - 1:
        # pad
        b0 = backoff[-1] if backoff else 10.0
        while len(backoff) < n - 1:
            b0 = b0 * 1.5
            backoff.append(b0)

    last: Optional[BaseException] = None
    for attempt in range(1, n + 1):
        try:
            return fn(*args, **fn_kwargs)
        except Exception as e:
            last = e
            if not is_transient_broker_error(e) or attempt >= n:
                raise
            if out_counts is not None:
                out_counts["retried"] = int(out_counts.get("retried", 0)) + 1
            delay = float(backoff[attempt - 1]) if attempt - 1 < len(backoff) else 10.0
            msg = str(e).replace("\n", " ")[:500]
            print(
                f"[BROKER_RETRY] step={step} attempt={attempt} error_type={_err_type(e)} message={msg}",
                flush=True,
            )
            time.sleep(delay)
    if last is not None:
        raise last
    raise RuntimeError("call_with_transient_retry: empty failure")


def transient_failure_kind(exc: BaseException) -> str:
    """Short label for [LOOP] DEGRADED reason=."""
    if exc is None:
        return "unknown"
    t = _err_type(exc)
    s = str(exc).lower()
    if "gai" in t.lower() or "name resolution" in s or "getaddrinfo" in s or "eai_again" in s:
        return "dns"
    if "read timed out" in s or "timeout" in s or "timed out" in s:
        return "broker_timeout"
    if "connection reset" in s or isinstance(exc, ConnectionResetError):
        return "connection_reset"
    if is_transient_broker_error(exc):
        return "network"
    return "network"
