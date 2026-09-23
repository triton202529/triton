"""Read-only current paper holdings from positions_snapshot.csv.

A stable read returns one byte string plus file_exists, resolved_path,
mtime_ns, size, SHA-256, and snapshot_ts. The cached parser receives those
values as normal arguments and parses only that byte string.
Rendering does not call the broker, refresh data, or write the snapshot.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

CURRENT_HOLDINGS_TITLE = "Current paper holdings"
CURRENT_HOLDINGS_SOURCE = "Source: positions_snapshot.csv"
CURRENT_HOLDINGS_UNAVAILABLE = "Current holdings unavailable"

_PARSE_KEYS: List[Tuple[Any, ...]] = []


def snapshot_as_of_label(snapshot_ts: str) -> str:
    return f"Snapshot as of: {snapshot_ts}"


def _symbol(row: Dict[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "").strip().upper()


def _fail(status: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "status": status, "message": message, "rows": [], "snapshot_ts": None}


def _valid_snapshot_ts(value: str) -> bool:
    text = value.strip()
    if "T" not in text:
        return False
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        datetime.fromisoformat(iso)
    except ValueError:
        return False
    return True


def _parse_text(text: str) -> Dict[str, Any]:
    try:
        reader = csv.DictReader(StringIO(text))
        fields = list(reader.fieldnames or [])
        raw_rows = list(reader)
    except csv.Error as exc:
        return _fail("malformed", f"Canonical position snapshot is malformed: {exc}")
    if "symbol" not in fields and "ticker" not in fields:
        return _fail("malformed", "Canonical position snapshot is missing a symbol column.")
    if "qty" not in fields:
        return _fail("malformed", "Canonical position snapshot is missing a qty column.")
    if "snapshot_ts" not in fields:
        return _fail("malformed", "Canonical position snapshot is missing snapshot_ts.")
    rows: List[Dict[str, str]] = []
    seen = set()
    stamps = set()
    for raw in raw_rows:
        if not any(str(value or "").strip() for value in raw.values()):
            continue
        symbol = _symbol(raw)
        qty = str(raw.get("qty") or "").strip()
        stamp = str(raw.get("snapshot_ts") or "").strip()
        if not symbol or not qty or not stamp:
            return _fail(
                "malformed",
                "Canonical position snapshot has a row without symbol, qty, or snapshot_ts.",
            )
        if symbol.startswith("^"):
            return _fail(
                "malformed",
                f"Canonical position snapshot contains index pseudo-position {symbol}.",
            )
        if symbol in seen:
            return _fail(
                "malformed",
                f"Canonical position snapshot contains duplicate symbol {symbol}.",
            )
        if not _valid_snapshot_ts(stamp):
            return _fail(
                "malformed",
                f"Canonical position snapshot has an invalid snapshot_ts for {symbol}.",
            )
        try:
            number = Decimal(qty)
        except InvalidOperation:
            return _fail(
                "malformed",
                f"Canonical position snapshot has a non-numeric qty for {symbol}.",
            )
        if not number.is_finite() or number < 0:
            return _fail(
                "malformed",
                f"Canonical position snapshot has a negative or non-finite qty for {symbol}.",
            )
        seen.add(symbol)
        stamps.add(stamp)
        rows.append(
            {
                "symbol": symbol,
                "qty": qty,
                "snapshot_ts": stamp,
                "market_value": str(raw.get("market_value") or ""),
                "value": str(raw.get("value") or ""),
                "avg_entry_price": str(raw.get("avg_entry_price") or ""),
                "current_price": str(raw.get("current_price") or ""),
                "unrealized_pl": str(raw.get("unrealized_pl") or ""),
                "unrealized_pnl": str(raw.get("unrealized_pnl") or ""),
                "company_name": str(raw.get("company_name") or raw.get("name") or ""),
            }
        )
    if not rows:
        return _fail("malformed", "Canonical position snapshot has no position rows.")
    if len(stamps) != 1:
        return _fail("malformed", "Canonical position snapshot has mixed snapshot timestamps.")
    rows.sort(key=lambda row: row["symbol"])
    return {
        "ok": True,
        "status": "ok",
        "message": None,
        "rows": rows,
        "snapshot_ts": next(iter(stamps)),
    }


def _read_file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _stamp(path: Path) -> Optional[Tuple[int, int]]:
    try:
        info = path.stat()
    except OSError:
        return None
    return (int(info.st_mtime_ns), int(info.st_size))


def _stamp_before(path: Path) -> Optional[Tuple[int, int]]:
    return _stamp(path)


def _stamp_after(path: Path) -> Optional[Tuple[int, int]]:
    return _stamp(path)


def _unavailable(status: str, detail: str, **extra: Any) -> Dict[str, Any]:
    body = {
        "ok": False,
        "status": status,
        "file_exists": status != "missing",
        "resolved_path": extra.get("resolved_path"),
        "mtime_ns": extra.get("mtime_ns"),
        "byte_size": extra.get("byte_size"),
        "sha256": extra.get("sha256"),
        "snapshot_ts": None,
        "message": f"{CURRENT_HOLDINGS_UNAVAILABLE}. {detail}",
        "row_count": 0,
    }
    return body


def _read_stable_snapshot(path: Path) -> Dict[str, Any]:
    """Read one byte string whose hash matches the stat taken around that read."""
    resolved = str(path.resolve())
    if not path.is_file():
        return _unavailable(
            "missing",
            "Canonical position snapshot is missing.",
            resolved_path=resolved,
        )
    for _attempt in (1, 2):
        before = _stamp_before(path)
        try:
            raw = _read_file_bytes(path)
        except OSError:
            raw = None
        after = _stamp_after(path)
        if raw is None:
            continue
        if before is None or after != before or before[1] != len(raw):
            continue
        return {
            "ok": True,
            "raw": raw,
            "file_exists": True,
            "mtime_ns": before[0],
            "byte_size": before[1],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "resolved_path": str(path.resolve()),
        }
    return _unavailable(
        "unstable",
        "Canonical position snapshot changed while reading.",
        resolved_path=resolved,
    )


def snapshot_identity(path: Path) -> Dict[str, Any]:
    """Identity fields for the cache key. Does not enter the cached loader."""
    stable = _read_stable_snapshot(Path(path))
    if not stable.get("ok"):
        return stable
    raw = stable["raw"]
    base = {
        "ok": False,
        "status": "malformed",
        "file_exists": True,
        "resolved_path": stable["resolved_path"],
        "mtime_ns": stable["mtime_ns"],
        "byte_size": stable["byte_size"],
        "sha256": stable["sha256"],
        "snapshot_ts": None,
        "message": f"{CURRENT_HOLDINGS_UNAVAILABLE}. Canonical position snapshot is empty.",
        "row_count": 0,
        "raw": raw,
    }
    if not raw.strip():
        return base
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        base["message"] = (
            f"{CURRENT_HOLDINGS_UNAVAILABLE}. Canonical position snapshot is not valid UTF-8."
        )
        return base
    parsed = _parse_text(text)
    if not parsed.get("ok"):
        base["message"] = f"{CURRENT_HOLDINGS_UNAVAILABLE}. {parsed.get('message')}"
        return base
    if hashlib.sha256(raw).hexdigest() != stable["sha256"] or len(raw) != stable["byte_size"]:
        return _unavailable(
            "unstable",
            "Canonical position snapshot changed while reading.",
            resolved_path=stable["resolved_path"],
        )
    base.update(
        ok=True,
        status="ok",
        snapshot_ts=parsed.get("snapshot_ts"),
        message=None,
        row_count=len(parsed.get("rows") or []),
    )
    return base


def _failure_view(ident: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": False,
        "status": ident.get("status") or "malformed",
        "message": ident.get("message") or f"{CURRENT_HOLDINGS_UNAVAILABLE}.",
        "rows": [],
        "identity": {key: value for key, value in ident.items() if key != "raw"},
        "snapshot_ts": None,
        "title": CURRENT_HOLDINGS_TITLE,
        "source_label": CURRENT_HOLDINGS_SOURCE,
        "as_of_label": None,
    }


@st.cache_data(show_spinner=False)
def cached_parse(
    raw_bytes: bytes,
    file_exists: bool,
    resolved_path: str,
    mtime_ns: int,
    size: int,
    sha256: str,
    snapshot_ts: str,
) -> List[Dict[str, str]]:
    """Parse raw_bytes only. Do not open, stat, or read resolved_path.

    Empty bytes are not an empty portfolio. A false file_exists, a hash
    mismatch, or a timestamp mismatch raises and returns no rows.
    """
    if file_exists is not True:
        raise ValueError("snapshot identity is not an existing file")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if len(raw_bytes) != size or digest != sha256:
        raise ValueError("snapshot bytes do not match the cache identity")
    if not raw_bytes.strip():
        raise ValueError("empty snapshot bytes are not a portfolio")
    parsed = _parse_text(raw_bytes.decode("utf-8-sig"))
    if not parsed.get("ok") or not parsed.get("rows"):
        raise ValueError(str(parsed.get("message") or "malformed snapshot"))
    if parsed.get("snapshot_ts") != snapshot_ts:
        raise ValueError("snapshot_ts does not match the parsed bytes")
    rows = list(parsed["rows"])
    if any(row.get("snapshot_ts") != snapshot_ts for row in rows):
        raise ValueError("row timestamp does not match the cache identity")
    _PARSE_KEYS.append((True, resolved_path, mtime_ns, size, sha256, snapshot_ts, digest))
    return rows


def load_portfolio_holdings(path: Path) -> Dict[str, Any]:
    """Return current paper holdings, or a visible failure with no rows."""
    ident = snapshot_identity(path)
    if not ident.get("ok"):
        return _failure_view(ident)
    stamp = str(ident["snapshot_ts"])
    try:
        rows = cached_parse(
            bytes(ident["raw"]),
            bool(ident["file_exists"]),
            str(ident["resolved_path"]),
            int(ident["mtime_ns"]),
            int(ident["byte_size"]),
            str(ident["sha256"]),
            stamp,
        )
    except (UnicodeDecodeError, ValueError):
        failed = _unavailable(
            "unstable",
            "Canonical position snapshot changed while reading.",
            resolved_path=ident.get("resolved_path"),
            mtime_ns=ident.get("mtime_ns"),
            byte_size=ident.get("byte_size"),
            sha256=ident.get("sha256"),
        )
        return _failure_view(failed)
    if any(row.get("snapshot_ts") != stamp for row in rows):
        return _failure_view(
            _unavailable(
                "unstable",
                "Canonical position snapshot changed while reading.",
                resolved_path=ident.get("resolved_path"),
            )
        )
    public_identity = {key: value for key, value in ident.items() if key != "raw"}
    return {
        "ok": True,
        "status": "ok",
        "message": None,
        "rows": list(rows),
        "identity": public_identity,
        "snapshot_ts": stamp,
        "title": CURRENT_HOLDINGS_TITLE,
        "source_label": CURRENT_HOLDINGS_SOURCE,
        "as_of_label": snapshot_as_of_label(stamp),
    }


def holdings_quantity_pairs(loaded: Dict[str, Any]) -> List[Dict[str, str]]:
    """Exact symbol and qty strings from a successful load."""
    if not loaded.get("ok"):
        return []
    return [{"symbol": row["symbol"], "qty": row["qty"]} for row in loaded.get("rows") or []]
