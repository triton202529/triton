"""Canonical security identity for internal joins.

TRITON has one identity key for equality and joins. Provider-specific symbols
stay at adapters. Broker-accepted symbols stay on the wire.

This module is the single alias policy. Do not add a second normalizer.

Explicit class-share allowlist only — not punctuation stripping.
BRK.B and BRK-B are the same security. HD/DBA/BITO/TSLA/GLD are unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Identity form = broker-accepted class-share form (Alpaca: BRK.B / BF.B).
CLASS_SHARE_IDENTITY_ALIASES: Dict[str, str] = {
    "BRK-B": "BRK.B",
    "BF-B": "BF.B",
}

# Market-data / Yahoo-style aliases. Never send these blindly to the broker.
CLASS_SHARE_MARKET_DATA_ALIASES: Dict[str, str] = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# Broker submission aliases (same mapping as historical place_live_orders.SYMBOL_ALIASES).
CLASS_SHARE_BROKER_ALIASES: Dict[str, str] = dict(CLASS_SHARE_IDENTITY_ALIASES)

COLLISION_REASON = "SYMBOL_IDENTITY_COLLISION"


class SymbolIdentityCollision(ValueError):
    """Two distinct raw symbols collapsed to one identity with conflicting presence."""

    def __init__(
        self,
        collisions: Sequence[Mapping[str, Any]],
        *,
        source: str = "",
    ) -> None:
        self.collisions = [dict(c) for c in collisions]
        self.source = str(source or "")
        self.reason_code = COLLISION_REASON
        msg = (
            f"{COLLISION_REASON} source={self.source or 'unspecified'} "
            f"count={len(self.collisions)} evidence={self.collisions}"
        )
        super().__init__(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "source": self.source,
            "collisions": list(self.collisions),
            "fail_closed": True,
            "summed": False,
            "duplicate_position": False,
        }


def _strip_upper(symbol: Any) -> str:
    if symbol is None:
        return ""
    s = str(symbol).strip().upper()
    if s in ("", "NAN", "NONE", "NULL"):
        return ""
    return s


def identity_key(symbol: Any) -> str:
    """Canonical internal identity for equality and joins."""
    s = _strip_upper(symbol)
    if not s:
        return ""
    return CLASS_SHARE_IDENTITY_ALIASES.get(s, s)


def broker_submit_symbol(symbol: Any) -> str:
    """Broker-accepted symbol for order submission (Alpaca class-share dots)."""
    s = _strip_upper(symbol)
    if not s:
        return ""
    return CLASS_SHARE_BROKER_ALIASES.get(s, s)


def market_data_symbol(symbol: Any) -> str:
    """Provider/Yahoo-style alias. Never a broker submission symbol."""
    ident = identity_key(symbol)
    if not ident:
        return ""
    return CLASS_SHARE_MARKET_DATA_ALIASES.get(ident, ident)


def collect_identity_collisions(
    raw_symbols: Iterable[Any],
) -> List[Dict[str, Any]]:
    """Return identities that appear under two or more distinct raw forms."""
    by_ident: Dict[str, set] = {}
    for raw in raw_symbols:
        s = _strip_upper(raw)
        if not s:
            continue
        ident = identity_key(s)
        by_ident.setdefault(ident, set()).add(s)
    out: List[Dict[str, Any]] = []
    for ident, forms in sorted(by_ident.items()):
        if len(forms) > 1:
            out.append(
                {
                    "identity": ident,
                    "raw_forms": sorted(forms),
                    "reason_code": COLLISION_REASON,
                }
            )
    return out


def assert_no_identity_collisions(
    raw_symbols: Iterable[Any],
    *,
    source: str = "",
) -> None:
    collisions = collect_identity_collisions(raw_symbols)
    if collisions:
        raise SymbolIdentityCollision(collisions, source=source)


def assert_unique_identities(
    raw_symbols: Iterable[Any],
    *,
    source: str = "",
) -> List[str]:
    """One row per identity. Distinct alias forms and repeated spellings fail closed."""
    raws = [symbol for symbol in (_strip_upper(value) for value in raw_symbols) if symbol]
    assert_no_identity_collisions(raws, source=source)
    if len(raws) != len(set(raws)):
        dupes = sorted({symbol for symbol in raws if raws.count(symbol) > 1})
        raise SymbolIdentityCollision(
            [
                {
                    "identity": identity_key(symbol),
                    "raw_forms": [symbol],
                    "reason_code": COLLISION_REASON,
                }
                for symbol in dupes
            ],
            source=source,
        )
    return raws


def candidate_display_map(candidate_symbols: Iterable[Any]) -> Dict[str, str]:
    """Map identity_key to the candidate's own spelling.

    Joins use identity_key. Artifacts keep this spelling. Two candidate
    spellings for one identity fail closed. Symbols outside the explicit
    class-share allowlist are not folded.
    """
    raws = assert_unique_identities(candidate_symbols, source="candidate")
    return {identity_key(symbol): symbol for symbol in raws}


def candidate_display_symbol(symbol: Any, display: Mapping[str, str]) -> str:
    """Return the candidate spelling when this symbol is an explicit alias of one."""
    raw = _strip_upper(symbol)
    if not raw:
        return ""
    chosen = display.get(identity_key(raw))
    if chosen:
        return str(chosen)
    return raw


def restore_candidate_spellings(
    symbols: Iterable[Any],
    candidate_symbols: Iterable[Any],
) -> List[str]:
    """Replace alias forms with the candidate spelling without adding rows."""
    display = candidate_display_map(candidate_symbols)
    raws = [_strip_upper(symbol) for symbol in symbols]
    assert_unique_identities([symbol for symbol in raws if symbol], source="artifact")
    restored = [candidate_display_symbol(symbol, display) if symbol else "" for symbol in raws]
    assert_unique_identities([symbol for symbol in restored if symbol], source="artifact")
    return restored


def fold_qty_map(
    rows: Sequence[Tuple[Any, float]],
    *,
    source: str = "positions",
) -> Dict[str, float]:
    """Fold (raw_symbol, qty) into identity -> qty.

    Same raw form twice: last write wins (existing broker-loop behavior).
    Distinct raw forms of one identity: fail closed — never sum, never duplicate.
    """
    raw_forms: List[str] = []
    last_qty: Dict[str, float] = {}
    last_raw: Dict[str, str] = {}
    for raw, qty in rows:
        s = _strip_upper(raw)
        if not s:
            continue
        try:
            q = float(qty)
        except (TypeError, ValueError):
            continue
        raw_forms.append(s)
        last_qty[s] = q
        last_raw[s] = s
    assert_no_identity_collisions(raw_forms, source=source)
    out: Dict[str, float] = {}
    for s, q in last_qty.items():
        out[identity_key(s)] = q
    return out


def fold_qty_map_with_provenance(
    rows: Sequence[Tuple[Any, float]],
    *,
    source: str = "positions",
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Like fold_qty_map, plus identity -> last raw form (audit)."""
    qty = fold_qty_map(rows, source=source)
    provenance: Dict[str, str] = {}
    for raw, _q in rows:
        s = _strip_upper(raw)
        if not s:
            continue
        provenance[identity_key(s)] = s
    return qty, provenance
