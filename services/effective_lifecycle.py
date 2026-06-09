"""
Phase 152 — Exit logic refinement helpers.

Shared gates for position age protection, drag decay, rotation confirmation,
winner protection, and exit-confidence scoring. Consumed by manage_positions,
performance_risk_overlay, portfolio_allocation_engine, and apply_signal_lifecycle.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXIT_LOGIC_PATH = ROOT / "config" / "exit_logic.json"
DEFAULT_VALIDATION_PATH = ROOT / "data" / "results" / "exit_logic_refinement_validation.json"

PROTECTED_EXIT_KINDS = frozenset(
    {"ROTATE_EXIT", "TRIM_PRIORITY", "HIGH_DRAG", "HIGH_DRAG_PROTECTION"}
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    ts = str(ts).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            raw = ts[:26] if "." in ts else ts
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def default_exit_logic_dict() -> Dict[str, Any]:
    return {
        "enabled": True,
        "min_holding_hours_before_exit": 24.0,
        "rotation_min_edge_delta": 0.05,
        "drag_decay": {
            "enabled": True,
            "decay_hours": 168.0,
            "historical_weight_max": 1.0,
            "historical_weight_min": 0.2,
            "current_unrealized_boost": 0.4,
            "force_exit_adjusted_threshold": -50.0,
        },
        "winner_protection_enabled": True,
        "exit_confidence_enabled": True,
        "exit_confidence_trim_threshold": 0.45,
        "exit_confidence_exit_threshold": 0.65,
        "catastrophic_signal_delta_pct": -0.015,
        "catastrophic_exit_confidence_max": 0.35,
        "severe_force_exit_unrealized_pl_lt": -200.0,
        "severe_force_exit_total_pl_lt": -300.0,
    }


@dataclass
class DragDecayConfig:
    enabled: bool = True
    decay_hours: float = 168.0
    historical_weight_max: float = 1.0
    historical_weight_min: float = 0.2
    current_unrealized_boost: float = 0.4
    force_exit_adjusted_threshold: float = -50.0


@dataclass
class ExitLogicConfig:
    enabled: bool = True
    min_holding_hours_before_exit: float = 24.0
    rotation_min_edge_delta: float = 0.05
    drag_decay: DragDecayConfig = field(default_factory=DragDecayConfig)
    winner_protection_enabled: bool = True
    exit_confidence_enabled: bool = True
    exit_confidence_trim_threshold: float = 0.45
    exit_confidence_exit_threshold: float = 0.65
    catastrophic_signal_delta_pct: float = -0.015
    catastrophic_exit_confidence_max: float = 0.35
    severe_force_exit_unrealized_pl_lt: float = -200.0
    severe_force_exit_total_pl_lt: float = -300.0


@dataclass
class ExitLogicMetrics:
    position_age_blocks: int = 0
    rotation_blocks: int = 0
    winner_protections: int = 0
    drag_adjustments: int = 0
    exit_confidence_reviews: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "position_age_blocks": int(self.position_age_blocks),
            "rotation_blocks": int(self.rotation_blocks),
            "winner_protections": int(self.winner_protections),
            "drag_adjustments": int(self.drag_adjustments),
            "exit_confidence_reviews": int(self.exit_confidence_reviews),
        }


def exit_logic_from_dict(d: Mapping[str, Any]) -> ExitLogicConfig:
    base = default_exit_logic_dict()
    merged: Dict[str, Any] = dict(base)
    merged.update(dict(d))
    dd_raw = merged.get("drag_decay") or {}
    if not isinstance(dd_raw, dict):
        dd_raw = {}
    dd_base = dict(base["drag_decay"])
    dd_base.update(dd_raw)
    return ExitLogicConfig(
        enabled=bool(merged.get("enabled", True)),
        min_holding_hours_before_exit=float(merged["min_holding_hours_before_exit"]),
        rotation_min_edge_delta=float(merged["rotation_min_edge_delta"]),
        drag_decay=DragDecayConfig(
            enabled=bool(dd_base.get("enabled", True)),
            decay_hours=float(dd_base.get("decay_hours", 168.0)),
            historical_weight_max=float(dd_base.get("historical_weight_max", 1.0)),
            historical_weight_min=float(dd_base.get("historical_weight_min", 0.2)),
            current_unrealized_boost=float(dd_base.get("current_unrealized_boost", 0.4)),
            force_exit_adjusted_threshold=float(
                dd_base.get("force_exit_adjusted_threshold", -50.0)
            ),
        ),
        winner_protection_enabled=bool(merged.get("winner_protection_enabled", True)),
        exit_confidence_enabled=bool(merged.get("exit_confidence_enabled", True)),
        exit_confidence_trim_threshold=float(merged["exit_confidence_trim_threshold"]),
        exit_confidence_exit_threshold=float(merged["exit_confidence_exit_threshold"]),
        catastrophic_signal_delta_pct=float(merged["catastrophic_signal_delta_pct"]),
        catastrophic_exit_confidence_max=float(merged["catastrophic_exit_confidence_max"]),
        severe_force_exit_unrealized_pl_lt=float(merged["severe_force_exit_unrealized_pl_lt"]),
        severe_force_exit_total_pl_lt=float(merged["severe_force_exit_total_pl_lt"]),
    )


def load_exit_logic_config(path: Optional[Path] = None) -> ExitLogicConfig:
    p = path or DEFAULT_EXIT_LOGIC_PATH
    merged = default_exit_logic_dict()
    if p.is_file():
        try:
            raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(raw, dict):
                merged.update(raw)
        except Exception:
            pass
    return exit_logic_from_dict(merged)


def load_position_entry_times(results_dir: Path) -> Dict[str, datetime]:
    """Most recent filled BUY timestamp per symbol from order/trade logs."""
    out: Dict[str, datetime] = {}

    def _consider(sym: str, ts: Optional[datetime]) -> None:
        sym = str(sym or "").strip().upper()
        if not sym or ts is None:
            return
        prev = out.get(sym)
        if prev is None or ts > prev:
            out[sym] = ts

    def _read_csv(path: Path) -> List[Dict[str, str]]:
        if not path.is_file():
            return []
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []

    for row in _read_csv(results_dir / "live_orders_log.csv"):
        sym = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or "").lower()
        status = str(row.get("status") or "").lower()
        if side != "buy" or status != "filled":
            continue
        qty = _safe_float(row.get("filled_qty") or row.get("qty"))
        if qty <= 0:
            continue
        _consider(sym, _parse_ts(str(row.get("timestamp") or "")))

    for row in _read_csv(results_dir / "recent_orders.csv"):
        sym = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or row.get("type") or "").lower()
        status = str(row.get("status") or "").lower()
        if "buy" not in side or status != "filled":
            continue
        qty = _safe_float(row.get("filled_qty") or row.get("qty"))
        if qty <= 0:
            continue
        ts = str(row.get("filled_at") or row.get("updated_at") or row.get("submitted_at") or "")
        _consider(sym, _parse_ts(ts))

    for row in _read_csv(results_dir / "trade_log.csv"):
        sym = str(row.get("ticker") or row.get("symbol") or "").upper()
        action = str(row.get("action") or "").upper()
        if action != "BUY":
            continue
        qty = _safe_float(row.get("quantity"))
        if qty <= 0:
            continue
        date_s = str(row.get("date") or "")
        _consider(sym, _parse_ts(f"{date_s}T16:00:00Z" if date_s else ""))

    return out


def compute_hours_held(
    symbol: str,
    entry_times: Mapping[str, datetime],
    *,
    now: Optional[datetime] = None,
) -> float:
    sym = str(symbol or "").strip().upper()
    entry = entry_times.get(sym)
    if entry is None:
        return 0.0
    ref = now or _utc_now()
    if entry.tzinfo is None:
        entry = entry.replace(tzinfo=timezone.utc)
    delta = ref - entry
    return round(max(0.0, delta.total_seconds() / 3600.0), 4)


def compute_current_position_pl(
    *,
    qty: float,
    avg_entry_price: Optional[float],
    current_price: Optional[float],
    unrealized_pl: Optional[float] = None,
) -> float:
    if unrealized_pl is not None:
        return float(unrealized_pl)
    try:
        if qty > 0 and avg_entry_price and current_price:
            return round(float(qty) * (float(current_price) - float(avg_entry_price)), 4)
    except (TypeError, ValueError):
        pass
    return 0.0


def is_force_exit_flag(risk_flag: str) -> bool:
    return "FORCE_EXIT" in str(risk_flag or "").upper()


def is_severe_risk_condition(
    *,
    total_pl: Optional[float],
    unrealized_pl: Optional[float],
    risk_flag: str,
    config: ExitLogicConfig,
) -> bool:
    if is_force_exit_flag(risk_flag):
        return True
    if unrealized_pl is not None and unrealized_pl < config.severe_force_exit_unrealized_pl_lt:
        return True
    if total_pl is not None and total_pl < config.severe_force_exit_total_pl_lt:
        return True
    return False


def is_catastrophic_signal_deterioration(
    *,
    delta_pct: float,
    confidence: float,
    config: ExitLogicConfig,
) -> bool:
    if delta_pct <= config.catastrophic_signal_delta_pct:
        return True
    if confidence <= config.catastrophic_exit_confidence_max and delta_pct < -0.005:
        return True
    return False


def apply_drag_decay(
    *,
    historical_total_pl: float,
    current_unrealized_pl: float,
    hours_held: float,
    confidence: float,
    config: ExitLogicConfig,
) -> Tuple[float, float, float, bool]:
    """
    Weight current evidence over historical cumulative drag for newer positions.

    Returns (historical_weight, current_weight, adjusted_score, should_force_exit).
    """
    dd = config.drag_decay
    if not config.enabled or not dd.enabled:
        should = historical_total_pl < dd.force_exit_adjusted_threshold
        if current_unrealized_pl > 0 and historical_total_pl < 0:
            should = False
        return 1.0, 0.0, historical_total_pl, should

    decay_hours = max(1.0, float(dd.decay_hours))
    age_factor = min(1.0, max(0.0, hours_held / decay_hours))
    hist_w = (
        dd.historical_weight_min
        + (dd.historical_weight_max - dd.historical_weight_min) * age_factor
    )
    curr_boost = dd.current_unrealized_boost if current_unrealized_pl > 0 else 0.0
    curr_w = max(0.0, (1.0 - hist_w) + curr_boost)
    conf_penalty = max(0.0, (0.55 - confidence)) * abs(historical_total_pl) * 0.05
    adjusted = hist_w * historical_total_pl + curr_w * current_unrealized_pl - conf_penalty

    should_force = adjusted <= dd.force_exit_adjusted_threshold
    if current_unrealized_pl > 0 and historical_total_pl < 0:
        should_force = False
    if current_unrealized_pl >= 0 and hours_held < config.min_holding_hours_before_exit:
        should_force = False

    return round(hist_w, 6), round(curr_w, 6), round(adjusted, 4), bool(should_force)


def find_rotation_replacement(
    exited_sym: str,
    alloc_rows: Mapping[str, Mapping[str, Any]],
    alloc_summary: Mapping[str, Any],
    held_symbols: Set[str],
) -> Tuple[str, float, float, float, float]:
    """Return (candidate_sym, conf_delta, edge_delta, alloc_delta, candidate_alloc_score)."""
    exited = alloc_rows.get(exited_sym, {})
    exited_conf = _safe_float(exited.get("confidence"))
    exited_edge = _safe_float(exited.get("edge_score"))
    exited_alloc = _safe_float(exited.get("allocation_score"))

    candidate_sym = ""
    candidate_conf = 0.0
    candidate_edge = 0.0
    candidate_alloc = 0.0

    top_opps = alloc_summary.get("top_5_opportunities") or []
    for opp in top_opps:
        if not isinstance(opp, dict):
            continue
        ticker = str(opp.get("ticker") or "").upper()
        action = str(opp.get("recommended_action") or "").upper()
        if (
            ticker
            and ticker != exited_sym
            and ticker not in held_symbols
            and action
            in (
                "ADD",
                "BUY",
                "OPEN",
            )
        ):
            candidate_sym = ticker
            candidate_conf = _safe_float(opp.get("confidence"))
            candidate_edge = _safe_float(opp.get("edge_score"))
            candidate_alloc = _safe_float(opp.get("allocation_score"))
            break

    if not candidate_sym:
        best_score = -999.0
        for sym, row in alloc_rows.items():
            if sym == exited_sym or sym in held_symbols:
                continue
            action = str(row.get("recommended_action") or "").upper()
            if action not in ("ADD", "BUY", "OPEN", "WATCH"):
                continue
            score = _safe_float(row.get("allocation_score"))
            if score > best_score:
                best_score = score
                candidate_sym = sym
                candidate_conf = _safe_float(row.get("confidence"))
                candidate_edge = _safe_float(row.get("edge_score"))
                candidate_alloc = score

    conf_delta = round(candidate_conf - exited_conf, 6) if candidate_sym else 0.0
    edge_delta = round(candidate_edge - exited_edge, 6) if candidate_sym else 0.0
    alloc_delta = round(candidate_alloc - exited_alloc, 6) if candidate_sym else 0.0
    return candidate_sym, conf_delta, edge_delta, alloc_delta, candidate_alloc


def check_rotation_confirmation(
    *,
    symbol: str,
    candidate: str,
    edge_delta: float,
    conf_delta: float,
    alloc_delta: float,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
) -> bool:
    """Return True if rotation is allowed (replacement meaningfully better)."""
    if not config.enabled:
        return True
    required = float(config.rotation_min_edge_delta)
    composite = edge_delta + max(0.0, conf_delta) * 0.5 + max(0.0, alloc_delta) * 0.25
    if composite >= required and edge_delta > 0:
        return True
    metrics.rotation_blocks += 1
    print(
        f"[ROTATION_BLOCK] symbol={symbol} candidate={candidate or 'NONE'} "
        f"edge_delta={edge_delta} required_delta={required} "
        f"reason=INSUFFICIENT_IMPROVEMENT",
        flush=True,
    )
    return False


def check_position_age_protection(
    *,
    symbol: str,
    hours_held: float,
    exit_kind: str,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
    emergency: bool,
) -> bool:
    """Return True if exit is allowed."""
    if not config.enabled or emergency:
        return True
    if exit_kind not in PROTECTED_EXIT_KINDS and exit_kind != "TRIM_PRIORITY":
        return True
    minimum = float(config.min_holding_hours_before_exit)
    if hours_held >= minimum:
        return True
    metrics.position_age_blocks += 1
    print(
        f"[POSITION_AGE_PROTECTION] symbol={symbol} hours_held={hours_held} "
        f"minimum_required={minimum} exit_blocked=true",
        flush=True,
    )
    return False


def check_winner_protection(
    *,
    symbol: str,
    current_pl: float,
    exit_kind: str,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
    emergency: bool,
) -> bool:
    """Return True if exit is allowed."""
    if not config.enabled or not config.winner_protection_enabled or emergency:
        return True
    if exit_kind not in PROTECTED_EXIT_KINDS and exit_kind != "TRIM_PRIORITY":
        return True
    if current_pl <= 0:
        return True
    metrics.winner_protections += 1
    print(
        f"[WINNER_PROTECTION] symbol={symbol} current_pl={current_pl} "
        f"exit_blocked=true reason=POSITIVE_POSITION",
        flush=True,
    )
    return False


def compute_exit_confidence(
    *,
    hours_held: float,
    current_pl: float,
    risk_flag: str,
    delta_pct: float,
    confidence: float,
    replacement_edge_delta: float,
    drag_score: float,
    config: ExitLogicConfig,
) -> Tuple[float, str]:
    score = 0.0
    min_h = float(config.min_holding_hours_before_exit)
    if hours_held >= min_h:
        score += 0.2
    elif hours_held >= min_h * 0.5:
        score += 0.1

    if current_pl < -100:
        score += 0.25
    elif current_pl < 0:
        score += 0.12
    elif current_pl > 0:
        score -= 0.18

    if is_force_exit_flag(risk_flag):
        score += 0.35
    elif "TRIM_PRIORITY" in str(risk_flag or "").upper():
        score += 0.12

    if delta_pct <= config.catastrophic_signal_delta_pct:
        score += 0.28
    elif delta_pct < -0.002:
        score += 0.08

    if confidence <= config.catastrophic_exit_confidence_max:
        score += 0.12

    if replacement_edge_delta > config.rotation_min_edge_delta:
        score += 0.1
    elif replacement_edge_delta <= 0:
        score -= 0.08

    if drag_score < -50:
        score += 0.12

    score = max(0.0, min(1.0, round(score, 4)))
    if score >= config.exit_confidence_exit_threshold:
        rec = "EXIT"
    elif score >= config.exit_confidence_trim_threshold:
        rec = "TRIM"
    else:
        rec = "HOLD"
    return score, rec


def advisory_exit_confidence_allows(
    *,
    symbol: str,
    recommendation: str,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
    emergency: bool,
    exit_kind: str,
    score: float,
) -> bool:
    """Advisory gate: block protected non-emergency exits when confidence says HOLD."""
    if not config.enabled or not config.exit_confidence_enabled or emergency:
        return True
    if exit_kind not in PROTECTED_EXIT_KINDS and exit_kind != "TRIM_PRIORITY":
        return True
    metrics.exit_confidence_reviews += 1
    print(
        f"[EXIT_CONFIDENCE] symbol={symbol} score={score} recommendation={recommendation}",
        flush=True,
    )
    return recommendation != "HOLD"


def evaluate_protected_exit(
    *,
    symbol: str,
    exit_kind: str,
    hours_held: float,
    current_pl: float,
    total_pl: float,
    unrealized_pl: float,
    risk_flag: str,
    delta_pct: float,
    confidence: float,
    replacement_edge_delta: float,
    drag_score: float,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
    force_exit_active: bool = False,
) -> bool:
    """
    Combined gate for protected exit paths. Returns True when exit may proceed.
    """
    if not config.enabled:
        return True

    emergency = (
        force_exit_active
        or is_severe_risk_condition(
            total_pl=total_pl,
            unrealized_pl=unrealized_pl,
            risk_flag=risk_flag,
            config=config,
        )
        or is_catastrophic_signal_deterioration(
            delta_pct=delta_pct, confidence=confidence, config=config
        )
    )

    if not check_position_age_protection(
        symbol=symbol,
        hours_held=hours_held,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        return False

    if not check_winner_protection(
        symbol=symbol,
        current_pl=current_pl,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        return False

    score, rec = compute_exit_confidence(
        hours_held=hours_held,
        current_pl=current_pl,
        risk_flag=risk_flag,
        delta_pct=delta_pct,
        confidence=confidence,
        replacement_edge_delta=replacement_edge_delta,
        drag_score=drag_score,
        config=config,
    )
    return advisory_exit_confidence_allows(
        symbol=symbol,
        recommendation=rec,
        config=config,
        metrics=metrics,
        emergency=emergency,
        exit_kind=exit_kind,
        score=score,
    )


def write_validation_json(metrics: ExitLogicMetrics, path: Optional[Path] = None) -> Path:
    out = path or DEFAULT_VALIDATION_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.to_dict()
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, out)
    return out
