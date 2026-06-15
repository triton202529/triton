"""
authority_labels.py
Phase 195F — Operator-facing authority label constants (presentation only).

These labels clarify artifact tier in dashboards, reports, and advisory outputs.
They do not change execution, lifecycle, allocation, or risk decision behavior.
"""

from __future__ import annotations

from typing import Any, Dict

RAW_SIGNAL_LIFECYCLE = "RAW_SIGNAL_LIFECYCLE"
EFFECTIVE_LIFECYCLE_AUTHORITY = "EFFECTIVE_LIFECYCLE_AUTHORITY"
ADAPTIVE_RISK_INTELLIGENCE = "ADAPTIVE_RISK_INTELLIGENCE"
MASTER_EXECUTION_GATE_AUTHORITY = "MASTER_EXECUTION_GATE_AUTHORITY"

RAW_LIFECYCLE_SUBTITLE = "Advisory only — not execution authority"
EFFECTIVE_LIFECYCLE_SUBTITLE = "Operational lifecycle authority"
ADAPTIVE_RISK_SUBTITLE = "Supporting risk intelligence only — not final execution authority"
MASTER_GATE_SUBTITLE = "Final execution-risk authority"

RAW_LIFECYCLE_SOURCE = "data/results/signal_lifecycle.csv"
EFFECTIVE_LIFECYCLE_SOURCE = "data/results/signal_lifecycle_effective.csv"
ADAPTIVE_RISK_SOURCE = "data/results/adaptive_risk_state.json"
MASTER_GATE_SOURCE = "data/results/master_execution_gate.json"


def lifecycle_raw_metadata() -> Dict[str, Any]:
    return {
        "authority_label": RAW_SIGNAL_LIFECYCLE,
        "authority_tier": "advisory",
        "execution_authority": False,
        "source": RAW_LIFECYCLE_SOURCE,
        "subtitle": RAW_LIFECYCLE_SUBTITLE,
    }


def lifecycle_effective_metadata() -> Dict[str, Any]:
    return {
        "authority_label": EFFECTIVE_LIFECYCLE_AUTHORITY,
        "authority_tier": "operational",
        "execution_authority": True,
        "source": EFFECTIVE_LIFECYCLE_SOURCE,
        "subtitle": EFFECTIVE_LIFECYCLE_SUBTITLE,
    }


def adaptive_risk_metadata() -> Dict[str, Any]:
    return {
        "authority_label": ADAPTIVE_RISK_INTELLIGENCE,
        "authority_tier": "advisory",
        "execution_authority": False,
        "source": ADAPTIVE_RISK_SOURCE,
        "subtitle": ADAPTIVE_RISK_SUBTITLE,
    }


def master_gate_metadata() -> Dict[str, Any]:
    return {
        "authority_label": MASTER_EXECUTION_GATE_AUTHORITY,
        "authority_tier": "operational",
        "execution_authority": True,
        "source": MASTER_GATE_SOURCE,
        "subtitle": MASTER_GATE_SUBTITLE,
    }


def authority_label_block() -> Dict[str, Any]:
    """Mixed-view authority hierarchy for operator-facing JSON summaries."""
    return {
        "raw_signal_lifecycle": lifecycle_raw_metadata(),
        "effective_lifecycle_authority": lifecycle_effective_metadata(),
        "adaptive_risk_intelligence": adaptive_risk_metadata(),
        "master_execution_gate_authority": master_gate_metadata(),
    }
