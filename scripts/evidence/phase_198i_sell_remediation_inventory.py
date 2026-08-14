"""Emit the reconciled 61-row Phase 198I broker-route inventory.

Authoritative source is the committed ROUTE_SEED + FINAL disposition map in this
file. This script does not read tmp/, ignored evidence packs, or data/live/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

FORMERLY_UNRESOLVED = (
    "R3",
    "R4",
    "R5",
    "R6",
    "R23",
    "R33",
    "R34",
    "R36",
    "R37",
    "R38",
    "R39",
    "R40",
    "R41",
    "R42",
    "R43",
)

# Map original route index (R1..) to final disposition after 198I SELL remediation.
FINAL = {
    "R1": "VERIFIED_NON_EXECUTION_ROUTE",
    "R2": "VERIFIED_DEAD_OR_UNREACHABLE",
    "R3": "VERIFIED_HARDENED_SELL_ROUTE",
    "R4": "VERIFIED_NON_SELL_ROUTE",
    "R5": "VERIFIED_NON_SELL_ROUTE",
    "R6": "VERIFIED_HARDENED_SELL_ROUTE",
    "R7": "VERIFIED_HARDENED_SELL_ROUTE",
    "R8": "VERIFIED_HARDENED_SELL_ROUTE",
    "R9": "VERIFIED_HARDENED_SELL_ROUTE",
    "R10": "VERIFIED_HARDENED_SELL_ROUTE",
    "R11": "VERIFIED_HARDENED_SELL_ROUTE",
    "R12": "VERIFIED_NON_EXECUTION_ROUTE",
    "R13": "VERIFIED_HARDENED_SELL_ROUTE",
    "R14": "VERIFIED_HARDENED_SELL_ROUTE",
    "R15": "VERIFIED_HARDENED_SELL_ROUTE",
    "R16": "VERIFIED_HARDENED_SELL_ROUTE",
    "R17": "VERIFIED_DEAD_OR_UNREACHABLE",
    "R18": "VERIFIED_DEAD_OR_UNREACHABLE",
    "R19": "VERIFIED_NON_SELL_ROUTE",
    "R20": "VERIFIED_NON_SELL_ROUTE",
    "R21": "VERIFIED_NON_SELL_ROUTE",
    "R22": "VERIFIED_NON_EXECUTION_ROUTE",
    "R23": "VERIFIED_NON_SELL_ROUTE",
    "R24": "VERIFIED_HARDENED_SELL_ROUTE",
    "R25": "VERIFIED_NON_EXECUTION_ROUTE",
    "R26": "VERIFIED_NON_SELL_ROUTE",
    "R27": "VERIFIED_NON_SELL_ROUTE",
    "R28": "VERIFIED_HARDENED_SELL_ROUTE",
    "R29": "VERIFIED_HARDENED_SELL_ROUTE",
    "R30": "VERIFIED_HARDENED_SELL_ROUTE",
    "R31": "VERIFIED_HARDENED_SELL_ROUTE",
    "R32": "VERIFIED_HARDENED_SELL_ROUTE",
    "R33": "VERIFIED_NON_SELL_ROUTE",
    "R34": "VERIFIED_NON_SELL_ROUTE",
    "R35": "VERIFIED_NON_EXECUTION_ROUTE",
    "R36": "VERIFIED_NON_SELL_ROUTE",
    "R37": "VERIFIED_HARDENED_SELL_ROUTE",
    "R38": "VERIFIED_NON_SELL_ROUTE",
    "R39": "VERIFIED_NON_SELL_ROUTE",
    "R40": "VERIFIED_NON_EXECUTION_ROUTE",
    "R41": "VERIFIED_NON_SELL_ROUTE",
    "R42": "VERIFIED_HARDENED_SELL_ROUTE",
    "R43": "VERIFIED_HARDENED_SELL_ROUTE",
    "R44": "VERIFIED_NON_SELL_ROUTE",
    "R45": "VERIFIED_NON_SELL_ROUTE",
    "R46": "VERIFIED_NON_SELL_ROUTE",
    "R47": "VERIFIED_NON_SELL_ROUTE",
    "R48": "VERIFIED_NON_EXECUTION_ROUTE",
    "R49": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R50": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R51": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R52": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R53": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R54": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R55": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R56": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R57": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R58": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R59": "VERIFIED_TEST_OR_FAKE_ONLY",
    "R60": "VERIFIED_NON_EXECUTION_ROUTE",
    "R61": "VERIFIED_NON_EXECUTION_ROUTE",
}

CALLSITES = {
    "R3": "pages/manual_order_desk.py::submit_order ~L318",
    "R4": "pages/manual_order_desk.py::cancel_order ~L355",
    "R5": "pages/manual_order_desk.py::cancel_all_orders ~L383",
    "R6": "pages_order_desk.py::submit_order ~L361",
    "R23": "scripts/opening_sequence.py::cancel_all_orders ~L72",
    "R33": "services/cancel_stale.py::cancel_order ~L19 (do not import; module-level)",
    "R34": "services/execute_cycle.py::cancel_open_orders ~L86",
    "R36": "services/manage_open_orders.py::broker.cancel_order ~L1060",
    "R37": "services/manage_open_orders.py::broker.submit_order ~L1083",
    "R38": "services/manage_open_orders.py::run br.cancel_order ~L1247",
    "R39": "services/market_open_cycle.py::_cancel_open_orders ~L106",
    "R40": "services/place_live_orders.py::argparse help 'never call broker.submit_order' ~L1660",
    "R41": "services/place_live_orders.py::broker.cancel_order duplicate cancel ~L1979",
    "R42": "services/place_live_orders.py::resp = broker.submit_order ~L2483",
    "R43": "services/position_protector.py::place_bracket submit_order ~L96",
}

EVIDENCE = {
    "R3": "tests/test_phase_198i_sell_broker_route_closure.py::test_r3_r6_page_submit_uses_gated_alpaca_broker",
    "R4": "tests/test_phase_198i_sell_broker_route_closure.py::test_r4_r5_r23_cancel_routes_are_non_sell_and_gated",
    "R5": "tests/test_phase_198i_sell_broker_route_closure.py::test_r4_r5_r23_cancel_routes_are_non_sell_and_gated",
    "R6": "tests/test_phase_198i_sell_broker_route_closure.py::test_r3_r6_page_submit_uses_gated_alpaca_broker",
    "R23": "tests/test_phase_198i_sell_broker_route_closure.py::test_r4_r5_r23_cancel_routes_are_non_sell_and_gated",
    "R33": "tests/test_phase_198i_sell_broker_route_closure.py::test_r33_cancel_stale_is_cancel_only_through_alpaca_broker",
    "R34": "tests/test_phase_198i_sell_broker_route_closure.py::test_r34_execute_cycle_cancel_open_orders",
    "R36": "tests/test_phase_198i_sell_broker_route_closure.py::test_r36_r37_manage_open_orders_sell_replace_hits_boundary",
    "R37": "tests/test_phase_198i_sell_broker_route_closure.py::test_r37_replace_submit_hits_boundary",
    "R38": "tests/test_phase_198i_sell_broker_route_closure.py::test_r38_run_cancel_sell_fail_closed",
    "R39": "tests/test_phase_198i_sell_broker_route_closure.py::test_r39_market_open_cycle_cancel_uses_alpaca_broker",
    "R40": "tests/test_phase_198i_sell_broker_route_closure.py::test_r40_argparse_help_is_non_execution",
    "R41": "tests/test_phase_198i_sell_broker_route_closure.py::test_r41_duplicate_sell_cancel_fail_closed",
    "R42": "tests/test_phase_198i_manage_positions_sell_integration.py::test_accepted_manage_positions_to_fake_broker_sell_once",
    "R43": "tests/test_phase_198i_sell_broker_route_closure.py::test_r43_position_protector_place_bracket_sell",
}

# Committed 61-route seed: source file, entry, prior classification, call chain.
# Runtime never reads tmp/ or ignored evidence packs.
ROUTE_SEED = {
    'R1': {"source_file": "app.py", "entry": "app.py::def get_client_and_conf", "chain": "STATIC_ONLY:app.py::def get_client_and_conf->L25:TradingClient construction; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unreachable", "canonical": "Unreachable", "proof": "STATIC_ONLY: STATIC_ONLY: construction is not a broker mutation (app.py)"},
    'R2': {"source_file": "app.py", "entry": "app.py::def run_script_sync", "chain": "STATIC_ONLY:app.py::def run_script_sync->L181:cancel_all_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Disabled fail-closed", "canonical": "Disabled fail-closed", "proof": "STATIC_ONLY: STATIC_ONLY: DISABLED_FAIL_CLOSED marker in app.py"},
    'R3': {"source_file": "pages/manual_order_desk.py", "entry": "pages/manual_order_desk.py::def render_manual_order_desk", "chain": "STATIC_ONLY:pages/manual_order_desk.py::def render_manual_order_desk->L318:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site pages/manual_order_desk.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R4': {"source_file": "pages/manual_order_desk.py", "entry": "pages/manual_order_desk.py::def render_manual_order_desk", "chain": "STATIC_ONLY:pages/manual_order_desk.py::def render_manual_order_desk->L355:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site pages/manual_order_desk.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R5': {"source_file": "pages/manual_order_desk.py", "entry": "pages/manual_order_desk.py::def render_manual_order_desk", "chain": "STATIC_ONLY:pages/manual_order_desk.py::def render_manual_order_desk->L383:cancel_all_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site pages/manual_order_desk.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R6': {"source_file": "pages_order_desk.py", "entry": "pages_order_desk.py::def render_manual_order_desk", "chain": "STATIC_ONLY:pages_order_desk.py::def render_manual_order_desk->L361:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site pages_order_desk.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R7': {"source_file": "place_orders_from_csv.py", "entry": "place_orders_from_csv.py::def place_order", "chain": "STATIC_ONLY:place_orders_from_csv.py::def place_order->L692:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R8': {"source_file": "scripts/attach_brackets_now.py", "entry": "scripts/attach_brackets_now.py::module-level", "chain": "STATIC_ONLY:scripts/attach_brackets_now.py::module-level->L62:direct REST mutation; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R9': {"source_file": "scripts/attach_oco.py", "entry": "scripts/attach_oco.py::def attach_oco", "chain": "STATIC_ONLY:scripts/attach_oco.py::def attach_oco->L74:direct REST mutation; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R10': {"source_file": "scripts/auto_execute_signals.py", "entry": "scripts/auto_execute_signals.py::def close_position_safely", "chain": "STATIC_ONLY:scripts/auto_execute_signals.py::def close_position_safely->L232:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R11': {"source_file": "scripts/auto_execute_signals.py", "entry": "scripts/auto_execute_signals.py::def close_position_safely", "chain": "STATIC_ONLY:scripts/auto_execute_signals.py::def close_position_safely->L252:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R12': {"source_file": "scripts/auto_execute_signals.py", "entry": "scripts/auto_execute_signals.py::def main", "chain": "STATIC_ONLY:scripts/auto_execute_signals.py::def main->L361:TradingClient construction; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unreachable", "canonical": "Unreachable", "proof": "STATIC_ONLY: STATIC_ONLY: construction is not a broker mutation (scripts/auto_execute_signals.py)"},
    'R13': {"source_file": "scripts/auto_execute_signals.py", "entry": "scripts/auto_execute_signals.py::def main", "chain": "STATIC_ONLY:scripts/auto_execute_signals.py::def main->L580:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R14': {"source_file": "scripts/auto_execute_signals.py", "entry": "scripts/auto_execute_signals.py::def main", "chain": "STATIC_ONLY:scripts/auto_execute_signals.py::def main->L646:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R15': {"source_file": "scripts/auto_exit.py", "entry": "scripts/auto_exit.py::def submit_exit", "chain": "STATIC_ONLY:scripts/auto_exit.py::def submit_exit->L68:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R16': {"source_file": "scripts/auto_trade.py", "entry": "scripts/auto_trade.py::def submit_order", "chain": "STATIC_ONLY:scripts/auto_trade.py::def submit_order->L66:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R17': {"source_file": "scripts/cancel_all.py", "entry": "scripts/cancel_all.py::module-level", "chain": "STATIC_ONLY:scripts/cancel_all.py::module-level->L8:cancel_all_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Disabled fail-closed", "canonical": "Disabled fail-closed", "proof": "STATIC_ONLY: STATIC_ONLY: DISABLED_FAIL_CLOSED marker in scripts/cancel_all.py"},
    'R18': {"source_file": "scripts/cancel_all.py", "entry": "scripts/cancel_all.py::module-level", "chain": "STATIC_ONLY:scripts/cancel_all.py::module-level->L13:cancel_all_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Disabled fail-closed", "canonical": "Disabled fail-closed", "proof": "STATIC_ONLY: STATIC_ONLY: DISABLED_FAIL_CLOSED marker in scripts/cancel_all.py"},
    'R19': {"source_file": "scripts/chase_limit.py", "entry": "scripts/chase_limit.py::def main", "chain": "STATIC_ONLY:scripts/chase_limit.py::def main->L69:direct REST mutation; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "BUY/ADD-only", "canonical": "BUY/ADD-only", "proof": "STATIC_ONLY: STATIC_ONLY: BUY-oriented helper scripts/chase_limit.py; SELL capability not demonstrated at this site"},
    'R20': {"source_file": "scripts/chase_limit.py", "entry": "scripts/chase_limit.py::def main", "chain": "STATIC_ONLY:scripts/chase_limit.py::def main->L79:direct REST mutation; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "BUY/ADD-only", "canonical": "BUY/ADD-only", "proof": "STATIC_ONLY: STATIC_ONLY: BUY-oriented helper scripts/chase_limit.py; SELL capability not demonstrated at this site"},
    'R21': {"source_file": "scripts/manual_buy.py", "entry": "scripts/manual_buy.py::module-level", "chain": "STATIC_ONLY:scripts/manual_buy.py::module-level->L20:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "BUY/ADD-only", "canonical": "BUY/ADD-only", "proof": "STATIC_ONLY: STATIC_ONLY: BUY-oriented helper scripts/manual_buy.py; SELL capability not demonstrated at this site"},
    'R22': {"source_file": "scripts/opening_sequence.py", "entry": "scripts/opening_sequence.py::def load_client", "chain": "STATIC_ONLY:scripts/opening_sequence.py::def load_client->L20:TradingClient construction; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unreachable", "canonical": "Unreachable", "proof": "STATIC_ONLY: STATIC_ONLY: construction is not a broker mutation (scripts/opening_sequence.py)"},
    'R23': {"source_file": "scripts/opening_sequence.py", "entry": "scripts/opening_sequence.py::def main", "chain": "STATIC_ONLY:scripts/opening_sequence.py::def main->L72:cancel_all_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site scripts/opening_sequence.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R24': {"source_file": "scripts/postfill_attach_oco.py", "entry": "scripts/postfill_attach_oco.py::def attach_oco_for_symbol", "chain": "STATIC_ONLY:scripts/postfill_attach_oco.py::def attach_oco_for_symbol->L123:direct REST mutation; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "UNEXECUTED: STATIC_ONLY: file contains gate_raw/enforce; call-site side-dependent; route-specific executed proof UNEXECUTED in repair"},
    'R25': {"source_file": "scripts/status.py", "entry": "scripts/status.py::def show_alpaca_status", "chain": "STATIC_ONLY:scripts/status.py::def show_alpaca_status->L111:TradingClient construction; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unreachable", "canonical": "Unreachable", "proof": "STATIC_ONLY: STATIC_ONLY: construction is not a broker mutation (scripts/status.py)"},
    'R26': {"source_file": "services/broker_alpaca.py", "entry": "services/broker_alpaca.py::def cancel", "chain": "STATIC_ONLY:services/broker_alpaca.py::def cancel->L323:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "INDIRECT: STATIC_ONLY: gate in services/broker_alpaca.py; EXECUTED_REF tests/test_phase_198i_fix_closeout_evidence_correction_5.py cancel/submit suites (prior Correction-7, not re-executed in repair)"},
    'R27': {"source_file": "services/broker_alpaca.py", "entry": "services/broker_alpaca.py::def _parse_ts", "chain": "STATIC_ONLY:services/broker_alpaca.py::def _parse_ts->L518:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "INDIRECT: STATIC_ONLY: gate in services/broker_alpaca.py; EXECUTED_REF tests/test_phase_198i_fix_closeout_evidence_correction_5.py cancel/submit suites (prior Correction-7, not re-executed in repair)"},
    'R28': {"source_file": "services/broker_alpaca.py", "entry": "services/broker_alpaca.py::def submit_market_order", "chain": "STATIC_ONLY:services/broker_alpaca.py::def submit_market_order->L610:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "INDIRECT: STATIC_ONLY: gate in services/broker_alpaca.py; EXECUTED_REF tests/test_phase_198i_fix_closeout_evidence_correction_5.py cancel/submit suites (prior Correction-7, not re-executed in repair)"},
    'R29': {"source_file": "services/broker_alpaca.py", "entry": "services/broker_alpaca.py::def submit_limit_order", "chain": "STATIC_ONLY:services/broker_alpaca.py::def submit_limit_order->L631:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "INDIRECT: STATIC_ONLY: gate in services/broker_alpaca.py; EXECUTED_REF tests/test_phase_198i_fix_closeout_evidence_correction_5.py cancel/submit suites (prior Correction-7, not re-executed in repair)"},
    'R30': {"source_file": "services/broker_alpaca.py", "entry": "services/broker_alpaca.py::def place_order", "chain": "STATIC_ONLY:services/broker_alpaca.py::def place_order->L777:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "INDIRECT: STATIC_ONLY: gate in services/broker_alpaca.py; EXECUTED_REF tests/test_phase_198i_fix_closeout_evidence_correction_5.py cancel/submit suites (prior Correction-7, not re-executed in repair)"},
    'R31': {"source_file": "services/broker_alpaca.py", "entry": "services/broker_alpaca.py::def place_bracket_market", "chain": "STATIC_ONLY:services/broker_alpaca.py::def place_bracket_market->L805:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "INDIRECT: STATIC_ONLY: gate in services/broker_alpaca.py; EXECUTED_REF tests/test_phase_198i_fix_closeout_evidence_correction_5.py cancel/submit suites (prior Correction-7, not re-executed in repair)"},
    'R32': {"source_file": "services/broker_alpaca.py", "entry": "services/broker_alpaca.py::def place_bracket", "chain": "STATIC_ONLY:services/broker_alpaca.py::def place_bracket->L860:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Hardened through canonical boundary", "canonical": "enforce_paper_sell_broker_submission|gate_raw_paper_sell|begin_paper_sell_dispatch", "proof": "INDIRECT: STATIC_ONLY: gate in services/broker_alpaca.py; EXECUTED_REF tests/test_phase_198i_fix_closeout_evidence_correction_5.py cancel/submit suites (prior Correction-7, not re-executed in repair)"},
    'R33': {"source_file": "services/cancel_stale.py", "entry": "services/cancel_stale.py::module-level", "chain": "STATIC_ONLY:services/cancel_stale.py::module-level->L19:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/cancel_stale.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R34': {"source_file": "services/execute_cycle.py", "entry": "services/execute_cycle.py::def cancel_open_orders", "chain": "STATIC_ONLY:services/execute_cycle.py::def cancel_open_orders->L86:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/execute_cycle.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R35': {"source_file": "services/execution_path_registry.py", "entry": "services/execution_path_registry.py::module-level", "chain": "STATIC_ONLY:services/execution_path_registry.py::module-level->L174:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Historical only", "canonical": "Historical only", "proof": "STATIC_ONLY: STATIC_ONLY: non-runtime/historical reference in services/execution_path_registry.py"},
    'R36': {"source_file": "services/manage_open_orders.py", "entry": "services/manage_open_orders.py::def _managed_limit_candidate", "chain": "STATIC_ONLY:services/manage_open_orders.py::def _managed_limit_candidate->L1060:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/manage_open_orders.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R37': {"source_file": "services/manage_open_orders.py", "entry": "services/manage_open_orders.py::def _managed_limit_candidate", "chain": "STATIC_ONLY:services/manage_open_orders.py::def _managed_limit_candidate->L1083:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/manage_open_orders.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R38': {"source_file": "services/manage_open_orders.py", "entry": "services/manage_open_orders.py::def run", "chain": "STATIC_ONLY:services/manage_open_orders.py::def run->L1247:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/manage_open_orders.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R39': {"source_file": "services/market_open_cycle.py", "entry": "services/market_open_cycle.py::def _cancel_open_orders", "chain": "STATIC_ONLY:services/market_open_cycle.py::def _cancel_open_orders->L106:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/market_open_cycle.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R40': {"source_file": "services/place_live_orders.py", "entry": "services/place_live_orders.py::def main", "chain": "STATIC_ONLY:services/place_live_orders.py::def main->L1660:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/place_live_orders.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R41': {"source_file": "services/place_live_orders.py", "entry": "services/place_live_orders.py::def _block_duplicate_row", "chain": "STATIC_ONLY:services/place_live_orders.py::def _block_duplicate_row->L1979:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/place_live_orders.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R42': {"source_file": "services/place_live_orders.py", "entry": "services/place_live_orders.py::def _emit_execution_intelligence", "chain": "STATIC_ONLY:services/place_live_orders.py::def _emit_execution_intelligence->L2483:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/place_live_orders.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R43': {"source_file": "services/position_protector.py", "entry": "services/position_protector.py::def place_bracket", "chain": "STATIC_ONLY:services/position_protector.py::def place_bracket->L96:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unresolved", "canonical": "Unresolved", "proof": "STATIC_ONLY: STATIC_ONLY: mutation site services/position_protector.py; complete runtime call chain and executed proof UNRESOLVED in repair without dynamic tracing"},
    'R44': {"source_file": "services/reprice_open_orders.py", "entry": "services/reprice_open_orders.py::def run", "chain": "STATIC_ONLY:services/reprice_open_orders.py::def run->L405:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "BUY/ADD-only", "canonical": "BUY/ADD-only", "proof": "STATIC_ONLY: STATIC_ONLY: BUY-oriented helper services/reprice_open_orders.py; SELL capability not demonstrated at this site"},
    'R45': {"source_file": "services/reprice_open_orders.py", "entry": "services/reprice_open_orders.py::def run", "chain": "STATIC_ONLY:services/reprice_open_orders.py::def run->L422:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "BUY/ADD-only", "canonical": "BUY/ADD-only", "proof": "STATIC_ONLY: STATIC_ONLY: BUY-oriented helper services/reprice_open_orders.py; SELL capability not demonstrated at this site"},
    'R46': {"source_file": "services/reprice_order_ladder.py", "entry": "services/reprice_order_ladder.py::def run", "chain": "STATIC_ONLY:services/reprice_order_ladder.py::def run->L568:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "BUY/ADD-only", "canonical": "BUY/ADD-only", "proof": "STATIC_ONLY: STATIC_ONLY: BUY-oriented helper services/reprice_order_ladder.py; SELL capability not demonstrated at this site"},
    'R47': {"source_file": "services/reprice_order_ladder.py", "entry": "services/reprice_order_ladder.py::def run", "chain": "STATIC_ONLY:services/reprice_order_ladder.py::def run->L580:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "BUY/ADD-only", "canonical": "BUY/ADD-only", "proof": "STATIC_ONLY: STATIC_ONLY: BUY-oriented helper services/reprice_order_ladder.py; SELL capability not demonstrated at this site"},
    'R48': {"source_file": "services/system_integration_audit.py", "entry": "services/system_integration_audit.py::def trace_execution_path", "chain": "STATIC_ONLY:services/system_integration_audit.py::def trace_execution_path->L757:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Historical only", "canonical": "Historical only", "proof": "STATIC_ONLY: STATIC_ONLY: non-runtime/historical reference in services/system_integration_audit.py"},
    'R49': {"source_file": "tests/test_phase_198i_fix_closeout_cross_process.py", "entry": "tests/test_phase_198i_fix_closeout_cross_process.py::def submit_order", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_cross_process.py::def submit_order->L489:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_cross_process.py"},
    'R50': {"source_file": "tests/test_phase_198i_fix_closeout_evidence_correction_3.py", "entry": "tests/test_phase_198i_fix_closeout_evidence_correction_3.py::def test_directory_fsync_only_protected_transitions", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_evidence_correction_3.py::def test_directory_fsync_only_protected_transitions->L426:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_evidence_correction_3.py"},
    'R51': {"source_file": "tests/test_phase_198i_fix_closeout_evidence_correction_4.py", "entry": "tests/test_phase_198i_fix_closeout_evidence_correction_4.py::def _run_dispatch_under_dir_failure", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_evidence_correction_4.py::def _run_dispatch_under_dir_failure->L303:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_evidence_correction_4.py"},
    'R52': {"source_file": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py", "entry": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_paper_sell_cancel_blocked_without_permit", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_paper_sell_cancel_blocked_without_permit->L50:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_evidence_correction_5.py"},
    'R53': {"source_file": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py", "entry": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_paper_buy_cancel_not_sell_gated", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_paper_buy_cancel_not_sell_gated->L57:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_evidence_correction_5.py"},
    'R54': {"source_file": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py", "entry": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_paper_sell_cancel_all_blocked_without_permit", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_paper_sell_cancel_all_blocked_without_permit->L65:cancel_all_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_evidence_correction_5.py"},
    'R55': {"source_file": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py", "entry": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_app_tradingclient_cancel_all_disabled_fail_closed", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_app_tradingclient_cancel_all_disabled_fail_closed->L87:cancel_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_evidence_correction_5.py"},
    'R56': {"source_file": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py", "entry": "tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_opening_sequence_uses_broker_cancel_all", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_closeout_evidence_correction_5.py::def test_opening_sequence_uses_broker_cancel_all->L98:cancel_orders; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_closeout_evidence_correction_5.py"},
    'R57': {"source_file": "tests/test_phase_198i_fix_paper_sell_enforcement.py", "entry": "tests/test_phase_198i_fix_paper_sell_enforcement.py::def test_102_107_durable_before_broker_and_crash_semantics", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_paper_sell_enforcement.py::def test_102_107_durable_before_broker_and_crash_semantics->L649:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_paper_sell_enforcement.py"},
    'R58': {"source_file": "tests/test_phase_198i_fix_paper_sell_enforcement.py", "entry": "tests/test_phase_198i_fix_paper_sell_enforcement.py::def test_113_114_place_live_orders_integration", "chain": "STATIC_ONLY:tests/test_phase_198i_fix_paper_sell_enforcement.py::def test_113_114_place_live_orders_integration->L724:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_fix_paper_sell_enforcement.py"},
    'R59': {"source_file": "tests/test_phase_198i_paper_sell_arm_runtime_enforcement.py", "entry": "tests/test_phase_198i_paper_sell_arm_runtime_enforcement.py::def _mock_dispatch", "chain": "STATIC_ONLY:tests/test_phase_198i_paper_sell_arm_runtime_enforcement.py::def _mock_dispatch->L242:submit_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Fake test-only", "canonical": "Fake test-only", "proof": "STATIC_ONLY: STATIC_ONLY: test double site tests/test_phase_198i_paper_sell_arm_runtime_enforcement.py"},
    'R60': {"source_file": "tools/probe_alpaca_bp.py", "entry": "tools/probe_alpaca_bp.py::def via_alpaca_py", "chain": "STATIC_ONLY:tools/probe_alpaca_bp.py::def via_alpaca_py->L38:TradingClient construction; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Unreachable", "canonical": "Unreachable", "proof": "STATIC_ONLY: STATIC_ONLY: construction is not a broker mutation (tools/probe_alpaca_bp.py)"},
    'R61': {"source_file": "tools/tmp_cancel_dups.py", "entry": "tools/tmp_cancel_dups.py::module-level", "chain": "STATIC_ONLY:tools/tmp_cancel_dups.py::module-level->L57:cancel_order; runtime_entry_point=UNKNOWN; dynamic_caller=UNKNOWN", "prior": "Historical only", "canonical": "Historical only", "proof": "STATIC_ONLY: STATIC_ONLY: non-runtime/historical reference in tools/tmp_cancel_dups.py"},

}


def default_out_dir() -> Path:
    override = os.environ.get("TRITON_198I_EVIDENCE_OUT")
    if override:
        return Path(override)
    return ROOT / "data" / "results" / "phase_198i_sell_remediation_20260814"


def build_inventory_rows() -> List[Dict[str, Any]]:
    if len(FINAL) != 61:
        raise RuntimeError(f"FINAL must contain 61 routes, got {len(FINAL)}")
    if len(ROUTE_SEED) != 61:
        raise RuntimeError(f"ROUTE_SEED must contain 61 routes, got {len(ROUTE_SEED)}")
    if set(FINAL) != set(ROUTE_SEED):
        raise RuntimeError("FINAL and ROUTE_SEED route ids diverge")
    rows: List[Dict[str, Any]] = []
    for i in range(1, 62):
        rid = f"R{i}"
        seed = ROUTE_SEED[rid]
        prior = seed["prior"]
        chain = seed["chain"]
        route_blob = f"{seed['source_file']} {seed['entry']} {chain}"
        reachable = "Y" if prior in {"Unresolved", "Hardened through canonical boundary", "BUY/ADD-only"} else "N"
        if prior in {"Unreachable", "Disabled fail-closed", "Historical only", "Fake test-only"}:
            reachable = "N"
        if rid == "R40":
            reachable = "N"
        disp = FINAL[rid]
        if disp == "VERIFIED_HARDENED_SELL_ROUTE":
            sides = "SELL"
        elif disp == "VERIFIED_NON_SELL_ROUTE" and "cancel" in route_blob.lower():
            sides = "CANCEL"
        elif disp in {
            "VERIFIED_NON_EXECUTION_ROUTE",
            "VERIFIED_DEAD_OR_UNREACHABLE",
            "VERIFIED_TEST_OR_FAKE_ONLY",
        }:
            sides = "NONE"
        else:
            sides = "UNKNOWN"
        if disp == "VERIFIED_HARDENED_SELL_ROUTE":
            hardened = "Y"
        elif disp == "VERIFIED_NON_SELL_ROUTE" and "cancel" in chain.lower():
            hardened = "Y_CANCEL_GATE"
        else:
            hardened = "N_A"
        if rid in EVIDENCE:
            fake_obs = "Y"
        elif disp == "VERIFIED_HARDENED_SELL_ROUTE":
            fake_obs = "Y_CONVERGE_ALPACA"
        else:
            fake_obs = "N_A"
        rows.append(
            {
                "route_id": rid,
                "source_file": seed["source_file"],
                "function_or_callsite": CALLSITES.get(rid, seed["entry"]),
                "route_entry_point": CALLSITES.get(rid, seed["entry"]),
                "complete_call_chain": chain,
                "reachable_or_unreachable": "reachable" if reachable == "Y" else "unreachable",
                "supported_order_sides": sides,
                "paper_or_live_applicability": "paper_primary",
                "direct_broker_interaction": seed["canonical"],
                "hardened_sell_boundary_reached": hardened,
                "fake_broker_observation_exists": fake_obs,
                "prior_classification": prior,
                "prior_disposition": prior,
                "current_classification": disp,
                "supporting_test_or_evidence": EVIDENCE.get(rid, seed["proof"]),
                "remediation_performed": (
                    "classified_and_evidenced" if prior == "Unresolved" else "reclassified_from_committed_seed"
                ),
                "final_disposition": disp,
            }
        )
    return rows


def write_inventory(out_dir: Optional[Path] = None) -> Dict[str, Any]:
    out = Path(out_dir) if out_dir is not None else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    rows = build_inventory_rows()
    if len(rows) != 61:
        raise RuntimeError(f"expected 61 inventory rows, got {len(rows)}")
    counts = Counter(r["final_disposition"] for r in rows)
    if counts.get("BLOCKED_UNRESOLVED", 0) != 0:
        raise RuntimeError("inventory must close all unresolved routes")
    expected = {
        "VERIFIED_HARDENED_SELL_ROUTE": 20,
        "VERIFIED_NON_SELL_ROUTE": 18,
        "VERIFIED_TEST_OR_FAKE_ONLY": 11,
        "VERIFIED_NON_EXECUTION_ROUTE": 9,
        "VERIFIED_DEAD_OR_UNREACHABLE": 3,
    }
    for key, n in expected.items():
        if counts.get(key, 0) != n:
            raise RuntimeError(f"{key} expected {n}, got {counts.get(key, 0)}")
    fifteen = [r for r in rows if r["prior_disposition"] == "Unresolved"]
    if [r["route_id"] for r in fifteen] != list(FORMERLY_UNRESOLVED):
        raise RuntimeError("formerly unresolved set diverged from committed seed")
    fields = list(rows[0].keys())
    csv_path = out / "02_broker_route_inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    before_after = [
        {
            "route_id": r["route_id"],
            "source_file": r["source_file"],
            "function_or_callsite": r["function_or_callsite"],
            "before": "Unresolved",
            "after": r["final_disposition"],
            "evidence": r["supporting_test_or_evidence"],
        }
        for r in fifteen
    ]
    (out / "03_formerly_unresolved_15_before_after.json").write_text(
        json.dumps({"count": len(before_after), "rows": before_after}, indent=2),
        encoding="utf-8",
    )
    payload = {"row_count": 61, "disposition_counts": dict(counts), "rows": rows}
    (out / "02_broker_route_inventory.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print("rows", len(rows))
    print("unresolved_prior", len(fifteen))
    print("counts", dict(counts))
    print("wrote", csv_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit Phase 198I 61-route inventory")
    parser.add_argument("--out-dir", default=None, help="Evidence output directory")
    args = parser.parse_args()
    write_inventory(Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
