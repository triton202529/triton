"""Isolated Streamlit AppTest entry for paper SELL arm controls. No HTTP server."""

from dashboard.operator_console import render_operations_page
from dashboard.paper_sell_arm_controls import render_paper_sell_arm_controls

render_operations_page()
render_paper_sell_arm_controls()
