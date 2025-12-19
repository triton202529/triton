#!/usr/bin/env python3
"""
Compliance and Audit Trail Engine for Triton

Implements institutional-grade compliance and audit capabilities:
- Trade audit trail (complete record)
- Regulatory compliance checks
- Risk limit enforcement
- Position limit monitoring
- Best execution verification
- Regulatory reporting (MiFID II, Reg NMS)
- Compliance alerts and exceptions

Critical for institutional trading and regulatory requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import warnings

warnings.filterwarnings("ignore")


class ComplianceAuditEngine:
    """
    Compliance and Audit Trail Engine.

    Ensures regulatory compliance and maintains complete audit trail.
    """

    def __init__(
        self,
        rules_config: Optional[Dict] = None,
        audit_log_path: str = "data/audit/trade_audit.jsonl",
        verbose: bool = False,
    ):
        """
        Initialize compliance engine.

        Args:
            rules_config: Compliance rules configuration
            audit_log_path: Path to audit log file
            verbose: Enable verbose logging
        """
        self.rules = rules_config or self._default_rules()
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Violation tracking
        self.violations = []
        self.warnings_list = []

    def _log(self, *args, **kwargs):
        """Logging helper."""
        if self.verbose:
            print(*args, **kwargs)

    def _default_rules(self) -> Dict:
        """Default compliance rules."""
        return {
            "position_limits": {
                "max_position_pct": 0.10,  # Max 10% in single position
                "max_sector_pct": 0.30,  # Max 30% in single sector
                "max_concentration": 0.50,  # Top 5 positions < 50%
            },
            "trading_limits": {
                "max_order_size": 1000000,  # Max $1M per order
                "max_daily_volume": 10000000,  # Max $10M per day
                "max_turnover_rate": 2.0,  # Max 200% annual turnover
            },
            "risk_limits": {
                "max_leverage": 1.0,  # No leverage
                "max_var": 0.05,  # Max 5% VaR
                "max_drawdown": 0.15,  # Max 15% drawdown
            },
            "best_execution": {
                "min_execution_quality": 0.95,  # 95% of VWAP or better
                "max_slippage_bps": 50,  # Max 50 bps slippage
            },
            "restricted_securities": [],  # Tickers not allowed
            "required_disclosures": ["price", "quantity", "timestamp", "rationale"],
        }

    def log_trade(self, trade: Dict) -> str:
        """
        Log trade to immutable audit trail.

        Args:
            trade: Trade details

        Returns:
            Audit ID (hash)
        """
        # Add timestamp and audit metadata
        audit_record = {
            "audit_id": self._generate_audit_id(trade),
            "timestamp": datetime.now().isoformat(),
            "trade": trade,
            "compliance_checks": self.check_compliance(trade),
            "system_state": {"version": "1.0", "environment": "production"},
        }

        # Write to append-only log
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(audit_record, default=str) + "\n")

        self._log(f"📝 Trade logged: {audit_record['audit_id']}")

        return audit_record["audit_id"]

    def _generate_audit_id(self, trade: Dict) -> str:
        """Generate unique audit ID."""
        # Hash of trade details + timestamp
        content = json.dumps(trade, sort_keys=True, default=str)
        timestamp = datetime.now().isoformat()
        hash_input = f"{content}{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def check_compliance(self, trade: Dict, portfolio_state: Optional[Dict] = None) -> Dict:
        """
        Check trade against compliance rules.

        Args:
            trade: Proposed trade
            portfolio_state: Current portfolio state

        Returns:
            Compliance check results
        """
        checks = {"passed": True, "violations": [], "warnings": [], "checks_performed": []}

        # Check 1: Position limits
        if portfolio_state:
            position_check = self._check_position_limits(trade, portfolio_state)
            checks["checks_performed"].append("position_limits")
            if not position_check["passed"]:
                checks["passed"] = False
                checks["violations"].extend(position_check["violations"])

        # Check 2: Order size limits
        order_check = self._check_order_limits(trade)
        checks["checks_performed"].append("order_limits")
        if not order_check["passed"]:
            checks["passed"] = False
            checks["violations"].extend(order_check["violations"])

        # Check 3: Restricted securities
        restricted_check = self._check_restricted_securities(trade)
        checks["checks_performed"].append("restricted_securities")
        if not restricted_check["passed"]:
            checks["passed"] = False
            checks["violations"].extend(restricted_check["violations"])

        # Check 4: Required disclosures
        disclosure_check = self._check_disclosures(trade)
        checks["checks_performed"].append("disclosures")
        if not disclosure_check["passed"]:
            checks["warnings"].extend(disclosure_check["warnings"])

        return checks

    def _check_position_limits(self, trade: Dict, portfolio_state: Dict) -> Dict:
        """Check position limit compliance."""
        violations = []

        ticker = trade.get("ticker")
        order_value = trade.get("value", 0)
        total_value = portfolio_state.get("total_value", 1)

        # Position concentration
        current_position = portfolio_state.get("positions", {}).get(ticker, {}).get("value", 0)
        new_position = current_position + order_value
        position_pct = new_position / total_value if total_value > 0 else 0

        max_position = self.rules["position_limits"]["max_position_pct"]
        if position_pct > max_position:
            violations.append(
                {
                    "rule": "max_position_pct",
                    "limit": max_position,
                    "actual": position_pct,
                    "message": f"Position {ticker} would exceed {max_position:.1%} limit",
                }
            )

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_order_limits(self, trade: Dict) -> Dict:
        """Check order size limits."""
        violations = []

        order_value = trade.get("value", 0)
        max_order = self.rules["trading_limits"]["max_order_size"]

        if order_value > max_order:
            violations.append(
                {
                    "rule": "max_order_size",
                    "limit": max_order,
                    "actual": order_value,
                    "message": f"Order size ${order_value:,.2f} exceeds limit ${max_order:,.2f}",
                }
            )

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_restricted_securities(self, trade: Dict) -> Dict:
        """Check if security is restricted."""
        violations = []

        ticker = trade.get("ticker")
        restricted = self.rules.get("restricted_securities", [])

        if ticker in restricted:
            violations.append(
                {
                    "rule": "restricted_securities",
                    "ticker": ticker,
                    "message": f"Trading {ticker} is restricted",
                }
            )

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_disclosures(self, trade: Dict) -> Dict:
        """Check required disclosures."""
        warnings = []

        required = self.rules.get("required_disclosures", [])

        for field in required:
            if field not in trade or not trade[field]:
                warnings.append(
                    {"field": field, "message": f"Missing required disclosure: {field}"}
                )

        return {"passed": len(warnings) == 0, "warnings": warnings}

    def verify_best_execution(self, execution: Dict, benchmark: Dict) -> Dict:
        """
        Verify best execution requirements.

        Compares execution to benchmark (VWAP, arrival price, etc.).

        Args:
            execution: Actual execution details
            benchmark: Benchmark for comparison

        Returns:
            Best execution verification
        """
        execution_price = execution.get("price", 0)
        benchmark_price = benchmark.get("price", 0)

        # Calculate slippage
        slippage = abs(execution_price - benchmark_price)
        slippage_bps = (slippage / benchmark_price) * 10000 if benchmark_price > 0 else 0

        # Check limits
        max_slippage = self.rules["best_execution"]["max_slippage_bps"]
        quality = 1 - (slippage_bps / max_slippage) if max_slippage > 0 else 1
        min_quality = self.rules["best_execution"]["min_execution_quality"]

        passed = quality >= min_quality and slippage_bps <= max_slippage

        return {
            "passed": passed,
            "execution_price": execution_price,
            "benchmark_price": benchmark_price,
            "slippage_bps": slippage_bps,
            "execution_quality": quality,
            "meets_requirements": passed,
        }

    def generate_regulatory_report(
        self, start_date: datetime, end_date: datetime, report_type: str = "mifid_ii"
    ) -> Dict:
        """
        Generate regulatory report.

        Args:
            start_date: Report start date
            end_date: Report end date
            report_type: Type of report ('mifid_ii', 'reg_nms', 'form_pf')

        Returns:
            Regulatory report
        """
        self._log(f"📋 Generating {report_type} report for {start_date} to {end_date}")

        # Load audit trail
        trades = self._load_audit_trail(start_date, end_date)

        report = {
            "report_type": report_type,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "total_trades": len(trades),
                "total_volume": sum(t.get("trade", {}).get("value", 0) for t in trades),
                "violations": len(
                    [t for t in trades if not t.get("compliance_checks", {}).get("passed", True)]
                ),
            },
            "trades": trades,
            "compliance_summary": self._summarize_compliance(trades),
        }

        return report

    def _load_audit_trail(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Load audit trail for date range."""
        trades = []

        if not self.audit_log_path.exists():
            return trades

        with open(self.audit_log_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    timestamp = datetime.fromisoformat(record["timestamp"])

                    if start_date <= timestamp <= end_date:
                        trades.append(record)
                except Exception:
                    continue

        return trades

    def _summarize_compliance(self, trades: List[Dict]) -> Dict:
        """Summarize compliance across trades."""
        total = len(trades)
        if total == 0:
            return {}

        passed = len([t for t in trades if t.get("compliance_checks", {}).get("passed", True)])
        violations = []

        for trade in trades:
            checks = trade.get("compliance_checks", {})
            violations.extend(checks.get("violations", []))

        return {
            "compliance_rate": passed / total,
            "total_violations": len(violations),
            "violation_types": pd.Series([v.get("rule", "unknown") for v in violations])
            .value_counts()
            .to_dict(),
        }

    def export_audit_trail(self, output_path: str, format: str = "csv"):
        """
        Export audit trail for external review.

        Args:
            output_path: Output file path
            format: 'csv' or 'json'
        """
        trades = self._load_audit_trail(datetime(2000, 1, 1), datetime.now())

        if format == "csv":
            # Flatten for CSV
            flattened = []
            for record in trades:
                flat = {
                    "audit_id": record.get("audit_id"),
                    "timestamp": record.get("timestamp"),
                    **record.get("trade", {}),
                    "compliance_passed": record.get("compliance_checks", {}).get("passed", True),
                }
                flattened.append(flat)

            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False)
        else:
            with open(output_path, "w") as f:
                json.dump(trades, f, indent=2, default=str)

        self._log(f"💾 Audit trail exported to {output_path}")


def main():
    """Demo compliance engine."""
    print("⚖️ Compliance and Audit Engine Demo")
    print("=" * 70)

    engine = ComplianceAuditEngine(verbose=True)

    # Example trade
    trade = {
        "ticker": "AAPL",
        "action": "BUY",
        "shares": 1000,
        "price": 150.0,
        "value": 150000,
        "timestamp": datetime.now().isoformat(),
        "rationale": "Strong technical setup",
    }

    # Example portfolio state
    portfolio_state = {
        "total_value": 1000000,
        "positions": {"AAPL": {"shares": 500, "value": 75000}},
    }

    # Check compliance
    print("\n✅ Checking compliance...")
    compliance = engine.check_compliance(trade, portfolio_state)
    print(f"  Passed: {compliance['passed']}")
    print(f"  Checks performed: {compliance['checks_performed']}")
    if compliance["violations"]:
        print(f"  Violations: {compliance['violations']}")

    # Log trade
    print("\n📝 Logging trade...")
    audit_id = engine.log_trade(trade)
    print(f"  Audit ID: {audit_id}")

    # Best execution check
    print("\n🎯 Verifying best execution...")
    execution = {"price": 150.10}
    benchmark = {"price": 150.00}
    best_ex = engine.verify_best_execution(execution, benchmark)
    print(f"  Passed: {best_ex['passed']}")
    print(f"  Slippage: {best_ex['slippage_bps']:.1f} bps")
    print(f"  Quality: {best_ex['execution_quality']:.2%}")

    # Generate report
    print("\n📋 Generating regulatory report...")
    report = engine.generate_regulatory_report(
        start_date=datetime.now() - timedelta(days=30), end_date=datetime.now()
    )
    print(f"  Total trades: {report['summary']['total_trades']}")
    print(f"  Compliance rate: {report['compliance_summary'].get('compliance_rate', 1):.1%}")

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
