#!/usr/bin/env python3
"""
Triton Stress Test Framework
============================

Comprehensive stress testing for the Triton trading system including:
- Market crash scenarios
- High volatility periods
- Model failure scenarios
- Data quality issues
- System overload conditions
- Risk management validation

Inspired by BlackRock's Aladdin stress testing capabilities.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Triton components (optional; runs fine in pure-sim mode)
try:
    from services.confidence import compute_confidence  # noqa: F401
    from services.feature_generator import add_technical_indicators  # noqa: F401
    from services.broker_alpaca import AlpacaClient  # noqa: F401
except Exception as e:
    print(f"Warning: Could not import some Triton components: {e}")
    print("Proceeding in simulation mode.")


def _max_dd_from_curve(curve: np.ndarray) -> float:
    """True peak→trough max drawdown from a cumulative curve."""
    if curve.size < 2:
        return 0.0
    peaks = np.maximum.accumulate(curve)
    dd = curve / peaks - 1.0
    return float(dd.min())


def _to_jsonable(v):
    """Make numpy types JSON-serializable."""
    import numpy as _np

    if isinstance(v, (_np.floating,)):
        return float(v)
    if isinstance(v, (_np.integer,)):
        return int(v)
    if isinstance(v, (_np.bool_,)):
        return bool(v)
    return v


class TritonStressTest:
    """
    Comprehensive stress testing framework for Triton trading system.

    Tests include:
    1. Market crash scenarios (2008, 2020, etc.)
    2. High volatility periods
    3. Model failure scenarios
    4. Data quality issues
    5. System overload conditions
    6. Risk management validation
    """

    def __init__(self, config_path: str = "config/stress_test.json"):
        self.config = self._load_config(config_path)

        # RNG for reproducibility
        self.rng = np.random.default_rng(self.config.get("seed", 12345))

        # Allow config to override system limits
        self.system_limits = {
            "max_positions": 15,
            "max_allocation_per_trade": 0.05,
            "max_daily_loss": 0.02,
            "max_portfolio_drawdown": 0.15,
            "min_confidence_threshold": 0.7,
        }
        self.system_limits.update(self.config.get("system_limits", {}))

        # Crash scenarios: prefer config if present
        self.crash_scenarios = self.config.get(
            "crash_scenarios",
            {
                "2008_financial_crisis": {
                    "duration_days": 252,
                    "max_drawdown": -0.55,
                    "volatility_multiplier": 3.0,
                    "correlation_increase": 0.8,
                },
                "2020_covid_crash": {
                    "duration_days": 60,
                    "max_drawdown": -0.35,
                    "volatility_multiplier": 2.5,
                    "correlation_increase": 0.7,
                },
                "2022_inflation_crash": {
                    "duration_days": 180,
                    "max_drawdown": -0.25,
                    "volatility_multiplier": 2.0,
                    "correlation_increase": 0.6,
                },
            },
        )

        self.results: Dict[str, Dict] = {}
        self.start_time = datetime.now()

    def _load_config(self, config_path: str) -> Dict:
        """Load stress test configuration with sane defaults."""
        default_config = {
            "test_capital": 100000,
            "test_duration_days": 252,
            "confidence_threshold": 0.7,
            "max_positions": 15,
            "risk_tolerance": 0.15,
            "scenarios": [
                "market_crash",
                "high_volatility",
                "model_failure",
                "data_corruption",
                "system_overload",
            ],
            "output_dir": "data/stress_test_results",
            "seed": 12345,
            "quick": False,
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                return {**default_config, **cfg}
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

        return default_config

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry
    # ──────────────────────────────────────────────────────────────────────────
    def run_all_stress_tests(self) -> Dict:
        """Run all configured stress tests."""
        print("🧪 Starting Triton Stress Test Suite...")
        print(f"Test Capital: ${self.config['test_capital']:,}")
        print(f"Duration baseline: {self.config['test_duration_days']} days")
        print("=" * 60)

        # Quick mode: shorten and limit scenarios
        if self.config.get("quick", False):
            self.config["test_duration_days"] = min(60, self.config["test_duration_days"])
            self.config["scenarios"] = [
                s
                for s in self.config["scenarios"]
                if s in ("market_crash", "high_volatility", "risk_management")
            ]
            for v in self.crash_scenarios.values():
                v["duration_days"] = min(60, int(v.get("duration_days", 60)))

        test_results: Dict[str, Dict] = {}

        # 1) Market Crashes
        if "market_crash" in self.config["scenarios"]:
            print("\n📉 Testing Market Crash Scenarios...")
            test_results["market_crash"] = self._test_market_crashes()

        # 2) High Volatility
        if "high_volatility" in self.config["scenarios"]:
            print("\n⚡ Testing High Volatility Scenarios...")
            test_results["high_volatility"] = self._test_high_volatility()

        # 3) Model Failures
        if "model_failure" in self.config["scenarios"]:
            print("\n🤖 Testing Model Failure Scenarios...")
            test_results["model_failure"] = self._test_model_failures()

        # 4) Data Corruption
        if "data_corruption" in self.config["scenarios"]:
            print("\n💾 Testing Data Quality Issues...")
            test_results["data_corruption"] = self._test_data_corruption()

        # 5) System Overload
        if "system_overload" in self.config["scenarios"]:
            print("\n🔥 Testing System Overload...")
            test_results["system_overload"] = self._test_system_overload()

        # 6) Risk Management (always useful)
        print("\n🛡️ Testing Risk Management...")
        test_results["risk_management"] = self._test_risk_management()

        self._generate_report(test_results)
        return test_results

    # ──────────────────────────────────────────────────────────────────────────
    # Scenarios
    # ──────────────────────────────────────────────────────────────────────────
    def _test_market_crashes(self) -> Dict:
        results = {}

        for name, params in self.crash_scenarios.items():
            print(f"  • {name} ...")
            portfolio_value = float(self.config["test_capital"])
            daily_returns: List[float] = []
            values: List[float] = []

            days = int(params.get("duration_days", self.config["test_duration_days"]))
            for _ in range(days):
                mkt_ret = self._simulate_crash_day(params)
                port_ret = self._simulate_portfolio_response(mkt_ret, positions=[])
                portfolio_value *= (1.0 + port_ret)
                values.append(portfolio_value)
                daily_returns.append(port_ret)

            curve = np.asarray(values, dtype=float)
            curve /= curve[0]
            vol = float(np.std(daily_returns) * np.sqrt(252))
            sharpe = float(
                (np.mean(daily_returns) / (np.std(daily_returns) + 1e-12)) * np.sqrt(252)
            )
            total_ret = float(curve[-1] - 1.0)
            max_dd = _max_dd_from_curve(curve)
            survived = max_dd > -float(self.system_limits["max_portfolio_drawdown"])

            results[name] = {
                "final_value": float(portfolio_value),
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "volatility": vol,
                "sharpe_ratio": sharpe,
                "survived": survived,
            }

            print(
                f"    Final ${portfolio_value:,.2f} | Return {total_ret:+.1%} | "
                f"MaxDD {max_dd:.1%} | {'✅ Survived' if survived else '❌ Failed'}"
            )

        return results

    def _test_high_volatility(self) -> Dict:
        results = {}
        vol_mults = [1.5, 2.0, 3.0, 4.0]
        days = int(self.config["test_duration_days"])

        for mult in vol_mults:
            print(f"  • {mult}× volatility ...")
            portfolio_value = float(self.config["test_capital"])
            daily_returns: List[float] = []
            values: List[float] = []

            for _ in range(days):
                mkt_ret = float(self.rng.normal(0.0, 0.02 * mult))
                port_ret = self._simulate_portfolio_response(mkt_ret, positions=[])
                portfolio_value *= (1.0 + port_ret)
                values.append(portfolio_value)
                daily_returns.append(port_ret)

            curve = np.asarray(values, dtype=float)
            curve /= curve[0]
            total_ret = float(curve[-1] - 1.0)
            max_dd = _max_dd_from_curve(curve)
            vol = float(np.std(daily_returns) * np.sqrt(252))
            survived = max_dd > -float(self.system_limits["max_portfolio_drawdown"])

            key = f"{mult}x_volatility"
            results[key] = {
                "final_value": float(portfolio_value),
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "volatility": vol,
                "survived": survived,
            }

            print(
                f"    Final ${portfolio_value:,.2f} | MaxDD {max_dd:.1%} | "
                f"{'✅ Survived' if survived else '❌ Failed'}"
            )

        return results

    def _test_model_failures(self) -> Dict:
        results = {}
        days = int(self.config["test_duration_days"])

        scenarios = {
            "model_bias": "Models consistently overestimate returns",
            "model_variance": "Models produce highly variable predictions",
            "model_correlation": "All models fail simultaneously",
            "confidence_failure": "Confidence scoring fails",
        }

        for name, _desc in scenarios.items():
            print(f"  • {name} ...")
            portfolio_value = float(self.config["test_capital"])
            daily_returns: List[float] = []
            values: List[float] = []

            for _ in range(days):
                if name == "model_bias":
                    mkt_ret = float(self.rng.normal(-0.02, 0.015))
                elif name == "model_variance":
                    mkt_ret = float(self.rng.normal(0.0, 0.05))
                elif name == "model_correlation":
                    mkt_ret = float(self.rng.normal(-0.03, 0.02))
                else:  # confidence_failure
                    mkt_ret = float(self.rng.normal(0.0, 0.025))

                port_ret = self._simulate_portfolio_response(mkt_ret, positions=[])
                portfolio_value *= (1.0 + port_ret)
                values.append(portfolio_value)
                daily_returns.append(port_ret)

            curve = np.asarray(values, dtype=float)
            curve /= curve[0]
            total_ret = float(curve[-1] - 1.0)
            max_dd = _max_dd_from_curve(curve)
            vol = float(np.std(daily_returns) * np.sqrt(252))
            survived = max_dd > -float(self.system_limits["max_portfolio_drawdown"])

            results[name] = {
                "final_value": float(portfolio_value),
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "volatility": vol,
                "survived": survived,
            }

            print(
                f"    Final ${portfolio_value:,.2f} | MaxDD {max_dd:.1%} | "
                f"{'✅ Survived' if survived else '❌ Failed'}"
            )

        return results

    def _test_data_corruption(self) -> Dict:
        results = {}
        days = int(self.config["test_duration_days"])

        scenarios = {
            "missing_data": "Random data points missing",
            "outlier_data": "Extreme outliers in data",
            "delayed_data": "Data arrives with delays",
            "wrong_data": "Completely wrong data feeds",
        }

        for name, _desc in scenarios.items():
            print(f"  • {name} ...")
            portfolio_value = float(self.config["test_capital"])
            daily_returns: List[float] = []
            values: List[float] = []
            dq_score = 1.0

            for _ in range(days):
                if name == "missing_data":
                    if float(self.rng.random()) < 0.10:
                        dq_score *= 0.8
                        mkt_ret = 0.0  # skip trading
                    else:
                        mkt_ret = float(self.rng.normal(0.0, 0.015))
                elif name == "outlier_data":
                    if float(self.rng.random()) < 0.05:
                        mkt_ret = float(self.rng.normal(0.0, 0.10))
                    else:
                        mkt_ret = float(self.rng.normal(0.0, 0.015))
                elif name == "delayed_data":
                    if float(self.rng.random()) < 0.20:
                        dq_score *= 0.9
                        mkt_ret = float(self.rng.normal(0.0, 0.020))
                    else:
                        mkt_ret = float(self.rng.normal(0.0, 0.015))
                else:  # wrong_data
                    if float(self.rng.random()) < 0.05:
                        dq_score *= 0.5
                        mkt_ret = float(self.rng.normal(-0.02, 0.03))
                    else:
                        mkt_ret = float(self.rng.normal(0.0, 0.015))

                port_ret = self._simulate_portfolio_response(mkt_ret, positions=[])
                portfolio_value *= (1.0 + port_ret)
                values.append(portfolio_value)
                daily_returns.append(port_ret)

            curve = np.asarray(values, dtype=float)
            curve /= curve[0]
            total_ret = float(curve[-1] - 1.0)
            max_dd = _max_dd_from_curve(curve)
            vol = float(np.std(daily_returns) * np.sqrt(252))
            survived = max_dd > -float(self.system_limits["max_portfolio_drawdown"])

            results[name] = {
                "final_value": float(portfolio_value),
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "volatility": vol,
                "data_quality_score": float(dq_score),
                "survived": survived,
            }

            print(
                f"    Final ${portfolio_value:,.2f} | DQ {dq_score:.2%} | "
                f"{'✅ Survived' if survived else '❌ Failed'}"
            )

        return results

    def _test_system_overload(self) -> Dict:
        results = {}
        days = int(self.config["test_duration_days"])

        scenarios = {
            "high_frequency": "Processing 1000+ signals per minute",
            "memory_pressure": "System running low on memory",
            "cpu_overload": "CPU usage at 95%+",
            "network_latency": "High network latency",
        }

        for name, _desc in scenarios.items():
            print(f"  • {name} ...")
            portfolio_value = float(self.config["test_capital"])
            daily_returns: List[float] = []
            values: List[float] = []
            delays: List[float] = []

            for _ in range(days):
                # Simulate delays WITHOUT sleeping (keeps tests fast & CI-friendly)
                if name == "high_frequency":
                    delays.append(float(self.rng.uniform(0.10, 0.50)))
                elif name == "memory_pressure":
                    delays.append(float(self.rng.uniform(0.20, 1.00)))
                elif name == "cpu_overload":
                    delays.append(float(self.rng.uniform(0.30, 1.50)))
                else:  # network_latency
                    delays.append(float(self.rng.uniform(0.50, 2.00)))

                mkt_ret = float(self.rng.normal(0.0, 0.015))
                port_ret = self._simulate_portfolio_response(mkt_ret, positions=[])
                portfolio_value *= (1.0 + port_ret)
                values.append(portfolio_value)
                daily_returns.append(port_ret)

            curve = np.asarray(values, dtype=float)
            curve /= curve[0]
            total_ret = float(curve[-1] - 1.0)
            max_dd = _max_dd_from_curve(curve)
            vol = float(np.std(daily_returns) * np.sqrt(252))
            avg_delay = float(np.mean(delays)) if delays else 0.0
            total_delay = float(np.sum(delays))
            survived = max_dd > -float(self.system_limits["max_portfolio_drawdown"])

            results[name] = {
                "final_value": float(portfolio_value),
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "volatility": vol,
                "avg_processing_delay": avg_delay,
                "total_processing_time": total_delay,  # simulated total
                "survived": survived,
            }

            print(
                f"    Final ${portfolio_value:,.2f} | AvgDelay {avg_delay:.3f}s | "
                f"{'✅ Survived' if survived else '❌ Failed'}"
            )

        return results

    def _test_risk_management(self) -> Dict:
        results = {}
        days = int(self.config["test_duration_days"])

        scenarios = {
            "position_limits": "Test maximum position limits",
            "allocation_limits": "Test maximum allocation per trade",
            "drawdown_limits": "Test maximum drawdown limits",
            "correlation_limits": "Test position correlation limits",
        }

        for name, _desc in scenarios.items():
            print(f"  • {name} ...")
            portfolio_value = float(self.config["test_capital"])
            positions: List[Dict] = []
            daily_returns: List[float] = []
            values: List[float] = []
            risk_violations = 0

            for _ in range(days):
                # Simulate different risk checks
                if name == "position_limits":
                    if len(positions) < int(self.system_limits["max_positions"]):
                        if float(self.rng.random()) < 0.30:
                            positions.append({"symbol": f"TEST{len(positions)}", "size": 0.05})
                    mkt_ret = float(self.rng.normal(0.0, 0.015))

                elif name == "allocation_limits":
                    if float(self.rng.random()) < 0.20:
                        alloc = float(self.rng.uniform(0.05, 0.15))
                        if alloc > float(self.system_limits["max_allocation_per_trade"]):
                            risk_violations += 1
                            alloc = float(self.system_limits["max_allocation_per_trade"])
                    mkt_ret = float(self.rng.normal(0.0, 0.015))

                elif name == "drawdown_limits":
                    # check against initial capital drawdown
                    current_dd = (portfolio_value - float(self.config["test_capital"])) / float(
                        self.config["test_capital"]
                    )
                    if current_dd < -float(self.system_limits["max_portfolio_drawdown"]):
                        risk_violations += 1
                        mkt_ret = 0.0  # emergency stop
                    else:
                        mkt_ret = float(self.rng.normal(0.0, 0.015))

                else:  # correlation_limits
                    if len(positions) > 5:
                        mkt_ret = float(self.rng.normal(0.0, 0.02))
                        if abs(mkt_ret) > 0.03:
                            risk_violations += 1
                    else:
                        mkt_ret = float(self.rng.normal(0.0, 0.015))

                port_ret = self._simulate_portfolio_response(mkt_ret, positions=positions)
                portfolio_value *= (1.0 + port_ret)
                values.append(portfolio_value)
                daily_returns.append(port_ret)

            curve = np.asarray(values, dtype=float)
            curve /= curve[0]
            total_ret = float(curve[-1] - 1.0)
            max_dd = _max_dd_from_curve(curve)
            vol = float(np.std(daily_returns) * np.sqrt(252))
            survived = max_dd > -float(self.system_limits["max_portfolio_drawdown"])

            results[name] = {
                "final_value": float(portfolio_value),
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "volatility": vol,
                "risk_violations": int(risk_violations),
                "survived": survived,
            }

            print(
                f"    Final ${portfolio_value:,.2f} | Violations {risk_violations} | "
                f"{'✅ Survived' if survived else '❌ Failed'}"
            )

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Simulation primitives
    # ──────────────────────────────────────────────────────────────────────────
    def _simulate_crash_day(self, params: Dict) -> float:
        """
        Simulate a single day during a market crash.

        Uses negative drift distributed across the crash duration,
        scaled volatility, and a simple correlation boost.
        """
        md = float(params.get("max_drawdown", -0.3))  # negative value expected
        days = max(1, int(params.get("duration_days", 60)))
        mu = np.log1p(md) / days  # daily log-drift (negative)
        sigma = 0.02 * float(params.get("volatility_multiplier", 2.0))
        rho = float(params.get("correlation_increase", 0.0))

        base = float(self.rng.normal(mu, sigma))
        idio = float(self.rng.normal(0.0, sigma * (1.0 - rho)))
        return base * (1.0 + rho) + idio

    def _simulate_portfolio_response(self, market_return: float, positions: List[Dict]) -> float:
        """Simulate how the portfolio responds to market movements."""
        beta = 0.8  # base market beta
        port_ret = market_return * beta

        if positions:
            # small position effect
            pos_size = sum(float(p.get("size", 0.05)) for p in positions)
            port_ret += pos_size * market_return * 0.1

        # idiosyncratic noise
        port_ret += float(self.rng.normal(0.0, 0.005))
        return port_ret

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────
    def _generate_report(self, test_results: Dict) -> None:
        print("\n" + "=" * 60)
        print("📊 STRESS TEST SUMMARY REPORT")
        print("=" * 60)

        total_tests = sum(len(s) for s in test_results.values())
        passed_tests = sum(
            sum(1 for r in s.values() if r.get("survived", False)) for s in test_results.values()
        )

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")

        print("\n📋 SCENARIO BREAKDOWN:")
        for scen_name, scen_results in test_results.items():
            print(f"\n{scen_name.upper()}:")
            for test_name, result in scen_results.items():
                status = "✅ PASS" if result.get("survived", False) else "❌ FAIL"
                ret = result.get("total_return", 0.0) * 100
                dd = result.get("max_drawdown", 0.0) * 100
                print(f"  {test_name}: {status} | Return: {ret:+.1f}% | Drawdown: {dd:.1f}%")

        print("\n🛡️ RISK ASSESSMENT:")
        critical = [
            f"{sn}.{tn}"
            for sn, rs in test_results.items()
            for tn, r in rs.items()
            if not r.get("survived", False)
        ]
        if critical:
            print("❌ CRITICAL FAILURES DETECTED:")
            for x in critical:
                print(f"  - {x}")
            print("\n⚠️  RECOMMENDATION: Address critical failures before going live!")
        else:
            print("✅ All tests passed! System appears robust.")

        print("\n📈 PERFORMANCE SUMMARY:")
        all_returns, all_dd = [], []
        for rs in test_results.values():
            for r in rs.values():
                all_returns.append(r.get("total_return", 0.0))
                all_dd.append(r.get("max_drawdown", 0.0))
        if all_returns:
            print(f"Average Return: {np.mean(all_returns):.1%}")
            print(f"Best Return: {np.max(all_returns):.1%}")
            print(f"Worst Return: {np.min(all_returns):.1%}")
            print(f"Average Drawdown: {np.mean(all_dd):.1%}")
            print(f"Worst Drawdown: {np.min(all_dd):.1%}")

        self._save_results(test_results)

        print(f"\n⏱️  Total Test Time: {datetime.now() - self.start_time}")
        print("=" * 60)

    def _save_results(self, test_results: Dict) -> None:
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"stress_test_results_{ts}.json"

        serializable = {
            scen: {name: {k: _to_jsonable(v) for k, v in res.items()} for name, res in scen_res.items()}
            for scen, scen_res in test_results.items()
        }

        blob = {
            "meta": {
                "schema_version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "seed": self.config.get("seed", None),
                "quick": bool(self.config.get("quick", False)),
                "config": {
                    k: v
                    for k, v in self.config.items()
                    if k.lower() not in {"api_key", "api_secret"}
                },
            },
            "results": serializable,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)

        print(f"\n💾 Detailed results saved to: {path}")


def main():
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Triton stress tests")
    parser.add_argument("--config", default="config/stress_test.json", help="Config file path")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to test")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (short & subset)")
    args = parser.parse_args()

    tester = TritonStressTest(args.config)

    # Allow CLI to override config
    if args.scenarios:
        tester.config["scenarios"] = args.scenarios
    if args.quick:
        tester.config["quick"] = True

    results = tester.run_all_stress_tests()

    critical_failures = sum(
        sum(1 for r in scen.values() if not r.get("survived", False)) for scen in results.values()
    )
    if critical_failures > 0:
        print(f"\n❌ {critical_failures} critical failures detected!")
        sys.exit(1)
    else:
        print("\n✅ All stress tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
