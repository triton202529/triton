# Triton Stress Testing Framework

A comprehensive stress testing suite for the Triton trading system, inspired by BlackRock's Aladdin risk management capabilities.

## Overview

This stress testing framework validates your Triton system's robustness under extreme market conditions, system failures, and edge cases. It's designed to ensure your automated trading system can survive real-world challenges.

## Features

### 🧪 Test Scenarios

1. **Market Crash Scenarios**
   - 2008 Financial Crisis simulation
   - 2020 COVID crash simulation
   - 2022 Inflation crash simulation
   - Custom crash scenarios

2. **High Volatility Periods**
   - 1.5x to 4x normal volatility
   - Extreme market movements
   - Volatility clustering effects

3. **Model Failure Scenarios**
   - Model bias testing
   - Model variance testing
   - Model correlation failures
   - Confidence scoring failures

4. **Data Quality Issues**
   - Missing data scenarios
   - Outlier data handling
   - Delayed data feeds
   - Wrong data feeds

5. **System Overload**
   - High-frequency processing
   - Memory pressure testing
   - CPU overload scenarios
   - Network latency issues

6. **Risk Management Validation**
   - Position limit enforcement
   - Allocation limit testing
   - Drawdown limit validation
   - Correlation limit testing

## Quick Start

### 1. Run Quick Validation
```bash
python quick_validation.py
```

### 2. Run Interactive Stress Tests
```bash
python run_stress_test.py
```

### 3. Run Specific Scenarios
```bash
# Market crash tests only
python services/stress_test.py --scenarios market_crash

# High volatility tests only
python services/stress_test.py --scenarios high_volatility

# All tests with custom config
python services/stress_test.py --config config/stress_test.json
```

## Configuration

Edit `config/stress_test.json` to customize:

```json
{
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
    "system_overload"
  ],
  "system_limits": {
    "max_positions": 15,
    "max_allocation_per_trade": 0.05,
    "max_daily_loss": 0.02,
    "max_portfolio_drawdown": 0.15,
    "min_confidence_threshold": 0.7
  }
}
```

## Test Results

### Success Criteria
- **Survived**: Portfolio drawdown < 15%
- **Passed**: All risk limits respected
- **Failed**: System breaches risk limits

### Sample Output
```
📊 STRESS TEST SUMMARY REPORT
============================================================
Total Tests: 25
Passed: 23
Failed: 2
Success Rate: 92.0%

📋 SCENARIO BREAKDOWN:

MARKET_CRASH:
  2008_financial_crisis: ✅ PASS | Return: -8.2% | Drawdown: -12.1%
  2020_covid_crash: ✅ PASS | Return: -5.1% | Drawdown: -8.3%
  2022_inflation_crash: ✅ PASS | Return: -3.2% | Drawdown: -6.7%

HIGH_VOLATILITY:
  1.5x_volatility: ✅ PASS | Return: 12.3% | Drawdown: -4.2%
  2.0x_volatility: ✅ PASS | Return: 8.7% | Drawdown: -7.1%
  3.0x_volatility: ❌ FAIL | Return: -2.1% | Drawdown: -18.3%
  4.0x_volatility: ❌ FAIL | Return: -8.9% | Drawdown: -22.7%
```

## Understanding Results

### Key Metrics

- **Total Return**: Portfolio performance during stress period
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized volatility of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Survived**: Whether system stayed within risk limits

### Risk Assessment

- **✅ PASS**: System handled stress scenario well
- **❌ FAIL**: System breached risk limits
- **⚠️ CRITICAL**: System failure could cause significant losses

## Customization

### Adding New Scenarios

1. **Create new test method** in `TritonStressTest` class:
```python
def _test_custom_scenario(self) -> Dict:
    """Test custom scenario."""
    # Implementation here
    return results
```

2. **Add to scenarios list** in config:
```json
"scenarios": ["market_crash", "custom_scenario"]
```

3. **Update main test runner**:
```python
if "custom_scenario" in self.config["scenarios"]:
    test_results["custom_scenario"] = self._test_custom_scenario()
```

### Modifying Risk Limits

Edit `system_limits` in config:
```json
"system_limits": {
  "max_positions": 20,
  "max_allocation_per_trade": 0.03,
  "max_daily_loss": 0.015,
  "max_portfolio_drawdown": 0.10
}
```

## Best Practices

### Before Going Live

1. **Run full stress test suite**
2. **Address all critical failures**
3. **Validate with paper trading**
4. **Start with small capital**
5. **Monitor closely initially**

### Regular Testing

- **Weekly**: Quick validation tests
- **Monthly**: Full stress test suite
- **After major changes**: Comprehensive testing
- **Before scaling**: Stress test with larger capital

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all Triton components are available
2. **Config Errors**: Validate JSON configuration
3. **Memory Issues**: Reduce test duration or capital
4. **Timeout Issues**: Increase timeout limits

### Debug Mode

Run with verbose output:
```bash
python services/stress_test.py --verbose
```

## Integration

### CI/CD Pipeline

Add to your GitHub Actions:
```yaml
- name: Run Stress Tests
  run: python services/stress_test.py --quick
```

### Monitoring

Integrate with your monitoring system:
```python
# Check stress test results
if stress_test_failed:
    send_alert("Critical stress test failure detected!")
```

## Performance

### Test Duration
- **Quick tests**: 2-5 minutes
- **Full suite**: 10-30 minutes
- **Comprehensive**: 30-60 minutes

### Resource Usage
- **CPU**: Moderate during simulation
- **Memory**: ~100MB typical usage
- **Disk**: Results saved to `data/stress_test_results/`

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review test output logs
3. Validate configuration files
4. Run quick validation first

---

**Remember**: Stress testing is crucial for automated trading systems. Never go live without comprehensive validation! 🚀
