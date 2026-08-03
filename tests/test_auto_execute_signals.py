import csv
import runpy
import sys
import types
from pathlib import Path


class FakeAccount:
    buying_power = "1000"


class FakePosition:
    market_value = "200"
    qty = "20"


class FakeTrade:
    price = "10"


class FakeOrder:
    id = "fake-order-id"


class FakeREST:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.orders = []
        FakeREST.instances.append(self)

    def get_account(self):
        return FakeAccount()

    def get_position(self, _ticker):
        return FakePosition()

    def get_latest_trade(self, _ticker):
        return FakeTrade()

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return FakeOrder()


def install_fake_alpaca(monkeypatch):
    alpaca_module = types.ModuleType("alpaca_trade_api")
    rest_module = types.ModuleType("alpaca_trade_api.rest")
    rest_module.REST = FakeREST
    alpaca_module.rest = rest_module

    monkeypatch.setitem(sys.modules, "alpaca_trade_api", alpaca_module)
    monkeypatch.setitem(sys.modules, "alpaca_trade_api.rest", rest_module)


def install_fake_dotenv(monkeypatch):
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)


def test_auto_execute_blocks_failed_risk_check(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    signals_dir = tmp_path / "data" / "predictions"
    results_dir = tmp_path / "data" / "results"
    signals_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (signals_dir / "signals.csv").write_text("ticker,signal\nRISKY,BUY\n")

    FakeREST.instances.clear()
    install_fake_alpaca(monkeypatch)
    install_fake_dotenv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(repo_root / "scripts"))

    runpy.run_path(str(repo_root / "scripts" / "auto_execute_signals.py"), run_name="__main__")

    api = FakeREST.instances[0]
    assert api.orders == []

    with (results_dir / "executed_trades.csv").open(newline="") as trade_log:
        rows = list(csv.DictReader(trade_log))

    assert len(rows) == 1
    assert rows[0]["ticker"] == "RISKY"
    assert rows[0]["status"] == "BLOCKED"
    assert rows[0]["note"] == "Too much exposure to RISKY (>10% of buying power)"
