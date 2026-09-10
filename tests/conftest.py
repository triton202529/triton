"""BN2A isolation: production-path and broker blocking only when marked."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from tests.isolation_guard import ProductionIsolationError, check_open

_REAL_OPEN = builtins.open
_REAL_PATH_OPEN = Path.open


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "bn2a_isolation: block production artifacts and broker calls for BN2A/ledger tests",
    )


def _guarded_open(file, mode="r", *args, **kwargs):
    try:
        check_open(Path(file), str(mode))
    except TypeError:
        pass
    return _REAL_OPEN(file, mode, *args, **kwargs)


def _guarded_path_open(self, mode="r", *args, **kwargs):
    check_open(Path(self), str(mode))
    return _REAL_PATH_OPEN(self, mode, *args, **kwargs)


def _block_broker(*_args, **_kwargs):
    raise ProductionIsolationError("forbidden_broker_contact")


@pytest.fixture(autouse=True)
def _bn2a_production_isolation(request, monkeypatch):
    if request.node.get_closest_marker("bn2a_isolation") is None:
        yield
        return
    monkeypatch.setattr(builtins, "open", _guarded_open)
    monkeypatch.setattr(Path, "open", _guarded_path_open)
    monkeypatch.setattr("requests.sessions.Session.request", _block_broker, raising=False)
    try:
        import services.broker_alpaca as broker

        monkeypatch.setattr(broker.AlpacaBroker, "_get", _block_broker, raising=False)
        monkeypatch.setattr(broker.AlpacaBroker, "_post", _block_broker, raising=False)
        monkeypatch.setattr(broker.AlpacaBroker, "_patch", _block_broker, raising=False)
        monkeypatch.setattr(broker.AlpacaBroker, "_delete", _block_broker, raising=False)
    except Exception:
        pass
    yield
