# services/contracts_registry.py
# ------------------------------------------------------------
# TRITON — Contracts Registry
#
# This file centralizes the list of DataContracts used by the validator.
#
# IMPORTANT:
# Your repo currently does NOT have any *contract registry* module,
# which is why tools/validate_contracts.py can't import it.
#
# This registry is intentionally minimal + critical:
#   - portfolio_history.csv (must have date + total_value)
#   - positions_snapshot.csv (must have symbol)
#
# You can add more contracts here over time.
# ------------------------------------------------------------

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.data_contracts import DataContract  # type: ignore


def _make_contract(**kwargs: Any) -> DataContract:
    """
    Create a DataContract safely even if the DataContract signature differs
    across repo versions. We only pass args that the constructor accepts.
    """
    sig = inspect.signature(DataContract)  # type: ignore[arg-type]
    accepted = set(sig.parameters.keys())

    filtered: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in accepted}

    # Some versions may name fields differently; add a couple safe aliases
    if "path" not in accepted and "file" in accepted and "path" in kwargs:
        filtered["file"] = kwargs["path"]

    return DataContract(**filtered)  # type: ignore[arg-type]


def get_contracts(project_root: Optional[Path] = None) -> List[DataContract]:
    """
    Returns the list of DataContract definitions.

    `project_root` is accepted for compatibility (some callers pass it),
    but contracts here use repo-relative paths so it's optional.
    """
    # NOTE: keep paths repo-relative; validator will join with project_root.
    return [
        _make_contract(
            name="Portfolio History",
            path=Path("data/results/portfolio_history.csv"),
            fmt="csv",
            required_cols=["date", "total_value"],
            optional_cols=[
                "portfolio_value",
                "equity",
                "cash",
                "buying_power",
                "long_mv",
                "short_mv",
                "net_mv",
                "timestamp",
                "date_utc",
                "timestamp_utc",
            ],
            min_rows=1,
            unique_keys=["date"],
        ),
        _make_contract(
            name="Positions Snapshot",
            path=Path("data/results/positions_snapshot.csv"),
            fmt="csv",
            required_cols=["symbol"],
            optional_cols=[
                "ticker",
                "date",
                "snapshot_ts",
                "qty",
                "current_price",
                "market_value",
                "value",
                "avg_entry_price",
                "cost_basis",
                "unrealized_pl",
                "unrealized_plpc",
                "exchange",
                "asset_class",
                "asset_marginable",
            ],
            min_rows=0,
            unique_keys=None,
        ),
    ]
