# services/snapshot_open_orders.py
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "results" / "open_orders_snapshot.json"
OUT_CSV = ROOT / "data" / "results" / "open_orders_snapshot.csv"

FIELDS = [
    "as_of",
    "id",
    "client_order_id",
    "symbol",
    "side",
    "type",
    "time_in_force",
    "qty",
    "filled_qty",
    "limit_price",
    "status",
    "created_at",
    "updated_at",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(mode: str = "paper") -> None:
    from services.broker_alpaca import AlpacaBroker  # type: ignore

    b = AlpacaBroker(mode=mode)
    oo = b.list_orders(status="open", nested=True, limit=2000) or []
    as_of = utc_now_iso()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"as_of": as_of, "mode": mode, "open_orders": oo}, indent=2), encoding="utf-8"
    )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for o in oo:
            row = {k: "" for k in FIELDS}
            row["as_of"] = as_of
            for k in FIELDS:
                if k == "as_of":
                    continue
                row[k] = o.get(k, "")
            w.writerow(row)

    print(f"wrote {OUT_JSON} and {OUT_CSV} open_orders={len(oo)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = ap.parse_args()
    main(mode=args.mode)
