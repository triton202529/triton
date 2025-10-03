# services/daily_fill_report.py
"""
Generate a daily fills summary from data/results/live_orders.csv.

Default: uses today's date (based on UTC timestamps in the log).
You can force a date with --date YYYY-MM-DD.

Outputs:
  - data/results/daily_report_<YYYYMMDD>.txt  (human-readable)
  - data/results/daily_report_<YYYYMMDD>.csv  (latest status per order id)

Examples:
  python -m services.daily_fill_report
  python -m services.daily_fill_report --date 2025-08-15
"""

import os
import re
import argparse
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

RESULTS_DIR = "data/results"
LOG_PATH = os.path.join(RESULTS_DIR, "live_orders.csv")

_fq_re = re.compile(r"filled_qty=([0-9.]+)")
_fa_re = re.compile(r"filled_avg=([0-9.]+)")


def _parse_note_fill(note: str):
    s = str(note) if isinstance(note, str) else ""
    fq = _fq_re.search(s)
    fa = _fa_re.search(s)
    fqv = float(fq.group(1)) if fq else None
    fav = float(fa.group(1)) if fa else None
    return fqv, fav


def parse_args():
    p = argparse.ArgumentParser(description="Daily fills summary from live_orders.csv")
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="YYYY-MM-DD (defaults to 'today' from UTC timestamps)",
    )
    return p.parse_args()


def main():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        print("No live_orders.csv found.")
        return

    d = pd.read_csv(LOG_PATH)
    if "timestamp" not in d.columns:
        print("live_orders.csv missing 'timestamp' column.")
        return

    d["ts"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d[d["ts"].notna()].copy()

    args = parse_args()
    if args.date:
        day = pd.to_datetime(args.date).date()
    else:
        # default to UTC 'today'
        day = datetime.now(timezone.utc).date()

    # restrict to orders where their latest row is from that date
    latest = d.sort_values("ts").groupby("broker_order_id", dropna=True).tail(1).copy()
    latest["ts_date"] = latest["ts"].dt.date
    day_latest = latest[latest["ts_date"] == day].copy()
    if day_latest.empty:
        print(f"No orders with latest status on {day}.")
        return

    # extract fills
    fqfa = day_latest["note"].apply(_parse_note_fill)
    day_latest["filled_qty"] = [x[0] for x in fqfa]
    day_latest["filled_avg"] = [x[1] for x in fqfa]
    day_latest["fill_pct"] = (day_latest["filled_qty"] / day_latest["qty"] * 100.0).round(1)

    # basic aggregates
    by_status = day_latest["status"].value_counts(dropna=False)
    total_orders = len(day_latest)
    filled_rows = day_latest[day_latest["status"].isin(["FILLED", "PARTIALLY_FILLED"])]
    est_filled_notional = (
        filled_rows["filled_qty"].fillna(0) * filled_rows["filled_avg"].fillna(0)
    ).sum()

    # outputs
    ymd = day.strftime("%Y%m%d")
    out_txt = os.path.join(RESULTS_DIR, f"daily_report_{ymd}.txt")
    out_csv = os.path.join(RESULTS_DIR, f"daily_report_{ymd}.csv")

    cols = [
        "session",
        "ticker",
        "side",
        "qty",
        "price",
        "status",
        "broker_order_id",
        "filled_qty",
        "filled_avg",
        "fill_pct",
        "timestamp",
    ]
    detail = day_latest.reindex(columns=[c for c in cols if c in day_latest.columns]).copy()
    detail.to_csv(out_csv, index=False)

    # human-readable text
    lines = []
    lines.append(f"DAILY FILL REPORT — {day.isoformat()}")
    lines.append("=" * 72)
    lines.append(f"Orders with latest status on this date: {total_orders}")
    lines.append("Status counts:")
    for s, n in by_status.items():
        lines.append(f"  - {s}: {n}")
    lines.append(f"Estimated filled notional (qty*avg): {est_filled_notional:,.2f}")
    lines.append("")
    lines.append("Latest status per order id:")
    lines.append(
        "session                ticker side qty   price   status            order_id                               filled_qty filled_avg fill_%"
    )
    for _, r in detail.iterrows():
        lines.append(
            f"{str(r.get('session','')):22} "
            f"{str(r.get('ticker','')):6} "
            f"{str(r.get('side','')):4} "
            f"{int(r.get('qty',0)):4d} "
            f"{('' if pd.isna(r.get('price')) else f'{float(r.get('price')):.2f}'):>7} "
            f"{str(r.get('status','')):12} "
            f"{str(r.get('broker_order_id','')):36} "
            f"{('' if pd.isna(r.get('filled_qty')) else f'{float(r.get('filled_qty')):.0f}'):>10} "
            f"{('' if pd.isna(r.get('filled_avg')) else f'{float(r.get('filled_avg')):.2f}'):>9} "
            f"{('' if pd.isna(r.get('fill_pct')) else f'{float(r.get('fill_pct')):.1f}'):>6}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_txt}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
