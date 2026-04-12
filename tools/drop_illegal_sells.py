from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# --- make project root importable (so "services" works) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.broker_alpaca import AlpacaBroker  # noqa

p = PROJECT_ROOT / "data" / "live" / "orders_today.csv"
df = pd.read_csv(p)
df.columns = [str(c).strip().lower() for c in df.columns]

# find columns
sym_col = next((c for c in ["symbol", "ticker", "sym"] if c in df.columns), None)
side_col = "side" if "side" in df.columns else None
qty_col = "qty" if "qty" in df.columns else ("quantity" if "quantity" in df.columns else None)

if not sym_col or not side_col:
    raise SystemExit(f"[BAD_SCHEMA] need symbol/ticker + side. cols={df.columns.tolist()}")

if not qty_col:
    df["qty"] = 1
    qty_col = "qty"

# positions (paper)
b = AlpacaBroker(mode="paper")
pos = b.list_positions() or []
held = {str(x.get("symbol", "")).strip().upper() for x in pos if str(x.get("symbol", "")).strip()}

df[sym_col] = df[sym_col].astype(str).str.strip().str.upper()
df[side_col] = df[side_col].astype(str).str.strip().str.upper()

is_sell = df[side_col].eq("SELL")
not_held = ~df[sym_col].isin(sorted(held))
drop = is_sell & not_held

dropped = int(drop.sum())
dropped_syms = sorted(set(df.loc[drop, sym_col].tolist())) if dropped else []

df2 = df[~drop].copy()
df2.to_csv(p, index=False)

print(f"[OK] Dropped {dropped} illegal SELL rows (not held). Saved -> {p}")
if dropped_syms:
    print("Dropped symbols:", ", ".join(dropped_syms))
