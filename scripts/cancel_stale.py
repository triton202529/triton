# scripts/cancel_stale.py
import argparse, json
from services.broker_alpaca import AlpacaBroker

def main():
    p = argparse.ArgumentParser(description="Cancel stale open orders with whitelist and cutoff rules.")
    p.add_argument("--mode", choices=["paper","live"], default="paper")
    p.add_argument("--whitelist", default="", help="Comma-separated symbols to keep fresh (e.g. V,MRK,MSFT,META,KO)")
    p.add_argument("--cutoff", default=None, help='ISO cutoff (e.g. "2025-09-01" or "2025-09-01T15:30:00")')
    p.add_argument("--older-than-days", type=int, default=None, dest="older")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    wl = [s.strip().upper() for s in args.whitelist.split(",") if s.strip()] or None
    b = AlpacaBroker(args.mode)
    out = b.cancel_open_orders(whitelist=wl, cutoff=args.cutoff, older_than_days=args.older, limit=args.limit, dry_run=args.dry_run)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
