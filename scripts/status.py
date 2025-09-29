#!/usr/bin/env python3
"""
Status checker for the Triton repo.

- Prints repo root and key data directories from config/alpha.yaml (if present).
- Reads Alpaca credentials from config/alpaca.json and (optionally) queries account.
- Uses UTF-8-SIG when reading text to tolerate BOM-encoded files on Windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml  # PyYAML
except Exception:
    yaml = None  # We'll fall back to defaults if YAML isn't available


def repo_root() -> Path:
    """Best-effort repo root (parent of /scripts)."""
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd()


def read_alpha_paths(root: Path) -> dict:
    """
    Read paths from config/alpha.yaml (UTF-8-SIG to tolerate BOM).
    Returns absolute paths for results/predictions/logs with sensible defaults.
    """
    defaults = {
        "results": "data/results",
        "predictions": "data/predictions",
        "logs": "data/logs",
    }
    alpha_yaml = root / "config" / "alpha.yaml"
    if alpha_yaml.exists() and yaml is not None:
        try:
            with alpha_yaml.open("r", encoding="utf-8-sig") as f:
                cfg = yaml.safe_load(f) or {}
            paths = (cfg.get("paths") or {})
            for k in defaults:
                if k in paths and isinstance(paths[k], str) and paths[k].strip():
                    defaults[k] = paths[k]
        except Exception as e:
            print(f"⚠️  Failed to parse config/alpha.yaml: {e}. Using defaults.")
    else:
        if not alpha_yaml.exists():
            print("⚠️  config/alpha.yaml not found; using default paths.")
        elif yaml is None:
            print("⚠️  PyYAML not installed; using default paths.")

    return {k: str((root / Path(v)).resolve()) for k, v in defaults.items()}


def print_dir_status(name: str, path_str: str) -> bool:
    """Prints a nice one-line status for a directory."""
    p = Path(path_str)
    ok = p.exists()
    label = f"{name:<11}"  # align like: results / predictions / logs
    print(f"{'✅' if ok else '❌'} {label} -> {path_str}")
    return ok


def read_alpaca(root: Path) -> dict | None:
    """
    Read Alpaca creds from config/alpaca.json (UTF-8-SIG to tolerate BOM).
    """
    alpaca_cfg = root / "config" / "alpaca.json"
    if not alpaca_cfg.exists():
        print("⚠️  config/alpaca.json not found; skipping Alpaca status.")
        return None
    try:
        with alpaca_cfg.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to read config/alpaca.json: {e}")
        return None


def show_alpaca_status(creds: dict | None) -> int:
    """
    Print a quick view of Alpaca configuration and (optionally) live account info.
    Never crashes the script if the network/creds are wrong.
    """
    if not creds:
        return 0

    print("\nAlpaca credentials:")
    key_id = creds.get("key_id", "<missing>")
    base_url = creds.get("base_url", "")
    paper_flag = creds.get("paper", None)

    if paper_flag is None:
        mode = "unknown"
    else:
        mode = "paper" if paper_flag else "live"

    print(f"  key_id : {key_id}")
    print(f"  mode   : {mode}")
    print(f"  base   : {base_url}")

    # Try to fetch account summary (optional)
    try:
        from alpaca.trading.client import TradingClient

        client = TradingClient(
            creds["key_id"],
            creds["secret_key"],
            paper=bool(paper_flag),
        )
        acct = client.get_account()
        # cash/equity are strings from API; print as-is
        print(f"  account: status={acct.status}, cash=${acct.cash}, equity=${acct.equity}")
        return 0
    except Exception as e:
        print(f"  ⚠️  Could not query account: {e.__class__.__name__}: {e}")
        return 1


def main() -> int:
    root = repo_root()
    print(f"Repo root: {root}")

    paths = read_alpha_paths(root)
    print()
    ok_results = print_dir_status("results", paths["results"])
    ok_preds = print_dir_status("predictions", paths["predictions"])
    ok_logs = print_dir_status("logs", paths["logs"])

    creds = read_alpaca(root)
    print()
    rc = show_alpaca_status(creds)

    # Non-zero exit if any core dirs missing or account fetch failed
    return 0 if (ok_results and ok_preds and ok_logs and rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
