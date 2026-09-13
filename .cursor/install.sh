#!/usr/bin/env bash
# Idempotent dependency setup for the Triton Cloud Agent environment.
set -euo pipefail

cd "$(dirname "$0")/.."

# Python virtualenv support is not always present in the base image.
if ! dpkg -s python3.12-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi

# Create the virtualenv only if it does not already exist.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt

# alpaca-trade-api's metadata pins pandas<1.6, which is incompatible with the
# rest of the modern stack. It runs fine against the pinned pandas 2.2, so it is
# installed without its (overly strict) dependency constraints.
.venv/bin/pip install --no-deps "alpaca-trade-api==3.2.0"

echo "Triton environment ready."
