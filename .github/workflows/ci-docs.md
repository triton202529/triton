# CI docs — GitHub Actions (CI)

Location: `.github/workflows/ci.yml`

## What the workflow does
- Runs on `push` and `pull_request` for branches `main` and `master`.
- Matrix tests on Python 3.11 and 3.12 (adjustable).
- Installs dependencies from `requirements.txt` and `requirements-dev.txt` (if present).
- Runs `pytest -q` to execute tests.
- Runs a fast smoke test: `python run_adaptive_pipeline.py --skip-subprocesses` (non-failing).
- Uploads `data/results/**` as artifacts so you can examine generated risk reports.

## Files to check-in
- `.github/workflows/ci.yml` — the workflow (already added).
- `requirements-dev.txt` — development dependencies (this file).
- `pytest.ini` — existing config (ensures collection from `tests/`).
- `tests/` — your test suite.

## How to modify
- To run full pipeline in CI (slower), remove `--skip-subprocesses`.
- To add additional Python versions, update the matrix under `strategy.matrix.python-version`.
- If you need system libraries (e.g. `gfortran`, `libpq-dev`), add apt installs in the `Install system dependencies` step.

## Viewing CI artifacts
- After a run completes, navigate to the workflow run in GitHub Actions UI.
- Look under `Artifacts` to download `triton-results` and inspect `data/results/risk_report.json` etc.

## Troubleshooting
- If CI fails due to missing system-level packages, install them in the `Install system dependencies` step.
- If tests import local packages (`services`), ensure tests run from repository root (workflow does `actions/checkout` so this is satisfied).
