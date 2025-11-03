Agents Guidance for ut-6350-predator-detection (scope: repo root)
Python 3.12; create venv: `python3 -m venv .venv && . .venv/bin/activate`.
Install deps: `pip install -r requirements.txt` (keep requirements.txt current via `make export-reqs`).
Run comparison: `make compare`; relocate non-animal images: `make filter`.
Tests: `make test`; single test: `make test-one FILE=tests/test_metrics.py TEST=test_function`.
Formatting: `make format`; lint check: `make lint` (black 120 cols + isort groups: stdlib, thirdparty, local).
Imports: absolute from `src`; no wildcard; one per line grouped stdlib/thirdparty/local.
Types: use PEP484; add `from __future__ import annotations` in new modules.
Naming: snake_case funcs/vars/modules; PascalCase classes; UPPER_CASE constants.
Functions <=50 lines; extract helpers instead of long scripts.
Errors: raise specific exceptions; no bare `except:`; include context; print only user-facing progress.
CLI guard every executable script with `if __name__ == "__main__":`.
State: avoid global mutable; pass dependencies explicitly.
Images may be reorganized: non-animal originals moved to `iNaturalist/not_detected/<original_folder>/` by filter script.
Performance: prefer vectorized/batch (Torch/Lightning) over Python loops.
Dependencies: only add when needed; version ranges `>=,<` preferred; pin if reproducibility required.
Commits: minimal diff; message explains WHY, not just WHAT.
Security: never commit secrets; configuration in requirements.txt or simple env vars.
No Cursor/Copilot rule files present; this doc supersedes other guidance.
