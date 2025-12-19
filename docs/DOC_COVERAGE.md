# Documentation Coverage

_Last updated: 2025-12-18_

## Top-Level Packages and Modules

- **`diffml`** – core library exposed via `diffml.__all__`. Modules:
  `bs_analytics`, `config`, `config_experiments`, `datasets_*`, `experiments_*`,
  `experiments_registry`, `losses`, `networks`, `run_experiment`, `simulation`,
  and `training`. Status: ✅ All public modules/classes/functions documented.
- **`scripts`** – automation/CLI entrypoints:
  - `scripts/run_all_experiments.py` – ✅ Expanded CLI docstrings, inline comment
    covering the `experiments_to_run` sentinel.
  - `scripts/run_experiment.py` – ✅ CLI docstring now clarifies behavior and exit
    codes.
- **`examples`** – runnable notebooks/scripts for demonstrations:
  - `examples/demo_path_dependent.py` – ✅ Demo helpers all documented; added an
    inline comment explaining the precision requirement.
- **`run_tests.py`** – ✅ Provides a documented helper for invoking pytest.

Packages under `configs/`, `docs/`, `paper/`, and `tests/` either store data or
are excluded from docstring enforcement (tests are covered by a Ruff per-file
ignore because they intentionally omit docstrings).

## Missing Docstrings

Automated checks (`python -m compileall` for syntax + a custom AST scan +
`ruff check --select D100,D101,D102,D103`) report **no missing public
docstrings** in the public API. Tests remain ignored by design.

## Updates in This PR

- Added `docs/DOCS_STYLE.md` to codify the NumPy docstring standard, public API
  definition, inline comment rules, and type-hint/Returns/Raises expectations.
- Documented the workflow inside `CONTRIBUTING.md`, including links to the new
  style guide and coverage checklist.
- Updated Ruff configuration to enforce pydocstyle checks on every module except
  tests.
- Expanded CLI docstrings (`scripts/run_all_experiments.py`, `scripts/run_experiment.py`)
  and added clarifying inline comments in both the CLI and demo scripts.

## Next Steps

- Keep this file updated whenever new public modules are added or docstring
  coverage changes.
- For new experiments/datasets, copy the template from `docs/DOCS_STYLE.md` and
  ensure `ruff check --select D` passes before opening a PR.
