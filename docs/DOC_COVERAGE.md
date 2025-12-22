# Documentation Coverage

_Last updated: 2025-12-22_

## Top-Level Packages and Modules

- **`diffml`** – core library exposed via `diffml.__all__`. Modules cover
  analytical utilities (`bs_analytics`), configuration helpers (`config`,
  `config_experiments`), dataset/experiment generators (`datasets_*`,
  `experiments_*`, `experiments_registry`), training utilities (`losses`,
  `networks`, `simulation`, `training`), and the CLI-friendly runner
  (`run_experiment`). All public modules, classes, and functions now use
  NumPy-style docstrings with imperative summaries and fully documented
  `Parameters`/`Returns`/`Raises` sections.
- **`diffml_article_replication`** – lightweight helper package for the paper
  replication scripts. Registry helpers and the standalone experiment runner are
  documented and linted with the same rules as the core package.
- **`scripts`** – automation entry points (`scripts/run_all_experiments.py`,
  `scripts/run_experiment.py`, `scripts/run_benchmark_digital.py`). Each script
  documents CLI arguments, side effects (e.g., creating output directories), and
  failure modes. Inline comments focus on invariants such as the sentinel for
  “run every experiment”.
- **`examples`** – demo notebooks and helper scripts. Example datasets include
  short docstrings that explain the financial context and the double-precision
  requirement.
- **`run_tests.py`** – documented wrapper describing how it discovers tests and
  which environment variables influence the run.

Packages under `configs/`, `docs/`, `paper/`, and `tests/` either store
configuration/prose or are intentionally excluded from docstring enforcement
(`tests/` carries a Ruff per-file ignore for `D`).

## Missing Docstrings

`ruff check src scripts examples --select D100,D101,D102,D103` currently reports
**no missing public docstrings**. Tests remain ignored by design.

## Updates in This PR

- Expanded `docs/DOCS_STYLE.md` and `CONTRIBUTING.md` with explicit guidance on
  what counts as the public API, how to structure inline comments, and how to
  keep this coverage report up to date.
- Enabled `pydocstyle`’s D401 rule by removing the global ignore from
  `ruff.toml`, ensuring that every summary line uses the imperative mood.
- Reworded `diffml.experiments_registry.register_experiment` and
  `diffml_article_replication.run_experiment.run_experiment_from_dict` so they
  comply with the stricter lint rules.

## Next Steps

- Update this report whenever new public modules are added or removed.
- For new experiments/datasets, copy the template from `docs/DOCS_STYLE.md` and
  ensure `poetry run ruff check --select D src scripts examples` passes before
  opening a PR.
