# DiffML Documentation Style Guide

This guide defines how we document the public API, inline commentary, and typing
expectations across the DiffML codebase. Follow it for every change that touches
`src/diffml`, `src/diffml_article_replication`, CLI entrypoints under `scripts/`,
or demo notebooks/scripts inside `examples/`.

## Scope and Definitions

- **Public packages/modules** – every Python module inside `src/diffml` or
  `src/diffml_article_replication` whose file name does not start with `_`,
  plus CLI helpers under `scripts/` and demo scripts under `examples/`.
  Modules exported in `diffml.__all__`, `diffml_article_replication.__all__`,
  or referenced by CLI entry points are always considered public even if they
  live deeper in the tree.
- **Public objects** – classes, functions, dataclasses, and constants that are
  not prefixed with `_` *and* are either imported in `diffml.__all__` /
  `diffml_article_replication.__all__`, referenced by the CLI/tests, or used by
  the documentation. Treat helper functions inside public modules as public
  unless the function name starts with `_`.
- **Private objects** – anything prefixed with `_` or contained in tests.
  Private helpers may still carry docstrings if it increases clarity but they
  are exempt from strict coverage.

Only the public surface is required to maintain full docstring and comment
coverage. Tests, build scripts, and `__pycache__` artifacts are excluded.
If you add a new public module, update `docs/DOC_COVERAGE.md` in the same PR.

## Docstring Standard (NumPy)

We use the **NumPy docstring style** everywhere. Docstrings must be accurate,
concise, have an imperative first line (D401), and reflect actual behavior /
types. Borrowing type hints from the signature is acceptable, but the *meaning*
of each argument must be described.

### Module Docstrings

- First statement in every public `*.py` file.
- Summary line + short paragraph describing responsibilities.
- Optional sections: `Notes`, `Examples`, `References`.
- Document relevant invariants (e.g., “All simulations run in float64.”).

### Class Docstrings

- Begin with a one-line summary of the class responsibility.
- Include `Parameters` and `Attributes` sections for dataclasses or classes with
  meaningful attributes.
- If the constructor accepts keyword-only arguments, document them in the
  `Parameters` section even if the class is a dataclass.
- Mention side effects (e.g., alters RNG state) and invariants (e.g., “expects
  1-D inputs”).

### Function and Method Docstrings

Follow this template:

```python
def example(a: Tensor, b: float = 0.0) -> Tensor:
    """Short summary in imperative mood.

    Longer description when needed.

    Parameters
    ----------
    a : Tensor
        Describe the expected shape, dtype, or units.
    b : float, optional
        Mention default behavior instead of just “optional”.

    Returns
    -------
    Tensor
        Describe shape/meaning, e.g., “Loss per sample”.

    Raises
    ------
    ValueError
        Describe when and why.
    RuntimeError
        Describe when applicable.
    """
```

- `Returns` and `Raises` sections are mandatory when the function returns a
  value or intentionally raises exceptions for invalid input.
- Include `Examples` when there is non-trivial setup, numerical stability
  concerns, or to illustrate CLI usage.
- Use ASCII math notation (e.g., `S_T = S_0 * exp(...)`) when referencing
  equations; prefer LaTeX-like notation inside backticks.
- Describe tensor shapes, broadcasting assumptions, units, and device behavior.

### Special Cases

- **Property methods** – may use a short summary if the behavior is obvious.
- **Context managers / generators** – mention yielded values in `Yields`
  (NumPy style) instead of `Returns`.
- **`__call__` methods** – document them like any public method since the class
  becomes callable.

## Inline Comments

Inline comments explain *why* a block exists, not *what* a line of code does.

- Use line comments sparingly for invariant checks, tricky math, and references
  to paper equations or research.
- Place the comment on the line above the code, capitalized, with punctuation.
- Avoid comments that restate the code (“# Increment i”).
- Use docstring `Notes` sections for longer explanations; inline comments should
  stay under ~80 characters when possible.
- Prefer referencing the originating paper/equation rather than duplicating long
  derivations in comments.

## Type Hints and Sections

- All public functions/classes must be fully type hinted (parameters and
  returns). Use `typing.Optional` or `| None` consistently (follow existing
  module style).
- Docstrings should not contradict type hints. Mention semantic constraints
  (e.g., “Must be positive”) even if the type hint already says `float`.
- Always add a `Returns` section unless the function returns `None`.
- Always add a `Raises` section when validation errors are possible. Reference
  the exact exception type.
- When a function mutates its input or performs I/O, document the side effects
  explicitly in the long description or a `Notes` section.

## Public API Checklist

Before opening a PR:

1. Every module/class/function in `src/diffml`, `scripts/`, and `examples/`
   without a leading `_` has a NumPy-style docstring that matches the template.
2. Inline comments only appear where they explain reasoning or invariants.
3. Type hints cover all public callables.
4. When you introduce new public APIs (including the replication package),
   update `docs/DOC_COVERAGE.md`.
5. Run `ruff check --select D --respect-gitignore` to verify lint enforcement.

## Enforcement

Ruff’s pydocstyle rules (`D1xx`, `D2xx`, etc.) are enabled in `ruff.toml`.
`tests/` modules keep docstring checks disabled, but all other code (including
scripts) must comply. CI will fail if:

- A public object lacks a docstring.
- Docstrings don’t follow the NumPy structure (e.g., missing blank line between
  summary and sections).
- Imperative first-line rules are violated (e.g., `D401`).

Use `ruff check src scripts examples --select D` locally to validate changes
before committing.

## Additional Resources

- `docs/DOC_COVERAGE.md` – tracks current docstring status per module.
- `CONTRIBUTING.md` – includes a contributor-facing summary of these rules.
- Pydocstyle reference: <https://numpydoc.readthedocs.io/en/latest/format.html>
