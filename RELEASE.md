# Release process for `diffml`

This document describes how to cut and publish a new release of
`diffml`.

The project uses:

* **Poetry** for packaging and dependency management.
* **Semantic Versioning** (`MAJOR.MINOR.PATCH`).
* **Git tags** of the form `vX.Y.Z`.
* **GitHub Actions** to build and publish the package when a tag is pushed.

---

## 1. Versioning policy

We follow [Semantic Versioning](https://semver.org/):

* **PATCH** (`X.Y.Z → X.Y.(Z+1)`): bug fixes or small internal changes that do not
  alter public APIs or expected behaviour.
* **MINOR** (`X.Y.Z → X.(Y+1).0`): backwards-compatible feature additions, new
  experiments, new datasets, new configuration options.
* **MAJOR** (`X.Y.Z → (X+1).0.0`): breaking changes to public APIs, behaviour,
  or configuration.

Pre-releases (e.g. `v0.2.0-rc.1`) can be used if needed, but the default flow
assumes stable tags (`vX.Y.Z`).

---

## 2. Branch model

A simple branch model is assumed:

* `main` – always releasable. Tagged releases are created from this branch.
* `develop` (optional) – integration branch for ongoing work; merged into
  `main` before cutting a release.

All releases **must** be tagged from `main`.

If you use a different model, adapt the references to match your workflow.

---

## 3. Pre-release checklist

Before cutting a release:

1. **Ensure `main` is up to date**

   ```bash
   git checkout main
   git pull origin main
   ```

2. **Confirm CI is green**

   * Latest commits on `main` should have passed the `CI` workflow:

     * Ruff
     * mypy
     * pytest

3. **Update documentation**

   * Ensure `README.md` reflects current usage.
   * Update `CONTRIBUTING.md` if the contribution or testing process changed.
   * Update any experiment descriptions if outputs or configuration changed.

4. **Update `CHANGELOG.md` (if present)**

   * Move entries from the “Unreleased” section to a new `vX.Y.Z` section.
   * Briefly describe:

     * New features
     * Fixes
     * Breaking changes (if any)

5. **Update the version in `pyproject.toml` via Poetry**

   Choose the appropriate bump:

   ```bash
   # One of:
   poetry version patch
   poetry version minor
   poetry version major
   ```

   Verify the new version:

   ```bash
   poetry version
   # Example output: diffml 0.2.0
   ```

6. **Sanity check build and tests locally**

   ```bash
   poetry install --with dev

   poetry run ruff check .
   poetry run mypy src tests
   poetry run pytest
   ```

   Optionally run the full experiment script:

   ```bash
   poetry run python scripts/run_all_experiments.py
   ```

---

## 4. Tagging and pushing a release

Assuming the new version is `0.2.0`:

1. **Commit all changes**

   ```bash
   git status   # ensure working tree is clean
   git add .
   git commit -m "chore: prepare release v0.2.0"
   ```

2. **Create an annotated tag**

   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   ```

3. **Push branch and tag**

   ```bash
   git push origin main
   git push origin v0.2.0
   ```

This will trigger the **Release** workflow (`.github/workflows/release.yml`),
which:

* Builds the package via Poetry (`poetry build`).
* Publishes the build artifacts to the configured index (PyPI or TestPyPI),
  using the token stored in `secrets.PYPI_API_TOKEN` (or equivalent).

Monitor the GitHub Actions tab to confirm the release job completes successfully.

---

## 5. PyPI / TestPyPI configuration

The release workflow is designed to use a PyPI-like index.

1. **Configure secrets in the repository settings**

   * `PYPI_API_TOKEN` (for real PyPI), or
   * `TEST_PYPI_API_TOKEN` (for TestPyPI), depending on how the workflow is set up.

2. **Check the `release.yml` configuration**

   * Confirm the action `pypa/gh-action-pypi-publish` points to the expected
     repository:

     * PyPI: default index
     * TestPyPI: `repository_url` set to the TestPyPI URL.

3. **First-time dry run (optional)**

   For an initial smoke test, you can:

   * Configure the workflow to target TestPyPI.
   * Cut a small release (e.g. `v0.1.0`).
   * Confirm that installation from TestPyPI works:

     ```bash
     pip install --index-url https://test.pypi.org/simple/ diffml-article-replication
     ```

   Once satisfied, switch the workflow to use real PyPI and the `PYPI_API_TOKEN`
   secret.

---

## 6. Hotfix releases

For urgent fixes:

1. Branch from the latest tag on `main`:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/0.2.1
   ```

2. Apply the fix, update tests, and bump the version patch:

   ```bash
   poetry version patch  # e.g. 0.2.1
   ```

3. Run tests and linters as usual.

4. Merge `hotfix/0.2.1` back into `main` (and `develop` if applicable).

5. Tag and push as in section 4.

---

## 7. Post-release steps

After a successful release:

* Update any **external references**:

  * Documentation or blog posts that refer to versions.
  * Example commands in READMEs that pin specific versions.

* If you maintain a **CHANGELOG**, start a new “Unreleased” section for future
  work.

* Optionally create a **GitHub Release** entry for the tag and paste:

  * Highlights of the release.
  * The corresponding section from `CHANGELOG.md`.

---

## 8. Troubleshooting

* **Release workflow fails**

  * Check the Actions log for:

    * Build errors (`poetry build`).
    * Authentication errors with PyPI (missing or invalid token).
  * Fix the issue, ensure `main` is updated, and re-push the tag or create a
    new one (e.g. `v0.2.1`).

* **Package not visible on PyPI**

  * Confirm the workflow used the intended index (PyPI vs TestPyPI).
  * Check that the project name in `pyproject.toml` matches what you search for.

If problems persist, open a GitHub issue with logs and exact commands used.
