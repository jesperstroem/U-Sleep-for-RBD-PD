# CI/CD Pipeline

This document describes the continuous integration and deployment setup for **usleep-rbdpd**.

---

## Overview

The pipeline uses **GitHub Actions** and runs automatically on every push and pull request.
It has three independent jobs that run in parallel:

| Job | What it checks | External deps needed? |
|-----|---------------|-----------------------|
| **Lint & Format** | Code style via Ruff | No |
| **Build Wheel** | Package metadata is valid and wheel builds cleanly | No |
| **Structural Tests** | File structure, packaging, notebook correctness | No |

> **Why no full integration tests in CI?**  
> The two core runtime dependencies (`sleep_preprocessing_pipeline` and `ml_architectures`)
> live on a private GitLab instance at `gitlab.au.dk` that is not reachable from GitHub
> Actions runners.  All checks that _do_ run in CI are network-independent.

---

## Pipeline Stages

### 1. Lint & Format (`lint`)

Runs [Ruff](https://docs.astral.sh/ruff/) — a fast Python linter and formatter.

- `ruff check .` — checks for PEP 8 violations and common errors
- `ruff format --check .` — verifies consistent formatting without modifying files

**Runs in ~30 seconds.**

### 2. Build Wheel (`build`)

Validates that the package can be built as a Python wheel.

```bash
pip install build
python -m build --wheel --no-isolation
```

The resulting `.whl` file is uploaded as a GitHub Actions artifact (retained 7 days).
This step confirms that `pyproject.toml` metadata is correct and setuptools can assemble
the distribution without fetching the private GitLab dependencies.

### 3. Structural Tests (`test`)

Runs `pytest tests/` using only the standard library — no private packages required.

Checks:
- Both model weight files exist in `weights/`
- `pyproject.toml` is present and contains valid TOML with the correct package name
- `setup.py` is absent (packaging migration completed)
- `demo.ipynb` exists, is valid JSON, and imports `matplotlib`, `USleep_Predictor`, and `plotHypnogram`

---

## Running Checks Locally

### Prerequisites

```bash
pip install "ruff>=0.9" "pytest>=8.0"
```

### Lint

```bash
ruff check .
```

### Format check (non-destructive)

```bash
ruff format --check .
```

### Auto-fix formatting

```bash
ruff format .
```

### Tests

```bash
pytest tests/ -v
```

### Pre-commit hooks (recommended)

Install [pre-commit](https://pre-commit.com/) once and let it run the linter and formatter
automatically before every commit:

```bash
pip install pre-commit
pre-commit install
```

After that, `git commit` will automatically run `ruff check --fix` and `ruff format`
on staged files.

To run all hooks manually against the entire repository:

```bash
pre-commit run --all-files
```

---

## Releases

There is no automated release/deployment workflow.  This repository is a demo/inference
tool with pre-trained weights committed directly.  When you want to tag a new release:

```bash
git tag v1.0.2
git push origin v1.0.2
```

GitHub will create a release entry.  You can attach the wheel artifact from the **Build**
job to the release via the GitHub UI.

---

## Required Secrets & Environment Variables

No secrets or environment variables are required for the CI pipeline itself.

If you add a future CD step (e.g. publishing to PyPI or GHCR), you would add these in
**Settings → Secrets and variables → Actions** on GitHub:

| Secret | Purpose |
|--------|---------|
| `PYPI_API_TOKEN` | Publish to PyPI via `twine` |
| `GITHUB_TOKEN` | Automatically provided by Actions; used for artifact uploads |

---

## File Reference

```
.github/
└── workflows/
    └── ci.yml          # The full CI pipeline definition

pyproject.toml          # Build config, dependencies, ruff & pytest settings
.pre-commit-config.yaml # Local pre-commit hooks (ruff + hygiene checks)
tests/
├── __init__.py
└── test_package.py     # Structural tests (no external deps)
docs/
└── ci-cd.md            # This file
```
