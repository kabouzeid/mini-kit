# Repository Guidelines

## Project Structure & Module Organization
`src/mini/` houses the library packages: `config/` for configuration loaders, `builder/` for object factories, and `trainer/` for training orchestration, hooks, and utilities. Keep new modules within the relevant subpackage and expose public APIs via each `__init__.py`. Tests live in `tests/` and mirror the source layout (`tests/test_builder.py`, etc.); add new tests alongside the feature you extend. `pyproject.toml` and `uv.lock` track dependencies and build metadata—update them when adding third-party packages.

## Build, Test, and Development Commands
Run `uv sync --all-extras --dev` once to create a virtual environment populated with runtime and developer dependencies. Use `uv run pytest` to execute the entire unit suite locally. `uv run ruff check src tests` lints the codebase; append `--fix` to auto-apply safe fixes. To publish or verify packaging, run `uv build`, which uses the configured `uv_build` backend.

## Coding Style & Naming Conventions
Python files use 4-space indentation and should be formatted with `ruff`. Module and function names stay `snake_case`, classes use `PascalCase`, and constants are `UPPER_SNAKE_CASE`. Prefer explicit type hints.

## Testing Guidelines
Write pytest tests named `test_*`. Group related assertions by behavior rather than implementation detail. Aim to cover edge cases for configuration parsing, builder registration, and trainer hooks. Run `uv run pytest` before pushing, and add parametrized tests when covering new combinations.

## Commit & Pull Request Guidelines
Commits follow short, imperative messages and git Conventional Commits. Keep each commit scoped to a logical change. Pull requests should reference the motivating issue when available, summarize behavior changes. Ensure lint and test commands pass before requesting review.
