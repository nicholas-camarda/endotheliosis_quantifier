# Repository Guidelines

## Project Structure & Module Organization
- `src/eq/`: Main package.
  - `core/`: Canonical loaders and constants (e.g., `BINARY_P2C=[0,1]`, mask threshold `127`).
  - `processing/`: Image conversion and patchification.
  - `pipeline/`: Orchestration and production runners.
  - `evaluation/`: Metrics and evaluators.
  - `quantification/`: Quantification workflows (stubs in progress).
  - `utils/`: Config, logging, hardware detection (MPS/CUDA), paths.
- `configs/`: YAML configs for mito pretraining and glomeruli fine‑tuning.
- `tests/`: Organized `unit/`, `integration/`, `evaluation/` suites.
- `logs/`: Example outputs/artifacts; do not rely on for state.

## Architecture Overview
- Data flow: raw images → patchification → binary masks (threshold 127) → segmentation → features → quantification.
- Core canon: `eq.core` provides `BINARY_P2C=[0,1]`, mask conversion, and loader helpers used across pipelines.
- Training stages: mitochondria pretraining → glomeruli fine‑tuning (see `configs/*.yaml`).
- Execution: CLI entry `eq` dispatches to `data-load`, `seg`, `quant-endo`, and `production` commands.
- Hardware: mode‑aware execution (auto/development/production) with MPS/CUDA detection and sensible batch sizes.

## Build, Test, and Development Commands
- Env: `conda env create -f environment.yml && conda activate eq`
- Dev install: `pip install -e .[dev]`
- Tests: `python -m pytest -q tests/`
- Lint/format: `ruff check . && ruff format .`
- Types: `mypy src/eq`
- CLI (after install):
  - Data prep: `eq data-load --data-dir <dir> --test-data-dir <dir> --cache-dir <dir>`
  - Train (seg): `eq seg --data-dir <dir> [--epochs 50 --batch-size 8]`
  - Quantification: `eq quant-endo --data-dir <dir> --segmentation-model <pkl>`
  - Production: `eq production --data-dir <dir> --test-data-dir <dir>`
  - Mode mgmt: `eq mode --show | --set development | --validate`

## Coding Style & Naming Conventions
- Python 3.9+. Indent 4 spaces. Prefer single quotes.
- Use `ruff` for lint/format (see `pyproject.toml`); follow `.agent-os/standards/code-style/python-style.md`.
- Names: packages/modules `snake_case`; classes `CapWords`; funcs/vars `snake_case`.
- Type annotate public APIs; keep utilities small and testable.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/unit`, `tests/integration`, `tests/evaluation`.
- Names: `test_*.py`; deterministic, offline‑runnable tests.
- Cover: loaders (`eq.core.data_loading`), pipelines/CLI, evaluators.
- Run before PR: `python -m pytest -q` + `ruff check` + `mypy`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`); history shows `feat`, `cleanup`, `chore`, `docs`.
- One logical change per commit; imperative subject; concise body.
- PRs: description, linked issues, test evidence (logs/screens), and config notes.
- CI bar: tests pass; lint/type checks clean.

## Security & Configuration Tips
- No datasets, models, or secrets in Git. Configure via `configs/*.yaml`.
- TensorFlow removed; stack is fastai/PyTorch. Use `eq mode` for MPS/CUDA/CPU‑aware runs; MPS fallback handled automatically.
