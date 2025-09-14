# Repository Guidelines

## Project Structure & Module Organization
- `src/eq/`: main package
  - `core/` (types, constants), `data_management/` (IO, loaders), `processing/` (preprocessing), `training/` (train_* scripts), `inference/`, `evaluation/`, `pipeline/`, `utils/`.
- `tests/`: pytest suites (e.g., `datablock_tests/`).
- `configs/`: YAML configs for training (`*.yaml`).
- `assets/`: example plots used in docs.
- `models/`: training outputs (checkpoints, plots).
- `data/`: `raw_data/` and `derived_data/` inputs.
- `output/`, `test_output/`: run artifacts during development.

## Build, Test, and Development Commands
- Create env: `mamba env create -f environment.yml && mamba activate eq`
- Install dev: `pip install -e .[dev]`
- Lint: `ruff check .` (static checks) • Format: `ruff format .`
- Tests: `python -m pytest -q`
- CLI help: `eq --help`
- Validate data names: `eq validate-naming --data-dir data/raw_data/your_project`
- Extract EM images: `eq extract-images --input-dir ... --output-dir data/derived_data/mito`
- Train (examples):
  - Mito: `python -m eq.training.train_mitochondria --data-dir data/derived_data/mito --model-dir models/segmentation/mitochondria --use-dynamic-patching`
  - Glomeruli: `python -m eq.training.train_glomeruli --data-dir data/raw_data/your_project --model-dir models/segmentation/glomeruli --use-dynamic-patching`

## Coding Style & Naming Conventions
- Python 3.10 with type hints where practical.
- Ruff controls lint and formatting (single quotes, spaces; see `pyproject.toml`).
- Modules/files: `snake_case.py`; functions/vars: `snake_case`; classes: `PascalCase`; constants: `UPPER_CASE` (see `eq.core.constants`).
- Keep public CLI in `eq.__main__` and `training/` modules; prefer small, testable utilities in `utils/`.

## Testing Guidelines
- Use pytest; place files under `tests/` named `test_*.py`.
- Prefer unit tests for loaders, transforms, and trainers; avoid filesystem writes outside `test_output/`.
- Run locally with `python -m pytest -q`; add focused tests near changed code.

## Commit & Pull Request Guidelines
- Use Conventional Commits style seen in history: `feat:`, `fix:`, `docs:`, `chore:`, `config:`; imperative, present tense; concise (<72 chars subject).
- Before PR: `ruff check .`, `ruff format --check .`, and `pytest` all pass.
- PR description: problem, approach, risks, and how to validate (include CLI examples). Link issues. Update docs/README when flags or outputs change. Include key plots (e.g., `training_loss.png`) if relevant.

## Security & Configuration Tips
- Do not commit data or secrets. Keep settings in `configs/*.yaml`; inspect hardware with `eq capabilities` and `eq mode --show`.
- Prefer `environment.yml` for reproducibility; pin extra tools via `[project.optional-dependencies].dev`.

