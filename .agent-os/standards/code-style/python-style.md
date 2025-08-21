# Python Code Style Guide — FFBayes

This guide reflects project conventions, your stated preferences, and our tooling.
It aims for clear, reproducible, publication-quality Python.

## Core Principles

- Write readable, explicit code
- Prefer clarity over cleverness; favor high-verbosity, self-documenting names
- Keep logic pure and testable; isolate I/O and side-effects
- Avoid hard-coded paths; resolve paths dynamically and portably
- Prefer small functions with single responsibility and explicit inputs/outputs

## Project Tooling and Conventions

- Ruff is used for linting and formatting
  - Lint: `E` (pycodestyle), `F` (pyflakes), `I` (isort)
  - Formatter: single quotes, 4-space indentation; line length handled by formatter
- Type checking: mypy (use type hints per PEP 484)
- Tests: pytest (write explicit, deterministic assertions)

## File Structure (this repo)

- Source code lives under `src/[project_name]`
- Tests live under `tests/`
- CLI entry points are defined in `pyproject.toml` under `[project.scripts]`
- Avoid top-level executable scripts; prefer module entry points

## Paths and Configuration

- Do not hard-code absolute paths
- Use `pathlib.Path` to construct paths portably
- For project-relative paths, anchor from a known file or the project root
- Use configuration files or environment variables for user-specific locations

Example (derive project root and use data folder):
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'datasets'
RESULTS_DIR = PROJECT_ROOT / 'results'
```

When accessing packaged resources, prefer `importlib.resources`.

```python
from importlib.resources import files

resource_path = files('ffbayes') / 'data' / 'schema.json'
```

## Documentation

- Use docstrings (PEP 257) on all public modules, classes, and functions
- Explain purpose, key parameters, return values, and important assumptions
- Focus on the “why” and behavior; keep concise and accessible

Example:
```python
def compute_vor(projections: list[float], replacement_level: float) -> list[float]:
    """Compute Value Over Replacement (VOR).

    Parameters
    ----------
    projections : list[float]
        Player projections in the same units as `replacement_level`.
    replacement_level : float
        Baseline level to compare against.

    Returns
    -------
    list[float]
        VOR values for each input projection.
    """
    return [p - replacement_level for p in projections]
```

## Testing (pytest)

- Write tests that assert specific, deterministic outcomes on test data
  - Do not only assert existence (e.g., column names); assert exact values
- Parametrize where appropriate; use fixtures for setup/teardown
- Separate production code from test code; keep tests simple and focused
- Control randomness with fixed seeds and/or dependency injection

Example assertions:
```python
import pytest
from ffbayes.utils.model_validation import clamp_probability

def test_clamp_probability_exact_values():
    assert clamp_probability(-0.1) == 0.0
    assert clamp_probability(0.5) == 0.5
    assert clamp_probability(1.5) == 1.0
```

Running tests:
```bash
pytest -q
pytest -q tests/test_some_module.py::test_specific_behavior
```

### Pytest patterns
- Use fixtures for reusable setup; keep fixtures focused and explicit
- Use `tmp_path` for filesystem isolation
- Use `@pytest.mark.parametrize` to cover input matrices succinctly

Examples:
```python
import pytest
from pathlib import Path

@pytest.fixture()
def small_numbers() -> list[int]:
    return [1, 2, 3]

@pytest.mark.parametrize(
    'raw, expected',
    [(-0.1, 0.0), (0.5, 0.5), (1.5, 1.0)],
)
def test_clamp_parametrized(raw: float, expected: float) -> None:
    from ffbayes.utils.model_validation import clamp_probability
    assert clamp_probability(raw) == expected

def test_tmp_path_roundtrip(tmp_path: Path) -> None:
    fp = tmp_path / 'sample.txt'
    fp.write_text('hello')
    assert fp.read_text() == 'hello'
```

## Type Checking (mypy)
- Run locally during development: `mypy src`
- Favor gradual typing: annotate public APIs first; expand inward
- Prefer precise types; avoid `Any`; use `TypedDict`, `Protocol`, `NamedTuple`, or `@dataclass`
- Use `# type: ignore[error-code]` sparingly and with justification

Minimal `pyproject.toml` config:
```toml
[tool.mypy]
python_version = "3.10"
check_untyped_defs = true
no_implicit_optional = true
warn_unused_ignores = true
warn_return_any = true
ignore_missing_imports = true
```

## Code Style

### Formatting
- 4 spaces per indentation level
- Single quotes for strings (enforced by Ruff)
- Let the formatter handle line length and wrapping
- Imports are grouped and sorted (stdlib, third-party, local) with blank lines between groups

### Naming
- Functions are verbs/verb-phrases (e.g., `load_draft_data`, `compute_vor`)
- Variables are descriptive nouns (avoid 1–2 character names)
- Constants use SCREAMING_SNAKE_CASE
- Classes use CapWords (e.g., `ModelRunner`)
- Avoid shadowing built-ins (e.g., use `list_values` instead of `list`)

### Types
- Annotate all public function signatures and class attributes
- Prefer precise types; avoid `Any` where feasible
- Use TypedDict/Protocol/NamedTuple/`dataclasses.dataclass` for structured data

### Control Flow and Errors
- Use guard clauses and early returns to avoid deep nesting
- Handle edge cases first; raise domain-appropriate exceptions (`ValueError`, `TypeError`)
- Never use bare `except:`; catch specific exceptions and re-raise when appropriate

### Logging
- Use the `logging` module instead of prints in library code
- Log actionable context; avoid noisy or sensitive data

```python
import logging
logger = logging.getLogger(__name__)

try:
    value = expensive_operation()
except ExternalServiceError as exc:
    logger.error('Failed external call: %s', exc)
    raise
```

## Data and Reproducibility

- Keep data loading/writing isolated from core logic
- Make pure functions the default; pass dependencies explicitly
- Seed randomness (`random`, `numpy`, `pymc`, etc.) for deterministic tests

## Performance and Memory

- Optimize only after measuring; prefer readability first
- Use vectorized operations and streaming where appropriate
- Add benchmarks or performance assertions when performance is a requirement

## Example Patterns

Pure function with type hints and clear naming:
```python
def normalize_scores(scores: list[float], min_value: float, max_value: float) -> list[float]:
    if max_value <= min_value:
        raise ValueError('max_value must be greater than min_value')
    span = max_value - min_value
    return [(s - min_value) / span for s in scores]
```

Path handling with `pathlib` and project root anchor:
```python
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

MODEL_CACHE = get_project_root() / 'results' / 'model_cache'
```

---

- Simplicity: prefer simple, clear solutions
- Consistency: apply patterns uniformly across the codebase
- Readability: write code that others can immediately understand
- Maintainability: structure code for easy modification and testing 