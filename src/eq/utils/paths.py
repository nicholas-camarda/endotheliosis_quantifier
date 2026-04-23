"""Path helpers for the repository's current raw/derived data layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

DEFAULT_DATA_PATH = "data/raw_data"
DEFAULT_OUTPUT_PATH = "data/derived_data"
DEFAULT_CACHE_PATH = "data/derived_data/cache"
DEFAULT_MODEL_PATH = "models"
DEFAULT_LOGS_PATH = "logs"
DEFAULT_RUNTIME_ROOT_ENV = "EQ_RUNTIME_ROOT"
DEFAULT_RUNTIME_OUTPUT_ENV = "EQ_RUNTIME_OUTPUT_PATH"
DEFAULT_RUNTIME_MODELS_ENV = "EQ_RUNTIME_MODEL_PATH"


def get_repo_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(raw_path: Union[str, Path]) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return get_repo_root() / path


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = _resolve_repo_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """Return the default raw-data directory for local training runs."""
    return _resolve_repo_path(os.getenv('EQ_DATA_PATH', DEFAULT_DATA_PATH))


def get_active_runtime_root() -> Path:
    """Return the active runtime root for local artifact-heavy workflows."""
    runtime_override = os.getenv(DEFAULT_RUNTIME_ROOT_ENV)
    if runtime_override:
        return _resolve_repo_path(runtime_override)

    runtime_candidate = Path.home() / "ProjectsRuntime" / get_repo_root().name
    if runtime_candidate.exists():
        return runtime_candidate

    return get_repo_root()


def get_output_path() -> Path:
    """Return the default derived-data/output directory."""
    return _resolve_repo_path(os.getenv('EQ_OUTPUT_PATH', DEFAULT_OUTPUT_PATH))


def get_runtime_output_path() -> Path:
    """Return the runtime output root, preferring the active runtime tree."""
    output_override = os.getenv(DEFAULT_RUNTIME_OUTPUT_ENV)
    if output_override:
        return _resolve_repo_path(output_override)
    return get_active_runtime_root() / "output"


def get_cache_path() -> Path:
    """Return the default cache directory used by the pipeline."""
    cache_override = os.getenv('EQ_CACHE_PATH')
    if cache_override:
        return _resolve_repo_path(cache_override)
    return _resolve_repo_path(DEFAULT_CACHE_PATH)


def get_models_path() -> Path:
    """Return the default model directory."""
    return _resolve_repo_path(os.getenv('EQ_MODEL_PATH', DEFAULT_MODEL_PATH))


def get_runtime_models_path() -> Path:
    """Return the runtime model root, preferring the active runtime tree."""
    model_override = os.getenv(DEFAULT_RUNTIME_MODELS_ENV)
    if model_override:
        return _resolve_repo_path(model_override)
    return get_active_runtime_root() / "models"


def get_logs_path() -> Path:
    """Return the default logs directory."""
    log_override = os.getenv('EQ_LOG_PATH') or os.getenv('EQ_LOGS_PATH')
    if log_override:
        return _resolve_repo_path(log_override)
    return _resolve_repo_path(DEFAULT_LOGS_PATH)


def find_files(
    directory: Union[str, Path],
    pattern: str = '*',
    recursive: bool = True,
) -> List[Path]:
    """Find files matching a pattern in a directory."""
    directory = Path(directory)
    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))


def get_relative_path(path: Union[str, Path], base: Union[str, Path, None] = None) -> str:
    """Return a path relative to ``base`` when possible."""
    path = Path(path)
    base_path = Path.cwd() if base is None else Path(base)

    try:
        return str(path.relative_to(base_path))
    except ValueError:
        return str(path)
