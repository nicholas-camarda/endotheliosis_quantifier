"""Path helpers for the repository's current raw/derived data layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union


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
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """Return the default raw-data directory for local training runs."""
    return _resolve_repo_path(os.getenv('EQ_DATA_PATH', 'data/raw_data'))


def get_output_path() -> Path:
    """Return the default derived-data/output directory."""
    return _resolve_repo_path(os.getenv('EQ_OUTPUT_PATH', 'data/derived_data'))


def get_cache_path() -> Path:
    """Return the default cache directory used by the pipeline."""
    cache_override = os.getenv('EQ_CACHE_PATH')
    if cache_override:
        return _resolve_repo_path(cache_override)
    return get_output_path() / 'cache'


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
