"""Path helpers for repository-local and runtime-root artifact layout."""

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
DEFAULT_DOX_LABEL_STUDIO_EXPORT_ENV = "EQ_DOX_LABEL_STUDIO_EXPORT"
DEFAULT_MR_SCORE_WORKBOOK_ENV = "EQ_MR_SCORE_WORKBOOK"
DEFAULT_MR_IMAGE_ROOT_ENV = "EQ_MR_IMAGE_ROOT"


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
    """Return the default raw-data directory, preferring the active runtime tree."""
    override = os.getenv('EQ_DATA_PATH')
    if override:
        return _resolve_repo_path(override)
    runtime_raw_data = get_active_runtime_root() / "raw_data"
    if runtime_raw_data.exists():
        return runtime_raw_data
    return _resolve_repo_path(DEFAULT_DATA_PATH)


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
    """Return the default derived-data directory, preferring the active runtime tree."""
    override = os.getenv('EQ_OUTPUT_PATH')
    if override:
        return _resolve_repo_path(override)
    runtime_derived_data = get_active_runtime_root() / "derived_data"
    if runtime_derived_data.exists():
        return runtime_derived_data
    return _resolve_repo_path(DEFAULT_OUTPUT_PATH)


def get_runtime_output_path() -> Path:
    """Return the runtime output root, preferring the active runtime tree."""
    output_override = os.getenv(DEFAULT_RUNTIME_OUTPUT_ENV)
    if output_override:
        return _resolve_repo_path(output_override)
    return get_active_runtime_root() / "output"


def _runtime_root_or_active(runtime_root: Union[str, Path, None] = None) -> Path:
    if runtime_root is None:
        return get_active_runtime_root()
    return _resolve_repo_path(runtime_root)


def get_runtime_raw_data_path(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the runtime raw-data root under the active runtime tree."""
    return _runtime_root_or_active(runtime_root) / "raw_data"


def get_runtime_cohorts_root(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the canonical runtime cohort root."""
    return get_runtime_raw_data_path(runtime_root) / "cohorts"


def get_runtime_cohort_path(cohort_id: str, runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the localized runtime directory for a cohort."""
    return get_runtime_cohorts_root(runtime_root) / str(cohort_id)


def get_runtime_cohort_manifest_path(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the unified runtime cohort manifest path."""
    return get_runtime_cohorts_root(runtime_root) / "manifest.csv"


def get_runtime_cohort_output_root(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the runtime cohort-output root."""
    if runtime_root is None:
        return get_runtime_output_path() / "cohorts"
    return _runtime_root_or_active(runtime_root) / "output" / "cohorts"


def get_runtime_cohort_output_path(cohort_id: str, runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the output directory for a specific cohort."""
    return get_runtime_cohort_output_root(runtime_root) / str(cohort_id)


def get_dox_label_studio_export_path() -> Path:
    """Return the latest Dox Label Studio export path used for cohort ingestion."""
    override = os.getenv(DEFAULT_DOX_LABEL_STUDIO_EXPORT_ENV)
    if override:
        return _resolve_repo_path(override)
    return (
        Path.home()
        / "Library/CloudStorage/OneDrive-Personal/phd/projects/VEGFRi and Dox/in-vivo mouse projects/kidney/2023-11-16_all-labeled-glom-data.json"
    )


def get_mr_score_workbook_path() -> Path:
    """Return the MR kidney score workbook path used for cohort ingestion."""
    override = os.getenv(DEFAULT_MR_SCORE_WORKBOOK_ENV)
    if override:
        return _resolve_repo_path(override)
    return (
        Path.home()
        / "Library/CloudStorage/OneDrive-Personal/phd/projects/VEGFRi and MR/in-vivo projects/kidney/VEGFRi and MR kidney scoring.xlsx"
    )


def get_mr_image_root_path() -> Path:
    """Return the MR external-drive whole-field TIFF root used for cohort ingestion."""
    override = os.getenv(DEFAULT_MR_IMAGE_ROOT_ENV)
    if override:
        return _resolve_repo_path(override)
    return Path(
        "/Volumes/USB EXT2020/older-years/2020-2024 PhD/projects/VEGFRi and MR/in-vivo projects/kidney/images"
    )


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
