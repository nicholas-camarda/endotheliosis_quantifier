"""Path helpers for repository-local and runtime-root artifact layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

DEFAULT_DATA_PATH = "raw_data"
DEFAULT_OUTPUT_PATH = "derived_data"
DEFAULT_CACHE_PATH = "derived_data/cache"
DEFAULT_MODEL_PATH = "models"
DEFAULT_LOGS_PATH = "logs"
DEFAULT_RUNTIME_ROOT_ENV = "EQ_RUNTIME_ROOT"
DEFAULT_RUNTIME_OUTPUT_ENV = "EQ_RUNTIME_OUTPUT_PATH"
DEFAULT_RUNTIME_MODELS_ENV = "EQ_RUNTIME_MODEL_PATH"
DEFAULT_DOX_LABEL_STUDIO_EXPORT_ENV = "EQ_DOX_LABEL_STUDIO_EXPORT"
DEFAULT_DOX_ASSIGNMENT_WORKBOOK_ENV = "EQ_DOX_ASSIGNMENT_WORKBOOK"
DEFAULT_DOX_SCORE_WORKBOOK_ENV = "EQ_DOX_SCORE_WORKBOOK"
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


def resolve_runtime_path(raw_path: Union[str, Path]) -> Path:
    """Resolve a runtime-root-relative path without creating repo-local artifacts."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return get_active_runtime_root() / path


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
    return get_active_runtime_root() / DEFAULT_DATA_PATH


def get_active_runtime_root() -> Path:
    """Return the active runtime root for local artifact-heavy workflows."""
    runtime_override = os.getenv(DEFAULT_RUNTIME_ROOT_ENV)
    if runtime_override:
        return _resolve_repo_path(runtime_override)

    return Path.home() / "ProjectsRuntime" / get_repo_root().name


def get_output_path() -> Path:
    """Return the default derived-data directory, preferring the active runtime tree."""
    override = os.getenv('EQ_OUTPUT_PATH')
    if override:
        return _resolve_repo_path(override)
    return get_active_runtime_root() / DEFAULT_OUTPUT_PATH


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


def get_runtime_cohort_manifest_summary_path(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the generated runtime cohort manifest summary path."""
    return _runtime_root_or_active(runtime_root) / "derived_data" / "cohort_manifest" / "manifest_summary.json"


def get_runtime_mitochondria_data_path(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the installed mitochondria full-image dataset root."""
    return get_runtime_raw_data_path(runtime_root) / "mitochondria_data"


def get_runtime_segmentation_evaluation_root(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the runtime segmentation-evaluation root."""
    if runtime_root is None:
        return get_runtime_output_path() / "segmentation_evaluation"
    return _runtime_root_or_active(runtime_root) / "output" / "segmentation_evaluation"


def get_runtime_segmentation_evaluation_path(
    result_name: str, runtime_root: Union[str, Path, None] = None
) -> Path:
    """Return a segmentation-evaluation directory."""
    return get_runtime_segmentation_evaluation_root(runtime_root) / str(result_name)


def get_runtime_predictions_root(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the runtime model-prediction root."""
    if runtime_root is None:
        return get_runtime_output_path() / "predictions"
    return _runtime_root_or_active(runtime_root) / "output" / "predictions"


def get_runtime_prediction_path(
    task_name: str, runtime_root: Union[str, Path, None] = None
) -> Path:
    """Return a task-specific model-prediction directory."""
    return get_runtime_predictions_root(runtime_root) / str(task_name)


def get_runtime_segmentation_evaluation_root(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the runtime segmentation-evaluation root."""
    if runtime_root is None:
        return get_runtime_output_path() / "segmentation_evaluation"
    return _runtime_root_or_active(runtime_root) / "output" / "segmentation_evaluation"


def get_runtime_segmentation_evaluation_path(
    result_name: str, runtime_root: Union[str, Path, None] = None
) -> Path:
    """Return a segmentation-evaluation directory."""
    return get_runtime_segmentation_evaluation_root(runtime_root) / str(result_name)


def get_runtime_predictions_root(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the runtime model-prediction root."""
    if runtime_root is None:
        return get_runtime_output_path() / "predictions"
    return _runtime_root_or_active(runtime_root) / "output" / "predictions"


def get_runtime_prediction_path(
    task_name: str, runtime_root: Union[str, Path, None] = None
) -> Path:
    """Return a task-specific model-prediction directory."""
    return get_runtime_predictions_root(runtime_root) / str(task_name)


def get_runtime_quantification_results_root(runtime_root: Union[str, Path, None] = None) -> Path:
    """Return the runtime quantification-result root."""
    if runtime_root is None:
        return get_runtime_output_path() / "quantification_results"
    return _runtime_root_or_active(runtime_root) / "output" / "quantification_results"


def get_runtime_quantification_result_path(
    result_name: str, runtime_root: Union[str, Path, None] = None
) -> Path:
    """Return a quantification-result directory."""
    return get_runtime_quantification_results_root(runtime_root) / str(result_name)


def get_dox_label_studio_export_path() -> Path:
    """Return the latest Dox Label Studio export path used for cohort ingestion."""
    override = os.getenv(DEFAULT_DOX_LABEL_STUDIO_EXPORT_ENV)
    if override:
        return _resolve_repo_path(override)
    return (
        Path.home()
        / "Library/CloudStorage/OneDrive-Personal/phd/projects/VEGFRi and Dox/in-vivo mouse projects/kidney/2023-11-16_all-labeled-glom-data.json"
    )


def get_dox_assignment_workbook_path() -> Path:
    """Return the Dox randomization/assignment workbook used for identity."""
    override = os.getenv(DEFAULT_DOX_ASSIGNMENT_WORKBOOK_ENV)
    if override:
        return _resolve_repo_path(override)
    return (
        Path.home()
        / "Library/CloudStorage/OneDrive-Personal/phd/projects/VEGFRi and Dox/in-vivo mouse projects/kidney/Rand_Assign.xlsx"
    )


def get_dox_score_workbook_path() -> Path:
    """Return the Dox image-level score workbook used for score-integrity audits."""
    override = os.getenv(DEFAULT_DOX_SCORE_WORKBOOK_ENV)
    if override:
        return _resolve_repo_path(override)
    return (
        Path.home()
        / "Library/CloudStorage/OneDrive-Personal/phd/projects/VEGFRi and Dox/in-vivo mouse projects/kidney/results/2023-11-16_all-labeled-glom-data_score-table-filtered.xlsx"
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
    return get_active_runtime_root() / DEFAULT_CACHE_PATH


def get_models_path() -> Path:
    """Return the default model directory."""
    model_override = os.getenv('EQ_MODEL_PATH')
    if model_override:
        return _resolve_repo_path(model_override)
    return get_runtime_models_path()


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
    return get_active_runtime_root() / DEFAULT_LOGS_PATH


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
