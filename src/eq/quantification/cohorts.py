"""Unified runtime cohort manifest and admission utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from eq.utils.paths import (
    get_active_runtime_root,
    get_dox_label_studio_export_path,
    get_mr_image_root_path,
    get_mr_score_workbook_path,
    get_runtime_cohort_manifest_path,
    get_runtime_cohort_manifest_summary_path,
    get_runtime_cohort_path,
)

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
MASK_SUFFIXES = {'.png', '.tif', '.tiff'}

COHORT_LAUREN_PREECLAMPSIA = 'lauren_preeclampsia'
LANE_MANUAL_MASK_CORE = 'manual_mask_core'
LANE_MANUAL_MASK_EXTERNAL = 'manual_mask_external'
LANE_SCORED_ONLY = 'scored_only'
LANE_MR_CONCORDANCE_ONLY = 'mr_concordance_only'
LEGACY_LANE_ALIASES = {
    'manual_mask': LANE_MANUAL_MASK_CORE,
    'masked_external': LANE_MANUAL_MASK_EXTERNAL,
}
TRAINING_MASK_LANES = {
    LANE_MANUAL_MASK_CORE,
    LANE_MANUAL_MASK_EXTERNAL,
    *LEGACY_LANE_ALIASES.keys(),
}

HUMAN_REQUIRED_COLUMNS = ('cohort_id', 'image_path', 'score')
SCORE_LOCATOR_COLUMNS = ('source_score_row', 'source_sample_id')
HUMAN_OPTIONAL_COLUMNS = (
    'mask_path',
    'treatment_group',
    'source_score_sheet',
    'score_path',
    'source_image_name',
    'source_sample_id',
    'source_score_row',
    'source_batch',
    'source_date',
    'score_reduction_method',
    'replicate_count',
)
GENERATED_COLUMNS = (
    'manifest_row_id',
    'harmonized_id',
    'join_status',
    'verification_status',
    'lane_assignment',
    'admission_status',
    'exclusion_reason',
    'image_sha256',
    'mask_sha256',
    'mapping_review_status',
    'discovery_surfaces',
)
ENRICHED_MANIFEST_COLUMNS = tuple(
    dict.fromkeys(
        HUMAN_REQUIRED_COLUMNS
        + SCORE_LOCATOR_COLUMNS
        + HUMAN_OPTIONAL_COLUMNS
        + GENERATED_COLUMNS
    )
)
NO_SOURCE_PATH_COLUMNS = (
    'original_source_path',
    'source_path',
    'original_image_path',
    'original_mask_path',
    'source_image_path',
    'source_mask_path',
)
DEFAULT_DISCOVERY_SURFACES = (
    'localized_runtime_assets',
    'score_workbook_or_labelstudio_export',
    'cohort_metadata_logs',
)
COHORT_DISCOVERY_REQUIREMENTS = {
    COHORT_LAUREN_PREECLAMPSIA: (
        'active_preeclampsia_runtime',
        'labelstudio_annotations',
    ),
    'vegfri_dox': ('latest_dox_label_studio_export', 'decoded_brushlabel_masks.csv'),
    'vegfri_mr': ('mr_workbook', 'external_drive_whole_field_tiffs'),
}
DEFAULT_IMAGE_MIN_COMPONENT_AREA = 64
DOX_MASK_QUALITY_MIN_FOREGROUND_FRACTION = 0.001
DOX_MASK_QUALITY_MAX_FOREGROUND_FRACTION = 0.25
DEFAULT_LAUREN_PREECLAMPSIA_PROJECT = (
    f'raw_data/cohorts/{COHORT_LAUREN_PREECLAMPSIA}'
)
DEFAULT_LAUREN_PREECLAMPSIA_ANNOTATIONS = (
    f'raw_data/cohorts/{COHORT_LAUREN_PREECLAMPSIA}/scores/labelstudio_annotations.json'
)
DEFAULT_MASKED_CORE_PROJECT = DEFAULT_LAUREN_PREECLAMPSIA_PROJECT
DEFAULT_MASKED_CORE_ANNOTATIONS = DEFAULT_LAUREN_PREECLAMPSIA_ANNOTATIONS


class CohortManifestError(RuntimeError):
    """Raised when a unified cohort manifest cannot be trusted."""


@dataclass(frozen=True)
class ManifestValidation:
    """Validation result for a unified manifest."""

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def raise_for_errors(self) -> None:
        if not self.passed:
            raise CohortManifestError('; '.join(self.errors))


@dataclass(frozen=True)
class CohortBuildResult:
    """Paths and row counts produced by a cohort build."""

    manifest_path: Path
    summary_path: Path
    rows: int
    status_counts: dict[str, int]
    lane_counts: dict[str, int]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _clean_str(value: Any) -> str:
    if _is_missing(value):
        return ''
    return str(value).strip()


def _slug(value: Any) -> str:
    text = _clean_str(value).lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return text or 'missing'


def _coerce_score(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_source_sample_id(image_name: str) -> str:
    stem = Path(_clean_str(image_name)).stem
    return stem.split('_', 1)[0] if '_' in stem else stem


def _sample_alpha_prefix(sample_id: str) -> str:
    match = re.match(r'([A-Za-z]+)', _clean_str(sample_id))
    return match.group(1).upper() if match else ''


def _clean_excel_label(value: Any) -> str:
    if _is_missing(value):
        return ''
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _clean_str(value)


def normalize_lane_assignment(value: Any) -> str:
    """Return the canonical lane name while accepting legacy manifests."""
    lane = _clean_str(value)
    return LEGACY_LANE_ALIASES.get(lane, lane)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest for a readable file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_relative(path: Path | str, runtime_root: Path | None = None) -> str:
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    path = Path(path).expanduser()
    try:
        return str(path.resolve().relative_to(runtime_root.resolve()))
    except ValueError:
        return str(path)


def _runtime_relative_or_empty(path: Path | str, runtime_root: Path | None = None) -> str:
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    path = Path(path).expanduser()
    try:
        return str(path.resolve().relative_to(runtime_root.resolve()))
    except ValueError:
        return ''


def resolve_runtime_asset_path(
    value: str | Path, runtime_root: Path | None = None
) -> Path:
    """Resolve a manifest asset path against the active runtime root."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return runtime_root / path


def canonical_manifest_columns() -> dict[str, tuple[str, ...]]:
    """Return the explicit unified-manifest schema groups."""
    return {
        'human_required': HUMAN_REQUIRED_COLUMNS,
        'score_locator_required_one_of': SCORE_LOCATOR_COLUMNS,
        'human_optional': HUMAN_OPTIONAL_COLUMNS,
        'pipeline_generated': GENERATED_COLUMNS,
        'enriched': ENRICHED_MANIFEST_COLUMNS,
    }


def validate_unified_manifest(
    manifest: pd.DataFrame, *, enriched: bool = False
) -> ManifestValidation:
    """Validate the unified manifest contract without guessing repairs."""
    errors: list[str] = []
    warnings: list[str] = []

    for column in NO_SOURCE_PATH_COLUMNS:
        if column in manifest.columns:
            errors.append(f'forbidden_source_path_column:{column}')

    required = set(HUMAN_REQUIRED_COLUMNS)
    if not enriched:
        required = {'cohort_id', 'score'}
    missing_required = sorted(required.difference(manifest.columns))
    if missing_required:
        errors.append(f'missing_required_columns:{",".join(missing_required)}')

    if manifest.empty:
        warnings.append('manifest_empty')
        return ManifestValidation(passed=not errors, errors=tuple(errors), warnings=tuple(warnings))

    if not any(column in manifest.columns for column in SCORE_LOCATOR_COLUMNS):
        errors.append('missing_score_locator:source_score_row_or_source_sample_id')
    elif all(
        manifest[column].map(_is_missing).all()
        for column in SCORE_LOCATOR_COLUMNS
        if column in manifest.columns
    ):
        errors.append('empty_score_locator:source_score_row_or_source_sample_id')

    if enriched:
        generated_missing = sorted(set(GENERATED_COLUMNS).difference(manifest.columns))
        if generated_missing:
            errors.append(f'missing_generated_columns:{",".join(generated_missing)}')
        if 'manifest_row_id' in manifest.columns and manifest['manifest_row_id'].duplicated().any():
            errors.append('duplicate_manifest_row_id')
        if {'cohort_id', 'harmonized_id'}.issubset(manifest.columns):
            duplicate_harmonized = manifest.duplicated(
                subset=['cohort_id', 'harmonized_id'], keep=False
            )
            if duplicate_harmonized.any():
                errors.append('duplicate_harmonized_id_within_cohort')

        if 'admission_status' in manifest.columns:
            admitted = manifest['admission_status'].astype(str).eq('admitted')
            if admitted.any():
                admitted_rows = manifest.loc[admitted]
                for column in ('score', 'image_path'):
                    if column not in admitted_rows.columns or admitted_rows[column].map(_is_missing).any():
                        errors.append(f'admitted_missing_{column}')
                if 'verification_status' not in admitted_rows.columns or not admitted_rows[
                    'verification_status'
                ].astype(str).eq('passed').all():
                    errors.append('admitted_without_passed_verification')
                if 'exclusion_reason' in admitted_rows.columns and not admitted_rows[
                    'exclusion_reason'
                ].map(_is_missing).all():
                    errors.append('admitted_with_exclusion_reason')

    return ManifestValidation(passed=not errors, errors=tuple(errors), warnings=tuple(warnings))


def _uniqueness_columns_for_cohort(
    cohort_rows: pd.DataFrame, candidate_columns: Sequence[str] | None = None
) -> tuple[str, ...]:
    candidates = tuple(
        column
        for column in (
            candidate_columns
            or (
                'source_image_name',
                'source_sample_id',
                'source_score_row',
                'image_path',
                'source_batch',
                'source_date',
            )
        )
        if column in cohort_rows.columns
    )
    if not candidates:
        raise CohortManifestError('no_candidate_columns_for_harmonized_id')

    for width in range(1, len(candidates) + 1):
        for start in range(0, len(candidates) - width + 1):
            selected = candidates[start : start + width]
            values = cohort_rows.loc[:, selected].astype(str).replace({'': np.nan})
            if values.isna().any(axis=None):
                continue
            if not values.duplicated().any():
                return selected

    values = cohort_rows.loc[:, candidates].astype(str).replace({'': np.nan})
    if values.isna().any(axis=None) or values.duplicated().any():
        raise CohortManifestError('cannot_derive_unique_harmonized_id')
    return candidates


def add_harmonized_ids(
    manifest: pd.DataFrame, candidate_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Append cohort-scoped harmonized IDs using the smallest unique discriminator set."""
    result = manifest.copy()
    harmonized_ids = pd.Series(index=result.index, dtype='object')
    discriminator_records: list[dict[str, str]] = []

    for cohort_id, cohort_rows in result.groupby('cohort_id', sort=True, dropna=False):
        selected = _uniqueness_columns_for_cohort(cohort_rows, candidate_columns)
        for index, row in cohort_rows.iterrows():
            parts = [_slug(cohort_id)]
            parts.extend(_slug(row.get(column, '')) for column in selected)
            harmonized_ids.at[index] = '__'.join(parts)
        discriminator_records.append(
            {
                'cohort_id': _clean_str(cohort_id),
                'harmonized_id_columns': ','.join(selected),
            }
        )

    result['harmonized_id'] = harmonized_ids
    result.attrs['harmonized_id_columns'] = discriminator_records
    return result


def _deduplicate_same_image_same_score(manifest: pd.DataFrame) -> pd.DataFrame:
    if not {'cohort_id', 'image_path', 'score'}.issubset(manifest.columns):
        return manifest
    duplicate_key = ['cohort_id', 'image_path', 'score']
    image_present = ~manifest['image_path'].map(_is_missing)
    duplicate_mask = manifest.duplicated(subset=duplicate_key, keep=False) & image_present
    if not duplicate_mask.any():
        return manifest
    work = manifest.copy()
    duplicate_counts = work.groupby(duplicate_key, dropna=False)['cohort_id'].transform('size')
    if 'duplicate_evidence_count' not in work.columns:
        work['duplicate_evidence_count'] = ''
    work.loc[duplicate_mask, 'duplicate_evidence_count'] = duplicate_counts[duplicate_mask].astype(int).astype(str)
    keep_mask = ~(
        work.duplicated(subset=duplicate_key, keep='first')
        & image_present.reindex(work.index, fill_value=False)
    )
    return work.loc[keep_mask].reset_index(drop=True)


def enrich_unified_manifest(
    manifest: pd.DataFrame,
    *,
    runtime_root: Path | None = None,
    require_readable_assets_for_admission: bool = True,
) -> pd.DataFrame:
    """Append generated manifest state, hashes, and fail-closed admission fields."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    work = _deduplicate_same_image_same_score(manifest.copy()).reset_index(drop=True)

    for column in ENRICHED_MANIFEST_COLUMNS:
        if column not in work.columns:
            work[column] = ''

    if 'lane_assignment' not in work.columns:
        work['lane_assignment'] = ''

    work = add_harmonized_ids(work)
    work['manifest_row_id'] = [
        f'{_slug(row.cohort_id)}__{index:06d}' for index, row in enumerate(work.itertuples(index=False), start=1)
    ]

    for index, row in work.iterrows():
        image_path_value = _clean_str(row.get('image_path'))
        mask_path_value = _clean_str(row.get('mask_path'))
        score = _coerce_score(row.get('score'))

        image_path = resolve_runtime_asset_path(image_path_value, runtime_root) if image_path_value else None
        mask_path = resolve_runtime_asset_path(mask_path_value, runtime_root) if mask_path_value else None
        image_exists = bool(image_path and image_path.exists() and image_path.is_file())
        mask_exists = bool(mask_path and mask_path.exists() and mask_path.is_file())

        if image_exists:
            work.at[index, 'image_sha256'] = sha256_file(image_path)
        if mask_exists:
            work.at[index, 'mask_sha256'] = sha256_file(mask_path)

        lane = normalize_lane_assignment(row.get('lane_assignment'))
        if not lane:
            if mask_path_value and mask_exists:
                lane = (
                    LANE_MANUAL_MASK_CORE
                    if _clean_str(row.get('cohort_id')) == COHORT_LAUREN_PREECLAMPSIA
                    else LANE_MANUAL_MASK_EXTERNAL
                )
            else:
                lane = LANE_SCORED_ONLY
        work.at[index, 'lane_assignment'] = lane

        preset_admission = _clean_str(row.get('admission_status'))
        if preset_admission in {'foreign', 'excluded'}:
            if not _clean_str(row.get('mapping_review_status')):
                work.at[index, 'mapping_review_status'] = 'not_sampled'
            if not _clean_str(row.get('discovery_surfaces')):
                work.at[index, 'discovery_surfaces'] = ';'.join(DEFAULT_DISCOVERY_SURFACES)
            continue

        locator_present = any(not _is_missing(row.get(column)) for column in SCORE_LOCATOR_COLUMNS)
        join_failures: list[str] = []
        if not image_path_value:
            join_failures.append('missing_image_path')
        elif require_readable_assets_for_admission and not image_exists:
            join_failures.append('image_unreadable')
        if score is None:
            join_failures.append('missing_score')
        if not locator_present:
            join_failures.append('missing_score_locator')
        if mask_path_value and require_readable_assets_for_admission and not mask_exists:
            join_failures.append('mask_unreadable')

        if join_failures:
            work.at[index, 'join_status'] = 'pending_discovery' if 'missing_score' in join_failures else 'failed'
            work.at[index, 'verification_status'] = 'pending_discovery'
            work.at[index, 'admission_status'] = 'unresolved'
            work.at[index, 'exclusion_reason'] = ';'.join(join_failures)
        else:
            work.at[index, 'join_status'] = 'joined'
            work.at[index, 'verification_status'] = 'passed'
            work.at[index, 'admission_status'] = 'admitted'
            work.at[index, 'exclusion_reason'] = ''

        if not _clean_str(row.get('mapping_review_status')):
            work.at[index, 'mapping_review_status'] = 'not_sampled'
        if not _clean_str(row.get('discovery_surfaces')):
            work.at[index, 'discovery_surfaces'] = ';'.join(DEFAULT_DISCOVERY_SURFACES)

    ordered = [column for column in ENRICHED_MANIFEST_COLUMNS if column in work.columns]
    extras = [column for column in work.columns if column not in ordered]
    return work.loc[:, ordered + extras]


def write_unified_manifest(
    manifest: pd.DataFrame,
    manifest_path: Path | None = None,
    *,
    runtime_root: Path | None = None,
) -> Path:
    """Write the canonical dataset-wide manifest."""
    manifest_path = (
        Path(manifest_path)
        if manifest_path
        else get_runtime_cohort_manifest_path(runtime_root)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return manifest_path


def load_runtime_cohort_manifest(
    manifest_path: Path | None = None, *, runtime_root: Path | None = None
) -> pd.DataFrame:
    """Load the runtime cohort manifest as the canonical downstream surface."""
    if manifest_path is None:
        manifest_path = get_runtime_cohort_manifest_path(runtime_root)
    return pd.read_csv(Path(manifest_path))


def summarize_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    """Return deterministic row counts by cohort, lane, and status."""
    if manifest.empty:
        return {'total_rows': 0, 'cohorts': {}, 'status_counts': {}, 'lane_counts': {}}

    cohorts: dict[str, Any] = {}
    for cohort_id, cohort_rows in manifest.groupby('cohort_id', sort=True):
        cohorts[str(cohort_id)] = {
            'rows': int(len(cohort_rows)),
            'lane_counts': cohort_rows['lane_assignment'].value_counts(dropna=False).to_dict()
            if 'lane_assignment' in cohort_rows
            else {},
            'admission_status_counts': cohort_rows['admission_status']
            .value_counts(dropna=False)
            .to_dict()
            if 'admission_status' in cohort_rows
            else {},
            'join_status_counts': cohort_rows['join_status'].value_counts(dropna=False).to_dict()
            if 'join_status' in cohort_rows
            else {},
        }

    return {
        'total_rows': int(len(manifest)),
        'cohorts': cohorts,
        'status_counts': manifest['admission_status'].value_counts(dropna=False).to_dict()
        if 'admission_status' in manifest
        else {},
        'lane_counts': manifest['lane_assignment'].value_counts(dropna=False).to_dict()
        if 'lane_assignment' in manifest
        else {},
    }


def write_manifest_summary(
    manifest: pd.DataFrame,
    output_path: Path | None = None,
    *,
    runtime_root: Path | None = None,
) -> Path:
    """Write manifest row-count summary as JSON."""
    output_path = (
        Path(output_path)
        if output_path
        else get_runtime_cohort_manifest_summary_path(runtime_root)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summarize_manifest(manifest), indent=2), encoding='utf-8')
    return output_path


def copy_asset_to_cohort(
    source_path: Path,
    cohort_id: str,
    subdir: str,
    *,
    runtime_root: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Copy one source asset into the localized runtime cohort tree."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    source_path = Path(source_path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = get_runtime_cohort_path(cohort_id, runtime_root) / subdir / (
        filename or source_path.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_file() and destination.stat().st_size == source_path.stat().st_size:
        return destination
    shutil.copy2(source_path, destination)
    return destination


def classify_foreign_rows(
    manifest: pd.DataFrame,
    allowed_sample_prefixes: Sequence[str],
    *,
    sample_column: str = 'source_sample_id',
) -> pd.DataFrame:
    """Classify rows outside the intended cohort sample prefixes as foreign."""
    result = manifest.copy()
    allowed = {prefix.upper() for prefix in allowed_sample_prefixes}
    for column in ('join_status', 'verification_status', 'admission_status', 'exclusion_reason'):
        if column not in result.columns:
            result[column] = ''
    for index, row in result.iterrows():
        sample = _clean_str(row.get(sample_column)).upper()
        prefix_match = re.match(r'([A-Z]+)', sample)
        prefix = prefix_match.group(1) if prefix_match else sample
        if prefix and prefix not in allowed:
            result.at[index, 'join_status'] = 'foreign_row'
            result.at[index, 'verification_status'] = 'excluded'
            result.at[index, 'admission_status'] = 'foreign'
            result.at[index, 'exclusion_reason'] = 'foreign_mixed_export_row'
    return result


def harmonize_localized_cohort(
    records: pd.DataFrame,
    cohort_id: str,
    *,
    runtime_root: Path | None = None,
    copy_assets: bool = True,
    source_audit_path: Path | None = None,
) -> pd.DataFrame:
    """Build runtime-local manifest rows from source asset records by copying assets."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    localized_rows: list[dict[str, Any]] = []
    source_audit_rows: list[dict[str, Any]] = []

    for row in records.itertuples(index=False):
        row_dict = row._asdict()
        source_image_value = _clean_str(row_dict.get('source_image_path') or row_dict.get('image_path'))
        source_image = Path(source_image_value) if source_image_value else None
        source_mask_value = _clean_str(row_dict.get('source_mask_path') or row_dict.get('mask_path'))
        source_mask = Path(source_mask_value) if source_mask_value else None
        source_image_name = _clean_str(
            row_dict.get('source_image_name') or (source_image.name if source_image else '')
        )
        harmonized_stem = _slug(
            row_dict.get('source_sample_id') or Path(source_image_name).stem or row_dict.get('source_score_row')
        )

        if source_image is None or not source_image.exists() or not source_image.is_file():
            localized_image = ''
            join_status = 'pending_discovery'
            exclusion_reason = 'source_image_missing'
        else:
            filename = f'{harmonized_stem}{source_image.suffix.lower()}'
            destination = (
                copy_asset_to_cohort(source_image, cohort_id, 'images', runtime_root=runtime_root, filename=filename)
                if copy_assets
                else source_image
            )
            localized_image = _runtime_relative(destination, runtime_root)
            join_status = 'localized'
            exclusion_reason = ''

        localized_mask = ''
        if source_mask and source_mask.exists():
            filename = f'{harmonized_stem}_mask{source_mask.suffix.lower()}'
            destination = (
                copy_asset_to_cohort(source_mask, cohort_id, 'masks', runtime_root=runtime_root, filename=filename)
                if copy_assets
                else source_mask
            )
            localized_mask = _runtime_relative(destination, runtime_root)

        localized = {
            key: value
            for key, value in row_dict.items()
            if key not in NO_SOURCE_PATH_COLUMNS
            and key not in {'source_image_path', 'source_mask_path'}
        }
        localized.update(
            {
                'cohort_id': cohort_id,
                'image_path': localized_image,
                'mask_path': localized_mask,
                'source_image_name': source_image_name,
                'source_sample_id': _clean_str(row_dict.get('source_sample_id') or harmonized_stem),
                'source_score_row': _clean_str(row_dict.get('source_score_row')),
                'join_status': join_status,
                'verification_status': 'pending_discovery' if exclusion_reason else 'pending',
                'admission_status': 'unresolved' if exclusion_reason else 'pending',
                'exclusion_reason': exclusion_reason,
            }
        )
        localized_rows.append(localized)
        source_audit_rows.append(
            {
                'cohort_id': cohort_id,
                'source_image_path': str(source_image) if source_image else '',
                'source_mask_path': str(source_mask) if source_mask else '',
                'runtime_image_path': localized_image,
                'runtime_mask_path': localized_mask,
                'copy_based': bool(copy_assets),
            }
        )

    if source_audit_path:
        source_audit_path = Path(source_audit_path)
        source_audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(source_audit_rows).to_csv(source_audit_path, index=False)
    return pd.DataFrame(localized_rows)


def inventory_decoded_dox_runtime(
    cohort_dir: Path | None = None, *, runtime_root: Path | None = None
) -> pd.DataFrame:
    """Inventory the existing decoded Dox brushlabel runtime surface."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    cohort_dir = Path(cohort_dir) if cohort_dir else runtime_root / 'raw_data' / 'cohorts' / 'vegfri_dox'
    ledger_path = cohort_dir / 'metadata' / 'decoded_brushlabel_masks.csv'
    if not ledger_path.exists():
        return pd.DataFrame(columns=ENRICHED_MANIFEST_COLUMNS)

    ledger = pd.read_csv(ledger_path)
    rows: list[dict[str, Any]] = []
    for row in ledger.itertuples(index=False):
        image_path = Path(getattr(row, 'image_path'))
        mask_path = Path(getattr(row, 'mask_path'))
        subject_prefix = _clean_str(getattr(row, 'subject_prefix', ''))
        rows.append(
            {
                'cohort_id': 'vegfri_dox',
                'image_path': _runtime_relative(image_path, runtime_root),
                'mask_path': _runtime_relative(mask_path, runtime_root),
                'score': '',
                'source_image_name': _clean_str(getattr(row, 'image_name', image_path.name)),
                'source_sample_id': subject_prefix,
                'source_score_row': _clean_str(getattr(row, 'task_id', '')),
                'source_score_sheet': 'latest_label_studio_brushlabel_export',
                'score_path': _runtime_relative(ledger_path, runtime_root),
                'treatment_group': treatment_group_from_dox_sample(subject_prefix),
                'lane_assignment': LANE_MANUAL_MASK_EXTERNAL,
                'join_status': 'pending_score_linkage',
                'verification_status': 'pending_discovery',
                'admission_status': 'unresolved',
                'exclusion_reason': 'missing_score_linkage',
                'discovery_surfaces': ';'.join(
                    (
                        'decoded_brushlabel_masks.csv',
                        'latest_dox_label_studio_export',
                        'older_dox_label_studio_exports',
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def treatment_group_from_dox_sample(sample_id: str) -> str:
    """Infer the Dox treatment group from the cohort-visible sample prefix."""
    text = _clean_str(sample_id).upper()
    match = re.match(r'([A-Z]+)', text)
    prefix = match.group(1) if match else text
    return {
        'M': 'vehicle',
        'S': 'sorafenib',
        'D': 'sor_plus_dox',
        'L': 'sor_plus_lis',
    }.get(prefix, '')


def _load_label_studio_choice_scores(annotation_source: Path) -> pd.DataFrame:
    """Load image-level choice scores from a Label Studio JSON export."""
    payload = json.loads(Path(annotation_source).read_text(encoding='utf-8'))
    rows: list[dict[str, Any]] = []
    for task in payload:
        file_upload = _clean_str(task.get('file_upload'))
        image_name = file_upload.split('-', 1)[-1] if '-' in file_upload else file_upload
        if not image_name:
            image_value = _clean_str(task.get('data', {}).get('image'))
            image_name = Path(image_value).name if image_value else ''
        source_sample_id = _extract_source_sample_id(image_name)

        annotations = task.get('annotations') or []
        if not annotations:
            rows.append(
                {
                    'task_id': task.get('id'),
                    'image_name': image_name,
                    'source_sample_id': source_sample_id,
                    'score': np.nan,
                    'score_status': 'missing_annotation',
                    'annotation_id': '',
                    'annotation_updated_at': '',
                }
            )
            continue

        for annotation in annotations:
            choices: list[Any] = []
            for result in annotation.get('result', []):
                if result.get('type') == 'choices':
                    choices.extend(result.get('value', {}).get('choices', []))
            score = _coerce_score(choices[-1]) if choices else None
            rows.append(
                {
                    'task_id': task.get('id'),
                    'image_name': image_name,
                    'source_sample_id': source_sample_id,
                    'score': score if score is not None else np.nan,
                    'score_status': 'ok' if score is not None else 'missing_score',
                    'annotation_id': annotation.get('id', ''),
                    'annotation_updated_at': annotation.get('updated_at', ''),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                'task_id',
                'image_name',
                'source_sample_id',
                'score',
                'score_status',
                'annotation_id',
                'annotation_updated_at',
            ]
        )
    scores = pd.DataFrame(rows)
    return scores.sort_values(
        ['image_name', 'annotation_updated_at'], na_position='first'
    ).drop_duplicates(subset=['task_id', 'image_name'], keep='last')


def build_lauren_preeclampsia_runtime_cohort(
    *,
    runtime_root: Path | None = None,
    project_dir: Path | None = None,
    annotation_source: Path | None = None,
) -> pd.DataFrame:
    """Build localized Lauren preeclampsia rows from the active runtime."""
    from eq.quantification.labelstudio_scores import recover_label_studio_score_table

    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    project_dir = (
        Path(project_dir)
        if project_dir
        else runtime_root / DEFAULT_LAUREN_PREECLAMPSIA_PROJECT
    )
    annotation_source = (
        Path(annotation_source)
        if annotation_source
        else runtime_root / DEFAULT_LAUREN_PREECLAMPSIA_ANNOTATIONS
    )
    if not project_dir.exists() or not annotation_source.exists():
        return pd.DataFrame(columns=ENRICHED_MANIFEST_COLUMNS)

    cohort_dir = runtime_root / 'raw_data' / 'cohorts' / COHORT_LAUREN_PREECLAMPSIA
    scores_dir = cohort_dir / 'scores'
    outputs = recover_label_studio_score_table(project_dir, annotation_source, scores_dir)
    score_table = pd.read_csv(outputs['scores'])
    score_table = score_table[
        (score_table['join_status'].astype(str) == 'ok')
        & (score_table['score_status'].astype(str) == 'ok')
    ].copy()

    records = pd.DataFrame(
        [
            {
                'source_image_path': row.raw_image_path,
                'source_mask_path': row.raw_mask_path,
                'source_image_name': row.image_name,
                'source_sample_id': row.image_stem,
                'source_score_row': row.source_task_id,
                'source_score_sheet': annotation_source.name,
                'score_path': _runtime_relative(outputs['scores'], runtime_root),
                'score': row.score,
                'lane_assignment': LANE_MANUAL_MASK_CORE,
                'discovery_surfaces': 'active_preeclampsia_runtime;labelstudio_annotations',
            }
            for row in score_table.itertuples(index=False)
        ]
    )
    return harmonize_localized_cohort(
        records,
        COHORT_LAUREN_PREECLAMPSIA,
        runtime_root=runtime_root,
        source_audit_path=cohort_dir / 'metadata' / 'source_audit.csv',
    )


def build_dox_runtime_cohort(
    *,
    runtime_root: Path | None = None,
    annotation_source: Path | None = None,
) -> pd.DataFrame:
    """Build Dox rows from decoded masks plus latest Label Studio choice scores."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    annotation_source = (
        Path(annotation_source) if annotation_source else get_dox_label_studio_export_path()
    )
    decoded = inventory_decoded_dox_runtime(runtime_root=runtime_root)
    if decoded.empty:
        return decoded
    if not annotation_source.exists():
        return decoded

    scores = _load_label_studio_choice_scores(annotation_source)
    scores_dir = runtime_root / 'raw_data' / 'cohorts' / 'vegfri_dox' / 'scores'
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_path = scores_dir / 'labelstudio_scores.csv'
    scores.to_csv(scores_path, index=False)

    score_lookup = scores.rename(columns={'task_id': 'source_score_row'}).copy()
    score_lookup['source_score_row'] = score_lookup['source_score_row'].astype(str)
    decoded['source_score_row'] = decoded['source_score_row'].astype(str)
    merged = decoded.drop(columns=['score'], errors='ignore').merge(
        score_lookup[
            [
                'source_score_row',
                'score',
                'score_status',
                'annotation_id',
                'annotation_updated_at',
            ]
        ],
        on='source_score_row',
        how='left',
    )
    merged['score_path'] = _runtime_relative(scores_path, runtime_root)
    merged['join_status'] = np.where(
        merged['score'].notna(), 'joined', 'pending_discovery'
    )
    merged['verification_status'] = np.where(
        merged['score'].notna(), 'pending', 'pending_discovery'
    )
    merged['admission_status'] = np.where(
        merged['score'].notna(), 'pending', 'unresolved'
    )
    merged['exclusion_reason'] = np.where(
        merged['score'].notna(), '', 'missing_score'
    )
    merged['lane_assignment'] = LANE_MANUAL_MASK_EXTERNAL

    decoded_task_ids = set(merged['source_score_row'].astype(str))
    extra_scores = scores[~scores['task_id'].astype(str).isin(decoded_task_ids)].copy()
    foreign_rows: list[dict[str, Any]] = []
    for row in extra_scores.itertuples(index=False):
        sample_id = _clean_str(row.source_sample_id)
        alpha_prefix = _sample_alpha_prefix(sample_id)
        if alpha_prefix == 'T':
            status = 'foreign'
            reason = 'foreign_mixed_export_row'
            join_status = 'foreign_row'
            verification_status = 'excluded'
        else:
            status = 'unresolved'
            reason = 'not_materialized_in_decoded_brushlabel_surface'
            join_status = 'pending_discovery'
            verification_status = 'pending_discovery'
        foreign_rows.append(
            {
                'cohort_id': 'vegfri_dox',
                'image_path': '',
                'mask_path': '',
                'score': row.score,
                'source_image_name': row.image_name,
                'source_sample_id': sample_id,
                'source_score_row': _clean_str(row.task_id),
                'source_score_sheet': annotation_source.name,
                'score_path': _runtime_relative(scores_path, runtime_root),
                'treatment_group': treatment_group_from_dox_sample(sample_id),
                'lane_assignment': LANE_SCORED_ONLY,
                'join_status': join_status,
                'verification_status': verification_status,
                'admission_status': status,
                'exclusion_reason': reason,
                'discovery_surfaces': ';'.join(
                    (
                        'latest_dox_label_studio_export',
                        'decoded_brushlabel_masks.csv',
                    )
                ),
            }
        )
    if foreign_rows:
        return pd.concat([merged, pd.DataFrame(foreign_rows)], ignore_index=True)
    return merged


def inventory_lauren_preeclampsia_from_score_table(
    score_table: pd.DataFrame, *, runtime_root: Path | None = None
) -> pd.DataFrame:
    """Translate the Lauren preeclampsia score table into unified-manifest rows."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    rows: list[dict[str, Any]] = []
    for row in score_table.itertuples(index=False):
        image_path = Path(_clean_str(getattr(row, 'raw_image_path', '') or getattr(row, 'image_path', '')))
        mask_path = Path(_clean_str(getattr(row, 'raw_mask_path', '') or getattr(row, 'mask_path', '')))
        if not image_path.exists():
            continue
        source_image_name = _clean_str(getattr(row, 'image_name', image_path.name))
        subject_image_id = _clean_str(
            getattr(row, 'subject_image_id', '') or getattr(row, 'image_stem', Path(source_image_name).stem)
        )
        rows.append(
            {
                'cohort_id': COHORT_LAUREN_PREECLAMPSIA,
                'image_path': _runtime_relative(image_path, runtime_root),
                'mask_path': _runtime_relative(mask_path, runtime_root) if mask_path.exists() else '',
                'score': getattr(row, 'score', ''),
                'source_image_name': source_image_name,
                'source_sample_id': subject_image_id,
                'source_score_row': _clean_str(getattr(row, 'source_task_id', '')),
                'source_score_sheet': Path(
                    _clean_str(getattr(row, 'annotation_source', ''))
                ).name,
                'score_path': _runtime_relative_or_empty(
                    _clean_str(getattr(row, 'annotation_source', '')), runtime_root
                ),
                'lane_assignment': LANE_MANUAL_MASK_CORE,
                'discovery_surfaces': 'active_preeclampsia_runtime;labelstudio_score_table',
            }
        )
    return pd.DataFrame(rows)


def inventory_image_files_as_unresolved(
    cohort_id: str,
    image_root: Path,
    *,
    runtime_root: Path | None = None,
    treatment_group: str = '',
    source_score_sheet: str = '',
    lane_assignment: str = LANE_SCORED_ONLY,
    discovery_surfaces: Iterable[str] = DEFAULT_DISCOVERY_SURFACES,
) -> pd.DataFrame:
    """Inventory image files when score linkage is not yet deterministically available."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    image_root = Path(image_root)
    if not image_root.exists():
        return pd.DataFrame(columns=ENRICHED_MANIFEST_COLUMNS)

    rows: list[dict[str, Any]] = []
    for image_path in sorted(path for path in image_root.rglob('*') if path.suffix.lower() in IMAGE_SUFFIXES):
        rows.append(
            {
                'cohort_id': cohort_id,
                'image_path': _runtime_relative(image_path, runtime_root),
                'score': '',
                'source_image_name': image_path.name,
                'source_sample_id': image_path.stem,
                'source_score_sheet': source_score_sheet,
                'treatment_group': treatment_group,
                'lane_assignment': lane_assignment,
                'join_status': 'pending_discovery',
                'verification_status': 'pending_discovery',
                'admission_status': 'unresolved',
                'exclusion_reason': 'missing_score_linkage',
                'discovery_surfaces': ';'.join(discovery_surfaces),
            }
        )
    return pd.DataFrame(rows)


def reduce_mr_replicates(
    workbook_path: Path, *, output_sidecar: Path | None = None
) -> pd.DataFrame:
    """Reduce MR workbook replicate grades to image-level medians.

    The parser is intentionally conservative: every numeric column is treated as
    a candidate image column and non-null numeric cells in that column are the
    raw within-image replicate vector.
    """
    workbook_path = Path(workbook_path)
    sheets = pd.read_excel(workbook_path, sheet_name=None, header=None)
    rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []

    for sheet_name, sheet in sheets.items():
        for column_index in sheet.columns:
            values = pd.to_numeric(sheet.iloc[1:, column_index], errors='coerce').dropna()
            if values.empty:
                continue
            image_label = _clean_excel_label(sheet.iloc[0, column_index])
            if not image_label:
                image_label = f'{sheet_name}_column_{column_index}'
            if 'replicate' in image_label.lower() and 'sample' in image_label.lower():
                continue
            replicate_values = [float(value) for value in values.tolist()]
            source_score_row = f'{sheet_name}:column:{column_index}'
            rows.append(
                {
                    'cohort_id': 'vegfri_mr',
                    'source_image_name': image_label,
                    'source_sample_id': image_label,
                    'source_score_row': source_score_row,
                    'source_score_sheet': sheet_name,
                    'score_path': '',
                    'score': float(np.median(replicate_values)),
                    'score_reduction_method': 'median_within_image_replicates',
                    'replicate_count': int(len(replicate_values)),
                    'lane_assignment': LANE_MR_CONCORDANCE_ONLY,
                }
            )
            for replicate_index, value in enumerate(replicate_values, start=1):
                replicate_rows.append(
                    {
                        'source_score_row': source_score_row,
                        'replicate_index': replicate_index,
                        'raw_score': value,
                    }
                )

    reduced = pd.DataFrame(rows)
    if output_sidecar:
        output_sidecar = Path(output_sidecar)
        output_sidecar.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(replicate_rows).to_csv(output_sidecar, index=False)
    return reduced


def _mr_image_lookup(image_root: Path) -> dict[str, Path]:
    image_root = Path(image_root)
    if not image_root.exists():
        return {}
    lookup: dict[str, Path] = {}
    for path in sorted(
        candidate
        for candidate in image_root.rglob('*')
        if candidate.is_file()
        and candidate.suffix.lower() in IMAGE_SUFFIXES
        and not candidate.name.startswith('._')
    ):
        lookup.setdefault(path.stem, path)
    return lookup


def build_mr_runtime_cohort(
    *,
    runtime_root: Path | None = None,
    workbook_path: Path | None = None,
    image_root: Path | None = None,
    copy_assets: bool = True,
) -> pd.DataFrame:
    """Build VEGFRi MR image-level rows from workbook medians and whole-field TIFFs."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    workbook_path = Path(workbook_path) if workbook_path else get_mr_score_workbook_path()
    image_root = Path(image_root) if image_root else get_mr_image_root_path()
    if not workbook_path.exists():
        return pd.DataFrame(columns=ENRICHED_MANIFEST_COLUMNS)

    cohort_dir = runtime_root / 'raw_data' / 'cohorts' / 'vegfri_mr'
    scores_dir = cohort_dir / 'scores'
    metadata_dir = cohort_dir / 'metadata'
    scores_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    localized_workbook = scores_dir / workbook_path.name
    if copy_assets:
        shutil.copy2(workbook_path, localized_workbook)
    else:
        localized_workbook = workbook_path

    replicate_sidecar = metadata_dir / 'mr_replicates.csv'
    reduced = reduce_mr_replicates(localized_workbook, output_sidecar=replicate_sidecar)
    if reduced.empty:
        return reduced

    lookup = _mr_image_lookup(image_root)
    records: list[dict[str, Any]] = []
    for row in reduced.itertuples(index=False):
        sample_id = _clean_str(row.source_sample_id)
        source_image = lookup.get(sample_id)
        records.append(
            {
                'source_image_path': str(source_image) if source_image else '',
                'source_image_name': source_image.name if source_image else f'{sample_id}.tif',
                'source_sample_id': sample_id,
                'source_score_row': _clean_str(row.source_score_row),
                'source_score_sheet': _clean_str(row.source_score_sheet),
                'source_batch': source_image.parent.name if source_image else _clean_str(row.source_score_sheet).lower(),
                'score_path': _runtime_relative(localized_workbook, runtime_root)
                if localized_workbook.exists()
                else '',
                'score': row.score,
                'score_reduction_method': row.score_reduction_method,
                'replicate_count': row.replicate_count,
                'lane_assignment': LANE_MR_CONCORDANCE_ONLY,
                'discovery_surfaces': 'mr_workbook;external_drive_whole_field_tiffs',
            }
        )

    source_audit = metadata_dir / 'source_audit.csv'
    localized = harmonize_localized_cohort(
        pd.DataFrame(records),
        'vegfri_mr',
        runtime_root=runtime_root,
        copy_assets=copy_assets,
        source_audit_path=source_audit,
    )
    acquisition_metadata = {
        'source_workbook': str(workbook_path),
        'source_image_root': str(image_root),
        'localized_workbook': _runtime_relative(localized_workbook, runtime_root)
        if localized_workbook.exists()
        else '',
        'source_tiff_count': len(lookup),
        'manifest_rows': int(len(localized)),
        'acquisition_regime': 'external-drive high-resolution whole-field TIFF batches',
        'phase_1_use': 'concordance_only_not_training_admission',
    }
    (metadata_dir / 'mr_acquisition_metadata.json').write_text(
        json.dumps(acquisition_metadata, indent=2),
        encoding='utf-8',
    )
    return localized


def verify_mapping_bundle(manifest: pd.DataFrame, *, runtime_root: Path | None = None) -> pd.DataFrame:
    """Apply the full fail-closed mapping-verification bundle."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    result = manifest.copy()
    for column in ('mapping_failure_reasons', 'verification_status', 'admission_status', 'exclusion_reason'):
        if column not in result.columns:
            result[column] = ''

    conflict_source = result[~result['image_path'].map(_is_missing)].copy()
    conflict_keys = conflict_source.groupby(['cohort_id', 'image_path'], dropna=False)['score'].nunique(dropna=True)
    conflicting_images = set(conflict_keys[conflict_keys > 1].index)

    for index, row in result.iterrows():
        preset_admission = _clean_str(row.get('admission_status'))
        if preset_admission in {'foreign', 'excluded'}:
            result.at[index, 'mapping_failure_reasons'] = _clean_str(row.get('exclusion_reason'))
            if not _clean_str(row.get('verification_status')):
                result.at[index, 'verification_status'] = 'excluded'
            continue

        failures: list[str] = []
        if _is_missing(row.get('harmonized_id')):
            failures.append('missing_harmonized_id')
        if _is_missing(row.get('image_path')):
            failures.append('missing_image_path')
        else:
            image_path = resolve_runtime_asset_path(row['image_path'], runtime_root)
            if not image_path.exists() or not image_path.is_file():
                failures.append('image_unreadable')
        if _coerce_score(row.get('score')) is None:
            failures.append('missing_score')
        if (row.get('cohort_id'), row.get('image_path')) in conflicting_images:
            failures.append('conflicting_duplicate_scores')
        if all(_is_missing(row.get(column)) for column in SCORE_LOCATOR_COLUMNS):
            failures.append('missing_locator_fields')
        if _is_missing(row.get('image_sha256')):
            failures.append('missing_image_hash')

        if failures:
            result.at[index, 'mapping_failure_reasons'] = ';'.join(failures)
            result.at[index, 'verification_status'] = 'failed' if 'conflicting_duplicate_scores' in failures else 'pending_discovery'
            result.at[index, 'admission_status'] = 'excluded' if 'conflicting_duplicate_scores' in failures else 'unresolved'
            result.at[index, 'exclusion_reason'] = ';'.join(failures)
        else:
            result.at[index, 'mapping_failure_reasons'] = ''
            result.at[index, 'verification_status'] = 'passed'
            if preset_admission != 'excluded':
                result.at[index, 'admission_status'] = 'admitted'
                result.at[index, 'exclusion_reason'] = ''
    return result


def apply_cohort_admission_policy(manifest: pd.DataFrame) -> pd.DataFrame:
    """Apply cohort lane policy after mapping verification."""
    result = manifest.copy()
    for index, row in result.iterrows():
        cohort_id = _clean_str(row.get('cohort_id'))
        lane = normalize_lane_assignment(row.get('lane_assignment'))
        result.at[index, 'lane_assignment'] = lane
        if cohort_id == 'vegfri_mr':
            result.at[index, 'lane_assignment'] = LANE_MR_CONCORDANCE_ONLY
            if _clean_str(row.get('admission_status')) == 'admitted':
                result.at[index, 'admission_status'] = 'evaluation_only'
                result.at[index, 'exclusion_reason'] = 'mr_phase1_concordance_only_not_training_admitted'
        elif lane == LANE_SCORED_ONLY and _clean_str(row.get('admission_status')) == 'admitted':
            result.at[index, 'admission_status'] = 'pending_transport_audit'
            result.at[index, 'exclusion_reason'] = 'scored_only_requires_segmentation_transport_audit'
    return result


def _default_dox_mask_quality_audit_path(runtime_root: Path) -> Path:
    return runtime_root / 'raw_data' / 'cohorts' / 'vegfri_dox' / 'metadata' / 'mask_quality_audit.csv'


def _default_dox_mask_quality_panel_dir(runtime_root: Path) -> Path:
    return runtime_root / 'raw_data' / 'cohorts' / 'vegfri_dox' / 'metadata' / 'mask_quality_panels'


def _mask_connected_component_count(mask_array: np.ndarray) -> int:
    binary = (mask_array > 0).astype(np.uint8)
    if not binary.any():
        return 0
    try:
        import cv2

        return int(cv2.connectedComponents(binary, connectivity=8)[0] - 1)
    except Exception:
        return 1


def _thumbnail_with_mask_overlay(image_path: Path, mask_path: Path, label: str, size: int = 180):
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    image.thumbnail((size, size))
    mask = mask.resize(image.size)
    overlay = Image.new('RGBA', image.size, (255, 0, 0, 0))
    mask_alpha = mask.point(lambda pixel: 110 if pixel > 0 else 0)
    overlay.putalpha(mask_alpha)
    composed = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')

    tile = Image.new('RGB', (size, size + 34), 'white')
    tile.paste(composed, ((size - composed.width) // 2, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((4, size + 4), label[:42], fill='black')
    return tile


def _write_mask_quality_review_panels(
    audit: pd.DataFrame,
    *,
    runtime_root: Path,
    output_dir: Path,
    page_size: int = 30,
    columns: int = 5,
) -> list[str]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: list[str] = []
    rows = audit.sort_values(
        ['mask_quality_decision', 'foreground_fraction', 'manifest_row_id']
    ).reset_index(drop=True)
    tile_width = 180
    tile_height = 214
    for page_index, start in enumerate(range(0, len(rows), page_size), start=1):
        page = rows.iloc[start : start + page_size]
        page_columns = min(columns, len(page))
        page_rows = int(math.ceil(len(page) / page_columns))
        sheet = Image.new('RGB', (page_columns * tile_width, page_rows * tile_height), 'white')
        for item_index, row in enumerate(page.itertuples(index=False)):
            image_path = resolve_runtime_asset_path(row.image_path, runtime_root)
            mask_path = resolve_runtime_asset_path(row.mask_path, runtime_root)
            label = (
                f'{row.source_sample_id} {row.score} '
                f'fg={float(row.foreground_fraction):.3f} {row.mask_quality_decision}'
            )
            tile = _thumbnail_with_mask_overlay(image_path, mask_path, label, size=tile_width)
            x = (item_index % page_columns) * tile_width
            y = (item_index // page_columns) * tile_height
            sheet.paste(tile, (x, y))
        panel_path = output_dir / f'dox_mask_quality_panel_{page_index:03d}.png'
        sheet.save(panel_path)
        panel_paths.append(_runtime_relative(panel_path, runtime_root))
    return panel_paths


def build_dox_mask_quality_audit(
    manifest: pd.DataFrame | None = None,
    *,
    runtime_root: Path | None = None,
    manifest_path: Path | None = None,
    audit_path: Path | None = None,
    panel_dir: Path | None = None,
) -> dict[str, Path]:
    """Build optional Dox mask import-provenance audit panels."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    manifest = (
        manifest.copy()
        if manifest is not None
        else load_runtime_cohort_manifest(manifest_path, runtime_root=runtime_root)
    )
    audit_path = Path(audit_path) if audit_path else _default_dox_mask_quality_audit_path(runtime_root)
    panel_dir = (
        Path(panel_dir)
        if panel_dir
        else _default_dox_mask_quality_panel_dir(runtime_root)
    )

    lane_series = manifest['lane_assignment'].map(normalize_lane_assignment)
    candidates = manifest[
        (manifest['cohort_id'].astype(str) == 'vegfri_dox')
        & (lane_series == LANE_MANUAL_MASK_EXTERNAL)
        & (manifest['admission_status'].astype(str) == 'admitted')
    ].copy()
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        failures: list[str] = []
        image_path = resolve_runtime_asset_path(row.image_path, runtime_root)
        mask_path = resolve_runtime_asset_path(row.mask_path, runtime_root)
        if not image_path.exists():
            failures.append('image_missing')
        if not mask_path.exists():
            failures.append('mask_missing')

        image_size = ''
        mask_size = ''
        foreground_fraction = np.nan
        component_count = 0
        if not failures:
            from PIL import Image

            image = Image.open(image_path)
            mask = Image.open(mask_path).convert('L')
            image_size = f'{image.size[0]}x{image.size[1]}'
            mask_size = f'{mask.size[0]}x{mask.size[1]}'
            mask_array = np.array(mask)
            foreground_fraction = float((mask_array > 0).mean())
            component_count = _mask_connected_component_count(mask_array)
            if image.size != mask.size:
                failures.append('image_mask_size_mismatch')
            if foreground_fraction < DOX_MASK_QUALITY_MIN_FOREGROUND_FRACTION:
                failures.append('foreground_fraction_too_low')
            if foreground_fraction > DOX_MASK_QUALITY_MAX_FOREGROUND_FRACTION:
                failures.append('foreground_fraction_too_high')
            if component_count <= 0:
                failures.append('no_mask_components')

        decision = 'approved' if not failures else 'blocked'
        rows.append(
            {
                'manifest_row_id': row.manifest_row_id,
                'harmonized_id': row.harmonized_id,
                'cohort_id': row.cohort_id,
                'source_sample_id': row.source_sample_id,
                'source_image_name': row.source_image_name,
                'image_path': row.image_path,
                'mask_path': row.mask_path,
                'score': row.score,
                'image_size': image_size,
                'mask_size': mask_size,
                'foreground_fraction': foreground_fraction,
                'component_count': component_count,
                'mask_quality_decision': decision,
                'mask_quality_failure_reasons': ';'.join(failures),
                'review_method': 'automated_non_degenerate_mask_gate_with_visual_panel',
            }
        )

    audit = pd.DataFrame(rows)
    panel_paths = (
        _write_mask_quality_review_panels(audit, runtime_root=runtime_root, output_dir=panel_dir)
        if not audit.empty
        else []
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    summary = {
        'total_rows': int(len(audit)),
        'decision_counts': audit['mask_quality_decision'].value_counts().to_dict()
        if not audit.empty
        else {},
        'foreground_fraction_min': float(audit['foreground_fraction'].min()) if not audit.empty else None,
        'foreground_fraction_median': float(audit['foreground_fraction'].median()) if not audit.empty else None,
        'foreground_fraction_max': float(audit['foreground_fraction'].max()) if not audit.empty else None,
        'panel_pages': panel_paths,
        'training_admission_rule': 'Dox manual-mask rows are accepted as first-class glomeruli training labels; this audit is provenance only',
    }
    summary_path = audit_path.with_name('mask_quality_summary.json')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {'audit': audit_path, 'summary': summary_path, 'panel_dir': panel_dir}


def apply_dox_mask_quality_approval(
    manifest: pd.DataFrame,
    *,
    runtime_root: Path | None = None,
    audit_path: Path | None = None,
) -> pd.DataFrame:
    """Attach optional Dox mask-audit provenance without changing admission."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    audit_path = Path(audit_path) if audit_path else _default_dox_mask_quality_audit_path(runtime_root)
    result = manifest.copy()
    for column in ('mask_quality_review_status', 'mask_quality_audit_path'):
        if column not in result.columns:
            result[column] = ''
    if not audit_path.exists():
        return result

    audit = pd.read_csv(audit_path).fillna('')
    approved = set(
        audit.loc[
            audit['mask_quality_decision'].astype(str).eq('approved'),
            'manifest_row_id',
        ].astype(str)
    )
    blocked = set(
        audit.loc[
            audit['mask_quality_decision'].astype(str).eq('blocked'),
            'manifest_row_id',
        ].astype(str)
    )
    audit_rel = _runtime_relative(audit_path, runtime_root)
    for index, row in result.iterrows():
        row_id = _clean_str(row.get('manifest_row_id'))
        if (
            _clean_str(row.get('cohort_id')) == 'vegfri_dox'
            and normalize_lane_assignment(row.get('lane_assignment')) == LANE_MANUAL_MASK_EXTERNAL
        ):
            if row_id in approved:
                result.at[index, 'mask_quality_review_status'] = 'approved'
                result.at[index, 'mask_quality_audit_path'] = audit_rel
            elif row_id in blocked:
                result.at[index, 'mask_quality_review_status'] = 'blocked'
                result.at[index, 'mask_quality_audit_path'] = audit_rel
    return result


def apply_discovery_reconciliation(
    manifest: pd.DataFrame,
    *,
    required_surfaces_by_cohort: dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Keep recoverable rows pending until declared discovery surfaces are exhausted."""
    requirements = required_surfaces_by_cohort or COHORT_DISCOVERY_REQUIREMENTS
    result = manifest.copy()
    for index, row in result.iterrows():
        admission = _clean_str(row.get('admission_status'))
        if admission in {'admitted', 'foreign', 'excluded', 'evaluation_only'}:
            continue
        cohort_id = _clean_str(row.get('cohort_id'))
        required = tuple(requirements.get(cohort_id, DEFAULT_DISCOVERY_SURFACES))
        discovered = {
            surface.strip()
            for surface in _clean_str(row.get('discovery_surfaces')).split(';')
            if surface.strip()
        }
        missing = [surface for surface in required if surface not in discovered]
        if missing:
            result.at[index, 'join_status'] = 'pending_discovery'
            result.at[index, 'verification_status'] = 'pending_discovery'
            result.at[index, 'admission_status'] = 'pending_discovery'
            result.at[index, 'exclusion_reason'] = 'pending_discovery:' + ','.join(missing)
        elif admission == 'pending_discovery':
            result.at[index, 'admission_status'] = 'unresolved'
            if not _clean_str(row.get('exclusion_reason')):
                result.at[index, 'exclusion_reason'] = 'discovery_exhausted'
    return result


def build_predicted_roi_grading_inputs(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    segmentation_artifact: str,
) -> Path:
    """Write predicted-ROI grading rows for admitted scored-only manifest rows."""
    lane_series = (
        manifest['lane_assignment'].map(normalize_lane_assignment)
        if 'lane_assignment' in manifest.columns
        else pd.Series([''] * len(manifest), index=manifest.index)
    )
    admission_series = (
        manifest['admission_status'].astype(str)
        if 'admission_status' in manifest.columns
        else pd.Series([''] * len(manifest), index=manifest.index)
    )
    eligible = manifest[
        (lane_series == LANE_SCORED_ONLY)
        & (admission_series.isin({'admitted', 'pending_transport_audit'}))
    ].copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if eligible.empty:
        eligible = pd.DataFrame(
            columns=[
                'manifest_row_id',
                'cohort_id',
                'harmonized_id',
                'image_path',
                'score',
                'predicted_roi_path',
                'segmentation_artifact',
                'artifact_family',
            ]
        )
    else:
        eligible['predicted_roi_path'] = ''
        eligible['segmentation_artifact'] = segmentation_artifact
        eligible['artifact_family'] = 'predicted_roi_grading'
    output_path = output_dir / 'predicted_roi_grading_inputs.csv'
    eligible.to_csv(output_path, index=False)
    return output_path


def build_predicted_roi_grading_inputs_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    segmentation_artifact: str,
) -> Path:
    """Build predicted-ROI inputs directly from the canonical runtime manifest."""
    manifest = load_runtime_cohort_manifest(manifest_path)
    return build_predicted_roi_grading_inputs(
        manifest,
        output_dir,
        segmentation_artifact=segmentation_artifact,
    )


def write_segmentation_transport_audit(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    segmentation_artifact: str,
    reviewed_rows: pd.DataFrame | None = None,
    transport_status: str = 'pending_review',
    failure_reason: str = '',
) -> Path:
    """Write a cohort-specific transport-audit artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        'segmentation_artifact': segmentation_artifact,
        'transport_status': transport_status,
        'failure_reason': failure_reason,
        'cohort_counts': summarize_manifest(manifest).get('cohorts', {}),
        'reviewed_rows': int(len(reviewed_rows)) if reviewed_rows is not None else 0,
    }
    output_path = output_dir / 'segmentation_transport_audit.json'
    output_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return output_path


def validate_segmentation_transport_inputs(
    manifest: pd.DataFrame,
    segmentation_outputs: pd.DataFrame,
    *,
    row_id_column: str = 'manifest_row_id',
) -> pd.DataFrame:
    """Block transport admission when segmentation or grading evidence is missing or invalid."""
    required = {row_id_column, 'segmentation_status', 'accepted_roi_count', 'grading_status'}
    missing_columns = sorted(required.difference(segmentation_outputs.columns))
    if missing_columns:
        raise CohortManifestError(f'missing_transport_columns:{",".join(missing_columns)}')

    output_lookup = segmentation_outputs.set_index(row_id_column)
    result = manifest.copy()
    for column in ('transport_status', 'transport_failure_reason'):
        if column not in result.columns:
            result[column] = ''

    for index, row in result.iterrows():
        row_id = _clean_str(row.get(row_id_column))
        if not row_id or row_id not in output_lookup.index:
            result.at[index, 'transport_status'] = 'blocked'
            result.at[index, 'transport_failure_reason'] = 'missing_segmentation_output'
            result.at[index, 'admission_status'] = 'excluded'
            result.at[index, 'exclusion_reason'] = 'missing_segmentation_output'
            continue
        output = output_lookup.loc[row_id]
        if isinstance(output, pd.DataFrame):
            output = output.iloc[0]
        failures: list[str] = []
        if _clean_str(output.get('segmentation_status')) != 'ok':
            failures.append('segmentation_not_ok')
        if int(output.get('accepted_roi_count', 0) or 0) <= 0:
            failures.append('degenerate_segmentation_output')
        if _clean_str(output.get('grading_status')) != 'ok':
            failures.append('grading_not_ok')
        if failures:
            result.at[index, 'transport_status'] = 'blocked'
            result.at[index, 'transport_failure_reason'] = ';'.join(failures)
            result.at[index, 'admission_status'] = 'excluded'
            result.at[index, 'exclusion_reason'] = ';'.join(failures)
        else:
            result.at[index, 'transport_status'] = 'passed'
            result.at[index, 'transport_failure_reason'] = ''
    return result


def write_mr_inference_contract(output_dir: Path) -> Path:
    """Write the explicit phase-1 MR TIFF inference and acceptance contract."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        'artifact_family': 'mr_phase1_inference_contract',
        'stages': [
            'whole_field_tiff_tiling',
            'glomerulus_segmentation',
            'component_area_filtering',
            'accepted_roi_extraction',
            'roi_grading',
            'image_level_median_aggregation',
            'human_vs_inferred_concordance',
        ],
        'acceptance_rules': {
            'minimum_component_area_pixels': DEFAULT_IMAGE_MIN_COMPONENT_AREA,
            'zero_accepted_rois': 'non_evaluable',
            'required_counts': ['accepted_roi_count', 'rejected_roi_count'],
        },
        'training_admission': 'not_allowed_in_phase_1',
    }
    output_path = output_dir / 'mr_inference_contract.json'
    output_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    return output_path


def build_mr_concordance_workflow(
    manifest: pd.DataFrame,
    inferred_roi_grades: pd.DataFrame,
    output_dir: Path,
    *,
    min_component_area: int = DEFAULT_IMAGE_MIN_COMPONENT_AREA,
) -> dict[str, Path]:
    """Aggregate accepted inferred ROI grades to image level and compare with MR human medians."""
    required = {'manifest_row_id', 'roi_grade', 'component_area'}
    missing = sorted(required.difference(inferred_roi_grades.columns))
    if missing:
        raise CohortManifestError(f'missing_mr_inference_columns:{",".join(missing)}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mr_manifest = manifest[
        (manifest['cohort_id'].astype(str) == 'vegfri_mr')
        & (manifest['admission_status'].astype(str).isin({'evaluation_only', 'admitted'}))
    ].copy()
    roi = inferred_roi_grades.copy()
    roi['component_area'] = pd.to_numeric(roi['component_area'], errors='coerce')
    roi['roi_grade'] = pd.to_numeric(roi['roi_grade'], errors='coerce')
    roi['accepted_roi'] = (roi['component_area'] >= min_component_area) & roi['roi_grade'].notna()

    accepted = roi[roi['accepted_roi']].copy()
    aggregate_rows: list[dict[str, Any]] = []
    for row in mr_manifest.itertuples(index=False):
        row_id = _clean_str(getattr(row, 'manifest_row_id'))
        row_rois = roi[roi['manifest_row_id'].astype(str) == row_id]
        accepted_rois = accepted[accepted['manifest_row_id'].astype(str) == row_id]
        if accepted_rois.empty:
            inferred_median = np.nan
            status = 'non_evaluable'
        else:
            inferred_median = float(accepted_rois['roi_grade'].median())
            status = 'ok'
        aggregate_rows.append(
            {
                'manifest_row_id': row_id,
                'harmonized_id': _clean_str(getattr(row, 'harmonized_id')),
                'source_sample_id': _clean_str(getattr(row, 'source_sample_id')),
                'human_image_median': float(getattr(row, 'score')),
                'inferred_image_median': inferred_median,
                'accepted_roi_count': int(len(accepted_rois)),
                'rejected_roi_count': int(len(row_rois) - len(accepted_rois)),
                'concordance_status': status,
            }
        )

    image_level = pd.DataFrame(aggregate_rows)
    metrics = mr_concordance_metrics(
        image_level['human_image_median'],
        image_level['inferred_image_median'],
    )
    image_level_path = output_dir / 'mr_image_level_concordance.csv'
    metrics_path = output_dir / 'mr_concordance_metrics.json'
    image_level.to_csv(image_level_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    contract_path = write_mr_inference_contract(output_dir)
    return {
        'image_level_concordance': image_level_path,
        'metrics': metrics_path,
        'inference_contract': contract_path,
    }


def mr_concordance_metrics(
    human_scores: Sequence[float], inferred_scores: Sequence[float]
) -> dict[str, float]:
    """Compute fixed MR phase-1 concordance metrics."""
    human = pd.Series(human_scores, dtype='float64')
    inferred = pd.Series(inferred_scores, dtype='float64')
    valid = human.notna() & inferred.notna()
    human = human[valid]
    inferred = inferred[valid]
    if human.empty:
        return {
            'n': 0.0,
            'mae': float('nan'),
            'spearman': float('nan'),
            'exact_agreement': float('nan'),
            'within_one_step_agreement': float('nan'),
        }
    diff = (human - inferred).abs()
    return {
        'n': float(len(human)),
        'mae': float(diff.mean()),
        'spearman': float(human.rank().corr(inferred.rank(), method='pearson')),
        'exact_agreement': float((diff == 0).mean()),
        'within_one_step_agreement': float((diff <= 1.0).mean()),
    }


def archive_retired_quantification_input_tree(path: Path) -> Path:
    """Mark an overlapping old quantification-input tree as retired."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    marker = path / 'RETIRED_ACTIVE_INPUT.md'
    marker.write_text(
        '# Retired Active Input\n\n'
        'This directory is a retired reference surface. Active scored cohort inputs live under '
        '`raw_data/cohorts/<cohort_id>/` with `raw_data/cohorts/manifest.csv` as the canonical table.\n',
        encoding='utf-8',
    )
    return marker


def build_current_accessible_cohorts(
    *,
    runtime_root: Path | None = None,
    manifest_path: Path | None = None,
    write_summary: bool = True,
) -> CohortBuildResult:
    """Build the unified manifest from currently accessible runtime cohort surfaces."""
    runtime_root = Path(runtime_root) if runtime_root else get_active_runtime_root()
    manifest_frames: list[pd.DataFrame] = []

    lauren_preeclampsia = build_lauren_preeclampsia_runtime_cohort(
        runtime_root=runtime_root
    )
    if not lauren_preeclampsia.empty:
        manifest_frames.append(lauren_preeclampsia)

    dox = build_dox_runtime_cohort(runtime_root=runtime_root)
    if not dox.empty:
        manifest_frames.append(dox)

    mr = build_mr_runtime_cohort(runtime_root=runtime_root)
    if not mr.empty:
        manifest_frames.append(mr)

    manifest = pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame(columns=ENRICHED_MANIFEST_COLUMNS)
    if manifest.empty:
        manifest_path = write_unified_manifest(
            manifest, manifest_path, runtime_root=runtime_root
        )
        summary_path = (
            write_manifest_summary(manifest, runtime_root=runtime_root)
            if write_summary
            else manifest_path.with_suffix('.summary.json')
        )
        return CohortBuildResult(manifest_path, summary_path, 0, {}, {})

    enriched = enrich_unified_manifest(manifest, runtime_root=runtime_root)
    verified = verify_mapping_bundle(enriched, runtime_root=runtime_root)
    policy_applied = apply_dox_mask_quality_approval(
        apply_discovery_reconciliation(apply_cohort_admission_policy(verified)),
        runtime_root=runtime_root,
    )
    manifest_path = write_unified_manifest(
        policy_applied, manifest_path, runtime_root=runtime_root
    )
    summary_path = (
        write_manifest_summary(policy_applied, runtime_root=runtime_root)
        if write_summary
        else manifest_path.with_suffix('.summary.json')
    )
    summary = summarize_manifest(policy_applied)
    return CohortBuildResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        rows=int(len(policy_applied)),
        status_counts={str(k): int(v) for k, v in summary.get('status_counts', {}).items()},
        lane_counts={str(k): int(v) for k, v in summary.get('lane_counts', {}).items()},
    )
