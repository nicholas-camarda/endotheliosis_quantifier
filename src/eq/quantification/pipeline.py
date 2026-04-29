"""Contract-first quantification pipeline for endotheliosis scoring."""

from __future__ import annotations

import json
import pickle
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from eq.core.constants import DEFAULT_IMAGE_SIZE
from eq.data_management.canonical_contract import (
    apply_migration_plan,
    build_migration_plan,
    canonical_image_filename,
    canonical_mask_filename,
    save_contract_report,
    subject_prefix_from_subject_image_id,
    validate_project_contract,
)
from eq.data_management.metadata_processor import MetadataProcessor
from eq.data_management.model_loading import load_model_safely
from eq.evaluation.quantification_metrics import calculate_quantification_metrics
from eq.inference.prediction_core import create_prediction_core
from eq.quantification.burden import (
    ALLOWED_SCORE_VALUES,
    BURDEN_COLUMN,
    derive_biological_grouping,
    evaluate_burden_index_table,
    validate_score_values,
)
from eq.quantification.endotheliosis_grade_model import (
    evaluate_endotheliosis_grade_model,
)
from eq.quantification.input_contract import (
    ResolvedQuantificationInputContract,
    label_contract_reference_for_scored_table,
    resolve_quantification_input_contract,
)
from eq.quantification.labelstudio_scores import (
    discover_label_studio_annotation_source,
    recover_label_studio_score_table,
)
from eq.quantification.learned_roi import evaluate_learned_roi_quantification
from eq.quantification.migration import generate_mapping_template, inventory_raw_project
from eq.quantification.ordinal import (
    NUMERICAL_INSTABILITY_PATTERNS,
    CanonicalOrdinalClassifier,
    build_grouped_ordinal_cohort_profile,
)
from eq.quantification.severe_aware_ordinal_estimator import (
    evaluate_severe_aware_ordinal_endotheliosis_estimator,
)
from eq.quantification.source_aware_estimator import (
    evaluate_source_aware_endotheliosis_estimator,
)
from eq.training.transfer_learning import _get_encoder_module
from eq.utils.logger import get_logger


class ContractPreparationError(RuntimeError):
    """Raised when the raw project contract is not ready for quantification."""


def _save_json(data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)
    return output_path


def _apply_score_label_overrides(
    scored_table: pd.DataFrame,
    label_overrides_path: Path | None,
    output_dir: Path,
    *,
    label_contract_reference: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Apply explicitly reviewed score overrides and write an audit surface."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'score_label_overrides_summary.json'
    if label_overrides_path is None:
        _save_json(
            {
                'label_overrides_path': 'none',
                'label_overrides': 'none',
                'score_column': '',
                'scored_rows': int(len(scored_table)),
                'override_rows': 0,
                'applied_rows': 0,
                'changed_rows': 0,
                'severe_boundary_changed_rows': 0,
                'allowed_score_values': ALLOWED_SCORE_VALUES.tolist(),
                'claim_boundary': (
                    'no reviewed label overrides supplied; base scored labels define target'
                ),
                'label_contract_reference': label_contract_reference or {},
            },
            summary_path,
        )
        return scored_table, {'score_label_overrides_summary': summary_path}
    label_overrides_path = Path(label_overrides_path)
    if not label_overrides_path.exists():
        raise FileNotFoundError(
            f'Label overrides file does not exist: {label_overrides_path}'
        )
    overrides = pd.read_csv(label_overrides_path)
    if 'subject_image_id' not in overrides.columns:
        raise ValueError('Label overrides must contain subject_image_id')
    score_column = next(
        (
            column
            for column in ['rubric_score', 'reviewer_score', 'reviewer_grade', 'score']
            if column in overrides.columns
        ),
        None,
    )
    if score_column is None:
        raise ValueError(
            'Label overrides must contain one score column: '
            'rubric_score, reviewer_score, reviewer_grade, or score'
        )
    work = scored_table.copy()
    overrides = overrides.copy()
    overrides['subject_image_id'] = overrides['subject_image_id'].astype(str)
    overrides['override_score'] = pd.to_numeric(
        overrides[score_column], errors='coerce'
    )
    invalid_score = overrides['override_score'].isna()
    if invalid_score.any():
        bad_ids = overrides.loc[invalid_score, 'subject_image_id'].head(10).tolist()
        raise ValueError(f'Label overrides contain nonnumeric scores for: {bad_ids}')
    validate_score_values(overrides['override_score'])
    duplicate_ids = overrides.loc[
        overrides['subject_image_id'].duplicated(keep=False), 'subject_image_id'
    ].unique()
    if len(duplicate_ids):
        raise ValueError(
            'Label overrides contain duplicate subject_image_id values: '
            f'{list(duplicate_ids[:10])}'
        )
    missing_ids = sorted(
        set(overrides['subject_image_id']) - set(work['subject_image_id'].astype(str))
    )
    if missing_ids:
        raise ValueError(
            'Label overrides reference rows absent from scored examples: '
            f'{missing_ids[:10]}'
        )

    lookup = overrides.set_index('subject_image_id')
    mask = work['subject_image_id'].astype(str).isin(lookup.index)
    original_scores = work.loc[mask, ['subject_image_id', 'score']].copy()
    original_scores['score'] = pd.to_numeric(original_scores['score'], errors='coerce')
    replacement = (
        work.loc[mask, 'subject_image_id'].astype(str).map(lookup['override_score'])
    )
    work['original_score_before_label_override'] = work['score']
    work['label_override_source'] = ''
    work.loc[mask, 'score'] = replacement.astype(float).to_numpy()
    work.loc[mask, 'label_override_source'] = str(label_overrides_path)
    validate_score_values(work['score'])

    audit = original_scores.rename(columns={'score': 'original_score'}).copy()
    audit['override_score'] = (
        audit['subject_image_id'].astype(str).map(lookup['override_score'])
    )
    audit['score_delta'] = audit['override_score'] - audit['original_score']
    for column in [
        'reviewer_confidence_1_5',
        'accepted_teaching',
        'review_flags',
        'review_notes',
        'review_source',
    ]:
        audit[column] = (
            audit['subject_image_id'].astype(str).map(lookup[column])
            if column in lookup.columns
            else ''
        )
    audit_path = output_dir / 'score_label_overrides_audit.csv'
    audit.to_csv(audit_path, index=False)
    contract_payload = dict(label_contract_reference or {})
    contract_payload['score_override_audit_path'] = str(audit_path)
    _save_json(
        {
            'label_overrides_path': str(label_overrides_path),
            'score_column': score_column,
            'scored_rows': int(len(work)),
            'override_rows': int(len(overrides)),
            'applied_rows': int(mask.sum()),
            'changed_rows': int((audit['score_delta'].abs() > 0).sum()),
            'severe_boundary_changed_rows': int(
                (
                    (audit['original_score'].astype(float) >= 2.0)
                    != (audit['override_score'].astype(float) >= 2.0)
                ).sum()
            ),
            'allowed_score_values': ALLOWED_SCORE_VALUES.tolist(),
            'claim_boundary': (
                'explicit reviewer label overrides only; no inferred labels applied'
            ),
            'label_contract_reference': contract_payload,
            'override_content_hash': contract_payload.get('label_overrides_hash'),
            'base_scored_input_hash': contract_payload.get('base_scored_input_hash'),
            'grouping_identity': contract_payload.get('grouping_identity', {}),
            'effective_target_definition_version': contract_payload.get(
                'target_definition_version'
            ),
        },
        summary_path,
    )
    return work, {
        'score_label_overrides_audit': audit_path,
        'score_label_overrides_summary': summary_path,
    }


def _score_to_class_index(score: float) -> int:
    matches = np.where(np.isclose(ALLOWED_SCORE_VALUES, float(score)))[0]
    if len(matches) != 1:
        raise ValueError(f'Unsupported endotheliosis score: {score}')
    return int(matches[0])


def _class_index_to_score(index: np.ndarray | int) -> np.ndarray | float:
    if isinstance(index, np.ndarray):
        return ALLOWED_SCORE_VALUES[index]
    return float(ALLOWED_SCORE_VALUES[int(index)])


def _score_probability_column_name(score: float) -> str:
    score_str = str(score).replace('.', '_')
    return f'prob_score_{score_str}'


def _score_label(score: float) -> str:
    return f'{score:g}'


def _burden_first_artifacts(model_artifacts: Dict[str, Path]) -> Dict[str, Path]:
    """Return explicit burden-first artifact keys without legacy generic model aliases."""
    return {
        key: value
        for key, value in model_artifacts.items()
        if key
        not in {
            'predictions',
            'metrics',
            'confusion_matrix',
            'model',
            'review_html',
            'review_examples',
            'review_assets_dir',
        }
    }


def _finite_matrix_status(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    return {
        'finite': bool(np.isfinite(values).all()),
        'nan_count': int(np.isnan(values).sum()),
        'posinf_count': int(np.isposinf(values).sum()),
        'neginf_count': int(np.isneginf(values).sum()),
    }


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def _find_canonical_path(
    root: Path, subject_image_id: str, is_mask: bool
) -> Optional[Path]:
    subject_dir = root / subject_prefix_from_subject_image_id(subject_image_id)
    if not subject_dir.exists():
        return None
    for suffix in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        candidate = subject_dir / (
            canonical_mask_filename(subject_image_id, suffix)
            if is_mask
            else canonical_image_filename(subject_image_id, suffix)
        )
        if candidate.exists():
            return candidate
    return None


def _threshold_mask(mask_array: np.ndarray) -> np.ndarray:
    return (mask_array > 127).astype(np.uint8)


def _component_angle(
    center_x: float, center_y: float, centroid_x: float, centroid_y: float
) -> float:
    dx = centroid_x - center_x
    dy = centroid_y - center_y
    return float((np.arctan2(-dx, -dy) + (2.0 * np.pi)) % (2.0 * np.pi))


def _extract_components(
    mask_array: np.ndarray, min_area: int = 64
) -> list[dict[str, Any]]:
    binary = _threshold_mask(mask_array)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = binary.shape
    center_x = width / 2.0
    center_y = height / 2.0

    components: list[dict[str, Any]] = []
    for label_index in range(1, num_labels):
        x, y, w, h, area = stats[label_index]
        if int(area) < min_area:
            continue
        centroid_x, centroid_y = centroids[label_index]
        component_mask = (labels == label_index).astype(np.uint8)
        components.append(
            {
                'component_index': label_index,
                'bbox_x': int(x),
                'bbox_y': int(y),
                'bbox_w': int(w),
                'bbox_h': int(h),
                'area': int(area),
                'centroid_x': float(centroid_x),
                'centroid_y': float(centroid_y),
                'distance_from_center': float(
                    np.hypot(centroid_x - center_x, centroid_y - center_y)
                ),
                'angle_from_top_ccw': _component_angle(
                    center_x, center_y, centroid_x, centroid_y
                ),
                'mask': component_mask,
            }
        )

    components.sort(
        key=lambda item: (item['angle_from_top_ccw'], item['distance_from_center'])
    )
    for rank, component in enumerate(components, start=1):
        component['glomerulus_id'] = rank
    return components


def _build_union_mask(
    mask_array: np.ndarray, min_component_area: int = 64
) -> dict[str, Any] | None:
    binary_mask = _threshold_mask(mask_array)
    if not binary_mask.any():
        return None

    components = _extract_components(mask_array, min_area=min_component_area)
    if components:
        union_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        for component in components:
            union_mask = np.maximum(union_mask, component['mask'].astype(np.uint8))
        component_count = len(components)
        largest_component_area = max(int(component['area']) for component in components)
        selection = 'union_mask'
    else:
        return None

    ys, xs = np.where(union_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    union_area = int(union_mask.sum())
    largest_fraction = float(largest_component_area / union_area) if union_area else 0.0

    return {
        'mask': union_mask,
        'bbox_x0': x0,
        'bbox_y0': y0,
        'bbox_x1': x1,
        'bbox_y1': y1,
        'component_count': int(component_count),
        'component_selection': selection,
        'union_area': union_area,
        'largest_component_area_fraction': largest_fraction,
        'bbox_width': int(x1 - x0),
        'bbox_height': int(y1 - y0),
    }


def build_scored_example_table(
    project_dir: Path, metadata_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Join standardized metadata to canonical raw images and masks."""
    rows: list[dict[str, Any]] = []
    images_root = Path(project_dir) / 'images'
    masks_root = Path(project_dir) / 'masks'

    metadata = metadata_df.rename(columns={'subject_id': 'subject_image_id'}).copy()
    metadata['subject_image_id'] = metadata['subject_image_id'].astype(str)

    for row in metadata.itertuples(index=False):
        subject_image_id = str(row.subject_image_id)
        image_path = _find_canonical_path(images_root, subject_image_id, is_mask=False)
        mask_path = _find_canonical_path(masks_root, subject_image_id, is_mask=True)
        if image_path and mask_path:
            join_status = 'ok'
        elif image_path:
            join_status = 'missing_mask'
        elif mask_path:
            join_status = 'missing_image'
        else:
            join_status = 'missing_image_and_mask'
        rows.append(
            {
                'subject_image_id': subject_image_id,
                'subject_prefix': subject_prefix_from_subject_image_id(
                    subject_image_id
                ),
                'glomerulus_id': int(row.glomerulus_id),
                'score': float(row.score),
                'raw_image_path': str(image_path) if image_path else '',
                'raw_mask_path': str(mask_path) if mask_path else '',
                'join_status': join_status,
                'roi_status': 'pending' if join_status == 'ok' else 'join_failed',
            }
        )

    scored_table = (
        pd.DataFrame(rows)
        .sort_values(['subject_image_id', 'glomerulus_id'])
        .reset_index(drop=True)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_table.to_csv(output_dir / 'scored_examples.csv', index=False)
    return scored_table


def build_image_level_scored_example_table(
    project_dir: Path, score_table: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Create one scored example per raw image/mask pair from Label Studio-derived scores."""
    rows: list[dict[str, Any]] = []

    for row in score_table.itertuples(index=False):
        join_status = str(row.join_status)
        score_status = str(row.score_status)
        roi_status = (
            'pending' if join_status == 'ok' and score_status == 'ok' else 'join_failed'
        )
        rows.append(
            {
                'subject_image_id': str(row.image_stem),
                'image_name': str(row.image_name),
                'subject_prefix': str(row.subject_prefix),
                'glomerulus_id': 1,
                'score': float(row.score) if pd.notna(row.score) else np.nan,
                'raw_image_path': str(row.raw_image_path or ''),
                'raw_mask_path': str(row.raw_mask_path or ''),
                'join_status': join_status,
                'score_status': score_status,
                'score_resolution': str(row.score_resolution),
                'roi_status': roi_status,
            }
        )

    scored_table = (
        pd.DataFrame(rows)
        .sort_values(['subject_prefix', 'image_name'])
        .reset_index(drop=True)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_table.to_csv(output_dir / 'scored_examples.csv', index=False)
    return scored_table


def _manifest_runtime_root(manifest_root: Path) -> Path:
    if manifest_root.name == 'cohorts' and manifest_root.parent.name == 'raw_data':
        return manifest_root.parent.parent
    return manifest_root


def _resolve_manifest_path(manifest_root: Path, raw_path: Any) -> str:
    text = str(raw_path or '').strip()
    if not text or text.lower() == 'nan':
        return ''
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == 'raw_data':
        return str(_manifest_runtime_root(manifest_root) / path)
    return str(manifest_root / path)


def _manifest_subject_group(row: pd.Series, image_path: Path) -> str:
    cohort_id = str(row.get('cohort_id') or 'unknown_cohort')
    source_sample_id = str(row.get('source_sample_id') or '').strip()
    if source_sample_id:
        if '_Image' in source_sample_id:
            subject = source_sample_id.split('_Image', 1)[0]
        else:
            subject = source_sample_id
    elif image_path.parent.name and image_path.parent.name != 'images':
        subject = image_path.parent.name
    else:
        subject = image_path.stem.split('_', 1)[0]
    return f'{cohort_id}:{subject}'


def _identity_token(value: Any, *, default: str = 'unknown') -> str:
    text = str(value or '').strip()
    if not text or text.lower() == 'nan':
        text = default
    token = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_').lower()
    return token or default


def _manifest_identity_fields(row: pd.Series, image_path: Path) -> dict[str, str]:
    manifest_subject_id = str(row.get('subject_id') or '').strip()
    manifest_sample_id = str(row.get('sample_id') or '').strip()
    manifest_image_id = str(row.get('image_id') or '').strip()
    if manifest_subject_id and manifest_sample_id and manifest_image_id:
        replicate_id = str(
            row.get('replicate_id') or Path(manifest_image_id).stem
        ).strip()
        return {
            'subject_id': manifest_subject_id,
            'sample_id': manifest_sample_id,
            'replicate_id': replicate_id,
            'image_id': manifest_image_id,
            'identity_resolution': str(
                row.get('identity_resolution') or 'manifest_owned_identity'
            ),
        }

    cohort_id = _identity_token(row.get('cohort_id'), default='unknown_cohort')
    source_sample_id = str(row.get('source_sample_id') or '').strip()
    source_image_name = str(row.get('source_image_name') or image_path.name).strip()
    subject_group = _manifest_subject_group(row, image_path)
    _, _, subject_text = subject_group.partition(':')

    dated_match = re.match(
        r'^(?P<subject>.+?)--(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})(?:_Image.*)?$',
        source_sample_id,
        flags=re.IGNORECASE,
    )
    if dated_match:
        subject_text = dated_match.group('subject')
        acquisition_text = dated_match.group('date')
    else:
        if '_Image' in source_sample_id:
            subject_text = source_sample_id.split('_Image', 1)[0]
        source_date = str(row.get('source_date') or '').strip()
        source_batch = str(row.get('source_batch') or '').strip()
        if source_date and source_date.lower() != 'nan':
            acquisition_text = source_date
        elif source_batch and source_batch.lower() != 'nan':
            acquisition_text = source_batch
        else:
            acquisition_text = 'undated'

    subject_id = f'{cohort_id}__{_identity_token(subject_text)}'
    sample_id = f'{subject_id}__{_identity_token(acquisition_text, default="undated")}'
    replicate_id = _identity_token(
        Path(source_image_name).stem, default=image_path.stem
    )
    image_id = f'{sample_id}__{replicate_id}'
    return {
        'subject_id': subject_id,
        'sample_id': sample_id,
        'replicate_id': replicate_id,
        'image_id': image_id,
        'identity_resolution': 'manifest_derived_acquisition_preserving',
    }


def build_manifest_scored_example_table(
    manifest_root: Path,
    output_dir: Path,
    *,
    cohort_ids: Optional[list[str]] = None,
    lane_assignments: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Create scored examples from the runtime cohort manifest."""
    manifest_root = Path(manifest_root)
    manifest_path = manifest_root / 'manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Cohort manifest not found: {manifest_path}')

    manifest = pd.read_csv(manifest_path)
    required_columns = {
        'cohort_id',
        'image_path',
        'mask_path',
        'score',
        'source_image_name',
        'source_sample_id',
        'manifest_row_id',
        'join_status',
        'admission_status',
        'lane_assignment',
    }
    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise ContractPreparationError(
            f'Cohort manifest is missing required quantification columns: {missing}'
        )

    selected = manifest[
        manifest['admission_status'].astype(str).eq('admitted')
        & manifest['image_path'].notna()
        & manifest['mask_path'].notna()
        & manifest['score'].notna()
    ].copy()
    selected = selected[selected['image_path'].astype(str).str.strip().ne('')]
    selected = selected[selected['mask_path'].astype(str).str.strip().ne('')]
    if cohort_ids:
        selected = selected[selected['cohort_id'].astype(str).isin(cohort_ids)]
    if lane_assignments:
        selected = selected[
            selected['lane_assignment'].astype(str).isin(lane_assignments)
        ]
    if selected.empty:
        raise ContractPreparationError(
            'Cohort manifest did not contain admitted scored image/mask rows for quantification'
        )

    rows: list[dict[str, Any]] = []
    for row in selected.sort_values(
        ['cohort_id', 'source_sample_id', 'source_image_name']
    ).itertuples(index=False):
        series = pd.Series(row._asdict())
        raw_image_path = Path(
            _resolve_manifest_path(manifest_root, series['image_path'])
        )
        raw_mask_path = Path(_resolve_manifest_path(manifest_root, series['mask_path']))
        score = float(series['score'])
        score_status_raw = str(series.get('score_status') or '').strip()
        score_status = (
            score_status_raw if score_status_raw and score_status_raw != 'nan' else 'ok'
        )
        join_status = (
            'ok'
            if raw_image_path.exists()
            and raw_mask_path.exists()
            and score_status == 'ok'
            else 'join_failed'
        )
        subject_image_id = str(series.get('manifest_row_id') or '').strip()
        if not subject_image_id or subject_image_id == 'nan':
            subject_image_id = f'{series["cohort_id"]}__{raw_image_path.stem}'
        identity_fields = _manifest_identity_fields(series, raw_image_path)
        rows.append(
            {
                'subject_image_id': subject_image_id,
                'image_name': str(
                    series.get('source_image_name') or raw_image_path.name
                ),
                'subject_prefix': _manifest_subject_group(series, raw_image_path),
                **identity_fields,
                'glomerulus_id': 1,
                'score': score,
                'raw_image_path': str(raw_image_path),
                'raw_mask_path': str(raw_mask_path),
                'join_status': join_status,
                'score_status': score_status,
                'score_resolution': 'manifest_admitted_score',
                'roi_status': 'pending'
                if join_status == 'ok' and score_status == 'ok'
                else 'join_failed',
                'cohort_id': str(series.get('cohort_id') or ''),
                'lane_assignment': str(series.get('lane_assignment') or ''),
                'manifest_row_id': subject_image_id,
                'harmonized_id': str(series.get('harmonized_id') or ''),
                'source_sample_id': str(series.get('source_sample_id') or ''),
                'source_image_name': str(series.get('source_image_name') or ''),
                'source_batch': str(series.get('source_batch') or ''),
                'source_date': str(series.get('source_date') or ''),
                'source_score_sheet': str(series.get('source_score_sheet') or ''),
                'score_path': str(series.get('score_path') or ''),
            }
        )

    scored_table = pd.DataFrame(rows).reset_index(drop=True)
    summary = {
        'manifest_path': str(manifest_path),
        'n_manifest_rows': int(len(manifest)),
        'n_scored_rows': int(len(scored_table)),
        'cohort_counts': scored_table['cohort_id'].value_counts(dropna=False).to_dict(),
        'lane_counts': scored_table['lane_assignment']
        .value_counts(dropna=False)
        .to_dict(),
        'join_status_counts': scored_table['join_status']
        .value_counts(dropna=False)
        .to_dict(),
        'score_status_counts': scored_table['score_status']
        .value_counts(dropna=False)
        .to_dict(),
        'score_value_counts': scored_table['score']
        .value_counts(dropna=False)
        .sort_index()
        .to_dict(),
        'identity_contract': {
            'row_key': 'manifest_row_id',
            'exact_image_key': 'image_id',
            'validation_group_key': 'subject_id',
            'subject_key': 'subject_id',
            'sample_key': 'sample_id',
            'identity_resolution': 'manifest_derived_acquisition_preserving',
            'n_subjects': int(scored_table['subject_id'].nunique()),
            'n_samples': int(scored_table['sample_id'].nunique()),
            'n_images': int(scored_table['image_id'].nunique()),
            'duplicate_image_ids': sorted(
                scored_table.loc[
                    scored_table['image_id'].duplicated(keep=False), 'image_id'
                ].unique()
            ),
            'subjects_with_multiple_samples': int(
                scored_table.groupby('subject_id')['sample_id'].nunique().gt(1).sum()
            ),
        },
    }
    duplicate_images = summary['identity_contract']['duplicate_image_ids']
    if duplicate_images:
        raise ContractPreparationError(
            f'Manifest identity contract produced duplicate image_id values: {duplicate_images[:10]}'
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_table.to_csv(output_dir / 'scored_examples.csv', index=False)
    _save_json(summary, output_dir / 'manifest_scored_examples_summary.json')
    return scored_table


def run_manifest_quantification(
    *,
    manifest_root: Path,
    segmentation_model_path: Path,
    output_dir: Path,
    stop_after: str = 'model',
    label_overrides_path: Path | None = None,
    input_contract: ResolvedQuantificationInputContract | None = None,
) -> Dict[str, Path]:
    """Run quantification from the runtime cohort manifest."""
    logger = get_logger('eq.quantification.pipeline')
    manifest_root = Path(manifest_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        'Starting manifest quantification: manifest_root=%s, output_dir=%s, stop_after=%s',
        manifest_root,
        output_dir,
        stop_after,
    )
    manifest_path = manifest_root / 'manifest.csv'
    mapping_template_path = output_dir / 'legacy_to_canonical_mapping_template.csv'
    pd.DataFrame(
        columns=[
            'subject_prefix',
            'legacy_image_stem',
            'image_name',
            'mask_name',
            'has_mask',
            'canonical_subject_image_id',
        ]
    ).to_csv(mapping_template_path, index=False)
    scored_table = build_manifest_scored_example_table(
        manifest_root, output_dir / 'scored_examples'
    )
    label_contract_reference = (
        label_contract_reference_for_scored_table(
            input_contract,
            scored_table,
            base_scored_input_path=manifest_path,
        )
        if input_contract is not None
        else None
    )
    scored_table, label_override_artifacts = _apply_score_label_overrides(
        scored_table,
        label_overrides_path,
        output_dir / 'scored_examples',
        label_contract_reference=label_contract_reference,
    )
    if label_override_artifacts:
        scored_table.to_csv(
            output_dir / 'scored_examples' / 'scored_examples.csv', index=False
        )
    logger.info(
        'Scored examples ready: rows=%d -> %s',
        len(scored_table),
        output_dir / 'scored_examples' / 'scored_examples.csv',
    )
    manifest_summary_path = (
        output_dir / 'scored_examples' / 'manifest_scored_examples_summary.json'
    )
    if stop_after == 'contract':
        return {
            'mapping_template': mapping_template_path,
            'manifest': manifest_path,
            'manifest_scored_summary': manifest_summary_path,
            'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
            **label_override_artifacts,
        }

    roi_table = extract_image_level_roi_crops(scored_table, output_dir / 'roi_crops')
    ok_rois = int(roi_table['roi_status'].astype(str).eq('ok').sum())
    logger.info(
        'ROI crops ready: rows=%d, ok=%d -> %s',
        len(roi_table),
        ok_rois,
        output_dir / 'roi_crops' / 'roi_scored_examples.csv',
    )
    if stop_after == 'roi':
        return {
            'mapping_template': mapping_template_path,
            'manifest': manifest_path,
            'manifest_scored_summary': manifest_summary_path,
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            **label_override_artifacts,
        }

    embedding_table = extract_embedding_table(
        roi_table=roi_table,
        segmentation_model_path=Path(segmentation_model_path),
        output_dir=output_dir / 'embeddings',
    )
    logger.info(
        'Embedding table ready: rows=%d, columns=%d -> %s',
        len(embedding_table),
        len(embedding_table.columns),
        output_dir / 'embeddings' / 'roi_embeddings.csv',
    )
    if stop_after == 'embeddings':
        return {
            'mapping_template': mapping_template_path,
            'manifest': manifest_path,
            'manifest_scored_summary': manifest_summary_path,
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
            **label_override_artifacts,
        }

    model_artifacts = evaluate_embedding_table(
        embedding_table,
        output_dir / 'ordinal_model',
        manifest_root=manifest_root,
        segmentation_model_path=segmentation_model_path,
        label_contract_reference=label_contract_reference,
    )
    logger.info(
        'Quantification model artifacts complete: %d artifacts under %s',
        len(model_artifacts),
        output_dir,
    )
    return {
        'mapping_template': mapping_template_path,
        'manifest': manifest_path,
        'manifest_scored_summary': manifest_summary_path,
        'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
        'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
        'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
        **label_override_artifacts,
        **_burden_first_artifacts(model_artifacts),
    }


def extract_roi_crops(
    scored_table: pd.DataFrame,
    output_dir: Path,
    padding: int = 32,
    min_component_area: int = 64,
) -> pd.DataFrame:
    """Extract deterministic glomerulus crops from canonical image+mask pairs."""
    output_dir = Path(output_dir)
    image_crop_dir = output_dir / 'images'
    mask_crop_dir = output_dir / 'masks'
    image_crop_dir.mkdir(parents=True, exist_ok=True)
    mask_crop_dir.mkdir(parents=True, exist_ok=True)

    result = scored_table.copy()
    result['roi_image_path'] = ''
    result['roi_mask_path'] = ''
    result['roi_bbox_x0'] = np.nan
    result['roi_bbox_y0'] = np.nan
    result['roi_bbox_x1'] = np.nan
    result['roi_bbox_y1'] = np.nan
    result['roi_area'] = np.nan
    result['roi_fill_fraction'] = np.nan
    result['roi_mean_intensity'] = np.nan
    result['roi_openness_score'] = np.nan

    for subject_image_id, subject_rows in result.groupby('subject_image_id', sort=True):
        first_row = subject_rows.iloc[0]
        if first_row['join_status'] != 'ok':
            continue

        image_path = Path(str(first_row['raw_image_path']))
        mask_path = Path(str(first_row['raw_mask_path']))
        image_array = np.array(Image.open(image_path).convert('RGB'))
        mask_array = np.array(Image.open(mask_path).convert('L'))
        if mask_array.shape[:2] != image_array.shape[:2]:
            result.loc[subject_rows.index, 'roi_status'] = 'image_mask_size_mismatch'
            continue

        components = _extract_components(mask_array, min_area=min_component_area)
        component_by_id = {
            int(component['glomerulus_id']): component for component in components
        }

        for index, scored_row in subject_rows.iterrows():
            glomerulus_id = int(scored_row['glomerulus_id'])
            component = component_by_id.get(glomerulus_id)
            if component is None:
                result.at[index, 'roi_status'] = 'component_below_min_area'
                continue

            x = int(component['bbox_x'])
            y = int(component['bbox_y'])
            w = int(component['bbox_w'])
            h = int(component['bbox_h'])
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(image_array.shape[1], x + w + padding)
            y1 = min(image_array.shape[0], y + h + padding)

            crop_image = image_array[y0:y1, x0:x1]
            crop_mask = (component['mask'][y0:y1, x0:x1] * 255).astype(np.uint8)

            crop_name = f'{subject_image_id}_g{glomerulus_id:03d}.png'
            image_crop_path = image_crop_dir / crop_name
            mask_crop_path = mask_crop_dir / crop_name
            Image.fromarray(crop_image).save(image_crop_path)
            Image.fromarray(crop_mask).save(mask_crop_path)

            gray_crop = np.array(Image.fromarray(crop_image).convert('L'))
            quant_metrics = calculate_quantification_metrics(crop_mask, gray_crop)
            fill_fraction = (
                float((crop_mask > 0).sum() / crop_mask.size) if crop_mask.size else 0.0
            )

            result.at[index, 'roi_status'] = 'ok'
            result.at[index, 'roi_image_path'] = str(image_crop_path)
            result.at[index, 'roi_mask_path'] = str(mask_crop_path)
            result.at[index, 'roi_bbox_x0'] = x0
            result.at[index, 'roi_bbox_y0'] = y0
            result.at[index, 'roi_bbox_x1'] = x1
            result.at[index, 'roi_bbox_y1'] = y1
            result.at[index, 'roi_area'] = int(component['area'])
            result.at[index, 'roi_fill_fraction'] = fill_fraction
            result.at[index, 'roi_mean_intensity'] = (
                float(gray_crop[crop_mask > 0].mean()) if (crop_mask > 0).any() else 0.0
            )
            result.at[index, 'roi_openness_score'] = float(quant_metrics.openness_score)

    result.to_csv(output_dir / 'roi_scored_examples.csv', index=False)
    return result


def extract_image_level_roi_crops(
    scored_table: pd.DataFrame,
    output_dir: Path,
    padding: int = 32,
    min_component_area: int = 64,
) -> pd.DataFrame:
    """Extract one union ROI crop per scored image using the full multi-component mask."""
    output_dir = Path(output_dir)
    image_crop_dir = output_dir / 'images'
    mask_crop_dir = output_dir / 'masks'
    image_crop_dir.mkdir(parents=True, exist_ok=True)
    mask_crop_dir.mkdir(parents=True, exist_ok=True)

    result = scored_table.copy()
    result['roi_image_path'] = ''
    result['roi_mask_path'] = ''
    result['roi_bbox_x0'] = np.nan
    result['roi_bbox_y0'] = np.nan
    result['roi_bbox_x1'] = np.nan
    result['roi_bbox_y1'] = np.nan
    result['roi_area'] = np.nan
    result['roi_fill_fraction'] = np.nan
    result['roi_mean_intensity'] = np.nan
    result['roi_openness_score'] = np.nan
    result['roi_component_count'] = np.nan
    result['roi_component_selection'] = ''
    result['roi_union_bbox_width'] = np.nan
    result['roi_union_bbox_height'] = np.nan
    result['roi_largest_component_area_fraction'] = np.nan

    for index, scored_row in result.iterrows():
        if str(scored_row.get('join_status', '')) != 'ok':
            continue
        if str(scored_row.get('score_status', 'ok')) != 'ok':
            result.at[index, 'roi_status'] = 'missing_score'
            continue

        image_path = Path(str(scored_row['raw_image_path']))
        mask_path = Path(str(scored_row['raw_mask_path']))
        image_array = np.array(Image.open(image_path).convert('RGB'))
        mask_array = np.array(Image.open(mask_path).convert('L'))
        if mask_array.shape[:2] != image_array.shape[:2]:
            result.at[index, 'roi_status'] = 'image_mask_size_mismatch'
            continue
        union = _build_union_mask(mask_array, min_component_area=min_component_area)
        if union is None:
            result.at[index, 'roi_status'] = 'component_below_min_area'
            continue

        result.at[index, 'roi_component_count'] = int(union['component_count'])
        result.at[index, 'roi_component_selection'] = str(union['component_selection'])

        x0 = max(0, int(union['bbox_x0']) - padding)
        y0 = max(0, int(union['bbox_y0']) - padding)
        x1 = min(image_array.shape[1], int(union['bbox_x1']) + padding)
        y1 = min(image_array.shape[0], int(union['bbox_y1']) + padding)

        crop_image = image_array[y0:y1, x0:x1]
        crop_mask = (union['mask'][y0:y1, x0:x1] * 255).astype(np.uint8)

        crop_name = f'{scored_row["subject_image_id"]}.png'
        image_crop_path = image_crop_dir / crop_name
        mask_crop_path = mask_crop_dir / crop_name
        Image.fromarray(crop_image).save(image_crop_path)
        Image.fromarray(crop_mask).save(mask_crop_path)

        gray_crop = np.array(Image.fromarray(crop_image).convert('L'))
        quant_metrics = calculate_quantification_metrics(crop_mask, gray_crop)
        fill_fraction = (
            float((crop_mask > 0).sum() / crop_mask.size) if crop_mask.size else 0.0
        )

        result.at[index, 'roi_status'] = 'ok'
        result.at[index, 'roi_image_path'] = str(image_crop_path)
        result.at[index, 'roi_mask_path'] = str(mask_crop_path)
        result.at[index, 'roi_bbox_x0'] = x0
        result.at[index, 'roi_bbox_y0'] = y0
        result.at[index, 'roi_bbox_x1'] = x1
        result.at[index, 'roi_bbox_y1'] = y1
        result.at[index, 'roi_area'] = int(union['union_area'])
        result.at[index, 'roi_fill_fraction'] = fill_fraction
        result.at[index, 'roi_mean_intensity'] = (
            float(gray_crop[crop_mask > 0].mean()) if (crop_mask > 0).any() else 0.0
        )
        result.at[index, 'roi_openness_score'] = float(quant_metrics.openness_score)
        result.at[index, 'roi_union_bbox_width'] = int(union['bbox_width'])
        result.at[index, 'roi_union_bbox_height'] = int(union['bbox_height'])
        result.at[index, 'roi_largest_component_area_fraction'] = float(
            union['largest_component_area_fraction']
        )

    result.to_csv(output_dir / 'roi_scored_examples.csv', index=False)
    return result


def _resolve_feature_map(encoded_output: Any) -> torch.Tensor:
    if isinstance(encoded_output, torch.Tensor):
        return encoded_output
    if isinstance(encoded_output, (list, tuple)):
        for item in reversed(encoded_output):
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError('Encoder output did not contain a tensor feature map')
    if isinstance(encoded_output, dict):
        for key in ('out', 'features', 'last_hidden_state'):
            value = encoded_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f'Unsupported encoder output type: {type(encoded_output)}')


def _prepare_encoder_for_forward(encoder: torch.nn.Module) -> torch.nn.Module:
    """Wrap encoder-like containers into a callable module for feature extraction."""
    if isinstance(encoder, torch.nn.ModuleList):
        if len(encoder) == 0:
            raise ValueError('Encoder ModuleList is empty')
        return encoder[0]
    return encoder


def extract_embedding_table(
    roi_table: pd.DataFrame,
    segmentation_model_path: Path,
    output_dir: Path,
    expected_size: int = DEFAULT_IMAGE_SIZE,
) -> pd.DataFrame:
    """Extract frozen encoder embeddings for ROI crops."""
    logger = get_logger('eq.quantification.embeddings')
    learn = load_model_safely(str(segmentation_model_path), model_type='glomeruli')
    learn.model.eval()
    encoder = _get_encoder_module(learn.model)
    if encoder is None:
        raise RuntimeError('Could not resolve encoder module from segmentation learner')
    encoder = _prepare_encoder_for_forward(encoder).to(
        next(learn.model.parameters()).device
    )
    encoder.eval()

    device = next(learn.model.parameters()).device
    prediction_core = create_prediction_core(expected_size)

    valid_rows = (
        roi_table[roi_table['roi_status'] == 'ok'].copy().reset_index(drop=True)
    )
    if valid_rows.empty:
        raise ContractPreparationError(
            'No ROI crops were extracted successfully; cannot build embeddings'
        )

    embeddings: list[np.ndarray] = []
    for row in valid_rows.itertuples(index=False):
        roi_image = Image.open(str(row.roi_image_path)).convert('RGB')
        tensor = prediction_core.preprocess_image_imagenet_normalized(roi_image).to(
            device
        )
        with torch.no_grad():
            feature_map = _resolve_feature_map(encoder(tensor))
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=1)
        embeddings.append(pooled.flatten().detach().cpu().numpy().astype(np.float32))

    embedding_matrix = np.vstack(embeddings)
    embedding_columns = [
        f'embedding_{index:04d}' for index in range(embedding_matrix.shape[1])
    ]
    embedding_df = pd.concat(
        [
            valid_rows.reset_index(drop=True),
            pd.DataFrame(embedding_matrix, columns=embedding_columns),
        ],
        axis=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_df.to_csv(output_dir / 'roi_embeddings.csv', index=False)
    _save_json(
        {
            'segmentation_model_path': str(segmentation_model_path),
            'expected_size': int(expected_size),
            'embedding_dim': int(embedding_matrix.shape[1]),
            'pooling': 'adaptive_avg_pool2d_1x1',
            'representation': 'frozen_segmentation_encoder',
            'preprocessing': 'imagenet_normalized_fastai',
        },
        output_dir / 'embedding_metadata.json',
    )
    logger.info('Saved ROI embeddings to %s', output_dir / 'roi_embeddings.csv')
    return embedding_df


def _save_preview_image(
    image: Image.Image, output_path: Path, max_side: int = 900
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    preview.save(output_path)
    return output_path


def _render_mask_overlay(
    raw_image_path: Path, raw_mask_path: Path, bbox: tuple[int, int, int, int]
) -> Image.Image:
    image = Image.open(raw_image_path).convert('RGB')
    mask = np.array(Image.open(raw_mask_path).convert('L'))
    image_array = np.array(image).astype(np.float32)
    positive = mask > 0
    overlay_color = np.array([255.0, 64.0, 64.0], dtype=np.float32)
    image_array[positive] = 0.65 * image_array[positive] + 0.35 * overlay_color
    overlay = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, outline=(32, 200, 255), width=4)
    return overlay


def _build_probability_rows(row: pd.Series) -> str:
    rows: list[str] = []
    for score in ALLOWED_SCORE_VALUES:
        column = _score_probability_column_name(float(score))
        probability = float(row.get(column, 0.0))
        rows.append(
            '<tr>'
            f'<td>{escape(_score_label(float(score)))}</td>'
            f'<td><div class="prob-bar"><span style="width:{probability * 100:.1f}%"></span></div></td>'
            f'<td>{probability:.3f}</td>'
            '</tr>'
        )
    return ''.join(rows)


def _review_narrative(row: pd.Series) -> str:
    return (
        f'Prediction was {float(row["predicted_score"]):.1f} against a true grade of {float(row["score"]):.1f}. '
        f'The full mask contained {int(row["roi_component_count"])} connected component(s) with fill fraction '
        f'{float(row["roi_fill_fraction"]):.3f} and openness score {float(row["roi_openness_score"]):.3f}. '
        'These are descriptive audit signals for review, not faithful feature attribution.'
    )


def _select_review_examples(
    predictions_df: pd.DataFrame, max_examples: int = 7
) -> pd.DataFrame:
    if predictions_df.empty:
        return predictions_df.copy()

    work = predictions_df.copy()
    selected: list[str] = []
    selected_rows: list[pd.Series] = []

    def take_rows(frame: pd.DataFrame, count: int, bucket: str) -> None:
        nonlocal selected, selected_rows
        for _, row in frame.iterrows():
            key = str(row['subject_image_id'])
            if key in selected:
                continue
            picked = row.copy()
            picked['selection_bucket'] = bucket
            selected_rows.append(picked)
            selected.append(key)
            if (
                len(selected_rows) >= max_examples
                or sum(
                    1 for item in selected_rows if item['selection_bucket'] == bucket
                )
                >= count
            ):
                break

    take_rows(
        work.sort_values(
            ['absolute_error', 'entropy', 'top_two_margin'],
            ascending=[False, False, True],
        ),
        count=2,
        bucket='highest_error',
    )
    take_rows(
        work.sort_values(
            ['entropy', 'top_two_margin', 'absolute_error'],
            ascending=[False, True, False],
        ),
        count=2,
        bucket='highest_uncertainty',
    )
    confident_correct = work[
        work['predicted_class'] == work['score_class']
    ].sort_values(['top_two_margin', 'entropy'], ascending=[False, True])
    take_rows(confident_correct, count=2, bucket='confident_correct')

    midpoint = float(np.median(ALLOWED_SCORE_VALUES))
    mid_range = work.assign(
        distance_to_midpoint=np.abs(work['score'] - midpoint),
        expected_distance_to_midpoint=np.abs(work['expected_score'] - midpoint),
    ).sort_values(
        [
            'distance_to_midpoint',
            'expected_distance_to_midpoint',
            'absolute_error',
            'entropy',
        ],
        ascending=[True, True, True, True],
    )
    take_rows(mid_range, count=1, bucket='representative_mid_range')

    if len(selected_rows) < min(max_examples, len(work)):
        filler = work.sort_values(
            ['absolute_error', 'entropy'], ascending=[False, False]
        )
        take_rows(filler, count=max_examples, bucket='additional_review_case')

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return work.head(min(max_examples, len(work))).copy()
    return selected_df.head(min(max_examples, len(work))).reset_index(drop=True)


def generate_html_review_report(
    predictions_df: pd.DataFrame,
    metrics_summary: dict[str, Any],
    output_dir: Path,
    max_examples: int = 7,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    assets_dir = output_dir / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_review_examples(predictions_df, max_examples=max_examples)
    selected_path = output_dir / 'selected_examples.csv'
    selected.to_csv(selected_path, index=False)

    cards: list[str] = []
    for example_index, row in selected.iterrows():
        raw_image_path = Path(str(row['raw_image_path']))
        raw_mask_path = Path(str(row['raw_mask_path']))
        roi_image_path = Path(str(row['roi_image_path']))
        bbox = (
            int(row['roi_bbox_x0']),
            int(row['roi_bbox_y0']),
            int(row['roi_bbox_x1']),
            int(row['roi_bbox_y1']),
        )

        raw_preview_path = assets_dir / f'example_{example_index:02d}_raw.png'
        overlay_preview_path = assets_dir / f'example_{example_index:02d}_overlay.png'
        roi_preview_path = assets_dir / f'example_{example_index:02d}_roi.png'

        raw_image = Image.open(raw_image_path).convert('RGB')
        raw_with_bbox = raw_image.copy()
        raw_draw = ImageDraw.Draw(raw_with_bbox)
        raw_draw.rectangle(bbox, outline=(32, 200, 255), width=4)
        _save_preview_image(raw_with_bbox, raw_preview_path)
        _save_preview_image(
            _render_mask_overlay(raw_image_path, raw_mask_path, bbox),
            overlay_preview_path,
        )
        _save_preview_image(
            Image.open(roi_image_path).convert('RGB'), roi_preview_path, max_side=500
        )

        probability_rows = _build_probability_rows(row)
        cards.append(
            f"""
            <section class="example-card">
              <h2>{escape(str(row['subject_image_id']))} <span class="bucket">{escape(str(row['selection_bucket']))}</span></h2>
              <div class="image-grid">
                <figure>
                  <img src="assets/{raw_preview_path.name}" alt="Raw image with ROI bounding box">
                  <figcaption>Raw image with union ROI bounding box</figcaption>
                </figure>
                <figure>
                  <img src="assets/{overlay_preview_path.name}" alt="Full mask overlay">
                  <figcaption>Full multi-component mask overlay</figcaption>
                </figure>
                <figure>
                  <img src="assets/{roi_preview_path.name}" alt="Union ROI crop">
                  <figcaption>Union ROI crop used for embeddings</figcaption>
                </figure>
              </div>
              <div class="summary-grid">
                <div><strong>True grade</strong><span>{float(row['score']):.1f}</span></div>
                <div><strong>Predicted grade</strong><span>{float(row['predicted_score']):.1f}</span></div>
                <div><strong>Expected score</strong><span>{float(row['expected_score']):.3f}</span></div>
                <div><strong>Absolute error</strong><span>{float(row['absolute_error']):.3f}</span></div>
                <div><strong>Top-two margin</strong><span>{float(row['top_two_margin']):.3f}</span></div>
                <div><strong>Entropy</strong><span>{float(row['entropy']):.3f}</span></div>
              </div>
              <p class="narrative">{escape(_review_narrative(row))}</p>
              <div class="detail-grid">
                <div>
                  <h3>Class Probabilities</h3>
                  <table>
                    <thead><tr><th>Grade</th><th>Probability</th><th>Value</th></tr></thead>
                    <tbody>{probability_rows}</tbody>
                  </table>
                </div>
                <div>
                  <h3>Audit Features</h3>
                  <table>
                    <tbody>
                      <tr><th>Mask components</th><td>{int(row['roi_component_count'])}</td></tr>
                      <tr><th>ROI area</th><td>{int(row['roi_area'])}</td></tr>
                      <tr><th>Fill fraction</th><td>{float(row['roi_fill_fraction']):.3f}</td></tr>
                      <tr><th>Mean intensity</th><td>{float(row['roi_mean_intensity']):.3f}</td></tr>
                      <tr><th>Openness score</th><td>{float(row['roi_openness_score']):.3f}</td></tr>
                      <tr><th>Union bbox</th><td>{int(row['roi_union_bbox_width'])} x {int(row['roi_union_bbox_height'])}</td></tr>
                      <tr><th>Largest-component area fraction</th><td>{float(row['roi_largest_component_area_fraction']):.3f}</td></tr>
                      <tr><th>ROI selection</th><td>{escape(str(row['roi_component_selection']))}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
            """
        )

    overall = metrics_summary.get('overall', {})
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Endotheliosis Review Report</title>
      <style>
        body {{ font-family: "Helvetica Neue", Arial, sans-serif; margin: 2rem auto; max-width: 1200px; color: #1f2933; background: #f7fafc; }}
        h1, h2, h3 {{ color: #102a43; }}
        .overall {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }}
        .overall div, .summary-grid div {{ background: white; border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }}
        .overall strong, .summary-grid strong {{ display: block; font-size: 0.85rem; color: #486581; margin-bottom: 0.25rem; }}
        .overall span, .summary-grid span {{ font-size: 1.1rem; font-weight: 600; }}
        .note {{ background: #fff7e6; border-left: 4px solid #d9822b; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
        .example-card {{ background: white; padding: 1.25rem; border-radius: 14px; margin-bottom: 1.25rem; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08); }}
        .image-grid, .detail-grid, .summary-grid {{ display: grid; gap: 1rem; }}
        .image-grid {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 1rem; }}
        .detail-grid {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
        img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #d9e2ec; background: #f0f4f8; }}
        figure {{ margin: 0; }}
        figcaption {{ font-size: 0.85rem; color: #52606d; margin-top: 0.4rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 0.45rem 0.4rem; border-bottom: 1px solid #e5e7eb; font-size: 0.92rem; }}
        .prob-bar {{ width: 100%; background: #e5e7eb; border-radius: 999px; height: 10px; overflow: hidden; }}
        .prob-bar span {{ display: block; height: 100%; background: linear-gradient(90deg, #2cb1bc, #127fbf); }}
        .bucket {{ font-size: 0.85rem; color: #486581; }}
        .narrative {{ color: #334e68; line-height: 1.5; }}
      </style>
    </head>
    <body>
      <h1>Endotheliosis Example Review</h1>
      <div class="note">
        The model target is the Label Studio image-level grade attached to the full image/mask pair. Probabilities,
        top-two margin, and entropy are confidence proxies for review. The audit features shown below are descriptive
        context, not faithful feature attribution or mechanistic explanation.
      </div>
      <section class="overall">
        <div><strong>Examples reviewed</strong><span>{len(selected)}</span></div>
        <div><strong>Total examples</strong><span>{int(metrics_summary.get('n_examples', len(predictions_df)))}</span></div>
        <div><strong>MAE</strong><span>{float(overall.get('mae', np.nan)):.3f}</span></div>
        <div><strong>Accuracy</strong><span>{float(overall.get('accuracy', np.nan)):.3f}</span></div>
        <div><strong>Within-one-bin</strong><span>{float(overall.get('within_one_bin_accuracy', np.nan)):.3f}</span></div>
        <div><strong>Quadratic weighted kappa</strong><span>{float(overall.get('quadratic_weighted_kappa', np.nan)):.3f}</span></div>
      </section>
      {''.join(cards)}
    </body>
    </html>
    """

    html_path = output_dir / 'ordinal_review.html'
    html_path.write_text(html, encoding='utf-8')
    return {
        'html': html_path,
        'selected_examples': selected_path,
        'assets_dir': assets_dir,
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _format_optional_float(value: Any, digits: int = 3) -> str:
    try:
        if value in ('', None) or not np.isfinite(float(value)):
            return 'not estimable'
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return 'not estimable'


def _select_quantification_review_examples(
    merged_df: pd.DataFrame, max_examples: int = 9
) -> pd.DataFrame:
    work = merged_df.copy()
    selected: list[pd.Series] = []
    used: set[tuple[str, str]] = set()

    def take(frame: pd.DataFrame, bucket: str, count: int = 1) -> None:
        nonlocal selected
        for _, row in frame.iterrows():
            key = (
                str(row.get('subject_image_id', '')),
                str(row.get('glomerulus_id', '')),
            )
            if key in used:
                continue
            row = row.copy()
            row['review_bucket'] = bucket
            selected.append(row)
            used.add(key)
            if (
                len([item for item in selected if item.get('review_bucket') == bucket])
                >= count
            ):
                break

    if 'stage_index_absolute_error' in work.columns:
        take(
            work.sort_values('stage_index_absolute_error', ascending=True),
            'representative_low_error',
        )
        take(
            work.sort_values('stage_index_absolute_error', ascending=False),
            'high_error',
        )
    if 'prediction_set_scores' in work.columns:
        set_sizes = work['prediction_set_scores'].map(
            lambda value: len([item for item in str(value).split('|') if item])
        )
        take(
            work.assign(_set_size=set_sizes).sort_values('_set_size', ascending=False),
            'high_uncertainty',
        )
    if 'cohort_id' in work.columns:
        for cohort, cohort_df in work.groupby('cohort_id'):
            take(
                cohort_df.sort_values('stage_index_absolute_error', ascending=True),
                f'cohort_{cohort}',
            )
    if 'score' in work.columns:
        take(work.sort_values('score', ascending=False), 'high_observed_burden')
    if len(selected) < min(max_examples, len(work)):
        take(
            work.sort_values(
                ['stage_index_absolute_error', BURDEN_COLUMN], ascending=[False, False]
            ),
            'additional_review_case',
            count=max_examples,
        )
    return (
        pd.DataFrame(selected)
        .head(min(max_examples, len(work)))
        .drop(columns=['_set_size'], errors='ignore')
        .reset_index(drop=True)
    )


def generate_combined_quantification_review(
    *,
    ordinal_predictions_path: Path,
    ordinal_metrics_path: Path,
    burden_artifacts: dict[str, Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Write a human-readable burden/ordinal quantification review bundle."""
    output_dir = Path(output_dir)
    assets_dir = output_dir / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    ordinal_predictions = pd.read_csv(ordinal_predictions_path)
    burden_predictions = pd.read_csv(burden_artifacts['burden_predictions'])
    ordinal_predictions = ordinal_predictions.rename(columns={'fold': 'ordinal_fold'})
    join_keys = [
        key
        for key in ['subject_image_id', 'glomerulus_id']
        if key in ordinal_predictions.columns and key in burden_predictions.columns
    ]
    if not join_keys:
        raise ContractPreparationError(
            'Cannot build quantification review: no shared prediction join keys'
        )
    if burden_predictions.duplicated(join_keys).any():
        raise ContractPreparationError(
            f'Cannot build quantification review: burden predictions are not one-to-one on {join_keys}'
        )
    if ordinal_predictions.duplicated(join_keys).any():
        raise ContractPreparationError(
            f'Cannot build quantification review: ordinal predictions are not one-to-one on {join_keys}'
        )
    merged = burden_predictions.merge(
        ordinal_predictions,
        on=join_keys,
        how='left',
        suffixes=('', '_ordinal'),
        validate='one_to_one',
    )
    if len(merged) != len(burden_predictions):
        raise ContractPreparationError(
            'Cannot build quantification review: merged row count does not match burden predictions'
        )
    if 'predicted_score' in merged.columns and merged['predicted_score'].isna().any():
        raise ContractPreparationError(
            'Cannot build quantification review: ordinal comparator fields are missing after merge'
        )
    review_examples = _select_quantification_review_examples(merged)
    review_examples_path = output_dir / 'review_examples.csv'
    review_examples.to_csv(review_examples_path, index=False)

    burden_metrics = _read_json_if_exists(burden_artifacts['burden_metrics'])
    ordinal_metrics = _read_json_if_exists(ordinal_metrics_path)
    grouping_audit = _read_json_if_exists(burden_artifacts['grouping_audit'])
    uncertainty = _read_json_if_exists(burden_artifacts['uncertainty_calibration'])
    threshold_support = pd.read_csv(burden_artifacts['threshold_support'])
    cohort_metrics = (
        pd.read_csv(burden_artifacts['cohort_metrics'])
        if burden_artifacts['cohort_metrics'].exists()
        and burden_artifacts['cohort_metrics'].stat().st_size > 0
        else pd.DataFrame()
    )
    final_cohort_metrics = (
        pd.read_csv(burden_artifacts['final_model_cohort_metrics'])
        if burden_artifacts.get('final_model_cohort_metrics')
        and burden_artifacts['final_model_cohort_metrics'].exists()
        and burden_artifacts['final_model_cohort_metrics'].stat().st_size > 0
        else pd.DataFrame()
    )
    cohort_metrics_for_results = (
        final_cohort_metrics if not final_cohort_metrics.empty else cohort_metrics
    )
    group_intervals = pd.read_csv(burden_artifacts['group_summary_intervals'])
    nearest_examples = (
        pd.read_csv(burden_artifacts['nearest_examples'])
        if burden_artifacts['nearest_examples'].exists()
        and burden_artifacts['nearest_examples'].stat().st_size > 0
        else pd.DataFrame()
    )
    signal_comparator = (
        pd.read_csv(burden_artifacts['signal_comparator_metrics'])
        if burden_artifacts.get('signal_comparator_metrics')
        and burden_artifacts['signal_comparator_metrics'].exists()
        and burden_artifacts['signal_comparator_metrics'].stat().st_size > 0
        else pd.DataFrame()
    )
    precision_candidate_summary = (
        _read_json_if_exists(burden_artifacts['precision_candidate_summary'])
        if burden_artifacts.get('precision_candidate_summary')
        else {}
    )
    morphology_candidate_summary = (
        _read_json_if_exists(burden_artifacts['morphology_candidate_summary'])
        if burden_artifacts.get('morphology_candidate_summary')
        else {}
    )
    morphology_diagnostics = (
        _read_json_if_exists(burden_artifacts['morphology_feature_diagnostics'])
        if burden_artifacts.get('morphology_feature_diagnostics')
        else {}
    )
    morphology_candidates = (
        pd.read_csv(burden_artifacts['morphology_candidate_metrics'])
        if burden_artifacts.get('morphology_candidate_metrics')
        and burden_artifacts['morphology_candidate_metrics'].exists()
        and burden_artifacts['morphology_candidate_metrics'].stat().st_size > 0
        else pd.DataFrame()
    )
    learned_roi_summary = (
        _read_json_if_exists(burden_artifacts['learned_roi_candidate_summary'])
        if burden_artifacts.get('learned_roi_candidate_summary')
        else {}
    )
    learned_roi_metrics = (
        pd.read_csv(burden_artifacts['learned_roi_candidate_metrics'])
        if burden_artifacts.get('learned_roi_candidate_metrics')
        and burden_artifacts['learned_roi_candidate_metrics'].exists()
        and burden_artifacts['learned_roi_candidate_metrics'].stat().st_size > 0
        else pd.DataFrame()
    )
    source_aware_verdict = (
        _read_json_if_exists(burden_artifacts['source_aware_estimator_verdict'])
        if burden_artifacts.get('source_aware_estimator_verdict')
        else {}
    )
    source_aware_metrics = (
        pd.read_csv(burden_artifacts['source_aware_metrics_by_split'])
        if burden_artifacts.get('source_aware_metrics_by_split')
        and burden_artifacts['source_aware_metrics_by_split'].exists()
        and burden_artifacts['source_aware_metrics_by_split'].stat().st_size > 0
        else pd.DataFrame()
    )
    source_aware_upstream = (
        _read_json_if_exists(burden_artifacts['source_aware_upstream_roi_adequacy'])
        if burden_artifacts.get('source_aware_upstream_roi_adequacy')
        else {}
    )
    severe_aware_verdict = (
        _read_json_if_exists(burden_artifacts['severe_aware_estimator_verdict'])
        if burden_artifacts.get('severe_aware_estimator_verdict')
        else {}
    )
    severe_aware_metrics = (
        pd.read_csv(burden_artifacts['severe_aware_metrics_by_split'])
        if burden_artifacts.get('severe_aware_metrics_by_split')
        and burden_artifacts['severe_aware_metrics_by_split'].exists()
        and burden_artifacts['severe_aware_metrics_by_split'].stat().st_size > 0
        else pd.DataFrame()
    )
    severe_threshold_metrics = (
        pd.read_csv(burden_artifacts['severe_aware_severe_threshold_metrics'])
        if burden_artifacts.get('severe_aware_severe_threshold_metrics')
        and burden_artifacts['severe_aware_severe_threshold_metrics'].exists()
        and burden_artifacts['severe_aware_severe_threshold_metrics'].stat().st_size > 0
        else pd.DataFrame()
    )
    grade_model_verdict = (
        _read_json_if_exists(burden_artifacts['endotheliosis_grade_model_verdict'])
        if burden_artifacts.get('endotheliosis_grade_model_verdict')
        else {}
    )
    grade_model_metrics = (
        pd.read_csv(burden_artifacts['endotheliosis_grade_model_candidate_metrics'])
        if burden_artifacts.get('endotheliosis_grade_model_candidate_metrics')
        and burden_artifacts['endotheliosis_grade_model_candidate_metrics'].exists()
        and burden_artifacts['endotheliosis_grade_model_candidate_metrics']
        .stat()
        .st_size
        > 0
        else pd.DataFrame()
    )

    support_status = str(burden_metrics.get('support_gate_status', 'unknown'))
    numerical_status = str(burden_metrics.get('numerical_stability_status', 'unknown'))
    overall = burden_metrics.get('overall', {})
    nominal_coverage = float(uncertainty.get('nominal_coverage', 0.9))
    empirical_coverage = float(
        uncertainty.get('overall', {}).get(
            'coverage', overall.get('prediction_set_coverage', 0.0)
        )
    )
    burden_interval_coverage = float(uncertainty.get('burden_interval_coverage', 0.0))
    coverage_gate_passed = empirical_coverage >= nominal_coverage
    numerical_acceptable = numerical_status == 'ok'
    operational_status = (
        'operational_candidate'
        if support_status == 'passed' and numerical_acceptable and coverage_gate_passed
        else 'exploratory_not_ready'
    )
    docs_ready = operational_status == 'operational_candidate'
    ordinal_overall = ordinal_metrics.get('overall', {})
    score_counts = burden_metrics.get('score_counts', {})
    score_distribution = ', '.join(
        f'{score}: {count}' for score, count in score_counts.items()
    )
    best_image_candidate = (
        precision_candidate_summary.get('best_image_level_candidate', {}) or {}
    )
    best_subject_candidate = (
        precision_candidate_summary.get('best_subject_level_candidate', {}) or {}
    )
    precision_recommendation = precision_candidate_summary.get('recommendation', '')
    morphology_best_image = (
        morphology_candidate_summary.get('best_image_level_candidate', {}) or {}
    )
    morphology_best_subject = (
        morphology_candidate_summary.get('best_subject_level_candidate', {}) or {}
    )
    learned_roi_image_readiness = learned_roi_summary.get('per_image_readiness', {})
    learned_roi_subject_readiness = learned_roi_summary.get(
        'subject_cohort_readiness', {}
    )
    learned_roi_ready = bool(learned_roi_summary.get('readme_docs_ready', False))
    learned_roi_track = str(learned_roi_summary.get('readme_docs_ready_track', ''))
    learned_roi_blockers = learned_roi_summary.get('blockers', [])
    source_aware_status = str(source_aware_verdict.get('overall_status', 'not_run'))
    source_aware_readme_ready = bool(
        source_aware_verdict.get('readme_snippet_eligible', False)
    )
    source_aware_scope_limiters = source_aware_verdict.get('scope_limiters', [])
    source_aware_hard_blockers = source_aware_verdict.get('hard_blockers', [])
    source_aware_reportable = source_aware_verdict.get('reportable_scopes', {})
    severe_aware_status = str(severe_aware_verdict.get('overall_status', 'not_run'))
    severe_aware_readme_ready = bool(
        severe_aware_verdict.get('readme_snippet_eligible', False)
    )
    severe_aware_reportable = severe_aware_verdict.get('reportable_scopes', {})
    severe_aware_scope_limiters = severe_aware_verdict.get('scope_limiters', [])
    severe_aware_hard_blockers = severe_aware_verdict.get('hard_blockers', [])
    grade_model_status = str(grade_model_verdict.get('overall_status', 'not_run'))
    grade_model_readme_ready = bool(
        grade_model_verdict.get('readme_facing_deployment_allowed', False)
    )
    grade_model_hard_blockers = grade_model_verdict.get('hard_blockers', [])
    severe_aware_figure_links = []
    for key, path in burden_artifacts.items():
        if key.startswith('severe_aware_figure_') and path.exists():
            rel = (
                Path('../burden_model/severe_aware_ordinal_estimator/summary/figures')
                / path.name
            )
            severe_aware_figure_links.append(
                f'<li><a href="{escape(str(rel))}">{escape(path.name)}</a></li>'
            )
    source_aware_figure_links = []
    for key, path in burden_artifacts.items():
        if key.startswith('source_aware_figure_') and path.exists():
            rel = (
                Path('../burden_model/source_aware_estimator/summary/figures')
                / path.name
            )
            source_aware_figure_links.append(
                f'<li><a href="{escape(str(rel))}">{escape(path.name)}</a></li>'
            )

    cohort_table_rows = []
    for _, row in cohort_metrics_for_results.iterrows():
        cohort_table_rows.append(
            '<tr>'
            f'<td>{escape(str(row.get("cohort_id", "")))}</td>'
            f'<td>{int(row.get("n_rows", 0))}</td>'
            f'<td>{int(row.get("n_subjects", row.get("n_samples", 0)))}</td>'
            f'<td>{_format_optional_float(row.get("subject_weighted_mean_predicted_burden", row.get("sample_weighted_mean_predicted_burden")))}</td>'
            f'<td>{_format_optional_float(row.get("stage_index_mae"))}</td>'
            '</tr>'
        )
    if not cohort_table_rows:
        cohort_table_rows.append(
            '<tr><td colspan="5">No cohort summaries available.</td></tr>'
        )

    signal_rows = []
    for _, row in signal_comparator.iterrows():
        signal_rows.append(
            '<tr>'
            f'<td>{escape(str(row.get("candidate_id", row.get("model_family", ""))))}</td>'
            f'<td>{escape(str(row.get("target_level", "")))}</td>'
            f'<td>{escape(str(row.get("feature_set", "")))}</td>'
            f'<td>{escape(str(row.get("validation_grouping", "")))}</td>'
            f'<td>{_format_optional_float(row.get("stage_index_mae"))}</td>'
            f'<td>{escape(str(row.get("candidate_status", "")))}</td>'
            '</tr>'
        )
    if not signal_rows:
        signal_rows.append(
            '<tr><td colspan="6">No precision candidate screen available.</td></tr>'
        )

    morphology_rows = []
    for _, row in morphology_candidates.iterrows():
        morphology_rows.append(
            '<tr>'
            f'<td>{escape(str(row.get("candidate_id", "")))}</td>'
            f'<td>{escape(str(row.get("target_level", "")))}</td>'
            f'<td>{escape(str(row.get("feature_set", "")))}</td>'
            f'<td>{escape(str(row.get("validation_grouping", "")))}</td>'
            f'<td>{_format_optional_float(row.get("stage_index_mae"))}</td>'
            f'<td>{escape(str(row.get("candidate_status", "")))}</td>'
            '</tr>'
        )
    if not morphology_rows:
        morphology_rows.append(
            '<tr><td colspan="6">No morphology candidate screen available.</td></tr>'
        )

    learned_roi_rows = []
    for _, row in learned_roi_metrics.iterrows():
        learned_roi_rows.append(
            '<tr>'
            f'<td>{escape(str(row.get("candidate_id", "")))}</td>'
            f'<td>{escape(str(row.get("target_level", "")))}</td>'
            f'<td>{escape(str(row.get("feature_set", "")))}</td>'
            f'<td>{escape(str(row.get("validation_grouping", "")))}</td>'
            f'<td>{_format_optional_float(row.get("stage_index_mae"))}</td>'
            f'<td>{_format_optional_float(row.get("grade_scale_mae"))}</td>'
            f'<td>{escape(str(row.get("candidate_status", "")))}</td>'
            '</tr>'
        )
    if not learned_roi_rows:
        learned_roi_rows.append(
            '<tr><td colspan="7">No learned ROI candidate screen available.</td></tr>'
        )

    support_rows = []
    for _, row in threshold_support.iterrows():
        support_rows.append(
            '<tr>'
            f'<td>{escape(str(row["stratum"]))}</td>'
            f'<td>&gt; {float(row["threshold"]):g}</td>'
            f'<td>{int(row["positive_groups"])}</td>'
            f'<td>{int(row["negative_groups"])}</td>'
            f'<td>{escape(str(row["support_status"]))}</td>'
            '</tr>'
        )

    group_rows = []
    for _, row in group_intervals.iterrows():
        group_rows.append(
            '<tr>'
            f'<td>{escape(str(row.get("stratum", "")))}</td>'
            f'<td>{escape(str(row.get("estimand", "")))}</td>'
            f'<td>{int(row.get("n_clusters", 0))}</td>'
            f'<td>{_format_optional_float(row.get("mean_burden_0_100"))}</td>'
            f'<td>{_format_optional_float(row.get("ci_low_0_100"))} to {_format_optional_float(row.get("ci_high_0_100"))}</td>'
            f'<td>{escape(str(row.get("status", "")))}</td>'
            '</tr>'
        )

    cards: list[str] = []
    for example_index, row in review_examples.iterrows():
        raw_image_path = Path(
            str(row.get('raw_image_path', row.get('raw_image_path_ordinal', '')))
        )
        raw_mask_path = Path(
            str(row.get('raw_mask_path', row.get('raw_mask_path_ordinal', '')))
        )
        roi_image_path = Path(
            str(row.get('roi_image_path', row.get('roi_image_path_ordinal', '')))
        )
        image_html = ''
        if (
            raw_image_path.exists()
            and raw_mask_path.exists()
            and roi_image_path.exists()
        ):
            bbox = (
                int(row.get('roi_bbox_x0', row.get('roi_bbox_x0_ordinal', 0))),
                int(row.get('roi_bbox_y0', row.get('roi_bbox_y0_ordinal', 0))),
                int(row.get('roi_bbox_x1', row.get('roi_bbox_x1_ordinal', 0))),
                int(row.get('roi_bbox_y1', row.get('roi_bbox_y1_ordinal', 0))),
            )
            raw_preview_path = (
                assets_dir / f'burden_example_{example_index:02d}_raw.png'
            )
            overlay_preview_path = (
                assets_dir / f'burden_example_{example_index:02d}_overlay.png'
            )
            roi_preview_path = (
                assets_dir / f'burden_example_{example_index:02d}_roi.png'
            )
            raw_image = Image.open(raw_image_path).convert('RGB')
            raw_with_bbox = raw_image.copy()
            raw_draw = ImageDraw.Draw(raw_with_bbox)
            raw_draw.rectangle(bbox, outline=(32, 200, 255), width=4)
            _save_preview_image(raw_with_bbox, raw_preview_path)
            _save_preview_image(
                _render_mask_overlay(raw_image_path, raw_mask_path, bbox),
                overlay_preview_path,
            )
            _save_preview_image(
                Image.open(roi_image_path).convert('RGB'),
                roi_preview_path,
                max_side=500,
            )
            image_html = f"""
            <div class="image-grid">
              <figure><img src="assets/{raw_preview_path.name}" alt="Raw image with ROI box"><figcaption>Raw image with ROI box</figcaption></figure>
              <figure><img src="assets/{overlay_preview_path.name}" alt="Mask overlay"><figcaption>Mask overlay</figcaption></figure>
              <figure><img src="assets/{roi_preview_path.name}" alt="ROI crop"><figcaption>ROI crop used for embeddings</figcaption></figure>
            </div>
            """
        subject_image_id = str(row.get('subject_image_id', 'unknown'))
        neighbor_rows = ''
        if (
            not nearest_examples.empty
            and 'subject_image_id' in nearest_examples.columns
        ):
            neighbors = nearest_examples[
                nearest_examples['subject_image_id'].astype(str) == subject_image_id
            ].head(3)
            for _, neighbor in neighbors.iterrows():
                neighbor_rows += (
                    '<tr>'
                    f'<td>{int(neighbor.get("neighbor_rank", 0))}</td>'
                    f'<td>{escape(str(neighbor.get("neighbor_subject_image_id", "")))}</td>'
                    f'<td>{_format_optional_float(neighbor.get("neighbor_score"), 1)}</td>'
                    f'<td>{_format_optional_float(neighbor.get("neighbor_distance"))}</td>'
                    '</tr>'
                )
        if not neighbor_rows:
            neighbor_rows = (
                '<tr><td colspan="4">No fold-pure nearest examples available.</td></tr>'
            )
        cohort_id = row.get('cohort_id', row.get('cohort_id_ordinal', ''))
        sample_id = row.get('sample_id', row.get('sample_id_ordinal', ''))
        prediction_source = row.get(
            'prediction_source', 'held_out_grouped_fold_prediction'
        )
        source_image = row.get('raw_image_path', row.get('raw_image_path_ordinal', ''))
        roi_image = row.get('roi_image_path', row.get('roi_image_path_ordinal', ''))

        threshold_rows = ''.join(
            '<tr>'
            f'<td>{column}</td>'
            f'<td>{_format_optional_float(row.get(column))}</td>'
            '</tr>'
            for column in [
                'prob_score_gt_0',
                'prob_score_gt_0p5',
                'prob_score_gt_1',
                'prob_score_gt_1p5',
                'prob_score_gt_2',
            ]
        )
        ordinal_prediction = row.get(
            'predicted_score', row.get('predicted_score_ordinal', '')
        )
        cards.append(
            f"""
            <section class="example-card">
              <h2>{escape(subject_image_id)} <span class="bucket">{escape(str(row.get('review_bucket', '')))}</span></h2>
              {image_html}
              <div class="summary-grid">
                <div><strong>Observed score</strong><span>{_format_optional_float(row.get('score'), 1)}</span></div>
                <div><strong>Burden index</strong><span>{_format_optional_float(row.get(BURDEN_COLUMN))}</span></div>
                <div><strong>Burden interval</strong><span>{_format_optional_float(row.get('burden_interval_low_0_100'))} to {_format_optional_float(row.get('burden_interval_high_0_100'))}</span></div>
                <div><strong>Prediction set</strong><span>{escape(str(row.get('prediction_set_scores', '')))}</span></div>
                <div><strong>Ordinal prediction</strong><span>{escape(str(ordinal_prediction))}</span></div>
                <div><strong>Fold</strong><span>{escape(str(row.get('fold', '')))}</span></div>
                <div><strong>Prediction source</strong><span>{escape(str(prediction_source))}</span></div>
                <div><strong>Cohort</strong><span>{escape(str(cohort_id))}</span></div>
                <div><strong>Sample</strong><span>{escape(str(sample_id))}</span></div>
              </div>
              <p class="provenance"><strong>Source image:</strong> {escape(str(source_image))}<br><strong>ROI image:</strong> {escape(str(roi_image))}</p>
              <div class="detail-grid">
                <div>
                  <h3>Threshold Profile</h3>
                  <table><tbody>{threshold_rows}</tbody></table>
                </div>
                <div>
                  <h3>Nearest Scored Examples</h3>
                  <table><thead><tr><th>Rank</th><th>Example</th><th>Score</th><th>Distance</th></tr></thead><tbody>{neighbor_rows}</tbody></table>
                </div>
              </div>
            </section>
            """
        )

    results_summary_csv = output_dir / 'results_summary.csv'
    summary_rows = [
        {
            'metric': 'operational_status',
            'value': operational_status,
            'interpretation': 'Burden model status from support and numerical-output gates',
        },
        {
            'metric': 'stage_index_mae',
            'value': overall.get('stage_index_mae'),
            'interpretation': 'Primary absolute error on 0-100 ordinal stage index',
        },
        {
            'metric': 'prediction_set_coverage',
            'value': overall.get('prediction_set_coverage'),
            'interpretation': 'Empirical score prediction-set coverage',
        },
        {
            'metric': 'nominal_prediction_set_coverage',
            'value': nominal_coverage,
            'interpretation': 'Nominal coverage target for score prediction sets',
        },
        {
            'metric': 'coverage_gate_passed',
            'value': coverage_gate_passed,
            'interpretation': 'Whether empirical prediction-set coverage met the nominal target',
        },
        {
            'metric': 'burden_interval_empirical_coverage',
            'value': burden_interval_coverage,
            'interpretation': 'Empirical coverage for burden interval bounds',
        },
        {
            'metric': 'ordinal_accuracy',
            'value': ordinal_overall.get('accuracy'),
            'interpretation': 'Ordinal comparator exact class accuracy',
        },
        {
            'metric': 'best_image_precision_candidate',
            'value': best_image_candidate.get('candidate_id', ''),
            'interpretation': (
                f'Stage-index MAE {_format_optional_float(best_image_candidate.get("stage_index_mae"))}'
            ),
        },
        {
            'metric': 'best_subject_precision_candidate',
            'value': best_subject_candidate.get('candidate_id', ''),
            'interpretation': (
                f'Stage-index MAE {_format_optional_float(best_subject_candidate.get("stage_index_mae"))}'
            ),
        },
        {
            'metric': 'best_image_morphology_candidate',
            'value': morphology_best_image.get('candidate_id', ''),
            'interpretation': (
                f'Stage-index MAE {_format_optional_float(morphology_best_image.get("stage_index_mae"))}'
            ),
        },
        {
            'metric': 'best_subject_morphology_candidate',
            'value': morphology_best_subject.get('candidate_id', ''),
            'interpretation': (
                f'Stage-index MAE {_format_optional_float(morphology_best_subject.get("stage_index_mae"))}'
            ),
        },
        {
            'metric': 'learned_roi_readme_docs_ready',
            'value': learned_roi_ready,
            'interpretation': f'Selected learned ROI track: {learned_roi_track or "none"}',
        },
        {
            'metric': 'learned_roi_blockers',
            'value': ' | '.join(map(str, learned_roi_blockers)),
            'interpretation': 'Explicit learned ROI readiness blockers',
        },
        {
            'metric': 'source_aware_status',
            'value': source_aware_status,
            'interpretation': 'Source-aware estimator verdict status',
        },
        {
            'metric': 'source_aware_upstream_roi_adequacy',
            'value': source_aware_upstream.get('status', ''),
            'interpretation': 'Upstream MR TIFF-to-ROI adequacy status',
        },
        {
            'metric': 'source_aware_hard_blockers',
            'value': ' | '.join(map(str, source_aware_hard_blockers)),
            'interpretation': 'Hard blockers for source-aware estimator claims',
        },
        {
            'metric': 'source_aware_scope_limiters',
            'value': ' | '.join(map(str, source_aware_scope_limiters)),
            'interpretation': 'Scope limiters for source-aware estimator claims',
        },
        {
            'metric': 'source_aware_image_reportable',
            'value': source_aware_reportable.get('image_level', ''),
            'interpretation': 'Whether image-level source-aware estimates are reportable within the current claim boundary',
        },
        {
            'metric': 'source_aware_subject_reportable',
            'value': source_aware_reportable.get('subject_level', ''),
            'interpretation': 'Whether subject-level source-aware estimates are reportable within the current claim boundary',
        },
        {
            'metric': 'source_aware_aggregate_current_data_reportable',
            'value': source_aware_reportable.get('aggregate_current_data', ''),
            'interpretation': 'Whether aggregate current-data source-aware estimates are reportable within the current claim boundary',
        },
        {
            'metric': 'source_aware_readme_snippet_eligible',
            'value': source_aware_readme_ready,
            'interpretation': 'Whether source-aware estimator results may enter README snippets',
        },
        {
            'metric': 'severe_aware_status',
            'value': severe_aware_status,
            'interpretation': 'Severe-aware ordinal estimator verdict status',
        },
        {
            'metric': 'severe_aware_output_type',
            'value': severe_aware_verdict.get('selected_output_type', ''),
            'interpretation': 'Whether P2 reports scalar burden, severe risk, ordinal set, subject-level aggregate, aggregate summary, or limited evidence',
        },
        {
            'metric': 'severe_aware_selected_threshold',
            'value': severe_aware_verdict.get('selected_severe_threshold', ''),
            'interpretation': 'Primary severe threshold used for P2 false-negative review',
        },
        {
            'metric': 'severe_aware_hard_blockers',
            'value': ' | '.join(map(str, severe_aware_hard_blockers)),
            'interpretation': 'Hard blockers for severe-aware estimator claims',
        },
        {
            'metric': 'severe_aware_scope_limiters',
            'value': ' | '.join(map(str, severe_aware_scope_limiters)),
            'interpretation': 'Scope limiters for severe-aware estimator claims',
        },
        {
            'metric': 'severe_aware_severe_risk_reportable',
            'value': severe_aware_reportable.get('severe_risk', ''),
            'interpretation': 'Whether severe-risk output is reportable within the current-data claim boundary',
        },
        {
            'metric': 'severe_aware_ordinal_set_reportable',
            'value': severe_aware_reportable.get('ordinal_prediction_set', ''),
            'interpretation': 'Whether ordinal prediction-set output is reportable within the current-data claim boundary',
        },
        {
            'metric': 'severe_aware_scalar_burden_reportable',
            'value': severe_aware_reportable.get('scalar_burden', ''),
            'interpretation': 'Whether P2 supports scalar burden reliability for severe endotheliosis',
        },
        {
            'metric': 'severe_aware_subject_level_reportable',
            'value': severe_aware_reportable.get('subject_level', ''),
            'interpretation': 'Whether subject-level severe-aware aggregation is reportable within the current-data claim boundary',
        },
        {
            'metric': 'severe_aware_aggregate_current_data_reportable',
            'value': severe_aware_reportable.get('aggregate_current_data', ''),
            'interpretation': 'Whether aggregate current-data severe-aware summaries are reportable',
        },
        {
            'metric': 'severe_aware_testing_availability',
            'value': severe_aware_verdict.get('testing_status', ''),
            'interpretation': 'Independent testing availability for severe-aware estimator',
        },
        {
            'metric': 'severe_aware_readme_snippet_eligible',
            'value': severe_aware_readme_ready,
            'interpretation': 'Whether severe-aware estimator results may enter README snippets',
        },
        {
            'metric': 'p3_endotheliosis_grade_model_status',
            'value': grade_model_status,
            'interpretation': 'P3 final product verdict status',
        },
        {
            'metric': 'p3_endotheliosis_grade_model_selected_candidate',
            'value': grade_model_verdict.get('selected_candidate_id', ''),
            'interpretation': 'P3 selected grade-model candidate, if any',
        },
        {
            'metric': 'p3_endotheliosis_grade_model_readme_ready',
            'value': grade_model_readme_ready,
            'interpretation': 'Whether P3 permits README-facing MR TIFF deployment language',
        },
        {
            'metric': 'p3_endotheliosis_grade_model_hard_blockers',
            'value': ' | '.join(map(str, grade_model_hard_blockers)),
            'interpretation': 'Hard blockers for P3 grade-model deployment claims',
        },
    ]
    if not source_aware_metrics.empty:
        for _, row in source_aware_metrics.iterrows():
            if str(row.get('candidate_id', '')) != str(
                source_aware_verdict.get('selected_image_candidate', '')
            ):
                continue
            summary_rows.append(
                {
                    'metric': f'source_aware_{row.get("split_label")}_stage_index_mae',
                    'value': row.get('stage_index_mae'),
                    'interpretation': f'Source-aware selected image candidate {row.get("candidate_id")} {row.get("split_label")}',
                }
            )
    if not severe_aware_metrics.empty:
        selected_severe_candidate = str(
            severe_aware_verdict.get('selected_image_candidate', '')
        )
        for _, row in severe_aware_metrics.iterrows():
            if str(row.get('candidate_id', '')) != selected_severe_candidate:
                continue
            summary_rows.append(
                {
                    'metric': f'severe_aware_{row.get("split_label")}_stage_index_mae',
                    'value': row.get('stage_index_mae'),
                    'interpretation': f'Severe-aware selected image candidate {row.get("candidate_id")} {row.get("split_label")}',
                }
            )
            summary_rows.append(
                {
                    'metric': f'severe_aware_{row.get("split_label")}_severe_false_negative_count',
                    'value': row.get('severe_false_negative_count'),
                    'interpretation': f'Severe false-negative count for selected candidate {row.get("candidate_id")} {row.get("split_label")}',
                }
            )
    if not grade_model_metrics.empty:
        selected_grade_candidate = str(
            grade_model_verdict.get('selected_candidate_id', '')
        )
        for _, row in grade_model_metrics.iterrows():
            if str(row.get('candidate_id', '')) != selected_grade_candidate:
                continue
            summary_rows.append(
                {
                    'metric': 'p3_selected_candidate_metric_label',
                    'value': row.get('metric_label'),
                    'interpretation': 'P3 selected candidate metrics are grouped out-of-fold development estimates',
                }
            )
            for metric_name in [
                'recall',
                'precision',
                'average_precision',
                'balanced_accuracy',
                'severe_band_recall',
            ]:
                if metric_name in row and pd.notna(row.get(metric_name)):
                    summary_rows.append(
                        {
                            'metric': f'p3_selected_candidate_{metric_name}',
                            'value': row.get(metric_name),
                            'interpretation': f'P3 selected candidate {metric_name}',
                        }
                    )
    pd.DataFrame(summary_rows).to_csv(results_summary_csv, index=False)

    results_summary_md = output_dir / 'results_summary.md'
    results_summary_text = f"""# Endotheliosis Quantification Results

## Verdict

- Operational status: `{operational_status}`
- README/docs-ready: `{docs_ready}`
- Claim boundary: predictive ordinal stage-burden index from image-level grades; not pixel-level percent endotheliosis and not a causal treatment effect.

## Primary Burden Metrics

- N examples: `{burden_metrics.get('n_examples')}`
- Subjects: `{burden_metrics.get('n_subject_groups')}`
- Stage-index MAE: `{_format_optional_float(overall.get('stage_index_mae'))}`
- Grade-scale MAE: `{_format_optional_float(overall.get('grade_scale_mae'))}`
- Prediction-set coverage: `{_format_optional_float(overall.get('prediction_set_coverage'))}`
- Nominal prediction-set coverage target: `{_format_optional_float(nominal_coverage)}`
- Coverage gate passed: `{coverage_gate_passed}`
- Burden interval empirical coverage: `{_format_optional_float(burden_interval_coverage)}`
- Average prediction-set size: `{_format_optional_float(overall.get('average_prediction_set_size'))}`
- Support gate: `{support_status}`
- Numerical stability: `{numerical_status}`
- Cohort composition notes: `{', '.join(map(str, burden_metrics.get('cohort_composition_notes', []))) or 'none'}`
- Backend warning messages: `{', '.join(map(str, burden_metrics.get('backend_warning_messages', []))) or 'none'}`

## Comparator Metrics

- Direct stage-index regression MAE: `{_format_optional_float(burden_metrics.get('direct_regression_comparator', {}).get('stage_index_mae'))}`
- Ordinal exact accuracy: `{_format_optional_float(ordinal_overall.get('accuracy'))}`
- Ordinal MAE: `{_format_optional_float(ordinal_overall.get('mae'))}`

## Precision Candidate Screen

- Best image-level candidate: `{best_image_candidate.get('candidate_id', '')}` with stage-index MAE `{_format_optional_float(best_image_candidate.get('stage_index_mae'))}`.
- Best subject-level candidate: `{best_subject_candidate.get('candidate_id', '')}` with stage-index MAE `{_format_optional_float(best_subject_candidate.get('stage_index_mae'))}`.
- Recommendation: {precision_recommendation or 'No precision candidate recommendation was generated.'}

## Morphology Feature Screen

- Feature rows: `{morphology_diagnostics.get('row_count', '')}`
- Feature count: `{morphology_diagnostics.get('feature_count', '')}`
- Best image-level morphology candidate: `{morphology_best_image.get('candidate_id', '')}` with stage-index MAE `{_format_optional_float(morphology_best_image.get('stage_index_mae'))}`.
- Best subject-level morphology candidate: `{morphology_best_subject.get('candidate_id', '')}` with stage-index MAE `{_format_optional_float(morphology_best_subject.get('stage_index_mae'))}`.
- Feature review: `../burden_model/primary_burden_index/evidence/morphology_feature_review/feature_review.html`

## Learned ROI Screen

- README/docs-ready: `{learned_roi_ready}`
- Ready track: `{learned_roi_track or 'none'}`
- Per-image readiness: `{learned_roi_image_readiness.get('status', '')}`
- Subject/cohort readiness: `{learned_roi_subject_readiness.get('status', '')}`
- Blockers: `{', '.join(map(str, learned_roi_blockers)) or 'none'}`
- Evidence review: `../burden_model/learned_roi/evidence/learned_roi_review.html`

## Source-Aware Estimator

- Status: `{source_aware_status}`
- Upstream ROI adequacy: `{source_aware_upstream.get('status', '')}`
- Selected image candidate: `{source_aware_verdict.get('selected_image_candidate', '')}`
- Selected subject candidate: `{source_aware_verdict.get('selected_subject_candidate', '')}`
- Reportable scopes: `{source_aware_reportable}`
- Hard blockers: `{', '.join(map(str, source_aware_hard_blockers)) or 'none'}`
- Scope limiters: `{', '.join(map(str, source_aware_scope_limiters)) or 'none'}`
- Metrics by split: `../burden_model/source_aware_estimator/summary/metrics_by_split.csv`
- Index: `../burden_model/source_aware_estimator/INDEX.md`

## Severe-Aware Ordinal Estimator

- Status: `{severe_aware_status}`
- Selected image candidate: `{severe_aware_verdict.get('selected_image_candidate', '')}`
- Selected subject candidate: `{severe_aware_verdict.get('selected_subject_candidate', '')}`
- Selected severe threshold: `{severe_aware_verdict.get('selected_severe_threshold', '')}`
- Selected output type: `{severe_aware_verdict.get('selected_output_type', '')}`
- Reportable scopes: `{severe_aware_reportable}`
- Hard blockers: `{', '.join(map(str, severe_aware_hard_blockers)) or 'none'}`
- Scope limiters: `{', '.join(map(str, severe_aware_scope_limiters)) or 'none'}`
- Metrics by split: `../burden_model/severe_aware_ordinal_estimator/summary/metrics_by_split.csv`
- Severe false-negative review: `../burden_model/severe_aware_ordinal_estimator/evidence/severe_false_negative_review.html`
- Index: `../burden_model/severe_aware_ordinal_estimator/INDEX.md`

## P3 Endotheliosis Grade Model

- Status: `{grade_model_status}`
- Selected candidate: `{grade_model_verdict.get('selected_candidate_id', '')}`
- Selected output type: `{grade_model_verdict.get('selected_output_type', '')}`
- Quantification gate passed: `{grade_model_verdict.get('quantification_gate_passed', '')}`
- Severe safety gate passed: `{grade_model_verdict.get('severe_safety_gate_passed', '')}`
- MR TIFF deployment gate passed: `{grade_model_verdict.get('mr_tiff_deployment_gate_passed', '')}`
- README-facing deployment allowed: `{grade_model_readme_ready}`
- Hard blockers: `{', '.join(map(str, grade_model_hard_blockers)) or 'none'}`
- Index: `../burden_model/endotheliosis_grade_model/INDEX.md`
- Final verdict: `../burden_model/endotheliosis_grade_model/summary/final_product_verdict.json`

## Documentation Recommendation

Use these results in README/docs only when `README/docs-ready` is `True`. The generated snippet is written every run, but it is not approval for reuse when the readiness flag is false.
"""
    results_summary_md.write_text(results_summary_text, encoding='utf-8')

    readme_snippet_path = output_dir / 'readme_results_snippet.md'
    snippet = f"""### Current Quantification Result

The current full-cohort quantification run reports an endotheliosis burden index as a predictive ordinal stage-burden score, not a pixel-level percent. Operational status: `{operational_status}`. README/docs-ready: `{docs_ready}`. Stage-index MAE: `{_format_optional_float(overall.get('stage_index_mae'))}`; prediction-set coverage: `{_format_optional_float(overall.get('prediction_set_coverage'))}` versus nominal target `{_format_optional_float(nominal_coverage)}`. Cohort composition notes: `{', '.join(map(str, burden_metrics.get('cohort_composition_notes', []))) or 'none'}`. Add a link to the published quantification review artifact before sharing.
"""
    if learned_roi_ready:
        snippet += (
            '\nLearned ROI result: '
            f'`{learned_roi_track}` track is README/docs-ready under the learned ROI gates. '
            'This remains a predictive grade-equivalent burden estimate, not tissue percent or mechanism.\n'
        )
    if source_aware_readme_ready:
        snippet += (
            '\nSource-aware estimator result: eligible under the current-data scoped verdict. '
            'This remains a predictive grade-equivalent burden estimate, not external validation.\n'
        )
    if severe_aware_readme_ready:
        snippet += (
            '\nSevere-aware estimator result: eligible under the severe-aware scoped verdict. '
            f'Output type: `{severe_aware_verdict.get("selected_output_type", "")}`. '
            'This remains predictive grade-equivalent, severe-risk, or ordinal-set evidence for current scored MR TIFF/ROI data, not external validation or tissue percent.\n'
        )
    if grade_model_readme_ready:
        snippet += (
            '\nP3 endotheliosis grade-model result: README-facing MR TIFF deployment is allowed '
            f'under verdict `{grade_model_status}`. '
            'This remains current-data and source-sensitive, not external validation.\n'
        )
    readme_snippet_path.write_text(snippet, encoding='utf-8')

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Endotheliosis Quantification Review</title>
      <style>
        body {{ font-family: "Helvetica Neue", Arial, sans-serif; margin: 2rem auto; max-width: 1280px; color: #1f2933; background: #f7fafc; }}
        h1, h2, h3 {{ color: #102a43; }}
        .note {{ background: #fff7e6; border-left: 4px solid #d9822b; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
        .status {{ background: white; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }}
        .summary-grid, .image-grid, .detail-grid {{ display: grid; gap: 1rem; }}
        .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 1rem; }}
        .summary-grid div {{ background: white; border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }}
        .summary-grid strong {{ display: block; font-size: 0.85rem; color: #486581; margin-bottom: 0.25rem; }}
        .summary-grid span {{ font-size: 1.1rem; font-weight: 600; }}
        .example-card {{ background: white; padding: 1.25rem; border-radius: 10px; margin-bottom: 1.25rem; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08); }}
        .image-grid {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 1rem; }}
        .detail-grid {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
        img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #d9e2ec; background: #f0f4f8; }}
        figure {{ margin: 0; }}
        figcaption {{ font-size: 0.85rem; color: #52606d; margin-top: 0.4rem; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 1rem; }}
        th, td {{ text-align: left; padding: 0.45rem 0.4rem; border-bottom: 1px solid #e5e7eb; font-size: 0.92rem; }}
        .bucket {{ font-size: 0.85rem; color: #486581; }}
      </style>
    </head>
    <body>
      <h1>Endotheliosis Quantification Review</h1>
      <h2>Endotheliosis burden index (0-100)</h2>
      <section class="status">
        <h2>Operational Verdict</h2>
        <p><strong>Status:</strong> {escape(operational_status)}</p>
        <p><strong>README/docs-ready:</strong> {docs_ready}</p>
        <p><strong>Support gate:</strong> {escape(support_status)}</p>
        <p><strong>Numerical status:</strong> {escape(numerical_status)}</p>
        <p><strong>Prediction-set coverage:</strong> {_format_optional_float(empirical_coverage)} versus nominal {_format_optional_float(nominal_coverage)}; gate passed: {coverage_gate_passed}</p>
        <p><strong>Cohort composition notes:</strong> {escape(', '.join(map(str, burden_metrics.get('cohort_composition_notes', []))) or 'none')}</p>
        <p><strong>Grouping key:</strong> {escape(str(grouping_audit.get('grouping_key', 'unknown')))} ({escape(str(grouping_audit.get('grouping_status', 'unknown')))})</p>
      </section>
      <div class="note">The burden output is a predictive ordinal stage-burden index from manually scored image-level grades. It is not pixel-level percent tissue involvement, not externally validated, and not a causal treatment-effect estimate.</div>
      <section class="summary-grid">
        <div><strong>Examples</strong><span>{burden_metrics.get('n_examples')}</span></div>
        <div><strong>Subjects</strong><span>{burden_metrics.get('n_subject_groups')}</span></div>
        <div><strong>Stage-index MAE</strong><span>{_format_optional_float(overall.get('stage_index_mae'))}</span></div>
        <div><strong>Grade-scale MAE</strong><span>{_format_optional_float(overall.get('grade_scale_mae'))}</span></div>
        <div><strong>Prediction-set coverage</strong><span>{_format_optional_float(overall.get('prediction_set_coverage'))}</span></div>
        <div><strong>Average set size</strong><span>{_format_optional_float(overall.get('average_prediction_set_size'))}</span></div>
        <div><strong>Ordinal accuracy</strong><span>{_format_optional_float(ordinal_overall.get('accuracy'))}</span></div>
        <div><strong>Score distribution</strong><span>{escape(score_distribution)}</span></div>
      </section>
      <h2>Comparator Summaries</h2>
      <table><thead><tr><th>Model</th><th>Metric</th><th>Value</th><th>Role</th></tr></thead><tbody>
        <tr><td>Burden cumulative threshold model</td><td>Stage-index MAE</td><td>{_format_optional_float(overall.get('stage_index_mae'))}</td><td>Primary candidate</td></tr>
        <tr><td>Direct stage-index regression</td><td>Stage-index MAE</td><td>{_format_optional_float(burden_metrics.get('direct_regression_comparator', {}).get('stage_index_mae'))}</td><td>Comparator</td></tr>
        <tr><td>Ordinal/multiclass comparator</td><td>Exact accuracy</td><td>{_format_optional_float(ordinal_overall.get('accuracy'))}</td><td>Comparator</td></tr>
        <tr><td>Ordinal/multiclass comparator</td><td>Grade-scale MAE</td><td>{_format_optional_float(ordinal_overall.get('mae'))}</td><td>Comparator</td></tr>
      </tbody></table>
      <h2>Precision Candidate Screen</h2>
      <p>{escape(str(precision_recommendation))}</p>
      <table><thead><tr><th>Candidate</th><th>Target level</th><th>Feature set</th><th>Validation</th><th>Stage-index MAE</th><th>Status</th></tr></thead><tbody>{''.join(signal_rows)}</tbody></table>
      <h2>Morphology Feature Screen</h2>
      <p>The morphology screen is exploratory until the operator review confirms the feature detections. Feature rows: {escape(str(morphology_diagnostics.get('row_count', '')))}; feature count: {escape(str(morphology_diagnostics.get('feature_count', '')))}.</p>
      <p><a href="../burden_model/primary_burden_index/evidence/morphology_feature_review/feature_review.html">Open morphology feature review</a></p>
      <table><thead><tr><th>Candidate</th><th>Target level</th><th>Feature set</th><th>Validation</th><th>Stage-index MAE</th><th>Status</th></tr></thead><tbody>{''.join(morphology_rows)}</tbody></table>
      <h2>Learned ROI Quantification</h2>
      <p><strong>README/docs-ready:</strong> {learned_roi_ready}; <strong>selected track:</strong> {escape(learned_roi_track or 'none')}.</p>
      <p><strong>Per-image readiness:</strong> {escape(str(learned_roi_image_readiness.get('status', '')))}; <strong>subject/cohort readiness:</strong> {escape(str(learned_roi_subject_readiness.get('status', '')))}.</p>
      <p><strong>Blockers:</strong> {escape(', '.join(map(str, learned_roi_blockers)) or 'none')}</p>
      <p><strong>Cohort diagnostic status:</strong> {escape(str(learned_roi_summary.get('cohort_diagnostics_status', '')))}</p>
      <p><a href="../burden_model/learned_roi/evidence/learned_roi_review.html">Open learned ROI evidence review</a></p>
      <table><thead><tr><th>Candidate</th><th>Target level</th><th>Feature set</th><th>Validation</th><th>Stage-index MAE</th><th>Grade-scale MAE</th><th>Status</th></tr></thead><tbody>{''.join(learned_roi_rows)}</tbody></table>
      <h2>Source-Aware Estimator</h2>
      <p><strong>Status:</strong> {escape(source_aware_status)}; <strong>upstream ROI adequacy:</strong> {escape(str(source_aware_upstream.get('status', '')))}</p>
      <p><strong>Selected image candidate:</strong> {escape(str(source_aware_verdict.get('selected_image_candidate', '')))}; <strong>selected subject candidate:</strong> {escape(str(source_aware_verdict.get('selected_subject_candidate', '')))}</p>
      <p><strong>Reportable scopes:</strong> {escape(str(source_aware_reportable))}</p>
      <p><strong>Hard blockers:</strong> {escape(', '.join(map(str, source_aware_hard_blockers)) or 'none')}</p>
      <p><strong>Scope limiters:</strong> {escape(', '.join(map(str, source_aware_scope_limiters)) or 'none')}</p>
      <p><a href="../burden_model/source_aware_estimator/INDEX.md">Open source-aware estimator index</a> | <a href="../burden_model/source_aware_estimator/summary/metrics_by_split.csv">Metrics by split</a></p>
      <ul>{''.join(source_aware_figure_links) or '<li>No source-aware figures available.</li>'}</ul>
      <h2>Severe-Aware Ordinal Estimator</h2>
      <p><strong>Status:</strong> {escape(severe_aware_status)}; <strong>selected output type:</strong> {escape(str(severe_aware_verdict.get('selected_output_type', '')))}</p>
      <p><strong>Selected image candidate:</strong> {escape(str(severe_aware_verdict.get('selected_image_candidate', '')))}; <strong>selected subject candidate:</strong> {escape(str(severe_aware_verdict.get('selected_subject_candidate', '')))}</p>
      <p><strong>Selected severe threshold:</strong> {escape(str(severe_aware_verdict.get('selected_severe_threshold', '')))}</p>
      <p><strong>Reportable scopes:</strong> {escape(str(severe_aware_reportable))}</p>
      <p><strong>Hard blockers:</strong> {escape(', '.join(map(str, severe_aware_hard_blockers)) or 'none')}</p>
      <p><strong>Scope limiters:</strong> {escape(', '.join(map(str, severe_aware_scope_limiters)) or 'none')}</p>
      <p><a href="../burden_model/severe_aware_ordinal_estimator/INDEX.md">Open severe-aware estimator index</a> | <a href="../burden_model/severe_aware_ordinal_estimator/summary/metrics_by_split.csv">Metrics by split</a> | <a href="../burden_model/severe_aware_ordinal_estimator/summary/severe_threshold_metrics.csv">Severe threshold metrics</a> | <a href="../burden_model/severe_aware_ordinal_estimator/evidence/severe_false_negative_review.html">Severe false-negative review</a></p>
      <ul>{''.join(severe_aware_figure_links) or '<li>No severe-aware figures available.</li>'}</ul>
      <h2>P3 Endotheliosis Grade Model</h2>
      <p><strong>Status:</strong> {escape(grade_model_status)}; <strong>selected output type:</strong> {escape(str(grade_model_verdict.get('selected_output_type', '')))}</p>
      <p><strong>Selected candidate:</strong> {escape(str(grade_model_verdict.get('selected_candidate_id', '')))}</p>
      <p><strong>Quantification gate passed:</strong> {escape(str(grade_model_verdict.get('quantification_gate_passed', '')))}; <strong>severe safety gate passed:</strong> {escape(str(grade_model_verdict.get('severe_safety_gate_passed', '')))}; <strong>MR TIFF deployment gate passed:</strong> {escape(str(grade_model_verdict.get('mr_tiff_deployment_gate_passed', '')))}</p>
      <p><strong>README-facing deployment allowed:</strong> {grade_model_readme_ready}</p>
      <p><strong>Hard blockers:</strong> {escape(', '.join(map(str, grade_model_hard_blockers)) or 'none')}</p>
      <p><a href="../burden_model/endotheliosis_grade_model/INDEX.md">Open P3 grade-model index</a> | <a href="../burden_model/endotheliosis_grade_model/summary/final_product_verdict.json">Final verdict JSON</a> | <a href="../burden_model/endotheliosis_grade_model/summary/candidate_coverage_matrix.csv">Candidate coverage</a></p>
      <h2>Artifact Links</h2>
      <p>Primary artifacts: <code>burden_model/primary_burden_index/model/burden_predictions.csv</code> for held-out grouped validation, <code>burden_model/primary_burden_index/model/final_model_predictions.csv</code> for the final full-cohort fitted model, <code>burden_model/primary_burden_index/model/burden_metrics.json</code>, <code>burden_model/primary_burden_index/calibration/uncertainty_calibration.json</code>, <code>burden_model/primary_burden_index/evidence/nearest_examples.csv</code>, <code>burden_model/primary_burden_index/candidates/precision_candidate_summary.json</code>, <code>burden_model/primary_burden_index/candidates/morphology_candidate_summary.json</code>, <code>burden_model/primary_burden_index/feature_sets/morphology_features.csv</code>, <code>burden_model/learned_roi/summary/estimator_verdict.json</code>, <code>burden_model/learned_roi/candidates/learned_roi_candidate_summary.json</code>, <code>burden_model/learned_roi/diagnostics/cohort_confounding_diagnostics.json</code>, <code>burden_model/endotheliosis_grade_model/INDEX.md</code>, <code>ordinal_model/ordinal_metrics.json</code>, and <code>ordinal_model/ordinal_predictions.csv</code>.</p>
      <h2>Final Full-Cohort Summaries</h2>
      <table><thead><tr><th>Cohort</th><th>Rows</th><th>Subjects</th><th>Subject-weighted burden</th><th>Stage-index MAE</th></tr></thead><tbody>{''.join(cohort_table_rows)}</tbody></table>
      <h2>Threshold Support</h2>
      <table><thead><tr><th>Stratum</th><th>Threshold</th><th>Positive groups</th><th>Negative groups</th><th>Status</th></tr></thead><tbody>{''.join(support_rows)}</tbody></table>
      <h2>Sample Summary Intervals</h2>
      <table><thead><tr><th>Stratum</th><th>Estimand</th><th>Clusters</th><th>Mean burden</th><th>95% CI</th><th>Status</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table>
      <h2>Reviewer Examples</h2>
      {''.join(cards)}
    </body>
    </html>
    """
    html_path = output_dir / 'quantification_review.html'
    html_path.write_text(html, encoding='utf-8')
    return {
        'quantification_review_html': html_path,
        'quantification_review_examples': review_examples_path,
        'quantification_results_summary_md': results_summary_md,
        'quantification_results_summary_csv': results_summary_csv,
        'quantification_readme_snippet': readme_snippet_path,
        'quantification_review_assets_dir': assets_dir,
    }


def _write_burden_model_index(
    burden_output_dir: Path, artifacts: dict[str, Path]
) -> Path:
    index_path = Path(burden_output_dir) / 'INDEX.md'
    rows = [
        (
            'primary_burden_index',
            'Primary grouped cumulative-threshold burden-index evaluator',
            'primary_burden_index/INDEX.md',
        ),
        ('learned_roi', 'Capped learned-ROI candidate screen', 'learned_roi/INDEX.md'),
        (
            'source_aware_estimator',
            'Source-aware estimator verdict bundle',
            'source_aware_estimator/INDEX.md',
        ),
        (
            'severe_aware_ordinal_estimator',
            'Severe-aware ordinal estimator verdict bundle',
            'severe_aware_ordinal_estimator/INDEX.md',
        ),
        (
            'endotheliosis_grade_model',
            'P3 grade-model selector and final product verdict',
            'endotheliosis_grade_model/INDEX.md',
        ),
    ]
    table_rows = '\n'.join(
        f'| `{slug}` | {description} | `{first_read}` |'
        for slug, description, first_read in rows
    )
    generated = sorted(
        key
        for key, value in artifacts.items()
        if isinstance(value, Path) and value.exists()
    )
    text = f"""# Burden Model Artifacts

This directory is organized by model or estimator subtree. Open the subtree index first, then follow its summary or typed artifact folders.

| Subtree | Role | First read |
| --- | --- | --- |
{table_rows}

`summary/` means first-read verdict material inside an estimator subtree. `summaries/` means aggregate tables such as cohort summaries or interval summaries.

Generated artifact keys: {', '.join(generated)}
"""
    index_path.write_text(text, encoding='utf-8')
    return index_path


def _evaluate_ordinal_embedding_table(
    embedding_df: pd.DataFrame, output_dir: Path, n_splits: int = 3
) -> Dict[str, Path]:
    """Train and evaluate the ordinal comparator on frozen embeddings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_columns = [
        column for column in embedding_df.columns if column.startswith('embedding_')
    ]
    if not embedding_columns:
        raise ValueError('Embedding table does not contain embedding columns')

    work_df = embedding_df.copy().reset_index(drop=True)
    work_df['score_class'] = work_df['score'].map(_score_to_class_index)
    group_column, groups_series, grouping_audit = derive_biological_grouping(work_df)
    work_df[group_column] = groups_series.reset_index(drop=True)

    x = work_df[embedding_columns].to_numpy(dtype=np.float64)
    y = work_df['score_class'].to_numpy(dtype=np.int64)
    groups = groups_series.astype(str).to_numpy()
    class_indices = np.arange(len(ALLOWED_SCORE_VALUES), dtype=np.int64)

    unique_groups = np.unique(groups)
    split_count = min(max(2, n_splits), len(unique_groups))
    if split_count < 2:
        raise ContractPreparationError(
            'Need at least two subject groups for grouped evaluation'
        )

    group_kfold = GroupKFold(n_splits=split_count)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    fold_warning_messages: list[dict[str, Any]] = []
    cohort_profile = build_grouped_ordinal_cohort_profile(
        y,
        groups,
        len(embedding_columns),
        classes=class_indices,
        score_values=ALLOWED_SCORE_VALUES,
    )

    for fold_index, (train_idx, test_idx) in enumerate(
        group_kfold.split(x, y, groups=groups), start=1
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        model = CanonicalOrdinalClassifier(classes=class_indices).fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        pred_class = probabilities.argmax(axis=1)
        pred_score = _class_index_to_score(pred_class)
        true_score = _class_index_to_score(y_test)
        expected_score = np.sum(probabilities * ALLOWED_SCORE_VALUES, axis=1)
        sorted_probabilities = np.sort(probabilities, axis=1)
        top_two_margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        entropy = _entropy(probabilities)

        fold_df = (
            work_df.iloc[test_idx]
            .drop(columns=['score_class', *embedding_columns], errors='ignore')
            .copy()
        )
        fold_df['fold'] = fold_index
        fold_df['score_class'] = y[test_idx]
        fold_df['predicted_score'] = pred_score
        fold_df['predicted_class'] = pred_class
        fold_df['expected_score'] = expected_score
        fold_df['top_two_margin'] = top_two_margin
        fold_df['entropy'] = entropy
        fold_df['absolute_error'] = np.abs(
            fold_df['score'].to_numpy(dtype=np.float64) - pred_score.astype(np.float64)
        )
        fold_df['prediction_error'] = pred_score.astype(np.float64) - fold_df[
            'score'
        ].to_numpy(dtype=np.float64)
        for class_index, score_value in enumerate(ALLOWED_SCORE_VALUES):
            fold_df[_score_probability_column_name(float(score_value))] = probabilities[
                :, class_index
            ]
        predictions.append(fold_df)

        fold_metrics.append(
            {
                'fold': fold_index,
                'num_examples': int(len(test_idx)),
                'mae': float(mean_absolute_error(true_score, pred_score)),
                'accuracy': float(accuracy_score(y_test, pred_class)),
                'within_one_bin_accuracy': float(
                    np.mean(np.abs(y_test - pred_class) <= 1)
                ),
                'quadratic_weighted_kappa': float(
                    cohen_kappa_score(y_test, pred_class, weights='quadratic')
                ),
            }
        )
        fold_warning_messages.append(
            {
                'fold': int(fold_index),
                'messages': model.warning_messages_,
                'train_class_counts': model.training_class_counts_,
                'test_class_counts': {
                    str(int(class_index)): int(np.sum(y_test == class_index))
                    for class_index in class_indices
                },
                'train_threshold_positive_counts': {
                    f'>{int(class_index)}': int(np.sum(y_train > class_index))
                    for class_index in class_indices[:-1]
                },
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    probability_columns = [
        _score_probability_column_name(float(score_value))
        for score_value in ALLOWED_SCORE_VALUES
    ]
    probability_output_status = _finite_matrix_status(
        predictions_df[probability_columns].to_numpy(dtype=np.float64)
    )
    predictions_path = output_dir / 'ordinal_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

    if 'score_class' in predictions_df.columns:
        merged_predictions = predictions_df.copy()
    else:
        merged_predictions = predictions_df.merge(
            work_df[['subject_image_id', 'glomerulus_id', 'score_class']],
            on=['subject_image_id', 'glomerulus_id'],
            how='left',
        )
    confusion = confusion_matrix(
        merged_predictions['score_class'],
        merged_predictions['predicted_class'],
        labels=list(range(len(ALLOWED_SCORE_VALUES))),
    )
    confusion_df = pd.DataFrame(
        confusion,
        index=[f'true_{score:g}' for score in ALLOWED_SCORE_VALUES],
        columns=[f'pred_{score:g}' for score in ALLOWED_SCORE_VALUES],
    )
    confusion_path = output_dir / 'ordinal_confusion_matrix.csv'
    confusion_df.to_csv(confusion_path)

    summary = {
        'n_examples': int(len(work_df)),
        'n_subject_groups': int(len(unique_groups)),
        'n_splits': int(split_count),
        'grouping_key': group_column,
        'grouping_audit': grouping_audit,
        'fold_metrics': fold_metrics,
        'overall': {
            'mae': float(
                mean_absolute_error(
                    predictions_df['score'], predictions_df['predicted_score']
                )
            ),
            'accuracy': float(
                accuracy_score(
                    merged_predictions['score_class'],
                    merged_predictions['predicted_class'],
                )
            ),
            'within_one_bin_accuracy': float(
                np.mean(
                    np.abs(
                        merged_predictions['score_class']
                        - merged_predictions['predicted_class']
                    )
                    <= 1
                )
            ),
            'quadratic_weighted_kappa': float(
                cohen_kappa_score(
                    merged_predictions['score_class'],
                    merged_predictions['predicted_class'],
                    weights='quadratic',
                )
            ),
        },
    }

    scaler = StandardScaler().fit(x)
    final_model = CanonicalOrdinalClassifier(classes=class_indices).fit(
        scaler.transform(x), y
    )
    combined_warning_messages = list(
        dict.fromkeys(
            [
                *[
                    message
                    for fold_entry in fold_warning_messages
                    for message in fold_entry['messages']
                ],
                *final_model.warning_messages_,
            ]
        )
    )
    full_target_class_support = all(
        int(count) > 0 for count in cohort_profile['class_counts'].values()
    )
    certification_blockers: list[str] = []
    if not probability_output_status['finite']:
        certification_blockers.append('numerical_instability')
    if not full_target_class_support:
        certification_blockers.append('missing_target_class_support')
    summary['ordinal_model'] = final_model.metadata()
    summary['cohort_profile'] = cohort_profile
    summary['stability'] = {
        'warning_patterns': list(NUMERICAL_INSTABILITY_PATTERNS),
        'fold_warning_messages': fold_warning_messages,
        'final_model_warning_messages': final_model.warning_messages_,
        'zero_unresolved_warning_gate_passed': not combined_warning_messages,
        'probability_output_status': probability_output_status,
        'numerical_stability_status': (
            'nonfinite_output'
            if not probability_output_status['finite']
            else 'backend_warnings_outputs_finite'
            if combined_warning_messages
            else 'ok'
        ),
        'backend_warning_messages': combined_warning_messages,
        'full_target_class_support': full_target_class_support,
        'certification_status': (
            'supported' if not certification_blockers else 'incomplete'
        ),
        'certification_blockers': certification_blockers,
    }
    metrics_path = _save_json(summary, output_dir / 'ordinal_metrics.json')
    model_path = output_dir / 'ordinal_embedding_model.pkl'
    with model_path.open('wb') as handle:
        pickle.dump(
            {
                'allowed_scores': ALLOWED_SCORE_VALUES.tolist(),
                'embedding_columns': embedding_columns,
                'scaler': scaler,
                'model': final_model,
                'model_metadata': final_model.metadata(),
            },
            handle,
        )

    report_artifacts = generate_html_review_report(
        predictions_df=predictions_df,
        metrics_summary=summary,
        output_dir=output_dir / 'review_report',
    )

    return {
        'predictions': predictions_path,
        'ordinal_predictions': predictions_path,
        'confusion_matrix': confusion_path,
        'ordinal_confusion_matrix': confusion_path,
        'metrics': metrics_path,
        'ordinal_metrics': metrics_path,
        'model': model_path,
        'ordinal_model': model_path,
        'review_html': report_artifacts['html'],
        'ordinal_review_html': report_artifacts['html'],
        'review_examples': report_artifacts['selected_examples'],
        'ordinal_review_examples': report_artifacts['selected_examples'],
        'review_assets_dir': report_artifacts['assets_dir'],
        'ordinal_review_assets_dir': report_artifacts['assets_dir'],
    }


def evaluate_embedding_table(
    embedding_df: pd.DataFrame,
    output_dir: Path,
    n_splits: int = 3,
    manifest_root: Path | None = None,
    segmentation_model_path: Path | None = None,
    label_contract_reference: dict[str, Any] | None = None,
) -> Dict[str, Path]:
    """Train burden-index model and retained ordinal comparator."""
    logger = get_logger('eq.quantification.pipeline')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_output = output_dir.parent
    logger.info(
        'Starting quantification model evaluation: rows=%d, ordinal_dir=%s, burden_dir=%s',
        len(embedding_df),
        output_dir,
        parent_output / 'burden_model',
    )
    logger.info('Evaluating ordinal comparator -> %s', output_dir)
    ordinal_artifacts = _evaluate_ordinal_embedding_table(
        embedding_df, output_dir, n_splits=n_splits
    )
    logger.info(
        'Ordinal comparator complete: predictions=%s, metrics=%s',
        ordinal_artifacts['predictions'],
        ordinal_artifacts['metrics'],
    )
    logger.info(
        'Evaluating burden-index and morphology candidates -> %s',
        parent_output / 'burden_model',
    )
    burden_artifacts = evaluate_burden_index_table(
        embedding_df, parent_output / 'burden_model', n_splits=n_splits
    )
    logger.info(
        'Burden evaluation complete: predictions=%s, metrics=%s',
        burden_artifacts['burden_predictions'],
        burden_artifacts['burden_metrics'],
    )
    logger.info(
        'Evaluating learned ROI phase-1 candidates -> %s',
        parent_output / 'burden_model' / 'learned_roi',
    )
    learned_roi_artifacts = evaluate_learned_roi_quantification(
        embedding_df, parent_output / 'burden_model', n_splits=n_splits
    )
    burden_artifacts.update(learned_roi_artifacts)
    logger.info(
        'Learned ROI evaluation complete: summary=%s',
        learned_roi_artifacts['learned_roi_candidate_summary'],
    )
    logger.info(
        'Evaluating source-aware estimator -> %s',
        parent_output / 'burden_model' / 'source_aware_estimator',
    )
    source_aware_artifacts = evaluate_source_aware_endotheliosis_estimator(
        embedding_df, parent_output / 'burden_model', n_splits=n_splits
    )
    burden_artifacts.update(source_aware_artifacts)
    logger.info(
        'Source-aware estimator complete: verdict=%s',
        source_aware_artifacts['source_aware_estimator_verdict'],
    )
    logger.info(
        'Evaluating severe-aware ordinal estimator -> %s',
        parent_output / 'burden_model' / 'severe_aware_ordinal_estimator',
    )
    severe_aware_artifacts = evaluate_severe_aware_ordinal_endotheliosis_estimator(
        embedding_df, parent_output / 'burden_model', n_splits=n_splits
    )
    burden_artifacts.update(severe_aware_artifacts)
    logger.info(
        'Severe-aware ordinal estimator complete: verdict=%s',
        severe_aware_artifacts['severe_aware_estimator_verdict'],
    )
    logger.info(
        'Evaluating P3 endotheliosis grade-model selector -> %s',
        parent_output / 'burden_model' / 'endotheliosis_grade_model',
    )
    grade_model_artifacts = evaluate_endotheliosis_grade_model(
        embedding_df,
        parent_output / 'burden_model',
        n_splits=n_splits,
        change_dir=Path('openspec/changes/p3-functional-severe-ordinal-quantification'),
        manifest_root=manifest_root,
        segmentation_model_path=segmentation_model_path,
        label_contract_reference=label_contract_reference,
    )
    burden_artifacts.update(grade_model_artifacts)
    logger.info(
        'P3 endotheliosis grade-model selector complete: verdict=%s',
        grade_model_artifacts['endotheliosis_grade_model_verdict'],
    )
    burden_model_index = _write_burden_model_index(
        parent_output / 'burden_model', burden_artifacts
    )
    burden_artifacts['burden_model_index'] = burden_model_index
    logger.info('Burden-model first-read index written -> %s', burden_model_index)
    logger.info(
        'Generating combined quantification review -> %s',
        parent_output / 'quantification_review' / 'quantification_review.html',
    )
    review_artifacts = generate_combined_quantification_review(
        ordinal_predictions_path=ordinal_artifacts['predictions'],
        ordinal_metrics_path=ordinal_artifacts['metrics'],
        burden_artifacts=burden_artifacts,
        output_dir=parent_output / 'quantification_review',
    )
    logger.info(
        'Combined quantification review complete: html=%s, summary=%s',
        review_artifacts['quantification_review_html'],
        review_artifacts['quantification_results_summary_md'],
    )
    merged_artifacts: Dict[str, Path] = dict(ordinal_artifacts)
    merged_artifacts.update(burden_artifacts)
    merged_artifacts.update(review_artifacts)
    return merged_artifacts


def run_contract_first_quantification(
    project_dir: Path,
    segmentation_model_path: Path,
    output_dir: Path,
    mapping_file: Optional[Path] = None,
    annotation_source: Optional[str | Path] = None,
    score_source: str = 'auto',
    apply_migration: bool = False,
    stop_after: str = 'model',
    label_overrides_path: Path | None = None,
    input_contract: ResolvedQuantificationInputContract | None = None,
) -> Dict[str, Path]:
    """Prepare the quantification contract and run the embedding-first scorer."""
    logger = get_logger('eq.quantification.pipeline')
    project_dir = Path(project_dir)
    segmentation_model_path = Path(segmentation_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        'Starting contract-first quantification: project_dir=%s, segmentation_model=%s, output_dir=%s, score_source=%s, stop_after=%s',
        project_dir,
        segmentation_model_path,
        output_dir,
        score_source,
        stop_after,
    )

    if input_contract is None:
        input_contract = resolve_quantification_input_contract(
            data_dir=project_dir,
            segmentation_model=segmentation_model_path,
            output_dir=output_dir,
            mapping_file=mapping_file,
            annotation_source=annotation_source,
            score_source=score_source,
            label_overrides_path=label_overrides_path,
        )
    score_source = input_contract.score_source
    annotation_source = input_contract.annotation_source
    mapping_file = input_contract.mapping_file
    label_overrides_path = input_contract.label_overrides_path

    manifest_path = project_dir / 'manifest.csv'
    if manifest_path.exists():
        if score_source == 'labelstudio' or annotation_source is not None:
            raise ContractPreparationError(
                'Manifest roots require manifest-mode scoring; score_source=labelstudio '
                'and annotation_source are raw-project Label Studio options and are not '
                'supported for manifest.csv roots.'
            )
        logger.info(
            'Manifest-backed quantification detected: manifest=%s', manifest_path
        )
        return run_manifest_quantification(
            manifest_root=project_dir,
            segmentation_model_path=segmentation_model_path,
            output_dir=output_dir,
            stop_after=stop_after,
            label_overrides_path=label_overrides_path,
            input_contract=input_contract,
        )

    inventory_path = output_dir / 'raw_inventory.csv'
    inventory_raw_project(project_dir).to_csv(inventory_path, index=False)
    logger.info('Raw project inventory written: %s', inventory_path)
    mapping_template_path = generate_mapping_template(
        project_dir, output_dir / 'legacy_to_canonical_mapping_template.csv'
    )
    logger.info('Mapping template written: %s', mapping_template_path)

    if (
        annotation_source is None
        and score_source in {'auto', 'labelstudio'}
        and not (score_source == 'auto' and manifest_path.exists())
    ):
        annotation_source = discover_label_studio_annotation_source(project_dir)

    use_labelstudio_scores = score_source == 'labelstudio' or (
        score_source == 'auto' and annotation_source is not None
    )

    if use_labelstudio_scores:
        if annotation_source is None:
            raise FileNotFoundError(
                'No Label Studio annotation source was found. Provide --annotation-source or use --score-source spreadsheet.'
            )

        score_outputs = recover_label_studio_score_table(
            project_dir=project_dir,
            annotation_source=annotation_source,
            output_dir=output_dir / 'labelstudio_scores',
        )
        score_table = pd.read_csv(score_outputs['scores'])
        logger.info(
            'Recovered Label Studio scores: rows=%d, scores=%s, summary=%s',
            len(score_table),
            score_outputs['scores'],
            score_outputs['summary'],
        )
        validation_summary = json.loads(
            score_outputs['summary'].read_text(encoding='utf-8')
        )
        if validation_summary.get('join_status_counts', {}).get('ok', 0) == 0:
            raise ContractPreparationError(
                'No scored image/mask pairs joined successfully from the Label Studio export'
            )

        scored_table = build_image_level_scored_example_table(
            project_dir=project_dir,
            score_table=score_table,
            output_dir=output_dir / 'scored_examples',
        )
        label_contract_reference = label_contract_reference_for_scored_table(
            input_contract,
            scored_table,
            base_scored_input_path=score_outputs['scores'],
            annotation_artifact_path=score_outputs['scores'],
        )
        scored_table, label_override_artifacts = _apply_score_label_overrides(
            scored_table,
            label_overrides_path,
            output_dir / 'scored_examples',
            label_contract_reference=label_contract_reference,
        )
        if label_override_artifacts:
            scored_table.to_csv(
                output_dir / 'scored_examples' / 'scored_examples.csv', index=False
            )
        logger.info('Scored examples ready: rows=%d', len(scored_table))
        if stop_after == 'contract':
            return {
                'raw_inventory': inventory_path,
                'mapping_template': mapping_template_path,
                'labelstudio_scores': score_outputs['scores'],
                'labelstudio_summary': score_outputs['summary'],
                'duplicate_annotations': score_outputs['duplicates'],
                'scored_examples': output_dir
                / 'scored_examples'
                / 'scored_examples.csv',
                **label_override_artifacts,
            }

        roi_table = extract_image_level_roi_crops(
            scored_table, output_dir / 'roi_crops'
        )
        logger.info('ROI crops ready: rows=%d', len(roi_table))
        if stop_after == 'roi':
            return {
                'raw_inventory': inventory_path,
                'mapping_template': mapping_template_path,
                'labelstudio_scores': score_outputs['scores'],
                'labelstudio_summary': score_outputs['summary'],
                'duplicate_annotations': score_outputs['duplicates'],
                'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
                **label_override_artifacts,
            }

        embedding_table = extract_embedding_table(
            roi_table=roi_table,
            segmentation_model_path=segmentation_model_path,
            output_dir=output_dir / 'embeddings',
        )
        logger.info(
            'Embedding table ready: rows=%d, columns=%d',
            len(embedding_table),
            len(embedding_table.columns),
        )
        if stop_after == 'embeddings':
            return {
                'raw_inventory': inventory_path,
                'mapping_template': mapping_template_path,
                'labelstudio_scores': score_outputs['scores'],
                'labelstudio_summary': score_outputs['summary'],
                'duplicate_annotations': score_outputs['duplicates'],
                'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
                'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
                **label_override_artifacts,
            }

        model_artifacts = evaluate_embedding_table(
            embedding_table,
            output_dir / 'ordinal_model',
            segmentation_model_path=segmentation_model_path,
            label_contract_reference=label_contract_reference,
        )
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'labelstudio_scores': score_outputs['scores'],
            'labelstudio_summary': score_outputs['summary'],
            'duplicate_annotations': score_outputs['duplicates'],
            'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
            **label_override_artifacts,
            **_burden_first_artifacts(model_artifacts),
        }

    metadata_file = project_dir / 'metadata' / 'subject_metadata.xlsx'
    if not metadata_file.exists():
        logger.error('Metadata file not found: %s', metadata_file)
        raise FileNotFoundError(f'Metadata file not found: {metadata_file}')

    metadata_output_dir = output_dir / 'metadata'
    processor = MetadataProcessor()
    metadata_df = processor.process_glomeruli_scoring_matrix(
        metadata_file, output_path=metadata_output_dir / 'standardized_metadata.csv'
    )
    logger.info(
        'Metadata standardized: rows=%d, output=%s',
        len(metadata_df),
        metadata_output_dir / 'standardized_metadata.csv',
    )

    migration_plan = build_migration_plan(
        project_dir, metadata_df, mapping_file=mapping_file
    )
    migration_plan_path = output_dir / 'contract_migration_plan.csv'
    migration_plan.to_csv(migration_plan_path, index=False)

    if apply_migration:
        migration_plan = apply_migration_plan(migration_plan)
        migration_plan_path = output_dir / 'contract_migration_applied.csv'
        migration_plan.to_csv(migration_plan_path, index=False)

    validation_report = validate_project_contract(
        project_dir, metadata_df, require_canonical=True
    )
    validation_path = save_contract_report(
        validation_report, output_dir / 'canonical_contract_validation.json'
    )
    logger.info(
        'Canonical contract validation status=%s report=%s',
        validation_report['overall_status'],
        validation_path,
    )

    if validation_report['overall_status'] != 'PASS':
        unresolved = migration_plan[
            migration_plan['status'].isin(
                {
                    'invalid_name',
                    'unmapped_legacy',
                    'target_collision',
                    'mapped_id_missing_from_metadata',
                    'canonical_missing_from_metadata',
                }
            )
        ]
        logger.error(
            'Canonical contract validation failed; see %s and %s',
            migration_plan_path,
            validation_path,
        )
        if not unresolved.empty:
            unresolved.to_csv(
                output_dir / 'canonical_contract_unresolved.csv', index=False
            )
        raise ContractPreparationError(
            'Canonical contract validation failed. Review the migration report and unresolved rows before ROI extraction.'
        )

    scored_table = build_scored_example_table(
        project_dir, metadata_df, output_dir / 'scored_examples'
    )
    label_contract_reference = label_contract_reference_for_scored_table(
        input_contract,
        scored_table,
        base_scored_input_path=metadata_output_dir / 'standardized_metadata.csv',
    )
    scored_table, label_override_artifacts = _apply_score_label_overrides(
        scored_table,
        label_overrides_path,
        output_dir / 'scored_examples',
        label_contract_reference=label_contract_reference,
    )
    if label_override_artifacts:
        scored_table.to_csv(
            output_dir / 'scored_examples' / 'scored_examples.csv', index=False
        )
    logger.info('Scored examples ready: rows=%d', len(scored_table))
    if stop_after == 'contract':
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'metadata': metadata_output_dir / 'standardized_metadata.csv',
            'migration_plan': migration_plan_path,
            'validation': validation_path,
            'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
            **label_override_artifacts,
        }

    roi_table = extract_roi_crops(scored_table, output_dir / 'roi_crops')
    logger.info('ROI crops ready: rows=%d', len(roi_table))
    if stop_after == 'roi':
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'metadata': metadata_output_dir / 'standardized_metadata.csv',
            'migration_plan': migration_plan_path,
            'validation': validation_path,
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            **label_override_artifacts,
        }

    embedding_table = extract_embedding_table(
        roi_table=roi_table,
        segmentation_model_path=segmentation_model_path,
        output_dir=output_dir / 'embeddings',
    )
    logger.info(
        'Embedding table ready: rows=%d, columns=%d',
        len(embedding_table),
        len(embedding_table.columns),
    )
    if stop_after == 'embeddings':
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'metadata': metadata_output_dir / 'standardized_metadata.csv',
            'migration_plan': migration_plan_path,
            'validation': validation_path,
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
            **label_override_artifacts,
        }

    model_artifacts = evaluate_embedding_table(
        embedding_table,
        output_dir / 'ordinal_model',
        segmentation_model_path=segmentation_model_path,
        label_contract_reference=label_contract_reference,
    )
    return {
        'raw_inventory': inventory_path,
        'mapping_template': mapping_template_path,
        'metadata': metadata_output_dir / 'standardized_metadata.csv',
        'migration_plan': migration_plan_path,
        'validation': validation_path,
        'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
        'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
        'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
        **label_override_artifacts,
        **_burden_first_artifacts(model_artifacts),
    }


def prepare_quantification_contract(
    raw_project_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    mapping_file: Optional[Path] = None,
    annotation_source: Optional[str | Path] = None,
    score_source: str = 'auto',
    migrate: bool = False,
    dry_run: bool = True,
) -> Dict[str, Path]:
    """Build the contract artifacts needed before embeddings and scoring."""
    raw_project_dir = Path(raw_project_dir)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = output_dir / 'raw_inventory.csv'
    inventory_raw_project(raw_project_dir).to_csv(inventory_path, index=False)
    mapping_template_path = generate_mapping_template(
        raw_project_dir, output_dir / 'legacy_to_canonical_mapping_template.csv'
    )

    if score_source not in {'auto', 'labelstudio', 'spreadsheet'}:
        raise ValueError(f'Unsupported score_source: {score_source}')

    if annotation_source is None and score_source in {'auto', 'labelstudio'}:
        annotation_source = discover_label_studio_annotation_source(raw_project_dir)

    use_labelstudio_scores = score_source == 'labelstudio' or (
        score_source == 'auto' and annotation_source is not None
    )

    if use_labelstudio_scores:
        if annotation_source is None:
            raise FileNotFoundError(
                'No Label Studio annotation source was found. Provide --annotation-source or use --score-source spreadsheet.'
            )

        score_outputs = recover_label_studio_score_table(
            project_dir=raw_project_dir,
            annotation_source=annotation_source,
            output_dir=output_dir / 'labelstudio_scores',
        )
        score_table = pd.read_csv(score_outputs['scores'])
        scored_table = build_image_level_scored_example_table(
            raw_project_dir, score_table, output_dir / 'scored_examples'
        )
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'labelstudio_scores': score_outputs['scores'],
            'labelstudio_summary': score_outputs['summary'],
            'duplicate_annotations': score_outputs['duplicates'],
            'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
        }

    metadata_output_dir = output_dir / 'metadata'
    processor = MetadataProcessor()
    metadata_df = processor.process_glomeruli_scoring_matrix(
        metadata_path, output_path=metadata_output_dir / 'standardized_metadata.csv'
    )

    migration_plan = build_migration_plan(
        raw_project_dir, metadata_df, mapping_file=mapping_file
    )
    migration_plan_path = output_dir / 'contract_migration_plan.csv'
    migration_plan.to_csv(migration_plan_path, index=False)

    if migrate and not dry_run:
        applied_plan = apply_migration_plan(migration_plan)
        migration_plan_path = output_dir / 'contract_migration_applied.csv'
        applied_plan.to_csv(migration_plan_path, index=False)

    validation_report = validate_project_contract(
        raw_project_dir, metadata_df, require_canonical=True
    )
    validation_path = save_contract_report(
        validation_report, output_dir / 'canonical_contract_validation.json'
    )
    scored_table = build_scored_example_table(
        raw_project_dir, metadata_df, output_dir / 'scored_examples'
    )

    outputs: Dict[str, Path] = {
        'raw_inventory': inventory_path,
        'mapping_template': mapping_template_path,
        'metadata': metadata_output_dir / 'standardized_metadata.csv',
        'migration_plan': migration_plan_path,
        'validation': validation_path,
        'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
    }
    unresolved = migration_plan[
        migration_plan['status'].isin(
            {
                'invalid_name',
                'unmapped_legacy',
                'target_collision',
                'mapped_id_missing_from_metadata',
                'canonical_missing_from_metadata',
            }
        )
    ]
    if not unresolved.empty:
        unresolved_path = output_dir / 'canonical_contract_unresolved.csv'
        unresolved.to_csv(unresolved_path, index=False)
        outputs['unresolved'] = unresolved_path
    if not scored_table.empty:
        outputs['scored_examples'] = (
            output_dir / 'scored_examples' / 'scored_examples.csv'
        )
    return outputs


def run_endotheliosis_scoring_pipeline(
    raw_project_dir: Path,
    metadata_path: Path,
    segmentation_model_path: Path,
    output_dir: Path,
    mapping_file: Optional[Path] = None,
    annotation_source: Optional[str | Path] = None,
    score_source: str = 'auto',
    migrate: bool = False,
    dry_run: bool = True,
) -> Dict[str, Path]:
    """Run the full contract-first scoring flow using the existing quantification pipeline."""
    output_dir = Path(output_dir)
    contract_outputs = prepare_quantification_contract(
        raw_project_dir=raw_project_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        mapping_file=mapping_file,
        annotation_source=annotation_source,
        score_source=score_source,
        migrate=migrate,
        dry_run=dry_run,
    )

    if 'validation' in contract_outputs:
        validation = json.loads(
            Path(contract_outputs['validation']).read_text(encoding='utf-8')
        )
        if validation.get('overall_status') != 'PASS':
            raise ContractPreparationError(
                'Canonical contract validation failed. Review the migration plan, validation report, and unresolved rows first.'
            )
    elif 'labelstudio_summary' in contract_outputs:
        summary = json.loads(
            Path(contract_outputs['labelstudio_summary']).read_text(encoding='utf-8')
        )
        if summary.get('join_status_counts', {}).get('ok', 0) == 0:
            raise ContractPreparationError(
                'Label Studio score recovery did not produce any joined scored image/mask pairs.'
            )
    else:
        raise ContractPreparationError(
            'Quantification contract preparation did not produce a recognizable validation artifact.'
        )

    return run_contract_first_quantification(
        project_dir=Path(raw_project_dir),
        segmentation_model_path=Path(segmentation_model_path),
        output_dir=output_dir,
        mapping_file=mapping_file,
        annotation_source=annotation_source,
        score_source=score_source,
        apply_migration=migrate and not dry_run,
        stop_after='model',
    )
