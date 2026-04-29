"""P3 endotheliosis grade-model selector.

This module consumes existing burden-model artifacts and current ROI rows, then
writes a contained grade-model selector subtree. It treats current data metrics
as grouped out-of-fold development estimates only.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from eq.data_management.model_loading import load_model_safely
from eq.evaluation.quantification_metrics import calculate_quantification_metrics
from eq.inference.prediction_core import create_prediction_core
from eq.quantification.burden import ALLOWED_SCORE_VALUES, validate_score_values
from eq.quantification.modeling_contracts import (
    GROUPED_DEVELOPMENT_METRIC_LABEL,
    build_artifact_manifest,
    capture_fit_warnings,
    choose_recall_threshold,
    save_json,
    to_finite_numeric_matrix,
)
from eq.quantification.morphology_features import (
    MORPHOLOGY_FEATURE_COLUMNS,
    _extract_feature_row,
)

GRADE_MODEL_ROOT_NAME = 'endotheliosis_grade_model'
PRIMARY_SEVERE_THRESHOLD = 2.0
THREE_BAND_LABELS = ('none_low', 'mild_mod', 'severe')
FOUR_BAND_LABELS = ('0', '0.5', '1_1.5', '2_3')
FINAL_STATUS_VALUES = {
    'readme_facing_deployable_mr_tiff_grade_model',
    'readme_facing_deployable_mr_tiff_severe_triage',
    'model_ready_pending_mr_tiff_deployment_smoke',
    'diagnostic_only_current_data_model',
    'current_data_insufficient',
}
IDENTITY_COLUMNS = [
    'subject_id',
    'subject_image_id',
    'glomerulus_id',
    'cohort_id',
    'score',
    'roi_image_path',
    'roi_mask_path',
]
OPTIONAL_IDENTITY_COLUMNS = ['sample_id', 'image_id', 'raw_image_path', 'raw_mask_path']
ROI_QC_COLUMNS = [
    'roi_area',
    'roi_fill_fraction',
    'roi_mean_intensity',
    'roi_openness_score',
    'roi_component_count',
    'roi_union_bbox_width',
    'roi_union_bbox_height',
    'roi_largest_component_area_fraction',
]
FIRST_CLASS_FAMILIES = {
    'three_band_ordinal_model': 'required',
    'four_band_ordinal_model': 'required_when_supported',
    'severe_triage_model': 'required',
    'aggregate_grade_model': 'required',
    'embedding_grade_model': 'required',
}
FAMILY_REQUIRED_DIRS = [
    'summary',
    'diagnostics',
    'predictions',
    'model',
    'evidence',
    'internal',
]


class EndotheliosisGradeModelError(RuntimeError):
    """Raised when P3 grade-model inputs violate the contract."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family_id: str
    target_kind: str
    feature_family: str
    feature_columns: tuple[str, ...]
    model_kind: str
    threshold_target: float | None = None
    regularization_c: float = 1.0
    mr_computable: bool = True


def grade_model_output_paths(burden_output_dir: Path) -> dict[str, Path]:
    """Return canonical P3 selector output paths."""
    root = Path(burden_output_dir) / GRADE_MODEL_ROOT_NAME
    return {
        'root': root,
        'summary': root / 'summary',
        'diagnostics': root / 'diagnostics',
        'splits': root / 'splits',
        'predictions': root / 'predictions',
        'internal': root / 'internal',
        'evidence': root / 'evidence',
        'model': root / 'model',
        'deployment': root / 'deployment',
        'index': root / 'INDEX.md',
        'development_folds': root / 'splits' / 'development_folds.csv',
        'feature_diagnostics': root / 'internal' / 'feature_diagnostics.json',
        'candidate_metrics': root / 'internal' / 'candidate_metrics.csv',
        'candidate_configs': root / 'internal' / 'candidate_configs.json',
        'autonomous_loop_log': root / 'internal' / 'autonomous_loop_log.json',
        'development_oof_predictions': root
        / 'predictions'
        / 'development_oof_predictions.csv',
        'final_training_predictions': root
        / 'predictions'
        / 'final_model_training_predictions.csv',
        'final_verdict_json': root / 'summary' / 'final_product_verdict.json',
        'final_verdict_md': root / 'summary' / 'final_product_verdict.md',
        'candidate_coverage_matrix': root / 'summary' / 'candidate_coverage_matrix.csv',
        'model_selection_table': root / 'summary' / 'model_selection_table.csv',
        'development_oof_metrics': root / 'summary' / 'development_oof_metrics.csv',
        'ordinal_feasibility': root / 'summary' / 'ordinal_feasibility.json',
        'severe_threshold_selection': root
        / 'summary'
        / 'severe_threshold_selection.json',
        'input_artifact_index': root / 'summary' / 'input_artifact_index.json',
        'artifact_manifest': root / 'summary' / 'artifact_manifest.json',
        'executive_summary': root / 'summary' / 'executive_summary.md',
        'selector_diagnostics': root / 'diagnostics' / 'selector_diagnostics.json',
        'family_gate_diagnostics': root
        / 'diagnostics'
        / 'candidate_family_gate_diagnostics.json',
        'hard_blockers': root / 'diagnostics' / 'hard_blockers.json',
        'final_model': root / 'model' / 'final_model.joblib',
        'final_model_metadata': root / 'model' / 'final_model_metadata.json',
        'inference_schema': root / 'model' / 'inference_schema.json',
        'deployment_smoke_predictions': root
        / 'model'
        / 'deployment_smoke_predictions.csv',
        'mr_smoke_manifest': root / 'deployment' / 'mr_tiff_smoke_manifest.csv',
        'mr_smoke_predictions': root / 'deployment' / 'mr_tiff_smoke_predictions.csv',
        'mr_smoke_report': root / 'deployment' / 'mr_tiff_smoke_report.html',
        'segmentation_quantification_contract': root
        / 'deployment'
        / 'segmentation_quantification_contract.json',
        'dox_smoke_manifest': root
        / 'deployment'
        / 'dox_scored_no_mask_smoke_manifest.csv',
        'dox_smoke_predictions': root
        / 'deployment'
        / 'dox_scored_no_mask_smoke_predictions.csv',
        'dox_smoke_summary': root
        / 'deployment'
        / 'dox_scored_no_mask_smoke_summary.csv',
        'dox_smoke_threshold_curve': root
        / 'deployment'
        / 'dox_scored_no_mask_smoke_threshold_curve.csv',
        'dox_smoke_report': root
        / 'deployment'
        / 'dox_scored_no_mask_smoke_report.html',
        'dox_smoke_contract': root
        / 'deployment'
        / 'dox_scored_no_mask_smoke_contract.json',
        'dox_overcall_triage_queue': root
        / 'deployment'
        / 'dox_scored_no_mask_overcall_triage_queue.csv',
        'dox_overcall_triage_report': root
        / 'deployment'
        / 'dox_scored_no_mask_overcall_triage_report.html',
        'dox_overcall_review_diagnostic': root
        / 'deployment'
        / 'dox_scored_no_mask_overcall_review_diagnostic.json',
        'dox_overcall_review_interpretation': root
        / 'deployment'
        / 'dox_scored_no_mask_first12_review_interpretation.csv',
        'error_review': root / 'evidence' / 'error_review.html',
        'severe_false_negative_review': root
        / 'evidence'
        / 'severe_false_negative_review.html',
        'ordinal_confusion_review': root / 'evidence' / 'ordinal_confusion_review.html',
    }


def _save_json(data: Any, output_path: Path) -> Path:
    return save_json(data, output_path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _prepare_dirs(paths: dict[str, Path], burden_output_dir: Path) -> None:
    for key, path in paths.items():
        if key in {
            'index',
            'development_folds',
            'feature_diagnostics',
            'candidate_metrics',
            'candidate_configs',
            'autonomous_loop_log',
            'development_oof_predictions',
            'final_training_predictions',
            'final_verdict_json',
            'final_verdict_md',
            'candidate_coverage_matrix',
            'model_selection_table',
            'development_oof_metrics',
            'ordinal_feasibility',
            'severe_threshold_selection',
            'input_artifact_index',
            'artifact_manifest',
            'executive_summary',
            'selector_diagnostics',
            'family_gate_diagnostics',
            'hard_blockers',
            'final_model',
            'final_model_metadata',
            'inference_schema',
            'deployment_smoke_predictions',
            'mr_smoke_manifest',
            'mr_smoke_predictions',
            'mr_smoke_report',
            'segmentation_quantification_contract',
            'dox_smoke_manifest',
            'dox_smoke_predictions',
            'dox_smoke_summary',
            'dox_smoke_threshold_curve',
            'dox_smoke_report',
            'dox_smoke_contract',
            'dox_overcall_triage_queue',
            'dox_overcall_triage_report',
            'dox_overcall_review_diagnostic',
            'dox_overcall_review_interpretation',
            'error_review',
            'severe_false_negative_review',
            'ordinal_confusion_review',
        }:
            continue
        path.mkdir(parents=True, exist_ok=True)
    for family_id in FIRST_CLASS_FAMILIES:
        family_root = Path(burden_output_dir) / family_id
        family_root.mkdir(parents=True, exist_ok=True)
        for dirname in FAMILY_REQUIRED_DIRS:
            (family_root / dirname).mkdir(parents=True, exist_ok=True)


def _artifact_record(path: Path, role: str, source: str, status: str) -> dict[str, str]:
    return {
        'path': str(path),
        'role': role,
        'source_subtree': source,
        'access_status': status,
        'exists': str(path.exists()),
    }


def _input_artifacts(burden_output_dir: Path) -> list[dict[str, str]]:
    root = Path(burden_output_dir)
    candidates = [
        (
            root / 'primary_burden_index' / 'model' / 'burden_predictions.csv',
            'primary burden OOF predictions',
            'primary_burden_index',
        ),
        (
            root / 'primary_burden_index' / 'feature_sets' / 'morphology_features.csv',
            'deterministic morphology feature table',
            'primary_burden_index',
        ),
        (
            root
            / 'primary_burden_index'
            / 'diagnostics'
            / 'morphology_feature_diagnostics.json',
            'morphology readiness diagnostics',
            'primary_burden_index',
        ),
        (
            root / 'learned_roi' / 'feature_sets' / 'learned_roi_features.csv',
            'learned ROI feature table',
            'learned_roi',
        ),
        (
            root / 'learned_roi' / 'candidates' / 'learned_roi_candidate_summary.json',
            'learned ROI candidate summary',
            'learned_roi',
        ),
        (
            root / 'source_aware_estimator' / 'summary' / 'estimator_verdict.json',
            'source-aware verdict',
            'source_aware_estimator',
        ),
        (
            root
            / 'severe_aware_ordinal_estimator'
            / 'summary'
            / 'estimator_verdict.json',
            'P2 severe-aware verdict',
            'severe_aware_ordinal_estimator',
        ),
        (
            root
            / 'severe_aware_ordinal_estimator'
            / 'evidence'
            / 'severe_false_negative_adjudications.json',
            'P2 false-negative adjudications',
            'severe_aware_ordinal_estimator',
        ),
    ]
    return [
        _artifact_record(
            path, role, source, 'read_only_consumed' if path.exists() else 'missing'
        )
        for path, role, source in candidates
    ]


def _merge_optional_table(
    base: pd.DataFrame,
    path: Path,
    *,
    prefix: str,
    protected: Sequence[str],
    required_join_keys: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        return base, {'path': str(path), 'available': False, 'merged_columns': []}
    table = pd.read_csv(path)
    missing_required = [
        key
        for key in required_join_keys
        if key not in base.columns or key not in table.columns
    ]
    if missing_required:
        return base, {
            'path': str(path),
            'available': True,
            'merged_columns': [],
            'join_status': 'missing_required_join_keys',
            'required_join_keys': list(required_join_keys),
            'missing_required_join_keys': missing_required,
            'row_count': int(len(table)),
        }
    join_keys = [
        key
        for key in ['subject_image_id', 'glomerulus_id']
        if key in base and key in table
    ]
    if not join_keys:
        return base, {
            'path': str(path),
            'available': True,
            'merged_columns': [],
            'join_status': 'no_shared_join_keys',
        }
    rename: dict[str, str] = {}
    drop_columns: list[str] = []
    for column in table.columns:
        if column in join_keys:
            continue
        if column in protected:
            drop_columns.append(column)
            continue
        if column in base.columns:
            rename[column] = f'{prefix}{column}'
    if drop_columns:
        table = table.drop(columns=drop_columns)
    table = table.rename(columns=rename)
    before = set(base.columns)
    duplicate_rows = table.duplicated(subset=join_keys, keep=False)
    if duplicate_rows.any():
        duplicate_keys = (
            table.loc[duplicate_rows, join_keys]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .to_dict(orient='records')
        )
        return base, {
            'path': str(path),
            'available': True,
            'merged_columns': [],
            'join_status': 'duplicate_join_keys',
            'join_keys': join_keys,
            'duplicate_join_keys': duplicate_keys,
            'row_count': int(len(table)),
        }
    merged = base.merge(table, on=join_keys, how='left', validate='many_to_one')
    return merged, {
        'path': str(path),
        'available': True,
        'join_keys': join_keys,
        'join_status': 'merged',
        'merged_columns': sorted(set(merged.columns) - before),
        'row_count': int(len(table)),
    }


def _load_adjudications(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    records = payload if isinstance(payload, list) else payload.get('records', [])
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get('subject_image_id', '')).strip()
        if key:
            result[key] = dict(record)
    return result


def _adjudicated_severe(
    row: pd.Series, adjudications: dict[str, dict[str, Any]]
) -> bool:
    original = bool(float(row['score']) >= PRIMARY_SEVERE_THRESHOLD)
    record = adjudications.get(str(row.get('subject_image_id', '')))
    if not record:
        return original
    adjudication = str(record.get('grade_adjudication', '')).lower()
    failure_source = str(record.get('failure_source', '')).lower()
    if 'too_high' in adjudication or 'not_severe' in failure_source:
        return False
    if 'correct' in adjudication or 'valid_grade' in failure_source:
        return True
    return original


def _reconstruct_feature_frame(
    embedding_df: pd.DataFrame, burden_output_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    missing = [
        column for column in IDENTITY_COLUMNS if column not in embedding_df.columns
    ]
    if missing:
        raise EndotheliosisGradeModelError(
            f'P3 candidate rows are missing required identity columns: {missing}'
        )
    validate_score_values(embedding_df['score'])
    protected = [*IDENTITY_COLUMNS, *OPTIONAL_IDENTITY_COLUMNS]
    work = embedding_df.copy().reset_index(drop=True)
    work['score'] = work['score'].astype(float)
    diagnostics: dict[str, Any] = {'source_tables': []}
    hard_blockers: list[str] = []

    root = Path(burden_output_dir)
    morphology_path = (
        root / 'primary_burden_index' / 'feature_sets' / 'morphology_features.csv'
    )
    learned_path = root / 'learned_roi' / 'feature_sets' / 'learned_roi_features.csv'
    work, merge_diag = _merge_optional_table(
        work,
        morphology_path,
        prefix='morphology_source_',
        protected=protected,
        required_join_keys=('subject_image_id', 'glomerulus_id'),
    )
    diagnostics['source_tables'].append(merge_diag)
    if not merge_diag.get('available'):
        hard_blockers.append('missing_primary_burden_morphology_features')
    elif merge_diag.get('join_status') != 'merged':
        hard_blockers.append('unusable_primary_burden_morphology_features_join_keys')
    work, merge_diag = _merge_optional_table(
        work,
        learned_path,
        prefix='learned_source_',
        protected=protected,
        required_join_keys=('subject_image_id', 'glomerulus_id'),
    )
    diagnostics['source_tables'].append(merge_diag)
    if not merge_diag.get('available'):
        hard_blockers.append('missing_learned_roi_features')
    elif merge_diag.get('join_status') != 'merged':
        hard_blockers.append('unusable_learned_roi_features_join_keys')

    adjudication_path = (
        root
        / 'severe_aware_ordinal_estimator'
        / 'evidence'
        / 'severe_false_negative_adjudications.json'
    )
    adjudications = _load_adjudications(adjudication_path)
    work['original_severe'] = work['score'].astype(float) >= PRIMARY_SEVERE_THRESHOLD
    work['adjudicated_severe'] = [
        _adjudicated_severe(row, adjudications) for _, row in work.iterrows()
    ]
    work['severe_target_source'] = np.where(
        work['subject_image_id'].astype(str).isin(adjudications),
        'p2_false_negative_adjudication',
        'original_score_gte_2',
    )

    for column in IDENTITY_COLUMNS:
        missing_values = work[column].isna() | (work[column].astype(str).str.len() == 0)
        if missing_values.any():
            hard_blockers.append(f'missing_values_in_{column}')
    for column in ['roi_image_path', 'roi_mask_path']:
        missing_paths = [
            value
            for value in work[column].astype(str).tolist()
            if not value or not Path(value).exists()
        ]
        if missing_paths:
            hard_blockers.append(f'{column}_paths_missing_on_disk:{len(missing_paths)}')

    numeric = work.select_dtypes(include=[np.number, bool])
    nonfinite = {
        column: int(
            (~np.isfinite(pd.to_numeric(numeric[column], errors='coerce'))).sum()
        )
        for column in numeric.columns
    }
    near_zero = []
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors='coerce').dropna()
        if len(values) and float(values.std(ddof=0)) == 0.0:
            near_zero.append(column)
    diagnostics.update(
        {
            'row_count': int(len(work)),
            'subject_count': int(work['subject_id'].astype(str).nunique()),
            'cohort_counts': work['cohort_id'].astype(str).value_counts().to_dict(),
            'score_counts': {
                f'{score:g}': int(np.isclose(work['score'].astype(float), score).sum())
                for score in ALLOWED_SCORE_VALUES
            },
            'original_severe_count': int(work['original_severe'].sum()),
            'adjudicated_severe_count': int(work['adjudicated_severe'].sum()),
            'adjudication_path': str(adjudication_path),
            'adjudication_available': adjudication_path.exists(),
            'adjudication_record_count': int(len(adjudications)),
            'nonfinite_counts': nonfinite,
            'near_zero_variance_columns': sorted(near_zero),
            'hard_blockers': sorted(set(hard_blockers)),
        }
    )
    return work, diagnostics, sorted(set(hard_blockers))


def _three_band(score: float) -> str:
    if score < 1.0:
        return 'none_low'
    if score < 2.0:
        return 'mild_mod'
    return 'severe'


def _four_band(score: float) -> str:
    if np.isclose(score, 0.0):
        return '0'
    if np.isclose(score, 0.5):
        return '0.5'
    if score < 2.0:
        return '1_1.5'
    return '2_3'


def _deterministic_development_folds(
    frame: pd.DataFrame, n_splits: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subject_rows = []
    for subject_id, group in frame.groupby('subject_id', sort=True):
        scores = sorted(float(value) for value in group['score'].dropna().unique())
        subject_rows.append(
            {
                'subject_id': str(subject_id),
                'cohort_id': str(group['cohort_id'].astype(str).mode().iloc[0]),
                'row_count': int(len(group)),
                'max_score': float(np.max(group['score'].astype(float))),
                'severe_positive_rows': int(group['adjudicated_severe'].sum()),
                'score_values': '|'.join(f'{value:g}' for value in scores),
            }
        )
    subjects = pd.DataFrame(subject_rows)
    split_count = min(max(2, int(n_splits)), len(subjects))
    if split_count < 2:
        raise EndotheliosisGradeModelError(
            'Need at least two subjects for grouped development validation'
        )
    subjects = subjects.sort_values(
        ['severe_positive_rows', 'max_score', 'row_count', 'subject_id'],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    fold_load = [
        {'rows': 0, 'severe': 0, 'subjects': 0, 'score_sum': 0.0}
        for _ in range(split_count)
    ]
    assignments: dict[str, int] = {}
    for _, row in subjects.iterrows():
        fold_index = min(
            range(split_count),
            key=lambda idx: (
                fold_load[idx]['severe'],
                fold_load[idx]['rows'],
                fold_load[idx]['score_sum'],
                idx,
            ),
        )
        assignments[str(row['subject_id'])] = fold_index + 1
        fold_load[fold_index]['rows'] += int(row['row_count'])
        fold_load[fold_index]['severe'] += int(row['severe_positive_rows'])
        fold_load[fold_index]['subjects'] += 1
        fold_load[fold_index]['score_sum'] += float(row['max_score'])
    result = frame[
        [
            *[column for column in IDENTITY_COLUMNS if column in frame.columns],
            'original_severe',
            'adjudicated_severe',
        ]
    ].copy()
    result['fold'] = result['subject_id'].astype(str).map(assignments).astype(int)
    result['three_band'] = result['score'].astype(float).map(_three_band)
    result['four_band'] = result['score'].astype(float).map(_four_band)
    coverage_rows = []
    for fold_index, group in result.groupby('fold', sort=True):
        scores = set(float(value) for value in group['score'].dropna().unique())
        absent = [
            float(value)
            for value in ALLOWED_SCORE_VALUES
            if not any(np.isclose(value, observed) for observed in scores)
        ]
        coverage_rows.append(
            {
                'fold': int(fold_index),
                'row_count': int(len(group)),
                'subject_count': int(group['subject_id'].nunique()),
                'severe_positive_rows': int(group['adjudicated_severe'].sum()),
                'absent_scores': [f'{value:g}' for value in absent],
                'coverage_status': 'complete'
                if not absent
                else 'subject_group_limited',
            }
        )
    diagnostics = {
        'split_count': int(split_count),
        'assignment_method': 'deterministic_subject_greedy_stratification',
        'metric_label': 'grouped_out_of_fold_development_estimate',
        'no_internal_locked_test_split': True,
        'fold_coverage': coverage_rows,
    }
    return result, diagnostics


def _numeric_feature_columns(
    frame: pd.DataFrame, candidates: Sequence[str]
) -> list[str]:
    columns = []
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors='coerce')
        if values.notna().any() and float(values.fillna(0.0).std(ddof=0)) > 0.0:
            columns.append(column)
    return columns


def _clip_scaled_features(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.clip(values, -20.0, 20.0), nan=0.0, posinf=20.0, neginf=-20.0
    )


def _feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    embedding_columns = [
        column for column in frame.columns if column.startswith('embedding_')
    ]
    learned_columns = [
        column
        for column in frame.columns
        if column.startswith('learned_')
        or column.startswith('encoder_')
        or column.startswith('roi_texture_')
    ]
    morphology_columns = _numeric_feature_columns(frame, MORPHOLOGY_FEATURE_COLUMNS)
    roi_columns = _numeric_feature_columns(frame, ROI_QC_COLUMNS)
    interaction_columns: list[str] = []
    interaction_pairs = [
        ('morph_open_space_density', 'morph_rbc_like_color_burden'),
        ('morph_slit_like_area_fraction', 'morph_rbc_like_color_burden'),
        ('morph_slit_like_area_fraction', 'morph_lumen_detectability_score'),
        (
            'morph_nuclear_mesangial_confounder_area_fraction',
            'morph_slit_like_area_fraction',
        ),
        ('roi_fill_fraction', 'morph_open_space_density'),
        ('roi_component_count', 'morph_pale_lumen_max_area_fraction'),
    ]
    for left, right in interaction_pairs:
        if left in frame.columns and right in frame.columns:
            name = f'interaction__{left}__x__{right}'
            frame[name] = pd.to_numeric(frame[left], errors='coerce').fillna(
                0.0
            ) * pd.to_numeric(frame[right], errors='coerce').fillna(0.0)
            interaction_columns.append(name)
    aggregate_columns = _numeric_feature_columns(
        frame,
        [
            'roi_component_count',
            'roi_area',
            'roi_union_bbox_width',
            'roi_union_bbox_height',
            'roi_largest_component_area_fraction',
        ],
    )
    return {
        'roi_qc': roi_columns,
        'morphology': morphology_columns,
        'roi_qc_morphology': [*roi_columns, *morphology_columns],
        'severe_interactions': [
            *roi_columns,
            *morphology_columns,
            *_numeric_feature_columns(frame, interaction_columns),
        ],
        'learned_roi': _numeric_feature_columns(frame, learned_columns),
        'embedding_reduced': _numeric_feature_columns(frame, embedding_columns),
        'learned_morphology': _numeric_feature_columns(
            frame, [*learned_columns, *morphology_columns]
        ),
        'embedding_morphology': _numeric_feature_columns(
            frame, [*embedding_columns, *morphology_columns]
        ),
        'aggregate': aggregate_columns,
        'hybrid': _numeric_feature_columns(
            frame,
            [*roi_columns, *morphology_columns, *learned_columns, *embedding_columns],
        ),
        '_source_learned': _numeric_feature_columns(frame, learned_columns),
        '_source_embedding': _numeric_feature_columns(frame, embedding_columns),
        '_source_interactions': _numeric_feature_columns(frame, interaction_columns),
    }


def _classifier(
    model_kind: str, n_features: int, n_classes: int, regularization_c: float = 1.0
) -> Pipeline:
    if model_kind == 'extra_trees':
        return Pipeline(
            [
                (
                    'model',
                    ExtraTreesClassifier(
                        n_estimators=100,
                        max_depth=4,
                        class_weight='balanced',
                        random_state=17,
                    ),
                )
            ]
        )
    steps: list[tuple[str, Any]] = [
        ('scale', StandardScaler()),
        (
            'clip_scaled_features',
            FunctionTransformer(_clip_scaled_features, validate=False),
        ),
    ]
    if model_kind == 'pca_logistic':
        k_features = max(1, min(8, n_features, n_classes * 4))
        steps.append(
            ('feature_select', SelectKBest(score_func=f_classif, k=k_features))
        )
    steps.append(
        (
            'model',
            LogisticRegression(
                class_weight='balanced',
                max_iter=2000,
                solver='lbfgs',
                C=float(regularization_c),
                random_state=17,
            ),
        )
    )
    return Pipeline(steps)


def _choose_threshold(
    probabilities: np.ndarray, target: np.ndarray, target_recall: float
) -> float:
    return choose_recall_threshold(probabilities, target, target_recall)


def _evaluate_severe_candidate(
    frame: pd.DataFrame, folds: pd.Series, spec: CandidateSpec
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = frame['adjudicated_severe'].astype(bool).to_numpy()
    x = to_finite_numeric_matrix(frame, spec.feature_columns)
    probabilities = np.zeros(len(frame), dtype=np.float64)
    thresholds = np.zeros(len(frame), dtype=np.float64)
    fold_rows = []
    warning_records: list[dict[str, str]] = []
    for fold in sorted(folds.unique()):
        test_mask = folds == fold
        train_mask = ~test_mask
        model = _classifier(spec.model_kind, x.shape[1], 2, spec.regularization_c)

        def fit_and_predict() -> tuple[np.ndarray, np.ndarray]:
            model.fit(x[train_mask], y[train_mask])
            train_prob = model.predict_proba(x[train_mask])[:, 1]
            fold_probabilities = model.predict_proba(x[test_mask])[:, 1]
            return train_prob, fold_probabilities

        (train_prob, fold_probabilities), warning_result = capture_fit_warnings(
            fit_and_predict
        )
        warning_records.extend(
            {'fold': str(int(fold)), **record}
            for record in warning_result.warning_messages
        )
        threshold = _choose_threshold(
            train_prob, y[train_mask], float(spec.threshold_target or 0.8)
        )
        probabilities[test_mask] = fold_probabilities
        thresholds[test_mask] = threshold
        fold_pred = probabilities[test_mask] >= threshold
        fold_rows.append(
            {
                'fold': int(fold),
                'candidate_id': spec.candidate_id,
                'threshold': float(threshold),
                'recall': float(recall_score(y[test_mask], fold_pred, zero_division=0)),
                'precision': float(
                    precision_score(y[test_mask], fold_pred, zero_division=0)
                ),
                'false_negatives': int(np.sum(y[test_mask] & ~fold_pred)),
                'false_positives': int(np.sum(~y[test_mask] & fold_pred)),
            }
        )
    predicted = probabilities >= thresholds
    metrics = {
        'candidate_id': spec.candidate_id,
        'family_id': spec.family_id,
        'target_kind': spec.target_kind,
        'feature_family': spec.feature_family,
        'model_kind': spec.model_kind,
        'threshold_target': float(spec.threshold_target or 0.8),
        'feature_count': int(x.shape[1]),
        'metric_label': GROUPED_DEVELOPMENT_METRIC_LABEL,
        'regularization_c': float(spec.regularization_c),
        'mr_computable': bool(spec.mr_computable),
        'auroc': _safe_roc_auc(y, probabilities),
        'average_precision': _safe_average_precision(y, probabilities),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'false_negatives': int(np.sum(y & ~predicted)),
        'false_positives': int(np.sum(~y & predicted)),
        'threshold': float(np.median(thresholds)),
        'finite_output': bool(np.isfinite(probabilities).all()),
        'warning_status': 'warnings_recorded'
        if warning_records
        else 'no_warnings_recorded',
        'warning_count': int(len(warning_records)),
        'warning_messages': warning_records[:10],
        'fold_metrics': fold_rows,
    }
    predictions = frame[
        [column for column in IDENTITY_COLUMNS if column in frame]
    ].copy()
    predictions['candidate_id'] = spec.candidate_id
    predictions['fold'] = folds.to_numpy()
    predictions['observed_severe'] = y
    predictions['predicted_severe_probability'] = probabilities
    predictions['selected_threshold'] = thresholds
    predictions['predicted_severe'] = predicted
    return predictions, metrics


def _evaluate_ordinal_candidate(
    frame: pd.DataFrame, folds: pd.Series, spec: CandidateSpec, labels: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_column = 'three_band' if spec.target_kind == 'three_band' else 'four_band'
    y = frame[target_column].astype(str).to_numpy()
    x = to_finite_numeric_matrix(frame, spec.feature_columns)
    predicted = np.empty(len(frame), dtype=object)
    probabilities = np.zeros((len(frame), len(labels)), dtype=np.float64)
    warning_records: list[dict[str, str]] = []
    for fold in sorted(folds.unique()):
        test_mask = folds == fold
        train_mask = ~test_mask
        model = _classifier(
            spec.model_kind, x.shape[1], len(labels), spec.regularization_c
        )

        def fit_and_predict() -> tuple[np.ndarray, np.ndarray]:
            model.fit(x[train_mask], y[train_mask])
            fold_predicted = model.predict(x[test_mask])
            fold_prob = model.predict_proba(x[test_mask])
            return fold_predicted, fold_prob

        (fold_predicted, fold_prob), warning_result = capture_fit_warnings(
            fit_and_predict
        )
        predicted[test_mask] = fold_predicted
        warning_records.extend(
            {'fold': str(int(fold)), **record}
            for record in warning_result.warning_messages
        )
        class_to_index = {label: index for index, label in enumerate(model.classes_)}
        for label_index, label in enumerate(labels):
            if label in class_to_index:
                probabilities[test_mask, label_index] = fold_prob[
                    :, class_to_index[label]
                ]
    severe_true = y == 'severe' if spec.target_kind == 'three_band' else y == '2_3'
    severe_pred = (
        predicted == 'severe'
        if spec.target_kind == 'three_band'
        else predicted == '2_3'
    )
    metrics = {
        'candidate_id': spec.candidate_id,
        'family_id': spec.family_id,
        'target_kind': spec.target_kind,
        'feature_family': spec.feature_family,
        'model_kind': spec.model_kind,
        'feature_count': int(x.shape[1]),
        'metric_label': GROUPED_DEVELOPMENT_METRIC_LABEL,
        'regularization_c': float(spec.regularization_c),
        'mr_computable': bool(spec.mr_computable),
        'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'severe_band_recall': float(
            recall_score(severe_true, severe_pred, zero_division=0)
        ),
        'non_adjacent_error_rate': _non_adjacent_error_rate(y, predicted, labels),
        'adjacent_accuracy': 1.0 - _non_adjacent_error_rate(y, predicted, labels),
        'confusion_matrix': confusion_matrix(
            y, predicted, labels=list(labels)
        ).tolist(),
        'finite_output': bool(np.isfinite(probabilities).all()),
        'warning_status': 'warnings_recorded'
        if warning_records
        else 'no_warnings_recorded',
        'warning_count': int(len(warning_records)),
        'warning_messages': warning_records[:10],
    }
    predictions = frame[
        [column for column in IDENTITY_COLUMNS if column in frame]
    ].copy()
    predictions['candidate_id'] = spec.candidate_id
    predictions['fold'] = folds.to_numpy()
    predictions['observed_band'] = y
    predictions['predicted_band'] = predicted
    for index, label in enumerate(labels):
        predictions[f'prob_{label}'] = probabilities[:, index]
    return predictions, metrics


def _safe_roc_auc(y: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probabilities))


def _safe_average_precision(y: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, probabilities))


def _non_adjacent_error_rate(
    observed: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> float:
    order = {label: index for index, label in enumerate(labels)}
    errors = [
        abs(order[str(obs)] - order[str(pred)]) > 1
        for obs, pred in zip(observed, predicted)
    ]
    return float(np.mean(errors)) if errors else 0.0


def _candidate_specs(feature_sets: dict[str, list[str]]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    severe_logistic_families = [
        'roi_qc',
        'morphology',
        'roi_qc_morphology',
        'severe_interactions',
    ]
    for family in severe_logistic_families:
        if feature_sets[family]:
            for target_recall in [0.8, 0.9, 0.95]:
                for regularization_c in [0.01, 0.1, 1.0, 10.0]:
                    specs.append(
                        CandidateSpec(
                            candidate_id=(
                                f'{family}_severe_recall_{target_recall:g}'
                                f'_c{regularization_c:g}'
                            ),
                            family_id='severe_triage_model',
                            target_kind='severe',
                            feature_family=family,
                            feature_columns=tuple(feature_sets[family]),
                            model_kind='logistic',
                            threshold_target=target_recall,
                            regularization_c=regularization_c,
                            mr_computable=True,
                        )
                    )
    learned_source_available = bool(feature_sets['_source_learned'])
    embedding_source_available = bool(feature_sets['_source_embedding'])
    learned_families = (
        ['learned_roi', 'learned_morphology'] if learned_source_available else []
    )
    embedding_families = (
        ['embedding_reduced', 'embedding_morphology']
        if embedding_source_available
        else []
    )
    for family in [*learned_families, *embedding_families]:
        source_columns = (
            feature_sets['_source_embedding']
            if family.startswith('embedding')
            else feature_sets['_source_learned']
        )
        if feature_sets[family] and source_columns:
            specs.append(
                CandidateSpec(
                    candidate_id=f'{family}_severe_gate',
                    family_id='embedding_grade_model',
                    target_kind='severe',
                    feature_family=family,
                    feature_columns=tuple(feature_sets[family]),
                    model_kind='pca_logistic',
                    threshold_target=0.9,
                    mr_computable=False,
                )
            )
    if feature_sets['aggregate']:
        specs.append(
            CandidateSpec(
                candidate_id='aggregate_aware_severe_gate',
                family_id='aggregate_grade_model',
                target_kind='severe',
                feature_family='aggregate',
                feature_columns=tuple(feature_sets['aggregate']),
                model_kind='logistic',
                threshold_target=0.9,
                mr_computable=True,
            )
        )
    if feature_sets['roi_qc_morphology']:
        specs.append(
            CandidateSpec(
                candidate_id='tree_exploratory_severe_gate',
                family_id='severe_triage_model',
                target_kind='severe',
                feature_family='roi_qc_morphology',
                feature_columns=tuple(feature_sets['roi_qc_morphology']),
                model_kind='extra_trees',
                threshold_target=0.9,
                mr_computable=True,
            )
        )
    ordinal_families = ['roi_qc_morphology']
    if learned_source_available:
        ordinal_families.append('learned_roi')
    if embedding_source_available:
        ordinal_families.append('embedding_reduced')
    if learned_source_available or embedding_source_available:
        ordinal_families.append('hybrid')
    for family in ordinal_families:
        source_ok = (
            True
            if family == 'roi_qc_morphology'
            else bool(feature_sets['_source_learned'])
            if family == 'learned_roi'
            else bool(feature_sets['_source_embedding'])
            if family == 'embedding_reduced'
            else bool(
                feature_sets['_source_learned'] or feature_sets['_source_embedding']
            )
        )
        if feature_sets[family] and source_ok:
            model_kind = (
                'pca_logistic'
                if family in {'embedding_reduced', 'hybrid'}
                else 'logistic'
            )
            mr_computable = family == 'roi_qc_morphology'
            specs.append(
                CandidateSpec(
                    candidate_id=f'{family}_three_band_ordinal',
                    family_id='three_band_ordinal_model',
                    target_kind='three_band',
                    feature_family=family,
                    feature_columns=tuple(feature_sets[family]),
                    model_kind=model_kind,
                    mr_computable=mr_computable,
                )
            )
            specs.append(
                CandidateSpec(
                    candidate_id=f'{family}_four_band_ordinal',
                    family_id='four_band_ordinal_model',
                    target_kind='four_band',
                    feature_family=family,
                    feature_columns=tuple(feature_sets[family]),
                    model_kind=model_kind,
                    mr_computable=mr_computable,
                )
            )
    return specs


def _evaluate_candidates(
    frame: pd.DataFrame, folds: pd.DataFrame, specs: Sequence[CandidateSpec]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame['three_band'] = frame['score'].astype(float).map(_three_band)
    frame['four_band'] = frame['score'].astype(float).map(_four_band)
    fold_keys = [
        key
        for key in ['subject_image_id', 'glomerulus_id']
        if key in frame and key in folds
    ]
    if not fold_keys:
        raise EndotheliosisGradeModelError(
            'Development folds are missing row join keys'
        )
    fold_series = frame[fold_keys].merge(
        folds[[*fold_keys, 'fold']], on=fold_keys, how='left', validate='many_to_one'
    )['fold']
    if fold_series.isna().any():
        raise EndotheliosisGradeModelError(
            'Development folds do not cover every candidate row'
        )
    metric_rows = []
    prediction_frames = []
    for spec in specs:
        if len(spec.feature_columns) == 0:
            continue
        if spec.target_kind == 'severe':
            predictions, metrics = _evaluate_severe_candidate(frame, fold_series, spec)
        elif spec.target_kind == 'three_band':
            predictions, metrics = _evaluate_ordinal_candidate(
                frame, fold_series, spec, THREE_BAND_LABELS
            )
        else:
            predictions, metrics = _evaluate_ordinal_candidate(
                frame, fold_series, spec, FOUR_BAND_LABELS
            )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def _baseline_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    severe = frame['adjudicated_severe'].astype(bool).to_numpy()
    severe_prior = float(np.mean(severe))
    severe_pred = np.full(len(severe), severe_prior >= 0.5, dtype=bool)
    rows = [
        {
            'candidate_id': 'empirical_prior_severe_baseline',
            'family_id': 'baseline',
            'target_kind': 'severe',
            'feature_family': 'none',
            'model_kind': 'empirical_prior',
            'metric_label': 'grouped_out_of_fold_development_estimate',
            'average_precision': severe_prior,
            'recall': float(recall_score(severe, severe_pred, zero_division=0)),
            'precision': float(precision_score(severe, severe_pred, zero_division=0)),
            'finite_output': True,
        }
    ]
    for target, labels in [
        ('three_band', THREE_BAND_LABELS),
        ('four_band', FOUR_BAND_LABELS),
        ('six_bin', tuple(f'{value:g}' for value in ALLOWED_SCORE_VALUES)),
    ]:
        observed = (
            frame['score'].astype(float).map(_three_band).astype(str).to_numpy()
            if target == 'three_band'
            else frame['score'].astype(float).map(_four_band).astype(str).to_numpy()
            if target == 'four_band'
            else frame['score'].astype(float).map(lambda value: f'{value:g}').to_numpy()
        )
        majority = pd.Series(observed).mode().iloc[0]
        predicted = np.full(len(observed), majority, dtype=object)
        rows.append(
            {
                'candidate_id': f'majority_{target}_baseline',
                'family_id': 'baseline',
                'target_kind': target,
                'feature_family': 'none',
                'model_kind': 'majority_class',
                'metric_label': 'grouped_out_of_fold_development_estimate',
                'accuracy': float(accuracy_score(observed, predicted)),
                'balanced_accuracy': float(
                    balanced_accuracy_score(observed, predicted)
                ),
                'finite_output': True,
                'label_order': '|'.join(labels),
            }
        )
    return rows


def _select_product(
    metrics: pd.DataFrame, hard_blockers: Sequence[str]
) -> dict[str, Any]:
    if metrics.empty:
        return {
            'overall_status': 'current_data_insufficient',
            'selected_candidate_id': '',
            'selected_family_id': '',
            'selected_output_type': '',
            'quantification_gate_passed': False,
            'severe_safety_gate_passed': False,
            'mr_tiff_deployment_gate_passed': False,
            'hard_blockers': list(hard_blockers),
            'claim_boundary': 'current data insufficient; no external validation',
        }
    severe_rows = metrics[metrics['target_kind'] == 'severe'].copy()
    severe_rows['severe_gate_passed'] = (
        (severe_rows['recall'].fillna(0.0) >= 0.8)
        & (severe_rows['precision'].fillna(0.0) >= 0.25)
    ) | (
        (severe_rows['recall'].fillna(0.0) >= 0.9)
        & (severe_rows['precision'].fillna(0.0) >= 0.15)
    )
    severe_pass = severe_rows[severe_rows['severe_gate_passed']]
    severe_mr_pass = severe_pass[
        severe_pass['mr_computable'].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
    ]
    best_severe = (
        severe_rows.sort_values(
            ['severe_gate_passed', 'recall', 'precision', 'average_precision'],
            ascending=[False, False, False, False],
        ).iloc[0]
        if not severe_rows.empty
        else None
    )
    ordinal_rows = metrics[
        metrics['target_kind'].isin(['three_band', 'four_band'])
    ].copy()
    ordinal_rows['ordinal_gate_passed'] = (
        (ordinal_rows['balanced_accuracy'].fillna(0.0) >= 0.5)
        & (ordinal_rows['severe_band_recall'].fillna(0.0) >= 0.8)
        & ordinal_rows['finite_output'].fillna(False).astype(bool)
    )
    ordinal_pass = ordinal_rows[ordinal_rows['ordinal_gate_passed']]
    ordinal_mr_pass = ordinal_pass[
        ordinal_pass['mr_computable'].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
    ]
    best_ordinal = (
        ordinal_rows.sort_values(
            ['ordinal_gate_passed', 'balanced_accuracy', 'severe_band_recall'],
            ascending=[False, False, False],
        ).iloc[0]
        if not ordinal_rows.empty
        else None
    )
    selected = None
    if not ordinal_mr_pass.empty and not severe_mr_pass.empty:
        selected = ordinal_mr_pass.sort_values(
            ['balanced_accuracy', 'severe_band_recall'], ascending=[False, False]
        ).iloc[0]
        status = 'model_ready_pending_mr_tiff_deployment_smoke'
        output_type = 'ordinal_grade_band'
        quant_pass = True
    elif not severe_mr_pass.empty:
        selected = severe_mr_pass.sort_values(
            ['recall', 'precision', 'average_precision'],
            ascending=[False, False, False],
        ).iloc[0]
        status = 'model_ready_pending_mr_tiff_deployment_smoke'
        output_type = 'severe_risk_triage'
        quant_pass = True
    elif not ordinal_pass.empty or not severe_pass.empty:
        selected = (
            ordinal_pass.sort_values(
                ['balanced_accuracy', 'severe_band_recall'], ascending=[False, False]
            ).iloc[0]
            if not ordinal_pass.empty
            else severe_pass.sort_values(
                ['recall', 'precision', 'average_precision'],
                ascending=[False, False, False],
            ).iloc[0]
        )
        status = 'diagnostic_only_current_data_model'
        output_type = str(selected['target_kind'])
        quant_pass = False
    elif best_ordinal is not None or best_severe is not None:
        selected = best_ordinal if best_ordinal is not None else best_severe
        status = 'diagnostic_only_current_data_model'
        output_type = str(selected['target_kind'])
        quant_pass = False
    else:
        status = 'current_data_insufficient'
        output_type = ''
        quant_pass = False

    if not quant_pass:
        deployment_blockers = [
            'no_candidate_passed_quantification_gate',
            'severe_safety_gate_failed',
            'ordinal_grade_gate_failed',
        ]
    else:
        deployment_blockers = (
            ['selected_candidate_feature_schema_not_mr_tiff_computable']
            if selected is not None and not bool(selected.get('mr_computable', False))
            else ['mr_tiff_segmentation_to_quantification_path_not_proven']
        )
    return {
        'overall_status': status,
        'selected_candidate_id': ''
        if selected is None
        else str(selected['candidate_id']),
        'selected_family_id': '' if selected is None else str(selected['family_id']),
        'selected_output_type': output_type,
        'quantification_gate_passed': bool(quant_pass),
        'severe_safety_gate_passed': bool(not severe_pass.empty),
        'mr_tiff_deployment_gate_passed': False,
        'readme_facing_deployment_allowed': False,
        'hard_blockers': list(dict.fromkeys([*hard_blockers, *deployment_blockers])),
        'strongest_severe_candidate': {}
        if best_severe is None
        else best_severe.to_dict(),
        'strongest_ordinal_candidate': {}
        if best_ordinal is None
        else best_ordinal.to_dict(),
        'minimum_additional_data_recommendations': [
            'Add or adjudicate source-diverse grade 2/3 MR TIFF examples with accepted glomerulus masks before deployment.',
            'Add per-image or per-glomerulus labels for multi-component images so aggregate-label ambiguity can be measured directly.',
            'Stabilize severe-risk modeling enough to clear recall/precision gates without numeric warning burden.',
        ],
        'claim_boundary': (
            'current-data grouped development evidence only; source-sensitive; '
            'no external validation and no README-facing MR TIFF deployment claim'
        ),
    }


def _write_family_artifacts(
    burden_output_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_diagnostics: dict[str, Any],
    hard_blockers: Sequence[str],
) -> None:
    for family_id, requirement in FIRST_CLASS_FAMILIES.items():
        family_root = Path(burden_output_dir) / family_id
        family_metrics = metrics[metrics['family_id'] == family_id].copy()
        family_predictions = predictions[
            predictions['candidate_id'].isin(family_metrics['candidate_id'].astype(str))
        ].copy()
        family_metrics.to_csv(family_root / 'summary' / 'metrics.csv', index=False)
        family_predictions.to_csv(
            family_root / 'predictions' / 'development_oof_predictions.csv', index=False
        )
        diagnostics = {
            'family_id': family_id,
            'required_or_exploratory': requirement,
            'candidate_count': int(len(family_metrics)),
            'run_status': 'evaluated' if not family_metrics.empty else 'hard_blocked',
            'hard_blockers': list(hard_blockers) if family_metrics.empty else [],
        }
        for name in [
            'input_support',
            'feature_diagnostics',
            'fold_diagnostics',
            'source_sensitivity',
            'gate_diagnostics',
        ]:
            payload = (
                diagnostics if name != 'feature_diagnostics' else feature_diagnostics
            )
            _save_json(payload, family_root / 'diagnostics' / f'{name}.json')
        if family_id == 'embedding_grade_model':
            _save_json(
                {'status': 'diagnostic_required_for_embedding_candidates'},
                family_root / 'diagnostics' / 'embedding_source_predictability.json',
            )
        if family_id == 'aggregate_grade_model':
            _save_json(
                {'status': 'image_level_aggregate_labels_used_for_gate_evaluation'},
                family_root / 'diagnostics' / 'aggregate_label_diagnostics.json',
            )
        if family_metrics.empty:
            _save_json(
                {'hard_blockers': list(hard_blockers)},
                family_root / 'diagnostics' / 'hard_blockers.json',
            )
        (family_root / 'INDEX.md').write_text(
            f"""# {family_id}

This first-class model-family subtree contains grouped development evidence for
`{family_id}`. Metrics are current-data grouped out-of-fold estimates, not
external validation.

- `summary/metrics.csv`: candidate metrics for this family.
- `predictions/development_oof_predictions.csv`: out-of-fold predictions.
- `diagnostics/`: input, feature, fold, source-sensitivity, and gate diagnostics.
""",
            encoding='utf-8',
        )


def _candidate_coverage(
    burden_output_dir: Path, metrics: pd.DataFrame, verdict: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for family_id, requirement in FIRST_CLASS_FAMILIES.items():
        family_metrics = metrics[metrics['family_id'] == family_id]
        selected = family_id == verdict.get('selected_family_id')
        rows.append(
            {
                'family_id': family_id,
                'subtree_path': f'burden_model/{family_id}',
                'required_or_exploratory': requirement,
                'run_status': 'evaluated'
                if not family_metrics.empty
                else 'hard_blocked',
                'candidate_ids': '|'.join(
                    sorted(family_metrics['candidate_id'].astype(str).tolist())
                ),
                'metrics_path': f'burden_model/{family_id}/summary/metrics.csv',
                'diagnostics_path': f'burden_model/{family_id}/diagnostics',
                'predictions_path': (
                    f'burden_model/{family_id}/predictions/'
                    'development_oof_predictions.csv'
                ),
                'gate_status': 'selected' if selected else 'not_selected',
                'selected': bool(selected),
                'failure_or_exclusion_reason': ''
                if selected
                else 'did_not_win_selector'
                if not family_metrics.empty
                else 'required_inputs_missing_or_no_estimable_features',
            }
        )
    return pd.DataFrame(rows)


def _write_final_model_if_supported(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    verdict: dict[str, Any],
    feature_sets: dict[str, list[str]],
    paths: dict[str, Path],
) -> None:
    if verdict['overall_status'] not in {
        'model_ready_pending_mr_tiff_deployment_smoke'
    }:
        _clear_final_model_artifacts(paths)
        return
    selected_id = verdict.get('selected_candidate_id', '')
    row = metrics[metrics['candidate_id'] == selected_id]
    if row.empty:
        return
    metric = row.iloc[0]
    columns = feature_sets.get(str(metric['feature_family']), [])
    if not columns:
        return
    x = to_finite_numeric_matrix(frame, columns)
    if str(metric['target_kind']) == 'severe':
        y = frame['adjudicated_severe'].astype(bool).to_numpy()
        model = _classifier(
            str(metric['model_kind']),
            x.shape[1],
            2,
            float(metric.get('regularization_c', 1.0)),
        )
        target_definition = 'adjudicated_score_gte_2_severe'
    else:
        target_column = (
            frame['score'].astype(float).map(_three_band)
            if str(metric['target_kind']) == 'three_band'
            else frame['score'].astype(float).map(_four_band)
        )
        y = target_column.astype(str).to_numpy()
        model = _classifier(
            str(metric['model_kind']),
            x.shape[1],
            len(np.unique(y)),
            float(metric.get('regularization_c', 1.0)),
        )
        target_definition = str(metric['target_kind'])

    def fit_final_model() -> None:
        model.fit(x, y)

    _, warning_result = capture_fit_warnings(fit_final_model)
    warning_records = warning_result.warning_messages
    paths['model'].mkdir(parents=True, exist_ok=True)
    with paths['final_model'].open('wb') as handle:
        pickle.dump(model, handle)
    metadata = {
        'selected_candidate_id': selected_id,
        'selected_family_id': verdict.get('selected_family_id'),
        'target_definition': target_definition,
        'feature_columns': columns,
        'regularization_c': float(metric.get('regularization_c', 1.0)),
        'operating_threshold': float(metric.get('threshold', 0.5))
        if pd.notna(metric.get('threshold', np.nan))
        else 0.5,
        'mr_computable_feature_schema': bool(metric.get('mr_computable', False)),
        'metric_label': GROUPED_DEVELOPMENT_METRIC_LABEL,
        'claim_boundary': verdict.get('claim_boundary'),
        'source_family_subtree': f'burden_model/{verdict.get("selected_family_id")}',
        'diagnostics_path': f'burden_model/{verdict.get("selected_family_id")}/diagnostics',
        'predictions_path': (
            f'burden_model/{verdict.get("selected_family_id")}/predictions/'
            'development_oof_predictions.csv'
        ),
        'refit_warning_status': 'warnings_recorded'
        if warning_records
        else 'no_warnings_recorded',
        'refit_warning_count': warning_result.warning_count,
        'refit_warning_messages': warning_records[:10],
    }
    _save_json(metadata, paths['final_model_metadata'])
    _save_json(
        {
            'feature_columns': columns,
            'required_identity_columns': IDENTITY_COLUMNS,
            'target_definition': target_definition,
            'input_unit': 'image_level_roi_or_aggregate_roi',
        },
        paths['inference_schema'],
    )
    training_predictions = frame[
        [column for column in IDENTITY_COLUMNS if column in frame.columns]
    ].copy()
    training_predictions['selected_candidate_id'] = selected_id
    training_predictions['prediction_source'] = 'final_model_full_current_data_refit'
    training_predictions.to_csv(paths['final_training_predictions'], index=False)
    pd.DataFrame(
        [
            {
                'status': 'not_run',
                'reason': 'mr_tiff_segmentation_to_quantification_path_not_proven',
            }
        ]
    ).to_csv(paths['deployment_smoke_predictions'], index=False)


def _write_reviews(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    verdict: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    del frame
    observed_severe = predictions.get(
        'observed_severe', pd.Series(False, index=predictions.index)
    ).astype(bool)
    predicted_severe = predictions.get(
        'predicted_severe', pd.Series(True, index=predictions.index)
    ).astype(bool)
    severe = predictions[observed_severe & (~predicted_severe)].head(50)
    html_rows = []
    for _, row in severe.iterrows():
        html_rows.append(
            '<tr>'
            f'<td>{escape(str(row.get("subject_image_id", "")))}</td>'
            f'<td>{escape(str(row.get("cohort_id", "")))}</td>'
            f'<td>{escape(str(row.get("score", "")))}</td>'
            f'<td>{escape(str(row.get("predicted_severe_probability", "")))}</td>'
            '</tr>'
        )
    if not html_rows:
        html_rows.append(
            '<tr><td colspan="4">No severe false negatives recorded.</td></tr>'
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>P3 Severe False Negative Review</title></head>
<body>
<h1>P3 Severe False Negative Review</h1>
<p>Verdict: {escape(str(verdict.get('overall_status')))}</p>
<table><thead><tr><th>Image</th><th>Cohort</th><th>Score</th><th>Probability</th></tr></thead>
<tbody>{''.join(html_rows)}</tbody></table>
</body></html>
"""
    paths['severe_false_negative_review'].write_text(html, encoding='utf-8')
    paths['error_review'].write_text(
        html.replace('Severe False Negative', 'Error'), encoding='utf-8'
    )
    paths['ordinal_confusion_review'].write_text(
        '<!doctype html><html><body><h1>P3 Ordinal Confusion Review</h1>'
        '<p>See predictions/development_oof_predictions.csv for candidate rows.</p>'
        '</body></html>',
        encoding='utf-8',
    )


def _write_index_and_summary(
    paths: dict[str, Path], verdict: dict[str, Any], metrics: pd.DataFrame
) -> None:
    selected = verdict.get('selected_candidate_id') or 'none'
    paths['index'].write_text(
        f"""# Endotheliosis Grade Model

Open `summary/final_product_verdict.md` first.

Current status: `{verdict['overall_status']}`.

Selected candidate: `{selected}`.

Current-data grouped development metrics are not external validation. README-facing
MR TIFF deployment language is blocked unless `readme_facing_deployment_allowed` is
true in `summary/final_product_verdict.json`.

Dox scored-no-mask smoke status: `{verdict.get('dox_scored_no_mask_smoke_status', 'not_run')}`.
""",
        encoding='utf-8',
    )
    paths['final_verdict_md'].write_text(
        f"""# P3 Final Product Verdict

Status: `{verdict['overall_status']}`

Selected candidate: `{selected}`

Claim boundary: {verdict['claim_boundary']}

MR TIFF deployment gate passed: `{verdict['mr_tiff_deployment_gate_passed']}`

Dox scored-no-mask smoke gate passed: `{verdict.get('dox_scored_no_mask_smoke_gate_passed', False)}`

Hard blockers:

{chr(10).join(f'- {item}' for item in verdict.get('hard_blockers', [])) or '- none'}
""",
        encoding='utf-8',
    )
    paths['executive_summary'].write_text(
        f"""# P3 Executive Summary

P3 reconstructed current scored ROI rows, generated deterministic subject-grouped
development folds, evaluated severe, ordinal, embedding-heavy, and aggregate-aware
candidate families, and wrote a selector verdict.

Final status: `{verdict['overall_status']}`.

Selected candidate: `{selected}`.

Candidate rows evaluated: {len(metrics)}.

Dox scored-no-mask smoke status: `{verdict.get('dox_scored_no_mask_smoke_status', 'not_run')}`.

The result is current-data and source-sensitive. It does not establish external
validation, causal evidence, or a README-facing MR TIFF deployment unless the MR
deployment gate passes.
""",
        encoding='utf-8',
    )


def _write_artifact_manifest(paths: dict[str, Path]) -> None:
    root = paths['root']
    records = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            records.append(
                {
                    'relative_path': str(path.relative_to(root)),
                    'role': 'p3_grade_model_artifact',
                    'consumer': 'runtime_review',
                    'required': True,
                    'reportability': 'current_data_internal',
                    'exists': True,
                }
            )
    _save_json(
        {
            'root': str(root),
            'manifest_complete': True,
            'artifact_count': len(records),
            'artifacts': records,
        },
        paths['artifact_manifest'],
    )


def _resolve_runtime_path(path: str | Path, runtime_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return runtime_root / resolved


def _infer_runtime_root_from_burden_output(burden_output_dir: Path) -> Path:
    burden_output_dir = Path(burden_output_dir).resolve()
    parts = burden_output_dir.parts
    if 'output' in parts:
        output_index = parts.index('output')
        if output_index > 0:
            return Path(*parts[:output_index])
    return burden_output_dir.parents[1]


def _write_dox_smoke_blocked(
    paths: dict[str, Path],
    *,
    status: str,
    reason: str,
    manifest_rows: int = 0,
    write_predictions: bool = True,
) -> dict[str, Any]:
    contract = {
        'stage': 'dox_scored_no_mask_smoke',
        'status': status,
        'passed': False,
        'reason': reason,
        'manifest_rows': int(manifest_rows),
    }
    _save_json(contract, paths['dox_smoke_contract'])
    pd.DataFrame([contract]).to_csv(paths['dox_smoke_summary'], index=False)
    if write_predictions:
        pd.DataFrame([contract]).to_csv(paths['dox_smoke_predictions'], index=False)
    paths['dox_smoke_report'].write_text(
        '<!doctype html><html><body>'
        '<h1>Dox scored-no-mask smoke</h1>'
        f'<p>Status: {escape(status)}</p>'
        f'<p>Reason: {escape(reason)}</p>'
        '</body></html>',
        encoding='utf-8',
    )
    return contract


def _build_union_mask_from_binary(
    binary_mask: np.ndarray, *, min_component_area: int
) -> dict[str, Any] | None:
    binary = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if not binary.any():
        return None
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components: list[dict[str, Any]] = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        components.append(
            {
                'label_index': label_index,
                'area': area,
                'mask': (labels == label_index).astype(np.uint8),
            }
        )
    if not components:
        return None
    union_mask = np.zeros_like(binary, dtype=np.uint8)
    for component in components:
        union_mask = np.maximum(union_mask, component['mask'])
    ys, xs = np.where(union_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    union_area = int(union_mask.sum())
    largest_component_area = max(int(component['area']) for component in components)
    return {
        'mask': union_mask,
        'bbox_x0': x0,
        'bbox_y0': y0,
        'bbox_x1': x1,
        'bbox_y1': y1,
        'component_count': int(len(components)),
        'component_selection': 'predicted_union_mask',
        'union_area': union_area,
        'largest_component_area_fraction': float(largest_component_area / union_area),
        'bbox_width': int(x1 - x0),
        'bbox_height': int(y1 - y0),
    }


def _extract_prediction_roi_row(
    *,
    source_row: pd.Series,
    image_path: Path,
    probability: np.ndarray,
    output_dir: Path,
    threshold: float,
    min_component_area: int,
    padding: int,
) -> dict[str, Any]:
    image_array = np.asarray(Image.open(image_path).convert('RGB'))
    core = create_prediction_core(expected_size=int(probability.shape[0]))
    resized_probability = core.resize_prediction_to_match(
        probability.astype(np.float32), image_array.shape[:2]
    )
    binary_mask = (resized_probability > float(threshold)).astype(np.uint8)
    union = _build_union_mask_from_binary(
        binary_mask, min_component_area=min_component_area
    )
    row: dict[str, Any] = {
        'subject_id': str(source_row.get('source_sample_id') or ''),
        'subject_image_id': str(source_row.get('manifest_row_id') or ''),
        'glomerulus_id': 1,
        'cohort_id': str(source_row.get('cohort_id') or 'vegfri_dox'),
        'score': float(source_row.get('score')),
        'raw_image_path': str(image_path),
        'segmentation_threshold': float(threshold),
        'segmentation_foreground_fraction': float(binary_mask.mean())
        if binary_mask.size
        else 0.0,
        'segmentation_probability_min': float(np.nanmin(resized_probability)),
        'segmentation_probability_max': float(np.nanmax(resized_probability)),
        'segmentation_probability_mean': float(np.nanmean(resized_probability)),
    }
    if union is None:
        row.update(
            {
                'roi_status': 'component_not_found',
                'roi_image_path': '',
                'roi_mask_path': '',
            }
        )
        return row

    roi_root = output_dir / 'dox_scored_no_mask_smoke_rois'
    image_crop_dir = roi_root / 'images'
    mask_crop_dir = roi_root / 'masks'
    image_crop_dir.mkdir(parents=True, exist_ok=True)
    mask_crop_dir.mkdir(parents=True, exist_ok=True)

    x0 = max(0, int(union['bbox_x0']) - padding)
    y0 = max(0, int(union['bbox_y0']) - padding)
    x1 = min(image_array.shape[1], int(union['bbox_x1']) + padding)
    y1 = min(image_array.shape[0], int(union['bbox_y1']) + padding)
    crop_image = image_array[y0:y1, x0:x1]
    crop_mask = (union['mask'][y0:y1, x0:x1] * 255).astype(np.uint8)
    crop_name = f'{row["subject_image_id"]}.png'
    image_crop_path = image_crop_dir / crop_name
    mask_crop_path = mask_crop_dir / crop_name
    Image.fromarray(crop_image).save(image_crop_path)
    Image.fromarray(crop_mask).save(mask_crop_path)

    gray_crop = np.asarray(Image.fromarray(crop_image).convert('L'))
    quant_metrics = calculate_quantification_metrics(crop_mask, gray_crop)
    row.update(
        {
            'roi_status': 'ok',
            'roi_image_path': str(image_crop_path),
            'roi_mask_path': str(mask_crop_path),
            'roi_bbox_x0': int(x0),
            'roi_bbox_y0': int(y0),
            'roi_bbox_x1': int(x1),
            'roi_bbox_y1': int(y1),
            'roi_area': int(union['union_area']),
            'roi_fill_fraction': float((crop_mask > 0).sum() / crop_mask.size)
            if crop_mask.size
            else 0.0,
            'roi_mean_intensity': float(gray_crop[crop_mask > 0].mean())
            if (crop_mask > 0).any()
            else 0.0,
            'roi_openness_score': float(quant_metrics.openness_score),
            'roi_component_count': int(union['component_count']),
            'roi_component_selection': str(union['component_selection']),
            'roi_union_bbox_width': int(union['bbox_width']),
            'roi_union_bbox_height': int(union['bbox_height']),
            'roi_largest_component_area_fraction': float(
                union['largest_component_area_fraction']
            ),
        }
    )
    return row


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _predict_tiled_segmentation_probability(
    *,
    model: Any,
    image: Image.Image,
    tile_size: int = 512,
    stride: int = 512,
    expected_size: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_array = np.asarray(image.convert('RGB'))
    height, width = image_array.shape[:2]
    probability = np.zeros((height, width), dtype=np.float32)
    tile_count = 0
    core = create_prediction_core(expected_size=expected_size)
    for y0 in _tile_starts(height, tile_size, stride):
        for x0 in _tile_starts(width, tile_size, stride):
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            tile = Image.fromarray(image_array[y0:y1, x0:x1]).convert('RGB')
            tile_probability, _audit = core.predict_segmentation_probability(
                model,
                tile,
                foreground_channel=1,
                imagenet_normalize=True,
            )
            tile_probability = core.resize_prediction_to_match(
                tile_probability.astype(np.float32), (y1 - y0, x1 - x0)
            )
            probability[y0:y1, x0:x1] = np.maximum(
                probability[y0:y1, x0:x1], tile_probability
            )
            tile_count += 1
    return probability, {
        'tiling_policy': 'overlapping_max_probability_merge',
        'tile_size': int(tile_size),
        'stride': int(stride),
        'expected_size': int(expected_size),
        'tile_count': int(tile_count),
        'source_height': int(height),
        'source_width': int(width),
        'probability_min': float(np.nanmin(probability)),
        'probability_max': float(np.nanmax(probability)),
        'probability_mean': float(np.nanmean(probability)),
    }


def _segmentation_inference_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _dox_smoke_threshold_curve(ok_frame: pd.DataFrame) -> pd.DataFrame:
    if ok_frame.empty:
        return pd.DataFrame(
            columns=[
                'threshold',
                'predicted_severe_rows',
                'true_positive_rows',
                'false_positive_rows',
                'false_negative_rows',
                'precision',
                'recall',
            ]
        )
    y_true = ok_frame['observed_severe'].astype(bool).to_numpy()
    probability = ok_frame['predicted_severe_probability'].astype(float).to_numpy()
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 101),
                probability[np.isfinite(probability)],
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = probability >= float(threshold)
        true_positive = int((predicted & y_true).sum())
        false_positive = int((predicted & ~y_true).sum())
        false_negative = int((~predicted & y_true).sum())
        predicted_count = int(predicted.sum())
        rows.append(
            {
                'threshold': float(threshold),
                'predicted_severe_rows': predicted_count,
                'true_positive_rows': true_positive,
                'false_positive_rows': false_positive,
                'false_negative_rows': false_negative,
                'precision': float(true_positive / predicted_count)
                if predicted_count
                else np.nan,
                'recall': float(true_positive / max(1, int(y_true.sum()))),
            }
        )
    return pd.DataFrame(rows).sort_values('threshold')


def _image_tag(path: Any, label: str) -> str:
    if pd.isna(path) or not str(path):
        return '<div class="missing">missing</div>'
    return (
        '<figure>'
        f'<img src="{escape(str(path))}" alt="{escape(label)}">'
        f'<figcaption>{escape(label)}</figcaption>'
        '</figure>'
    )


def _dox_review_section(title: str, rows: pd.DataFrame, limit: int = 24) -> str:
    cards = []
    for _, row in rows.head(limit).iterrows():
        cards.append(
            '<article class="case">'
            '<h3>'
            f'{escape(str(row.get("review_priority", "")))}: '
            f'{escape(str(row.get("subject_image_id", "")))}'
            '</h3>'
            '<div class="media">'
            f'{_image_tag(row.get("roi_image_path"), "ROI")}'
            f'{_image_tag(row.get("roi_mask_path"), "Predicted mask")}'
            '</div>'
            '<table>'
            f'<tr><th>Review bucket</th><td>{escape(str(row.get("review_bucket", "")))}</td></tr>'
            f'<tr><th>Human score</th><td>{escape(str(row.get("score", "")))}</td></tr>'
            f'<tr><th>Observed severe</th><td>{escape(str(row.get("observed_severe", "")))}</td></tr>'
            f'<tr><th>Predicted severe</th><td>{escape(str(row.get("predicted_severe", "")))}</td></tr>'
            '<tr><th>Risk</th><td>'
            f'{escape(format(float(row.get("predicted_severe_probability", np.nan)), ".3f"))}'
            '</td></tr>'
            f'<tr><th>ROI status</th><td>{escape(str(row.get("roi_status", "")))}</td></tr>'
            f'<tr><th>Components</th><td>{escape(str(row.get("roi_component_count", "")))}</td></tr>'
            '</table>'
            '</article>'
        )
    if not cards:
        cards.append('<p>No rows in this bucket.</p>')
    return f'<section><h2>{escape(title)}</h2>{"".join(cards)}</section>'


def _write_dox_smoke_review(
    *,
    predictions: pd.DataFrame,
    threshold_curve: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    ok = predictions[predictions['roi_status'].astype(str).eq('ok')].copy()
    probability = ok.get(
        'predicted_severe_probability', pd.Series(np.nan, index=ok.index)
    ).astype(float)
    observed = ok.get('observed_severe', pd.Series(False, index=ok.index)).astype(bool)
    predicted = ok.get('predicted_severe', pd.Series(False, index=ok.index)).astype(bool)
    true_severe = ok[observed].sort_values(
        'predicted_severe_probability', ascending=False
    )
    false_positive = ok[predicted & ~observed].sort_values(
        'predicted_severe_probability', ascending=False
    )
    false_negative = ok[observed & ~predicted].sort_values(
        'predicted_severe_probability'
    )
    low_risk_true_negative = ok[(~predicted) & (~observed)].sort_values(
        'predicted_severe_probability'
    )
    missing = predictions[~predictions['roi_status'].astype(str).eq('ok')]
    candidate_thresholds = threshold_curve[
        threshold_curve['recall'].fillna(0.0) >= 0.9
    ].sort_values(['predicted_severe_rows', 'threshold'], ascending=[True, False])
    threshold_rows = []
    for _, row in candidate_thresholds.head(12).iterrows():
        threshold_rows.append(
            '<tr>'
            f'<td>{float(row["threshold"]):.3f}</td>'
            f'<td>{int(row["predicted_severe_rows"])}</td>'
            f'<td>{int(row["false_positive_rows"])}</td>'
            f'<td>{int(row["false_negative_rows"])}</td>'
            f'<td>{float(row["recall"]):.3f}</td>'
            f'<td>{float(row["precision"]):.3f}</td>'
            '</tr>'
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Dox scored-no-mask smoke review</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; }}
.summary, table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
.case {{ border: 1px solid #ccc; padding: 12px; margin: 12px 0; }}
.media {{ display: flex; gap: 12px; flex-wrap: wrap; }}
figure {{ margin: 0; }}
img {{ max-width: 360px; max-height: 260px; border: 1px solid #ddd; object-fit: contain; }}
figcaption {{ font-size: 12px; color: #555; }}
</style></head><body>
<h1>Dox scored-no-mask smoke review</h1>
<table class="summary">
<tr><th>Status</th><td>{escape(str(summary.get('status')))}</td></tr>
<tr><th>Accepted ROIs</th><td>{summary.get('accepted_roi_rows')} / {summary.get('manifest_rows')}</td></tr>
<tr><th>Observed severe</th><td>{summary.get('observed_severe_rows')}</td></tr>
<tr><th>Predicted severe</th><td>{summary.get('predicted_severe_rows')}</td></tr>
<tr><th>False negatives</th><td>{summary.get('false_negative_rows')}</td></tr>
<tr><th>Recall</th><td>{summary.get('severe_recall')}</td></tr>
<tr><th>Precision</th><td>{summary.get('severe_precision')}</td></tr>
<tr><th>Reason</th><td>{escape(str(summary.get('reason')))}</td></tr>
</table>
<h2>Thresholds With Recall At Least 0.90</h2>
<table><thead><tr><th>Threshold</th><th>Flagged</th><th>False positives</th><th>False negatives</th><th>Recall</th><th>Precision</th></tr></thead>
<tbody>{''.join(threshold_rows) or '<tr><td colspan="6">No threshold reaches recall 0.90.</td></tr>'}</tbody></table>
{_dox_review_section('Human severe examples', true_severe)}
{_dox_review_section('Highest-risk false positives', false_positive)}
{_dox_review_section('False negatives at operating threshold', false_negative)}
{_dox_review_section('Segmentation misses', missing)}
{_dox_review_section('Lowest-risk true negatives', low_risk_true_negative)}
</body></html>
"""
    paths['dox_smoke_report'].write_text(html, encoding='utf-8')


def _dox_overcall_feature_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.copy()


def _bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    true_values = {'true', '1', 'yes', 'y'}
    false_values = {'false', '0', 'no', 'n', 'nan', 'none', ''}
    return normalized.map(
        lambda value: True
        if value in true_values
        else False
        if value in false_values
        else default
    ).astype(bool)


def _assign_k_center_clusters(x_scaled: np.ndarray, cluster_count: int) -> np.ndarray:
    if cluster_count <= 1 or len(x_scaled) == 0:
        return np.zeros(len(x_scaled), dtype=int)
    distances_to_median = np.linalg.norm(x_scaled - np.median(x_scaled, axis=0), axis=1)
    center_indices = [int(np.argmin(distances_to_median))]
    while len(center_indices) < cluster_count:
        centers = x_scaled[center_indices]
        distances = np.linalg.norm(x_scaled[:, None, :] - centers[None, :, :], axis=2)
        nearest_distance = distances.min(axis=1)
        nearest_distance[center_indices] = -1.0
        center_indices.append(int(np.argmax(nearest_distance)))
    centers = x_scaled[center_indices]
    distances = np.linalg.norm(x_scaled[:, None, :] - centers[None, :, :], axis=2)
    return distances.argmin(axis=1).astype(int)


def _dox_overcall_triage_queue(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _dox_overcall_feature_frame(predictions)
    frame['review_bucket'] = ''
    frame['review_priority'] = np.nan
    predicted_severe = _bool_series(frame, 'predicted_severe')
    observed_severe = _bool_series(frame, 'observed_severe')
    false_positive_mask = (
        frame['roi_status'].astype(str).eq('ok')
        & predicted_severe
        & ~observed_severe
    )
    false_positive = frame[false_positive_mask].copy()
    review_rows: list[pd.DataFrame] = []
    if not false_positive.empty:
        feature_columns = [
            column
            for column in [
                *ROI_QC_COLUMNS,
                *MORPHOLOGY_FEATURE_COLUMNS,
                'segmentation_foreground_fraction',
                'segmentation_probability_mean',
                'segmentation_probability_max',
            ]
            if column in false_positive.columns
        ]
        x = to_finite_numeric_matrix(false_positive, feature_columns, finite_bound=1e6)
        cluster_count = min(12, max(1, len(false_positive) // 8), len(false_positive))
        if cluster_count > 1:
            x_mean = x.mean(axis=0)
            x_std = x.std(axis=0)
            x_scaled = np.divide(
                x - x_mean,
                np.where(x_std > 0, x_std, 1.0),
                out=np.zeros_like(x),
                where=np.isfinite(x),
            )
            x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
            x_scaled = np.clip(x_scaled, -10.0, 10.0)
            labels = _assign_k_center_clusters(x_scaled, cluster_count)
        else:
            labels = np.zeros(len(false_positive), dtype=int)
        false_positive['overcall_cluster'] = labels
        cluster_representatives = []
        for cluster_id, cluster_df in false_positive.groupby('overcall_cluster'):
            cluster_df = cluster_df.copy()
            cluster_df['review_bucket'] = 'cluster_representative_false_positive'
            cluster_df['review_priority'] = 10 + int(cluster_id)
            representative = cluster_df.sort_values(
                'predicted_severe_probability', ascending=False
            ).head(1)
            representative['cluster_size'] = len(cluster_df)
            cluster_representatives.append(representative)
        review_rows.extend(cluster_representatives)
        high_conf = false_positive.sort_values(
            'predicted_severe_probability', ascending=False
        ).head(18)
        high_conf = high_conf.copy()
        high_conf['review_bucket'] = 'highest_confidence_false_positive'
        high_conf['review_priority'] = range(100, 100 + len(high_conf))
        review_rows.append(high_conf)
        boundary = false_positive.assign(
            boundary_distance=(
                false_positive['predicted_severe_probability'].astype(float) - 0.5
            ).abs()
        ).sort_values('boundary_distance').head(10)
        boundary = boundary.copy()
        boundary['review_bucket'] = 'threshold_boundary_false_positive'
        boundary['review_priority'] = range(200, 200 + len(boundary))
        review_rows.append(boundary)
    missing = frame[~frame['roi_status'].astype(str).eq('ok')].copy()
    if not missing.empty:
        missing['review_bucket'] = 'segmentation_miss'
        missing['review_priority'] = range(300, 300 + len(missing))
        review_rows.append(missing)
    true_severe = frame[observed_severe].copy()
    if not true_severe.empty:
        true_severe['review_bucket'] = 'human_severe_reference'
        true_severe['review_priority'] = range(400, 400 + len(true_severe))
        review_rows.append(true_severe)
    if not review_rows:
        return pd.DataFrame()
    queue = pd.concat(review_rows, ignore_index=True, sort=False)
    queue = queue.drop_duplicates(subset=['subject_image_id', 'review_bucket'])
    queue['reviewer_score'] = ''
    queue['reviewer_roi_usable'] = ''
    queue['reviewer_overcall_reason'] = ''
    queue['reviewer_action'] = ''
    queue['reviewer_notes'] = ''
    preferred_columns = [
        'review_priority',
        'review_bucket',
        'overcall_cluster',
        'cluster_size',
        'subject_id',
        'subject_image_id',
        'score',
        'observed_severe',
        'predicted_severe',
        'predicted_severe_probability',
        'roi_status',
        'segmentation_foreground_fraction',
        'roi_component_count',
        'roi_area',
        'roi_fill_fraction',
        'roi_openness_score',
        'reviewer_score',
        'reviewer_roi_usable',
        'reviewer_overcall_reason',
        'reviewer_action',
        'reviewer_notes',
        'morph_pale_lumen_area_fraction',
        'morph_slit_like_area_fraction',
        'morph_rbc_like_color_burden',
        'raw_image_path',
        'roi_image_path',
        'roi_mask_path',
    ]
    columns = [column for column in preferred_columns if column in queue.columns]
    return queue.sort_values('review_priority')[columns]


DOX_REVIEWER_COLUMNS = [
    'reviewer_score',
    'reviewer_roi_usable',
    'reviewer_overcall_reason',
    'reviewer_action',
    'reviewer_notes',
]


def _clear_final_model_artifacts(paths: dict[str, Path]) -> None:
    for stale_path in [
        paths['final_model'],
        paths['final_model_metadata'],
        paths['inference_schema'],
        paths['deployment_smoke_predictions'],
        paths['final_training_predictions'],
    ]:
        if stale_path.exists():
            stale_path.unlink()


def _preserve_existing_dox_review_annotations(
    queue: pd.DataFrame, existing_queue_path: Path
) -> pd.DataFrame:
    if not existing_queue_path.exists() or queue.empty:
        return queue
    existing = pd.read_csv(existing_queue_path)
    key_columns = ['subject_image_id', 'review_bucket']
    if not all(column in existing.columns for column in key_columns):
        return queue
    reviewer_columns = [
        column
        for column in DOX_REVIEWER_COLUMNS
        if column in existing.columns and column in queue.columns
    ]
    if not reviewer_columns:
        return queue
    annotation_source = existing[key_columns + reviewer_columns].drop_duplicates(
        subset=key_columns, keep='last'
    )
    merged = queue.drop(columns=reviewer_columns).merge(
        annotation_source,
        on=key_columns,
        how='left',
        validate='many_to_one',
    )
    for column in reviewer_columns:
        merged[column] = merged[column].fillna('')
    return merged[queue.columns]


def _write_dox_overcall_review_diagnostic(paths: dict[str, Path]) -> dict[str, Any]:
    queue_path = paths['dox_overcall_triage_queue']
    if not queue_path.exists():
        diagnostic = {
            'status': 'not_available',
            'reason': 'dox_overcall_triage_queue_missing',
            'overcall_confirmed': False,
        }
        _save_json(diagnostic, paths['dox_overcall_review_diagnostic'])
        return diagnostic
    queue = pd.read_csv(queue_path)
    cluster = queue[
        queue.get('review_bucket', pd.Series('', index=queue.index))
        .astype(str)
        .eq('cluster_representative_false_positive')
    ].copy()
    if cluster.empty:
        diagnostic = {
            'status': 'not_available',
            'reason': 'no_cluster_representative_false_positive_rows',
            'overcall_confirmed': False,
        }
        _save_json(diagnostic, paths['dox_overcall_review_diagnostic'])
        return diagnostic
    cluster['reviewer_score_numeric'] = pd.to_numeric(
        cluster.get('reviewer_score', pd.Series(np.nan, index=cluster.index)),
        errors='coerce',
    )
    cluster['reviewer_roi_usable_normalized'] = (
        cluster.get('reviewer_roi_usable', pd.Series('', index=cluster.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    reviewed = cluster[
        cluster['reviewer_score_numeric'].notna()
        | cluster['reviewer_roi_usable_normalized'].isin(['yes', 'no', 'uncertain'])
    ].copy()
    reviewed['reviewer_severe'] = (
        reviewed['reviewer_score_numeric'] >= PRIMARY_SEVERE_THRESHOLD
    )
    reviewed['reviewer_roi_usable_yes'] = reviewed[
        'reviewer_roi_usable_normalized'
    ].eq('yes')
    reviewed['reviewer_roi_usable_or_uncertain'] = reviewed[
        'reviewer_roi_usable_normalized'
    ].isin(['yes', 'uncertain'])
    usable_yes = reviewed[reviewed['reviewer_roi_usable_yes']]
    usable_or_uncertain = reviewed[reviewed['reviewer_roi_usable_or_uncertain']]
    usable_yes_nonsevere = usable_yes[~usable_yes['reviewer_severe']]
    usable_or_uncertain_nonsevere = usable_or_uncertain[
        ~usable_or_uncertain['reviewer_severe']
    ]
    reviewed[
        [
            column
            for column in [
                'review_priority',
                'review_bucket',
                'subject_image_id',
                'score',
                'predicted_severe_probability',
                'reviewer_score',
                'reviewer_roi_usable',
                'reviewer_severe',
                'roi_status',
                'raw_image_path',
                'roi_image_path',
                'roi_mask_path',
            ]
            if column in reviewed.columns
        ]
    ].to_csv(paths['dox_overcall_review_interpretation'], index=False)
    usable_yes_count = int(len(usable_yes))
    usable_or_uncertain_count = int(len(usable_or_uncertain))
    overcall_confirmed = bool(
        len(reviewed) >= 8
        and usable_yes_count >= 5
        and int(usable_yes['reviewer_severe'].sum()) == 0
        and (len(usable_yes_nonsevere) / max(1, usable_yes_count)) >= 0.75
    )
    diagnostic = {
        'status': 'reviewed' if len(reviewed) else 'not_reviewed',
        'reviewed_cluster_representative_rows': int(len(reviewed)),
        'usable_yes_rows': usable_yes_count,
        'usable_or_uncertain_rows': usable_or_uncertain_count,
        'unusable_no_rows': int(
            reviewed['reviewer_roi_usable_normalized'].eq('no').sum()
        ),
        'reviewer_severe_rows': int(reviewed['reviewer_severe'].sum()),
        'reviewer_severe_usable_yes_rows': int(
            usable_yes['reviewer_severe'].sum()
        ),
        'reviewer_nonsevere_usable_yes_rows': int(len(usable_yes_nonsevere)),
        'reviewer_nonsevere_usable_or_uncertain_rows': int(
            len(usable_or_uncertain_nonsevere)
        ),
        'overcall_confirmed': overcall_confirmed,
        'interpretation': (
            'reviewed_cluster_representatives_confirm_usable_nonsevere_overcalls'
            if overcall_confirmed
            else 'reviewed_cluster_representatives_do_not_confirm_overcall_blocker'
        ),
        'selection_action': (
            'reject_current_selected_candidate_as_dox_overcaller'
            if overcall_confirmed
            else 'no_selection_change'
        ),
        'review_interpretation_path': str(paths['dox_overcall_review_interpretation']),
    }
    _save_json(diagnostic, paths['dox_overcall_review_diagnostic'])
    return diagnostic


def _write_dox_overcall_triage_report(
    *, predictions: pd.DataFrame, paths: dict[str, Path]
) -> None:
    queue = _dox_overcall_triage_queue(predictions)
    queue = _preserve_existing_dox_review_annotations(
        queue, paths['dox_overcall_triage_queue']
    )
    queue.to_csv(paths['dox_overcall_triage_queue'], index=False)
    _write_dox_overcall_review_diagnostic(paths)
    section = _dox_review_section('Rows in CSV order', queue, limit=len(queue))
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Dox overcall triage</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; }}
td, th {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
table {{ border-collapse: collapse; }}
.case {{ border: 1px solid #ccc; padding: 12px; margin: 12px 0; }}
.media {{ display: flex; gap: 12px; flex-wrap: wrap; }}
figure {{ margin: 0; }}
img {{ max-width: 360px; max-height: 260px; border: 1px solid #ddd; object-fit: contain; }}
figcaption {{ font-size: 12px; color: #555; }}
</style></head><body>
	<h1>Dox overcall triage</h1>
	<p>Queue rows: {len(queue)}. This HTML is rendered in the same order as the CSV: sort by review_priority ascending.</p>
	<p>Fill reviewer_score with 0, 0.5, 1, 1.5, 2, or 3. Fill reviewer_roi_usable with yes/no/uncertain. Fill reviewer_overcall_reason with one of: true_severe_label_should_increase, nonsevere_model_overcall, poor_segmentation_or_crop, artifact_or_edge, uncertain. Fill reviewer_action with one of: keep_as_false_positive, update_label, exclude_bad_roi, needs_second_review.</p>
	{section if not queue.empty else '<p>No overcall triage rows were generated.</p>'}
</body></html>
"""
    paths['dox_overcall_triage_report'].write_text(html, encoding='utf-8')


def _run_dox_scored_no_mask_smoke(
    *,
    paths: dict[str, Path],
    burden_output_dir: Path,
    manifest_root: Path | None,
    segmentation_model_path: Path | None,
    threshold: float = 0.75,
    min_component_area: int = 64,
    padding: int = 32,
) -> dict[str, Any]:
    if manifest_root is None:
        return _write_dox_smoke_blocked(
            paths,
            status='not_run',
            reason='manifest_root_not_provided',
        )
    runtime_root = _infer_runtime_root_from_burden_output(burden_output_dir)
    manifest_root = _resolve_runtime_path(manifest_root, runtime_root)
    smoke_manifest_path = (
        manifest_root
        / 'vegfri_dox'
        / 'metadata'
        / 'dox_scored_no_mask_smoke_manifest.csv'
    )
    if not smoke_manifest_path.exists():
        return _write_dox_smoke_blocked(
            paths,
            status='not_run',
            reason='dox_scored_no_mask_smoke_manifest_missing',
        )
    smoke_manifest = pd.read_csv(smoke_manifest_path)
    smoke_manifest.to_csv(paths['dox_smoke_manifest'], index=False)
    if smoke_manifest.empty:
        return _write_dox_smoke_blocked(
            paths,
            status='not_run',
            reason='dox_scored_no_mask_smoke_manifest_empty',
        )
    if segmentation_model_path is None:
        return _write_dox_smoke_blocked(
            paths,
            status='blocked',
            reason='segmentation_model_path_not_provided',
            manifest_rows=len(smoke_manifest),
        )
    segmentation_model_path = _resolve_runtime_path(segmentation_model_path, runtime_root)
    required_model_paths = [
        paths['final_model'],
        paths['final_model_metadata'],
        paths['inference_schema'],
        segmentation_model_path,
    ]
    missing = [str(path) for path in required_model_paths if not path.exists()]
    if missing:
        return _write_dox_smoke_blocked(
            paths,
            status='blocked',
            reason=f'missing_required_model_artifacts: {"|".join(missing)}',
            manifest_rows=len(smoke_manifest),
        )

    schema = _read_json(paths['inference_schema'])
    feature_columns = list(schema.get('feature_columns') or [])
    learner = load_model_safely(str(segmentation_model_path), model_type='glomeruli')
    inference_device = _segmentation_inference_device()
    learner.model.to(inference_device)
    learner.model.eval()
    rows: list[dict[str, Any]] = []
    prediction_audits: list[dict[str, Any]] = []
    for source_row in smoke_manifest.to_dict(orient='records'):
        row_series = pd.Series(source_row)
        image_path = _resolve_runtime_path(str(source_row['image_path']), runtime_root)
        if not image_path.exists():
            rows.append(
                {
                    'subject_image_id': str(source_row.get('manifest_row_id') or ''),
                    'score': source_row.get('score'),
                    'roi_status': 'image_missing',
                    'raw_image_path': str(image_path),
                }
            )
            continue
        image = Image.open(image_path).convert('RGB')
        probability, audit = _predict_tiled_segmentation_probability(
            model=learner.model,
            image=image,
            tile_size=512,
            stride=512,
            expected_size=256,
        )
        prediction_audits.append(audit)
        rows.append(
            _extract_prediction_roi_row(
                source_row=row_series,
                image_path=image_path,
                probability=probability,
                output_dir=paths['deployment'],
                threshold=threshold,
                min_component_area=min_component_area,
                padding=padding,
            )
        )
    feature_frame = pd.DataFrame(rows)
    ok_frame = feature_frame[feature_frame['roi_status'].astype(str).eq('ok')].copy()
    if ok_frame.empty:
        feature_frame.to_csv(paths['dox_smoke_predictions'], index=False)
        return _write_dox_smoke_blocked(
            paths,
            status='failed',
            reason='segmentation_produced_no_accepted_rois',
            manifest_rows=len(smoke_manifest),
            write_predictions=False,
        )

    with paths['final_model'].open('rb') as handle:
        final_model = pickle.load(handle)
    x = to_finite_numeric_matrix(ok_frame, feature_columns)
    if hasattr(final_model, 'predict_proba'):
        probabilities = final_model.predict_proba(x)
        classes = list(getattr(final_model, 'classes_', []))
        positive_index = classes.index(True) if True in classes else -1
        ok_frame['predicted_severe_probability'] = probabilities[:, positive_index]
    elif hasattr(final_model, 'decision_function'):
        scores = final_model.decision_function(x)
        ok_frame['predicted_severe_probability'] = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise EndotheliosisGradeModelError(
            'Final P3 model does not expose predict_proba or decision_function.'
        )
    metadata = _read_json(paths['final_model_metadata'])
    threshold_value = float(
        metadata.get('operating_threshold')
        or metadata.get('selected_threshold')
        or 0.5
    )
    ok_frame['predicted_severe'] = (
        ok_frame['predicted_severe_probability'].astype(float) >= threshold_value
    )
    ok_frame['observed_severe'] = ok_frame['score'].astype(float) >= PRIMARY_SEVERE_THRESHOLD
    ok_frame['absolute_score_distance_to_severe_boundary'] = (
        ok_frame['score'].astype(float) - PRIMARY_SEVERE_THRESHOLD
    ).abs()
    predictions = feature_frame.merge(
        ok_frame[
            [
                'subject_image_id',
                'predicted_severe_probability',
                'predicted_severe',
                'observed_severe',
                'absolute_score_distance_to_severe_boundary',
            ]
        ],
        on='subject_image_id',
        how='left',
    )
    predictions.to_csv(paths['dox_smoke_predictions'], index=False)
    threshold_curve = _dox_smoke_threshold_curve(ok_frame)
    threshold_curve.to_csv(paths['dox_smoke_threshold_curve'], index=False)
    total = int(len(predictions))
    ok_count = int(predictions['roi_status'].astype(str).eq('ok').sum())
    observed_severe = ok_frame['observed_severe'].astype(bool)
    predicted_severe = ok_frame['predicted_severe'].astype(bool)
    severe_count = int(observed_severe.sum())
    false_negative_count = int((observed_severe & ~predicted_severe).sum())
    recall = (
        float(recall_score(observed_severe, predicted_severe, zero_division=0))
        if severe_count
        else np.nan
    )
    precision = float(
        precision_score(observed_severe, predicted_severe, zero_division=0)
    )
    severe_precision_gate = (
        bool(precision >= 0.15) if severe_count else bool(predicted_severe.sum() == 0)
    )
    failure_reasons = []
    if ok_count != total:
        failure_reasons.append('segmentation_missing_accepted_rois')
    if severe_count and false_negative_count:
        failure_reasons.append('dox_smoke_severe_false_negatives')
    if not severe_precision_gate:
        failure_reasons.append('dox_smoke_severe_precision_below_0.15')
    passed = bool(ok_count == total and severe_precision_gate and false_negative_count == 0)
    summary = {
        'stage': 'dox_scored_no_mask_smoke',
        'status': 'passed' if passed else 'failed',
        'passed': passed,
        'manifest_rows': total,
        'accepted_roi_rows': ok_count,
        'segmentation_model_path': str(segmentation_model_path),
        'segmentation_inference_device': str(inference_device),
        'segmentation_threshold': float(threshold),
        'selected_candidate_id': metadata.get('selected_candidate_id', ''),
        'feature_columns': '|'.join(feature_columns),
        'prediction_threshold': threshold_value,
        'human_grade_comparison': 'image_level_score_gte_2_vs_predicted_severe',
        'observed_severe_rows': severe_count,
        'predicted_severe_rows': int(predicted_severe.sum()),
        'false_negative_rows': false_negative_count,
        'severe_recall': recall,
        'severe_precision': precision,
        'reason': '' if passed else '|'.join(failure_reasons),
    }
    pd.DataFrame([summary]).to_csv(paths['dox_smoke_summary'], index=False)
    _save_json(
        {
            **summary,
            'threshold_curve_path': str(paths['dox_smoke_threshold_curve']),
            'prediction_audits': prediction_audits[:5],
        },
        paths['dox_smoke_contract'],
    )
    _write_dox_smoke_review(
        predictions=predictions,
        threshold_curve=threshold_curve,
        summary=summary,
        paths=paths,
    )
    _write_dox_overcall_triage_report(predictions=predictions, paths=paths)
    return summary


def evaluate_endotheliosis_grade_model(
    embedding_df: pd.DataFrame,
    burden_output_dir: Path,
    n_splits: int = 3,
    change_dir: Path | None = None,
    manifest_root: Path | None = None,
    segmentation_model_path: Path | None = None,
) -> dict[str, Path]:
    """Evaluate the P3 grade-model selector and write contained artifacts."""
    del change_dir
    burden_output_dir = Path(burden_output_dir)
    paths = grade_model_output_paths(burden_output_dir)
    _prepare_dirs(paths, burden_output_dir)

    input_index = _input_artifacts(burden_output_dir)
    _save_json(input_index, paths['input_artifact_index'])
    frame, feature_diagnostics, hard_blockers = _reconstruct_feature_frame(
        embedding_df, burden_output_dir
    )
    folds, fold_diagnostics = _deterministic_development_folds(frame, n_splits)
    folds.to_csv(paths['development_folds'], index=False)
    feature_sets = _feature_sets(frame)
    feature_diagnostics['feature_sets'] = {
        key: {'column_count': len(value), 'columns': value}
        for key, value in feature_sets.items()
    }
    feature_diagnostics['fold_diagnostics'] = fold_diagnostics
    _save_json(feature_diagnostics, paths['feature_diagnostics'])

    specs = _candidate_specs(feature_sets)
    metrics, predictions = _evaluate_candidates(frame, folds, specs)
    baseline = pd.DataFrame(_baseline_metrics(frame))
    metrics = pd.concat([baseline, metrics], ignore_index=True, sort=False)
    metrics.to_csv(paths['candidate_metrics'], index=False)
    predictions.to_csv(paths['development_oof_predictions'], index=False)
    _save_json(
        {
            'candidate_specs': [
                {
                    'candidate_id': spec.candidate_id,
                    'family_id': spec.family_id,
                    'target_kind': spec.target_kind,
                    'feature_family': spec.feature_family,
                    'feature_columns': list(spec.feature_columns),
                    'model_kind': spec.model_kind,
                    'threshold_target': spec.threshold_target,
                }
                for spec in specs
            ],
            'threshold_selection': 'training_fold_only',
            'dimensionality_control': (
                'sklearn_pipeline_feature_selection_fit_inside_training_fold'
            ),
            'standardization_transform': (
                'StandardScaler fit inside training fold, followed by finite '
                'clipping to [-20, 20] before model fitting'
            ),
        },
        paths['candidate_configs'],
    )

    verdict = _select_product(metrics, hard_blockers)
    if verdict['overall_status'] not in FINAL_STATUS_VALUES:
        raise EndotheliosisGradeModelError(
            f'Unsupported final status: {verdict["overall_status"]}'
        )
    _write_family_artifacts(
        burden_output_dir, metrics, predictions, feature_diagnostics, hard_blockers
    )
    coverage = _candidate_coverage(burden_output_dir, metrics, verdict)
    coverage.to_csv(paths['candidate_coverage_matrix'], index=False)
    metrics.to_csv(paths['model_selection_table'], index=False)
    metrics.to_csv(paths['development_oof_metrics'], index=False)
    _save_json(
        {
            'decision': 'ordinal_bands_deployable'
            if verdict.get('selected_output_type') == 'ordinal_grade_band'
            else 'ordinal_diagnostic_only'
            if 'ordinal' in str(verdict.get('selected_output_type'))
            else 'ordinal_unsupported_or_not_selected',
            'three_band_labels': THREE_BAND_LABELS,
            'four_band_labels': FOUR_BAND_LABELS,
            'metric_label': 'grouped_out_of_fold_development_estimate',
        },
        paths['ordinal_feasibility'],
    )
    threshold_metrics = metrics[metrics['target_kind'] == 'severe'].copy()
    threshold_metrics.to_json(
        paths['severe_threshold_selection'], orient='records', indent=2
    )
    _save_json(
        {
            'events': [
                {
                    'event': 'candidate_evaluated',
                    'candidate_id': str(row['candidate_id']),
                    'family_id': str(row['family_id']),
                    'target_kind': str(row['target_kind']),
                }
                for _, row in metrics.iterrows()
            ],
            'stop_rule': verdict['overall_status'],
        },
        paths['autonomous_loop_log'],
    )
    _save_json(
        {
            'selector_status': verdict['overall_status'],
            'metric_label': 'grouped_out_of_fold_development_estimate',
            'no_external_validation_claim': True,
            'no_internal_locked_test_split': True,
        },
        paths['selector_diagnostics'],
    )
    _save_json(coverage.to_dict(orient='records'), paths['family_gate_diagnostics'])
    _write_final_model_if_supported(frame, metrics, verdict, feature_sets, paths)
    dox_smoke = _run_dox_scored_no_mask_smoke(
        paths=paths,
        burden_output_dir=burden_output_dir,
        manifest_root=manifest_root,
        segmentation_model_path=segmentation_model_path,
    )
    verdict['dox_scored_no_mask_smoke_gate_passed'] = bool(
        dox_smoke.get('passed', False)
    )
    verdict['dox_scored_no_mask_smoke_status'] = str(
        dox_smoke.get('status', 'not_run')
    )
    dox_overcall_review = _write_dox_overcall_review_diagnostic(paths)
    verdict['dox_overcall_review_status'] = str(
        dox_overcall_review.get('status', 'not_available')
    )
    verdict['dox_overcall_confirmed'] = bool(
        dox_overcall_review.get('overcall_confirmed', False)
    )
    if (
        verdict.get('overall_status') == 'model_ready_pending_mr_tiff_deployment_smoke'
        and not bool(dox_smoke.get('passed', False))
        and str(dox_smoke.get('reason', '')) != 'manifest_root_not_provided'
    ):
        verdict['hard_blockers'] = list(
            dict.fromkeys(
                [
                    *verdict.get('hard_blockers', []),
                    'dox_scored_no_mask_smoke_not_passed',
                    'mr_tiff_deployment_blocked_until_dox_smoke_passes',
                ]
            )
        )
    if (
        verdict.get('overall_status') == 'model_ready_pending_mr_tiff_deployment_smoke'
        and bool(dox_overcall_review.get('overcall_confirmed', False))
    ):
        verdict['overall_status'] = 'diagnostic_only_current_data_model'
        verdict['quantification_gate_passed'] = False
        verdict['severe_safety_gate_passed'] = False
        verdict['selected_output_type'] = 'severe_risk_triage_diagnostic_only'
        verdict['claim_boundary'] = (
            'Dox reviewed cluster representatives confirm usable non-severe '
            'overcalls; current selected severe-risk model is diagnostic only.'
        )
        verdict['hard_blockers'] = list(
            dict.fromkeys(
                [
                    *verdict.get('hard_blockers', []),
                    'dox_review_confirmed_selected_candidate_overcalls_nonsevere_usable_rois',
                    'selected_severe_triage_candidate_rejected_after_dox_review',
                ]
            )
        )
        _clear_final_model_artifacts(paths)
    _save_json(verdict, paths['final_verdict_json'])
    _save_json(verdict.get('hard_blockers', []), paths['hard_blockers'])
    _write_reviews(frame, predictions, verdict, paths)
    _write_index_and_summary(paths, verdict, metrics)
    _write_artifact_manifest(paths)

    return {
        'endotheliosis_grade_model_index': paths['index'],
        'endotheliosis_grade_model_verdict': paths['final_verdict_json'],
        'endotheliosis_grade_model_verdict_md': paths['final_verdict_md'],
        'endotheliosis_grade_model_candidate_metrics': paths['candidate_metrics'],
        'endotheliosis_grade_model_development_folds': paths['development_folds'],
        'endotheliosis_grade_model_predictions': paths['development_oof_predictions'],
        'endotheliosis_grade_model_candidate_coverage': paths[
            'candidate_coverage_matrix'
        ],
        'endotheliosis_grade_model_dox_smoke_manifest': paths['dox_smoke_manifest'],
        'endotheliosis_grade_model_dox_smoke_predictions': paths[
            'dox_smoke_predictions'
        ],
        'endotheliosis_grade_model_dox_smoke_summary': paths['dox_smoke_summary'],
        'endotheliosis_grade_model_dox_smoke_threshold_curve': paths[
            'dox_smoke_threshold_curve'
        ],
        'endotheliosis_grade_model_dox_smoke_report': paths['dox_smoke_report'],
        'endotheliosis_grade_model_dox_smoke_contract': paths['dox_smoke_contract'],
        'endotheliosis_grade_model_dox_overcall_triage_queue': paths[
            'dox_overcall_triage_queue'
        ],
        'endotheliosis_grade_model_dox_overcall_triage_report': paths[
            'dox_overcall_triage_report'
        ],
        'endotheliosis_grade_model_dox_overcall_review_diagnostic': paths[
            'dox_overcall_review_diagnostic'
        ],
        'endotheliosis_grade_model_dox_overcall_review_interpretation': paths[
            'dox_overcall_review_interpretation'
        ],
        'endotheliosis_grade_model_artifact_manifest': paths['artifact_manifest'],
        'endotheliosis_grade_model_executive_summary': paths['executive_summary'],
    }
