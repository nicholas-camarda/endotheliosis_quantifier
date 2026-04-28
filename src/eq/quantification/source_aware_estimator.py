"""Source-aware practical endotheliosis burden estimator."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from eq.quantification.burden import ALLOWED_SCORE_VALUES

SOURCE_AWARE_ROOT_NAME = 'source_aware_estimator'
STAGE_INDEX_BY_SCORE = {
    float(score): float(index * 20.0)
    for index, score in enumerate(ALLOWED_SCORE_VALUES)
}
REQUIRED_SCORE_VALUES = set(float(score) for score in ALLOWED_SCORE_VALUES)
PREDICTION_INTERVAL_COVERAGE = 0.90
SUMMARY_FIGURES = [
    'metrics_by_split.png',
    'predicted_vs_observed.png',
    'calibration_by_score.png',
    'source_performance.png',
    'uncertainty_width_distribution.png',
    'reliability_label_counts.png',
]
REQUIRED_RELATIVE_ARTIFACTS = [
    'INDEX.md',
    'summary/estimator_verdict.json',
    'summary/estimator_verdict.md',
    'summary/metrics_by_split.csv',
    'summary/metrics_by_split.json',
    'summary/artifact_manifest.json',
    *[f'summary/figures/{name}' for name in SUMMARY_FIGURES],
    'predictions/image_predictions.csv',
    'predictions/subject_predictions.csv',
    'diagnostics/upstream_roi_adequacy.json',
    'diagnostics/source_sensitivity.json',
    'diagnostics/reliability_labels.json',
    'evidence/source_aware_estimator_review.html',
    'internal/candidate_metrics.csv',
    'internal/candidate_summary.json',
]


class SourceAwareEstimatorError(RuntimeError):
    """Raised when source-aware estimator inputs violate the contract."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    target_level: str
    feature_family: str
    source_handling: str
    feature_columns: tuple[str, ...]


def source_aware_output_paths(burden_output_dir: Path) -> dict[str, Path]:
    root = Path(burden_output_dir) / SOURCE_AWARE_ROOT_NAME
    return {
        'root': root,
        'summary': root / 'summary',
        'figures': root / 'summary' / 'figures',
        'predictions': root / 'predictions',
        'diagnostics': root / 'diagnostics',
        'evidence': root / 'evidence',
        'internal': root / 'internal',
        'index': root / 'INDEX.md',
        'verdict_json': root / 'summary' / 'estimator_verdict.json',
        'verdict_md': root / 'summary' / 'estimator_verdict.md',
        'metrics_by_split_csv': root / 'summary' / 'metrics_by_split.csv',
        'metrics_by_split_json': root / 'summary' / 'metrics_by_split.json',
        'artifact_manifest': root / 'summary' / 'artifact_manifest.json',
        'image_predictions': root / 'predictions' / 'image_predictions.csv',
        'subject_predictions': root / 'predictions' / 'subject_predictions.csv',
        'upstream_roi_adequacy': root / 'diagnostics' / 'upstream_roi_adequacy.json',
        'source_sensitivity': root / 'diagnostics' / 'source_sensitivity.json',
        'reliability_labels': root / 'diagnostics' / 'reliability_labels.json',
        'evidence_review': root / 'evidence' / 'source_aware_estimator_review.html',
        'candidate_metrics': root / 'internal' / 'candidate_metrics.csv',
        'candidate_summary': root / 'internal' / 'candidate_summary.json',
    }


def _save_json(data: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return output_path


def _stage_targets(scores: pd.Series) -> np.ndarray:
    return scores.astype(float).map(STAGE_INDEX_BY_SCORE).to_numpy(dtype=np.float64)


def _nearest_supported_score(stage_index: np.ndarray) -> np.ndarray:
    values = np.asarray(list(STAGE_INDEX_BY_SCORE.values()), dtype=np.float64)
    scores = np.asarray(list(STAGE_INDEX_BY_SCORE.keys()), dtype=np.float64)
    index = np.abs(stage_index[:, None] - values[None, :]).argmin(axis=1)
    return scores[index]


def _validate_inputs(df: pd.DataFrame) -> list[str]:
    required = {'subject_id', 'score', 'roi_image_path', 'roi_mask_path'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SourceAwareEstimatorError(
            f'Source-aware estimator input is missing required columns: {missing}'
        )
    scores = set(pd.to_numeric(df['score'], errors='coerce').dropna().astype(float))
    unsupported = sorted(
        score for score in scores if score not in REQUIRED_SCORE_VALUES
    )
    if unsupported:
        raise SourceAwareEstimatorError(
            f'Unsupported score values for source-aware estimator: {unsupported}; '
            f'supported={sorted(REQUIRED_SCORE_VALUES)}'
        )
    if df['subject_id'].astype(str).str.strip().eq('').any():
        raise SourceAwareEstimatorError('Blank subject_id values are not supported.')
    if 'cohort_id' not in df.columns:
        df['cohort_id'] = ''
    notes: list[str] = []
    for column in ['sample_id', 'image_id', 'subject_image_id']:
        if column not in df.columns:
            notes.append(f'missing_optional_identity_column:{column}')
    return notes


def _safe_numeric_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    safe: list[str] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors='coerce')
        if values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all():
            if float(values.std(ddof=0)) > 0:
                safe.append(column)
    return safe


def _load_feature_table(
    embedding_df: pd.DataFrame, burden_output_dir: Path
) -> pd.DataFrame:
    learned_path = (
        Path(burden_output_dir)
        / 'learned_roi'
        / 'feature_sets'
        / 'learned_roi_features.csv'
    )
    base = embedding_df.copy().reset_index(drop=True)
    if learned_path.exists():
        learned = pd.read_csv(learned_path)
        learned_feature_columns = [
            column for column in learned.columns if column.startswith('learned_')
        ]
        identity_keys = [
            key
            for key in ['subject_image_id', 'subject_id', 'sample_id', 'image_id']
            if key in base.columns and key in learned.columns
        ]
        if identity_keys and len(learned) == len(base):
            merged = base.merge(
                learned[identity_keys + learned_feature_columns],
                on=identity_keys,
                how='left',
                validate='one_to_one',
            )
            return merged
    return base


def _candidate_specs(feature_df: pd.DataFrame) -> list[CandidateSpec]:
    roi_columns = _safe_numeric_columns(
        feature_df,
        [
            column
            for column in [
                'roi_area',
                'roi_fill_fraction',
                'roi_mean_intensity',
                'roi_openness_score',
                'roi_component_count',
                'roi_union_bbox_width',
                'roi_union_bbox_height',
                'roi_largest_component_area_fraction',
            ]
            if column in feature_df.columns
        ],
    )
    learned_qc = _safe_numeric_columns(
        feature_df,
        [
            column
            for column in feature_df.columns
            if column.startswith('learned_simple_roi_qc_')
        ],
    )
    learned_encoder = _safe_numeric_columns(
        feature_df,
        [
            column
            for column in feature_df.columns
            if column.startswith('learned_current_glomeruli_encoder_')
        ],
    )
    embedding_columns = _safe_numeric_columns(
        feature_df,
        [column for column in feature_df.columns if column.startswith('embedding_')],
    )
    qc = learned_qc or roi_columns
    learned = learned_encoder or embedding_columns
    hybrid = list(dict.fromkeys([*qc, *learned]))
    specs = [
        CandidateSpec('pooled_roi_qc', 'image', 'roi_qc', 'pooled', tuple(qc)),
        CandidateSpec(
            'pooled_learned_roi', 'image', 'learned_roi', 'pooled', tuple(learned)
        ),
        CandidateSpec('pooled_hybrid', 'image', 'hybrid', 'pooled', tuple(hybrid)),
        CandidateSpec(
            'source_adjusted_roi_qc', 'image', 'roi_qc', 'source_indicator', tuple(qc)
        ),
        CandidateSpec(
            'source_adjusted_hybrid',
            'image',
            'hybrid',
            'source_indicator',
            tuple(hybrid),
        ),
        CandidateSpec(
            'within_source_calibrated_hybrid',
            'image',
            'hybrid',
            'within_source_calibration',
            tuple(hybrid),
        ),
        CandidateSpec(
            'subject_source_adjusted_hybrid',
            'subject',
            'hybrid',
            'source_indicator',
            tuple(hybrid),
        ),
    ]
    return [spec for spec in specs if spec.feature_columns]


def _design_matrix(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    source_handling: str,
    source_categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    matrix = (
        df.loc[:, list(feature_columns)].apply(pd.to_numeric).to_numpy(dtype=np.float64)
    )
    names = list(feature_columns)
    if source_handling == 'source_indicator':
        source = df['cohort_id'].astype(str).fillna('')
        if source_categories is None:
            source_categories = sorted(value for value in source.unique() if value)
        indicator_blocks = []
        for category in source_categories:
            indicator_blocks.append(
                (source == category).astype(float).to_numpy()[:, None]
            )
            names.append(f'source_indicator__{category}')
        if indicator_blocks:
            matrix = np.hstack([matrix, *indicator_blocks])
    return matrix, names


def _fit_predict_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    source_handling: str,
    target_column: str = 'observed_stage_index',
) -> tuple[np.ndarray, list[str]]:
    source_categories = sorted(
        value
        for value in train_df['cohort_id'].astype(str).fillna('').unique()
        if value
    )
    x_train, _ = _design_matrix(
        train_df,
        feature_columns,
        source_handling=source_handling
        if source_handling == 'source_indicator'
        else 'pooled',
        source_categories=source_categories,
    )
    x_test, _ = _design_matrix(
        test_df,
        feature_columns,
        source_handling=source_handling
        if source_handling == 'source_indicator'
        else 'pooled',
        source_categories=source_categories,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    warnings_seen: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = Ridge(alpha=1.0, solver='svd')
        model.fit(x_train, train_df[target_column].to_numpy(dtype=np.float64))
        pred = model.predict(x_test)
    warnings_seen.extend(str(item.message) for item in caught)
    if source_handling == 'within_source_calibration':
        train_pred = model.predict(x_train)
        residual = train_df[target_column].to_numpy(dtype=np.float64) - train_pred
        offsets = (
            pd.DataFrame(
                {'cohort_id': train_df['cohort_id'].astype(str), 'residual': residual}
            )
            .groupby('cohort_id')['residual']
            .mean()
            .to_dict()
        )
        pred = (
            pred + test_df['cohort_id'].astype(str).map(offsets).fillna(0.0).to_numpy()
        )
    return np.clip(pred, 0.0, 100.0), list(dict.fromkeys(warnings_seen))


def _raise_if_subject_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_subjects = set(train_df['subject_id'].astype(str))
    test_subjects = set(test_df['subject_id'].astype(str))
    leaked = sorted(train_subjects & test_subjects)
    if leaked:
        preview = ', '.join(leaked[:5])
        raise SourceAwareEstimatorError(
            'subject_id_validation_leakage detected in source-aware estimator: '
            f'{preview}'
        )


def _prediction_set(predicted_stage: np.ndarray, residual_quantile: float) -> list[str]:
    values = np.asarray(list(STAGE_INDEX_BY_SCORE.values()), dtype=np.float64)
    scores = list(STAGE_INDEX_BY_SCORE.keys())
    sets: list[str] = []
    for value in predicted_stage:
        included = [
            f'{score:g}'
            for score, stage in zip(scores, values)
            if abs(float(stage) - float(value)) <= residual_quantile
        ]
        if not included:
            included = [f'{_nearest_supported_score(np.array([value]))[0]:g}']
        sets.append('|'.join(included))
    return sets


def _coverage_from_sets(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty or 'prediction_set_scores' not in df.columns:
        return np.nan, np.nan
    covered = []
    sizes = []
    for _, row in df.iterrows():
        values = [item for item in str(row['prediction_set_scores']).split('|') if item]
        sizes.append(len(values))
        covered.append(f'{float(row["score"]):g}' in values)
    return float(np.mean(covered)), float(np.mean(sizes))


def _metric_row(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    target_level: str,
    feature_family: str,
    source_handling: str,
    split_label: str,
    split_description: str,
    eligible_for_model_selection: bool,
    warning_messages: list[str],
) -> dict[str, Any]:
    if predictions.empty:
        return {
            'candidate_id': candidate_id,
            'target_level': target_level,
            'feature_family': feature_family,
            'source_handling_mode': source_handling,
            'split_label': split_label,
            'split_description': split_description,
            'row_count': 0,
            'subject_count': 0,
            'source_scope': '',
            'stage_index_mae': np.nan,
            'grade_scale_mae': np.nan,
            'coverage': np.nan,
            'average_interval_or_prediction_set_width': np.nan,
            'finite_output_status': 'empty',
            'warning_count': len(warning_messages),
            'intended_use': 'not_available',
            'eligible_for_model_selection': eligible_for_model_selection,
        }
    predicted_stage = predictions['predicted_stage_index'].to_numpy(dtype=np.float64)
    finite_output = bool(np.isfinite(predicted_stage).all())
    predicted_score = (
        _nearest_supported_score(predicted_stage)
        if finite_output
        else np.full(len(predictions), np.nan)
    )
    coverage, set_size = _coverage_from_sets(predictions)
    if np.isnan(set_size) and {'interval_low_0_100', 'interval_high_0_100'}.issubset(
        predictions.columns
    ):
        set_size = float(
            (
                predictions['interval_high_0_100'] - predictions['interval_low_0_100']
            ).mean()
        )
    return {
        'candidate_id': candidate_id,
        'target_level': target_level,
        'feature_family': feature_family,
        'source_handling_mode': source_handling,
        'split_label': split_label,
        'split_description': split_description,
        'row_count': int(len(predictions)),
        'subject_count': int(predictions['subject_id'].nunique())
        if 'subject_id' in predictions.columns
        else 0,
        'source_scope': '|'.join(sorted(predictions['cohort_id'].astype(str).unique()))
        if 'cohort_id' in predictions.columns
        else '',
        'stage_index_mae': float(
            mean_absolute_error(predictions['observed_stage_index'], predicted_stage)
        )
        if finite_output
        else np.nan,
        'grade_scale_mae': float(
            mean_absolute_error(predictions['score'].astype(float), predicted_score)
        )
        if finite_output and 'score' in predictions.columns
        else np.nan,
        'coverage': coverage,
        'average_interval_or_prediction_set_width': set_size,
        'finite_output_status': 'finite' if finite_output else 'nonfinite',
        'warning_count': len(warning_messages),
        'intended_use': 'current_data_scope_labeled_estimation',
        'eligible_for_model_selection': bool(eligible_for_model_selection),
    }


def _image_oof_predictions(
    df: pd.DataFrame, spec: CandidateSpec, n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    groups = df['subject_id'].astype(str)
    split_count = min(max(2, n_splits), groups.nunique())
    splitter = GroupKFold(n_splits=split_count)
    predictions = []
    all_warnings: list[str] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(df, groups=groups), start=1
    ):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        _raise_if_subject_leakage(train_df, test_df)
        pred, warning_messages = _fit_predict_ridge(
            train_df,
            test_df,
            spec.feature_columns,
            source_handling=spec.source_handling,
        )
        all_warnings.extend(warning_messages)
        fold_df = test_df.copy()
        fold_df['fold'] = int(fold)
        fold_df['predicted_stage_index'] = pred
        fold_df['prediction_source'] = 'validation_subject_heldout'
        predictions.append(fold_df)
    result = pd.concat(predictions, ignore_index=True)
    residual_q = float(
        np.quantile(
            np.abs(result['observed_stage_index'] - result['predicted_stage_index']),
            PREDICTION_INTERVAL_COVERAGE,
        )
    )
    result['interval_low_0_100'] = np.clip(
        result['predicted_stage_index'] - residual_q, 0.0, 100.0
    )
    result['interval_high_0_100'] = np.clip(
        result['predicted_stage_index'] + residual_q, 0.0, 100.0
    )
    result['prediction_set_scores'] = _prediction_set(
        result['predicted_stage_index'].to_numpy(dtype=float), residual_q
    )
    result['residual_quantile_coverage'] = PREDICTION_INTERVAL_COVERAGE
    return result, list(dict.fromkeys(all_warnings))


def _training_apparent_predictions(
    df: pd.DataFrame, spec: CandidateSpec
) -> tuple[pd.DataFrame, list[str]]:
    pred, warning_messages = _fit_predict_ridge(
        df, df, spec.feature_columns, source_handling=spec.source_handling
    )
    result = df.copy()
    result['fold'] = 0
    result['predicted_stage_index'] = pred
    result['prediction_source'] = 'training_apparent'
    residual_q = float(
        np.quantile(
            np.abs(result['observed_stage_index'] - result['predicted_stage_index']),
            PREDICTION_INTERVAL_COVERAGE,
        )
    )
    result['interval_low_0_100'] = np.clip(
        result['predicted_stage_index'] - residual_q, 0.0, 100.0
    )
    result['interval_high_0_100'] = np.clip(
        result['predicted_stage_index'] + residual_q, 0.0, 100.0
    )
    result['prediction_set_scores'] = _prediction_set(
        result['predicted_stage_index'].to_numpy(dtype=float), residual_q
    )
    result['residual_quantile_coverage'] = PREDICTION_INTERVAL_COVERAGE
    return result, warning_messages


def _subject_table(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    aggregations: dict[str, Any] = {
        column: 'mean' for column in feature_columns if column in df.columns
    }
    aggregations.update(
        {
            'observed_stage_index': 'mean',
            'score': 'mean',
            'cohort_id': lambda value: (
                value.mode().iloc[0] if not value.mode().empty else str(value.iloc[0])
            ),
        }
    )
    for column in ['sample_id', 'image_id', 'subject_image_id']:
        if column in df.columns:
            aggregations[column] = 'first'
    return df.groupby('subject_id', as_index=False).agg(aggregations)


def _subject_oof_predictions(
    df: pd.DataFrame, spec: CandidateSpec, n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    subject_df = _subject_table(df, spec.feature_columns)
    split_count = min(max(2, n_splits), len(subject_df))
    splitter = KFold(n_splits=split_count, shuffle=True, random_state=11)
    predictions = []
    all_warnings: list[str] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(subject_df), start=1):
        train_df = subject_df.iloc[train_idx].copy()
        test_df = subject_df.iloc[test_idx].copy()
        _raise_if_subject_leakage(train_df, test_df)
        pred, warning_messages = _fit_predict_ridge(
            train_df,
            test_df,
            spec.feature_columns,
            source_handling=spec.source_handling,
        )
        all_warnings.extend(warning_messages)
        fold_df = test_df.copy()
        fold_df['fold'] = int(fold)
        fold_df['predicted_stage_index'] = pred
        fold_df['prediction_source'] = 'validation_subject_heldout'
        predictions.append(fold_df)
    result = pd.concat(predictions, ignore_index=True)
    residual_q = float(
        np.quantile(
            np.abs(result['observed_stage_index'] - result['predicted_stage_index']),
            PREDICTION_INTERVAL_COVERAGE,
        )
    )
    result['interval_low_0_100'] = np.clip(
        result['predicted_stage_index'] - residual_q, 0.0, 100.0
    )
    result['interval_high_0_100'] = np.clip(
        result['predicted_stage_index'] + residual_q, 0.0, 100.0
    )
    return result, list(dict.fromkeys(all_warnings))


def _add_reliability_labels(df: pd.DataFrame, known_sources: set[str]) -> pd.DataFrame:
    result = df.copy()
    labels = []
    for _, row in result.iterrows():
        row_labels = []
        source = str(row.get('cohort_id', '') or '')
        if not source or source not in known_sources:
            row_labels.append('unknown_source')
        width = float(row.get('interval_high_0_100', np.nan)) - float(
            row.get('interval_low_0_100', np.nan)
        )
        if np.isfinite(width) and width >= 60:
            row_labels.append('wide_uncertainty')
        predicted = float(row.get('predicted_stage_index', np.nan))
        if np.isfinite(predicted) and 70 <= predicted <= 90:
            row_labels.append('transitional_score_region')
        if not row_labels:
            row_labels.append('standard_current_data')
        result_labels = '|'.join(dict.fromkeys(row_labels))
        labels.append(result_labels)
    result['reliability_label'] = labels
    return result


def _source_sensitivity(
    selected_predictions: pd.DataFrame,
    feature_df: pd.DataFrame,
    selected_spec: CandidateSpec,
) -> dict[str, Any]:
    by_source = []
    for source, source_df in selected_predictions.groupby('cohort_id', dropna=False):
        coverage, set_size = _coverage_from_sets(source_df)
        by_source.append(
            {
                'cohort_id': str(source),
                'row_count': int(len(source_df)),
                'subject_count': int(source_df['subject_id'].nunique()),
                'score_counts': {
                    f'{float(score):g}': int(count)
                    for score, count in source_df['score']
                    .value_counts()
                    .sort_index()
                    .items()
                },
                'stage_index_mae': float(
                    mean_absolute_error(
                        source_df['observed_stage_index'],
                        source_df['predicted_stage_index'],
                    )
                ),
                'grade_scale_mae': float(
                    mean_absolute_error(
                        source_df['score'].astype(float),
                        _nearest_supported_score(
                            source_df['predicted_stage_index'].to_numpy(dtype=float)
                        ),
                    )
                ),
                'coverage': coverage,
                'average_prediction_set_size': set_size,
            }
        )
    leave_source_rows = []
    sources = sorted(str(value) for value in feature_df['cohort_id'].dropna().unique())
    for source in sources:
        train_df = feature_df[feature_df['cohort_id'].astype(str) != source].copy()
        test_df = feature_df[feature_df['cohort_id'].astype(str) == source].copy()
        if train_df.empty or test_df.empty or train_df['score'].nunique() < 2:
            leave_source_rows.append(
                {
                    'train_source': f'not_{source}',
                    'test_source': source,
                    'status': 'non_estimable',
                    'reason': 'insufficient_train_or_test_support',
                }
            )
            continue
        pred, messages = _fit_predict_ridge(
            train_df, test_df, selected_spec.feature_columns, source_handling='pooled'
        )
        tmp = test_df.copy()
        tmp['predicted_stage_index'] = pred
        leave_source_rows.append(
            {
                'train_source': f'not_{source}',
                'test_source': source,
                'row_count': int(len(tmp)),
                'subject_count': int(tmp['subject_id'].nunique()),
                'stage_index_mae': float(
                    mean_absolute_error(tmp['observed_stage_index'], pred)
                ),
                'grade_scale_mae': float(
                    mean_absolute_error(
                        tmp['score'].astype(float), _nearest_supported_score(pred)
                    )
                ),
                'warning_count': len(messages),
                'qualitative_degradation_status': 'current_data_sensitivity_only',
                'status': 'estimated',
            }
        )
    return {
        'row_count': int(len(selected_predictions)),
        'subject_count': int(selected_predictions['subject_id'].nunique()),
        'by_cohort': by_source,
        'leave_source_out': leave_source_rows,
        'interpretation': 'current-data source sensitivity; not external validation',
    }


def _upstream_roi_adequacy(df: pd.DataFrame, burden_output_dir: Path) -> dict[str, Any]:
    embedding_metadata_path = (
        Path(burden_output_dir).parent / 'embeddings' / 'embedding_metadata.json'
    )
    embedding_metadata = (
        json.loads(embedding_metadata_path.read_text(encoding='utf-8'))
        if embedding_metadata_path.exists()
        else {}
    )
    roi_status = (
        df['roi_status'].astype(str)
        if 'roi_status' in df.columns
        else pd.Series(['ok'] * len(df))
    )
    cohort_counts = (
        df['cohort_id'].astype(str).value_counts(dropna=False).to_dict()
        if 'cohort_id' in df.columns
        else {}
    )
    subject_counts = (
        df.groupby('cohort_id')['subject_id'].nunique().to_dict()
        if {'cohort_id', 'subject_id'}.issubset(df.columns)
        else {}
    )
    usable = int(roi_status.eq('ok').sum())
    failed = int(len(df) - usable)
    availability = {
        column: int(df[column].astype(str).str.strip().ne('').sum())
        for column in [
            'roi_image_path',
            'roi_mask_path',
            'raw_image_path',
            'raw_mask_path',
        ]
        if column in df.columns
    }
    status = 'adequate_current_data' if usable > 0 and failed == 0 else 'limited'
    return {
        'status': status,
        'total_input_rows': int(len(df)),
        'scored_rows': int(df['score'].notna().sum()) if 'score' in df.columns else 0,
        'usable_roi_rows': usable,
        'failed_roi_rows': failed,
        'roi_status_counts': roi_status.value_counts(dropna=False).to_dict(),
        'row_counts_by_cohort': cohort_counts,
        'subject_counts_by_cohort': {str(k): int(v) for k, v in subject_counts.items()},
        'path_availability_counts': availability,
        'segmentation_model_path': embedding_metadata.get(
            'segmentation_model_path', ''
        ),
        'segmentation_provenance_status': 'present'
        if embedding_metadata.get('segmentation_model_path')
        else 'missing_from_embedding_metadata',
        'mask_context': 'manual_or_manifest_mask_contract; exact source reported by cohort manifest when available',
        'reportable_scope_support': {
            'image_level': usable > 0,
            'subject_level': int(df['subject_id'].nunique()) >= 2
            if 'subject_id' in df.columns
            else False,
            'aggregate_current_data': usable > 0,
        },
    }


def _write_reliability_labels(path: Path) -> dict[str, Any]:
    labels = {
        'standard_current_data': 'Prediction is within the standard known-source current-data scope.',
        'wide_uncertainty': 'Prediction interval or prediction set is broad.',
        'transitional_score_region': 'Prediction lies near the high-ambiguity score-2 transition region.',
        'source_sensitive': 'Source/cohort diagnostics indicate source-dependent behavior.',
        'underpowered_stratum': 'Source or score stratum has limited support.',
        'leave_source_out_degraded': 'Leave-source-out sensitivity degrades materially.',
        'numerical_warning_scope_limiter': 'Candidate emitted nonfatal backend warnings with finite outputs.',
        'unknown_source': 'Source context is missing or not represented in the current training/evaluation sources.',
    }
    data = {
        'labels': labels,
        'hard_blockers': [
            'broken_joins',
            'unsupported_scores',
            'nonfinite_selected_predictions',
            'subject_id_validation_leakage',
            'untraceable_provenance',
            'missing_required_verdict_or_index',
            'claims_outside_grade_equivalent_prediction',
        ],
        'scope_limiters': [
            'score_specific_undercoverage',
            'wide_prediction_sets',
            'source_sensitivity',
            'small_source_score_strata',
            'leave_source_out_degradation',
            'weak_single_image_reliability',
            'candidate_numerical_warnings_outputs_finite',
            'unknown_source',
        ],
    }
    _save_json(data, path)
    return data


def _write_figures(
    *,
    paths: dict[str, Path],
    metrics_by_split: pd.DataFrame,
    image_predictions: pd.DataFrame,
    source_sensitivity: dict[str, Any],
) -> list[Path]:
    figures_dir = paths['figures']
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    plt.figure(figsize=(8, 4.5))
    metric_plot = metrics_by_split[
        metrics_by_split['candidate_id'].isin(
            image_predictions['candidate_id'].unique()
        )
    ].copy()
    if not metric_plot.empty:
        labels = metric_plot['candidate_id'] + '\\n' + metric_plot['split_label']
        plt.bar(labels, metric_plot['stage_index_mae'].astype(float))
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Stage-index MAE')
    plt.title('Metrics by Split')
    plt.tight_layout()
    path = figures_dir / 'metrics_by_split.png'
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    plt.figure(figsize=(5.5, 5))
    plt.scatter(
        image_predictions['observed_stage_index'],
        image_predictions['predicted_stage_index'],
        c=pd.Categorical(image_predictions['cohort_id']).codes,
        alpha=0.75,
    )
    plt.plot([0, 100], [0, 100], color='black', linestyle='--', linewidth=1)
    plt.xlabel('Observed stage index')
    plt.ylabel('Predicted stage index')
    plt.title('Predicted vs Observed')
    plt.tight_layout()
    path = figures_dir / 'predicted_vs_observed.png'
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    plt.figure(figsize=(7, 4.5))
    calibration_rows = []
    for score, group in image_predictions.groupby('score'):
        coverage, set_size = _coverage_from_sets(group)
        calibration_rows.append(
            {'score': float(score), 'coverage': coverage, 'average_set_size': set_size}
        )
    calibration = pd.DataFrame(calibration_rows)
    if not calibration.empty:
        plt.bar(calibration['score'].astype(str), calibration['coverage'])
        plt.axhline(PREDICTION_INTERVAL_COVERAGE, color='black', linestyle='--')
        plt.ylim(0, 1.05)
    plt.xlabel('Observed score')
    plt.ylabel('Prediction-set coverage')
    plt.title('Calibration by Score')
    plt.tight_layout()
    path = figures_dir / 'calibration_by_score.png'
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    plt.figure(figsize=(7, 4.5))
    source_rows = pd.DataFrame(source_sensitivity.get('by_cohort', []))
    if not source_rows.empty:
        plt.bar(source_rows['cohort_id'].astype(str), source_rows['stage_index_mae'])
        plt.ylabel('Stage-index MAE')
    plt.title('Source Performance')
    plt.tight_layout()
    path = figures_dir / 'source_performance.png'
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    plt.figure(figsize=(7, 4.5))
    widths = (
        image_predictions['interval_high_0_100']
        - image_predictions['interval_low_0_100']
    )
    plt.hist(widths, bins=min(12, max(3, int(len(widths) ** 0.5))))
    plt.xlabel('Prediction interval width')
    plt.ylabel('Rows')
    plt.title('Uncertainty Width Distribution')
    plt.tight_layout()
    path = figures_dir / 'uncertainty_width_distribution.png'
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    plt.figure(figsize=(7, 4.5))
    label_counts = image_predictions['reliability_label'].str.get_dummies('|').sum()
    if not label_counts.empty:
        plt.bar(label_counts.index, label_counts.values)
        plt.xticks(rotation=45, ha='right')
    plt.ylabel('Rows')
    plt.title('Reliability Label Counts')
    plt.tight_layout()
    path = figures_dir / 'reliability_label_counts.png'
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)
    return written


def _write_evidence_review(
    path: Path, *, verdict: dict[str, Any], image_predictions: pd.DataFrame
) -> Path:
    examples = []
    work = image_predictions.copy()
    work['abs_error'] = np.abs(
        work['observed_stage_index'] - work['predicted_stage_index']
    )
    buckets = [
        ('representative_correct', work.sort_values('abs_error', ascending=True)),
        ('high_error', work.sort_values('abs_error', ascending=False)),
        (
            'high_uncertainty',
            work.assign(
                width=work['interval_high_0_100'] - work['interval_low_0_100']
            ).sort_values('width', ascending=False),
        ),
        (
            'score_2_like',
            work.iloc[(work['predicted_stage_index'] - 80.0).abs().argsort()],
        ),
    ]
    seen = set()
    for bucket, frame in buckets:
        for _, row in frame.head(3).iterrows():
            key = str(row.get('subject_image_id', row.name))
            if key in seen:
                continue
            seen.add(key)
            examples.append((bucket, row))
            break
    cards = []
    for bucket, row in examples:
        cards.append(
            '<section>'
            f'<h2>{escape(bucket)}: {escape(str(row.get("subject_image_id", "")))}</h2>'
            f'<p>Observed score: {float(row.get("score", np.nan)):.2f}; '
            f'predicted burden: {float(row.get("predicted_stage_index", np.nan)):.2f}; '
            f'reliability: {escape(str(row.get("reliability_label", "")))}</p>'
            f'<p>Source: {escape(str(row.get("cohort_id", "")))}; '
            f'ROI: {escape(str(row.get("roi_image_path", "")))}</p>'
            '</section>'
        )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Source-Aware Estimator Review</title></head>
<body>
<h1>Source-Aware Endotheliosis Estimator Review</h1>
<p>This evidence supports predictive grade-equivalent burden review only; it is not proof of closed-lumen mechanism.</p>
<p>Verdict: {escape(str(verdict.get('overall_status', '')))}</p>
{''.join(cards)}
</body></html>
"""
    path.write_text(html, encoding='utf-8')
    return path


def _artifact_manifest(root: Path) -> dict[str, Any]:
    entries = []
    for relative in REQUIRED_RELATIVE_ARTIFACTS:
        path = root / relative
        role = relative.split('/', 1)[0] if '/' in relative else 'index'
        entries.append(
            {
                'relative_path': relative,
                'role': role,
                'consumer': 'human_review_and_tests',
                'reportability': 'human_facing'
                if relative.startswith(('INDEX', 'summary/', 'evidence/'))
                else 'diagnostic_or_internal',
                'required': True,
                'exists': path.exists(),
            }
        )
    return {
        'artifact_count': len(entries),
        'allowed_first_pass_artifacts': REQUIRED_RELATIVE_ARTIFACTS,
        'artifacts': entries,
        'manifest_complete': all(entry['exists'] for entry in entries),
    }


def _write_index(path: Path, verdict: dict[str, Any], manifest: dict[str, Any]) -> Path:
    lines = [
        '# Source-Aware Endotheliosis Estimator',
        '',
        'Open `summary/estimator_verdict.md` first.',
        '',
        f'Overall status: `{verdict.get("overall_status", "")}`',
        '',
        'Claim boundary: predictive grade-equivalent burden for the current scored MR TIFF/ROI data only.',
        '',
        '## Artifact Manifest',
        '',
    ]
    for artifact in manifest['artifacts']:
        lines.append(
            f'- `{artifact["relative_path"]}`: {artifact["role"]} ({artifact["consumer"]})'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _write_verdict_md(path: Path, verdict: dict[str, Any]) -> Path:
    figures = '\n'.join(f'- `figures/{name}`' for name in SUMMARY_FIGURES)
    text = f"""# Source-Aware Estimator Verdict

Overall status: `{verdict['overall_status']}`

README snippet eligible: `{verdict['readme_snippet_eligible']}`

## Reportable Scopes

```json
{json.dumps(verdict['reportable_scopes'], indent=2)}
```

## Hard Blockers

{', '.join(verdict['hard_blockers']) or 'none'}

## Scope Limiters

{', '.join(verdict['scope_limiters']) or 'none'}

## Summary Figures

{figures}

The figures and metrics are current-data predictive evidence only. Apparent training metrics are optimistic and are not independent testing.
"""
    path.write_text(text, encoding='utf-8')
    return path


def evaluate_source_aware_endotheliosis_estimator(
    embedding_df: pd.DataFrame, burden_output_dir: Path, *, n_splits: int = 3
) -> dict[str, Path]:
    """Evaluate contained source-aware current-data endotheliosis estimator."""
    paths = source_aware_output_paths(burden_output_dir)
    for key in [
        'summary',
        'figures',
        'predictions',
        'diagnostics',
        'evidence',
        'internal',
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)
    work_df = embedding_df.copy().reset_index(drop=True)
    input_notes = _validate_inputs(work_df)
    feature_df = _load_feature_table(work_df, burden_output_dir)
    feature_df['score'] = pd.to_numeric(feature_df['score'], errors='raise').astype(
        float
    )
    feature_df['observed_stage_index'] = _stage_targets(feature_df['score'])
    feature_df['cohort_id'] = feature_df['cohort_id'].fillna('').astype(str)
    known_sources = set(value for value in feature_df['cohort_id'].unique() if value)

    upstream = _upstream_roi_adequacy(feature_df, burden_output_dir)
    _save_json(upstream, paths['upstream_roi_adequacy'])
    reliability_definitions = _write_reliability_labels(paths['reliability_labels'])

    metrics_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    image_predictions_by_candidate: dict[str, pd.DataFrame] = {}
    subject_predictions_by_candidate: dict[str, pd.DataFrame] = {}

    for spec in _candidate_specs(feature_df):
        if spec.target_level == 'image':
            validation_pred, validation_warnings = _image_oof_predictions(
                feature_df, spec, n_splits
            )
            validation_pred['candidate_id'] = spec.candidate_id
            validation_pred['target_level'] = spec.target_level
            validation_pred = _add_reliability_labels(validation_pred, known_sources)
            training_pred, training_warnings = _training_apparent_predictions(
                feature_df, spec
            )
            training_pred['candidate_id'] = spec.candidate_id
            training_pred['target_level'] = spec.target_level
            training_pred = _add_reliability_labels(training_pred, known_sources)
            image_predictions_by_candidate[spec.candidate_id] = validation_pred
            metrics_rows.append(
                _metric_row(
                    training_pred,
                    candidate_id=spec.candidate_id,
                    target_level=spec.target_level,
                    feature_family=spec.feature_family,
                    source_handling=spec.source_handling,
                    split_label='training_apparent',
                    split_description='Full-data apparent fit; optimistic diagnostic only',
                    eligible_for_model_selection=False,
                    warning_messages=training_warnings,
                )
            )
            metrics_rows.append(
                _metric_row(
                    validation_pred,
                    candidate_id=spec.candidate_id,
                    target_level=spec.target_level,
                    feature_family=spec.feature_family,
                    source_handling=spec.source_handling,
                    split_label='validation_subject_heldout',
                    split_description='Grouped out-of-fold validation by subject_id',
                    eligible_for_model_selection=True,
                    warning_messages=validation_warnings,
                )
            )
            testing_row = _metric_row(
                validation_pred.iloc[0:0],
                candidate_id=spec.candidate_id,
                target_level=spec.target_level,
                feature_family=spec.feature_family,
                source_handling=spec.source_handling,
                split_label='testing_not_available_current_data_sensitivity',
                split_description='No predeclared independent held-out test partition exists; source sensitivity reported separately',
                eligible_for_model_selection=False,
                warning_messages=[],
            )
            testing_row['finite_output_status'] = 'not_applicable'
            metrics_rows.append(testing_row)
            candidate_summaries.append(
                {
                    'candidate_id': spec.candidate_id,
                    'target_level': spec.target_level,
                    'feature_family': spec.feature_family,
                    'source_handling_mode': spec.source_handling,
                    'feature_count': len(spec.feature_columns),
                    'warning_messages': validation_warnings,
                }
            )
        else:
            validation_pred, validation_warnings = _subject_oof_predictions(
                feature_df, spec, n_splits
            )
            validation_pred['candidate_id'] = spec.candidate_id
            validation_pred['target_level'] = spec.target_level
            validation_pred = _add_reliability_labels(validation_pred, known_sources)
            subject_predictions_by_candidate[spec.candidate_id] = validation_pred
            metrics_rows.append(
                _metric_row(
                    validation_pred,
                    candidate_id=spec.candidate_id,
                    target_level=spec.target_level,
                    feature_family=spec.feature_family,
                    source_handling=spec.source_handling,
                    split_label='validation_subject_heldout',
                    split_description='One row per held-out subject',
                    eligible_for_model_selection=True,
                    warning_messages=validation_warnings,
                )
            )
            testing_row = _metric_row(
                validation_pred.iloc[0:0],
                candidate_id=spec.candidate_id,
                target_level=spec.target_level,
                feature_family=spec.feature_family,
                source_handling=spec.source_handling,
                split_label='testing_not_available_current_data_sensitivity',
                split_description='No predeclared independent held-out test partition exists',
                eligible_for_model_selection=False,
                warning_messages=[],
            )
            testing_row['finite_output_status'] = 'not_applicable'
            metrics_rows.append(testing_row)
            candidate_summaries.append(
                {
                    'candidate_id': spec.candidate_id,
                    'target_level': spec.target_level,
                    'feature_family': spec.feature_family,
                    'source_handling_mode': spec.source_handling,
                    'feature_count': len(spec.feature_columns),
                    'warning_messages': validation_warnings,
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    selection_metrics = metrics[
        (metrics['split_label'] == 'validation_subject_heldout')
        & (metrics['finite_output_status'] == 'finite')
    ].copy()
    image_selection = selection_metrics[selection_metrics['target_level'] == 'image']
    subject_selection = selection_metrics[
        selection_metrics['target_level'] == 'subject'
    ]
    if image_selection.empty:
        raise SourceAwareEstimatorError(
            'No finite image-level source-aware candidates.'
        )
    best_image_id = str(
        image_selection.sort_values(
            ['grade_scale_mae', 'stage_index_mae', 'candidate_id']
        ).iloc[0]['candidate_id']
    )
    best_subject_id = (
        str(
            subject_selection.sort_values(['stage_index_mae', 'candidate_id']).iloc[0][
                'candidate_id'
            ]
        )
        if not subject_selection.empty
        else ''
    )
    selected_image_predictions = image_predictions_by_candidate[best_image_id]
    selected_subject_predictions = (
        subject_predictions_by_candidate[best_subject_id]
        if best_subject_id
        else pd.DataFrame()
    )
    source_diag = _source_sensitivity(
        selected_image_predictions,
        feature_df,
        next(
            spec
            for spec in _candidate_specs(feature_df)
            if spec.candidate_id == best_image_id
        ),
    )
    _save_json(source_diag, paths['source_sensitivity'])

    hard_blockers: list[str] = []
    if not np.isfinite(
        selected_image_predictions['predicted_stage_index'].to_numpy(dtype=float)
    ).all():
        hard_blockers.append('nonfinite_selected_predictions')
    if not upstream['reportable_scope_support']['image_level']:
        hard_blockers.append('upstream_roi_inadequate_for_image_level')
    scope_limiters: list[str] = []
    if (
        selected_image_predictions['reliability_label']
        .str.contains('wide_uncertainty')
        .any()
    ):
        scope_limiters.append('wide_uncertainty')
    if (
        selected_image_predictions['reliability_label']
        .str.contains('transitional_score_region')
        .any()
    ):
        scope_limiters.append('score_2_like_transitional_region')
    if any(row.get('status') == 'estimated' for row in source_diag['leave_source_out']):
        scope_limiters.append('current_data_leave_source_out_sensitivity_only')
    warning_count = int(
        metrics.loc[
            (metrics['candidate_id'] == best_image_id)
            & (metrics['split_label'] == 'validation_subject_heldout'),
            'warning_count',
        ]
        .fillna(0)
        .max()
    )
    if warning_count:
        scope_limiters.append('numerical_warning_scope_limiter')
    if input_notes:
        scope_limiters.extend(input_notes)

    reportable_scopes = {
        'image_level': bool(
            not hard_blockers and upstream['reportable_scope_support']['image_level']
        ),
        'subject_level': bool(
            best_subject_id and upstream['reportable_scope_support']['subject_level']
        ),
        'aggregate_current_data': bool(
            upstream['reportable_scope_support']['aggregate_current_data']
        ),
    }
    readme_eligible = False
    verdict = {
        'overall_status': 'blocked'
        if hard_blockers
        else 'limited_current_data_estimator',
        'selected_image_candidate': best_image_id,
        'selected_subject_candidate': best_subject_id,
        'upstream_roi_adequacy_status': upstream['status'],
        'hard_blockers': hard_blockers,
        'scope_limiters': list(dict.fromkeys(scope_limiters)),
        'reportable_scopes': reportable_scopes,
        'non_reportable_scopes': [
            scope for scope, enabled in reportable_scopes.items() if not enabled
        ],
        'readme_snippet_eligible': readme_eligible,
        'testing_status': 'testing_not_available_current_data_sensitivity',
        'next_action': 'review_current_data_source_aware_estimator_outputs',
        'claim_boundary': 'predictive grade-equivalent endotheliosis burden for current scored MR TIFF/ROI data; not tissue percent, closed-capillary percent, causal evidence, or external validation',
    }

    metrics.to_csv(paths['metrics_by_split_csv'], index=False)
    _save_json(metrics.to_dict(orient='records'), paths['metrics_by_split_json'])
    candidate_metrics = metrics[
        metrics['split_label'] == 'validation_subject_heldout'
    ].copy()
    candidate_metrics.to_csv(paths['candidate_metrics'], index=False)
    _save_json(
        {
            'candidate_count': len(candidate_summaries),
            'selected_image_candidate': best_image_id,
            'selected_subject_candidate': best_subject_id,
            'candidates': candidate_summaries,
        },
        paths['candidate_summary'],
    )
    selected_image_predictions.to_csv(paths['image_predictions'], index=False)
    if selected_subject_predictions.empty:
        pd.DataFrame().to_csv(paths['subject_predictions'], index=False)
    else:
        selected_subject_predictions.to_csv(paths['subject_predictions'], index=False)
    _write_figures(
        paths=paths,
        metrics_by_split=metrics,
        image_predictions=selected_image_predictions,
        source_sensitivity=source_diag,
    )
    _write_evidence_review(
        paths['evidence_review'],
        verdict=verdict,
        image_predictions=selected_image_predictions,
    )
    _save_json(verdict, paths['verdict_json'])
    _write_verdict_md(paths['verdict_md'], verdict)
    manifest = _artifact_manifest(paths['root'])
    _write_index(paths['index'], verdict, manifest)
    _save_json(manifest, paths['artifact_manifest'])
    manifest = _artifact_manifest(paths['root'])
    _save_json(manifest, paths['artifact_manifest'])
    _write_index(paths['index'], verdict, manifest)
    return {
        'source_aware_index': paths['index'],
        'source_aware_estimator_verdict': paths['verdict_json'],
        'source_aware_estimator_verdict_md': paths['verdict_md'],
        'source_aware_metrics_by_split': paths['metrics_by_split_csv'],
        'source_aware_metrics_by_split_json': paths['metrics_by_split_json'],
        'source_aware_artifact_manifest': paths['artifact_manifest'],
        'source_aware_image_predictions': paths['image_predictions'],
        'source_aware_subject_predictions': paths['subject_predictions'],
        'source_aware_upstream_roi_adequacy': paths['upstream_roi_adequacy'],
        'source_aware_source_sensitivity': paths['source_sensitivity'],
        'source_aware_reliability_labels': paths['reliability_labels'],
        'source_aware_evidence_review': paths['evidence_review'],
        'source_aware_candidate_metrics': paths['candidate_metrics'],
        'source_aware_candidate_summary': paths['candidate_summary'],
        **{
            f'source_aware_figure_{Path(name).stem}': paths['figures'] / name
            for name in SUMMARY_FIGURES
        },
    }
