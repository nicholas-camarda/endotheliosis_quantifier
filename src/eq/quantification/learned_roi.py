"""Learned ROI representation screens for endotheliosis burden."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from eq.quantification.burden import (
    ALLOWED_SCORE_VALUES,
    BURDEN_COLUMN,
    PREDICTION_SET_COVERAGE,
    score_to_stage_index,
)
from eq.quantification.learned_roi_review import write_learned_roi_review
from eq.quantification.ordinal import NUMERICAL_INSTABILITY_PATTERNS

LEARNED_ROI_OUTPUT_GROUPS = {
    'summary': 'summary',
    'validation': 'validation',
    'calibration': 'calibration',
    'summaries': 'summaries',
    'evidence': 'evidence',
    'candidates': 'candidates',
    'diagnostics': 'diagnostics',
    'feature_sets': 'feature_sets',
}
LEARNED_ROI_ROOT_NAME = 'learned_roi'
FIT_ALLOWED_PROVIDERS = {'current_glomeruli_encoder', 'simple_roi_qc'}
AUDIT_ONLY_PROVIDERS = {'torchvision_resnet18_imagenet', 'timm_dino_or_vit'}
BASELINE_AVERAGE_SET_SIZE = 5.308
IMAGE_READY_MAX_AVERAGE_SET_SIZE = 4.0


class LearnedROIError(RuntimeError):
    """Raised when learned ROI quantification inputs violate the contract."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    target_level: str
    feature_set: str
    feature_columns: tuple[str, ...]


def learned_roi_output_paths(burden_output_dir: Path) -> dict[str, Path]:
    """Return canonical learned ROI grouped output directories."""
    root = Path(burden_output_dir) / LEARNED_ROI_ROOT_NAME
    paths = {key: root / dirname for key, dirname in LEARNED_ROI_OUTPUT_GROUPS.items()}
    paths['root'] = root
    paths['index'] = root / 'INDEX.md'
    paths['estimator_verdict'] = paths['summary'] / 'estimator_verdict.json'
    paths['estimator_verdict_md'] = paths['summary'] / 'estimator_verdict.md'
    paths['artifact_manifest'] = paths['summary'] / 'artifact_manifest.json'
    return paths


def _save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def _relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_learned_roi_first_read_artifacts(
    *, paths: dict[str, Path], summary: dict[str, Any], artifacts: dict[str, Path]
) -> dict[str, Path]:
    root = paths['root']
    manifest = {
        'estimator': 'learned_roi',
        'first_read': {
            'index': 'INDEX.md',
            'verdict_json': 'summary/estimator_verdict.json',
            'verdict_markdown': 'summary/estimator_verdict.md',
        },
        'artifact_contract': summary['artifact_contract'],
        'artifacts': {
            key: _relative_to_root(path, root)
            for key, path in artifacts.items()
            if path.exists()
        },
    }
    verdict = {
        'estimator': 'learned_roi',
        'candidate_count': summary['candidate_count'],
        'readme_docs_ready': summary['readme_docs_ready'],
        'readme_docs_ready_track': summary['readme_docs_ready_track'],
        'per_image_status': summary['per_image_readiness'].get('status', ''),
        'subject_cohort_status': summary['subject_cohort_readiness'].get('status', ''),
        'best_image_level_candidate': summary['best_image_level_candidate'].get(
            'candidate_id', ''
        ),
        'best_subject_level_candidate': summary['best_subject_level_candidate'].get(
            'candidate_id', ''
        ),
        'cohort_diagnostics_status': summary['cohort_diagnostics_status'],
        'blockers': summary['blockers'],
        'next_action': summary['next_action'],
        'claim_boundary': summary['claim_boundary'],
    }
    verdict_path = _save_json(verdict, paths['estimator_verdict'])
    manifest_path = _save_json(manifest, paths['artifact_manifest'])
    verdict_md = f"""# Learned ROI Summary

- README/docs-ready: `{verdict['readme_docs_ready']}`
- Ready track: `{verdict['readme_docs_ready_track'] or 'none'}`
- Per-image status: `{verdict['per_image_status']}`
- Subject/cohort status: `{verdict['subject_cohort_status']}`
- Best image-level candidate: `{verdict['best_image_level_candidate']}`
- Best subject-level candidate: `{verdict['best_subject_level_candidate']}`
- Cohort diagnostics: `{verdict['cohort_diagnostics_status']}`
- Next action: `{verdict['next_action']}`

Claim boundary: {verdict['claim_boundary']}
"""
    paths['estimator_verdict_md'].write_text(verdict_md, encoding='utf-8')
    index_text = """# Learned ROI

This subtree contains the capped learned-ROI candidate screen.

## First Read

- `summary/estimator_verdict.json`: machine-readable readiness verdict.
- `summary/estimator_verdict.md`: compact reviewer-facing verdict.
- `summary/artifact_manifest.json`: generated artifact map.
- `candidates/learned_roi_candidate_summary.json`: full candidate-screen summary.
- `validation/`: image-level and subject-level validation predictions.
- `calibration/`: prediction-set and uncertainty calibration.
- `diagnostics/`: provider and cohort-confounding diagnostics.
- `evidence/`: review HTML and nearest examples.
- `feature_sets/`: learned ROI feature tables.

`summary/` contains first-read verdict material. Aggregate subject/cohort interval outputs remain under `summaries/`.
"""
    paths['index'].write_text(index_text, encoding='utf-8')
    return {
        'learned_roi_index': paths['index'],
        'learned_roi_estimator_verdict': verdict_path,
        'learned_roi_estimator_verdict_md': paths['estimator_verdict_md'],
        'learned_roi_artifact_manifest': manifest_path,
    }


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ''


def _provider_audit_entry(provider_id: str) -> dict[str, Any]:
    if provider_id == 'current_glomeruli_encoder':
        return {
            'provider_id': provider_id,
            'status': 'available_fit_allowed',
            'fit_allowed': True,
            'module': 'existing_embeddings_csv',
            'package_version': '',
            'model_weight_provenance': 'embeddings/roi_embeddings.csv',
            'network_or_download_required': False,
            'failure_message': '',
        }
    if provider_id == 'simple_roi_qc':
        return {
            'provider_id': provider_id,
            'status': 'available_fit_allowed',
            'fit_allowed': True,
            'module': 'eq.quantification.learned_roi',
            'package_version': '',
            'model_weight_provenance': 'derived_from_roi_table',
            'network_or_download_required': False,
            'failure_message': '',
        }
    if provider_id == 'torchvision_resnet18_imagenet':
        available = importlib.util.find_spec('torchvision') is not None
        return {
            'provider_id': provider_id,
            'status': 'available_audit_only' if available else 'unavailable',
            'fit_allowed': False,
            'module': 'torchvision',
            'package_version': _package_version('torchvision') if available else '',
            'model_weight_provenance': 'not_loaded_phase_1_audit_only',
            'network_or_download_required': False,
            'failure_message': ''
            if available
            else 'torchvision import specification not found',
        }
    if provider_id == 'timm_dino_or_vit':
        available = importlib.util.find_spec('timm') is not None
        return {
            'provider_id': provider_id,
            'status': 'available_audit_only' if available else 'unavailable',
            'fit_allowed': False,
            'module': 'timm',
            'package_version': _package_version('timm') if available else '',
            'model_weight_provenance': 'not_loaded_phase_1_audit_only',
            'network_or_download_required': False,
            'failure_message': ''
            if available
            else 'timm import specification not found',
        }
    return {
        'provider_id': provider_id,
        'status': 'failed',
        'fit_allowed': False,
        'module': '',
        'package_version': '',
        'model_weight_provenance': '',
        'network_or_download_required': False,
        'failure_message': f'Unknown provider: {provider_id}',
    }


def audit_learned_roi_providers(output_path: Path) -> dict[str, Any]:
    """Audit provider availability without fitting optional providers."""
    providers = [
        _provider_audit_entry('current_glomeruli_encoder'),
        _provider_audit_entry('simple_roi_qc'),
        _provider_audit_entry('torchvision_resnet18_imagenet'),
        _provider_audit_entry('timm_dino_or_vit'),
    ]
    payload = {
        'phase': 'phase_1_capped_candidate_screen',
        'status_values': [
            'available_fit_allowed',
            'available_audit_only',
            'unavailable',
            'failed',
        ],
        'fit_allowed_provider_ids': sorted(FIT_ALLOWED_PROVIDERS),
        'audit_only_provider_ids': sorted(AUDIT_ONLY_PROVIDERS),
        'providers': providers,
        'network_download_policy': 'not_allowed_in_phase_1',
    }
    _save_json(payload, output_path)
    return payload


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith('embedding_')]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _image_stats(path: Any) -> dict[str, float]:
    try:
        image = Image.open(Path(str(path))).convert('RGB')
    except (FileNotFoundError, OSError, ValueError):
        return {
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'red_mean': 0.0,
            'green_mean': 0.0,
            'blue_mean': 0.0,
        }
    arr = np.asarray(image, dtype=np.float32)
    return {
        'mean_intensity': float(arr.mean()),
        'std_intensity': float(arr.std()),
        'red_mean': float(arr[:, :, 0].mean()),
        'green_mean': float(arr[:, :, 1].mean()),
        'blue_mean': float(arr[:, :, 2].mean()),
    }


def build_learned_roi_feature_table(
    embedding_df: pd.DataFrame, feature_sets_dir: Path, diagnostics_dir: Path
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Write phase-1 learned ROI feature tables from existing embeddings."""
    embedding_columns = _embedding_columns(embedding_df)
    if not embedding_columns:
        raise LearnedROIError('Embedding table does not contain embedding columns.')
    required = {'subject_id', 'score'}
    missing = sorted(required - set(embedding_df.columns))
    if missing:
        raise LearnedROIError(f'Embedding table is missing required columns: {missing}')

    identity_columns = [
        column
        for column in [
            'subject_id',
            'sample_id',
            'image_id',
            'subject_image_id',
            'glomerulus_id',
            'cohort_id',
            'score',
            'roi_image_path',
            'roi_mask_path',
            'raw_image_path',
            'raw_mask_path',
            'fold',
        ]
        if column in embedding_df.columns
    ]
    feature_blocks: list[pd.DataFrame] = [embedding_df[identity_columns].copy()]
    feature_blocks.append(
        pd.DataFrame(
            {
                f'learned_current_glomeruli_encoder_{column.removeprefix("embedding_")}': _safe_numeric(
                    embedding_df[column]
                ).fillna(0.0)
                for column in embedding_columns
            },
            index=embedding_df.index,
        )
    )

    roi_source_columns = [
        'roi_area',
        'roi_fill_fraction',
        'roi_mean_intensity',
        'roi_openness_score',
        'roi_component_count',
        'roi_union_bbox_width',
        'roi_union_bbox_height',
        'roi_largest_component_area_fraction',
    ]
    simple_qc_features: dict[str, pd.Series] = {}
    for column in roi_source_columns:
        if column in embedding_df.columns:
            simple_qc_features[f'learned_simple_roi_qc_{column}'] = _safe_numeric(
                embedding_df[column]
            ).fillna(0.0)

    if 'roi_image_path' in embedding_df.columns:
        image_stat_rows = [
            _image_stats(path) for path in embedding_df['roi_image_path']
        ]
        image_stats = pd.DataFrame(image_stat_rows)
        for column in image_stats.columns:
            simple_qc_features[f'learned_simple_roi_qc_roi_{column}'] = image_stats[
                column
            ]
    if simple_qc_features:
        feature_blocks.append(
            pd.DataFrame(simple_qc_features, index=embedding_df.index)
        )

    features = pd.concat(feature_blocks, axis=1).copy()

    provider_columns = [
        column for column in features.columns if column.startswith('learned_')
    ]
    nonfinite_by_column = {
        column: int((~np.isfinite(features[column].to_numpy(dtype=np.float64))).sum())
        for column in provider_columns
    }
    for column, nonfinite_count in nonfinite_by_column.items():
        if nonfinite_count:
            raise LearnedROIError(
                f'Learned ROI feature column contains nonfinite values: {column}'
            )

    feature_sets_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_sets_dir / 'learned_roi_features.csv'
    metadata_path = feature_sets_dir / 'learned_roi_feature_metadata.json'
    diagnostics_path = diagnostics_dir / 'learned_roi_feature_diagnostics.json'
    features.to_csv(feature_path, index=False)

    provider_feature_counts = {
        'current_glomeruli_encoder': int(
            len(
                [
                    column
                    for column in provider_columns
                    if column.startswith('learned_current_glomeruli_encoder_')
                ]
            )
        ),
        'simple_roi_qc': int(
            len(
                [
                    column
                    for column in provider_columns
                    if column.startswith('learned_simple_roi_qc_')
                ]
            )
        ),
    }
    metadata = {
        'provider_ids': ['current_glomeruli_encoder', 'simple_roi_qc'],
        'provider_feature_counts': provider_feature_counts,
        'preprocessing_transforms': {
            'current_glomeruli_encoder': 'reuse_existing_roi_embeddings_csv',
            'simple_roi_qc': 'roi_scalar_features_plus_basic_rgb_statistics',
        },
        'image_size_policy': 'reuse_existing_roi_crop_size',
        'mask_use_policy': 'use_existing_roi_mask_geometry_columns_when_present',
        'package_versions': {'numpy': np.__version__, 'pandas': pd.__version__},
        'feature_table_shape': [int(features.shape[0]), int(features.shape[1])],
    }
    _save_json(metadata, metadata_path)

    matrix = features[provider_columns].to_numpy(dtype=np.float64)
    variances = np.var(matrix, axis=0) if matrix.size else np.array([])
    diagnostics = {
        'row_count': int(len(features)),
        'subject_count': int(features['subject_id'].nunique()),
        'provider_count': 2,
        'feature_count': int(len(provider_columns)),
        'feature_count_by_provider': provider_feature_counts,
        'nonfinite_counts': nonfinite_by_column,
        'zero_variance_count': int(np.sum(np.isclose(variances, 0.0))),
        'near_zero_variance_count': int(np.sum(variances < 1e-12)),
        'missingness_counts': {
            column: int(features[column].isna().sum()) for column in features.columns
        },
        'feature_ranges': {
            column: {
                'min': float(features[column].min()),
                'max': float(features[column].max()),
            }
            for column in provider_columns
        },
        'matrix_rank': int(np.linalg.matrix_rank(matrix)) if matrix.size else 0,
        'singular_values': [
            float(value) for value in np.linalg.svd(matrix, compute_uv=False)[:25]
        ]
        if matrix.size
        else [],
    }
    _save_json(diagnostics, diagnostics_path)
    return features, {
        'learned_roi_features': feature_path,
        'learned_roi_feature_metadata': metadata_path,
        'learned_roi_feature_diagnostics': diagnostics_path,
    }


def _matching_warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for warning_message in caught:
        text = str(warning_message.message)
        lower = text.lower()
        if any(pattern in lower for pattern in NUMERICAL_INSTABILITY_PATTERNS):
            messages.append(text)
    return list(dict.fromkeys(messages))


def _feature_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return (
        df.loc[:, list(columns)]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _top_variance_columns(
    df: pd.DataFrame, columns: Sequence[str], max_columns: int
) -> list[str]:
    if len(columns) <= max_columns:
        return list(columns)
    matrix = _feature_matrix(df, columns)
    variances = np.var(matrix, axis=0)
    order = np.argsort(variances)[::-1][:max_columns]
    column_list = list(columns)
    return [column_list[int(index)] for index in order]


def _score_from_stage_index(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    stage_values = score_to_stage_index(ALLOWED_SCORE_VALUES)
    result = np.zeros(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        nearest = int(np.argmin(np.abs(stage_values - value)))
        result[index] = ALLOWED_SCORE_VALUES[nearest]
    return result


def _conformal_quantile(values: np.ndarray, coverage: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 100.0
    quantile_level = min(1.0, np.ceil((len(finite) + 1) * coverage) / len(finite))
    return float(np.quantile(finite, quantile_level, method='higher'))


def _prediction_set_from_interval(low: float, high: float) -> str:
    stage_values = score_to_stage_index(ALLOWED_SCORE_VALUES)
    selected = [
        float(score)
        for score, stage_value in zip(ALLOWED_SCORE_VALUES, stage_values)
        if float(low) <= float(stage_value) <= float(high)
    ]
    if not selected:
        midpoint = (float(low) + float(high)) / 2.0
        selected = [
            float(ALLOWED_SCORE_VALUES[int(np.argmin(abs(stage_values - midpoint)))])
        ]
    return '|'.join(f'{value:g}' for value in sorted(set(selected)))


def _prediction_set_contains(prediction_set: str, score: float) -> bool:
    return any(
        np.isclose(float(value), score)
        for value in str(prediction_set).split('|')
        if value != ''
    )


def _empirical_coverage(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {'n': 0, 'coverage': None, 'average_set_size': None}
    contains = [
        _prediction_set_contains(row['prediction_set_scores'], float(row['score']))
        for _, row in predictions.iterrows()
    ]
    set_sizes = [
        len([value for value in str(item).split('|') if value != ''])
        for item in predictions['prediction_set_scores']
    ]
    return {
        'n': int(len(predictions)),
        'coverage': float(np.mean(contains)),
        'average_set_size': float(np.mean(set_sizes)),
    }


def _grouped_image_predictions(
    df: pd.DataFrame, columns: Sequence[str], candidate_id: str, n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    x = _feature_matrix(df, columns)
    scores = df['score'].astype(float).to_numpy()
    stage_targets = score_to_stage_index(scores)
    groups = df['subject_id'].astype(str).to_numpy()
    split_count = min(max(2, n_splits), len(np.unique(groups)))
    predictions = np.zeros(len(df), dtype=np.float64)
    folds = np.zeros(len(df), dtype=np.int64)
    residual_quantiles: dict[int, float] = {}
    warning_messages: list[str] = []

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=split_count).split(x, stage_targets, groups=groups), start=1
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = Ridge(alpha=1.0, solver='svd')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x_train, stage_targets[train_idx])
            fold_predictions = model.predict(x_test)
        warning_messages.extend(_matching_warning_messages(caught))
        fold_predictions = np.clip(fold_predictions, 0.0, 100.0)
        predictions[test_idx] = fold_predictions
        folds[test_idx] = fold
        residual_quantiles[fold] = _conformal_quantile(
            np.abs(stage_targets[test_idx] - fold_predictions), PREDICTION_SET_COVERAGE
        )

    result = df.drop(
        columns=[column for column in df.columns if column.startswith('learned_')],
        errors='ignore',
    ).copy()
    result['candidate_id'] = candidate_id
    result['provider_id'] = candidate_id.removeprefix('image_')
    result['fold'] = folds
    result['observed_stage_index'] = stage_targets
    result[BURDEN_COLUMN] = predictions
    result['predicted_score'] = _score_from_stage_index(predictions)
    result['stage_index_absolute_error'] = np.abs(stage_targets - predictions)
    result['grade_scale_absolute_error'] = np.abs(
        scores - result['predicted_score'].to_numpy(dtype=np.float64)
    )
    lows = []
    highs = []
    sets = []
    for value, fold in zip(predictions, folds):
        quantile = residual_quantiles[int(fold)]
        low = float(np.clip(value - quantile, 0.0, 100.0))
        high = float(np.clip(value + quantile, 0.0, 100.0))
        lows.append(low)
        highs.append(high)
        sets.append(_prediction_set_from_interval(low, high))
    result['burden_interval_low_0_100'] = lows
    result['burden_interval_high_0_100'] = highs
    result['burden_interval_coverage'] = PREDICTION_SET_COVERAGE
    result['prediction_set_scores'] = sets
    result['prediction_source'] = 'learned_roi_held_out_subject_fold'
    result['prediction_set_method'] = 'grouped_oof_stage_residual_conformal_interval'
    return result, list(dict.fromkeys(warning_messages))


def _subject_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    work = df.copy()
    if 'cohort_id' not in work.columns:
        work['cohort_id'] = 'unknown_cohort'
    grouped = work.groupby('subject_id', dropna=False)
    numeric = grouped[list(columns)].mean().reset_index()
    meta = grouped.agg(
        score=('score', 'mean'),
        cohort_id=(
            'cohort_id',
            lambda values: str(values.iloc[0]) if len(values) else '',
        ),
        n_image_rows=('score', 'size'),
    ).reset_index()
    return meta.merge(numeric, on='subject_id', how='left', validate='one_to_one')


def _kfold_subject_predictions(
    subject_df: pd.DataFrame, columns: Sequence[str], candidate_id: str, n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    x = _feature_matrix(subject_df, columns)
    stage_targets = score_to_stage_index(
        _score_from_stage_index(
            subject_df['score'].to_numpy(dtype=np.float64) * 100.0 / 3.0
        )
    )
    observed_stage = subject_df['score'].to_numpy(dtype=np.float64) * (100.0 / 3.0)
    split_count = min(max(2, n_splits), len(subject_df))
    predictions = np.zeros(len(subject_df), dtype=np.float64)
    folds = np.zeros(len(subject_df), dtype=np.int64)
    warnings_seen: list[str] = []
    for fold, (train_idx, test_idx) in enumerate(
        KFold(n_splits=split_count, shuffle=True, random_state=0).split(x), start=1
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = Ridge(alpha=1.0, solver='svd')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x_train, observed_stage[train_idx])
            fold_predictions = model.predict(x_test)
        warnings_seen.extend(_matching_warning_messages(caught))
        predictions[test_idx] = np.clip(fold_predictions, 0.0, 100.0)
        folds[test_idx] = fold
    result = subject_df[['subject_id', 'cohort_id', 'n_image_rows', 'score']].copy()
    result['candidate_id'] = candidate_id
    result['fold'] = folds
    result['observed_subject_stage_index_mean'] = observed_stage
    result['predicted_subject_burden_0_100'] = predictions
    result['subject_stage_index_absolute_error'] = np.abs(observed_stage - predictions)
    result['predicted_score'] = predictions / (100.0 / 3.0)
    result['grade_scale_absolute_error'] = np.abs(
        result['score'] - result['predicted_score']
    )
    result['prediction_source'] = 'learned_roi_subject_kfold_prediction'
    result['subject_row_contract'] = 'one_row_per_subject_id'
    return result, list(dict.fromkeys(warnings_seen))


def _finite_status(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        'finite': bool(np.isfinite(arr).all()),
        'nan_count': int(np.isnan(arr).sum()),
        'posinf_count': int(np.isposinf(arr).sum()),
        'neginf_count': int(np.isneginf(arr).sum()),
    }


def _candidate_metric_row(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    target_level: str,
    feature_set: str,
    feature_count: int,
    n_subjects: int,
    warning_messages: Sequence[str],
) -> dict[str, Any]:
    if target_level == 'image':
        observed = predictions['observed_stage_index'].to_numpy(dtype=np.float64)
        predicted = predictions[BURDEN_COLUMN].to_numpy(dtype=np.float64)
        coverage = _empirical_coverage(predictions)
        target_definition = 'image_stage_index_0_100'
        validation = 'subject_id_groupkfold'
    else:
        observed = predictions['observed_subject_stage_index_mean'].to_numpy(
            dtype=np.float64
        )
        predicted = predictions['predicted_subject_burden_0_100'].to_numpy(
            dtype=np.float64
        )
        coverage = {'coverage': '', 'average_set_size': ''}
        target_definition = 'mean_subject_stage_index_0_100'
        validation = 'subject_kfold_one_row_per_subject_id'
    finite = _finite_status(predicted)
    return {
        'candidate_id': candidate_id,
        'target_level': target_level,
        'target_definition': target_definition,
        'model_family': 'ridge_regression',
        'feature_set': feature_set,
        'validation_grouping': validation,
        'n_rows': int(len(predictions)),
        'n_subjects': int(n_subjects),
        'feature_count': int(feature_count),
        'stage_index_mae': float(mean_absolute_error(observed, predicted)),
        'grade_scale_mae': float(predictions['grade_scale_absolute_error'].mean()),
        'prediction_set_coverage': coverage.get('coverage', ''),
        'average_prediction_set_size': coverage.get('average_set_size', ''),
        'prediction_set_status': 'grouped_oof_calibrated'
        if target_level == 'image'
        else 'not_applicable_subject_regression',
        'prediction_output_finite': bool(finite['finite']),
        'nan_count': int(finite['nan_count']),
        'posinf_count': int(finite['posinf_count']),
        'neginf_count': int(finite['neginf_count']),
        'backend_warning_count': int(len(warning_messages)),
        'backend_warning_messages': ' | '.join(dict.fromkeys(warning_messages)),
        'candidate_status': 'valid_finite'
        if finite['finite'] and not warning_messages
        else 'finite_with_backend_warnings'
        if finite['finite']
        else 'invalid_nonfinite_output',
        'stage_index_recoding_note': '0_100_stage_index_is_operational_ordinal_recoding_not_tissue_percent',
    }


def _candidate_specs(feature_df: pd.DataFrame) -> list[CandidateSpec]:
    encoder = tuple(
        column
        for column in feature_df.columns
        if column.startswith('learned_current_glomeruli_encoder_')
    )
    qc = tuple(
        column
        for column in feature_df.columns
        if column.startswith('learned_simple_roi_qc_')
    )
    return [
        CandidateSpec(
            'image_current_glomeruli_encoder',
            'image',
            'current_glomeruli_encoder',
            encoder,
        ),
        CandidateSpec('image_simple_roi_qc', 'image', 'simple_roi_qc', qc),
        CandidateSpec(
            'image_current_glomeruli_encoder_plus_simple_roi_qc',
            'image',
            'current_glomeruli_encoder_plus_simple_roi_qc',
            encoder + qc,
        ),
        CandidateSpec(
            'subject_current_glomeruli_encoder',
            'subject',
            'current_glomeruli_encoder',
            encoder,
        ),
        CandidateSpec('subject_simple_roi_qc', 'subject', 'simple_roi_qc', qc),
        CandidateSpec(
            'subject_current_glomeruli_encoder_plus_simple_roi_qc',
            'subject',
            'current_glomeruli_encoder_plus_simple_roi_qc',
            encoder + qc,
        ),
    ]


def _coverage_by_score(predictions: pd.DataFrame) -> dict[str, Any]:
    return {
        f'{score:g}': _empirical_coverage(
            predictions[np.isclose(predictions['score'].astype(float), score)]
        )
        for score in ALLOWED_SCORE_VALUES
    }


def _coverage_by_cohort(predictions: pd.DataFrame) -> dict[str, Any]:
    if 'cohort_id' not in predictions.columns:
        return {}
    return {
        str(cohort): _empirical_coverage(subset)
        for cohort, subset in predictions.groupby('cohort_id')
    }


def _grouped_bootstrap_intervals(subject_predictions: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    strata = [('overall', subject_predictions)]
    if 'cohort_id' in subject_predictions.columns:
        strata.extend(
            (f'cohort:{cohort}', subset)
            for cohort, subset in subject_predictions.groupby('cohort_id')
        )
    for stratum, subset in strata:
        values = subset['predicted_subject_burden_0_100'].to_numpy(dtype=np.float64)
        n = len(values)
        if n >= 3:
            boot = [
                float(np.mean(rng.choice(values, size=n, replace=True)))
                for _ in range(1000)
            ]
            low, high = np.quantile(boot, [0.025, 0.975])
            status = 'estimable'
        else:
            low = high = np.nan
            status = 'unstable_small_subject_count'
        rows.append(
            {
                'stratum': stratum,
                'n_subjects': int(n),
                'mean_predicted_subject_burden_0_100': float(np.mean(values))
                if n
                else '',
                'ci_low_0_100': float(low) if np.isfinite(low) else '',
                'ci_high_0_100': float(high) if np.isfinite(high) else '',
                'interval_type': 'grouped_bootstrap_confidence_interval',
                'status': status,
            }
        )
    return {'rows': rows}


def _cohort_predictability_screen(
    df: pd.DataFrame, columns: Sequence[str]
) -> dict[str, Any]:
    if 'cohort_id' not in df.columns or df['cohort_id'].nunique() < 2:
        return {'status': 'not_estimable_single_cohort'}
    counts = df['cohort_id'].astype(str).value_counts()
    if counts.min() < 3:
        return {'status': 'not_estimable_small_cohort_count'}
    selected_columns = _top_variance_columns(df, columns, max_columns=16)
    x = _feature_matrix(df, selected_columns)
    y = df['cohort_id'].astype(str).to_numpy()
    split_count = min(3, int(counts.min()))
    predictions = np.empty(len(df), dtype=object)
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=0)
    warning_messages: list[str] = []
    for train_idx, test_idx in splitter.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x_train, y[train_idx])
            predictions[test_idx] = model.predict(x_test)
        warning_messages.extend(_matching_warning_messages(caught))
    return {
        'status': 'estimated',
        'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
        'n_cohorts': int(len(counts)),
        'cohort_counts': {str(key): int(value) for key, value in counts.items()},
        'feature_count': int(len(selected_columns)),
        'warning_messages': list(dict.fromkeys(warning_messages)),
    }


def _leave_one_cohort_diagnostic(
    df: pd.DataFrame, columns: Sequence[str]
) -> dict[str, Any]:
    if 'cohort_id' not in df.columns or df['cohort_id'].nunique() < 2:
        return {'status': 'not_estimable_single_cohort'}
    rows: list[dict[str, Any]] = []
    selected_columns = _top_variance_columns(df, columns, max_columns=16)
    x_all = _feature_matrix(df, selected_columns)
    stage_targets = score_to_stage_index(df['score'].astype(float).to_numpy())
    for cohort in sorted(df['cohort_id'].astype(str).unique()):
        test_mask = df['cohort_id'].astype(str).to_numpy() == cohort
        train_mask = ~test_mask
        if train_mask.sum() < 10 or test_mask.sum() < 10:
            rows.append(
                {
                    'heldout_cohort': cohort,
                    'status': 'not_estimable_small_row_count',
                    'train_rows': int(train_mask.sum()),
                    'test_rows': int(test_mask.sum()),
                }
            )
            continue
        model = Ridge(alpha=1.0, solver='svd')
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all[train_mask])
        x_test = scaler.transform(x_all[test_mask])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x_train, stage_targets[train_mask])
            pred = np.clip(model.predict(x_test), 0.0, 100.0)
        warning_messages = _matching_warning_messages(caught)
        finite = _finite_status(pred)
        rows.append(
            {
                'heldout_cohort': cohort,
                'status': 'estimated' if finite['finite'] else 'nonfinite_predictions',
                'train_rows': int(train_mask.sum()),
                'test_rows': int(test_mask.sum()),
                'stage_index_mae': float(
                    mean_absolute_error(stage_targets[test_mask], pred)
                )
                if finite['finite']
                else '',
                'feature_count': int(len(selected_columns)),
                'warning_messages': warning_messages,
            }
        )
    return {'status': 'estimated', 'rows': rows}


def _write_cohort_diagnostics(
    feature_df: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    selected_columns: Sequence[str],
    output_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    by_cohort = {}
    if 'cohort_id' in feature_df.columns:
        for cohort, subset in feature_df.groupby('cohort_id'):
            by_cohort[str(cohort)] = {
                'n_rows': int(len(subset)),
                'n_subjects': int(subset['subject_id'].nunique()),
                'score_distribution': {
                    f'{score:g}': int(
                        np.isclose(subset['score'].astype(float), score).sum()
                    )
                    for score in ALLOWED_SCORE_VALUES
                },
            }
    coverage = _coverage_by_cohort(selected_predictions)
    residuals = {}
    for cohort, subset in (
        selected_predictions.groupby('cohort_id')
        if 'cohort_id' in selected_predictions.columns
        else []
    ):
        residuals[str(cohort)] = {
            'n_rows': int(len(subset)),
            'stage_index_mae': float(subset['stage_index_absolute_error'].mean()),
            'grade_scale_mae': float(subset['grade_scale_absolute_error'].mean()),
        }
    cohort_predictability = _cohort_predictability_screen(feature_df, selected_columns)
    leave_one = _leave_one_cohort_diagnostic(feature_df, selected_columns)

    blockers: list[str] = []
    for cohort, stats in coverage.items():
        if (
            int(stats.get('n') or 0) >= 30
            and float(stats.get('coverage') or 0.0) < 0.80
        ):
            blockers.append(f'cohort_coverage_below_0p80:{cohort}')
    mae_values = [
        float(stats['grade_scale_mae'])
        for stats in residuals.values()
        if stats.get('grade_scale_mae') != ''
    ]
    if mae_values and max(mae_values) - min(mae_values) >= 0.35:
        blockers.append('cohort_grade_scale_mae_difference_ge_0p35')
    if (
        cohort_predictability.get('status') == 'estimated'
        and float(cohort_predictability.get('balanced_accuracy') or 0.0) >= 0.80
    ):
        blockers.append('selected_features_predict_cohort_balanced_accuracy_ge_0p80')
    if cohort_predictability.get('warning_messages'):
        blockers.append('cohort_predictability_numerical_instability_warnings')
    for row in leave_one.get('rows', []):
        if row.get('status') == 'nonfinite_predictions':
            blockers.append(f'leave_one_cohort_nonfinite:{row.get("heldout_cohort")}')
        if row.get('warning_messages'):
            blockers.append(
                f'leave_one_cohort_numerical_instability_warnings:{row.get("heldout_cohort")}'
            )

    payload = {
        'cohort_counts': by_cohort,
        'candidate_residual_summaries_by_cohort': residuals,
        'prediction_set_coverage_by_cohort': coverage,
        'average_prediction_set_size_by_cohort': {
            cohort: stats.get('average_set_size') for cohort, stats in coverage.items()
        },
        'cohort_predictability_screen': cohort_predictability,
        'leave_one_cohort_out': leave_one,
        'readiness_blockers': blockers,
        'status': 'blocked' if blockers else 'passed',
    }
    _save_json(payload, output_path)
    return payload, blockers


def _readiness_summary(
    *,
    best_image: dict[str, Any],
    best_subject: dict[str, Any],
    image_predictions: pd.DataFrame,
    subject_predictions: pd.DataFrame,
    cohort_blockers: Sequence[str],
    image_warnings: Sequence[str],
    subject_warnings: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_blockers: list[str] = []
    coverage = _empirical_coverage(image_predictions)
    if float(coverage.get('coverage') or 0.0) < 0.88:
        image_blockers.append('overall_prediction_set_coverage_below_0p88')
    if (
        float(coverage.get('average_set_size') or 999.0)
        > IMAGE_READY_MAX_AVERAGE_SET_SIZE
    ):
        image_blockers.append('average_prediction_set_size_above_4p0')
    for score, stats in _coverage_by_score(image_predictions).items():
        if (
            int(stats.get('n') or 0) >= 30
            and float(stats.get('coverage') or 0.0) < 0.80
        ):
            image_blockers.append(f'observed_score_coverage_below_0p80:{score}')
    for cohort, stats in _coverage_by_cohort(image_predictions).items():
        if (
            int(stats.get('n') or 0) >= 30
            and float(stats.get('coverage') or 0.0) < 0.80
        ):
            image_blockers.append(f'cohort_coverage_below_0p80:{cohort}')
    if image_warnings:
        image_blockers.append('unresolved_numerical_instability_warnings')
    image_blockers.extend(cohort_blockers)

    subject_blockers: list[str] = []
    if subject_predictions['subject_id'].duplicated().any():
        subject_blockers.append('subject_predictions_not_one_row_per_subject_id')
    if subject_warnings:
        subject_blockers.append('unresolved_numerical_instability_warnings')
    subject_blockers.extend(cohort_blockers)
    return (
        {
            'status': 'ready' if not image_blockers else 'blocked',
            'readme_docs_ready': not image_blockers,
            'blockers': list(dict.fromkeys(image_blockers)),
            'selected_candidate_id': best_image.get('candidate_id', ''),
            'coverage': coverage,
            'thresholds': {
                'overall_coverage_min': 0.88,
                'stratum_coverage_min': 0.80,
                'average_prediction_set_size_max': IMAGE_READY_MAX_AVERAGE_SET_SIZE,
            },
        },
        {
            'status': 'ready' if not subject_blockers else 'blocked',
            'readme_docs_ready': not subject_blockers,
            'blockers': list(dict.fromkeys(subject_blockers)),
            'selected_candidate_id': best_subject.get('candidate_id', ''),
            'claim_boundary': 'subject/cohort readiness does not imply operationally precise per-image predictions',
        },
    )


def _write_calibration(predictions: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    payload = {
        'calibration_method': 'grouped_out_of_fold_stage_residual_prediction_sets',
        'nominal_coverage': PREDICTION_SET_COVERAGE,
        'overall': _empirical_coverage(predictions),
        'by_cohort': _coverage_by_cohort(predictions),
        'by_observed_score': _coverage_by_score(predictions),
        'baseline_comparison': {
            'current_baseline_average_prediction_set_size': BASELINE_AVERAGE_SET_SIZE,
            'phase_1_ready_average_prediction_set_size_max': IMAGE_READY_MAX_AVERAGE_SET_SIZE,
            'materially_narrowed': float(
                _empirical_coverage(predictions).get('average_set_size') or 999.0
            )
            <= IMAGE_READY_MAX_AVERAGE_SET_SIZE,
        },
    }
    _save_json(payload, output_path)
    return payload


def _nearest_examples(
    predictions: pd.DataFrame,
    feature_df: pd.DataFrame,
    columns: Sequence[str],
    output_path: Path,
) -> Path:
    x = _feature_matrix(feature_df, columns)
    groups = feature_df['subject_id'].astype(str).to_numpy()
    ids = feature_df['subject_image_id'].astype(str).to_numpy()
    scores = feature_df['score'].astype(float).to_numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    neighbors = NearestNeighbors(n_neighbors=min(10, len(feature_df))).fit(x_scaled)
    distances, indices = neighbors.kneighbors(x_scaled)
    rows: list[dict[str, Any]] = []
    for row_index, subject_image_id in enumerate(ids):
        added = 0
        for distance, neighbor_index in zip(distances[row_index], indices[row_index]):
            neighbor_index = int(neighbor_index)
            if (
                neighbor_index == row_index
                or groups[neighbor_index] == groups[row_index]
            ):
                continue
            rows.append(
                {
                    'subject_image_id': subject_image_id,
                    'candidate_id': str(predictions.iloc[row_index]['candidate_id']),
                    'neighbor_rank': int(added + 1),
                    'neighbor_subject_image_id': ids[neighbor_index],
                    'neighbor_score': float(scores[neighbor_index]),
                    'neighbor_distance': float(distance),
                    'neighbor_cohort_id': str(
                        feature_df.iloc[neighbor_index].get('cohort_id', '')
                    ),
                    'neighbor_roi_image_path': str(
                        feature_df.iloc[neighbor_index].get('roi_image_path', '')
                    ),
                    'neighbor_raw_image_path': str(
                        feature_df.iloc[neighbor_index].get('raw_image_path', '')
                    ),
                }
            )
            added += 1
            if added >= 3:
                break
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def evaluate_learned_roi_quantification(
    embedding_df: pd.DataFrame, burden_output_dir: Path, n_splits: int = 3
) -> dict[str, Path]:
    """Evaluate the capped phase-1 learned ROI candidate screen."""
    paths = learned_roi_output_paths(burden_output_dir)
    file_keys = {
        'index',
        'estimator_verdict',
        'estimator_verdict_md',
        'artifact_manifest',
    }
    for key, path in paths.items():
        if key in file_keys:
            continue
        path.mkdir(parents=True, exist_ok=True)

    provider_audit_path = paths['diagnostics'] / 'provider_audit.json'
    provider_audit = audit_learned_roi_providers(provider_audit_path)
    feature_df, feature_artifacts = build_learned_roi_feature_table(
        embedding_df, paths['feature_sets'], paths['diagnostics']
    )

    metric_rows: list[dict[str, Any]] = []
    image_predictions: list[pd.DataFrame] = []
    subject_prediction_rows: list[pd.DataFrame] = []
    warning_by_candidate: dict[str, list[str]] = {}
    image_candidate_predictions: dict[str, pd.DataFrame] = {}
    subject_candidate_predictions: dict[str, pd.DataFrame] = {}

    for spec in _candidate_specs(feature_df):
        if not spec.feature_columns:
            continue
        if spec.target_level == 'image':
            predictions, warning_messages = _grouped_image_predictions(
                feature_df, spec.feature_columns, spec.candidate_id, n_splits
            )
            image_predictions.append(predictions)
            image_candidate_predictions[spec.candidate_id] = predictions
            n_subjects = int(feature_df['subject_id'].nunique())
        else:
            subject_df = _subject_table(feature_df, spec.feature_columns)
            predictions, warning_messages = _kfold_subject_predictions(
                subject_df, spec.feature_columns, spec.candidate_id, n_splits
            )
            subject_prediction_rows.append(predictions)
            subject_candidate_predictions[spec.candidate_id] = predictions
            n_subjects = int(subject_df['subject_id'].nunique())
        warning_by_candidate[spec.candidate_id] = warning_messages
        metric_rows.append(
            _candidate_metric_row(
                predictions,
                candidate_id=spec.candidate_id,
                target_level=spec.target_level,
                feature_set=spec.feature_set,
                feature_count=len(spec.feature_columns),
                n_subjects=n_subjects,
                warning_messages=warning_messages,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = paths['candidates'] / 'learned_roi_candidate_metrics.csv'
    metrics.to_csv(metrics_path, index=False)
    if metrics.empty:
        raise LearnedROIError('No phase-1 learned ROI candidates were fitted.')

    image_metrics = metrics[metrics['target_level'] == 'image'].copy()
    subject_metrics = metrics[metrics['target_level'] == 'subject'].copy()
    best_image = (
        image_metrics.sort_values(
            ['grade_scale_mae', 'stage_index_mae', 'candidate_id']
        )
        .iloc[0]
        .to_dict()
    )
    best_subject = (
        subject_metrics.sort_values(
            ['grade_scale_mae', 'stage_index_mae', 'candidate_id']
        )
        .iloc[0]
        .to_dict()
    )
    selected_image_predictions = image_candidate_predictions[
        str(best_image['candidate_id'])
    ]
    selected_subject_predictions = subject_candidate_predictions[
        str(best_subject['candidate_id'])
    ]
    all_image_predictions = pd.concat(image_predictions, ignore_index=True)
    all_subject_predictions = pd.concat(subject_prediction_rows, ignore_index=True)

    predictions_path = paths['validation'] / 'learned_roi_predictions.csv'
    all_image_predictions.to_csv(predictions_path, index=False)
    subject_predictions_path = (
        paths['validation'] / 'learned_roi_subject_predictions.csv'
    )
    all_subject_predictions.to_csv(subject_predictions_path, index=False)

    calibration_path = paths['calibration'] / 'learned_roi_calibration.json'
    _write_calibration(selected_image_predictions, calibration_path)
    subject_summary_path = (
        paths['summaries'] / 'learned_roi_subject_summary_intervals.json'
    )
    _save_json(
        _grouped_bootstrap_intervals(selected_subject_predictions), subject_summary_path
    )

    selected_spec = next(
        spec
        for spec in _candidate_specs(feature_df)
        if spec.candidate_id == best_image['candidate_id']
    )
    cohort_diagnostics_path = (
        paths['diagnostics'] / 'cohort_confounding_diagnostics.json'
    )
    cohort_diagnostics, cohort_blockers = _write_cohort_diagnostics(
        feature_df,
        selected_image_predictions,
        selected_spec.feature_columns,
        cohort_diagnostics_path,
    )
    image_readiness, subject_readiness = _readiness_summary(
        best_image=best_image,
        best_subject=best_subject,
        image_predictions=selected_image_predictions,
        subject_predictions=selected_subject_predictions,
        cohort_blockers=cohort_blockers,
        image_warnings=warning_by_candidate.get(str(best_image['candidate_id']), []),
        subject_warnings=warning_by_candidate.get(
            str(best_subject['candidate_id']), []
        ),
    )

    nearest_path = paths['evidence'] / 'learned_roi_nearest_examples.csv'
    _nearest_examples(
        selected_image_predictions,
        feature_df,
        selected_spec.feature_columns,
        nearest_path,
    )

    review_artifacts = write_learned_roi_review(
        selected_predictions=selected_image_predictions,
        nearest_examples_path=nearest_path,
        output_dir=paths['evidence'],
        candidate_summary={
            'best_image_level_candidate': best_image,
            'best_subject_level_candidate': best_subject,
            'per_image_readiness': image_readiness,
            'subject_cohort_readiness': subject_readiness,
        },
    )

    readme_ready = bool(
        image_readiness['readme_docs_ready'] or subject_readiness['readme_docs_ready']
    )
    summary = {
        'artifact_contract': {
            'metrics': 'candidates/learned_roi_candidate_metrics.csv',
            'predictions': 'validation/learned_roi_predictions.csv',
            'provider_audit': 'diagnostics/provider_audit.json',
            'calibration': 'calibration/learned_roi_calibration.json',
            'cohort_diagnostics': 'diagnostics/cohort_confounding_diagnostics.json',
            'review_html': 'evidence/learned_roi_review.html',
        },
        'candidate_count': int(len(metrics)),
        'phase_1_candidate_ids': sorted(metrics['candidate_id'].astype(str).tolist()),
        'provider_availability': provider_audit,
        'best_image_level_candidate': best_image,
        'best_subject_level_candidate': best_subject,
        'per_image_readiness': image_readiness,
        'subject_cohort_readiness': subject_readiness,
        'cohort_diagnostics_status': cohort_diagnostics.get('status'),
        'readme_docs_ready': readme_ready,
        'readme_docs_ready_track': 'image'
        if image_readiness['readme_docs_ready']
        else 'subject_cohort'
        if subject_readiness['readme_docs_ready']
        else '',
        'blockers': list(
            dict.fromkeys(
                [
                    *image_readiness.get('blockers', []),
                    *subject_readiness.get('blockers', []),
                ]
            )
        ),
        'next_action': 'promote_ready_track_with_claim_boundaries'
        if readme_ready
        else 'keep_learned_roi_exploratory_and_review_failed_gates',
        'claim_boundary': 'predictive grade-equivalent endotheliosis burden; not tissue percent, closed-capillary percent, causal evidence, or mechanistic proof',
    }
    summary_path = paths['candidates'] / 'learned_roi_candidate_summary.json'
    _save_json(summary, summary_path)

    first_read_artifacts = _write_learned_roi_first_read_artifacts(
        paths=paths,
        summary=summary,
        artifacts={
            'learned_roi_candidate_summary': summary_path,
            'learned_roi_candidate_metrics': metrics_path,
            'learned_roi_predictions': predictions_path,
            'learned_roi_subject_predictions': subject_predictions_path,
            'learned_roi_calibration': calibration_path,
            'learned_roi_subject_summary_intervals': subject_summary_path,
            'learned_roi_cohort_confounding_diagnostics': cohort_diagnostics_path,
            'learned_roi_nearest_examples': nearest_path,
            **review_artifacts,
            **feature_artifacts,
        },
    )

    return {
        'learned_roi_provider_audit': provider_audit_path,
        **first_read_artifacts,
        **feature_artifacts,
        'learned_roi_candidate_metrics': metrics_path,
        'learned_roi_predictions': predictions_path,
        'learned_roi_subject_predictions': subject_predictions_path,
        'learned_roi_calibration': calibration_path,
        'learned_roi_subject_summary_intervals': subject_summary_path,
        'learned_roi_cohort_confounding_diagnostics': cohort_diagnostics_path,
        'learned_roi_candidate_summary': summary_path,
        'learned_roi_nearest_examples': nearest_path,
        **review_artifacts,
    }
