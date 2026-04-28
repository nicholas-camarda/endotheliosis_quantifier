"""Burden-index modeling for image-level endotheliosis quantification."""

from __future__ import annotations

import json
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from eq.quantification.feature_review import write_morphology_feature_review
from eq.quantification.morphology_features import (
    MORPHOLOGY_FEATURE_COLUMNS,
    write_morphology_feature_tables,
)
from eq.quantification.ordinal import NUMERICAL_INSTABILITY_PATTERNS
from eq.utils.logger import get_logger

ALLOWED_SCORE_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64)
CUMULATIVE_THRESHOLDS = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
THRESHOLD_PROBABILITY_COLUMNS = [
    'prob_score_gt_0',
    'prob_score_gt_0p5',
    'prob_score_gt_1',
    'prob_score_gt_1p5',
    'prob_score_gt_2',
]
BURDEN_COLUMN = 'endotheliosis_burden_0_100'
STAGE_INDEX_TARGETS = {0.0: 0.0, 0.5: 20.0, 1.0: 40.0, 1.5: 60.0, 2.0: 80.0, 3.0: 100.0}
PREDICTION_SET_COVERAGE = 0.90
PRIMARY_BURDEN_INDEX_ROOT_NAME = 'primary_burden_index'
BURDEN_OUTPUT_GROUPS = {
    'primary_model': 'model',
    'validation': 'validation',
    'calibration': 'calibration',
    'summaries': 'summaries',
    'evidence': 'evidence',
    'candidates': 'candidates',
    'diagnostics': 'diagnostics',
    'feature_sets': 'feature_sets',
}
LEGACY_FLAT_BURDEN_FILES = [
    'grouping_audit.json',
    'threshold_support.csv',
    'burden_predictions.csv',
    'threshold_metrics.csv',
    'calibration_bins.csv',
    'uncertainty_calibration.json',
    'cohort_metrics.csv',
    'group_summary_intervals.csv',
    'validation_design.json',
    'prediction_explanations.csv',
    'nearest_examples.csv',
    'signal_comparator_metrics.csv',
    'subject_level_candidate_predictions.csv',
    'precision_candidate_summary.json',
    'burden_metrics.json',
    'final_model_predictions.csv',
    'final_model_cohort_metrics.csv',
    'final_model_group_summary_intervals.csv',
    'cohort_stability.csv',
    'burden_model.joblib',
]


def burden_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return the canonical grouped burden artifact directories."""
    root = Path(output_dir) / PRIMARY_BURDEN_INDEX_ROOT_NAME
    paths = {key: root / dirname for key, dirname in BURDEN_OUTPUT_GROUPS.items()}
    paths['root'] = root
    paths['index'] = root / 'INDEX.md'
    paths['morphology_review'] = paths['evidence'] / 'morphology_feature_review'
    return paths


def _prepare_burden_output_contract(output_dir: Path) -> dict[str, Path]:
    """Create grouped output directories and remove stale flat artifacts."""
    paths = burden_output_paths(output_dir)
    file_keys = {'index'}
    for key, directory in paths.items():
        if key in file_keys:
            continue
        directory.mkdir(parents=True, exist_ok=True)
    for filename in LEGACY_FLAT_BURDEN_FILES:
        stale_path = Path(output_dir) / filename
        if stale_path.is_file():
            stale_path.unlink()
    return paths


def _write_primary_burden_index(index_path: Path) -> Path:
    text = """# Primary Burden Index

This subtree contains the original grouped cumulative-threshold burden-index evaluator.

## First Read

- `model/burden_metrics.json`: validation metrics, support gates, calibration contract, and model provenance.
- `model/burden_predictions.csv`: subject-held-out grouped validation predictions.
- `model/final_model_predictions.csv`: final full-cohort fitted predictions.
- `summaries/`: aggregate cohort and interval summary tables.
- `validation/`: grouped validation design, threshold support, threshold metrics, and cohort stability.
- `calibration/`: calibration bins and uncertainty calibration.
- `evidence/`: reviewer evidence, nearest examples, and morphology feature review.
- `candidates/`: exploratory precision and morphology candidate screens.
- `feature_sets/`: deterministic feature tables used by candidate screens.
- `diagnostics/`: feature diagnostics and other machine-readable checks.

`summaries/` contains aggregate tables. Estimator `summary/` directories elsewhere under `burden_model/` contain first-read verdict bundles.
"""
    index_path.write_text(text, encoding='utf-8')
    return index_path


class BurdenModelError(ValueError):
    """Raised when burden-index modeling inputs violate the target contract."""


@dataclass
class _ConstantThresholdModel:
    probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((x.shape[0], 2), dtype=np.float64)
        probabilities[:, 1] = self.probability
        probabilities[:, 0] = 1.0 - self.probability
        return probabilities


def validate_score_values(scores: Sequence[float] | pd.Series) -> None:
    """Fail closed when unsupported image-level score values are present."""
    observed = pd.Series(scores).dropna().astype(float).unique()
    unsupported = sorted(
        float(value)
        for value in observed
        if not np.any(np.isclose(ALLOWED_SCORE_VALUES, value))
    )
    if unsupported:
        raise BurdenModelError(
            'Unsupported endotheliosis score values: '
            f'{unsupported}. Supported rubric: {ALLOWED_SCORE_VALUES.tolist()}'
        )


def score_to_stage_index(
    scores: Sequence[float] | pd.Series | np.ndarray,
) -> np.ndarray:
    validate_score_values(scores)
    values = np.asarray(scores, dtype=np.float64)
    result = np.zeros_like(values, dtype=np.float64)
    for score, stage_index in STAGE_INDEX_TARGETS.items():
        result[np.isclose(values, score)] = stage_index
    return result


def threshold_targets(scores: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    validate_score_values(values)
    return np.column_stack(
        [values > threshold for threshold in CUMULATIVE_THRESHOLDS]
    ).astype(np.int64)


def burden_from_threshold_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CUMULATIVE_THRESHOLDS):
        raise BurdenModelError(
            f'Expected threshold probability matrix with {len(CUMULATIVE_THRESHOLDS)} columns'
        )
    if not np.all(np.isfinite(probabilities)):
        raise BurdenModelError('Threshold probabilities must be finite')
    if np.any((probabilities < -1e-12) | (probabilities > 1.0 + 1e-12)):
        raise BurdenModelError('Threshold probabilities must be within [0, 1]')
    return 100.0 * probabilities.mean(axis=1)


def monotonic_threshold_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Project independent threshold probabilities to a valid cumulative sequence."""
    projected = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    for column_index in range(1, projected.shape[1]):
        projected[:, column_index] = np.minimum(
            projected[:, column_index - 1], projected[:, column_index]
        )
    return projected


def score_probabilities_from_thresholds(
    threshold_probabilities: np.ndarray,
) -> np.ndarray:
    p = monotonic_threshold_probabilities(threshold_probabilities)
    score_probabilities = np.zeros(
        (p.shape[0], len(ALLOWED_SCORE_VALUES)), dtype=np.float64
    )
    score_probabilities[:, 0] = 1.0 - p[:, 0]
    score_probabilities[:, 1] = p[:, 0] - p[:, 1]
    score_probabilities[:, 2] = p[:, 1] - p[:, 2]
    score_probabilities[:, 3] = p[:, 2] - p[:, 3]
    score_probabilities[:, 4] = p[:, 3] - p[:, 4]
    score_probabilities[:, 5] = p[:, 4]
    score_probabilities = np.clip(score_probabilities, 0.0, 1.0)
    row_sums = score_probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return score_probabilities / row_sums


def prediction_sets_from_score_probabilities(
    score_probabilities: np.ndarray,
    *,
    coverage: float = PREDICTION_SET_COVERAGE,
    conformity_quantiles: Sequence[float] | np.ndarray | None = None,
) -> list[str]:
    prediction_sets: list[str] = []
    quantiles = (
        np.asarray(conformity_quantiles, dtype=np.float64)
        if conformity_quantiles is not None
        else np.full(score_probabilities.shape[0], coverage, dtype=np.float64)
    )
    for row, quantile in zip(score_probabilities, quantiles):
        order = np.argsort(row)[::-1]
        total = 0.0
        selected: list[float] = []
        for index in order:
            selected.append(float(ALLOWED_SCORE_VALUES[index]))
            total += float(row[index])
            if total >= float(quantile):
                break
        selected = sorted(selected)
        prediction_sets.append('|'.join(f'{value:g}' for value in selected))
    return prediction_sets


def _prediction_set_contains(prediction_set: str, score: float) -> bool:
    values = [float(item) for item in str(prediction_set).split('|') if item != '']
    return any(np.isclose(value, score) for value in values)


def _matching_warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for warning_message in caught:
        text = str(warning_message.message)
        lower = text.lower()
        if any(pattern in lower for pattern in NUMERICAL_INSTABILITY_PATTERNS):
            messages.append(text)
    return list(dict.fromkeys(messages))


def _finite_matrix_status(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    return {
        'finite': bool(np.isfinite(values).all()),
        'nan_count': int(np.isnan(values).sum()),
        'posinf_count': int(np.isposinf(values).sum()),
        'neginf_count': int(np.isneginf(values).sum()),
    }


def _predict_threshold_probability(
    model: Any, x_test: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        probabilities = model.predict_proba(x_test)[:, 1]
    return probabilities, _matching_warning_messages(caught)


def _sanitize_threshold_label(threshold: float) -> str:
    return f'{threshold:g}'.replace('.', 'p')


def derive_biological_grouping(
    embedding_df: pd.DataFrame,
) -> tuple[str, pd.Series, dict[str, Any]]:
    if (
        'subject_id' in embedding_df.columns
        and embedding_df['subject_id'].notna().all()
        and embedding_df['subject_id'].astype(str).str.strip().ne('').all()
    ):
        key = 'subject_id'
        groups = embedding_df[key].astype(str)
        status = 'certified_from_manifest_subject_id'
    else:
        raise BurdenModelError('Embedding table must contain subject_id')
    audit = {
        'grouping_key': key,
        'grouping_status': status,
        'n_rows': int(len(embedding_df)),
        'n_subjects': int(groups.nunique()),
        'n_samples': int(embedding_df['sample_id'].nunique())
        if 'sample_id' in embedding_df.columns
        else 0,
        'subject_id_present': bool('subject_id' in embedding_df.columns),
        'sample_id_present': bool('sample_id' in embedding_df.columns),
        'manifest_owned_grouping_contract': key == 'subject_id',
        'certified_for_grouped_validation': True,
    }
    return key, groups, audit


def _conformal_quantile(values: np.ndarray, coverage: float) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return 1.0
    quantile_level = min(
        1.0, np.ceil((len(finite_values) + 1) * coverage) / len(finite_values)
    )
    return float(np.quantile(finite_values, quantile_level, method='higher'))


def _aps_conformity_scores(
    score_probabilities: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    conformity = np.zeros(score_probabilities.shape[0], dtype=np.float64)
    for row_index, (row, score) in enumerate(zip(score_probabilities, scores)):
        true_positions = np.where(np.isclose(ALLOWED_SCORE_VALUES, score))[0]
        if len(true_positions) != 1:
            raise BurdenModelError(f'Unsupported endotheliosis score: {score}')
        true_position = int(true_positions[0])
        total = 0.0
        for class_index in np.argsort(row)[::-1]:
            total += float(row[class_index])
            if int(class_index) == true_position:
                break
        conformity[row_index] = min(max(total, 0.0), 1.0)
    return conformity


def _fit_threshold_model(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[Any, list[str], str]:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    if positives == 0 or negatives == 0:
        probability = float(np.mean(y_train)) if len(y_train) else 0.0
        return _ConstantThresholdModel(probability), [], 'constant_single_class'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = LogisticRegression(
            penalty='l2', C=0.1, max_iter=4000, solver='lbfgs', random_state=0
        )
        model.fit(x_train, y_train)
    return model, _matching_warning_messages(caught), 'penalized_binary_logistic'


def _threshold_positive_counts(scores: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cohort_values: list[str | None] = [None]
    return pd.DataFrame(rows)


def _write_threshold_support(
    embedding_df: pd.DataFrame, groups: pd.Series, output_path: Path
) -> pd.DataFrame:
    scores = embedding_df['score'].astype(float).to_numpy()
    group_values = groups.astype(str).to_numpy()
    cohort_series = (
        embedding_df['cohort_id'].astype(str)
        if 'cohort_id' in embedding_df.columns
        else pd.Series(['all'] * len(embedding_df))
    )
    strata: list[tuple[str, np.ndarray]] = [
        ('overall', np.ones(len(embedding_df), dtype=bool))
    ]
    for cohort in sorted(cohort_series.unique()):
        strata.append((f'cohort:{cohort}', cohort_series.to_numpy() == cohort))

    rows: list[dict[str, Any]] = []
    for stratum, mask in strata:
        stratum_scores = scores[mask]
        stratum_groups = group_values[mask]
        for threshold in CUMULATIVE_THRESHOLDS:
            positive = stratum_scores > threshold
            negative = ~positive
            positive_groups = set(stratum_groups[positive])
            negative_groups = set(stratum_groups[negative])
            underpowered = len(positive_groups) < 2 or len(negative_groups) < 2
            rows.append(
                {
                    'stratum': stratum,
                    'threshold': float(threshold),
                    'positive_rows': int(positive.sum()),
                    'negative_rows': int(negative.sum()),
                    'positive_groups': int(len(positive_groups)),
                    'negative_groups': int(len(negative_groups)),
                    'support_status': 'underpowered' if underpowered else 'estimable',
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def _empirical_coverage(
    predictions: pd.DataFrame, mask: pd.Series | np.ndarray
) -> dict[str, Any]:
    subset = predictions.loc[mask].copy()
    if subset.empty:
        return {'n': 0, 'coverage': None, 'average_set_size': None}
    contains = [
        _prediction_set_contains(row['prediction_set_scores'], float(row['score']))
        for _, row in subset.iterrows()
    ]
    set_sizes = [
        len([item for item in str(value).split('|') if item != ''])
        for value in subset['prediction_set_scores']
    ]
    return {
        'n': int(len(subset)),
        'coverage': float(np.mean(contains)),
        'average_set_size': float(np.mean(set_sizes)),
    }


def _write_uncertainty_calibration(
    predictions: pd.DataFrame, output_path: Path, group_column: str
) -> Path:
    by_cohort: dict[str, Any] = {}
    if 'cohort_id' in predictions.columns:
        for cohort, subset in predictions.groupby('cohort_id'):
            by_cohort[str(cohort)] = _empirical_coverage(
                predictions, predictions.index.isin(subset.index)
            )
    by_score: dict[str, Any] = {}
    for score in ALLOWED_SCORE_VALUES:
        by_score[f'{score:g}'] = _empirical_coverage(
            predictions, np.isclose(predictions['score'].astype(float), score)
        )
    residuals = np.abs(
        predictions['observed_stage_index'].astype(float)
        - predictions[BURDEN_COLUMN].astype(float)
    )
    payload = {
        'calibration_method': 'grouped_out_of_fold_probability_prediction_sets',
        'prediction_set_method': 'grouped_fold_row_aps_conformal_quantile',
        'burden_interval_method': 'grouped_fold_row_absolute_residual_conformal_quantile',
        'grouping_column': group_column,
        'nominal_coverage': PREDICTION_SET_COVERAGE,
        'overall': _empirical_coverage(
            predictions, np.ones(len(predictions), dtype=bool)
        ),
        'by_cohort': by_cohort,
        'by_observed_score': by_score,
        'absolute_stage_residual_summary': {
            'min': float(residuals.min()),
            'median': float(residuals.median()),
            'p90': float(residuals.quantile(0.90)),
            'max': float(residuals.max()),
        },
        'burden_interval_coverage': float(
            np.mean(
                (
                    predictions['observed_stage_index']
                    >= predictions['burden_interval_low_0_100']
                )
                & (
                    predictions['observed_stage_index']
                    <= predictions['burden_interval_high_0_100']
                )
            )
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return output_path


def _cluster_label(group_column: str) -> str:
    return 'subject' if group_column == 'subject_id' else 'sample'


def _write_group_summary_intervals(
    predictions: pd.DataFrame, group_column: str, output_path: Path
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    strata = [('overall', predictions)]
    if 'cohort_id' in predictions.columns:
        strata.extend(
            (f'cohort:{cohort}', df) for cohort, df in predictions.groupby('cohort_id')
        )

    for stratum, df in strata:
        sample_means = df.groupby(group_column)[BURDEN_COLUMN].mean()
        values = sample_means.to_numpy(dtype=np.float64)
        n_clusters = len(values)
        status = 'estimable' if n_clusters >= 3 else 'unstable_small_cluster_count'
        if n_clusters:
            mean_value = float(np.mean(values))
            if n_clusters >= 3:
                boot = [
                    float(np.mean(rng.choice(values, size=n_clusters, replace=True)))
                    for _ in range(1000)
                ]
                low, high = np.quantile(boot, [0.025, 0.975])
            else:
                low, high = np.nan, np.nan
        else:
            mean_value, low, high = np.nan, np.nan, np.nan
        rows.append(
            {
                'stratum': stratum,
                'estimand': f'mean_of_{_cluster_label(group_column)}_level_mean_burden_indices',
                'resampling_unit': group_column,
                'weighting_rule': f'equal_weight_per_{_cluster_label(group_column)}',
                'n_clusters': int(n_clusters),
                'mean_burden_0_100': mean_value,
                'ci_low_0_100': float(low) if np.isfinite(low) else '',
                'ci_high_0_100': float(high) if np.isfinite(high) else '',
                'interval_type': 'grouped_bootstrap_confidence_interval',
                'status': status,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def _write_cohort_metrics(
    predictions: pd.DataFrame, group_column: str, output_path: Path
) -> pd.DataFrame:
    if 'cohort_id' not in predictions.columns:
        output_path.write_text('', encoding='utf-8')
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cohort, df in predictions.groupby('cohort_id'):
        sample_means = df.groupby(group_column)[BURDEN_COLUMN].mean()
        rows.append(
            {
                'cohort_id': cohort,
                'n_rows': int(len(df)),
                'n_subjects': int(df['subject_id'].nunique())
                if 'subject_id' in df.columns
                else int(df[group_column].nunique()),
                'n_samples': int(df['sample_id'].nunique())
                if 'sample_id' in df.columns
                else int(df[group_column].nunique()),
                'stage_index_mae': float(
                    mean_absolute_error(df['observed_stage_index'], df[BURDEN_COLUMN])
                ),
                'grade_scale_mae': float(
                    mean_absolute_error(df['score'], df['predicted_stage_score'])
                ),
                'image_row_mean_predicted_burden': float(df[BURDEN_COLUMN].mean()),
                f'{_cluster_label(group_column)}_weighted_mean_predicted_burden': float(
                    sample_means.mean()
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def _fit_direct_regression_comparator(
    x: np.ndarray, stage_targets: np.ndarray, groups: np.ndarray, split_count: int
) -> dict[str, Any]:
    predictions = np.zeros(len(stage_targets), dtype=np.float64)
    warning_messages: list[str] = []
    for train_idx, test_idx in GroupKFold(n_splits=split_count).split(
        x, stage_targets, groups=groups
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = Ridge(alpha=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x_train, stage_targets[train_idx])
            fold_predictions = model.predict(x_test)
        warning_messages.extend(_matching_warning_messages(caught))
        predictions[test_idx] = np.clip(fold_predictions, 0.0, 100.0)
    return {
        'model_family': 'direct_stage_index_ridge_regression',
        'stage_index_mae': float(mean_absolute_error(stage_targets, predictions)),
        'prediction_output_status': _finite_matrix_status(predictions),
        'backend_warning_messages': list(dict.fromkeys(warning_messages)),
    }


def _candidate_status(finite_status: dict[str, Any], warning_count: int) -> str:
    if not finite_status['finite']:
        return 'invalid_nonfinite_output'
    if warning_count:
        return 'finite_with_backend_warnings'
    return 'valid_finite'


def _feature_matrix(work_df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return (
        work_df.loc[:, list(columns)]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _grouped_ridge_predictions(
    features: np.ndarray,
    stage_targets: np.ndarray,
    groups: np.ndarray,
    split_count: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    predictions = np.zeros(len(stage_targets), dtype=np.float64)
    fold_assignments = np.zeros(len(stage_targets), dtype=np.int64)
    warning_messages: list[str] = []
    for train_idx, test_idx in GroupKFold(n_splits=split_count).split(
        features, stage_targets, groups=groups
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        model = Ridge(alpha=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x_train, stage_targets[train_idx])
            fold_predictions = model.predict(x_test)
        warning_messages.extend(_matching_warning_messages(caught))
        predictions[test_idx] = np.clip(fold_predictions, 0.0, 100.0)
        fold_assignments[test_idx] = int(np.max(fold_assignments) + 1)
    return predictions, list(dict.fromkeys(warning_messages)), fold_assignments


def _kfold_ridge_predictions(
    features: np.ndarray, stage_targets: np.ndarray, split_count: int
) -> tuple[np.ndarray, list[str], np.ndarray]:
    predictions = np.zeros(len(stage_targets), dtype=np.float64)
    fold_assignments = np.zeros(len(stage_targets), dtype=np.int64)
    warning_messages: list[str] = []
    splitter = KFold(n_splits=split_count, shuffle=True, random_state=0)
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(features), start=1
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        model = Ridge(alpha=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x_train, stage_targets[train_idx])
            fold_predictions = model.predict(x_test)
        warning_messages.extend(_matching_warning_messages(caught))
        predictions[test_idx] = np.clip(fold_predictions, 0.0, 100.0)
        fold_assignments[test_idx] = fold_index
    return predictions, list(dict.fromkeys(warning_messages)), fold_assignments


def _mean_baseline_predictions(
    stage_targets: np.ndarray, split_count: int
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros(len(stage_targets), dtype=np.float64)
    fold_assignments = np.zeros(len(stage_targets), dtype=np.int64)
    splitter = KFold(n_splits=split_count, shuffle=True, random_state=0)
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(stage_targets), start=1
    ):
        predictions[test_idx] = float(np.mean(stage_targets[train_idx]))
        fold_assignments[test_idx] = fold_index
    return predictions, fold_assignments


def _candidate_row(
    *,
    candidate_id: str,
    target_level: str,
    target_definition: str,
    model_family: str,
    feature_set: str,
    validation_grouping: str,
    n_rows: int,
    n_subjects: int,
    feature_count: int,
    stage_targets: np.ndarray,
    predictions: np.ndarray,
    warning_messages: Sequence[str],
    intended_use: str,
) -> dict[str, Any]:
    finite_status = _finite_matrix_status(predictions)
    observed_scores = np.interp(stage_targets, [0.0, 100.0], [0.0, 3.0])
    predicted_scores = np.interp(predictions, [0.0, 100.0], [0.0, 3.0])
    return {
        'candidate_id': candidate_id,
        'target_level': target_level,
        'target_definition': target_definition,
        'model_family': model_family,
        'feature_set': feature_set,
        'validation_grouping': validation_grouping,
        'n_rows': int(n_rows),
        'n_subjects': int(n_subjects),
        'feature_count': int(feature_count),
        'stage_index_mae': float(mean_absolute_error(stage_targets, predictions)),
        'grade_scale_mae': float(
            mean_absolute_error(observed_scores, predicted_scores)
        ),
        'prediction_set_coverage': '',
        'average_prediction_set_size': '',
        'prediction_set_status': 'not_applicable_direct_regression_screen',
        'cohort_metric_status': 'global_candidate_screen; inspect cohort summaries for primary burden artifacts',
        'prediction_output_finite': bool(finite_status['finite']),
        'nan_count': int(finite_status['nan_count']),
        'posinf_count': int(finite_status['posinf_count']),
        'neginf_count': int(finite_status['neginf_count']),
        'backend_warning_count': int(len(warning_messages)),
        'backend_warning_messages': ' | '.join(dict.fromkeys(warning_messages)),
        'candidate_status': _candidate_status(finite_status, len(warning_messages)),
        'intended_use': intended_use,
    }


def _write_signal_comparator_screen(
    work_df: pd.DataFrame,
    embedding_columns: Sequence[str],
    stage_targets: np.ndarray,
    groups: np.ndarray,
    split_count: int,
    output_path: Path,
) -> pd.DataFrame:
    """Screen image-level and subject-level candidates for precision follow-up."""
    rows: list[dict[str, Any]] = []
    roi_columns = [
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
        if column in work_df.columns
    ]

    image_feature_sets: list[tuple[str, str, list[str]]] = [
        ('image_embedding_only_ridge', 'frozen_embedding', list(embedding_columns))
    ]
    if roi_columns:
        image_feature_sets.extend(
            [
                ('image_roi_scalar_only_ridge', 'roi_scalar', roi_columns),
                (
                    'image_embedding_plus_roi_scalar_ridge',
                    'frozen_embedding_plus_roi_scalar',
                    list(embedding_columns) + roi_columns,
                ),
            ]
        )
    for candidate_id, feature_set, columns in image_feature_sets:
        features = _feature_matrix(work_df, columns)
        predictions, warning_messages, _folds = _grouped_ridge_predictions(
            features, stage_targets, groups, split_count
        )
        rows.append(
            _candidate_row(
                candidate_id=candidate_id,
                target_level='image',
                target_definition='image_stage_index_0_100',
                model_family='ridge_regression',
                feature_set=feature_set,
                validation_grouping='subject_id_groupkfold',
                n_rows=len(work_df),
                n_subjects=len(np.unique(groups)),
                feature_count=features.shape[1],
                stage_targets=stage_targets,
                predictions=predictions,
                warning_messages=warning_messages,
                intended_use='per_image_precision_screen_not_primary_model',
            )
        )

    subject_candidate_path = output_path.with_name(
        'subject_level_candidate_predictions.csv'
    )
    summary_path = output_path.with_name('precision_candidate_summary.json')
    subject_predictions, subject_rows = _write_subject_level_candidate_screen(
        work_df=work_df,
        embedding_columns=embedding_columns,
        roi_columns=roi_columns,
        stage_targets=stage_targets,
        split_count=split_count,
        output_path=subject_candidate_path,
    )
    rows.extend(subject_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    _write_precision_candidate_summary(
        result, subject_predictions, output_path=summary_path
    )
    return result


def _write_subject_level_candidate_screen(
    *,
    work_df: pd.DataFrame,
    embedding_columns: Sequence[str],
    roi_columns: Sequence[str],
    stage_targets: np.ndarray,
    split_count: int,
    output_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    subject_df = work_df.copy()
    if 'cohort_id' not in subject_df.columns:
        subject_df['cohort_id'] = ''
    subject_df['observed_stage_index'] = stage_targets
    feature_columns = list(embedding_columns) + list(roi_columns)
    aggregation: dict[str, Any] = {'cohort_id': 'first', 'observed_stage_index': 'mean'}
    for column in feature_columns:
        aggregation[column] = 'mean'
    grouped = subject_df.groupby('subject_id', as_index=False).agg(aggregation)
    grouped = grouped.rename(
        columns={'observed_stage_index': 'observed_subject_stage_index_mean'}
    )
    target = grouped['observed_subject_stage_index_mean'].to_numpy(dtype=np.float64)
    subject_split_count = min(max(2, split_count), len(grouped))
    if subject_split_count < 2:
        raise BurdenModelError('Need at least two subjects for subject-level screening')

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    baseline_predictions, baseline_folds = _mean_baseline_predictions(
        target, subject_split_count
    )
    prediction_rows.append(
        _subject_candidate_prediction_rows(
            grouped,
            candidate_id='subject_global_mean_baseline',
            model_family='training_fold_global_mean',
            feature_set='none',
            predictions=baseline_predictions,
            folds=baseline_folds,
        )
    )
    metric_rows.append(
        _candidate_row(
            candidate_id='subject_global_mean_baseline',
            target_level='subject',
            target_definition='mean_subject_stage_index_0_100',
            model_family='training_fold_global_mean',
            feature_set='none',
            validation_grouping='subject_kfold',
            n_rows=len(grouped),
            n_subjects=len(grouped),
            feature_count=0,
            stage_targets=target,
            predictions=baseline_predictions,
            warning_messages=[],
            intended_use='subject_cohort_summary_baseline',
        )
    )

    subject_feature_sets: list[tuple[str, str, list[str]]] = [
        (
            'subject_embedding_only_ridge',
            'mean_frozen_embedding',
            list(embedding_columns),
        )
    ]
    if roi_columns:
        subject_feature_sets.extend(
            [
                ('subject_roi_scalar_only_ridge', 'mean_roi_scalar', list(roi_columns)),
                (
                    'subject_embedding_plus_roi_scalar_ridge',
                    'mean_frozen_embedding_plus_mean_roi_scalar',
                    list(embedding_columns) + list(roi_columns),
                ),
            ]
        )
    for candidate_id, feature_set, columns in subject_feature_sets:
        features = _feature_matrix(grouped, columns)
        predictions, warning_messages, folds = _kfold_ridge_predictions(
            features, target, subject_split_count
        )
        prediction_rows.append(
            _subject_candidate_prediction_rows(
                grouped,
                candidate_id=candidate_id,
                model_family='ridge_regression',
                feature_set=feature_set,
                predictions=predictions,
                folds=folds,
            )
        )
        metric_rows.append(
            _candidate_row(
                candidate_id=candidate_id,
                target_level='subject',
                target_definition='mean_subject_stage_index_0_100',
                model_family='ridge_regression',
                feature_set=feature_set,
                validation_grouping='subject_kfold',
                n_rows=len(grouped),
                n_subjects=len(grouped),
                feature_count=features.shape[1],
                stage_targets=target,
                predictions=predictions,
                warning_messages=warning_messages,
                intended_use='subject_or_cohort_burden_summary_screen',
            )
        )

    result = pd.concat(prediction_rows, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result, metric_rows


def _subject_candidate_prediction_rows(
    grouped: pd.DataFrame,
    *,
    candidate_id: str,
    model_family: str,
    feature_set: str,
    predictions: np.ndarray,
    folds: np.ndarray,
) -> pd.DataFrame:
    rows = grouped[
        ['subject_id', 'cohort_id', 'observed_subject_stage_index_mean']
    ].copy()
    rows['candidate_id'] = candidate_id
    rows['model_family'] = model_family
    rows['feature_set'] = feature_set
    rows['fold'] = folds.astype(int)
    rows['predicted_subject_burden_0_100'] = predictions
    rows['subject_stage_index_absolute_error'] = np.abs(
        rows['observed_subject_stage_index_mean'].astype(float).to_numpy() - predictions
    )
    rows['prediction_source'] = 'held_out_subject_level_prediction'
    finite_status = _finite_matrix_status(predictions)
    rows['prediction_output_finite'] = bool(finite_status['finite'])
    return rows


def _best_candidate(metrics: pd.DataFrame, target_level: str) -> dict[str, Any] | None:
    subset = metrics[
        (metrics['target_level'] == target_level)
        & (metrics['prediction_output_finite'].astype(bool))
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(['stage_index_mae', 'backend_warning_count'])
    return subset.iloc[0].to_dict()


def _write_precision_candidate_summary(
    metrics: pd.DataFrame, subject_predictions: pd.DataFrame, output_path: Path
) -> Path:
    best_image = _best_candidate(metrics, 'image')
    best_subject = _best_candidate(metrics, 'subject')
    recommendation = 'Use the expanded screen as follow-up evidence only; no candidate replaces the primary burden model until it is integrated with calibrated prediction sets.'
    if best_subject is not None and best_image is not None:
        if float(best_subject['stage_index_mae']) < float(
            best_image['stage_index_mae']
        ):
            recommendation = 'Subject-level aggregation is the strongest follow-up direction for cohort burden summaries; it does not replace per-image prediction sets.'
        else:
            recommendation = 'Image-level precision remains the strongest follow-up direction; retain subject-heldout validation and improve calibrated uncertainty before promotion.'
    payload = {
        'artifact_contract': {
            'canonical_signal_screen': 'candidates/signal_comparator_metrics.csv',
            'subject_level_predictions': 'candidates/subject_level_candidate_predictions.csv',
        },
        'candidate_count': int(len(metrics)),
        'subject_prediction_rows': int(len(subject_predictions)),
        'best_image_level_candidate': best_image,
        'best_subject_level_candidate': best_subject,
        'nonfinite_candidate_count': int(
            (~metrics['prediction_output_finite'].astype(bool)).sum()
        ),
        'backend_warning_candidate_count': int(
            (metrics['backend_warning_count'].astype(int) > 0).sum()
        ),
        'recommendation': recommendation,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return output_path


def _morphology_feature_readiness(work_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether deterministic morphology features are biologically reviewable."""
    blockers: list[str] = []
    slit = pd.to_numeric(
        work_df.get('morph_slit_like_area_fraction', pd.Series(dtype=float)),
        errors='coerce',
    ).fillna(0.0)
    border = pd.to_numeric(
        work_df.get('morph_border_false_slit_area_fraction', pd.Series(dtype=float)),
        errors='coerce',
    ).fillna(0.0)
    boundary_overlap = pd.to_numeric(
        work_df.get('morph_slit_boundary_overlap_fraction', pd.Series(dtype=float)),
        errors='coerce',
    ).fillna(0.0)
    nuclear = pd.to_numeric(
        work_df.get(
            'morph_nuclear_mesangial_confounder_area_fraction', pd.Series(dtype=float)
        ),
        errors='coerce',
    ).fillna(0.0)
    score = pd.to_numeric(work_df.get('score', pd.Series(dtype=float)), errors='coerce')
    zero_score = score.eq(0.0)
    zero_score_slit_positive_fraction = (
        float(slit[zero_score].gt(0).mean()) if zero_score.any() else None
    )
    overall_slit_positive_fraction = float(slit.gt(0).mean()) if len(slit) else 0.0
    mean_boundary_overlap = (
        float(boundary_overlap.mean()) if len(boundary_overlap) else 0.0
    )
    mean_border_false_slit_area_fraction = float(border.mean()) if len(border) else 0.0
    mean_nuclear_confounder_area_fraction = (
        float(nuclear.mean()) if len(nuclear) else 0.0
    )
    if (
        zero_score_slit_positive_fraction is not None
        and zero_score_slit_positive_fraction > 0.50
    ):
        blockers.append('accepted_slit_signal_is_common_in_score_0_images')
    if overall_slit_positive_fraction > 0.95:
        blockers.append('accepted_slit_signal_is_nearly_ubiquitous')
    if mean_boundary_overlap > 0.20:
        blockers.append('slit_signal_has_high_boundary_overlap')
    if mean_nuclear_confounder_area_fraction > 0.10:
        blockers.append('nuclear_mesangial_confounder_burden_is_high')
    return {
        'status': 'failed_visual_feature_readiness' if blockers else 'passed',
        'blockers': blockers,
        'metrics': {
            'overall_slit_positive_fraction': overall_slit_positive_fraction,
            'zero_score_slit_positive_fraction': zero_score_slit_positive_fraction,
            'mean_slit_boundary_overlap_fraction': mean_boundary_overlap,
            'mean_border_false_slit_area_fraction': mean_border_false_slit_area_fraction,
            'mean_nuclear_mesangial_confounder_area_fraction': mean_nuclear_confounder_area_fraction,
        },
        'decision': (
            'do_not_promote_morphology_candidates_or_use_slit_features_for_biology_claims'
            if blockers
            else 'morphology_features_are_reviewable_for_candidate_modeling'
        ),
    }


def _write_morphology_candidate_screen(
    *,
    work_df: pd.DataFrame,
    embedding_columns: Sequence[str],
    morphology_columns: Sequence[str],
    stage_targets: np.ndarray,
    groups: np.ndarray,
    split_count: int,
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    image_feature_sets = [
        (
            'image_morphology_only_ridge',
            'morphology_features',
            list(morphology_columns),
        ),
        (
            'image_embedding_plus_morphology_ridge',
            'frozen_embedding_plus_morphology_features',
            list(embedding_columns) + list(morphology_columns),
        ),
    ]
    for candidate_id, feature_set, columns in image_feature_sets:
        features = _feature_matrix(work_df, columns)
        predictions, warning_messages, folds = _grouped_ridge_predictions(
            features, stage_targets, groups, split_count
        )
        rows.append(
            _candidate_row(
                candidate_id=candidate_id,
                target_level='image',
                target_definition='image_stage_index_0_100',
                model_family='ridge_regression',
                feature_set=feature_set,
                validation_grouping='subject_id_groupkfold',
                n_rows=len(work_df),
                n_subjects=len(np.unique(groups)),
                feature_count=features.shape[1],
                stage_targets=stage_targets,
                predictions=predictions,
                warning_messages=warning_messages,
                intended_use='morphology_precision_screen_not_primary_model',
            )
        )
        image_predictions = work_df[
            [
                column
                for column in [
                    'subject_image_id',
                    'subject_id',
                    'sample_id',
                    'image_id',
                    'cohort_id',
                    'score',
                ]
                if column in work_df.columns
            ]
        ].copy()
        image_predictions['candidate_id'] = candidate_id
        image_predictions['fold'] = folds
        image_predictions['observed_stage_index'] = stage_targets
        image_predictions['predicted_burden_0_100'] = predictions
        image_predictions['stage_index_absolute_error'] = np.abs(
            stage_targets - predictions
        )
        image_predictions['prediction_source'] = 'held_out_subject_morphology_screen'
        prediction_rows.append(image_predictions)

    subject_df = work_df.copy()
    if 'cohort_id' not in subject_df.columns:
        subject_df['cohort_id'] = ''
    subject_df['observed_stage_index'] = stage_targets
    subject_feature_sets = [
        (
            'subject_morphology_only_ridge',
            'mean_morphology_features',
            list(morphology_columns),
        ),
        (
            'subject_embedding_plus_morphology_ridge',
            'mean_frozen_embedding_plus_mean_morphology_features',
            list(embedding_columns) + list(morphology_columns),
        ),
    ]
    aggregation: dict[str, Any] = {'cohort_id': 'first', 'observed_stage_index': 'mean'}
    for column in set(list(embedding_columns) + list(morphology_columns)):
        aggregation[column] = 'mean'
    grouped = subject_df.groupby('subject_id', as_index=False).agg(aggregation)
    grouped = grouped.rename(
        columns={'observed_stage_index': 'observed_subject_stage_index_mean'}
    )
    target = grouped['observed_subject_stage_index_mean'].to_numpy(dtype=np.float64)
    subject_split_count = min(max(2, split_count), len(grouped))
    for candidate_id, feature_set, columns in subject_feature_sets:
        features = _feature_matrix(grouped, columns)
        predictions, warning_messages, folds = _kfold_ridge_predictions(
            features, target, subject_split_count
        )
        prediction_rows.append(
            _subject_candidate_prediction_rows(
                grouped,
                candidate_id=candidate_id,
                model_family='ridge_regression',
                feature_set=feature_set,
                predictions=predictions,
                folds=folds,
            )
        )
        rows.append(
            _candidate_row(
                candidate_id=candidate_id,
                target_level='subject',
                target_definition='mean_subject_stage_index_0_100',
                model_family='ridge_regression',
                feature_set=feature_set,
                validation_grouping='subject_kfold',
                n_rows=len(grouped),
                n_subjects=len(grouped),
                feature_count=features.shape[1],
                stage_targets=target,
                predictions=predictions,
                warning_messages=warning_messages,
                intended_use='morphology_subject_or_cohort_burden_screen',
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_path, index=False)
    predictions_path = output_path.with_name(
        'subject_morphology_candidate_predictions.csv'
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(predictions_path, index=False)
    summary_path = output_path.with_name('morphology_candidate_summary.json')
    feature_readiness = _morphology_feature_readiness(work_df)
    summary = {
        'artifact_contract': {
            'metrics': 'candidates/morphology_candidate_metrics.csv',
            'predictions': 'candidates/subject_morphology_candidate_predictions.csv',
        },
        'candidate_count': int(len(metrics)),
        'feature_columns': list(morphology_columns),
        'best_image_level_candidate': _best_candidate(metrics, 'image'),
        'best_subject_level_candidate': _best_candidate(metrics, 'subject'),
        'feature_readiness': feature_readiness,
        'selection_status': 'blocked_by_visual_feature_readiness'
        if feature_readiness['status'] != 'passed'
        else 'exploratory_until_operator_feature_review_and_calibration_gates_pass',
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return metrics


def _write_cohort_stability(
    validation_metrics: pd.DataFrame, final_metrics: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    value_column = 'subject_weighted_mean_predicted_burden'
    if validation_metrics.empty or final_metrics.empty:
        output_path.write_text('', encoding='utf-8')
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    final_lookup = final_metrics.set_index('cohort_id')
    for row in validation_metrics.itertuples(index=False):
        cohort_id = str(getattr(row, 'cohort_id'))
        if cohort_id not in final_lookup.index:
            continue
        final_row = final_lookup.loc[cohort_id]
        validation_mean = float(getattr(row, value_column))
        final_mean = float(final_row[value_column])
        absolute_difference = abs(final_mean - validation_mean)
        rows.append(
            {
                'cohort_id': cohort_id,
                'validation_prediction_source': 'subject_heldout_grouped_fold_prediction',
                'final_prediction_source': 'final_model_full_cohort_fit',
                'validation_subject_weighted_mean_burden': validation_mean,
                'final_subject_weighted_mean_burden': final_mean,
                'absolute_difference': absolute_difference,
                'stability_gate': 'passed'
                if absolute_difference <= 5.0
                else 'review_required',
                'stability_rule': 'final_vs_subject_heldout_mean_difference_le_5_burden_points',
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def evaluate_burden_index_table(
    embedding_df: pd.DataFrame, output_dir: Path, n_splits: int = 3
) -> dict[str, Path]:
    """Fit grouped cumulative-threshold burden models and write artifacts."""
    logger = get_logger('eq.quantification.burden')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        'Starting burden-index evaluation for %d rows -> %s',
        len(embedding_df),
        output_dir,
    )
    output_paths = _prepare_burden_output_contract(output_dir)
    logger.info(
        'Prepared grouped burden output directories: %s',
        ', '.join(f'{key}={path.name}' for key, path in output_paths.items()),
    )
    primary_model_dir = output_paths['primary_model']
    validation_dir = output_paths['validation']
    calibration_dir = output_paths['calibration']
    summaries_dir = output_paths['summaries']
    evidence_dir = output_paths['evidence']
    candidates_dir = output_paths['candidates']
    diagnostics_dir = output_paths['diagnostics']
    feature_sets_dir = output_paths['feature_sets']
    morphology_review_dir = output_paths['morphology_review']
    if 'score' not in embedding_df.columns:
        raise BurdenModelError('Embedding table must contain score')
    validate_score_values(embedding_df['score'])
    embedding_columns = [
        column for column in embedding_df.columns if column.startswith('embedding_')
    ]
    if not embedding_columns:
        raise BurdenModelError('Embedding table does not contain embedding columns')

    group_column, groups, grouping_audit = derive_biological_grouping(embedding_df)
    work_df = embedding_df.copy().reset_index(drop=True)
    work_df[group_column] = groups.reset_index(drop=True)
    logger.info(
        'Using %s grouped validation with %d subjects and %d samples',
        group_column,
        int(groups.nunique()),
        int(work_df['sample_id'].nunique()) if 'sample_id' in work_df.columns else 0,
    )
    logger.info(
        'Extracting morphology features for %d ROI rows -> %s',
        len(work_df),
        feature_sets_dir / 'morphology_features.csv',
    )
    morphology_df, morphology_artifacts = write_morphology_feature_tables(
        work_df, feature_sets_dir, diagnostics_dir
    )
    logger.info(
        'Morphology features written: rows=%d, features=%d, diagnostics=%s',
        len(morphology_df),
        len(MORPHOLOGY_FEATURE_COLUMNS),
        morphology_artifacts['morphology_feature_diagnostics'],
    )
    for column in ['morphology_feature_status', *MORPHOLOGY_FEATURE_COLUMNS]:
        work_df[column] = morphology_df[column].to_numpy()
    logger.info(
        'Generating morphology feature review -> %s',
        morphology_review_dir / 'feature_review.html',
    )
    morphology_review_artifacts = write_morphology_feature_review(
        morphology_df, morphology_review_dir
    )
    logger.info(
        'Morphology review written: html=%s, adjudication=%s',
        morphology_review_artifacts['morphology_feature_review_html'],
        morphology_review_artifacts['morphology_operator_adjudication_agreement'],
    )
    scores = work_df['score'].astype(float).to_numpy()
    stage_targets = score_to_stage_index(scores)
    target_matrix = threshold_targets(scores)
    x = work_df[embedding_columns].to_numpy(dtype=np.float64)
    unique_groups = np.unique(groups.astype(str).to_numpy())
    split_count = min(max(2, n_splits), len(unique_groups))
    if split_count < 2:
        raise BurdenModelError(
            'Need at least two biological groups for grouped burden evaluation'
        )
    logger.info(
        'Fitting grouped cumulative-threshold burden model: folds=%d, thresholds=%s',
        split_count,
        ', '.join(f'>{threshold:g}' for threshold in CUMULATIVE_THRESHOLDS),
    )

    grouping_audit['n_splits'] = int(split_count)
    grouping_audit_path = validation_dir / 'grouping_audit.json'
    grouping_audit_path.write_text(
        json.dumps(grouping_audit, indent=2), encoding='utf-8'
    )
    threshold_support = _write_threshold_support(
        work_df, groups, validation_dir / 'threshold_support.csv'
    )

    raw_threshold_probabilities = np.zeros(
        (len(work_df), len(CUMULATIVE_THRESHOLDS)), dtype=np.float64
    )
    fold_assignments = np.zeros(len(work_df), dtype=np.int64)
    fold_warning_messages: list[dict[str, Any]] = []
    fold_train_indices: dict[int, np.ndarray] = {}
    fold_conformity_quantiles: dict[int, float] = {}
    fold_residual_quantiles: dict[int, float] = {}

    splitter = GroupKFold(n_splits=split_count)
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(x, scores, groups=groups), start=1
    ):
        logger.info(
            'Burden fold %d/%d: train_rows=%d, test_rows=%d, train_subjects=%d, test_subjects=%d',
            fold_index,
            split_count,
            len(train_idx),
            len(test_idx),
            len(np.unique(groups.iloc[train_idx].astype(str))),
            len(np.unique(groups.iloc[test_idx].astype(str))),
        )
        fold_train_indices[fold_index] = train_idx
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        fold_assignments[test_idx] = fold_index
        fold_messages: list[str] = []
        for threshold_index, threshold in enumerate(CUMULATIVE_THRESHOLDS):
            model, messages, family = _fit_threshold_model(
                x_train, target_matrix[train_idx, threshold_index]
            )
            fold_messages.extend(messages)
            fold_probabilities, predict_messages = _predict_threshold_probability(
                model, x_test
            )
            fold_messages.extend(predict_messages)
            raw_threshold_probabilities[test_idx, threshold_index] = fold_probabilities
        fold_threshold_probabilities = monotonic_threshold_probabilities(
            raw_threshold_probabilities[test_idx]
        )
        fold_score_probabilities = score_probabilities_from_thresholds(
            fold_threshold_probabilities
        )
        fold_burden_values = burden_from_threshold_probabilities(
            fold_threshold_probabilities
        )
        fold_conformity = _aps_conformity_scores(
            fold_score_probabilities, scores[test_idx]
        )
        fold_residuals = np.abs(stage_targets[test_idx] - fold_burden_values)
        fold_conformity_quantiles[fold_index] = _conformal_quantile(
            fold_conformity, PREDICTION_SET_COVERAGE
        )
        fold_residual_quantiles[fold_index] = _conformal_quantile(
            fold_residuals, PREDICTION_SET_COVERAGE
        )
        fold_warning_messages.append(
            {
                'fold': int(fold_index),
                'messages': list(dict.fromkeys(fold_messages)),
                'train_group_count': int(
                    len(np.unique(groups.iloc[train_idx].astype(str)))
                ),
                'test_group_count': int(
                    len(np.unique(groups.iloc[test_idx].astype(str)))
                ),
            }
        )

    raw_threshold_probability_status = _finite_matrix_status(
        raw_threshold_probabilities
    )
    threshold_probabilities = monotonic_threshold_probabilities(
        raw_threshold_probabilities
    )
    threshold_probability_status = _finite_matrix_status(threshold_probabilities)
    burden_values = burden_from_threshold_probabilities(threshold_probabilities)
    score_probabilities = score_probabilities_from_thresholds(threshold_probabilities)
    predicted_score_indices = np.argmax(score_probabilities, axis=1)
    predicted_scores = ALLOWED_SCORE_VALUES[predicted_score_indices]
    row_conformity_quantiles = np.array(
        [fold_conformity_quantiles[int(fold_index)] for fold_index in fold_assignments],
        dtype=np.float64,
    )
    row_residual_quantiles = np.array(
        [fold_residual_quantiles[int(fold_index)] for fold_index in fold_assignments],
        dtype=np.float64,
    )
    prediction_sets = prediction_sets_from_score_probabilities(
        score_probabilities, conformity_quantiles=row_conformity_quantiles
    )

    predictions = work_df.drop(columns=embedding_columns, errors='ignore').copy()
    predictions['fold'] = fold_assignments
    predictions['observed_stage_index'] = stage_targets
    for index, column in enumerate(THRESHOLD_PROBABILITY_COLUMNS):
        predictions[column] = threshold_probabilities[:, index]
    for index, score in enumerate(ALLOWED_SCORE_VALUES):
        predictions[f'prob_stage_score_{score:g}'.replace('.', 'p')] = (
            score_probabilities[:, index]
        )
    predictions[BURDEN_COLUMN] = burden_values
    predictions['predicted_stage_score'] = predicted_scores
    predictions['stage_index_absolute_error'] = np.abs(stage_targets - burden_values)
    predictions['grade_scale_absolute_error'] = np.abs(scores - predicted_scores)
    predictions['prediction_set_scores'] = prediction_sets
    predictions['burden_interval_low_0_100'] = np.clip(
        burden_values - row_residual_quantiles, 0.0, 100.0
    )
    predictions['burden_interval_high_0_100'] = np.clip(
        burden_values + row_residual_quantiles, 0.0, 100.0
    )
    predictions['burden_interval_coverage'] = PREDICTION_SET_COVERAGE
    predictions['burden_interval_method'] = (
        'grouped_fold_row_absolute_residual_conformal_quantile'
    )
    predictions['prediction_set_method'] = 'grouped_fold_row_aps_conformal_quantile'
    predictions['prediction_source'] = 'held_out_grouped_fold_prediction'

    predictions_path = primary_model_dir / 'burden_predictions.csv'
    predictions.to_csv(predictions_path, index=False)
    logger.info('Held-out burden predictions written -> %s', predictions_path)

    threshold_rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(CUMULATIVE_THRESHOLDS):
        target = target_matrix[:, index]
        prob = threshold_probabilities[:, index]
        row = {
            'threshold': float(threshold),
            'probability_column': THRESHOLD_PROBABILITY_COLUMNS[index],
            'brier_score': float(np.mean((prob - target) ** 2)),
            'positive_rows': int(target.sum()),
            'negative_rows': int(len(target) - target.sum()),
        }
        if len(np.unique(target)) == 2:
            row['roc_auc'] = float(roc_auc_score(target, prob))
        else:
            row['roc_auc'] = ''
        threshold_rows.append(row)
    threshold_metrics_path = validation_dir / 'threshold_metrics.csv'
    pd.DataFrame(threshold_rows).to_csv(threshold_metrics_path, index=False)

    calibration_bins_path = calibration_dir / 'calibration_bins.csv'
    calibration_df = predictions.copy()
    calibration_df['burden_bin'] = pd.cut(
        calibration_df[BURDEN_COLUMN], bins=np.linspace(0, 100, 6), include_lowest=True
    )
    calibration_summary = (
        calibration_df.groupby('burden_bin', observed=False)
        .agg(
            n=('score', 'size'),
            mean_predicted_burden=(BURDEN_COLUMN, 'mean'),
            mean_observed_stage_index=('observed_stage_index', 'mean'),
        )
        .reset_index()
    )
    calibration_summary['burden_bin'] = calibration_summary['burden_bin'].astype(str)
    calibration_summary.to_csv(calibration_bins_path, index=False)

    uncertainty_path = _write_uncertainty_calibration(
        predictions, calibration_dir / 'uncertainty_calibration.json', group_column
    )
    cohort_metrics_path = summaries_dir / 'cohort_metrics.csv'
    _write_cohort_metrics(predictions, group_column, cohort_metrics_path)
    group_summary_path = summaries_dir / 'group_summary_intervals.csv'
    _write_group_summary_intervals(predictions, group_column, group_summary_path)
    validation_design_path = validation_dir / 'validation_design.json'
    validation_design_path.write_text(
        json.dumps(
            {
                'primary_validation': 'subject_heldout_grouped_cross_validation',
                'grouping_column': group_column,
                'subject_grouping_required': group_column == 'subject_id',
                'n_subjects': int(predictions['subject_id'].nunique())
                if 'subject_id' in predictions.columns
                else int(len(unique_groups)),
                'n_samples': int(predictions['sample_id'].nunique())
                if 'sample_id' in predictions.columns
                else 0,
                'n_images': int(len(predictions)),
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    prediction_explanations_path = evidence_dir / 'prediction_explanations.csv'
    explanation_columns = [
        column
        for column in [
            'subject_image_id',
            'glomerulus_id',
            'cohort_id',
            'lane_assignment',
            group_column,
            'fold',
            'score',
            'observed_stage_index',
            BURDEN_COLUMN,
            'prediction_set_scores',
            'burden_interval_low_0_100',
            'burden_interval_high_0_100',
            'prediction_set_method',
            'prediction_source',
            *THRESHOLD_PROBABILITY_COLUMNS,
            'raw_image_path',
            'raw_mask_path',
            'roi_image_path',
        ]
        if column in predictions.columns
    ]
    predictions[explanation_columns].to_csv(prediction_explanations_path, index=False)

    nearest_rows: list[dict[str, Any]] = []
    for fold_index in sorted(fold_train_indices):
        test_idx = np.where(fold_assignments == fold_index)[0]
        train_idx = fold_train_indices[fold_index]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        scaler = StandardScaler().fit(x[train_idx])
        train_x = scaler.transform(x[train_idx])
        test_x = scaler.transform(x[test_idx])
        neighbors = NearestNeighbors(n_neighbors=min(10, len(train_idx))).fit(train_x)
        distances, indices = neighbors.kneighbors(test_x)
        for local_row, source_index in enumerate(test_idx):
            added = 0
            source_group = str(groups.iloc[source_index])
            for distance, neighbor_local_index in zip(
                distances[local_row], indices[local_row]
            ):
                neighbor_index = int(train_idx[int(neighbor_local_index)])
                if str(groups.iloc[neighbor_index]) == source_group:
                    continue
                nearest_rows.append(
                    {
                        'subject_image_id': predictions.loc[source_index].get(
                            'subject_image_id', source_index
                        ),
                        'fold': int(fold_index),
                        'neighbor_rank': int(added + 1),
                        'neighbor_subject_image_id': predictions.loc[
                            neighbor_index
                        ].get('subject_image_id', neighbor_index),
                        'neighbor_score': float(scores[neighbor_index]),
                        'neighbor_distance': float(distance),
                        'neighbor_cohort_id': predictions.loc[neighbor_index].get(
                            'cohort_id', ''
                        ),
                        'neighbor_lane_assignment': predictions.loc[neighbor_index].get(
                            'lane_assignment', ''
                        ),
                        'neighbor_raw_image_path': predictions.loc[neighbor_index].get(
                            'raw_image_path', ''
                        ),
                        'neighbor_raw_mask_path': predictions.loc[neighbor_index].get(
                            'raw_mask_path', ''
                        ),
                        'neighbor_roi_image_path': predictions.loc[neighbor_index].get(
                            'roi_image_path', ''
                        ),
                    }
                )
                added += 1
                if added >= 3:
                    break
    nearest_examples_path = evidence_dir / 'nearest_examples.csv'
    pd.DataFrame(nearest_rows).to_csv(nearest_examples_path, index=False)

    direct_regression = _fit_direct_regression_comparator(
        x, stage_targets, groups.astype(str).to_numpy(), split_count
    )
    signal_comparator_path = candidates_dir / 'signal_comparator_metrics.csv'
    subject_level_candidate_path = (
        candidates_dir / 'subject_level_candidate_predictions.csv'
    )
    precision_candidate_summary_path = (
        candidates_dir / 'precision_candidate_summary.json'
    )
    signal_comparator = _write_signal_comparator_screen(
        work_df,
        embedding_columns,
        stage_targets,
        groups.astype(str).to_numpy(),
        split_count,
        signal_comparator_path,
    )
    logger.info('Precision candidate screen written -> %s', signal_comparator_path)
    morphology_candidate_path = candidates_dir / 'morphology_candidate_metrics.csv'
    subject_morphology_candidate_path = (
        candidates_dir / 'subject_morphology_candidate_predictions.csv'
    )
    morphology_candidate_summary_path = (
        candidates_dir / 'morphology_candidate_summary.json'
    )
    morphology_candidate_metrics = _write_morphology_candidate_screen(
        work_df=work_df,
        embedding_columns=embedding_columns,
        morphology_columns=MORPHOLOGY_FEATURE_COLUMNS,
        stage_targets=stage_targets,
        groups=groups.astype(str).to_numpy(),
        split_count=split_count,
        output_path=morphology_candidate_path,
    )
    logger.info('Morphology candidate screen written -> %s', morphology_candidate_path)
    support_blockers = sorted(
        set(
            threshold_support.loc[
                threshold_support['support_status'] != 'estimable', 'stratum'
            ]
        )
    )
    overall_support_blockers = [
        blocker for blocker in support_blockers if blocker == 'overall'
    ]
    stratum_support_blockers = [
        blocker for blocker in support_blockers if blocker != 'overall'
    ]
    cohort_composition_notes = [
        f'{blocker} has incomplete within-cohort threshold support; this is a cohort-composition note, not a full-cohort modeling blocker.'
        for blocker in stratum_support_blockers
    ]
    numerical_warnings = list(
        dict.fromkeys(
            message
            for fold in fold_warning_messages
            for message in fold.get('messages', [])
        )
    )
    numerical_stability_status = (
        'nonfinite_output'
        if not threshold_probability_status['finite']
        else 'backend_warnings_outputs_finite'
        if numerical_warnings
        else 'ok'
    )
    support_gate_status = 'blocked' if overall_support_blockers else 'passed'
    overall = {
        'stage_index_mae': float(
            mean_absolute_error(
                predictions['observed_stage_index'], predictions[BURDEN_COLUMN]
            )
        ),
        'grade_scale_mae': float(
            mean_absolute_error(
                predictions['score'], predictions['predicted_stage_score']
            )
        ),
        'prediction_set_coverage': _empirical_coverage(
            predictions, np.ones(len(predictions), dtype=bool)
        )['coverage'],
        'average_prediction_set_size': _empirical_coverage(
            predictions, np.ones(len(predictions), dtype=bool)
        )['average_set_size'],
    }
    interval_coverage = float(
        np.mean(
            (
                predictions['observed_stage_index']
                >= predictions['burden_interval_low_0_100']
            )
            & (
                predictions['observed_stage_index']
                <= predictions['burden_interval_high_0_100']
            )
        )
    )
    overall['burden_interval_empirical_coverage'] = interval_coverage
    metrics = {
        'canonical_module': 'eq.quantification.burden',
        'model_family': 'grouped_cumulative_threshold_logistic',
        'n_examples': int(len(predictions)),
        'n_subject_groups': int(len(unique_groups)),
        'n_splits': int(split_count),
        'allowed_score_values': ALLOWED_SCORE_VALUES.tolist(),
        'thresholds': CUMULATIVE_THRESHOLDS.tolist(),
        'threshold_probability_columns': THRESHOLD_PROBABILITY_COLUMNS,
        'burden_column': BURDEN_COLUMN,
        'monotonic_correction_method': 'cumulative_min_projection',
        'overall': overall,
        'score_counts': {
            f'{score:g}': int(np.sum(np.isclose(scores, score)))
            for score in ALLOWED_SCORE_VALUES
        },
        'threshold_positive_counts': {
            f'score_gt_{_sanitize_threshold_label(threshold)}': int(
                np.sum(scores > threshold)
            )
            for threshold in CUMULATIVE_THRESHOLDS
        },
        'fold_warning_messages': fold_warning_messages,
        'raw_threshold_probability_status': raw_threshold_probability_status,
        'threshold_probability_status': threshold_probability_status,
        'fold_group_conformity_quantiles': {
            str(key): float(value)
            for key, value in sorted(fold_conformity_quantiles.items())
        },
        'fold_group_residual_quantiles': {
            str(key): float(value)
            for key, value in sorted(fold_residual_quantiles.items())
        },
        'numerical_stability_status': numerical_stability_status,
        'backend_warning_messages': numerical_warnings,
        'support_gate_status': support_gate_status,
        'overall_support_gate_status': 'blocked'
        if overall_support_blockers
        else 'passed',
        'stratified_support_gate_status': 'composition_note'
        if stratum_support_blockers
        else 'passed',
        'support_gate_blockers': overall_support_blockers,
        'overall_support_gate_blockers': overall_support_blockers,
        'stratum_support_gate_blockers': [],
        'cohort_composition_notes': cohort_composition_notes,
        'cohort_composition_strata': stratum_support_blockers,
        'direct_regression_comparator': direct_regression,
        'signal_comparator_screen': signal_comparator.to_dict(orient='records'),
        'morphology_candidate_screen': morphology_candidate_metrics.to_dict(
            orient='records'
        ),
        'morphology_feature_artifacts': {
            key: str(value.relative_to(output_dir))
            for key, value in morphology_artifacts.items()
        },
        'morphology_feature_review_artifacts': {
            key: str(value.relative_to(output_dir))
            for key, value in morphology_review_artifacts.items()
            if value.is_file()
        },
    }
    if precision_candidate_summary_path.exists():
        precision_summary = json.loads(
            precision_candidate_summary_path.read_text(encoding='utf-8')
        )
        precision_summary['current_primary_burden_model'] = {
            'model_family': metrics['model_family'],
            'target_level': 'image',
            'target_definition': 'image_stage_index_0_100_from_threshold_probabilities',
            'validation_grouping': f'{group_column}_groupkfold',
            'stage_index_mae': overall['stage_index_mae'],
            'grade_scale_mae': overall['grade_scale_mae'],
            'prediction_set_coverage': overall['prediction_set_coverage'],
            'average_prediction_set_size': overall['average_prediction_set_size'],
            'numerical_stability_status': numerical_stability_status,
            'support_gate_status': support_gate_status,
        }
        precision_candidate_summary_path.write_text(
            json.dumps(precision_summary, indent=2), encoding='utf-8'
        )
    metrics_path = primary_model_dir / 'burden_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    logger.info(
        'Burden metrics written -> %s (support=%s, numerical=%s, coverage=%.3f)',
        metrics_path,
        support_gate_status,
        numerical_stability_status,
        float(overall['prediction_set_coverage']),
    )

    final_scaler = StandardScaler().fit(x)
    final_models: dict[str, Any] = {}
    final_x = final_scaler.transform(x)
    final_raw_threshold_probabilities = np.zeros(
        (len(work_df), len(CUMULATIVE_THRESHOLDS)), dtype=np.float64
    )
    for index, threshold in enumerate(CUMULATIVE_THRESHOLDS):
        model, _messages, _family = _fit_threshold_model(
            final_x, target_matrix[:, index]
        )
        final_models[THRESHOLD_PROBABILITY_COLUMNS[index]] = model
        final_raw_threshold_probabilities[:, index], _predict_messages = (
            _predict_threshold_probability(model, final_x)
        )
    final_threshold_probabilities = monotonic_threshold_probabilities(
        final_raw_threshold_probabilities
    )
    final_score_probabilities = score_probabilities_from_thresholds(
        final_threshold_probabilities
    )
    final_burden_values = burden_from_threshold_probabilities(
        final_threshold_probabilities
    )
    final_predicted_scores = ALLOWED_SCORE_VALUES[
        np.argmax(final_score_probabilities, axis=1)
    ]
    validation_conformity_quantile = _conformal_quantile(
        _aps_conformity_scores(score_probabilities, scores), PREDICTION_SET_COVERAGE
    )
    validation_residual_quantile = _conformal_quantile(
        np.abs(stage_targets - burden_values), PREDICTION_SET_COVERAGE
    )
    final_predictions = work_df.drop(columns=embedding_columns, errors='ignore').copy()
    final_predictions['observed_stage_index'] = stage_targets
    for index, column in enumerate(THRESHOLD_PROBABILITY_COLUMNS):
        final_predictions[column] = final_threshold_probabilities[:, index]
    for index, score in enumerate(ALLOWED_SCORE_VALUES):
        final_predictions[f'prob_stage_score_{score:g}'.replace('.', 'p')] = (
            final_score_probabilities[:, index]
        )
    final_predictions[BURDEN_COLUMN] = final_burden_values
    final_predictions['predicted_stage_score'] = final_predicted_scores
    final_predictions['apparent_stage_index_absolute_error'] = np.abs(
        stage_targets - final_burden_values
    )
    final_predictions['apparent_grade_scale_absolute_error'] = np.abs(
        scores - final_predicted_scores
    )
    final_predictions['prediction_set_scores'] = (
        prediction_sets_from_score_probabilities(
            final_score_probabilities,
            conformity_quantiles=np.full(
                len(final_predictions), validation_conformity_quantile
            ),
        )
    )
    final_predictions['burden_interval_low_0_100'] = np.clip(
        final_burden_values - validation_residual_quantile, 0.0, 100.0
    )
    final_predictions['burden_interval_high_0_100'] = np.clip(
        final_burden_values + validation_residual_quantile, 0.0, 100.0
    )
    final_predictions['burden_interval_coverage'] = PREDICTION_SET_COVERAGE
    final_predictions['burden_interval_method'] = (
        'validation_residual_quantile_applied_to_final_full_cohort_fit'
    )
    final_predictions['prediction_set_method'] = (
        'validation_aps_quantile_applied_to_final_full_cohort_fit'
    )
    final_predictions['prediction_source'] = 'final_model_full_cohort_fit'
    final_predictions_path = primary_model_dir / 'final_model_predictions.csv'
    final_predictions.to_csv(final_predictions_path, index=False)
    logger.info(
        'Final full-cohort burden predictions written -> %s', final_predictions_path
    )
    final_cohort_metrics_path = summaries_dir / 'final_model_cohort_metrics.csv'
    _write_cohort_metrics(final_predictions, group_column, final_cohort_metrics_path)
    final_group_summary_path = summaries_dir / 'final_model_group_summary_intervals.csv'
    _write_group_summary_intervals(
        final_predictions, group_column, final_group_summary_path
    )
    cohort_stability_path = validation_dir / 'cohort_stability.csv'
    _write_cohort_stability(
        _read_csv_if_nonempty(cohort_metrics_path),
        _read_csv_if_nonempty(final_cohort_metrics_path),
        cohort_stability_path,
    )
    model_path = primary_model_dir / 'burden_model.joblib'
    with model_path.open('wb') as handle:
        pickle.dump(
            {
                'allowed_score_values': ALLOWED_SCORE_VALUES.tolist(),
                'thresholds': CUMULATIVE_THRESHOLDS.tolist(),
                'embedding_columns': embedding_columns,
                'scaler': final_scaler,
                'threshold_models': final_models,
                'model_metadata': metrics,
                'prediction_contract': {
                    'validation_predictions': 'primary_burden_index/model/burden_predictions.csv',
                    'final_full_cohort_predictions': 'primary_burden_index/model/final_model_predictions.csv',
                    'final_prediction_source': 'final_model_full_cohort_fit',
                    'validation_prediction_source': 'held_out_grouped_fold_prediction',
                },
            },
            handle,
        )
    logger.info('Serialized burden model written -> %s', model_path)
    primary_burden_index = _write_primary_burden_index(output_paths['index'])
    logger.info(
        'Primary burden-index first-read index written -> %s', primary_burden_index
    )

    return {
        'primary_burden_index': output_paths['root'],
        'primary_burden_index_index': primary_burden_index,
        'burden_predictions': predictions_path,
        'burden_metrics': metrics_path,
        'threshold_metrics': threshold_metrics_path,
        'threshold_support': validation_dir / 'threshold_support.csv',
        'calibration_bins': calibration_bins_path,
        'uncertainty_calibration': uncertainty_path,
        'grouping_audit': grouping_audit_path,
        'prediction_explanations': prediction_explanations_path,
        'nearest_examples': nearest_examples_path,
        'cohort_metrics': cohort_metrics_path,
        'group_summary_intervals': group_summary_path,
        'validation_design': validation_design_path,
        'signal_comparator_metrics': signal_comparator_path,
        'subject_level_candidate_predictions': subject_level_candidate_path,
        'precision_candidate_summary': precision_candidate_summary_path,
        'morphology_features': morphology_artifacts['morphology_features'],
        'morphology_feature_metadata': morphology_artifacts[
            'morphology_feature_metadata'
        ],
        'subject_morphology_features': morphology_artifacts[
            'subject_morphology_features'
        ],
        'morphology_feature_diagnostics': morphology_artifacts[
            'morphology_feature_diagnostics'
        ],
        'morphology_feature_review_html': morphology_review_artifacts[
            'morphology_feature_review_html'
        ],
        'morphology_feature_review_cases': morphology_review_artifacts[
            'morphology_feature_review_cases'
        ],
        'morphology_operator_adjudication_template': morphology_review_artifacts[
            'morphology_operator_adjudication_template'
        ],
        'morphology_operator_adjudication_agreement': morphology_review_artifacts[
            'morphology_operator_adjudication_agreement'
        ],
        'morphology_candidate_metrics': morphology_candidate_path,
        'subject_morphology_candidate_predictions': subject_morphology_candidate_path,
        'morphology_candidate_summary': morphology_candidate_summary_path,
        'final_model_predictions': final_predictions_path,
        'final_model_cohort_metrics': final_cohort_metrics_path,
        'final_model_group_summary_intervals': final_group_summary_path,
        'cohort_stability': cohort_stability_path,
        'burden_model': model_path,
    }
