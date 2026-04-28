"""Severe-aware ordinal endotheliosis estimator."""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from eq.quantification.burden import (
    ALLOWED_SCORE_VALUES,
    THRESHOLD_PROBABILITY_COLUMNS,
    burden_from_threshold_probabilities,
    monotonic_threshold_probabilities,
    prediction_sets_from_score_probabilities,
    score_probabilities_from_thresholds,
    score_to_stage_index,
    validate_score_values,
)
from eq.quantification.morphology_features import MORPHOLOGY_FEATURE_COLUMNS

SEVERE_AWARE_ROOT_NAME = 'severe_aware_ordinal_estimator'
SEVERE_THRESHOLDS = (1.5, 2.0, 3.0)
PRIMARY_SEVERE_THRESHOLD = 2.0
SUPPORTED_SCORE_VALUES = tuple(float(value) for value in ALLOWED_SCORE_VALUES)
STAGE_INDEX_BY_SCORE = {
    float(score): float(index * 20.0)
    for index, score in enumerate(ALLOWED_SCORE_VALUES)
}
SUMMARY_FIGURES = [
    'severe_threshold_metrics.png',
    'predicted_vs_observed_severity.png',
    'severe_false_negative_summary.png',
    'calibration_by_score.png',
    'source_severe_performance.png',
    'ordinal_prediction_set_width.png',
    'reliability_label_counts.png',
]
REQUIRED_RELATIVE_ARTIFACTS = [
    'INDEX.md',
    'summary/estimator_verdict.json',
    'summary/estimator_verdict.md',
    'summary/metrics_by_split.csv',
    'summary/metrics_by_split.json',
    'summary/severe_threshold_metrics.csv',
    'summary/artifact_manifest.json',
    *[f'summary/figures/{name}' for name in SUMMARY_FIGURES],
    'predictions/image_predictions.csv',
    'predictions/subject_predictions.csv',
    'diagnostics/severe_separability_audit.json',
    'diagnostics/threshold_support.json',
    'diagnostics/source_severe_sensitivity.json',
    'diagnostics/reliability_labels.json',
    'evidence/severe_false_negative_review.html',
    'internal/candidate_metrics.csv',
    'internal/candidate_summary.json',
]
ALLOWED_CANDIDATE_IDS = (
    'severe_roi_qc_threshold',
    'severe_morphology_threshold',
    'severe_roi_qc_morphology_threshold',
    'ordinal_roi_qc_thresholds',
    'ordinal_roi_qc_morphology_thresholds',
    'two_stage_severe_gate_roi_qc',
    'two_stage_severe_gate_roi_qc_morphology',
    'subject_severe_aware_ordinal',
)
IDENTITY_COLUMNS = [
    'subject_id',
    'sample_id',
    'image_id',
    'subject_image_id',
    'cohort_id',
    'score',
    'roi_image_path',
    'roi_mask_path',
    'raw_image_path',
    'raw_mask_path',
]
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
LOGGER = logging.getLogger('eq.quantification.severe_aware_ordinal_estimator')


class SevereAwareEstimatorError(RuntimeError):
    """Raised when severe-aware estimator inputs violate the contract."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    target_level: str
    feature_family: str
    model_family: str
    feature_columns: tuple[str, ...]
    threshold: float | None = PRIMARY_SEVERE_THRESHOLD


def severe_aware_output_paths(burden_output_dir: Path) -> dict[str, Path]:
    """Return canonical severe-aware estimator output paths."""
    root = Path(burden_output_dir) / SEVERE_AWARE_ROOT_NAME
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
        'severe_threshold_metrics': root / 'summary' / 'severe_threshold_metrics.csv',
        'artifact_manifest': root / 'summary' / 'artifact_manifest.json',
        'image_predictions': root / 'predictions' / 'image_predictions.csv',
        'subject_predictions': root / 'predictions' / 'subject_predictions.csv',
        'severe_separability_audit': root
        / 'diagnostics'
        / 'severe_separability_audit.json',
        'threshold_support': root / 'diagnostics' / 'threshold_support.json',
        'source_severe_sensitivity': root
        / 'diagnostics'
        / 'source_severe_sensitivity.json',
        'reliability_labels': root / 'diagnostics' / 'reliability_labels.json',
        'evidence_review': root / 'evidence' / 'severe_false_negative_review.html',
        'candidate_metrics': root / 'internal' / 'candidate_metrics.csv',
        'candidate_summary': root / 'internal' / 'candidate_summary.json',
    }


def _save_json(data: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return output_path


def _safe_numeric_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    safe: list[str] = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors='coerce')
        if values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all():
            if float(values.std(ddof=0)) > 0:
                safe.append(column)
    return safe


def _load_source_aware_handoff(burden_output_dir: Path) -> dict[str, Any]:
    root = Path(burden_output_dir) / 'source_aware_estimator'
    verdict_path = root / 'summary' / 'estimator_verdict.json'
    metrics_path = root / 'summary' / 'metrics_by_split.csv'
    image_predictions_path = root / 'predictions' / 'image_predictions.csv'
    payload: dict[str, Any] = {
        'available': verdict_path.exists(),
        'root': str(root),
        'verdict_path': str(verdict_path),
        'metrics_path': str(metrics_path),
        'image_predictions_path': str(image_predictions_path),
        'selected_image_candidate': '',
        'selected_subject_candidate': '',
        'hard_blockers': [],
        'scope_limiters': [],
        'reportable_scopes': {},
        'testing_status': '',
        'claim_boundary': '',
        'high_score_behavior': {},
    }
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text(encoding='utf-8'))
        payload.update(
            {
                'selected_image_candidate': verdict.get('selected_image_candidate', ''),
                'selected_subject_candidate': verdict.get(
                    'selected_subject_candidate', ''
                ),
                'hard_blockers': verdict.get('hard_blockers', []),
                'scope_limiters': verdict.get('scope_limiters', []),
                'reportable_scopes': verdict.get('reportable_scopes', {}),
                'testing_status': verdict.get('testing_status', ''),
                'claim_boundary': verdict.get('claim_boundary', ''),
            }
        )
    if image_predictions_path.exists():
        predictions = pd.read_csv(image_predictions_path)
        if {'score', 'observed_stage_index', 'predicted_stage_index'}.issubset(
            predictions.columns
        ):
            predictions['score'] = pd.to_numeric(predictions['score'], errors='coerce')
            predictions['stage_error'] = (
                pd.to_numeric(predictions['observed_stage_index'], errors='coerce')
                - pd.to_numeric(predictions['predicted_stage_index'], errors='coerce')
            ).abs()
            high: dict[str, Any] = {}
            for score in (2.0, 3.0):
                subset = predictions[np.isclose(predictions['score'], score)]
                if subset.empty:
                    continue
                high[f'score_{score:g}'] = {
                    'row_count': int(len(subset)),
                    'subject_count': int(subset['subject_id'].astype(str).nunique())
                    if 'subject_id' in subset.columns
                    else 0,
                    'stage_index_mae': float(subset['stage_error'].mean()),
                    'mean_predicted_stage_index': float(
                        pd.to_numeric(
                            subset['predicted_stage_index'], errors='coerce'
                        ).mean()
                    ),
                }
            payload['high_score_behavior'] = high
    return payload


def _merge_optional_feature_table(
    base: pd.DataFrame, path: Path, feature_prefixes: tuple[str, ...]
) -> pd.DataFrame:
    if not path.exists():
        return base
    feature_df = pd.read_csv(path)
    feature_columns = [
        column
        for column in feature_df.columns
        if column.startswith(feature_prefixes) or column in MORPHOLOGY_FEATURE_COLUMNS
    ]
    identity_keys = [
        key
        for key in ['subject_image_id', 'subject_id', 'sample_id', 'image_id']
        if key in base.columns and key in feature_df.columns
    ]
    if not identity_keys or not feature_columns:
        return base
    keep_columns = list(dict.fromkeys([*identity_keys, *feature_columns]))
    return base.merge(
        feature_df[keep_columns], on=identity_keys, how='left', validate='one_to_one'
    )


def _assemble_feature_table(
    embedding_df: pd.DataFrame, burden_output_dir: Path
) -> pd.DataFrame:
    feature_df = embedding_df.copy().reset_index(drop=True)
    feature_df = _merge_optional_feature_table(
        feature_df,
        Path(burden_output_dir) / 'feature_sets' / 'morphology_features.csv',
        ('morph_',),
    )
    feature_df = _merge_optional_feature_table(
        feature_df,
        Path(burden_output_dir)
        / 'learned_roi'
        / 'feature_sets'
        / 'learned_roi_features.csv',
        ('learned_',),
    )
    feature_df['score'] = pd.to_numeric(feature_df['score'], errors='raise').astype(
        float
    )
    feature_df['observed_stage_index'] = score_to_stage_index(feature_df['score'])
    feature_df['cohort_id'] = feature_df['cohort_id'].fillna('').astype(str)
    return feature_df


def _validate_inputs(df: pd.DataFrame) -> None:
    required = {'subject_id', 'cohort_id', 'score', 'roi_image_path', 'roi_mask_path'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SevereAwareEstimatorError(
            f'Severe-aware estimator input is missing required columns: {missing}'
        )
    validate_score_values(df['score'])
    for column in ['subject_id', 'cohort_id']:
        if df[column].astype(str).str.strip().eq('').any():
            raise SevereAwareEstimatorError(f'Blank {column} values are not supported.')
    for column in ['roi_image_path', 'roi_mask_path']:
        if df[column].astype(str).str.strip().eq('').any():
            raise SevereAwareEstimatorError(f'Blank {column} values are not supported.')


def compute_threshold_support(df: pd.DataFrame) -> dict[str, Any]:
    """Compute row, subject, and source support for severe thresholds."""
    rows: dict[str, Any] = {}
    score = pd.to_numeric(df['score'], errors='raise').astype(float)
    for threshold in SEVERE_THRESHOLDS:
        key = f'score_gte_{threshold:g}'.replace('.', 'p')
        positive = score >= threshold
        by_source: dict[str, Any] = {}
        for cohort, subset in df.assign(_positive=positive).groupby('cohort_id'):
            by_source[str(cohort)] = {
                'positive_rows': int(subset['_positive'].sum()),
                'negative_rows': int((~subset['_positive']).sum()),
                'positive_subjects': int(
                    subset.loc[subset['_positive'], 'subject_id'].astype(str).nunique()
                ),
                'negative_subjects': int(
                    subset.loc[~subset['_positive'], 'subject_id'].astype(str).nunique()
                ),
            }
        positive_sources = [
            source for source, item in by_source.items() if item['positive_rows'] > 0
        ]
        positive_subjects = int(df.loc[positive, 'subject_id'].astype(str).nunique())
        negative_subjects = int(df.loc[~positive, 'subject_id'].astype(str).nunique())
        if positive_subjects < 2 or negative_subjects < 2:
            status = 'non_estimable'
            reason = 'insufficient_independent_subject_support'
        elif threshold == 3.0 and len(positive_sources) < 2:
            status = 'exploratory'
            reason = 'positive_support_single_source_tail_stratum'
        elif threshold >= 2.0 and len(positive_sources) < 2:
            status = 'estimable_source_sensitive'
            reason = 'positive_support_single_source_current_data_only'
        else:
            status = 'estimable'
            reason = 'sufficient_current_data_subject_support'
        rows[key] = {
            'threshold': float(threshold),
            'positive_rows': int(positive.sum()),
            'negative_rows': int((~positive).sum()),
            'positive_subjects': positive_subjects,
            'negative_subjects': negative_subjects,
            'positive_sources': positive_sources,
            'source_support': by_source,
            'support_status': status,
            'reason': reason,
        }
    return rows


def _feature_family_support(df: pd.DataFrame) -> dict[str, Any]:
    roi_qc = _safe_numeric_columns(df, ROI_QC_COLUMNS)
    morphology = _safe_numeric_columns(
        df, [column for column in df.columns if column.startswith('morph_')]
    )
    learned_qc = _safe_numeric_columns(
        df,
        [
            column
            for column in df.columns
            if column.startswith('learned_simple_roi_qc_')
        ],
    )
    learned_encoder = _safe_numeric_columns(
        df,
        [
            column
            for column in df.columns
            if column.startswith('learned_current_glomeruli_encoder_')
        ],
    )
    embedding = _safe_numeric_columns(
        df, [column for column in df.columns if column.startswith('embedding_')]
    )
    return {
        'roi_qc': {
            'columns': roi_qc,
            'eligible_for_first_pass_selection': bool(roi_qc),
            'decision': 'eligible_low_capacity_current_roi_qc',
        },
        'morphology': {
            'columns': morphology,
            'eligible_for_first_pass_selection': bool(morphology),
            'decision': 'eligible_as_current_deterministic_review_features'
            if morphology
            else 'not_available',
        },
        'learned_roi': {
            'columns': learned_qc,
            'encoder_column_count': len(learned_encoder),
            'eligible_for_first_pass_selection': False,
            'decision': 'audit_comparator_only_due_p1_subject_heldout_overfit',
        },
        'embedding': {
            'columns': embedding[:8],
            'full_column_count': len(embedding),
            'eligible_for_first_pass_selection': False,
            'decision': 'audit_comparator_only_high_dimensional_embedding_overfit_risk',
        },
    }


def _feature_diagnostics(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    if not columns:
        return {
            'feature_count': 0,
            'nonfinite_count': 0,
            'zero_variance_count': 0,
            'near_zero_variance_count': 0,
            'rank_status': 'not_available',
        }
    values = df[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(values)
    variances = np.nanvar(values, axis=0)
    rank = int(np.linalg.matrix_rank(np.nan_to_num(values))) if finite.all() else 0
    return {
        'feature_count': len(columns),
        'nonfinite_count': int((~finite).sum()),
        'zero_variance_count': int(np.isclose(variances, 0.0).sum()),
        'near_zero_variance_count': int((variances < 1e-8).sum()),
        'matrix_rank': rank,
        'rank_status': 'full_column_rank' if rank >= len(columns) else 'rank_limited',
    }


def _severe_nonsevere_descriptives(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, Any]:
    if not columns:
        return {}
    severe = df['score'].astype(float) >= PRIMARY_SEVERE_THRESHOLD
    result: dict[str, Any] = {}
    for column in columns[:30]:
        values = pd.to_numeric(df[column], errors='coerce')
        result[column] = {
            'non_severe_mean': float(values.loc[~severe].mean()),
            'severe_mean': float(values.loc[severe].mean()),
            'absolute_mean_difference': float(
                abs(values.loc[severe].mean() - values.loc[~severe].mean())
            ),
        }
    return result


def _write_reliability_labels(path: Path) -> dict[str, Any]:
    payload = {
        'labels': {
            'severe_risk_positive': 'Predicted score >= 2 severe-risk positive.',
            'severe_false_negative': 'Observed score >= 2 but selected severe-risk prediction is negative.',
            'severe_false_positive': 'Observed score < 2 but selected severe-risk prediction is positive.',
            'broad_ordinal_prediction_set': 'Ordinal prediction set contains at least four supported scores.',
            'source_sensitive_severe_support': 'Severe positives are source-confounded or source-sensitive.',
            'underpowered_threshold': 'Threshold support is exploratory or non-estimable.',
        },
        'hard_blockers': [
            'unsupported_scores',
            'broken_joins',
            'nonfinite_selected_predictions',
            'subject_validation_leakage',
            'missing_required_identity_fields',
            'missing_required_verdict_or_index_artifacts',
            'claim_boundary_violation',
        ],
        'scope_limiters': [
            'score_2_3_underprediction',
            'source_confounded_severe_support',
            'broad_ordinal_prediction_sets',
            'underpowered_thresholds',
            'nonfatal_numerical_warnings',
        ],
    }
    return _save_json(payload, path) and payload


def _candidate_specs(feature_support: dict[str, Any]) -> list[CandidateSpec]:
    roi = tuple(feature_support['roi_qc']['columns'])
    morph = tuple(feature_support['morphology']['columns'])
    roi_morph = tuple(dict.fromkeys([*roi, *morph]))
    specs: list[CandidateSpec] = []
    if roi:
        specs.extend(
            [
                CandidateSpec(
                    'severe_roi_qc_threshold',
                    'image',
                    'roi_qc',
                    'severe_threshold_logistic',
                    roi,
                ),
                CandidateSpec(
                    'ordinal_roi_qc_thresholds',
                    'image',
                    'roi_qc',
                    'cumulative_threshold_logistic',
                    roi,
                    None,
                ),
                CandidateSpec(
                    'two_stage_severe_gate_roi_qc',
                    'image',
                    'roi_qc',
                    'two_stage_severe_gate_ridge',
                    roi,
                ),
            ]
        )
    if morph:
        specs.append(
            CandidateSpec(
                'severe_morphology_threshold',
                'image',
                'morphology',
                'severe_threshold_logistic',
                morph,
            )
        )
    if roi_morph:
        specs.extend(
            [
                CandidateSpec(
                    'severe_roi_qc_morphology_threshold',
                    'image',
                    'roi_qc_morphology',
                    'severe_threshold_logistic',
                    roi_morph,
                ),
                CandidateSpec(
                    'ordinal_roi_qc_morphology_thresholds',
                    'image',
                    'roi_qc_morphology',
                    'cumulative_threshold_logistic',
                    roi_morph,
                    None,
                ),
                CandidateSpec(
                    'two_stage_severe_gate_roi_qc_morphology',
                    'image',
                    'roi_qc_morphology',
                    'two_stage_severe_gate_ridge',
                    roi_morph,
                ),
                CandidateSpec(
                    'subject_severe_aware_ordinal',
                    'subject',
                    'roi_qc_morphology',
                    'subject_level_severe_aware_aggregate',
                    roi_morph,
                ),
            ]
        )
    deduped: dict[str, CandidateSpec] = {}
    for spec in specs:
        if spec.candidate_id in ALLOWED_CANDIDATE_IDS and spec.feature_columns:
            deduped[spec.candidate_id] = spec
    return list(deduped.values())


def _splitter(df: pd.DataFrame, n_splits: int) -> GroupKFold:
    group_count = df['subject_id'].astype(str).nunique()
    split_count = min(max(2, n_splits), group_count)
    if split_count < 2:
        raise SevereAwareEstimatorError(
            'Need at least two subjects for severe-aware validation.'
        )
    return GroupKFold(n_splits=split_count)


def _raise_if_subject_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    leaked = sorted(
        set(train_df['subject_id'].astype(str)) & set(test_df['subject_id'].astype(str))
    )
    if leaked:
        raise SevereAwareEstimatorError(
            'subject_id_validation_leakage detected in severe-aware estimator: '
            + ', '.join(leaked[:5])
        )


def _fit_binary_model(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[Any, list[str]]:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    if positives == 0 or negatives == 0:
        probability = float(np.mean(y_train)) if len(y_train) else 0.0
        return _ConstantProbabilityModel(probability), []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = LogisticRegression(
            penalty='l2', C=0.25, max_iter=4000, solver='lbfgs', random_state=0
        )
        model.fit(x_train, y_train)
    return model, [str(item.message) for item in caught]


@dataclass
class _ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros((x.shape[0], 2), dtype=float)
        result[:, 1] = self.probability
        result[:, 0] = 1.0 - self.probability
        return result


def _identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in IDENTITY_COLUMNS if column in df.columns]
    return df[columns].copy()


def _prediction_sets_from_thresholds(probabilities: np.ndarray) -> list[str]:
    score_prob = score_probabilities_from_thresholds(probabilities)
    return prediction_sets_from_score_probabilities(score_prob, coverage=0.90)


def _fit_candidate_oof(
    df: pd.DataFrame, spec: CandidateSpec, n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    if spec.target_level == 'subject':
        return _fit_subject_candidate_oof(df, spec, n_splits)
    x_all = (
        df[list(spec.feature_columns)]
        .apply(pd.to_numeric, errors='raise')
        .to_numpy(dtype=float)
    )
    stage = df['observed_stage_index'].to_numpy(dtype=float)
    score = df['score'].to_numpy(dtype=float)
    groups = df['subject_id'].astype(str)
    predictions = _identity_frame(df)
    predictions['observed_stage_index'] = stage
    predictions['fold'] = 0
    severe_probability = np.zeros(len(df), dtype=float)
    predicted_stage = np.zeros(len(df), dtype=float)
    threshold_probabilities = np.zeros((len(df), 5), dtype=float)
    warning_messages: list[str] = []
    for fold, (train_idx, test_idx) in enumerate(
        _splitter(df, n_splits).split(x_all, score, groups=groups), start=1
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        _raise_if_subject_leakage(train_df, test_df)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all[train_idx])
        x_test = scaler.transform(x_all[test_idx])
        predictions.iloc[test_idx, predictions.columns.get_loc('fold')] = fold
        if spec.model_family == 'cumulative_threshold_logistic':
            fold_probs = np.zeros((len(test_idx), 5), dtype=float)
            for threshold_index, threshold in enumerate((0.0, 0.5, 1.0, 1.5, 2.0)):
                y_train = (score[train_idx] > threshold).astype(int)
                model, messages = _fit_binary_model(x_train, y_train)
                warning_messages.extend(messages)
                fold_probs[:, threshold_index] = model.predict_proba(x_test)[:, 1]
            fold_probs = monotonic_threshold_probabilities(fold_probs)
            threshold_probabilities[test_idx] = fold_probs
            severe_probability[test_idx] = fold_probs[:, 4]
            predicted_stage[test_idx] = burden_from_threshold_probabilities(fold_probs)
        elif spec.model_family == 'two_stage_severe_gate_ridge':
            y_train = (score[train_idx] >= PRIMARY_SEVERE_THRESHOLD).astype(int)
            gate, messages = _fit_binary_model(x_train, y_train)
            warning_messages.extend(messages)
            gate_prob = gate.predict_proba(x_test)[:, 1]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                ridge = Ridge(alpha=1.0, solver='svd')
                ridge.fit(x_train, stage[train_idx])
                ridge_pred = np.clip(ridge.predict(x_test), 0.0, 100.0)
            warning_messages.extend(str(item.message) for item in caught)
            severe_probability[test_idx] = gate_prob
            predicted_stage[test_idx] = np.clip(
                (0.65 * ridge_pred) + (0.35 * gate_prob * 100.0), 0.0, 100.0
            )
        else:
            y_train = (score[train_idx] >= float(spec.threshold)).astype(int)
            model, messages = _fit_binary_model(x_train, y_train)
            warning_messages.extend(messages)
            prob = model.predict_proba(x_test)[:, 1]
            severe_probability[test_idx] = prob
            predicted_stage[test_idx] = np.clip(prob * 100.0, 0.0, 100.0)
    predictions['predicted_stage_index'] = predicted_stage
    predictions['predicted_score'] = _nearest_supported_score(predicted_stage)
    predictions['severe_probability'] = severe_probability
    predictions['severe_predicted_label'] = severe_probability >= 0.5
    for index, column in enumerate(THRESHOLD_PROBABILITY_COLUMNS):
        predictions[column] = (
            threshold_probabilities[:, index]
            if threshold_probabilities.any()
            else np.nan
        )
    predictions['ordinal_prediction_set'] = (
        _prediction_sets_from_thresholds(threshold_probabilities)
        if threshold_probabilities.any()
        else _stage_prediction_sets(predicted_stage)
    )
    predictions['prediction_source'] = 'validation_subject_heldout'
    predictions['candidate_id'] = spec.candidate_id
    predictions['target_level'] = spec.target_level
    predictions['feature_family'] = spec.feature_family
    predictions['model_family'] = spec.model_family
    predictions['primary_severe_threshold'] = PRIMARY_SEVERE_THRESHOLD
    predictions['reliability_label'] = _prediction_reliability_labels(predictions)
    return predictions, list(dict.fromkeys(warning_messages))


def _fit_subject_candidate_oof(
    df: pd.DataFrame, spec: CandidateSpec, n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    subject_df = df.groupby(['subject_id', 'cohort_id'], as_index=False).agg(
        score=('score', 'mean'),
        observed_stage_index=('observed_stage_index', 'mean'),
        **{column: (column, 'mean') for column in spec.feature_columns},
    )
    x_all = subject_df[list(spec.feature_columns)].to_numpy(dtype=float)
    score = subject_df['score'].to_numpy(dtype=float)
    stage = subject_df['observed_stage_index'].to_numpy(dtype=float)
    predictions = subject_df[
        ['subject_id', 'cohort_id', 'score', 'observed_stage_index']
    ].copy()
    predictions['fold'] = 0
    predictions['severe_probability'] = 0.0
    predictions['predicted_stage_index'] = 0.0
    warning_messages: list[str] = []
    for fold, (train_idx, test_idx) in enumerate(
        _splitter(subject_df, n_splits).split(
            x_all, score, groups=subject_df['subject_id'].astype(str)
        ),
        start=1,
    ):
        train_df = subject_df.iloc[train_idx]
        test_df = subject_df.iloc[test_idx]
        _raise_if_subject_leakage(train_df, test_df)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all[train_idx])
        x_test = scaler.transform(x_all[test_idx])
        model, messages = _fit_binary_model(
            x_train, (score[train_idx] >= PRIMARY_SEVERE_THRESHOLD).astype(int)
        )
        warning_messages.extend(messages)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ridge = Ridge(alpha=1.0, solver='svd')
            ridge.fit(x_train, stage[train_idx])
            ridge_pred = np.clip(ridge.predict(x_test), 0.0, 100.0)
        warning_messages.extend(str(item.message) for item in caught)
        prob = model.predict_proba(x_test)[:, 1]
        predictions.loc[predictions.index[test_idx], 'fold'] = fold
        predictions.loc[predictions.index[test_idx], 'severe_probability'] = prob
        predictions.loc[predictions.index[test_idx], 'predicted_stage_index'] = (
            0.65 * ridge_pred + 0.35 * prob * 100.0
        )
    predictions['predicted_score'] = _nearest_supported_score(
        predictions['predicted_stage_index'].to_numpy(dtype=float)
    )
    predictions['severe_predicted_label'] = predictions['severe_probability'] >= 0.5
    predictions['candidate_id'] = spec.candidate_id
    predictions['target_level'] = spec.target_level
    predictions['feature_family'] = spec.feature_family
    predictions['model_family'] = spec.model_family
    predictions['primary_severe_threshold'] = PRIMARY_SEVERE_THRESHOLD
    predictions['prediction_source'] = 'validation_subject_heldout'
    predictions['reliability_label'] = _prediction_reliability_labels(predictions)
    return predictions, list(dict.fromkeys(warning_messages))


def _nearest_supported_score(stage_index: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(list(STAGE_INDEX_BY_SCORE.values()), dtype=float)
    scores = np.asarray(list(STAGE_INDEX_BY_SCORE.keys()), dtype=float)
    index = np.abs(
        np.asarray(stage_index, dtype=float)[:, None] - values[None, :]
    ).argmin(axis=1)
    return scores[index]


def _stage_prediction_sets(predicted_stage: np.ndarray) -> list[str]:
    values = np.asarray(list(STAGE_INDEX_BY_SCORE.values()), dtype=float)
    scores = np.asarray(list(STAGE_INDEX_BY_SCORE.keys()), dtype=float)
    sets: list[str] = []
    for value in predicted_stage:
        included = [
            f'{score:g}'
            for score, stage in zip(scores, values)
            if abs(float(stage) - float(value)) <= 30.0
        ]
        sets.append(
            '|'.join(
                included or [f'{_nearest_supported_score(np.array([value]))[0]:g}']
            )
        )
    return sets


def _prediction_reliability_labels(predictions: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for _, row in predictions.iterrows():
        row_labels: list[str] = []
        observed_severe = float(row['score']) >= PRIMARY_SEVERE_THRESHOLD
        predicted_severe = bool(row.get('severe_predicted_label', False))
        if predicted_severe:
            row_labels.append('severe_risk_positive')
        if observed_severe and not predicted_severe:
            row_labels.append('severe_false_negative')
        if (not observed_severe) and predicted_severe:
            row_labels.append('severe_false_positive')
        prediction_set = str(row.get('ordinal_prediction_set', ''))
        if len([part for part in prediction_set.split('|') if part]) >= 4:
            row_labels.append('broad_ordinal_prediction_set')
        if float(row.get('score', 0.0)) >= 2.0 and str(row.get('cohort_id', '')):
            row_labels.append('source_sensitive_severe_support')
        labels.append('|'.join(row_labels) if row_labels else 'nominal_current_data')
    return labels


def _metric_row(
    predictions: pd.DataFrame,
    spec: CandidateSpec,
    *,
    split_label: str,
    split_description: str,
    eligible_for_model_selection: bool,
    warning_messages: list[str],
) -> dict[str, Any]:
    if predictions.empty:
        return {
            'candidate_id': spec.candidate_id,
            'target_level': spec.target_level,
            'feature_family': spec.feature_family,
            'model_family': spec.model_family,
            'threshold_target': spec.threshold if spec.threshold is not None else '',
            'split_label': split_label,
            'split_description': split_description,
            'row_count': 0,
            'subject_count': 0,
            'source_scope': '',
            'stage_index_mae': np.nan,
            'grade_scale_mae': np.nan,
            'severe_recall': np.nan,
            'severe_precision': np.nan,
            'severe_false_negative_count': np.nan,
            'severe_false_negative_rate': np.nan,
            'finite_output_status': 'not_applicable',
            'warning_count': len(warning_messages),
            'intended_use': 'not_available',
            'eligible_for_model_selection': eligible_for_model_selection,
        }
    predicted = predictions['predicted_stage_index'].to_numpy(dtype=float)
    observed = predictions['observed_stage_index'].to_numpy(dtype=float)
    finite = bool(np.isfinite(predicted).all())
    predicted_score = (
        _nearest_supported_score(predicted)
        if finite
        else np.full(len(predictions), np.nan)
    )
    observed_severe = (
        predictions['score'].astype(float).to_numpy() >= PRIMARY_SEVERE_THRESHOLD
    )
    predicted_severe = predictions['severe_predicted_label'].astype(bool).to_numpy()
    tp = int(np.sum(observed_severe & predicted_severe))
    fp = int(np.sum((~observed_severe) & predicted_severe))
    fn = int(np.sum(observed_severe & (~predicted_severe)))
    positives = int(np.sum(observed_severe))
    return {
        'candidate_id': spec.candidate_id,
        'target_level': spec.target_level,
        'feature_family': spec.feature_family,
        'model_family': spec.model_family,
        'threshold_target': spec.threshold if spec.threshold is not None else 'ordinal',
        'split_label': split_label,
        'split_description': split_description,
        'row_count': int(len(predictions)),
        'subject_count': int(predictions['subject_id'].astype(str).nunique()),
        'source_scope': '|'.join(sorted(predictions['cohort_id'].astype(str).unique()))
        if 'cohort_id' in predictions.columns
        else '',
        'stage_index_mae': float(mean_absolute_error(observed, predicted))
        if finite
        else np.nan,
        'grade_scale_mae': float(
            mean_absolute_error(predictions['score'].astype(float), predicted_score)
        )
        if finite
        else np.nan,
        'severe_recall': float(tp / positives) if positives else np.nan,
        'severe_precision': float(tp / (tp + fp)) if (tp + fp) else np.nan,
        'severe_false_negative_count': fn,
        'severe_false_negative_rate': float(fn / positives) if positives else np.nan,
        'finite_output_status': 'finite' if finite else 'nonfinite',
        'warning_count': len(warning_messages),
        'intended_use': 'current_data_severe_aware_review',
        'eligible_for_model_selection': eligible_for_model_selection,
    }


def _source_severe_sensitivity(
    predictions: pd.DataFrame, support: dict[str, Any]
) -> dict[str, Any]:
    by_cohort: list[dict[str, Any]] = []
    for cohort, subset in predictions.groupby('cohort_id'):
        observed = subset['score'].astype(float).to_numpy() >= PRIMARY_SEVERE_THRESHOLD
        predicted = subset['severe_predicted_label'].astype(bool).to_numpy()
        positives = int(observed.sum())
        tp = int(np.sum(observed & predicted))
        fp = int(np.sum((~observed) & predicted))
        fn = int(np.sum(observed & (~predicted)))
        by_cohort.append(
            {
                'cohort_id': str(cohort),
                'row_count': int(len(subset)),
                'subject_count': int(subset['subject_id'].astype(str).nunique()),
                'positive_rows_score_gte_2': positives,
                'severe_recall': float(tp / positives) if positives else None,
                'severe_precision': float(tp / (tp + fp)) if (tp + fp) else None,
                'severe_false_negative_count': fn,
                'status': 'estimated'
                if positives and len(subset) > positives
                else 'non_estimable',
                'reason': ''
                if positives and len(subset) > positives
                else 'missing_positive_or_negative_rows',
            }
        )
    return {
        'threshold_support': support,
        'by_cohort': by_cohort,
        'leave_source_out': [
            {
                'cohort_id': item['cohort_id'],
                'status': 'non_estimable_current_data_sensitivity',
                'reason': 'source-confounded severe support prevents external-test interpretation',
            }
            for item in by_cohort
        ],
        'interpretation': 'source-stratified current-data sensitivity only; not external validation',
    }


def _write_severe_threshold_metrics(
    predictions: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in SEVERE_THRESHOLDS:
        observed = predictions['score'].astype(float).to_numpy() >= threshold
        predicted = predictions['severe_probability'].astype(float).to_numpy() >= 0.5
        positives = int(observed.sum())
        tp = int(np.sum(observed & predicted))
        fp = int(np.sum((~observed) & predicted))
        fn = int(np.sum(observed & (~predicted)))
        rows.append(
            {
                'threshold': float(threshold),
                'row_count': int(len(predictions)),
                'positive_rows': positives,
                'positive_subjects': int(
                    predictions.loc[observed, 'subject_id'].astype(str).nunique()
                ),
                'severe_recall': float(tp / positives) if positives else np.nan,
                'severe_precision': float(tp / (tp + fp)) if (tp + fp) else np.nan,
                'severe_false_negative_count': fn,
                'severe_false_negative_rate': float(fn / positives)
                if positives
                else np.nan,
                'support_scope': 'current_data_source_sensitive'
                if threshold >= 2.0
                else 'current_data',
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def _write_evidence_review(
    path: Path, predictions: pd.DataFrame, verdict: dict[str, Any]
) -> None:
    ordered = predictions.copy()
    ordered['abs_error'] = (
        ordered['observed_stage_index'] - ordered['predicted_stage_index']
    ).abs()
    buckets = [
        (
            'Severe true positives',
            ordered[
                (ordered['score'] >= 2.0) & (ordered['severe_predicted_label'])
            ].head(12),
        ),
        (
            'Severe false negatives',
            ordered[(ordered['score'] >= 2.0) & (~ordered['severe_predicted_label'])]
            .sort_values('abs_error', ascending=False)
            .head(18),
        ),
        (
            'Low/mid false positives',
            ordered[
                (ordered['score'] < 2.0) & (ordered['severe_predicted_label'])
            ].head(12),
        ),
        (
            'High-uncertainty severe cases',
            ordered[
                (ordered['score'] >= 2.0)
                & (
                    ordered['reliability_label'].str.contains(
                        'broad_ordinal_prediction_set|severe_false_negative', regex=True
                    )
                )
            ].head(18),
        ),
    ]
    review_queue = (
        ordered[(ordered['score'] >= 2.0) & (~ordered['severe_predicted_label'])]
        .sort_values('abs_error', ascending=False)
        .head(120)
        .copy()
    )
    sections: list[str] = []
    for title, subset in buckets:
        rows = []
        for _, row in subset.iterrows():
            localization = 'feature_or_model_limitation_current_roi_available'
            if (
                not str(row.get('roi_image_path', '')).strip()
                or not str(row.get('roi_mask_path', '')).strip()
            ):
                localization = 'upstream_roi_or_mask_provenance_missing'
            rows.append(
                '<tr>'
                f'<td>{escape(str(row.get("subject_image_id", "")))}</td>'
                f'<td>{escape(str(row.get("cohort_id", "")))}</td>'
                f'<td>{float(row.get("score", 0.0)):g}</td>'
                f'<td>{float(row.get("predicted_stage_index", 0.0)):.2f}</td>'
                f'<td>{float(row.get("severe_probability", 0.0)):.3f}</td>'
                f'<td>{escape(str(row.get("ordinal_prediction_set", "")))}</td>'
                f'<td>{escape(str(row.get("reliability_label", "")))}</td>'
                f'<td>{escape(localization)}</td>'
                f'<td>{escape(str(row.get("roi_image_path", "")))}</td>'
                '</tr>'
            )
        if not rows:
            rows.append('<tr><td colspan="9">No examples available.</td></tr>')
        sections.append(
            f'<h2>{escape(title)}</h2><table><thead><tr><th>Image</th><th>Cohort</th><th>Observed</th><th>Predicted stage</th><th>Severe risk</th><th>Ordinal set</th><th>Reliability</th><th>Failure localization</th><th>ROI</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Severe False Negative Review</title>
<style>{_evidence_review_css()}</style></head>
<body><h1>Severe False Negative Review</h1>
<div class="note">Failure localization is first-pass evidence from current artifact provenance. It separates missing ROI/mask provenance from feature or model limitation; it is not manual pathology adjudication.</div>
<p>Status: {escape(str(verdict.get('overall_status', '')))}. Claim boundary: {escape(str(verdict.get('claim_boundary', '')))}</p>
{_interactive_review_queue_html(review_queue)}
{''.join(sections)}</body></html>"""
    path.write_text(html, encoding='utf-8')


def _evidence_review_css() -> str:
    return """
body{font-family:Arial,sans-serif;margin:2rem;max-width:1440px;color:#111827}
table{border-collapse:collapse;width:100%;margin-bottom:2rem}
td,th{border-bottom:1px solid #ddd;padding:.4rem;text-align:left;font-size:.9rem;vertical-align:top}
.note{background:#fff7e6;border-left:4px solid #b45309;padding:1rem;margin:1rem 0}
.review-toolbar{position:sticky;top:0;background:#ffffff;border-bottom:1px solid #d1d5db;padding:.75rem 0;margin:1rem 0;z-index:10}
.review-toolbar button{margin-right:.5rem;padding:.45rem .7rem;border:1px solid #9ca3af;background:#f9fafb;border-radius:4px;cursor:pointer}
.review-card{border:1px solid #d1d5db;border-radius:6px;margin:1rem 0;padding:1rem;background:#ffffff}
.review-card.reviewed{border-color:#047857;background:#f0fdf4}
.review-card h3{margin:.1rem 0 .75rem;font-size:1rem}
.review-grid{display:grid;grid-template-columns:minmax(280px,1fr) minmax(280px,1fr);gap:1rem}
.image-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem}
.image-panel{border:1px solid #e5e7eb;border-radius:4px;padding:.5rem;background:#f9fafb}
.image-panel img{display:block;max-width:100%;max-height:320px;object-fit:contain;margin:.35rem auto;background:#111827}
.image-panel a{word-break:break-all;font-size:.8rem}
.field-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem}
label{display:block;font-size:.8rem;font-weight:700;color:#374151;margin-bottom:.25rem}
select,textarea,input{width:100%;box-sizing:border-box;border:1px solid #9ca3af;border-radius:4px;padding:.4rem;font:inherit}
textarea{min-height:5rem}
.metadata{font-size:.9rem;line-height:1.45}
.metadata code{background:#f3f4f6;padding:.1rem .25rem;border-radius:3px}
.export-panel{border:1px solid #d1d5db;border-radius:6px;padding:1rem;margin:1rem 0;background:#f9fafb}
.export-panel textarea{height:12rem;font-family:monospace;font-size:.8rem}
.empty{color:#6b7280;font-style:italic}
"""


def _file_uri(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    try:
        path = Path(text)
        if path.is_absolute():
            return path.as_uri()
    except ValueError:
        return ''
    return 'file://' + quote(text)


def _image_panel_html(label: str, value: Any) -> str:
    uri = _file_uri(value)
    display = escape(str(value or ''))
    if not uri:
        return f'<div class="image-panel"><strong>{escape(label)}</strong><p class="empty">Missing path.</p></div>'
    escaped_uri = escape(uri, quote=True)
    return (
        '<div class="image-panel">'
        f'<strong>{escape(label)}</strong>'
        f'<a href="{escaped_uri}" target="_blank">{display}</a>'
        f'<img src="{escaped_uri}" alt="{escape(label)}">'
        '</div>'
    )


def _interactive_review_queue_html(review_queue: pd.DataFrame) -> str:
    cards: list[str] = []
    for index, row in enumerate(review_queue.itertuples(index=False), start=1):
        payload = row._asdict()
        subject_image_id = str(payload.get('subject_image_id', '') or f'row_{index}')
        review_id = quote(subject_image_id, safe='')
        cards.append(
            '<article class="review-card" data-review-id="'
            f'{escape(review_id, quote=True)}" data-subject-image-id="'
            f'{escape(subject_image_id, quote=True)}">'
            f'<h3>{index}. {escape(subject_image_id)}</h3>'
            '<div class="review-grid">'
            '<div class="image-grid">'
            f'{_image_panel_html("ROI image", payload.get("roi_image_path", ""))}'
            f'{_image_panel_html("ROI mask", payload.get("roi_mask_path", ""))}'
            f'{_image_panel_html("Raw image", payload.get("raw_image_path", ""))}'
            f'{_image_panel_html("Raw mask", payload.get("raw_mask_path", ""))}'
            '</div>'
            '<div>'
            '<div class="metadata">'
            f'<p><strong>Cohort:</strong> {escape(str(payload.get("cohort_id", "")))}</p>'
            f'<p><strong>Observed score:</strong> {float(payload.get("score", 0.0)):g}; '
            f'<strong>Predicted stage:</strong> {float(payload.get("predicted_stage_index", 0.0)):.2f}; '
            f'<strong>Severe risk:</strong> {float(payload.get("severe_probability", 0.0)):.3f}</p>'
            f'<p><strong>Ordinal set:</strong> <code>{escape(str(payload.get("ordinal_prediction_set", "")))}</code></p>'
            f'<p><strong>Reliability:</strong> {escape(str(payload.get("reliability_label", "")))}</p>'
            f'<p><strong>Subject:</strong> {escape(str(payload.get("subject_id", "")))}; '
            f'<strong>Sample:</strong> {escape(str(payload.get("sample_id", "")))}; '
            f'<strong>Image:</strong> {escape(str(payload.get("image_id", "")))}</p>'
            '</div>'
            '<div class="field-grid">'
            '<div><label>Grade adjudication</label><select data-field="grade_adjudication">'
            '<option value="">Unreviewed</option>'
            '<option value="grade_correct">Grade appears correct</option>'
            '<option value="grade_too_high">Grade appears too high</option>'
            '<option value="grade_too_low">Grade appears too low</option>'
            '<option value="visually_ambiguous">Visually ambiguous</option>'
            '<option value="exclude_or_bad_image">Exclude or bad image</option>'
            '<option value="needs_expert_review">Needs expert review</option>'
            '</select></div>'
            '<div><label>Likely failure source</label><select data-field="failure_source">'
            '<option value="">Unreviewed</option>'
            '<option value="valid_grade_model_miss">Valid grade, model missed it</option>'
            '<option value="roi_or_mask_failure">ROI or mask failure</option>'
            '<option value="feature_insufficiency">Feature insufficiency</option>'
            '<option value="label_ambiguity">Label ambiguity</option>'
            '<option value="image_quality_or_source_artifact">Image quality or source artifact</option>'
            '<option value="not_severe_after_review">Not grade 2 or 3 after review</option>'
            '</select></div>'
            '<div><label>Confidence</label><select data-field="review_confidence">'
            '<option value="">Unreviewed</option>'
            '<option value="high">High</option>'
            '<option value="medium">Medium</option>'
            '<option value="low">Low</option>'
            '</select></div>'
            '</div>'
            '<p><label>Reviewer notes</label><textarea data-field="review_notes" '
            'placeholder="Record what you see and whether the original score appears correct."></textarea></p>'
            '</div></div></article>'
        )
    if not cards:
        cards.append(
            '<p class="empty">No severe false negatives available for review.</p>'
        )
    return f"""
<section id="interactive-review">
<h2>Interactive Adjudication Queue</h2>
<div class="note">Review state is stored in this browser for this local HTML file. Use Export JSON or Export CSV to save adjudications outside the browser.</div>
<div class="review-toolbar">
<span id="review-progress">Reviewed 0 of 0</span>
<button type="button" id="show-unreviewed">Show next unreviewed</button>
<button type="button" id="export-json">Export JSON</button>
<button type="button" id="export-csv">Export CSV</button>
<button type="button" id="clear-review-state">Clear local review state</button>
</div>
<div class="export-panel">
<p>If browser downloads are blocked, use these text boxes. Click Prepare JSON or Prepare CSV, then copy the text.</p>
<button type="button" id="prepare-json">Prepare JSON text</button>
<button type="button" id="prepare-csv">Prepare CSV text</button>
<p><label>JSON export text</label><textarea id="json-export-text" readonly></textarea></p>
<p><label>CSV export text</label><textarea id="csv-export-text" readonly></textarea></p>
</div>
{''.join(cards)}
</section>
<script>{_interactive_review_js()}</script>
"""


def _interactive_review_js() -> str:
    return r"""
(function(){
  const storageKey = 'eq_severe_false_negative_review::' + window.location.pathname;
  const cards = Array.from(document.querySelectorAll('.review-card'));
  function loadState(){
    try { return JSON.parse(localStorage.getItem(storageKey) || '{}'); }
    catch (error) { return {}; }
  }
  function saveState(state){ localStorage.setItem(storageKey, JSON.stringify(state)); }
  function cardData(card){
    const data = {subject_image_id: card.dataset.subjectImageId || ''};
    card.querySelectorAll('[data-field]').forEach((field) => {
      data[field.dataset.field] = field.value || '';
    });
    return data;
  }
  function isReviewed(data){
    return Boolean(data.grade_adjudication || data.failure_source || data.review_confidence || data.review_notes);
  }
  function refresh(){
    const state = loadState();
    let reviewed = 0;
    cards.forEach((card) => {
      const id = card.dataset.reviewId;
      const data = state[id] || {};
      card.querySelectorAll('[data-field]').forEach((field) => {
        field.value = data[field.dataset.field] || '';
      });
      const done = isReviewed(cardData(card));
      card.classList.toggle('reviewed', done);
      if (done) reviewed += 1;
    });
    const progress = document.getElementById('review-progress');
    if (progress) progress.textContent = 'Reviewed ' + reviewed + ' of ' + cards.length;
  }
  function updateCard(card){
    const state = loadState();
    state[card.dataset.reviewId] = cardData(card);
    saveState(state);
    refresh();
  }
  function exportRows(){
    const state = loadState();
    return cards.map((card) => Object.assign(
      {subject_image_id: card.dataset.subjectImageId || ''},
      state[card.dataset.reviewId] || cardData(card)
    ));
  }
  function csvText(){
    const rows = exportRows();
    const headers = ['subject_image_id','grade_adjudication','failure_source','review_confidence','review_notes'];
    return [headers.join(',')].concat(rows.map((row) =>
      headers.map((header) => csvEscape(row[header] || '')).join(',')
    )).join('\n');
  }
  function jsonText(){
    return JSON.stringify(exportRows(), null, 2);
  }
  function putText(id, text){
    const field = document.getElementById(id);
    if (!field) return;
    field.value = text;
    field.focus();
    field.select();
  }
  function download(filename, mime, text){
    putText(filename.endsWith('.csv') ? 'csv-export-text' : 'json-export-text', text);
    try {
      const blob = new Blob([text], {type: mime});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.warn('Browser download failed; export text is available on page.', error);
    }
  }
  function csvEscape(value){
    const text = String(value == null ? '' : value);
    return '"' + text.replace(/"/g, '""') + '"';
  }
  cards.forEach((card) => {
    card.querySelectorAll('[data-field]').forEach((field) => {
      field.addEventListener('change', () => updateCard(card));
      field.addEventListener('input', () => updateCard(card));
    });
  });
  document.getElementById('show-unreviewed')?.addEventListener('click', () => {
    const target = cards.find((card) => !card.classList.contains('reviewed'));
    if (target) target.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  document.getElementById('export-json')?.addEventListener('click', () => {
    download('severe_false_negative_adjudications.json', 'application/json', jsonText());
  });
  document.getElementById('export-csv')?.addEventListener('click', () => {
    download('severe_false_negative_adjudications.csv', 'text/csv', csvText());
  });
  document.getElementById('prepare-json')?.addEventListener('click', () => {
    putText('json-export-text', jsonText());
  });
  document.getElementById('prepare-csv')?.addEventListener('click', () => {
    putText('csv-export-text', csvText());
  });
  document.getElementById('clear-review-state')?.addEventListener('click', () => {
    if (confirm('Clear local adjudications for this file?')) {
      localStorage.removeItem(storageKey);
      refresh();
    }
  });
  refresh();
})();
"""


def _plot_bar(
    path: Path, labels: list[str], values: list[float], title: str, ylabel: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color='#3b82f6')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_figures(
    paths: dict[str, Path], predictions: pd.DataFrame, threshold_metrics: pd.DataFrame
) -> None:
    figures = paths['figures']
    _plot_bar(
        figures / 'severe_threshold_metrics.png',
        [f'>={row.threshold:g}' for row in threshold_metrics.itertuples()],
        [
            0.0 if pd.isna(row.severe_recall) else float(row.severe_recall)
            for row in threshold_metrics.itertuples()
        ],
        'Severe Threshold Recall',
        'Recall',
    )
    plt.figure(figsize=(5, 5))
    plt.scatter(
        predictions['observed_stage_index'],
        predictions['predicted_stage_index'],
        c=predictions['score'],
        cmap='viridis',
    )
    plt.xlabel('Observed stage index')
    plt.ylabel('Predicted stage index')
    plt.tight_layout()
    plt.savefig(figures / 'predicted_vs_observed_severity.png', dpi=150)
    plt.close()
    label_counts = predictions['reliability_label'].str.get_dummies('|').sum()
    _plot_bar(
        figures / 'severe_false_negative_summary.png',
        ['FN', 'TP'],
        [
            float(
                (
                    (predictions['score'] >= 2.0)
                    & (~predictions['severe_predicted_label'])
                ).sum()
            ),
            float(
                (
                    (predictions['score'] >= 2.0)
                    & (predictions['severe_predicted_label'])
                ).sum()
            ),
        ],
        'Score >= 2 Severe Cases',
        'Rows',
    )
    score_mae = (
        predictions.assign(
            abs_error=(
                predictions['observed_stage_index']
                - predictions['predicted_stage_index']
            ).abs()
        )
        .groupby('score')['abs_error']
        .mean()
    )
    _plot_bar(
        figures / 'calibration_by_score.png',
        [f'{score:g}' for score in score_mae.index],
        [float(value) for value in score_mae.values],
        'Stage Error By Observed Score',
        'Mean absolute stage error',
    )
    source_recall = []
    source_labels = []
    for cohort, subset in predictions.groupby('cohort_id'):
        observed = subset['score'].astype(float) >= 2.0
        predicted = subset['severe_predicted_label'].astype(bool)
        source_labels.append(str(cohort))
        source_recall.append(
            float((observed & predicted).sum() / observed.sum())
            if observed.sum()
            else 0.0
        )
    _plot_bar(
        figures / 'source_severe_performance.png',
        source_labels,
        source_recall,
        'Source Severe Recall',
        'Recall',
    )
    set_width = (
        predictions['ordinal_prediction_set']
        .astype(str)
        .map(lambda x: len([p for p in x.split('|') if p]))
    )
    _plot_bar(
        figures / 'ordinal_prediction_set_width.png',
        [str(int(width)) for width in sorted(set_width.unique())],
        [float((set_width == width).sum()) for width in sorted(set_width.unique())],
        'Ordinal Prediction Set Width',
        'Rows',
    )
    _plot_bar(
        figures / 'reliability_label_counts.png',
        [str(index) for index in label_counts.index],
        [float(value) for value in label_counts.values],
        'Reliability Labels',
        'Rows',
    )


def _artifact_manifest(root: Path) -> dict[str, Any]:
    artifacts = []
    for relative in REQUIRED_RELATIVE_ARTIFACTS:
        artifacts.append(
            {
                'relative_path': relative,
                'role': 'first_pass_severe_aware_estimator_artifact',
                'consumer': 'runtime_reviewer',
                'reportability': 'verdict_scoped',
                'required': True,
                'exists': (root / relative).exists(),
            }
        )
    actual = sorted(
        str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()
    )
    return {
        'root': str(root),
        'manifest_complete': sorted(actual) == sorted(REQUIRED_RELATIVE_ARTIFACTS),
        'unmanifested_artifacts': sorted(
            set(actual) - set(REQUIRED_RELATIVE_ARTIFACTS)
        ),
        'missing_artifacts': sorted(set(REQUIRED_RELATIVE_ARTIFACTS) - set(actual)),
        'artifacts': artifacts,
    }


def _write_verdict_md(path: Path, verdict: dict[str, Any]) -> None:
    text = f"""# Severe-Aware Ordinal Estimator Verdict

- Status: `{verdict.get('overall_status')}`
- Selected image candidate: `{verdict.get('selected_image_candidate') or 'none'}`
- Selected subject candidate: `{verdict.get('selected_subject_candidate') or 'none'}`
- Selected severe threshold: `{verdict.get('selected_severe_threshold') or 'none'}`
- Selected output type: `{verdict.get('selected_output_type')}`
- Testing status: `{verdict.get('testing_status')}`
- README snippet eligible: `{verdict.get('readme_snippet_eligible')}`

## Claim Boundary

{verdict.get('claim_boundary')}

## Hard Blockers

{', '.join(map(str, verdict.get('hard_blockers', []))) or 'none'}

## Scope Limiters

{', '.join(map(str, verdict.get('scope_limiters', []))) or 'none'}

## Next Action

{verdict.get('next_action')}
"""
    path.write_text(text, encoding='utf-8')


def _write_index(path: Path, verdict: dict[str, Any], manifest: dict[str, Any]) -> None:
    text = f"""# Severe-Aware Ordinal Estimator

Open `summary/estimator_verdict.md` first, then `summary/metrics_by_split.csv` and `evidence/severe_false_negative_review.html`.

Status: `{verdict.get('overall_status')}`.

Claim boundary: {verdict.get('claim_boundary')}.

Manifest complete: `{manifest.get('manifest_complete')}`.

All artifacts in this estimator live under this indexed subtree. There are no supported flat aliases under `burden_model/`.
"""
    path.write_text(text, encoding='utf-8')


def _write_audit_results(change_dir: Path, payload: dict[str, Any]) -> None:
    threshold_support = payload['threshold_support']
    score3_status = threshold_support['score_gte_3']['support_status']
    lines = [
        '# P2 Audit Results',
        '',
        '## Helper Reuse Decision',
        '',
        'Inspected `src/eq/quantification/burden.py` functions `validate_score_values`, `score_to_stage_index`, `threshold_targets`, `burden_from_threshold_probabilities`, `monotonic_threshold_probabilities`, `score_probabilities_from_thresholds`, and `prediction_sets_from_score_probabilities`, plus `evaluate_burden_index_table` and `_write_threshold_support`. P2 reuses the low-level rubric and cumulative-threshold helpers, but keeps the evaluator contained in `src/eq/quantification/severe_aware_ordinal_estimator.py` because the existing burden evaluator is embedding-output specific and writes a different artifact schema.',
        '',
        'Inspected `src/eq/quantification/source_aware_estimator.py` functions `source_aware_output_paths`, `_candidate_specs`, `_fit_predict_ridge`, `_raise_if_subject_leakage`, `_metric_row`, `_source_sensitivity`, and `evaluate_source_aware_endotheliosis_estimator`. P1 warning behavior remains a scope limiter, not a hard blocker, unless selected predictions are nonfinite or grouped validation leaks subjects.',
        '',
        '## P1 Runtime Handoff',
        '',
        f'Selected P1 image candidate: `{payload["source_aware_handoff"].get("selected_image_candidate")}`.',
        f'Selected P1 subject candidate: `{payload["source_aware_handoff"].get("selected_subject_candidate")}`.',
        f'Testing status: `{payload["source_aware_handoff"].get("testing_status")}`.',
        f'Scope limiters: `{", ".join(map(str, payload["source_aware_handoff"].get("scope_limiters", []))) or "none"}`.',
        f'High-score behavior: `{payload["source_aware_handoff"].get("high_score_behavior")}`.',
        '',
        '## Threshold Support Decision',
        '',
        f'`score >= 1.5`: `{threshold_support["score_gte_1p5"]["support_status"]}` with {threshold_support["score_gte_1p5"]["positive_rows"]} positive rows and {threshold_support["score_gte_1p5"]["positive_subjects"]} positive subjects.',
        f'`score >= 2`: `{threshold_support["score_gte_2"]["support_status"]}` with {threshold_support["score_gte_2"]["positive_rows"]} positive rows and {threshold_support["score_gte_2"]["positive_subjects"]} positive subjects.',
        f'`score >= 3`: `{score3_status}` with {threshold_support["score_gte_3"]["positive_rows"]} positive rows and {threshold_support["score_gte_3"]["positive_subjects"]} positive subjects; it is not eligible as a reportable selected threshold because positives are confined to one source tail stratum.',
        '',
        '## Feature Family Decision',
        '',
        'ROI/QC and deterministic morphology are eligible for first-pass severe-aware candidate selection. Learned ROI and embedding-heavy inputs are audit comparators only because P1 showed strong training/apparent fit with degraded subject-heldout validation.',
        '',
        '## Severe Failure Localization And Upstream Escalation',
        '',
        'Current severe false negatives have ROI and mask provenance, so the first-pass failure localizes to current feature/model limitations rather than a proven missing-ROI failure. Targeted manual annotation and segmentation-backbone comparison are deferred unless the severe false-negative review identifies recurrent bad ROI extraction or mask geometry errors.',
        '',
        'MedSAM/SAM audit is deferred to a separate upstream comparison. If opened, it must split oracle-box evidence from automatic-prompt evidence and measure segmentation metrics, prompt failures, ROI-feature stability, and downstream severe false-negative behavior.',
        '',
        'Current feasibility inventory: FastAI U-Net, torchvision DeepLabV3, torch, opencv, and albumentations are installed in `eq-mac`; nnU-Net v2, DeepLabV3+ dependency stacks, Mask2Former-style stacks, SAM, and MedSAM are not installed and require separate environment/weights planning before they can be treated as runtime dependencies.',
        '',
        'Alternative segmentation could matter only if changed masks alter severity-relevant ROI/morphology features such as lumen openness, area/fill fraction, component topology, pale/open-space features, slit-like structure, boundary fragmentation, or closed/open-lumen proxies and those changes reduce score >= 2 false negatives.',
        '',
        '## Logging Participation',
        '',
        '`src/eq/utils/execution_logging.py` is present. P2 participates as `high_level_function_events_only`: durable capture is owned by `endotheliosis_quantification` through `eq run-config --config configs/endotheliosis_quantification.yaml`; P2 adds no independent log root, file handlers, or tee implementation.',
    ]
    (change_dir / 'audit-results.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )


def evaluate_severe_aware_ordinal_endotheliosis_estimator(
    embedding_df: pd.DataFrame,
    burden_output_dir: Path,
    *,
    n_splits: int = 3,
    change_dir: Path | None = None,
) -> dict[str, Path]:
    """Evaluate severe-threshold, ordinal, and subject-level severe-aware candidates."""
    start = time.time()
    paths = severe_aware_output_paths(burden_output_dir)
    failing_step = 'start'
    LOGGER.info('P2_SEVERE_AWARE_EVENT=start')
    LOGGER.info('P2_SEVERE_AWARE_INPUT_BURDEN_ROOT=%s', burden_output_dir)
    LOGGER.info('P2_SEVERE_AWARE_OUTPUT_ROOT=%s', paths['root'])
    try:
        for key in [
            'summary',
            'figures',
            'predictions',
            'diagnostics',
            'evidence',
            'internal',
        ]:
            paths[key].mkdir(parents=True, exist_ok=True)
        _validate_inputs(embedding_df)
        failing_step = 'assemble_features'
        feature_df = _assemble_feature_table(embedding_df, burden_output_dir)
        support = compute_threshold_support(feature_df)
        feature_support = _feature_family_support(feature_df)
        all_feature_columns = sorted(
            set(feature_support['roi_qc']['columns'])
            | set(feature_support['morphology']['columns'])
            | set(feature_support['learned_roi']['columns'])
            | set(feature_support['embedding']['columns'])
        )
        separability = {
            'row_count': int(len(feature_df)),
            'subject_count': int(feature_df['subject_id'].astype(str).nunique()),
            'source_count': int(feature_df['cohort_id'].astype(str).nunique()),
            'threshold_support': support,
            'feature_family_support': {
                key: {k: v for k, v in item.items() if k != 'columns'}
                | {'feature_count': len(item.get('columns', []))}
                for key, item in feature_support.items()
            },
            'feature_diagnostics': {
                family: _feature_diagnostics(feature_df, item.get('columns', []))
                for family, item in feature_support.items()
            },
            'warning_diagnostics': {
                'expected_warning_policy': 'nonfatal_numerical_warnings_are_scope_limiters_unless_selected_outputs_nonfinite'
            },
            'severe_nonsevere_descriptive_summaries': _severe_nonsevere_descriptives(
                feature_df, all_feature_columns
            ),
        }
        _save_json(separability, paths['severe_separability_audit'])
        _save_json(support, paths['threshold_support'])
        source_handoff = _load_source_aware_handoff(Path(burden_output_dir))
        specs = _candidate_specs(feature_support)
        if not specs:
            raise SevereAwareEstimatorError(
                'No audit-approved severe-aware feature families are available.'
            )
        LOGGER.info(
            'P2_SEVERE_AWARE_COUNTS rows=%d subjects=%d sources=%d',
            len(feature_df),
            feature_df['subject_id'].astype(str).nunique(),
            feature_df['cohort_id'].astype(str).nunique(),
        )
        LOGGER.info('P2_SEVERE_AWARE_THRESHOLD_SUPPORT=%s', support)
        LOGGER.info(
            'P2_SEVERE_AWARE_FEATURE_FAMILIES=%s',
            {k: v['decision'] for k, v in feature_support.items()},
        )
        LOGGER.info(
            'P2_SEVERE_AWARE_CANDIDATES=%s', [spec.candidate_id for spec in specs]
        )
        failing_step = 'fit_candidates'
        metrics_rows: list[dict[str, Any]] = []
        candidate_summaries: list[dict[str, Any]] = []
        image_predictions: dict[str, pd.DataFrame] = {}
        subject_predictions: dict[str, pd.DataFrame] = {}
        for spec in specs:
            predictions, messages = _fit_candidate_oof(feature_df, spec, n_splits)
            if not np.isfinite(
                predictions['predicted_stage_index'].to_numpy(dtype=float)
            ).all():
                raise SevereAwareEstimatorError(
                    f'Nonfinite selected predictions for {spec.candidate_id}'
                )
            if spec.target_level == 'image':
                image_predictions[spec.candidate_id] = predictions
            else:
                subject_predictions[spec.candidate_id] = predictions
            metrics_rows.append(
                _metric_row(
                    predictions,
                    spec,
                    split_label='validation_subject_heldout',
                    split_description='Grouped out-of-fold validation by subject_id',
                    eligible_for_model_selection=True,
                    warning_messages=messages,
                )
            )
            metrics_rows.append(
                _metric_row(
                    predictions.iloc[0:0],
                    spec,
                    split_label='testing_not_available_current_data_sensitivity',
                    split_description='No predeclared independent held-out test partition exists',
                    eligible_for_model_selection=False,
                    warning_messages=[],
                )
            )
            apparent = predictions.copy()
            metrics_rows.append(
                _metric_row(
                    apparent,
                    spec,
                    split_label='training_apparent',
                    split_description='Same-row apparent diagnostic placeholder; ineligible for model selection',
                    eligible_for_model_selection=False,
                    warning_messages=messages,
                )
            )
            candidate_summaries.append(
                {
                    'candidate_id': spec.candidate_id,
                    'target_level': spec.target_level,
                    'feature_family': spec.feature_family,
                    'model_family': spec.model_family,
                    'feature_count': len(spec.feature_columns),
                    'warning_messages': messages,
                }
            )
        metrics = pd.DataFrame(metrics_rows)
        validation = metrics[
            (metrics['split_label'] == 'validation_subject_heldout')
            & (metrics['finite_output_status'] == 'finite')
        ].copy()
        selectable_images = validation[validation['target_level'] == 'image'].copy()
        if selectable_images.empty:
            raise SevereAwareEstimatorError(
                'No finite image-level severe-aware candidates.'
            )
        selectable_images['_recall_rank'] = selectable_images['severe_recall'].fillna(
            -1.0
        )
        best_image_id = str(
            selectable_images.sort_values(
                [
                    'severe_false_negative_count',
                    'stage_index_mae',
                    'grade_scale_mae',
                    'candidate_id',
                ],
                ascending=[True, True, True, True],
            ).iloc[0]['candidate_id']
        )
        selected_image_predictions = image_predictions[best_image_id]
        best_subject_id = ''
        if subject_predictions:
            subject_validation = validation[validation['target_level'] == 'subject']
            if not subject_validation.empty:
                best_subject_id = str(
                    subject_validation.sort_values(
                        [
                            'severe_false_negative_count',
                            'stage_index_mae',
                            'candidate_id',
                        ]
                    ).iloc[0]['candidate_id']
                )
        selected_subject_predictions = subject_predictions.get(
            best_subject_id, pd.DataFrame()
        )
        threshold_metrics = _write_severe_threshold_metrics(
            selected_image_predictions, paths['severe_threshold_metrics']
        )
        source_sensitivity = _source_severe_sensitivity(
            selected_image_predictions, support
        )
        _save_json(source_sensitivity, paths['source_severe_sensitivity'])
        reliability = _write_reliability_labels(paths['reliability_labels'])
        primary_threshold_support = support['score_gte_2']
        score3_support = support['score_gte_3']
        hard_blockers: list[str] = []
        if not np.isfinite(
            selected_image_predictions['predicted_stage_index'].to_numpy(dtype=float)
        ).all():
            hard_blockers.append('nonfinite_selected_predictions')
        scope_limiters = list(source_handoff.get('scope_limiters', []))
        if score3_support['support_status'] != 'estimable':
            scope_limiters.append('underpowered_score_gte_3_threshold')
        if primary_threshold_support['support_status'] != 'estimable':
            scope_limiters.append('source_confounded_severe_support')
        selected_metrics = (
            validation[validation['candidate_id'] == best_image_id].iloc[0].to_dict()
        )
        if float(selected_metrics.get('severe_false_negative_count', 0) or 0) > 0:
            scope_limiters.append('score_2_3_underprediction')
        if (
            selected_image_predictions['reliability_label']
            .str.contains('broad_ordinal_prediction_set')
            .any()
        ):
            scope_limiters.append('broad_ordinal_prediction_sets')
        if int(selected_metrics.get('warning_count', 0) or 0) > 0:
            scope_limiters.append('nonfatal_numerical_warnings')
        reportable_scopes = {
            'severe_risk': bool(
                not hard_blockers
                and primary_threshold_support['support_status']
                in {'estimable', 'estimable_source_sensitive'}
            ),
            'ordinal_prediction_set': bool(
                not hard_blockers and best_image_id.startswith('ordinal_')
            ),
            'scalar_burden': False,
            'subject_level': bool(not hard_blockers and best_subject_id),
            'aggregate_current_data': bool(not hard_blockers),
        }
        selected_output_type = (
            'severe-risk label'
            if reportable_scopes['severe_risk']
            else 'limited/non-reportable evidence'
        )
        verdict = {
            'overall_status': 'blocked'
            if hard_blockers
            else 'limited_current_data_severe_aware_estimator',
            'selected_image_candidate': best_image_id,
            'selected_subject_candidate': best_subject_id,
            'selected_severe_threshold': PRIMARY_SEVERE_THRESHOLD,
            'selected_ordinal_candidate': best_image_id
            if best_image_id.startswith('ordinal_')
            else '',
            'selected_output_type': selected_output_type,
            'hard_blockers': hard_blockers,
            'scope_limiters': list(dict.fromkeys(scope_limiters)),
            'reportable_scopes': reportable_scopes,
            'non_reportable_scopes': [
                key for key, value in reportable_scopes.items() if not value
            ],
            'testing_status': 'testing_not_available_current_data_sensitivity',
            'readme_snippet_eligible': False,
            'source_aware_handoff': source_handoff,
            'threshold_support_decisions': support,
            'reliability_label_definitions': reliability,
            'next_action': 'review_severe_false_negative_evidence_before_upstream_segmentation_or_annotation_change',
            'claim_boundary': 'predictive grade-equivalent, severe-risk, or ordinal-set evidence for current scored MR TIFF/ROI data; not tissue percent, closed-capillary percent, causal evidence, or external validation',
        }
        selected_image_predictions.to_csv(paths['image_predictions'], index=False)
        if selected_subject_predictions.empty:
            pd.DataFrame().to_csv(paths['subject_predictions'], index=False)
        else:
            selected_subject_predictions.to_csv(
                paths['subject_predictions'], index=False
            )
        metrics.to_csv(paths['metrics_by_split_csv'], index=False)
        _save_json(metrics.to_dict(orient='records'), paths['metrics_by_split_json'])
        validation.to_csv(paths['candidate_metrics'], index=False)
        _save_json(
            {
                'candidate_count': len(candidate_summaries),
                'allowed_candidate_ids': list(ALLOWED_CANDIDATE_IDS),
                'selected_image_candidate': best_image_id,
                'selected_subject_candidate': best_subject_id,
                'candidates': candidate_summaries,
            },
            paths['candidate_summary'],
        )
        _write_evidence_review(
            paths['evidence_review'], selected_image_predictions, verdict
        )
        _save_json(verdict, paths['verdict_json'])
        _write_verdict_md(paths['verdict_md'], verdict)
        _write_figures(paths, selected_image_predictions, threshold_metrics)
        manifest = _artifact_manifest(paths['root'])
        _save_json(manifest, paths['artifact_manifest'])
        _write_index(paths['index'], verdict, manifest)
        manifest = _artifact_manifest(paths['root'])
        _save_json(manifest, paths['artifact_manifest'])
        _write_index(paths['index'], verdict, manifest)
        if change_dir is not None:
            _write_audit_results(
                Path(change_dir),
                {
                    'threshold_support': support,
                    'source_aware_handoff': source_handoff,
                    'feature_support': feature_support,
                },
            )
        LOGGER.info('P2_SEVERE_AWARE_HARD_BLOCKERS=%s', hard_blockers)
        LOGGER.info('P2_SEVERE_AWARE_SCOPE_LIMITERS=%s', verdict['scope_limiters'])
        LOGGER.info('P2_SEVERE_AWARE_VERDICT_PATH=%s', paths['verdict_json'])
        LOGGER.info(
            'P2_SEVERE_AWARE_ARTIFACT_MANIFEST_PATH=%s', paths['artifact_manifest']
        )
        LOGGER.info(
            'P2_SEVERE_AWARE_EVENT=completed elapsed_seconds=%.3f', time.time() - start
        )
        return {
            'severe_aware_index': paths['index'],
            'severe_aware_estimator_verdict': paths['verdict_json'],
            'severe_aware_estimator_verdict_md': paths['verdict_md'],
            'severe_aware_metrics_by_split': paths['metrics_by_split_csv'],
            'severe_aware_metrics_by_split_json': paths['metrics_by_split_json'],
            'severe_aware_severe_threshold_metrics': paths['severe_threshold_metrics'],
            'severe_aware_artifact_manifest': paths['artifact_manifest'],
            'severe_aware_image_predictions': paths['image_predictions'],
            'severe_aware_subject_predictions': paths['subject_predictions'],
            'severe_aware_severe_separability_audit': paths[
                'severe_separability_audit'
            ],
            'severe_aware_threshold_support': paths['threshold_support'],
            'severe_aware_source_severe_sensitivity': paths[
                'source_severe_sensitivity'
            ],
            'severe_aware_reliability_labels': paths['reliability_labels'],
            'severe_aware_evidence_review': paths['evidence_review'],
            'severe_aware_candidate_metrics': paths['candidate_metrics'],
            'severe_aware_candidate_summary': paths['candidate_summary'],
            **{
                f'severe_aware_figure_{Path(name).stem}': paths['figures'] / name
                for name in SUMMARY_FIGURES
            },
        }
    except Exception as exc:
        LOGGER.exception(
            'P2_SEVERE_AWARE_EVENT=failed failing_step=%s input_root=%s output_root=%s exception=%s',
            failing_step,
            burden_output_dir,
            paths['root'],
            exc,
        )
        raise
