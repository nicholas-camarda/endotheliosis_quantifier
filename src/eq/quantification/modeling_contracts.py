"""Shared contracts for quantification model selection and artifact writing."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

GROUPED_DEVELOPMENT_METRIC_LABEL = 'grouped_out_of_fold_development_estimate'


@dataclass(frozen=True)
class WarningCaptureResult:
    """Warnings captured during a fit/predict operation."""

    warning_status: str
    warning_count: int
    warning_messages: list[dict[str, str]]


@dataclass(frozen=True)
class EstimabilityResult:
    """Bounded target-support result for a candidate or grouped fold."""

    estimable: bool
    blocker: str
    class_counts: dict[str, int]


@dataclass(frozen=True)
class SourceSupportResult:
    """Source/cohort support diagnostics for source-sensitive model evidence."""

    status: str
    source_column: str
    source_counts: dict[str, int]
    target_source_counts: dict[str, dict[str, int]]
    hard_blockers: list[str]
    diagnostics: list[dict[str, Any]]


def target_estimability(
    target: Sequence[Any] | np.ndarray | pd.Series,
    *,
    min_classes: int = 2,
    min_rows_per_class: int = 1,
    blocker: str = 'target_not_estimable',
) -> EstimabilityResult:
    """Check whether a target vector has enough class support for sklearn fitting."""
    series = pd.Series(target).dropna().astype(str)
    counts = {str(key): int(value) for key, value in series.value_counts().sort_index().items()}
    estimable = (
        len(series) > 0
        and len(counts) >= int(min_classes)
        and all(value >= int(min_rows_per_class) for value in counts.values())
    )
    return EstimabilityResult(estimable=estimable, blocker='' if estimable else blocker, class_counts=counts)


def source_stratified_target_support(
    frame: pd.DataFrame,
    target: Sequence[Any] | np.ndarray | pd.Series,
    *,
    source_columns: Sequence[str] = ('cohort_id', 'source_cohort', 'source_id'),
    min_sources: int = 2,
    min_sources_per_target_class: int = 2,
) -> SourceSupportResult:
    """Diagnose whether target support is separable from cohort/source identity."""
    source_column = next((column for column in source_columns if column in frame.columns), '')
    if not source_column:
        return SourceSupportResult(
            status='source_column_missing',
            source_column='',
            source_counts={},
            target_source_counts={},
            hard_blockers=[],
            diagnostics=[
                hard_blocker_payload(
                    'source_column_missing',
                    scope='source_stratified_support',
                    details={'checked_columns': list(source_columns)},
                )
            ],
        )
    work = pd.DataFrame(
        {
            'target': pd.Series(target).astype(str).reset_index(drop=True),
            'source': frame[source_column].astype(str).reset_index(drop=True),
        }
    ).dropna()
    source_counts = {
        str(key): int(value) for key, value in work['source'].value_counts().sort_index().items()
    }
    target_source_counts: dict[str, dict[str, int]] = {}
    diagnostics: list[dict[str, Any]] = []
    hard_blockers: list[str] = []
    if len(source_counts) < int(min_sources):
        hard_blockers.append('single_source_candidate_evidence')
        diagnostics.append(
            hard_blocker_payload(
                'single_source_candidate_evidence',
                scope='source_stratified_support',
                details={'source_column': source_column, 'source_counts': source_counts},
            )
        )
    for target_value, group in work.groupby('target'):
        counts = {
            str(key): int(value)
            for key, value in group['source'].value_counts().sort_index().items()
        }
        target_source_counts[str(target_value)] = counts
        if len(counts) < int(min_sources_per_target_class):
            diagnostics.append(
                hard_blocker_payload(
                    'target_class_source_concentrated',
                    scope='source_stratified_support',
                    details={
                        'source_column': source_column,
                        'target_class': str(target_value),
                        'source_counts': counts,
                    },
                )
            )
    status = (
        'hard_blocked'
        if hard_blockers
        else 'source_concentration_diagnosed'
        if diagnostics
        else 'source_stratified_support_ok'
    )
    return SourceSupportResult(
        status=status,
        source_column=source_column,
        source_counts=source_counts,
        target_source_counts=target_source_counts,
        hard_blockers=hard_blockers,
        diagnostics=diagnostics,
    )


def hard_blocker_payload(
    blocker: str,
    *,
    scope: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable hard-blocker payload for diagnostics and verdicts."""
    return {
        'blocker': blocker,
        'scope': scope,
        'severity': 'hard_blocker',
        'details': details or {},
    }


def insufficient_data_verdict(
    blockers: Sequence[str],
    *,
    claim_boundary: str = 'current data insufficient; no external validation',
) -> dict[str, Any]:
    """Return the shared bounded verdict for unestimable quantification runs."""
    return {
        'overall_status': 'current_data_insufficient',
        'selected_candidate_id': '',
        'selected_family_id': '',
        'selected_output_type': '',
        'quantification_gate_passed': False,
        'severe_safety_gate_passed': False,
        'mr_tiff_deployment_gate_passed': False,
        'readme_facing_deployment_allowed': False,
        'hard_blockers': list(dict.fromkeys(blockers)),
        'claim_boundary': claim_boundary,
    }


def empty_candidate_metrics_frame() -> pd.DataFrame:
    """Return the canonical empty candidate-metrics table."""
    return pd.DataFrame(
        columns=[
            'candidate_id',
            'family_id',
            'target_kind',
            'feature_family',
            'model_kind',
            'metric_label',
            'finite_output',
            'hard_blockers',
            'fold_metrics',
        ]
    )


def empty_candidate_predictions_frame(identity_columns: Sequence[str]) -> pd.DataFrame:
    """Return the canonical empty candidate-predictions table."""
    return pd.DataFrame(columns=[*identity_columns, 'candidate_id', 'fold'])


def save_supported_sklearn_model(model: Any, output_path: Path) -> Path:
    """Write a supported sklearn artifact with serialization matching its suffix."""
    output_path = Path(output_path)
    if output_path.suffix != '.joblib':
        raise ValueError(
            f"Supported sklearn artifacts must use a .joblib filename, got: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def save_json(data: Any, output_path: Path) -> Path:
    """Write JSON with a stable directory-creation contract."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return output_path


def to_finite_numeric_matrix(
    frame: pd.DataFrame, columns: Sequence[str], *, finite_bound: float | None = None
) -> np.ndarray:
    """Return a finite float matrix for a declared feature column set."""
    x = frame.loc[:, list(columns)].apply(pd.to_numeric, errors='coerce')
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matrix = x.to_numpy(dtype=np.float64)
    if finite_bound is not None:
        matrix = np.clip(matrix, -float(finite_bound), float(finite_bound))
    return matrix


def choose_recall_threshold(
    probabilities: np.ndarray, target: np.ndarray, target_recall: float
) -> float:
    """Choose the highest-precision threshold that satisfies a recall target."""
    thresholds = np.unique(np.clip(probabilities, 0.0, 1.0))
    thresholds = np.unique(np.concatenate(([0.0], thresholds, [1.0])))
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        rows.append(
            (
                float(threshold),
                float(recall_score(target, predicted, zero_division=0)),
                float(precision_score(target, predicted, zero_division=0)),
            )
        )
    passing = [row for row in rows if row[1] >= target_recall]
    if passing:
        return max(passing, key=lambda row: (row[2], row[0]))[0]
    return max(rows, key=lambda row: (row[1], row[2], row[0]))[0]


def capture_fit_warnings(
    operation: Callable[[], Any],
) -> tuple[Any, WarningCaptureResult]:
    """Run an operation while recording warnings for artifact diagnostics."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = operation()
    records = [
        {'category': warning.category.__name__, 'message': str(warning.message)}
        for warning in caught
    ]
    return result, WarningCaptureResult(
        warning_status='warnings_recorded' if records else 'no_warnings_recorded',
        warning_count=len(records),
        warning_messages=records[:10],
    )


def build_artifact_manifest(root: Path, *, role: str, consumer: str) -> dict[str, Any]:
    """Build a manifest for all files under a runtime artifact root."""
    root = Path(root)
    records = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            records.append(
                {
                    'relative_path': str(path.relative_to(root)),
                    'role': role,
                    'consumer': consumer,
                    'required': True,
                    'reportability': 'current_data_internal',
                    'exists': True,
                }
            )
    return {
        'root': str(root),
        'manifest_complete': True,
        'artifact_count': len(records),
        'artifacts': records,
    }
