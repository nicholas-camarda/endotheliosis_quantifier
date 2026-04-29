"""Shared contracts for quantification model selection and artifact writing."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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
