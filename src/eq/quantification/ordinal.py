"""Ordinal modeling on frozen encoder embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class _ThresholdModel:
    is_constant: bool
    constant_probability: float
    model: LogisticRegression | None


class OrdinalThresholdClassifier:
    """A simple cumulative-threshold ordinal classifier built from binary logits."""

    def __init__(self, classes: np.ndarray | None = None, max_iter: int = 1000, random_state: int = 0):
        self._preset_classes = np.array(classes, copy=True) if classes is not None else None
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_: list[_ThresholdModel] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OrdinalThresholdClassifier':
        self.classes_ = np.array(self._preset_classes, copy=True) if self._preset_classes is not None else np.sort(np.unique(y))
        self.models_ = []
        for threshold in self.classes_[:-1]:
            binary_target = (y > threshold).astype(int)
            unique_values = np.unique(binary_target)
            if len(unique_values) == 1:
                self.models_.append(
                    _ThresholdModel(
                        is_constant=True,
                        constant_probability=float(unique_values[0]),
                        model=None,
                    )
                )
                continue
            model = LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            model.fit(X, binary_target)
            self.models_.append(
                _ThresholdModel(
                    is_constant=False,
                    constant_probability=0.0,
                    model=model,
                )
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError('Model must be fit before predict_proba.')

        gt_probs = []
        for threshold_model in self.models_:
            if threshold_model.is_constant:
                prob = np.full(X.shape[0], threshold_model.constant_probability, dtype=float)
            else:
                assert threshold_model.model is not None
                prob = threshold_model.model.predict_proba(X)[:, 1]
            gt_probs.append(prob)

        if not gt_probs:
            return np.ones((X.shape[0], 1), dtype=float)

        gt = np.column_stack(gt_probs)
        class_probs = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        class_probs[:, 0] = 1.0 - gt[:, 0]
        for index in range(1, len(self.classes_) - 1):
            class_probs[:, index] = gt[:, index - 1] - gt[:, index]
        class_probs[:, -1] = gt[:, -1]
        class_probs = np.clip(class_probs, 0.0, 1.0)
        row_sums = class_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return class_probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        class_indices = probs.argmax(axis=1)
        assert self.classes_ is not None
        return self.classes_[class_indices]


def run_grouped_ordinal_experiment(
    embeddings_path: Path,
    embedding_metadata_path: Path,
    output_dir: Path,
    n_splits: int = 5,
) -> dict[str, Path]:
    """Train and evaluate the first ordinal model on frozen embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(embeddings_path)
    metadata = pd.read_csv(embedding_metadata_path)
    if len(metadata) != X.shape[0]:
        raise ValueError('Embedding matrix row count does not match embedding metadata row count.')

    score_values = np.array(sorted(metadata['score'].astype(float).unique().tolist()), dtype=float)
    score_to_index = {score: index for index, score in enumerate(score_values.tolist())}
    y = metadata['score'].map(score_to_index).to_numpy(dtype=int)
    groups = metadata['subject_prefix'].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError('At least two subject groups are required for grouped ordinal evaluation.')

    splitter = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        classifier = OrdinalThresholdClassifier(classes=np.arange(len(score_values)))
        classifier.fit(X_train, y_train)
        probs = classifier.predict_proba(X_test)
        pred_idx = probs.argmax(axis=1)
        pred_score = score_values[pred_idx]
        expected_score = probs @ score_values
        truth_score = score_values[y_test]
        fold_metadata = metadata.iloc[test_idx].copy()
        fold_metadata['fold'] = fold_index
        fold_metadata['predicted_score'] = pred_score
        fold_metadata['expected_score'] = expected_score
        fold_metadata['true_score'] = truth_score
        predictions.append(fold_metadata)
        fold_metrics.append(
            {
                'fold': fold_index,
                'n_test': int(len(test_idx)),
                'mae_predicted': float(mean_absolute_error(truth_score, pred_score)),
                'mae_expected': float(mean_absolute_error(truth_score, expected_score)),
                'quadratic_kappa': float(cohen_kappa_score(y_test, pred_idx, weights='quadratic')),
                'exact_accuracy': float(np.mean(pred_idx == y_test)),
                'within_half_point_accuracy': float(np.mean(np.abs(pred_score - truth_score) <= 0.5)),
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_path = output_dir / 'ordinal_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

    metrics = {
        'score_values': score_values.tolist(),
        'fold_metrics': fold_metrics,
        'overall_mae_predicted': float(mean_absolute_error(predictions_df['true_score'], predictions_df['predicted_score'])),
        'overall_mae_expected': float(mean_absolute_error(predictions_df['true_score'], predictions_df['expected_score'])),
        'overall_quadratic_kappa': float(
            cohen_kappa_score(
                predictions_df['true_score'].map(score_to_index),
                predictions_df['predicted_score'].map(score_to_index),
                weights='quadratic',
            )
        ),
        'confusion_matrix': confusion_matrix(
            predictions_df['true_score'].map(score_to_index),
            predictions_df['predicted_score'].map(score_to_index),
        ).tolist(),
    }
    metrics_path = output_dir / 'ordinal_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {
        'predictions': predictions_path,
        'metrics': metrics_path,
    }
