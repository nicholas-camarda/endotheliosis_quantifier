"""Canonical ordinal modeling on frozen encoder embeddings."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

NUMERICAL_INSTABILITY_PATTERNS = ('overflow', 'divide by zero', 'invalid value')


@dataclass
class _ConstantOrdinalModel:
    class_index: int


def _matching_warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for warning_message in caught:
        text = str(warning_message.message)
        lower = text.lower()
        if any(pattern in lower for pattern in NUMERICAL_INSTABILITY_PATTERNS):
            messages.append(text)
    return list(dict.fromkeys(messages))


def _class_count_map(y: np.ndarray, classes: np.ndarray) -> dict[str, int]:
    return {
        str(int(class_index)): int(np.sum(y == class_index)) for class_index in classes
    }


def _threshold_positive_count_map(y: np.ndarray, classes: np.ndarray) -> dict[str, int]:
    return {
        f'>{int(class_index)}': int(np.sum(y > class_index))
        for class_index in classes[:-1]
    }


def build_grouped_ordinal_cohort_profile(
    y: np.ndarray,
    groups: np.ndarray,
    embedding_dim: int,
    *,
    classes: np.ndarray,
    score_values: np.ndarray | None = None,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    profile: dict[str, Any] = {
        'n_examples': int(len(y)),
        'embedding_dim': int(embedding_dim),
        'n_subject_groups': int(len(unique_groups)),
        'class_counts': _class_count_map(y, classes),
        'threshold_positive_counts': _threshold_positive_count_map(y, classes),
        'subject_group_counts': {
            str(group): int(np.sum(groups == group)) for group in unique_groups
        },
    }
    if score_values is not None:
        profile['score_values'] = [float(score) for score in score_values.tolist()]
        profile['score_value_counts'] = {
            str(float(score_values[int(class_index)])): int(np.sum(y == class_index))
            for class_index in classes
        }
    return profile


class CanonicalOrdinalClassifier:
    """Strongly regularized penalized multiclass logistic baseline."""

    def __init__(
        self,
        classes: np.ndarray | None = None,
        *,
        max_iter: int = 4000,
        random_state: int = 0,
        regularization_strength: float = 0.1,
        max_selected_features: int = 16,
    ) -> None:
        self._preset_classes = (
            np.array(classes, copy=True) if classes is not None else None
        )
        self.max_iter = max_iter
        self.random_state = random_state
        self.regularization_strength = regularization_strength
        self.max_selected_features = max_selected_features
        self.classes_: np.ndarray | None = None
        self.model_: LogisticRegression | _ConstantOrdinalModel | None = None
        self.selected_feature_indices_: np.ndarray | None = None
        self.fit_warning_messages_: list[str] = []
        self.predict_warning_messages_: list[str] = []
        self.training_class_counts_: dict[str, int] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'CanonicalOrdinalClassifier':
        self.classes_ = (
            np.array(self._preset_classes, copy=True)
            if self._preset_classes is not None
            else np.sort(np.unique(y))
        )
        self.fit_warning_messages_ = []
        self.predict_warning_messages_ = []
        self.training_class_counts_ = _class_count_map(y, self.classes_)
        self.selected_feature_indices_ = None
        observed_classes = np.unique(y)
        if len(observed_classes) == 1:
            self.model_ = _ConstantOrdinalModel(class_index=int(observed_classes[0]))
            return self

        transformed_x = x
        max_features = min(self.max_selected_features, x.shape[1])
        if max_features >= 1 and max_features < x.shape[1]:
            feature_variances = np.var(x, axis=0)
            self.selected_feature_indices_ = np.argsort(feature_variances)[::-1][
                :max_features
            ]
            transformed_x = x[:, self.selected_feature_indices_]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = LogisticRegression(
                penalty='l2',
                C=self.regularization_strength,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver='lbfgs',
                multi_class='multinomial',
            )
            model.fit(transformed_x, y)
        self.fit_warning_messages_ = _matching_warning_messages(caught)
        self.model_ = model
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.model_ is None:
            raise RuntimeError('Model must be fit before predict_proba.')

        if isinstance(self.model_, _ConstantOrdinalModel):
            probabilities = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
            class_position = int(
                np.where(self.classes_ == self.model_.class_index)[0][0]
            )
            probabilities[:, class_position] = 1.0
            return probabilities

        transformed_x = (
            x[:, self.selected_feature_indices_]
            if self.selected_feature_indices_ is not None
            else x
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fitted_probabilities = self.model_.predict_proba(transformed_x)
        self.predict_warning_messages_ = _matching_warning_messages(caught)

        probabilities = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
        for source_index, class_index in enumerate(self.model_.classes_):
            target_index = int(np.where(self.classes_ == class_index)[0][0])
            probabilities[:, target_index] = fitted_probabilities[:, source_index]
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probabilities / row_sums

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        class_indices = probabilities.argmax(axis=1)
        assert self.classes_ is not None
        return self.classes_[class_indices]

    @property
    def warning_messages_(self) -> list[str]:
        return list(
            dict.fromkeys(
                [*self.fit_warning_messages_, *self.predict_warning_messages_]
            )
        )

    def metadata(self) -> dict[str, Any]:
        if self.classes_ is None:
            raise RuntimeError('Model must be fit before metadata is available.')

        metadata: dict[str, Any] = {
            'canonical_module': 'eq.quantification.ordinal',
            'estimator_class': self.__class__.__name__,
            'classes': [int(class_index) for class_index in self.classes_.tolist()],
            'training_class_counts': self.training_class_counts_,
            'fit_warning_messages': list(self.fit_warning_messages_),
            'predict_warning_messages': list(self.predict_warning_messages_),
        }
        if isinstance(self.model_, _ConstantOrdinalModel):
            metadata['estimator_family'] = 'constant_single_class'
            metadata['constant_class_index'] = int(self.model_.class_index)
            return metadata

        metadata['estimator_family'] = 'penalized_multiclass_logistic'
        metadata['regularization'] = {
            'solver': 'lbfgs',
            'penalty': 'l2',
            'C': float(self.regularization_strength),
            'max_iter': int(self.max_iter),
            'multi_class': 'multinomial',
        }
        metadata['feature_reduction'] = (
            {
                'kind': 'top_variance_feature_selection',
                'n_components': int(len(self.selected_feature_indices_)),
            }
            if self.selected_feature_indices_ is not None
            else {'kind': 'none', 'n_components': None}
        )
        return metadata


def run_grouped_ordinal_experiment(
    embeddings_path: Path,
    embedding_metadata_path: Path,
    output_dir: Path,
    n_splits: int = 5,
) -> dict[str, Path]:
    """Train and evaluate the canonical ordinal model on frozen embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(embeddings_path)
    metadata = pd.read_csv(embedding_metadata_path)
    if len(metadata) != x.shape[0]:
        raise ValueError(
            'Embedding matrix row count does not match embedding metadata row count.'
        )

    score_values = np.array(
        sorted(metadata['score'].astype(float).unique().tolist()), dtype=float
    )
    score_to_index = {score: index for index, score in enumerate(score_values.tolist())}
    y = metadata['score'].map(score_to_index).to_numpy(dtype=int)
    groups = metadata['subject_prefix'].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            'At least two subject groups are required for grouped ordinal evaluation.'
        )

    splitter = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
    classes = np.arange(len(score_values), dtype=int)
    cohort_profile = build_grouped_ordinal_cohort_profile(
        y, groups, x.shape[1], classes=classes, score_values=score_values
    )
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    fold_warning_messages: list[dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        classifier = CanonicalOrdinalClassifier(classes=classes)
        classifier.fit(x_train, y_train)
        probs = classifier.predict_proba(x_test)
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
        fold_warning_messages.append(
            {
                'fold': int(fold_index),
                'messages': classifier.warning_messages_,
                'train_class_counts': _class_count_map(y_train, classes),
                'test_class_counts': _class_count_map(y_test, classes),
                'train_threshold_positive_counts': _threshold_positive_count_map(
                    y_train, classes
                ),
            }
        )
        fold_metrics.append(
            {
                'fold': fold_index,
                'n_test': int(len(test_idx)),
                'mae_predicted': float(mean_absolute_error(truth_score, pred_score)),
                'mae_expected': float(mean_absolute_error(truth_score, expected_score)),
                'quadratic_kappa': float(
                    cohen_kappa_score(y_test, pred_idx, weights='quadratic')
                ),
                'exact_accuracy': float(np.mean(pred_idx == y_test)),
                'within_half_point_accuracy': float(
                    np.mean(np.abs(pred_score - truth_score) <= 0.5)
                ),
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_path = output_dir / 'ordinal_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

    final_scaler = StandardScaler().fit(x)
    final_classifier = CanonicalOrdinalClassifier(classes=classes).fit(
        final_scaler.transform(x), y
    )
    combined_warning_messages = list(
        dict.fromkeys(
            [
                *[
                    message
                    for fold_entry in fold_warning_messages
                    for message in fold_entry['messages']
                ],
                *final_classifier.warning_messages_,
            ]
        )
    )
    full_target_class_support = all(
        int(count) > 0 for count in cohort_profile['class_counts'].values()
    )
    certification_blockers: list[str] = []
    if combined_warning_messages:
        certification_blockers.append('numerical_instability')
    if not full_target_class_support:
        certification_blockers.append('missing_target_class_support')
    metrics = {
        'score_values': score_values.tolist(),
        'fold_metrics': fold_metrics,
        'overall_mae_predicted': float(
            mean_absolute_error(
                predictions_df['true_score'], predictions_df['predicted_score']
            )
        ),
        'overall_mae_expected': float(
            mean_absolute_error(
                predictions_df['true_score'], predictions_df['expected_score']
            )
        ),
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
        'ordinal_model': final_classifier.metadata(),
        'cohort_profile': cohort_profile,
        'stability': {
            'warning_patterns': list(NUMERICAL_INSTABILITY_PATTERNS),
            'fold_warning_messages': fold_warning_messages,
            'final_model_warning_messages': final_classifier.warning_messages_,
            'zero_unresolved_warning_gate_passed': not combined_warning_messages,
            'full_target_class_support': full_target_class_support,
            'certification_status': (
                'supported' if not certification_blockers else 'incomplete'
            ),
            'certification_blockers': certification_blockers,
        },
    }
    metrics_path = output_dir / 'ordinal_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {'predictions': predictions_path, 'metrics': metrics_path}
