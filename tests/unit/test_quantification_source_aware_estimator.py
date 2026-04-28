import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import eq.quantification.source_aware_estimator as source_aware_module
from eq.quantification.source_aware_estimator import (
    REQUIRED_RELATIVE_ARTIFACTS,
    SUMMARY_FIGURES,
    SourceAwareEstimatorError,
    evaluate_source_aware_endotheliosis_estimator,
)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _embedding_df(tmp_path: Path) -> pd.DataFrame:
    rows = []
    scores = [0, 0.5, 1, 1.5, 2, 3]
    for subject_index in range(8):
        if subject_index == 7:
            cohort_id = ''
        else:
            cohort_id = 'cohort_a' if subject_index < 4 else 'cohort_b'
        for image_index in range(3):
            score = float(scores[(subject_index + image_index) % len(scores)])
            image_path = tmp_path / 'images' / f's{subject_index}_{image_index}.png'
            mask_path = tmp_path / 'masks' / f's{subject_index}_{image_index}.png'
            _write_image(image_path, 40 + subject_index * 10 + image_index)
            _write_image(mask_path, 255)
            rows.append(
                {
                    'subject_id': f'{cohort_id or "unknown"}__s{subject_index}',
                    'sample_id': f'{cohort_id or "unknown"}__s{subject_index}__sample',
                    'image_id': f'{cohort_id or "unknown"}__s{subject_index}__image{image_index}',
                    'subject_image_id': f'{cohort_id or "unknown"}__{subject_index}_{image_index}',
                    'cohort_id': cohort_id,
                    'score': score,
                    'roi_status': 'ok',
                    'roi_image_path': str(image_path),
                    'roi_mask_path': str(mask_path),
                    'raw_image_path': str(image_path),
                    'raw_mask_path': str(mask_path),
                    'roi_area': 100 + subject_index,
                    'roi_fill_fraction': 0.5 + image_index * 0.01,
                    'roi_mean_intensity': 80 + subject_index,
                    'roi_openness_score': 0.1 * image_index,
                    'roi_component_count': 1,
                    'roi_union_bbox_width': 16,
                    'roi_union_bbox_height': 16,
                    'roi_largest_component_area_fraction': 1.0,
                    'embedding_0000': float(subject_index),
                    'embedding_0001': float(image_index),
                    'embedding_0002': score,
                    'embedding_0003': float(subject_index % 2),
                }
            )
    return pd.DataFrame(rows)


def test_source_aware_estimator_writes_contained_manifested_artifacts(tmp_path):
    artifacts = evaluate_source_aware_endotheliosis_estimator(
        _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
    )

    for key in [
        'source_aware_index',
        'source_aware_estimator_verdict',
        'source_aware_estimator_verdict_md',
        'source_aware_metrics_by_split',
        'source_aware_metrics_by_split_json',
        'source_aware_artifact_manifest',
        'source_aware_image_predictions',
        'source_aware_subject_predictions',
        'source_aware_upstream_roi_adequacy',
        'source_aware_source_sensitivity',
        'source_aware_reliability_labels',
        'source_aware_evidence_review',
        'source_aware_candidate_metrics',
        'source_aware_candidate_summary',
    ]:
        assert artifacts[key].exists(), key

    root = tmp_path / 'burden_model' / 'source_aware_estimator'
    manifest = json.loads(
        artifacts['source_aware_artifact_manifest'].read_text(encoding='utf-8')
    )
    assert manifest['manifest_complete'] is True
    assert sorted(item['relative_path'] for item in manifest['artifacts']) == sorted(
        REQUIRED_RELATIVE_ARTIFACTS
    )
    for figure_name in SUMMARY_FIGURES:
        assert (root / 'summary' / 'figures' / figure_name).exists()

    actual_files = sorted(
        str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()
    )
    assert actual_files == sorted(REQUIRED_RELATIVE_ARTIFACTS)
    assert not (
        tmp_path / 'burden_model' / 'source_aware_estimator_summary.json'
    ).exists()


def test_source_aware_estimator_metrics_labels_and_grouping(tmp_path):
    artifacts = evaluate_source_aware_endotheliosis_estimator(
        _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
    )

    metrics = pd.read_csv(artifacts['source_aware_metrics_by_split'])
    assert {
        'training_apparent',
        'validation_subject_heldout',
        'testing_not_available_current_data_sensitivity',
    }.issubset(set(metrics['split_label']))
    assert 'testing_explicit_heldout' not in set(metrics['split_label'])
    assert not metrics[metrics['split_label'] == 'training_apparent'][
        'eligible_for_model_selection'
    ].any()
    assert metrics[metrics['split_label'] == 'validation_subject_heldout'][
        'eligible_for_model_selection'
    ].any()

    predictions = pd.read_csv(artifacts['source_aware_image_predictions'])
    assert predictions.groupby('subject_id')['fold'].nunique().max() == 1
    assert predictions['reliability_label'].str.contains('unknown_source').any()
    assert predictions['prediction_source'].eq('validation_subject_heldout').all()

    reliability = json.loads(
        artifacts['source_aware_reliability_labels'].read_text(encoding='utf-8')
    )
    assert 'unknown_source' in reliability['labels']
    assert 'transitional_score_region' in reliability['labels']
    assert 'score_specific_undercoverage' in reliability['scope_limiters']
    assert 'unsupported_scores' in reliability['hard_blockers']


def test_source_aware_estimator_verdict_and_upstream_adequacy(tmp_path):
    artifacts = evaluate_source_aware_endotheliosis_estimator(
        _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
    )

    verdict = json.loads(
        artifacts['source_aware_estimator_verdict'].read_text(encoding='utf-8')
    )
    assert verdict['testing_status'] == 'testing_not_available_current_data_sensitivity'
    assert {'image_level', 'subject_level', 'aggregate_current_data'}.issubset(
        verdict['reportable_scopes']
    )
    assert verdict['readme_snippet_eligible'] is False
    assert (
        verdict['claim_boundary']
        == 'predictive grade-equivalent endotheliosis burden for current scored MR TIFF/ROI data; not tissue percent, closed-capillary percent, causal evidence, or external validation'
    )

    upstream = json.loads(
        artifacts['source_aware_upstream_roi_adequacy'].read_text(encoding='utf-8')
    )
    assert upstream['total_input_rows'] == 24
    assert upstream['usable_roi_rows'] == 24
    assert upstream['failed_roi_rows'] == 0
    assert upstream['roi_status_counts'] == {'ok': 24}


def test_source_aware_estimator_rejects_unsupported_scores(tmp_path):
    df = _embedding_df(tmp_path)
    df.loc[0, 'score'] = 2.5
    with pytest.raises(SourceAwareEstimatorError, match='Unsupported score values'):
        evaluate_source_aware_endotheliosis_estimator(df, tmp_path / 'burden_model')


def test_source_aware_estimator_rejects_nonfinite_candidate_predictions(
    tmp_path, monkeypatch
):
    def nonfinite_fit(*args, **kwargs):
        test_df = args[1]
        return np.full(len(test_df), np.nan), []

    monkeypatch.setattr(source_aware_module, '_fit_predict_ridge', nonfinite_fit)
    with pytest.raises(SourceAwareEstimatorError, match='No finite image-level'):
        evaluate_source_aware_endotheliosis_estimator(
            _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
        )


def test_source_aware_estimator_rejects_subject_leakage(tmp_path, monkeypatch):
    class LeakyGroupKFold:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def split(self, df, groups=None):
            yield np.arange(0, len(df) - 1), np.arange(1, len(df))

    monkeypatch.setattr(source_aware_module, 'GroupKFold', LeakyGroupKFold)
    with pytest.raises(
        SourceAwareEstimatorError, match='subject_id_validation_leakage'
    ):
        evaluate_source_aware_endotheliosis_estimator(
            _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
        )
