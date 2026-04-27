import json

import numpy as np
import pandas as pd
import pytest

from eq.quantification.burden import (
    ALLOWED_SCORE_VALUES,
    BurdenModelError,
    burden_from_threshold_probabilities,
    evaluate_burden_index_table,
    monotonic_threshold_probabilities,
    score_to_stage_index,
    threshold_targets,
    validate_score_values,
)


def test_six_bin_score_validation_rejects_unsupported_2p5():
    validate_score_values([0, 0.5, 1, 1.5, 2, 3])

    with pytest.raises(BurdenModelError, match='2.5'):
        validate_score_values([0, 1, 2.5, 3])


def test_stage_index_and_threshold_targets_follow_six_bin_rubric():
    scores = np.array([0, 0.5, 1, 1.5, 2, 3], dtype=float)

    np.testing.assert_allclose(score_to_stage_index(scores), [0, 20, 40, 60, 80, 100])
    targets = threshold_targets(scores)

    assert targets.shape == (6, 5)
    np.testing.assert_array_equal(targets[0], [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(targets[-1], [1, 1, 1, 1, 1])


def test_monotonic_projection_and_burden_formula():
    raw = np.array([[0.2, 0.8, 0.4, 0.5, 0.1]], dtype=float)

    projected = monotonic_threshold_probabilities(raw)

    np.testing.assert_allclose(projected, [[0.2, 0.2, 0.2, 0.2, 0.1]])
    np.testing.assert_allclose(burden_from_threshold_probabilities(projected), [18.0])


def test_evaluate_burden_index_table_writes_expected_artifacts(tmp_path):
    rows = []
    scores = ALLOWED_SCORE_VALUES.tolist()
    for subject_index in range(6):
        for image_index in range(2):
            score = scores[(subject_index + image_index) % len(scores)]
            rows.append(
                {
                    'subject_image_id': f'M{subject_index}--2023-06-12_Image{image_index}',
                    'subject_prefix': f'M{subject_index}--2023-06-12',
                    'subject_id': f'cohort__2023_06_12__m{subject_index}',
                    'sample_id': f'cohort__2023_06_12__m{subject_index}__image{image_index}',
                    'image_id': f'cohort__m{subject_index}__2023_06_12__image{image_index}',
                    'cohort_id': 'cohort_a' if subject_index < 3 else 'cohort_b',
                    'lane_assignment': 'manual_mask_core',
                    'glomerulus_id': 1,
                    'score': score,
                    'embedding_0000': float(subject_index),
                    'embedding_0001': float(image_index),
                    'embedding_0002': float(score),
                }
            )
    artifacts = evaluate_burden_index_table(pd.DataFrame(rows), tmp_path / 'burden')

    for key in [
        'burden_predictions',
        'burden_metrics',
        'threshold_metrics',
        'threshold_support',
        'uncertainty_calibration',
        'grouping_audit',
        'nearest_examples',
        'group_summary_intervals',
        'final_model_predictions',
        'final_model_cohort_metrics',
        'final_model_group_summary_intervals',
        'signal_comparator_metrics',
        'subject_level_candidate_predictions',
        'precision_candidate_summary',
    ]:
        assert artifacts[key].exists()

    predictions = pd.read_csv(artifacts['burden_predictions'])
    assert 'animal_id' not in predictions.columns
    assert predictions['subject_id'].str.contains('2023_06_12').all()
    assert predictions.groupby('subject_id')['fold'].nunique().max() == 1
    assert (predictions['prob_score_gt_0'] >= predictions['prob_score_gt_0p5']).all()
    assert {
        'burden_interval_low_0_100',
        'burden_interval_high_0_100',
        'burden_interval_coverage',
        'burden_interval_method',
        'prediction_set_scores',
        'prediction_set_method',
        'prediction_source',
    }.issubset(predictions.columns)
    assert predictions['burden_interval_low_0_100'].between(0, 100).all()
    assert predictions['burden_interval_high_0_100'].between(0, 100).all()
    allowed_tokens = {f'{value:g}' for value in ALLOWED_SCORE_VALUES}
    for prediction_set in predictions['prediction_set_scores']:
        assert set(str(prediction_set).split('|')).issubset(allowed_tokens)

    support = pd.read_csv(artifacts['threshold_support'])
    assert {'positive_groups', 'negative_groups', 'support_status'}.issubset(
        support.columns
    )
    assert {'overall', 'cohort:cohort_a', 'cohort:cohort_b'}.issubset(
        set(support['stratum'])
    )

    group_summary = pd.read_csv(artifacts['group_summary_intervals'])
    overall = group_summary[group_summary['stratum'] == 'overall'].iloc[0]
    assert overall['resampling_unit'] == 'subject_id'
    assert overall['weighting_rule'] == 'equal_weight_per_subject'
    assert int(overall['n_clusters']) == predictions['subject_id'].nunique()

    nearest = pd.read_csv(artifacts['nearest_examples'])
    subject_by_image = predictions.set_index('subject_image_id')['subject_id'].to_dict()
    assert not nearest.empty
    assert {
        'neighbor_raw_image_path',
        'neighbor_raw_mask_path',
        'neighbor_roi_image_path',
    }.issubset(nearest.columns)
    for _, row in nearest.iterrows():
        assert (
            subject_by_image[row['subject_image_id']]
            != subject_by_image[row['neighbor_subject_image_id']]
        )

    metrics = json.loads(artifacts['burden_metrics'].read_text(encoding='utf-8'))
    assert metrics['threshold_probability_status']['finite'] is True
    assert metrics['fold_group_conformity_quantiles']
    assert metrics['fold_group_residual_quantiles']
    assert 'signal_comparator_screen' in metrics

    signal = pd.read_csv(artifacts['signal_comparator_metrics'])
    assert {
        'candidate_id',
        'target_level',
        'target_definition',
        'feature_set',
        'validation_grouping',
        'stage_index_mae',
        'candidate_status',
    }.issubset(signal.columns)
    assert {
        'image_embedding_only_ridge',
        'subject_global_mean_baseline',
        'subject_embedding_only_ridge',
    }.issubset(set(signal['candidate_id']))
    assert {'image', 'subject'}.issubset(set(signal['target_level']))

    subject_candidates = pd.read_csv(artifacts['subject_level_candidate_predictions'])
    assert {
        'subject_id',
        'cohort_id',
        'observed_subject_stage_index_mean',
        'predicted_subject_burden_0_100',
        'subject_stage_index_absolute_error',
        'candidate_id',
        'fold',
    }.issubset(subject_candidates.columns)
    assert (
        subject_candidates['subject_id'].nunique()
        == predictions['subject_id'].nunique()
    )

    precision_summary = json.loads(
        artifacts['precision_candidate_summary'].read_text(encoding='utf-8')
    )
    assert precision_summary['candidate_count'] == len(signal)
    assert precision_summary['current_primary_burden_model']['target_level'] == 'image'
    assert precision_summary['best_subject_level_candidate']

    uncertainty = json.loads(
        artifacts['uncertainty_calibration'].read_text(encoding='utf-8')
    )
    assert (
        uncertainty['prediction_set_method']
        == 'grouped_fold_row_aps_conformal_quantile'
    )
    assert uncertainty['grouping_column'] == 'subject_id'
    assert 'by_cohort' in uncertainty
    assert 'by_observed_score' in uncertainty

    final_predictions = pd.read_csv(artifacts['final_model_predictions'])
    assert len(final_predictions) == len(predictions)
    assert (
        final_predictions['prediction_source'].eq('final_model_full_cohort_fit').all()
    )
