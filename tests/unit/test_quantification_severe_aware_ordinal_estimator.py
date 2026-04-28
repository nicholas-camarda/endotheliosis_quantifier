import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import eq.quantification.severe_aware_ordinal_estimator as severe_module
from eq.quantification.severe_aware_ordinal_estimator import (
    REQUIRED_RELATIVE_ARTIFACTS,
    SUMMARY_FIGURES,
    SevereAwareEstimatorError,
    compute_threshold_support,
    evaluate_severe_aware_ordinal_endotheliosis_estimator,
)
from eq.utils.execution_logging import ExecutionLogContext, execution_log_context


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _embedding_df(tmp_path: Path) -> pd.DataFrame:
    rows = []
    scores = [0, 0.5, 1, 1.5, 2, 3]
    for subject_index in range(12):
        cohort_id = 'cohort_a' if subject_index < 6 else 'cohort_b'
        for image_index in range(3):
            score = float(scores[(subject_index + image_index) % len(scores)])
            image_path = tmp_path / 'images' / f's{subject_index}_{image_index}.png'
            mask_path = tmp_path / 'masks' / f's{subject_index}_{image_index}.png'
            _write_image(image_path, 40 + subject_index * 10 + image_index)
            _write_image(mask_path, 255)
            rows.append(
                {
                    'subject_id': f'{cohort_id}__s{subject_index}',
                    'sample_id': f'{cohort_id}__s{subject_index}__sample',
                    'image_id': f'{cohort_id}__s{subject_index}__image{image_index}',
                    'subject_image_id': f'{cohort_id}__{subject_index}_{image_index}',
                    'cohort_id': cohort_id,
                    'score': score,
                    'roi_image_path': str(image_path),
                    'roi_mask_path': str(mask_path),
                    'raw_image_path': str(image_path),
                    'raw_mask_path': str(mask_path),
                    'roi_area': 100 + score * 20 + subject_index,
                    'roi_fill_fraction': 0.45 + score * 0.05,
                    'roi_mean_intensity': 80 + score * 5,
                    'roi_openness_score': 0.1 * image_index,
                    'roi_component_count': 1 + int(score >= 2),
                    'roi_union_bbox_width': 16,
                    'roi_union_bbox_height': 16,
                    'roi_largest_component_area_fraction': 1.0,
                    'embedding_0000': float(subject_index),
                    'embedding_0001': float(image_index),
                    'embedding_0002': score,
                }
            )
    return pd.DataFrame(rows)


def _write_source_aware_handoff(root: Path) -> None:
    summary = root / 'source_aware_estimator' / 'summary'
    predictions = root / 'source_aware_estimator' / 'predictions'
    summary.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    (summary / 'estimator_verdict.json').write_text(
        json.dumps(
            {
                'selected_image_candidate': 'pooled_roi_qc',
                'selected_subject_candidate': 'subject_source_adjusted_hybrid',
                'hard_blockers': [],
                'scope_limiters': ['wide_uncertainty'],
                'reportable_scopes': {'image_level': True},
                'testing_status': 'testing_not_available_current_data_sensitivity',
                'claim_boundary': 'predictive current-data evidence',
            }
        ),
        encoding='utf-8',
    )
    pd.DataFrame(
        {
            'subject_id': ['a', 'b'],
            'score': [2.0, 3.0],
            'observed_stage_index': [80.0, 100.0],
            'predicted_stage_index': [40.0, 35.0],
        }
    ).to_csv(predictions / 'image_predictions.csv', index=False)
    pd.DataFrame({'candidate_id': ['pooled_roi_qc']}).to_csv(
        summary / 'metrics_by_split.csv', index=False
    )


def test_threshold_support_marks_score_three_exploratory_when_single_source(tmp_path):
    df = _embedding_df(tmp_path)
    df.loc[df['score'] == 3.0, 'cohort_id'] = 'cohort_b'

    support = compute_threshold_support(df)

    assert support['score_gte_2']['positive_subjects'] >= 2
    assert support['score_gte_3']['support_status'] == 'exploratory'
    assert (
        support['score_gte_3']['reason']
        == 'positive_support_single_source_tail_stratum'
    )


def test_severe_aware_estimator_writes_contained_manifested_artifacts(tmp_path):
    burden_root = tmp_path / 'burden_model'
    _write_source_aware_handoff(burden_root)

    artifacts = evaluate_severe_aware_ordinal_endotheliosis_estimator(
        _embedding_df(tmp_path), burden_root, n_splits=3, change_dir=tmp_path
    )

    root = burden_root / 'severe_aware_ordinal_estimator'
    manifest = json.loads(artifacts['severe_aware_artifact_manifest'].read_text())
    assert manifest['manifest_complete'] is True
    assert sorted(item['relative_path'] for item in manifest['artifacts']) == sorted(
        REQUIRED_RELATIVE_ARTIFACTS
    )
    actual_files = sorted(
        str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()
    )
    assert actual_files == sorted(REQUIRED_RELATIVE_ARTIFACTS)
    for figure_name in SUMMARY_FIGURES:
        assert (root / 'summary' / 'figures' / figure_name).exists()
    assert not (burden_root / 'severe_aware_estimator_summary.json').exists()
    assert (tmp_path / 'audit-results.md').exists()


def test_metrics_labels_false_negatives_and_readme_eligibility(tmp_path):
    burden_root = tmp_path / 'burden_model'
    _write_source_aware_handoff(burden_root)
    artifacts = evaluate_severe_aware_ordinal_endotheliosis_estimator(
        _embedding_df(tmp_path), burden_root, n_splits=3
    )

    metrics = pd.read_csv(artifacts['severe_aware_metrics_by_split'])
    assert {
        'training_apparent',
        'validation_subject_heldout',
        'testing_not_available_current_data_sensitivity',
    }.issubset(set(metrics['split_label']))
    assert not metrics[metrics['split_label'] == 'training_apparent'][
        'eligible_for_model_selection'
    ].any()
    assert 'severe_false_negative_count' in metrics.columns

    predictions = pd.read_csv(artifacts['severe_aware_image_predictions'])
    assert predictions.groupby('subject_id')['fold'].nunique().max() == 1
    assert 'reliability_label' in predictions.columns
    assert (
        predictions[predictions['score'] >= 2.0]['reliability_label']
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )

    verdict = json.loads(artifacts['severe_aware_estimator_verdict'].read_text())
    assert verdict['readme_snippet_eligible'] is False
    assert verdict['testing_status'] == 'testing_not_available_current_data_sensitivity'
    assert 'external validation' in verdict['claim_boundary']


def test_false_negative_review_is_interactive_adjudication_surface(tmp_path):
    image_path = tmp_path / 'images' / 'severe.png'
    mask_path = tmp_path / 'masks' / 'severe.png'
    _write_image(image_path, 160)
    _write_image(mask_path, 255)
    review_path = tmp_path / 'review.html'
    severe_module._write_evidence_review(
        review_path,
        pd.DataFrame(
            [
                {
                    'subject_id': 'subject_1',
                    'sample_id': 'sample_1',
                    'image_id': 'image_1',
                    'subject_image_id': 'subject_1__image_1',
                    'cohort_id': 'cohort_b',
                    'score': 2.0,
                    'observed_stage_index': 80.0,
                    'predicted_stage_index': 20.0,
                    'severe_probability': 0.2,
                    'severe_predicted_label': False,
                    'ordinal_prediction_set': '0|0.5|1',
                    'reliability_label': 'severe_false_negative',
                    'roi_image_path': str(image_path),
                    'roi_mask_path': str(mask_path),
                    'raw_image_path': str(image_path),
                    'raw_mask_path': str(mask_path),
                }
            ]
        ),
        {'overall_status': 'limited', 'claim_boundary': 'predictive evidence'},
    )

    html = review_path.read_text(encoding='utf-8')

    assert 'Interactive Adjudication Queue' in html
    assert 'Grade appears correct' in html
    assert 'Valid grade, model missed it' in html
    assert 'localStorage' in html
    assert 'Export CSV' in html
    assert '<img src="file://' in html


def test_existing_false_negative_adjudications_are_summarized(tmp_path):
    burden_root = tmp_path / 'burden_model'
    _write_source_aware_handoff(burden_root)
    adjudication_path = (
        burden_root
        / 'severe_aware_ordinal_estimator'
        / 'evidence'
        / 'severe_false_negative_adjudications.json'
    )
    adjudication_path.parent.mkdir(parents=True, exist_ok=True)
    adjudication_path.write_text(
        json.dumps(
            [
                {
                    'subject_image_id': 'cohort_b__1_1',
                    'grade_adjudication': 'grade_correct',
                    'failure_source': 'valid_grade_model_miss',
                    'review_confidence': 'high',
                    'review_notes': '',
                },
                {
                    'subject_image_id': 'cohort_b__2_0',
                    'grade_adjudication': 'grade_too_high',
                    'failure_source': 'not_severe_after_review',
                    'review_confidence': 'medium',
                    'review_notes': '1.5',
                },
            ]
        ),
        encoding='utf-8',
    )

    artifacts = evaluate_severe_aware_ordinal_endotheliosis_estimator(
        _embedding_df(tmp_path), burden_root, n_splits=3
    )

    summary = json.loads(
        artifacts['severe_aware_adjudication_summary'].read_text(encoding='utf-8')
    )
    adjudications = pd.read_csv(artifacts['severe_aware_adjudications_csv'])
    verdict = json.loads(artifacts['severe_aware_estimator_verdict'].read_text())

    assert summary['reviewed_count'] == 2
    assert summary['adjudicated_still_severe_count'] == 1
    assert summary['adjudicated_not_severe_count'] == 1
    assert 'adjudicated_selected_threshold_metrics' in summary
    assert (
        'severe_false_negative_count'
        in summary['adjudicated_selected_threshold_metrics']
    )
    rerun_verdict = json.loads(
        artifacts['severe_aware_adjudicated_rerun_verdict'].read_text()
    )
    rerun_metrics = pd.read_csv(artifacts['severe_aware_adjudicated_rerun_metrics'])
    assert (
        rerun_verdict['overall_status']
        == 'adjudicated_current_data_severe_threshold_rerun'
    )
    assert not rerun_metrics.empty
    assert set(rerun_metrics['split_label']) == {
        'validation_subject_heldout_adjudicated'
    }
    assert adjudications['proposed_score'].dropna().tolist() == [1.5]
    assert (
        verdict['next_action']
        == 'rerun_or_interpret_p2_with_adjudicated_severe_false_negative_labels'
    )


def test_nonfinite_predictions_are_hard_failures(tmp_path, monkeypatch):
    def nonfinite_fit(*args, **kwargs):
        df = args[0]
        predictions = df[['subject_id', 'cohort_id', 'score']].copy()
        predictions['observed_stage_index'] = 0.0
        predictions['predicted_stage_index'] = np.nan
        predictions['severe_predicted_label'] = False
        return predictions, []

    monkeypatch.setattr(severe_module, '_fit_candidate_oof', nonfinite_fit)
    with pytest.raises(
        SevereAwareEstimatorError, match='Nonfinite selected predictions'
    ):
        evaluate_severe_aware_ordinal_endotheliosis_estimator(
            _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
        )


def test_subject_leakage_is_a_hard_failure(tmp_path, monkeypatch):
    class LeakyGroupKFold:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def split(self, x, y=None, groups=None):
            yield np.arange(0, len(x) - 1), np.arange(1, len(x))

    monkeypatch.setattr(severe_module, 'GroupKFold', LeakyGroupKFold)
    with pytest.raises(
        SevereAwareEstimatorError, match='subject_id_validation_leakage'
    ):
        evaluate_severe_aware_ordinal_endotheliosis_estimator(
            _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
        )


def test_logger_events_and_no_independent_log_root(tmp_path, caplog):
    logger = logging.getLogger('eq.quantification.severe_aware_ordinal_estimator')
    logger.handlers.clear()
    logger.propagate = True
    caplog.set_level(
        logging.INFO, logger='eq.quantification.severe_aware_ordinal_estimator'
    )
    logger.addHandler(caplog.handler)

    evaluate_severe_aware_ordinal_endotheliosis_estimator(
        _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
    )

    text = caplog.text
    assert 'P2_SEVERE_AWARE_EVENT=start' in text
    assert 'P2_SEVERE_AWARE_OUTPUT_ROOT=' in text
    assert 'P2_SEVERE_AWARE_THRESHOLD_SUPPORT=' in text
    assert 'P2_SEVERE_AWARE_CANDIDATES=' in text
    assert 'P2_SEVERE_AWARE_VERDICT_PATH=' in text
    assert 'P2_SEVERE_AWARE_EVENT=completed' in text
    assert not (Path.cwd() / 'logs').exists()
    assert not (tmp_path / 'logs').exists()


def test_existing_execution_log_context_captures_severe_aware_events(tmp_path):
    log_path = tmp_path / 'runtime' / 'logs' / 'run_config' / 'p2' / 'run.log'
    logging.getLogger('eq.quantification.severe_aware_ordinal_estimator').setLevel(
        logging.INFO
    )
    context = ExecutionLogContext(
        surface='run_config',
        run_id='p2',
        log_path=log_path,
        runtime_root=tmp_path / 'runtime',
        config_path=Path('configs/endotheliosis_quantification.yaml'),
    )

    with execution_log_context(context, logger_name='eq'):
        evaluate_severe_aware_ordinal_endotheliosis_estimator(
            _embedding_df(tmp_path), tmp_path / 'burden_model', n_splits=3
        )

    text = log_path.read_text(encoding='utf-8')
    assert 'P2_SEVERE_AWARE_EVENT=start' in text
    assert 'P2_SEVERE_AWARE_VERDICT_PATH=' in text
    assert 'P2_SEVERE_AWARE_EVENT=completed' in text
    assert not (
        tmp_path / 'burden_model' / 'severe_aware_ordinal_estimator' / 'logs'
    ).exists()
