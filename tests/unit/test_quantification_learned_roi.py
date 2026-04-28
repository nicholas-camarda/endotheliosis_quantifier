import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eq.quantification.learned_roi import (
    audit_learned_roi_providers,
    build_learned_roi_feature_table,
    evaluate_learned_roi_quantification,
)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _embedding_df(tmp_path: Path) -> pd.DataFrame:
    rows = []
    scores = [0, 0.5, 1, 1.5, 2, 3]
    for subject_index in range(8):
        cohort_id = 'cohort_a' if subject_index < 4 else 'cohort_b'
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


def test_provider_audit_marks_optional_providers_audit_only_or_unavailable(tmp_path):
    audit_path = tmp_path / 'provider_audit.json'
    audit = audit_learned_roi_providers(audit_path)

    assert audit_path.exists()
    providers = {entry['provider_id']: entry for entry in audit['providers']}
    assert providers['current_glomeruli_encoder']['status'] == 'available_fit_allowed'
    assert providers['simple_roi_qc']['status'] == 'available_fit_allowed'
    assert providers['torchvision_resnet18_imagenet']['fit_allowed'] is False
    assert providers['timm_dino_or_vit']['fit_allowed'] is False
    assert providers['torchvision_resnet18_imagenet']['status'] in {
        'available_audit_only',
        'unavailable',
    }
    assert providers['timm_dino_or_vit']['status'] in {
        'available_audit_only',
        'unavailable',
    }


def test_feature_table_preserves_identity_and_finite_provider_prefixes(tmp_path):
    feature_df, artifacts = build_learned_roi_feature_table(
        _embedding_df(tmp_path), tmp_path / 'feature_sets', tmp_path / 'diagnostics'
    )

    assert artifacts['learned_roi_features'].exists()
    assert artifacts['learned_roi_feature_metadata'].exists()
    assert artifacts['learned_roi_feature_diagnostics'].exists()
    assert {
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
    }.issubset(feature_df.columns)
    provider_columns = [
        column for column in feature_df.columns if column.startswith('learned_')
    ]
    assert any(
        column.startswith('learned_current_glomeruli_encoder_')
        for column in provider_columns
    )
    assert any(
        column.startswith('learned_simple_roi_qc_') for column in provider_columns
    )
    assert np.isfinite(feature_df[provider_columns].to_numpy(dtype=float)).all()


def test_evaluate_learned_roi_writes_capped_candidates_and_gates(tmp_path):
    artifacts = evaluate_learned_roi_quantification(
        _embedding_df(tmp_path), tmp_path / 'burden', n_splits=3
    )

    for key in [
        'learned_roi_provider_audit',
        'learned_roi_index',
        'learned_roi_estimator_verdict',
        'learned_roi_estimator_verdict_md',
        'learned_roi_artifact_manifest',
        'learned_roi_features',
        'learned_roi_feature_metadata',
        'learned_roi_feature_diagnostics',
        'learned_roi_candidate_metrics',
        'learned_roi_predictions',
        'learned_roi_subject_predictions',
        'learned_roi_calibration',
        'learned_roi_cohort_confounding_diagnostics',
        'learned_roi_candidate_summary',
        'learned_roi_review_html',
        'learned_roi_review_examples',
        'learned_roi_nearest_examples',
        'learned_roi_attribution_status',
    ]:
        assert artifacts[key].exists(), key

    metrics = pd.read_csv(artifacts['learned_roi_candidate_metrics'])
    expected_candidates = {
        'image_current_glomeruli_encoder',
        'image_simple_roi_qc',
        'image_current_glomeruli_encoder_plus_simple_roi_qc',
        'subject_current_glomeruli_encoder',
        'subject_simple_roi_qc',
        'subject_current_glomeruli_encoder_plus_simple_roi_qc',
    }
    assert set(metrics['candidate_id']) == expected_candidates
    assert (
        not metrics['candidate_id']
        .str.contains('torchvision|timm|dino|conch|uni')
        .any()
    )
    assert {'stage_index_mae', 'grade_scale_mae'}.issubset(metrics.columns)

    predictions = pd.read_csv(artifacts['learned_roi_predictions'])
    assert predictions.groupby('subject_id')['fold'].nunique().max() == 1
    assert {
        'prediction_set_scores',
        'burden_interval_low_0_100',
        'burden_interval_high_0_100',
        'prediction_source',
    }.issubset(predictions.columns)

    subject_predictions = pd.read_csv(artifacts['learned_roi_subject_predictions'])
    selected_subject = subject_predictions[
        subject_predictions['candidate_id']
        == 'subject_current_glomeruli_encoder_plus_simple_roi_qc'
    ]
    assert selected_subject['subject_id'].is_unique

    calibration = json.loads(
        artifacts['learned_roi_calibration'].read_text(encoding='utf-8')
    )
    assert calibration['nominal_coverage'] == 0.9
    assert 'by_cohort' in calibration
    assert 'by_observed_score' in calibration
    assert (
        calibration['baseline_comparison'][
            'current_baseline_average_prediction_set_size'
        ]
        == 5.308
    )

    cohort_diagnostics = json.loads(
        artifacts['learned_roi_cohort_confounding_diagnostics'].read_text(
            encoding='utf-8'
        )
    )
    assert 'cohort_predictability_screen' in cohort_diagnostics
    assert 'readiness_blockers' in cohort_diagnostics

    summary = json.loads(
        artifacts['learned_roi_candidate_summary'].read_text(encoding='utf-8')
    )
    assert summary['phase_1_candidate_ids'] == sorted(expected_candidates)
    assert (
        summary['per_image_readiness']['thresholds']['average_prediction_set_size_max']
        == 4.0
    )
    assert (
        summary['claim_boundary']
        == 'predictive grade-equivalent endotheliosis burden; not tissue percent, closed-capillary percent, causal evidence, or mechanistic proof'
    )
    verdict = json.loads(
        artifacts['learned_roi_estimator_verdict'].read_text(encoding='utf-8')
    )
    assert verdict['estimator'] == 'learned_roi'
    assert verdict['candidate_count'] == len(expected_candidates)
    assert verdict['claim_boundary'] == summary['claim_boundary']
    manifest = json.loads(
        artifacts['learned_roi_artifact_manifest'].read_text(encoding='utf-8')
    )
    assert manifest['first_read']['verdict_json'] == 'summary/estimator_verdict.json'
    assert 'learned_roi_candidate_summary' in manifest['artifacts']
    assert (tmp_path / 'burden' / 'learned_roi' / 'INDEX.md').exists()

    review_html = artifacts['learned_roi_review_html'].read_text(encoding='utf-8')
    assert 'Learned ROI Quantification Review' in review_html
    assert 'not proof of closed-lumen biology' in review_html
    assert not (tmp_path / 'burden' / 'learned_roi_candidate_summary.json').exists()
    assert not (tmp_path / 'burden' / 'learned_roi_predictions.csv').exists()
    assert not (tmp_path / 'burden' / 'learned_roi' / 'primary_model').exists()
