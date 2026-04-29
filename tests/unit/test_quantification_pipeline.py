import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from eq.quantification.labelstudio_scores import recover_label_studio_score_table
from eq.quantification.ordinal import CanonicalOrdinalClassifier
from eq.quantification.pipeline import (
    ALLOWED_SCORE_VALUES,
    _apply_score_label_overrides,
    _prepare_encoder_for_forward,
    build_image_level_scored_example_table,
    build_manifest_scored_example_table,
    build_scored_example_table,
    evaluate_embedding_table,
    extract_image_level_roi_crops,
    extract_roi_crops,
)


def _make_rect_mask(
    shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x0, y0, x1, y1 in rectangles:
        mask[y0:y1, x0:x1] = 255
    return mask


def test_build_scored_table_and_extract_rois(tmp_path: Path):
    project_dir = tmp_path / 'project'
    image_dir = project_dir / 'images' / 'T19'
    mask_dir = project_dir / 'masks' / 'T19'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[10:40, 50:80] = 120
    image[70:100, 20:50] = 200
    Image.fromarray(image).save(image_dir / 'T19-1.jpg')

    mask = _make_rect_mask((120, 120), [(50, 10, 80, 40), (20, 70, 50, 100)])
    Image.fromarray(mask).save(mask_dir / 'T19-1_mask.jpg')

    metadata_df = pd.DataFrame(
        {'subject_id': ['T19-1', 'T19-1'], 'glomerulus_id': [1, 2], 'score': [0.0, 1.0]}
    )

    scored = build_scored_example_table(project_dir, metadata_df, tmp_path / 'out')
    assert set(scored['join_status']) == {'ok'}

    roi_table = extract_roi_crops(scored, tmp_path / 'roi')
    assert set(roi_table['roi_status']) == {'ok'}
    assert roi_table['roi_image_path'].astype(bool).all()
    assert roi_table['roi_area'].notna().all()


def test_recover_labelstudio_scores_prefers_latest_and_backfills_missing_grade(
    tmp_path: Path,
):
    project_dir = tmp_path / 'project'
    image_dir = project_dir / 'images' / 'T19'
    mask_dir = project_dir / 'masks' / 'T19'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = _make_rect_mask((64, 64), [(16, 16, 48, 48)])
    for name in ['T19_Image0.jpg', 'T19_Image1.jpg']:
        Image.fromarray(image).save(image_dir / name)
        Image.fromarray(mask).save(mask_dir / f'{Path(name).stem}_mask.jpg')

    annotation_payload = [
        {
            'id': 1,
            'file_upload': 'aaaa-T19_Image0.jpg',
            'created_at': '2023-04-06T16:31:14.000000Z',
            'updated_at': '2023-04-06T16:31:42.000000Z',
            'annotations': [
                {
                    'updated_at': '2023-04-06T16:31:42.000000Z',
                    'result': [
                        {
                            'type': 'brushlabels',
                            'value': {'brushlabels': ['glomerulus']},
                        },
                        {'type': 'choices', 'value': {'choices': ['0.5']}},
                    ],
                }
            ],
        },
        {
            'id': 2,
            'file_upload': 'bbbb-T19_Image0.jpg',
            'created_at': '2023-04-10T17:33:32.000000Z',
            'updated_at': '2023-04-10T20:37:47.000000Z',
            'annotations': [
                {
                    'updated_at': '2023-04-10T20:37:47.000000Z',
                    'result': [
                        {
                            'type': 'brushlabels',
                            'value': {'brushlabels': ['glomerulus']},
                        },
                        {'type': 'choices', 'value': {'choices': ['1.0']}},
                    ],
                }
            ],
        },
        {
            'id': 3,
            'file_upload': 'cccc-T19_Image1.jpg',
            'created_at': '2023-04-06T16:31:14.000000Z',
            'updated_at': '2023-04-06T16:31:42.000000Z',
            'annotations': [
                {
                    'updated_at': '2023-04-06T16:31:42.000000Z',
                    'result': [
                        {
                            'type': 'brushlabels',
                            'value': {'brushlabels': ['glomerulus']},
                        },
                        {'type': 'choices', 'value': {'choices': ['0']}},
                    ],
                }
            ],
        },
        {
            'id': 4,
            'file_upload': 'dddd-T19_Image1.jpg',
            'created_at': '2023-04-10T17:33:32.000000Z',
            'updated_at': '2023-04-10T20:37:47.000000Z',
            'annotations': [
                {
                    'updated_at': '2023-04-10T20:37:47.000000Z',
                    'result': [
                        {
                            'type': 'brushlabels',
                            'value': {'brushlabels': ['glomerulus']},
                        }
                    ],
                }
            ],
        },
    ]
    annotation_path = tmp_path / 'annotations.json'
    annotation_path.write_text(json.dumps(annotation_payload), encoding='utf-8')

    outputs = recover_label_studio_score_table(
        project_dir, annotation_path, tmp_path / 'scores'
    )
    score_table = pd.read_csv(outputs['scores'])

    image0 = score_table.loc[score_table['image_name'] == 'T19_Image0.jpg'].iloc[0]
    image1 = score_table.loc[score_table['image_name'] == 'T19_Image1.jpg'].iloc[0]
    assert image0['score'] == 1.0
    assert image0['score_resolution'] == 'latest_annotation'
    assert image1['score'] == 0.0
    assert image1['score_resolution'] == 'latest_missing_grade_backfilled'
    assert set(score_table['join_status']) == {'ok'}


def test_build_image_level_scored_table_and_extract_rois(tmp_path: Path):
    project_dir = tmp_path / 'project'
    image_dir = project_dir / 'images' / 'T19'
    mask_dir = project_dir / 'masks' / 'T19'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[20:50, 20:50] = 180
    image[70:100, 70:100] = 120
    Image.fromarray(image).save(image_dir / 'T19_Image5.jpg')

    mask = _make_rect_mask((120, 120), [(25, 25, 45, 45), (75, 75, 95, 95)])
    Image.fromarray(mask).save(mask_dir / 'T19_Image5_mask.jpg')

    score_table = pd.DataFrame(
        {
            'image_name': ['T19_Image5.jpg'],
            'image_stem': ['T19_Image5'],
            'subject_prefix': ['T19'],
            'score': [1.0],
            'score_status': ['ok'],
            'score_resolution': ['latest_annotation'],
            'raw_image_path': [str(image_dir / 'T19_Image5.jpg')],
            'raw_mask_path': [str(mask_dir / 'T19_Image5_mask.jpg')],
            'join_status': ['ok'],
        }
    )

    scored = build_image_level_scored_example_table(
        project_dir, score_table, tmp_path / 'out'
    )
    assert scored.loc[0, 'subject_image_id'] == 'T19_Image5'
    assert scored.loc[0, 'glomerulus_id'] == 1

    roi_table = extract_image_level_roi_crops(scored, tmp_path / 'roi', padding=0)
    assert roi_table.loc[0, 'roi_status'] == 'ok'
    assert roi_table.loc[0, 'roi_component_selection'] == 'union_mask'
    assert roi_table.loc[0, 'roi_component_count'] == 2
    assert roi_table.loc[0, 'roi_bbox_x0'] == 25
    assert roi_table.loc[0, 'roi_bbox_y0'] == 25
    assert roi_table.loc[0, 'roi_bbox_x1'] == 95
    assert roi_table.loc[0, 'roi_bbox_y1'] == 95
    assert 0.4 < roi_table.loc[0, 'roi_largest_component_area_fraction'] < 0.6
    assert Path(str(roi_table.loc[0, 'roi_image_path'])).exists()


def test_manifest_scored_examples_use_all_admitted_mask_paired_rows(tmp_path: Path):
    cohorts = tmp_path / 'raw_data' / 'cohorts'
    lauren_images = cohorts / 'lauren_preeclampsia' / 'images'
    lauren_masks = cohorts / 'lauren_preeclampsia' / 'masks'
    dox_images = cohorts / 'vegfri_dox' / 'images' / 'M1'
    dox_masks = cohorts / 'vegfri_dox' / 'masks' / 'M1'
    for path in (lauren_images, lauren_masks, dox_images, dox_masks):
        path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
        lauren_images / 't19_image0.jpg'
    )
    Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 255).save(
        lauren_masks / 't19_image0_mask.jpg'
    )
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
        dox_images / 'M1_Image0.jpg'
    )
    Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 255).save(
        dox_masks / 'M1_Image0_mask.png'
    )
    pd.DataFrame(
        [
            {
                'cohort_id': 'lauren_preeclampsia',
                'image_path': 'raw_data/cohorts/lauren_preeclampsia/images/t19_image0.jpg',
                'mask_path': 'raw_data/cohorts/lauren_preeclampsia/masks/t19_image0_mask.jpg',
                'score': 0.5,
                'source_image_name': 'T19_Image0.jpg',
                'source_sample_id': 'T19_Image0',
                'manifest_row_id': 'lauren_preeclampsia__000001',
                'join_status': 'joined',
                'admission_status': 'admitted',
                'lane_assignment': 'manual_mask_core',
                'score_status': '',
                'source_score_sheet': 'labelstudio_annotations.json',
                'score_path': 'raw_data/cohorts/lauren_preeclampsia/scores/labelstudio_scores.csv',
            },
            {
                'cohort_id': 'vegfri_dox',
                'image_path': 'raw_data/cohorts/vegfri_dox/images/M1/M1_Image0.jpg',
                'mask_path': 'raw_data/cohorts/vegfri_dox/masks/M1/M1_Image0_mask.png',
                'score': 1.0,
                'source_image_name': 'M1_Image0.jpg',
                'source_sample_id': 'M1',
                'manifest_row_id': 'vegfri_dox__000001',
                'join_status': 'joined',
                'admission_status': 'admitted',
                'lane_assignment': 'manual_mask_external',
                'score_status': 'ok',
                'source_score_sheet': 'labelstudio_scores.csv',
                'score_path': 'raw_data/cohorts/vegfri_dox/scores/labelstudio_scores.csv',
            },
            {
                'cohort_id': 'vegfri_mr',
                'image_path': 'raw_data/cohorts/vegfri_mr/images/MR1.tif',
                'mask_path': '',
                'score': 2.0,
                'source_image_name': 'MR1.tif',
                'source_sample_id': 'MR1',
                'manifest_row_id': 'vegfri_mr__000001',
                'join_status': 'joined',
                'admission_status': 'evaluation_only',
                'lane_assignment': 'mr_concordance_only',
                'score_status': 'ok',
                'source_score_sheet': 'MR workbook',
                'score_path': 'raw_data/cohorts/vegfri_mr/scores/workbook.xlsx',
            },
        ]
    ).to_csv(cohorts / 'manifest.csv', index=False)

    scored = build_manifest_scored_example_table(cohorts, tmp_path / 'out')

    assert len(scored) == 2
    assert set(scored['cohort_id']) == {'lauren_preeclampsia', 'vegfri_dox'}
    assert set(scored['lane_assignment']) == {
        'manual_mask_core',
        'manual_mask_external',
    }
    assert scored['join_status'].eq('ok').all()
    assert set(scored['subject_prefix']) == {'lauren_preeclampsia:T19', 'vegfri_dox:M1'}
    assert set(scored['subject_id']) == {'lauren_preeclampsia__t19', 'vegfri_dox__m1'}
    assert set(scored['sample_id']) == {
        'lauren_preeclampsia__t19__undated',
        'vegfri_dox__m1__undated',
    }
    assert scored['image_id'].is_unique
    summary = json.loads(
        (tmp_path / 'out' / 'manifest_scored_examples_summary.json').read_text()
    )
    assert summary['n_scored_rows'] == 2
    assert summary['cohort_counts'] == {'lauren_preeclampsia': 1, 'vegfri_dox': 1}
    assert summary['identity_contract']['validation_group_key'] == 'subject_id'
    assert summary['identity_contract']['duplicate_image_ids'] == []


def test_score_label_overrides_replace_scores_and_write_audit(tmp_path: Path):
    scored = pd.DataFrame(
        {'subject_image_id': ['case_1', 'case_2'], 'score': [0.5, 2.0]}
    )
    overrides = tmp_path / 'rubric_overrides.csv'
    pd.DataFrame(
        {
            'subject_image_id': ['case_1'],
            'rubric_score': [2.0],
            'reviewer_confidence_1_5': ['high'],
            'accepted_teaching': [True],
            'review_flags': ['RBCs'],
        }
    ).to_csv(overrides, index=False)

    updated, artifacts = _apply_score_label_overrides(
        scored, overrides, tmp_path / 'out'
    )

    assert updated.loc[updated['subject_image_id'] == 'case_1', 'score'].item() == 2.0
    assert (
        updated.loc[
            updated['subject_image_id'] == 'case_1',
            'original_score_before_label_override',
        ].item()
        == 0.5
    )
    audit = pd.read_csv(artifacts['score_label_overrides_audit'])
    assert audit.loc[0, 'original_score'] == 0.5
    assert audit.loc[0, 'override_score'] == 2.0
    summary = json.loads(
        artifacts['score_label_overrides_summary'].read_text(encoding='utf-8')
    )
    assert summary['applied_rows'] == 1
    assert summary['severe_boundary_changed_rows'] == 1


def test_canonical_ordinal_classifier_outputs_probability_simplex():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    model = CanonicalOrdinalClassifier(classes=np.arange(7)).fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (6, 7)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(6), atol=1e-6)


def test_canonical_ordinal_classifier_handles_missing_training_classes():
    x = np.array([[0.0], [0.5], [1.0], [1.5]], dtype=np.float64)
    y = np.array([0, 1, 1, 3], dtype=np.int64)
    model = CanonicalOrdinalClassifier(classes=np.arange(5)).fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (4, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4), atol=1e-6)
    assert np.all(probabilities[:, 2] == 0.0)
    assert np.all(probabilities[:, 4] == 0.0)


def test_prepare_encoder_for_forward_wraps_module_list():
    module_list = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    wrapped = _prepare_encoder_for_forward(module_list)
    assert isinstance(wrapped, torch.nn.Identity)


def test_evaluate_embedding_table_runs_with_grouped_subject_splits(tmp_path: Path):
    rows = []
    for subject_index, subject_prefix in enumerate(['T19', 'T20', 'T21', 'T22']):
        for image_index in range(1, 3):
            score = float(
                ALLOWED_SCORE_VALUES[
                    (subject_index + image_index) % len(ALLOWED_SCORE_VALUES)
                ]
            )
            rows.append(
                {
                    'subject_image_id': f'{subject_prefix}-{image_index}',
                    'subject_prefix': subject_prefix,
                    'subject_id': f'cohort__{subject_prefix.lower()}',
                    'cohort_id': 'cohort',
                    'sample_id': f'cohort__{subject_prefix.lower()}__image{image_index}',
                    'glomerulus_id': 1,
                    'score': score,
                    'raw_image_path': str(
                        tmp_path / f'{subject_prefix}-{image_index}.jpg'
                    ),
                    'raw_mask_path': str(
                        tmp_path / f'{subject_prefix}-{image_index}_mask.jpg'
                    ),
                    'roi_image_path': str(
                        tmp_path / f'{subject_prefix}-{image_index}_roi.jpg'
                    ),
                    'roi_mask_path': str(
                        tmp_path / f'{subject_prefix}-{image_index}_roi_mask.jpg'
                    ),
                    'roi_bbox_x0': 0,
                    'roi_bbox_y0': 0,
                    'roi_bbox_x1': 20,
                    'roi_bbox_y1': 20,
                    'roi_area': 100,
                    'roi_fill_fraction': 0.25,
                    'roi_mean_intensity': 0.5 + 0.1 * image_index,
                    'roi_openness_score': 0.2 + 0.05 * subject_index,
                    'roi_component_count': 2,
                    'roi_component_selection': 'union_mask',
                    'roi_union_bbox_width': 20,
                    'roi_union_bbox_height': 20,
                    'roi_largest_component_area_fraction': 0.5,
                    'embedding_0000': float(subject_index),
                    'embedding_0001': float(image_index),
                    'embedding_0002': float(subject_index + image_index),
                }
            )

            raw_image = np.full(
                (24, 24, 3), fill_value=40 * (subject_index + 1), dtype=np.uint8
            )
            raw_mask = _make_rect_mask((24, 24), [(4, 4, 10, 10), (14, 14, 20, 20)])
            roi_image = raw_image[2:22, 2:22]
            roi_mask = raw_mask[2:22, 2:22]
            Image.fromarray(raw_image).save(
                tmp_path / f'{subject_prefix}-{image_index}.jpg'
            )
            Image.fromarray(raw_mask).save(
                tmp_path / f'{subject_prefix}-{image_index}_mask.jpg'
            )
            Image.fromarray(roi_image).save(
                tmp_path / f'{subject_prefix}-{image_index}_roi.jpg'
            )
            Image.fromarray(roi_mask).save(
                tmp_path / f'{subject_prefix}-{image_index}_roi_mask.jpg'
            )

    embedding_df = pd.DataFrame(rows)
    artifacts = evaluate_embedding_table(embedding_df, tmp_path / 'model')
    assert artifacts['metrics'].exists()
    assert artifacts['predictions'].exists()
    assert artifacts['confusion_matrix'].exists()
    assert artifacts['review_html'].exists()
    assert artifacts['review_examples'].exists()

    predictions_df = pd.read_csv(artifacts['predictions'])
    for probability_column in [
        'prob_score_0_0',
        'prob_score_0_5',
        'prob_score_1_0',
        'prob_score_1_5',
        'prob_score_2_0',
        'prob_score_3_0',
    ]:
        assert probability_column in predictions_df.columns
    assert 'prob_score_2_5' not in predictions_df.columns
    assert {
        'expected_score',
        'top_two_margin',
        'entropy',
        'absolute_error',
        'prediction_error',
    }.issubset(predictions_df.columns)
    metrics = json.loads(artifacts['metrics'].read_text(encoding='utf-8'))
    assert metrics['n_examples'] == len(embedding_df)
    assert metrics['n_subject_groups'] == 4
    assert metrics['grouping_key'] == 'subject_id'
    assert metrics['ordinal_model']['estimator_class'] == 'CanonicalOrdinalClassifier'
    assert metrics['stability']['zero_unresolved_warning_gate_passed'] is True
    assert metrics['cohort_profile']['embedding_dim'] == 3
    assert metrics['cohort_profile']['n_examples'] == len(embedding_df)
    assert metrics['stability']['certification_status'] == 'incomplete'
    assert (
        'missing_target_class_support' in metrics['stability']['certification_blockers']
    )

    html = artifacts['review_html'].read_text(encoding='utf-8')
    assert 'descriptive audit signals' in html
    assert html.count('class="example-card"') == 7

    for key in [
        'burden_predictions',
        'burden_metrics',
        'threshold_metrics',
        'threshold_support',
        'calibration_bins',
        'uncertainty_calibration',
        'grouping_audit',
        'prediction_explanations',
        'nearest_examples',
        'cohort_metrics',
        'group_summary_intervals',
        'final_model_predictions',
        'final_model_cohort_metrics',
        'final_model_group_summary_intervals',
        'signal_comparator_metrics',
        'subject_level_candidate_predictions',
        'precision_candidate_summary',
        'morphology_features',
        'morphology_feature_metadata',
        'subject_morphology_features',
        'morphology_feature_diagnostics',
        'morphology_feature_review_html',
        'morphology_feature_review_cases',
        'morphology_operator_adjudication_template',
        'morphology_operator_adjudication_agreement',
        'morphology_candidate_metrics',
        'subject_morphology_candidate_predictions',
        'morphology_candidate_summary',
        'burden_model',
        'burden_model_index',
        'primary_burden_index_index',
        'quantification_review_html',
        'quantification_review_examples',
        'quantification_results_summary_md',
        'quantification_results_summary_csv',
        'quantification_readme_snippet',
        'source_aware_index',
        'source_aware_estimator_verdict',
        'source_aware_metrics_by_split',
        'source_aware_artifact_manifest',
        'source_aware_image_predictions',
        'source_aware_subject_predictions',
        'source_aware_upstream_roi_adequacy',
    ]:
        assert artifacts[key].exists(), key

    burden_predictions = pd.read_csv(artifacts['burden_predictions'])
    assert {
        'prob_score_gt_0',
        'prob_score_gt_0p5',
        'prob_score_gt_1',
        'prob_score_gt_1p5',
        'prob_score_gt_2',
        'endotheliosis_burden_0_100',
        'prediction_set_scores',
        'burden_interval_low_0_100',
        'burden_interval_high_0_100',
    }.issubset(burden_predictions.columns)
    assert (
        burden_predictions['prob_score_gt_0'] >= burden_predictions['prob_score_gt_0p5']
    ).all()
    assert (
        burden_predictions['prob_score_gt_0p5'] >= burden_predictions['prob_score_gt_1']
    ).all()

    review_html = artifacts['quantification_review_html'].read_text(encoding='utf-8')
    assert 'Operational Verdict' in review_html
    assert 'Final Full-Cohort Summaries' in review_html
    assert 'Reviewer Examples' in review_html
    assert 'predictive ordinal stage-burden index' in review_html
    assert 'Endotheliosis burden index (0-100)' in review_html
    assert 'Comparator Summaries' in review_html
    assert 'Precision Candidate Screen' in review_html
    assert 'Morphology Feature Screen' in review_html
    assert 'Source-Aware Estimator' in review_html
    assert 'Metrics by split' in review_html
    assert 'Direct stage-index regression' in review_html
    assert 'held_out_grouped_fold_prediction' in review_html
    assert (
        'burden_model/primary_burden_index/model/burden_predictions.csv' in review_html
    )
    assert (
        'burden_model/primary_burden_index/model/final_model_predictions.csv'
        in review_html
    )
    assert (
        'burden_model/primary_burden_index/feature_sets/morphology_features.csv'
        in review_html
    )
    assert artifacts['morphology_candidate_metrics'].parent.name == 'candidates'
    assert artifacts['burden_model'].parent.name == 'model'
    assert artifacts['burden_model'].parents[1].name == 'primary_burden_index'
    assert not (
        tmp_path / 'quantification_results' / 'burden_model' / 'primary_model'
    ).exists()
    assert not (
        tmp_path / 'quantification_results' / 'burden_model' / 'summaries'
    ).exists()
    source_verdict = json.loads(
        artifacts['source_aware_estimator_verdict'].read_text(encoding='utf-8')
    )
    assert source_verdict['testing_status'] == (
        'testing_not_available_current_data_sensitivity'
    )
    source_metrics = pd.read_csv(artifacts['source_aware_metrics_by_split'])
    assert 'training_apparent' in set(source_metrics['split_label'])
    assert 'validation_subject_heldout' in set(source_metrics['split_label'])

    review_examples = pd.read_csv(artifacts['quantification_review_examples'])
    assert '_set_size' not in review_examples.columns
    assert 'ordinal_fold' in review_examples.columns
    assert review_examples['predicted_score'].notna().all()

    snippet = artifacts['quantification_readme_snippet'].read_text(encoding='utf-8')
    assert 'quantification_review/quantification_review.html' not in snippet
    assert 'README/docs-ready' in snippet
