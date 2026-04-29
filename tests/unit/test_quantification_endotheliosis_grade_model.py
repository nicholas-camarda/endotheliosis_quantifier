import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eq.quantification.endotheliosis_grade_model import (
    FIRST_CLASS_FAMILIES,
    _write_dox_overcall_review_diagnostic,
    _write_final_model_if_supported,
    evaluate_endotheliosis_grade_model,
    grade_model_output_paths,
)
from eq.quantification.morphology_features import MORPHOLOGY_FEATURE_COLUMNS


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _embedding_df(tmp_path: Path) -> pd.DataFrame:
    rows = []
    scores = [0, 0.5, 1, 1.5, 2, 3]
    for subject_index in range(12):
        cohort_id = 'vegfri_dox' if subject_index >= 6 else 'lauren_preeclampsia'
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
                    'glomerulus_id': 1,
                    'cohort_id': cohort_id,
                    'score': score,
                    'roi_image_path': str(image_path),
                    'roi_mask_path': str(mask_path),
                    'raw_image_path': str(image_path),
                    'raw_mask_path': str(mask_path),
                    'roi_area': 100 + score * 20 + subject_index,
                    'roi_fill_fraction': 0.45 + score * 0.05,
                    'roi_mean_intensity': 80 + score * 5,
                    'roi_openness_score': 0.1 * image_index + 0.05 * score,
                    'roi_component_count': 1 + int(score >= 2),
                    'roi_union_bbox_width': 16,
                    'roi_union_bbox_height': 16,
                    'roi_largest_component_area_fraction': 1.0 - 0.03 * image_index,
                    'embedding_0000': float(subject_index),
                    'embedding_0001': float(image_index),
                    'embedding_0002': score,
                    'embedding_0003': float(score >= 2),
                }
            )
    return pd.DataFrame(rows)


def _write_upstream_artifacts(burden_root: Path, df: pd.DataFrame) -> None:
    morphology_dir = burden_root / 'primary_burden_index' / 'feature_sets'
    morphology_dir.mkdir(parents=True, exist_ok=True)
    morphology = df[['subject_image_id', 'glomerulus_id']].copy()
    for index, column in enumerate(MORPHOLOGY_FEATURE_COLUMNS):
        morphology[column] = df['score'].astype(float) * (index + 1) + index
    morphology.to_csv(morphology_dir / 'morphology_features.csv', index=False)

    learned_dir = burden_root / 'learned_roi' / 'feature_sets'
    learned_dir.mkdir(parents=True, exist_ok=True)
    learned = df[['subject_image_id', 'glomerulus_id']].copy()
    learned['learned_roi_signal'] = df['score'].astype(float)
    learned['learned_roi_quality'] = df['roi_fill_fraction'].astype(float)
    learned.to_csv(learned_dir / 'learned_roi_features.csv', index=False)

    adjudication_dir = burden_root / 'severe_aware_ordinal_estimator' / 'evidence'
    adjudication_dir.mkdir(parents=True, exist_ok=True)
    severe_row = df[df['score'] >= 2].iloc[0]
    adjudication_dir.joinpath('severe_false_negative_adjudications.json').write_text(
        json.dumps(
            [
                {
                    'subject_image_id': severe_row['subject_image_id'],
                    'grade_adjudication': 'grade_too_high',
                    'failure_source': 'not_severe_after_review',
                }
            ]
        ),
        encoding='utf-8',
    )


def test_grade_model_writes_grouped_folds_and_selector_artifacts(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path)
    _write_upstream_artifacts(burden_root, df)
    label_contract = {
        'target_definition_version': 'test_v1',
        'label_overrides_path': 'derived_data/overrides.csv',
        'label_overrides_hash': 'abc123',
        'grouping_identity': {'validation_group_key': 'subject_id'},
    }

    artifacts = evaluate_endotheliosis_grade_model(
        df,
        burden_root,
        n_splits=3,
        label_contract_reference=label_contract,
    )

    root = burden_root / 'endotheliosis_grade_model'
    folds = pd.read_csv(artifacts['endotheliosis_grade_model_development_folds'])
    assert folds.groupby('subject_id')['fold'].nunique().max() == 1
    assert 'adjudicated_severe' in folds.columns

    verdict = json.loads(artifacts['endotheliosis_grade_model_verdict'].read_text())
    assert verdict['label_contract_reference'] == label_contract
    assert verdict['overall_status'] in {
        'model_ready_pending_mr_tiff_deployment_smoke',
        'diagnostic_only_current_data_model',
        'current_data_insufficient',
    }
    assert verdict['readme_facing_deployment_allowed'] is False
    assert (
        'mr_tiff_segmentation_to_quantification_path_not_proven'
        in verdict['hard_blockers']
    )
    assert (root / 'summary' / 'candidate_coverage_matrix.csv').exists()
    assert (root / 'summary' / 'input_artifact_index.json').exists()
    metrics = pd.read_csv(root / 'summary' / 'development_oof_metrics.csv')
    assert set(metrics['metric_label'].dropna()) == {
        'grouped_out_of_fold_development_estimate'
    }
    severe_metrics = metrics[metrics['target_kind'] == 'severe']
    assert severe_metrics['finite_output'].all()
    assert {'warning_status', 'warning_count'}.issubset(metrics.columns)
    feature_diagnostics = json.loads(
        (root / 'internal' / 'feature_diagnostics.json').read_text(encoding='utf-8')
    )
    fold_coverage = feature_diagnostics['fold_diagnostics']['fold_coverage']
    assert all('coverage_status' in row for row in fold_coverage)
    assert not (root / 'feature_sets').exists()
    assert not (root / 'validation').exists()
    assert not (root / 'calibration').exists()
    assert not (root / 'summaries').exists()


def test_final_model_metadata_records_label_contract_reference(tmp_path):
    paths = grade_model_output_paths(tmp_path / 'burden_model')
    paths['final_training_predictions'].parent.mkdir(parents=True, exist_ok=True)
    label_contract = {'target_definition_version': 'test_v1'}
    frame = pd.DataFrame(
        {
            'subject_id': ['s1', 's2', 's3', 's4'],
            'subject_image_id': ['i1', 'i2', 'i3', 'i4'],
            'glomerulus_id': [1, 1, 1, 1],
            'cohort_id': ['c', 'c', 'c', 'c'],
            'score': [0.0, 0.5, 2.0, 3.0],
            'adjudicated_severe': [False, False, True, True],
            'feature_a': [0.1, 0.2, 1.0, 1.2],
        }
    )
    metrics = pd.DataFrame(
        [
            {
                'candidate_id': 'candidate_1',
                'feature_family': 'test',
                'target_kind': 'severe',
                'model_kind': 'logistic_l2',
                'regularization_c': 1.0,
                'threshold': 0.5,
                'mr_computable': True,
            }
        ]
    )
    verdict = {
        'overall_status': 'model_ready_pending_mr_tiff_deployment_smoke',
        'selected_candidate_id': 'candidate_1',
        'selected_family_id': 'test_family',
        'claim_boundary': 'test',
    }

    _write_final_model_if_supported(
        frame,
        metrics,
        verdict,
        {'test': ['feature_a']},
        paths,
        label_contract_reference=label_contract,
    )

    metadata = json.loads(paths['final_model_metadata'].read_text(encoding='utf-8'))
    assert metadata['label_contract_reference'] == label_contract


def test_first_class_family_subtrees_are_not_selector_internal_only(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path)
    _write_upstream_artifacts(burden_root, df)

    evaluate_endotheliosis_grade_model(df, burden_root, n_splits=3)

    for family_id in FIRST_CLASS_FAMILIES:
        family_root = burden_root / family_id
        assert (family_root / 'INDEX.md').exists()
        for dirname in [
            'summary',
            'diagnostics',
            'predictions',
            'model',
            'evidence',
            'internal',
        ]:
            assert (family_root / dirname).is_dir()
        for diagnostic in [
            'input_support.json',
            'feature_diagnostics.json',
            'fold_diagnostics.json',
            'source_sensitivity.json',
            'gate_diagnostics.json',
        ]:
            assert (family_root / 'diagnostics' / diagnostic).exists()


def test_missing_required_upstream_artifacts_are_hard_blockers(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path)

    artifacts = evaluate_endotheliosis_grade_model(df, burden_root, n_splits=3)

    verdict = json.loads(artifacts['endotheliosis_grade_model_verdict'].read_text())
    assert 'missing_primary_burden_morphology_features' in verdict['hard_blockers']
    assert 'missing_learned_roi_features' in verdict['hard_blockers']
    blockers = json.loads(
        (
            burden_root
            / 'endotheliosis_grade_model'
            / 'diagnostics'
            / 'hard_blockers.json'
        ).read_text(encoding='utf-8')
    )
    assert 'missing_primary_burden_morphology_features' in blockers


def test_optional_feature_tables_do_not_override_input_scores(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path)
    _write_upstream_artifacts(burden_root, df)
    changed_id = df.loc[0, 'subject_image_id']
    df.loc[df['subject_image_id'] == changed_id, 'score'] = 3.0

    for relative_path in [
        'primary_burden_index/feature_sets/morphology_features.csv',
        'learned_roi/feature_sets/learned_roi_features.csv',
    ]:
        path = burden_root / relative_path
        table = pd.read_csv(path)
        table['score'] = 0.0
        table.to_csv(path, index=False)

    evaluate_endotheliosis_grade_model(df, burden_root, n_splits=3)

    predictions = pd.read_csv(
        burden_root
        / 'endotheliosis_grade_model'
        / 'predictions'
        / 'development_oof_predictions.csv'
    )
    observed_scores = predictions.loc[
        predictions['subject_image_id'] == changed_id, 'score'
    ].unique()
    assert observed_scores.tolist() == [3.0]


def test_optional_row_level_features_join_on_glomerulus_id(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path)
    shared_subject_image_id = df.loc[0, 'subject_image_id']
    df.loc[1, 'subject_image_id'] = shared_subject_image_id
    df.loc[0, 'glomerulus_id'] = 1
    df.loc[1, 'glomerulus_id'] = 2
    _write_upstream_artifacts(burden_root, df)

    evaluate_endotheliosis_grade_model(df, burden_root, n_splits=3)

    diagnostics = json.loads(
        (
            burden_root
            / 'endotheliosis_grade_model'
            / 'internal'
            / 'feature_diagnostics.json'
        ).read_text(encoding='utf-8')
    )
    source_tables = {
        Path(row['path']).name: row
        for row in diagnostics['source_tables']
        if row.get('available')
    }
    assert source_tables['learned_roi_features.csv']['join_keys'] == [
        'subject_image_id',
        'glomerulus_id',
    ]
    assert source_tables['learned_roi_features.csv']['join_status'] == 'merged'

    predictions = pd.read_csv(
        burden_root
        / 'endotheliosis_grade_model'
        / 'predictions'
        / 'development_oof_predictions.csv'
    )
    matched = predictions[predictions['subject_image_id'] == shared_subject_image_id]
    assert sorted(matched['glomerulus_id'].astype(int).unique().tolist()) == [1, 2]


def test_row_level_optional_features_without_glomerulus_id_are_hard_blocked(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path)
    df.loc[1, 'subject_image_id'] = df.loc[0, 'subject_image_id']
    df.loc[0, 'glomerulus_id'] = 1
    df.loc[1, 'glomerulus_id'] = 2
    _write_upstream_artifacts(burden_root, df)

    learned_path = (
        burden_root / 'learned_roi' / 'feature_sets' / 'learned_roi_features.csv'
    )
    learned = pd.read_csv(learned_path).drop(columns=['glomerulus_id'])
    learned.to_csv(learned_path, index=False)

    artifacts = evaluate_endotheliosis_grade_model(df, burden_root, n_splits=3)

    verdict = json.loads(artifacts['endotheliosis_grade_model_verdict'].read_text())
    assert 'unusable_learned_roi_features_join_keys' in verdict['hard_blockers']

    diagnostics = json.loads(
        (
            burden_root
            / 'endotheliosis_grade_model'
            / 'internal'
            / 'feature_diagnostics.json'
        ).read_text(encoding='utf-8')
    )
    learned_diag = [
        row
        for row in diagnostics['source_tables']
        if Path(row['path']).name == 'learned_roi_features.csv'
    ][0]
    assert learned_diag['join_status'] == 'missing_required_join_keys'
    assert learned_diag['missing_required_join_keys'] == ['glomerulus_id']


def test_embedding_candidates_require_embedding_columns(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path).drop(
        columns=[
            column
            for column in _embedding_df(tmp_path).columns
            if column.startswith('embedding_')
        ]
    )
    _write_upstream_artifacts(burden_root, df)

    artifacts = evaluate_endotheliosis_grade_model(df, burden_root, n_splits=3)

    metrics = pd.read_csv(
        burden_root
        / 'endotheliosis_grade_model'
        / 'summary'
        / 'development_oof_metrics.csv'
    )
    candidate_ids = '|'.join(metrics['candidate_id'].astype(str))
    assert 'embedding_' not in candidate_ids
    assert artifacts['endotheliosis_grade_model_artifact_manifest'].exists()


def test_failed_quantification_gates_remove_stale_final_model_artifacts(tmp_path):
    burden_root = tmp_path / 'burden_model'
    df = _embedding_df(tmp_path)
    for column in [
        'roi_area',
        'roi_fill_fraction',
        'roi_mean_intensity',
        'roi_openness_score',
        'roi_component_count',
        'roi_union_bbox_width',
        'roi_union_bbox_height',
        'roi_largest_component_area_fraction',
    ]:
        df[column] = 1.0
    _write_upstream_artifacts(burden_root, df)
    morphology_path = (
        burden_root
        / 'primary_burden_index'
        / 'feature_sets'
        / 'morphology_features.csv'
    )
    morphology = pd.read_csv(morphology_path)
    for column in MORPHOLOGY_FEATURE_COLUMNS:
        morphology[column] = 1.0
    morphology.to_csv(morphology_path, index=False)
    model_dir = burden_root / 'endotheliosis_grade_model' / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    stale_model = model_dir / 'final_model.joblib'
    stale_schema = model_dir / 'inference_schema.json'
    stale_model.write_text('stale', encoding='utf-8')
    stale_schema.write_text('{}', encoding='utf-8')

    artifacts = evaluate_endotheliosis_grade_model(df, burden_root, n_splits=3)

    verdict = json.loads(artifacts['endotheliosis_grade_model_verdict'].read_text())
    if verdict['overall_status'] in {
        'diagnostic_only_current_data_model',
        'current_data_insufficient',
    }:
        assert not stale_model.exists()
        assert not stale_schema.exists()


def test_reviewed_dox_cluster_reps_confirm_overcall_blocker(tmp_path):
    burden_root = tmp_path / 'burden_model'
    paths = grade_model_output_paths(burden_root)
    paths['deployment'].mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(12):
        rows.append(
            {
                'review_priority': 10 + index,
                'review_bucket': 'cluster_representative_false_positive',
                'subject_image_id': f'dox_{index}',
                'score': 0.0,
                'predicted_severe_probability': 0.9,
                'roi_status': 'ok',
                'reviewer_score': 0.0 if index != 11 else 2.0,
                'reviewer_roi_usable': 'yes' if index < 7 else 'uncertain',
                'raw_image_path': '',
                'roi_image_path': '',
                'roi_mask_path': '',
            }
        )
    pd.DataFrame(rows).to_csv(paths['dox_overcall_triage_queue'], index=False)

    diagnostic = _write_dox_overcall_review_diagnostic(paths)

    assert diagnostic['overcall_confirmed'] is True
    assert diagnostic['usable_yes_rows'] == 7
    assert diagnostic['reviewer_nonsevere_usable_yes_rows'] == 7
    assert (
        diagnostic['selection_action']
        == 'reject_current_selected_candidate_as_dox_overcaller'
    )
    assert paths['dox_overcall_review_diagnostic'].exists()
    assert paths['dox_overcall_review_interpretation'].exists()
