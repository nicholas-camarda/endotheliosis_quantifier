import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from eq.quantification.learned_roi import evaluate_learned_roi_quantification
from eq.run_config import run_config

ATLAS_RUNNER_NAME = 'run_label_free_roi_embedding_atlas'
ATLAS_MODULE_REASON = (
    'eq.quantification.embedding_atlas is the planned implementation module for '
    'OpenSpec change label-free-roi-embedding-atlas'
)


def _atlas_module():
    return pytest.importorskip(
        'eq.quantification.embedding_atlas', reason=ATLAS_MODULE_REASON
    )


def _run_atlas(config: dict):
    atlas = _atlas_module()
    runner = getattr(atlas, ATLAS_RUNNER_NAME)
    return runner(config)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _atlas_output_root(quantification_root: Path) -> Path:
    return quantification_root / 'burden_model' / 'embedding_atlas'


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _base_roi_rows(tmp_path: Path, *, rows_per_subject: int = 2) -> pd.DataFrame:
    rows = []
    scores = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    for subject_index in range(6):
        cohort_id = 'source_a' if subject_index < 3 else 'source_b'
        treatment = 'vehicle' if subject_index % 2 == 0 else 'dox'
        for image_index in range(rows_per_subject):
            row_index = subject_index * rows_per_subject + image_index
            score = scores[row_index % len(scores)]
            roi_path = tmp_path / 'roi_assets' / f'roi_{row_index}.png'
            mask_path = tmp_path / 'roi_assets' / f'mask_{row_index}.png'
            _write_image(roi_path, 40 + row_index)
            _write_image(mask_path, 255)
            rows.append(
                {
                    'roi_row_id': f'roi-{row_index:03d}',
                    'subject_id': f'subject-{subject_index:02d}',
                    'sample_id': f'sample-{subject_index:02d}',
                    'image_id': f'image-{row_index:03d}',
                    'subject_image_id': f'subject-{subject_index:02d}__image-{image_index}',
                    'glomerulus_id': image_index + 1,
                    'cohort_id': cohort_id,
                    'source_workbook': f'{cohort_id}.xlsx',
                    'treatment_group': treatment,
                    'lane_assignment': f'lane-{subject_index % 2}',
                    'reviewer_id': f'reviewer-{image_index % 2}',
                    'score': score,
                    'severe_label': bool(score >= 2.0),
                    'manual_label_override': '',
                    'roi_image_path': str(roi_path),
                    'roi_mask_path': str(mask_path),
                    'raw_image_path': str(roi_path),
                    'raw_mask_path': str(mask_path),
                    'roi_area': 120.0 + row_index,
                    'roi_fill_fraction': 0.55 + 0.01 * image_index,
                    'roi_mean_intensity': 80.0 + subject_index,
                    'roi_openness_score': 0.08 * image_index,
                    'roi_component_count': 1,
                    'roi_union_bbox_width': 16,
                    'roi_union_bbox_height': 16,
                    'roi_largest_component_area_fraction': 0.98,
                    'rbc_heavy_flag': bool(row_index % 5 == 0),
                    'low_quality_flag': False,
                    'mask_adequacy_status': 'adequate',
                    'roi_geometry_contract_version': 'oracle_hardened_v1',
                    'roi_preprocessing_version': 'frozen_encoder_roi_preprocess_v1',
                    'roi_threshold_policy': 'fail_closed_union_roi_threshold_v1',
                    'roi_status': 'valid_union_roi',
                    'artifact_provenance_id': 'oracle-current-input-contract-v1',
                    'embedding_model_id': 'current_glomeruli_encoder',
                    'embedding_preprocessing_version': 'encoder_preprocess_v1',
                    'feature_lineage_json': json.dumps(
                        {
                            'embedding_0000': 'frozen_encoder_embedding',
                            'embedding_0001': 'frozen_encoder_embedding',
                            'embedding_0002': 'frozen_encoder_embedding',
                            'embedding_0003': 'frozen_encoder_embedding',
                            'roi_area': 'roi_qc_measurement',
                            'roi_fill_fraction': 'roi_qc_measurement',
                            'roi_mean_intensity': 'roi_qc_measurement',
                            'roi_openness_score': 'roi_qc_measurement',
                            'roi_component_count': 'roi_qc_measurement',
                        }
                    ),
                    'embedding_0000': float(subject_index),
                    'embedding_0001': float(image_index),
                    'embedding_0002': float(subject_index % 3),
                    'embedding_0003': float((subject_index + image_index) % 2),
                }
            )
    return pd.DataFrame(rows)


def _write_atlas_inputs(
    tmp_path: Path,
    *,
    missing_columns: tuple[str, ...] = (),
    include_learned_roi: bool = True,
    unapproved_feature_lineage: bool = False,
) -> tuple[Path, pd.DataFrame]:
    quantification_root = tmp_path / 'quantification_results' / 'fixture_run'
    frame = _base_roi_rows(tmp_path)
    if unapproved_feature_lineage:
        frame['feature_lineage_json'] = frame['feature_lineage_json'].map(
            lambda value: json.dumps(
                {**json.loads(value), 'embedding_0003': 'human_score_derived'}
            )
        )
    if missing_columns:
        frame = frame.drop(columns=list(missing_columns))

    embeddings_dir = quantification_root / 'embeddings'
    roi_dir = quantification_root / 'roi_crops'
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(embeddings_dir / 'roi_embeddings.csv', index=False)
    frame[
        [
            column
            for column in frame.columns
            if not column.startswith('embedding_') or column == 'embedding_model_id'
        ]
    ].to_csv(roi_dir / 'roi_scored_examples.csv', index=False)

    if include_learned_roi:
        learned_dir = (
            quantification_root / 'burden_model' / 'learned_roi' / 'feature_sets'
        )
        learned_dir.mkdir(parents=True, exist_ok=True)
        learned = frame[['roi_row_id', 'subject_image_id', 'glomerulus_id']].copy()
        learned['learned_roi_texture_signal'] = np.linspace(0.0, 1.0, len(learned))
        learned['learned_roi_qc_signal'] = frame['roi_fill_fraction'].to_numpy(
            dtype=float
        )
        learned['learned_roi_feature_lineage_json'] = json.dumps(
            {
                'learned_roi_texture_signal': 'learned_roi_feature',
                'learned_roi_qc_signal': 'learned_roi_feature',
            }
        )
        learned.to_csv(learned_dir / 'learned_roi_features.csv', index=False)

    return quantification_root, frame


def _base_atlas_config(quantification_root: Path) -> dict:
    return {
        'workflow_id': 'label_free_roi_embedding_atlas',
        'quantification_output_root': str(quantification_root),
        'input_artifacts': {
            'roi_embeddings': 'embeddings/roi_embeddings.csv',
            'roi_scored_examples': 'roi_crops/roi_scored_examples.csv',
            'learned_roi_features': 'burden_model/learned_roi/feature_sets/learned_roi_features.csv',
        },
        'identity_columns': [
            'roi_row_id',
            'subject_id',
            'subject_image_id',
            'glomerulus_id',
            'roi_image_path',
        ],
        'required_provenance_columns': [
            'roi_geometry_contract_version',
            'roi_preprocessing_version',
            'roi_threshold_policy',
            'roi_status',
            'artifact_provenance_id',
            'embedding_model_id',
            'embedding_preprocessing_version',
            'feature_lineage_json',
        ],
        'feature_allowlist': [
            'embedding_0000',
            'embedding_0001',
            'embedding_0002',
            'embedding_0003',
            'roi_area',
            'roi_fill_fraction',
            'roi_mean_intensity',
            'roi_openness_score',
            'roi_component_count',
            'learned_roi_texture_signal',
            'learned_roi_qc_signal',
        ],
        'feature_spaces': {
            'encoder_standardized': {'prefixes': ['embedding_'], 'scaling': 'standard'},
            'encoder_pca': {
                'source_space': 'encoder_standardized',
                'pca_components': 2,
            },
            'roi_qc_standardized': {
                'columns': [
                    'roi_area',
                    'roi_fill_fraction',
                    'roi_mean_intensity',
                    'roi_openness_score',
                    'roi_component_count',
                ],
                'scaling': 'standard',
            },
            'learned_roi_standardized': {
                'prefixes': ['learned_roi_'],
                'scaling': 'standard',
            },
        },
        'clustering': {
            'methods': ['kmeans', 'gaussian_mixture', 'hdbscan'],
            'cluster_counts': [2, 3],
            'random_state': 17,
        },
        'stability': {
            'resampling_unit': 'subject_id',
            'n_resamples': 3,
            'random_state': 19,
        },
        'interpretation_thresholds': {
            'min_rows': 3,
            'min_subjects': 2,
            'min_stability': 0.6,
            'max_source_fraction': 0.8,
            'max_artifact_fraction': 0.4,
            'min_grade_association_strength': 0.25,
            'max_missing_asset_fraction': 0.2,
        },
        'review': {
            'max_representatives_per_cluster': 2,
            'include_severe_boundary_queue': True,
            'exclude_same_subject_neighbors': True,
        },
    }


def test_label_blinding_blocks_denied_allowlist_columns_and_writes_audit(tmp_path):
    quantification_root, _ = _write_atlas_inputs(tmp_path)
    config = _base_atlas_config(quantification_root)
    config['feature_allowlist'] = [*config['feature_allowlist'], 'score', 'cohort_id']

    _run_atlas(config)

    atlas_root = _atlas_output_root(quantification_root)
    verdict = _read_json(atlas_root / 'summary' / 'atlas_verdict.json')
    audit = _read_json(atlas_root / 'diagnostics' / 'label_blinding_audit.json')

    assert verdict['workflow_status'] == 'failed'
    assert any(
        blocker in verdict['blockers']
        for blocker in ['label_leakage', 'source_leakage']
    )
    assert {'score', 'cohort_id'}.issubset(set(audit['denied_columns']))
    assert {'score', 'cohort_id'}.issubset(set(audit['excluded_label_like_columns']))
    assert audit['denied_column_entered_feature_matrix'] is False


@pytest.mark.parametrize(
    ('missing_column', 'expected_blocker'),
    [
        ('subject_id', 'missing_identity_columns'),
        ('roi_geometry_contract_version', 'stale_or_incomplete_provenance'),
        ('artifact_provenance_id', 'stale_or_incomplete_provenance'),
    ],
)
def test_missing_identity_or_provenance_fails_closed(
    tmp_path, missing_column, expected_blocker
):
    quantification_root, _ = _write_atlas_inputs(
        tmp_path, missing_columns=(missing_column,)
    )

    _run_atlas(_base_atlas_config(quantification_root))

    atlas_root = _atlas_output_root(quantification_root)
    verdict = _read_json(atlas_root / 'summary' / 'atlas_verdict.json')
    assert verdict['workflow_status'] == 'failed'
    assert expected_blocker in verdict['blockers']
    assert missing_column in json.dumps(verdict)
    assert not (atlas_root / 'clusters' / 'cluster_assignments.csv').exists()


def test_unapproved_feature_lineage_fails_before_clustering(tmp_path):
    quantification_root, _ = _write_atlas_inputs(
        tmp_path, unapproved_feature_lineage=True
    )

    _run_atlas(_base_atlas_config(quantification_root))

    atlas_root = _atlas_output_root(quantification_root)
    verdict = _read_json(atlas_root / 'summary' / 'atlas_verdict.json')
    audit = _read_json(atlas_root / 'diagnostics' / 'label_blinding_audit.json')
    assert verdict['workflow_status'] == 'failed'
    assert 'unapproved_feature_lineage' in verdict['blockers']
    assert 'embedding_0003' in json.dumps(audit['unapproved_feature_lineage'])
    assert not (atlas_root / 'clusters' / 'cluster_assignments.csv').exists()


def test_feature_space_and_method_availability_schemas(tmp_path):
    quantification_root, _ = _write_atlas_inputs(tmp_path)

    _run_atlas(_base_atlas_config(quantification_root))

    atlas_root = _atlas_output_root(quantification_root)
    manifest = _read_json(atlas_root / 'feature_space' / 'feature_space_manifest.json')
    feature_spaces = {
        entry['feature_space_id']: entry for entry in manifest['feature_spaces']
    }
    assert {
        'encoder_standardized',
        'encoder_pca',
        'roi_qc_standardized',
        'learned_roi_standardized',
    }.issubset(feature_spaces)
    for feature_space in feature_spaces.values():
        assert {
            'row_count',
            'subject_count',
            'feature_count',
            'missing_value_count',
            'nonfinite_count',
            'zero_variance_count',
            'near_zero_variance_count',
            'scaling_policy',
            'package_versions',
        }.issubset(feature_space)
    assert feature_spaces['encoder_pca']['pca_policy']['component_count'] == 2

    availability = _read_json(atlas_root / 'diagnostics' / 'method_availability.json')
    methods = {entry['method_id']: entry for entry in availability['methods']}
    assert {
        'sklearn_pca',
        'kmeans',
        'gaussian_mixture',
        'nearest_neighbors',
        'hdbscan',
        'umap_learn',
    }.issubset(methods)
    for method in methods.values():
        assert {'available', 'method_role', 'fit_eligible', 'failure_reason'}.issubset(
            method
        )
    assert methods['hdbscan']['method_role'] == 'optional_clustering'
    if not methods['hdbscan']['available']:
        assert methods['hdbscan']['fit_eligible'] is False
        assert methods['hdbscan']['failure_reason']


def test_cluster_assignments_stability_and_posthoc_schema(tmp_path):
    quantification_root, _ = _write_atlas_inputs(tmp_path)

    _run_atlas(_base_atlas_config(quantification_root))

    atlas_root = _atlas_output_root(quantification_root)
    assignments = pd.read_csv(atlas_root / 'clusters' / 'cluster_assignments.csv')
    assert {
        'roi_row_id',
        'subject_id',
        'feature_space_id',
        'method_id',
        'cluster_id',
        'assignment_confidence',
        'assignment_distance',
        'is_outlier_or_noise',
    }.issubset(assignments.columns)
    assert not {
        'severity_label',
        'predicted_score',
        'candidate_severity_like_group',
    }.intersection(assignments.columns)

    stability = _read_json(atlas_root / 'stability' / 'cluster_stability.json')
    assert stability['resampling_unit'] == 'subject_id'
    assert stability['row_count'] == len(assignments['roi_row_id'].drop_duplicates())
    assert stability['subject_count'] == assignments['subject_id'].nunique()
    for entry in stability['results']:
        assert {
            'feature_space_id',
            'method_id',
            'stability_metrics',
            'non_estimable_reasons',
        }.issubset(entry)

    posthoc = _read_json(
        atlas_root / 'diagnostics' / 'cluster_posthoc_diagnostics.json'
    )
    assert {
        'original_score_distribution',
        'severe_nonsevere_distribution',
        'cohort_source_distribution',
        'roi_qc_summaries',
        'mask_roi_adequacy',
        'cluster_interpretations',
    }.issubset(posthoc)
    allowed_labels = {
        'candidate_morphology_group',
        'candidate_severity_like_group',
        'artifact_or_quality_group',
        'source_sensitive_group',
        'unstable_group',
        'insufficient_support',
    }
    assert {
        row['interpretation_label'] for row in posthoc['cluster_interpretations']
    }.issubset(allowed_labels)


@pytest.mark.parametrize(
    ('threshold_key', 'threshold_value', 'expected_blocker'),
    [
        ('min_subjects', 99, 'min_subjects'),
        ('min_stability', 1.01, 'min_stability'),
        ('max_source_fraction', 0.01, 'max_source_fraction'),
        ('max_artifact_fraction', 0.0, 'max_artifact_fraction'),
        ('min_grade_association_strength', 1.01, 'min_grade_association_strength'),
        ('max_missing_asset_fraction', 0.0, 'max_missing_asset_fraction'),
    ],
)
def test_posthoc_interpretation_gates_block_severity_like_labels(
    tmp_path, threshold_key, threshold_value, expected_blocker
):
    quantification_root, _ = _write_atlas_inputs(tmp_path)
    config = _base_atlas_config(quantification_root)
    config['interpretation_thresholds'] = copy.deepcopy(
        config['interpretation_thresholds']
    )
    config['interpretation_thresholds'][threshold_key] = threshold_value

    _run_atlas(config)

    posthoc = _read_json(
        _atlas_output_root(quantification_root)
        / 'diagnostics'
        / 'cluster_posthoc_diagnostics.json'
    )
    severity_like = [
        row
        for row in posthoc['cluster_interpretations']
        if row['interpretation_label'] == 'candidate_severity_like_group'
    ]
    assert severity_like == []
    assert expected_blocker in json.dumps(posthoc['cluster_interpretations'])


def test_review_queue_first_read_artifacts_and_no_label_overrides(tmp_path):
    quantification_root, _ = _write_atlas_inputs(tmp_path)

    _run_atlas(_base_atlas_config(quantification_root))

    atlas_root = _atlas_output_root(quantification_root)
    for path in [
        atlas_root / 'INDEX.md',
        atlas_root / 'summary' / 'atlas_verdict.json',
        atlas_root / 'summary' / 'atlas_summary.md',
        atlas_root / 'summary' / 'artifact_manifest.json',
        atlas_root / 'evidence' / 'embedding_atlas_review.html',
        atlas_root / 'review_queue' / 'atlas_adjudication_queue.csv',
    ]:
        assert path.exists(), path

    queue = pd.read_csv(atlas_root / 'review_queue' / 'atlas_adjudication_queue.csv')
    assert {
        'review_priority',
        'reason_code',
        'roi_row_id',
        'cluster_id',
        'nearest_neighbor_evidence',
        'original_score',
        'reviewed_anchor_evidence',
        'roi_image_path',
        'roi_path_provenance',
    }.issubset(queue.columns)
    assert not list(atlas_root.rglob('*label*override*'))
    assert 'descriptive morphology clustering' in (atlas_root / 'INDEX.md').read_text(
        encoding='utf-8'
    )


def test_run_config_dispatches_label_free_roi_embedding_atlas(tmp_path):
    quantification_root, _ = _write_atlas_inputs(tmp_path)
    runtime_root = tmp_path / 'runtime'
    relative_quantification_root = (
        Path('output') / 'quantification_results' / 'fixture_run'
    )
    target_root = runtime_root / relative_quantification_root
    target_root.parent.mkdir(parents=True, exist_ok=True)
    quantification_root.rename(target_root)
    config_path = tmp_path / 'atlas_config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                **_base_atlas_config(relative_quantification_root),
                'workflow': 'label_free_roi_embedding_atlas',
                'run': {
                    'name': 'fixture_label_free_roi_embedding_atlas',
                    'runtime_root_default': str(runtime_root),
                },
            }
        ),
        encoding='utf-8',
    )

    run_config(config_path)

    verdict = _read_json(
        _atlas_output_root(target_root) / 'summary' / 'atlas_verdict.json'
    )
    assert verdict['workflow_status'] == 'completed'


def test_atlas_writes_no_generated_artifacts_under_repo_root(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_candidates = [
        repo_root / 'embedding_atlas',
        repo_root / 'burden_model' / 'embedding_atlas',
        repo_root / 'summary' / 'atlas_verdict.json',
    ]
    before = {path: path.exists() for path in repo_root_candidates}
    quantification_root, _ = _write_atlas_inputs(tmp_path)

    _run_atlas(_base_atlas_config(quantification_root))

    assert _atlas_output_root(quantification_root).exists()
    after = {path: path.exists() for path in repo_root_candidates}
    assert after == before


def test_supervised_learned_roi_quantification_does_not_require_atlas_artifacts(
    tmp_path,
):
    quantification_root, frame = _write_atlas_inputs(
        tmp_path, include_learned_roi=False
    )
    burden_root = quantification_root / 'burden_model'
    atlas_root = burden_root / 'embedding_atlas'
    assert not atlas_root.exists()

    artifacts = evaluate_learned_roi_quantification(frame, burden_root, n_splits=3)

    assert artifacts['learned_roi_estimator_verdict'].exists()
    assert artifacts['learned_roi_features'].exists()
    assert not atlas_root.exists()
