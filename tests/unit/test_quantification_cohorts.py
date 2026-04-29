from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eq.quantification.cohorts import (
    apply_cohort_admission_policy,
    apply_discovery_reconciliation,
    apply_dox_mask_quality_approval,
    archive_retired_quantification_input_tree,
    build_dox_mask_quality_audit,
    build_dox_scored_only_resolution_audit,
    build_mr_concordance_workflow,
    build_mr_runtime_cohort,
    build_predicted_roi_grading_inputs,
    build_predicted_roi_grading_inputs_from_manifest,
    canonical_manifest_columns,
    classify_foreign_rows,
    enrich_unified_manifest,
    harmonize_localized_cohort,
    mr_concordance_metrics,
    reduce_mr_replicates,
    validate_segmentation_transport_inputs,
    validate_unified_manifest,
    verify_mapping_bundle,
    write_mr_inference_contract,
)


def _write_file(path: Path, payload: bytes = b'data') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_image(path: Path, shape: tuple[int, int, int] = (16, 16, 3)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(path)
    return path


def _write_mask(path: Path, shape: tuple[int, int] = (16, 16)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:12, 4:12] = 255
    Image.fromarray(mask).save(path)
    return path


def test_manifest_schema_and_minimal_input_validation():
    schema = canonical_manifest_columns()

    assert 'cohort_id' in schema['human_required']
    assert 'harmonized_id' in schema['pipeline_generated']
    assert 'image_sha256' in schema['pipeline_generated']

    manifest = pd.DataFrame(
        [{'cohort_id': 'vegfri_dox', 'score': 1.5, 'source_sample_id': 'M1'}]
    )

    validation = validate_unified_manifest(manifest)

    assert validation.passed


def test_enrichment_generates_hashes_and_minimal_harmonized_ids(tmp_path):
    runtime_root = tmp_path / 'runtime'
    image = _write_file(
        runtime_root / 'raw_data/cohorts/vegfri_dox/images/M1/M1_Image0.jpg'
    )
    mask = _write_file(
        runtime_root / 'raw_data/cohorts/vegfri_dox/masks/M1/M1_Image0_mask.png'
    )
    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'vegfri_dox',
                'image_path': 'raw_data/cohorts/vegfri_dox/images/M1/M1_Image0.jpg',
                'mask_path': 'raw_data/cohorts/vegfri_dox/masks/M1/M1_Image0_mask.png',
                'score': 2.0,
                'source_image_name': image.name,
                'source_sample_id': 'M1',
                'source_score_row': 'task-1',
            }
        ]
    )

    enriched = enrich_unified_manifest(manifest, runtime_root=runtime_root)
    validation = validate_unified_manifest(enriched, enriched=True)

    assert validation.passed
    assert enriched.loc[0, 'admission_status'] == 'admitted'
    assert enriched.loc[0, 'verification_status'] == 'passed'
    assert enriched.loc[0, 'lane_assignment'] == 'manual_mask_external'
    assert enriched.loc[0, 'harmonized_id'] == 'vegfri_dox__m1_image0_jpg'
    assert len(enriched.loc[0, 'image_sha256']) == 64
    assert len(enriched.loc[0, 'mask_sha256']) == 64


def test_foreign_mixed_export_rows_are_explicit_not_merged():
    manifest = pd.DataFrame(
        [
            {'cohort_id': 'vegfri_dox', 'source_sample_id': 'M1', 'score': 1.0},
            {'cohort_id': 'vegfri_dox', 'source_sample_id': 'Lauren42', 'score': 1.0},
        ]
    )

    classified = classify_foreign_rows(manifest, ['M', 'S', 'D', 'L'])

    assert classified.loc[0, 'admission_status'] != 'foreign'
    assert classified.loc[1, 'admission_status'] == 'foreign'
    assert classified.loc[1, 'exclusion_reason'] == 'foreign_mixed_export_row'


def test_harmonization_copies_assets_without_source_paths_in_manifest(tmp_path):
    runtime_root = tmp_path / 'runtime'
    source_image = _write_file(tmp_path / 'source' / 'A image.JPG')
    source_mask = _write_file(tmp_path / 'source' / 'A mask.PNG')
    records = pd.DataFrame(
        [
            {
                'source_image_path': str(source_image),
                'source_mask_path': str(source_mask),
                'source_sample_id': 'A1',
                'source_score_row': 'row-1',
                'score': 1.0,
            }
        ]
    )

    localized = harmonize_localized_cohort(
        records,
        'example',
        runtime_root=runtime_root,
        source_audit_path=runtime_root
        / 'raw_data/cohorts/example/metadata/source_audit.csv',
    )

    assert 'source_image_path' not in localized.columns
    assert 'source_mask_path' not in localized.columns
    assert source_image.exists()
    assert source_mask.exists()
    assert (runtime_root / localized.loc[0, 'image_path']).exists()
    assert (runtime_root / localized.loc[0, 'mask_path']).exists()
    assert (
        runtime_root / 'raw_data/cohorts/example/metadata/source_audit.csv'
    ).exists()


def test_harmonization_keeps_missing_source_image_unresolved(tmp_path):
    runtime_root = tmp_path / 'runtime'
    records = pd.DataFrame(
        [
            {
                'source_image_path': '',
                'source_image_name': 'missing.tif',
                'source_sample_id': 'missing',
                'source_score_row': 'row-1',
                'score': 1.0,
            }
        ]
    )

    localized = harmonize_localized_cohort(
        records, 'example', runtime_root=runtime_root
    )

    assert localized.loc[0, 'image_path'] == ''
    assert localized.loc[0, 'admission_status'] == 'unresolved'
    assert localized.loc[0, 'exclusion_reason'] == 'source_image_missing'


def test_mapping_verification_rejects_conflicting_duplicate_scores(tmp_path):
    runtime_root = tmp_path / 'runtime'
    _write_file(runtime_root / 'raw_data/cohorts/example/images/A1.jpg')
    base = pd.DataFrame(
        [
            {
                'cohort_id': 'example',
                'image_path': 'raw_data/cohorts/example/images/A1.jpg',
                'score': 1.0,
                'source_image_name': 'A1.jpg',
                'source_sample_id': 'A1',
                'source_score_row': 'row-1',
            },
            {
                'cohort_id': 'example',
                'image_path': 'raw_data/cohorts/example/images/A1.jpg',
                'score': 2.0,
                'source_image_name': 'A1.jpg',
                'source_sample_id': 'A1',
                'source_score_row': 'row-2',
            },
        ]
    )
    enriched = enrich_unified_manifest(base, runtime_root=runtime_root)

    verified = verify_mapping_bundle(enriched, runtime_root=runtime_root)

    assert set(verified['admission_status']) == {'excluded'}
    assert all(
        'conflicting_duplicate_scores' in reason
        for reason in verified['exclusion_reason']
    )


def test_missing_image_rows_remain_row_level_evidence(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'vegfri_dox',
                'image_path': '',
                'score': 1.0,
                'source_sample_id': 'T1',
                'source_score_row': 'task-1',
                'admission_status': 'foreign',
                'verification_status': 'excluded',
                'exclusion_reason': 'foreign_mixed_export_row',
            },
            {
                'cohort_id': 'vegfri_dox',
                'image_path': '',
                'score': 1.0,
                'source_sample_id': 'T2',
                'source_score_row': 'task-2',
                'admission_status': 'foreign',
                'verification_status': 'excluded',
                'exclusion_reason': 'foreign_mixed_export_row',
            },
            {
                'cohort_id': 'vegfri_dox',
                'image_path': '',
                'score': 2.0,
                'source_sample_id': 'M99',
                'source_score_row': 'task-3',
                'admission_status': 'unresolved',
                'exclusion_reason': 'not_materialized_in_decoded_brushlabel_surface',
            },
        ]
    )

    enriched = enrich_unified_manifest(manifest, runtime_root=tmp_path)
    verified = verify_mapping_bundle(enriched, runtime_root=tmp_path)

    assert len(verified) == 3
    assert verified['source_score_row'].tolist() == ['task-1', 'task-2', 'task-3']
    assert verified['admission_status'].tolist() == ['foreign', 'foreign', 'unresolved']


def test_mr_policy_and_predicted_roi_outputs_remain_separate(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'vegfri_mr',
                'lane_assignment': 'scored_only',
                'admission_status': 'admitted',
            },
            {
                'cohort_id': 'vegfri_dox',
                'lane_assignment': 'scored_only',
                'admission_status': 'admitted',
                'manifest_row_id': 'dox-1',
                'harmonized_id': 'vegfri_dox__m1',
                'image_path': 'raw_data/cohorts/vegfri_dox/images/M1.jpg',
                'score': 1.5,
            },
        ]
    )

    policy = apply_cohort_admission_policy(manifest)
    predicted_path = build_predicted_roi_grading_inputs(
        policy,
        tmp_path / 'output/quantification_results/vegfri_dox/predicted_roi',
        segmentation_artifact='segmentation.pkl',
    )
    predicted = pd.read_csv(predicted_path)

    assert policy.loc[0, 'lane_assignment'] == 'mr_concordance_only'
    assert policy.loc[0, 'admission_status'] == 'evaluation_only'
    assert predicted['cohort_id'].tolist() == ['vegfri_dox']
    assert predicted['artifact_family'].tolist() == ['predicted_roi_grading']


def test_predicted_roi_builder_consumes_manifest_path_without_source_roots(tmp_path):
    manifest_path = tmp_path / 'manifest.csv'
    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'external',
                'lane_assignment': 'scored_only',
                'admission_status': 'pending_transport_audit',
                'manifest_row_id': 'row-1',
                'harmonized_id': 'external__row_1',
                'image_path': 'raw_data/cohorts/external/images/row_1.tif',
                'score': 1.0,
            }
        ]
    )
    manifest.to_csv(manifest_path, index=False)

    predicted_path = build_predicted_roi_grading_inputs_from_manifest(
        manifest_path, tmp_path / 'output', segmentation_artifact='segmentation.pkl'
    )
    predicted = pd.read_csv(predicted_path)

    assert predicted['manifest_row_id'].tolist() == ['row-1']
    assert 'source_image_path' not in predicted.columns


def test_discovery_reconciliation_keeps_recoverable_rows_pending():
    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'vegfri_mr',
                'join_status': 'failed',
                'verification_status': 'pending_discovery',
                'admission_status': 'unresolved',
                'exclusion_reason': 'missing_image_path',
                'discovery_surfaces': 'mr_workbook',
            },
            {
                'cohort_id': 'vegfri_mr',
                'join_status': 'failed',
                'verification_status': 'pending_discovery',
                'admission_status': 'pending_discovery',
                'exclusion_reason': '',
                'discovery_surfaces': 'mr_workbook;external_drive_whole_field_tiffs',
            },
        ]
    )

    reconciled = apply_discovery_reconciliation(manifest)

    assert reconciled.loc[0, 'admission_status'] == 'pending_discovery'
    assert 'external_drive_whole_field_tiffs' in reconciled.loc[0, 'exclusion_reason']
    assert reconciled.loc[1, 'admission_status'] == 'unresolved'


def test_manual_mask_external_dox_rows_remain_training_admitted():
    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'vegfri_dox',
                'lane_assignment': 'manual_mask_external',
                'admission_status': 'admitted',
                'exclusion_reason': '',
            }
        ]
    )

    policy = apply_cohort_admission_policy(manifest)

    assert policy.loc[0, 'admission_status'] == 'admitted'
    assert policy.loc[0, 'exclusion_reason'] == ''


def test_dox_mask_quality_audit_records_provenance_for_admitted_manual_external_rows(
    tmp_path,
):
    runtime_root = tmp_path / 'runtime'
    image = _write_image(
        runtime_root / 'raw_data/cohorts/vegfri_dox/images/M1/M1_Image0.jpg'
    )
    mask = _write_mask(
        runtime_root / 'raw_data/cohorts/vegfri_dox/masks/M1/M1_Image0_mask.png'
    )
    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'vegfri_dox',
                'image_path': 'raw_data/cohorts/vegfri_dox/images/M1/M1_Image0.jpg',
                'mask_path': 'raw_data/cohorts/vegfri_dox/masks/M1/M1_Image0_mask.png',
                'score': 1.0,
                'source_image_name': image.name,
                'source_sample_id': 'M1',
                'source_score_row': 'task-1',
            }
        ]
    )
    enriched = enrich_unified_manifest(manifest, runtime_root=runtime_root)
    policy = apply_cohort_admission_policy(
        verify_mapping_bundle(enriched, runtime_root=runtime_root)
    )

    outputs = build_dox_mask_quality_audit(policy, runtime_root=runtime_root)
    audited = pd.read_csv(outputs['audit'])
    approved = apply_dox_mask_quality_approval(policy, runtime_root=runtime_root)

    assert audited['mask_quality_decision'].tolist() == ['approved']
    assert approved.loc[0, 'admission_status'] == 'admitted'
    assert approved.loc[0, 'mask_quality_review_status'] == 'approved'
    assert (
        outputs['panel_dir']
        == runtime_root / 'raw_data/cohorts/vegfri_dox/metadata/mask_quality_panels'
    )
    assert list(outputs['panel_dir'].glob('*.png'))


def test_dox_scored_only_resolution_audit_defines_clean_smoke_set(tmp_path):
    runtime_root = tmp_path / 'runtime'
    upload_root = tmp_path / 'label-studio' / 'media' / 'upload' / '1'
    clean_image = _write_image(upload_root / 'abc123-M1_Image0.jpg')
    duplicate_image = _write_image(upload_root / 'def456-T19_Image0.jpg')
    manifest = pd.DataFrame(
        [
            {
                'manifest_row_id': 'clean-row',
                'cohort_id': 'vegfri_dox',
                'lane_assignment': 'scored_only',
                'admission_status': 'unresolved',
                'join_status': 'failed',
                'exclusion_reason': 'missing_image_path;missing_image_hash',
                'source_image_name': 'M1_Image0.jpg',
                'source_sample_id': 'M1',
                'source_score_row': 'task-clean',
                'score': 1.0,
            },
            {
                'manifest_row_id': 'duplicate-a',
                'cohort_id': 'vegfri_dox',
                'lane_assignment': 'scored_only',
                'admission_status': 'foreign',
                'join_status': 'foreign_row',
                'exclusion_reason': 'foreign_mixed_export_row',
                'source_image_name': 'T19_Image0.jpg',
                'source_sample_id': 'T19',
                'source_score_row': 'task-dup-a',
                'score': 0.5,
            },
            {
                'manifest_row_id': 'duplicate-b',
                'cohort_id': 'vegfri_dox',
                'lane_assignment': 'scored_only',
                'admission_status': 'foreign',
                'join_status': 'foreign_row',
                'exclusion_reason': 'foreign_mixed_export_row',
                'source_image_name': 'T19_Image0.jpg',
                'source_sample_id': 'T19',
                'source_score_row': 'task-dup-b',
                'score': 1.0,
            },
            {
                'manifest_row_id': 'missing-score',
                'cohort_id': 'vegfri_dox',
                'lane_assignment': 'scored_only',
                'admission_status': 'foreign',
                'join_status': 'foreign_row',
                'exclusion_reason': 'foreign_mixed_export_row',
                'source_image_name': 'missing_score.jpg',
                'source_sample_id': 'T20',
                'source_score_row': 'task-missing-score',
                'score': np.nan,
            },
        ]
    )

    outputs = build_dox_scored_only_resolution_audit(
        manifest, runtime_root=runtime_root, upload_root=upload_root.parent
    )
    audit = pd.read_csv(outputs['audit'])
    smoke = pd.read_csv(outputs['smoke_manifest'])

    assert outputs['counts']['scored_only_rows'] == 4
    assert outputs['counts']['clean_smoke_rows'] == 1
    assert smoke['manifest_row_id'].tolist() == ['clean-row']
    localized_path = runtime_root / smoke.loc[0, 'image_path']
    assert localized_path.exists()
    assert localized_path.read_bytes() == clean_image.read_bytes()
    assert smoke['labelstudio_image_path'].tolist() == [str(clean_image)]
    assert smoke['mask_path'].fillna('').tolist() == ['']
    updated_manifest = pd.read_csv(runtime_root / 'raw_data/cohorts/manifest.csv')
    clean_row = updated_manifest[
        updated_manifest['manifest_row_id'].astype(str) == 'clean-row'
    ].iloc[0]
    assert clean_row['eligible_dox_scored_no_mask_smoke'] is True or bool(
        clean_row['eligible_dox_scored_no_mask_smoke']
    )
    assert clean_row['image_path'] == smoke.loc[0, 'image_path']
    assert clean_row['mask_path'] == '' or pd.isna(clean_row['mask_path'])
    duplicate_name = duplicate_image.name.split('-', 1)[1]
    duplicate_rows = audit[audit['source_image_name'] == duplicate_name]
    assert duplicate_rows['duplicate_source_image_name'].all()
    assert duplicate_rows['conflicting_scores_for_source_image_name'].all()
    assert audit.loc[
        audit['manifest_row_id'] == 'missing-score', 'missing_score'
    ].item()


def test_transport_validation_blocks_missing_degenerate_or_invalid_outputs():
    manifest = pd.DataFrame(
        [
            {
                'manifest_row_id': 'missing',
                'admission_status': 'pending_transport_audit',
            },
            {
                'manifest_row_id': 'degenerate',
                'admission_status': 'pending_transport_audit',
            },
            {'manifest_row_id': 'ok', 'admission_status': 'pending_transport_audit'},
        ]
    )
    outputs = pd.DataFrame(
        [
            {
                'manifest_row_id': 'degenerate',
                'segmentation_status': 'ok',
                'accepted_roi_count': 0,
                'grading_status': 'ok',
            },
            {
                'manifest_row_id': 'ok',
                'segmentation_status': 'ok',
                'accepted_roi_count': 2,
                'grading_status': 'ok',
            },
        ]
    )

    validated = validate_segmentation_transport_inputs(manifest, outputs)

    assert validated.loc[0, 'transport_failure_reason'] == 'missing_segmentation_output'
    assert (
        validated.loc[1, 'transport_failure_reason'] == 'degenerate_segmentation_output'
    )
    assert validated.loc[2, 'transport_status'] == 'passed'


def test_mr_replicates_reduce_to_image_level_sidecar(tmp_path):
    workbook = tmp_path / 'mr_scores.xlsx'
    sidecar = tmp_path / 'mr_replicates.csv'
    raw = pd.DataFrame([['image_a', 'image_b'], [1.0, 2.0], [3.0, 2.0], [None, 4.0]])
    raw.to_excel(workbook, index=False, header=False)

    reduced = reduce_mr_replicates(workbook, output_sidecar=sidecar)
    sidecar_rows = pd.read_csv(sidecar)

    assert reduced['cohort_id'].tolist() == ['vegfri_mr', 'vegfri_mr']
    assert reduced['score_reduction_method'].unique().tolist() == [
        'median_within_image_replicates'
    ]
    assert reduced['score'].tolist() == [2.0, 2.0]
    assert reduced['replicate_count'].tolist() == [2, 3]
    assert len(sidecar_rows) == 5


def test_mr_runtime_cohort_uses_workbook_medians_and_localized_tiffs(tmp_path):
    runtime_root = tmp_path / 'runtime'
    image_root = tmp_path / 'mr_images'
    _write_file(image_root / 'batch1' / '7007.tif')
    _write_file(image_root / 'batch1' / '7010.tif')
    workbook = tmp_path / 'mr_scores.xlsx'
    raw = pd.DataFrame(
        [['SampleID \\ Replicate', 7007.0, 7010.0], [1, 0.0, 1.0], [2, 2.0, 3.0]]
    )
    raw.to_excel(workbook, sheet_name='Batch1', index=False, header=False)

    cohort = build_mr_runtime_cohort(
        runtime_root=runtime_root, workbook_path=workbook, image_root=image_root
    )
    enriched = enrich_unified_manifest(cohort, runtime_root=runtime_root)
    policy = apply_cohort_admission_policy(
        verify_mapping_bundle(enriched, runtime_root=runtime_root)
    )

    assert cohort['source_sample_id'].tolist() == ['7007', '7010']
    assert cohort['score'].tolist() == [1.0, 2.0]
    assert all((runtime_root / path).exists() for path in cohort['image_path'])
    assert policy['admission_status'].tolist() == ['evaluation_only', 'evaluation_only']
    assert (
        runtime_root
        / 'raw_data/cohorts/vegfri_mr/metadata/mr_acquisition_metadata.json'
    ).exists()


def test_mr_inference_contract_and_concordance_workflow(tmp_path):
    contract_path = write_mr_inference_contract(tmp_path / 'contract')
    contract_text = contract_path.read_text()

    manifest = pd.DataFrame(
        [
            {
                'cohort_id': 'vegfri_mr',
                'admission_status': 'evaluation_only',
                'manifest_row_id': 'mr-1',
                'harmonized_id': 'vegfri_mr__one',
                'source_sample_id': 'one',
                'score': 1.0,
            },
            {
                'cohort_id': 'vegfri_mr',
                'admission_status': 'evaluation_only',
                'manifest_row_id': 'mr-2',
                'harmonized_id': 'vegfri_mr__two',
                'source_sample_id': 'two',
                'score': 2.0,
            },
        ]
    )
    inferred = pd.DataFrame(
        [
            {'manifest_row_id': 'mr-1', 'roi_grade': 1.0, 'component_area': 100},
            {'manifest_row_id': 'mr-1', 'roi_grade': 3.0, 'component_area': 120},
            {'manifest_row_id': 'mr-1', 'roi_grade': 0.0, 'component_area': 10},
            {'manifest_row_id': 'mr-2', 'roi_grade': 2.0, 'component_area': 20},
        ]
    )

    outputs = build_mr_concordance_workflow(
        manifest, inferred, tmp_path / 'mr_concordance', min_component_area=64
    )
    image_level = pd.read_csv(outputs['image_level_concordance'])

    assert 'whole_field_tiff_tiling' in contract_text
    assert image_level.loc[0, 'inferred_image_median'] == 2.0
    assert image_level.loc[0, 'accepted_roi_count'] == 2
    assert image_level.loc[1, 'concordance_status'] == 'non_evaluable'
    assert outputs['metrics'].exists()


def test_archive_retired_quantification_input_tree_writes_marker(tmp_path):
    old_tree = tmp_path / 'old_quantification_inputs'
    old_tree.mkdir()

    marker = archive_retired_quantification_input_tree(old_tree)

    assert marker.exists()
    assert 'raw_data/cohorts/<cohort_id>/' in marker.read_text()


def test_mr_concordance_metrics_are_fixed():
    metrics = mr_concordance_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 2.0])

    assert metrics['n'] == 3.0
    assert metrics['mae'] == 2 / 3
    assert metrics['exact_agreement'] == 1 / 3
    assert metrics['within_one_step_agreement'] == 1.0
