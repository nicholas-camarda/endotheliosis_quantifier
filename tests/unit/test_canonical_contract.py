from pathlib import Path

import pandas as pd

from eq.data_management.backup_utils import create_backup_snapshot
from eq.data_management.canonical_contract import (
    apply_migration_plan,
    build_migration_plan,
    parse_subject_image_filename,
    validate_project_contract,
)
from eq.data_management.canonical_naming import (
    parse_subject_image_stem as parse_legacy_compatible_stem,
)


def test_parse_subject_image_filename_supports_canonical_and_legacy():
    parsed_new = parse_subject_image_filename('T19-1.jpg')
    assert parsed_new is not None
    assert parsed_new.naming == 'canonical'
    assert parsed_new.subject_image_id == 'T19-1'

    parsed_old = parse_subject_image_filename('T19_Image99_mask.jpg')
    assert parsed_old is not None
    assert parsed_old.naming == 'legacy'
    assert parsed_old.legacy_stem == 'T19_Image99'


def test_canonical_parsers_reject_trailing_junk_and_accept_non_t_subjects():
    contract_parsed = parse_subject_image_filename('M1-2.jpg')
    naming_parsed = parse_legacy_compatible_stem('M1-2')

    assert contract_parsed is not None
    assert contract_parsed.subject_prefix == 'M1'
    assert naming_parsed is not None
    assert naming_parsed.subject_prefix == 'M1'
    assert parse_subject_image_filename('M1-2-extra.jpg') is None
    assert parse_legacy_compatible_stem('M1-2-extra') is None


def test_build_migration_plan_uses_mapping_and_apply(tmp_path: Path):
    project_dir = tmp_path / 'project'
    image_dir = project_dir / 'images' / 'T19'
    mask_dir = project_dir / 'masks' / 'T19'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / 'T19_Image99.jpg').write_bytes(b'image')
    (mask_dir / 'T19_Image99_mask.jpg').write_bytes(b'mask')

    metadata_df = pd.DataFrame(
        {'subject_id': ['T19-1'], 'glomerulus_id': [1], 'score': [0.5]}
    )
    mapping_file = tmp_path / 'mapping.csv'
    pd.DataFrame(
        {'legacy_stem': ['T19_Image99'], 'subject_image_id': ['T19-1']}
    ).to_csv(mapping_file, index=False)

    plan = build_migration_plan(project_dir, metadata_df, mapping_file=mapping_file)
    assert set(plan['status']) == {'rename_required'}

    applied = apply_migration_plan(plan)
    assert set(applied['status']) == {'renamed'}
    assert (project_dir / 'images' / 'T19' / 'T19-1.jpg').exists()
    assert (project_dir / 'masks' / 'T19' / 'T19-1_mask.jpg').exists()


def test_validate_project_contract_flags_legacy_and_missing_pairs(tmp_path: Path):
    project_dir = tmp_path / 'project'
    image_dir = project_dir / 'images' / 'T19'
    mask_dir = project_dir / 'masks' / 'T19'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / 'T19_Image0.jpg').write_bytes(b'image')
    (mask_dir / 'T19-2_mask.jpg').write_bytes(b'mask')

    metadata_df = pd.DataFrame(
        {'subject_id': ['T19-1', 'T19-2'], 'glomerulus_id': [1, 1], 'score': [0.0, 1.0]}
    )
    report = validate_project_contract(project_dir, metadata_df, require_canonical=True)
    assert report['overall_status'] == 'FAIL'
    assert report['masks_without_images'] == ['T19-2']
    assert any('Legacy image filename' in error for error in report['errors'])


def test_create_backup_snapshot_writes_manifests(tmp_path: Path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    (source_dir / 'a.txt').write_text('alpha', encoding='utf-8')

    artifact = create_backup_snapshot(
        [source_dir], tmp_path / 'backup', 'demo', timestamp='20260402_130000'
    )
    assert artifact.backup_root.exists()
    assert artifact.manifest_files.exists()
    assert artifact.manifest_sha256.exists()
    assert 'source/a.txt' in artifact.manifest_files.read_text(encoding='utf-8')
