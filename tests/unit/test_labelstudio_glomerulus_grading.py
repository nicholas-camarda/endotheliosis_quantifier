from pathlib import Path

import pandas as pd
import pytest

from eq.labelstudio.glomerulus_grading import (
    LabelStudioGlomerulusContractError,
    load_glomerulus_grading_records,
    prepare_rollup_records,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / 'fixtures'
    / 'labelstudio_glomerulus_instances'
)


def test_loads_multiple_complete_glomeruli_and_cutoff_exclusion():
    records = load_glomerulus_grading_records(
        FIXTURE_DIR / 'valid_multi_glomerulus_export.json'
    )

    assert list(records['glomerulus_instance_id']) == ['glom_a', 'glom_b', 'glom_c']
    assert records.loc[0, 'image_id'] == 'T19_Image0'
    assert records.loc[0, 'region_id'] == 'glom_a'
    assert records.loc[0, 'human_grade'] == 0.5
    assert records.loc[0, 'completeness_status'] == 'complete'
    assert records.loc[0, 'grader_user_id'] == '7'
    assert records.loc[0, 'grader_email'] == 'grader@example.edu'
    assert records.loc[0, 'annotation_id'] == '5001'
    assert records.loc[2, 'completeness_status'] == 'excluded'
    assert records.loc[2, 'exclusion_reason'] == 'cutoff_partial_glomerulus'
    assert pd.isna(records.loc[2, 'human_grade'])


def test_rejects_grade_without_region_link():
    with pytest.raises(
        LabelStudioGlomerulusContractError,
        match='grade-to-region linkage|legacy baseline',
    ):
        load_glomerulus_grading_records(
            FIXTURE_DIR / 'missing_region_link_export.json'
        )


def test_rejects_cutoff_region_with_grade():
    with pytest.raises(
        LabelStudioGlomerulusContractError, match='excluded-region-grade'
    ):
        load_glomerulus_grading_records(
            FIXTURE_DIR / 'invalid_cutoff_grade_export.json'
        )


def test_rejects_duplicate_active_grades_for_one_region():
    with pytest.raises(LabelStudioGlomerulusContractError, match='duplicate-grade'):
        load_glomerulus_grading_records(FIXTURE_DIR / 'duplicate_grade_export.json')


def test_rejects_missing_grader_provenance():
    with pytest.raises(LabelStudioGlomerulusContractError, match='missing-provenance'):
        load_glomerulus_grading_records(FIXTURE_DIR / 'missing_grader_export.json')


def test_rollup_records_exclude_cutoff_candidates():
    records = load_glomerulus_grading_records(
        FIXTURE_DIR / 'valid_multi_glomerulus_export.json'
    )

    rollup = prepare_rollup_records(records)

    assert list(rollup['source_glomerulus_record_id']) == [
        'T19_Image0::glom_a::5001::7',
        'T19_Image0::glom_b::5001::7',
    ]
    assert set(rollup['human_grade']) == {0.5, 2.0}
    assert 'glom_c' not in set(rollup['glomerulus_instance_id'])
