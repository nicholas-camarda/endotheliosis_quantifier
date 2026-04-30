from pathlib import Path

import numpy as np
import pytest

from eq.evaluation.run_medsam_automatic_glomeruli_prompts_workflow import (
    AUTOMATIC_METRIC_FIELDS,
    PROPOSAL_BOX_FIELDS,
    classify_automatic_prompt_result,
    derive_proposal_boxes,
    ensure_evaluation_output_path,
    proposal_recall_row,
)


def test_derive_proposal_boxes_filters_pads_clips_and_records_skips():
    probability = np.zeros((12, 12), dtype=np.float32)
    probability[0:3, 0:3] = 0.8
    probability[6:11, 7:12] = 0.9
    probability[5, 1] = 0.95

    boxes, decisions = derive_proposal_boxes(
        probability,
        threshold=0.5,
        min_component_area=4,
        max_component_area=40,
        padding=2,
        merge_iou=0.25,
        max_boxes=10,
    )

    assert boxes == [
        {
            'proposal_index': 1,
            'bbox_x0': 0,
            'bbox_y0': 0,
            'bbox_x1': 5,
            'bbox_y1': 5,
            'component_area': 9,
            'threshold': 0.5,
            'decision': 'generated',
        },
        {
            'proposal_index': 2,
            'bbox_x0': 5,
            'bbox_y0': 4,
            'bbox_x1': 12,
            'bbox_y1': 12,
            'component_area': 25,
            'threshold': 0.5,
            'decision': 'generated',
        },
    ]
    assert {
        decision['decision_reason']
        for decision in decisions
        if decision['decision'] == 'skipped'
    } == {'component_area_below_minimum'}


def test_derive_proposal_boxes_merges_overlapping_padded_boxes():
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[2:6, 2:6] = 0.8
    probability[2:6, 8:12] = 0.8

    boxes, decisions = derive_proposal_boxes(
        probability,
        threshold=0.5,
        min_component_area=4,
        max_component_area=100,
        padding=3,
        merge_iou=0.20,
        max_boxes=10,
    )

    assert len(boxes) == 1
    assert boxes[0]['bbox_x0'] == 0
    assert boxes[0]['bbox_x1'] == 15
    assert any(decision['decision'] == 'merged' for decision in decisions)


def test_derive_proposal_boxes_records_overflow():
    probability = np.zeros((20, 20), dtype=np.float32)
    for index, x0 in enumerate([1, 6, 11], start=1):
        probability[1:4, x0 : x0 + 3] = 0.9

    boxes, decisions = derive_proposal_boxes(
        probability,
        threshold=0.5,
        min_component_area=4,
        max_component_area=100,
        padding=0,
        merge_iou=0.25,
        max_boxes=2,
    )

    assert len(boxes) == 2
    assert sum(1 for decision in decisions if decision['decision'] == 'overflow') == 1


def test_proposal_recall_row_counts_matched_and_missed_components():
    manual = np.zeros((12, 12), dtype=np.uint8)
    manual[1:5, 1:5] = 1
    manual[7:11, 7:11] = 1
    boxes = [{'bbox_x0': 0, 'bbox_y0': 0, 'bbox_x1': 6, 'bbox_y1': 6}]

    row = proposal_recall_row(
        manual_mask=manual,
        proposal_boxes=boxes,
        manifest_row_id='row-1',
        cohort_id='vegfri_dox',
        lane_assignment='manual_mask_external',
        candidate_family='transfer',
        candidate_artifact='model.pkl',
        threshold=0.5,
        min_component_area=4,
    )

    assert row['manual_component_count'] == 2
    assert row['matched_manual_component_count'] == 1
    assert row['missed_manual_component_count'] == 1
    assert row['proposal_count'] == 1
    assert row['proposal_recall'] == pytest.approx(0.5)


def test_automatic_gate_classification_recommends_prompt_or_fine_tuning():
    passing = classify_automatic_prompt_result(
        proposal_recall=0.96,
        auto_dice=0.91,
        prompt_failure_count=0,
        min_proposal_recall=0.90,
        min_auto_dice=0.85,
        max_prompt_failures=0,
    )
    assert passing['failure_mode'] == 'none_detected'
    assert passing['recommended_generated_mask_source'] == 'medsam_automatic_glomeruli'
    assert (
        passing['fine_tuning_recommendation']
        == 'not_recommended_prompt_based_generation_first'
    )

    localization_failure = classify_automatic_prompt_result(
        proposal_recall=0.70,
        auto_dice=0.91,
        prompt_failure_count=0,
        min_proposal_recall=0.90,
        min_auto_dice=0.85,
        max_prompt_failures=0,
    )
    assert localization_failure['failure_mode'] == 'proposal_localization'
    assert (
        localization_failure['fine_tuning_recommendation']
        == 'improve_box_proposer_before_fine_tuning'
    )

    boundary_failure = classify_automatic_prompt_result(
        proposal_recall=0.95,
        auto_dice=0.75,
        prompt_failure_count=0,
        min_proposal_recall=0.90,
        min_auto_dice=0.85,
        max_prompt_failures=0,
    )
    assert boundary_failure['failure_mode'] == 'medsam_boundary_quality'
    assert (
        boundary_failure['fine_tuning_recommendation']
        == 'open_medsam_sam_fine_tuning_change'
    )


def test_output_path_guard_rejects_raw_data(tmp_path: Path):
    ok = ensure_evaluation_output_path(
        tmp_path,
        'output/segmentation_evaluation/medsam_automatic_glomeruli_prompts/run',
    )
    assert (
        ok
        == tmp_path
        / 'output/segmentation_evaluation/medsam_automatic_glomeruli_prompts/run'
    )

    with pytest.raises(ValueError, match='raw_data'):
        ensure_evaluation_output_path(
            tmp_path, 'raw_data/cohorts/vegfri_dox/masks/medsam_auto'
        )


def test_automatic_metric_fields_include_transition_metadata():
    assert 'prompt_mode' in AUTOMATIC_METRIC_FIELDS
    assert 'proposal_threshold' in AUTOMATIC_METRIC_FIELDS
    assert 'mask_source' in AUTOMATIC_METRIC_FIELDS


def test_proposal_box_fields_cover_generated_and_skipped_rows():
    assert 'bbox_x0' in PROPOSAL_BOX_FIELDS
    assert 'bbox_y1' in PROPOSAL_BOX_FIELDS
    assert 'proposal_index' in PROPOSAL_BOX_FIELDS
    assert 'decision_reason' in PROPOSAL_BOX_FIELDS
