from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eq.evaluation.run_medsam_manual_glomeruli_comparison_workflow import (
    DEFAULT_METRIC_FIELDS,
    derive_oracle_boxes,
    ensure_evaluation_output_path,
    metric_row,
    select_pilot_inputs,
)


def _write_image(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def _write_mask(path: Path, foreground_box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((16, 16), dtype=np.uint8)
    x0, y0, x1, y1 = foreground_box
    arr[y0:y1, x0:x1] = 255
    Image.fromarray(arr).save(path)


def _manifest_row(
    *,
    manifest_row_id: str,
    cohort_id: str,
    subject: str,
    image_path: str,
    mask_path: str,
    lane_assignment: str,
) -> dict[str, str]:
    return {
        'manifest_row_id': manifest_row_id,
        'cohort_id': cohort_id,
        'source_sample_id': subject,
        'admission_status': 'admitted',
        'lane_assignment': lane_assignment,
        'image_path': image_path,
        'mask_path': mask_path,
    }


def test_select_pilot_inputs_balances_cohorts_and_subjects(tmp_path: Path):
    runtime_root = tmp_path
    rows = []
    for cohort, lane in [
        ('vegfri_dox', 'manual_mask_external'),
        ('lauren_preeclampsia', 'manual_mask_core'),
    ]:
        for index in range(3):
            image_rel = f'raw_data/cohorts/{cohort}/images/{index}.png'
            mask_rel = f'raw_data/cohorts/{cohort}/masks/{index}_mask.png'
            _write_image(runtime_root / image_rel)
            _write_mask(runtime_root / mask_rel, (2, 2, 8, 8))
            rows.append(
                _manifest_row(
                    manifest_row_id=f'{cohort}__{index:06d}',
                    cohort_id=cohort,
                    subject=f'subject_{index}',
                    image_path=image_rel,
                    mask_path=mask_rel,
                    lane_assignment=lane,
                )
            )

    selected = select_pilot_inputs(pd.DataFrame(rows), runtime_root, pilot_size=4)

    assert selected['cohort_id'].value_counts().to_dict() == {
        'lauren_preeclampsia': 2,
        'vegfri_dox': 2,
    }
    assert selected['selection_rank'].tolist() == [1, 2, 3, 4]
    assert set(selected['selection_reason']) == {
        'deterministic_cohort_subject_balanced_pilot'
    }


def test_derive_oracle_boxes_pads_and_clips_components():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[0:3, 0:3] = 1
    mask[6:10, 7:11] = 1

    boxes, skipped = derive_oracle_boxes(mask, min_component_area=4, padding=2)

    assert skipped == []
    assert boxes == [
        {
            'component_index': 1,
            'bbox_x0': 0,
            'bbox_y0': 0,
            'bbox_x1': 5,
            'bbox_y1': 5,
            'component_area': 9,
            'padding': 2,
        },
        {
            'component_index': 2,
            'bbox_x0': 5,
            'bbox_y0': 4,
            'bbox_x1': 12,
            'bbox_y1': 12,
            'component_area': 16,
            'padding': 2,
        },
    ]


def test_derive_oracle_boxes_reports_skipped_components():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 1

    boxes, skipped = derive_oracle_boxes(mask, min_component_area=4, padding=1)

    assert boxes == []
    assert skipped == [
        {
            'component_index': 1,
            'component_area': 1,
            'skip_reason': 'component_area_below_minimum',
        }
    ]


def test_output_path_must_stay_outside_raw_data(tmp_path: Path):
    runtime_root = tmp_path

    ok = ensure_evaluation_output_path(
        runtime_root,
        'output/segmentation_evaluation/medsam_manual_glomeruli_comparison/run',
    )
    assert (
        ok
        == runtime_root
        / 'output/segmentation_evaluation/medsam_manual_glomeruli_comparison/run'
    )

    with pytest.raises(ValueError, match='raw_data'):
        ensure_evaluation_output_path(
            runtime_root, 'raw_data/cohorts/vegfri_dox/masks/generated'
        )


def test_metric_row_has_expected_schema():
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[1:3, 1:3] = 1
    prediction = np.zeros((4, 4), dtype=np.uint8)
    prediction[1:3, 2:4] = 1

    row = metric_row(
        method='medsam_oracle',
        candidate_artifact='',
        manifest_row_id='row-1',
        cohort_id='vegfri_dox',
        lane_assignment='manual_mask_external',
        manual_mask=truth,
        predicted_mask=prediction,
    )

    assert list(row.keys()) == DEFAULT_METRIC_FIELDS
    assert row['dice'] == pytest.approx(0.5)
    assert row['jaccard'] == pytest.approx(1 / 3)
    assert row['precision'] == pytest.approx(0.5)
    assert row['recall'] == pytest.approx(0.5)
