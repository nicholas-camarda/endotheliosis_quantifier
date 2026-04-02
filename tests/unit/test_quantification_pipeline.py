import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from eq.quantification.labelstudio_scores import recover_label_studio_score_table
from eq.quantification.pipeline import (
    ALLOWED_SCORE_VALUES,
    OrdinalThresholdModel,
    _prepare_encoder_for_forward,
    build_image_level_scored_example_table,
    build_scored_example_table,
    evaluate_embedding_table,
    extract_image_level_roi_crops,
    extract_roi_crops,
)


def _make_rect_mask(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
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
        {
            'subject_id': ['T19-1', 'T19-1'],
            'glomerulus_id': [1, 2],
            'score': [0.0, 1.0],
        }
    )

    scored = build_scored_example_table(project_dir, metadata_df, tmp_path / 'out')
    assert set(scored['join_status']) == {'ok'}

    roi_table = extract_roi_crops(scored, tmp_path / 'roi')
    assert set(roi_table['roi_status']) == {'ok'}
    assert roi_table['roi_image_path'].astype(bool).all()
    assert roi_table['roi_area'].notna().all()


def test_recover_labelstudio_scores_prefers_latest_and_backfills_missing_grade(tmp_path: Path):
    project_dir = tmp_path / 'project'
    image_dir = project_dir / 'images' / 'T19'
    mask_dir = project_dir / 'masks' / 'T19'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = _make_rect_mask((64, 64), [(16, 16, 48, 48)])
    for name in ['T19_Image0.jpg', 'T19_Image1.jpg']:
        Image.fromarray(image).save(image_dir / name)
        Image.fromarray(mask).save(mask_dir / f"{Path(name).stem}_mask.jpg")

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
                        {'type': 'brushlabels', 'value': {'brushlabels': ['glomerulus']}},
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
                        {'type': 'brushlabels', 'value': {'brushlabels': ['glomerulus']}},
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
                        {'type': 'brushlabels', 'value': {'brushlabels': ['glomerulus']}},
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
                        {'type': 'brushlabels', 'value': {'brushlabels': ['glomerulus']}},
                    ],
                }
            ],
        },
    ]
    annotation_path = tmp_path / 'annotations.json'
    annotation_path.write_text(json.dumps(annotation_payload), encoding='utf-8')

    outputs = recover_label_studio_score_table(project_dir, annotation_path, tmp_path / 'scores')
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
    image[30:90, 30:90] = 180
    Image.fromarray(image).save(image_dir / 'T19_Image5.jpg')

    mask = _make_rect_mask((120, 120), [(35, 35, 85, 85)])
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

    scored = build_image_level_scored_example_table(project_dir, score_table, tmp_path / 'out')
    assert scored.loc[0, 'subject_image_id'] == 'T19_Image5'
    assert scored.loc[0, 'glomerulus_id'] == 1

    roi_table = extract_image_level_roi_crops(scored, tmp_path / 'roi')
    assert roi_table.loc[0, 'roi_status'] == 'ok'
    assert roi_table.loc[0, 'roi_component_selection'] == 'single_component'
    assert Path(str(roi_table.loc[0, 'roi_image_path'])).exists()


def test_ordinal_threshold_model_outputs_probability_simplex():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    model = OrdinalThresholdModel(n_classes=7).fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (6, 7)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(6), atol=1e-6)


def test_prepare_encoder_for_forward_wraps_module_list():
    module_list = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    wrapped = _prepare_encoder_for_forward(module_list)
    assert isinstance(wrapped, torch.nn.Identity)


def test_evaluate_embedding_table_runs_with_grouped_subject_splits(tmp_path: Path):
    rows = []
    for subject_index, subject_prefix in enumerate(['T19', 'T20', 'T21']):
        for image_index in range(1, 3):
            score = float(ALLOWED_SCORE_VALUES[(subject_index + image_index) % len(ALLOWED_SCORE_VALUES)])
            rows.append(
                {
                    'subject_image_id': f'{subject_prefix}-{image_index}',
                    'subject_prefix': subject_prefix,
                    'glomerulus_id': 1,
                    'score': score,
                    'embedding_0000': float(subject_index),
                    'embedding_0001': float(image_index),
                    'embedding_0002': float(subject_index + image_index),
                }
            )
    embedding_df = pd.DataFrame(rows)
    artifacts = evaluate_embedding_table(embedding_df, tmp_path / 'model')
    assert artifacts['metrics'].exists()
    assert artifacts['predictions'].exists()
    assert artifacts['confusion_matrix'].exists()
