import numpy as np

from eq.training.mitochondria_validation_examples import (
    _candidate_crop_boxes,
    _center_truth_fraction,
    _edge_contact_fraction,
    _select_examples,
)


def test_candidate_crop_boxes_include_grid_and_foreground_centers():
    mask = np.zeros((768, 1024), dtype=np.uint8)
    mask[300:340, 410:450] = 1

    boxes = _candidate_crop_boxes(mask, crop_size=256)

    assert (0, 0, 256, 256) in boxes
    assert any(
        left <= 430 <= right and top <= 320 <= bottom
        for left, top, right, bottom in boxes
    )
    assert len(boxes) == len(set(boxes))


def test_select_examples_prefers_high_dice_distinct_images():
    rows = [
        {
            'image_name': 'a.tif',
            'dice': 0.9,
            'jaccard': 0.8,
            'truth_foreground_fraction': 0.1,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.5,
        },
        {
            'image_name': 'a.tif',
            'dice': 0.95,
            'jaccard': 0.9,
            'truth_foreground_fraction': 0.1,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.5,
        },
        {
            'image_name': 'b.tif',
            'dice': 0.85,
            'jaccard': 0.7,
            'truth_foreground_fraction': 0.1,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.5,
        },
        {
            'image_name': 'c.tif',
            'dice': 0.75,
            'jaccard': 0.6,
            'truth_foreground_fraction': 0.1,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.5,
        },
    ]

    selected = _select_examples(rows, count=2)

    assert [row['image_name'] for row in selected] == ['a.tif', 'b.tif']
    assert selected[0]['dice'] == 0.95


def test_select_examples_prefers_visual_crops_before_tiny_edge_fragments():
    rows = [
        {
            'image_name': 'edge.tif',
            'dice': 0.99,
            'jaccard': 0.98,
            'truth_foreground_fraction': 0.02,
            'edge_contact_fraction': 0.10,
            'center_truth_fraction': 0.0,
        },
        {
            'image_name': 'visual.tif',
            'dice': 0.92,
            'jaccard': 0.85,
            'truth_foreground_fraction': 0.10,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.70,
        },
    ]

    selected = _select_examples(rows, count=1)

    assert selected[0]['image_name'] == 'visual.tif'


def test_select_examples_prefers_diverse_visual_crops():
    base_crop = np.arange(16, dtype=np.uint8).reshape(4, 4)
    different_crop = np.flipud(base_crop)
    rows = [
        {
            'image_name': 'a.tif',
            'crop_box': '0|0|4|4',
            'dice': 0.99,
            'jaccard': 0.98,
            'truth_foreground_fraction': 0.1,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.5,
            '_image_crop': base_crop,
        },
        {
            'image_name': 'b.tif',
            'crop_box': '0|0|4|4',
            'dice': 0.98,
            'jaccard': 0.97,
            'truth_foreground_fraction': 0.1,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.5,
            '_image_crop': base_crop.copy(),
        },
        {
            'image_name': 'c.tif',
            'crop_box': '4|0|8|4',
            'dice': 0.9,
            'jaccard': 0.82,
            'truth_foreground_fraction': 0.1,
            'edge_contact_fraction': 0.0,
            'center_truth_fraction': 0.5,
            '_image_crop': different_crop,
        },
    ]

    selected = _select_examples(rows, count=2)

    assert [row['image_name'] for row in selected] == ['a.tif', 'c.tif']


def test_visual_crop_metrics_measure_edge_and_centered_foreground():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3:5, 3:5] = 1
    assert _edge_contact_fraction(mask) == 0.0
    assert _center_truth_fraction(mask) == 1.0

    edge_mask = np.zeros((8, 8), dtype=np.uint8)
    edge_mask[0, 2:6] = 1
    assert _edge_contact_fraction(edge_mask) == 1.0
    assert _center_truth_fraction(edge_mask) == 0.0
