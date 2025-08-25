import numpy as np

from eq.metrics.segmentation_metrics import (
    calculate_all_metrics,
    calculate_batch_metrics,
    dice_coefficient,
    f1_score,
    iou_score,
    precision_score,
    recall_score,
)


def test_single_sample_metrics_exact_values():
    # Simple 4x4 case with known overlap
    # Ground truth has a 2x2 square in the top-left
    gt = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # Prediction overlaps 3 of the 4 pixels and adds 1 false positive
    pred = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # Manual expectations
    tp = 3  # overlap pixels
    fp = 1  # extra predicted pixel at (1,2)
    fn = 1  # missed gt pixel at (1,1)
    tn = 16 - (tp + fp + fn)

    expected_precision = tp / (tp + fp)
    expected_recall = tp / (tp + fn)
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)

    intersection = tp
    union = tp + fp + fn
    expected_iou = intersection / union
    expected_dice = 2 * intersection / (2 * intersection + fp + fn)

    # Pixel accuracy for binary masks
    expected_pixel_acc = (tp + tn) / 16.0

    # Assertions using individual functions
    assert np.isclose(precision_score(pred, gt), expected_precision)
    assert np.isclose(recall_score(pred, gt), expected_recall)
    assert np.isclose(f1_score(pred, gt), expected_f1)
    assert np.isclose(iou_score(pred, gt), expected_iou)
    assert np.isclose(dice_coefficient(pred, gt), expected_dice)

    # Assertions using aggregate helper
    metrics = calculate_all_metrics(pred, gt, include_hausdorff=False)
    assert np.isclose(metrics.dice_coefficient, expected_dice)
    assert np.isclose(metrics.iou_score, expected_iou)
    assert np.isclose(metrics.precision, expected_precision)
    assert np.isclose(metrics.recall, expected_recall)
    assert np.isclose(metrics.f1_score, expected_f1)


def test_batch_metrics_average_and_individuals():
    # Two samples with distinct overlaps to test averaging
    gt1 = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    pred1 = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # Second sample: perfect match
    gt2 = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    pred2 = gt2.copy()

    # Compute individual metrics explicitly for verification
    m1 = calculate_all_metrics(pred1, gt1)
    m2 = calculate_all_metrics(pred2, gt2)

    # Batch calculation
    avg, individuals = calculate_batch_metrics([pred1, pred2], [gt1, gt2])

    # Individuals round-trip
    assert len(individuals) == 2
    assert np.isclose(individuals[0].dice_coefficient, m1.dice_coefficient)
    assert np.isclose(individuals[1].dice_coefficient, m2.dice_coefficient)

    # Averages are arithmetic means
    assert np.isclose(avg.dice_coefficient, (m1.dice_coefficient + m2.dice_coefficient) / 2.0)
    assert np.isclose(avg.iou_score, (m1.iou_score + m2.iou_score) / 2.0)
    assert np.isclose(avg.precision, (m1.precision + m2.precision) / 2.0)
    assert np.isclose(avg.recall, (m1.recall + m2.recall) / 2.0)
    assert np.isclose(avg.f1_score, (m1.f1_score + m2.f1_score) / 2.0)


