import numpy as np

from eq.evaluation.glomeruli_evaluator import evaluate_glomeruli_model


class DummyLearner:
    """Minimal stub mimicking fastai learner's predict API."""
    def __init__(self, pred_masks):
        self._preds = pred_masks
        self._idx = 0

    def predict(self, pil_image):
        # Return a tuple similar to fastai: (category, tensor, probabilities)
        # Here we just return the mask as numpy; evaluator handles numpy arrays
        pred = self._preds[self._idx]
        self._idx = (self._idx + 1) % len(self._preds)
        return (None, pred, None)


def test_evaluate_glomeruli_model_metrics_and_outputs(tmp_path):
    # Create two simple 4x4 samples
    # Sample 1: partial overlap
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
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # Sample 2: perfect match
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

    val_images = np.stack([
        np.repeat(gt1[..., None], 3, axis=-1),  # Make 3-channel images from masks for simplicity
        np.repeat(gt2[..., None], 3, axis=-1),
    ]).astype(np.float32)
    val_masks = np.stack([gt1, gt2]).astype(np.uint8)

    # Dummy learner cycles through provided predictions
    learn = DummyLearner([pred1, pred2])

    metrics = evaluate_glomeruli_model(
        learn=learn,
        val_images=val_images,
        val_masks=val_masks,
        output_dir=str(tmp_path),
        model_name="glom_eval_test",
    )

    # Manually compute expectations for sample 1
    tp = 3
    fp = 0
    fn = 1
    dice1 = 2 * tp / (2 * tp + fp + fn)
    iou1 = tp / (tp + fp + fn)
    pix_acc1 = (tp + (16 - (tp + fp + fn))) / 16.0

    # Sample 2 perfect
    dice2 = 1.0
    iou2 = 1.0
    pix_acc2 = 1.0

    # Averages
    exp_dice_mean = (dice1 + dice2) / 2.0
    exp_iou_mean = (iou1 + iou2) / 2.0
    exp_pix_mean = (pix_acc1 + pix_acc2) / 2.0

    assert np.isclose(metrics["dice_mean"], exp_dice_mean)
    assert np.isclose(metrics["iou_mean"], exp_iou_mean)
    assert np.isclose(metrics["pixel_acc_mean"], exp_pix_mean)
    assert metrics["num_samples"] == 2

    # Artifacts exist
    import os
    out_dir = tmp_path / "glom_eval_test"
    assert os.path.exists(out_dir / "sample_predictions.png")
    assert os.path.exists(out_dir / "evaluation_summary.txt")


