#!/usr/bin/env python3
"""
Glomeruli evaluation utility mirroring mitochondria evaluation patterns.
Computes Dice, IoU, Pixel Accuracy; saves sample predictions and summary.
Respects QUICK_TEST indicator requirements.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def _compute_pixel_accuracy(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    correct = np.sum(true_mask == pred_mask)
    total = true_mask.size
    return float(correct / total)


def evaluate_glomeruli_model(learn, val_images: np.ndarray, val_masks: np.ndarray,
                             output_dir: str, model_name: str) -> Dict[str, float]:
    """
    Evaluate a trained glomeruli segmentation model on validation arrays.

    Args:
        learn: FastAI learner (or compatible) with predict(image) API
        val_images: Validation images as numpy array (N, H, W[, C]) normalized 0-1
        val_masks: Validation masks as numpy array (N, H, W[, 1]) binary 0/1
        output_dir: Base output directory
        model_name: Folder name to write artifacts under output_dir

    Returns:
        Metrics dict with means/stds and sample count
    """
    import os

    from PIL import Image

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    dice_scores = []
    iou_scores = []
    pixel_accuracies = []

    for i in range(len(val_images)):
        img = val_images[i]
        true_mask = val_masks[i]

        # Ensure image 3-channel for PIL
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)

        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        pred_result = learn.predict(img_pil)

        # Extract prediction tensor from fastai tuple
        if isinstance(pred_result, tuple) and len(pred_result) >= 2:
            pred_tensor = pred_result[1]
        else:
            pred_tensor = pred_result

        # Convert to numpy
        if hasattr(pred_tensor, 'numpy'):
            pred_mask = pred_tensor.numpy()
        elif hasattr(pred_tensor, 'cpu'):
            pred_mask = pred_tensor.cpu().numpy()
        else:
            pred_mask = np.asarray(pred_tensor)

        # Squeeze ground truth to 2D
        true_binary = (np.squeeze(true_mask) > 0.5).astype(np.float32)

        # Resize prediction if necessary
        if pred_mask.shape != true_binary.shape:
            from scipy.ndimage import zoom
            scale_factors = [true_binary.shape[j] / pred_mask.shape[j] for j in range(len(true_binary.shape))]
            pred_mask = zoom(pred_mask, scale_factors, order=1)

        pred_binary = (pred_mask > 0.5).astype(np.float32)

        intersection = float(np.sum(true_binary * pred_binary))
        dice = (2.0 * intersection) / (np.sum(true_binary) + np.sum(pred_binary) + 1e-7)
        union = float(np.sum(true_binary) + np.sum(pred_binary) - intersection)
        iou = intersection / (union + 1e-7)
        pix_acc = _compute_pixel_accuracy(true_binary, pred_binary)

        dice_scores.append(dice)
        iou_scores.append(iou)
        pixel_accuracies.append(pix_acc)

    metrics = {
        'dice_mean': float(np.mean(dice_scores)) if dice_scores else 0.0,
        'dice_std': float(np.std(dice_scores)) if dice_scores else 0.0,
        'iou_mean': float(np.mean(iou_scores)) if iou_scores else 0.0,
        'iou_std': float(np.std(iou_scores)) if iou_scores else 0.0,
        'pixel_acc_mean': float(np.mean(pixel_accuracies)) if pixel_accuracies else 0.0,
        'pixel_acc_std': float(np.std(pixel_accuracies)) if pixel_accuracies else 0.0,
        'num_samples': int(len(val_images)),
    }

    # Sample predictions grid
    predictions_plot = output_path / "sample_predictions.png"
    try:
        n_show = min(4, len(val_images))
        if n_show > 0:
            fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
            if n_show == 1:
                axes = np.array([axes])

            is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
            title = 'Glomeruli Evaluation: Image | Ground Truth | Prediction'
            if is_quick_test:
                title = 'TESTING RUN - ' + title
            fig.suptitle(title, fontsize=16)

            preds = []
            for i in range(n_show):
                img = val_images[i]
                if img.ndim == 3 and img.shape[-1] == 1:
                    img_disp = img.squeeze()
                elif img.ndim == 2:
                    img_disp = img
                else:
                    img_disp = img

                axes[i, 0].imshow(img_disp, cmap='gray')
                axes[i, 0].set_title(f'Image {i+1}')
                axes[i, 0].axis('off')

                mask = val_masks[i]
                axes[i, 1].imshow(np.squeeze(mask), cmap='gray')
                axes[i, 1].set_title(f'Ground Truth {i+1}')
                axes[i, 1].axis('off')

                # Predict for display
                img_for_pred = img
                if img_for_pred.ndim == 3 and img_for_pred.shape[-1] == 1:
                    img_for_pred = np.repeat(img_for_pred, 3, axis=-1)
                elif img_for_pred.ndim == 2:
                    img_for_pred = np.repeat(img_for_pred[..., None], 3, axis=-1)
                img_pil = Image.fromarray((img_for_pred * 255).astype(np.uint8))
                pred = learn.predict(img_pil)
                if isinstance(pred, tuple) and len(pred) >= 2:
                    pred_tensor = pred[1]
                else:
                    pred_tensor = pred
                pred_np = pred_tensor.cpu().numpy() if hasattr(pred_tensor, 'cpu') else np.asarray(pred_tensor)
                axes[i, 2].imshow(pred_np, cmap='gray')
                axes[i, 2].set_title(f'Prediction {i+1}')
                axes[i, 2].axis('off')

            plt.tight_layout()
            plt.savefig(predictions_plot, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Warning: Could not save sample predictions plot: {e}")

    # Summary text file with testing indicator when applicable
    summary_path = output_path / "evaluation_summary.txt"
    try:
        with open(summary_path, 'w') as f:
            f.write("Glomeruli Segmentation Model Evaluation Summary\n")
            f.write("================================================\n\n")
            if os.getenv('QUICK_TEST', 'false').lower() == 'true':
                f.write("TESTING RUN - QUICK_TEST MODE\n")
                f.write("This is a TESTING run. DO NOT use for production.\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Evaluation samples: {len(val_images)}\n")
            f.write(f"Output directory: {output_path}\n\n")
            f.write("QUANTITATIVE EVALUATION METRICS:\n")
            f.write("================================\n")
            f.write(f"Dice Score:      {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}\n")
            f.write(f"IoU Score:       {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}\n")
            f.write(f"Pixel Accuracy:  {metrics['pixel_acc_mean']:.4f} ± {metrics['pixel_acc_std']:.4f}\n")
            f.write(f"Sample Count:    {metrics['num_samples']}\n")
    except Exception as e:
        print(f"Warning: Could not write evaluation summary: {e}")

    return metrics


