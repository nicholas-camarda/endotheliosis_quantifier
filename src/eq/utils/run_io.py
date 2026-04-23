#!/usr/bin/env python3
"""
Run I/O Utilities for Training Pipeline

This module provides standardized functions for saving training outputs,
ensuring consistent directory structure and file naming across all training scripts.
"""

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import torch
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner

from eq.utils.logger import get_logger

logger = get_logger("eq.run_io")


def save_splits(output_dir: Path, model_folder_name: str, data_info: Dict[str, Any]) -> None:
    """Save data splits manifest with prefixed filename."""
    try:
        split_manifest = {
            "stage": data_info.get("stage", "unknown"),
            "training_mode": data_info.get("training_mode"),
            "data_root": data_info.get("data_root"),
            "generated_at": datetime.now().isoformat(),
            "train_images": [str(p) for p in data_info.get("train_items", [])],
            "valid_images": [str(p) for p in data_info.get("valid_items", [])],
            "counts": {
                "train": int(len(data_info.get("train_items", []))),
                "valid": int(len(data_info.get("valid_items", [])))
            }
        }
        
        splits_file = output_dir / f"{model_folder_name}_splits.json"
        split_manifest = {k: v for k, v in split_manifest.items() if v is not None}
        with open(splits_file, 'w') as f:
            json.dump(split_manifest, f, indent=2)
        logger.info(f"Wrote split manifest to {splits_file}")
    except Exception as e:
        logger.warning(f"Could not write split manifest: {e}")


def attach_best_model_callback(model_folder_name: str, monitor: str = 'valid_loss', comp: Optional[Callable] = None) -> SaveModelCallback:
    """Create SaveModelCallback with prefixed filename.
    monitor: metric name to monitor (e.g., 'valid_loss' or 'dice')
    comp: comparator; if None, fastai infers by monitor name
    """
    kwargs = {
        'monitor': monitor,
        'fname': f'{model_folder_name}_best_model',
        'with_opt': False,
    }
    if comp is not None:
        kwargs['comp'] = comp
    return SaveModelCallback(**kwargs)


def save_plots(learn: Learner, output_dir: Path, model_folder_name: str) -> None:
    """Save all training plots with prefixed filenames."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure matplotlib is available for subsequent plots
    try:
        plt.close('all')
        values = None
        if hasattr(learn.recorder, 'values') and learn.recorder.values:
            import numpy as _np
            values = _np.array(learn.recorder.values)
        if values is not None and values.shape[1] >= 2:
            epochs = range(len(values))
            # Ensure ax is typed as an Axes (not ndarray) for linters
            from matplotlib.axes import Axes
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = cast(Axes, ax)
            ax.plot(epochs, values[:, 0], label='Train', color='tab:blue')
            ax.plot(epochs, values[:, 1], label='Valid', color='tab:orange')
            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        else:
            # Fallback to fastai helper (still on a fresh figure)
            plt.figure(figsize=(10, 6))
            learn.recorder.plot_loss()
        plt.savefig(output_dir / f"{model_folder_name}_training_loss.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training loss plot saved to: {output_dir / f'{model_folder_name}_training_loss.png'}")
    except Exception as e:
        logger.warning(f"Could not save training loss plot: {e}")
    
    # Learning rate schedule plot
    try:
        if hasattr(learn.recorder, 'lrs') and learn.recorder.lrs:
            plt.figure(figsize=(10, 6))
            plt.plot(learn.recorder.lrs)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Batch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(output_dir / f"{model_folder_name}_lr_schedule.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Learning rate plot saved to: {output_dir / f'{model_folder_name}_lr_schedule.png'}")
        else:
            logger.info("Learning rate data not available for plotting")
    except Exception as e:
        logger.warning(f"Could not save learning rate plot: {e}")
    
    # Training metrics plot
    try:
        if hasattr(learn.recorder, 'values') and learn.recorder.values:
            import matplotlib.pyplot as plt
            import numpy as np
            
            values = np.array(learn.recorder.values)
            epoch_range = range(len(values))
            
            plt.figure(figsize=(12, 8))
            
            # Plot losses
            plt.subplot(2, 2, 1)
            plt.plot(epoch_range, values[:, 0], label='Train Loss', color='blue')
            plt.plot(epoch_range, values[:, 1], label='Valid Loss', color='red')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot dice score if available
            if values.shape[1] > 2:
                plt.subplot(2, 2, 2)
                plt.plot(epoch_range, values[:, 2], label='Dice Score', color='green')
                plt.title('Dice Score')
                plt.xlabel('Epoch')
                plt.ylabel('Dice Score')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_folder_name}_metrics.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Training metrics plot saved to: {output_dir / f'{model_folder_name}_metrics.png'}")
        else:
            logger.info("Metrics data not available for plotting")
    except Exception as e:
        logger.warning(f"Could not save metrics plot: {e}")

    # Validation predictions plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get a few validation samples in a deterministic order
        val_dl = learn.dls.valid.new(shuffle=False, drop_last=False)
        batch = next(iter(val_dl))
        images, masks = batch
        
        # Make predictions aligned with the same non-shuffled dataloader
        with learn.no_bar():
            preds = learn.get_preds(dl=val_dl)
        
        # Prepare item paths for raw image display (if available)
        item_paths = getattr(learn.dls.valid_ds, 'items', None)
        # Try to import mask getter to resolve mask filenames alongside images
        try:
            from eq.data_management.datablock_loader import (
                default_get_y_path as _get_mask_path,
            )
        except Exception:
            _get_mask_path = None
        
        # Plot first few predictions: add a RAW column for sanity
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        try:
            # Add model folder name into the figure title for easier README embedding
            fig.suptitle(f"Validation Predictions – {model_folder_name}", fontsize=14)
        except Exception:
            pass
        
        # Always decode to display-ready space; never assume channels
        for i in range(min(3, len(images))):
            dec_x, _ = learn.dls.decode((images[i], masks[i]))
            # Convert decoded image to numpy robustly
            try:
                if hasattr(dec_x, 'cpu') and hasattr(dec_x, 'numpy'):
                    img_np = dec_x.cpu().numpy()
                else:
                    img_np = np.asarray(dec_x)
            except Exception:
                # Fallback: try to extract underlying PIL image if wrapped
                try:
                    img_np = np.asarray(getattr(dec_x, 'image', dec_x))
                except Exception:
                    logger.warning("Failed to convert decoded image to numpy; using zeros")
                    img_np = np.zeros((images[i].shape[-2], images[i].shape[-1], 3), dtype=np.float32)

            # Squeeze any leading singleton batch dims (e.g., 1x3xHxW)
            while img_np.ndim > 3 and img_np.shape[0] == 1:
                img_np = np.squeeze(img_np, axis=0)

            # Ensure channels-last RGB for plotting (handle CHW and grayscale)
            if img_np.ndim == 3:
                # If channel-first (C,H,W) convert to (H,W,C)
                if img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
                    img_np = np.transpose(img_np, (1, 2, 0))
                # If last dim is single channel, repeat to 3
                if img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)
                # If last dim > 3, take first 3 channels for visualization
                if img_np.shape[-1] > 3:
                    img_np = img_np[..., :3]
            elif img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)

            # Convert to float and scale to [0,1] robustly
            img_np = img_np.astype(np.float32, copy=False)
            vmin = float(np.nanmin(img_np)) if img_np.size else 0.0
            vmax = float(np.nanmax(img_np)) if img_np.size else 1.0
            if img_np.dtype == np.uint8:
                img_np = img_np / 255.0
            elif vmax > 1.0 or vmin < 0.0:
                # Apply per-image min-max scaling when outside [0,1]
                rng = vmax - vmin
                if rng > 1e-8:
                    img_np = (img_np - vmin) / rng
                else:
                    img_np = np.zeros_like(img_np, dtype=np.float32)
            else:
                # Already in [0,1]; ensure numeric stability
                img_np = np.clip(img_np, 0.0, 1.0)

            # If still flat (e.g., all zeros), attempt manual denorm from original batch tensor
            if float(np.nanmax(img_np) - np.nanmin(img_np)) < 1e-6:
                try:
                    xb = images[i]
                    if hasattr(xb, 'detach'):
                        xb = xb.detach().cpu().float()
                    xb_np = xb.numpy()
                    # Squeeze potential batch and ensure CHW
                    while xb_np.ndim > 3 and xb_np.shape[0] == 1:
                        xb_np = np.squeeze(xb_np, axis=0)
                    # If HWC, transpose to CHW for denorm
                    if xb_np.ndim == 3 and xb_np.shape[0] not in (1, 3) and xb_np.shape[-1] in (1, 3):
                        xb_np = np.transpose(xb_np, (2, 0, 1))
                    # Denormalize using imagenet stats
                    from fastai.vision.all import imagenet_stats
                    mean, std = imagenet_stats
                    mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
                    std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
                    if xb_np.shape[0] == 1:
                        # replicate single channel
                        xb_np = np.repeat(xb_np, 3, axis=0)
                    denorm = xb_np * std + mean
                    img_np = np.transpose(denorm, (1, 2, 0))
                    img_np = np.clip(img_np, 0.0, 1.0)
                except Exception as _e:
                    logger.warning(f"Manual denorm failed: {_e}")

            # If dynamic range is extremely low, apply contrast stretching for visibility (debug-only)
            try:
                dr = float(np.nanmax(img_np) - np.nanmin(img_np)) if img_np.size else 0.0
                if dr < 0.05:
                    p1 = float(np.percentile(img_np, 1))
                    p99 = float(np.percentile(img_np, 99))
                    if p99 > p1:
                        img_np = np.clip((img_np - p1) / (p99 - p1), 0.0, 1.0)
                        logger.info(f"Input[{i}] contrast-stretched (p1={p1:.4f}, p99={p99:.4f})")
                    else:
                        # Fallback linear stretch across min/max
                        mn = float(np.nanmin(img_np))
                        mx = float(np.nanmax(img_np))
                        rng = mx - mn
                        if rng > 1e-8:
                            img_np = np.clip((img_np - mn) / rng, 0.0, 1.0)
                            logger.info(f"Input[{i}] contrast-stretched (min={mn:.4f}, max={mx:.4f})")
            except Exception as _e:
                logger.warning(f"Contrast stretch failed: {_e}")

            logger.info(
                f"Img[{i}] shape={img_np.shape}, dtype={img_np.dtype}, min={float(np.nanmin(img_np)):.4f}, max={float(np.nanmax(img_np)):.4f}"
            )
            # Try to display the RAW image from disk (left-most column)
            if item_paths is not None and len(item_paths) > i:
                try:
                    raw_path = Path(item_paths[i])
                    # Prefer PIL directly to avoid any pipeline transforms
                    from PIL import Image as _PIL
                    raw_img = _PIL.open(raw_path)
                    raw_np = np.asarray(raw_img)
                    # Ensure HWC and scale to [0,1]
                    if raw_np.ndim == 2:
                        raw_np = np.stack([raw_np] * 3, axis=-1)
                    elif raw_np.ndim == 3 and raw_np.shape[-1] == 1:
                        raw_np = np.repeat(raw_np, 3, axis=-1)
                    if raw_np.dtype == np.uint8:
                        raw_np = raw_np.astype(np.float32) / 255.0
                    else:
                        # Min-max stretch floats for visibility
                        rmin, rmax = float(np.nanmin(raw_np)), float(np.nanmax(raw_np))
                        rrng = rmax - rmin
                        if rrng > 1e-8:
                            raw_np = np.clip((raw_np - rmin) / rrng, 0.0, 1.0)
                        else:
                            raw_np = np.zeros_like(raw_np, dtype=np.float32)
                    # Resolve mask path for logging/annotation if possible
                    mask_name = "(unknown)"
                    if _get_mask_path is not None:
                        try:
                            mask_path = Path(_get_mask_path(raw_path))
                            mask_name = mask_path.name
                        except Exception as _e2:
                            logger.warning(f"Failed to resolve mask path for {raw_path.name}: {_e2}")

                    logger.info(
                        f"PAIR[{i}] image={raw_path.name} mask={mask_name} | RAW shape={raw_np.shape} dtype={raw_np.dtype} min={float(np.nanmin(raw_np)):.4f} max={float(np.nanmax(raw_np)):.4f}"
                    )
                    axes[i, 0].imshow(raw_np)
                    axes[i, 0].set_title(f'Raw {i+1}\n{raw_path.name}', fontsize=8)
                    axes[i, 0].axis('off')
                except Exception as _e:
                    logger.warning(f"Failed to load raw image for index {i}: {_e}")
                    axes[i, 0].axis('off')
            else:
                axes[i, 0].axis('off')

            # Model input decoded (second column)
            axes[i, 1].imshow(img_np)
            axes[i, 1].set_title(f'Input {i+1}')
            axes[i, 1].axis('off')
            
            # Ground truth mask
            gt_mask = masks[i]
            if hasattr(gt_mask, 'detach'):
                gt_mask = gt_mask.detach().cpu().numpy()
            else:
                gt_mask = np.asarray(gt_mask)
            # Ensure 2D and binary
            while gt_mask.ndim > 2 and gt_mask.shape[0] == 1:
                gt_mask = np.squeeze(gt_mask, axis=0)
            if gt_mask.ndim == 3:
                gt_mask = np.squeeze(gt_mask)
            if gt_mask.max() > 1:
                gt_mask = (gt_mask > 0).astype(np.uint8)
            logger.info(f"GT[{i}] unique={np.unique(gt_mask)} shape={gt_mask.shape}")
            axes[i, 2].imshow(gt_mask, cmap='gray')
            # If we resolved mask_name above, annotate it; otherwise keep generic title
            try:
                if 'mask_name' in locals() and isinstance(mask_name, str):
                    axes[i, 2].set_title(f'Ground Truth {i+1}\n{mask_name}', fontsize=8)
                else:
                    axes[i, 2].set_title(f'Ground Truth {i+1}')
            except Exception:
                axes[i, 2].set_title(f'Ground Truth {i+1}')
            axes[i, 2].axis('off')
            
            # Prediction
            pred_i = preds[0][i]
            if hasattr(pred_i, 'detach'):
                pred_i = pred_i.detach().cpu()
            # Squeeze any leading singleton dims
            while pred_i.ndim > 3 and pred_i.shape[0] == 1:
                pred_i = pred_i[0]
            # Argmax over channel/class dim if present
            if pred_i.ndim == 3 and pred_i.shape[0] > 1:
                pred_mask = pred_i.argmax(dim=0).numpy()
            else:
                pred_mask = pred_i.numpy()
            if pred_mask.ndim == 3:
                pred_mask = np.squeeze(pred_mask)
            if pred_mask.max() > 1:
                pred_mask = (pred_mask > 0).astype(np.uint8)
            logger.info(f"Pred[{i}] unique={np.unique(pred_mask)} shape={pred_mask.shape}")
            axes[i, 3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Prediction {i+1}')
            axes[i, 3].axis('off')
            
            # Overlay - show underlying image with transparent color-coded error overlay
            # First show the denormalized input image as base layer
            axes[i, 4].imshow(img_np)
            
            # Build color-coded error mask
            overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
            
            # Green for true positives (both GT and pred are positive)
            tp_mask = (gt_mask > 0) & (pred_mask > 0)
            overlay[tp_mask] = [0.0, 1.0, 0.0]  # Green
            
            # Red for false positives (pred positive, GT negative)
            fp_mask = (gt_mask == 0) & (pred_mask > 0)
            overlay[fp_mask] = [1.0, 0.0, 0.0]  # Red
            
            # Blue for false negatives (GT positive, pred negative)
            fn_mask = (gt_mask > 0) & (pred_mask == 0)
            overlay[fn_mask] = [0.0, 0.0, 1.0]  # Blue
            logger.info(
                f"TP={int(tp_mask.sum())} FP={int(fp_mask.sum())} FN={int(fn_mask.sum())}; overlay shape={overlay.shape}"
            )
            
            # Now overlay with alpha transparency on top of the image
            axes[i, 4].imshow(overlay, alpha=0.4, vmin=0, vmax=1)
            axes[i, 4].set_title(f'Overlay {i+1}\n(Green=TP, Red=FP, Blue=FN)')
            axes[i, 4].axis('off')
        
        # Leave room for the suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / f"{model_folder_name}_validation_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Validation predictions saved to: {output_dir / f'{model_folder_name}_validation_predictions.png'}")
    except Exception as e:
        logger.warning(f"Could not save validation predictions: {e}")


def save_training_history(learn: Learner, output_dir: Path, model_folder_name: str, 
                         hyperparams: Dict[str, Any]) -> None:
    """Save training history as TSV with prefixed filename."""
    try:
        history_file = output_dir / f"{model_folder_name}_training_history.tsv"
        
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # Write header with training configuration
            writer.writerow(['# Training Configuration'])
            # Ensure deterministic order for reproducibility
            for key in sorted(hyperparams.keys()):
                writer.writerow([key, hyperparams[key]])
            # Single blank separator line
            writer.writerow([""])  # Empty row separator
            
            # Write training metrics if available
            if hasattr(learn, 'recorder') and learn.recorder:
                if hasattr(learn.recorder, 'values') and learn.recorder.values:
                    # Build metric header once. FastAI's recorder.metric_names commonly includes
                    # ['train_loss', 'valid_loss', 'time'] already; we guard against duplicates.
                    metric_names: List[str] = ['epoch']
                    base_names: List[str] = []
                    if hasattr(learn.recorder, 'metric_names') and learn.recorder.metric_names:
                        # Filter out potential duplicate/empty entries and normalize
                        base_names = [str(n).strip() for n in learn.recorder.metric_names if str(n).strip()]
                        # FastAI includes 'time' for display only; it is not present in values. Remove it.
                        base_names = [n for n in base_names if n.lower() != 'time']
                    else:
                        base_names = ['train_loss', 'valid_loss']
                    # Remove 'epoch' if present and deduplicate while preserving order
                    seen = set()
                    deduped: List[str] = []
                    for n in base_names:
                        if n.lower() == 'epoch':
                            continue
                        if n not in seen:
                            seen.add(n)
                            deduped.append(n)
                    metric_names.extend(deduped)
                    
                    # Section header and header row
                    writer.writerow(['# Training Metrics'])
                    writer.writerow(metric_names)
                    
                    # Data rows: ensure values length matches header-1 (epoch)
                    for epoch_index, values in enumerate(learn.recorder.values):
                        try:
                            value_cells: List[str] = []
                            for v in values[:len(metric_names) - 1]:
                                if isinstance(v, (int, float)):
                                    value_cells.append(f"{float(v)}")
                                else:
                                    # Convert tensors or others to string safely
                                    try:
                                        value_cells.append(f"{float(v.detach().cpu().item())}")  # type: ignore[attr-defined]
                                    except Exception:
                                        value_cells.append(str(v))
                            # Pad with N/A if recorder provided fewer columns than header
                            while len(value_cells) < len(metric_names) - 1:
                                value_cells.append('N/A')
                            row = [epoch_index] + value_cells
                            writer.writerow(row)
                        except Exception as _e:
                            logger.warning(f"Failed to write history row for epoch {epoch_index}: {_e}")
                    
                    # Single blank separator line
                    writer.writerow([""])  # Empty row separator
                
                # Write final summary
                writer.writerow(['# Final Summary'])
                if hasattr(learn.recorder, 'values') and learn.recorder.values:
                    final_values = learn.recorder.values[-1]
                    # Safely extract by position with guards
                    final_train = final_values[0] if len(final_values) > 0 else 'N/A'
                    final_valid = final_values[1] if len(final_values) > 1 else 'N/A'
                    def _to_str(x: Any) -> str:
                        try:
                            return f"{float(x)}"
                        except Exception:
                            try:
                                return f"{float(x.detach().cpu().item())}"  # type: ignore[attr-defined]
                            except Exception:
                                return str(x)
                    writer.writerow(['final_train_loss', _to_str(final_train)])
                    writer.writerow(['final_valid_loss', _to_str(final_valid)])
                    if len(final_values) > 2:
                        writer.writerow(['final_dice_score', _to_str(final_values[2])])
        
        logger.info(f"Training history saved to: {history_file}")
    except Exception as e:
        logger.warning(f"Could not save training history: {e}")


def _package_version(package_name: str) -> Optional[str]:
    try:
        module = __import__(package_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def _git_state() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    state: Dict[str, Any] = {"repo_root": str(repo_root)}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        state["commit"] = commit
    except Exception:
        state["commit"] = None
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        state["dirty"] = bool(dirty.strip())
    except Exception:
        state["dirty"] = None
    return state


def _build_run_metadata(config_path: Optional[str], extra_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    package_versions = {
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "fastai": _package_version("fastai"),
        "numpy": _package_version("numpy"),
    }
    metadata: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "package_versions": package_versions,
        "code": _git_state(),
    }
    if config_path:
        metadata["config_path"] = str(config_path)
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def save_run_metadata(
    output_dir: Path,
    model_folder_name: str,
    config_path: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save run metadata and optionally copy config file."""
    metadata = _build_run_metadata(config_path, extra_metadata)
    metadata_file = output_dir / f"{model_folder_name}_run_metadata.txt"

    with open(metadata_file, 'w') as f:
        f.write(f"Run generated at: {metadata['generated_at']}\n")
        f.write(f"Command: {metadata['command']}\n")
        f.write(f"Training mode: {metadata.get('training_mode', 'unknown')}\n")
        f.write(f"Data root: {metadata.get('data_root', 'unknown')}\n")
        f.write(f"Python version: {metadata['python']['version']}\n")
        f.write(f"PyTorch version: {metadata['package_versions']['torch']}\n")
        f.write(f"Torchvision version: {metadata['package_versions'].get('torchvision') or 'Not available'}\n")
        f.write(f"FastAI version: {metadata['package_versions'].get('fastai') or 'Not available'}\n")
        f.write(f"NumPy version: {metadata['package_versions'].get('numpy') or 'Not available'}\n")
        f.write(f"Git commit: {metadata['code'].get('commit') or 'unknown'}\n")
        f.write(f"Git dirty: {metadata['code'].get('dirty')}\n")
        if config_path:
            f.write(f"Config file: {config_path}\n")
        if "model_path" in metadata:
            f.write(f"Model path: {metadata['model_path']}\n")

    metadata_json_file = output_dir / f"{model_folder_name}_run_metadata.json"
    with open(metadata_json_file, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if not metadata_json_file.exists():
        raise RuntimeError(f"Required run metadata was not written: {metadata_json_file}")

    logger.info(f"Run metadata saved to: {metadata_file}")
    logger.info(f"Run metadata JSON saved to: {metadata_json_file}")

    # Copy config file if provided
    if config_path and Path(config_path).exists():
        config_file = output_dir / f"{model_folder_name}_config.yaml"
        import shutil
        shutil.copy2(config_path, config_file)
        logger.info(f"Config file copied to: {config_file}")


def metadata_path_for_model(model_path: Union[str, Path]) -> Path:
    """Return the adjacent metadata JSON path for an exported model artifact."""
    path = Path(model_path)
    return path.with_name(f"{path.stem}_run_metadata.json")


def load_supported_segmentation_artifact_metadata(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate metadata for a supported segmentation artifact."""
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        raise ValueError(f"Missing required segmentation artifact metadata: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get("training_mode") != "dynamic_full_image_patching":
        raise ValueError(
            f"Unsupported segmentation artifact training mode for {model_path}: "
            f"{metadata.get('training_mode')!r}"
        )
    if metadata.get("artifact_status") == "compatibility_only":
        raise ValueError(f"Compatibility-only segmentation artifact is not supported for transfer: {model_path}")
    required = ["command", "code", "package_versions", "data_root", "model_path"]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Segmentation artifact metadata missing required keys: {', '.join(missing)}")
    return metadata


def export_final_model(learn: Learner, output_dir: Path, model_folder_name: str) -> Path:
    """Export final model with prefixed filename and return the path."""
    export_fname = f"{model_folder_name}.pkl"
    learn.export(export_fname)
    model_path = output_dir / export_fname
    logger.info(f"Model saved to: {model_path}")
    return model_path
