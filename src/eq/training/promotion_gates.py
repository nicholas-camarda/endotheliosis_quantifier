"""Promotion-gate helpers for segmentation model quality review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


CropBox = Tuple[int, int, int, int]


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a mask as a 2D binary uint8 array."""
    arr = np.asarray(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def audit_binary_masks(mask_paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Audit foreground/background coverage for binary masks."""
    fractions: List[float] = []
    paths = [Path(path) for path in mask_paths]
    for path in paths:
        mask = load_binary_mask(path)
        fractions.append(float(mask.mean()))

    if not fractions:
        return {
            "count": 0,
            "background_only_count": 0,
            "full_foreground_count": 0,
            "background_only_rate": 0.0,
            "full_foreground_rate": 0.0,
            "foreground_fraction": {},
        }

    values = np.asarray(fractions, dtype=np.float64)
    return {
        "count": len(fractions),
        "background_only_count": int(np.sum(values == 0.0)),
        "full_foreground_count": int(np.sum(values == 1.0)),
        "background_only_rate": float(np.mean(values == 0.0)),
        "full_foreground_rate": float(np.mean(values == 1.0)),
        "foreground_fraction": {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        },
    }


def binary_dice_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute pixel-aggregated Dice and Jaccard for binary masks."""
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    intersection = float(np.logical_and(truth, pred).sum())
    truth_sum = float(truth.sum())
    pred_sum = float(pred.sum())
    union = float(np.logical_or(truth, pred).sum())
    dice_denom = truth_sum + pred_sum
    return {
        "dice": 1.0 if dice_denom == 0 else float((2.0 * intersection) / dice_denom),
        "jaccard": 1.0 if union == 0 else float(intersection / union),
    }


def trivial_baseline_metrics(masks: Sequence[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compute all-background and all-foreground baselines over validation masks."""
    if not masks:
        raise ValueError("At least one validation mask is required for baseline metrics.")
    truth = np.concatenate([np.asarray(mask).astype(bool).reshape(-1) for mask in masks])
    all_background = np.zeros_like(truth, dtype=bool)
    all_foreground = np.ones_like(truth, dtype=bool)
    return {
        "all_background": binary_dice_jaccard(truth, all_background),
        "all_foreground": binary_dice_jaccard(truth, all_foreground),
    }


def _crop_boxes(height: int, width: int, crop_size: int) -> List[CropBox]:
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    ys = sorted(set([0, max(0, height - crop_size), height // 2 - crop_size // 2]))
    xs = sorted(set([0, max(0, width - crop_size), width // 2 - crop_size // 2]))
    boxes: List[CropBox] = []
    seen = set()
    for top in ys:
        for left in xs:
            top = int(np.clip(top, 0, max(0, height - crop_size)))
            left = int(np.clip(left, 0, max(0, width - crop_size)))
            box = (left, top, left + crop_size, top + crop_size)
            if box not in seen:
                boxes.append(box)
                seen.add(box)
    return boxes


def _positive_center_crop_box(mask: np.ndarray, crop_size: int) -> CropBox | None:
    pos = np.column_stack(np.where(mask > 0))
    if pos.size == 0:
        return None
    y_min, x_min = pos.min(axis=0)
    y_max, x_max = pos.max(axis=0)
    center_y = int((y_min + y_max) // 2)
    center_x = int((x_min + x_max) // 2)
    height, width = mask.shape
    top = int(np.clip(center_y - crop_size // 2, 0, max(0, height - crop_size)))
    left = int(np.clip(center_x - crop_size // 2, 0, max(0, width - crop_size)))
    return (left, top, left + crop_size, top + crop_size)


def _crop_labels(crop: np.ndarray) -> List[str]:
    frac = float(crop.mean()) if crop.size else 0.0
    if frac == 0.0:
        return ["background"]
    touches_edge = bool(
        crop[0, :].any() or crop[-1, :].any() or crop[:, 0].any() or crop[:, -1].any()
    )
    labels = ["positive"]
    if touches_edge:
        labels.insert(0, "boundary")
    return labels


def deterministic_validation_manifest(
    mask_paths: Iterable[str | Path],
    crop_size: int,
    examples_per_category: int = 2,
) -> List[Dict[str, Any]]:
    """Select deterministic background, boundary, and positive validation crops."""
    categories: Dict[str, List[Dict[str, Any]]] = {
        "background": [],
        "boundary": [],
        "positive": [],
    }
    for path in sorted(Path(p) for p in mask_paths):
        mask = load_binary_mask(path)
        height, width = mask.shape
        boxes = _crop_boxes(height, width, crop_size)
        positive_box = _positive_center_crop_box(mask, crop_size)
        if positive_box is not None and positive_box not in boxes:
            boxes.append(positive_box)
        for box in boxes:
            left, top, right, bottom = box
            crop = mask[top:bottom, left:right]
            frac = float(crop.mean()) if crop.size else 0.0
            labels = _crop_labels(crop)
            for category in labels:
                if len(categories[category]) >= examples_per_category:
                    continue
                categories[category].append(
                    {
                        "mask_path": str(path),
                        "crop_box": [left, top, right, bottom],
                        "category": category,
                        "foreground_fraction": frac,
                    }
                )

    manifest: List[Dict[str, Any]] = []
    for category in ("background", "boundary", "positive"):
        if len(categories[category]) < examples_per_category:
            raise ValueError(
                f"Deterministic validation manifest lacks required {category} examples: "
                f"found {len(categories[category])}, required {examples_per_category}."
            )
        manifest.extend(categories[category])
    return manifest


def evaluate_prediction_degeneracy(
    truth_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    foreground_fraction_tolerance: float = 0.02,
    baseline_margin: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate whether predictions are all-background or all-foreground degenerate."""
    if len(truth_masks) != len(pred_masks):
        raise ValueError("truth_masks and pred_masks must have the same length.")
    if not truth_masks:
        raise ValueError("At least one truth/prediction pair is required.")

    pred_fractions = [float(np.asarray(mask).astype(bool).mean()) for mask in pred_masks]
    baselines = trivial_baseline_metrics([np.asarray(mask).astype(bool) for mask in truth_masks])
    truth = np.concatenate([np.asarray(mask).astype(bool).reshape(-1) for mask in truth_masks])
    pred = np.concatenate([np.asarray(mask).astype(bool).reshape(-1) for mask in pred_masks])
    candidate = binary_dice_jaccard(truth, pred)

    reasons: List[str] = []
    if all(frac <= foreground_fraction_tolerance for frac in pred_fractions):
        reasons.append("predictions_are_all_background")
    if all(frac >= 1.0 - foreground_fraction_tolerance for frac in pred_fractions):
        reasons.append("predictions_are_all_foreground")
    for baseline_name, baseline_metrics in baselines.items():
        for metric_name in ("dice", "jaccard"):
            if candidate[metric_name] <= baseline_metrics[metric_name] + baseline_margin:
                reasons.append(f"candidate_does_not_clear_{baseline_name}_{metric_name}_baseline")

    return {
        "blocked": bool(reasons),
        "reasons": reasons,
        "candidate": candidate,
        "baselines": baselines,
        "prediction_foreground_fraction": {
            "min": float(np.min(pred_fractions)),
            "median": float(np.median(pred_fractions)),
            "max": float(np.max(pred_fractions)),
        },
    }


def audit_manifest_crops(manifest: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize deterministic validation manifest crop coverage."""
    if not manifest:
        return {"count": 0, "categories": {}, "foreground_fraction": {}}
    fractions = np.asarray([float(item["foreground_fraction"]) for item in manifest], dtype=np.float64)
    categories: Dict[str, int] = {}
    for item in manifest:
        category = str(item["category"])
        categories[category] = categories.get(category, 0) + 1
    return {
        "count": len(manifest),
        "categories": categories,
        "background_only_rate": float(np.mean(fractions == 0.0)),
        "full_foreground_rate": float(np.mean(fractions == 1.0)),
        "foreground_fraction": {
            "min": float(np.min(fractions)),
            "median": float(np.median(fractions)),
            "p75": float(np.percentile(fractions, 75)),
            "p95": float(np.percentile(fractions, 95)),
            "max": float(np.max(fractions)),
        },
    }


def evaluate_glomeruli_promotion_candidate(
    truth_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    manifest: Sequence[Dict[str, Any]],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a single pass/block report for a glomeruli promotion candidate."""
    reasons: List[str] = []
    if provenance.get("training_mode") != "dynamic_full_image_patching":
        reasons.append("artifact_training_mode_is_not_dynamic_full_image_patching")
    if provenance.get("scientific_promotion_status") == "promoted":
        reasons.append("artifact_already_claims_promotion_before_gate")

    manifest_audit = audit_manifest_crops(manifest)
    for category in ("background", "boundary", "positive"):
        if manifest_audit.get("categories", {}).get(category, 0) == 0:
            reasons.append(f"manifest_missing_{category}_examples")

    degeneracy = evaluate_prediction_degeneracy(truth_masks, pred_masks)
    reasons.extend(degeneracy["reasons"])
    return {
        "blocked": bool(reasons),
        "reasons": reasons,
        "manifest_audit": manifest_audit,
        "prediction_review": degeneracy,
        "provenance": provenance,
    }
