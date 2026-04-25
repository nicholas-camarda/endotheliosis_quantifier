"""Pytest-oriented audit helpers for segmentation validation evidence.

These helpers intentionally do not expose a CLI.  They provide small,
deterministic checks that tests and candidate-comparison reporting can compose
when deciding whether a segmentation result supports promotion-facing claims.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from eq.training.promotion_gates import binary_dice_jaccard, binary_precision_recall

RUNTIME_USE_AVAILABLE = "available_research_use"
RUNTIME_USE_UNAVAILABLE = "unavailable"

PROMOTION_ELIGIBLE = "promotion_eligible"
PROMOTION_NOT_ELIGIBLE = "not_promotion_eligible"
PROMOTION_AUDIT_MISSING = "audit_missing"
PROMOTION_INSUFFICIENT = "insufficient_evidence_for_promotion"

MITO_SCOPE_HELDOUT = "heldout_test_preserved"
MITO_SCOPE_ALL_AVAILABLE = "all_available_pretraining"
MITO_INFERENCE_HELDOUT = "heldout_evaluable"
MITO_INFERENCE_NOT_APPLICABLE = "not_applicable_for_inference_claim"
MITO_INFERENCE_AUDIT_MISSING = "audit_missing"

ROOT_CAUSE_CLASSES = (
    "image_mask_pairing_error",
    "transform_alignment_error",
    "mask_binarization_error",
    "class_channel_or_threshold_error",
    "resize_policy_artifact",
    "split_or_panel_bias",
    "training_signal_insufficient",
    "mitochondria_base_defect",
    "negative_background_supervision_missing",
    "true_model_underfit",
)


def _path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(Path(str(value)).expanduser())


def _subject_from_path(value: Any) -> str:
    path = Path(str(value))
    if path.parent != path and path.parent.name not in {"images", "masks"}:
        return path.parent.name
    stem = path.stem
    stem = stem.removesuffix("_mask")
    match = re.match(r"(?P<subject>.+?)_image\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group("subject")
    return stem


def _truthy_path_set(values: Iterable[Any]) -> set[str]:
    return {_path_key(value) for value in values if _path_key(value)}


def _summary(values: Sequence[float]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_csv_rows(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write row dictionaries with stable field ordering."""
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def audit_paired_root_contract(data_root: str | Path) -> Dict[str, Any]:
    """Audit an `images/` plus `masks/` full-image root without fallback pairing."""
    root = Path(data_root).expanduser()
    images_root = root / "images"
    masks_root = root / "masks"
    if not images_root.is_dir() or not masks_root.is_dir():
        return {
            "ok": False,
            "reason": "missing_images_or_masks_directory",
            "data_root": str(root),
            "image_count": 0,
            "mask_count": 0,
            "unpaired_images": [],
            "unpaired_masks": [],
        }

    image_paths = sorted(
        path for path in images_root.rglob("*") if path.is_file() and not path.name.startswith(".")
    )
    mask_paths = sorted(
        path for path in masks_root.rglob("*") if path.is_file() and not path.name.startswith(".")
    )

    def image_key(path: Path) -> str:
        return str(path.relative_to(images_root).with_suffix(""))

    def mask_key(path: Path) -> str:
        rel = path.relative_to(masks_root)
        stem = rel.stem
        if stem.endswith("_mask"):
            stem = stem.removesuffix("_mask")
        elif stem.startswith("mask_"):
            stem = stem.removeprefix("mask_")
        return str(rel.with_name(stem).with_suffix(""))

    image_keys = {image_key(path): path for path in image_paths}
    mask_keys = {mask_key(path): path for path in mask_paths}
    unpaired_images = [str(image_keys[key]) for key in sorted(set(image_keys) - set(mask_keys))]
    unpaired_masks = [str(mask_keys[key]) for key in sorted(set(mask_keys) - set(image_keys))]
    return {
        "ok": not unpaired_images and not unpaired_masks,
        "reason": "ok" if not unpaired_images and not unpaired_masks else "unpaired_images_or_masks",
        "data_root": str(root),
        "image_count": len(image_paths),
        "mask_count": len(mask_paths),
        "paired_count": len(set(image_keys) & set(mask_keys)),
        "unpaired_images": unpaired_images,
        "unpaired_masks": unpaired_masks,
    }


def audit_manifest_rows(rows: Sequence[Mapping[str, Any]], runtime_root: str | Path | None = None) -> Dict[str, Any]:
    """Audit admitted manifest rows supplied by tests or runtime integration."""
    root = Path(runtime_root).expanduser() if runtime_root else None
    checked_rows: list[Dict[str, Any]] = []
    missing_pairs: list[Dict[str, Any]] = []
    by_cohort: Dict[str, int] = defaultdict(int)
    by_lane: Dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        image_path = Path(str(row.get("image_path", "")))
        mask_path = Path(str(row.get("mask_path", "")))
        if root is not None and not image_path.is_absolute():
            image_path = root / image_path
        if root is not None and not mask_path.is_absolute():
            mask_path = root / mask_path
        cohort_id = str(row.get("cohort_id", "unknown") or "unknown")
        lane = str(row.get("lane_assignment", "unknown") or "unknown")
        by_cohort[cohort_id] += 1
        by_lane[lane] += 1
        record = {
            "row_index": index,
            "cohort_id": cohort_id,
            "lane_assignment": lane,
            "subject_id": str(row.get("subject_id") or _subject_from_path(image_path)),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_exists": image_path.exists(),
            "mask_exists": mask_path.exists(),
        }
        checked_rows.append(record)
        if not record["image_exists"] or not record["mask_exists"]:
            missing_pairs.append(record)
    return {
        "ok": not missing_pairs,
        "row_count": len(rows),
        "missing_pair_count": len(missing_pairs),
        "by_cohort_id": dict(sorted(by_cohort.items())),
        "by_lane_assignment": dict(sorted(by_lane.items())),
        "rows": checked_rows,
        "missing_pairs": missing_pairs,
    }


def audit_split_overlap(
    *,
    train_images: Iterable[Any] | None,
    valid_images: Iterable[Any] | None = None,
    promotion_manifest: Sequence[Mapping[str, Any]] | None = None,
    subject_id_func: Callable[[Any], str] | None = None,
) -> Dict[str, Any]:
    """Compare candidate split provenance against deterministic promotion rows."""
    train_set = _truthy_path_set(train_images or [])
    valid_set = _truthy_path_set(valid_images or [])
    manifest_images = _truthy_path_set(
        (row.get("image_path") for row in (promotion_manifest or []))
    )
    if not train_set:
        return {
            "status": PROMOTION_AUDIT_MISSING,
            "promotion_evidence_status": PROMOTION_AUDIT_MISSING,
            "reason": "missing_train_split_provenance",
            "runtime_use_status": RUNTIME_USE_AVAILABLE,
            "train_image_overlap_count": 0,
            "subject_overlap_count": 0,
            "overlapping_train_images": [],
            "overlapping_train_subjects": [],
        }

    subject_getter = subject_id_func or _subject_from_path
    train_subjects = {subject_getter(path) for path in train_set}
    manifest_subjects = {subject_getter(path) for path in manifest_images}
    image_overlap = sorted(train_set & manifest_images)
    subject_overlap = sorted(train_subjects & manifest_subjects)
    blocked = bool(image_overlap or subject_overlap)
    return {
        "status": PROMOTION_NOT_ELIGIBLE if blocked else PROMOTION_ELIGIBLE,
        "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE if blocked else PROMOTION_ELIGIBLE,
        "reason": (
            "train_evaluation_overlap"
            if image_overlap
            else "train_evaluation_subject_overlap"
            if subject_overlap
            else "ok"
        ),
        "runtime_use_status": RUNTIME_USE_AVAILABLE,
        "train_image_count": len(train_set),
        "valid_image_count": len(valid_set),
        "manifest_image_count": len(manifest_images),
        "train_image_overlap_count": len(image_overlap),
        "subject_overlap_count": len(subject_overlap),
        "overlapping_train_images": image_overlap,
        "overlapping_train_subjects": subject_overlap,
    }


def classify_artifact_status(
    provenance: Mapping[str, Any] | None,
    *,
    loadable: bool = True,
    requires_transfer_base: bool = False,
) -> Dict[str, Any]:
    """Return runtime-use and promotion-evidence status on separate axes."""
    if not loadable:
        return {
            "runtime_use_status": RUNTIME_USE_UNAVAILABLE,
            "promotion_evidence_status": PROMOTION_AUDIT_MISSING,
            "reasons": ["artifact_not_loadable"],
        }
    metadata = dict(provenance or {})
    reasons: list[str] = []
    if not metadata:
        reasons.append("missing_run_metadata")
    if not metadata.get("train_images"):
        reasons.append("missing_train_split_provenance")
    if not metadata.get("valid_images"):
        reasons.append("missing_valid_split_provenance")
    if not (metadata.get("source_image_size_summary") or {}).get("count"):
        reasons.append("missing_source_image_size_summary")
    if not (metadata.get("source_mask_size_summary") or {}).get("count"):
        reasons.append("missing_source_mask_size_summary")
    if requires_transfer_base and not metadata.get("transfer_base_artifact_path") and not metadata.get("base_model_path"):
        reasons.append("missing_transfer_base_provenance")
    if requires_transfer_base:
        base_metadata = metadata.get("transfer_base_metadata") or {}
        if not base_metadata.get("mitochondria_training_scope") and not metadata.get("transfer_base_mitochondria_training_scope"):
            reasons.append("missing_transfer_base_mitochondria_scope")
        if not base_metadata.get("mitochondria_inference_claim_status"):
            reasons.append("missing_transfer_base_inference_claim_status")
    promotion_status = PROMOTION_AUDIT_MISSING if reasons else PROMOTION_INSUFFICIENT
    return {
        "runtime_use_status": RUNTIME_USE_AVAILABLE,
        "promotion_evidence_status": promotion_status,
        "reasons": reasons,
    }


def source_size_summary(image_paths: Iterable[Any], mask_paths: Iterable[Any] | None = None) -> Dict[str, Any]:
    """Summarize source image and mask dimensions without loading tensor libraries."""
    image_sizes: list[tuple[int, int]] = []
    mask_sizes: list[tuple[int, int]] = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                image_sizes.append(tuple(int(v) for v in img.size))
        except Exception:
            continue
    for path in mask_paths or []:
        try:
            with Image.open(path) as mask:
                mask_sizes.append(tuple(int(v) for v in mask.size))
        except Exception:
            continue

    def summarize_sizes(values: Sequence[tuple[int, int]]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        widths = [float(width) for width, _ in values]
        heights = [float(height) for _, height in values]
        return {"count": len(values), "width": _summary(widths), "height": _summary(heights)}

    return {
        "source_image_size_summary": summarize_sizes(image_sizes),
        "source_mask_size_summary": summarize_sizes(mask_sizes),
    }


def manifest_context_summary(data_root: str | Path) -> Dict[str, Any]:
    """Summarize manifest rows for manifest-backed cohort roots, if present."""
    root = Path(data_root).expanduser()
    manifest_path = root / "manifest.csv" if root.name == "cohorts" else root.parent / "manifest.csv"
    if not manifest_path.exists():
        return {
            "manifest_path": None,
            "manifest_rows": 0,
            "manifest_training_eligible_rows": 0,
            "manifest_cohort_counts": {},
            "manifest_lane_counts": {},
        }
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in reader]
    eligible = [
        row
        for row in rows
        if row.get("admission_status") == "admitted"
        and row.get("image_path")
        and row.get("mask_path")
        and row.get("lane_assignment")
    ]
    cohort_counts: Dict[str, int] = defaultdict(int)
    lane_counts: Dict[str, int] = defaultdict(int)
    for row in eligible:
        cohort_counts[row.get("cohort_id") or "unknown"] += 1
        lane_counts[row.get("lane_assignment") or "unknown"] += 1
    return {
        "manifest_path": str(manifest_path),
        "manifest_rows": len(rows),
        "manifest_training_eligible_rows": len(eligible),
        "manifest_cohort_counts": dict(sorted(cohort_counts.items())),
        "manifest_lane_counts": dict(sorted(lane_counts.items())),
    }


def resize_policy_record(
    *,
    source_image_size: Sequence[int] | None = None,
    source_mask_size: Sequence[int] | None = None,
    crop_size: int,
    output_size: int,
    aspect_ratio_policy: str = "squish",
    resize_method: str = "ResizeMethod.Squish",
    image_interpolation: str = "fastai_default_image",
    mask_interpolation: str = "fastai_default_mask",
    mask_binarization_after_resize: str = "mask_values_thresholded_to_binary",
    prediction_resize_back_method: str = "PredictionCore.resize_prediction_to_match",
    threshold_resize_order: str = "resize_probability_then_threshold",
    split: str | None = None,
    category: str | None = None,
    cohort_id: str | None = None,
    lane_assignment: str | None = None,
) -> Dict[str, Any]:
    """Build one structured resize-policy audit row."""
    crop = int(crop_size)
    output = int(output_size)
    return {
        "split": split,
        "category": category,
        "cohort_id": cohort_id,
        "lane_assignment": lane_assignment,
        "source_image_size": list(source_image_size) if source_image_size is not None else None,
        "source_mask_size": list(source_mask_size) if source_mask_size is not None else None,
        "crop_size": crop,
        "output_size": output,
        "crop_to_output_resize_ratio": float(crop / output) if output else None,
        "aspect_ratio_policy": aspect_ratio_policy,
        "resize_method": resize_method,
        "image_interpolation": image_interpolation,
        "mask_interpolation": mask_interpolation,
        "mask_binarization_after_resize": mask_binarization_after_resize,
        "prediction_resize_back_method": prediction_resize_back_method,
        "threshold_resize_order": threshold_resize_order,
    }


def audit_resize_policy_parity(
    training_policy: Mapping[str, Any],
    evaluation_policy: Mapping[str, Any],
    keys: Sequence[str] = (
        "crop_size",
        "output_size",
        "aspect_ratio_policy",
        "resize_method",
        "mask_binarization_after_resize",
        "prediction_resize_back_method",
        "threshold_resize_order",
    ),
) -> Dict[str, Any]:
    mismatches = [
        {
            "field": key,
            "training": training_policy.get(key),
            "evaluation": evaluation_policy.get(key),
        }
        for key in keys
        if training_policy.get(key) != evaluation_policy.get(key)
    ]
    return {
        "ok": not mismatches,
        "promotion_evidence_status": PROMOTION_ELIGIBLE if not mismatches else PROMOTION_NOT_ELIGIBLE,
        "reason": "ok" if not mismatches else "resize_policy_mismatch",
        "mismatches": mismatches,
    }


def audit_preprocessing_parity(
    *,
    learner_consistent_preprocessing: bool,
    supported_threshold_semantics: bool,
    training_policy: Mapping[str, Any] | None = None,
    evaluation_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    reasons: list[str] = []
    if not learner_consistent_preprocessing:
        reasons.append("evaluation_preprocessing_not_learner_consistent")
    if not supported_threshold_semantics:
        reasons.append("unsupported_threshold_semantics")
    resize = None
    if training_policy is not None and evaluation_policy is not None:
        resize = audit_resize_policy_parity(training_policy, evaluation_policy)
        if not resize["ok"]:
            reasons.append("resize_policy_mismatch")
    return {
        "ok": not reasons,
        "promotion_evidence_status": PROMOTION_ELIGIBLE if not reasons else PROMOTION_NOT_ELIGIBLE,
        "reasons": reasons,
        "resize_policy_parity": resize,
    }


def mask_binary_values(mask: Any) -> set[int]:
    arr = np.asarray(mask)
    return {int(value) for value in np.unique(arr)}


def check_binary_mask_preservation(
    before: Any,
    after: Any,
    *,
    foreground_fraction_tolerance: float = 0.05,
) -> Dict[str, Any]:
    before_arr = (np.asarray(before) > 0).astype(np.uint8)
    after_arr = np.asarray(after)
    after_values = mask_binary_values(after_arr)
    after_bin = (after_arr > 0).astype(np.uint8)
    before_fraction = float(before_arr.mean()) if before_arr.size else 0.0
    after_fraction = float(after_bin.mean()) if after_bin.size else 0.0
    area_delta = abs(after_fraction - before_fraction)
    ok = after_values.issubset({0, 1, 255}) and area_delta <= float(foreground_fraction_tolerance)
    return {
        "ok": ok,
        "before_foreground_fraction": before_fraction,
        "after_foreground_fraction": after_fraction,
        "foreground_fraction_delta": float(area_delta),
        "after_values": sorted(after_values),
        "promotion_evidence_status": PROMOTION_ELIGIBLE if ok else PROMOTION_NOT_ELIGIBLE,
        "reason": "ok" if ok else "mask_resize_or_binarization_changed_binary_semantics",
    }


def audit_transform_alignment(
    image_marker: Any,
    mask_marker: Any,
    *,
    tolerance_pixels: float = 0.0,
) -> Dict[str, Any]:
    """Check whether two fixture markers occupy the same bounding box."""
    image_arr = np.asarray(image_marker)
    mask_arr = np.asarray(mask_marker)
    if image_arr.ndim == 3:
        image_arr = image_arr[..., 0]
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]

    def bbox(arr: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.column_stack(np.where(arr > 0))
        if coords.size == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    image_box = bbox(image_arr)
    mask_box = bbox(mask_arr)
    if image_box is None or mask_box is None:
        return {
            "ok": False,
            "reason": "missing_alignment_marker",
            "image_bbox": image_box,
            "mask_bbox": mask_box,
        }
    delta = max(abs(a - b) for a, b in zip(image_box, mask_box))
    ok = delta <= float(tolerance_pixels)
    return {
        "ok": ok,
        "reason": "ok" if ok else "transform_alignment_error",
        "image_bbox": image_box,
        "mask_bbox": mask_box,
        "max_bbox_delta_pixels": float(delta),
    }


def audit_datablock_sampling(
    dls: Any,
    *,
    crop_size: int,
    output_size: int,
    min_pos_pixels: int,
    batches: int = 2,
    output_csv: Path | None = None,
) -> Dict[str, Any]:
    """Sample FastAI DataLoaders and summarize foreground burden by split."""
    rows: list[Dict[str, Any]] = []

    def visit(dl: Any, split: str) -> None:
        iterator = iter(dl)
        for batch_index in range(int(batches)):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                continue
            yb = batch[1]
            try:
                import torch

                y = yb.detach().cpu() if torch.is_tensor(yb) else torch.as_tensor(yb)
            except Exception:
                y = np.asarray(yb)
            if hasattr(y, "numpy"):
                y = y.numpy()
            arr = np.asarray(y)
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0]
            for example_index, mask in enumerate(arr):
                mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
                foreground_pixels = int(mask_bin.sum())
                rows.append(
                    {
                        "split": split,
                        "batch_index": batch_index,
                        "example_index": example_index,
                        "crop_size": int(crop_size),
                        "output_size": int(output_size),
                        "foreground_fraction": float(mask_bin.mean()) if mask_bin.size else 0.0,
                        "foreground_pixels": foreground_pixels,
                        "any_positive": foreground_pixels > 0,
                        "meets_min_pos_pixels": foreground_pixels >= int(min_pos_pixels),
                    }
                )

    if hasattr(dls, "train"):
        visit(dls.train, "train")
    if hasattr(dls, "valid"):
        visit(dls.valid, "valid")
    if output_csv is not None:
        write_csv_rows(rows, output_csv)
    by_split: Dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(float(row["foreground_fraction"]))
    return {
        "rows": rows,
        "by_split": {split: _summary(values) for split, values in sorted(by_split.items())},
        "foreground_heavy_validation_panel": bool(
            by_split.get("valid") and np.median(by_split["valid"]) > 0.5
        ),
    }


def audit_dynamic_patching_datablock(
    data_root: str | Path,
    *,
    bs: int = 2,
    crop_size: int,
    output_size: int,
    positive_focus_p: float,
    min_pos_pixels: int,
    pos_crop_attempts: int,
    stage: str = "glomeruli",
    batches: int = 2,
    output_csv: Path | None = None,
) -> Dict[str, Any]:
    """Build supported dynamic-patching DataLoaders and audit sampled crops."""
    from eq.data_management.datablock_loader import (
        build_segmentation_dls_dynamic_patching,
    )

    dls = build_segmentation_dls_dynamic_patching(
        data_root,
        bs=bs,
        num_workers=0,
        crop_size=crop_size,
        output_size=output_size,
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
        stage=stage,
    )
    return audit_datablock_sampling(
        dls,
        crop_size=crop_size,
        output_size=output_size,
        min_pos_pixels=min_pos_pixels,
        batches=batches,
        output_csv=output_csv,
    )


def aggregate_metric_by_category(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Aggregate prediction rows by family, category, cohort, and lane."""
    grouped: Dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("family", "")),
            str(row.get("category", "unknown")),
            str(row.get("cohort_id", "unknown")),
            str(row.get("lane_assignment", "unknown")),
        )
        grouped[key].append(row)
    results: list[Dict[str, Any]] = []
    metric_names = ("dice", "jaccard", "precision", "recall")
    for (family, category, cohort_id, lane), group_rows in sorted(grouped.items()):
        record: Dict[str, Any] = {
            "family": family,
            "category": category,
            "cohort_id": cohort_id,
            "lane_assignment": lane,
            "count": len(group_rows),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
            record[metric] = float(np.mean(values)) if values else None
        record["truth_foreground_fraction_median"] = float(
            np.median([float(row.get("truth_foreground_fraction", 0.0)) for row in group_rows])
        )
        record["prediction_foreground_fraction_median"] = float(
            np.median([float(row.get("prediction_foreground_fraction", 0.0)) for row in group_rows])
        )
        results.append(record)
    return results


def audit_prediction_shapes(
    rows: Sequence[Mapping[str, Any]],
    *,
    background_false_positive_limit: float = 0.02,
    positive_overcoverage_ratio_limit: float = 1.5,
    positive_undercoverage_ratio_limit: float = 0.5,
    minimum_category_dice: float = 0.3,
    minimum_category_jaccard: float = 0.15,
    minimum_positive_precision: float = 0.1,
    minimum_positive_recall: float = 0.5,
) -> Dict[str, Any]:
    """Detect broad oversegmentation and category-specific failure."""
    audit_rows: list[Dict[str, Any]] = []
    reasons_by_family: Dict[str, set[str]] = defaultdict(set)
    category_gate = audit_category_gates(
        rows,
        background_false_positive_limit=background_false_positive_limit,
        positive_overcoverage_ratio_limit=positive_overcoverage_ratio_limit,
        positive_undercoverage_ratio_limit=positive_undercoverage_ratio_limit,
        minimum_category_dice=minimum_category_dice,
        minimum_category_jaccard=minimum_category_jaccard,
        minimum_positive_precision=minimum_positive_precision,
        minimum_positive_recall=minimum_positive_recall,
    )
    gate_reasons_by_key: Dict[tuple[str, str, str], set[str]] = defaultdict(set)
    def _manifest_key(value: Any) -> str:
        return "" if value is None else str(value)

    for gate_row in category_gate["rows"]:
        if str(gate_row.get("gate_passed")) == "False" or gate_row.get("gate_passed") is False:
            key = (
                str(gate_row.get("family", "unknown")),
                str(gate_row.get("category", "unknown")),
                _manifest_key(gate_row.get("manifest_index")),
            )
            gate_reasons_by_key[key].add(str(gate_row.get("failure_reason") or gate_row.get("gate_name")))
    for row in rows:
        family = str(row.get("family", "unknown"))
        category = str(row.get("category", "unknown"))
        manifest_index = _manifest_key(row.get("manifest_index"))
        truth_fg = float(row.get("truth_foreground_fraction", 0.0) or 0.0)
        pred_fg = float(row.get("prediction_foreground_fraction", 0.0) or 0.0)
        overcoverage_ratio = None if truth_fg <= 0 else float(pred_fg / truth_fg)
        reasons = sorted(gate_reasons_by_key.get((family, category, manifest_index), set()))
        if reasons:
            reasons.append("category_metric_failure")
        for reason in reasons:
            reasons_by_family[family].add(reason)
        audit_rows.append(
            {
                **dict(row),
                "foreground_overcoverage_ratio": overcoverage_ratio,
                "shape_gate_failed": bool(reasons),
                "shape_gate_reasons": "|".join(reasons),
            }
        )
    families = sorted({str(row.get("family", "unknown")) for row in rows} | set(reasons_by_family))
    family_status = {
        family: {
            "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE if reasons_by_family.get(family) else PROMOTION_ELIGIBLE,
            "reasons": sorted(reasons_by_family.get(family, set())),
        }
        for family in families
    }
    return {
        "rows": audit_rows,
        "family_status": family_status,
        "blocked": any(status["reasons"] for status in family_status.values()),
        "category_gate": category_gate,
    }


def audit_category_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    background_false_positive_limit: float = 0.02,
    minimum_background_pixel_accuracy: float = 0.98,
    positive_overcoverage_ratio_limit: float = 1.5,
    positive_undercoverage_ratio_limit: float = 0.5,
    minimum_category_dice: float = 0.3,
    minimum_category_jaccard: float = 0.15,
    minimum_positive_precision: float = 0.1,
    minimum_positive_recall: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate category-specific gates without using empty-background Dice as the background gate."""
    gate_rows: list[Dict[str, Any]] = []
    reasons_by_family: Dict[str, set[str]] = defaultdict(set)
    categories_by_family: Dict[str, set[str]] = defaultdict(set)

    def add_gate(
        *,
        row: Mapping[str, Any],
        gate_name: str,
        metric_name: str,
        observed_value: float | None,
        required_comparator: str,
        required_value: float,
        rationale: str,
        failure_reason: str,
    ) -> None:
        family = str(row.get("family", "unknown"))
        category = str(row.get("category", "unknown"))
        categories_by_family[family].add(category)
        if observed_value is None:
            passed = False
        elif required_comparator == "<=":
            passed = observed_value <= required_value
        elif required_comparator == ">=":
            passed = observed_value >= required_value
        else:
            raise ValueError(f"Unsupported category gate comparator: {required_comparator}")
        if not passed:
            reasons_by_family[family].add(failure_reason)
        gate_rows.append(
            {
                "family": family,
                "category": category,
                "manifest_index": row.get("manifest_index"),
                "cohort_id": row.get("cohort_id"),
                "lane_assignment": row.get("lane_assignment"),
                "image_path": row.get("image_path"),
                "mask_path": row.get("mask_path"),
                "crop_box": row.get("crop_box"),
                "gate_name": gate_name,
                "metric_name": metric_name,
                "observed_value": observed_value,
                "required_comparator": required_comparator,
                "required_value": required_value,
                "gate_passed": passed,
                "threshold": row.get("threshold"),
                "threshold_policy_status": row.get("threshold_policy_status"),
                "rationale": rationale,
                "failure_reason": "" if passed else failure_reason,
            }
        )

    for row in rows:
        category = str(row.get("category", "unknown"))
        truth_fg = float(row.get("truth_foreground_fraction", 0.0) or 0.0)
        pred_fg = float(row.get("prediction_foreground_fraction", 0.0) or 0.0)
        pixel_accuracy = row.get("pixel_accuracy")
        pixel_accuracy_value = None if pixel_accuracy in (None, "") else float(pixel_accuracy)
        if category == "background":
            add_gate(
                row=row,
                gate_name="background_false_positive_control",
                metric_name="prediction_foreground_fraction",
                observed_value=pred_fg,
                required_comparator="<=",
                required_value=background_false_positive_limit,
                rationale="True-background crops are gated by predicted foreground fraction, not empty-mask Dice/Jaccard.",
                failure_reason="background_false_positive_foreground_excess",
            )
            add_gate(
                row=row,
                gate_name="background_pixel_correctness",
                metric_name="pixel_accuracy",
                observed_value=pixel_accuracy_value,
                required_comparator=">=",
                required_value=minimum_background_pixel_accuracy,
                rationale="Background crops should remain mostly background at the selected threshold.",
                failure_reason="background_pixel_accuracy_low",
            )
            continue

        if category not in {"positive", "boundary"}:
            continue

        dice = float(row.get("dice", 0.0) or 0.0)
        jaccard = float(row.get("jaccard", 0.0) or 0.0)
        precision = float(row.get("precision", 0.0) or 0.0)
        recall = float(row.get("recall", 0.0) or 0.0)
        ratio = None if truth_fg <= 0 else float(pred_fg / truth_fg)
        add_gate(
            row=row,
            gate_name="foreground_overlap_dice",
            metric_name="dice",
            observed_value=dice,
            required_comparator=">=",
            required_value=minimum_category_dice,
            rationale="Foreground-containing crops require minimum overlap.",
            failure_reason="low_foreground_dice",
        )
        add_gate(
            row=row,
            gate_name="foreground_overlap_jaccard",
            metric_name="jaccard",
            observed_value=jaccard,
            required_comparator=">=",
            required_value=minimum_category_jaccard,
            rationale="Foreground-containing crops require minimum overlap.",
            failure_reason="low_foreground_jaccard",
        )
        add_gate(
            row=row,
            gate_name="foreground_precision",
            metric_name="precision",
            observed_value=precision,
            required_comparator=">=",
            required_value=minimum_positive_precision,
            rationale="Foreground-containing crops should not be dominated by false-positive foreground.",
            failure_reason="low_foreground_precision",
        )
        add_gate(
            row=row,
            gate_name="foreground_recall",
            metric_name="recall",
            observed_value=recall,
            required_comparator=">=",
            required_value=minimum_positive_recall,
            rationale="Foreground-containing crops should preserve most annotated foreground.",
            failure_reason="low_foreground_recall",
        )
        add_gate(
            row=row,
            gate_name="foreground_size_overcoverage",
            metric_name="prediction_to_truth_foreground_ratio",
            observed_value=ratio,
            required_comparator="<=",
            required_value=positive_overcoverage_ratio_limit,
            rationale="Foreground prediction area should not substantially exceed annotated area.",
            failure_reason="positive_or_boundary_overcoverage",
        )
        add_gate(
            row=row,
            gate_name="foreground_size_undercoverage",
            metric_name="prediction_to_truth_foreground_ratio",
            observed_value=ratio,
            required_comparator=">=",
            required_value=positive_undercoverage_ratio_limit,
            rationale="Foreground prediction area should not substantially undershoot annotated area.",
            failure_reason="positive_or_boundary_undercoverage",
        )

    family_status = {
        family: {
            "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE if reasons_by_family.get(family) else PROMOTION_ELIGIBLE,
            "reasons": sorted(reasons_by_family.get(family, set())),
            "categories_evaluated": sorted(categories_by_family.get(family, set())),
            "failed_gate_count": sum(
                1
                for row in gate_rows
                if row["family"] == family and row["gate_passed"] is False
            ),
            "gate_count": sum(1 for row in gate_rows if row["family"] == family),
        }
        for family in sorted(categories_by_family)
    }
    return {
        "rows": gate_rows,
        "family_status": family_status,
        "blocked": any(status["reasons"] for status in family_status.values()),
    }


def metrics_for_masks(
    truth_mask: Any,
    pred_mask: Any,
) -> Dict[str, float]:
    metrics = binary_dice_jaccard(np.asarray(truth_mask), np.asarray(pred_mask))
    metrics.update(binary_precision_recall(np.asarray(truth_mask), np.asarray(pred_mask)))
    return metrics


def failure_reproduction_row(
    *,
    candidate_family: str,
    panel_id: str,
    artifact_path: str | Path | None,
    image_path: str | Path | None,
    mask_path: str | Path | None,
    crop_box: Sequence[int] | str | None,
    resize_policy: Mapping[str, Any] | None,
    threshold: float | None,
    prediction_tensor_shape: Sequence[int] | None,
    overlay_path: str | Path | None,
    root_causes: Sequence[str] | None = None,
    remediation_path: str | None = None,
) -> Dict[str, Any]:
    invalid = [cause for cause in (root_causes or []) if cause not in ROOT_CAUSE_CLASSES]
    if invalid:
        raise ValueError(f"Unknown root-cause class(es): {', '.join(invalid)}")

    def known(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        return bool(text) and not text.startswith(("not_recorded", "unknown", "audit_missing"))

    return {
        "candidate_family": candidate_family,
        "panel_id": panel_id,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "image_path": str(image_path) if image_path is not None else None,
        "mask_path": str(mask_path) if mask_path is not None else None,
        "crop_box": json.dumps(crop_box) if not isinstance(crop_box, str) else crop_box,
        "resize_policy": json.dumps(dict(resize_policy or {}), sort_keys=True),
        "threshold": threshold,
        "prediction_tensor_shape": (
            json.dumps([int(value) for value in prediction_tensor_shape])
            if prediction_tensor_shape is not None
            else None
        ),
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
        "traceable": all(
            known(value)
            for value in (artifact_path, image_path, mask_path, crop_box, overlay_path)
        ),
        "root_causes": "|".join(root_causes or []),
        "remediation_path": remediation_path,
    }


def validation_prediction_panel_trace_rows(
    *,
    model_folder_name: str,
    candidate_family: str,
    artifact_path: str | Path | None,
    image_paths: Sequence[str | Path | None],
    mask_paths: Sequence[str | Path | None],
    overlay_path: str | Path,
    prediction_tensor_shapes: Sequence[Sequence[int] | None],
    resize_policy: Mapping[str, Any] | None,
    threshold: float | None,
    crop_boxes: Sequence[Sequence[int] | str | None] | None = None,
) -> list[Dict[str, Any]]:
    """Build trace rows for the multi-panel validation-predictions PNG."""
    rows: list[Dict[str, Any]] = []
    crop_values = list(crop_boxes or [])
    for index, image_path in enumerate(image_paths):
        crop_box = crop_values[index] if index < len(crop_values) else "not_recorded_validation_datablock_crop"
        mask_path = mask_paths[index] if index < len(mask_paths) else None
        pred_shape = (
            prediction_tensor_shapes[index]
            if index < len(prediction_tensor_shapes)
            else None
        )
        rows.append(
            failure_reproduction_row(
                candidate_family=candidate_family,
                panel_id=f"{model_folder_name}-validation-{index + 1}",
                artifact_path=artifact_path,
                image_path=image_path,
                mask_path=mask_path,
                crop_box=crop_box,
                resize_policy=resize_policy,
                threshold=threshold,
                prediction_tensor_shape=pred_shape,
                overlay_path=overlay_path,
                root_causes=[],
                remediation_path="audit_trace_only_not_a_root_cause_classification",
            )
        )
    return rows


def classify_root_causes(
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map audit evidence flags to explicit root-cause classes."""
    causes: list[str] = []
    mapping = {
        "pairing_error": "image_mask_pairing_error",
        "transform_alignment_error": "transform_alignment_error",
        "mask_binarization_error": "mask_binarization_error",
        "class_channel_or_threshold_error": "class_channel_or_threshold_error",
        "resize_policy_artifact": "resize_policy_artifact",
        "split_or_panel_bias": "split_or_panel_bias",
        "training_signal_insufficient": "training_signal_insufficient",
        "mitochondria_base_defect": "mitochondria_base_defect",
        "negative_background_supervision_missing": "negative_background_supervision_missing",
        "true_model_underfit": "true_model_underfit",
    }
    for key, cause in mapping.items():
        if bool(evidence.get(key)):
            causes.append(cause)
    if not causes and bool(evidence.get("poor_performance")):
        causes.append("true_model_underfit")
    remediation = remediation_path_for_root_causes(causes)
    return {
        "root_causes": causes,
        "remediation_path": remediation,
        "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE if causes else PROMOTION_ELIGIBLE,
    }


def remediation_path_for_root_causes(root_causes: Sequence[str]) -> str:
    causes = set(root_causes)
    if causes & {
        "image_mask_pairing_error",
        "transform_alignment_error",
        "mask_binarization_error",
        "class_channel_or_threshold_error",
        "resize_policy_artifact",
        "split_or_panel_bias",
    }:
        return "fix_code_or_evaluation_then_regenerate_evidence"
    if "negative_background_supervision_missing" in causes:
        return "apply_p2_negative_glomeruli_crop_supervision_then_reevaluate"
    if "mitochondria_base_defect" in causes:
        return "rebuild_or_relabel_mitochondria_transfer_base_then_reevaluate_glomeruli"
    if "training_signal_insufficient" in causes:
        return "audit_training_data_and_sampler_then_train_new_candidates"
    if "true_model_underfit" in causes:
        return "train_new_candidate_after_audit_gates_pass"
    return "no_remediation_required"


def build_mitochondria_training_provenance(
    *,
    data_root: str | Path,
    train_items: Iterable[Any],
    valid_items: Iterable[Any],
    image_size: int,
    crop_size: int | None = None,
    output_size: int | None = None,
    positive_focus_p: float | None = None,
    min_pos_pixels: int | None = None,
    pos_crop_attempts: int | None = None,
    negative_crop_provenance: Mapping[str, Any] | None = None,
    augmentation_policy: Mapping[str, Any] | None = None,
    command: str | None = None,
) -> Dict[str, Any]:
    """Build mitochondria transfer-base training-scope metadata."""
    root = Path(data_root).expanduser()
    fitted_items = [_path_key(path) for path in list(train_items) + list(valid_items)]
    fitted_masks: list[str] = []
    try:
        from eq.data_management.standard_getters import get_y_full

        for item in fitted_items:
            try:
                fitted_masks.append(str(get_y_full(Path(item))))
            except Exception:
                continue
    except Exception:
        fitted_masks = []
    testing_root = root.parent / "testing" if root.name == "training" else root / "testing"
    training_root = root if root.name == "training" else root / "training"

    def count_images(image_root: Path) -> int:
        images_root = image_root / "images"
        if not images_root.is_dir():
            return 0
        return len([path for path in images_root.rglob("*") if path.is_file() and not path.name.startswith(".")])

    testing_prefix = _path_key(testing_root)
    testing_included = any(path.startswith(testing_prefix) for path in fitted_items if testing_prefix)
    scope = MITO_SCOPE_ALL_AVAILABLE if testing_included else MITO_SCOPE_HELDOUT
    inference_status = (
        MITO_INFERENCE_NOT_APPLICABLE if testing_included else MITO_INFERENCE_HELDOUT
    )
    return {
        "mitochondria_training_scope": scope,
        "mitochondria_inference_claim_status": inference_status,
        "mitochondria_physical_training_image_count": count_images(training_root),
        "mitochondria_physical_testing_image_count": count_images(testing_root),
        "mitochondria_testing_root_included_in_fitting": testing_included,
        "actual_pretraining_image_paths": fitted_items,
        "actual_pretraining_mask_paths": fitted_masks,
        "split_policy": "internal_dynamic_train_validation_split",
        "resize_policy": resize_policy_record(
            crop_size=int(crop_size or image_size),
            output_size=int(output_size or image_size),
        ),
        "positive_focus_p": positive_focus_p,
        "min_pos_pixels": min_pos_pixels,
        "pos_crop_attempts": pos_crop_attempts,
        "training_command": command,
    }


def validate_mitochondria_scope_for_claim(
    metadata: Mapping[str, Any],
    *,
    claim_type: str,
) -> Dict[str, Any]:
    scope = metadata.get("mitochondria_training_scope")
    if claim_type == "heldout_mitochondria_performance" and scope != MITO_SCOPE_HELDOUT:
        return {
            "ok": False,
            "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE,
            "reason": "mitochondria_testing_was_used_for_pretraining",
        }
    if scope == MITO_SCOPE_ALL_AVAILABLE:
        return {
            "ok": True,
            "promotion_evidence_status": PROMOTION_INSUFFICIENT,
            "reason": "all_available_pretraining_allowed_for_representation_only",
        }
    if scope == MITO_SCOPE_HELDOUT:
        return {
            "ok": True,
            "promotion_evidence_status": PROMOTION_INSUFFICIENT,
            "reason": "heldout_test_preserved",
        }
    return {
        "ok": False,
        "promotion_evidence_status": PROMOTION_AUDIT_MISSING,
        "reason": "missing_mitochondria_training_scope",
    }


def build_glomeruli_training_provenance(
    *,
    data_root: str | Path,
    train_items: Iterable[Any],
    valid_items: Iterable[Any],
    seed: int,
    split_seed: int,
    crop_size: int,
    output_size: int,
    candidate_family: str,
    training_mode: str,
    splitter_name: str = "RandomSplitter",
    transfer_base_artifact_path: str | Path | None = None,
    transfer_base_metadata: Mapping[str, Any] | None = None,
    positive_focus_p: float | None = None,
    min_pos_pixels: int | None = None,
    pos_crop_attempts: int | None = None,
    negative_crop_provenance: Mapping[str, Any] | None = None,
    augmentation_policy: Mapping[str, Any] | None = None,
    command: str | None = None,
) -> Dict[str, Any]:
    def _item_path(item: Any) -> Any:
        if isinstance(item, dict) and item.get("__eq_negative_crop_record__"):
            return item.get("source_image_path")
        return item

    train_paths = [_path_key(_item_path(path)) for path in train_items]
    valid_paths = [_path_key(_item_path(path)) for path in valid_items]
    negative_train_crop_ids = [
        str(item.get("negative_crop_id"))
        for item in train_items
        if isinstance(item, dict) and item.get("__eq_negative_crop_record__")
    ]
    all_paths = train_paths + valid_paths
    mask_paths: list[str] = []
    try:
        from eq.data_management.standard_getters import get_y_full

        for image_path in all_paths:
            try:
                mask_paths.append(str(get_y_full(Path(image_path))))
            except Exception:
                continue
    except Exception:
        mask_paths = []
    resize_policy = resize_policy_record(crop_size=crop_size, output_size=output_size)
    return {
        "data_root": str(Path(data_root).expanduser()),
        "training_mode": training_mode,
        "candidate_family": candidate_family,
        "seed": int(seed),
        "split_seed": int(split_seed),
        "splitter_name": splitter_name,
        "train_images": train_paths,
        "valid_images": valid_paths,
        "negative_train_crop_ids": negative_train_crop_ids,
        **manifest_context_summary(data_root),
        **source_size_summary(all_paths, mask_paths),
        "crop_size": int(crop_size),
        "image_size": int(output_size),
        "output_size": int(output_size),
        "crop_to_output_resize_ratio": resize_policy["crop_to_output_resize_ratio"],
        "resize_policy": resize_policy,
        "aspect_ratio_policy": resize_policy["aspect_ratio_policy"],
        "resize_method": resize_policy["resize_method"],
        "image_interpolation": resize_policy["image_interpolation"],
        "mask_interpolation": resize_policy["mask_interpolation"],
        "mask_binarization_after_resize": resize_policy["mask_binarization_after_resize"],
        "prediction_resize_back_method": resize_policy["prediction_resize_back_method"],
        "threshold_resize_order": resize_policy["threshold_resize_order"],
        "positive_focus_p": positive_focus_p,
        "min_pos_pixels": min_pos_pixels,
        "pos_crop_attempts": pos_crop_attempts,
        "augmentation_settings": {
            "fastai_aug_transforms": True,
            "spatial_crop_transform": "CropTransform",
        },
        "augmentation_policy": dict(augmentation_policy or {
            "variant": "fastai_default",
            "config_controls_active": False,
            "gaussian_noise_active": False,
        }),
        "learner_preprocessing": {
            "int_to_float_tensor": True,
            "normalize": "imagenet_stats",
            "mask_preprocess_transform": "binary_threshold_uint8_masks",
        },
        **dict(negative_crop_provenance or {
            "negative_crop_supervision_status": "absent",
            "negative_crop_manifest_path": None,
            "negative_crop_manifest_sha256": None,
            "negative_crop_count": 0,
            "mask_derived_background_crop_count": 0,
            "curated_negative_crop_count": 0,
            "negative_crop_source_image_count": 0,
            "negative_crop_review_protocol_version": "",
            "negative_crop_sampler_weight": 0.0,
        }),
        "transfer_base_artifact_path": str(transfer_base_artifact_path) if transfer_base_artifact_path else None,
        "transfer_base_metadata": dict(transfer_base_metadata or {}),
        "transfer_base_mitochondria_training_scope": (
            dict(transfer_base_metadata or {}).get("mitochondria_training_scope")
        ),
        "training_command": command,
    }


CLAIM_METRIC_RE = re.compile(r"\b(Dice|Jaccard|precision|recall)\b|(?:\d\.\d{2,3})", re.IGNORECASE)
UNSUPPORTED_CLAIM_RE = re.compile(r"\b(external validation|clinical readiness|promoted default|deployment-ready)\b", re.IGNORECASE)


def classify_claim_text(text: str) -> str:
    lowered = text.lower()
    if "causal" in lowered or "causes" in lowered:
        return "causal"
    if "external validation" in lowered or "generaliz" in lowered:
        return "external-validity related"
    if "predict" in lowered or "prognos" in lowered:
        return "predictive/prognostic"
    if "associate" in lowered or "correlat" in lowered:
        return "associational"
    return "descriptive"


def documentation_claim_audit(
    documents: Mapping[str, str],
    *,
    cited_report_status: str,
    cited_report_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Audit README/onboarding segmentation performance claims."""
    rows: list[Dict[str, Any]] = []
    eligible = cited_report_status == PROMOTION_ELIGIBLE
    for document_path, text in documents.items():
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not CLAIM_METRIC_RE.search(line) and not UNSUPPORTED_CLAIM_RE.search(line):
                continue
            claim_type = classify_claim_text(line)
            unsupported_language = bool(UNSUPPORTED_CLAIM_RE.search(line))
            has_metric_claim = bool(CLAIM_METRIC_RE.search(line))
            status = PROMOTION_ELIGIBLE
            reason = "ok"
            if unsupported_language:
                status = PROMOTION_NOT_ELIGIBLE
                reason = "unsupported_external_or_promotion_language"
            elif has_metric_claim and not eligible:
                status = PROMOTION_NOT_ELIGIBLE
                reason = "metric_claim_cites_non_promotion_eligible_report"
            rows.append(
                {
                    "document_path": document_path,
                    "line_number": line_number,
                    "claim_type": claim_type,
                    "cited_report_path": str(cited_report_path) if cited_report_path else None,
                    "cited_report_status": cited_report_status,
                    "promotion_evidence_status": status,
                    "reason": reason,
                    "claim_text": line.strip(),
                }
            )
    blocked = any(row["promotion_evidence_status"] != PROMOTION_ELIGIBLE for row in rows)
    return {
        "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE if blocked else PROMOTION_ELIGIBLE,
        "blocked": blocked,
        "rows": rows,
    }


def write_documentation_claim_audit(
    audit: Mapping[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# Documentation Claim Audit",
        "",
        f"- Promotion evidence status: `{audit.get('promotion_evidence_status')}`",
        f"- Blocked: `{audit.get('blocked')}`",
        "",
        "| Document | Line | Status | Reason | Claim type |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in audit.get("rows", []):
        lines.append(
            "| {document_path} | {line_number} | `{promotion_evidence_status}` | `{reason}` | {claim_type} |".format(
                **row
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ValidationAuditReport:
    """Small structured report payload for pytest/debug rendering."""

    implementation_audit: Mapping[str, Any]
    statistical_validity: Mapping[str, Any]
    scientific_interpretation: Mapping[str, Any]
    robustness_tests: Mapping[str, Any]
    documentation_consistency: Mapping[str, Any]

    def to_markdown(self) -> str:
        sections = [
            ("Implementation Audit", self.implementation_audit),
            ("Statistical Validity", self.statistical_validity),
            ("Scientific Interpretation", self.scientific_interpretation),
            ("Robustness Tests", self.robustness_tests),
            ("Documentation Consistency", self.documentation_consistency),
        ]
        lines = ["# Segmentation Validation Audit", ""]
        for title, payload in sections:
            lines.extend([f"## {title}", "", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
        return "\n".join(lines)
