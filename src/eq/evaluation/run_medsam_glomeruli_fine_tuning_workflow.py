"""Run MedSAM glomeruli fine-tuning workflow from YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from eq.evaluation.medsam_glomeruli_workflow import (
    _file_hash,
    _runtime_path,
    _runtime_root,
    _write_csv,
    ensure_evaluation_output_path,
    load_binary_mask,
    metric_row,
)
from eq.evaluation.medsam_torch_runtime import (
    medsam_subprocess_extra_env,
    resolve_medsam_torch_python,
    training_backend,
)
from eq.evaluation.run_medsam_automatic_glomeruli_prompts_workflow import (
    AUTOMATIC_MASK_SOURCE,
    AUTOMATIC_METRIC_FIELDS,
    PROPOSAL_BOX_FIELDS,
    _automatic_metric_row,
    _candidate_paths,
    _predict_probability,
    _select_best_setting,
    derive_proposal_boxes,
    proposal_recall_row,
)
from eq.evaluation.run_medsam_manual_glomeruli_comparison_workflow import (
    _run_medsam_batch,
    derive_oracle_boxes,
)
from eq.utils.paths import (
    ensure_not_under_runtime_raw_data,
    get_runtime_generated_masks_glomeruli_manifest_path,
    get_runtime_medsam_finetuned_release_path,
    get_runtime_medsam_glomeruli_checkpoint_path,
)

WORKFLOW_ID = "medsam_glomeruli_fine_tuning"
DEFAULT_RUN_ID = "pilot_medsam_glomeruli_fine_tuning"
DEFAULT_OUTPUT_DIR = "output/segmentation_evaluation/medsam_glomeruli_fine_tuning"
DEFAULT_MANIFEST_PATH = "raw_data/cohorts/manifest.csv"
MASK_SOURCE = "medsam_finetuned_glomeruli"


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    if payload.get("workflow") != WORKFLOW_ID:
        raise ValueError(f"Fine-tuning config must use `workflow: {WORKFLOW_ID}`.")
    return payload


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _resolve_output_paths(config: dict[str, Any], runtime_root: Path) -> dict[str, Path]:
    run_cfg = _mapping(config, "run")
    outputs = _mapping(config, "outputs")
    run_id = str(run_cfg.get("name") or DEFAULT_RUN_ID)
    mask_release_id = str(outputs.get("mask_release_id") or run_id)
    checkpoint_id = str(outputs.get("checkpoint_id") or run_id)
    evaluation_dir = ensure_evaluation_output_path(
        runtime_root,
        outputs.get("evaluation_dir", f"{DEFAULT_OUTPUT_DIR}/{run_id}"),
    )
    checkpoint_root = _runtime_path(
        runtime_root,
        outputs.get("checkpoint_root", str(get_runtime_medsam_glomeruli_checkpoint_path(checkpoint_id, runtime_root))),
    )
    generated_release_root = _runtime_path(
        runtime_root,
        outputs.get(
            "generated_mask_release_root",
            str(get_runtime_medsam_finetuned_release_path(mask_release_id, runtime_root)),
        ),
    )
    generated_registry_path = _runtime_path(
        runtime_root,
        outputs.get(
            "generated_mask_registry_path",
            str(get_runtime_generated_masks_glomeruli_manifest_path(runtime_root)),
        ),
    )
    ensure_not_under_runtime_raw_data(generated_release_root, runtime_root)
    ensure_not_under_runtime_raw_data(generated_registry_path, runtime_root)
    return {
        "evaluation_dir": evaluation_dir,
        "checkpoint_root": checkpoint_root,
        "generated_release_root": generated_release_root,
        "generated_registry_path": generated_registry_path,
        "run_id": Path(run_id),
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strongest_group_id(row: pd.Series) -> str:
    for key in ("source_sample_id", "subject_id", "animal_id", "manifest_row_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _admitted_manual_rows(
    manifest: pd.DataFrame,
    runtime_root: Path,
    *,
    lane_assignments: list[str],
) -> pd.DataFrame:
    frame = manifest.fillna("").copy()
    eligible = frame[
        (frame["admission_status"].astype(str) == "admitted")
        & frame["lane_assignment"].astype(str).isin(lane_assignments)
        & frame["image_path"].astype(str).str.strip().ne("")
        & frame["mask_path"].astype(str).str.strip().ne("")
    ].copy()
    eligible["image_path_resolved"] = eligible["image_path"].map(
        lambda p: str(_runtime_path(runtime_root, p))
    )
    eligible["mask_path_resolved"] = eligible["mask_path"].map(
        lambda p: str(_runtime_path(runtime_root, p))
    )
    eligible = eligible[
        eligible["image_path_resolved"].map(lambda p: Path(p).exists())
        & eligible["mask_path_resolved"].map(lambda p: Path(p).exists())
    ].copy()
    eligible["split_group_id"] = eligible.apply(_strongest_group_id, axis=1)
    eligible = eligible.sort_values(
        ["cohort_id", "split_group_id", "manifest_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    return eligible


def _build_deterministic_splits(
    rows: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    group_order = (
        rows[["split_group_id"]]
        .drop_duplicates()
        .sort_values(["split_group_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    total_groups = len(group_order)
    train_cut = max(1, int(total_groups * train_fraction))
    validation_cut = max(train_cut + 1, int(total_groups * (train_fraction + validation_fraction)))
    assignments: dict[str, str] = {}
    for index, group in enumerate(group_order["split_group_id"].tolist()):
        if index < train_cut:
            assignments[group] = "train"
        elif index < validation_cut:
            assignments[group] = "validation"
        else:
            assignments[group] = "test"
    output = rows.copy()
    output["split"] = output["split_group_id"].map(assignments).fillna("test")
    output["selection_rank"] = output.groupby("split").cumcount() + 1
    output["selection_reason"] = "deterministic_grouped_assignment"
    return output


def _required_split_fields() -> list[str]:
    return [
        "manifest_row_id",
        "cohort_id",
        "lane_assignment",
        "source_sample_id",
        "split_group_id",
        "image_path",
        "mask_path",
        "split",
        "selection_rank",
        "selection_reason",
    ]


def _write_split_manifests(
    frame: pd.DataFrame, evaluation_dir: Path
) -> dict[str, Path]:
    split_dir = evaluation_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        target = split_dir / f"{split}.csv"
        subset = frame[frame["split"] == split].copy()
        subset[_required_split_fields()].to_csv(target, index=False)
        paths[split] = target
    return paths


def _validate_split_manifest(path: Path, runtime_root: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Configured split manifest does not exist: {path}")
    frame = pd.read_csv(path).fillna("")
    missing = [field for field in _required_split_fields() if field not in frame.columns]
    if missing:
        raise ValueError(f"Split manifest is missing required fields {missing}: {path}")
    for record in frame.to_dict(orient="records"):
        image_path = _runtime_path(runtime_root, record["image_path"])
        mask_path = _runtime_path(runtime_root, record["mask_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Split image path missing: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Split mask path missing: {mask_path}")
    return _hash_file(path)


def _fixed_evaluation_rows(split_paths: dict[str, str], runtime_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("validation", "test"):
        path_text = split_paths.get(split, "")
        if not path_text:
            continue
        frame = pd.read_csv(path_text).fillna("")
        frame["split"] = split
        frame["image_path_resolved"] = frame["image_path"].map(
            lambda value: str(_runtime_path(runtime_root, value))
        )
        frame["mask_path_resolved"] = frame["mask_path"].map(
            lambda value: str(_runtime_path(runtime_root, value))
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=_required_split_fields())
    return pd.concat(frames, ignore_index=True)


def _split_rows(split_paths: dict[str, str], runtime_root: Path, split: str) -> pd.DataFrame:
    path_text = split_paths.get(split, "")
    if not path_text:
        return pd.DataFrame(columns=_required_split_fields())
    frame = pd.read_csv(path_text).fillna("")
    frame["split"] = split
    frame["image_path_resolved"] = frame["image_path"].map(
        lambda value: str(_runtime_path(runtime_root, value))
    )
    frame["mask_path_resolved"] = frame["mask_path"].map(
        lambda value: str(_runtime_path(runtime_root, value))
    )
    return frame


def _write_fixed_baseline_metrics(
    *,
    fixed_rows: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    metrics_dir = output_dir / "baseline_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in fixed_rows.to_dict(orient="records"):
        manual_mask = load_binary_mask(Path(str(record["mask_path_resolved"])))
        rows.extend(
            _baseline_metric_rows(
                manual_mask=manual_mask,
                manifest_row_id=str(record["manifest_row_id"]),
                cohort_id=str(record["cohort_id"]),
                lane_assignment=str(record["lane_assignment"]),
            )
        )
    metrics_path = metrics_dir / "trivial_baseline_metrics.csv"
    if rows:
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        grouped = (
            pd.DataFrame(rows)
            .groupby("method", dropna=False)[["dice", "jaccard", "precision", "recall"]]
            .mean()
            .reset_index()
        )
    else:
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        grouped = pd.DataFrame(
            columns=["method", "dice", "jaccard", "precision", "recall"]
        )
    grouped_path = metrics_dir / "trivial_baseline_metric_by_source.csv"
    grouped.to_csv(grouped_path, index=False)
    return {
        "metric_rows": len(rows),
        "trivial_baseline_metrics": str(metrics_path),
        "trivial_baseline_metric_by_source": str(grouped_path),
    }


def _baseline_execution_plan(
    *,
    fixed_rows: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    proposal = _mapping(config, "proposal")
    medsam = _mapping(config, "medsam")
    current_segmenter = _mapping(config, "current_segmenter")
    return {
        "fixed_evaluation_rows": int(len(fixed_rows)),
        "automatic_medsam": {
            "status": "configured",
            "mask_source": AUTOMATIC_MASK_SOURCE,
            "proposal_thresholds": proposal.get("thresholds", []),
            "proposal_source": "current_segmenter_candidates",
            "medsam_checkpoint": str(medsam.get("base_checkpoint", "")),
        },
        "oracle_prompt_medsam": {
            "status": "configured_if_enabled",
            "reference": "manual_mask_component_boxes",
            "medsam_checkpoint": str(medsam.get("base_checkpoint", "")),
        },
        "current_segmenter": {
            "status": "configured",
            "comparison_threshold": current_segmenter.get("comparison_threshold", ""),
            "candidate_paths": {
                "transfer": str(current_segmenter.get("transfer_model_path", "")),
                "scratch": str(current_segmenter.get("scratch_model_path", "")),
            },
        },
        "trivial_baselines": {
            "status": "computed_from_fixed_manual_masks",
            "methods": ["trivial_all_background", "trivial_all_foreground"],
        },
    }


def _load_rgb_resized(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _load_mask_resized(path: Path, image_size: int) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize((int(image_size), int(image_size)), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def _prepare_medsam_npy_data(
    *,
    train_rows: pd.DataFrame,
    output_root: Path,
    image_size: int,
) -> dict[str, Any]:
    imgs_dir = output_root / "imgs"
    gts_dir = output_root / "gts"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    gts_dir.mkdir(parents=True, exist_ok=True)
    image_files: list[str] = []
    mask_files: list[str] = []
    skipped: list[dict[str, str]] = []
    manifest_rows: list[dict[str, Any]] = []
    for record in train_rows.fillna("").to_dict(orient="records"):
        row_id = str(record.get("manifest_row_id", "")).strip()
        image_path = Path(str(record.get("image_path_resolved") or record["image_path"]))
        mask_path = Path(str(record.get("mask_path_resolved") or record["mask_path"]))
        stem = row_id or image_path.stem
        target_name = f"{stem}.npy"
        if not image_path.exists() or not mask_path.exists():
            skipped.append(
                {
                    "manifest_row_id": row_id,
                    "reason": "missing_image_or_mask",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )
            continue
        image = _load_rgb_resized(image_path, image_size)
        mask = _load_mask_resized(mask_path, image_size)
        if int(mask.sum()) <= 0:
            skipped.append(
                {
                    "manifest_row_id": row_id,
                    "reason": "empty_mask_after_resize",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )
            continue
        image_target = imgs_dir / target_name
        mask_target = gts_dir / target_name
        np.save(image_target, image)
        np.save(mask_target, mask)
        image_files.append(str(image_target))
        mask_files.append(str(mask_target))
        manifest_rows.append(
            {
                "manifest_row_id": row_id,
                "source_image_path": str(image_path),
                "source_mask_path": str(mask_path),
                "npy_image_path": str(image_target),
                "npy_mask_path": str(mask_target),
                "image_size": int(image_size),
                "mask_foreground_pixels": int(mask.sum()),
            }
        )
    manifest_path = output_root / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    skipped_path = output_root / "skipped.csv"
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    return {
        "output_root": str(output_root),
        "prepared_count": len(image_files),
        "skipped_count": len(skipped),
        "manifest": str(manifest_path),
        "skipped": str(skipped_path),
        "image_files": image_files,
        "mask_files": mask_files,
    }


def _baseline_metric_rows(
    *,
    manual_mask: np.ndarray,
    manifest_row_id: str,
    cohort_id: str,
    lane_assignment: str,
) -> list[dict[str, Any]]:
    manual = (np.asarray(manual_mask) > 0).astype(np.uint8)
    return [
        metric_row(
            method="trivial_all_background",
            candidate_artifact="trivial_baseline",
            manifest_row_id=manifest_row_id,
            cohort_id=cohort_id,
            lane_assignment=lane_assignment,
            manual_mask=manual,
            predicted_mask=np.zeros_like(manual, dtype=np.uint8),
        ),
        metric_row(
            method="trivial_all_foreground",
            candidate_artifact="trivial_baseline",
            manifest_row_id=manifest_row_id,
            cohort_id=cohort_id,
            lane_assignment=lane_assignment,
            manual_mask=manual,
            predicted_mask=np.ones_like(manual, dtype=np.uint8),
        ),
    ]


_FIXED_SPLIT_ROW_FIELDS = [
    "manifest_row_id",
    "cohort_id",
    "lane_assignment",
    "source_sample_id",
    "image_path",
    "mask_path",
    "image_path_resolved",
    "mask_path_resolved",
    "selection_rank",
    "selection_reason",
]


def _preflight_medsam_inference_paths(config: dict[str, Any]) -> dict[str, Path]:
    medsam = _mapping(config, "medsam")
    python_path = resolve_medsam_torch_python(config)
    repo_path = Path(str(medsam.get("repo", "")).strip()).expanduser()
    checkpoint = Path(str(medsam.get("base_checkpoint", "")).strip()).expanduser()
    script_path = repo_path / "MedSAM_Inference.py"
    bundle = {
        "medsam_python": python_path,
        "medsam_repo": repo_path,
        "checkpoint": checkpoint,
        "medsam_script": script_path,
    }
    for label, path in bundle.items():
        if not path.exists():
            raise FileNotFoundError(f"MedSAM inference preflight missing {label}: {path}")
    return bundle


def _baseline_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    frame = pd.DataFrame(rows)
    return {
        "rows": int(len(frame)),
        "mean_dice": float(frame["dice"].mean()),
        "mean_jaccard": float(frame["jaccard"].mean()),
        "mean_precision": float(frame["precision"].mean()),
        "mean_recall": float(frame["recall"].mean()),
    }


def _run_external_fixed_split_baselines(
    *,
    fixed_rows: pd.DataFrame,
    config: dict[str, Any],
    baseline_dir: Path,
    _runtime_root: Path,
) -> dict[str, Any]:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_eval_cfg = _mapping(config, "baseline_evaluation")
    proposal_cfg = _mapping(config, "proposal")
    tiling_cfg = _mapping(config, "tiling")
    current_segmenter_cfg = _mapping(config, "current_segmenter")
    medsam_cfg = _mapping(config, "medsam")

    run_auto = bool(baseline_eval_cfg.get("automatic_medsam", True))
    run_oracle = bool(baseline_eval_cfg.get("oracle_prompt_medsam", True))
    run_current = bool(baseline_eval_cfg.get("current_segmenter", True))

    medsam_inference = bool(run_auto or run_oracle)
    need_segmenters = bool(run_auto or run_current)

    summary: dict[str, Any] = {
        "status": "completed",
        "automatic_medsam": {"enabled": run_auto},
        "oracle_prompt_medsam": {"enabled": run_oracle},
        "current_segmenter": {"enabled": run_current},
    }

    if fixed_rows.empty:
        summary["status"] = "skipped_empty_fixed_split"
        return summary

    bundle: dict[str, Path] = {}
    if medsam_inference:
        bundle = _preflight_medsam_inference_paths(config)

    tile_size = int(tiling_cfg.get("tile_size", 512))
    stride = int(tiling_cfg.get("stride", 512))
    expected_size = int(tiling_cfg.get("expected_size", 256))
    thresholds = [float(value) for value in proposal_cfg.get("thresholds", [0.2, 0.35, 0.5])]
    min_area = int(proposal_cfg.get("min_component_area", 2000))
    max_area = int(proposal_cfg.get("max_component_area", 750000))
    padding = int(proposal_cfg.get("padding", 16))
    merge_iou = float(proposal_cfg.get("merge_iou", 0.25))
    max_boxes = int(proposal_cfg.get("max_boxes_per_image", 20))
    comparison_threshold = float(current_segmenter_cfg.get("comparison_threshold", 0.75))

    recall_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    proposal_by_setting: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    current_metrics: list[dict[str, Any]] = []
    manual_masks: dict[str, np.ndarray] = {}

    current_models: dict[str, Any] = {}
    candidate_paths = _candidate_paths(current_segmenter_cfg)
    if need_segmenters:
        from eq.data_management.model_loading import load_model_safely

        for _, model_path in candidate_paths.items():
            if not model_path.exists():
                raise FileNotFoundError(f"Missing current segmenter artifact: {model_path}")
        for family, model_path in candidate_paths.items():
            learner = load_model_safely(str(model_path), model_type="glomeruli")
            learner.model.eval()
            current_models[family] = learner.model

    for record in fixed_rows.fillna("").to_dict(orient="records"):
        row_id = str(record["manifest_row_id"])
        image_path = Path(str(record["image_path_resolved"]))
        manual_mask = load_binary_mask(Path(str(record["mask_path_resolved"])))
        manual_masks[row_id] = manual_mask

        base_record = {field: record.get(field, "") for field in _FIXED_SPLIT_ROW_FIELDS}

        if need_segmenters:
            for family, model in current_models.items():
                model_path = candidate_paths[family]
                probability, _audit = _predict_probability(
                    model=model,
                    image_path=image_path,
                    tile_size=tile_size,
                    stride=stride,
                    expected_size=expected_size,
                )
                if run_current:
                    current_mask = (
                        probability >= comparison_threshold
                    ).astype(np.uint8)
                    current_metrics.append(
                        metric_row(
                            method=f"current_segmenter_{family}",
                            candidate_artifact=str(model_path),
                            manifest_row_id=row_id,
                            cohort_id=str(record["cohort_id"]),
                            lane_assignment=str(record["lane_assignment"]),
                            manual_mask=manual_mask,
                            predicted_mask=current_mask,
                        )
                    )
                if run_auto:
                    for threshold in thresholds:
                        boxes, decisions = derive_proposal_boxes(
                            probability,
                            threshold=threshold,
                            min_component_area=min_area,
                            max_component_area=max_area,
                            padding=padding,
                            merge_iou=merge_iou,
                            max_boxes=max_boxes,
                        )
                        proposal_by_setting[(row_id, family, threshold)] = boxes
                        overflow_count = sum(
                            1 for decision in decisions if decision.get("decision") == "overflow"
                        )
                        for decision in decisions:
                            proposal_rows.append(
                                {
                                    **base_record,
                                    "candidate_family": family,
                                    "candidate_artifact": str(model_path),
                                    "threshold": threshold,
                                    **decision,
                                }
                            )
                        recall_row = proposal_recall_row(
                            manual_mask=manual_mask,
                            proposal_boxes=boxes,
                            manifest_row_id=row_id,
                            cohort_id=str(record["cohort_id"]),
                            lane_assignment=str(record["lane_assignment"]),
                            candidate_family=family,
                            candidate_artifact=str(model_path),
                            threshold=threshold,
                            min_component_area=min_area,
                        )
                        recall_row["overflow_count"] = int(overflow_count)
                        recall_rows.append(recall_row)

    if proposal_rows:
        pd.DataFrame(proposal_rows).reindex(columns=PROPOSAL_BOX_FIELDS).to_csv(
            baseline_dir / "proposal_boxes.csv",
            index=False,
        )
        summary["proposal_boxes_csv"] = str(baseline_dir / "proposal_boxes.csv")
    if recall_rows:
        pd.DataFrame(recall_rows).to_csv(
            baseline_dir / "proposal_recall.csv",
            index=False,
        )
        summary["proposal_recall_csv"] = str(baseline_dir / "proposal_recall.csv")

    if current_metrics:
        path = baseline_dir / "current_segmenter_metrics.csv"
        pd.DataFrame(current_metrics).to_csv(path, index=False)
        summary["current_segmenter_metrics_csv"] = str(path)
        summary["current_segmenter"].update(_baseline_metric_summary(current_metrics))

    device = str(medsam_cfg.get("device", "cpu"))
    checkpoint_str = str(bundle.get("checkpoint", ""))

    prompt_failures: list[dict[str, Any]] = []
    automatic_metrics: list[dict[str, Any]] = []

    if run_auto:
        if not recall_rows:
            summary["automatic_medsam"]["status"] = "skipped_no_proposals"
        else:
            best_family, best_threshold = _select_best_setting(recall_rows)
            summary["automatic_medsam"]["selected_candidate_family"] = best_family
            summary["automatic_medsam"]["selected_threshold"] = float(best_threshold)
            auto_mask_root = baseline_dir / "automatic_medsam_masks" / best_family
            auto_mask_root.mkdir(parents=True, exist_ok=True)
            medsam_auto_items: list[dict[str, Any]] = []
            for record in fixed_rows.fillna("").to_dict(orient="records"):
                row_id = str(record["manifest_row_id"])
                manual_mask = manual_masks[row_id]
                boxes = proposal_by_setting.get(
                    (row_id, best_family, best_threshold), []
                )
                medsam_auto_items.append(
                    {
                        "manifest_row_id": row_id,
                        "image_path": str(record["image_path_resolved"]),
                        "height": int(manual_mask.shape[0]),
                        "width": int(manual_mask.shape[1]),
                        "boxes": boxes,
                        "output_path": str(
                            auto_mask_root
                            / f"{row_id}_{best_family}_{best_threshold:g}_medsam_auto.png"
                        ),
                    }
                )
            batch_dir = baseline_dir / "medsam_batch_automatic"
            batch_dir.mkdir(parents=True, exist_ok=True)
            auto_failures = _run_medsam_batch(
                medsam_python=bundle["medsam_python"],
                medsam_repo=bundle["medsam_repo"],
                checkpoint=bundle["checkpoint"],
                device=device,
                items=medsam_auto_items,
                output_dir=batch_dir,
            )
            for failure in auto_failures:
                failure["baseline_prompt_lane"] = "automatic_medsam"
            prompt_failures.extend(auto_failures)

            items_by_id = {item["manifest_row_id"]: item for item in medsam_auto_items}
            for record in fixed_rows.fillna("").to_dict(orient="records"):
                row_id = str(record["manifest_row_id"])
                manual_mask = manual_masks[row_id]
                auto_path = Path(items_by_id[row_id]["output_path"])
                if auto_path.exists():
                    automatic_metrics.append(
                        _automatic_metric_row(
                            prompt_mode="automatic_current_segmenter_boxes",
                            proposal_threshold=float(best_threshold),
                            candidate_family=str(best_family),
                            mask_source=AUTOMATIC_MASK_SOURCE,
                            method="medsam_automatic",
                            candidate_artifact=checkpoint_str,
                            manifest_row_id=row_id,
                            cohort_id=str(record["cohort_id"]),
                            lane_assignment=str(record["lane_assignment"]),
                            manual_mask=manual_mask,
                            predicted_mask=load_binary_mask(auto_path),
                        )
                    )

            if automatic_metrics:
                path = baseline_dir / "automatic_medsam_metrics.csv"
                _write_csv(path, automatic_metrics, fieldnames=AUTOMATIC_METRIC_FIELDS)
                summary["automatic_medsam_metrics_csv"] = str(path)
                summary["automatic_medsam"].update(
                    _baseline_metric_summary(automatic_metrics)
                )

    oracle_metrics: list[dict[str, Any]] = []
    if run_oracle:
        oracle_rows_csv: list[dict[str, Any]] = []
        oracle_mask_dir = baseline_dir / "oracle_medsam_masks"
        oracle_mask_dir.mkdir(parents=True, exist_ok=True)
        medsam_oracle_items: list[dict[str, Any]] = []
        for record in fixed_rows.fillna("").to_dict(orient="records"):
            row_id = str(record["manifest_row_id"])
            manual_mask = manual_masks[row_id]
            base_record = {field: record.get(field, "") for field in _FIXED_SPLIT_ROW_FIELDS}
            boxes, skipped = derive_oracle_boxes(
                manual_mask,
                min_component_area=min_area,
                padding=padding,
            )
            for box in boxes:
                oracle_rows_csv.append({**base_record, **box, "skip_reason": ""})
            for skip in skipped:
                oracle_rows_csv.append({**base_record, **skip})
            medsam_oracle_items.append(
                {
                    "manifest_row_id": row_id,
                    "image_path": str(record["image_path_resolved"]),
                    "height": int(manual_mask.shape[0]),
                    "width": int(manual_mask.shape[1]),
                    "boxes": boxes,
                    "output_path": str(
                        oracle_mask_dir / f"{row_id}_medsam_oracle_union.png"
                    ),
                }
            )
        pd.DataFrame(oracle_rows_csv).to_csv(
            baseline_dir / "oracle_boxes.csv",
            index=False,
        )
        summary["oracle_boxes_csv"] = str(baseline_dir / "oracle_boxes.csv")

        oracle_batch_dir = baseline_dir / "medsam_batch_oracle"
        oracle_batch_dir.mkdir(parents=True, exist_ok=True)
        oracle_failures = _run_medsam_batch(
            medsam_python=bundle["medsam_python"],
            medsam_repo=bundle["medsam_repo"],
            checkpoint=bundle["checkpoint"],
            device=device,
            items=medsam_oracle_items,
            output_dir=oracle_batch_dir,
        )
        for failure in oracle_failures:
            failure["baseline_prompt_lane"] = "oracle_prompt_medsam"
        prompt_failures.extend(oracle_failures)

        for record in fixed_rows.fillna("").to_dict(orient="records"):
            row_id = str(record["manifest_row_id"])
            manual_mask = manual_masks[row_id]
            oracle_path = oracle_mask_dir / f"{row_id}_medsam_oracle_union.png"
            if oracle_path.exists():
                oracle_metrics.append(
                    metric_row(
                        method="medsam_oracle",
                        candidate_artifact=checkpoint_str,
                        manifest_row_id=row_id,
                        cohort_id=str(record["cohort_id"]),
                        lane_assignment=str(record["lane_assignment"]),
                        manual_mask=manual_mask,
                        predicted_mask=load_binary_mask(oracle_path),
                    )
                )

        if oracle_metrics:
            path = baseline_dir / "oracle_medsam_metrics.csv"
            pd.DataFrame(oracle_metrics).to_csv(path, index=False)
            summary["oracle_medsam_metrics_csv"] = str(path)
            summary["oracle_prompt_medsam"].update(_baseline_metric_summary(oracle_metrics))

    if prompt_failures:
        path = baseline_dir / "prompt_failures.csv"
        pd.DataFrame(prompt_failures).to_csv(path, index=False)
        summary["prompt_failures_csv"] = str(path)
        summary["prompt_failure_count"] = int(len(prompt_failures))

    return summary


def _mean_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return 0.0 if value in ("", None) else float(value)


def _classify_generated_mask_adoption(
    *,
    fine_tuned_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
    current_auto_metrics: dict[str, Any],
    current_segmenter_metrics: dict[str, Any],
    trivial_baseline_metrics: dict[str, Any],
    prompt_failure_count: int,
    gates: dict[str, Any],
    overlay_review_status: str,
) -> dict[str, Any]:
    fine_dice = _mean_metric(fine_tuned_metrics, "dice")
    fine_jaccard = _mean_metric(fine_tuned_metrics, "jaccard")
    oracle_dice = _mean_metric(oracle_metrics, "dice")
    oracle_gap = max(0.0, oracle_dice - fine_dice) if oracle_dice else 0.0
    improves_auto = fine_dice > _mean_metric(current_auto_metrics, "dice")
    improves_segmenter = fine_dice > _mean_metric(current_segmenter_metrics, "dice")
    beats_trivial = fine_dice > _mean_metric(trivial_baseline_metrics, "dice")
    reliability_passed = (
        int(prompt_failure_count) <= int(gates.get("max_prompt_failures", 0))
        and str(overlay_review_status) == "passed"
        and beats_trivial
    )
    oracle_level = (
        fine_dice >= float(gates.get("min_dice", 0.90))
        and fine_jaccard >= float(gates.get("min_jaccard", 0.82))
        and oracle_gap <= float(gates.get("max_oracle_dice_gap", 0.05))
    )
    if oracle_level and improves_auto and improves_segmenter and reliability_passed:
        adoption_tier = "oracle_level_preferred"
        recommended_source = MASK_SOURCE
        failure_mode = "none_detected"
    elif improves_auto and improves_segmenter and reliability_passed:
        adoption_tier = "improved_candidate_not_oracle"
        recommended_source = ""
        failure_mode = "oracle_gap"
    else:
        adoption_tier = "blocked"
        recommended_source = ""
        failure_mode = "downstream_integration"
        if not oracle_level:
            failure_mode = "oracle_gap"
        if int(prompt_failure_count) > int(gates.get("max_prompt_failures", 0)):
            failure_mode = "prompt_geometry"
        if not beats_trivial:
            failure_mode = "training_quality"
    return {
        "adoption_tier": adoption_tier,
        "primary_generated_mask_transition_status": adoption_tier,
        "recommended_generated_mask_source": recommended_source,
        "oracle_dice_gap": oracle_gap,
        "oracle_level_gates_passed": bool(oracle_level),
        "improves_current_auto": bool(improves_auto),
        "improves_current_segmenter": bool(improves_segmenter),
        "beats_trivial_baseline": bool(beats_trivial),
        "failure_mode": failure_mode,
    }


def _release_manifest_fields() -> list[str]:
    return [
        "generated_mask_id",
        "mask_release_id",
        "mask_source",
        "adoption_tier",
        "cohort_id",
        "lane_assignment",
        "source_sample_id",
        "source_image_path",
        "reference_mask_path",
        "generated_mask_path",
        "checkpoint_id",
        "checkpoint_path",
        "proposal_source",
        "proposal_threshold",
        "run_id",
        "generation_status",
        "provenance_path",
    ]


def _package_generated_mask_release(
    *,
    release_dir: Path,
    mask_release_id: str,
    run_id: str,
    checkpoint_id: str,
    checkpoint_path: Path,
    proposal_source: str,
    proposal_threshold: float,
    adoption_tier: str,
    generated_masks: list[dict[str, Any]],
) -> dict[str, Path]:
    ensure_not_under_runtime_raw_data(release_dir, release_dir.parents[4])
    masks_dir = release_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = release_dir / "provenance.json"
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(generated_masks, start=1):
        source_mask = Path(str(item["generated_mask_path"]))
        target_mask = masks_dir / source_mask.name
        if source_mask.exists():
            shutil.copy2(source_mask, target_mask)
            status = str(item.get("generation_status") or "generated")
        else:
            status = "missing_source"
        rows.append(
            {
                "generated_mask_id": f"{mask_release_id}:{index:05d}",
                "mask_release_id": mask_release_id,
                "mask_source": MASK_SOURCE,
                "adoption_tier": adoption_tier,
                "cohort_id": str(item.get("cohort_id", "")),
                "lane_assignment": str(item.get("lane_assignment", "")),
                "source_sample_id": str(item.get("source_sample_id", "")),
                "source_image_path": str(item.get("source_image_path", "")),
                "reference_mask_path": str(item.get("reference_mask_path", "")),
                "generated_mask_path": str(target_mask),
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_path),
                "proposal_source": proposal_source,
                "proposal_threshold": float(proposal_threshold),
                "run_id": run_id,
                "generation_status": status,
                "provenance_path": str(provenance_path),
            }
        )
    manifest_path = release_dir / "manifest.csv"
    pd.DataFrame(rows, columns=_release_manifest_fields()).to_csv(
        manifest_path, index=False
    )
    provenance = {
        "mask_release_id": mask_release_id,
        "mask_source": MASK_SOURCE,
        "adoption_tier": adoption_tier,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "proposal_source": proposal_source,
        "proposal_threshold": float(proposal_threshold),
        "manifest_path": str(manifest_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (release_dir / "INDEX.md").write_text(
        "\n".join(
            [
                f"# MedSAM Fine-Tuned Glomeruli Release: {mask_release_id}",
                "",
                f"- Mask source: `{MASK_SOURCE}`",
                f"- Adoption tier: `{adoption_tier}`",
                f"- Manifest: `{manifest_path}`",
                f"- Provenance: `{provenance_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "release_dir": release_dir,
        "manifest": manifest_path,
        "provenance": provenance_path,
        "index": release_dir / "INDEX.md",
    }


def _update_generated_mask_registry(registry_path: Path, release_manifest_path: Path) -> None:
    release_frame = pd.read_csv(release_manifest_path).fillna("")
    release_frame["release_manifest_path"] = str(release_manifest_path)
    columns = [
        "generated_mask_id",
        "mask_release_id",
        "mask_source",
        "adoption_tier",
        "cohort_id",
        "lane_assignment",
        "source_sample_id",
        "source_image_path",
        "reference_mask_path",
        "generated_mask_path",
        "release_manifest_path",
        "checkpoint_id",
        "checkpoint_path",
        "proposal_source",
        "proposal_threshold",
        "run_id",
        "generation_status",
        "provenance_path",
    ]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        existing = pd.read_csv(registry_path).fillna("")
        combined = pd.concat([existing, release_frame[columns]], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["generated_mask_id"], keep="last"
        )
    else:
        combined = release_frame[columns]
    combined.to_csv(registry_path, index=False)


def _build_checkpoint_provenance(
    *,
    training_command: list[str],
    environment: dict[str, str],
    medsam_repo: Path,
    base_checkpoint: Path,
    split_manifest_paths: dict[str, Path],
    split_manifest_hashes: dict[str, str],
    adaptation_mode: str,
    frozen_components: list[str],
    trainable_components: list[str],
    hyperparameters: dict[str, Any],
    checkpoint_files: list[Path],
    training_status: str,
    local_feasibility_status: str,
) -> dict[str, Any]:
    checkpoint_file_strings = [str(path) for path in checkpoint_files if path.exists()]
    return {
        "training_command": training_command,
        "environment": environment,
        "medsam_repo": str(medsam_repo),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_hash": _file_hash(base_checkpoint),
        "split_manifest_paths": {
            key: str(value) for key, value in split_manifest_paths.items()
        },
        "split_manifest_hashes": split_manifest_hashes,
        "adaptation_mode": adaptation_mode,
        "frozen_components": frozen_components,
        "trainable_components": trainable_components,
        "hyperparameters": hyperparameters,
        "checkpoint_files": checkpoint_file_strings,
        "training_status": training_status,
        "local_feasibility_status": local_feasibility_status,
        "supported_checkpoint": bool(
            training_status == "completed" and checkpoint_file_strings
        ),
    }


def _build_adaptation_command(
    *,
    medsam_python: Path,
    entrypoint: Path,
    train_npy_root: Path,
    work_dir: Path,
    base_checkpoint: Path,
    task_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> list[str]:
    return [
        str(medsam_python),
        str(entrypoint),
        "-i",
        str(train_npy_root),
        "-task_name",
        str(task_name),
        "-model_type",
        "vit_b",
        "-checkpoint",
        str(base_checkpoint),
        "--load_pretrain",
        "True",
        "-pretrain_model_path",
        str(base_checkpoint),
        "-work_dir",
        str(work_dir),
        "-num_epochs",
        str(int(epochs)),
        "-batch_size",
        str(int(batch_size)),
        "-lr",
        str(float(learning_rate)),
        "--device",
        str(device),
    ]


def _training_command_for_config(
    *,
    config: dict[str, Any],
    preflight: dict[str, str],
    train_npy_root: Path,
    work_dir: Path,
    training_cfg: dict[str, Any],
) -> list[str]:
    medsam = _mapping(config, "medsam")
    device = str(medsam.get("device", "cpu"))
    backend = preflight.get("training_backend", "upstream_medsam")
    python_path = Path(preflight["medsam_python"])
    repo = Path(preflight["medsam_repo"])
    ckpt = Path(preflight["base_checkpoint"])
    epochs = int(training_cfg.get("epochs", 10))
    batch_size = int(training_cfg.get("batch_size", 2))
    lr = float(training_cfg.get("learning_rate", 0.0001))
    if backend == "eq_native_adapter":
        return [
            str(python_path),
            "-m",
            "eq.evaluation.medsam_glomeruli_adapter",
            "--medsam-repo",
            str(repo),
            "--train-npy-root",
            str(train_npy_root),
            "--checkpoint",
            str(ckpt),
            "--work-dir",
            str(work_dir),
            "--model-type",
            str(training_cfg.get("model_type", "vit_b")),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(lr),
            "--weight-decay",
            str(float(training_cfg.get("weight_decay", 0.01))),
            "--bbox-shift",
            str(int(training_cfg.get("bbox_shift", 20))),
            "--device",
            device,
            "--max-examples",
            str(int(training_cfg.get("max_examples", 0))),
            "--seed",
            str(int(training_cfg.get("seed", 2023))),
        ]
    entrypoint = Path(preflight["entrypoint"])
    return _build_adaptation_command(
        medsam_python=python_path,
        entrypoint=entrypoint,
        train_npy_root=train_npy_root,
        work_dir=work_dir,
        base_checkpoint=ckpt,
        task_name=str(training_cfg.get("task_name", "eq_glomeruli")),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
    )


def _run_local_feasibility_smoke(
    *,
    command: list[str],
    env: dict[str, str],
    timeout_seconds: int,
    image_size: int,
    batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        memory_failure = "memory" in (completed.stderr or "").lower() or "oom" in (
            completed.stderr or ""
        ).lower()
        status = (
            "local_feasible"
            if completed.returncode == 0 and not memory_failure
            else "requires_external_accelerator"
        )
        return {
            "local_feasibility_status": status,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "elapsed_seconds": time.time() - started,
            "device": env.get("PYTORCH_DEVICE", ""),
            "image_size": int(image_size),
            "batch_size": int(batch_size),
            "memory_failure": bool(memory_failure),
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "local_feasibility_status": "requires_external_accelerator",
            "returncode": "",
            "stdout": "",
            "stderr": str(exc),
            "elapsed_seconds": time.time() - started,
            "device": env.get("PYTORCH_DEVICE", ""),
            "image_size": int(image_size),
            "batch_size": int(batch_size),
            "memory_failure": "memory" in str(exc).lower(),
        }


def _preflight_training_dependencies(config: dict[str, Any]) -> dict[str, str]:
    medsam = _mapping(config, "medsam")
    training = _mapping(config, "training")
    python_path = resolve_medsam_torch_python(config)
    repo_path = Path(str(medsam.get("repo", "")).strip()).expanduser()
    base_checkpoint = Path(str(medsam.get("base_checkpoint", "")).strip()).expanduser()
    backend = training_backend(config)
    entrypoint = str(training.get("entrypoint", "train_one_gpu.py")).strip()
    if not repo_path.exists():
        raise FileNotFoundError(f"Configured MedSAM repo does not exist: {repo_path}")
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"Configured MedSAM base checkpoint does not exist: {base_checkpoint}"
        )
    if backend == "upstream_medsam":
        entrypoint_path = (repo_path / entrypoint).resolve()
        if not entrypoint_path.exists():
            raise FileNotFoundError(
                f"Configured MedSAM entrypoint does not exist: {entrypoint_path}"
            )
        entrypoint_str = str(entrypoint_path)
    else:
        entrypoint_str = ""
    adaptation_mode = str(medsam.get("adaptation_mode", "")).strip()
    if not adaptation_mode:
        raise ValueError("medsam.adaptation_mode is required")
    frozen_components = medsam.get("frozen_components", [])
    trainable_components = medsam.get("trainable_components", [])
    if not isinstance(frozen_components, list) or not isinstance(trainable_components, list):
        raise ValueError("medsam.frozen_components and medsam.trainable_components must be lists")
    if not trainable_components:
        raise ValueError("medsam.trainable_components must include at least one component")
    return {
        "medsam_python": str(python_path),
        "medsam_repo": str(repo_path),
        "base_checkpoint": str(base_checkpoint),
        "entrypoint": entrypoint_str,
        "adaptation_mode": adaptation_mode,
        "training_backend": backend,
    }


def run_medsam_glomeruli_fine_tuning_workflow(
    config_path: Path, *, dry_run: bool = False
) -> dict[str, Path]:
    config = _load_config(config_path)
    runtime_root = _runtime_root(config)
    medsam = _mapping(config, "medsam")
    inputs = _mapping(config, "inputs")
    split_cfg = _mapping(inputs, "split_manifests")
    paths = _resolve_output_paths(config, runtime_root)
    outputs = _mapping(config, "outputs")
    run_cfg = _mapping(config, "run")
    lane_assignments = [
        str(v) for v in inputs.get("lane_assignments", ["manual_mask_core", "manual_mask_external"])
    ]
    manifest_path = _runtime_path(runtime_root, inputs.get("manifest_path", DEFAULT_MANIFEST_PATH))
    paths["evaluation_dir"].mkdir(parents=True, exist_ok=True)
    generated_splits = False
    split_hashes: dict[str, str] = {}
    split_paths: dict[str, str] = {}
    explicit_train = str(split_cfg.get("train", "")).strip()
    explicit_validation = str(split_cfg.get("validation", "")).strip()
    explicit_test = str(split_cfg.get("test", "")).strip()
    if explicit_train and explicit_validation and explicit_test:
        for key, raw in (
            ("train", explicit_train),
            ("validation", explicit_validation),
            ("test", explicit_test),
        ):
            resolved = _runtime_path(runtime_root, raw)
            split_hashes[key] = _validate_split_manifest(resolved, runtime_root)
            split_paths[key] = str(resolved)
    else:
        manifest = pd.read_csv(manifest_path)
        admitted = _admitted_manual_rows(
            manifest, runtime_root, lane_assignments=lane_assignments
        )
        split_frame = _build_deterministic_splits(
            admitted,
            train_fraction=float(inputs.get("train_fraction", 0.70)),
            validation_fraction=float(inputs.get("validation_fraction", 0.15)),
        )
        manifest_outputs = _write_split_manifests(split_frame, paths["evaluation_dir"])
        for key, manifest_file in manifest_outputs.items():
            split_hashes[key] = _hash_file(manifest_file)
            split_paths[key] = str(manifest_file)
        generated_splits = True
    preflight = _preflight_training_dependencies(config)
    training_cfg = _mapping(config, "training")
    fixed_rows = _fixed_evaluation_rows(split_paths, runtime_root)
    train_rows = _split_rows(split_paths, runtime_root, "train")
    train_npy_root = _runtime_path(
        runtime_root,
        training_cfg.get(
            "preprocessed_npy_root", "derived_data/medsam_glomeruli/npy_data"
        ),
    )
    training_data = _prepare_medsam_npy_data(
        train_rows=train_rows,
        output_root=train_npy_root,
        image_size=int(training_cfg.get("image_size", 1024)),
    )
    baseline_outputs = _write_fixed_baseline_metrics(
        fixed_rows=fixed_rows,
        output_dir=paths["evaluation_dir"],
    )
    baseline_plan = _baseline_execution_plan(fixed_rows=fixed_rows, config=config)
    baseline_eval_cfg = _mapping(config, "baseline_evaluation")
    external_allowed = bool(baseline_eval_cfg.get("external_baselines", True))
    external_baselines_summary: dict[str, Any]
    if dry_run:
        external_baselines_summary = {
            "status": "skipped_dry_run",
            "reason": "External MedSAM and current-segmenter baselines are skipped during dry-run.",
        }
    elif not external_allowed:
        external_baselines_summary = {
            "status": "skipped_config",
            "reason": "baseline_evaluation.external_baselines=false",
        }
    else:
        external_baselines_summary = _run_external_fixed_split_baselines(
            fixed_rows=fixed_rows,
            config=config,
            baseline_dir=paths["evaluation_dir"] / "baseline_metrics",
            _runtime_root=runtime_root,
        )
    training_command = _training_command_for_config(
        config=config,
        preflight=preflight,
        train_npy_root=train_npy_root,
        work_dir=paths["checkpoint_root"],
        training_cfg=training_cfg,
    )
    local_feasibility = {
        "local_feasibility_status": "requires_external_accelerator"
        if dry_run
        else "not_run",
        "dry_run": bool(dry_run),
        "reason": "Dry run records the feasibility contract without launching MedSAM training."
        if dry_run
        else "Local feasibility smoke command not requested.",
    }
    if (not dry_run) and bool(training_cfg.get("run_local_feasibility_smoke", False)):
        backend = preflight.get("training_backend", "upstream_medsam")
        smoke_env = {
            **medsam_subprocess_extra_env(device=str(medsam.get("device", "cpu"))),
            "PYTORCH_DEVICE": str(medsam.get("device", "cpu")),
        }
        if backend == "eq_native_adapter":
            smoke_training_cfg = dict(training_cfg)
            smoke_training_cfg["epochs"] = int(training_cfg.get("smoke_epochs", 1))
            smoke_training_cfg["batch_size"] = int(training_cfg.get("smoke_batch_size", 1))
            smoke_training_cfg["max_examples"] = int(training_cfg.get("smoke_examples", 2))
            smoke_command = _training_command_for_config(
                config=config,
                preflight=preflight,
                train_npy_root=train_npy_root,
                work_dir=paths["checkpoint_root"],
                training_cfg=smoke_training_cfg,
            )
        else:
            smoke_command = training_command + [
                "-num_epochs",
                str(int(training_cfg.get("smoke_epochs", 1))),
                "-batch_size",
                str(int(training_cfg.get("smoke_batch_size", 1))),
            ]
        local_feasibility = _run_local_feasibility_smoke(
            command=smoke_command,
            env=smoke_env,
            timeout_seconds=int(training_cfg.get("smoke_timeout_seconds", 300)),
            image_size=int(training_cfg.get("image_size", 1024)),
            batch_size=int(training_cfg.get("smoke_batch_size", 1)),
        )
    checkpoint_files = list(paths["checkpoint_root"].glob("*.pth")) if paths["checkpoint_root"].exists() else []
    training_status = "not_started_dry_run" if dry_run else "not_started"
    if checkpoint_files:
        training_status = "completed"
    checkpoint_provenance = _build_checkpoint_provenance(
        training_command=training_command,
        environment={
            **medsam_subprocess_extra_env(device=str(medsam.get("device", "cpu"))),
            "PYTORCH_DEVICE": str(medsam.get("device", "cpu")),
        },
        medsam_repo=Path(preflight["medsam_repo"]),
        base_checkpoint=Path(preflight["base_checkpoint"]),
        split_manifest_paths={key: Path(value) for key, value in split_paths.items()},
        split_manifest_hashes=split_hashes,
        adaptation_mode=preflight["adaptation_mode"],
        frozen_components=[str(value) for value in medsam.get("frozen_components", [])],
        trainable_components=[
            str(value) for value in medsam.get("trainable_components", [])
        ],
        hyperparameters=training_cfg,
        checkpoint_files=checkpoint_files,
        training_status=training_status,
        local_feasibility_status=str(
            local_feasibility.get("local_feasibility_status", "")
        ),
    )
    paths["checkpoint_root"].mkdir(parents=True, exist_ok=True)
    checkpoint_provenance_path = paths["checkpoint_root"] / "provenance.json"
    checkpoint_provenance_path.write_text(
        json.dumps(checkpoint_provenance, indent=2), encoding="utf-8"
    )
    summary = {
        "workflow": WORKFLOW_ID,
        "run_id": str(run_cfg.get("name") or DEFAULT_RUN_ID),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "runtime_root": str(runtime_root),
        "manifest_path": str(manifest_path),
        "dry_run": bool(dry_run),
        "medsam_python": preflight["medsam_python"],
        "medsam_repo": preflight["medsam_repo"],
        "base_checkpoint": preflight["base_checkpoint"],
        "base_checkpoint_hash": _file_hash(Path(str(medsam.get("base_checkpoint", "")).strip()).expanduser())
        if str(medsam.get("base_checkpoint", "")).strip()
        else "",
        "paths": {
            "evaluation_dir": str(paths["evaluation_dir"]),
            "checkpoint_root": str(paths["checkpoint_root"]),
            "generated_release_root": str(paths["generated_release_root"]),
            "generated_registry_path": str(paths["generated_registry_path"]),
        },
        "adoption_gates": _mapping(config, "adoption_gates"),
        "dependency_preflight": preflight,
        "baseline_evaluation": {
            **baseline_plan,
            "outputs": baseline_outputs,
            "external_baselines": external_baselines_summary,
            "dry_run_note": external_baselines_summary.get("reason", "")
            if dry_run
            else "",
        },
        "training_data": training_data,
        "local_feasibility": local_feasibility,
        "training_command": training_command,
        "checkpoint_provenance": str(checkpoint_provenance_path),
        "checkpoint_provenance_summary": {
            "training_status": checkpoint_provenance["training_status"],
            "supported_checkpoint": checkpoint_provenance["supported_checkpoint"],
            "checkpoint_files": checkpoint_provenance["checkpoint_files"],
        },
        "split_manifests": split_paths,
        "split_manifest_hashes": split_hashes,
        "generated_splits": generated_splits,
        "note": "Fine-tuned checkpoint inference/evaluation tasks run after training produces supported checkpoint artifacts.",
    }
    summary_path = paths["evaluation_dir"] / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not dry_run and bool(outputs.get("create_checkpoint_dir_on_start", False)):
        paths["checkpoint_root"].mkdir(parents=True, exist_ok=True)
    return {"evaluation_dir": paths["evaluation_dir"], "summary": summary_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MedSAM glomeruli fine-tuning workflow."
    )
    parser.add_argument("--config", default="configs/medsam_glomeruli_fine_tuning.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_medsam_glomeruli_fine_tuning_workflow(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
