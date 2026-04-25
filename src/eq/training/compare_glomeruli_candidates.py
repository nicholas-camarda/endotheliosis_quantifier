#!/usr/bin/env python3
"""Compare glomeruli transfer and scratch candidates under one deterministic promotion contract."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from eq.core.constants import DEFAULT_VAL_RATIO
from eq.data_management.datablock_loader import (
    get_items_full_images,
    validate_supported_segmentation_training_root,
)
from eq.data_management.model_loading import load_model_safely
from eq.data_management.standard_getters import get_y_full
from eq.evaluation.segmentation_metrics import pixel_accuracy
from eq.inference.prediction_core import create_prediction_core
from eq.training.promotion_gates import (
    audit_manifest_crops,
    binary_dice_jaccard,
    binary_precision_recall,
    deterministic_validation_manifest,
    evaluate_glomeruli_promotion_candidate,
    resolve_image_path_for_mask,
    trivial_baseline_metrics,
)
from eq.training.segmentation_validation_audit import (
    PROMOTION_AUDIT_MISSING,
    PROMOTION_ELIGIBLE,
    PROMOTION_INSUFFICIENT,
    PROMOTION_NOT_ELIGIBLE,
    RUNTIME_USE_AVAILABLE,
    aggregate_metric_by_category,
    audit_prediction_shapes,
    audit_preprocessing_parity,
    audit_resize_policy_parity,
    audit_split_overlap,
    classify_artifact_status,
    classify_root_causes,
    documentation_claim_audit,
    failure_reproduction_row,
    resize_policy_record,
    write_csv_rows,
    write_documentation_claim_audit,
)
from eq.utils.logger import get_logger
from eq.utils.paths import (
    get_runtime_models_path,
    get_runtime_segmentation_evaluation_path,
)
from eq.utils.run_io import metadata_path_for_model

TIE_MARGIN = 0.02
DEFAULT_EXAMPLES_PER_CATEGORY = 2
COMPARE_PREDICTION_THRESHOLD = 0.01

logger = get_logger("eq.glomeruli_candidate_comparison")


def _generated_run_id(seed: int) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_all_manual_mask_glomeruli_seed{seed}"


@dataclass
class CandidateRuntime:
    family: str
    role: str
    model_path: Path | None
    seed: int | None
    command: list[str] | None
    status: str
    error: str | None = None


def _read_metadata_if_available(model_path: Path | None) -> Dict[str, Any]:
    if model_path is None:
        return {}
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as handle:
        return json.load(handle)


def _split_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_splits.json")


def _validation_trace_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_validation_prediction_trace.csv")


def _read_validation_trace_rows(
    model_path: str | Path | None,
    *,
    root_causes: Sequence[str],
    remediation_path: str | None,
) -> list[Dict[str, Any]]:
    if not model_path:
        return []
    trace_path = _validation_trace_path_for_model(Path(model_path).expanduser())
    if not trace_path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with trace_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = dict(row)
            record["trace_source"] = str(trace_path)
            if not str(record.get("root_causes") or "").strip() and root_causes:
                record["root_causes"] = "|".join(root_causes)
            if not str(record.get("remediation_path") or "").strip() and remediation_path:
                record["remediation_path"] = remediation_path
            rows.append(record)
    return rows


def _merge_split_sidecar(
    provenance: Dict[str, Any],
    *,
    model_path: Path | None,
) -> Dict[str, Any]:
    if model_path is None:
        return provenance
    split_path = _split_path_for_model(model_path)
    if not split_path.exists():
        return provenance
    with split_path.open() as handle:
        split_metadata = json.load(handle)
    merged = dict(provenance)
    for key in (
        "train_images",
        "valid_images",
        "train_subjects",
        "valid_subjects",
        "split_seed",
        "splitter_name",
        "data_root",
        "training_mode",
        "crop_size",
        "output_size",
        "image_size",
        "counts",
    ):
        if key not in split_metadata:
            continue
        current = merged.get(key)
        incoming = split_metadata[key]
        if current not in (None, "", [], {}) and incoming not in (None, "", [], {}) and current != incoming:
            raise ValueError(
                f"Conflicting {key!r} between run metadata and split sidecar for {model_path}: "
                f"{current!r} != {incoming!r}"
            )
        merged[key] = incoming if current in (None, "", [], {}) else current
    merged["split_sidecar_path"] = str(split_path)
    return merged


def _build_fallback_provenance(runtime: CandidateRuntime) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "family": runtime.family,
        "comparison_role": runtime.role,
        "artifact_path": str(runtime.model_path) if runtime.model_path else None,
        "artifact_status": "comparison_only_missing_metadata" if runtime.model_path else "unavailable",
        "scientific_promotion_status": "not_evaluated" if runtime.model_path else "unavailable",
        "training_mode": "unknown",
        "seed": runtime.seed,
    }
    if runtime.command:
        provenance["comparison_command"] = runtime.command
    if runtime.error:
        provenance["error"] = runtime.error
    return provenance


def _merged_provenance(runtime: CandidateRuntime) -> Dict[str, Any]:
    metadata = _read_metadata_if_available(runtime.model_path)
    if not metadata:
        return _merge_split_sidecar(
            _build_fallback_provenance(runtime),
            model_path=runtime.model_path,
        )
    provenance = dict(metadata)
    provenance.setdefault("artifact_path", str(runtime.model_path) if runtime.model_path else None)
    provenance.setdefault("artifact_status", metadata.get("artifact_status", "supported_runtime"))
    provenance.setdefault(
        "scientific_promotion_status",
        metadata.get("scientific_promotion_status", "not_evaluated"),
    )
    invocation = provenance.get("invocation", {})
    if runtime.command and "comparison_command" not in provenance:
        provenance["comparison_command"] = runtime.command
    provenance["comparison_role"] = runtime.role
    provenance["family"] = runtime.family
    provenance["seed"] = invocation.get("seed", runtime.seed)
    return _merge_split_sidecar(provenance, model_path=runtime.model_path)


def _candidate_requires_transfer_base(runtime: CandidateRuntime, provenance: Dict[str, Any]) -> bool:
    family = str(runtime.family)
    candidate_family = str(provenance.get("candidate_family", ""))
    return family == "transfer" or candidate_family == "mitochondria_transfer"


def _promotion_status_from_reasons(reasons: Sequence[str]) -> str:
    if any("missing" in reason or "audit" in reason for reason in reasons):
        return PROMOTION_AUDIT_MISSING
    if reasons:
        return PROMOTION_NOT_ELIGIBLE
    return PROMOTION_INSUFFICIENT


def _heldout_mask_paths_for_runtimes(runtimes: Sequence[CandidateRuntime]) -> List[Path] | None:
    valid_sets: list[set[str]] = []
    train_subject_sets: list[set[str]] = []
    for runtime in runtimes:
        provenance = _merged_provenance(runtime)
        valid_images = provenance.get("valid_images") or []
        if not valid_images:
            return None
        valid_sets.append({str(Path(path).expanduser()) for path in valid_images})
        train_subject_sets.append(
            {_subject_id_from_image_path(Path(path).expanduser()) for path in provenance.get("train_images") or []}
        )
    if not valid_sets:
        return None
    shared_images = sorted(set.intersection(*valid_sets))
    if not shared_images:
        return []
    train_subjects = set.union(*train_subject_sets) if train_subject_sets else set()
    mask_paths: List[Path] = []
    for image_path in shared_images:
        if _subject_id_from_image_path(Path(image_path).expanduser()) in train_subjects:
            continue
        try:
            mask_paths.append(Path(get_y_full(Path(image_path))))
        except Exception:
            continue
    return mask_paths


def _build_deterministic_manifest(
    *,
    data_root: Path,
    runtimes: Sequence[CandidateRuntime],
    crop_size: int,
    examples_per_category: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    heldout_mask_paths = _heldout_mask_paths_for_runtimes(runtimes)
    manifest_source = "shared_candidate_validation_split"
    if heldout_mask_paths is None:
        heldout_mask_paths = _validation_mask_paths(data_root)
        manifest_source = "full_data_root_audit_missing_split_provenance"
    elif not heldout_mask_paths:
        return [], {
            "manifest_source": manifest_source,
            "promotion_evidence_status": PROMOTION_INSUFFICIENT,
            "decision_reason": "insufficient_heldout_category_support",
            "count": 0,
            "categories": {},
        }
    try:
        manifest = deterministic_validation_manifest(
            heldout_mask_paths,
            crop_size=crop_size,
            examples_per_category=examples_per_category,
        )
        audit = audit_manifest_crops(manifest)
        audit["manifest_source"] = manifest_source
        audit["promotion_evidence_status"] = (
            PROMOTION_ELIGIBLE
            if manifest_source == "shared_candidate_validation_split"
            else PROMOTION_AUDIT_MISSING
        )
        return manifest, audit
    except ValueError as exc:
        return [], {
            "manifest_source": manifest_source,
            "promotion_evidence_status": PROMOTION_INSUFFICIENT,
            "decision_reason": "insufficient_heldout_category_support",
            "error": str(exc),
            "count": 0,
            "categories": {},
        }


def _manifest_metadata_index(data_root: Path) -> Dict[str, Dict[str, str]]:
    manifest_path = data_root / "manifest.csv" if data_root.name == "cohorts" else data_root.parent / "manifest.csv"
    if not manifest_path.exists():
        return {}
    runtime_root = data_root.parents[1] if data_root.name == "cohorts" and data_root.parent.name == "raw_data" else data_root.parents[2]
    index: Dict[str, Dict[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata = {
                "cohort_id": str(row.get("cohort_id") or ""),
                "lane_assignment": str(row.get("lane_assignment") or ""),
            }
            for key in ("image_path", "mask_path"):
                value = str(row.get(key) or "")
                if not value:
                    continue
                path = Path(value).expanduser()
                absolute = path if path.is_absolute() else runtime_root / path
                index[str(absolute)] = metadata
                index[value] = metadata
    return index


def _annotate_manifest_with_context(manifest: Sequence[Dict[str, Any]], data_root: Path) -> List[Dict[str, Any]]:
    context = _manifest_metadata_index(data_root)
    if not context:
        return [dict(row) for row in manifest]
    annotated: List[Dict[str, Any]] = []
    for row in manifest:
        record = dict(row)
        metadata = context.get(str(row.get("image_path"))) or context.get(str(row.get("mask_path"))) or {}
        if metadata:
            record.setdefault("cohort_id", metadata.get("cohort_id"))
            record.setdefault("lane_assignment", metadata.get("lane_assignment"))
        annotated.append(record)
    return annotated


def _provenance_resize_policy(provenance: Dict[str, Any], *, fallback_crop_size: int, fallback_output_size: int) -> Dict[str, Any]:
    policy = provenance.get("resize_policy")
    if isinstance(policy, dict):
        return dict(policy)
    return resize_policy_record(
        crop_size=int(provenance.get("crop_size") or fallback_crop_size),
        output_size=int(provenance.get("output_size") or provenance.get("image_size") or fallback_output_size),
    )


def _candidate_audit_rows(
    *,
    summary: Dict[str, Any],
    manifest: Sequence[Dict[str, Any]],
    expected_size: int,
) -> Dict[str, Any]:
    provenance = summary.get("provenance", {})
    split_integrity = audit_split_overlap(
        train_images=provenance.get("train_images"),
        valid_images=provenance.get("valid_images"),
        promotion_manifest=manifest,
    )
    training_policy = _provenance_resize_policy(
        provenance,
        fallback_crop_size=expected_size,
        fallback_output_size=expected_size,
    )
    resize_rows = []
    resize_ratio = float(training_policy.get("crop_to_output_resize_ratio") or 1.0)
    resize_sensitivity_status = (
        "resize_benefit_unproven"
        if resize_ratio != 1.0
        else "no_downsample_policy"
    )
    resize_infeasibility_reason = (
        "no_no_downsample_or_less_downsample_candidate_artifact_available"
        if resize_ratio != 1.0
        else ""
    )
    for row in summary.get("prediction_rows", []):
        resize_rows.append(
            {
                "family": summary["family"],
                "resize_sensitivity_status": resize_sensitivity_status,
                "resize_sensitivity_comparison": "current_policy_vs_no_downsample_or_less_downsample",
                "resize_sensitivity_infeasibility_reason": resize_infeasibility_reason,
                "heldout_dice": row.get("dice"),
                "heldout_jaccard": row.get("jaccard"),
                "truth_foreground_fraction": row.get("truth_foreground_fraction"),
                "prediction_foreground_fraction": row.get("prediction_foreground_fraction"),
                **resize_policy_record(
                    crop_size=int(training_policy.get("crop_size") or expected_size),
                    output_size=int(training_policy.get("output_size") or expected_size),
                    category=row.get("category"),
                    cohort_id=row.get("cohort_id"),
                    lane_assignment=row.get("lane_assignment"),
                    split="deterministic_promotion",
                ),
            }
        )
    if not resize_rows:
        resize_rows.append(
            {
                "family": summary["family"],
                "resize_sensitivity_status": resize_sensitivity_status,
                "resize_sensitivity_comparison": "current_policy_vs_no_downsample_or_less_downsample",
                "resize_sensitivity_infeasibility_reason": (
                    resize_infeasibility_reason or "deterministic_promotion_manifest_unavailable"
                ),
                "heldout_dice": None,
                "heldout_jaccard": None,
                "truth_foreground_fraction": None,
                "prediction_foreground_fraction": None,
                **resize_policy_record(
                    crop_size=int(training_policy.get("crop_size") or expected_size),
                    output_size=int(training_policy.get("output_size") or expected_size),
                    split="deterministic_promotion_unavailable",
                ),
            }
        )
    eval_policy = resize_policy_record(
        crop_size=int(training_policy.get("crop_size") or expected_size),
        output_size=int(training_policy.get("output_size") or expected_size),
    )
    resize_parity = audit_resize_policy_parity(training_policy, eval_policy)
    preprocessing_parity = audit_preprocessing_parity(
        learner_consistent_preprocessing=True,
        supported_threshold_semantics=True,
        training_policy=training_policy,
        evaluation_policy=eval_policy,
    )
    prediction_shape = audit_prediction_shapes(summary.get("prediction_rows", []))
    resize_sensitivity = {
        "status": resize_sensitivity_status,
        "promotion_evidence_status": PROMOTION_INSUFFICIENT if resize_ratio != 1.0 else PROMOTION_ELIGIBLE,
        "required_comparison": (
            "current_policy_vs_no_downsample_or_less_downsample"
            if resize_ratio != 1.0
            else None
        ),
        "infeasibility_reason": resize_infeasibility_reason or None,
    }
    artifact_status = classify_artifact_status(
        provenance,
        loadable=bool(summary.get("available")),
        requires_transfer_base=_candidate_requires_transfer_base(
            CandidateRuntime(
                family=summary["family"],
                role=summary["comparison_role"],
                model_path=Path(summary["artifact_path"]) if summary.get("artifact_path") else None,
                seed=summary.get("seed"),
                command=summary.get("command"),
                status=summary.get("status", "unavailable"),
            ),
            provenance,
        ),
    )
    transfer_base_metadata = provenance.get("transfer_base_metadata") or {}
    transfer_base_report = {
        "artifact_path": provenance.get("transfer_base_artifact_path") or provenance.get("base_model_path"),
        "mitochondria_training_scope": provenance.get("transfer_base_mitochondria_training_scope")
        or transfer_base_metadata.get("mitochondria_training_scope"),
        "mitochondria_inference_claim_status": transfer_base_metadata.get("mitochondria_inference_claim_status"),
        "physical_training_image_count": transfer_base_metadata.get("mitochondria_physical_training_image_count"),
        "physical_testing_image_count": transfer_base_metadata.get("mitochondria_physical_testing_image_count"),
        "actual_fitted_image_count": len(transfer_base_metadata.get("actual_pretraining_image_paths") or []),
        "actual_fitted_mask_count": len(transfer_base_metadata.get("actual_pretraining_mask_paths") or []),
        "resize_policy": transfer_base_metadata.get("resize_policy"),
    }
    has_background_false_positive = any(
        "background_false_positive" in reason
        for status in prediction_shape["family_status"].values()
        for reason in status["reasons"]
    )
    has_negative_background_supervision = (
        str(provenance.get("negative_crop_supervision_status") or "").strip().lower() == "present"
    )
    root_cause = classify_root_causes(
        {
            "split_or_panel_bias": split_integrity["promotion_evidence_status"] != PROMOTION_ELIGIBLE,
            "resize_policy_artifact": resize_parity["promotion_evidence_status"] != PROMOTION_ELIGIBLE
            or resize_sensitivity["promotion_evidence_status"] != PROMOTION_ELIGIBLE,
            "class_channel_or_threshold_error": not preprocessing_parity["ok"],
            "negative_background_supervision_missing": has_background_false_positive
            and not has_negative_background_supervision,
            "training_signal_insufficient": bool(summary.get("gate", {}).get("blocked"))
            or has_background_false_positive,
            "mitochondria_base_defect": (
                _candidate_requires_transfer_base(
                    CandidateRuntime(
                        family=summary["family"],
                        role=summary["comparison_role"],
                        model_path=Path(summary["artifact_path"]) if summary.get("artifact_path") else None,
                        seed=summary.get("seed"),
                        command=summary.get("command"),
                        status=summary.get("status", "unavailable"),
                    ),
                    provenance,
                )
                and artifact_status["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING
            ),
        }
    )
    reproduction_rows = [
        failure_reproduction_row(
            candidate_family=summary["family"],
            panel_id=f"{summary['family']}-{row.get('manifest_index', index)}",
            artifact_path=summary.get("artifact_path"),
            image_path=row.get("image_path"),
            mask_path=row.get("mask_path"),
            crop_box=row.get("crop_box"),
            resize_policy=training_policy,
            threshold=COMPARE_PREDICTION_THRESHOLD,
            prediction_tensor_shape=row.get("prediction_tensor_shape"),
            overlay_path=row.get("review_panel_path"),
            root_causes=root_cause["root_causes"],
            remediation_path=root_cause["remediation_path"],
        )
        for index, row in enumerate(summary.get("prediction_rows", []))
    ]
    reproduction_rows.extend(
        _read_validation_trace_rows(
            summary.get("artifact_path"),
            root_causes=root_cause["root_causes"],
            remediation_path=root_cause["remediation_path"],
        )
    )
    reproduction_rows.extend(
        _read_validation_trace_rows(
            transfer_base_report.get("artifact_path"),
            root_causes=[],
            remediation_path="audit_trace_only_not_a_root_cause_classification",
        )
    )
    return {
        "split_integrity": split_integrity,
        "resize_policy": resize_rows,
        "resize_policy_parity": resize_parity,
        "resize_sensitivity": resize_sensitivity,
        "preprocessing_parity": preprocessing_parity,
        "prediction_shape": prediction_shape,
        "artifact_status": artifact_status,
        "transfer_base_report": transfer_base_report,
        "root_cause": root_cause,
        "failure_reproduction": reproduction_rows,
    }


def _discover_model_path(family_dir: Path) -> Path:
    candidates = sorted(family_dir.rglob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model artifact produced under {family_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_training_command(command: list[str], family: str, role: str, model_root: Path, seed: int) -> CandidateRuntime:
    try:
        logger.info("Running candidate training command: %s", " ".join(command))
        subprocess.run(command, check=True)
        family_dir = model_root / family
        return CandidateRuntime(
            family=family,
            role=role,
            model_path=_discover_model_path(family_dir),
            seed=seed,
            command=command,
            status="available",
        )
    except Exception as exc:
        return CandidateRuntime(
            family=family,
            role=role,
            model_path=None,
            seed=seed,
            command=command,
            status="unavailable",
            error=str(exc),
        )


def _candidate_command(
    *,
    data_dir: Path,
    model_dir: Path,
    model_name: str,
    epochs: int,
    learning_rate: float,
    image_size: int,
    crop_size: int,
    batch_size: int | None,
    loss_name: str | None,
    seed: int,
    from_scratch: bool,
    base_model: Path | None = None,
    split_manifest_path: Path | None = None,
    negative_crop_manifest_path: Path | None = None,
    negative_crop_sampler_weight: float = 0.0,
    augmentation_variant: str = "fastai_default",
    device: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eq.training.train_glomeruli",
        "--data-dir",
        str(data_dir),
        "--model-dir",
        str(model_dir),
        "--model-name",
        model_name,
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--image-size",
        str(image_size),
        "--crop-size",
        str(crop_size),
        "--seed",
        str(seed),
    ]
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if loss_name:
        command.extend(["--loss", loss_name])
    if split_manifest_path is not None:
        command.extend(["--split-manifest", str(split_manifest_path)])
    if negative_crop_manifest_path is not None:
        command.extend(
            [
                "--negative-crop-manifest",
                str(negative_crop_manifest_path),
                "--negative-crop-sampler-weight",
                str(negative_crop_sampler_weight),
            ]
        )
    command.extend(["--augmentation-variant", augmentation_variant])
    if device:
        command.extend(["--device", device])
    if from_scratch:
        command.append("--from-scratch")
    else:
        if base_model is None:
            raise ValueError("Transfer candidate requires an explicit base_model path.")
        command.extend(["--base-model", str(base_model)])
    return command


def _subject_id_from_image_path(image_path: Path) -> str:
    if image_path.parent != image_path and image_path.parent.name not in {"images", "masks"}:
        return image_path.parent.name
    stem = image_path.stem.removesuffix("_mask")
    match = re.match(r"(?P<subject>.+?)_image\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group("subject")
    return stem


def _write_shared_training_split(data_root: Path, output_root: Path, *, seed: int) -> Path:
    """Persist the one train/validation split used by fresh comparison candidates."""
    image_paths = [str(Path(path).expanduser()) for path in get_items_full_images(data_root)]
    if len(image_paths) < 2:
        raise ValueError("Candidate comparison requires at least two image/mask pairs for an explicit split.")
    subject_to_images: dict[str, list[str]] = defaultdict(list)
    for image_path in image_paths:
        subject_to_images[_subject_id_from_image_path(Path(image_path))].append(image_path)
    if len(subject_to_images) < 2:
        raise ValueError(
            "Candidate comparison requires at least two resolved subjects for a subject-held-out split."
        )
    rng = np.random.default_rng(int(seed))
    subjects = np.asarray(sorted(subject_to_images))
    rng.shuffle(subjects)
    target_valid_count = max(1, int(round(len(image_paths) * float(DEFAULT_VAL_RATIO))))
    valid_subjects: list[str] = []
    valid_image_count = 0
    for subject in subjects:
        if len(valid_subjects) >= len(subjects) - 1:
            break
        valid_subjects.append(str(subject))
        valid_image_count += len(subject_to_images[str(subject)])
        if valid_image_count >= target_valid_count:
            break
    valid_subject_set = set(valid_subjects)
    train_subjects = [subject for subject in sorted(subject_to_images) if subject not in valid_subject_set]
    train_images = [
        image
        for subject in train_subjects
        for image in sorted(subject_to_images[subject])
    ]
    valid_images = [
        image
        for subject in sorted(valid_subject_set)
        for image in sorted(subject_to_images[subject])
    ]
    payload = {
        "split_seed": int(seed),
        "splitter_name": "explicit_shared_subject_split",
        "valid_pct": float(DEFAULT_VAL_RATIO),
        "train_subjects": train_subjects,
        "valid_subjects": sorted(valid_subject_set),
        "train_images": train_images,
        "valid_images": valid_images,
        "counts": {"train": len(train_images), "valid": len(valid_images)},
    }
    split_path = output_root / "shared_candidate_training_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return split_path


def _crop_array(arr: np.ndarray, crop_box: Sequence[int]) -> np.ndarray:
    left, top, right, bottom = [int(value) for value in crop_box]
    return arr[top:bottom, left:right]


def _load_crop_pair(item: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    image_path = Path(item.get("image_path") or resolve_image_path_for_mask(item["mask_path"]))
    mask_path = Path(item["mask_path"])
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    truth_crop = (_crop_array(mask, item["crop_box"]) > 0).astype(np.uint8)
    image_crop = _crop_array(image, item["crop_box"])
    return image_crop, truth_crop


def _predict_crop_with_audit(
    learn: Any,
    image_crop: np.ndarray,
    truth_shape: tuple[int, int],
    expected_size: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    core = create_prediction_core(expected_size)
    device = next(learn.model.parameters()).device
    pil_image = Image.fromarray(image_crop).convert("RGB")
    resized = pil_image.resize((expected_size, expected_size), Image.Resampling.BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.no_grad():
        raw_output = learn.model(tensor.to(device))
    if raw_output.shape[1] == 2:
        pred_prob = torch.softmax(raw_output, dim=1)[:, 1]
    else:
        pred_prob = torch.sigmoid(raw_output)
    pred_np = pred_prob.squeeze().detach().cpu().numpy()
    pred_resized = core.resize_prediction_to_match(pred_np, truth_shape)
    return (pred_resized > COMPARE_PREDICTION_THRESHOLD).astype(np.uint8), {
        "input_tensor_shape": [int(value) for value in tensor.shape],
        "raw_output_shape": [int(value) for value in raw_output.shape],
        "prediction_probability_shape": [int(value) for value in pred_np.shape],
        "prediction_resized_shape": [int(value) for value in pred_resized.shape],
        "threshold": float(COMPARE_PREDICTION_THRESHOLD),
        "inference_preprocessing": "deterministic_resize_imagenet_normalize",
        "threshold_resize_order": "resize_probability_then_threshold",
    }


def _predict_crop(learn: Any, image_crop: np.ndarray, truth_shape: tuple[int, int], expected_size: int) -> np.ndarray:
    pred, _audit = _predict_crop_with_audit(
        learn,
        image_crop,
        truth_shape,
        expected_size,
    )
    return pred


def _overlay_mask(image_crop: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = image_crop.astype(np.float32).copy()
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    active = mask.astype(bool)
    base[active] = (0.6 * base[active]) + (0.4 * overlay[active])
    return np.clip(base, 0, 255).astype(np.uint8)


def _save_review_panel(
    *,
    asset_path: Path,
    image_crop: np.ndarray,
    truth_crop: np.ndarray,
    pred_crop: np.ndarray,
) -> None:
    truth_overlay = _overlay_mask(image_crop, truth_crop, (0, 255, 0))
    pred_overlay = _overlay_mask(image_crop, pred_crop, (255, 0, 0))
    panel = np.concatenate([image_crop, truth_overlay, pred_overlay], axis=1)
    Image.fromarray(panel).save(asset_path)


def _safe_slug(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "".join(chars).strip("_")
    return slug or "item"


def _aggregate_metrics(truth_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]) -> Dict[str, float]:
    truth = np.concatenate([mask.astype(bool).reshape(-1) for mask in truth_masks])
    pred = np.concatenate([mask.astype(bool).reshape(-1) for mask in pred_masks])
    metrics = binary_dice_jaccard(truth, pred)
    metrics.update(binary_precision_recall(truth, pred))
    metrics["pixel_accuracy"] = pixel_accuracy(pred.astype(np.uint8), truth.astype(np.uint8))
    return metrics


def _evaluate_runtime(
    runtime: CandidateRuntime,
    manifest: Sequence[Dict[str, Any]],
    asset_dir: Path,
    expected_size: int,
) -> Dict[str, Any]:
    provenance = _merged_provenance(runtime)
    summary: Dict[str, Any] = {
        "family": runtime.family,
        "comparison_role": runtime.role,
        "available": runtime.status == "available" and runtime.model_path is not None,
        "status": runtime.status,
        "artifact_path": str(runtime.model_path) if runtime.model_path else None,
        "seed": provenance.get("seed", runtime.seed),
        "provenance": provenance,
        "command": runtime.command,
        "error": runtime.error,
    }
    if not summary["available"]:
        summary["gate"] = {"blocked": True, "reasons": ["candidate_family_unavailable"]}
        summary["metrics"] = {}
        summary["prediction_rows"] = []
        return summary

    prediction_rows: List[Dict[str, Any]] = []
    if not manifest:
        summary["gate"] = {
            "blocked": True,
            "reasons": ["insufficient_heldout_category_support"],
        }
        summary["metrics"] = {}
        summary["baselines"] = {}
        summary["prediction_rows"] = prediction_rows
        summary["runtime_use_status"] = RUNTIME_USE_AVAILABLE
        summary["promotion_evidence_status"] = PROMOTION_INSUFFICIENT
        return summary

    learn = load_model_safely(str(runtime.model_path), model_type="glomeruli")
    learn.model.eval()
    truth_masks: List[np.ndarray] = []
    pred_masks: List[np.ndarray] = []
    for index, item in enumerate(manifest):
        image_crop, truth_crop = _load_crop_pair(item)
        pred_crop, predict_audit = _predict_crop_with_audit(
            learn,
            image_crop,
            truth_crop.shape,
            expected_size=expected_size,
        )
        truth_masks.append(truth_crop)
        pred_masks.append(pred_crop)
        metrics = binary_dice_jaccard(truth_crop, pred_crop)
        metrics.update(binary_precision_recall(truth_crop, pred_crop))
        image_name = str(item.get("image_name") or Path(str(item.get("image_path", ""))).name)
        subject_id = str(item.get("subject_id") or Path(str(item.get("image_path", ""))).parent.name)
        category = str(item.get("category"))
        asset_path = asset_dir / (
            f"{runtime.family}_{index:02d}_{_safe_slug(category)}_{_safe_slug(subject_id)}_{_safe_slug(Path(image_name).stem)}.png"
        )
        _save_review_panel(
            asset_path=asset_path,
            image_crop=image_crop,
            truth_crop=truth_crop,
            pred_crop=pred_crop,
        )
        prediction_rows.append(
            {
                "family": runtime.family,
                "comparison_role": runtime.role,
                "seed": provenance.get("seed", runtime.seed),
                "image_path": item.get("image_path"),
                "image_name": image_name,
                "subject_id": subject_id,
                "mask_path": item.get("mask_path"),
                "category": category,
                "cohort_id": item.get("cohort_id"),
                "lane_assignment": item.get("lane_assignment"),
                "manifest_index": index,
                "crop_box": json.dumps(item.get("crop_box")),
                "truth_foreground_fraction": float(truth_crop.mean()),
                "prediction_foreground_fraction": float(pred_crop.mean()),
                "prediction_tensor_shape": predict_audit["raw_output_shape"],
                "prediction_probability_shape": predict_audit["prediction_probability_shape"],
                "prediction_resized_shape": predict_audit["prediction_resized_shape"],
                "threshold_resize_order": predict_audit["threshold_resize_order"],
                "dice": metrics["dice"],
                "jaccard": metrics["jaccard"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "review_panel_path": str(asset_path),
            }
        )

    gate = evaluate_glomeruli_promotion_candidate(
        truth_masks,
        pred_masks,
        manifest,
        provenance,
    )
    baselines = trivial_baseline_metrics(truth_masks)
    summary["gate"] = gate
    summary["metrics"] = _aggregate_metrics(truth_masks, pred_masks)
    summary["baselines"] = baselines
    summary["prediction_rows"] = prediction_rows
    return summary


def _within_tie_margin(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], *, margin: float = TIE_MARGIN) -> bool:
    return (
        abs(float(metrics_a.get("dice", 0.0)) - float(metrics_b.get("dice", 0.0))) <= margin
        and abs(float(metrics_a.get("jaccard", 0.0)) - float(metrics_b.get("jaccard", 0.0))) <= margin
    )


def _strictly_better_on_both(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], *, margin: float = TIE_MARGIN) -> bool:
    return (
        float(metrics_a.get("dice", -1.0)) > float(metrics_b.get("dice", -1.0)) + margin
        and float(metrics_a.get("jaccard", -1.0)) > float(metrics_b.get("jaccard", -1.0)) + margin
    )


def determine_promotion_decision(
    candidate_summaries: Sequence[Dict[str, Any]],
    compatibility_summary: Dict[str, Any] | None = None,
    *,
    tie_margin: float = TIE_MARGIN,
) -> Dict[str, Any]:
    transfer = next((row for row in candidate_summaries if row["family"] == "transfer"), None)
    scratch = next((row for row in candidate_summaries if row["family"] == "scratch"), None)
    unavailable = [
        row["family"]
        for row in candidate_summaries
        if row["family"] in {"transfer", "scratch"} and not row["available"]
    ]
    promotable = [
        row
        for row in candidate_summaries
        if row["family"] in {"transfer", "scratch"}
        and row["available"]
        and not row["gate"]["blocked"]
        and row.get("promotion_evidence_status") in {None, PROMOTION_ELIGIBLE}
    ]

    result: Dict[str, Any] = {
        "decision_state": None,
        "decision_reason": None,
        "promoted_family": None,
        "research_use_candidates": [],
        "tie_margin": {"dice": tie_margin, "jaccard": tie_margin},
    }

    if unavailable:
        result["decision_state"] = "insufficient_evidence"
        result["decision_reason"] = f"candidate_family_unavailable:{','.join(sorted(unavailable))}"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in promotable
        ]
        return result

    if not promotable:
        if any(
            row.get("promotion_evidence_status") == PROMOTION_INSUFFICIENT
            or "insufficient_heldout_category_support" in row.get("gate", {}).get("reasons", [])
            for row in candidate_summaries
        ):
            result["decision_state"] = "insufficient_evidence"
            result["decision_reason"] = "insufficient_heldout_category_support"
        else:
            result["decision_state"] = "not_promotion_eligible"
            result["decision_reason"] = "no_candidate_cleared_promotion_gates"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in candidate_summaries
            if row["family"] in {"transfer", "scratch"} and row.get("runtime_use_status") == RUNTIME_USE_AVAILABLE
        ]
        return result

    if compatibility_summary and compatibility_summary.get("available") and compatibility_summary.get("metrics"):
        filtered: List[Dict[str, Any]] = []
        for row in promotable:
            if _within_tie_margin(row["metrics"], compatibility_summary["metrics"], margin=tie_margin):
                row.setdefault("decision_notes", []).append(
                    "candidate_does_not_materially_exceed_compatibility_reference"
                )
            else:
                filtered.append(row)
        if not filtered:
            result["decision_state"] = "not_promotion_eligible"
            result["decision_reason"] = "no_candidate_materially_exceeded_compatibility_reference"
            return result
        promotable = filtered

    if len(promotable) == 1:
        winner = promotable[0]
        result["decision_state"] = "promoted"
        result["decision_reason"] = "exactly_one_candidate_cleared_promotion_gates"
        result["promoted_family"] = winner["family"]
        return result

    first, second = promotable[0], promotable[1]
    if _within_tie_margin(first["metrics"], second["metrics"], margin=tie_margin):
        result["decision_state"] = "insufficient_evidence"
        result["decision_reason"] = "transfer_and_scratch_are_within_practical_tie_margin"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in promotable
        ]
        return result

    if _strictly_better_on_both(first["metrics"], second["metrics"], margin=tie_margin):
        winner = first
    elif _strictly_better_on_both(second["metrics"], first["metrics"], margin=tie_margin):
        winner = second
    else:
        result["decision_state"] = "insufficient_evidence"
        result["decision_reason"] = "mixed_metric_signal_between_transfer_and_scratch"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in promotable
        ]
        return result

    result["decision_state"] = "promoted"
    result["decision_reason"] = "one_candidate_outperformed_the_other_on_shared_manifest_metrics"
    result["promoted_family"] = winner["family"]
    return result


def _decision_promotion_status(decision: Dict[str, Any], candidate_summaries: Sequence[Dict[str, Any]]) -> str:
    statuses = [row.get("promotion_evidence_status") for row in candidate_summaries]
    if PROMOTION_AUDIT_MISSING in statuses:
        return PROMOTION_AUDIT_MISSING
    if decision.get("decision_state") == "promoted":
        return PROMOTION_ELIGIBLE
    if decision.get("decision_state") == "insufficient_evidence":
        return PROMOTION_INSUFFICIENT
    return PROMOTION_NOT_ELIGIBLE


def _write_manifest(manifest: Sequence[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(list(manifest), handle, indent=2, sort_keys=True)


def _write_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_report(
    *,
    output_path: Path,
    decision: Dict[str, Any],
    manifest_audit: Dict[str, Any],
    candidate_summaries: Sequence[Dict[str, Any]],
    compatibility_summary: Dict[str, Any] | None,
) -> None:
    manifest_categories = manifest_audit.get("categories", {})
    lines = [
        "# Glomeruli Candidate Comparison Report",
        "",
        f"- Decision: `{decision['decision_state']}`",
        f"- Reason: `{decision['decision_reason']}`",
        f"- Promoted family: `{decision.get('promoted_family')}`",
        "",
        "## Manifest",
        "",
        f"- Total crops: `{manifest_audit.get('count')}`",
        f"- Unique images: `{manifest_audit.get('unique_images')}`",
        f"- Unique subjects: `{manifest_audit.get('unique_subjects')}`",
        (
            f"- Category counts: `background={manifest_categories.get('background', 0)}` "
            f"`boundary={manifest_categories.get('boundary', 0)}` "
            f"`positive={manifest_categories.get('positive', 0)}`"
        ),
        "- Review panels show: `raw crop | truth overlay (green) | prediction overlay (red)`",
        "",
        "## Candidates",
        "",
    ]
    for summary in candidate_summaries:
        metrics = summary.get("metrics", {})
        split_integrity = summary.get("split_integrity", {})
        root_cause = summary.get("root_cause", {})
        resize_sensitivity = summary.get("resize_sensitivity", {})
        provenance = summary.get("provenance", {})
        transfer_base_report = summary.get("transfer_base_report", {})
        lines.extend(
            [
                f"### {summary['family']}",
                f"- Available: `{summary['available']}`",
                f"- Runtime use status: `{summary.get('runtime_use_status')}`",
                f"- Promotion evidence status: `{summary.get('promotion_evidence_status')}`",
                f"- Role: `{summary['comparison_role']}`",
                f"- Artifact: `{summary['artifact_path']}`",
                f"- Seed: `{summary.get('seed')}`",
                f"- Gate blocked: `{summary['gate']['blocked']}`",
                f"- Gate reasons: `{', '.join(summary['gate'].get('reasons', []))}`",
                f"- Split integrity: `{split_integrity.get('reason')}` "
                f"`train_image_overlap={split_integrity.get('train_image_overlap_count')}` "
                f"`subject_overlap={split_integrity.get('subject_overlap_count')}`",
                f"- Transfer base artifact: `{transfer_base_report.get('artifact_path') or provenance.get('transfer_base_artifact_path') or provenance.get('base_model_path')}`",
                f"- Transfer base mitochondria scope: `{transfer_base_report.get('mitochondria_training_scope')}`",
                f"- Transfer base inference claim status: `{transfer_base_report.get('mitochondria_inference_claim_status')}`",
                f"- Transfer base physical counts: `training={transfer_base_report.get('physical_training_image_count')}` `testing={transfer_base_report.get('physical_testing_image_count')}`",
                f"- Transfer base fitted counts: `images={transfer_base_report.get('actual_fitted_image_count')}` `masks={transfer_base_report.get('actual_fitted_mask_count')}`",
                f"- Transfer base resize policy: `{json.dumps(transfer_base_report.get('resize_policy'), sort_keys=True) if transfer_base_report.get('resize_policy') is not None else None}`",
                f"- Resize sensitivity status: `{resize_sensitivity.get('status')}`",
                f"- Negative crop supervision status: `{provenance.get('negative_crop_supervision_status', 'absent')}`",
                f"- Negative crop manifest: `{provenance.get('negative_crop_manifest_path')}`",
                f"- Negative crop counts: `total={provenance.get('negative_crop_count', 0)}` "
                f"`mask_derived_background={provenance.get('mask_derived_background_crop_count', 0)}` "
                f"`curated={provenance.get('curated_negative_crop_count', 0)}`",
                f"- Negative crop sampler weight: `{provenance.get('negative_crop_sampler_weight', 0.0)}`",
                f"- Augmentation policy: `{json.dumps(provenance.get('augmentation_policy', {}), sort_keys=True)}`",
                f"- Root causes: `{', '.join(root_cause.get('root_causes', []))}`",
                f"- Remediation path: `{root_cause.get('remediation_path')}`",
                (
                    f"- Metrics: `dice={metrics.get('dice')}` `jaccard={metrics.get('jaccard')}` "
                    f"`precision={metrics.get('precision')}` `recall={metrics.get('recall')}`"
                ),
                "",
            ]
        )
    if compatibility_summary:
        lines.extend(
            [
                "## Compatibility Reference",
                "",
                f"- Available: `{compatibility_summary['available']}`",
                f"- Artifact: `{compatibility_summary['artifact_path']}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_report(
    *,
    output_path: Path,
    decision: Dict[str, Any],
    manifest_audit: Dict[str, Any],
    candidate_summaries: Sequence[Dict[str, Any]],
) -> None:
    manifest_categories = manifest_audit.get("categories", {})
    candidate_sections: List[str] = []
    for summary in candidate_summaries:
        metrics = summary.get("metrics", {})
        prediction_rows = summary.get("prediction_rows", [])
        gallery = []
        for row in prediction_rows:
            image_label = html.escape(str(row.get("image_name") or Path(str(row.get("image_path", ""))).name))
            subject_label = html.escape(str(row.get("subject_id", "")))
            category_label = html.escape(str(row.get("category", "")))
            crop_box = html.escape(str(row.get("crop_box", "")))
            manifest_index = int(row.get("manifest_index", 0))
            gallery.append(
                "<figure class='panel-card'>"
                f"<img src=\"{html.escape(str(Path('review_assets') / Path(row['review_panel_path']).name))}\" alt=\"{html.escape(summary['family'])} panel {manifest_index}\" />"
                "<figcaption>"
                f"<strong>Panel {manifest_index:02d}</strong> | {category_label} | {subject_label} | {image_label}<br>"
                f"crop={crop_box}<br>"
                f"dice={float(row['dice']):.3f} | jaccard={float(row['jaccard']):.3f} | "
                f"precision={float(row['precision']):.3f} | recall={float(row['recall']):.3f}<br>"
                f"truth_fg={float(row['truth_foreground_fraction']):.3f} | pred_fg={float(row['prediction_foreground_fraction']):.3f}<br>"
                "panel order: raw | truth overlay | prediction overlay"
                "</figcaption>"
                "</figure>"
            )
        gate_reasons = summary["gate"].get("reasons", [])
        split_integrity = summary.get("split_integrity", {})
        root_cause = summary.get("root_cause", {})
        resize_sensitivity = summary.get("resize_sensitivity", {})
        provenance = summary.get("provenance", {})
        transfer_base_report = summary.get("transfer_base_report", {})
        reason_items = (
            "".join(f"<li>{html.escape(reason)}</li>" for reason in gate_reasons)
            if gate_reasons
            else "<li>none</li>"
        )
        candidate_sections.append(
            "<section class='candidate-section'>"
            f"<h2>{html.escape(summary['family'])}</h2>"
            "<div class='summary-grid'>"
            f"<div><strong>Available</strong><br>{html.escape(str(summary['available']))}</div>"
            f"<div><strong>Runtime use</strong><br>{html.escape(str(summary.get('runtime_use_status')))}</div>"
            f"<div><strong>Promotion evidence</strong><br>{html.escape(str(summary.get('promotion_evidence_status')))}</div>"
            f"<div><strong>Seed</strong><br>{html.escape(str(summary.get('seed')))}</div>"
            f"<div><strong>Blocked</strong><br>{html.escape(str(summary['gate']['blocked']))}</div>"
            f"<div><strong>Role</strong><br>{html.escape(str(summary['comparison_role']))}</div>"
            f"<div><strong>Dice</strong><br>{float(metrics.get('dice', 0.0)):.3f}</div>"
            f"<div><strong>Jaccard</strong><br>{float(metrics.get('jaccard', 0.0)):.3f}</div>"
            f"<div><strong>Precision</strong><br>{float(metrics.get('precision', 0.0)):.3f}</div>"
            f"<div><strong>Recall</strong><br>{float(metrics.get('recall', 0.0)):.3f}</div>"
            "</div>"
            f"<p class='artifact'><strong>Artifact:</strong> {html.escape(str(summary.get('artifact_path')))}</p>"
            f"<p class='artifact'><strong>Split integrity:</strong> {html.escape(str(split_integrity.get('reason')))} "
            f"train image overlap={html.escape(str(split_integrity.get('train_image_overlap_count')))} "
            f"subject overlap={html.escape(str(split_integrity.get('subject_overlap_count')))}</p>"
            f"<p class='artifact'><strong>Transfer base:</strong> {html.escape(str(transfer_base_report.get('artifact_path') or provenance.get('transfer_base_artifact_path') or provenance.get('base_model_path')))} "
            f"scope={html.escape(str(transfer_base_report.get('mitochondria_training_scope')))} "
            f"inference_claim={html.escape(str(transfer_base_report.get('mitochondria_inference_claim_status')))} "
            f"physical_train={html.escape(str(transfer_base_report.get('physical_training_image_count')))} "
            f"physical_test={html.escape(str(transfer_base_report.get('physical_testing_image_count')))} "
            f"fitted_images={html.escape(str(transfer_base_report.get('actual_fitted_image_count')))} "
            f"fitted_masks={html.escape(str(transfer_base_report.get('actual_fitted_mask_count')))} "
            f"resize={html.escape(str(transfer_base_report.get('resize_policy')))}</p>"
            f"<p class='artifact'><strong>Resize sensitivity:</strong> {html.escape(str(resize_sensitivity.get('status')))}</p>"
            f"<p class='artifact'><strong>Root causes:</strong> {html.escape(', '.join(root_cause.get('root_causes', [])))} "
            f"<strong>Remediation:</strong> {html.escape(str(root_cause.get('remediation_path')))}</p>"
            "<div class='gate-box'>"
            "<strong>Gate reasons</strong>"
            f"<ul>{reason_items}</ul>"
            "</div>"
            "<div class='gallery'>"
            + "".join(gallery)
            + "</div>"
            "</section>"
        )
    html_text = (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.45;color:#1f2937;background:#f8fafc;}"
        "h1,h2,h3{margin:0 0 12px 0;color:#0f172a;}"
        ".hero,.candidate-section,.manifest-box{background:#fff;border:1px solid #dbe3ec;border-radius:12px;padding:18px;margin-bottom:18px;box-shadow:0 1px 2px rgba(15,23,42,.04);}"
        ".summary-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0 14px 0;}"
        ".summary-grid div{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px;}"
        ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:16px;}"
        ".panel-card{margin:0;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;}"
        ".panel-card img{width:100%;height:auto;border-radius:8px;border:1px solid #e2e8f0;background:#fff;}"
        ".panel-card figcaption{font-size:13px;margin-top:10px;color:#334155;}"
        ".artifact{font-size:13px;word-break:break-word;}"
        ".gate-box{background:#fff7ed;border:1px solid #fdba74;border-radius:10px;padding:12px;margin:12px 0 16px 0;}"
        ".manifest-meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:12px;}"
        ".manifest-meta div{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px;}"
        ".legend{color:#475569;font-size:14px;margin-top:8px;}"
        "@media (max-width: 1100px){.gallery{grid-template-columns:1fr;}.summary-grid,.manifest-meta{grid-template-columns:repeat(2,minmax(0,1fr));}}"
        "</style></head><body>"
        "<section class='hero'>"
        "<h1>Glomeruli Candidate Comparison Report</h1>"
        f"<p><strong>Decision:</strong> {html.escape(str(decision['decision_state']))} "
        f"<strong>Reason:</strong> {html.escape(str(decision['decision_reason']))} "
        f"<strong>Promoted family:</strong> {html.escape(str(decision.get('promoted_family')))}</p>"
        "</section>"
        "<section class='manifest-box'>"
        "<h2>Manifest Coverage</h2>"
        "<p class='legend'>This report is scored on a deterministic crop manifest. Each review panel shows <strong>raw crop | truth overlay | prediction overlay</strong>.</p>"
        "<div class='manifest-meta'>"
        f"<div><strong>Total crops</strong><br>{int(manifest_audit.get('count', 0))}</div>"
        f"<div><strong>Unique images</strong><br>{int(manifest_audit.get('unique_images', 0))}</div>"
        f"<div><strong>Unique subjects</strong><br>{int(manifest_audit.get('unique_subjects', 0))}</div>"
        f"<div><strong>Category counts</strong><br>background={int(manifest_categories.get('background', 0))} | boundary={int(manifest_categories.get('boundary', 0))} | positive={int(manifest_categories.get('positive', 0))}</div>"
        "</div>"
        "</section>"
        + "".join(candidate_sections)
        + "</body></html>"
    )
    output_path.write_text(html_text, encoding="utf-8")


def _validation_mask_paths(data_root: Path) -> List[Path]:
    image_paths = get_items_full_images(data_root)
    mask_paths: List[Path] = []
    for image_path in image_paths:
        mask_paths.append(get_y_full(image_path))
    if not mask_paths:
        raise ValueError(
            f"No validation masks found for candidate comparison under {data_root}. "
            "Use a supported paired full-image root or the manifest-backed "
            "raw_data/cohorts registry root with admitted masked rows."
        )
    return mask_paths


def compare_glomeruli_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = validate_supported_segmentation_training_root(args.data_dir, stage="glomeruli")
    run_id = str(args.run_id or _generated_run_id(args.seed))
    output_root = Path(args.output_dir).expanduser() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_dir).expanduser()
    model_root.mkdir(parents=True, exist_ok=True)
    asset_dir = output_root / "review_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    shared_split_path = None
    if not (args.transfer_model_path and args.scratch_model_path):
        shared_split_path = _write_shared_training_split(data_root, output_root, seed=args.seed)

    if args.transfer_model_path:
        transfer_runtime = CandidateRuntime(
            family="transfer",
            role="candidate",
            model_path=Path(args.transfer_model_path).expanduser(),
            seed=None,
            command=None,
            status="available" if Path(args.transfer_model_path).expanduser().exists() else "unavailable",
            error=None if Path(args.transfer_model_path).expanduser().exists() else "transfer_model_path_not_found",
        )
    else:
        transfer_runtime = _run_training_command(
            _candidate_command(
                data_dir=data_root,
                model_dir=model_root,
                model_name=args.transfer_model_name,
                epochs=args.transfer_epochs,
                learning_rate=args.learning_rate,
                image_size=args.image_size,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                loss_name=args.loss,
                seed=args.seed,
                from_scratch=False,
                base_model=Path(args.transfer_base_model).expanduser() if args.transfer_base_model else None,
                split_manifest_path=shared_split_path,
                negative_crop_manifest_path=Path(args.negative_crop_manifest).expanduser() if args.negative_crop_manifest else None,
                negative_crop_sampler_weight=args.negative_crop_sampler_weight,
                augmentation_variant=args.augmentation_variant,
                device=args.device,
            ),
            family="transfer",
            role="candidate",
            model_root=model_root,
            seed=args.seed,
        )

    if args.scratch_model_path:
        scratch_runtime = CandidateRuntime(
            family="scratch",
            role="candidate",
            model_path=Path(args.scratch_model_path).expanduser(),
            seed=None,
            command=None,
            status="available" if Path(args.scratch_model_path).expanduser().exists() else "unavailable",
            error=None if Path(args.scratch_model_path).expanduser().exists() else "scratch_model_path_not_found",
        )
    else:
        scratch_runtime = _run_training_command(
            _candidate_command(
                data_dir=data_root,
                model_dir=model_root,
                model_name=args.scratch_model_name,
                epochs=args.scratch_epochs,
                learning_rate=args.learning_rate,
                image_size=args.image_size,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                loss_name=args.loss,
                seed=args.seed,
                from_scratch=True,
                split_manifest_path=shared_split_path,
                negative_crop_manifest_path=Path(args.negative_crop_manifest).expanduser() if args.negative_crop_manifest else None,
                negative_crop_sampler_weight=args.negative_crop_sampler_weight,
                augmentation_variant=args.augmentation_variant,
                device=args.device,
            ),
            family="scratch",
            role="candidate",
            model_root=model_root,
            seed=args.seed,
        )

    candidate_runtimes = [transfer_runtime, scratch_runtime]
    manifest, manifest_audit = _build_deterministic_manifest(
        data_root=data_root,
        runtimes=candidate_runtimes,
        crop_size=args.crop_size,
        examples_per_category=args.examples_per_category,
    )
    manifest = _annotate_manifest_with_context(manifest, data_root)
    manifest_path = output_root / "deterministic_validation_manifest.json"
    _write_manifest(manifest, manifest_path)

    candidate_summaries = [
        _evaluate_runtime(transfer_runtime, manifest, asset_dir, expected_size=args.image_size),
        _evaluate_runtime(scratch_runtime, manifest, asset_dir, expected_size=args.image_size),
    ]
    candidate_audits: Dict[str, Dict[str, Any]] = {}
    for summary in candidate_summaries:
        audit = _candidate_audit_rows(
            summary=summary,
            manifest=manifest,
            expected_size=args.image_size,
        )
        candidate_audits[summary["family"]] = audit
        summary["runtime_use_status"] = audit["artifact_status"]["runtime_use_status"]
        audit_statuses = [
            audit["artifact_status"]["promotion_evidence_status"],
            audit["split_integrity"]["promotion_evidence_status"],
            audit["prediction_shape"]["family_status"].get(summary["family"], {}).get(
                "promotion_evidence_status",
                PROMOTION_ELIGIBLE,
            ),
            audit["preprocessing_parity"]["promotion_evidence_status"],
            audit["resize_sensitivity"]["promotion_evidence_status"],
            manifest_audit.get("promotion_evidence_status", PROMOTION_ELIGIBLE),
        ]
        if PROMOTION_AUDIT_MISSING in audit_statuses:
            summary["promotion_evidence_status"] = PROMOTION_AUDIT_MISSING
        elif PROMOTION_NOT_ELIGIBLE in audit_statuses or summary.get("gate", {}).get("blocked"):
            summary["promotion_evidence_status"] = PROMOTION_NOT_ELIGIBLE
        elif PROMOTION_INSUFFICIENT in audit_statuses:
            summary["promotion_evidence_status"] = PROMOTION_INSUFFICIENT
        else:
            summary["promotion_evidence_status"] = PROMOTION_ELIGIBLE
        summary["split_integrity"] = audit["split_integrity"]
        summary["prediction_shape_audit"] = audit["prediction_shape"]
        summary["resize_policy_parity"] = audit["resize_policy_parity"]
        summary["resize_sensitivity"] = audit["resize_sensitivity"]
        summary["preprocessing_parity"] = audit["preprocessing_parity"]
        summary["transfer_base_report"] = audit["transfer_base_report"]
        summary["root_cause"] = audit["root_cause"]
        extra_gate_reasons = []
        if audit["split_integrity"]["promotion_evidence_status"] != PROMOTION_ELIGIBLE:
            extra_gate_reasons.append(audit["split_integrity"]["reason"])
        for status in audit["prediction_shape"]["family_status"].values():
            extra_gate_reasons.extend(status.get("reasons", []))
        if audit["artifact_status"]["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING:
            extra_gate_reasons.extend(audit["artifact_status"]["reasons"])
        if audit["resize_sensitivity"]["promotion_evidence_status"] != PROMOTION_ELIGIBLE:
            extra_gate_reasons.append(audit["resize_sensitivity"]["status"])
        if extra_gate_reasons:
            summary.setdefault("gate", {}).setdefault("reasons", [])
            summary["gate"]["reasons"] = sorted(set(summary["gate"]["reasons"] + extra_gate_reasons))
            summary["gate"]["blocked"] = True

    compatibility_summary = None
    if args.compat_model_path:
        compatibility_runtime = CandidateRuntime(
            family="compatibility_reference",
            role="compatibility_reference",
            model_path=Path(args.compat_model_path).expanduser(),
            seed=None,
            command=None,
            status="available" if Path(args.compat_model_path).expanduser().exists() else "unavailable",
            error=None if Path(args.compat_model_path).expanduser().exists() else "compat_model_path_not_found",
        )
        compatibility_summary = _evaluate_runtime(
            compatibility_runtime,
            manifest,
            asset_dir,
            expected_size=args.image_size,
        )

    decision = determine_promotion_decision(
        candidate_summaries,
        compatibility_summary=compatibility_summary,
        tie_margin=TIE_MARGIN,
    )
    decision["promotion_evidence_status"] = _decision_promotion_status(decision, candidate_summaries)

    candidate_rows = []
    for summary in candidate_summaries:
        candidate_rows.append(
            {
                "family": summary["family"],
                "comparison_role": summary["comparison_role"],
                "available": summary["available"],
                "artifact_path": summary["artifact_path"],
                "seed": summary.get("seed"),
                "runtime_use_status": summary.get("runtime_use_status"),
                "promotion_evidence_status": summary.get("promotion_evidence_status"),
                "blocked": summary["gate"]["blocked"],
                "reasons": "|".join(summary["gate"].get("reasons", [])),
                "root_causes": "|".join(summary.get("root_cause", {}).get("root_causes", [])),
                "remediation_path": summary.get("root_cause", {}).get("remediation_path"),
                "negative_crop_supervision_status": summary.get("provenance", {}).get("negative_crop_supervision_status"),
                "negative_crop_manifest_path": summary.get("provenance", {}).get("negative_crop_manifest_path"),
                "negative_crop_manifest_sha256": summary.get("provenance", {}).get("negative_crop_manifest_sha256"),
                "negative_crop_count": summary.get("provenance", {}).get("negative_crop_count"),
                "mask_derived_background_crop_count": summary.get("provenance", {}).get("mask_derived_background_crop_count"),
                "curated_negative_crop_count": summary.get("provenance", {}).get("curated_negative_crop_count"),
                "negative_crop_sampler_weight": summary.get("provenance", {}).get("negative_crop_sampler_weight"),
                "augmentation_policy": json.dumps(summary.get("provenance", {}).get("augmentation_policy", {}), sort_keys=True),
                "dice": summary.get("metrics", {}).get("dice"),
                "jaccard": summary.get("metrics", {}).get("jaccard"),
                "precision": summary.get("metrics", {}).get("precision"),
                "recall": summary.get("metrics", {}).get("recall"),
            }
        )

    prediction_rows = [
        row
        for summary in candidate_summaries
        for row in summary.get("prediction_rows", [])
    ]
    metric_by_category_rows = aggregate_metric_by_category(prediction_rows)
    prediction_shape_rows = [
        row
        for audit in candidate_audits.values()
        for row in audit["prediction_shape"].get("rows", [])
    ]
    resize_policy_rows = [
        row
        for audit in candidate_audits.values()
        for row in audit.get("resize_policy", [])
    ]
    failure_reproduction_rows = [
        row
        for audit in candidate_audits.values()
        for row in audit.get("failure_reproduction", [])
    ]
    decision_payload = {
        "decision": decision,
        "run_id": run_id,
        "output_dir": str(output_root),
        "model_dir": str(model_root),
        "manifest_path": str(manifest_path),
        "candidate_manifest": list(manifest),
        "manifest_audit": manifest_audit,
        "candidate_summaries": candidate_summaries,
        "candidate_audits": candidate_audits,
        "metric_by_category_path": str(output_root / "metric_by_category.csv"),
        "prediction_shape_audit_path": str(output_root / "prediction_shape_audit.csv"),
        "resize_policy_audit_path": str(output_root / "resize_policy_audit.csv"),
        "failure_reproduction_audit_path": str(output_root / "failure_reproduction_audit.csv"),
        "documentation_claim_audit_path": str(output_root / "documentation_claim_audit.md"),
        "compatibility_summary": compatibility_summary,
    }
    report_json = output_root / "promotion_report.json"
    report_md = output_root / "promotion_report.md"
    report_html = output_root / "promotion_report.html"
    _write_csv(candidate_rows, output_root / "candidate_summary.csv")
    _write_csv(prediction_rows, output_root / "candidate_predictions.csv")
    write_csv_rows(metric_by_category_rows, output_root / "metric_by_category.csv")
    write_csv_rows(prediction_shape_rows, output_root / "prediction_shape_audit.csv")
    write_csv_rows(resize_policy_rows, output_root / "resize_policy_audit.csv")
    write_csv_rows(failure_reproduction_rows, output_root / "failure_reproduction_audit.csv")
    docs_to_audit: Dict[str, str] = {}
    for doc_path in (Path("README.md"), Path("docs/ONBOARDING_GUIDE.md")):
        if doc_path.exists():
            docs_to_audit[str(doc_path)] = doc_path.read_text(encoding="utf-8")
    doc_audit = documentation_claim_audit(
        docs_to_audit,
        cited_report_status=decision["promotion_evidence_status"],
        cited_report_path=report_json,
    )
    write_documentation_claim_audit(doc_audit, output_root / "documentation_claim_audit.md")
    decision_payload["documentation_claim_audit"] = doc_audit
    report_json.write_text(json.dumps(decision_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(
        output_path=report_md,
        decision=decision,
        manifest_audit=manifest_audit,
        candidate_summaries=candidate_summaries,
        compatibility_summary=compatibility_summary,
    )
    _write_html_report(
        output_path=report_html,
        decision=decision,
        manifest_audit=manifest_audit,
        candidate_summaries=candidate_summaries,
    )
    return decision_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare glomeruli transfer and scratch candidates")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Supported glomeruli root: raw_data/cohorts, raw_data/cohorts/<cohort_id>, or an active paired raw_data project root",
    )
    parser.add_argument(
        "--output-dir",
        default=str(get_runtime_segmentation_evaluation_path("glomeruli_candidate_comparison")),
        help=(
            "Root directory for candidate-comparison runs "
            "(the run id is always appended)"
        ),
    )
    parser.add_argument("--run-id", help="Run directory name to create under --output-dir")
    parser.add_argument(
        "--model-dir",
        default=str(get_runtime_models_path() / "segmentation" / "glomeruli"),
        help="Root model-artifact directory for trained candidates; train_glomeruli writes transfer/ and scratch/ underneath",
    )
    parser.add_argument("--transfer-base-model", help="Explicit mitochondria base artifact for transfer training")
    parser.add_argument("--transfer-model-path", help="Existing transfer candidate artifact to evaluate instead of training")
    parser.add_argument("--scratch-model-path", help="Existing scratch candidate artifact to evaluate instead of training")
    parser.add_argument("--compat-model-path", help="Optional non-promoted comparison artifact path")
    parser.add_argument("--seed", type=int, default=42, help="Explicit seed for the initial one-seed-per-family workflow")
    parser.add_argument("--transfer-epochs", type=int, default=30, help="Transfer candidate epochs")
    parser.add_argument("--scratch-epochs", type=int, default=50, help="Scratch candidate epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional shared batch size override")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Shared training learning rate")
    parser.add_argument("--image-size", type=int, default=256, help="Model input size")
    parser.add_argument("--crop-size", type=int, default=512, help="Deterministic validation crop size and training crop size")
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        help="Device forwarded to fresh candidate training commands. Omit to auto-select cuda, then mps, then cpu.",
    )
    parser.add_argument("--loss", default="", help="Optional loss override forwarded to the training CLI")
    parser.add_argument("--negative-crop-manifest", help="Validated negative/background crop manifest forwarded to fresh candidate training")
    parser.add_argument("--negative-crop-sampler-weight", type=float, default=0.0, help="Deterministic negative/background crop manifest sampling weight")
    parser.add_argument("--augmentation-variant", default="fastai_default", choices=["fastai_default", "spatial_only", "current_plus_lighting"], help="Recorded augmentation policy variant for candidate provenance")
    parser.add_argument("--examples-per-category", type=int, default=DEFAULT_EXAMPLES_PER_CATEGORY, help="Deterministic manifest examples per category")
    parser.add_argument("--transfer-model-name", default="glomeruli_transfer_candidate", help="Transfer candidate model name prefix")
    parser.add_argument("--scratch-model-name", default="glomeruli_scratch_candidate", help="Scratch candidate model name prefix")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_glomeruli_candidates(args)


if __name__ == "__main__":
    main()
