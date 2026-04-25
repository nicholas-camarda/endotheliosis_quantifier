#!/usr/bin/env python3
"""Deterministic probability and threshold audit for glomeruli overcoverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from eq.data_management.datablock_loader import validate_supported_segmentation_training_root
from eq.data_management.model_loading import load_model_safely
from eq.evaluation.segmentation_metrics import pixel_accuracy
from eq.inference.prediction_core import create_prediction_core
from eq.training.compare_glomeruli_candidates import (
    CandidateRuntime,
    _annotate_manifest_with_context,
    _build_deterministic_manifest,
    _load_crop_pair,
)
from eq.training.promotion_gates import binary_dice_jaccard, binary_precision_recall
from eq.training.segmentation_validation_audit import resize_policy_record
from eq.utils.paths import get_runtime_segmentation_evaluation_path

DEFAULT_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50)
REQUIRED_CATEGORIES = ("background", "boundary", "positive")
ROOT_CAUSES = {
    "threshold_policy_artifact",
    "training_signal_insufficient",
    "resize_policy_artifact",
    "augmentation_policy_artifact",
    "insufficient_current_namespace_artifacts",
    "inconclusive_short_run_only",
}
BACKGROUND_FALSE_POSITIVE_LIMIT = 0.02


@dataclass(frozen=True)
class AuditCandidate:
    family: str
    model_path: Path


def parse_thresholds(raw: str | Sequence[float] | None) -> list[float]:
    """Return a sorted unique threshold list."""
    if raw is None or raw == "":
        values = list(DEFAULT_THRESHOLDS)
    elif isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]
    if not values:
        raise ValueError("At least one threshold is required.")
    bad = [value for value in values if not 0.0 <= value <= 1.0]
    if bad:
        raise ValueError(f"Thresholds must be in [0, 1]; got {bad}.")
    return sorted(set(values))


def file_sha256(path: Path) -> str | None:
    """Hash a readable file without treating hash failures as model-load success."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def default_output_dir(run_id: str) -> Path:
    return get_runtime_segmentation_evaluation_path("glomeruli_overcoverage_audit") / run_id


def validate_candidate_paths(candidates: Sequence[AuditCandidate]) -> None:
    missing = [f"{candidate.family}:{candidate.model_path}" for candidate in candidates if not candidate.model_path.exists()]
    if missing:
        raise FileNotFoundError("Missing glomeruli overcoverage audit candidate path(s): " + ", ".join(missing))


def _write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _truth_binary(truth_crop: np.ndarray) -> np.ndarray:
    return np.asarray(truth_crop).astype(bool)


def _binary_metrics(truth_crop: np.ndarray, pred_crop: np.ndarray) -> dict[str, float]:
    metrics = binary_dice_jaccard(truth_crop, pred_crop)
    metrics.update(binary_precision_recall(truth_crop, pred_crop))
    metrics["pixel_accuracy"] = pixel_accuracy(np.asarray(pred_crop).astype(np.uint8), np.asarray(truth_crop).astype(np.uint8))
    return metrics


def predict_foreground_probability(
    learn: Any,
    image_crop: np.ndarray,
    truth_shape: tuple[int, int],
    expected_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return foreground probability resized back to truth shape."""
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
    return pred_resized.astype(np.float32), {
        "input_tensor_shape": [int(value) for value in tensor.shape],
        "raw_output_shape": [int(value) for value in raw_output.shape],
        "prediction_probability_shape": [int(value) for value in pred_np.shape],
        "prediction_resized_shape": [int(value) for value in pred_resized.shape],
        "inference_preprocessing": "deterministic_resize_imagenet_normalize",
        "threshold_resize_order": "resize_probability_then_threshold",
    }


def probability_quantile_row(
    *,
    candidate_family: str,
    crop_id: str,
    item: Mapping[str, Any],
    prob: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(prob, dtype=np.float64).reshape(-1)
    row: dict[str, Any] = {
        "candidate_family": candidate_family,
        "category": item.get("category"),
        "crop_id": crop_id,
        "source_image_identifier": item.get("subject_id") or item.get("image_name") or item.get("image_path"),
        "image_path": item.get("image_path"),
        "mask_path": item.get("mask_path"),
        "crop_box": json.dumps(item.get("crop_box")),
        "foreground_probability_min": float(np.min(values)),
        "foreground_probability_p01": float(np.percentile(values, 1)),
        "foreground_probability_p05": float(np.percentile(values, 5)),
        "foreground_probability_p10": float(np.percentile(values, 10)),
        "foreground_probability_p25": float(np.percentile(values, 25)),
        "foreground_probability_p50": float(np.percentile(values, 50)),
        "foreground_probability_p75": float(np.percentile(values, 75)),
        "foreground_probability_p90": float(np.percentile(values, 90)),
        "foreground_probability_p95": float(np.percentile(values, 95)),
        "foreground_probability_p99": float(np.percentile(values, 99)),
        "foreground_probability_max": float(np.max(values)),
        "foreground_probability_mean": float(np.mean(values)),
    }
    for threshold in thresholds:
        row[f"area_probability_ge_{threshold:g}"] = float(np.mean(values >= float(threshold)))
    return row


def threshold_metric_row(
    *,
    candidate_family: str,
    crop_id: str,
    item: Mapping[str, Any],
    truth_crop: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = (np.asarray(prob) >= float(threshold)).astype(np.uint8)
    truth = _truth_binary(truth_crop)
    metrics = _binary_metrics(truth, pred)
    truth_fraction = float(truth.mean())
    pred_fraction = float(pred.mean())
    category = str(item.get("category") or "")
    return {
        "candidate_family": candidate_family,
        "category": category,
        "threshold": float(threshold),
        "crop_id": crop_id,
        "image_path": item.get("image_path"),
        "mask_path": item.get("mask_path"),
        "crop_box": json.dumps(item.get("crop_box")),
        "crop_count": 1,
        "dice": metrics["dice"],
        "jaccard": metrics["jaccard"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "predicted_foreground_fraction": pred_fraction,
        "truth_foreground_fraction": truth_fraction,
        "false_positive_foreground_fraction": pred_fraction if category == "background" else "",
        "prediction_to_truth_foreground_ratio": "" if truth_fraction == 0.0 else float(pred_fraction / truth_fraction),
    }


def aggregate_threshold_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["candidate_family"]), str(row["category"]), float(row["threshold"]))
        grouped.setdefault(key, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for (family, category, threshold), values in sorted(grouped.items()):
        numeric = ("dice", "jaccard", "precision", "recall", "pixel_accuracy", "predicted_foreground_fraction", "truth_foreground_fraction")
        record: dict[str, Any] = {
            "candidate_family": family,
            "category": category,
            "threshold": threshold,
            "crop_count": len(values),
        }
        for key in numeric:
            record[key] = float(np.mean([float(row[key]) for row in values]))
        if category == "background":
            fp_values = [float(row["false_positive_foreground_fraction"]) for row in values]
            record["false_positive_foreground_fraction"] = float(np.mean(fp_values))
        else:
            ratios = [
                float(row["prediction_to_truth_foreground_ratio"])
                for row in values
                if str(row["prediction_to_truth_foreground_ratio"]) != ""
            ]
            record["prediction_to_truth_foreground_ratio"] = "" if not ratios else float(np.mean(ratios))
        aggregated.append(record)
    return aggregated


def background_false_positive_curve(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        if str(row.get("category")) != "background":
            continue
        grouped.setdefault((str(row["candidate_family"]), float(row["threshold"])), []).append(
            float(row["false_positive_foreground_fraction"])
        )
    output: list[dict[str, Any]] = []
    for (family, threshold), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "candidate_family": family,
                "threshold": threshold,
                "background_crop_count": int(arr.size),
                "false_positive_foreground_fraction_min": float(np.min(arr)),
                "false_positive_foreground_fraction_median": float(np.median(arr)),
                "false_positive_foreground_fraction_mean": float(np.mean(arr)),
                "false_positive_foreground_fraction_max": float(np.max(arr)),
            }
        )
    return output


def classify_overcoverage(
    *,
    probability_rows: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    load_failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    causes: list[str] = []
    evidence: list[str] = []
    if load_failures:
        causes.append("insufficient_current_namespace_artifacts")
        evidence.append("one_or_more_candidate_artifacts_failed_to_load")

    background_p50 = [
        float(row["foreground_probability_p50"])
        for row in probability_rows
        if str(row.get("category")) == "background"
    ]
    positive_p50 = [
        float(row["foreground_probability_p50"])
        for row in probability_rows
        if str(row.get("category")) in {"boundary", "positive"}
    ]
    threshold_001_background = [
        float(row["false_positive_foreground_fraction"])
        for row in threshold_rows
        if str(row.get("category")) == "background" and abs(float(row.get("threshold")) - 0.01) < 1e-9
    ]
    higher_background = [
        float(row["false_positive_foreground_fraction"])
        for row in threshold_rows
        if str(row.get("category")) == "background" and float(row.get("threshold")) >= 0.10
    ]
    if threshold_001_background and max(threshold_001_background) > BACKGROUND_FALSE_POSITIVE_LIMIT:
        higher_ok = bool(higher_background) and min(higher_background) <= BACKGROUND_FALSE_POSITIVE_LIMIT
        separable = bool(background_p50 and positive_p50) and np.median(background_p50) < np.median(positive_p50)
        if higher_ok or separable:
            causes.append("threshold_policy_artifact")
            evidence.append("background_false_positive_excess_at_0.01_improves_or_probabilities_are_separable")
        else:
            causes.append("training_signal_insufficient")
            evidence.append("background_false_positive_excess_persists_across_thresholds_or_probabilities_overlap")

    if not probability_rows and not load_failures:
        causes.append("inconclusive_short_run_only")
        evidence.append("no_probability_rows_available")

    if not causes:
        causes.append("inconclusive_short_run_only")
        evidence.append("threshold_and_probability_evidence_did_not_identify_a_single_root_cause")

    ordered = [cause for cause in ROOT_CAUSES if cause in set(causes)]
    return {"root_causes": ordered, "evidence": evidence}


def _array_to_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr)
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 1)
        data = (data * 255).astype(np.uint8)
    Image.fromarray(data).save(path)


def _overlay(image_crop: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = image_crop.astype(np.float32).copy()
    color_arr = np.zeros_like(base)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    active = np.asarray(mask).astype(bool)
    base[active] = 0.55 * base[active] + 0.45 * color_arr[active]
    return np.clip(base, 0, 255).astype(np.uint8)


def write_review_panels(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for row in rows:
        crop_id = str(row["crop_id"])
        family = str(row["candidate_family"])
        category = str(row["category"])
        image_crop = np.asarray(row["image_crop"])
        truth_crop = np.asarray(row["truth_crop"])
        prob = np.asarray(row["probability"])
        raw_path = output_dir / f"{family}_{crop_id}_raw.png"
        truth_path = output_dir / f"{family}_{crop_id}_truth.png"
        prob_path = output_dir / f"{family}_{crop_id}_probability.png"
        _array_to_png(raw_path, image_crop)
        _array_to_png(truth_path, _overlay(image_crop, truth_crop, (0, 255, 0)))
        _array_to_png(prob_path, prob)
        threshold_imgs = []
        for threshold in row["thresholds"]:
            pred = (prob >= float(threshold)).astype(np.uint8)
            pred_path = output_dir / f"{family}_{crop_id}_thr_{float(threshold):g}.png"
            _array_to_png(pred_path, _overlay(image_crop, pred, (255, 0, 0)))
            threshold_imgs.append((threshold, pred_path.name))
        threshold_html = "".join(
            f"<figure><img src='{html.escape(name)}' alt='threshold {float(threshold):g}'><figcaption>threshold={float(threshold):g}</figcaption></figure>"
            for threshold, name in threshold_imgs
        )
        cards.append(
            "<section class='card'>"
            f"<h2>{html.escape(family)} | {html.escape(category)} | {html.escape(crop_id)}</h2>"
            "<div class='grid'>"
            f"<figure><img src='{html.escape(raw_path.name)}' alt='raw'><figcaption>raw crop</figcaption></figure>"
            f"<figure><img src='{html.escape(truth_path.name)}' alt='truth'><figcaption>ground truth</figcaption></figure>"
            f"<figure><img src='{html.escape(prob_path.name)}' alt='probability'><figcaption>foreground probability</figcaption></figure>"
            f"{threshold_html}"
            "</div>"
            "</section>"
        )
    html_text = (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172033;background:#f8fafc;}"
        ".card{background:white;border:1px solid #dbe3ec;border-radius:8px;padding:16px;margin-bottom:18px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;}"
        "figure{margin:0;}img{width:100%;height:auto;border:1px solid #dbe3ec;border-radius:6px;background:white;}figcaption{font-size:13px;color:#475569;margin-top:6px;}"
        "</style></head><body><h1>Glomeruli Overcoverage Audit Review Panels</h1>"
        + "".join(cards)
        + "</body></html>"
    )
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def _empty_artifact_rows(*, image_size: int, crop_size: int, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resize_rows = [
        {
            **resize_policy_record(crop_size=crop_size, output_size=image_size),
            "screening_policy": "current_policy",
            "device": device,
            "batch_size": "",
            "runtime_failure": "",
        }
    ]
    training_rows = [
        {
            "changed_axis": "none_no_training_audit",
            "candidate_family": "",
            "run_id": "",
            "epochs": 0,
            "model_path": "",
            "comparison_artifact_path": "",
            "category_level_overcoverage_outcome": "not_applicable_no_training_audit",
        }
    ]
    return resize_rows, training_rows


def run_overcoverage_audit(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = parse_thresholds(args.thresholds)
    run_id = str(args.run_id)
    candidates = [
        AuditCandidate("transfer", Path(args.transfer_model_path).expanduser()),
        AuditCandidate("scratch", Path(args.scratch_model_path).expanduser()),
    ]
    validate_candidate_paths(candidates)
    data_root = validate_supported_segmentation_training_root(args.data_dir, stage="glomeruli")
    output_root = Path(args.output_dir).expanduser() / run_id if args.output_dir else default_output_dir(run_id)
    output_root.mkdir(parents=True, exist_ok=True)

    runtimes = [
        CandidateRuntime(family=candidate.family, role="overcoverage_audit", model_path=candidate.model_path, seed=None, command=None, status="available")
        for candidate in candidates
    ]
    manifest, manifest_audit = _build_deterministic_manifest(
        data_root=data_root,
        runtimes=runtimes,
        crop_size=int(args.crop_size),
        examples_per_category=int(args.examples_per_category),
    )
    manifest = _annotate_manifest_with_context(manifest, data_root)
    manifest_path = output_root / "deterministic_validation_manifest.json"
    manifest_path.write_text(json.dumps(list(manifest), indent=2, sort_keys=True), encoding="utf-8")

    candidate_inputs = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output_root),
        "device": args.device,
        "image_size": int(args.image_size),
        "crop_size": int(args.crop_size),
        "thresholds": thresholds,
        "deterministic_manifest_path": str(manifest_path),
        "deterministic_manifest_source": manifest_audit.get("manifest_source"),
        "negative_crop_manifest_path": str(Path(args.negative_crop_manifest).expanduser()) if args.negative_crop_manifest else None,
        "candidates": [
            {
                "candidate_family": candidate.family,
                "model_path": str(candidate.model_path),
                "model_sha256": file_sha256(candidate.model_path),
            }
            for candidate in candidates
        ],
    }
    (output_root / "candidate_inputs.json").write_text(json.dumps(candidate_inputs, indent=2, sort_keys=True), encoding="utf-8")

    probability_rows: list[dict[str, Any]] = []
    threshold_detail_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    load_failures: list[dict[str, Any]] = []

    for candidate in candidates:
        try:
            learn = load_model_safely(str(candidate.model_path), model_type="glomeruli")
            learn.model.eval()
        except Exception as exc:
            load_failures.append(
                {
                    "candidate_family": candidate.family,
                    "model_path": str(candidate.model_path),
                    "error": str(exc),
                    "root_cause": "insufficient_current_namespace_artifacts",
                }
            )
            continue
        for index, item in enumerate(manifest):
            image_crop, truth_crop = _load_crop_pair(item)
            prob, predict_audit = predict_foreground_probability(
                learn,
                image_crop,
                truth_crop.shape,
                int(args.image_size),
            )
            crop_id = f"{index:03d}_{str(item.get('category'))}"
            probability_rows.append(
                {
                    **probability_quantile_row(
                        candidate_family=candidate.family,
                        crop_id=crop_id,
                        item=item,
                        prob=prob,
                        thresholds=thresholds,
                    ),
                    **predict_audit,
                }
            )
            for threshold in thresholds:
                threshold_detail_rows.append(
                    threshold_metric_row(
                        candidate_family=candidate.family,
                        crop_id=crop_id,
                        item=item,
                        truth_crop=truth_crop,
                        prob=prob,
                        threshold=float(threshold),
                    )
                )
            panel_rows.append(
                {
                    "candidate_family": candidate.family,
                    "crop_id": crop_id,
                    "category": item.get("category"),
                    "image_crop": image_crop,
                    "truth_crop": truth_crop,
                    "probability": prob,
                    "thresholds": thresholds,
                }
            )

    threshold_rows = aggregate_threshold_rows(threshold_detail_rows)
    background_rows = background_false_positive_curve(threshold_detail_rows)
    resize_rows, training_rows = _empty_artifact_rows(image_size=int(args.image_size), crop_size=int(args.crop_size), device=str(args.device))
    root_cause = classify_overcoverage(
        probability_rows=probability_rows,
        threshold_rows=threshold_detail_rows,
        load_failures=load_failures,
    )
    audit_summary = {
        "run_id": run_id,
        "output_dir": str(output_root),
        "candidate_inputs_path": str(output_root / "candidate_inputs.json"),
        "manifest_path": str(manifest_path),
        "manifest_audit": manifest_audit,
        "thresholds": thresholds,
        "root_causes": root_cause["root_causes"],
        "root_cause_evidence": root_cause["evidence"],
        "load_failures": load_failures,
        "artifacts": {
            "probability_quantiles": str(output_root / "probability_quantiles.csv"),
            "threshold_sweep": str(output_root / "threshold_sweep.csv"),
            "threshold_sweep_by_crop": str(output_root / "threshold_sweep_by_crop.csv"),
            "background_false_positive_curve": str(output_root / "background_false_positive_curve.csv"),
            "resize_policy_comparison": str(output_root / "resize_policy_comparison.csv"),
            "training_signal_ablation_summary": str(output_root / "training_signal_ablation_summary.csv"),
            "review_panels": str(output_root / "review_panels" / "index.html"),
        },
    }

    _write_csv(probability_rows, output_root / "probability_quantiles.csv")
    _write_csv(threshold_rows, output_root / "threshold_sweep.csv")
    _write_csv(threshold_detail_rows, output_root / "threshold_sweep_by_crop.csv")
    _write_csv(background_rows, output_root / "background_false_positive_curve.csv")
    _write_csv(resize_rows, output_root / "resize_policy_comparison.csv")
    _write_csv(training_rows, output_root / "training_signal_ablation_summary.csv")
    write_review_panels(rows=panel_rows, output_dir=output_root / "review_panels")
    (output_root / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2, sort_keys=True), encoding="utf-8")
    return audit_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit glomeruli overcoverage across probability thresholds")
    parser.add_argument("--run-id", required=True, help="Run directory name for this audit")
    parser.add_argument("--transfer-model-path", required=True, help="Current-namespace transfer candidate artifact")
    parser.add_argument("--scratch-model-path", required=True, help="Current-namespace scratch/no-mito-base candidate artifact")
    parser.add_argument("--data-dir", required=True, help="Supported glomeruli raw-data root or manifest-backed cohorts root")
    parser.add_argument("--output-dir", help="Optional output root; run id is appended when supplied")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS), help="Comma-separated threshold grid")
    parser.add_argument("--image-size", type=int, default=256, help="Model input size")
    parser.add_argument("--crop-size", type=int, default=512, help="Deterministic validation crop size")
    parser.add_argument("--examples-per-category", type=int, default=2, help="Examples per background/boundary/positive category")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="cpu", help="Device label recorded in audit provenance")
    parser.add_argument("--negative-crop-manifest", help="Optional validated negative/background crop manifest path")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_overcoverage_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
