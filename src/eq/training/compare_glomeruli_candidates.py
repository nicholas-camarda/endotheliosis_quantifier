#!/usr/bin/env python3
"""Compare glomeruli transfer and scratch candidates under one deterministic promotion contract."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

from eq.data_management.datablock_loader import validate_supported_segmentation_training_root
from eq.data_management.model_loading import load_model_safely
from eq.evaluation.segmentation_metrics import pixel_accuracy
from eq.inference.prediction_core import create_prediction_core
from eq.training.promotion_gates import (
    audit_manifest_crops,
    binary_dice_jaccard,
    binary_precision_recall,
    deterministic_validation_manifest,
    evaluate_glomeruli_promotion_candidate,
    trivial_baseline_metrics,
)
from eq.utils.logger import get_logger
from eq.utils.paths import get_runtime_output_path
from eq.utils.run_io import metadata_path_for_model


TIE_MARGIN = 0.02
DEFAULT_EXAMPLES_PER_CATEGORY = 2
COMPARE_PREDICTION_THRESHOLD = 0.01

logger = get_logger("eq.glomeruli_candidate_comparison")


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
        return _build_fallback_provenance(runtime)
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
    return provenance


def _discover_model_path(family_dir: Path) -> Path:
    candidates = sorted(family_dir.rglob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model artifact produced under {family_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_training_command(command: list[str], family: str, role: str, output_dir: Path, seed: int) -> CandidateRuntime:
    try:
        logger.info("Running candidate training command: %s", " ".join(command))
        subprocess.run(command, check=True)
        family_dir = output_dir / family
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
    if from_scratch:
        command.append("--from-scratch")
    else:
        if base_model is None:
            raise ValueError("Transfer candidate requires an explicit base_model path.")
        command.extend(["--base-model", str(base_model)])
    return command


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


def _predict_crop(learn: Any, image_crop: np.ndarray, truth_shape: tuple[int, int], expected_size: int) -> np.ndarray:
    core = create_prediction_core(expected_size)
    pil_image = Image.fromarray(image_crop)
    batch = learn.dls.test_dl([pil_image]).one_batch()
    tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
    device = next(learn.model.parameters()).device
    with torch.no_grad():
        raw_output = learn.model(tensor.to(device))
    if raw_output.shape[1] == 2:
        pred_prob = torch.softmax(raw_output, dim=1)[:, 1]
    else:
        pred_prob = torch.sigmoid(raw_output)
    pred_np = pred_prob.squeeze().detach().cpu().numpy()
    pred_resized = core.resize_prediction_to_match(pred_np, truth_shape)
    return (pred_resized > COMPARE_PREDICTION_THRESHOLD).astype(np.uint8)


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

    learn = load_model_safely(str(runtime.model_path), model_type="glomeruli")
    learn.model.eval()
    truth_masks: List[np.ndarray] = []
    pred_masks: List[np.ndarray] = []
    prediction_rows: List[Dict[str, Any]] = []
    for index, item in enumerate(manifest):
        image_crop, truth_crop = _load_crop_pair(item)
        pred_crop = _predict_crop(learn, image_crop, truth_crop.shape, expected_size=expected_size)
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
                "manifest_index": index,
                "crop_box": json.dumps(item.get("crop_box")),
                "truth_foreground_fraction": float(truth_crop.mean()),
                "prediction_foreground_fraction": float(pred_crop.mean()),
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
        if row["family"] in {"transfer", "scratch"} and row["available"] and not row["gate"]["blocked"]
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
        result["decision_state"] = "blocked"
        result["decision_reason"] = "no_candidate_cleared_promotion_gates"
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
            result["decision_state"] = "blocked"
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
        lines.extend(
            [
                f"### {summary['family']}",
                f"- Available: `{summary['available']}`",
                f"- Role: `{summary['comparison_role']}`",
                f"- Artifact: `{summary['artifact_path']}`",
                f"- Seed: `{summary.get('seed')}`",
                f"- Gate blocked: `{summary['gate']['blocked']}`",
                f"- Gate reasons: `{', '.join(summary['gate'].get('reasons', []))}`",
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
            f"<div><strong>Seed</strong><br>{html.escape(str(summary.get('seed')))}</div>"
            f"<div><strong>Blocked</strong><br>{html.escape(str(summary['gate']['blocked']))}</div>"
            f"<div><strong>Role</strong><br>{html.escape(str(summary['comparison_role']))}</div>"
            f"<div><strong>Dice</strong><br>{float(metrics.get('dice', 0.0)):.3f}</div>"
            f"<div><strong>Jaccard</strong><br>{float(metrics.get('jaccard', 0.0)):.3f}</div>"
            f"<div><strong>Precision</strong><br>{float(metrics.get('precision', 0.0)):.3f}</div>"
            f"<div><strong>Recall</strong><br>{float(metrics.get('recall', 0.0)):.3f}</div>"
            "</div>"
            f"<p class='artifact'><strong>Artifact:</strong> {html.escape(str(summary.get('artifact_path')))}</p>"
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


def compare_glomeruli_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = validate_supported_segmentation_training_root(args.data_dir, stage="glomeruli")
    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = output_root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    asset_dir = output_root / "review_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    manifest = deterministic_validation_manifest(
        sorted(path for path in (data_root / "masks").rglob("*") if path.is_file()),
        crop_size=args.crop_size,
        examples_per_category=args.examples_per_category,
    )
    manifest_audit = audit_manifest_crops(manifest)
    manifest_path = output_root / "deterministic_validation_manifest.json"
    _write_manifest(manifest, manifest_path)

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
                model_dir=candidate_root,
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
            ),
            family="transfer",
            role="candidate",
            output_dir=candidate_root,
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
                model_dir=candidate_root,
                model_name=args.scratch_model_name,
                epochs=args.scratch_epochs,
                learning_rate=args.learning_rate,
                image_size=args.image_size,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                loss_name=args.loss,
                seed=args.seed,
                from_scratch=True,
            ),
            family="scratch",
            role="candidate",
            output_dir=candidate_root,
            seed=args.seed,
        )

    candidate_summaries = [
        _evaluate_runtime(transfer_runtime, manifest, asset_dir, expected_size=args.image_size),
        _evaluate_runtime(scratch_runtime, manifest, asset_dir, expected_size=args.image_size),
    ]

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

    candidate_rows = []
    for summary in candidate_summaries:
        candidate_rows.append(
            {
                "family": summary["family"],
                "comparison_role": summary["comparison_role"],
                "available": summary["available"],
                "artifact_path": summary["artifact_path"],
                "seed": summary.get("seed"),
                "blocked": summary["gate"]["blocked"],
                "reasons": "|".join(summary["gate"].get("reasons", [])),
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
    decision_payload = {
        "decision": decision,
        "manifest_path": str(manifest_path),
        "manifest_audit": manifest_audit,
        "candidate_summaries": candidate_summaries,
        "compatibility_summary": compatibility_summary,
    }
    report_json = output_root / "promotion_report.json"
    report_md = output_root / "promotion_report.md"
    report_html = output_root / "promotion_report.html"
    _write_csv(candidate_rows, output_root / "candidate_summary.csv")
    _write_csv(prediction_rows, output_root / "candidate_predictions.csv")
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
    parser.add_argument("--data-dir", required=True, help="Supported raw_data/.../training_pairs root")
    parser.add_argument(
        "--output-dir",
        default=str(get_runtime_output_path() / "glomeruli_candidate_comparison"),
        help=(
            "Directory for candidate comparison artifacts "
            "(defaults to the active runtime output root)"
        ),
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
    parser.add_argument("--loss", default="", help="Optional loss override forwarded to the training CLI")
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
