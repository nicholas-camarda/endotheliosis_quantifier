#!/usr/bin/env python3
"""Current glomeruli prediction utility for explicit model and data-root evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eq.data_management.datablock_loader import validate_supported_segmentation_training_root
from eq.data_management.standard_getters import get_y_full
from eq.evaluation.evaluate_glomeruli_model import GlomeruliModelEvaluator
from eq.utils.logger import get_logger
from eq.utils.paths import get_runtime_output_path


logger = get_logger("eq.glomeruli_prediction")


def _load_optional_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg if isinstance(cfg, dict) else {}


def run_glomeruli_prediction(
    *,
    model_path: str | None = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    max_images: int = 10,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate a glomeruli artifact on explicit image/mask pairs from a supported training root.

    The dedicated training module CLI is the canonical control surface for glomeruli training.
    This utility is only a small explicit evaluation helper. A YAML config may fill missing
    values, but it is not the authoritative workflow contract.
    """

    cfg = _load_optional_config(config_path)
    if not model_path:
        model_path = cfg.get("model", {}).get("checkpoint_path")
    if not data_dir:
        data_dir = cfg.get("data", {}).get("processed", {}).get("train_dir")
    if not model_path or not data_dir:
        raise ValueError(
            "run_glomeruli_prediction requires explicit model_path and data_dir, "
            "either directly or via an optional overlay config."
        )

    training_root = validate_supported_segmentation_training_root(data_dir, stage="glomeruli")
    image_paths = sorted((training_root / "images").rglob("*"))
    image_paths = [path for path in image_paths if path.is_file()]
    if not image_paths:
        raise ValueError(f"No images found under {training_root / 'images'}")

    output_path = (
        Path(output_dir).expanduser()
        if output_dir
        else get_runtime_output_path() / "glomeruli_prediction_eval"
    )
    evaluator = GlomeruliModelEvaluator(str(model_path), str(output_path))

    rows: list[dict[str, Any]] = []
    for image_path in image_paths[:max_images]:
        mask_path = get_y_full(image_path)
        result = evaluator.evaluate_single_image(str(image_path), str(mask_path))
        result["image_path"] = str(image_path)
        result["mask_path"] = str(mask_path)
        rows.append(result)

    if not rows:
        raise RuntimeError("No evaluable image/mask pairs were found.")

    summary = {
        "model_path": str(model_path),
        "data_dir": str(training_root),
        "max_images": max_images,
        "samples_evaluated": len(rows),
        "mean_dice": float(sum(row.get("dice", 0.0) for row in rows) / len(rows)),
        "mean_iou": float(sum(row.get("iou", 0.0) for row in rows) / len(rows)),
        "mean_pixel_acc": float(sum(row.get("pixel_acc", 0.0) for row in rows) / len(rows)),
        "config_path": str(config_path) if config_path else None,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "glomeruli_prediction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a glomeruli artifact on a supported training root")
    parser.add_argument("--model-path", help="Explicit glomeruli artifact path")
    parser.add_argument(
        "--data-dir",
        help="Supported glomeruli root: raw_data/cohorts, raw_data/cohorts/<cohort_id>, or an active paired raw_data project root",
    )
    parser.add_argument("--output-dir", help="Directory for evaluation artifacts")
    parser.add_argument("--max-images", type=int, default=10, help="Maximum number of paired images to evaluate")
    parser.add_argument("--config", help="Optional YAML overlay for filling missing values only")
    args = parser.parse_args()

    run_glomeruli_prediction(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        config_path=args.config,
    )
