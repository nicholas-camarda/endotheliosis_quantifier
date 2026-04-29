"""Run the glomeruli candidate-comparison workflow from YAML."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from eq.data_management.negative_glomeruli_crops import (
    generate_mask_derived_background_manifest,
    validate_negative_crop_manifest,
)
from eq.training.compare_glomeruli_candidates import (
    resize_screening_decision_from_rows,
)
from eq.utils.execution_logging import (
    direct_execution_log_context,
    run_logged_subprocess,
    runtime_root_environment,
)

LOGGER = logging.getLogger("eq.training.run_glomeruli_candidate_comparison_workflow")


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def _runtime_root(config: dict[str, Any]) -> Path:
    run_cfg = config.get("run", {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    env_name = str(run_cfg.get("runtime_root_env") or "EQ_RUNTIME_ROOT")
    runtime_value = os.environ.get(env_name) or run_cfg.get("runtime_root_default")
    if not runtime_value:
        raise ValueError(
            f"Runtime root is not set. Export {env_name} or set run.runtime_root_default."
        )
    return Path(str(runtime_value)).expanduser()


def _runtime_path(runtime_root: Path, raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return runtime_root / path


def _emit(message: str) -> None:
    LOGGER.info("%s", message)
    print(message, flush=True)


def _run(
    command: list[str],
    env: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    run_logged_subprocess(command, env=env, dry_run=dry_run, logger=LOGGER)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _resize_screen_attempt_rows(
    *,
    run_id: str,
    run_output_dir: Path,
    attempt: dict[str, Any],
    runtime_status: str,
    failure_reason: str,
    log_path: Path,
) -> list[dict[str, Any]]:
    base = {
        "row_type": "attempt",
        "run_id": run_id,
        "crop_size": attempt.get("crop_size"),
        "image_size": attempt.get("image_size"),
        "output_size": attempt.get("image_size"),
        "crop_to_output_resize_ratio": (
            float(attempt["crop_size"]) / float(attempt["image_size"])
            if attempt.get("crop_size") and attempt.get("image_size")
            else None
        ),
        "batch_size": attempt.get("batch_size"),
        "device": attempt.get("device"),
        "runtime_status": runtime_status,
        "failure_reason": failure_reason,
        "log_path": str(log_path),
        "output_dir": str(run_output_dir),
    }
    if runtime_status != "completed":
        return [{**base, "candidate_family": "all"}]

    candidate_rows = _read_csv_rows(run_output_dir / "candidate_summary.csv")
    prediction_rows = _read_csv_rows(run_output_dir / "candidate_predictions.csv")
    result_rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        family = str(candidate.get("family") or "")
        family_predictions = [
            row
            for row in prediction_rows
            if str(row.get("family") or "") == family
        ]
        positive_boundary = [
            row
            for row in family_predictions
            if str(row.get("category") or "") in {"positive", "boundary"}
        ]
        ratio_errors: list[float] = []
        for row in positive_boundary:
            truth_fg = _float_or_none(row.get("truth_foreground_fraction"))
            pred_fg = _float_or_none(row.get("prediction_foreground_fraction"))
            if truth_fg is not None and truth_fg > 0 and pred_fg is not None:
                ratio_errors.append(abs((pred_fg / truth_fg) - 1.0))
        result_rows.append(
            {
                **base,
                "candidate_family": family,
                "threshold_policy_status": candidate.get("threshold_policy_status"),
                "selected_threshold": candidate.get("threshold"),
                "category_gate_status": candidate.get("category_gate_status"),
                "category_gate_failed_count": candidate.get("category_gate_failed_count"),
                "background_false_positive_foreground_fraction": candidate.get("background_false_positive_foreground_fraction"),
                "positive_boundary_dice": _mean([
                    value
                    for value in (_float_or_none(row.get("dice")) for row in positive_boundary)
                    if value is not None
                ]),
                "positive_boundary_recall": _mean([
                    value
                    for value in (_float_or_none(row.get("recall")) for row in positive_boundary)
                    if value is not None
                ]),
                "positive_boundary_precision": _mean([
                    value
                    for value in (_float_or_none(row.get("precision")) for row in positive_boundary)
                    if value is not None
                ]),
                "positive_boundary_prediction_to_truth_ratio_abs_error": _mean(ratio_errors),
                "aggregate_dice": candidate.get("dice"),
                "aggregate_jaccard": candidate.get("jaccard"),
                "aggregate_precision": candidate.get("precision"),
                "aggregate_recall": candidate.get("recall"),
            }
        )
    return result_rows or [{**base, "candidate_family": "all", "failure_reason": "completed_but_candidate_summary_empty"}]


def _exact_artifact_path(runtime_root: Path, raw_path: Any, *, dry_run: bool) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError("An exact artifact path is required for this supported handoff.")
    if any(token in text for token in ("*", "?", "[")) or text.startswith("latest_"):
        raise ValueError(f"Exact artifact path required; unsupported selector: {text}")
    path = _runtime_path(runtime_root, text)
    if not dry_run and not path.exists():
        raise FileNotFoundError(f"Configured artifact path does not exist: {path}")
    if not dry_run and not path.is_file():
        raise ValueError(f"Configured artifact path is not a file: {path}")
    return path


def _validate_resize_screening_config(resize_screening: dict[str, Any]) -> None:
    if not bool(resize_screening.get("enabled", False)):
        return
    if resize_screening.get("fallback_run_id"):
        raise ValueError("resize_screening.fallback_run_id is not supported; use a separate explicit workflow config.")
    attempts = resize_screening.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("resize_screening.attempts must be a non-empty list when resize screening is enabled.")
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("Each resize_screening.attempts item must be a mapping.")
        if str(attempt.get("run_if") or "always") == "primary_failed":
            raise ValueError("resize_screening run_if: primary_failed is not supported; use a separate explicit workflow config.")


def run_glomeruli_candidate_comparison_workflow(
    config_path: Path, *, dry_run: bool = False
) -> Path:
    """Run manifest refresh, mitochondria base training, and candidate comparison."""
    config = _load_config(config_path)
    if config.get("workflow") != "glomeruli_candidate_comparison":
        raise ValueError(
            "Candidate-comparison workflow config must use "
            "`workflow: glomeruli_candidate_comparison`."
        )
    runtime_root = _runtime_root(config)
    run_id = str(config["run"]["name"])
    paths = config["paths"]
    mito = config["mitochondria"]
    transfer = config["glomeruli_transfer"]
    scratch = config["glomeruli_scratch"]
    candidate_training = config["candidate_training"]
    comparison = config["comparison"]
    negative_cfg = config.get("negative_background_supervision") or {}
    augmentation_audit = config.get("augmentation_audit") or {}
    train_candidates = bool(comparison.get("train_candidates", True))
    _validate_resize_screening_config(config.get("resize_screening") or {})

    env = os.environ.copy()
    env["EQ_RUNTIME_ROOT"] = str(runtime_root)
    required_env = config.get("run", {}).get("required_env", {})
    if isinstance(required_env, dict):
        env.update({str(key): str(value) for key, value in required_env.items()})

    python = str(config.get("run", {}).get("python") or sys.executable)
    if not Path(python).exists():
        raise FileNotFoundError(f"Configured Python does not exist: {python}")
    training_device = str(config.get("run", {}).get("training_device") or "").strip()
    if not training_device:
        raise ValueError("run.training_device must be set explicitly, for example `mps` on macOS.")

    command = [
        sys.executable,
        "-m",
        "eq.training.run_glomeruli_candidate_comparison_workflow",
        "--config",
        str(config_path),
    ]
    if dry_run:
        command.append("--dry-run")

    with runtime_root_environment(runtime_root), direct_execution_log_context(
        surface="glomeruli_candidate_comparison",
        config_run_name=run_id,
        runtime_root=runtime_root,
        dry_run=dry_run,
        config_path=config_path,
        command=command,
        workflow="glomeruli_candidate_comparison",
        logger_name="eq",
    ) as log_context:
        _emit(f"EXECUTION_LOG={log_context.log_path}")
        _emit(f"CONFIG={config_path}")
        _emit(f"WORKFLOW={config['workflow']}")
        _emit(f"RUN_ID={run_id}")
        _emit(f"RUNTIME_ROOT={runtime_root}")

        _run([python, "-m", "eq", "cohort-manifest"], env, dry_run=dry_run)

        if bool(mito.get("enabled", True)):
            _run(
                [
                    python,
                    "-m",
                    "eq.training.train_mitochondria",
                    "--data-dir",
                    str(_runtime_path(runtime_root, paths["mito_data_dir"])),
                    "--model-dir",
                    str(_runtime_path(runtime_root, paths["mito_model_dir"])),
                    "--model-name",
                    str(mito["model_name"]),
                    "--epochs",
                    str(mito["epochs"]),
                    "--batch-size",
                    str(mito["batch_size"]),
                    "--learning-rate",
                    str(mito["learning_rate"]),
                    "--image-size",
                    str(mito["image_size"]),
                    "--device",
                    training_device,
                ],
                env,
                dry_run=dry_run,
            )

        mito_base_model = None
        if train_candidates:
            mito_base_model = _exact_artifact_path(
                runtime_root,
                transfer.get("base_model_artifact_path") or transfer.get("base_model"),
                dry_run=dry_run,
            )
        _emit(f"MITO_BASE_MODEL={mito_base_model}")

        negative_manifest_path: Path | None = None
        negative_sampler_weight = 0.0
        if bool(negative_cfg.get("enabled", False)):
            mask_cfg = negative_cfg.get("mask_derived_background") or {}
            curated_cfg = negative_cfg.get("curated_negative_manifest") or {}
            if bool(mask_cfg.get("enabled", False)):
                curation_id = str(mask_cfg.get("curation_id") or f"{run_id}_mask_background")
                negative_manifest_path = (
                    runtime_root
                    / "derived_data"
                    / "glomeruli_negative_crops"
                    / "manifests"
                    / f"{curation_id}.csv"
                )
                negative_sampler_weight = float(mask_cfg.get("sampler_weight", 0.0))
                _emit(f"NEGATIVE_CROP_MANIFEST={negative_manifest_path}")
                if not dry_run:
                    audit = generate_mask_derived_background_manifest(
                        data_root=_runtime_path(runtime_root, paths["glomeruli_data_dir"]),
                        manifest_path=negative_manifest_path,
                        curation_id=curation_id,
                        crop_size=int(candidate_training["crop_size"]),
                        crops_per_image_limit=int(mask_cfg.get("crops_per_image_limit", 2)),
                        min_foreground_pixels=int(mask_cfg.get("min_foreground_pixels", 0)),
                        seed=int(config["run"]["seed"]),
                    )
                    _emit(f"NEGATIVE_CROP_AUDIT={audit}")
            if bool(curated_cfg.get("enabled", False)):
                manifest_value = curated_cfg.get("manifest_path")
                if not manifest_value:
                    raise ValueError(
                        "negative_background_supervision.curated_negative_manifest.manifest_path "
                        "is required when curated negatives are enabled."
                    )
                negative_manifest_path = _runtime_path(runtime_root, str(manifest_value))
                negative_sampler_weight = float(curated_cfg.get("sampler_weight", negative_sampler_weight))
                _emit(f"CURATED_NEGATIVE_CROP_MANIFEST={negative_manifest_path}")
                if not dry_run:
                    validate_negative_crop_manifest(negative_manifest_path)

        augmentation_variant = str(augmentation_audit.get("variant") or "fastai_default")

        def _comparison_command(
            *,
            command_run_id: str,
            image_size: int,
            crop_size: int,
            batch_size: Any,
            resize_screening_summary: Path | None = None,
        ) -> list[str]:
            command = [
                    python,
                    "-m",
                    "eq.training.compare_glomeruli_candidates",
                    "--data-dir",
                    str(_runtime_path(runtime_root, paths["glomeruli_data_dir"])),
                    "--output-dir",
                    str(_runtime_path(runtime_root, paths["comparison_output_dir"])),
                    "--model-dir",
                    str(_runtime_path(runtime_root, paths["glomeruli_model_dir"])),
                    "--run-id",
                    command_run_id,
                    "--seed",
                    str(config["run"]["seed"]),
                    "--transfer-epochs",
                    str(transfer["epochs"]),
                    "--scratch-epochs",
                    str(scratch["epochs"]),
                    "--batch-size",
                    str(batch_size),
                    "--learning-rate",
                    str(candidate_training["learning_rate"]),
                    "--image-size",
                    str(image_size),
                    "--crop-size",
                    str(crop_size),
                    "--device",
                    training_device,
                    "--examples-per-category",
                    str(comparison["examples_per_category"]),
                    "--transfer-model-name",
                    str(transfer["model_name"]),
                    "--scratch-model-name",
                    str(scratch["model_name"]),
            ]
            if train_candidates:
                command.extend(["--transfer-base-model", str(mito_base_model)])
            else:
                transfer_model_path = comparison.get("transfer_model_path")
                scratch_model_path = comparison.get("scratch_model_path")
                if not transfer_model_path or not scratch_model_path:
                    raise ValueError(
                        "comparison.transfer_model_path and comparison.scratch_model_path "
                        "are required when comparison.train_candidates is false."
                    )
                command.extend(
                    [
                        "--transfer-model-path",
                        str(_runtime_path(runtime_root, str(transfer_model_path))),
                        "--scratch-model-path",
                        str(_runtime_path(runtime_root, str(scratch_model_path))),
                    ]
                )
            if comparison.get("overcoverage_audit_dir"):
                command.extend(
                    [
                        "--overcoverage-audit-dir",
                        str(_runtime_path(runtime_root, str(comparison["overcoverage_audit_dir"]))),
                    ]
                )
            if resize_screening_summary is not None:
                command.extend(["--resize-screening-summary", str(resize_screening_summary)])
            if comparison.get("prediction_threshold") is not None:
                command.extend(["--prediction-threshold", str(comparison["prediction_threshold"])])
            if candidate_training.get("loss"):
                command.extend(["--loss", str(candidate_training["loss"])])
            if negative_manifest_path is not None:
                command.extend(
                    [
                        "--negative-crop-manifest",
                        str(negative_manifest_path),
                        "--negative-crop-sampler-weight",
                        str(negative_sampler_weight),
                    ]
                )
            command.extend(["--augmentation-variant", augmentation_variant])
            return command

        if bool(comparison.get("enabled", True)):
            resize_screening = config.get("resize_screening") or {}
            if bool(resize_screening.get("enabled", False)):
                attempts = resize_screening.get("attempts")
                comparison_output_root = _runtime_path(runtime_root, paths["comparison_output_dir"])
                summary_run_id = str(resize_screening.get("summary_run_id") or "p0_resize_screening_summary")
                summary_dir = comparison_output_root / summary_run_id
                summary_path = summary_dir / "resize_policy_screening_summary.csv"
                reference_run_id = str(resize_screening.get("reference_run_id") or "p0_resize_screen_current_512to256")
                primary_run_id = str(resize_screening.get("primary_run_id") or "p0_resize_screen_512to512")
                attempt_rows: list[dict[str, Any]] = []
                attempt_status: dict[str, str] = {}
                for raw_attempt in attempts:
                    attempt = dict(raw_attempt)
                    attempt_run_id = str(attempt["run_id"])
                    run_if = str(attempt.get("run_if") or "always")
                    if run_if != "always":
                        raise ValueError(f"Unsupported resize_screening run_if value for {attempt_run_id}: {run_if}")
                    attempt.setdefault("crop_size", candidate_training["crop_size"])
                    attempt.setdefault("image_size", candidate_training["image_size"])
                    attempt.setdefault("batch_size", candidate_training["batch_size"])
                    attempt.setdefault("device", training_device)
                    command = _comparison_command(
                        command_run_id=attempt_run_id,
                        image_size=int(attempt["image_size"]),
                        crop_size=int(attempt["crop_size"]),
                        batch_size=attempt["batch_size"],
                    )
                    run_output_dir = comparison_output_root / attempt_run_id
                    runtime_status = "completed"
                    failure_reason = ""
                    try:
                        _run(command, env, dry_run=dry_run)
                        if dry_run:
                            runtime_status = "dry_run"
                            failure_reason = "dry_run_no_artifacts_generated"
                    except subprocess.CalledProcessError as exc:
                        runtime_status = "failed"
                        failure_reason = f"returncode={exc.returncode}"
                        _emit(f"RESIZE_SCREEN_ATTEMPT_FAILED run_id={attempt_run_id} {failure_reason}")
                        if not bool(resize_screening.get("record_failures", True)):
                            raise
                    attempt_status[attempt_run_id] = runtime_status
                    attempt_rows.extend(
                        _resize_screen_attempt_rows(
                            run_id=attempt_run_id,
                            run_output_dir=run_output_dir,
                            attempt=attempt,
                            runtime_status=runtime_status,
                            failure_reason=failure_reason,
                            log_path=log_context.log_path,
                        )
                    )
                decision_row = resize_screening_decision_from_rows(
                    attempt_rows,
                    reference_run_id=reference_run_id,
                    primary_run_id=primary_run_id,
                )
                decision_row.update(
                    {
                        "run_id": "resize_screen_decision",
                        "candidate_family": "all",
                        "runtime_status": "completed",
                        "log_path": str(log_context.log_path),
                        "output_dir": str(summary_dir),
                    }
                )
                summary_rows = attempt_rows + [decision_row]
                _write_csv_rows(summary_rows, summary_path)
                _emit(f"RESIZE_SCREENING_SUMMARY={summary_path}")
                if bool(resize_screening.get("update_reference_with_summary", True)) and not dry_run:
                    reference_attempt = next(
                        (dict(row) for row in attempts if isinstance(row, dict) and str(row.get("run_id")) == reference_run_id),
                        None,
                    )
                    if reference_attempt is None:
                        raise ValueError(f"Reference resize-screen attempt is missing: {reference_run_id}")
                    reference_attempt.setdefault("crop_size", candidate_training["crop_size"])
                    reference_attempt.setdefault("image_size", candidate_training["image_size"])
                    reference_attempt.setdefault("batch_size", candidate_training["batch_size"])
                    _run(
                        _comparison_command(
                            command_run_id=reference_run_id,
                            image_size=int(reference_attempt["image_size"]),
                            crop_size=int(reference_attempt["crop_size"]),
                            batch_size=reference_attempt["batch_size"],
                            resize_screening_summary=summary_path,
                        ),
                        env,
                        dry_run=dry_run,
                    )
                return summary_path

            command = _comparison_command(
                command_run_id=run_id,
                image_size=int(candidate_training["image_size"]),
                crop_size=int(candidate_training["crop_size"]),
                batch_size=candidate_training["batch_size"],
            )
            _run(
                command,
                env,
                dry_run=dry_run,
            )

    return mito_base_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run glomeruli candidate comparison from YAML."
    )
    parser.add_argument(
        "--config",
        default="configs/glomeruli_candidate_comparison.yaml",
        help="Glomeruli candidate-comparison YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without launching training.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_glomeruli_candidate_comparison_workflow(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
