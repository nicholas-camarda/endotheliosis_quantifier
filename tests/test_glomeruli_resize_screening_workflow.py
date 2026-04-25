import csv
import subprocess
import sys
from pathlib import Path

import yaml

from eq.training.run_glomeruli_candidate_comparison_workflow import (
    run_glomeruli_candidate_comparison_workflow,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_resize_screening_workflow_records_failed_primary_and_runs_fallback(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime"
    config_path = tmp_path / "resize_screen.yaml"
    comparison_root = runtime_root / "output" / "segmentation_evaluation" / "glomeruli_candidate_comparison"
    config = {
        "workflow": "glomeruli_candidate_comparison",
        "run": {
            "name": "resize_screen_test",
            "seed": 42,
            "runtime_root_default": str(runtime_root),
            "python": sys.executable,
            "training_device": "mps",
        },
        "paths": {
            "mito_data_dir": "raw_data/mitochondria_data/training",
            "glomeruli_data_dir": "raw_data/cohorts",
            "mito_model_dir": "models/segmentation/mitochondria",
            "glomeruli_model_dir": "models/segmentation/glomeruli",
            "comparison_output_dir": "output/segmentation_evaluation/glomeruli_candidate_comparison",
        },
        "mitochondria": {
            "enabled": False,
            "model_name": "mito",
            "epochs": 0,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "image_size": 256,
        },
        "glomeruli_transfer": {"model_name": "transfer", "epochs": 1},
        "glomeruli_scratch": {"model_name": "scratch", "epochs": 1},
        "candidate_training": {
            "batch_size": 2,
            "learning_rate": 1e-3,
            "image_size": 256,
            "crop_size": 512,
        },
        "negative_background_supervision": {"enabled": False},
        "augmentation_audit": {"variant": "fastai_default"},
        "comparison": {
            "enabled": True,
            "train_candidates": False,
            "transfer_model_path": "models/transfer.pkl",
            "scratch_model_path": "models/scratch.pkl",
            "overcoverage_audit_dir": "output/audit",
            "examples_per_category": 3,
        },
        "resize_screening": {
            "enabled": True,
            "record_failures": True,
            "update_reference_with_summary": True,
            "summary_run_id": "p0_resize_screening_summary",
            "reference_run_id": "p0_resize_screen_current_512to256",
            "primary_run_id": "p0_resize_screen_512to512",
            "fallback_run_id": "p0_resize_screen_512to384",
            "attempts": [
                {
                    "run_id": "p0_resize_screen_current_512to256",
                    "crop_size": 512,
                    "image_size": 256,
                    "batch_size": 2,
                    "run_if": "always",
                },
                {
                    "run_id": "p0_resize_screen_512to512",
                    "crop_size": 512,
                    "image_size": 512,
                    "batch_size": 1,
                    "run_if": "always",
                },
                {
                    "run_id": "p0_resize_screen_512to384",
                    "crop_size": 512,
                    "image_size": 384,
                    "batch_size": 1,
                    "run_if": "primary_failed",
                },
            ],
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, env, *, dry_run, log_handle):
        commands.append(command)
        if "eq.training.compare_glomeruli_candidates" not in command:
            return
        run_id = command[command.index("--run-id") + 1]
        if run_id == "p0_resize_screen_512to512":
            raise subprocess.CalledProcessError(137, command)
        run_output = comparison_root / run_id
        _write_csv(
            run_output / "candidate_summary.csv",
            [
                {
                    "family": "transfer",
                    "threshold_policy_status": "validation_derived_threshold",
                    "threshold": "0.25",
                    "category_gate_status": "promotion_eligible",
                    "category_gate_failed_count": "0",
                    "background_false_positive_foreground_fraction": "0.001",
                    "dice": "0.8",
                    "jaccard": "0.7",
                    "precision": "0.8",
                    "recall": "0.8",
                }
            ],
        )
        _write_csv(
            run_output / "candidate_predictions.csv",
            [
                {
                    "family": "transfer",
                    "category": "positive",
                    "dice": "0.8",
                    "recall": "0.8",
                    "precision": "0.8",
                    "truth_foreground_fraction": "0.2",
                    "prediction_foreground_fraction": "0.2",
                },
                {
                    "family": "transfer",
                    "category": "boundary",
                    "dice": "0.8",
                    "recall": "0.8",
                    "precision": "0.8",
                    "truth_foreground_fraction": "0.2",
                    "prediction_foreground_fraction": "0.2",
                },
            ],
        )

    monkeypatch.setattr(
        "eq.training.run_glomeruli_candidate_comparison_workflow._run",
        fake_run,
    )

    summary_path = run_glomeruli_candidate_comparison_workflow(config_path)

    rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
    run_ids = {row["run_id"] for row in rows}
    assert "p0_resize_screen_512to512" in run_ids
    assert "p0_resize_screen_512to384" in run_ids
    assert any(row["runtime_status"] == "failed" and row["failure_reason"] == "returncode=137" for row in rows)
    assert any(row["row_type"] == "decision" for row in rows)
    reference_commands = [
        command
        for command in commands
        if "eq.training.compare_glomeruli_candidates" in command
        and command[command.index("--run-id") + 1] == "p0_resize_screen_current_512to256"
    ]
    assert any("--resize-screening-summary" in command for command in reference_commands)
