import sys
from pathlib import Path

import pytest
import yaml

from eq.training.run_glomeruli_candidate_comparison_workflow import (
    run_glomeruli_candidate_comparison_workflow,
)


def test_resize_screening_fallback_config_is_rejected(tmp_path, monkeypatch):
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
    def fake_run(command, env, *, dry_run):
        raise AssertionError(f"resize fallback config should fail before subprocess launch: {command}")

    monkeypatch.setattr(
        "eq.training.run_glomeruli_candidate_comparison_workflow._run",
        fake_run,
    )

    with pytest.raises(ValueError, match="fallback_run_id"):
        run_glomeruli_candidate_comparison_workflow(config_path)
