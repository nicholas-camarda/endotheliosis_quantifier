from pathlib import Path

import pytest

from eq.run_config import run_glomeruli_transfer_config


def _base_transfer_config(runtime_root: Path, pretrained_model: dict[str, str]) -> dict:
    return {
        "workflow": "segmentation_glomeruli_transfer",
        "run": {
            "name": "handoff_test",
            "runtime_root_default": str(runtime_root),
            "python": "/usr/bin/python3",
        },
        "pretrained_model": pretrained_model,
        "data": {"processed": {"train_dir": "raw_data/cohorts", "crop_size": 512}},
        "model": {
            "output_dir": "models/segmentation/glomeruli",
            "model_name": "glom",
            "input_size": [256, 256],
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": 1e-3,
            },
        },
        "reproducibility": {"random_seed": 42},
    }


def test_run_config_rejects_latest_glob_handoff(tmp_path):
    config = _base_transfer_config(
        tmp_path / "runtime",
        {"artifact_glob": "models/segmentation/mitochondria/**/*.pkl"},
    )

    with pytest.raises(ValueError, match="artifact_glob is not supported"):
        run_glomeruli_transfer_config(config, dry_run=True)


def test_run_config_accepts_exact_artifact_path_for_dry_run(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, env, *, dry_run):
        commands.append(command)

    monkeypatch.setattr("eq.run_config._run", fake_run)
    runtime_root = tmp_path / "runtime"
    config = _base_transfer_config(
        runtime_root,
        {
            "artifact_path": (
                "models/segmentation/mitochondria/base_run/base_run.pkl"
            )
        },
    )

    run_glomeruli_transfer_config(config, dry_run=True)

    assert commands
    command = commands[0]
    assert command[command.index("--base-model") + 1] == str(
        runtime_root / "models/segmentation/mitochondria/base_run/base_run.pkl"
    )
