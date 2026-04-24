"""Run the fixed-loader full segmentation retraining workflow from YAML."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


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


def _run(command: list[str], env: dict[str, str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def _lr_token(learning_rate: Any) -> str:
    return f"{float(learning_rate):.0e}".replace("-0", "-")


def _mito_run_pattern(config: dict[str, Any]) -> str:
    mito = config["mitochondria"]
    return (
        f"{mito['model_name']}-"
        f"pretrain_e{mito['epochs']}_"
        f"b{mito['batch_size']}_"
        f"lr{_lr_token(mito['learning_rate'])}_"
        f"sz{mito['image_size']}"
    )


def _latest_pkl(runtime_root: Path, config: dict[str, Any]) -> Path:
    model_dir = _runtime_path(runtime_root, str(config["paths"]["mito_model_dir"]))
    run_pattern = _mito_run_pattern(config)
    matches = sorted(
        Path(path)
        for path in glob.glob(str(model_dir / f"{run_pattern}" / "*.pkl"))
    )
    if not matches:
        raise FileNotFoundError(
            f"No mitochondria base artifact found for {run_pattern} under {model_dir}"
        )
    return matches[-1]


def run_fixedloader_full(config_path: Path, *, dry_run: bool = False) -> Path:
    """Run manifest refresh, mitochondria base training, and candidate comparison."""
    config = _load_config(config_path)
    runtime_root = _runtime_root(config)
    paths = config["paths"]
    mito = config["mitochondria"]
    transfer = config["glomeruli_transfer"]
    scratch = config["glomeruli_scratch"]
    candidate_training = config["candidate_training"]
    comparison = config["comparison"]

    env = os.environ.copy()
    env["EQ_RUNTIME_ROOT"] = str(runtime_root)
    required_env = config.get("run", {}).get("required_env", {})
    if isinstance(required_env, dict):
        env.update({str(key): str(value) for key, value in required_env.items()})

    python = str(config.get("run", {}).get("python") or sys.executable)
    if not Path(python).exists():
        raise FileNotFoundError(f"Configured Python does not exist: {python}")

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
            ],
            env,
            dry_run=dry_run,
        )

    mito_base_model = _runtime_path(runtime_root, "DRY_RUN_MITO_BASE.pkl") if dry_run else _latest_pkl(runtime_root, config)
    print(f"MITO_BASE_MODEL={mito_base_model}", flush=True)

    if bool(comparison.get("enabled", True)):
        _run(
            [
                python,
                "-m",
                "eq.training.compare_glomeruli_candidates",
                "--data-dir",
                str(_runtime_path(runtime_root, paths["glomeruli_data_dir"])),
                "--output-dir",
                str(_runtime_path(runtime_root, paths["comparison_output_dir"])),
                "--transfer-base-model",
                str(mito_base_model),
                "--seed",
                str(config["run"]["seed"]),
                "--transfer-epochs",
                str(transfer["epochs"]),
                "--scratch-epochs",
                str(scratch["epochs"]),
                "--batch-size",
                str(candidate_training["batch_size"]),
                "--learning-rate",
                str(candidate_training["learning_rate"]),
                "--image-size",
                str(candidate_training["image_size"]),
                "--crop-size",
                str(candidate_training["crop_size"]),
                "--examples-per-category",
                str(comparison["examples_per_category"]),
                "--transfer-model-name",
                str(transfer["model_name"]),
                "--scratch-model-name",
                str(scratch["model_name"]),
            ],
            env,
            dry_run=dry_run,
        )

    return mito_base_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-loader full segmentation retraining from YAML."
    )
    parser.add_argument(
        "--config",
        default="configs/segmentation_fixedloader_full_retrain.yaml",
        help="Fixed-loader full retraining YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without launching training.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_fixedloader_full(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
