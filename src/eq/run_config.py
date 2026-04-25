"""Run repository workflow YAML configs through one CLI entrypoint."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_WORKFLOWS = {
    "segmentation_glomeruli_transfer",
    "full_segmentation_retrain",
    "segmentation_mitochondria_pretraining",
}


def load_workflow_config(config_path: Path) -> dict[str, Any]:
    """Load a workflow config and require an explicit supported workflow."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    workflow = str(config.get("workflow") or "")
    if workflow not in SUPPORTED_WORKFLOWS:
        supported = ", ".join(sorted(SUPPORTED_WORKFLOWS))
        raise ValueError(
            f"Unsupported or missing config workflow {workflow!r}. Supported workflows: {supported}"
        )
    config["_config_path"] = str(config_path)
    return config


def run_config(config_path: Path, *, dry_run: bool = False) -> None:
    """Run a supported workflow config."""
    config = load_workflow_config(config_path)
    workflow = str(config["workflow"])
    if workflow == "full_segmentation_retrain":
        from eq.training.run_full_segmentation_retrain import run_full_segmentation_retrain

        run_full_segmentation_retrain(config_path, dry_run=dry_run)
        return
    if workflow == "segmentation_mitochondria_pretraining":
        run_mitochondria_pretraining_config(config, dry_run=dry_run)
        return
    if workflow == "segmentation_glomeruli_transfer":
        run_glomeruli_transfer_config(config, dry_run=dry_run)
        return
    raise AssertionError(f"Unhandled supported workflow: {workflow}")


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


def _runtime_path(runtime_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return runtime_root / path


def _runner_env(config: dict[str, Any], runtime_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["EQ_RUNTIME_ROOT"] = str(runtime_root)
    required_env = config.get("run", {}).get("required_env", {})
    if isinstance(required_env, dict):
        env.update({str(key): str(value) for key, value in required_env.items()})
    return env


def _python(config: dict[str, Any]) -> str:
    python = str(config.get("run", {}).get("python") or sys.executable)
    if not Path(python).exists():
        raise FileNotFoundError(f"Configured Python does not exist: {python}")
    return python


def _run(command: list[str], env: dict[str, str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def _model_input_size(model_cfg: dict[str, Any], default: int = 256) -> int:
    raw_size = model_cfg.get("input_size", default)
    if isinstance(raw_size, (list, tuple)):
        return int(raw_size[0])
    return int(raw_size)


def _latest_artifact_from_glob(runtime_root: Path, raw_glob: str) -> Path:
    glob_path = _runtime_path(runtime_root, raw_glob)
    matches = sorted(Path(path) for path in glob.glob(str(glob_path), recursive=True))
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No artifact matched configured glob: {glob_path}")
    return matches[-1]


def run_mitochondria_pretraining_config(
    config: dict[str, Any], *, dry_run: bool = False
) -> None:
    """Run mitochondria pretraining from a workflow YAML config."""
    runtime_root = _runtime_root(config)
    env = _runner_env(config, runtime_root)
    python = _python(config)
    data_cfg = config.get("data", {})
    processed_cfg = data_cfg.get("processed", {}) if isinstance(data_cfg, dict) else {}
    model_cfg = config.get("model", {})
    training_cfg = model_cfg.get("training", {}) if isinstance(model_cfg, dict) else {}

    _run(
        [
            python,
            "-m",
            "eq.training.train_mitochondria",
            "--config",
            str(config.get("_config_path", "")),
            "--data-dir",
            str(_runtime_path(runtime_root, processed_cfg["train_dir"])),
            "--model-dir",
            str(_runtime_path(runtime_root, model_cfg["output_dir"])),
            "--model-name",
            str(model_cfg["model_name"]),
            "--epochs",
            str(training_cfg["epochs"]),
            "--batch-size",
            str(training_cfg["batch_size"]),
            "--learning-rate",
            str(training_cfg["learning_rate"]),
            "--image-size",
            str(_model_input_size(model_cfg)),
        ],
        env,
        dry_run=dry_run,
    )


def run_glomeruli_transfer_config(
    config: dict[str, Any], *, dry_run: bool = False
) -> None:
    """Run glomeruli transfer training from a workflow YAML config."""
    runtime_root = _runtime_root(config)
    env = _runner_env(config, runtime_root)
    python = _python(config)
    data_cfg = config.get("data", {})
    processed_cfg = data_cfg.get("processed", {}) if isinstance(data_cfg, dict) else {}
    model_cfg = config.get("model", {})
    training_cfg = model_cfg.get("training", {}) if isinstance(model_cfg, dict) else {}
    pretrained_cfg = config.get("pretrained_model", {})
    if not isinstance(pretrained_cfg, dict):
        raise ValueError("pretrained_model must be a mapping for glomeruli transfer config")
    base_model = (
        _runtime_path(runtime_root, "DRY_RUN_BASE_MODEL.pkl")
        if dry_run
        else _latest_artifact_from_glob(runtime_root, str(pretrained_cfg["artifact_glob"]))
    )
    if not dry_run and not base_model.exists():
        raise FileNotFoundError(f"Configured base model does not exist: {base_model}")

    _run(
        [
            python,
            "-m",
            "eq.training.train_glomeruli",
            "--config",
            str(config.get("_config_path", "")),
            "--data-dir",
            str(_runtime_path(runtime_root, processed_cfg["train_dir"])),
            "--model-dir",
            str(_runtime_path(runtime_root, model_cfg["output_dir"])),
            "--model-name",
            str(model_cfg["model_name"]),
            "--base-model",
            str(base_model),
            "--epochs",
            str(training_cfg["epochs"]),
            "--batch-size",
            str(training_cfg["batch_size"]),
            "--learning-rate",
            str(training_cfg["learning_rate"]),
            "--image-size",
            str(_model_input_size(model_cfg)),
            "--crop-size",
            str(processed_cfg.get("crop_size", _model_input_size(model_cfg))),
            "--seed",
            str(config.get("reproducibility", {}).get("random_seed", 42)),
        ],
        env,
        dry_run=dry_run,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a repository workflow YAML config.")
    parser.add_argument("--config", required=True, help="Workflow YAML config to run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training or analysis.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_config(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
