"""Run high-resolution glomeruli concordance workflow from YAML."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eq.quantification.cohorts import (
    build_mr_concordance_workflow,
    load_runtime_cohort_manifest,
)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    if payload.get("workflow") != "highres_glomeruli_concordance":
        raise ValueError("High-resolution config must use `workflow: highres_glomeruli_concordance`.")
    return payload


def _runtime_root(config: dict[str, Any]) -> Path:
    run_cfg = config.get("run", {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    env_name = str(run_cfg.get("runtime_root_env") or "EQ_RUNTIME_ROOT")
    runtime_value = os.environ.get(env_name) or run_cfg.get("runtime_root_default")
    if not runtime_value:
        raise ValueError(f"Runtime root is not set. Export {env_name} or set run.runtime_root_default.")
    return Path(str(runtime_value)).expanduser()


def _runtime_path(runtime_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return runtime_root / path


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_path(runtime_root: Path, section: dict[str, Any], key: str) -> Path:
    value = section.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required high-resolution concordance input: {key}")
    return _runtime_path(runtime_root, value)


def run_highres_glomeruli_concordance_workflow(
    config_path: Path, *, dry_run: bool = False
) -> dict[str, Path]:
    """Validate explicit large-field inputs and write MR-like concordance artifacts."""
    config = _load_config(config_path)
    runtime_root = _runtime_root(config)
    run_cfg = _mapping(config, "run")
    inputs = _mapping(config, "inputs")
    outputs = _mapping(config, "outputs")
    preprocessing = _mapping(config, "preprocessing")
    run_id = str(run_cfg.get("name") or "highres_concordance")
    python = str(run_cfg.get("python") or sys.executable)

    segmentation_artifact = _required_path(runtime_root, inputs, "segmentation_artifact")
    manifest_path = _required_path(runtime_root, inputs, "manifest_path")
    inferred_roi_grades = _required_path(runtime_root, inputs, "inferred_roi_grades")
    evaluation_dir = _runtime_path(
        runtime_root,
        outputs.get("evaluation_dir", f"output/segmentation_evaluation/highres_glomeruli_concordance/{run_id}"),
    )
    predictions_dir = _runtime_path(
        runtime_root,
        outputs.get("predictions_dir", f"output/predictions/highres_glomeruli_concordance/{run_id}"),
    )

    print("WORKFLOW=highres_glomeruli_concordance", flush=True)
    print(f"PYTHON={python}", flush=True)
    print(f"SEGMENTATION_ARTIFACT={segmentation_artifact}", flush=True)
    print(f"MANIFEST={manifest_path}", flush=True)
    print(f"INFERRED_ROI_GRADES={inferred_roi_grades}", flush=True)
    print(f"TILE_SIZE={preprocessing.get('tile_size')}", flush=True)
    print(f"EVALUATION_DIR={evaluation_dir}", flush=True)
    print(f"PREDICTIONS_DIR={predictions_dir}", flush=True)
    if dry_run:
        return {
            "evaluation_dir": evaluation_dir,
            "predictions_dir": predictions_dir,
        }

    for label, path in {
        "segmentation_artifact": segmentation_artifact,
        "manifest_path": manifest_path,
        "inferred_roi_grades": inferred_roi_grades,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Required high-resolution {label} does not exist: {path}")

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_runtime_cohort_manifest(manifest_path)
    roi_grades = pd.read_csv(inferred_roi_grades)
    outputs_map = build_mr_concordance_workflow(
        manifest,
        roi_grades,
        evaluation_dir,
        min_component_area=int(preprocessing.get("min_component_area", 2000)),
    )
    provenance_path = evaluation_dir / "workflow_provenance.json"
    provenance = {
        "workflow": "highres_glomeruli_concordance",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "segmentation_artifact": str(segmentation_artifact),
        "manifest_path": str(manifest_path),
        "inferred_roi_grades": str(inferred_roi_grades),
        "preprocessing": preprocessing,
        "evaluation_dir": str(evaluation_dir),
        "predictions_dir": str(predictions_dir),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    outputs_map["provenance"] = provenance_path
    outputs_map["predictions_dir"] = predictions_dir
    return outputs_map


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run high-resolution glomeruli concordance from YAML.")
    parser.add_argument("--config", default="configs/highres_glomeruli_concordance.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_highres_glomeruli_concordance_workflow(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
