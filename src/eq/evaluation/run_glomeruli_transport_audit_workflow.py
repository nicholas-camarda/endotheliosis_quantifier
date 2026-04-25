"""Run a standard glomeruli transport-audit workflow from YAML."""

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
    load_runtime_cohort_manifest,
    validate_segmentation_transport_inputs,
    write_segmentation_transport_audit,
)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    if payload.get("workflow") != "glomeruli_transport_audit":
        raise ValueError("Transport-audit config must use `workflow: glomeruli_transport_audit`.")
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


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_path(runtime_root: Path, section: dict[str, Any], key: str) -> Path:
    value = section.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required transport-audit input: {key}")
    return _runtime_path(runtime_root, value)


def run_glomeruli_transport_audit_workflow(
    config_path: Path, *, dry_run: bool = False
) -> dict[str, Path]:
    """Validate explicit transport inputs and write standard transport-audit artifacts."""
    config = _load_config(config_path)
    runtime_root = _runtime_root(config)
    run_cfg = _required_mapping(config, "run")
    inputs = _required_mapping(config, "inputs")
    outputs = _required_mapping(config, "outputs")
    run_id = str(run_cfg.get("name") or "transport_audit")
    python = str(run_cfg.get("python") or sys.executable)

    segmentation_artifact = _required_path(runtime_root, inputs, "segmentation_artifact")
    manifest_path = _required_path(runtime_root, inputs, "manifest_path")
    segmentation_outputs_path = _required_path(runtime_root, inputs, "segmentation_outputs")
    evaluation_dir = _runtime_path(
        runtime_root,
        outputs.get("evaluation_dir", f"output/segmentation_evaluation/glomeruli_transport_audit/{run_id}"),
    )
    predictions_dir = _runtime_path(
        runtime_root,
        outputs.get("predictions_dir", f"output/predictions/glomeruli_transport_audit/{run_id}"),
    )

    print(f"WORKFLOW=glomeruli_transport_audit", flush=True)
    print(f"PYTHON={python}", flush=True)
    print(f"SEGMENTATION_ARTIFACT={segmentation_artifact}", flush=True)
    print(f"MANIFEST={manifest_path}", flush=True)
    print(f"SEGMENTATION_OUTPUTS={segmentation_outputs_path}", flush=True)
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
        "segmentation_outputs": segmentation_outputs_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Required transport-audit {label} does not exist: {path}")

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_runtime_cohort_manifest(manifest_path)
    cohort_id = str(inputs.get("cohort_id") or "").strip()
    if cohort_id:
        manifest = manifest[manifest["cohort_id"].astype(str) == cohort_id].copy()
    segmentation_outputs = pd.read_csv(segmentation_outputs_path)
    reviewed = validate_segmentation_transport_inputs(manifest, segmentation_outputs)
    reviewed_path = evaluation_dir / "transport_validated_manifest.csv"
    reviewed.to_csv(reviewed_path, index=False)
    audit_path = write_segmentation_transport_audit(
        reviewed,
        evaluation_dir,
        segmentation_artifact=str(segmentation_artifact),
        reviewed_rows=reviewed,
        transport_status="complete",
    )
    provenance_path = evaluation_dir / "workflow_provenance.json"
    provenance = {
        "workflow": "glomeruli_transport_audit",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "segmentation_artifact": str(segmentation_artifact),
        "manifest_path": str(manifest_path),
        "segmentation_outputs": str(segmentation_outputs_path),
        "evaluation_dir": str(evaluation_dir),
        "predictions_dir": str(predictions_dir),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "audit": audit_path,
        "validated_manifest": reviewed_path,
        "provenance": provenance_path,
        "predictions_dir": predictions_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standard glomeruli transport audit from YAML.")
    parser.add_argument("--config", default="configs/glomeruli_transport_audit.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_glomeruli_transport_audit_workflow(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
