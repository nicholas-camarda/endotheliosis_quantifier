"""Run the canonical endotheliosis quantification workflow from YAML."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eq.quantification.input_contract import (
    ResolvedQuantificationInputContract,
    resolve_quantification_input_contract,
    validate_committed_label_override_path,
)
from eq.quantification.pipeline import run_contract_first_quantification
from eq.utils.execution_logging import (
    direct_execution_log_context,
    runtime_root_environment,
)

LOGGER = logging.getLogger(
    'eq.quantification.run_endotheliosis_quantification_workflow'
)


def _emit(message: str) -> None:
    LOGGER.info('%s', message)
    print(message, flush=True)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f'Config does not exist: {config_path}')
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Config must be a mapping: {config_path}')
    if payload.get('workflow') != 'endotheliosis_quantification':
        raise ValueError(
            'Quantification config must use `workflow: endotheliosis_quantification`.'
        )
    return payload


def _runtime_root(config: dict[str, Any]) -> Path:
    run_cfg = config.get('run', {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    env_name = str(run_cfg.get('runtime_root_env') or 'EQ_RUNTIME_ROOT')
    runtime_value = os.environ.get(env_name) or run_cfg.get('runtime_root_default')
    if not runtime_value:
        raise ValueError(
            f'Runtime root is not set. Export {env_name} or set run.runtime_root_default.'
        )
    return Path(str(runtime_value)).expanduser()


def _runtime_path(runtime_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return runtime_root / path


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a mapping')
    return value


def _required_path(runtime_root: Path, section: dict[str, Any], key: str) -> Path:
    value = section.get(key)
    if value in (None, ''):
        raise ValueError(f'Missing required quantification input: {key}')
    return _runtime_path(runtime_root, value)


def resolve_endotheliosis_quantification_contract(
    *,
    data_dir: Path,
    segmentation_model: Path,
    output_dir: Path,
    mapping_file: Path | None = None,
    annotation_source: str | Path | None = None,
    score_source: str = 'auto',
    label_overrides_path: Path | None = None,
) -> ResolvedQuantificationInputContract:
    return resolve_quantification_input_contract(
        data_dir=data_dir,
        segmentation_model=segmentation_model,
        output_dir=output_dir,
        mapping_file=mapping_file,
        annotation_source=annotation_source,
        score_source=score_source,
        label_overrides_path=label_overrides_path,
    )


def run_endotheliosis_quantification_inputs(
    *,
    data_dir: Path,
    segmentation_model: Path,
    output_dir: Path,
    mapping_file: Path | None = None,
    annotation_source: str | Path | None = None,
    score_source: str = 'auto',
    label_overrides_path: Path | None = None,
    apply_migration: bool = False,
    stop_after: str = 'model',
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run the canonical contract-first quantification engine from explicit inputs."""
    contract = resolve_endotheliosis_quantification_contract(
        data_dir=Path(data_dir),
        segmentation_model=Path(segmentation_model),
        output_dir=Path(output_dir),
        mapping_file=Path(mapping_file) if mapping_file else None,
        annotation_source=annotation_source,
        score_source=score_source,
        label_overrides_path=Path(label_overrides_path)
        if label_overrides_path
        else None,
    )
    data_dir = contract.data_dir
    segmentation_model = contract.segmentation_model
    output_dir = contract.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_contract_first_quantification(
        project_dir=data_dir,
        segmentation_model_path=segmentation_model,
        output_dir=output_dir,
        mapping_file=Path(mapping_file) if mapping_file else None,
        annotation_source=annotation_source,
        score_source=contract.score_source,
        label_overrides_path=contract.label_overrides_path,
        apply_migration=apply_migration,
        stop_after=stop_after,
        input_contract=contract,
    )
    provenance_path = output_dir / 'workflow_provenance.json'
    payload = {
        'workflow': 'endotheliosis_quantification',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'data_dir': str(data_dir),
        'segmentation_model': str(segmentation_model),
        'output_dir': str(output_dir),
        'label_overrides_path': str(label_overrides_path)
        if label_overrides_path
        else '',
        'label_contract_reference': contract.reference(),
        'outputs': {key: str(value) for key, value in result.items()},
    }
    if provenance:
        payload.update(provenance)
    provenance_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    result['provenance'] = provenance_path
    return result


def run_endotheliosis_quantification_workflow(
    config_path: Path, *, dry_run: bool = False
) -> dict[str, Path]:
    """Run contract-first quantification without launching segmentation workflows."""
    config = _load_config(config_path)
    runtime_root = _runtime_root(config)
    run_cfg = _mapping(config, 'run')
    inputs = _mapping(config, 'inputs')
    outputs = _mapping(config, 'outputs')
    options = _mapping(config, 'options')
    run_id = str(run_cfg.get('name') or 'endotheliosis_quantification')
    python = str(run_cfg.get('python') or sys.executable)

    data_dir = _required_path(runtime_root, inputs, 'data_dir')
    segmentation_model = _required_path(runtime_root, inputs, 'segmentation_model')
    output_dir = _runtime_path(
        runtime_root,
        outputs.get('quantification_dir', f'output/quantification_results/{run_id}'),
    )
    mapping_file = inputs.get('mapping_file')
    annotation_source = inputs.get('annotation_source')
    label_overrides = inputs.get('label_overrides')
    validate_committed_label_override_path(label_overrides)
    input_score_source = str(inputs.get('score_source', '') or '')
    option_score_source = str(options.get('score_source', 'auto'))
    if input_score_source and input_score_source != option_score_source:
        raise ValueError(
            'Quantification config has divergent inputs.score_source and '
            f'options.score_source: {input_score_source!r} != {option_score_source!r}'
        )

    command = [
        sys.executable,
        '-m',
        'eq.quantification.run_endotheliosis_quantification_workflow',
        '--config',
        str(config_path),
    ]
    if dry_run:
        command.append('--dry-run')
    with (
        runtime_root_environment(runtime_root),
        direct_execution_log_context(
            surface='endotheliosis_quantification',
            config_run_name=run_id,
            runtime_root=runtime_root,
            dry_run=dry_run,
            config_path=config_path,
            command=command,
            workflow='endotheliosis_quantification',
            logger_name='eq',
        ) as log_context,
    ):
        _emit(f'EXECUTION_LOG={log_context.log_path}')
        _emit('WORKFLOW=endotheliosis_quantification')
        _emit(f'PYTHON={python}')
        _emit(f'DATA_DIR={data_dir}')
        _emit(f'SEGMENTATION_MODEL={segmentation_model}')
        _emit(f'OUTPUT_DIR={output_dir}')
        _emit(f'STOP_AFTER={options.get("stop_after", "model")}')
        if dry_run:
            return {'quantification_dir': output_dir}

        return run_endotheliosis_quantification_inputs(
            data_dir=data_dir,
            segmentation_model=segmentation_model,
            output_dir=output_dir,
            mapping_file=_runtime_path(runtime_root, mapping_file)
            if mapping_file
            else None,
            annotation_source=annotation_source,
            score_source=option_score_source,
            label_overrides_path=_runtime_path(runtime_root, label_overrides)
            if label_overrides
            else None,
            apply_migration=bool(options.get('apply_migration', False)),
            stop_after=str(options.get('stop_after', 'model')),
            provenance={
                'run_id': run_id,
                'config_path': str(config_path),
                'log_path': str(log_context.log_path),
            },
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run endotheliosis quantification from YAML.'
    )
    parser.add_argument('--config', default='configs/endotheliosis_quantification.yaml')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_endotheliosis_quantification_workflow(Path(args.config), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
