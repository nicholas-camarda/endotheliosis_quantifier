"""Run repository workflow YAML configs through one CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from eq.utils.execution_logging import (
    execution_log_context,
    make_execution_log_context,
    run_logged_subprocess,
    runtime_root_environment,
)

SUPPORTED_WORKFLOWS = {
    'endotheliosis_quantification',
    'medsam_glomeruli_fine_tuning',
    'glomeruli_candidate_comparison',
    'glomeruli_transport_audit',
    'highres_glomeruli_concordance',
    'label_free_roi_embedding_atlas',
    'medsam_automatic_glomeruli_prompts',
    'medsam_manual_glomeruli_comparison',
    'segmentation_glomeruli_transfer',
    'segmentation_mitochondria_pretraining',
}

RETIRED_WORKFLOWS = {
    'full_segmentation_retrain',
    'segmentation_fixedloader_full_retrain',
}
LOGGER = logging.getLogger('eq.run_config')


def load_workflow_config(config_path: Path) -> dict[str, Any]:
    """Load a workflow config and require an explicit supported workflow."""
    if not config_path.exists():
        raise FileNotFoundError(f'Config does not exist: {config_path}')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Config must be a YAML mapping: {config_path}')
    workflow = str(config.get('workflow') or '')
    if workflow in RETIRED_WORKFLOWS:
        supported = ', '.join(sorted(SUPPORTED_WORKFLOWS))
        raise ValueError(
            f'Retired mixed workflow {workflow!r} is no longer supported. '
            f'Use one of the split workflow families instead: {supported}'
        )
    if workflow not in SUPPORTED_WORKFLOWS:
        supported = ', '.join(sorted(SUPPORTED_WORKFLOWS))
        raise ValueError(
            f'Unsupported or missing config workflow {workflow!r}. Supported workflows: {supported}'
        )
    config['_config_path'] = str(config_path)
    return config


def run_config(config_path: Path, *, dry_run: bool = False) -> None:
    """Run a supported workflow config."""
    config = load_workflow_config(config_path)
    workflow = str(config['workflow'])
    runtime_root = _runtime_root(config)
    run_cfg = config.get('run', {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    run_name = str(run_cfg.get('name') or workflow)
    command = [sys.executable, '-m', 'eq', 'run-config', '--config', str(config_path)]
    if dry_run:
        command.append('--dry-run')

    with runtime_root_environment(runtime_root):
        context = make_execution_log_context(
            surface='run_config',
            mode='run_config',
            config_run_name=run_name,
            runtime_root=runtime_root,
            dry_run=dry_run,
            config_path=config_path,
            command=command,
        )
        with execution_log_context(context, logger_name='eq', workflow=workflow):
            LOGGER.info('WORKFLOW_ID=%s', workflow)
            LOGGER.info('CONFIG_PATH=%s', config_path)
            LOGGER.info('RUNTIME_ROOT=%s', runtime_root)
            if workflow == 'glomeruli_candidate_comparison':
                from eq.training.run_glomeruli_candidate_comparison_workflow import (
                    run_glomeruli_candidate_comparison_workflow,
                )

                run_glomeruli_candidate_comparison_workflow(
                    config_path, dry_run=dry_run
                )
                return
            if workflow == 'glomeruli_transport_audit':
                from eq.evaluation.run_glomeruli_transport_audit_workflow import (
                    run_glomeruli_transport_audit_workflow,
                )

                run_glomeruli_transport_audit_workflow(config_path, dry_run=dry_run)
                return
            if workflow == 'highres_glomeruli_concordance':
                from eq.evaluation.run_highres_glomeruli_concordance_workflow import (
                    run_highres_glomeruli_concordance_workflow,
                )

                run_highres_glomeruli_concordance_workflow(config_path, dry_run=dry_run)
                return
            if workflow == 'endotheliosis_quantification':
                from eq.quantification.run_endotheliosis_quantification_workflow import (
                    run_endotheliosis_quantification_workflow,
                )

                run_endotheliosis_quantification_workflow(config_path, dry_run=dry_run)
                return
            if workflow == 'label_free_roi_embedding_atlas':
                from eq.quantification.embedding_atlas import (
                    run_label_free_roi_embedding_atlas,
                )

                run_label_free_roi_embedding_atlas(config_path, dry_run=dry_run)
                return
            if workflow == 'medsam_manual_glomeruli_comparison':
                from eq.evaluation.run_medsam_manual_glomeruli_comparison_workflow import (
                    run_medsam_manual_glomeruli_comparison_workflow,
                )

                run_medsam_manual_glomeruli_comparison_workflow(
                    config_path, dry_run=dry_run
                )
                return
            if workflow == 'medsam_automatic_glomeruli_prompts':
                from eq.evaluation.run_medsam_automatic_glomeruli_prompts_workflow import (
                    run_medsam_automatic_glomeruli_prompts_workflow,
                )

                run_medsam_automatic_glomeruli_prompts_workflow(
                    config_path, dry_run=dry_run
                )
                return
            if workflow == 'medsam_glomeruli_fine_tuning':
                from eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow import (
                    run_medsam_glomeruli_fine_tuning_workflow,
                )

                run_medsam_glomeruli_fine_tuning_workflow(config_path, dry_run=dry_run)
                return
            if workflow == 'segmentation_mitochondria_pretraining':
                run_mitochondria_pretraining_config(config, dry_run=dry_run)
                return
            if workflow == 'segmentation_glomeruli_transfer':
                run_glomeruli_transfer_config(config, dry_run=dry_run)
                return
            raise AssertionError(f'Unhandled supported workflow: {workflow}')


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


def _runner_env(config: dict[str, Any], runtime_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env['EQ_RUNTIME_ROOT'] = str(runtime_root)
    required_env = config.get('run', {}).get('required_env', {})
    if isinstance(required_env, dict):
        env.update({str(key): str(value) for key, value in required_env.items()})
    return env


def _python(config: dict[str, Any]) -> str:
    python = str(config.get('run', {}).get('python') or sys.executable)
    if not Path(python).exists():
        raise FileNotFoundError(f'Configured Python does not exist: {python}')
    return python


def _run(command: list[str], env: dict[str, str], *, dry_run: bool) -> None:
    started = time.time()
    LOGGER.info('COMMAND=%s', ' '.join(command))
    try:
        run_logged_subprocess(command, env=env, dry_run=dry_run, logger=LOGGER)
    except Exception:
        LOGGER.exception(
            'COMMAND_STATUS=failed ELAPSED_SECONDS=%.3f', time.time() - started
        )
        raise
    LOGGER.info('COMMAND_STATUS=completed ELAPSED_SECONDS=%.3f', time.time() - started)


def _model_input_size(model_cfg: dict[str, Any], default: int = 256) -> int:
    raw_size = model_cfg.get('input_size', default)
    if isinstance(raw_size, (list, tuple)):
        return int(raw_size[0])
    return int(raw_size)


def _exact_artifact_path(runtime_root: Path, raw_path: Any, *, dry_run: bool) -> Path:
    text = str(raw_path or '').strip()
    if not text:
        raise ValueError(
            'Supported artifact handoff requires pretrained_model.artifact_path.'
        )
    if any(token in text for token in ('*', '?', '[')):
        raise ValueError(
            'Supported artifact handoff requires an exact artifact_path, not a glob '
            f'or latest-artifact selector: {text}'
        )
    path = _runtime_path(runtime_root, text)
    if not dry_run and not path.exists():
        raise FileNotFoundError(f'Configured artifact_path does not exist: {path}')
    if not dry_run and not path.is_file():
        raise ValueError(f'Configured artifact_path is not a file: {path}')
    return path


def run_mitochondria_pretraining_config(
    config: dict[str, Any], *, dry_run: bool = False
) -> None:
    """Run mitochondria pretraining from a workflow YAML config."""
    runtime_root = _runtime_root(config)
    env = _runner_env(config, runtime_root)
    python = _python(config)
    data_cfg = config.get('data', {})
    processed_cfg = data_cfg.get('processed', {}) if isinstance(data_cfg, dict) else {}
    model_cfg = config.get('model', {})
    training_cfg = model_cfg.get('training', {}) if isinstance(model_cfg, dict) else {}

    _run(
        [
            python,
            '-m',
            'eq.training.train_mitochondria',
            '--config',
            str(config.get('_config_path', '')),
            '--data-dir',
            str(_runtime_path(runtime_root, processed_cfg['train_dir'])),
            '--model-dir',
            str(_runtime_path(runtime_root, model_cfg['output_dir'])),
            '--model-name',
            str(model_cfg['model_name']),
            '--epochs',
            str(training_cfg['epochs']),
            '--batch-size',
            str(training_cfg['batch_size']),
            '--learning-rate',
            str(training_cfg['learning_rate']),
            '--image-size',
            str(_model_input_size(model_cfg)),
            '--split-seed',
            str(
                processed_cfg.get(
                    'random_seed',
                    config.get('reproducibility', {}).get(
                        'random_seed', config.get('run', {}).get('seed', 42)
                    ),
                )
            ),
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
    data_cfg = config.get('data', {})
    processed_cfg = data_cfg.get('processed', {}) if isinstance(data_cfg, dict) else {}
    model_cfg = config.get('model', {})
    training_cfg = model_cfg.get('training', {}) if isinstance(model_cfg, dict) else {}
    pretrained_cfg = config.get('pretrained_model', {})
    if not isinstance(pretrained_cfg, dict):
        raise ValueError(
            'pretrained_model must be a mapping for glomeruli transfer config'
        )
    if 'artifact_glob' in pretrained_cfg:
        raise ValueError(
            'pretrained_model.artifact_glob is not supported. Configure an exact '
            'pretrained_model.artifact_path.'
        )
    base_model = _exact_artifact_path(
        runtime_root, pretrained_cfg.get('artifact_path'), dry_run=dry_run
    )

    _run(
        [
            python,
            '-m',
            'eq.training.train_glomeruli',
            '--config',
            str(config.get('_config_path', '')),
            '--data-dir',
            str(_runtime_path(runtime_root, processed_cfg['train_dir'])),
            '--model-dir',
            str(_runtime_path(runtime_root, model_cfg['output_dir'])),
            '--model-name',
            str(model_cfg['model_name']),
            '--base-model',
            str(base_model),
            '--epochs',
            str(training_cfg['epochs']),
            '--batch-size',
            str(training_cfg['batch_size']),
            '--learning-rate',
            str(training_cfg['learning_rate']),
            '--image-size',
            str(_model_input_size(model_cfg)),
            '--crop-size',
            str(processed_cfg.get('crop_size', _model_input_size(model_cfg))),
            '--seed',
            str(config.get('reproducibility', {}).get('random_seed', 42)),
            '--split-seed',
            str(
                processed_cfg.get(
                    'random_seed',
                    config.get('data', {})
                    .get('split', {})
                    .get(
                        'random_seed',
                        config.get('reproducibility', {}).get('random_seed', 42),
                    ),
                )
            ),
        ],
        env,
        dry_run=dry_run,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run a repository workflow YAML config.'
    )
    parser.add_argument('--config', required=True, help='Workflow YAML config to run.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without launching training or analysis.',
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_config(Path(args.config), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
