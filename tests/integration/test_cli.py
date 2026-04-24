"""Small CLI smoke tests for the current unified interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'eq', *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_help():
    result = run_cli('--help')
    assert result.returncode == 0
    assert 'Endotheliosis Quantifier Pipeline' in result.stdout
    assert 'data-load' in result.stdout
    assert 'train-segmenter' not in result.stdout
    assert 'seg' not in result.stdout
    assert 'capabilities' in result.stdout
    assert 'mode' in result.stdout
    assert 'organize-lucchi' in result.stdout


def test_cli_no_command_shows_help_and_fails():
    result = run_cli()
    assert result.returncode == 1
    assert 'Endotheliosis Quantifier Pipeline' in result.stdout


def test_cli_data_load_help():
    result = run_cli('data-load', '--help')
    assert result.returncode == 0
    assert '--data-dir' in result.stdout
    assert '--test-data-dir' in result.stdout
    assert '--cache-dir' in result.stdout


def test_cli_organize_lucchi_help():
    result = run_cli('organize-lucchi', '--help')
    assert result.returncode == 0
    assert '--input-dir' in result.stdout
    assert '--output-dir' in result.stdout
    assert 'raw_data/mitochondria_data' in result.stdout


def test_cli_run_config_help():
    result = run_cli('run-config', '--help')
    assert result.returncode == 0
    assert '--config' in result.stdout
    assert '--dry-run' in result.stdout


def test_cli_run_config_dry_runs_all_committed_configs():
    for config_name in (
        'mito_pretraining_config.yaml',
        'glomeruli_finetuning_config.yaml',
        'segmentation_fixedloader_full_retrain.yaml',
    ):
        result = run_cli('run-config', '--config', f'configs/{config_name}', '--dry-run')
        assert result.returncode == 0, result.stderr
        assert '/eq-mac/bin/python' in result.stdout


def test_cli_mode_show_runs():
    result = run_cli('mode', '--show')
    assert result.returncode == 0
    assert 'Environment Mode:' in result.stdout


def test_cli_capabilities_runs():
    result = run_cli('capabilities')
    assert result.returncode == 0
    assert 'Hardware Capability Report' in result.stdout


def test_cli_rejects_invalid_global_mode():
    result = run_cli('--mode', 'invalid_mode', 'capabilities')
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr
