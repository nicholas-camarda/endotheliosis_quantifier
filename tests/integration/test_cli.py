"""Small CLI smoke tests for the current unified interface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    env.setdefault("EQ_RUNTIME_ROOT", "/tmp/eq_cli_smoke_runtime")
    return subprocess.run(
        [sys.executable, '-m', 'eq', *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_cli_help():
    result = run_cli('--help')
    assert result.returncode == 0
    assert 'Endotheliosis Quantifier Pipeline' in result.stdout
    assert 'data-load' not in result.stdout
    assert 'process-data' not in result.stdout
    assert 'extract-features' not in result.stdout
    assert 'quantify' not in result.stdout
    assert 'audit-derived' not in result.stdout
    assert 'train-segmenter' not in result.stdout
    assert 'quant-endo' in result.stdout
    assert 'prepare-quant-contract' in result.stdout
    assert 'capabilities' in result.stdout
    assert 'mode' in result.stdout
    assert 'organize-lucchi' in result.stdout


def test_cli_no_command_shows_help_and_fails():
    result = run_cli()
    assert result.returncode == 1
    assert 'Endotheliosis Quantifier Pipeline' in result.stdout


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


def test_direct_quantification_help_exposes_reviewed_label_contract():
    for command in ('quant-endo', 'prepare-quant-contract'):
        result = run_cli(command, '--help')
        assert result.returncode == 0
        assert '--label-overrides' in result.stdout


def test_direct_quantification_commands_fail_without_reviewed_label_contract(tmp_path):
    for command in ('quant-endo', 'prepare-quant-contract'):
        result = run_cli(
            command,
            '--data-dir',
            str(tmp_path / 'data'),
            '--segmentation-model',
            str(tmp_path / 'model.pkl'),
            '--output-dir',
            str(tmp_path / 'out'),
        )
        assert result.returncode != 0
        assert 'requires --label-overrides' in (result.stdout + result.stderr)
        assert 'eq run-config --config configs/endotheliosis_quantification.yaml' in (
            result.stdout + result.stderr
        )


def test_cli_run_config_dry_runs_all_committed_configs():
    for config_name in (
        'mito_pretraining_config.yaml',
        'glomeruli_finetuning_config.yaml',
        'glomeruli_candidate_comparison.yaml',
        'glomeruli_transport_audit.yaml',
        'highres_glomeruli_concordance.yaml',
        'endotheliosis_quantification.yaml',
    ):
        result = run_cli('run-config', '--config', f'configs/{config_name}', '--dry-run')
        assert result.returncode == 0, result.stderr
        assert '/eq-mac/bin/python' in result.stdout
        assert 'LOG_PATH=' in result.stdout
        assert 'logs/run_config' in result.stdout
        assert 'EXECUTION_STATUS=completed' in result.stdout


def test_cli_run_config_rejects_retired_mixed_workflow(tmp_path):
    config = tmp_path / "retired.yaml"
    config.write_text("workflow: segmentation_fixedloader_full_retrain\n", encoding="utf-8")

    result = run_cli('run-config', '--config', str(config), '--dry-run')

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert 'Retired mixed workflow' in combined
    assert 'glomeruli_candidate_comparison' in combined


def test_cli_downstream_workflows_fail_without_explicit_inputs(tmp_path):
    config = tmp_path / "transport_missing_model.yaml"
    config.write_text(
        "\n".join(
            [
                "workflow: glomeruli_transport_audit",
                "run:",
                "  runtime_root_default: /tmp/eq_cli_smoke_runtime",
                "inputs:",
                "  manifest_path: raw_data/cohorts/manifest.csv",
                "  segmentation_outputs: output/predictions/example.csv",
                "outputs: {}",
            ]
        ),
        encoding="utf-8",
    )

    result = run_cli('run-config', '--config', str(config), '--dry-run')

    assert result.returncode != 0
    assert 'Missing required transport-audit input: segmentation_artifact' in (
        result.stdout + result.stderr
    )


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
