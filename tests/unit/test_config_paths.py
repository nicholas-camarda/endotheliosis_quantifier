from __future__ import annotations

import json
from pathlib import Path

from eq.data_management.output_manager import OutputManager
from eq.utils.config_manager import ConfigManager
from eq.utils.paths import (
    get_cache_path,
    get_data_path,
    get_logs_path,
    get_models_path,
    get_output_path,
    get_repo_root,
)


def test_path_helpers_resolve_from_repo_root(monkeypatch):
    monkeypatch.delenv("EQ_DATA_PATH", raising=False)
    monkeypatch.delenv("EQ_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("EQ_CACHE_PATH", raising=False)
    monkeypatch.delenv("EQ_MODEL_PATH", raising=False)
    monkeypatch.delenv("EQ_LOG_PATH", raising=False)
    monkeypatch.delenv("EQ_LOGS_PATH", raising=False)

    repo_root = get_repo_root()

    assert get_data_path() == repo_root / "data/raw_data"
    assert get_output_path() == repo_root / "data/derived_data"
    assert get_cache_path() == repo_root / "data/derived_data/cache"
    assert get_models_path() == repo_root / "models"
    assert get_logs_path() == repo_root / "logs"


def test_config_manager_uses_resolved_paths_and_reloadable_global_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"

    monkeypatch.setenv("EQ_DATA_PATH", "custom/raw")
    monkeypatch.setenv("EQ_OUTPUT_PATH", "custom/derived")
    monkeypatch.setenv("EQ_CACHE_PATH", "custom/cache")
    monkeypatch.setenv("EQ_MODEL_PATH", "custom/models")
    monkeypatch.setenv("EQ_LOG_PATH", "custom/logs")

    manager = ConfigManager(config_path=config_path)

    repo_root = Path.cwd()
    assert manager.global_config.data_path == str(repo_root / "custom/raw")
    assert manager.global_config.output_path == str(repo_root / "custom/derived")
    assert manager.global_config.cache_path == str(repo_root / "custom/cache")
    assert manager.global_config.model_path == str(repo_root / "custom/models")
    assert manager.global_config.log_file == str(repo_root / "custom/logs/eq.log")

    config_data = {
        "global": {
            "data_path": "saved/raw",
            "output_path": "saved/derived",
            "cache_path": "saved/cache",
            "model_path": "saved/models",
            "log_file": "saved/logs/eq.log",
            "log_level": "DEBUG",
            "default_mode": "linux_gpu",
        },
        "modes": {},
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    monkeypatch.delenv("EQ_DATA_PATH", raising=False)
    monkeypatch.delenv("EQ_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("EQ_CACHE_PATH", raising=False)
    monkeypatch.delenv("EQ_MODEL_PATH", raising=False)
    monkeypatch.delenv("EQ_LOG_PATH", raising=False)

    reloaded = ConfigManager(config_path=config_path)

    assert reloaded.global_config.data_path == str(repo_root / "saved/raw")
    assert reloaded.global_config.output_path == str(repo_root / "saved/derived")
    assert reloaded.global_config.cache_path == str(repo_root / "saved/cache")
    assert reloaded.global_config.model_path == str(repo_root / "saved/models")
    assert reloaded.global_config.log_file == str(repo_root / "saved/logs/eq.log")
    assert reloaded.global_config.log_level == "DEBUG"
    assert reloaded.global_config.default_mode == "linux_gpu"

    monkeypatch.setenv("EQ_OUTPUT_PATH", "env/derived")
    monkeypatch.setenv("EQ_MODEL_PATH", "env/models")
    monkeypatch.setenv("EQ_LOG_PATH", "env/logs")
    overridden = ConfigManager(config_path=config_path)

    assert overridden.global_config.output_path == str(repo_root / "env/derived")
    assert overridden.global_config.model_path == str(repo_root / "env/models")
    assert overridden.global_config.log_file == str(repo_root / "env/logs/eq.log")


def test_output_manager_resolves_relative_base_dir_and_writes_clean_summary(tmp_path):
    manager = OutputManager(base_output_dir=str(tmp_path / "output"))
    output_dirs = manager.create_output_directory("Example Data")

    manager.create_run_summary(
        output_dirs,
        {
            "data_source": "Example Data",
            "run_type": "smoke",
            "timestamp": "2026-04-02_120000",
            "created_at": "2026-04-02T12:00:00",
            "config": {"batch_size": 4},
            "results": {"dice": 0.9},
        },
    )

    summary = (output_dirs["main"] / "run_summary.md").read_text(encoding="utf-8")

    assert summary.startswith("# Pipeline Run Summary\n\n## Run Information")
    assert "```json\n{\n  \"batch_size\": 4\n}\n```" in summary
    assert "```json\n{\n  \"dice\": 0.9\n}\n```" in summary
    assert "\n            ## Run Information" not in summary
