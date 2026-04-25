import json
from datetime import datetime, timedelta
from pathlib import Path

from eq.data_management.output_manager import OutputManager
from eq.utils.config_manager import ConfigManager
from eq.utils.paths import (
    get_cache_path,
    get_data_path,
    get_dox_label_studio_export_path,
    get_logs_path,
    get_models_path,
    get_mr_image_root_path,
    get_mr_score_workbook_path,
    get_output_path,
    get_runtime_quantification_result_path,
    get_runtime_quantification_results_root,
    get_repo_root,
    get_runtime_cohort_manifest_path,
    get_runtime_cohort_manifest_summary_path,
    get_runtime_cohort_path,
    get_runtime_cohorts_root,
    get_runtime_mitochondria_data_path,
    get_runtime_output_path,
    get_runtime_prediction_path,
    get_runtime_predictions_root,
    get_runtime_raw_data_path,
    get_runtime_segmentation_evaluation_path,
    get_runtime_segmentation_evaluation_root,
)


def test_path_helpers_resolve_from_repo_root(monkeypatch):
    monkeypatch.delenv("EQ_DATA_PATH", raising=False)
    monkeypatch.delenv("EQ_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("EQ_CACHE_PATH", raising=False)
    monkeypatch.delenv("EQ_MODEL_PATH", raising=False)
    monkeypatch.delenv("EQ_LOG_PATH", raising=False)
    monkeypatch.delenv("EQ_LOGS_PATH", raising=False)
    monkeypatch.delenv("EQ_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("EQ_RUNTIME_OUTPUT_PATH", raising=False)

    repo_root = get_repo_root()
    runtime_root = Path.home() / "ProjectsRuntime" / repo_root.name
    expected_raw = runtime_root / "raw_data"
    expected_derived = runtime_root / "derived_data"

    assert get_data_path() == expected_raw
    assert get_output_path() == expected_derived
    assert get_cache_path() == runtime_root / "derived_data/cache"
    assert get_models_path() == runtime_root / "models"
    assert get_logs_path() == runtime_root / "logs"
    active_root = runtime_root
    assert get_runtime_raw_data_path() == active_root / "raw_data"
    assert get_runtime_output_path() == active_root / "output"
    assert get_runtime_cohorts_root() == active_root / "raw_data/cohorts"
    assert get_runtime_cohort_manifest_path() == active_root / "raw_data/cohorts/manifest.csv"
    assert get_runtime_cohort_manifest_summary_path() == active_root / "derived_data/cohort_manifest/manifest_summary.json"
    assert get_runtime_cohort_path("vegfri_dox") == active_root / "raw_data/cohorts/vegfri_dox"
    assert get_runtime_mitochondria_data_path() == active_root / "raw_data/mitochondria_data"
    assert get_runtime_segmentation_evaluation_root() == active_root / "output/segmentation_evaluation"
    assert (
        get_runtime_segmentation_evaluation_path("glomeruli_candidate_comparison")
        == active_root / "output/segmentation_evaluation/glomeruli_candidate_comparison"
    )
    assert get_runtime_predictions_root() == active_root / "output/predictions"
    assert get_runtime_prediction_path("glomeruli") == active_root / "output/predictions/glomeruli"
    assert get_runtime_quantification_results_root() == active_root / "output/quantification_results"
    assert get_runtime_quantification_result_path("lauren_preeclampsia") == active_root / "output/quantification_results/lauren_preeclampsia"


def test_path_helpers_prefer_runtime_root_when_present(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "raw_data").mkdir(parents=True)
    (runtime_root / "derived_data").mkdir(parents=True)

    monkeypatch.setenv("EQ_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.delenv("EQ_DATA_PATH", raising=False)
    monkeypatch.delenv("EQ_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("EQ_RUNTIME_OUTPUT_PATH", raising=False)

    assert get_data_path() == runtime_root / "raw_data"
    assert get_output_path() == runtime_root / "derived_data"
    assert get_runtime_raw_data_path() == runtime_root / "raw_data"
    assert get_runtime_cohorts_root() == runtime_root / "raw_data/cohorts"
    assert get_runtime_cohort_manifest_path() == runtime_root / "raw_data/cohorts/manifest.csv"
    assert get_runtime_cohort_manifest_summary_path() == runtime_root / "derived_data/cohort_manifest/manifest_summary.json"
    assert get_runtime_cohort_path("lauren_preeclampsia") == runtime_root / "raw_data/cohorts/lauren_preeclampsia"
    assert get_runtime_mitochondria_data_path() == runtime_root / "raw_data/mitochondria_data"
    assert get_runtime_output_path() == runtime_root / "output"
    assert get_runtime_segmentation_evaluation_root() == runtime_root / "output/segmentation_evaluation"
    assert (
        get_runtime_segmentation_evaluation_path("glomeruli_candidate_comparison")
        == runtime_root / "output/segmentation_evaluation/glomeruli_candidate_comparison"
    )
    assert get_runtime_predictions_root() == runtime_root / "output/predictions"
    assert get_runtime_prediction_path("mitochondria") == runtime_root / "output/predictions/mitochondria"
    assert get_runtime_quantification_results_root() == runtime_root / "output/quantification_results"
    assert get_runtime_quantification_result_path("lauren_preeclampsia") == runtime_root / "output/quantification_results/lauren_preeclampsia"


def test_runtime_cohort_helpers_accept_explicit_runtime_root(tmp_path):
    runtime_root = tmp_path / "explicit_runtime"

    assert get_runtime_raw_data_path(runtime_root) == runtime_root / "raw_data"
    assert get_runtime_cohorts_root(runtime_root) == runtime_root / "raw_data/cohorts"
    assert get_runtime_cohort_manifest_path(runtime_root) == runtime_root / "raw_data/cohorts/manifest.csv"
    assert get_runtime_cohort_manifest_summary_path(runtime_root) == runtime_root / "derived_data/cohort_manifest/manifest_summary.json"
    assert get_runtime_cohort_path("vegfri_dox", runtime_root) == runtime_root / "raw_data/cohorts/vegfri_dox"
    assert get_runtime_mitochondria_data_path(runtime_root) == runtime_root / "raw_data/mitochondria_data"
    assert get_runtime_segmentation_evaluation_root(runtime_root) == runtime_root / "output/segmentation_evaluation"
    assert (
        get_runtime_segmentation_evaluation_path("glomeruli", runtime_root)
        == runtime_root / "output/segmentation_evaluation/glomeruli"
    )
    assert get_runtime_predictions_root(runtime_root) == runtime_root / "output/predictions"
    assert get_runtime_prediction_path("glomeruli", runtime_root) == runtime_root / "output/predictions/glomeruli"
    assert get_runtime_quantification_results_root(runtime_root) == runtime_root / "output/quantification_results"
    assert get_runtime_quantification_result_path("vegfri_dox", runtime_root) == runtime_root / "output/quantification_results/vegfri_dox"


def test_external_cohort_source_paths_are_overridable(monkeypatch, tmp_path):
    dox_export = tmp_path / "dox.json"
    mr_workbook = tmp_path / "mr.xlsx"
    mr_images = tmp_path / "mr_images"

    monkeypatch.setenv("EQ_DOX_LABEL_STUDIO_EXPORT", str(dox_export))
    monkeypatch.setenv("EQ_MR_SCORE_WORKBOOK", str(mr_workbook))
    monkeypatch.setenv("EQ_MR_IMAGE_ROOT", str(mr_images))

    assert get_dox_label_studio_export_path() == dox_export
    assert get_mr_score_workbook_path() == mr_workbook
    assert get_mr_image_root_path() == mr_images


def test_current_docs_and_configs_use_operation_specific_output_roots():
    checked_paths = [
        Path("README.md"),
        Path("docs/OUTPUT_STRUCTURE.md"),
        Path("docs/ONBOARDING_GUIDE.md"),
        Path("configs/glomeruli_candidate_comparison.yaml"),
        Path("configs/glomeruli_transport_audit.yaml"),
        Path("configs/highres_glomeruli_concordance.yaml"),
        Path("configs/endotheliosis_quantification.yaml"),
        Path("configs/glomeruli_finetuning_config.yaml"),
        Path("configs/mito_pretraining_config.yaml"),
        Path("openspec/specs/scored-only-quantification-cohort/spec.md"),
        Path("openspec/specs/glomeruli-candidate-comparison/spec.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in checked_paths)

    assert "output/segmentation_results" not in text
    assert "output/segmentation_evaluation" in text
    assert "output/predictions" in text
    assert "masked_core" not in text
    assert "masked-core" not in text
    assert "admitted after mask-quality review" not in text
    assert "mask-quality gate" not in text
    assert "explicit mask-quality" not in text


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


def test_output_manager_cleanup_uses_run_metadata_created_at(tmp_path):
    manager = OutputManager(base_output_dir=str(tmp_path / "output"))

    stale_dirs = manager.create_output_directory("Old Data")
    fresh_dirs = manager.create_output_directory("New Data")

    old_created_at = (datetime.now() - timedelta(days=45)).isoformat()
    new_created_at = datetime.now().isoformat()

    (stale_dirs["main"] / "run_metadata.json").write_text(
        json.dumps({"created_at": old_created_at}, indent=2),
        encoding="utf-8",
    )
    (fresh_dirs["main"] / "run_metadata.json").write_text(
        json.dumps({"created_at": new_created_at}, indent=2),
        encoding="utf-8",
    )

    manager.cleanup_old_runs(max_age_days=30)

    assert not stale_dirs["main"].exists()
    assert fresh_dirs["main"].exists()
