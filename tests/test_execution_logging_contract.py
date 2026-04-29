from __future__ import annotations

import logging
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
import yaml

from eq.utils.execution_logging import (
    ExecutionLogContext,
    derive_run_id,
    direct_execution_log_context,
    execution_log_context,
    resolve_execution_log_path,
    run_logged_subprocess,
)
from eq.utils.logger import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_execution_log_path_preserves_run_config_and_direct_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("EQ_RUNTIME_ROOT", str(runtime_root))

    run_config_path = resolve_execution_log_path(
        surface="run_config",
        run_id="example_run",
        mode="run_config",
        timestamp="2026-04-28_120000",
        dry_run=True,
    )
    direct_path = resolve_execution_log_path(
        surface="train_glomeruli",
        run_id="candidate_a",
        mode="direct",
        timestamp="2026-04-28_120001",
    )

    assert run_config_path == (
        runtime_root / "logs" / "run_config" / "example_run" / "2026-04-28_120000_dry_run.log"
    )
    assert direct_path == (
        runtime_root / "logs" / "direct" / "train_glomeruli" / "candidate_a" / "2026-04-28_120001.log"
    )
    assert REPO_ROOT not in run_config_path.parents
    assert REPO_ROOT not in direct_path.parents
    for unsafe in ("", "../bad", "bad/name", "bad\\name", ".", ".."):
        with pytest.raises(ValueError):
            resolve_execution_log_path(
                surface=unsafe,
                run_id="run",
                mode="direct",
                timestamp="2026-04-28_120000",
            )


def test_run_id_derivation_order() -> None:
    assert derive_run_id(explicit_run_id="explicit", config_run_name="config")[0:2] == (
        "explicit",
        "run_id",
    )
    assert derive_run_id(config_run_name="config", output_stem="out")[0:2] == (
        "config",
        "config_run_name",
    )
    assert derive_run_id(output_stem="/tmp/model.pkl")[0:2] == ("model", "output_stem")
    generated, source = derive_run_id(timestamp="2026-04-28_120000")
    assert generated == "run_2026-04-28_120000"
    assert source == "generated_timestamp"


def test_execution_log_context_records_start_success_and_removes_only_own_handler(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "runtime" / "logs" / "direct" / "surface" / "run" / "run.log"
    logger = logging.getLogger("eq.test.execution.success")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(StringIO())
    logger.addHandler(stream_handler)
    context = ExecutionLogContext(surface="surface", run_id="run", log_path=log_path)

    with execution_log_context(context, logger_name=logger.name) as active:
        assert active.log_path == log_path
        logger.info("CUSTOM_EVENT=inside")
        assert stream_handler in logger.handlers
        assert any(getattr(handler, "_eq_execution_handler", False) for handler in logger.handlers)

    assert stream_handler in logger.handlers
    assert not any(getattr(handler, "_eq_execution_handler", False) for handler in logger.handlers)
    text = log_path.read_text(encoding="utf-8")
    assert "SURFACE=surface" in text
    assert "RUN_ID=run" in text
    assert "CUSTOM_EVENT=inside" in text
    assert "EXECUTION_STATUS=completed" in text


def test_execution_log_context_records_failure_and_reraises_without_handler_leak(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "runtime" / "logs" / "direct" / "surface" / "run" / "run.log"
    logger = logging.getLogger("eq.test.execution.failure")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    context = ExecutionLogContext(surface="surface", run_id="run", log_path=log_path)

    with pytest.raises(RuntimeError, match="boom"):
        with execution_log_context(context, logger_name=logger.name):
            logger.info("CUSTOM_EVENT=before_failure")
            raise RuntimeError("boom")

    assert not any(getattr(handler, "_eq_execution_handler", False) for handler in logger.handlers)
    text = log_path.read_text(encoding="utf-8")
    assert "CUSTOM_EVENT=before_failure" in text
    assert "EXECUTION_STATUS=failed" in text
    assert "EXCEPTION_TYPE=RuntimeError" in text
    assert "EXCEPTION=boom" in text


def test_setup_logging_does_not_erase_active_execution_log_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime" / "logs" / "direct" / "surface" / "run" / "run.log"
    context = ExecutionLogContext(surface="surface", run_id="run", log_path=log_path)

    with execution_log_context(context, logger_name="eq"):
        logger = setup_logging(verbose=True)
        logger.info("AFTER_SETUP_LOGGING=still_captured")

    text = log_path.read_text(encoding="utf-8")
    assert "AFTER_SETUP_LOGGING=still_captured" in text
    assert "EXECUTION_STATUS=completed" in text


def test_colored_console_logging_does_not_leak_ansi_codes_to_execution_log(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "runtime" / "logs" / "direct" / "surface" / "run" / "run.log"
    context = ExecutionLogContext(surface="surface", run_id="run", log_path=log_path)
    logger = setup_logging(verbose=True)

    with execution_log_context(context, logger_name="eq"):
        logger.info("COLOR_SHOULD_STAY_ON_CONSOLE_ONLY")

    text = log_path.read_text(encoding="utf-8")
    assert "\033[" not in text
    assert "INFO - COLOR_SHOULD_STAY_ON_CONSOLE_ONLY" in text


def test_run_logged_subprocess_tees_stdout_and_stderr_on_success(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime" / "logs" / "direct" / "surface" / "run" / "run.log"
    context = ExecutionLogContext(surface="surface", run_id="run", log_path=log_path)

    with execution_log_context(context, logger_name="eq.subprocess.success"):
        run_logged_subprocess(
            [
                sys.executable,
                "-c",
                "import sys; print('stdout_line'); print('stderr_line', file=sys.stderr)",
            ],
            logger=logging.getLogger("eq.subprocess.success"),
            console=StringIO(),
        )

    text = log_path.read_text(encoding="utf-8")
    assert "SUBPROCESS_OUTPUT=stdout_line" in text
    assert "SUBPROCESS_OUTPUT=stderr_line" in text
    assert "SUBPROCESS_RETURN_CODE=0" in text
    assert "SUBPROCESS_STATUS=completed" in text


def test_run_logged_subprocess_logs_nonzero_return_code_and_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime" / "logs" / "direct" / "surface" / "run" / "run.log"
    context = ExecutionLogContext(surface="surface", run_id="run", log_path=log_path)

    with pytest.raises(subprocess.CalledProcessError):
        with execution_log_context(context, logger_name="eq.subprocess.failure"):
            run_logged_subprocess(
                [sys.executable, "-c", "import sys; print('before_exit'); sys.exit(7)"],
                logger=logging.getLogger("eq.subprocess.failure"),
                console=StringIO(),
            )

    text = log_path.read_text(encoding="utf-8")
    assert "SUBPROCESS_OUTPUT=before_exit" in text
    assert "SUBPROCESS_RETURN_CODE=7" in text
    assert "SUBPROCESS_STATUS=failed" in text
    assert "EXECUTION_STATUS=failed" in text


def test_repeated_execution_contexts_do_not_duplicate_log_lines(tmp_path: Path) -> None:
    logger = logging.getLogger("eq.test.execution.duplicates")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    paths = [
        tmp_path / "runtime" / "logs" / "direct" / "surface" / f"run_{index}" / "run.log"
        for index in range(2)
    ]

    for index, path in enumerate(paths):
        context = ExecutionLogContext(surface="surface", run_id=f"run_{index}", log_path=path)
        with execution_log_context(context, logger_name=logger.name):
            logger.info("UNIQUE_EVENT=%s", index)

    for index, path in enumerate(paths):
        text = path.read_text(encoding="utf-8")
        assert text.count(f"UNIQUE_EVENT={index}") == 1
    assert not logger.handlers


def _run_eq_config(config_path: Path, runtime_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["EQ_RUNTIME_ROOT"] = str(runtime_root)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "eq", "run-config", "--config", str(config_path), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_config_dry_runs_all_committed_configs_create_runtime_logs(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    for config_name in (
        "mito_pretraining_config.yaml",
        "glomeruli_finetuning_config.yaml",
        "glomeruli_candidate_comparison.yaml",
        "glomeruli_transport_audit.yaml",
        "highres_glomeruli_concordance.yaml",
        "endotheliosis_quantification.yaml",
    ):
        result = _run_eq_config(REPO_ROOT / "configs" / config_name, runtime_root)
        assert result.returncode == 0, result.stdout + result.stderr
        combined = result.stdout + result.stderr
        assert "LOG_PATH=" in combined
        assert "logs/run_config" in combined

    log_paths = sorted((runtime_root / "logs" / "run_config").glob("*/*.log"))
    assert len(log_paths) >= 6
    assert all("EXECUTION_STATUS=completed" in path.read_text(encoding="utf-8") for path in log_paths)
    assert not (REPO_ROOT / "logs").exists()


def _write_workflow_config(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_direct_workflow_dry_runs_create_direct_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("EQ_RUNTIME_ROOT", str(runtime_root))

    candidate_config = _write_workflow_config(
        tmp_path / "candidate.yaml",
        {
            "workflow": "glomeruli_candidate_comparison",
            "run": {
                "name": "candidate_direct",
                "seed": 42,
                "runtime_root_default": str(runtime_root),
                "python": sys.executable,
                "training_device": "cpu",
            },
            "paths": {
                "mito_data_dir": "raw_data/mitochondria_data/training",
                "glomeruli_data_dir": "raw_data/cohorts",
                "mito_model_dir": "models/segmentation/mitochondria",
                "glomeruli_model_dir": "models/segmentation/glomeruli",
                "comparison_output_dir": "output/segmentation_evaluation/glomeruli_candidate_comparison",
            },
            "mitochondria": {
                "enabled": False,
                "model_name": "mito",
                "epochs": 0,
                "batch_size": 1,
                "learning_rate": 1e-3,
                "image_size": 256,
            },
            "glomeruli_transfer": {
                "model_name": "transfer",
                "epochs": 1,
                "base_model_artifact_path": "models/segmentation/mitochondria/base.pkl",
            },
            "glomeruli_scratch": {"model_name": "scratch", "epochs": 1},
            "candidate_training": {
                "batch_size": 1,
                "learning_rate": 1e-3,
                "image_size": 256,
                "crop_size": 512,
            },
            "comparison": {"enabled": True, "examples_per_category": 1},
        },
    )
    transport_config = _write_workflow_config(
        tmp_path / "transport.yaml",
        {
            "workflow": "glomeruli_transport_audit",
            "run": {"name": "transport_direct", "runtime_root_default": str(runtime_root)},
            "inputs": {
                "segmentation_artifact": "models/glom.pkl",
                "manifest_path": "raw_data/cohorts/manifest.csv",
                "segmentation_outputs": "output/predictions.csv",
            },
            "outputs": {},
        },
    )
    highres_config = _write_workflow_config(
        tmp_path / "highres.yaml",
        {
            "workflow": "highres_glomeruli_concordance",
            "run": {"name": "highres_direct", "runtime_root_default": str(runtime_root)},
            "inputs": {
                "segmentation_artifact": "models/glom.pkl",
                "manifest_path": "raw_data/cohorts/manifest.csv",
                "inferred_roi_grades": "output/roi.csv",
            },
            "outputs": {},
            "preprocessing": {"tile_size": 512},
        },
    )
    quant_config = _write_workflow_config(
        tmp_path / "quant.yaml",
        {
            "workflow": "endotheliosis_quantification",
            "run": {"name": "quant_direct", "runtime_root_default": str(runtime_root)},
            "inputs": {
                "data_dir": "raw_data/endotheliosis",
                "segmentation_model": "models/glom.pkl",
            },
            "outputs": {},
            "options": {"stop_after": "model"},
        },
    )

    from eq.evaluation.run_glomeruli_transport_audit_workflow import (
        run_glomeruli_transport_audit_workflow,
    )
    from eq.evaluation.run_highres_glomeruli_concordance_workflow import (
        run_highres_glomeruli_concordance_workflow,
    )
    from eq.quantification.run_endotheliosis_quantification_workflow import (
        run_endotheliosis_quantification_workflow,
    )
    from eq.training.run_glomeruli_candidate_comparison_workflow import (
        run_glomeruli_candidate_comparison_workflow,
    )

    run_glomeruli_candidate_comparison_workflow(candidate_config, dry_run=True)
    run_glomeruli_transport_audit_workflow(transport_config, dry_run=True)
    run_highres_glomeruli_concordance_workflow(highres_config, dry_run=True)
    run_endotheliosis_quantification_workflow(quant_config, dry_run=True)

    expected_surfaces = {
        "glomeruli_candidate_comparison",
        "glomeruli_transport_audit",
        "highres_glomeruli_concordance",
        "endotheliosis_quantification",
    }
    actual_surfaces = {
        path.parent.parent.name for path in (runtime_root / "logs" / "direct").glob("*/*/*.log")
    }
    assert expected_surfaces <= actual_surfaces


def test_direct_training_and_comparison_mains_attach_direct_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("EQ_RUNTIME_ROOT", str(runtime_root))

    import eq.training.compare_glomeruli_candidates as compare
    import eq.training.train_glomeruli as train_glomeruli
    import eq.training.train_mitochondria as train_mitochondria

    monkeypatch.setattr(
        train_mitochondria,
        "get_segmentation_training_batch_size",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        train_mitochondria,
        "train_mitochondria_with_datablock",
        lambda **kwargs: (object(), tmp_path / "mito_model.pkl"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_mitochondria",
            "--data-dir",
            str(tmp_path / "mito_data"),
            "--model-dir",
            str(tmp_path / "models"),
            "--model-name",
            "mito_direct",
            "--epochs",
            "1",
        ],
    )
    train_mitochondria.main()

    monkeypatch.setattr(
        train_glomeruli,
        "get_segmentation_training_batch_size",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(train_glomeruli, "train_glomeruli_with_datablock", lambda **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_glomeruli",
            "--data-dir",
            str(tmp_path / "glom_data"),
            "--model-dir",
            str(tmp_path / "models"),
            "--model-name",
            "glom_direct",
            "--from-scratch",
            "--epochs",
            "1",
        ],
    )
    train_glomeruli.main()

    monkeypatch.setattr(compare, "compare_glomeruli_candidates", lambda args: {"run_id": args.run_id})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_glomeruli_candidates",
            "--data-dir",
            str(tmp_path / "glom_data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--model-dir",
            str(tmp_path / "models"),
            "--run-id",
            "compare_direct",
        ],
    )
    compare.main()

    logs = sorted((runtime_root / "logs" / "direct").glob("*/*/*.log"))
    surfaces = {path.parent.parent.name for path in logs}
    assert {"train_mitochondria", "train_glomeruli", "compare_glomeruli_candidates"} <= surfaces
    assert any("RUN_ID=mito_direct" in path.read_text(encoding="utf-8") for path in logs)
    assert any("RUN_ID=glom_direct" in path.read_text(encoding="utf-8") for path in logs)
    assert any("RUN_ID=compare_direct" in path.read_text(encoding="utf-8") for path in logs)
