from pathlib import Path

import pytest

from eq.utils import run_io


class _Recorder:
    values = [[0.2, 0.1]]
    metric_names = ["train_loss", "valid_loss"]


class _Learner:
    recorder = _Recorder()

    def export(self, path):
        Path(path).write_text("model", encoding="utf-8")


class _NoExportLearner:
    def export(self, path):
        return None


def test_required_split_manifest_write_failure_names_path(tmp_path: Path):
    output_dir = tmp_path / "missing"
    expected_path = output_dir / "model_splits.json"

    with pytest.raises(RuntimeError, match=str(expected_path)):
        run_io.save_splits(output_dir, "model", {"train_images": [], "valid_images": []})


def test_required_training_history_write_failure_names_path(tmp_path: Path):
    output_dir = tmp_path / "missing"
    expected_path = output_dir / "model_training_history.tsv"

    with pytest.raises(RuntimeError, match=str(expected_path)):
        run_io.save_training_history(_Learner(), output_dir, "model", {"epochs": 1})


def test_required_model_export_verifies_exported_file_exists(tmp_path: Path):
    expected_path = tmp_path / "model.pkl"

    with pytest.raises(RuntimeError, match=str(expected_path)):
        run_io.export_final_model(_NoExportLearner(), tmp_path, "model")


def test_required_run_metadata_write_failure_names_path(tmp_path: Path):
    output_dir = tmp_path / "missing"
    expected_path = output_dir / "model_run_metadata.txt"

    with pytest.raises(RuntimeError, match=str(expected_path)):
        run_io.save_run_metadata(output_dir, "model")


def test_optional_plot_failure_warns_with_artifact_path(tmp_path: Path, monkeypatch):
    warnings = []
    expected_path = tmp_path / "model_training_loss.png"

    monkeypatch.setattr(run_io.logger, "warning", lambda message: warnings.append(message))

    run_io.save_plots(object(), tmp_path, "model")

    assert any(str(expected_path) in message for message in warnings)
