import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from eq.training import glomeruli_overcoverage_audit as audit
from eq.utils.run_io import metadata_path_for_model


def _make_training_root(root: Path) -> None:
    images = root / "images"
    masks = root / "masks"
    for subject in ["T19", "T20", "T21"]:
        (images / subject).mkdir(parents=True, exist_ok=True)
        (masks / subject).mkdir(parents=True, exist_ok=True)
    samples = [
        ("T19", "empty", np.zeros((32, 32), dtype=np.uint8)),
        ("T20", "boundary", np.pad(np.ones((8, 8), dtype=np.uint8), ((0, 24), (0, 24)))),
        ("T21", "positive", np.pad(np.ones((8, 8), dtype=np.uint8), ((12, 12), (12, 12)))),
    ]
    for subject, stem, mask in samples:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = 160
        image[..., 1] = 120
        image[..., 2] = 110
        Image.fromarray(image).save(images / subject / f"{stem}.jpg")
        Image.fromarray((mask * 255).astype(np.uint8)).save(masks / subject / f"{stem}_mask.png")


class _FakeLearn:
    def __init__(self, foreground_probability: float):
        self.model = _FakeModel(foreground_probability)


class _FakeModel(torch.nn.Module):
    def __init__(self, foreground_probability: float):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.foreground_logit = float(np.log(foreground_probability / (1.0 - foreground_probability)))

    def forward(self, tensor):
        batch, _channels, height, width = tensor.shape
        background = torch.zeros((batch, 1, height, width), device=tensor.device)
        foreground = torch.full((batch, 1, height, width), self.foreground_logit, device=tensor.device)
        return torch.cat([background, foreground], dim=1)


def _write_supported_metadata(model_path: Path, *, family: str) -> None:
    metadata_path = metadata_path_for_model(model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_status": "supported_runtime",
                "scientific_promotion_status": "not_evaluated",
                "candidate_family": family,
                "training_mode": "dynamic_full_image_patching",
                "crop_size": 32,
                "output_size": 16,
                "invocation": {"seed": 42},
            }
        ),
        encoding="utf-8",
    )


def test_parse_thresholds_defaults_and_validation():
    assert audit.parse_thresholds(None) == [0.01, 0.05, 0.1, 0.25, 0.5]
    assert audit.parse_thresholds("0.5,0.01,0.5") == [0.01, 0.5]
    with pytest.raises(ValueError, match="Thresholds must be in"):
        audit.parse_thresholds("-0.1,0.5")


def test_missing_candidate_paths_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing glomeruli overcoverage audit candidate path"):
        audit.validate_candidate_paths(
            [
                audit.AuditCandidate("transfer", tmp_path / "missing_transfer.pkl"),
                audit.AuditCandidate("scratch", tmp_path / "missing_scratch.pkl"),
            ]
        )


def test_overcoverage_audit_writes_required_artifacts(tmp_path, monkeypatch):
    training_root = tmp_path / "raw_data" / "project" / "training_pairs"
    _make_training_root(training_root)
    transfer_model = tmp_path / "transfer.pkl"
    scratch_model = tmp_path / "scratch.pkl"
    transfer_model.write_text("transfer")
    scratch_model.write_text("scratch")
    _write_supported_metadata(transfer_model, family="transfer")
    _write_supported_metadata(scratch_model, family="scratch")

    def fake_load_model_safely(path, model_type):
        if Path(path).name == "transfer.pkl":
            return _FakeLearn(0.03)
        return _FakeLearn(0.20)

    monkeypatch.setattr(audit, "load_model_safely", fake_load_model_safely)

    output_root = tmp_path / "audit_output"
    summary = audit.run_overcoverage_audit(
        Namespace(
            run_id="unit_overcoverage",
            transfer_model_path=str(transfer_model),
            scratch_model_path=str(scratch_model),
            data_dir=str(training_root),
            output_dir=str(output_root),
            thresholds="0.01,0.05,0.10,0.25,0.50",
            image_size=16,
            crop_size=32,
            examples_per_category=1,
            device="cpu",
            negative_crop_manifest=None,
        )
    )

    run_output = output_root / "unit_overcoverage"
    assert summary["output_dir"] == str(run_output)
    assert "threshold_policy_artifact" in summary["root_causes"]
    for filename in [
        "audit_summary.json",
        "candidate_inputs.json",
        "deterministic_validation_manifest.json",
        "probability_quantiles.csv",
        "threshold_sweep.csv",
        "threshold_sweep_by_crop.csv",
        "background_false_positive_curve.csv",
        "resize_policy_comparison.csv",
        "training_signal_ablation_summary.csv",
    ]:
        assert (run_output / filename).exists(), filename
    assert (run_output / "review_panels" / "index.html").exists()
    probability_header = (run_output / "probability_quantiles.csv").read_text().splitlines()[0]
    assert "foreground_probability_p50" in probability_header
    assert "area_probability_ge_0.01" in probability_header
    threshold_text = (run_output / "threshold_sweep.csv").read_text()
    assert "false_positive_foreground_fraction" in threshold_text
    assert "prediction_to_truth_foreground_ratio" in threshold_text
    candidate_inputs = json.loads((run_output / "candidate_inputs.json").read_text())
    assert candidate_inputs["thresholds"] == [0.01, 0.05, 0.1, 0.25, 0.5]
    assert candidate_inputs["candidates"][0]["model_sha256"]


def test_load_failure_is_recorded_without_shim(tmp_path, monkeypatch):
    training_root = tmp_path / "raw_data" / "project" / "training_pairs"
    _make_training_root(training_root)
    transfer_model = tmp_path / "transfer.pkl"
    scratch_model = tmp_path / "scratch.pkl"
    transfer_model.write_text("transfer")
    scratch_model.write_text("scratch")
    _write_supported_metadata(transfer_model, family="transfer")
    _write_supported_metadata(scratch_model, family="scratch")

    def fake_load_model_safely(path, model_type):
        if Path(path).name == "transfer.pkl":
            raise RuntimeError("legacy namespace missing")
        return _FakeLearn(0.20)

    monkeypatch.setattr(audit, "load_model_safely", fake_load_model_safely)

    summary = audit.run_overcoverage_audit(
        Namespace(
            run_id="load_failure",
            transfer_model_path=str(transfer_model),
            scratch_model_path=str(scratch_model),
            data_dir=str(training_root),
            output_dir=str(tmp_path / "audit_output"),
            thresholds=None,
            image_size=16,
            crop_size=32,
            examples_per_category=1,
            device="cpu",
            negative_crop_manifest=None,
        )
    )

    assert "insufficient_current_namespace_artifacts" in summary["root_causes"]
    assert summary["load_failures"][0]["candidate_family"] == "transfer"
    assert "legacy namespace missing" in summary["load_failures"][0]["error"]


def test_default_output_dir_uses_runtime_root(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime" / "endotheliosis_quantifier"
    monkeypatch.setenv("EQ_RUNTIME_ROOT", str(runtime_root))

    assert audit.default_output_dir("audit_run") == (
        runtime_root / "output" / "segmentation_evaluation" / "glomeruli_overcoverage_audit" / "audit_run"
    )
