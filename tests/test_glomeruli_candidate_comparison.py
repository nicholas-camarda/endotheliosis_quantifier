from argparse import Namespace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from eq.training.compare_glomeruli_candidates import (
    COMPARE_PREDICTION_THRESHOLD,
    _validation_mask_paths,
    _predict_crop,
    build_arg_parser,
    compare_glomeruli_candidates,
    determine_promotion_decision,
)


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
        image[..., 1] = 120
        Image.fromarray(image).save(images / subject / f"{stem}.jpg")
        Image.fromarray((mask * 255).astype(np.uint8)).save(masks / subject / f"{stem}_mask.png")


def test_determine_promotion_decision_marks_tie_as_insufficient_evidence():
    summaries = [
        {
            "family": "transfer",
            "available": True,
            "artifact_path": "/tmp/transfer.pkl",
            "gate": {"blocked": False},
            "metrics": {"dice": 0.80, "jaccard": 0.70},
        },
        {
            "family": "scratch",
            "available": True,
            "artifact_path": "/tmp/scratch.pkl",
            "gate": {"blocked": False},
            "metrics": {"dice": 0.81, "jaccard": 0.705},
        },
    ]

    decision = determine_promotion_decision(summaries)

    assert decision["decision_state"] == "insufficient_evidence"
    assert decision["decision_reason"] == "transfer_and_scratch_are_within_practical_tie_margin"
    assert len(decision["research_use_candidates"]) == 2


def test_determine_promotion_decision_marks_unavailable_family_as_insufficient_evidence():
    summaries = [
        {
            "family": "transfer",
            "available": False,
            "artifact_path": None,
            "gate": {"blocked": True},
            "metrics": {},
        },
        {
            "family": "scratch",
            "available": True,
            "artifact_path": "/tmp/scratch.pkl",
            "gate": {"blocked": False},
            "metrics": {"dice": 0.9, "jaccard": 0.82},
        },
    ]

    decision = determine_promotion_decision(summaries)

    assert decision["decision_state"] == "insufficient_evidence"
    assert decision["decision_reason"] == "candidate_family_unavailable:transfer"


def test_compare_glomeruli_candidates_writes_reports_for_tied_candidates(tmp_path, monkeypatch):
    training_root = tmp_path / "raw_data" / "project" / "training_pairs"
    _make_training_root(training_root)
    output_root = tmp_path / "comparison"
    transfer_model = tmp_path / "transfer.pkl"
    scratch_model = tmp_path / "scratch.pkl"
    transfer_model.write_text("transfer")
    scratch_model.write_text("scratch")

    def fake_evaluate(runtime, manifest, asset_dir, expected_size):
        asset_dir.mkdir(parents=True, exist_ok=True)
        review_panel = asset_dir / f"{runtime.family}_00.png"
        Image.fromarray(np.zeros((32, 96, 3), dtype=np.uint8)).save(review_panel)
        metrics = {"dice": 0.80 if runtime.family == "transfer" else 0.81, "jaccard": 0.70 if runtime.family == "transfer" else 0.705, "precision": 0.8, "recall": 0.8}
        return {
            "family": runtime.family,
            "comparison_role": runtime.role,
            "available": True,
            "status": "available",
            "artifact_path": str(runtime.model_path),
            "seed": 42,
            "provenance": {"seed": 42, "training_mode": "dynamic_full_image_patching"},
            "command": runtime.command,
            "error": None,
            "gate": {"blocked": False, "reasons": []},
            "metrics": metrics,
            "prediction_rows": [
                {
                    "family": runtime.family,
                    "comparison_role": runtime.role,
                    "seed": 42,
                    "image_path": manifest[0]["image_path"],
                    "mask_path": manifest[0]["mask_path"],
                    "category": manifest[0]["category"],
                    "crop_box": json.dumps(manifest[0]["crop_box"]),
                    "truth_foreground_fraction": manifest[0]["foreground_fraction"],
                    "prediction_foreground_fraction": manifest[0]["foreground_fraction"],
                    "dice": metrics["dice"],
                    "jaccard": metrics["jaccard"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "review_panel_path": str(review_panel),
                }
            ],
        }

    monkeypatch.setattr(
        "eq.training.compare_glomeruli_candidates._evaluate_runtime",
        fake_evaluate,
    )

    payload = compare_glomeruli_candidates(
        Namespace(
            data_dir=str(training_root),
            output_dir=str(output_root),
            transfer_base_model=None,
            transfer_model_path=str(transfer_model),
            scratch_model_path=str(scratch_model),
            compat_model_path=None,
            seed=42,
            transfer_epochs=30,
            scratch_epochs=50,
            batch_size=None,
            learning_rate=1e-3,
            image_size=256,
            crop_size=32,
            loss="",
            examples_per_category=1,
            transfer_model_name="transfer_candidate",
            scratch_model_name="scratch_candidate",
        )
    )

    assert payload["decision"]["decision_state"] == "insufficient_evidence"
    assert (output_root / "deterministic_validation_manifest.json").exists()
    assert (output_root / "candidate_summary.csv").exists()
    assert (output_root / "candidate_predictions.csv").exists()
    assert (output_root / "promotion_report.json").exists()
    assert (output_root / "promotion_report.md").exists()
    assert (output_root / "promotion_report.html").exists()
    html_report = (output_root / "promotion_report.html").read_text()
    assert "Manifest Coverage" in html_report
    assert "panel order: raw | truth overlay | prediction overlay" in html_report


def test_candidate_comparison_uses_manifest_backed_cohort_registry_masks(tmp_path):
    runtime_root = tmp_path / "runtime"
    cohorts_root = runtime_root / "raw_data" / "cohorts"
    rows = []
    samples = [
        ("masked_core", "manual_mask", "core_empty", np.zeros((32, 32), dtype=np.uint8)),
        (
            "vegfri_dox",
            "masked_external",
            "dox_boundary",
            np.pad(np.ones((8, 8), dtype=np.uint8), ((0, 24), (0, 24))),
        ),
        (
            "vegfri_dox",
            "masked_external",
            "dox_positive",
            np.pad(np.ones((8, 8), dtype=np.uint8), ((12, 12), (12, 12))),
        ),
    ]
    for cohort_id, lane, stem, mask in samples:
        image_dir = cohorts_root / cohort_id / "images"
        mask_dir = cohorts_root / cohort_id / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_dir / f"{stem}.jpg")
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_dir / f"{stem}_mask.png")
        rows.append(
            {
                "cohort_id": cohort_id,
                "lane_assignment": lane,
                "admission_status": "admitted",
                "image_path": f"raw_data/cohorts/{cohort_id}/images/{stem}.jpg",
                "mask_path": f"raw_data/cohorts/{cohort_id}/masks/{stem}_mask.png",
            }
        )
    rows.append(
        {
            "cohort_id": "vegfri_mr",
            "lane_assignment": "mr_concordance_only",
            "admission_status": "evaluation_only",
            "image_path": "raw_data/cohorts/vegfri_mr/images/heldout.tif",
            "mask_path": "",
        }
    )
    import pandas as pd

    pd.DataFrame(rows).to_csv(cohorts_root / "manifest.csv", index=False)

    mask_paths = _validation_mask_paths(cohorts_root)

    assert len(mask_paths) == 3
    assert {path.name for path in mask_paths} == {
        "core_empty_mask.png",
        "dox_boundary_mask.png",
        "dox_positive_mask.png",
    }


def test_compare_parser_defaults_to_runtime_output_root(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime" / "endotheliosis_quantifier"
    runtime_root.mkdir(parents=True)
    monkeypatch.setenv("EQ_RUNTIME_ROOT", str(runtime_root))

    parser = build_arg_parser()
    args = parser.parse_args(["--data-dir", str(tmp_path / "raw_data" / "project" / "training_pairs")])

    assert args.output_dir == str(runtime_root / "output" / "glomeruli_candidate_comparison")


def test_predict_crop_uses_learner_preprocessing_and_underconfident_threshold(monkeypatch):
    class FakeCore:
        def resize_prediction_to_match(self, pred_mask, target_shape):
            return pred_mask

    class FakeTestDl:
        def one_batch(self):
            return (torch.zeros((1, 3, 2, 2), dtype=torch.float32),)

    class FakeDls:
        def test_dl(self, items):
            assert len(items) == 1
            return FakeTestDl()

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            fg = torch.full((1, 2, 2), -4.0, dtype=torch.float32, device=x.device)
            bg = torch.zeros((1, 2, 2), dtype=torch.float32, device=x.device)
            return torch.stack([bg, fg], dim=1)

    class FakeLearn:
        def __init__(self):
            self.dls = FakeDls()
            self.model = FakeModel()

    monkeypatch.setattr(
        "eq.training.compare_glomeruli_candidates.create_prediction_core",
        lambda expected_size: FakeCore(),
    )

    pred = _predict_crop(
        FakeLearn(),
        image_crop=np.zeros((2, 2, 3), dtype=np.uint8),
        truth_shape=(2, 2),
        expected_size=2,
    )

    assert COMPARE_PREDICTION_THRESHOLD == 0.01
    assert pred.shape == (2, 2)
    assert pred.sum() == 4
