import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eq.training.compare_glomeruli_candidates import (
    COMPARE_PREDICTION_THRESHOLD,
    CandidateRuntime,
    _annotate_manifest_with_context,
    _merged_provenance,
    _candidate_command,
    _predict_crop,
    _validation_mask_paths,
    _write_shared_training_split,
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


def test_merged_provenance_loads_adjacent_split_sidecar(tmp_path):
    model_path = tmp_path / "candidate.pkl"
    model_path.write_text("model")
    (tmp_path / "candidate_run_metadata.json").write_text(
        json.dumps(
            {
                "training_mode": "dynamic_full_image_patching",
                "artifact_status": "supported_runtime",
                "scientific_promotion_status": "not_evaluated",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "candidate_splits.json").write_text(
        json.dumps(
            {
                "train_images": ["/data/train_a.jpg"],
                "valid_images": ["/data/valid_a.jpg"],
                "data_root": "/data",
                "crop_size": 512,
                "output_size": 256,
            }
        ),
        encoding="utf-8",
    )

    provenance = _merged_provenance(
        CandidateRuntime(
            family="transfer",
            role="candidate",
            model_path=model_path,
            seed=42,
            command=None,
            status="available",
        )
    )

    assert provenance["train_images"] == ["/data/train_a.jpg"]
    assert provenance["valid_images"] == ["/data/valid_a.jpg"]
    assert provenance["split_sidecar_path"] == str(tmp_path / "candidate_splits.json")
    assert provenance["crop_size"] == 512


def test_compare_glomeruli_candidates_writes_reports_for_tied_candidates(tmp_path, monkeypatch):
    training_root = tmp_path / "raw_data" / "project" / "training_pairs"
    _make_training_root(training_root)
    output_root = tmp_path / "comparison"
    model_root = tmp_path / "models" / "segmentation" / "glomeruli"
    transfer_model = tmp_path / "transfer.pkl"
    scratch_model = tmp_path / "scratch.pkl"
    transfer_model.write_text("transfer")
    scratch_model.write_text("scratch")
    valid_images = sorted(str(path) for path in (training_root / "images").rglob("*.jpg"))
    source_size_summary = {
        "source_image_size_summary": {
            "count": len(valid_images),
            "width": {"count": len(valid_images), "min": 32.0, "median": 32.0, "p75": 32.0, "p95": 32.0, "max": 32.0},
            "height": {"count": len(valid_images), "min": 32.0, "median": 32.0, "p75": 32.0, "p95": 32.0, "max": 32.0},
        },
        "source_mask_size_summary": {
            "count": len(valid_images),
            "width": {"count": len(valid_images), "min": 32.0, "median": 32.0, "p75": 32.0, "p95": 32.0, "max": 32.0},
            "height": {"count": len(valid_images), "min": 32.0, "median": 32.0, "p75": 32.0, "p95": 32.0, "max": 32.0},
        },
    }
    transfer_base_metadata = {
        "mitochondria_training_scope": "all_available_pretraining",
        "mitochondria_inference_claim_status": "not_applicable_for_inference_claim",
        "mitochondria_physical_training_image_count": 165,
        "mitochondria_physical_testing_image_count": 0,
        "actual_pretraining_image_paths": ["mito_train.tif"],
        "actual_pretraining_mask_paths": ["mito_train_mask.tif"],
        "resize_policy": {"crop_size": 256, "output_size": 256},
    }
    split_metadata = {
        "training_mode": "dynamic_full_image_patching",
        "artifact_status": "supported_runtime",
        "scientific_promotion_status": "not_evaluated",
        "data_root": str(training_root),
        "model_path": "",
        "command": "unit test",
        "code": {"commit": "test"},
        "package_versions": {"torch": "test"},
        "train_images": [str(tmp_path / "held_out_subject_guard" / "train.jpg")],
        "valid_images": valid_images,
        **source_size_summary,
        "transfer_base_artifact_path": str(tmp_path / "mitochondria.pkl"),
        "transfer_base_metadata": transfer_base_metadata,
        "transfer_base_mitochondria_training_scope": "all_available_pretraining",
    }
    (tmp_path / "transfer_run_metadata.json").write_text(
        json.dumps({**split_metadata, "model_path": str(transfer_model)}),
        encoding="utf-8",
    )
    (tmp_path / "scratch_run_metadata.json").write_text(
        json.dumps({**split_metadata, "model_path": str(scratch_model), "transfer_base_artifact_path": None}),
        encoding="utf-8",
    )
    trace_header = (
        "candidate_family,panel_id,artifact_path,image_path,mask_path,crop_box,resize_policy,"
        "threshold,prediction_tensor_shape,overlay_path,traceable,root_causes,remediation_path\n"
    )
    (tmp_path / "transfer_validation_prediction_trace.csv").write_text(
        trace_header
        + f"mitochondria_transfer,transfer-validation-1,{transfer_model},{valid_images[0]},mask.png,"
        + '"not_recorded_validation_datablock_crop","{}",0.01,"[2,32,32]",panel.png,False,,\n',
        encoding="utf-8",
    )
    (tmp_path / "scratch_validation_prediction_trace.csv").write_text(
        trace_header
        + f"scratch,scratch-validation-1,{scratch_model},{valid_images[0]},mask.png,"
        + '"not_recorded_validation_datablock_crop","{}",0.01,"[2,32,32]",panel.png,False,,\n',
        encoding="utf-8",
    )

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
            "provenance": {
                "seed": 42,
                "training_mode": "dynamic_full_image_patching",
                "train_images": [str(tmp_path / "held_out_subject_guard" / "train.jpg")],
                "valid_images": valid_images,
                **source_size_summary,
                "transfer_base_artifact_path": str(tmp_path / "mitochondria.pkl")
                if runtime.family == "transfer"
                else None,
                "transfer_base_metadata": transfer_base_metadata if runtime.family == "transfer" else {},
                "transfer_base_mitochondria_training_scope": (
                    "all_available_pretraining" if runtime.family == "transfer" else None
                ),
            },
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
                    "cohort_id": manifest[0].get("cohort_id"),
                    "lane_assignment": manifest[0].get("lane_assignment"),
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
            model_dir=str(model_root),
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
            negative_crop_manifest=None,
            negative_crop_sampler_weight=0.0,
            augmentation_variant="fastai_default",
            examples_per_category=1,
            transfer_model_name="transfer_candidate",
            scratch_model_name="scratch_candidate",
            run_id="unit_test_run",
        )
    )

    run_output = output_root / "unit_test_run"
    assert payload["decision"]["decision_state"] == "insufficient_evidence"
    assert payload["run_id"] == "unit_test_run"
    assert payload["output_dir"] == str(run_output)
    assert payload["model_dir"] == str(model_root)
    assert (run_output / "deterministic_validation_manifest.json").exists()
    assert (run_output / "candidate_summary.csv").exists()
    assert (run_output / "candidate_predictions.csv").exists()
    assert (run_output / "metric_by_category.csv").exists()
    assert (run_output / "prediction_shape_audit.csv").exists()
    assert (run_output / "resize_policy_audit.csv").exists()
    assert (run_output / "failure_reproduction_audit.csv").exists()
    assert (run_output / "documentation_claim_audit.md").exists()
    assert (run_output / "promotion_report.json").exists()
    assert (run_output / "promotion_report.md").exists()
    assert (run_output / "promotion_report.html").exists()
    report = json.loads((run_output / "promotion_report.json").read_text())
    assert report["decision"]["promotion_evidence_status"] == "insufficient_evidence_for_promotion"
    transfer_summary = next(row for row in report["candidate_summaries"] if row["family"] == "transfer")
    assert transfer_summary["transfer_base_report"]["mitochondria_inference_claim_status"] == "not_applicable_for_inference_claim"
    assert transfer_summary["transfer_base_report"]["physical_training_image_count"] == 165
    metric_header = (run_output / "metric_by_category.csv").read_text().splitlines()[0]
    assert "cohort_id" in metric_header
    assert "lane_assignment" in metric_header
    resize_header = (run_output / "resize_policy_audit.csv").read_text().splitlines()[0]
    assert "resize_sensitivity_infeasibility_reason" in resize_header
    assert "heldout_dice" in resize_header
    failure_text = (run_output / "failure_reproduction_audit.csv").read_text()
    assert "trace_source" in failure_text
    assert "transfer-validation-1" in failure_text
    candidate_summary = (run_output / "candidate_summary.csv").read_text()
    assert "negative_crop_supervision_status" in candidate_summary
    assert "augmentation_policy" in candidate_summary
    markdown_report = (run_output / "promotion_report.md").read_text()
    assert "Negative crop supervision status" in markdown_report
    assert "Augmentation policy" in markdown_report
    html_report = (run_output / "promotion_report.html").read_text()
    assert "Manifest Coverage" in html_report
    assert "Promotion evidence" in html_report
    assert "inference_claim=not_applicable_for_inference_claim" in html_report
    assert "panel order: raw | truth overlay | prediction overlay" in html_report


def test_candidate_comparison_writes_shared_split_before_fresh_training(tmp_path):
    training_root = tmp_path / "raw_data" / "project" / "training_pairs"
    _make_training_root(training_root)
    output_root = tmp_path / "comparison" / "unit"

    split_path = _write_shared_training_split(training_root, output_root, seed=42)
    split = json.loads(split_path.read_text())

    assert split_path.exists()
    assert split["splitter_name"] == "explicit_shared_subject_split"
    assert split["train_images"]
    assert split["valid_images"]
    assert split["train_subjects"]
    assert split["valid_subjects"]
    assert not (set(split["train_images"]) & set(split["valid_images"]))
    assert not (set(split["train_subjects"]) & set(split["valid_subjects"]))

    command = _candidate_command(
        data_dir=training_root,
        model_dir=tmp_path / "models",
        model_name="candidate",
        epochs=1,
        learning_rate=1e-3,
        image_size=256,
        crop_size=32,
        batch_size=1,
        loss_name=None,
        seed=42,
        from_scratch=True,
        split_manifest_path=split_path,
        device="mps",
    )

    assert "--split-manifest" in command
    assert str(split_path) in command
    assert command[command.index("--device") + 1] == "mps"


def test_shared_split_resolves_flat_image_subjects(tmp_path):
    training_root = tmp_path / "raw_data" / "cohorts" / "lauren_preeclampsia"
    images = training_root / "images"
    masks = training_root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for stem in ["t19_image0", "t19_image1", "t20_image0", "t20_image1"]:
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(images / f"{stem}.jpg")
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(masks / f"{stem}_mask.jpg")

    split_path = _write_shared_training_split(training_root, tmp_path / "comparison", seed=42)
    split = json.loads(split_path.read_text())

    assert not (set(split["train_subjects"]) & set(split["valid_subjects"]))
    for image_path in split["train_images"]:
        assert not any(f"/{subject}_" in image_path for subject in split["valid_subjects"])
    for image_path in split["valid_images"]:
        assert any(f"/{subject}_" in image_path for subject in split["valid_subjects"])


def test_manifest_context_is_propagated_to_deterministic_rows(tmp_path):
    runtime_root = tmp_path / "runtime"
    cohorts_root = runtime_root / "raw_data" / "cohorts"
    image_dir = cohorts_root / "vegfri_dox" / "images"
    mask_dir = cohorts_root / "vegfri_dox" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image_path = image_dir / "sample.jpg"
    mask_path = mask_dir / "sample_mask.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(mask_path)

    import pandas as pd

    pd.DataFrame(
        [
            {
                "cohort_id": "vegfri_dox",
                "lane_assignment": "manual_mask_external",
                "admission_status": "admitted",
                "image_path": "raw_data/cohorts/vegfri_dox/images/sample.jpg",
                "mask_path": "raw_data/cohorts/vegfri_dox/masks/sample_mask.png",
            }
        ]
    ).to_csv(cohorts_root / "manifest.csv", index=False)

    annotated = _annotate_manifest_with_context(
        [
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "category": "background",
                "crop_box": [0, 0, 32, 32],
            }
        ],
        cohorts_root,
    )

    assert annotated[0]["cohort_id"] == "vegfri_dox"
    assert annotated[0]["lane_assignment"] == "manual_mask_external"



def test_candidate_comparison_uses_manifest_backed_cohort_registry_masks(tmp_path):
    runtime_root = tmp_path / "runtime"
    cohorts_root = runtime_root / "raw_data" / "cohorts"
    rows = []
    samples = [
        (
            "lauren_preeclampsia",
            "manual_mask_core",
            "lauren_empty",
            np.zeros((32, 32), dtype=np.uint8),
        ),
        (
            "vegfri_dox",
            "manual_mask_external",
            "dox_boundary",
            np.pad(np.ones((8, 8), dtype=np.uint8), ((0, 24), (0, 24))),
        ),
        (
            "vegfri_dox",
            "manual_mask_external",
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
        "lauren_empty_mask.png",
        "dox_boundary_mask.png",
        "dox_positive_mask.png",
    }


def test_compare_parser_defaults_to_runtime_output_root(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime" / "endotheliosis_quantifier"
    runtime_root.mkdir(parents=True)
    monkeypatch.setenv("EQ_RUNTIME_ROOT", str(runtime_root))

    parser = build_arg_parser()
    args = parser.parse_args(["--data-dir", str(tmp_path / "raw_data" / "project" / "training_pairs")])

    assert args.output_dir == str(runtime_root / "output" / "segmentation_evaluation" / "glomeruli_candidate_comparison")
    assert args.model_dir == str(runtime_root / "models" / "segmentation" / "glomeruli")


def test_predict_crop_uses_deterministic_preprocessing_and_underconfident_threshold(monkeypatch):
    class FakeCore:
        def resize_prediction_to_match(self, pred_mask, target_shape):
            return pred_mask

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.seen_shape = None

        def forward(self, x):
            self.seen_shape = tuple(x.shape)
            fg = torch.full((1, 2, 2), -4.0, dtype=torch.float32, device=x.device)
            bg = torch.zeros((1, 2, 2), dtype=torch.float32, device=x.device)
            return torch.stack([bg, fg], dim=1)

    class FakeLearn:
        def __init__(self):
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
