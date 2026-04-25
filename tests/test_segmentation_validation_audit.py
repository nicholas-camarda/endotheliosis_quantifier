import csv
from pathlib import Path

import numpy as np
from PIL import Image

from eq.training.segmentation_validation_audit import (
    MITO_INFERENCE_NOT_APPLICABLE,
    MITO_SCOPE_ALL_AVAILABLE,
    PROMOTION_AUDIT_MISSING,
    PROMOTION_ELIGIBLE,
    PROMOTION_NOT_ELIGIBLE,
    RUNTIME_USE_AVAILABLE,
    ValidationAuditReport,
    aggregate_metric_by_category,
    audit_datablock_sampling,
    audit_dynamic_patching_datablock,
    audit_category_gates,
    audit_manifest_rows,
    audit_paired_root_contract,
    audit_prediction_shapes,
    audit_preprocessing_parity,
    audit_resize_policy_parity,
    audit_split_overlap,
    audit_transform_alignment,
    build_glomeruli_training_provenance,
    build_mitochondria_training_provenance,
    check_binary_mask_preservation,
    classify_artifact_status,
    classify_root_causes,
    documentation_claim_audit,
    failure_reproduction_row,
    resize_policy_record,
    validation_prediction_panel_trace_rows,
    validate_mitochondria_scope_for_claim,
    write_csv_rows,
    write_documentation_claim_audit,
)


def test_paired_root_contract_fails_closed_for_unpaired_files(tmp_path: Path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(images / "a.jpg")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(masks / "b_mask.png")

    audit = audit_paired_root_contract(tmp_path)

    assert audit["ok"] is False
    assert audit["reason"] == "unpaired_images_or_masks"
    assert audit["unpaired_images"] == [str(images / "a.jpg")]
    assert audit["unpaired_masks"] == [str(masks / "b_mask.png")]


def test_manifest_rows_report_missing_pairs_and_lane_counts(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    image_path = runtime_root / "raw_data" / "cohorts" / "c1" / "images" / "s1.jpg"
    mask_path = runtime_root / "raw_data" / "cohorts" / "c1" / "masks" / "s1_mask.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_path)

    audit = audit_manifest_rows(
        [
            {
                "cohort_id": "c1",
                "lane_assignment": "manual_mask_core",
                "image_path": "raw_data/cohorts/c1/images/s1.jpg",
                "mask_path": "raw_data/cohorts/c1/masks/s1_mask.png",
            },
            {
                "cohort_id": "c2",
                "lane_assignment": "manual_mask_external",
                "image_path": "raw_data/cohorts/c2/images/missing.jpg",
                "mask_path": "raw_data/cohorts/c2/masks/missing_mask.png",
            },
        ],
        runtime_root=runtime_root,
    )

    assert audit["ok"] is False
    assert audit["missing_pair_count"] == 1
    assert audit["by_cohort_id"] == {"c1": 1, "c2": 1}
    assert audit["by_lane_assignment"]["manual_mask_core"] == 1


def test_split_overlap_blocks_promotion_but_keeps_runtime_use_available():
    manifest = [
        {"image_path": "/runtime/raw_data/cohorts/c1/images/s1/a.jpg"},
        {"image_path": "/runtime/raw_data/cohorts/c2/images/s2/b.jpg"},
    ]

    audit = audit_split_overlap(
        train_images=["/runtime/raw_data/cohorts/c1/images/s1/a.jpg"],
        valid_images=["/runtime/raw_data/cohorts/c2/images/s2/b.jpg"],
        promotion_manifest=manifest,
    )

    assert audit["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE
    assert audit["runtime_use_status"] == RUNTIME_USE_AVAILABLE
    assert audit["train_image_overlap_count"] == 1


def test_split_overlap_detects_flat_filename_subject_overlap():
    manifest = [
        {
            "image_path": "/runtime/raw_data/cohorts/lauren_preeclampsia/images/t19_image1.jpg",
            "mask_path": "/runtime/raw_data/cohorts/lauren_preeclampsia/masks/t19_image1_mask.jpg",
        }
    ]

    audit = audit_split_overlap(
        train_images=["/runtime/raw_data/cohorts/lauren_preeclampsia/images/t19_image0.jpg"],
        valid_images=["/runtime/raw_data/cohorts/lauren_preeclampsia/images/t19_image1.jpg"],
        promotion_manifest=manifest,
    )

    assert audit["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE
    assert audit["reason"] == "train_evaluation_subject_overlap"
    assert audit["subject_overlap_count"] == 1
    assert audit["overlapping_train_subjects"] == ["t19"]


def test_missing_split_provenance_is_audit_missing_not_unusable():
    audit = audit_split_overlap(train_images=[], valid_images=[], promotion_manifest=[])

    assert audit["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING
    assert audit["runtime_use_status"] == RUNTIME_USE_AVAILABLE


def test_artifact_status_uses_two_axes_for_missing_transfer_provenance():
    status = classify_artifact_status(
        {"train_images": ["/a.jpg"], "valid_images": ["/b.jpg"]},
        loadable=True,
        requires_transfer_base=True,
    )

    assert status["runtime_use_status"] == RUNTIME_USE_AVAILABLE
    assert status["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING
    assert "missing_transfer_base_provenance" in status["reasons"]


def test_incomplete_promotion_contract_fields_are_audit_missing():
    status = classify_artifact_status(
        {
            "artifact_status": "supported_runtime",
            "train_images": ["/runtime/raw_data/cohorts/c1/images/a.jpg"],
            "valid_images": ["/runtime/raw_data/cohorts/c1/images/b.jpg"],
            "transfer_base_artifact_path": "/runtime/models/mito.pkl",
            "transfer_base_mitochondria_training_scope": "all_available_pretraining",
        },
        loadable=True,
        requires_transfer_base=True,
    )

    assert status["runtime_use_status"] == RUNTIME_USE_AVAILABLE
    assert status["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING
    assert "missing_source_image_size_summary" in status["reasons"]
    assert "missing_source_mask_size_summary" in status["reasons"]
    assert "missing_transfer_base_inference_claim_status" in status["reasons"]


def test_resize_policy_records_and_detects_mismatch():
    training = resize_policy_record(crop_size=512, output_size=256)
    evaluation = resize_policy_record(
        crop_size=512,
        output_size=512,
        threshold_resize_order="threshold_then_resize_binary_mask",
    )

    audit = audit_resize_policy_parity(training, evaluation)

    assert training["crop_to_output_resize_ratio"] == 2.0
    assert audit["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE
    assert {row["field"] for row in audit["mismatches"]} >= {
        "output_size",
        "threshold_resize_order",
    }


def test_preprocessing_parity_fails_for_bespoke_evaluation_path():
    audit = audit_preprocessing_parity(
        learner_consistent_preprocessing=False,
        supported_threshold_semantics=True,
    )

    assert audit["ok"] is False
    assert audit["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE
    assert "evaluation_preprocessing_not_learner_consistent" in audit["reasons"]


def test_binary_mask_preservation_catches_nonbinary_resize_values():
    before = np.zeros((8, 8), dtype=np.uint8)
    before[2:6, 2:6] = 1
    after = before.copy()
    after[3, 3] = 2

    audit = check_binary_mask_preservation(before, after)

    assert audit["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE
    assert 2 in audit["after_values"]


def test_transform_alignment_uses_marker_bounding_boxes():
    image_marker = np.zeros((12, 12), dtype=np.uint8)
    mask_marker = np.zeros((12, 12), dtype=np.uint8)
    image_marker[3:6, 4:7] = 255
    mask_marker[3:6, 4:7] = 1

    aligned = audit_transform_alignment(image_marker, mask_marker)
    mask_marker_shifted = np.roll(mask_marker, 2, axis=0)
    shifted = audit_transform_alignment(image_marker, mask_marker_shifted)

    assert aligned["ok"] is True
    assert shifted["ok"] is False
    assert shifted["reason"] == "transform_alignment_error"


class _FakeLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


class _FakeDls:
    def __init__(self):
        train_masks = np.zeros((2, 4, 4), dtype=np.uint8)
        train_masks[0, 1:3, 1:3] = 1
        valid_masks = np.ones((2, 4, 4), dtype=np.uint8)
        self.train = _FakeLoader([(np.zeros((2, 3, 4, 4)), train_masks)])
        self.valid = _FakeLoader([(np.zeros((2, 3, 4, 4)), valid_masks)])


def test_datablock_sampling_audit_writes_crop_foreground_distribution(tmp_path: Path):
    output_csv = tmp_path / "datablock_sampling_audit.csv"

    audit = audit_datablock_sampling(
        _FakeDls(),
        crop_size=512,
        output_size=256,
        min_pos_pixels=4,
        output_csv=output_csv,
    )

    assert output_csv.exists()
    assert audit["by_split"]["train"]["count"] == 2
    assert audit["foreground_heavy_validation_panel"] is True
    rows = list(csv.DictReader(output_csv.open()))
    assert rows[0]["crop_size"] == "512"


def test_dynamic_patching_datablock_audit_uses_supported_loader(monkeypatch, tmp_path: Path):
    calls = []

    def fake_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeDls()

    monkeypatch.setattr(
        "eq.data_management.datablock_loader.build_segmentation_dls_dynamic_patching",
        fake_builder,
    )

    audit = audit_dynamic_patching_datablock(
        tmp_path,
        crop_size=512,
        output_size=256,
        positive_focus_p=0.6,
        min_pos_pixels=64,
        pos_crop_attempts=10,
    )

    assert calls
    assert calls[0][1]["stage"] == "glomeruli"
    assert audit["by_split"]["valid"]["count"] == 2


def test_prediction_shape_audit_blocks_background_false_positive_and_overcoverage():
    rows = [
        {
            "family": "transfer",
            "category": "background",
            "truth_foreground_fraction": 0.0,
            "prediction_foreground_fraction": 0.15,
            "dice": 0.0,
            "jaccard": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        },
        {
            "family": "transfer",
            "category": "positive",
            "truth_foreground_fraction": 0.2,
            "prediction_foreground_fraction": 0.7,
            "dice": 0.2,
            "jaccard": 0.1,
            "precision": 0.1,
            "recall": 0.8,
        },
    ]

    audit = audit_prediction_shapes(rows)

    reasons = audit["family_status"]["transfer"]["reasons"]
    assert audit["blocked"] is True
    assert "background_false_positive_foreground_excess" in reasons
    assert "positive_or_boundary_overcoverage" in reasons


def test_category_gate_audit_does_not_fail_background_on_empty_mask_dice():
    rows = [
        {
            "family": "transfer",
            "category": "background",
            "manifest_index": 0,
            "truth_foreground_fraction": 0.0,
            "prediction_foreground_fraction": 0.0004,
            "pixel_accuracy": 0.9996,
            "dice": 0.0,
            "jaccard": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 0.25,
            "threshold_policy_status": "validation_derived_threshold",
        }
    ]

    audit = audit_category_gates(rows)
    shape_audit = audit_prediction_shapes(rows)

    assert audit["blocked"] is False
    assert audit["family_status"]["transfer"]["reasons"] == []
    assert shape_audit["blocked"] is False
    assert shape_audit["family_status"]["transfer"]["reasons"] == []


def test_category_gate_audit_blocks_positive_overlap_failures():
    rows = [
        {
            "family": "scratch",
            "category": "positive",
            "manifest_index": 1,
            "truth_foreground_fraction": 0.5,
            "prediction_foreground_fraction": 0.1,
            "pixel_accuracy": 0.6,
            "dice": 0.2,
            "jaccard": 0.1,
            "precision": 0.3,
            "recall": 0.2,
            "threshold": 0.25,
            "threshold_policy_status": "validation_derived_threshold",
        }
    ]

    audit = audit_category_gates(rows)
    reasons = audit["family_status"]["scratch"]["reasons"]

    assert audit["blocked"] is True
    assert "low_foreground_dice" in reasons
    assert "low_foreground_jaccard" in reasons
    assert "low_foreground_recall" in reasons
    assert "positive_or_boundary_undercoverage" in reasons


def test_metric_by_category_aggregates_by_family_cohort_and_lane():
    rows = [
        {
            "family": "scratch",
            "category": "positive",
            "cohort_id": "c1",
            "lane_assignment": "manual_mask_core",
            "dice": 0.4,
            "jaccard": 0.25,
            "precision": 0.5,
            "recall": 0.6,
            "truth_foreground_fraction": 0.2,
            "prediction_foreground_fraction": 0.3,
        },
        {
            "family": "scratch",
            "category": "positive",
            "cohort_id": "c1",
            "lane_assignment": "manual_mask_core",
            "dice": 0.6,
            "jaccard": 0.35,
            "precision": 0.7,
            "recall": 0.8,
            "truth_foreground_fraction": 0.4,
            "prediction_foreground_fraction": 0.5,
        },
    ]

    result = aggregate_metric_by_category(rows)

    assert len(result) == 1
    assert result[0]["dice"] == 0.5
    assert result[0]["cohort_id"] == "c1"


def test_failure_reproduction_row_requires_known_root_cause_class():
    row = failure_reproduction_row(
        candidate_family="transfer",
        panel_id="panel-1",
        artifact_path="/models/transfer.pkl",
        image_path="/images/a.jpg",
        mask_path="/masks/a_mask.png",
        crop_box=[0, 0, 512, 512],
        resize_policy=resize_policy_record(crop_size=512, output_size=256),
        threshold=0.01,
        prediction_tensor_shape=[1, 2, 256, 256],
        overlay_path="/report/panel.png",
        root_causes=["resize_policy_artifact"],
    )

    assert row["traceable"] is True
    assert row["root_causes"] == "resize_policy_artifact"
    assert "512" in row["resize_policy"]


def test_validation_prediction_panel_trace_rows_mark_missing_crop_untraceable():
    rows = validation_prediction_panel_trace_rows(
        model_folder_name="model_a",
        candidate_family="mitochondria_transfer",
        artifact_path="/models/model_a.pkl",
        image_paths=["/images/a.jpg"],
        mask_paths=["/masks/a_mask.png"],
        overlay_path="/models/model_a_validation_predictions.png",
        prediction_tensor_shapes=[[2, 256, 256]],
        resize_policy=resize_policy_record(crop_size=512, output_size=256),
        threshold=None,
    )

    assert len(rows) == 1
    assert rows[0]["panel_id"] == "model_a-validation-1"
    assert rows[0]["prediction_tensor_shape"] == "[2, 256, 256]"
    assert rows[0]["crop_box"] == "not_recorded_validation_datablock_crop"
    assert rows[0]["traceable"] is False


def test_root_cause_classifier_maps_defect_to_remediation():
    result = classify_root_causes(
        {
            "resize_policy_artifact": True,
            "negative_background_supervision_missing": True,
        }
    )

    assert result["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE
    assert result["root_causes"] == [
        "resize_policy_artifact",
        "negative_background_supervision_missing",
    ]
    assert result["remediation_path"] == "fix_code_or_evaluation_then_regenerate_evidence"


def test_mitochondria_all_available_pretraining_disables_inference_claim(tmp_path: Path):
    training = tmp_path / "mitochondria_data" / "training"
    testing = tmp_path / "mitochondria_data" / "testing"
    for root in (training, testing):
        (root / "images").mkdir(parents=True)
    train_item = training / "images" / "train.tif"
    test_item = testing / "images" / "test.tif"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(train_item)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(test_item)

    provenance = build_mitochondria_training_provenance(
        data_root=training,
        train_items=[train_item, test_item],
        valid_items=[],
        image_size=256,
        crop_size=256,
        output_size=256,
    )
    claim = validate_mitochondria_scope_for_claim(
        provenance,
        claim_type="heldout_mitochondria_performance",
    )

    assert provenance["mitochondria_training_scope"] == MITO_SCOPE_ALL_AVAILABLE
    assert provenance["mitochondria_inference_claim_status"] == MITO_INFERENCE_NOT_APPLICABLE
    assert claim["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE


def test_glomeruli_training_provenance_records_split_and_resize_contract():
    provenance = build_glomeruli_training_provenance(
        data_root="/runtime/raw_data/cohorts",
        train_items=["/runtime/raw_data/cohorts/c1/images/a.jpg"],
        valid_items=["/runtime/raw_data/cohorts/c2/images/b.jpg"],
        seed=42,
        split_seed=42,
        crop_size=512,
        output_size=256,
        candidate_family="mitochondria_transfer",
        training_mode="dynamic_full_image_patching",
        transfer_base_artifact_path="/models/mito.pkl",
        transfer_base_metadata={"mitochondria_training_scope": MITO_SCOPE_ALL_AVAILABLE},
        positive_focus_p=0.6,
        min_pos_pixels=64,
        pos_crop_attempts=10,
    )

    assert provenance["splitter_name"] == "RandomSplitter"
    assert provenance["train_images"] == ["/runtime/raw_data/cohorts/c1/images/a.jpg"]
    assert provenance["transfer_base_mitochondria_training_scope"] == MITO_SCOPE_ALL_AVAILABLE
    assert provenance["resize_policy"]["crop_to_output_resize_ratio"] == 2.0


def test_documentation_claim_audit_blocks_non_eligible_metric_claim(tmp_path: Path):
    audit = documentation_claim_audit(
        {
            "README.md": "| Candidate | Dice | Jaccard |\n| transfer | 0.95 | 0.91 |",
            "docs/ONBOARDING_GUIDE.md": "This is internal deterministic validation.",
        },
        cited_report_status=PROMOTION_NOT_ELIGIBLE,
        cited_report_path="/runtime/output/promotion_report.md",
    )
    output_path = tmp_path / "documentation_claim_audit.md"
    write_documentation_claim_audit(audit, output_path)

    assert audit["promotion_evidence_status"] == PROMOTION_NOT_ELIGIBLE
    assert output_path.exists()
    assert "metric_claim_cites_non_promotion_eligible_report" in output_path.read_text()


def test_readme_and_onboarding_do_not_publish_audit_missing_segmentation_metrics():
    docs = {
        "README.md": Path("README.md").read_text(encoding="utf-8"),
        "docs/ONBOARDING_GUIDE.md": Path("docs/ONBOARDING_GUIDE.md").read_text(encoding="utf-8"),
    }

    audit = documentation_claim_audit(
        docs,
        cited_report_status=PROMOTION_AUDIT_MISSING,
        cited_report_path="/runtime/output/promotion_report.md",
    )

    assert audit["blocked"] is False, audit["rows"]


def test_write_csv_rows_and_review_lane_report(tmp_path: Path):
    csv_path = tmp_path / "metric_by_category.csv"
    write_csv_rows([{"family": "transfer", "dice": 0.5}], csv_path)
    report = ValidationAuditReport(
        implementation_audit={"direct_evidence": "split checked"},
        statistical_validity={"direct_evidence": "held-out only"},
        scientific_interpretation={"inference": "internal validation only"},
        robustness_tests={"direct_evidence": "overcoverage gate"},
        documentation_consistency={"direct_evidence": "README audited"},
    ).to_markdown()

    assert csv_path.exists()
    assert "Implementation Audit" in report
    assert "Documentation Consistency" in report
    assert PROMOTION_ELIGIBLE == "promotion_eligible"
