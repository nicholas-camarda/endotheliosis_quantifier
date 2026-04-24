import inspect
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from eq.data_management import datablock_loader
from eq.data_management.datablock_loader import (
    build_segmentation_dls_dynamic_patching,
    get_items_full_images,
    validate_supported_segmentation_training_root,
)
from eq.data_management.standard_getters import get_y_full
from eq.training import transfer_learning
from eq.training.promotion_gates import (
    audit_binary_masks,
    audit_manifest_crops,
    deterministic_validation_manifest,
    evaluate_glomeruli_promotion_candidate,
    evaluate_prediction_degeneracy,
    evaluate_prediction_review,
    trivial_baseline_metrics,
)


def _make_full_image_root(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)


def test_training_root_validator_accepts_full_image_root(tmp_path: Path):
    _make_full_image_root(tmp_path)

    assert validate_supported_segmentation_training_root(tmp_path) == tmp_path


def test_training_root_validator_rejects_static_patch_root(tmp_path: Path):
    (tmp_path / "image_patches").mkdir()
    (tmp_path / "mask_patches").mkdir()

    with pytest.raises(ValueError, match="Unsupported static patch training root"):
        validate_supported_segmentation_training_root(tmp_path, stage="mitochondria")


def test_training_root_validator_rejects_mixed_static_and_full_image_root(tmp_path: Path):
    _make_full_image_root(tmp_path)
    (tmp_path / "image_patches").mkdir()
    (tmp_path / "mask_patches").mkdir()

    with pytest.raises(ValueError, match="Unsupported static patch training root"):
        validate_supported_segmentation_training_root(tmp_path, stage="mitochondria")


def test_training_root_validator_rejects_missing_full_image_dirs(tmp_path: Path):
    (tmp_path / "images").mkdir()

    with pytest.raises(ValueError, match="Missing required full-image directories"):
        validate_supported_segmentation_training_root(tmp_path, stage="glomeruli")


def test_glomeruli_training_root_validator_requires_supported_raw_data_project_or_cohort_root(tmp_path: Path):
    invalid_root = tmp_path / "clean_backup"
    _make_full_image_root(invalid_root)

    with pytest.raises(ValueError, match="raw_data"):
        validate_supported_segmentation_training_root(invalid_root, stage="glomeruli")

    backup_root = tmp_path / "raw_data" / "source_imports" / "clean_backup"
    _make_full_image_root(backup_root)

    with pytest.raises(ValueError, match="clean_backup"):
        validate_supported_segmentation_training_root(backup_root, stage="glomeruli")

    project_data_root = tmp_path / "raw_data" / "cohorts" / "lauren_preeclampsia"
    _make_full_image_root(project_data_root)

    assert (
        validate_supported_segmentation_training_root(project_data_root, stage="glomeruli")
        == project_data_root
    )

    training_pairs_root = tmp_path / "raw_data" / "project" / "training_pairs"
    _make_full_image_root(training_pairs_root)

    assert (
        validate_supported_segmentation_training_root(training_pairs_root, stage="glomeruli")
        == training_pairs_root
    )

    cohort_root = tmp_path / "raw_data" / "cohorts" / "vegfri_dox"
    _make_full_image_root(cohort_root)

    assert (
        validate_supported_segmentation_training_root(cohort_root, stage="glomeruli")
        == cohort_root
    )

    cohort_registry_root = tmp_path / "raw_data" / "cohorts"
    cohort_registry_root.mkdir(parents=True, exist_ok=True)
    (cohort_registry_root / "manifest.csv").write_text("cohort_id\n", encoding="utf-8")

    assert (
        validate_supported_segmentation_training_root(cohort_registry_root, stage="glomeruli")
        == cohort_registry_root
    )


def test_dynamic_dls_uses_selected_training_root_only(tmp_path: Path, monkeypatch):
    training_root = tmp_path / "mitochondria_data" / "training"
    testing_root = tmp_path / "mitochondria_data" / "testing"
    _make_full_image_root(training_root)
    _make_full_image_root(testing_root)
    seen_paths = []

    def fake_get_items(path):
        seen_paths.append(("items", Path(path)))
        return [Path(path) / "images" / "sample.png"]

    class FakeDataBlock:
        def dataloaders(self, path, **kwargs):
            seen_paths.append(("dataloaders", Path(path)))
            return object()

    monkeypatch.setattr(datablock_loader, "get_items_full_images", fake_get_items)
    monkeypatch.setattr(
        datablock_loader,
        "build_segmentation_datablock_dynamic_patching",
        lambda **kwargs: FakeDataBlock(),
    )

    result = build_segmentation_dls_dynamic_patching(training_root, bs=1, num_workers=0)

    assert result is not None
    assert seen_paths == [("items", training_root), ("dataloaders", training_root)]
    assert testing_root not in [path for _, path in seen_paths]


def test_cohort_training_root_uses_manifest_admitted_rows_only(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    cohort_root = runtime_root / "raw_data" / "cohorts" / "vegfri_dox"
    image_dir = cohort_root / "images" / "M1"
    mask_dir = cohort_root / "masks" / "M1"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for stem in ("approved", "unresolved"):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / f"{stem}.jpg")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(mask_dir / f"{stem}_mask.png")

    manifest = pd.DataFrame(
        [
            {
                "cohort_id": "vegfri_dox",
                "lane_assignment": "manual_mask_external",
                "admission_status": "admitted",
                "image_path": "raw_data/cohorts/vegfri_dox/images/M1/approved.jpg",
                "mask_path": "raw_data/cohorts/vegfri_dox/masks/M1/approved_mask.png",
            },
            {
                "cohort_id": "vegfri_dox",
                "lane_assignment": "manual_mask_external",
                "admission_status": "unresolved",
                "image_path": "raw_data/cohorts/vegfri_dox/images/M1/unresolved.jpg",
                "mask_path": "raw_data/cohorts/vegfri_dox/masks/M1/unresolved_mask.png",
            },
        ]
    )
    manifest.to_csv(runtime_root / "raw_data" / "cohorts" / "manifest.csv", index=False)

    items = get_items_full_images(cohort_root)

    assert items == [image_dir / "approved.jpg"]


def test_cohort_registry_training_root_uses_all_admitted_masked_rows(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    cohorts_root = runtime_root / "raw_data" / "cohorts"
    rows = []
    for cohort_id, lane, stem in [
        ("lauren_preeclampsia", "manual_mask_core", "lauren"),
        ("vegfri_dox", "manual_mask_external", "dox"),
        ("vegfri_mr", "mr_concordance_only", "mr"),
    ]:
        image_dir = cohorts_root / cohort_id / "images"
        mask_dir = cohorts_root / cohort_id / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / f"{stem}.jpg")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(mask_dir / f"{stem}_mask.png")
        rows.append(
            {
                "cohort_id": cohort_id,
                "lane_assignment": lane,
                "admission_status": "admitted" if cohort_id != "vegfri_mr" else "evaluation_only",
                "image_path": f"raw_data/cohorts/{cohort_id}/images/{stem}.jpg",
                "mask_path": f"raw_data/cohorts/{cohort_id}/masks/{stem}_mask.png",
            }
        )
    rows.append(
        {
            "cohort_id": "vegfri_dox",
            "lane_assignment": "manual_mask_external",
            "admission_status": "unresolved",
            "image_path": "raw_data/cohorts/vegfri_dox/images/unresolved.jpg",
            "mask_path": "raw_data/cohorts/vegfri_dox/masks/unresolved_mask.png",
        }
    )
    pd.DataFrame(rows).to_csv(cohorts_root / "manifest.csv", index=False)

    assert validate_supported_segmentation_training_root(cohorts_root, stage="glomeruli") == cohorts_root
    items = get_items_full_images(cohorts_root)

    assert items == [
        cohorts_root / "lauren_preeclampsia/images/lauren.jpg",
        cohorts_root / "vegfri_dox/images/dox.jpg",
    ]


def test_dynamic_dls_fails_clearly_when_no_valid_pairs_exist(tmp_path: Path):
    _make_full_image_root(tmp_path)

    with pytest.raises(ValueError, match="No valid image-mask pairs found"):
        build_segmentation_dls_dynamic_patching(tmp_path, bs=1, num_workers=0)

    image_path = tmp_path / "images" / "orphan.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    with pytest.raises(ValueError, match="Unpaired full-image training root"):
        build_segmentation_dls_dynamic_patching(tmp_path, bs=1, num_workers=0)


def test_dynamic_dls_positive_focus_crops_pairs_and_preserves_image_intensity(tmp_path: Path):
    _make_full_image_root(tmp_path)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[..., 0] = 40
    image[..., 1] = 90
    image[..., 2] = 140
    image[80:110, 80:110, :] = [220, 180, 120]
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[80:110, 80:110] = 255
    Image.fromarray(image).save(tmp_path / "images" / "sample.jpg")
    Image.fromarray(mask).save(tmp_path / "masks" / "sample_mask.png")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dls = build_segmentation_dls_dynamic_patching(
        tmp_path,
        bs=1,
        num_workers=0,
        crop_size=64,
        output_size=32,
        positive_focus_p=1.0,
        min_pos_pixels=64,
        pos_crop_attempts=10,
    )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    x_batch, y_batch = dls.one_batch()
    foreground_fraction = float((y_batch > 0).float().mean())

    assert foreground_fraction > 0.15
    assert torch.unique(y_batch).tolist() == [0, 1]
    assert float(x_batch.max() - x_batch.min()) > 0.5


def test_training_cli_help_does_not_expose_static_patch_toggle():
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "src" / "eq" / "training" / "train_mitochondria.py",
        repo_root / "src" / "eq" / "training" / "train_glomeruli.py",
    ]

    for script in scripts:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        assert "--no-dynamic-patching" not in result.stdout
        assert "--use-dynamic-patching" not in result.stdout
        assert "images/" in result.stdout
        assert "masks/" in result.stdout


def test_glomeruli_training_cli_requires_explicit_family_selection(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "raw_data" / "project" / "training_pairs"
    _make_full_image_root(data_root)
    script = repo_root / "src" / "eq" / "training" / "train_glomeruli.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(data_root),
            "--model-dir",
            str(tmp_path / "models"),
            "--epochs",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "explicit family selection" in (result.stderr + result.stdout)


def test_package_cli_does_not_expose_unsupported_training_routes():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "eq", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "train-segmenter" not in result.stdout
    assert "seg " not in result.stdout
    assert "--no-dynamic-patching" not in result.stdout
    assert "--use-dynamic-patching" not in result.stdout


def test_glomeruli_config_and_docs_use_raw_data_training_contract():
    repo_root = Path(__file__).resolve().parents[1]
    files = [
        repo_root / "configs" / "glomeruli_finetuning_config.yaml",
        repo_root / "README.md",
        repo_root / "docs" / "ONBOARDING_GUIDE.md",
        repo_root / "docs" / "TECHNICAL_LAB_NOTEBOOK.md",
        repo_root / "docs" / "SEGMENTATION_ENGINEERING_GUIDE.md",
    ]

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "raw_data" in text
        assert "raw_data/cohorts" in text
        assert "derived_data/glomeruli_data/training_full_images" not in text


def test_get_y_full_does_not_match_by_substring_suffix(tmp_path: Path):
    _make_full_image_root(tmp_path)
    image_dir = tmp_path / "images" / "T30"
    mask_dir = tmp_path / "masks" / "T30"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "T30_Image9.jpg"
    mask_path = mask_dir / "T30_Image999_mask.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_path)

    with pytest.raises(FileNotFoundError):
        get_y_full(image_path)


def test_get_y_full_accepts_lucchi_groundtruth_naming(tmp_path: Path):
    _make_full_image_root(tmp_path)
    image_path = tmp_path / "images" / "training_86.tif"
    mask_path = tmp_path / "masks" / "training_groundtruth_86.tif"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_path)

    assert get_y_full(image_path) == mask_path


def test_transfer_learning_api_has_no_static_patch_switch():
    assert "use_dynamic_patching" not in inspect.signature(
        transfer_learning.load_model_for_transfer_learning
    ).parameters
    assert "use_dynamic_patching" not in inspect.signature(
        transfer_learning.transfer_learn_glomeruli
    ).parameters


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def test_glomeruli_mask_audit_reports_foreground_background_coverage(tmp_path: Path):
    empty = np.zeros((16, 16), dtype=np.uint8)
    boundary = np.zeros((16, 16), dtype=np.uint8)
    boundary[4:12, 4:12] = 1
    full = np.ones((16, 16), dtype=np.uint8)
    paths = []
    for name, mask in [("empty.png", empty), ("boundary.png", boundary), ("full.png", full)]:
        path = tmp_path / name
        _write_mask(path, mask)
        paths.append(path)

    audit = audit_binary_masks(paths)

    assert audit["count"] == 3
    assert audit["background_only_count"] == 1
    assert audit["full_foreground_count"] == 1
    assert audit["foreground_fraction"]["median"] == pytest.approx(0.25)


def test_trivial_baseline_metrics_include_all_background_and_all_foreground():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :] = 1

    baselines = trivial_baseline_metrics([mask])

    assert set(baselines) == {"all_background", "all_foreground"}
    assert baselines["all_background"]["dice"] == 0.0
    assert baselines["all_foreground"]["dice"] == pytest.approx(2 / 3)
    assert baselines["all_foreground"]["jaccard"] == pytest.approx(0.5)


def test_trivial_baselines_handle_empty_partial_and_full_masks():
    empty = np.zeros((4, 4), dtype=np.uint8)
    partial = np.zeros((4, 4), dtype=np.uint8)
    partial[:2, :] = 1
    full = np.ones((4, 4), dtype=np.uint8)

    baselines = trivial_baseline_metrics([empty, partial, full])

    assert baselines["all_background"]["dice"] == pytest.approx(0.0)
    assert baselines["all_background"]["jaccard"] == pytest.approx(0.0)
    assert baselines["all_foreground"]["dice"] == pytest.approx(2 / 3)
    assert baselines["all_foreground"]["jaccard"] == pytest.approx(0.5)


def test_deterministic_validation_manifest_is_stable(tmp_path: Path):
    empty = np.zeros((16, 16), dtype=np.uint8)
    boundary = np.zeros((16, 16), dtype=np.uint8)
    boundary[0:4, 0:4] = 1
    positive = np.zeros((16, 16), dtype=np.uint8)
    positive[6:10, 6:10] = 1
    paths = []
    for name, mask in [("empty.png", empty), ("boundary.png", boundary), ("positive.png", positive)]:
        path = tmp_path / name
        _write_mask(path, mask)
        paths.append(path)

    first = deterministic_validation_manifest(paths, crop_size=16, examples_per_category=1)
    second = deterministic_validation_manifest(paths, crop_size=16, examples_per_category=1)

    assert first == second
    assert {item["category"] for item in first} == {"background", "boundary", "positive"}


def test_deterministic_validation_manifest_has_required_categories(tmp_path: Path):
    empty = np.zeros((16, 16), dtype=np.uint8)
    boundary = np.zeros((16, 16), dtype=np.uint8)
    boundary[0:4, 0:4] = 1
    positive = np.zeros((16, 16), dtype=np.uint8)
    positive[6:10, 6:10] = 1
    paths = []
    for name, mask in [("z_positive.png", positive), ("a_empty.png", empty), ("m_boundary.png", boundary)]:
        path = tmp_path / name
        _write_mask(path, mask)
        paths.append(path)

    first = deterministic_validation_manifest(paths, crop_size=16, examples_per_category=1)
    second = deterministic_validation_manifest(reversed(paths), crop_size=16, examples_per_category=1)

    assert first == second
    assert [item["category"] for item in first] == ["background", "boundary", "positive"]
    assert all(item["crop_box"] == [0, 0, 16, 16] for item in first)
    assert all("image_path" in item for item in first)
    audit = audit_manifest_crops(first)
    assert audit["categories"] == {"background": 1, "boundary": 1, "positive": 1}


def test_deterministic_validation_manifest_spreads_examples_across_images_when_available(tmp_path: Path):
    paths = []
    for name in ["T19_Image0", "T20_Image0", "T21_Image0"]:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0:4, 0:4] = 1
        mask[14:18, 14:18] = 1
        subject = name.split("_")[0]
        mask_path = tmp_path / "masks" / subject / f"{name}_mask.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = tmp_path / "images" / subject / f"{name}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
        _write_mask(mask_path, mask)
        paths.append(mask_path)

    manifest = deterministic_validation_manifest(paths, crop_size=8, examples_per_category=2)

    positives = [item for item in manifest if item["category"] == "positive"]
    boundaries = [item for item in manifest if item["category"] == "boundary"]
    assert len({item["image_path"] for item in positives}) == 2
    assert len({item["image_path"] for item in boundaries}) == 2
    assert len({item["subject_id"] for item in positives}) == 2
    assert len({item["subject_id"] for item in boundaries}) == 2
    assert all(item["category"] in {"background", "boundary", "positive"} for item in manifest)


def test_degenerate_all_foreground_predictions_block_promotion():
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[4:8, 4:8] = 1
    all_foreground = np.ones((16, 16), dtype=np.uint8)

    review = evaluate_prediction_degeneracy([truth], [all_foreground])

    assert review["blocked"] is True
    assert "predictions_are_all_foreground" in review["reasons"]
    assert "candidate_does_not_clear_all_foreground_dice_baseline" in review["reasons"]


def test_degenerate_all_background_predictions_block_promotion():
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[4:8, 4:8] = 1
    all_background = np.zeros((16, 16), dtype=np.uint8)

    review = evaluate_prediction_degeneracy([truth], [all_background])

    assert review["blocked"] is True
    assert "predictions_are_all_background" in review["reasons"]
    assert review["prediction_foreground_fraction"]["max"] == 0.0


def test_prediction_degeneracy_requires_matching_pair_counts():
    with pytest.raises(ValueError, match="same length"):
        evaluate_prediction_degeneracy([np.zeros((4, 4), dtype=np.uint8)], [])


def test_glomeruli_promotion_candidate_report_blocks_missing_gate_evidence(tmp_path: Path):
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[4:8, 4:8] = 1
    pred = np.ones((16, 16), dtype=np.uint8)
    manifest = [{"category": "background", "foreground_fraction": 0.0}]
    provenance = {"training_mode": "static_patch", "scientific_promotion_status": "not_evaluated"}

    report = evaluate_glomeruli_promotion_candidate([truth], [pred], manifest, provenance)

    assert report["blocked"] is True
    assert "artifact_training_mode_is_not_dynamic_full_image_patching" in report["reasons"]
    assert "manifest_missing_boundary_examples" in report["reasons"]
    assert "manifest_missing_positive_examples" in report["reasons"]
    assert "predictions_are_all_foreground" in report["reasons"]


def test_prediction_review_blocks_undersegmented_positive_examples():
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[2:14, 2:14] = 1
    pred = np.zeros((16, 16), dtype=np.uint8)
    pred[7:9, 7:9] = 1

    review = evaluate_prediction_review(
        [truth],
        [pred],
        manifest=[{"category": "positive", "foreground_fraction": float(truth.mean())}],
    )

    assert review["blocked"] is True
    assert (
        "prediction_does_not_reasonably_cover_positive_glomerulus_examples"
        in review["reasons"]
    )
