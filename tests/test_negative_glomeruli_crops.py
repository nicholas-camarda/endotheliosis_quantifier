from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eq.data_management.datablock_loader import build_segmentation_dls_dynamic_patching
from eq.data_management.negative_glomeruli_crops import (
    CURATED_NEGATIVE_LABEL,
    PROPOSAL_STATUS,
    REQUIRED_FIELDS,
    file_sha256,
    generate_mask_derived_background_manifest,
    generate_mr_tiff_review_batch,
    validate_negative_crop_manifest,
)


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _paired_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw_data" / "toy_project" / "training_pairs"
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[40:55, 40:55] = 255
    _write_image(root / "images" / "sample_image0.png", image)
    _write_image(root / "masks" / "sample_image0_mask.png", mask)
    return root


def test_mask_derived_background_manifest_has_zero_overlap(tmp_path: Path) -> None:
    root = _paired_root(tmp_path)
    manifest = tmp_path / "derived_data/glomeruli_negative_crops/manifests/test.csv"

    audit = generate_mask_derived_background_manifest(
        data_root=root,
        manifest_path=manifest,
        curation_id="test",
        crop_size=24,
        crops_per_image_limit=2,
        min_foreground_pixels=0,
        seed=1,
    )

    validation = validate_negative_crop_manifest(manifest)
    assert audit["negative_crop_count"] == validation.negative_crop_count
    assert validation.mask_derived_background_crop_count > 0
    assert validation.curated_negative_crop_count == 0
    for row in validation.rows:
        mask = np.asarray(Image.open(root / "masks" / "sample_image0_mask.png"))
        crop = mask[
            int(row["crop_y_min"]): int(row["crop_y_max"]),
            int(row["crop_x_min"]): int(row["crop_x_max"]),
        ]
        assert int((crop > 0).sum()) == 0


def test_unreviewed_mr_tiff_proposal_is_rejected_for_training(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/cohorts/vegfri_mr/images/mr0.tif"
    _write_image(source, np.full((32, 32, 3), 128, dtype=np.uint8))
    manifest = tmp_path / "derived_data/glomeruli_negative_crops/manifests/proposal.csv"
    row = {field: "" for field in REQUIRED_FIELDS}
    row.update(
        {
            "negative_crop_id": "proposal_0",
            "source_image_path": str(source),
            "source_image_sha256": file_sha256(source),
            "source_cohort_id": "vegfri_mr",
            "crop_x_min": 0,
            "crop_y_min": 0,
            "crop_x_max": 16,
            "crop_y_max": 16,
            "coordinate_frame": "source_image_pixels_xyxy",
            "label": CURATED_NEGATIVE_LABEL,
            "annotation_status": PROPOSAL_STATUS,
            "reviewer_id": "pending",
            "reviewed_at_utc": "2026-04-25T00:00:00Z",
            "review_batch_id": "batch0",
            "review_protocol_version": "mr-negative-review-v1",
            "negative_scope": "crop_only",
            "source_mapping_method": "manual_review",
            "source_mapping_status": "pending_review",
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_FIELDS))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="not trainable"):
        validate_negative_crop_manifest(manifest)


def test_mr_review_batch_generates_review_assets_but_not_trainable_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw_data/cohorts/vegfri_mr/images"
    _write_image(source_dir / "mr0.tif", np.full((48, 48, 3), 128, dtype=np.uint8))
    manifest = tmp_path / "derived_data/glomeruli_negative_crops/manifests/mr_review.csv"
    assets = tmp_path / "derived_data/glomeruli_negative_crops/review_assets/mr_review"

    audit = generate_mr_tiff_review_batch(
        source_images_dir=source_dir,
        manifest_path=manifest,
        review_assets_dir=assets,
        curation_id="mr_review",
        crop_size=24,
        proposals_per_image=2,
        seed=1,
    )

    assert audit["proposal_count"] == 2
    assert audit["trainable"] is False
    assert len(list(assets.glob("*.png"))) == 2
    with pytest.raises(ValueError, match="not trainable"):
        validate_negative_crop_manifest(manifest)


def test_negative_manifest_adds_training_only_zero_mask_samples(tmp_path: Path) -> None:
    root = _paired_root(tmp_path)
    manifest = tmp_path / "derived_data/glomeruli_negative_crops/manifests/test.csv"
    generate_mask_derived_background_manifest(
        data_root=root,
        manifest_path=manifest,
        curation_id="test",
        crop_size=24,
        crops_per_image_limit=1,
        min_foreground_pixels=0,
        seed=1,
    )

    dls = build_segmentation_dls_dynamic_patching(
        root,
        bs=2,
        num_workers=0,
        crop_size=24,
        output_size=24,
        negative_crop_manifest_path=manifest,
        negative_crop_sampler_weight=1.0,
        device="cpu",
    )

    assert any(isinstance(item, dict) for item in dls.train_ds.items)
    assert not any(isinstance(item, dict) for item in dls.valid_ds.items)
    xb, yb = dls.one_batch()
    assert xb.shape[-2:] == (24, 24)
    assert yb.shape[-2:] == (24, 24)
