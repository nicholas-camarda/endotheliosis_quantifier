"""Negative/background crop manifests for glomeruli training."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from eq.data_management.datablock_loader import (
    get_items_full_images,
    training_item_image_path,
    training_item_mask_path,
)

MASK_DERIVED_LABEL = "mask_derived_background"
CURATED_NEGATIVE_LABEL = "negative_glomerulus"
MASK_DERIVED_STATUS = "mask_validated_background"
CURATED_REVIEWED_STATUS = "reviewed_negative"
PROPOSAL_STATUS = "proposed_review_only"
NEGATIVE_SCOPE = "crop_only"

REQUIRED_FIELDS = (
    "negative_crop_id",
    "source_image_path",
    "source_image_sha256",
    "source_cohort_id",
    "crop_x_min",
    "crop_y_min",
    "crop_x_max",
    "crop_y_max",
    "coordinate_frame",
    "label",
    "annotation_status",
    "reviewer_id",
    "reviewed_at_utc",
    "review_batch_id",
    "review_protocol_version",
    "negative_scope",
    "source_mapping_method",
    "source_mapping_status",
    "notes",
)


@dataclass(frozen=True)
class NegativeCropValidation:
    """Validated negative/background crop manifest summary."""

    manifest_path: Path
    manifest_sha256: str
    rows: list[dict[str, Any]]
    mask_derived_background_crop_count: int
    curated_negative_crop_count: int
    negative_crop_source_image_count: int
    review_protocol_versions: list[str]

    @property
    def negative_crop_count(self) -> int:
        return len(self.rows)

    def provenance(self, *, sampler_weight: float) -> dict[str, Any]:
        return {
            "negative_crop_supervision_status": "present" if self.rows else "absent",
            "negative_crop_manifest_path": str(self.manifest_path) if self.rows else None,
            "negative_crop_manifest_sha256": self.manifest_sha256 if self.rows else None,
            "negative_crop_count": self.negative_crop_count,
            "mask_derived_background_crop_count": self.mask_derived_background_crop_count,
            "curated_negative_crop_count": self.curated_negative_crop_count,
            "negative_crop_source_image_count": self.negative_crop_source_image_count,
            "negative_crop_review_protocol_version": "|".join(self.review_protocol_versions),
            "negative_crop_sampler_weight": float(sampler_weight),
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REQUIRED_FIELDS})


def _int_field(row: dict[str, Any], field: str, *, row_number: int) -> int:
    try:
        return int(row[field])
    except Exception as exc:
        raise ValueError(f"Row {row_number} has invalid integer field {field!r}: {row.get(field)!r}") from exc


def validate_negative_crop_manifest(
    manifest_path: str | Path,
    *,
    require_trainable: bool = True,
) -> NegativeCropValidation:
    """Validate a negative/background crop manifest for training use."""

    path = Path(manifest_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Negative crop manifest does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Negative crop manifest is not a file: {path}")

    rows = _read_rows(path)
    missing_header = [field for field in REQUIRED_FIELDS if field not in (rows[0].keys() if rows else REQUIRED_FIELDS)]
    if missing_header:
        raise ValueError(f"Negative crop manifest missing required columns: {', '.join(missing_header)}")

    accepted: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        missing = [field for field in REQUIRED_FIELDS if str(row.get(field, "")).strip() == "" and field != "notes"]
        if missing:
            raise ValueError(f"Row {index} missing required fields: {', '.join(missing)}")

        source_path = Path(row["source_image_path"]).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Row {index} source image does not exist: {source_path}")
        expected_sha = str(row["source_image_sha256"]).strip()
        observed_sha = file_sha256(source_path)
        if expected_sha != observed_sha:
            raise ValueError(f"Row {index} source image sha256 mismatch for {source_path}")

        x_min = _int_field(row, "crop_x_min", row_number=index)
        y_min = _int_field(row, "crop_y_min", row_number=index)
        x_max = _int_field(row, "crop_x_max", row_number=index)
        y_max = _int_field(row, "crop_y_max", row_number=index)
        if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Row {index} has invalid crop box: {(x_min, y_min, x_max, y_max)}")
        with Image.open(source_path) as image:
            width, height = image.size
        if x_max > width or y_max > height:
            raise ValueError(f"Row {index} crop box exceeds source image bounds: {(width, height)}")

        label = str(row["label"]).strip()
        status = str(row["annotation_status"]).strip()
        if str(row["negative_scope"]).strip() != NEGATIVE_SCOPE:
            raise ValueError(f"Row {index} negative_scope must be {NEGATIVE_SCOPE!r}")
        if label == MASK_DERIVED_LABEL:
            if status != MASK_DERIVED_STATUS:
                raise ValueError(f"Row {index} mask-derived background requires {MASK_DERIVED_STATUS!r}")
        elif label == CURATED_NEGATIVE_LABEL:
            if status != CURATED_REVIEWED_STATUS:
                if require_trainable:
                    raise ValueError(
                        f"Row {index} curated MR/TIFF crop is not trainable until annotation_status="
                        f"{CURATED_REVIEWED_STATUS!r}; observed {status!r}"
                    )
                continue
        else:
            raise ValueError(f"Row {index} unsupported negative crop label: {label!r}")
        if status == PROPOSAL_STATUS and require_trainable:
            raise ValueError(f"Row {index} is a review proposal, not trainable negative supervision")

        normalized = dict(row)
        normalized.update(
            {
                "crop_x_min": x_min,
                "crop_y_min": y_min,
                "crop_x_max": x_max,
                "crop_y_max": y_max,
            }
        )
        accepted.append(normalized)

    labels = [str(row["label"]) for row in accepted]
    versions = sorted({str(row["review_protocol_version"]) for row in accepted})
    return NegativeCropValidation(
        manifest_path=path,
        manifest_sha256=file_sha256(path),
        rows=accepted,
        mask_derived_background_crop_count=labels.count(MASK_DERIVED_LABEL),
        curated_negative_crop_count=labels.count(CURATED_NEGATIVE_LABEL),
        negative_crop_source_image_count=len({str(row["source_image_path"]) for row in accepted}),
        review_protocol_versions=versions,
    )


def _zero_mask_crop(mask_path: Path, box: tuple[int, int, int, int], *, min_foreground_pixels: int) -> bool:
    with Image.open(mask_path) as mask:
        crop = np.asarray(mask.crop(box))
    return int((crop > 0).sum()) <= int(min_foreground_pixels)


def _candidate_boxes(width: int, height: int, crop_size: int, *, rng: np.random.Generator, attempts: int) -> Iterable[tuple[int, int, int, int]]:
    if width < crop_size or height < crop_size:
        yield (0, 0, width, height)
        return
    # Include deterministic corners, then random boxes for diversity.
    corners = [
        (0, 0),
        (width - crop_size, 0),
        (0, height - crop_size),
        (width - crop_size, height - crop_size),
    ]
    for left, top in corners:
        yield (left, top, left + crop_size, top + crop_size)
    for _ in range(attempts):
        left = int(rng.integers(0, width - crop_size + 1))
        top = int(rng.integers(0, height - crop_size + 1))
        yield (left, top, left + crop_size, top + crop_size)


def generate_mask_derived_background_manifest(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    curation_id: str,
    crop_size: int,
    crops_per_image_limit: int,
    min_foreground_pixels: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate mask-proven background crop rows from admitted paired images."""

    output_path = Path(manifest_path).expanduser()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    rejected_overlap = 0
    rejected_geometry = 0
    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for image_index, item in enumerate(get_items_full_images(Path(data_root).expanduser())):
        image_path = training_item_image_path(item)
        mask_path = training_item_mask_path(item)
        with Image.open(image_path) as image:
            width, height = image.size
        accepted_for_image = 0
        for box in _candidate_boxes(width, height, int(crop_size), rng=rng, attempts=25):
            if accepted_for_image >= int(crops_per_image_limit):
                break
            x_min, y_min, x_max, y_max = box
            if x_max <= x_min or y_max <= y_min:
                rejected_geometry += 1
                continue
            if not _zero_mask_crop(mask_path, box, min_foreground_pixels=min_foreground_pixels):
                rejected_overlap += 1
                continue
            rows.append(
                {
                    "negative_crop_id": f"{curation_id}_{image_index:05d}_{accepted_for_image:02d}",
                    "source_image_path": str(image_path),
                    "source_image_sha256": file_sha256(image_path),
                    "source_cohort_id": image_path.parents[1].name if image_path.parent.name == "images" else image_path.parent.parent.name,
                    "crop_x_min": x_min,
                    "crop_y_min": y_min,
                    "crop_x_max": x_max,
                    "crop_y_max": y_max,
                    "coordinate_frame": "source_image_pixels_xyxy",
                    "label": MASK_DERIVED_LABEL,
                    "annotation_status": MASK_DERIVED_STATUS,
                    "reviewer_id": "mask_zero_overlap_validator",
                    "reviewed_at_utc": reviewed_at,
                    "review_batch_id": curation_id,
                    "review_protocol_version": "mask-derived-background-v1",
                    "negative_scope": NEGATIVE_SCOPE,
                    "source_mapping_method": "paired_segmentation_mask_zero_overlap",
                    "source_mapping_status": "validated_zero_foreground",
                    "notes": f"mask_path={mask_path}",
                }
            )
            accepted_for_image += 1

    _write_rows(output_path, rows)
    validation = validate_negative_crop_manifest(output_path)
    audit = {
        "curation_id": curation_id,
        "manifest_path": str(output_path),
        "negative_crop_count": validation.negative_crop_count,
        "mask_derived_background_crop_count": validation.mask_derived_background_crop_count,
        "source_image_count": validation.negative_crop_source_image_count,
        "rejected_overlap_count": rejected_overlap,
        "rejected_geometry_count": rejected_geometry,
        "crop_size": int(crop_size),
        "crops_per_image_limit": int(crops_per_image_limit),
        "min_foreground_pixels": int(min_foreground_pixels),
        "manifest_sha256": validation.manifest_sha256,
    }
    audit_path = output_path.parents[1] / "audits" / f"{curation_id}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def generate_mr_tiff_review_batch(
    *,
    source_images_dir: str | Path,
    manifest_path: str | Path,
    review_assets_dir: str | Path,
    curation_id: str,
    crop_size: int,
    proposals_per_image: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate review-only MR/TIFF crop proposals and thumbnail assets."""

    source_root = Path(source_images_dir).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"MR/TIFF source image directory does not exist: {source_root}")
    output_path = Path(manifest_path).expanduser()
    asset_root = Path(review_assets_dir).expanduser()
    asset_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    image_paths = sorted(
        path for path in source_root.rglob("*")
        if path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    )
    for image_index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            accepted = 0
            for box in _candidate_boxes(width, height, int(crop_size), rng=rng, attempts=max(25, proposals_per_image * 5)):
                if accepted >= int(proposals_per_image):
                    break
                crop = image.crop(box)
                crop_id = f"{curation_id}_{image_index:05d}_{accepted:02d}"
                asset_path = asset_root / f"{crop_id}.png"
                crop.save(asset_path)
                x_min, y_min, x_max, y_max = box
                rows.append(
                    {
                        "negative_crop_id": crop_id,
                        "source_image_path": str(image_path),
                        "source_image_sha256": file_sha256(image_path),
                        "source_cohort_id": "vegfri_mr",
                        "crop_x_min": x_min,
                        "crop_y_min": y_min,
                        "crop_x_max": x_max,
                        "crop_y_max": y_max,
                        "coordinate_frame": "source_image_pixels_xyxy",
                        "label": CURATED_NEGATIVE_LABEL,
                        "annotation_status": PROPOSAL_STATUS,
                        "reviewer_id": "pending_review",
                        "reviewed_at_utc": reviewed_at,
                        "review_batch_id": curation_id,
                        "review_protocol_version": "mr-negative-review-v1",
                        "negative_scope": NEGATIVE_SCOPE,
                        "source_mapping_method": "manual_review_required",
                        "source_mapping_status": "pending_review",
                        "notes": f"review_asset={asset_path}",
                    }
                )
                accepted += 1

    _write_rows(output_path, rows)
    audit = {
        "curation_id": curation_id,
        "manifest_path": str(output_path),
        "review_assets_dir": str(asset_root),
        "proposal_count": len(rows),
        "source_image_count": len(image_paths),
        "annotation_status": PROPOSAL_STATUS,
        "trainable": False,
        "crop_size": int(crop_size),
        "proposals_per_image": int(proposals_per_image),
    }
    audit_path = output_path.parents[1] / "audits" / f"{curation_id}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def weighted_negative_rows(rows: list[dict[str, Any]], sampler_weight: float) -> list[dict[str, Any]]:
    """Apply a deterministic manifest-row weight without stochastic DataLoader magic."""

    weight = float(sampler_weight)
    if weight <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: str(row["negative_crop_id"]))
    if weight < 1:
        keep = max(1, int(np.ceil(len(ordered) * weight)))
        return ordered[:keep]
    whole = int(np.floor(weight))
    frac = weight - whole
    weighted = ordered * whole
    if frac > 0:
        weighted.extend(ordered[: int(np.ceil(len(ordered) * frac))])
    return weighted


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manage glomeruli negative/background crop manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mask_parser = subparsers.add_parser("generate-mask-background", help="Generate mask-derived background crop manifest")
    mask_parser.add_argument("--data-root", required=True)
    mask_parser.add_argument("--manifest-path", required=True)
    mask_parser.add_argument("--curation-id", required=True)
    mask_parser.add_argument("--crop-size", type=int, required=True)
    mask_parser.add_argument("--crops-per-image-limit", type=int, default=2)
    mask_parser.add_argument("--min-foreground-pixels", type=int, default=0)
    mask_parser.add_argument("--seed", type=int, default=42)

    review_parser = subparsers.add_parser("generate-mr-review-batch", help="Generate review-only MR/TIFF crop proposals")
    review_parser.add_argument("--source-images-dir", required=True)
    review_parser.add_argument("--manifest-path", required=True)
    review_parser.add_argument("--review-assets-dir", required=True)
    review_parser.add_argument("--curation-id", required=True)
    review_parser.add_argument("--crop-size", type=int, required=True)
    review_parser.add_argument("--proposals-per-image", type=int, default=2)
    review_parser.add_argument("--seed", type=int, default=42)

    validate_parser = subparsers.add_parser("validate", help="Validate trainable negative/background crop manifest")
    validate_parser.add_argument("--manifest-path", required=True)

    args = parser.parse_args()
    if args.command == "generate-mask-background":
        result = generate_mask_derived_background_manifest(
            data_root=args.data_root,
            manifest_path=args.manifest_path,
            curation_id=args.curation_id,
            crop_size=args.crop_size,
            crops_per_image_limit=args.crops_per_image_limit,
            min_foreground_pixels=args.min_foreground_pixels,
            seed=args.seed,
        )
    elif args.command == "generate-mr-review-batch":
        result = generate_mr_tiff_review_batch(
            source_images_dir=args.source_images_dir,
            manifest_path=args.manifest_path,
            review_assets_dir=args.review_assets_dir,
            curation_id=args.curation_id,
            crop_size=args.crop_size,
            proposals_per_image=args.proposals_per_image,
            seed=args.seed,
        )
    else:
        validation = validate_negative_crop_manifest(args.manifest_path)
        result = validation.provenance(sampler_weight=1.0)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
