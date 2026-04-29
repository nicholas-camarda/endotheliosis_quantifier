"""
FastAI v2 DataBlock loader for binary segmentation.

Supported segmentation training uses full-image `images/` and `masks/`
directories with on-the-fly dynamic patching.
"""

import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np
import torch
from fastai.data.block import DataBlock
from fastai.data.transforms import ItemTransform, RandomSplitter, Transform
from fastai.vision.all import (
    ImageBlock,
    IntToFloatTensor,
    MaskBlock,
    Normalize,
    PILImage,
    PILMask,
    Resize,
    ResizeMethod,
    TensorImage,
    TensorMask,
    aug_transforms,
    get_image_files,
    imagenet_stats,
)
from PIL import Image

from eq.core.constants import (
    DEFAULT_FLIP_VERT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_LIGHTING,
    DEFAULT_MAX_ROTATE,
    DEFAULT_MAX_WARP,
    DEFAULT_MAX_ZOOM,
    DEFAULT_MIN_POS_PIXELS,
    DEFAULT_MIN_ZOOM,
    DEFAULT_POS_CROP_ATTEMPTS,
    DEFAULT_POSITIVE_CROP_JITTER,
    DEFAULT_POSITIVE_FOCUS_P,
    DEFAULT_VAL_RATIO,
)
from eq.utils.logger import get_logger

TRAINING_MODE_DYNAMIC_FULL_IMAGE = "dynamic_full_image_patching"
STATIC_PATCH_DIR_NAMES = (
    "image_patches",
    "mask_patches",
    "image_patch_validation",
    "mask_patch_validation",
)
GLOMERULI_TRAINING_STAGES = {"glomeruli", "glomeruli_transfer"}
TRAINING_MASK_LANES = {
    "manual_mask_core",
    "manual_mask_external",
    # Legacy names retained so current runtime manifests continue to load until
    # they are migrated to the clearer cohort/lane vocabulary.
    "manual_mask",
    "masked_external",
}
SUPPORTED_AUGMENTATION_VARIANTS = {"fastai_default", "spatial_only", "no_aug"}
BLOCKED_RAW_PROJECT_ROOT_PARTS = {
    "_retired",
    "backup",
    "backup_before_reorganization",
    "clean_backup",
    "old",
}


def resolve_segmentation_training_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve the requested segmentation training device.

    CPU remains supported for small tests and explicit low-resource runs. A
    workflow that requires MPS or CUDA must pass that device explicitly so it
    fails instead of silently using CPU.
    """
    if device is not None:
        requested = torch.device(device)
        if requested.type == "mps" and not (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("Requested segmentation training device `mps`, but MPS is unavailable.")
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested segmentation training device `cuda`, but CUDA is unavailable.")
        return requested
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augmentation_policy_for_variant(variant: str) -> dict[str, Any]:
    """Return the explicit augmentation policy for supported segmentation variants."""
    key = (variant or "fastai_default").strip().lower()
    if key not in SUPPORTED_AUGMENTATION_VARIANTS:
        raise ValueError(
            f"Unsupported augmentation variant {variant!r}. "
            f"Use one of {sorted(SUPPORTED_AUGMENTATION_VARIANTS)}."
        )
    if key == "no_aug":
        return {
            "variant": key,
            "fastai_aug_transforms": False,
            "config_controls_active": True,
            "gaussian_noise_active": False,
            "max_rotate": 0,
            "flip_vert": False,
            "min_zoom": 1.0,
            "max_zoom": 1.0,
            "max_warp": 0.0,
            "max_lighting": 0.0,
        }
    policy = {
        "variant": key,
        "fastai_aug_transforms": True,
        "config_controls_active": True,
        "gaussian_noise_active": False,
        "max_rotate": int(DEFAULT_MAX_ROTATE),
        "flip_vert": DEFAULT_FLIP_VERT,
        "min_zoom": DEFAULT_MIN_ZOOM,
        "max_zoom": DEFAULT_MAX_ZOOM,
        "max_warp": DEFAULT_MAX_WARP,
        "max_lighting": DEFAULT_MAX_LIGHTING,
    }
    if key == "spatial_only":
        policy["max_lighting"] = 0.0
    return policy


def default_get_y_path(x: Union[str, Path]) -> Path:
    """Resolve full-image masks for FastAI learner serialization ABI."""
    from eq.data_management.standard_getters import get_y_full

    return Path(get_y_full(x))


def _is_raw_data_cohort_root(root: Path) -> bool:
    return root.parent.name == "cohorts" and "raw_data" in set(root.parts)


def _is_raw_data_cohort_registry_root(root: Path) -> bool:
    return root.name == "cohorts" and root.parent.name == "raw_data"


def _is_raw_data_training_pairs_root(root: Path) -> bool:
    return root.name == "training_pairs" and "raw_data" in set(root.parts)


def _is_raw_data_project_root(root: Path) -> bool:
    if "raw_data" not in set(root.parts):
        return False
    if set(root.parts) & BLOCKED_RAW_PROJECT_ROOT_PARTS:
        return False
    if _is_raw_data_training_pairs_root(root) or _is_raw_data_cohort_root(root):
        return False
    return (root / "images").is_dir() and (root / "masks").is_dir()


def _runtime_root_for_cohort_root(root: Path) -> Path:
    # <runtime_root>/raw_data/cohorts/<cohort_id>
    return root.parents[2]


def _runtime_root_for_cohort_registry_root(root: Path) -> Path:
    # <runtime_root>/raw_data/cohorts
    return root.parents[1]


def _manifest_pair_record(image_path: Path, mask_path: Path, *, manifest_row: str | None = None) -> dict[str, Any]:
    return {
        "__eq_manifest_pair_record__": True,
        "source_image_path": str(image_path),
        "source_mask_path": str(mask_path),
        "manifest_row": manifest_row,
    }


def _is_manifest_pair_record(item: Any) -> bool:
    return isinstance(item, dict) and item.get("__eq_manifest_pair_record__") is True


def training_item_image_path(item: Any) -> Path:
    if _is_manifest_pair_record(item) or _is_negative_crop_record(item):
        return Path(item["source_image_path"]).expanduser()
    return Path(item).expanduser()


def training_item_mask_path(item: Any) -> Path:
    if _is_manifest_pair_record(item):
        return Path(item["source_mask_path"]).expanduser()
    from eq.data_management.standard_getters import get_y_full as _get_y_full

    return _get_y_full(item)


def _validate_readable_image(path: Path, *, role: str, row_label: str, image_path: Path, mask_path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest-backed training row {row_label} has missing {role}: "
            f"image_path={image_path}; mask_path={mask_path}"
        )
    try:
        with Image.open(path) as handle:
            handle.verify()
    except Exception as exc:
        raise ValueError(
            f"Manifest-backed training row {row_label} has unreadable {role}: "
            f"image_path={image_path}; mask_path={mask_path}"
        ) from exc


def _manifest_admitted_cohort_images(root: Path) -> Optional[List[Any]]:
    if _is_raw_data_cohort_registry_root(root):
        manifest_path = root / "manifest.csv"
        runtime_root = _runtime_root_for_cohort_registry_root(root)
        cohort_name = None
    elif _is_raw_data_cohort_root(root):
        manifest_path = root.parent / "manifest.csv"
        runtime_root = _runtime_root_for_cohort_root(root)
        cohort_name = root.name
    else:
        return None
    if not manifest_path.exists():
        return None

    import pandas as pd

    manifest = pd.read_csv(manifest_path).fillna("")
    lane_filter = manifest["lane_assignment"].astype(str).isin(TRAINING_MASK_LANES)
    cohort_rows = manifest[
        (manifest["admission_status"].astype(str) == "admitted")
        & lane_filter
        & (~manifest["image_path"].map(lambda value: str(value).strip() == ""))
        & (~manifest["mask_path"].map(lambda value: str(value).strip() == ""))
    ].copy()
    if cohort_name is not None:
        cohort_rows = cohort_rows[cohort_rows["cohort_id"].astype(str) == cohort_name]
    items: list[dict[str, Any]] = []
    for row_index, row in cohort_rows.iterrows():
        row_dict = row.to_dict()
        row_id = str(row_dict.get("manifest_row_id") or row_dict.get("row_id") or row_index)
        row_label = f"{row_id}"
        image_path = runtime_root / str(row["image_path"])
        mask_path = runtime_root / str(row["mask_path"])
        _validate_readable_image(
            image_path,
            role="image",
            row_label=row_label,
            image_path=image_path,
            mask_path=mask_path,
        )
        _validate_readable_image(
            mask_path,
            role="mask",
            row_label=row_label,
            image_path=image_path,
            mask_path=mask_path,
        )
        items.append(_manifest_pair_record(image_path, mask_path, manifest_row=row_label))
    return items


def validate_supported_segmentation_training_root(data_root: Union[str, Path], *, stage: str = "segmentation") -> Path:
    """Validate the supported full-image dynamic-patching training contract."""
    root = Path(data_root).expanduser()
    static_dirs = [name for name in STATIC_PATCH_DIR_NAMES if (root / name).exists()]
    is_glomeruli_stage = stage in GLOMERULI_TRAINING_STAGES
    is_manifest_cohort_registry = _is_raw_data_cohort_registry_root(root) and (root / "manifest.csv").is_file()
    has_images = (root / "images").is_dir()
    has_masks = (root / "masks").is_dir()

    if static_dirs:
        static_text = ", ".join(static_dirs)
        raise ValueError(
            f"Unsupported static patch training root for {stage}: {root}. "
            f"Found retired static patch directories: {static_text}. "
            "Supported segmentation training requires a full-image root with "
            "`images/` and `masks/` directories."
        )

    if is_glomeruli_stage and is_manifest_cohort_registry:
        return root

    missing = []
    if not has_images:
        missing.append("images/")
    if not has_masks:
        missing.append("masks/")
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Unsupported segmentation training root for {stage}: {root}. "
            f"Missing required full-image directories: {missing_text}. "
            "Supported segmentation training uses full-image `images/` and "
            "`masks/` with dynamic patching."
        )

    if is_glomeruli_stage:
        if not (
            _is_raw_data_training_pairs_root(root)
            or _is_raw_data_cohort_root(root)
            or _is_raw_data_project_root(root)
        ):
            raise ValueError(
                f"Unsupported glomeruli training root for {stage}: {root}. "
                "Supported glomeruli training uses an active paired project root "
                "under `raw_data/<project>/...`, a legacy standalone "
                "`raw_data/.../training_pairs` root, an admitted runtime cohort root "
                "under `raw_data/cohorts/<cohort_id>`, or the manifest-backed "
                "`raw_data/cohorts` registry root for all admitted masked rows. "
                "Raw backup trees such as `clean_backup` are source material, not "
                "direct training roots."
            )

    return root


class MaskPreprocessTransform(Transform):
    """Convert mask values from [0, 255] to [0, 1] for binary segmentation."""
    
    def encodes(self, x: TensorMask):
        if x.dtype == torch.uint8 and x.max() > 1:
            return (x > 127).long()
        return x

class CropTransform(ItemTransform):
    """Positive-aware synchronized crop for image-mask pairs."""
    def __init__(
        self,
        size: int = DEFAULT_IMAGE_SIZE,
        positive_focus_p: float = DEFAULT_POSITIVE_FOCUS_P,
        min_pos_pixels: int = DEFAULT_MIN_POS_PIXELS,
        max_attempts: int = DEFAULT_POS_CROP_ATTEMPTS,
        jitter: int = DEFAULT_POSITIVE_CROP_JITTER,
    ):
        self.size = int(size)
        self.positive_focus_p = float(positive_focus_p)
        self.min_pos_pixels = int(min_pos_pixels)
        self.max_attempts = int(max_attempts)
        self.jitter = int(jitter)

    def _to_numpy_mask(self, mask):
        try:
            import torch
            if hasattr(mask, 'detach') and isinstance(mask, torch.Tensor):
                m = mask.detach().cpu().numpy()
            else:
                m = np.array(mask)
            while m.ndim > 2 and m.shape[0] == 1:
                m = np.squeeze(m, axis=0)
            if m.ndim == 3:
                m = np.squeeze(m)
            return (m > 0).astype(np.uint8)
        except Exception:
            return None

    def _random_crop_coords(self, h: int, w: int):
        top = np.random.randint(0, max(1, h - self.size + 1))
        left = np.random.randint(0, max(1, w - self.size + 1))
        return top, left

    def _pos_centered_crop_coords(self, pos_idx, h: int, w: int):
        y, x = pos_idx[np.random.randint(0, len(pos_idx))]
        if self.jitter > 0:
            y += np.random.randint(-self.jitter, self.jitter + 1)
            x += np.random.randint(-self.jitter, self.jitter + 1)
        top = int(np.clip(y - self.size // 2, 0, max(0, h - self.size)))
        left = int(np.clip(x - self.size // 2, 0, max(0, w - self.size)))
        return top, left

    def encodes(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            return x

        img, mask = x
        if hasattr(img, 'shape'):
            h, w = img.shape[-2:]
        else:
            h, w = img.size[1], img.size[0]

        if h < self.size or w < self.size:
            from fastai.vision.all import Resize
            img = Resize(self.size)(img)
            mask = Resize(self.size)(mask)
            h, w = self.size, self.size

        use_pos_focus = np.random.rand() < self.positive_focus_p
        top, left = self._random_crop_coords(h, w)

        if use_pos_focus:
            m = self._to_numpy_mask(mask)
            if m is not None and m.max() > 0:
                pos_idx = np.column_stack(np.where(m > 0))
                for _ in range(self.max_attempts):
                    ty, lx = self._pos_centered_crop_coords(pos_idx, h, w)
                    crop = m[ty:ty + self.size, lx:lx + self.size]
                    if int(crop.sum()) >= self.min_pos_pixels:
                        top, left = ty, lx
                        break
                else:
                    top, left = self._pos_centered_crop_coords(pos_idx, h, w)

        if hasattr(img, 'crop') and img is not None and mask is not None:
            img_cropped = img.crop((left, top, left + self.size, top + self.size))
            mask_cropped = mask.crop((left, top, left + self.size, top + self.size))
            img_arr = np.asarray(img_cropped).copy()
            if img_arr.ndim == 2:
                img_arr = np.repeat(img_arr[..., None], 3, axis=-1)
            img_cropped = TensorImage(torch.from_numpy(img_arr).permute(2, 0, 1))
            mask_cropped = TensorMask(torch.from_numpy(np.asarray(mask_cropped).copy()))
        elif img is not None and mask is not None:
            img_cropped = img[..., top:top + self.size, left:left + self.size]
            mask_cropped = mask[..., top:top + self.size, left:left + self.size]
        else:
            img_cropped, mask_cropped = img, mask

        return img_cropped, mask_cropped
    
def get_items_full_images(path: Path) -> List[Any]:
    """
    Get image files from images/ directory for dynamic patching.
    Manifest-backed cohort roots use only admitted masked manifest rows; other
    roots explicitly use the local images/ directory.
    """
    # Explicitly use images/ directory - no fallbacks for safety
    root = Path(path)
    manifest_items = _manifest_admitted_cohort_images(root)
    if manifest_items is not None:
        logger = get_logger("eq.datablock_loader")
        logger.info(
            f"DataBlock (cohort full images): Found {len(manifest_items)} manifest-admitted image-mask pairs"
        )
        return manifest_items

    images_dir = root / "images"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"images/ directory not found in {path}. Expected: {images_dir}")
    
    all_images = get_image_files(images_dir)
    valid_images = []
    skipped_count = 0
    
    logger = get_logger("eq.datablock_loader")
    
    skipped_samples = []
    for img_path in all_images:
        # Use the full-image getter to check if mask exists
        try:
            from eq.data_management.standard_getters import get_y_full as _get_y_full
            _get_y_full(img_path)
            valid_images.append(img_path)
        except Exception:
            skipped_count += 1
            if len(skipped_samples) < 10:
                skipped_samples.append(Path(img_path).name)
            logger.debug(f"Skipping image {Path(img_path).name} - no corresponding mask found")
    
    if skipped_count > 0:
        examples = ", ".join(skipped_samples)
        raise ValueError(
            f"Unpaired full-image training root: found {skipped_count} image(s) without masks "
            f"under {images_dir}. Examples: {examples}. Supported training requires an explicit "
            "image/mask pair contract before model construction."
        )

    logger.info(f"DataBlock (dynamic patching): Found {len(valid_images)} valid image-mask pairs")
    return valid_images


def fixed_splitter_from_paths(
    *,
    train_images: Iterable[str | Path],
    valid_images: Iterable[str | Path],
) -> Callable[[list[Any]], tuple[list[int], list[int]]]:
    """Build a strict splitter from explicit image path provenance."""
    train_set = {str(Path(path).expanduser()) for path in train_images}
    valid_set = {str(Path(path).expanduser()) for path in valid_images}
    overlap = train_set & valid_set
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Explicit split has train/valid overlap: {examples}")
    if not train_set or not valid_set:
        raise ValueError("Explicit split requires non-empty train_images and valid_images.")

    def splitter(items: list[Any]) -> tuple[list[int], list[int]]:
        train_idx: list[int] = []
        valid_idx: list[int] = []
        unknown: list[str] = []
        for index, item in enumerate(items):
            key = str(training_item_image_path(item))
            if key in train_set:
                train_idx.append(index)
            elif key in valid_set:
                valid_idx.append(index)
            else:
                unknown.append(key)
        if unknown:
            examples = ", ".join(unknown[:5])
            raise ValueError(f"Explicit split does not cover {len(unknown)} item(s): {examples}")
        return train_idx, valid_idx

    return splitter


def fixed_splitter_from_manifest(split_manifest_path: str | Path) -> Callable[[list[Any]], tuple[list[int], list[int]]]:
    """Load a strict explicit train/validation splitter from a JSON manifest."""
    path = Path(split_manifest_path).expanduser()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return fixed_splitter_from_paths(
        train_images=payload.get("train_images") or [],
        valid_images=payload.get("valid_images") or [],
    )


def build_segmentation_datablock_dynamic_patching(
    codes: Optional[List[int]] = None,
    crop_size: int = DEFAULT_IMAGE_SIZE,
    output_size: Optional[int] = None,
    min_scale: float = 0.3,
    flip_p: float = 0.5,
    max_rotate: float = 10.0,
    splitter: Optional[Callable] = None,
    positive_focus_p: float = DEFAULT_POSITIVE_FOCUS_P,
    min_pos_pixels: int = DEFAULT_MIN_POS_PIXELS,
    pos_crop_attempts: int = DEFAULT_POS_CROP_ATTEMPTS,
    negative_crop_manifest_path: Optional[Union[str, Path]] = None,
    negative_crop_sampler_weight: float = 0.0,
    augmentation_variant: str = "fastai_default",
    split_seed: int = 42,
):
    """
    Create a DataBlock for binary segmentation with dynamic patching.
    
    Training behavior:
    1. Loads full images from images/ directory
    2. Applies augmentations to full images first
    3. Crops synchronized image/mask patches after augmentation
    4. Rejects pre-generated static patch roots before model construction
    
    Args:
        codes: Label codes for MaskBlock (binary by default)
        crop_size: Size of patches to crop (default 256)
        min_scale: Minimum scale for RandomResizedCrop
        flip_p: Probability of horizontal flip
        max_rotate: Maximum rotation in degrees
        splitter: Train/valid splitter (defaults to 80/20 RandomSplitter)
    """
    if codes is None:
        codes = [0, 1]
    
    if splitter is None:
        splitter = RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=int(split_seed))

    negative_records: list[dict[str, Any]] = []
    if negative_crop_manifest_path:
        from eq.data_management.negative_glomeruli_crops import (
            validate_negative_crop_manifest,
            weighted_negative_rows,
        )

        validation = validate_negative_crop_manifest(negative_crop_manifest_path)
        for row in weighted_negative_rows(validation.rows, negative_crop_sampler_weight):
            record = dict(row)
            record["__eq_negative_crop_record__"] = True
            negative_records.append(record)

    base_splitter = splitter

    def combined_get_items(path: Path) -> List[Any]:
        items = list(get_items_full_images(path))
        if negative_records:
            items.extend(negative_records)
        return items

    def combined_splitter(items: list[Any]) -> tuple[list[int], list[int]]:
        base_indices = [index for index, item in enumerate(items) if not _is_negative_crop_record(item)]
        negative_indices = [index for index, item in enumerate(items) if _is_negative_crop_record(item)]
        train_local, valid_local = base_splitter([items[index] for index in base_indices])
        train_indices = [base_indices[index] for index in train_local]
        valid_indices = [base_indices[index] for index in valid_local]
        # Negative/background manifest crops are training supervision, not held-out
        # validation evidence. Promotion evidence handles background crops separately.
        train_indices.extend(negative_indices)
        return train_indices, valid_indices
    
    # Item transforms: apply synchronized augmentations and cropping to image-mask pairs
    if output_size is None:
        output_size = crop_size

    item_tfms = [
        CropTransform(
            size=crop_size,
            positive_focus_p=positive_focus_p,
            min_pos_pixels=min_pos_pixels,
            max_attempts=pos_crop_attempts,
        ),
        # Resize the selected crop to network input size without changing crop center.
        Resize(output_size, method=ResizeMethod.Squish),
    ]
    
    # Batch transforms: apply named augmentation variants with ImageNet normalization.
    augmentation_policy = augmentation_policy_for_variant(augmentation_variant)
    batch_tfms = [IntToFloatTensor()]
    if augmentation_policy["fastai_aug_transforms"]:
        batch_tfms.extend(
            aug_transforms(
                # Do not pass size here; avoid RandomResizedCrop-like behavior.
                max_rotate=int(augmentation_policy["max_rotate"]),
                flip_vert=bool(augmentation_policy["flip_vert"]),
                min_zoom=float(augmentation_policy["min_zoom"]),
                max_zoom=float(augmentation_policy["max_zoom"]),
                max_warp=float(augmentation_policy["max_warp"]),
                max_lighting=float(augmentation_policy["max_lighting"]),
            )
        )
    batch_tfms.extend(
        [
            Normalize.from_stats(*imagenet_stats, cuda=False),
            MaskPreprocessTransform(),
        ]
    )
    
    block = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=codes)],  # Use codes for proper segmentation
        get_items=combined_get_items,
        get_x=_get_x_negative_or_path,
        get_y=_get_y_negative_or_full,
        splitter=combined_splitter,
        item_tfms=item_tfms, 
        batch_tfms=batch_tfms,
    )
    return block


def build_segmentation_dls_dynamic_patching(
    data_root: Union[str, Path], 
    bs: int = 8, 
    num_workers: int = 0,
    crop_size: int = DEFAULT_IMAGE_SIZE,
    output_size: Optional[int] = None,
    min_scale: float = 0.3,
    flip_p: float = 0.5,
    max_rotate: float = 10.0,
    positive_focus_p: float = DEFAULT_POSITIVE_FOCUS_P,
    min_pos_pixels: int = DEFAULT_MIN_POS_PIXELS,
    pos_crop_attempts: int = DEFAULT_POS_CROP_ATTEMPTS,
    stage: str = "segmentation",
    splitter: Optional[Callable] = None,
    negative_crop_manifest_path: Optional[Union[str, Path]] = None,
    negative_crop_sampler_weight: float = 0.0,
    augmentation_variant: str = "fastai_default",
    device: Optional[Union[str, torch.device]] = None,
    split_seed: int = 42,
):
    """
    Create DataLoaders for binary segmentation with dynamic patching.
    
    This approach:
    1. Loads full images from images/ directory
    2. Applies augmentations to full images
    3. Crops patches on-the-fly during training
    4. Provides much better augmentation diversity
    5. Uses positive-aware cropping for class balance
    
    Args:
        data_root: Path to data directory with images/ and masks/ subdirectories
        bs: Batch size
        num_workers: Number of workers for data loading
        crop_size: Size of patches to crop (default 256)
        min_scale: Minimum scale for RandomResizedCrop
        flip_p: Probability of horizontal flip
        max_rotate: Maximum rotation in degrees
        positive_focus_p: Probability to bias crops toward positive regions
        min_pos_pixels: Minimum positive pixels required in focused crops
        pos_crop_attempts: Maximum attempts to find sufficiently positive crops
        
    Returns:
        DataLoaders object
    """
    root = validate_supported_segmentation_training_root(data_root, stage=stage)

    # Preflight: ensure we have at least one valid image-mask pair
    prelim_items = get_items_full_images(root)
    if len(prelim_items) == 0:
        raise ValueError(
            (
                "No valid image-mask pairs found for dynamic patching. "
                "Ensure `<data_root>/images/` exists and corresponding masks reside under `masks/` "
                "using patterns `<stem>_mask<ext>` or `mask_<stem><ext>`."
            )
        )

    # Create DataBlock with dynamic patching
    logger = get_logger("eq.datablock_loader")
    try:
        logger.info(f"Dynamic patching sizes: crop_size={crop_size}, output_size={output_size if output_size is not None else crop_size}")
    except Exception:
        pass

    db = build_segmentation_datablock_dynamic_patching(
        crop_size=crop_size,
        output_size=output_size,
        min_scale=min_scale,
        flip_p=flip_p,
        max_rotate=max_rotate,
        splitter=splitter,
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
        negative_crop_manifest_path=negative_crop_manifest_path,
        negative_crop_sampler_weight=negative_crop_sampler_weight,
        augmentation_variant=augmentation_variant,
        split_seed=split_seed,
    )
    
    # Create DataLoaders on the required accelerator.  This is part of the
    # runtime contract: segmentation training must not silently run on CPU.
    training_device = resolve_segmentation_training_device(device)
    dls = db.dataloaders(root, bs=bs, num_workers=num_workers, device=training_device)
    
    logger = get_logger("eq.datablock_loader")
    logger.info(f"Segmentation DataLoaders device: {dls.device}")
    logger.info("✅ Dynamic patching DataLoaders created - positive-aware cropping enabled" if positive_focus_p > 0 else "✅ Dynamic patching DataLoaders created")
    # Crop coverage diagnostic
    try:
        pos_rate = estimate_positive_crop_rate(dls.valid, batches=5)
        min_pos_rate = estimate_min_positive_pixels_rate(dls.valid, min_pos_pixels=int(min_pos_pixels), batches=5)
        if pos_rate < 0.15:
            logger.warning(f"⚠️  Positive-any coverage (val): {pos_rate:.1%}; >=min_pos_pixels: {min_pos_rate:.1%}. Consider increasing positive_focus_p or min_pos_pixels.")
        else:
            logger.info(f"🔎 Positive crop coverage (val): any={pos_rate:.1%}, >=min_pos_pixels={min_pos_rate:.1%}")
    except Exception:
        pass
    
    return dls


def _is_negative_crop_record(item: Any) -> bool:
    return isinstance(item, dict) and item.get("__eq_negative_crop_record__") is True


def _negative_crop_box(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["crop_x_min"]),
        int(record["crop_y_min"]),
        int(record["crop_x_max"]),
        int(record["crop_y_max"]),
    )


def _get_x_negative_or_path(item):
    if _is_manifest_pair_record(item):
        return Path(item["source_image_path"])
    if not _is_negative_crop_record(item):
        return item
    image = PILImage.create(item["source_image_path"])
    return PILImage.create(image.crop(_negative_crop_box(item)))


def _get_y_negative_or_full(item):
    if _is_manifest_pair_record(item):
        return Path(item["source_mask_path"])
    if not _is_negative_crop_record(item):
        from eq.data_management.standard_getters import get_y_full as _get_y_full
        return _get_y_full(item)
    x_min, y_min, x_max, y_max = _negative_crop_box(item)
    height = y_max - y_min
    width = x_max - x_min
    zero_mask = np.zeros((height, width), dtype=np.uint8)
    from PIL import Image
    return PILMask.create(Image.fromarray(zero_mask))


def estimate_positive_crop_rate(dl, batches: int = 5) -> float:
    """Estimate fraction of crops that contain any positive pixels in a dataloader."""
    import torch
    seen = 0
    has_pos = 0
    it = iter(dl)
    for _ in range(batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        if torch.is_tensor(yb):
            y = yb
        else:
            try:
                y = torch.as_tensor(yb)
            except Exception:
                continue
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        b = y.shape[0]
        seen += b
        has_pos += (y.view(b, -1).max(dim=1).values > 0).sum().item()
    return (has_pos / seen) if seen > 0 else 0.0


def estimate_min_positive_pixels_rate(dl, min_pos_pixels: int, batches: int = 5) -> float:
    """Estimate fraction of crops with at least `min_pos_pixels` positive pixels."""
    import torch
    seen = 0
    meets_min = 0
    it = iter(dl)
    for _ in range(batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        if torch.is_tensor(yb):
            y = yb
        else:
            try:
                y = torch.as_tensor(yb)
            except Exception:
                continue
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        b = y.shape[0]
        seen += b
        meets_min += (y.view(b, -1).sum(dim=1) >= int(min_pos_pixels)).sum().item()
    return (meets_min / seen) if seen > 0 else 0.0
