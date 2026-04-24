"""
FastAI v2 DataBlock loader for binary segmentation.

Supported segmentation training uses full-image `images/` and `masks/`
directories with on-the-fly dynamic patching. Static patch helpers remain here
only for legacy audit, conversion, and historical artifact inspection.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from fastai.data.block import DataBlock
from fastai.data.transforms import ItemTransform, RandomSplitter, Transform
from fastai.vision.all import (
    FlipItem,
    ImageBlock,
    IntToFloatTensor,
    MaskBlock,
    Normalize,
    PILMask,
    RandomResizedCrop,
    Resize,
    ResizeMethod,
    TensorImage,
    TensorMask,
    Rotate,
    aug_transforms,
    get_image_files,
    imagenet_stats,
)

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
from eq.data_management.standard_getters import get_y_full, get_y_patch
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
BLOCKED_RAW_PROJECT_ROOT_PARTS = {
    "_retired",
    "backup",
    "backup_before_reorganization",
    "clean_backup",
    "old",
}


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
    return [
        runtime_root / image_path
        for image_path in cohort_rows["image_path"].astype(str).tolist()
        if (runtime_root / image_path).exists()
    ]


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
            "`images/` and `masks/` directories. Use static patch data only for "
            "legacy audit, conversion, or historical inspection workflows."
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


def default_get_y_path(x):
    """Resolve a mask path for either patch datasets or full-image datasets."""
    try:
        return get_y_patch(x)
    except FileNotFoundError:
        return get_y_full(x)


def default_get_y(x):
    """Backward-compatible mask getter returning a loaded ``PILMask`` instance."""
    return PILMask.create(default_get_y_path(x))


def get_items_standard(path: Path) -> List[Any]:
    """Legacy flexible item resolver retained for audit/conversion utilities."""
    root = Path(path)
    if (root / 'image_patches').exists():
        return get_items_patches(root)
    if (root / 'images').exists():
        return get_items_full_images(root)

    raise FileNotFoundError(
        f"Expected either image_patches/ or images/ under {root}."
    )


def build_segmentation_datablock(
    codes: Optional[List[int]] = None,
    get_items: Optional[Callable[[Path], List[Any]]] = None,
    get_y_func: Optional[Callable[[Path], Path]] = None,
    splitter: Optional[Callable] = None,
    item_tfms: Optional[List] = None,
    batch_tfms: Optional[List] = None,
):
    """
    Create a legacy DataBlock for binary segmentation over pre-generated patches.

    This is not a supported model-training builder. It is retained for legacy
    audit, conversion, and historical artifact inspection where static patch
    datasets must still be read.
    By default, this builder:
    - enumerates items via `get_items_patches` (images under `image_patches/`),
    - resolves masks via `get_y_patch` (under `mask_patches/`, with sensible fallbacks),
    - uses an 80/20 `RandomSplitter`,
    - applies basic resize and normalization.

    Args:
        codes: Label codes for MaskBlock (binary by default)
        get_items: Optional override for item listing; defaults to `get_items_patches`
        get_y_func: Optional override for mask resolver; defaults to `get_y_patch`
        splitter: Train/valid splitter (defaults to 80/20 RandomSplitter)
        item_tfms: Item transforms for augmentation (defaults to segmentation-appropriate transforms)
        batch_tfms: Batch transforms for normalization (defaults to standard transforms)
    """
    if codes is None:
        codes = [0, 1]
    
    if get_items is None:
        get_items = get_items_patches

    if splitter is None:
        splitter = RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42)
    
    if get_y_func is None:
        get_y_func = default_get_y_path

    # Modern FastAI v2 best practices: augmentations in item_tfms, normalization in batch_tfms
    if item_tfms is None:
        item_tfms = [
            Resize(DEFAULT_IMAGE_SIZE),
        ]

    if batch_tfms is None:
        batch_tfms = [
            IntToFloatTensor(),  # MUST be first in batch_tfms for augmentations to work
            *aug_transforms(
                size=DEFAULT_IMAGE_SIZE,
                max_rotate=DEFAULT_MAX_ROTATE,
                flip_vert=DEFAULT_FLIP_VERT,
                min_zoom=DEFAULT_MIN_ZOOM,
                max_zoom=DEFAULT_MAX_ZOOM,
                max_warp=DEFAULT_MAX_WARP,
                # Enable lighting augmentation for medical imaging robustness
                max_lighting=DEFAULT_MAX_LIGHTING,
            ),
            # Use ImageNet normalization - critical for transfer learning performance
            Normalize.from_stats(*imagenet_stats),
            MaskPreprocessTransform(),
        ]

    # Use list for blocks parameter as expected by FastAI v2
    block = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=codes)],  # Use codes for proper segmentation
        get_items=get_items,
        get_y=get_y_func,
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    return block


def build_segmentation_dls(data_root: Union[str, Path], bs: int = 8, num_workers: int = 0):
    """
    Create legacy DataLoaders for binary segmentation using static patches.

    This is not a supported model-training entrypoint. Supported training uses
    `build_segmentation_dls_dynamic_patching` with a validated full-image root.
    
    Args:
        data_root: Path to legacy data directory with image_patches/ and mask_patches/
        bs: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoaders object
    """
    # Create DataLoaders using static patches
    db = build_segmentation_datablock()
    dls = db.dataloaders(Path(data_root), bs=bs, num_workers=num_workers)

    # Preflight: enforce 1:1 mapping between image_patches and mask_patches
    try:
        root = Path(data_root)
        img_dir = root / 'image_patches'
        msk_dir = root / 'mask_patches'
        if img_dir.exists() and msk_dir.exists():
            img_paths = list(get_image_files(img_dir))
            # Build expected mask paths map
            missing = []
            for p in img_paths[:10000]:  # cap to avoid extreme scans
                try:
                    get_y_patch(p)
                except FileNotFoundError:
                    expected = msk_dir / f"{p.stem}_mask{p.suffix}"
                    missing.append((p, expected))
            if missing:
                examples = "\n".join([f"- image: {im} -> expected mask: {em}" for im, em in missing[:10]])
                raise ValueError(
                    "1:1 mapping check failed: some masks are missing.\n" +
                    f"Checked {min(len(img_paths),10000)} images, missing: {len(missing)}.\n" +
                    examples
                )

            # Mask content sanity check on a small validation sample
            try:
                logger = get_logger("eq.datablock_loader")
                val_items = list(getattr(dls.valid_ds, 'items', []))
                sample_items = val_items[:64]
                checked = 0
                all_zero = 0
                some_positive = 0
                for ip in sample_items:
                    ip = Path(ip)
                    try:
                        mp = get_y_patch(ip)
                    except FileNotFoundError:
                        continue
                    m = np.array(PILMask.create(mp))
                    if m.max() == 0:
                        all_zero += 1
                    else:
                        some_positive += 1
                    checked += 1
                if checked > 0 and all_zero/checked > 0.5:
                    logger.warning(f"⚠️  High empty-mask rate in validation: {all_zero/checked:.2%} ({all_zero}/{checked})")
                else:
                    logger.info(f"✅ Mask validation: {some_positive}/{checked} masks have positive pixels ({some_positive/checked:.1%})")
            except Exception:
                # Non-fatal; continue
                pass
    except Exception as _e:
        # Surface mapping issues clearly
        raise
    return dls


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
    

def get_items_patches(path: Path) -> List[Any]:
    """
    List images under `image_patches/` that have resolvable masks via `get_y_patch`.

    - Directory: expects `<data_root>/image_patches/`.
    - Validation: keeps only images for which `get_y_patch(image)` resolves a mask path.
    - Logging: reports counts of kept vs. skipped images.
    """
    images_dir = Path(path) / "image_patches"
    if not images_dir.exists():
        # Be explicit for the static patching pipeline; avoid silently falling back
        raise FileNotFoundError(f"image_patches/ directory not found in {path}. Expected: {images_dir}")

    all_images = get_image_files(images_dir)
    valid_images: List[Any] = []
    skipped_count = 0
    logger = get_logger("eq.datablock_loader")

    for img_path in all_images:
        try:
            # Only check that a mask can be resolved; do not load it
            get_y_patch(img_path)
            valid_images.append(img_path)
        except FileNotFoundError:
            skipped_count += 1
            logger.warning(f"Skipping image {Path(img_path).name} - no corresponding mask found")
            continue

    logger.info(
        f"DataBlock (static patches): Found {len(valid_images)} valid image-mask pairs, "
        f"skipped {skipped_count} images without masks"
    )
    return valid_images

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
    
    # Get all image files
    all_images = get_image_files(images_dir)
    valid_images = []
    skipped_count = 0
    
    logger = get_logger("eq.datablock_loader")
    
    skipped_samples = []
    for img_path in all_images:
        # Use the full-image getter to check if mask exists
        try:
            # This will raise FileNotFoundError if no mask is found
            from eq.data_management.standard_getters import get_y_full as _get_y_full
            _get_y_full(img_path)  # Just check if it exists, don't load it yet
            valid_images.append(img_path)
        except Exception:
            # Fallback: try the most common expected path directly to be extra tolerant
            try:
                img_path = Path(img_path)
                data_root = img_path.parent.parent  # .../images/<rel> → data_root
                rel = img_path.parent.relative_to(data_root / "images") if (data_root / "images").exists() else Path("")
                stem = img_path.stem
                # Prefer same extension as image, then png
                primary_ext = img_path.suffix.lower() or ".jpg"
                candidates = []
                masks_root = data_root / "masks"
                for ext in [primary_ext, ".png", ".jpg", ".jpeg"]:
                    # masks/<rel>/<stem>_mask<ext>
                    candidates.append((masks_root / rel / f"{stem}_mask{ext}"))
                    # masks/<rel>/mask_<stem><ext>
                    candidates.append((masks_root / rel / f"mask_{stem}{ext}"))
                hit = next((c for c in candidates if c.exists()), None)
                if hit is not None:
                    valid_images.append(img_path)
                    continue
            except Exception:
                pass
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
):
    """
    Create a DataBlock for binary segmentation with dynamic patching.
    
    Key differences from static patching:
    1. Loads full images from images/ directory (not patches)
    2. Applies augmentations to full images first
    3. Crops 256x256 patches after augmentation
    4. Much better augmentation diversity and performance
    
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
        splitter = RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42)
    
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
    
    # Batch transforms: apply additional augmentations with ImageNet normalization
    batch_tfms = [
        IntToFloatTensor(),
        *aug_transforms(
            # Do not pass size here; avoid RandomResizedCrop-like behavior
            max_rotate=int(DEFAULT_MAX_ROTATE),
            flip_vert=DEFAULT_FLIP_VERT,
            min_zoom=DEFAULT_MIN_ZOOM,
            max_zoom=DEFAULT_MAX_ZOOM,
            max_warp=DEFAULT_MAX_WARP,
            max_lighting=DEFAULT_MAX_LIGHTING,
        ),
        Normalize.from_stats(*imagenet_stats),
        MaskPreprocessTransform(),
    ]
    
    # Create DataBlock for full images with dynamic patching
    # Use the full-image getter
    from eq.data_management.standard_getters import get_y_full as _get_y_full
    block = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=codes)],  # Use codes for proper segmentation
        get_items=get_items_full_images,
        get_y=_get_y_full,
        splitter=splitter,
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
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
    )
    
    # Create DataLoaders
    dls = db.dataloaders(root, bs=bs, num_workers=num_workers)
    
    logger = get_logger("eq.datablock_loader")
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
