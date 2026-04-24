#!/usr/bin/env python3
"""
Standard Getter Functions for FastAI v2.

Roles:
- get_y_full: resolve mask path for a full image in `images/` (used by dynamic patching).

The public getter returns a Path to the corresponding mask or raises
FileNotFoundError if not found.
"""

from pathlib import Path
from typing import List

from eq.utils.logger import get_logger


def _mask_matches_image_stem(mask_path: Path, image_stem: str) -> bool:
    """Check mask/image pairing by exact normalized stem match."""
    mask_stem = mask_path.stem
    candidates = {
        mask_stem,
        mask_stem.removesuffix("_mask"),
        mask_stem.removeprefix("mask_"),
    }
    return image_stem in candidates


def get_y_full(x):
    """
    Resolve full-sized mask for a given image under data_root/images/ → data_root/masks/.

    Directory assumptions:
      - Image under `<data_root>/images/...`.
      - Mask under `<data_root>/masks/...`.

    Naming patterns tried (preserving subdirectories when present):
      - `<stem>_mask<ext>`
      - `mask_<stem><ext>`
      - Fallback: glob any file containing both `<stem>` and `mask`.

    Returns a Path or raises FileNotFoundError if no mask is found.
    """
    log = get_logger("eq.standard_getters")
    img_path = Path(x)

    # Find dataset root by locating the 'images' directory and stepping up one level
    p = img_path
    data_root = None
    for _ in range(5):
        if p.name == 'images':
            data_root = p.parent
            break
        p = p.parent
    if data_root is None:
        # Fallback to two levels up (Txx -> images -> root) for typical layout
        data_root = img_path.parent.parent.parent
    masks_root = data_root / "masks"

    cand_dirs: List[Path] = [masks_root]
    # Attempt to mirror substructure under images/
    try:
        rel = img_path.parent.relative_to(data_root / "images")
        cand_dirs += [masks_root / rel]
    except Exception:
        pass

    stem = img_path.stem
    names = [
        f"{stem}_mask",
        f"mask_{stem}",
        stem.replace("img_", "mask_"),
        stem.replace("training_", "training_groundtruth_", 1),
        stem.replace("testing_", "testing_groundtruth_", 1),
        stem.replace("train_", "train_label_", 1),
        stem.replace("test_", "test_label_", 1),
    ]

    # Build extension candidates
    ext_candidates = [".png", img_path.suffix.lower(), ".jpg", ".jpeg", ".tif", ".tiff"]
    seen = set()
    exts = []
    for e in ext_candidates:
        if e and e not in seen:
            exts.append(e)
            seen.add(e)

    tried = []
    for d in cand_dirs:
        for nm in names:
            for ext in exts:
                p = d / f"{nm}{ext}"
                tried.append(p)
                if p.exists():
                    return p

    # Fallback exact normalized-stem search
    for d in cand_dirs:
        if d.exists():
            for candidate in d.iterdir():
                if candidate.is_file() and _mask_matches_image_stem(candidate, stem):
                    return candidate

    raise FileNotFoundError(
        f"❌ No mask found for {img_path.name}. Looked under 'masks/' with common patterns."
    )
