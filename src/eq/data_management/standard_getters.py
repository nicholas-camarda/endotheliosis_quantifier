#!/usr/bin/env python3
"""
Standard Getter Functions for FastAI v2.

Roles:
- get_y_full: resolve mask path for a full image in `images/` (used by dynamic patching).

The public getter returns a Path to the corresponding mask or raises
FileNotFoundError if not found.
"""

from pathlib import Path


def get_y_full(x):
    """
    Resolve full-sized mask for a given image under data_root/images/ → data_root/masks/.

    Directory assumptions:
      - Image under `<data_root>/images/...`.
      - Mask under `<data_root>/masks/...`.

    Naming patterns tried (preserving subdirectories when present):
      - `<stem>_mask<ext>`
      - `mask_<stem><ext>`

    Returns a Path or raises FileNotFoundError if no mask is found.
    """
    img_path = Path(x)

    images_index = None
    for index, part in enumerate(img_path.parts):
        if part == "images":
            images_index = index
            break
    if images_index is None:
        raise FileNotFoundError(
            f"No mask found for {img_path}. Expected image path under `<data_root>/images/...`."
        )

    data_root = Path(*img_path.parts[:images_index])
    images_root = data_root / "images"
    masks_root = data_root / "masks"
    try:
        rel_parent = img_path.parent.relative_to(images_root)
    except ValueError as exc:
        raise FileNotFoundError(
            f"No mask found for {img_path}. Expected image path under `{images_root}`."
        ) from exc
    mask_dir = masks_root / rel_parent

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
    for nm in names:
        for ext in exts:
            p = mask_dir / f"{nm}{ext}"
            tried.append(p)
            if p.exists():
                return p

    raise FileNotFoundError(
        f"No mask found for {img_path}. Looked only under mirrored mask directory {mask_dir}."
    )
