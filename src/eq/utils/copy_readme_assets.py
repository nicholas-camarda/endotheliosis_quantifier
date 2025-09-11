#!/usr/bin/env python3
"""
Utility to copy key training artifacts to the assets/ folder for README embeddings.

It locates the most recent mitochondria model and the most recent glomeruli
models (transfer and scratch), then copies:
- training_loss.png
- validation_predictions.png

Destination structure (created if missing):
assets/
  mitochondria/
    training_loss.png
    validation_predictions.png
  glomeruli/
    transfer/
      training_loss.png
      validation_predictions.png
    scratch/
      training_loss.png
      validation_predictions.png

Run:
  python -m eq.utils.copy_readme_assets \
    --models-root models/segmentation \
    --assets-root assets

You can also override any of the specific model roots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import shutil


def _find_latest_dir(root: Path) -> Optional[Path]:
    """Return the newest directory (by mtime) under root, recursively if needed.

    We look for leaf directories containing PNG artifacts to avoid picking plain parents.
    """
    if not root.exists():
        return None

    candidates: List[Tuple[float, Path]] = []
    for p in root.rglob("*"):
        if p.is_dir():
            # Heuristic: prefer dirs that already have at least one PNG artifact
            has_png = any((p / f).exists() for f in p.glob("*.png"))
            if has_png:
                candidates.append((p.stat().st_mtime, p))
    if not candidates:
        # Fallback: just pick the newest dir anywhere under root
        for p in root.rglob("*"):
            if p.is_dir():
                candidates.append((p.stat().st_mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _copy_if_exists(src: Path, dst: Path) -> bool:
    try:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
    except Exception:
        pass
    return False


def copy_assets(models_root: Path, assets_root: Path,
                mito_root: Optional[Path] = None,
                glom_transfer_root: Optional[Path] = None,
                glom_scratch_root: Optional[Path] = None) -> int:
    """Copy artifacts into assets/.

    Returns process exit code: 0 on success (even if some files missing), 1 on fatal error.
    """
    try:
        mito_root = mito_root or (models_root / "mitochondria")
        glom_root = models_root / "glomeruli"
        glom_transfer_root = glom_transfer_root or (glom_root / "transfer")
        glom_scratch_root = glom_scratch_root or (glom_root / "scratch")

        # Find latest model directories
        mito_dir = _find_latest_dir(mito_root) or mito_root
        transfer_dir = _find_latest_dir(glom_transfer_root) or glom_transfer_root
        scratch_dir = _find_latest_dir(glom_scratch_root) or glom_scratch_root

        # Define artifact names we expect to exist (we keep training_loss.png per README)
        png_names = [
            ("training_loss.png", "training_loss.png"),
            ("validation_predictions.png", "validation_predictions.png"),
        ]

        # Copy mitochondria
        for src_name, dst_name in png_names:
            _copy_if_exists(
                next((p for p in mito_dir.glob(f"**/*{src_name}") if p.is_file()), mito_dir / src_name),
                assets_root / "mitochondria" / dst_name,
            )

        # Copy transfer glomeruli
        for src_name, dst_name in png_names:
            _copy_if_exists(
                next((p for p in transfer_dir.glob(f"**/*{src_name}") if p.is_file()), transfer_dir / src_name),
                assets_root / "glomeruli" / "transfer" / dst_name,
            )

        # Copy scratch glomeruli
        for src_name, dst_name in png_names:
            _copy_if_exists(
                next((p for p in scratch_dir.glob(f"**/*{src_name}") if p.is_file()), scratch_dir / src_name),
                assets_root / "glomeruli" / "scratch" / dst_name,
            )

        return 0
    except Exception:
        return 1


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Copy training artifacts to assets for README")
    parser.add_argument("--models-root", default="models/segmentation", help="Root of segmentation models")
    parser.add_argument("--assets-root", default="assets", help="Destination assets root")
    parser.add_argument("--mito-root", help="Override mitochondria models root")
    parser.add_argument("--glom-transfer-root", help="Override glomeruli transfer models root")
    parser.add_argument("--glom-scratch-root", help="Override glomeruli scratch models root")
    args = parser.parse_args(argv)

    exit_code = copy_assets(
        models_root=Path(args.models_root),
        assets_root=Path(args.assets_root),
        mito_root=Path(args.mito_root) if args.mito_root else None,
        glom_transfer_root=Path(args.glom_transfer_root) if args.glom_transfer_root else None,
        glom_scratch_root=Path(args.glom_scratch_root) if args.glom_scratch_root else None,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


