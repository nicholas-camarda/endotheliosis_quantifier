#!/usr/bin/env python3
"""Organize the Lucchi dataset into the repository's expected layout."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tifffile

from eq.utils.paths import get_runtime_mitochondria_data_path

logger = logging.getLogger(__name__)


def extract_tif_stack(tif_path: Path, output_dir: Path, prefix: str) -> int:
    """Extract a stacked TIF into individual files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(tif_path) as tif:
        count = len(tif.pages)
        for index, page in enumerate(tif.pages):
            output_path = output_dir / f'{prefix}_{index}.tif'
            tifffile.imwrite(output_path, page.asarray())

    logger.info('Extracted %s images from %s into %s', count, tif_path, output_dir)
    return count


def organize_lucchi_dataset(input_dir: str, output_dir: str | None = None) -> Path:
    """Convert Lucchi train/test TIF stacks into training/testing image directories."""
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else get_runtime_mitochondria_data_path()

    img_dir = input_path / 'img'
    label_dir = input_path / 'label'
    train_img_file = img_dir / 'train_im.tif'
    train_label_file = label_dir / 'train_label.tif'
    test_img_file = img_dir / 'test_im.tif'
    test_label_file = label_dir / 'test_label.tif'

    required_files = [train_img_file, train_label_file, test_img_file, test_label_file]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Lucchi dataset is incomplete. Missing: {missing}')

    training_images_dir = output_path / 'training' / 'images'
    training_masks_dir = output_path / 'training' / 'masks'
    testing_images_dir = output_path / 'testing' / 'images'
    testing_masks_dir = output_path / 'testing' / 'masks'
    cache_dir = output_path / 'cache'

    for path in [
        training_images_dir,
        training_masks_dir,
        testing_images_dir,
        testing_masks_dir,
        cache_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    extract_tif_stack(train_img_file, training_images_dir, 'training')
    extract_tif_stack(train_label_file, training_masks_dir, 'training_groundtruth')
    extract_tif_stack(test_img_file, testing_images_dir, 'testing')
    extract_tif_stack(test_label_file, testing_masks_dir, 'testing_groundtruth')

    readme_path = output_path / 'README.md'
    readme_path.write_text(
        '\n'.join(
            [
                '# Lucchi Dataset Layout',
                '',
                'This directory contains the Lucchi mitochondria dataset reorganized for local EQ workflows.',
                '',
                'Structure:',
                '```text',
                'mitochondria_data/',
                '├── training/',
                '│   ├── images/',
                '│   └── masks/',
                '├── testing/',
                '│   ├── images/',
                '│   └── masks/',
                '└── cache/',
                '```',
                '',
                'Each stacked TIF from the original dataset has been expanded into individual `.tif` files.',
            ]
        ),
        encoding='utf-8',
    )

    logger.info('Lucchi dataset organized under %s', output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Organize the Lucchi dataset for EQ.')
    parser.add_argument('--input-dir', required=True, help='Directory containing Lucchi img/ and label/ folders.')
    parser.add_argument(
        '--output-dir',
        default=str(get_runtime_mitochondria_data_path()),
        help='Output directory for the organized dataset.',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    organize_lucchi_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
