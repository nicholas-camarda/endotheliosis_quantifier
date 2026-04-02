from pathlib import Path

import pytest

from eq.utils.paths import get_output_path


def _get_derived_data_root() -> Path:
    derived_data_root = get_output_path()
    if not derived_data_root.exists():
        pytest.skip(f"Derived data directory not present: {derived_data_root}")
    return derived_data_root


def test_derived_data_root_exists():
    derived_data_root = _get_derived_data_root()
    assert derived_data_root.is_dir()


@pytest.mark.parametrize("dataset_name", ["mitochondria_data", "glomeruli_data"])
def test_dataset_training_layout_if_present(dataset_name: str):
    derived_data_root = _get_derived_data_root()
    dataset_dir = derived_data_root / dataset_name
    if not dataset_dir.exists():
        pytest.skip(f"Dataset not present: {dataset_dir}")

    training_dir = dataset_dir / "training"
    assert training_dir.is_dir()
    assert (training_dir / "image_patches").is_dir()
    assert (training_dir / "mask_patches").is_dir()


def test_any_patch_dataset_has_consistent_patch_counts():
    derived_data_root = _get_derived_data_root()
    candidates = []

    for dataset_dir in derived_data_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        direct_images = dataset_dir / "image_patches"
        direct_masks = dataset_dir / "mask_patches"
        if direct_images.is_dir() and direct_masks.is_dir():
            candidates.append((direct_images, direct_masks))

        nested_training = dataset_dir / "training"
        nested_images = nested_training / "image_patches"
        nested_masks = nested_training / "mask_patches"
        if nested_images.is_dir() and nested_masks.is_dir():
            candidates.append((nested_images, nested_masks))

    if not candidates:
        pytest.skip(f"No patch dataset layouts found under {derived_data_root}")

    image_patches_dir, mask_patches_dir = candidates[0]
    image_patches = sorted(path for path in image_patches_dir.iterdir() if path.is_file())
    mask_patches = sorted(path for path in mask_patches_dir.iterdir() if path.is_file())
    if not image_patches or not mask_patches:
        pytest.skip(f"No patch files present under {image_patches_dir.parent}")

    assert len(mask_patches) == len(image_patches)


def test_cache_directories_are_directories_when_present():
    derived_data_root = _get_derived_data_root()
    for dataset_name in ("mitochondria_data", "glomeruli_data"):
        cache_dir = derived_data_root / dataset_name / "cache"
        if cache_dir.exists():
            assert cache_dir.is_dir()
