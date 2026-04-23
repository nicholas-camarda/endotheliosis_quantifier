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


def test_mitochondria_full_image_layout_if_present():
    derived_data_root = _get_derived_data_root()
    dataset_dir = derived_data_root / "mitochondria_data"
    if not dataset_dir.exists():
        pytest.skip(f"Dataset not present: {dataset_dir}")

    for split in ("training", "testing"):
        split_dir = dataset_dir / split
        assert split_dir.is_dir()
        assert (split_dir / "images").is_dir()
        assert (split_dir / "masks").is_dir()
        assert not (split_dir / "image_patches").exists()
        assert not (split_dir / "mask_patches").exists()
        assert not (split_dir / "image_patch_validation").exists()
        assert not (split_dir / "mask_patch_validation").exists()


def test_segmentation_static_patch_dirs_are_not_active_if_present():
    derived_data_root = _get_derived_data_root()
    forbidden_names = {
        "image_patches",
        "mask_patches",
        "image_patch_validation",
        "mask_patch_validation",
    }

    for dataset_name in ("mitochondria_data", "glomeruli_data"):
        dataset_dir = derived_data_root / dataset_name
        if not dataset_dir.exists():
            continue
        active_static_dirs = [
            path
            for path in dataset_dir.rglob("*")
            if path.is_dir() and path.name in forbidden_names
        ]
        assert active_static_dirs == []


def test_projects_runtime_segmentation_static_patch_dirs_are_retired_when_runtime_present():
    runtime_root = Path.home() / "ProjectsRuntime" / "endotheliosis_quantifier" / "derived_data"
    if not runtime_root.exists():
        pytest.skip(f"ProjectsRuntime derived data root not present: {runtime_root}")

    forbidden_names = {
        "image_patches",
        "mask_patches",
        "image_patch_validation",
        "mask_patch_validation",
    }
    for dataset_name in ("mitochondria_data", "glomeruli_data"):
        dataset_dir = runtime_root / dataset_name
        if not dataset_dir.exists():
            continue
        active_static_dirs = [
            path
            for path in dataset_dir.rglob("*")
            if path.is_dir() and path.name in forbidden_names
        ]
        assert active_static_dirs == []


def test_cache_directories_are_directories_when_present():
    derived_data_root = _get_derived_data_root()
    for dataset_name in ("mitochondria_data", "glomeruli_data"):
        cache_dir = derived_data_root / dataset_name / "cache"
        if cache_dir.exists():
            assert cache_dir.is_dir()
