from pathlib import Path

import pandas as pd
import pytest

from eq.utils.paths import get_data_path


def _get_raw_data_root() -> Path:
    raw_data_root = get_data_path()
    if not raw_data_root.exists():
        pytest.skip(f'Raw data directory not present: {raw_data_root}')
    return raw_data_root


def test_raw_data_root_exists():
    raw_data_root = _get_raw_data_root()
    assert raw_data_root.is_dir()


def test_lauren_preeclampsia_cohort_structure_if_present():
    raw_data_root = _get_raw_data_root()
    project_dir = raw_data_root / 'cohorts' / 'lauren_preeclampsia'
    if not project_dir.exists():
        pytest.skip(f'Lauren preeclampsia cohort not present under {raw_data_root}')

    assert (project_dir / 'images').is_dir()
    assert (project_dir / 'masks').is_dir()


def test_subject_metadata_is_readable_if_present():
    raw_data_root = _get_raw_data_root()
    metadata_file = (
        raw_data_root
        / 'cohorts'
        / 'lauren_preeclampsia'
        / 'metadata'
        / 'subject_metadata.xlsx'
    )
    if not metadata_file.exists():
        pytest.skip(f'Metadata file not present: {metadata_file}')

    dataframe = pd.read_excel(metadata_file, nrows=5)
    assert list(dataframe.columns)


def test_raw_data_contains_nonempty_images_if_present():
    raw_data_root = _get_raw_data_root()
    image_root = raw_data_root / 'cohorts' / 'lauren_preeclampsia' / 'images'
    if not image_root.exists():
        pytest.skip(f'Project image tree not present: {image_root}')

    image_files = []
    for pattern in ('*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png'):
        image_files.extend(image_root.rglob(pattern))

    if not image_files:
        pytest.skip(f'No image files found under {image_root}')

    assert max(path.stat().st_size for path in image_files) > 0
