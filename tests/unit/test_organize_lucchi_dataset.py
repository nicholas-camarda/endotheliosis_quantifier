from pathlib import Path

import numpy as np
import tifffile

from eq.data_management.organize_lucchi_dataset import organize_lucchi_dataset


def _write_stack(path: Path, depth: int = 3) -> None:
    stack = np.stack(
        [np.full((8, 8), fill_value=index, dtype=np.uint8) for index in range(depth)],
        axis=0,
    )
    tifffile.imwrite(path, stack, photometric='minisblack')


def test_organize_lucchi_dataset(tmp_path: Path):
    input_dir = tmp_path / 'lucchi'
    (input_dir / 'img').mkdir(parents=True)
    (input_dir / 'label').mkdir(parents=True)

    _write_stack(input_dir / 'img' / 'train_im.tif')
    _write_stack(input_dir / 'label' / 'train_label.tif')
    _write_stack(input_dir / 'img' / 'test_im.tif')
    _write_stack(input_dir / 'label' / 'test_label.tif')

    output_dir = tmp_path / 'organized'
    result = organize_lucchi_dataset(str(input_dir), str(output_dir))

    assert result == output_dir
    assert sorted((output_dir / 'training' / 'images').glob('*.tif'))
    assert sorted((output_dir / 'training' / 'masks').glob('*.tif'))
    assert sorted((output_dir / 'testing' / 'images').glob('*.tif'))
    assert sorted((output_dir / 'testing' / 'masks').glob('*.tif'))
    assert (output_dir / 'cache').exists()
    readme_path = output_dir / 'README.md'
    assert readme_path.exists()
    assert 'mitochondria_data/' in readme_path.read_text(encoding='utf-8')
