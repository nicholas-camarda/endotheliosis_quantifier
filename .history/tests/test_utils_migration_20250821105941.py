import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


def create_dummy_tif(path: Path, size: tuple[int, int] = (8, 8)) -> None:
	arr = (np.arange(size[0] * size[1]).reshape(size).astype(np.uint8))
	img = Image.fromarray(arr)
	img.save(path, format='TIFF')


def create_dummy_jpg(path: Path, size: tuple[int, int] = (8, 8)) -> None:
	arr = (np.arange(size[0] * size[1]).reshape(size).astype(np.uint8))
	img = Image.fromarray(arr)
	img.save(path, format='JPEG')


def test_convert_tif_to_jpg(tmp_path: Path):
	from eq.io.convert_files_to_jpg import convert_tif_to_jpg

	inp = tmp_path / 'in' / 'A'
	out = tmp_path / 'out'
	inp.mkdir(parents=True)
	# two tifs
	create_dummy_tif(inp / 'one.tif')
	create_dummy_tif(inp / 'two.tif')

	convert_tif_to_jpg(str(inp.parent), str(out))

	# Expect files in out/A/ named A_one.jpg and A_two.jpg
	expected = { 'A_one.jpg', 'A_two.jpg' }
	produced = { p.name for p in (out / 'A').glob('*.jpg') }
	assert produced == expected


def test_patchify_image_dir(tmp_path: Path):
	from eq.patches.patchify_images import patchify_image_dir

	inp = tmp_path / 'in'
	out = tmp_path / 'out'
	inp.mkdir()
	# 8x8 image -> with square_size=4 expect 4 patches
	create_dummy_jpg(inp / 'img.jpg', size=(8, 8))

	patchify_image_dir(4, str(inp), str(out))

	files = list(out.glob('*.jpg'))
	names = { p.name for p in files }
	assert len(files) == 4
	assert {'img_0_0.jpg', 'img_0_1.jpg', 'img_1_0.jpg', 'img_1_1.jpg'} == names


def test_preprocess_images_extracts_rois(tmp_path: Path):
	from eq.features.preprocess_roi import preprocess_images

	top = tmp_path
	images_dir = top / 'train' / 'images' / 'P01'
	masks_dir = top / 'train' / 'masks' / 'P01'
	roi_dir = top / 'train' / 'rois'
	masks_dir.mkdir(parents=True)
	images_dir.mkdir(parents=True)
	roi_dir.mkdir(parents=True)

	# Create simple 256x256 image and circular mask
	img = np.zeros((256, 256, 3), dtype=np.uint8)
	cv2.rectangle(img, (80, 80), (176, 176), (255, 255, 255), -1)
	mask = np.zeros((256, 256), dtype=np.uint8)
	cv2.circle(mask, (128, 128), 40, 255, -1)

	cv2.imwrite(str(images_dir / 'img0.jpg'), img)
	cv2.imwrite(str(masks_dir / 'img0_mask.jpg'), mask)

	rois = preprocess_images(str(images_dir.parent), str(masks_dir.parent), str(roi_dir))
	# Expect at least one ROI file and non-empty array
	produced = list((roi_dir / 'P01').glob('*.jpg'))
	assert len(produced) >= 1
	assert rois is None or rois.shape[0] >= 1
