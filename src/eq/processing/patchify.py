# patchify_images.py

import os

import cv2
import numpy as np


def patchify_image_dir(square_size, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        ext_ = os.path.splitext(filename)[1]  # includes '.'
        input_path = os.path.join(input_dir, filename)
        if os.path.isdir(input_path):
            # recursively process subdirectories
            output_subdir = os.path.join(output_dir, filename)
            if not os.path.exists(output_subdir):
                os.mkdir(output_subdir)
            patchify_image_dir(square_size, input_path, output_subdir)
        elif filename.lower().endswith(('.tif', '.jpg', '.jpeg', '.png')):
            img = cv2.imread(input_path)
            h, w = img.shape[:2]
            for i in range(0, h, square_size):
                for j in range(0, w, square_size):
                    patch = img[i:i+square_size, j:j+square_size]
                    if patch.shape[0] != square_size or patch.shape[1] != square_size:
                        continue
                    output_filename = f"{os.path.splitext(filename)[0]}_{i//square_size}_{j//square_size}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, patch)


def patchify_image_and_mask_dirs(square_size, image_dir, mask_dir, output_dir):
    """
    Patchify images and (optionally) corresponding masks, writing both into the
    same subject directory. Mask patches get a `_mask` suffix in the filename.

    - image_dir: directory containing image files (jpg/tif)
    - mask_dir: directory containing mask files with *_mask.* naming; may be None or missing
    - output_dir: directory to write patches for this subject
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.tif', '.jpg', '.jpeg', '.png'))]
    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Attempt to locate corresponding mask
        base_stem = os.path.splitext(filename)[0]  # e.g., T19_Image0
        mask_path = None
        if mask_dir and os.path.exists(mask_dir):
            # Look for exact *_mask.* file
            for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                candidate = os.path.join(mask_dir, f"{base_stem}_mask{ext}")
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

        mask_img = None
        if mask_path is not None:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # If mask size mismatches, skip mask writing but still write image patches
            if mask_img is not None and (mask_img.shape[0] != h or mask_img.shape[1] != w):
                mask_img = None

        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                patch = img[i:i+square_size, j:j+square_size]
                if patch.shape[0] != square_size or patch.shape[1] != square_size:
                    continue
                out_name = f"{base_stem}_{i//square_size}_{j//square_size}.jpg"
                cv2.imwrite(os.path.join(output_dir, out_name), patch)

                if mask_img is not None:
                    mask_patch = mask_img[i:i+square_size, j:j+square_size]
                    if mask_patch.shape[0] != square_size or mask_patch.shape[1] != square_size:
                        continue
                    out_mask_name = f"{base_stem}_{i//square_size}_{j//square_size}_mask.jpg"
                    cv2.imwrite(os.path.join(output_dir, out_mask_name), mask_patch)