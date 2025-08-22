# patchify_images.py

import os

from skimage import io


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
            img = io.imread(input_path)
            h, w = img.shape[:2]
            for i in range(0, h, square_size):
                for j in range(0, w, square_size):
                    patch = img[i:i+square_size, j:j+square_size]
                    if patch.shape[0] != square_size or patch.shape[1] != square_size:
                        continue
                    output_filename = f"{os.path.splitext(filename)[0]}_{i//square_size}_{j//square_size}{ext_}"
                    output_path = os.path.join(output_dir, output_filename)
                    io.imsave(output_path, patch)
