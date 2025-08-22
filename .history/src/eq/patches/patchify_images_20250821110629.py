# patchify_images.py

import os

from patchify import patchify
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
        elif filename.endswith('.tif') or filename.endswith('.jpg'):
            img = io.imread(input_path)
            print(f"The filename is: {filename}")
            print(f"The image shape is: {img.shape}")

            patches = patchify(img, (square_size, square_size),
                               step=(square_size, square_size))

            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j]
                    output_filename = f"{os.path.splitext(filename)[0]}_{i}_{j}{ext_}"
                    output_path = os.path.join(output_dir, output_filename)
                    io.imsave(output_path, patch)
                    print(f"Saved patch {output_filename}")
