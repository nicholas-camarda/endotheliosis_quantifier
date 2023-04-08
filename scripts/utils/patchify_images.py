# patchify_images.py

import sys
from patchify import patchify
from skimage import io
import os

square_size = int(sys.argv[1])
input_dir = sys.argv[2]
output_dir = sys.argv[3]

# input_dir = 'mitochondria_data/testing/masks'
# output_dir = 'mitochondria_data/testing/mask_patches'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        filepath = os.path.join(input_dir, filename)
        img = io.imread(filepath)
        print(f"The filename is: {filename}")
        print(f"The image shape is: {img.shape}")

        patches = patchify(img, (square_size, square_size),
                           step=(square_size, square_size))

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j]
                output_path = os.path.join(
                    output_dir, f"{filename.split('.')[0]}_{i}_{j}.tif")
                io.imsave(output_path, patch)
                print(f"Saved patch {i}_{j}")
