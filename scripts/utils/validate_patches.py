# validate_patches.py

import sys
from patchify import patchify, unpatchify
from skimage import io
import os
import numpy as np

patch_size = int(sys.argv[1])
original_image_size = tuple(map(int, sys.argv[2].split(',')))
input_dir = sys.argv[3]
validation_dir = sys.argv[4]

if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

# Collect all patches for each original image
image_patches = {}
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith('.tif'):
        # Extract the original image name and patch number
        image_name = filename.rsplit('_', 2)[0]
        patch_number = 'patch_' + (filename.rsplit(
            '_', 2)[1] + "_" + filename.rsplit('_', 2)[2]).rsplit('.', 1)[0]
        # print(image_name)
        # print(patch_number)
        filepath = os.path.join(input_dir, filename)
        patch = io.imread(filepath)
        if image_name not in image_patches:
            image_patches[image_name] = {}
        image_patches[image_name][patch_number] = np.array(patch)

# for key, value in image_patches.items():
#     print(key, [(k, _) for k, _ in value.items()])
#     break


# print(image_patches)
# Reconstruct each original image from its patches
print("Now reconstructing images...")
for image_num, patches in image_patches.items():
    patches_for_this_image = []
    for patch_keys in sorted(patches.keys()):
        for j in patches[patch_keys]:
            patches_for_this_image.append(j)

    patches_for_this_image = np.array(
        patches_for_this_image).reshape(original_image_size[0]//patch_size,
                                        original_image_size[1]//patch_size,
                                        patch_size, patch_size)
    # print(patches_for_this_image.shape)
    # break
    img_reconstructed = unpatchify(patches_for_this_image, original_image_size)
    validation_path = os.path.join(
        validation_dir, f"{image_num}.tif")
    io.imsave(validation_path, img_reconstructed)
    print(f"Saved reconstructed image {image_num}")
