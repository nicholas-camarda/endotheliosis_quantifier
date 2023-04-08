#!/bin/bash
# i know this is kind of a lame solution..
# need to run this from home directory e.g., sh scripts/utils/run_split_images_and_get_patches.sh 

# Set patch size and original image size
patch_size=256
original_image_size="768,1024"

# set up all the directories...
input_dir_training_images="data/mitochondria_data/training/images"
output_dir_training_images="data/mitochondria_data/training/image_patches"
validation_dir_training_images="data/mitochondria_data/training/image_patch_validation"
training_images_file="data/mitochondria_data/training.tif"

input_dir_training_masks="data/mitochondria_data/training/masks"
output_dir_training_masks="data/mitochondria_data/training/mask_patches"
validation_dir_training_masks="data/mitochondria_data/training/mask_patch_validation"
training_mask_file="data/mitochondria_data/training_groundtruth.tif"

input_dir_testing_images="data/mitochondria_data/testing/images"
output_dir_testing_images="data/mitochondria_data/testing/image_patches"
validation_dir_testing_images="data/mitochondria_data/testing/image_patch_validation"
testing_images_file="data/mitochondria_data/testing.tif"

input_dir_testing_masks="data/mitochondria_data/testing/masks"
output_dir_testing_masks="data/mitochondria_data/testing/mask_patches"
validation_dir_testing_masks="data/mitochondria_data/testing/mask_patch_validation"
testing_mask_file="data/mitochondria_data/testing_groundtruth.tif"


# take them out of the tiff stack
sh scripts/utils/split_images.sh "$training_images_file" "$input_dir_training_images"

# Run patchify_images.py
python scripts/utils/patchify_images.py "$patch_size" "$input_dir_training_images" "$output_dir_training_images"

# Run validate_patches.py
python scripts/utils/validate_patches.py "$patch_size" "$original_image_size" "$output_dir_training_images" "$validation_dir_training_images"

# MASKS
# take them out of the tif stack
sh scripts/utils/split_images.sh "$training_mask_file" "$input_dir_training_masks"

# Run patchify_images.py
python scripts/utils/patchify_images.py "$patch_size" "$input_dir_training_masks" "$output_dir_training_masks"

# Run validate_patches.py
python scripts/utils/validate_patches.py "$patch_size" "$original_image_size" "$output_dir_training_masks" "$validation_dir_training_masks"


## FOR TESTING

# IMAGES
sh scripts/utils/split_images.sh "$testing_images_file" "$input_dir_testing_images"

# Run patchify_images.py
python scripts/utils/patchify_images.py "$patch_size" "$input_dir_testing_images" "$output_dir_testing_images"

# Run validate_patches.py
python scripts/utils/validate_patches.py "$patch_size" "$original_image_size" "$output_dir_testing_images" "$validation_dir_testing_images"

# MASKS
sh scripts/utils/split_images.sh "$testing_mask_file" "$input_dir_testing_masks"
# Run patchify_images.py
python scripts/utils/patchify_images.py "$patch_size" "$input_dir_testing_masks" "$output_dir_testing_masks"

# Run validate_patches.py
python scripts/utils/validate_patches.py "$patch_size" "$original_image_size" "$output_dir_testing_masks" "$validation_dir_testing_masks"

ls $input_dir_training_images | wc -l
ls $output_dir_training_images | wc -l
ls $validation_dir_training_images | wc -l

ls $input_dir_training_masks | wc -l
ls $output_dir_training_masks | wc -l
ls $validation_dir_training_masks | wc -l

ls $input_dir_testing_images | wc -l
ls $output_dir_testing_images | wc -l
ls $validation_dir_testing_images | wc -l

ls $input_dir_testing_masks | wc -l
ls $output_dir_testing_masks | wc -l
ls $validation_dir_testing_masks | wc -l
