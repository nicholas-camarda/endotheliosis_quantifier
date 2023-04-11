
from datetime import datetime
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
from typing import List

import segmentation_models as sm
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from skimage import io
import sys
from skimage.io import imread, imsave
from skimage.transform import resize
import pickle


def openness_score(mask, preprocessed_image, threshold_ratio=0.85):
    # Calculate the area of the glomerulus (white pixels in the mask)
    total_area = cv2.countNonZero(mask)
    print(f'Total area: {total_area}')

    # # Calculate the area of open capillaries (white pixels in the preprocessed image within the mask)
    # open_area = cv2.countNonZero(cv2.bitwise_and(preprocessed_image, mask))
    # print(f'Open area: {open_area}')

    # Find the maximum pixel value in the preprocessed image
    max_pixel_value = np.max(preprocessed_image)
    print(f'Max pixel value: {max_pixel_value}')

    # Calculate the threshold pixel value
    threshold_pixel_value = threshold_ratio * max_pixel_value
    print(f'Threshold pixel value: {threshold_pixel_value}')

    # Create a binary mask with maximum pixel values in the preprocessed image
    max_pixel_mask = (preprocessed_image >=
                      threshold_pixel_value).astype(np.uint8)

    print(max_pixel_mask.shape)
    print(mask.shape)

    # Calculate the area of open capillaries (maximum pixel value occurrences within the mask)
    open_area = cv2.countNonZero(cv2.bitwise_and(max_pixel_mask, mask))
    print(f'Open area: {open_area}')

    # Calculate the ratio of open area to total area
    score = open_area / total_area if total_area > 0 else 0

    return score


def grade_glomerulus(openness_score):
    # Define the threshold values for each grade based on your ground-truth data
    grade_thresholds = [0.6, 0.4, 0.2]  # 20% open, 40% open, 60% open

    # Grade the glomerulus based on the openness score
    for i, threshold in enumerate(grade_thresholds):
        if openness_score >= threshold:
            return i
    return len(grade_thresholds)


def load_pickled_data(file_path):
    # Open the pickle file
    with open(file_path, 'rb') as f:
        # Load the data from the file
        data = pickle.load(f)
    return data


top_data_directory = 'data/Lauren_PreEclampsia_Data'
cache_dir_path = os.path.join(top_data_directory, 'cache')
top_output_directory = 'output/segmentation_models'
file_name_with_ext = f'2023-04-10-glom_unet_xfer_seg_model-epochs75_batch8.hdf5'
file_name = os.path.splitext(file_name_with_ext)[0]

# make the output directories
final_output_path = os.path.join(
    top_output_directory, 'glomerulus_segmentation', file_name)
os.makedirs(final_output_path, exist_ok=True)
final_plots_dir = os.path.join(final_output_path, 'plots')
os.makedirs(final_plots_dir, exist_ok=True)

new_model_full_path = os.path.join(final_output_path, file_name_with_ext)

# load up the data
X_train = load_pickled_data(os.path.join(
    cache_dir_path, 'train_images.pickle'))
y_train = load_pickled_data(os.path.join(cache_dir_path, 'train_masks.pickle'))
X_val = load_pickled_data(os.path.join(cache_dir_path, 'val_images.pickle'))
y_val = load_pickled_data(os.path.join(cache_dir_path, 'val_masks.pickle'))
X_test = load_pickled_data(os.path.join(cache_dir_path, 'test_images.pickle'))

print(f'Training images shape: {X_train.shape}')
print(f'Training masks shape: {y_train.shape}')
print(f'Validation images shape: {X_val.shape}')
print(f'Validation masks shape: {y_val.shape}')
print(f'Testing images shape: {X_test.shape}')

# load the pretrained unet model
print(f"Loading pretrained model: {new_model_full_path}")
model = tf.keras.models.load_model(new_model_full_path, compile=False)
# print(model.summary())

print("Predicting on test set to generate binary masks...")
y_pred = model.predict(X_test)

# threshold to distinguish pixel is glom or not
binary_masks = y_pred > 0.5

# Loop over each test image and its corresponding binary mask
for i in range(len(X_test)):
    image = X_test[i]
    binary_mask = binary_masks[i].astype(np.uint8)

    # Use np.where to get the indices of the segmented region
    # indices = np.where(binary_mask)
    if np.count_nonzero(binary_mask) == 0:
        # binary mask is empty, skip this image
        continue

    print(i)

    # code to extract region of interest from image using binary_mask
    # ...
    region_of_interest = np.expand_dims(
        cv2.bitwise_and(image, image, mask=binary_mask), axis=-1)

    # Perform analysis on the segmented region as desired
    print(binary_mask.shape)
    print(region_of_interest.shape)
    print(image.shape)

    frac_open = openness_score(region_of_interest, image)
    print(f'Percent open: {frac_open*100}%')

    # Optionally, visualize the segmented region for debugging purposes
    # Plot the original image, binary mask, and region of interest side by side
    # Optionally, visualize the image, binary mask and region of interest in one plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Image")
    axs[1].imshow(binary_mask, cmap='gray')
    axs[1].set_title("Binary Mask")
    axs[2].imshow(region_of_interest, cmap='gray')
    axs[2].set_title("Region of Interest")
    plt.show()
    plt.clf()
    break
