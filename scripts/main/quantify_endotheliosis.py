
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


def openness_score(glomerulus_contour, preprocessed_image):
    # Create a binary mask with the same dimensions as the input image
    mask = np.zeros_like(preprocessed_image, dtype=np.uint8)

    # Fill the glomerulus contour with white color
    cv2.drawContours(mask, [glomerulus_contour], 0, 255, -1)

    # Calculate the area of the glomerulus (white pixels in the mask)
    total_area = cv2.countNonZero(mask)

    # Calculate the area of open capillaries (white pixels in the preprocessed image within the mask)
    open_area = cv2.countNonZero(cv2.bitwise_and(preprocessed_image, mask))

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
model = tf.keras.models.load_model(new_model_full_path, compile=False)
# print(model.summary())

y_pred = model.predict(X_test)

# threshold to distinguish pixel is glom or not
binary_mask = y_pred > 0.5

region_of_interest = X_test[binary_mask]
