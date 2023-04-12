
from numpy import absolute
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


def find_none_indices(arr):
    return np.argwhere(np.vectorize(lambda x: x is None)(arr))


def remove_none_elements(arr, none_indices):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input should be a numpy array")

    mask = np.ones(arr.shape[0], dtype=bool)
    mask[none_indices[:, 0]] = False
    return arr[mask]


def plot_image(index):
    test_img = X_test[index]

    # plot and save image
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.show()
    plt.close()


def plot_features(features, index):
    plt.figure(figsize=(12, 12))
    square = 8  # because the conv layer we are viewing is 8x8 = 64
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, index)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(features[index, :, :, ix-1], cmap='gray')
    plt.show()
    plt.close()


top_data_directory = 'data/Lauren_PreEclampsia_Data'
cache_dir_path = os.path.join(top_data_directory, 'cache')
top_output_directory = 'output/segmentation_models'
top_output_directory_regresion_models = 'output/regression_models'
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
color_images_train = load_pickled_data(os.path.join(cache_dir_path, 'train_images_color.pickle'))
train_image_masks = load_pickled_data(os.path.join(cache_dir_path, 'train_masks.pickle'))
color_images_val = load_pickled_data(os.path.join(cache_dir_path, 'val_images_color.pickle'))
val_image_masks = load_pickled_data(os.path.join(cache_dir_path, 'val_masks.pickle'))
color_images_test = load_pickled_data(os.path.join(cache_dir_path, 'test_images_color.pickle'))
scores = load_pickled_data(os.path.join(cache_dir_path, 'scores.pickle'))

# Combine the original training data into X and y,
# and then use it to train this new model
original_images = np.concatenate((color_images_train, color_images_val))
binary_masks = np.concatenate((train_image_masks, val_image_masks))

# Extract regions of interest (ROIs) from the original images using binary masks

X_temp = original_images * binary_masks
X_temp = np.array([preprocess_input(p) for p in X_temp])

y_temp = np.array(list(scores.values()))
none_indices = find_none_indices(y_temp)
X = remove_none_elements(X_temp, none_indices)
y = remove_none_elements(y_temp, none_indices)

print(f'ROI shape: {X.shape}')
print(f'Scores shape: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define input_shape for grayscale image (duplicated 3 times to trick)
height, width, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
input_shape = (height, width, channels)

# Load VGG16 model wothout classifier/fully connected layers
# Load imagenet weights that we are going to use as feature generators
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
    layer.trainable = False

new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv1').output)
new_model.summary()

print("Using VGG16 to extract features from ROIs...")
glomerular_features = new_model.predict(X_train)

# plot the example image
# plot_image(index = 1)
# Plot features to view them
# plot_features(features, index = 1)

# Reassign 'features' as X to make it easy to follow
xgboost_X = glomerular_features
# Make it compatible for Random Forest and match Y labels
# Compute the number of features per ROI
num_features = np.prod(xgboost_X.shape[1:])
# Reshape the features to have the same number of samples as y_train
xgboost_X = xgboost_X.reshape(len(y_train), num_features)
print(xgboost_X.shape)
# Reshape Y to match X
xgboost_y = y_train.reshape(-1)
print(xgboost_y.shape)

# XGBOOST
print("Running xgboost model training...")
# Documentation for GPU: https://xgboost.readthedocs.io/en/latest/gpu/index.html#
xgboost_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)  # Define gpu_id, otherwise it uses CPU and computation would be very slow.
# Train the model on training data
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgboost_model, xgboost_X, xgboost_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
xgboost_model.fit(xgboost_X, xgboost_y, verbose=True)
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

# Save the trained XGBoost model
os.makedirs(top_output_directory_regresion_models, exist_ok=True)
xgboost_model.save_model(os.path.join(top_output_directory_regresion_models, 'xgboost_model_glom_openness.model'))


if (False):
    # load the pretrained unet model
    print(f"Loading pretrained model: {new_model_full_path}")
    model = tf.keras.models.load_model(new_model_full_path, compile=False)
    # print(model.summary())

    print("Predicting on test set to generate binary masks...")
    binary_masks = model.predict(X_test)

    print('Identifying regions of interest in original images...')
    X = X_test[binary_masks > 0.5]
    y = scores

    # Convert the scores to a 0-1 floating-point scale
    y = y / 3

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create a pipeline with preprocessing and regression model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate the confidence interval
    alpha = 0.95
    squared_errors = (y_pred - y_test) ** 2
    mse = mean_squared_error(y_test, y_pred)
    confidence_interval = np.sqrt(stats.t.interval(alpha, len(
        y_test)-1, loc=np.mean(squared_errors), scale=stats.sem(squared_errors)))

    # Evaluate the model
    print(f"Mean squared error: {mse:.2f}")
    print(f"R2 score: {r2_score(y_test, y_pred):.2f}")
    print(f"Confidence interval: {confidence_interval}")
