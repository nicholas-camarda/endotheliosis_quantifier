
import os
import pickle
# import sys
from datetime import datetime
from math import sqrt
from typing import List

import cv2
# import keras
import lightgbm as lgb
import numpy as np
# import pandas as pd
# import segmentation_models as sm
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from matplotlib import pyplot as plt
from numpy import absolute
from scipy import stats
# from skimage import io
# from skimage.io import imread, imsave
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from tensorflow.keras.models import load_model


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


def make_regression_data(features, y):
    # Reassign 'features' as X to make it easy to follow
    X_ = features
    # Make it compatible for Random Forest and match Y labels
    # Compute the number of features per ROI
    num_features = np.prod(X_.shape[1:])
    # Reshape the features to have the same number of samples as y_train
    X_ = X_.reshape(len(y), num_features)
    # Reshape Y to match X
    y_ = y.reshape(-1)

    print(X_.shape)
    print(y_.shape)

    return X_, y_


def tune_learning_rate_and_estimators(elr_X_train, elr_y_train, elr_cv, device_type):
    # Define search space
    search_space = {
        'learning_rate': Real(0.0001, 0.3),
        'n_estimators': Integer(50, 400),
    }

    # Set up the LightGBM model with the device type (CPU or GPU)
    lgb_model = lgb.LGBMRegressor(device=device_type)

    # Set up BayesSearchCV with the specified search space
    bayes_search = BayesSearchCV(estimator=lgb_model, search_spaces=search_space, scoring='neg_mean_absolute_error', cv=elr_cv, n_jobs=4, n_iter=50)

    # Perform the search to find the best learning rate and number of estimators
    bayes_search.fit(elr_X_train, elr_y_train)

    # Get the best learning rate and number of estimators
    best_learning_rate = bayes_search.best_params_['learning_rate']
    best_n_estimators = bayes_search.best_params_['n_estimators']
    print(f'Best learning rate: {best_learning_rate}')
    print(f'Best number of estimators: {best_n_estimators}')

    return best_learning_rate, best_n_estimators


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Memory growth set")
    except RuntimeError as e:
        print(e)

top_data_directory = 'data/Lauren_PreEclampsia_Data'
cache_dir_path = os.path.join(top_data_directory, 'cache')
top_output_directory = 'output/segmentation_models'
top_output_directory_regresion_models = 'output/regression_models'
file_name_with_ext = f'2023-04-10-glom_unet_xfer_seg_model-epochs75_batch8.hdf5'
file_name = os.path.splitext(file_name_with_ext)[0]

val_perc = 0.2

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

# X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train: {X_train.shape}')
print(f'X_val: {X_val.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_val: {y_val.shape}')

# define input shape
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

print("Using VGG16 to extract features from ROIs in training set...")
glomerular_features = new_model.predict(X_train)
X_train_brr, y_train_brr = make_regression_data(glomerular_features, y_train)

print("Using VGG16 to extract features from ROIs in validation set...")
glomerular_features_val = new_model.predict(X_val)
X_val_brr, y_val_brr = make_regression_data(glomerular_features_val, y_val)

# Save the output
top_output_directory_regression_input_data = os.path.join(cache_dir_path, 'regression_input', 'bayesian_ridge_model')
os.makedirs(top_output_directory_regression_input_data, exist_ok=True)
data_to_save = ['X_train_regression', 'y_train_regression', 'X_val_regression', 'y_val_regression']
regr_data = [X_train_brr, y_train_brr, X_val_brr, y_val_brr]

for datum_name, datum in zip(data_to_save, regr_data):
    data_filepath = os.path.join(top_output_directory_regression_input_data, datum_name + '.pkl')
    with open(data_filepath, 'wb') as f:
        pickle.dump(datum, f)

print(f'Saved regression data input to: {top_output_directory_regression_input_data}')
