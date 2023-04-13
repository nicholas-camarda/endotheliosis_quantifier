import os
import pickle
from math import sqrt

import cv2
import numpy as np
import psutil
from sklearn.linear_model import (ARDRegression, BayesianRidge,
                                  LinearRegression, SGDRegressor)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (KFold, ShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_pickled_data(file_path):
    # Open the pickle file
    with open(file_path, 'rb') as f:
        # Load the data from the file
        data = pickle.load(f)
    return data


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


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")


top_data_directory = 'data/Lauren_PreEclampsia_Data'
regression_model = 'bayesian_ridge_model'
regression_cache_dir_path = os.path.join(top_data_directory, 'cache', 'regression_input', regression_model)
top_output_directory_regresion_models = 'output/regression_models'
directory_regression_models = os.path.join(top_output_directory_regresion_models, regression_model)

n_cv_splits = 5
n_cpu_jobs = 3  # can change this on better machine

# load up the data
X_train_brr = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_train_regression.pkl'))[:20]
y_train_brr = load_pickled_data(os.path.join(regression_cache_dir_path, 'y_train_regression.pkl'))[:20]
X_val_brr = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_val_regression.pkl'))
y_val_brr = load_pickled_data(os.path.join(regression_cache_dir_path, 'y_val_regression.pkl'))

print(f'X_train: {X_train_brr.shape}')
print(f'y_train: {y_train_brr.shape}')
print(f'X_val: {X_val_brr.shape}')
print(f'y_val: {y_val_brr.shape}')

# A generator function to yield batches of data


def data_generator(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


# raise ValueError("stop")
# Create an SGDRegressor instance
sgd_regressor = SGDRegressor(verbose=0, penalty='elasticnet')

# Define the number of epochs and batch size
# n_epochs = 100
# batch_size = 16
sgd_regressor.fit(X_train_brr, y_train_brr)

# # Train the SGDRegressor using a generator
# for epoch in range(1, n_epochs + 1):
#     print(f'Epoch {epoch}')
#     for X_batch, y_batch in data_generator(X_train_brr, y_train_brr, batch_size):
#         sgd_regressor.partial_fit(X_batch, y_batch)
#     print('-' * 50)

# Evaluate the model
y_pred = sgd_regressor.predict(X_val_brr)
mse = mean_squared_error(y_val_brr, y_pred)
print(f'Mean squared error: {mse}')

# print("Fitting ARDRegression model...")

# model = ARDRegression(verbose=True, compute_score=True, copy_X=False)
# model.fit(X_train_brr, y_train_brr)
# print("ARDRegression model fitted successfully.")

# # Save the trained Bayesian Ridge Regression model
# os.makedirs(directory_regression_models, exist_ok=True)
# model_filepath = os.path.join(directory_regression_models, 'ARDmodel-glom_openness.pkl')

# with open(model_filepath, 'wb') as f:
#     pickle.dump(model, f)

# print(f"ARD Regression model saved to {model_filepath}")

# # Make predictions on the test set and calculate the prediction variance
# y_pred_brr, y_pred_var_brr = model.predict(X_val_brr, return_std=True)

# # Compute the confidence intervals
# confidence_level = 0.95
# z = 1.96  # z-score for 95% confidence
# std_pred = np.sqrt(y_pred_var_brr)  # Calculate the standard deviation for each prediction
# lower_confidence_interval = y_pred_brr - z * std_pred
# upper_confidence_interval = y_pred_brr + z * std_pred

# # Save the predictions and confidence intervals
# predictions_filepath = os.path.join(directory_regression_models, 'validation_predictions_and_confidence_intervals.csv')
# np.savetxt(predictions_filepath, np.column_stack((y_pred_brr, lower_confidence_interval, upper_confidence_interval)), delimiter=',', header='prediction,lower_ci,upper_ci', comments='')

# print(f"Predictions and confidence intervals saved to {predictions_filepath}")

# # Evaluate the model's performance
# rmse = sqrt(mean_squared_error(y_val_brr, y_pred_brr))
# print(f"RMSE: {rmse}")


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


# Create and train the Bayesian Ridge Regression model
# print("Running Bayesian Ridge Regression model training...")
# bayesian_ridge_model = BayesianRidge(verbose=True, compute_score=True, copy_X=False)


# Set up K-Fold cross-validation
# kfold = KFold(n_splits=n_cv_splits, shuffle=True, random_state=1)
# kfold = ShuffleSplit(n_splits=n_cv_splits, test_size=0.2, random_state=42)


# Calculate the cross-validated scores
# print(f'Getting cross validation score with {n_cv_splits} splits...')
# scores_cval_brr = cross_val_score(bayesian_ridge_model, X_train_brr, y_train_brr,
#                                   scoring='neg_mean_squared_error', n_jobs=n_cpu_jobs,
#                                   verbose=2)

# Calculate the average score and standard deviation
# average_score = np.mean(scores_cval_brr)
# std_score = np.std(scores_cval_brr)

# print(f'Average score: {average_score}, Standard deviation: {std_score}')

# print_memory_usage()
# # Train the BayesianRidge model on the entire training data
# bayesian_ridge_model.fit(X_train_brr, y_train_brr)
# print_memory_usage()
# print("BayesianRidge model fitted successfully.")
