import os
import pickle
from math import sqrt

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model


def run_random_forest(X_train, y_train, X_val, y_val, model_output_directory, n_estimators=100, n_cpu_jobs=6):
    # Initialize the RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42, verbose=2, n_jobs=n_cpu_jobs)

    # Train the model on the training set
    regressor.fit(X_train, y_train)

    # Predict the scores on the validation set
    y_pred = regressor.predict(X_val)

    # Assuming you have evaluated the model on the validation set
    residuals = y_val - y_pred
    residual_std_dev = np.std(residuals)

    # Calculate the confidence interval
    confidence_interval = 1.96 * residual_std_dev

    # Create a pandas DataFrame with the predictions and confidence intervals
    data = {
        "Prediction": y_pred,
        "'True' Value": y_val,
        "Lower_CI": y_pred - confidence_interval,
        "Upper_CI": y_pred + confidence_interval,
    }

    output_df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file in the output folder
    os.makedirs(model_output_directory, exist_ok=True)
    output_file_path = os.path.join(model_output_directory, "predictions_with_confidence_intervals.csv")
    output_df.to_csv(output_file_path, index=False)

    print("Predictions with confidence intervals saved to:", output_file_path)

    # Calculate the mean squared error
    mse = mean_squared_error(y_val, y_pred)

    print(f"Mean squared error: {mse:.4f}\n")
    print(output_df)

    return regressor


def run_bayesian_ridge_regressor(X_train, y_train, X_val, y_val, model_output_directory, n_cv_splits=5, n_cpu_jobs=8):
    """Run Bayesian Ridge Regression model training and evaluation."""
    print("Running Bayesian Ridge Regression model training...")
    bayesian_ridge_model = BayesianRidge(verbose=True, compute_score=True)

    # Set up K-Fold cross-validation
    kfold = KFold(n_splits=n_cv_splits, shuffle=True, random_state=1)

    # Calculate the cross-validated scores
    print(f'Getting cross validation score with {n_cv_splits} splits...')
    scores_cval_brr = cross_val_score(bayesian_ridge_model, X_train, y_train,
                                      scoring='neg_mean_squared_error', n_jobs=n_cpu_jobs,
                                      verbose=2)

    # Calculate the average score and standard deviation
    average_score = np.mean(scores_cval_brr)
    std_score = np.std(scores_cval_brr)

    print(f'Average score: {average_score}, Standard deviation: {std_score}')

    # Train the BayesianRidge model on the entire training data
    bayesian_ridge_model.fit(X_train, y_train)
    print("BayesianRidge model fitted successfully.")

    model_filepath = os.path.join(model_output_directory, 'brr_model-glom_openness.pkl')
    with open(model_filepath, 'wb') as f:
        pickle.dump(bayesian_ridge_model, f)

    print(f"Bayesian Ridge Regression model saved to {model_filepath}")

    # Make predictions on the test set and calculate the prediction variance
    y_pred_brr, y_pred_var_brr = bayesian_ridge_model.predict(X_val, return_std=True)

    # Compute the confidence intervals
    confidence_level = 0.95
    z = 1.96  # z-score for 95% confidence
    std_pred = np.sqrt(y_pred_var_brr)  # Calculate the standard deviation for each prediction
    lower_confidence_interval = y_pred_brr - z * std_pred
    upper_confidence_interval = y_pred_brr + z * std_pred

    # Save the predictions and confidence intervals
    predictions_filepath = os.path.join(model_output_directory, 'brr_predictions.csv')
    np.savetxt(predictions_filepath, np.column_stack((y_pred_brr, lower_confidence_interval, upper_confidence_interval)), delimiter=',', header='prediction,lower_ci,upper_ci', comments='')

    print(f"Predictions and confidence intervals saved to {predictions_filepath}")

    # Evaluate the model's performance
    rmse = sqrt(mean_squared_error(y_val, y_pred_brr))
    print(f"RMSE: {rmse}")
    
    return bayesian_ridge_model


def openness_score(mask, preprocessed_image, threshold_ratio=0.85):
    """Calculate openness score for a glomerulus."""
    # Calculate the area of the glomerulus (white pixels in the mask)
    total_area = cv2.countNonZero(mask)
    print(f'Total area: {total_area}')

    # Find the maximum pixel value in the preprocessed image
    max_pixel_value = np.max(preprocessed_image)
    print(f'Max pixel value: {max_pixel_value}')

    # Calculate the threshold pixel value
    threshold_pixel_value = threshold_ratio * max_pixel_value
    print(f'Threshold pixel value: {threshold_pixel_value}')

    # Create a binary mask with maximum pixel values in the preprocessed image
    max_pixel_mask = (preprocessed_image >= threshold_pixel_value).astype(np.uint8)

    print(max_pixel_mask.shape)
    print(mask.shape)

    # Calculate the area of open capillaries (maximum pixel value occurrences within the mask)
    open_area = cv2.countNonZero(cv2.bitwise_and(max_pixel_mask, mask))
    print(f'Open area: {open_area}')

    # Calculate the ratio of open area to total area
    score = open_area / total_area if total_area > 0 else 0

    return score


def grade_glomerulus(openness_score):
    """Grade glomerulus based on openness score."""
    # Define the threshold values for each grade based on your ground-truth data
    grade_thresholds = [0.6, 0.4, 0.2]  # 20% open, 40% open, 60% open

    # Grade the glomerulus based on the openness score
    for i, threshold in enumerate(grade_thresholds):
        if openness_score >= threshold:
            return i
    return len(grade_thresholds)


def fit_umap_model(X_train, X_val, n_components=15, n_neighbors=8, n_epochs=30000, lr=1e-5):
    """Fit UMAP model for dimensionality reduction."""
    print("Fitting UMAP model to reduce dimensions of dataset...")
    print(f'Using n_components = {n_components}')

    # Create a scaler instance
    scaler = StandardScaler()

    # Fit the scaler to the training data
    scaler.fit(X_train)

    # Transform the training and validation data using the scaler
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create a UMAP instance
    umap_model = umap.UMAP(n_components=n_components,
                n_neighbors=n_neighbors,
                learning_rate=lr,
                n_epochs=n_epochs,
                verbose=True)

    # Fit UMAP on the training data
    X_train_umap = umap_model.fit_transform(X_train_scaled)
    X_val_umap = umap_model.transform(X_val_scaled)
    print("Done!")
    return X_train_umap, X_val_umap


def data_generator(X, y, batch_size):
    """Data generator for neural network training."""
    num_samples = len(X)
    while True:
        # Shuffle the data indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            # Load the batch data
            X_batch = np.array([X[i] for i in batch_indices])
            y_batch = np.array([y[i] for i in batch_indices])

            yield X_batch, y_batch


def run_neural_network(X_train, y_train, X_val, y_val, model_output_directory,
                       n_epochs, n_batch_size, input_shape=(256, 256, 64)):
    """Run neural network regression model."""
    input_layer = Input(shape=input_shape)

    # Add custom regression layers
    x = Flatten()(input_layer)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)  # Use a linear activation function for regression

    # Create the final regression model
    regression_model = Model(inputs=input_layer, outputs=predictions)

    # Compile and train the regression model
    regression_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

    train_gen = data_generator(X_train, y_train, n_batch_size)
    val_gen = data_generator(X_val, y_val, n_batch_size)

    steps_per_epoch = len(X_train) // n_batch_size
    validation_steps = len(X_val) // n_batch_size

    with tf.device("/GPU:0"):
        regression_model.fit(train_gen,
                             steps_per_epoch=steps_per_epoch,
                             epochs=n_epochs,
                             validation_data=val_gen,
                             validation_steps=validation_steps)
    print(regression_model.summary())

    model_filepath = os.path.join(model_output_directory, 'nn_model-glom_openness.pkl')
    with open(model_filepath, 'wb') as f:
        pickle.dump(regression_model, f)

    return regression_model








# Functions moved to eq.pipeline.quantify_endotheliosis module
# All side-effecting code removed for clean imports
