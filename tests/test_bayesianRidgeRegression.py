import os
import pickle
from math import sqrt

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate random data for testing
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate random test data
X_test, y_test = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)

# Create and train the Bayesian Ridge Regression model
print("Running Bayesian Ridge Regression model training...")
bayesian_ridge_model = BayesianRidge(verbose=True)
bayesian_ridge_model.fit(X_train, y_train)

# Save the trained Bayesian Ridge Regression model
top_output_directory_regression_models = 'output/regression_models/test'
os.makedirs(top_output_directory_regression_models, exist_ok=True)
model_filepath = os.path.join(top_output_directory_regression_models, 'test_bayesian_ridge_model.pkl')

with open(model_filepath, 'wb') as f:
    pickle.dump(bayesian_ridge_model, f)

print(f"Bayesian Ridge Regression model saved to {model_filepath}")

# Make predictions on the test set and calculate the prediction variance
y_pred, y_pred_var = bayesian_ridge_model.predict(X_test, return_std=True)

# Compute the confidence intervals
confidence_level = 0.95
z = 1.96  # z-score for 95% confidence
std_pred = np.sqrt(y_pred_var)  # Calculate the standard deviation for each prediction
lower_confidence_interval = y_pred - z * std_pred
upper_confidence_interval = y_pred + z * std_pred

# Save the predictions and confidence intervals
predictions_filepath = os.path.join(top_output_directory_regression_models, 'predictions_and_confidence_intervals.csv')
np.savetxt(predictions_filepath, np.column_stack((y_pred, lower_confidence_interval, upper_confidence_interval)), delimiter=',', header='prediction,lower_ci,upper_ci', comments='')

print(f"Predictions and confidence intervals saved to {predictions_filepath}")

# Evaluate the model's performance
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
