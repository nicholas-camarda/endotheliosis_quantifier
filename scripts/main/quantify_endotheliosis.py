# print("Using VGG16 to extract features from ROIs in validation set...")
# # Do the same thing for the validation set
# glomerular_features_validation = new_model.predict(X_val)
# X_val_brr, y_val_brr = make_regression_data(glomerular_features_validation, y_val)

# lightGBM
# Create and train the Bayesian Ridge Regression model
print("Running Bayesian Ridge Regression model training...")
bayesian_ridge_model = BayesianRidge(verbose=True)
# Set up K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

# Calculate the cross-validated scores
scores_cval_brr = cross_val_score(bayesian_ridge_model, X_train_brr, y_train_brr, cv=kfold, scoring='neg_mean_squared_error')

# Calculate the average score and standard deviation
average_score = np.mean(scores)
std_score = np.std(scores)

print(f'Average score: {average_score}, Standard deviation: {std_score}')

# Train the BayesianRidge model on the entire training data
bayesian_ridge_model.fit(X_train_brr, y_train_brr)

# Save the trained Bayesian Ridge Regression model
top_output_directory_regression_models = 'output/regression_models/bayesian_ridge_model'
os.makedirs(top_output_directory_regression_models, exist_ok=True)
model_filepath = os.path.join(top_output_directory_regression_models, 'bayesian_ridge_model-glom_openness.pkl')

with open(model_filepath, 'wb') as f:
    pickle.dump(bayesian_ridge_model, f)

print(f"Bayesian Ridge Regression model saved to {model_filepath}")

# Make predictions on the test set and calculate the prediction variance
y_pred_brr, y_pred_var_brr = bayesian_ridge_model.predict(X_val_brr, return_std=True)

# Compute the confidence intervals
confidence_level = 0.95
z = 1.96  # z-score for 95% confidence
std_pred = np.sqrt(y_pred_var_brr)  # Calculate the standard deviation for each prediction
lower_confidence_interval = y_pred_brr - z * std_pred
upper_confidence_interval = y_pred_brr + z * std_pred

# Save the predictions and confidence intervals
predictions_filepath = os.path.join(top_output_directory_regression_models, 'validation_predictions_and_confidence_intervals.csv')
np.savetxt(predictions_filepath, np.column_stack((y_pred_brr, lower_confidence_interval, upper_confidence_interval)), delimiter=',', header='prediction,lower_ci,upper_ci', comments='')

print(f"Predictions and confidence intervals saved to {predictions_filepath}")

# Evaluate the model's performance
rmse = sqrt(mean_squared_error(y_val, y_pred_brr))
print(f"RMSE: {rmse}")


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
