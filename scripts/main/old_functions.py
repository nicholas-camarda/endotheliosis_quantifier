def run_bayesian_ridge_regressor(X_train, y_train, X_val, y_val, model_output_directory, n_cv_splits=5, n_cpu_jobs=8):
    # Create and train the Bayesian Ridge Regression model
    print("Running Bayesian Ridge Regression model training...")
    bayesian_ridge_model = BayesianRidge(verbose=True, compute_score=True)

    # Set up K-Fold cross-validation
    kfold = KFold(n_splits=n_cv_splits, shuffle=True, random_state=1)
    # kfold = ShuffleSplit(n_splits=n_cv_splits, test_size=0.2, random_state=42)

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

    # bayesian_ridge_model = load_pickled_data(model_filepath)

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


def old_run_random_forest_regressor(X_train, y_train, X_val, y_val, model_output_directory, n_cv_splits=5, n_estimators=100, n_cpu_jobs=8):

    # Create and train the RandomForestRegressor model
    print("Running RandomForestRegressor model training...")
    random_forest_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    # Calculate the cross-validated scores
    print(f'Getting cross validation score with {n_cv_splits} splits...')
    scores_cval_rf = cross_val_score(random_forest_model, X_train, y_train, cv=n_cv_splits,
                                     scoring='neg_mean_squared_error', n_jobs=n_cpu_jobs,
                                     verbose=2)

    # Calculate the average score and standard deviation
    average_score = np.mean(scores_cval_rf)
    std_score = np.std(scores_cval_rf)

    print(f'Average score: {average_score}, Standard deviation: {std_score}')

    # Train the RandomForestRegressor model on the entire training data
    random_forest_model.fit(X_train, y_train)
    print("RandomForestRegressor model fitted successfully.")

    model_filepath = os.path.join(model_output_directory, 'rf_model-glom_openness.pkl')
    with open(model_filepath, 'wb') as f:
        pickle.dump(random_forest_model, f)

    # random_forest_model = load_pickled_data(model_filepath)

    print(f"Random Forest Regression model saved to {model_filepath}")

    # Make predictions on the test set
    y_pred_rf = random_forest_model.predict(X_val)

    # Save the predictions
    predictions_filepath = os.path.join(model_output_directory, 'validation_predictions_rf.csv')
    np.savetxt(predictions_filepath, y_pred_rf, delimiter=',', header='prediction', comments='')

    print(f"Predictions saved to {predictions_filepath}")

    # Evaluate the model's performance
    rmse = sqrt(mean_squared_error(y_val, y_pred_rf))
    print(f"RMSE: {rmse}")
    return random_forest_model


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


def get_label_counts(vector_):
    # Compute the unique counts of the elements in the y_val vector
    unique, counts = np.unique(vector_, return_counts=True)

    # Print the counts for each unique element
    for i in range(len(unique)):
        print("Validation Category {}: {}".format(unique[i], counts[i]))


def fit_umap_model(X_train, X_val, n_components=15, n_neighbors=8, n_epochs=30000, lr=1e-5):

    # n_components = min([X_train.shape[0], X_val.shape[0]])
    # n_neighbors = 8
    # n_umap_epochs = 30000
    # lr = 1e-5
    # Create a UMAP instance
    umap = UMAP(n_components=n_components,
                n_neighbors=n_neighbors,
                learning_rate=lr,
                n_epochs=n_epochs,
                verbose=True)

    print("Fitting UMAP model to reduce dimensions of dataset...")
    print(f'Using n_components = {n_components}')

    # Create a scaler instance
    scaler = StandardScaler()

    # Fit the scaler to the training data
    scaler.fit(X_train)

    # Transform the training and validation data using the scaler
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Fit UMAP on the training data
    X_train_umap = umap.fit_transform(X_train_scaled)
    X_val_umap = umap.transform(X_val_scaled)
    print("Done!")
    return X_train_umap, X_val_umap



def data_generator(X, y, batch_size):
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

