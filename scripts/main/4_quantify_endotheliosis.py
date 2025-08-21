import os
import pickle
from math import sqrt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (ARDRegression, BayesianRidge,
                                  LinearRegression, SGDRegressor)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (KFold, ShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import (Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from umap import UMAP


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


def load_pickled_data(file_path):
    # Open the pickle file
    with open(file_path, 'rb') as f:
        # Load the data from the file
        data = pickle.load(f)
    return data


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")


def plot_history(history, output_dir, file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, file_name+"_loss.png"))
    plt.clf()  # clear this figure after saving it

    # plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, file_name+"_accuracy.png"))
    # plt.show()
    plt.clf()  # clear this figure after saving it
    plt.close()


top_data_directory = 'data/Lauren_PreEclampsia_Data'
regression_cache_dir_path = os.path.join(top_data_directory, 'cache', 'features_and_scores')
top_output_directory_regresion_models = 'output/regression_models'
# directory_nn_model = os.path.join(top_output_directory_regresion_models, 'neural_network')
directory_rf_model = os.path.join(top_output_directory_regresion_models, 'rf-model')

# load up the glomeruli features from VGG16 and 0-3 grades for each of the images (potentially containing multiple glomeruli)
X_train = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_train_glom_features.pkl'))
# X_train = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_train_net_reg.pkl'))
y_train = load_pickled_data(os.path.join(regression_cache_dir_path, 'y_train_scores.pkl'))
X_val = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_val_glom_features.pkl'))
# X_val = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_val_net_reg.pkl'))
y_val = load_pickled_data(os.path.join(regression_cache_dir_path, 'y_val_scores.pkl'))

print(f'X_train: {X_train.shape}')
print(f'y_train: {y_train.shape}')
print(f'X_val: {X_val.shape}')
print(f'y_val: {y_val.shape}')


run_random_forest(X_train, y_train, X_val, y_val,
                  model_output_directory=directory_rf_model,
                  n_estimators=300)



# n_neural_net_epochs = 50
# n_batch_size = 32
# run_neural_network(X_train, y_train, X_val, y_val,
#                    model_output_directory=directory_nn_model,
#                    n_epochs=n_neural_net_epochs,
#                    n_batch_size=n_batch_size)


# # Choose the number of components to keep
# n_components = min([X_train.shape[0], X_val.shape[0]])


# # Create a UMAP instance
# umap = UMAP(n_components=n_components,
#             n_neighbors=8,
#             learning_rate=1e-5,
#             n_epochs=30000,
#             verbose=True)

# print("Fitting UMAP model to reduce dimensions of dataset...")
# print(f'Using n_components = {n_components}')

# # Create a scaler instance
# scaler = StandardScaler()

# # Fit the scaler to the training data
# scaler.fit(X_train)

# # Transform the training and validation data using the scaler
# X_train_scaled = scaler.transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# # Fit UMAP on the training data
# X_train_umap = umap.fit_transform(X_train_scaled)

# # Define the classes or labels for the data points
# classes = [0, 0.5, 1, 1.5, 2, 3]

# # Define the colors to use for each class
# colors = ["red", "orange", "green", "purple", "blue", "black"]

# # Create a scatter plot of the UMAP coordinates
# plt.figure(figsize=(8, 8))
# for i in range(len(classes)):
#     plt.scatter(X_train_umap[y_train == i, 0], X_train_umap[y_train == i, 1],
#                 color=colors[i], alpha=0.5, label=classes[i])
# plt.xlabel("UMAP Component 1")
# plt.ylabel("UMAP Component 2")
# plt.legend(loc="upper right")
# plt.title("UMAP Plot of Training Data")
# plt.show()


# # Transform the validation data
# X_val_umap = umap.transform(X_val_scaled)

# run_random_forest_regressor(X_train_umap, y_train, X_val_umap, y_val,
#                             model_output_directory=directory_rf_regression_models,
#                             n_cv_splits=n_cv_splits, n_estimators=100, n_cpu_jobs=n_cpu_jobs)
