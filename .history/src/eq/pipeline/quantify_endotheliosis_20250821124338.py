import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


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


# Functions moved to eq.pipeline.quantify_endotheliosis module
# All side-effecting code removed for clean imports
