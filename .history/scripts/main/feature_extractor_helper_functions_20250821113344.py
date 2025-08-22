import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Model, Sequential


def load_pickled_data(file_path):
    # Open the pickle file
    with open(file_path, 'rb') as f:
        # Load the data from the file
        data = pickle.load(f)
    return data


def plot_history(history, output_dir, file_name, metric):
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

    acc = history.history[metric]
    val_acc = history.history[f'val_{metric}']
    plt.plot(epochs, acc, 'y', label=f'Training {metric}')
    plt.plot(epochs, val_acc, 'r', label=f'Validation {metric}')
    plt.title(f'Training and validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(os.path.join(output_dir, file_name+f"_{metric}.png"))
    # plt.show()
    plt.clf()  # clear this figure after saving it
    plt.close()


def preprocess_images_to_rois(image_folder, mask_folder, output_folder, padding=5, size=256):

    rois = []
    for patient_folder in os.listdir(image_folder):
        patient_image_folder = os.path.join(image_folder, patient_folder)
        patient_mask_folder = os.path.join(mask_folder, patient_folder)

        # if there's no corresponding mask, continue...
        if not os.path.exists(patient_mask_folder):
            continue

        for filename_image, filename_mask in zip(sorted(os.listdir(patient_image_folder)), sorted(os.listdir(patient_mask_folder))):
            # Load image and mask
            image_file = os.path.join(patient_image_folder, filename_image)
            mask_file = os.path.join(patient_mask_folder, filename_mask)

            img = cv2.imread(image_file)
            img = cv2.resize(img, (size, size))
            mask = cv2.imread(mask_file, 0)
            mask = cv2.resize(mask, (size, size))

            filename = os.path.splitext(filename_image)[0]

            # Compute the intersection of the image and the mask
            masked_img = cv2.bitwise_and(img, img, mask=mask)

            # Find contours in the masked image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Extract ROIs using contours and save them
            roi_index = 0
            for cnt in contours:
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(cnt)

                # Add padding to the rectangle
                x, y, w, h = x - padding, y - padding, w + 2 * padding, h + 2 * padding

                # Extract the rectangular ROI
                roi = masked_img[y:y+h, x:x+w]

                # Check if the ROI is empty or out of the image boundaries, skip if it is
                if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                ROI_filename = f'{filename}_ROI_{roi_index}'
                print(ROI_filename)

                # Resize the ROI
                roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_CUBIC)
                rois.append(roi)

                # Save the ROI as an image
                patient_folder = filename.split("_")[0]
                roi_output_folder = os.path.join(output_folder, patient_folder)
                if not os.path.exists(roi_output_folder):
                    os.makedirs(roi_output_folder)
                roi_output_path = os.path.join(roi_output_folder, f"{ROI_filename}.jpg")
                cv2.imwrite(roi_output_path, roi)
                roi_index += 1

    return np.array(rois)


def expand_scores(score_dict, roi_output_folder):
    expanded_scores = []
    for key, value in score_dict.items():
        if value is not None:
            patient_folder = key.split('_')[0]
            image_name = patient_folder + '_' + key.split('_')[1] + '_ROI_'
            roi_folder = os.path.join(roi_output_folder, patient_folder)
            # print(patient_folder, image_name, roi_folder)

            if os.path.exists(roi_folder):
                roi_files = [f for f in os.listdir(roi_folder) if f.startswith(image_name)]
                # print(roi_files)
                num_rois = len(roi_files)
                expanded_scores.extend([value] * num_rois)
        # print("\n")
    return np.array(expanded_scores)


def extract_features(images):
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    # Preprocess the input images
    preprocessed_images = preprocess_input(images)

    # Extract features
    features = model.predict(preprocessed_images)

    # Flatten the features
    flattened_features = features.reshape((features.shape[0], -1))

    return flattened_features


# Load and preprocess the labeled data
top_data_directory = 'data/preeclampsia_data'
image_folder = os.path.join(top_data_directory, "train", "images")
mask_folder = os.path.join(top_data_directory, "train", "masks")
cache_dir = os.path.join(top_data_directory, "cache")
scores = load_pickled_data(os.path.join(cache_dir, 'scores.pickle'))
roi_train_output_folder = os.path.join(top_data_directory, "train", "rois")
roi_test_output_folder = os.path.join(top_data_directory, "test", "rois")
square_size = 256
# extract multiple ROIs from each image

rois = preprocess_images_to_rois(image_folder, mask_folder,
                                 output_folder=roi_train_output_folder,
                                 padding=5,
                                 size=square_size)

print(rois.shape)
raise ValueError("stop")

# expand the scores to match the dimensions of additional ROIs for each image
expanded_scores_array = expand_scores(scores, roi_output_folder=roi_train_output_folder)

# Convert endotheliosis scores to a continuous scale (0-1)
endotheliosis_scores = np.array(expanded_scores_array)  # Replace this with the actual scores for each ROI
endotheliosis_scores_continuous = endotheliosis_scores / 3.0
# print(endotheliosis_scores_continuous)

# Parameters
test_size = 0.2
random_seed = 42

# Load your preprocessed images (ROIs) and scores
X = extract_features(np.array(rois))
y = endotheliosis_scores_continuous


# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_seed)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

file_name = 'rf-glom_openness'
model_output_dir = 'output/regression_models/rf_model2'
os.makedirs(model_output_dir, exist_ok=True)

# Initialize the RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2, n_jobs=6)

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
output_file_path = os.path.join(model_output_dir, "predictions_with_confidence_intervals.csv")
output_df.to_csv(output_file_path, index=False)

print("Predictions with confidence intervals saved to:", output_file_path)

# Calculate the mean squared error
mse = mean_squared_error(y_val, y_pred)

print(f"Mean squared error: {mse:.4f}")
print(y_pred, y_val)
