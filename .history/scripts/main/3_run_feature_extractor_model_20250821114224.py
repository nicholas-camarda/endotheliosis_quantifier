
import os
import pickle

# import sys
import cv2
# import keras
import numpy as np
# import pandas as pd
# import segmentation_models as sm
import tensorflow as tf
import tensorflow_hub as hub
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from skimage import io
# from skimage.io import imread, imsave
# scikit-optimize
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


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


def upscale_image_srgan(image, scale=4, srgan_model_path="https://tfhub.dev/captain-pool/esrgan-tf2/1"):
    # Load the pre-trained SRGAN model
    model = hub.load(srgan_model_path)

    # Prepare the input image
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    # Calculate the new dimensions
    height, width = image.shape[1:3]
    new_height = height * scale
    new_width = width * scale

    # Resize the image to the desired scale
    image_resized = tf.image.resize(image, (new_height, new_width), method=tf.image.ResizeMethod.BICUBIC)

    # Upscale the image with SRGAN
    upscaled_image = model(image_resized)
    upscaled_image = tf.squeeze(upscaled_image)

    # Convert the output image to the original dtype (0 to 255)
    upscaled_image = tf.image.convert_image_dtype(upscaled_image, tf.uint8)

    return upscaled_image.numpy()


def expand_scores(score_dict, roi_output_folder):
    expanded_scores = []
    keys_is_none = []
    num_rois_cnt = 0
    for key, value in score_dict.items():

        patient_folder = key.split('_')[0]
        image_name = patient_folder + '_' + key.split('_')[1] + "_"
        roi_folder = os.path.join(roi_output_folder, patient_folder)
        # print(roi_folder)
        if os.path.exists(roi_folder):
            roi_files = [f for f in os.listdir(roi_folder) if f.startswith(image_name)]

            num_rois = len(roi_files)
            expanded_scores.extend([value] * num_rois)

            # print(image_name, num_rois)
            num_rois_cnt += num_rois
            if value is None:
                keys_is_none.append(key)

    # print(num_rois_cnt)
    return np.array(expanded_scores), np.array(keys_is_none)


def extract_features(images, input_shape):
    # Load the pre-trained ResNet50 model without the top layer
    print("Extracting features usng ResNet50 fine-tuning...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    # Preprocess the input images
    preprocessed_images = preprocess_input(images)

    # Extract features
    features = model.predict(preprocessed_images)

    # Flatten the features
    flattened_features = features.reshape((features.shape[0], -1))

    return flattened_features


def preprocess_images_to_rois(image_folder, mask_folder, output_folder, padding=5, size=256):

    rois = []
    for patient_folder in os.listdir(image_folder):
        patient_image_folder = os.path.join(image_folder, patient_folder)
        patient_mask_folder = os.path.join(mask_folder, patient_folder)

        # Check if the path is a directory, and if there's no corresponding mask, continue...
        if not os.path.isdir(patient_image_folder) or not os.path.exists(patient_mask_folder):
            continue

        image_files = sorted([f for f in os.listdir(patient_image_folder) if os.path.isfile(os.path.join(patient_image_folder, f))])
        mask_files = sorted([f for f in os.listdir(patient_mask_folder) if os.path.isfile(os.path.join(patient_mask_folder, f))])

        for filename_image in image_files:
            # get the mask filename based off of the corresponding image
            filename_mask = os.path.splitext(filename_image)[0] + "_mask.jpg"

            if filename_mask in mask_files:
                # Load image and mask such that they are paired!
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
                    # print(ROI_filename)

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


# Load and preprocess the labeled data
square_size = 256
top_data_directory = 'data/preeclampsia_data'
cache_dir_path = os.path.join(top_data_directory, 'cache')
top_output_directory_regresion_models = 'output/regression_models'

scores = load_pickled_data(os.path.join(cache_dir_path, 'scores.pickle'))

image_folder = os.path.join(top_data_directory, "train", "images")
mask_folder = os.path.join(top_data_directory, "train", "masks")

roi_train_output_folder = os.path.join(top_data_directory, "train", "rois")

roi_test_output_folder = os.path.join(top_data_directory, "test", "rois")


print(f'Using square size: {square_size}')
# square_size = 256
# test_file = os.path.join(roi_train_output_folder, "T19", "T19_Image0_ROI_0.jpg")
# roi = cv2.imread(test_file)
# roi = cv2.resize(roi, (square_size, square_size), interpolation=cv2.INTER_CUBIC)
# print(roi.shape)
# rescaled_image = upscale_image_srgan(roi)
# cv2.imshow("Rescaled Image", rescaled_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# raise ValueError("stop")

# Extract regions of interest (ROIs) from the original images using binary masks
X_temp = preprocess_images_to_rois(image_folder, mask_folder,
                                   output_folder=roi_train_output_folder,
                                   padding=5,
                                   size=square_size)


print(f'Initial X shape: {X_temp.shape}')
print(f'Initial y shape (before expansion to fit ROIs): {len(scores.values())}')

# expand the scores to match the dimensions of additional ROIs for each image
expanded_scores_array, _ = expand_scores(scores, roi_output_folder=roi_train_output_folder)
print(f'Expanded scores array: {expanded_scores_array.shape}')

none_indices = find_none_indices(expanded_scores_array)
# print(none_indices)

X = remove_none_elements(X_temp, none_indices)
y_temp = remove_none_elements(expanded_scores_array, none_indices)

# Convert endotheliosis scores to a continuous scale (0-1)
y = np.array(y_temp) / 3.0

print(f'ROIs shape after removing bad masks: {X.shape}')
print(f'Scores shape (after expansion and removing bad masks): {y.shape}')

# X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train: {X_train.shape}')
print(f'X_val: {X_val.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_val: {y_val.shape}')

# define input shape
height, width, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
input_shape = (height, width, channels)

glomerular_features = extract_features(X_train, input_shape=input_shape)
glomerular_features_val = extract_features(X_val, input_shape=input_shape)

print(f'Glomerular training features: {glomerular_features.shape}')
print(f'Glomerular validation features: {glomerular_features_val.shape}')

# Save the output
top_output_directory_regression_input_data = os.path.join(cache_dir_path, 'features_and_scores')
os.makedirs(top_output_directory_regression_input_data, exist_ok=True)
data_to_save = ['X_train_glom_features', 'y_train_scores', 'X_val_glom_features', 'y_val_scores']
regr_data = [glomerular_features, y_train, glomerular_features_val, y_val]

for datum_name, datum in zip(data_to_save, regr_data):
    data_filepath = os.path.join(top_output_directory_regression_input_data, datum_name + '.pkl')
    with open(data_filepath, 'wb') as f:
        pickle.dump(datum, f)

print(f'Saved glomerular features and scores data to: {top_output_directory_regression_input_data}')


# file_name_with_ext = f'2023-04-10-glom_unet_xfer_seg_model-epochs75_batch8.hdf5'
# file_name = os.path.splitext(file_name_with_ext)[0]
#

# top_output_directory = 'output/segmentation_models'

# make the output directories
# final_output_path = os.path.join(top_output_directory, 'glomerulus_segmentation', file_name)
# os.makedirs(final_output_path, exist_ok=True)
# final_plots_dir = os.path.join(final_output_path, 'plots')
# os.makedirs(final_plots_dir, exist_ok=True)

# new_model_full_path = os.path.join(final_output_path, file_name_with_ext)
# # Combine the original training data into X and y,
# # and then use it to train this new model
# original_images = np.concatenate((color_images_train, color_images_val))
# binary_masks = np.concatenate((train_image_masks, val_image_masks))


# X_train_brr, y_train_brr = make_regression_data(glomerular_features, y_train)
# X_val_brr, y_val_brr = make_regression_data(glomerular_features_val, y_val)

# # Save the output
# top_output_directory_regression_input_data = os.path.join(cache_dir_path, 'regression_input')
# os.makedirs(top_output_directory_regression_input_data, exist_ok=True)
# data_to_save = ['X_train_regression', 'y_train_regression', 'X_val_regression', 'y_val_regression']
# regr_data = [X_train_brr, y_train_brr, X_val_brr, y_val_brr]

# for datum_name, datum in zip(data_to_save, regr_data):
#     data_filepath = os.path.join(top_output_directory_regression_input_data, datum_name + '.pkl')
#     with open(data_filepath, 'wb') as f:
#         pickle.dump(datum, f)

# print(f'Saved regression data input to: {top_output_directory_regression_input_data}')
