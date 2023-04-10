from tensorflow.keras.metrics import MeanIoU
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
import os
import pandas as pd
from typing import List

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# my annotation class object


class Annotation:
    def __init__(self, image_name, rle_mask, score=None):
        self.image_name = image_name
        self.rle_mask = rle_mask
        self.score = score

    def __repr__(self):
        return f"Annotation(image_path={self.image_name}, annotations={self.rle_mask}, score={self.score})"

# my rle input stream class


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_binary_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    # Add an extra dimension to make it [height, width, 1]
    # image = np.expand_dims(image, axis=-1)
    # print(image.shape)

    return image

# extract the annotations from the json file produced by label-studio


def load_annotations_from_json(json_file):
    with open(json_file, 'r') as f:
        annotations_data = json.load(f)

    annotations = []
    for entry in annotations_data:
        image_name = entry['file_upload']
        # print(image_name)
        image_name = image_name.split('-')[1]
        # print(image_path)

        image_rle_mask = []
        score = None
        for annotation in entry['annotations']:
            # Find 'rle' values
            rle_values = [result['value']['rle']
                          for result in annotation['result'] if 'rle' in result['value']]
            if rle_values:
                image_rle_mask.extend(rle_values[0])

            # Find 'choices' values and extract the score
            choices_values = [result['value']['choices']
                              for result in annotation['result'] if 'choices' in result['value']]
            if choices_values and score is None:
                score = float(choices_values[0][0])

        annotation = Annotation(image_name, image_rle_mask, score)
        annotations.append(annotation)
        print(annotation.image_name, annotation.score)

    return annotations


def get_image_path_from_json(json_file):
    with open(json_file, 'r') as f:
        annotations_data = json.load(f)
    return [entry['file_upload'] for entry in annotations_data]

# extract the score from each annotation


def get_scores_from_annotations(annotations):
    labels = []
    for annotation in annotations:
        if annotation.score is not None:
            label = int(annotation.score)
        else:
            label = None
        labels.append(label)
    return labels

# Match up the name of the image that was uploaded to label
# and the image name you have in your data directory


def find_image_path(image_name, root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file == image_name:
                return os.path.join(root, file)
    return None


def openness_score(glomerulus_contour, preprocessed_image):
    # Create a binary mask with the same dimensions as the input image
    mask = np.zeros_like(preprocessed_image, dtype=np.uint8)

    # Fill the glomerulus contour with white color
    cv2.drawContours(mask, [glomerulus_contour], 0, 255, -1)

    # Calculate the area of the glomerulus (white pixels in the mask)
    total_area = cv2.countNonZero(mask)

    # Calculate the area of open capillaries (white pixels in the preprocessed image within the mask)
    open_area = cv2.countNonZero(cv2.bitwise_and(preprocessed_image, mask))

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


def save_binary_masks_as_images_loop(binary_masks, output_dir, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, binary_mask in enumerate(binary_masks):
        # Convert the binary mask to a uint8 image with values in [0, 255]
        mask_image = (binary_mask * 255).astype(np.uint8)

        # Save the binary mask as an image
        output_path = os.path.join(
            output_dir, f'{os.path.splitext(name)[0]}_mask.jpg')
        print(output_path)
        cv2.imwrite(output_path, mask_image)


def save_binary_masks_as_images(binary_mask, output_dir, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask_image = (binary_mask * 255).astype(np.uint8)
    # Save the binary mask as an image
    output_path = os.path.join(
        output_dir, f'{os.path.splitext(name)[0]}_mask.jpg')
    print(output_path)
    cv2.imwrite(output_path, mask_image)


# Load all the data up


def load_data(annotation_file, data_dir):
    """Load data, convert rle mask to 

    Args:
        annotation_file (_type_): _description_
        data_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = {}

    # Load the annotations
    annotations = load_annotations_from_json(annotation_file)

    for annotation in annotations:
        img_name = annotation.image_name
        img_path = find_image_path(img_name, root_directory=data_dir)

        if img_path is None:
            print(f"Image file {img_name} not found in the directory.")
            continue

        # Load the image and its dimensions
        img = np.array(cv2.imread(img_path))
        # print(img.shape)
        # convert to grayscale and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.
        img_height, img_width = img.shape[:2]

        # Decode the RLE mask to a binary image
        binary_mask = rle_to_binary_mask(
            annotation.rle_mask, img_height, img_width)
        binary_mask = binary_mask / 255.
        # print(f'Binary mask shape: {binary_mask.shape}')
        # binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY) / 255.

        score = annotation.score
        # Ensure that score is a float: [0.0, 0.5, 1.0, 1.5, etc]
        if score is not None:
            score = float(score)

        # Add the features and score to the dictionary
        if img_name not in data:

            data[img_name] = {'X': [], 'y': [], 'score': None}
        data[img_name]['X'].append(np.expand_dims(img, axis=-1))
        data[img_name]['y'].append(np.expand_dims(binary_mask, axis=-1))
        data[img_name]['score'] = score

    data[img_name]['X'] = np.array(data[img_name]['X'])
    data[img_name]['y'] = np.array(data[img_name]['y'])
    return data


training_data_top_dir = 'data/Lauren_PreEclampsia_Data/Lauren_PreEclampsia_jpg_training_data'
annotation_file = os.path.join(training_data_top_dir, 'annotations.json')
data_dir = os.path.join(training_data_top_dir,
                        'Lauren_PreEclampsia_Raw_Images')

model_path = 'output/segmentation_models/unet_binseg_50epoch_3960images_8batchsize/unet_binseg_50epoch_3960images_8batchsize.hdf5'
file_name = 'endotheliosis_seg.hdf5'

# crazy workflow but you need to
# 1 load in the data
data = load_data(annotation_file, data_dir)


names = [item for item in data.keys()]
X_test = np.vstack([data[item]['X'] for item in data.keys()])
y_test = np.vstack([data[item]['y'] for item in data.keys()])
scores = np.array([data[item]['score'] for item in data.keys()])
print(X_test.shape)
print(y_test.shape)
print(scores.shape)
print(names)

# write the binary masks to equivalent file structure
for i, name in enumerate(np.unique(names, axis=0)):
    dir_name = name.split("_")[0]
    save_binary_masks_as_images(y_test[i], output_dir=os.path.join(
        training_data_top_dir, 'Lauren_PreEclampsia_Masks', dir_name),
        name=name)

# patch the files

model = tf.keras.models.load_model(model_path, compile=False)
# Calculate IOU after the fact
print('Predicting on test set...')
y_pred = model.predict(X_test)
# threshold to distinguish pixel is mito or not
threshold = 0.5
y_pred_thresholded = y_pred > threshold

# # to calculate meanIoU, you need to say 2 classes.
# # weird that it's different from building the neural net
# n_classes_iou = 2  # note that this is different from when we made the neural net

# # IoU intersection over union aka the jaccard index
# # overlap between the predicted segmentation and the ground truth divided by
# # the area of union between pred seg and groundtruth
# # if intersection == union, then value is 1 and you have a great segmenter
# IOU_keras = MeanIoU(num_classes=n_classes_iou)
# IOU_keras.update_state(y_pred_thresholded, y_test)
# print("Mean IoU =", IOU_keras.result().numpy())

# # generate n unique random indices from the test dataset
# n = 20  # specify the number of images you want to generate
# random_indices = np.random.choice(len(X_test), size=n, replace=False)
# # create output directory if it doesn't exist
# predicitions_output_dir = f'output/segmentation_models/{file_name}/plots/predictions'
# if not os.path.exists(predicitions_output_dir):
#     os.makedirs(predicitions_output_dir)

# # generate and save n random images
# for test_img_number in random_indices:
#     # load a random image from the test dataset
#     test_img = X_test[test_img_number]
#     ground_truth = y_test[test_img_number]
#     test_img_input = np.expand_dims(test_img, 0)
#     prediction = (model.predict(test_img_input)[
#                   0, :, :, 0] > 0.5).astype(np.uint8)

#     # create file name based on index of image in dataset
#     file_name_predicition = f"{file_name}_prediction_{test_img_number}"

#     # plot and save image
#     plt.figure(figsize=(16, 8))
#     plt.subplot(231)
#     plt.title('Testing Image')
#     plt.imshow(test_img[:, :, 0], cmap='gray')
#     plt.subplot(232)
#     plt.title('Testing Label')
#     plt.imshow(ground_truth[:, :, 0], cmap='gray')
#     plt.subplot(233)
#     plt.title('Prediction on test image')
#     plt.imshow(prediction, cmap='gray')
#     plt.savefig(os.path.join(predicitions_output_dir,
#                 file_name_predicition+".png"))
#     plt.clf()
#     plt.close()
#     # plt.show()

# # # Test on one image
# # image_path = 'jpg_data/Lauren_PreEclampsia_Raw_Images/T30/T30_Image0.jpg'
# # result = process_new_image(image_path, endotheliosisQuantifierModel)
# # print("Results:")
# # print(json.dumps(result, indent=2))


# # RUN THIS IN A LOOP
# # images_dir = 'jpg_data/Lauren_PreEclampsia_Raw_Images'
# # output_dir = 'output'

# # # create an empty dataframe to store the results
# # results_df = pd.DataFrame(columns=['subdirectory', 'image', 'result'])

# # # loop through subdirectories and files
# # for subdir, dirs, files in os.walk(images_dir):
# #     print(subdir)
# #     for file in files:
# #         # check if file is a .jpg
# #         if file.endswith('.jpg') or file.endswith('.jpeg'):
# #             # construct the full path to the image
# #             image_path = os.path.join(subdir, file)
# #             # process the image and store the result
# #             result = process_new_image(image_path, endotheliosisQuantifierModel)
# #             print(result)
# #             # add the result to the dataframe
# #             results_df = results_df.append({'subdirectory': subdir,
# #                                             'image': file,
# #                                             'result': result},
# #                                            ignore_index=True)

# # # save the results to a CSV file
# # results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

# # # group the results by subdirectory and image, and calculate the mean result
# # avg_results_df = results_df.groupby(['subdirectory', 'image']).mean().reset_index()

# # # save the average results to a CSV file
# # avg_results_df.to_csv(os.path.join(output_dir, 'average_results.csv'), index=False)


# # print("Done!")
