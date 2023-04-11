
import shutil
import random
import os
import cv2
import numpy as np
import json
from typing import List
from patchify import patchify, unpatchify
from skimage import io
from skimage.transform import resize
from skimage.io import imread, imsave
import os
import re
from alive_progress import alive_bar


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
    return image

# extract the annotations from the json file produced by label-studio


def find_image_path(image_name, root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file == image_name:
                return os.path.join(root, file)
    return None


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
        # print(annotation.image_name, annotation.score)

    return annotations


def get_image_size(annotation, data_dir):
    img_name = annotation.image_name
    img_path = find_image_path(img_name, root_directory=data_dir)

    img_height, img_width = (0, 0)
    if img_path is None:
        print(f"Image file {img_name} not found in the directory.")
        return img_height, img_width

    # Load the image and its dimensions
    img = np.array(cv2.imread(img_path))
    img_height, img_width = img.shape[:2]
    return img_height, img_width


def get_image_path_from_json(json_file):
    with open(json_file, 'r') as f:
        annotations_data = json.load(f)
    return [entry['file_upload'] for entry in annotations_data]


def organize_data_into_subdirs(data_dir):
    """
    Sorts files in a directory by sample name and puts them in corresponding folders.

    Args:
    data_dir (str): path to directory containing the files to be sorted.

    Returns:
    files (list): list of paths to all files in data_dir
    """
    # List all files in data_dir
    file_list = os.listdir(data_dir)

    for file in file_list:
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(data_dir, file)):
            # Split the file name by '-' or '_' and get the sample name
            sample_name = file.split(
                '-')[0] if '-' in file else file.split('_')[0]

            # Create a new directory for the sample if it doesn't exist
            sample_dir = os.path.join(data_dir, sample_name)
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)

            # Move the file to the new directory
            file_path = os.path.join(data_dir, file)
            target_path = os.path.join(sample_dir, file)
            if not os.path.exists(target_path):
                shutil.move(file_path, target_path)
        else:
            print("Skipping. File(s) are already in the correct structure.")

    print("Done!")
    return file_list


def save_binary_masks_as_images(binary_mask, output_dir, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # assumes pixel values [0,255]
    mask_image = (binary_mask).astype(np.uint8)
    # Save the binary mask as an image
    output_path = os.path.join(
        output_dir, f'{os.path.splitext(name)[0]}_mask.jpg')
    # print(output_path)
    cv2.imwrite(output_path, mask_image)


# Load all the data up
def generate_binary_masks(annotation_file, data_dir):
    print("Writing binary mask images to .jpg using annotations file...")
    # Load the annotations
    annotations = load_annotations_from_json(annotation_file)

    with alive_bar(len(annotations)) as b:
        for annotation in annotations:
            sample_name = annotation.image_name
            img_height, img_width = get_image_size(annotation, data_dir)

            binary_mask = rle_to_binary_mask(
                annotation.rle_mask, img_height, img_width)

            dir_name = sample_name.split("_")[0]
            mask_top_dir = os.path.join(data_dir, "masks")
            bin_mask_img_output_dir = os.path.join(mask_top_dir, dir_name)
            save_binary_masks_as_images(binary_mask,
                                        output_dir=bin_mask_img_output_dir,
                                        name=sample_name)
            b()


def create_train_val_test_lists(data_dir, val_split=0.2):
    """
    Creates lists of training, validation, and testing image-mask pairs based on the given directory structure.

    Args:
        data_dir (str): Path to the directory containing the "images" and "masks" subdirectories.
        val_split (float): Proportion of the dataset to use for validation. Default is 0.2.

    Returns:
        Tuple: 5 lists containing (1) image file paths for training, (2) mask file paths for training,
            and (3) image file paths for validation, (4) mask file paths for validation,
            and (5) image file paths for testing (i.e., missing masks).
    """

    train_images = []
    train_masks = []
    test_images = []
    directories = os.listdir(os.path.join(data_dir, 'images'))
    num_directories = len(directories)
    with alive_bar(num_directories) as bar:
        for i, sample_dir in enumerate(directories):
            if os.path.isdir(os.path.join(data_dir, 'images', sample_dir)):

                sample_image_dir = os.path.join(data_dir, 'images', sample_dir)
                sample_mask_dir = os.path.join(data_dir, 'masks', sample_dir)

                if os.path.isdir(sample_mask_dir):
                    for image_file in sorted(os.listdir(sample_image_dir)):
                        if image_file.endswith('.jpg'):
                            image_path = os.path.join(
                                sample_image_dir, image_file)
                            mask_file_name = os.path.splitext(os.path.basename(image_path))[
                                0] + "_mask.jpg"
                            mask_path = os.path.join(
                                sample_mask_dir, mask_file_name)
                            # print(image_path, mask_path)

                            if os.path.isfile(mask_path):
                                train_images.append(image_path)
                                train_masks.append(mask_path)
                            else:
                                test_images.append(image_path)
                else:
                    for image_file in os.listdir(sample_image_dir):
                        if image_file.endswith('.jpg'):
                            test_images.append(os.path.join(
                                sample_image_dir, image_file))
            bar()

    # Split training data into training and validation sets
    num_val = int(len(train_images) * val_split)
    train_images, val_images = train_images[num_val:], train_images[:num_val]
    train_masks, val_masks = train_masks[num_val:], train_masks[:num_val]

    print(f'Training images length: {len(train_images)}')
    print(f'Training masks length: {len(train_masks)}')
    print(f'Validation images length: {len(val_images)}')
    print(f'Validation masks length: {len(val_masks)}')
    print(f'Testing images length: {len(test_images)}')

    return train_images, train_masks, val_images, val_masks, test_images


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
        img = resize(img, (256, 256), anti_aliasing=True)

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


other_testing_dir = 'data/Lauren_PreEclampsia_Data/test'
training_data_top_dir = 'data/Lauren_PreEclampsia_Data/train'
annotation_file = os.path.join(
    training_data_top_dir, '2023-04-10_annotations.json')
images_directory = os.path.join(training_data_top_dir, 'images')
masks_directory = os.path.join(training_data_top_dir, 'masks')

# this is how i trained the big model, so all must be resized to 256 x 256
square_size = 256

# generate the binary masks from the annotation file and raw image data
# generate_binary_masks(annotation_file=annotation_file,
#                       data_dir=training_data_top_dir)

organize_data_into_subdirs(data_dir=other_testing_dir)

train_images, train_masks, val_images, val_masks, test_images = create_train_val_test_lists(
    data_dir=training_data_top_dir, val_split=0.2)

# print(train_images)
# print(train_masks)
# print(val_images)
# print(val_masks)
# print(test_images)
# data = load_data(annotation_file, data_dir)


# names = [item for item in data.keys()]
# X_test = np.vstack([data[item]['X'] for item in data.keys()])
# y_test = np.vstack([data[item]['y'] for item in data.keys()])
# scores = np.array([data[item]['score'] for item in data.keys()])
# print(X_test.shape)
# print(y_test.shape)
# print(scores.shape)
# print(names)
