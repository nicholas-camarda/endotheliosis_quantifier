
import os
import cv2
import numpy as np
import json
from typing import List
from patchify import patchify, unpatchify
from skimage import io
import os
import sys


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
        print(annotation.image_name, annotation.score)

    return annotations


def get_image_path_from_json(json_file):
    with open(json_file, 'r') as f:
        annotations_data = json.load(f)
    return [entry['file_upload'] for entry in annotations_data]


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
square_size = 256

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


def patchify_image_dir(square_size, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        ext_ = os.path.splitext(filename)[1]  # includes '.'
        input_path = os.path.join(input_dir, filename)
        if os.path.isdir(input_path):
            # recursively process subdirectories
            output_subdir = os.path.join(output_dir, filename)
            if not os.path.exists(output_subdir):
                os.mkdir(output_subdir)
            patchify_image_dir(square_size, input_path, output_subdir)
        elif filename.endswith('.tif') or filename.endswith('.jpg'):
            img = io.imread(input_path)
            print(f"The filename is: {filename}")
            print(f"The image shape is: {img.shape}")

            patches = patchify(img, (square_size, square_size),
                               step=(square_size, square_size))

            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j]
                    output_filename = f"{os.path.splitext(filename)[0]}_{i}_{j}{ext_}"
                    output_path = os.path.join(output_dir, output_filename)
                    io.imsave(output_path, patch)
                    print(f"Saved patch {output_filename}")


# patchify_image_dir(square_size, data_dir, os.path.join(
#     training_data_top_dir, "image_patches"))
# patchify_image_dir(square_size,
#                    os.path.join(training_data_top_dir,
#                                 'Lauren_PreEclampsia_Masks'),
#                    os.path.join(training_data_top_dir, "mask_patches"))
