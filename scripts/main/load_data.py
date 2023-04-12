
import pickle
import shutil
import random
import os
import cv2
import numpy as np
import json
from typing import List
from sklearn.model_selection import train_test_split
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


def get_scores_from_annotations(annotations, cache_dir):
    scores = {}
    for annotation in sorted(annotations, key=lambda x: os.path.splitext(os.path.basename(x.image_name))[0]):
        name = os.path.splitext(os.path.basename(annotation.image_name))[0]
        if annotation.score is not None:
            score = float(annotation.score)
        else:
            score = None
        if name not in scores:
            scores[name] = 0.
        scores[name] = score

    sorted_scores = {k: v for k, v in sorted(
        scores.items(), key=lambda item: item[0])}

    pkl_file = os.path.join(cache_dir, 'scores.pickle')
    with open(pkl_file, 'wb') as f:
        pickle.dump(sorted_scores, f)
    return scores


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


def list_files(path):
    # Create an empty list to store file paths
    file_list = []

    # Walk the directory tree
    for root, dirs, files in os.walk(path):
        # Iterate over the files in the current directory
        for file in files:
            # Append the file path to the list
            file_list.append(os.path.join(root, file))

    # Return the list of file paths
    return file_list


def organize_data_into_subdirs(data_dir):
    """
    Sorts files in a directory by sample name and puts them in corresponding folders.

    Args:
    data_dir (str): path to directory containing the files to be sorted.

    Returns:
    files (list): list of paths to all files in data_dir
    all_samples (list): list of the names of the sample names of these files
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

    all_files = list_files(data_dir)
    all_sample_names_dict = {}
    for file in all_files:
        sample_name = file.split(
            '-')[0] if '-' in file else file.split('_')[0]
        if sample_name not in all_sample_names_dict:
            all_sample_names_dict[sample_name] = ""
        all_sample_names_dict[sample_name] = file

    print("Done!")
    return all_files, all_sample_names_dict


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


def check_sort_order(train_images, train_masks):

    # check if they are sorted in the same order
    if not arrays_sorted_same_order(train_images, train_masks):
        raise ValueError(
            "Images and masks arrays are not sorted in the same order!")

    # return all the arrays if sorted in the same order
    print("Images and masks are sorted in the correct order.")


def arrays_sorted_same_order(array1, array2):
    """
    Tests whether two arrays are sorted in the same order.

    Args:
        array1 (np.ndarray): First array to compare.
        array2 (np.ndarray): Second array to compare.

    Returns:
        bool: True if the arrays are sorted in the same order, False otherwise.
    """
    sort_idx_1 = np.argsort(array1)
    sort_idx_2 = np.argsort(array2)
    return np.array_equal(sort_idx_1, sort_idx_2)


def get_names_from_file_paths(paths):
    """Extract names from file paths

    Args:
        paths (list): list of file paths to be split
    """
    ids = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    return ids


def set_diff(list1, list2):
    """_summary_

    Args:
        list1 (list): reference list
        list2 (list): what is in the reference list, but not this one?

    Returns:
        _type_: diff, as a list
    """
    set1 = set(list1)
    set2 = set(list2)
    diff = set1 - set2
    return list(diff)


def create_train_val_test_lists(data_dir):
    """
    Creates lists of training, validation, and testing image-mask pairs based on the given directory structure.

    Args:
        data_dir (str): Path to the directory containing the "images" and "masks" subdirectories.

    Returns:
        Tuple: 3 lists containing (1) image file paths for training, (2) mask file paths for training,
            and (3) image file paths for testing (i.e., missing masks), 
            and 2 dicts containing sample - file path mappings for training and testing images
    """

    train_images = []
    train_masks = []
    test_images = []
    train_data_ids = {}
    test_data_ids = {}

    directories = os.listdir(os.path.join(data_dir, 'images'))
    num_directories = len(directories)
    print("Collecting training, validation, and test data paths...")
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
                            sample_id = os.path.splitext(
                                os.path.basename(image_path))[0]

                            if os.path.isfile(mask_path):
                                train_images.append(image_path)
                                train_masks.append(mask_path)
                                # make a dictionary that maps these data to sample_id for later organizing
                                if sample_id not in train_data_ids:
                                    train_data_ids[sample_id] = {'image_path': "", 'mask_path': ""}
                                train_data_ids[sample_id]['image_path'] = image_path
                                train_data_ids[sample_id]['mask_path'] = mask_path
                            else:
                                # make a dictionary that maps these data to sample_id for later organizing
                                test_images.append(image_path)
                                if sample_id not in test_data_ids:
                                    test_data_ids[sample_id] = ""
                                test_data_ids[sample_id] = image_path
                else:
                    for image_file in os.listdir(sample_image_dir):
                        if image_file.endswith('.jpg'):

                            image_path = os.path.join(
                                sample_image_dir, image_file)
                            test_images.append(image_path)

                            sample_id = os.path.splitext(
                                os.path.basename(image_path))[0]
                            if sample_id not in test_data_ids:
                                test_data_ids[sample_id] = ""
                            test_data_ids[sample_id] = image_path
            bar()

    # print(f'Training images length: {len(train_images)}')
    # print(f'Training masks length: {len(train_masks)}')
    # print(f'Testing images length: {len(test_images)}')
    # print(f'Training data dict length: {len(train_data_ids.items())}')
    # print(f'Testing data dict length: {len(test_data_ids.items())}')

    # just convert them to np.arrays, easier to work with
    train_images = sorted(np.array(train_images))
    train_masks = sorted(np.array(train_masks))
    test_images = sorted(np.array(test_images))

    # sort the dictionaries
    train_data_dict = dict(sorted(train_data_ids.items()))
    test_data_dict = dict(sorted(test_data_ids.items()))

    # check the sort order
    check_sort_order(train_images, train_masks)

    # check the sort order with the dictionaries
    train_image_names = get_names_from_file_paths(train_images)
    test_image_names = get_names_from_file_paths(test_images)
    if arrays_sorted_same_order(list(train_data_dict.keys()), train_image_names):
        print('Training dictionary and train images are in the same order.')
    else:
        print(set_diff(train_data_dict.keys(), train_image_names))
        print(set_diff(train_image_names, train_data_dict.keys()))
        print(train_image_names)
        print(train_data_dict.keys())
        raise ValueError(
            "Training dictionary and train images are not sorted in the same order!")

    if arrays_sorted_same_order(list(test_data_dict.keys()), test_image_names):
        print('Testing dictionary and test images are in the same order.')
    else:
        print(set_diff(test_data_dict.keys(), test_image_names))
        print(set_diff(test_image_names, test_data_dict.keys()))
        print(test_image_names)
        print(test_data_dict.keys())
        raise ValueError(
            "Testing dictionary and test images are not sorted in the same order!")

    print("Done!")
    return train_images, train_masks, test_images, train_data_dict, test_data_dict


def preprocess_data_image(path, size):

    # Load the image and its dimensions
    img = np.array(cv2.imread(path))
    # color image, resized, expanded dims, scaled
    color_img = resize(img, output_shape=(size, size))
    # print(color_img.shape
    color_img = color_img / 255.

    # bw image, resized, expanded dims, scaled
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = resize(bw_img, output_shape=(size, size))
    bw_img = np.expand_dims(bw_img, axis=-1)
    # print(bw_img.shape)
    bw_img = bw_img / 255.

    return bw_img, color_img


def preprocess_data_mask(path, size):

    # Load the image and its dimensions
    img = np.array(cv2.imread(path))
    # bw image, resized, expanded dims, scaled
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = resize(bw_img, output_shape=(size, size))
    bw_img = np.expand_dims(bw_img, axis=-1)
    # print(bw_img.shape)
    bw_img = bw_img / 255.

    return bw_img


def generate_final_dataset(train_images_paths, train_masks_paths, test_images_paths, train_data_dict, test_data_dict, size, cache_dir, val_split=0.2):
    print("Processing all the data...")
    os.makedirs(cache_dir, exist_ok=True)

    # MASKS DON"T HAVE COLOR - REMOVE
    # ORGANIZE OUTPUT SO THIS IS LOGICAL
    # Check if cached data exists and load it
    cache_paths = [
        os.path.join(cache_dir, f"{name}.pickle")
        for name in ["train_images", "train_masks", "test_images", "val_images", "val_masks", "train_data_dict", "test_data_dict",
                     "train_images_color", "val_images_color", "train_masks_color", "val_masks_color"]
    ]
    cache_exists = all(os.path.exists(path) for path in cache_paths)
    if cache_exists:
        print("Loading data from cache...")
        with open(cache_paths[0], "rb") as f:
            train_images = pickle.load(f)
        with open(cache_paths[1], "rb") as f:
            train_masks = pickle.load(f)
        with open(cache_paths[2], "rb") as f:
            test_images = pickle.load(f)
        with open(cache_paths[3], "rb") as f:
            val_images = pickle.load(f)
        with open(cache_paths[4], "rb") as f:
            val_masks = pickle.load(f)
        with open(cache_paths[5], "rb") as f:
            train_data_dict = pickle.load(f)
        with open(cache_paths[6], "rb") as f:
            test_data_dict = pickle.load(f)
        # color images
        with open(cache_paths[7], "rb") as f:
            train_images_color = pickle.load(f)
        with open(cache_paths[8], "rb") as f:
            val_images_color = pickle.load(f)
        with open(cache_paths[9], "rb") as f:
            train_masks_color = pickle.load(f)
        with open(cache_paths[10], "rb") as f:
            val_masks_color = pickle.load(f)

        print("Done!")
        return train_images, train_masks, val_images, val_masks, test_images, train_data_dict, test_data_dict, train_images_color, val_images_color, train_masks_color, val_masks_color

    # Otherwise, process the data

    # First, generate the training and validation split
    num_val = int(val_split * len(train_images_paths))
    train_images_all, train_images_all_color = np.array([preprocess_data_image(p, size, 'image') for p in train_images_paths])

    train_images = train_images_all[num_val:]
    val_images = train_images_all[:num_val]
    train_images_color = train_images_all_color[num_val:]
    val_images_color = train_images_all_color[:num_val]

    train_masks_all, train_masks_all_color = np.array([preprocess_data_mask(p, size, 'mask') for p in train_masks_paths])

    train_masks = train_masks_all[num_val:]
    val_masks = train_masks_all[:num_val]
    train_masks_color = train_masks_all_color[num_val:]
    val_masks_color = train_masks_all_color[:num_val]

    # Then, save everything
    # train and val (black and white images) images
    with open(cache_paths[0], "wb") as f:
        pickle.dump(train_images, f)
    print(f"Train images done with shape: {train_images.shape}")

    with open(cache_paths[1], "wb") as f:
        pickle.dump(train_masks, f)
    print(f"Train masks done with shape: {train_masks.shape}")

    # test images
    test_images, test_images_color = np.array([preprocess_data_image(p, size)
                                               for p in test_images_paths])
    with open(cache_paths[2], "wb") as f:
        pickle.dump(test_images, f)
    print(f"Test images done with shape: {test_images.shape}")

    with open(cache_paths[3], "wb") as f:
        pickle.dump(test_images_color, f)
    print(f"Test images (color) done with shape: {test_images_color.shape}")

    # val images
    with open(cache_paths[4], "wb") as f:
        pickle.dump(val_images, f)
    print(f"Validation images done with shape: {val_images.shape}")

    # val masks
    with open(cache_paths[5], "wb") as f:
        pickle.dump(val_masks, f)
    print(f"Validation masks done with shape: {val_masks.shape}")

    # train data dict
    with open(cache_paths[6], "wb") as f:
        pickle.dump(train_data_dict, f)
    print(f"Ordered training dictionary done with length: {len(train_data_dict.items())}.")

    # test data dict
    with open(cache_paths[7], "wb") as f:
        pickle.dump(test_data_dict, f)
    print(f"Ordered testing dictionary done with length: {len(test_data_dict.items())}.")

    # train images color
    with open(cache_paths[8], "wb") as f:
        pickle.dump(train_images_color, f)
    print(f"Train images (color) done with shape: {train_images_color.shape}")

    # val images color
    with open(cache_paths[9], "wb") as f:
        pickle.dump(val_images_color, f)
    print(f"Validation images (color) done with shape: {val_images_color.shape}")

    # train masks color
    with open(cache_paths[10], "wb") as f:
        pickle.dump(train_masks_color, f)
    print(f"Train masks (color) done with shape: {train_masks_color.shape}")

    with open(cache_paths[11], "wb") as f:
        pickle.dump(val_masks_color, f)
    print(f"Validation masks (color) done with shape: {val_masks.shape}")

    print("Done!")
    # return train_images, train_masks, val_images, val_masks,  test_images, train_data_dict, test_data_dict


top_data_directory = 'data/Lauren_PreEclampsia_Data'
testing_data_top_dir = os.path.join(top_data_directory, 'test')
training_data_top_dir = os.path.join(top_data_directory, 'train')
cache_dir_path = os.path.join(top_data_directory, 'cache')

annotation_file = os.path.join(
    training_data_top_dir, '2023-04-10_annotations.json')
images_directory = os.path.join(training_data_top_dir, 'images')
masks_directory = os.path.join(training_data_top_dir, 'masks')

# this is how i trained the big model, so all must be resized to 256 x 256
square_size = 256

# generate the binary masks from the annotation file and raw image data
# generate_binary_masks(annotation_file=annotation_file,
#                       data_dir=training_data_top_dir)

test_images_1_paths, test_data_dict_1 = organize_data_into_subdirs(
    data_dir=testing_data_top_dir)

train_images_paths, train_masks_paths, test_images_2_paths, train_data_dict, test_data_dict_2 = create_train_val_test_lists(
    data_dir=training_data_top_dir)

test_images_paths = np.concatenate((test_images_1_paths, test_images_2_paths))

# fix this, merge the two dictionaries
test_data_dict = dict(sorted({**test_data_dict_1, **test_data_dict_2}.items()))

generate_final_dataset(train_images_paths, train_masks_paths, test_images_paths, train_data_dict, test_data_dict, size=square_size, cache_dir=cache_dir_path)

# organize the scores at last
print('Writing scores-image mapping to cache dir...')
annotations = load_annotations_from_json(annotation_file)
scores = get_scores_from_annotations(annotations, cache_dir=cache_dir_path)

print('Done!')
