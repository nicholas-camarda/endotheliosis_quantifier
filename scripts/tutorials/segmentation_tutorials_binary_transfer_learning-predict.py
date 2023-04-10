# https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial118_binary_semantic_segmentation_using_unet.ipynb
# https://www.youtube.com/watch?v=oBIkr7CAE6g

import torch  # torch==1.9.1+cu111 for nvidia-cudnn-cu11 8.6.0.163
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import CustomObjectScope

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import glob
import gc

# these were divided into patches using patchify_images.py
image_directory_tif = 'data/mitochondria_data/training/image_patches/*.tif'
mask_directory_tif = 'data/mitochondria_data/training/mask_patches/*.tif'
image_test_directory_tif = 'data/mitochondria_data/testing/image_patches/*.tif'
mask_test_directory_tif = 'data/mitochondria_data/testing/mask_patches/*.tif'


# this is the total number of images in teh tif stack
num_images_desired = 330
# 330

# size of the image patch, square
SIZE = 256

# number of patches per image, in this case original was 768 x 1024 and we are working with 256x256
num_patches = 12

# number of training epochs
n_epochs = 50  # 50

n_batch_size = 8
# effective batch size = n_batch_size * n_gradients
# print(f'Effective batch size = {n_batch_size * n_gradients}')
# binary segmentation - for the neural nety
n_classes = 1

# learning rate getting fed to the Adam optimizer
LEARNING_RATE = 1e-3

num_images = num_images_desired * num_patches
final_model_fn_name = f'unet_binseg_{n_epochs}epoch_{num_images}images_{n_batch_size}batchsize.hdf5'


def generate_datasets(image_dir, mask_dir, type_data):
    # print("Reading the images...")
    image_names = glob.glob(image_dir)
    # print(len(image_names))
    image_names_sorted_subset = sorted(image_names)[0:num_images]
    images = np.array([cv2.imread(image, 0)
                       for image in image_names_sorted_subset])
    image_dataset = np.expand_dims(images, axis=3)

    # print("Reading the masks...")
    mask_names = glob.glob(mask_dir)
    # print(len(mask_names))
    mask_names_sorted_subset = sorted(mask_names)[0:num_images]

    masks = np.array([cv2.imread(mask, 0)
                     for mask in mask_names_sorted_subset])
    mask_dataset = np.expand_dims(masks, axis=3)

    # print(f"{type_data}: ")
    # print("Image data shape is: ", image_dataset.shape)
    # print("Mask data shape is: ", mask_dataset.shape)
    # print("Max pixel value in image before normalization is: ", image_dataset.max())
    # print("Labels in the mask are : ", np.unique(mask_dataset))

    # Normalize images, we need to scale them so that the labels are 0 and 1
    image_dataset = image_dataset / 255.
    # Do not normalize masks, just rescale to 0 to 1.
    mask_dataset = mask_dataset / 255.  # PIxel values will be 0 or 1

    return (image_dataset, mask_dataset)


# Training dataset
image_dataset_1, mask_dataset_1 = generate_datasets(
    image_directory_tif, mask_directory_tif, "TRAINING")

# Testing dataset
image_dataset_2, mask_dataset_2 = generate_datasets(
    image_test_directory_tif, mask_test_directory_tif, "TESTING")

image_dataset = np.concatenate((image_dataset_1, image_dataset_2), axis=0)
mask_dataset = np.concatenate((mask_dataset_1, mask_dataset_2), axis=0)


print("Total images in the original dataset are: ", len(image_dataset))
print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, mask_dataset, test_size=0.1, random_state=42)


def remove_extension(file_path):
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    return file_name_no_ext


file_name = remove_extension(final_model_fn_name)
print(file_name)

# this will get made in the plotting function
plots_output_dir = f'output/segmentation_models/{file_name}/plots'

# make the model output dir, just put it at the top of segmentation_models subdir
output_dir = f'output/segmentation_models/{file_name}'

# Load previously saved model
print('Loading previously trained model...')
print(final_model_fn_name)
model_path = os.path.join(
    output_dir, final_model_fn_name)

model = tf.keras.models.load_model(model_path, compile=False)

# Calculate IOU after the fact
print('Predicting on test set...')
y_pred = model.predict(X_test)
# threshold to distinguish pixel is mito or not
threshold = 0.5
y_pred_thresholded = y_pred > threshold

# to calculate meanIoU, you need to say 2 classes.
# weird that it's different from building the neural net
n_classes_iou = 2  # note that this is different from when we made the neural net

# IoU intersection over union aka the jaccard index
# overlap between the predicted segmentation and the ground truth divided by
# the area of union between pred seg and groundtruth
# if intersection == union, then value is 1 and you have a great segmenter
IOU_keras = MeanIoU(num_classes=n_classes_iou)
IOU_keras.update_state(y_pred_thresholded, y_test)
print("Mean IoU =", IOU_keras.result().numpy())

# generate n unique random indices from the test dataset
n = 20  # specify the number of images you want to generate
random_indices = np.random.choice(len(X_test), size=n, replace=False)
# create output directory if it doesn't exist
predicitions_output_dir = f'output/segmentation_models/{file_name}/plots/predictions'
if not os.path.exists(predicitions_output_dir):
    os.makedirs(predicitions_output_dir)

# generate and save n random images
for test_img_number in random_indices:
    # load a random image from the test dataset
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[
                  0, :, :, 0] > 0.5).astype(np.uint8)

    # create file name based on index of image in dataset
    file_name_predicition = f"{file_name}_prediction_{test_img_number}"

    # plot and save image
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')
    plt.savefig(os.path.join(predicitions_output_dir,
                file_name_predicition+".png"))
    plt.clf()
    plt.close()
    # plt.show()
gc.collect()

# # Create a callback that saves the model's weights every 5 epochs
# full_checkpoint_path = checkpoint_path+"_cp-{epoch:04d}.ckpt"
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=full_checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     save_freq=5*n_batch_size)

# # Save the weights using the `checkpoint_path` format
# model.save_weights(full_checkpoint_path.format(epoch=0))
