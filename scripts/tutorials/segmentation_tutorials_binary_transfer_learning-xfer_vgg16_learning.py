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
from keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
import segmentation_models as sm

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import glob
import gc


print(f'keras version: {keras.__version__}')
print(f'tensorflow version: {tf.__version__}')
print(f'torch version: {torch.__version__}')

# reset the TensorFlow graph
# print("Reseting tensorflow graph...")
# tf.compat.v1.reset_default_graph()

# start a clean session
print("Starting a clean session...")
tf.keras.backend.clear_session()

# print("Getting GPU memory information...")
gpus = tf.config.list_physical_devices('GPU')
t = torch.cuda.get_device_properties(0).total_memory/10e8
r = torch.cuda.memory_reserved(0)/10e8
a = torch.cuda.memory_allocated(0)/10e8
f = r-a  # free inside reserved
print(f'Total GPU memory (GB): {t}')
print(f'Reserved GPU memory: {r}')
print(f'Free (inside reserved) GPU memory: {f}')


# try limiting gpu memory growth to silence these warnings:
# Allocator (GPU_0_bfc) ran out of memory trying to allocate 784.28MiB with freed_by_count=0.
# The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
print("Disabling GPU memory growth to improve performance...")
tf.config.experimental.set_memory_growth(gpus[0], False)

# Review: maybe we can use imagenet VGG16 with final predicition layers
# removed to train new model on mitochondria, then use that model to train on kidneys

print("Starting analysis...")
# these were divided into patches using patchify_images.py
image_directory_tif = 'data/mitochondria_data/training/image_patches/*.tif'
mask_directory_tif = 'data/mitochondria_data/training/mask_patches/*.tif'
image_test_directory_tif = 'data/mitochondria_data/testing/image_patches/*.tif'
mask_test_directory_tif = 'data/mitochondria_data/testing/mask_patches/*.tif'


# this is the total number of images in teh tif stack
num_images_desired = 50
# 330

# size of the image patch, square
SIZE = 256

# number of patches per image, in this case original was 768 x 1024 and we are working with 256x256
num_patches = 12

# number of training epochs
n_epochs = 50  # 50

n_batch_size = 4
# effective batch size = n_batch_size * n_gradients
# print(f'Effective batch size = {n_batch_size * n_gradients}')
# binary segmentation - for the neural nety
n_classes = 1

# learning rate getting fed to the Adam optimizer
LEARNING_RATE = 1e-3

num_images = num_images_desired * num_patches
final_model_fn_name = f'vgg16_xfer_binseg_{n_epochs}epoch_{num_images}images_{n_batch_size}batchsize.hdf5'


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
image_dataset = np.concatenate([image_dataset] * 3, axis=-1)[0:num_images]
mask_dataset = np.concatenate((mask_dataset_1, mask_dataset_2), axis=0)
mask_dataset = np.concatenate([mask_dataset] * 3, axis=-1)[0:num_images]

print("Total images in the original dataset are: ", len(image_dataset))
print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, mask_dataset, test_size=0.2, random_state=42)

# Sanity check, view few mages
# image_number = random.randint(0, len(X_train)-1)
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number, :, :, 0], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number, :, :, 0], cmap='gray')
# plt.show()


def remove_extension(file_path):
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    return file_name_no_ext


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


# Define input_shape for grayscale image (duplicated 3 times to trick)
height, width, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
input_shape = (height, width, channels)

# remove dense classification layer
model = VGG16(weights='imagenet',
              include_top=False,
              input_shape=input_shape)

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in model.layers:
    layer.trainable = False

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#               loss='binary_crossentropy',
#               metrics=['accuracy', sm.metrics.IOUScore(threshold=0.5)])


new_model = Model(inputs=model.input,
                  outputs=model.get_layer('block1_conv2').output)
print("Done compiling.")
new_model.summary()


features = new_model.predict(X_train)

# Plot features to view them
plt.figure(figsize=(12, 12))
square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0, :, :, ix-1], cmap='gray')
        ix += 1
plt.show()


features = new_model.predict(X_train)


# binary_classification uses only 1 class
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# gray-scale image of 256 x 256 pixels, with binary segmentation
model = build_unet(input_shape=input_shape,
                   n_classes=n_classes)
#    n_gradients=n_gradients)


# categorical or multi_crossentropy for more classes
# don't use accuracy as a metric for segmentation, it'll give good accuracy almost always
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])  # MeanIoU(num_classes=2)

# # for plotting later
# file_name = remove_extension(final_model_fn_name)
# print(file_name)

# # this will get made in the plotting function
# plots_output_dir = f'output/segmentation_models/{file_name}/plots'

# # make the model output dir, just put it at the top of segmentation_models subdir
# output_dir = f'output/segmentation_models/{file_name}'
# # checkpoint_path = os.path.join(
# #     output_dir, f"checkpoints/{file_name}/{file_name}_")
# # checkpoint_dir = os.path.dirname(checkpoint_path)

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # if not os.path.exists(checkpoint_dir):
# #     os.makedirs(checkpoint_dir)

# if not os.path.exists(plots_output_dir):
#     os.makedirs(plots_output_dir)


# with tf.device("/GPU:0"):
#     print(
#         f"Using GPU device: {tf.config.list_physical_devices('GPU')}")

#     history = new_model.fit(X_train, y_train,
#                             batch_size=n_batch_size,
#                             verbose=1,
#                             epochs=n_epochs,
#                             # callbacks=[cp_callback],
#                             validation_data=(X_test, y_test),
#                             shuffle=False)

# plot_history(history, plots_output_dir, file_name)

# print("Saving model...")
# model_path = os.path.join(
#     output_dir, final_model_fn_name)
# model.save(model_path)
