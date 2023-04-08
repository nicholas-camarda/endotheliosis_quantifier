# https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial118_binary_semantic_segmentation_using_unet.ipynb
# https://www.youtube.com/watch?v=oBIkr7CAE6g

import tensorflow as tf
import keras
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import glob
from tensorflow.keras.metrics import MeanIoU

# these were divided into patches using patchify_images.py
image_directory_tif = 'data/mitochondria_data/training/image_patches/*.tif'
mask_directory_tif = 'data/mitochondria_data/training/mask_patches/*.tif'

SIZE = 256
num_images_desired = 1000
num_patches = 12
num_images = num_images_desired * num_patches

print("Reading the images...")
image_names = glob.glob(image_directory_tif)
print(len(image_names))
image_names_sorted_subset = sorted(image_names)[0:num_images]
images = np.array([cv2.imread(image, 0)
                  for image in image_names_sorted_subset])
image_dataset = np.expand_dims(images, axis=3)

print("Reading the masks...")
mask_names = glob.glob(mask_directory_tif)
print(len(mask_names))
mask_names_sorted_subset = sorted(mask_names)[0:num_images]

masks = np.array([cv2.imread(mask, 0) for mask in mask_names_sorted_subset])
mask_dataset = np.expand_dims(masks, axis=3)

print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))

# Normalize images, we need to scale them so that the labels are 0 and 1
image_dataset = image_dataset / 255.
# Do not normalize masks, just rescale to 0 to 1.
mask_dataset = mask_dataset / 255.  # PIxel values will be 0 or 1


X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, mask_dataset, test_size=0.20, random_state=42)


# Sanity check, view few mages
# image_number = random.randint(0, len(X_train)-1)
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number, :, :, 0], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number, :, :, 0], cmap='gray')
# plt.show()


print(keras.__version__)
print(tf.__version__)


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)  # Not in the original network.
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  # Not in the original network
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)  # Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


# binary_classification uses only 1 class
n_classes = 1
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# color image of 256 x 256 pixels, with binary segmentation
model = build_unet(input_shape=input_shape, n_classes=n_classes)
# print(my_unet.summary())

# categorical or multi_crossentropy for more classes
# don't use accuracy as a metric for segmentation, it'll give good accuracy but
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy', metrics=MeanIoU(num_classes=2))

with tf.device("/GPU:0"):
    history = model.fit(X_train, y_train,
                        batch_size=8,
                        verbose=1,
                        epochs=25,
                        validation_data=(X_test, y_test),
                        shuffle=False)
