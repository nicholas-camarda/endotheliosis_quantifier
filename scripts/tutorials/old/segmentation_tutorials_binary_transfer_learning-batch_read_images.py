# https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial118_binary_semantic_segmentation_using_unet.ipynb
# https://www.youtube.com/watch?v=oBIkr7CAE6g


import torch  # torch==1.9.1+cu111 for nvidia-cudnn-cu11 8.6.0.163
import tensorflow as tf
import keras

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

import random
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

# this is the total number of images in teh tif stack
# num_images_desired = 330
# 330

# size of the image patch, square
SIZE = 256

# number of patches per image, in this case original was 768 x 1024 and we are working with 256x256
# num_patches = 12

# number of training epochs
n_epochs = 50  # 50

n_batch_size = 16
# effective batch size = n_batch_size * n_gradients
# print(f'Effective batch size = {n_batch_size * n_gradients}')
# binary segmentation - for the neural nety
n_classes = 1

# learning rate getting fed to the Adam optimizer
LEARNING_RATE = 1e-3

# num_images = num_images_desired * num_patches
final_model_fn_name = f'unet_binseg_{n_epochs}epoch_{n_batch_size}batchsize-batch_load_experiment.hdf5'


# Define a function to perform additional preprocessing after datagen.
# For example, scale images, convert masks to categorical, etc.
def preprocess_data(img, mask, num_class):
    # Scale images
    img = img / 255.  # This can be done in ImageDataGenerator but showing it outside as an example
    if num_class > 1:
        # Convert mask to one-hot
        labelencoder = LabelEncoder()
        n, h, w, c = mask.shape
        mask = mask.reshape(-1, 1)
        mask = labelencoder.fit_transform(mask)
        mask = mask.reshape(n, h, w, c)
        mask = to_categorical(mask, num_class)
    else:
        mask = mask / 255.

    return (img, mask)


# Define the generator.
# We are not doing any rotation or zoom to make sure mask values are not interpolated.
# It is important to keep pixel values in mask as 0, 1, 2, 3, .....


def trainGenerator(train_img_path, train_mask_path, num_class, size):

    img_data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        color_mode='grayscale',
        # target_size=(size, size),
        batch_size=n_batch_size,
        seed=42)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        color_mode='grayscale',
        # target_size=(size, size),
        batch_size=n_batch_size,
        seed=42)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)


print("Reading images...")
# these were divided into patches using patchify_images.py
image_directory_tif1 = 'data/mitochondria_data/data_batched/train_images/'
mask_directory_tif1 = 'data/mitochondria_data/data_batched/train_masks/'
image_test_directory_tif1 = 'data/mitochondria_data/data_batched/val_images/'
mask_test_directory_tif1 = 'data/mitochondria_data/data_batched/val_masks/'

img_list = os.listdir('data/mitochondria_data/data_batched/train_images/train')
# Very important to sort as we want to match images and masks with same number.
img_list.sort()
msk_list = os.listdir('data/mitochondria_data/data_batched/train_masks/train')
msk_list.sort()
num_train_images = len(img_list)
print("Total number of training images are: ", num_train_images)
num_train_masks = len(msk_list)
print("Total number of training masks are: ", num_train_masks)


train_img_gen = trainGenerator(image_directory_tif1, mask_directory_tif1,
                               num_class=n_classes, size=SIZE)

val_img_gen = trainGenerator(image_test_directory_tif1, mask_test_directory_tif1,
                             num_class=n_classes, size=SIZE)

x, y = train_img_gen.__next__()
print(x.shape)
print(y.shape)
print("max value in image dataset is: ", x.max())
print("max value in mask dataset is: ", y.max())

# for i in range(0, 3):
#     image = x[i, :, :, 0]
#     mask = y[i, :, :, 0]
#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask, cmap='gray')
#     plt.show()

# # train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     image_dataset, mask_dataset, test_size=0.2, random_state=42)

# # Sanity check, view few mages
# # image_number = random.randint(0, len(X_train)-1)
# # plt.figure(figsize=(12, 6))
# # plt.subplot(121)
# # plt.imshow(X_train[image_number, :, :, 0], cmap='gray')
# # plt.subplot(122)
# # plt.imshow(y_train[image_number, :, :, 0], cmap='gray')
# # plt.show()


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
    # still initialize the base model
    model = Model(inputs, outputs, name="U-Net")
    # here is where the custom model creation needs to be modified to include the n_gradients parameter
    # model = CustomTrainStep(
    #     n_gradients, inputs=model.inputs, outputs=model.outputs)
    return model


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


steps_per_epoch = num_train_images//n_batch_size
val_steps_per_epoch = num_train_masks//n_batch_size

# gray scale = 1 channel

input_shape = (SIZE, SIZE, 1)
# gray-scale image of 256 x 256 pixels, with binary segmentation
model = build_unet(input_shape=input_shape,
                   n_classes=n_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])  # MeanIoU(num_classes=2)
model.summary()

with tf.device("/GPU:0"):
    history = model.fit(train_img_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        verbose=1,
                        validation_data=val_img_gen,
                        validation_steps=val_steps_per_epoch)

output_dir = 'output/segmentation_models'
model.save(os.path.join(output_dir, final_model_fn_name))

###
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
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# # categorical or multi_crossentropy for more classes
# # don't use accuracy as a metric for segmentation, it'll give good accuracy almost always
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])  # MeanIoU(num_classes=2)

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

#     history = model.fit(X_train, y_train,
#                         batch_size=n_batch_size,
#                         verbose=1,
#                         epochs=n_epochs,
#                         # callbacks=[cp_callback],
#                         validation_data=(X_test, y_test),
#                         shuffle=False)

# plot_history(history, plots_output_dir, file_name)

# print("Saving model...")
# model_path = os.path.join(
#     output_dir, final_model_fn_name)
# model.save(model_path)
