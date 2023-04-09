# https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial118_binary_semantic_segmentation_using_unet.ipynb
# https://www.youtube.com/watch?v=oBIkr7CAE6g

from keras.models import load_model
from tensorflow.keras.models import load_model
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


print(f'keras version: {keras.__version__}')
print(f'tensorflow version: {tf.__version__}')
# start a clean session
tf.keras.backend.clear_session()
# try limiting gpu memory growth to silence these warnings:
# Allocator (GPU_0_bfc) ran out of memory trying to allocate 784.28MiB with freed_by_count=0.
# The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Review: maybe we can use imagenet VGG16 with final predicition layers
# removed to train new model on mitochondria, then use that model to train on kidneys

# these were divided into patches using patchify_images.py
image_directory_tif = 'data/mitochondria_data/training/image_patches/*.tif'
mask_directory_tif = 'data/mitochondria_data/training/mask_patches/*.tif'

# Load the test dataset
image_directory_testing_tif = 'data/mitochondria_data/testing/image_patches/*.tif'
mask_directory_testing_tif = 'data/mitochondria_data/testing/mask_patches/*.tif'

# this is the total number of images in teh tif stack
num_images_desired = 165

# size of the image patch, square
SIZE = 256

# number of patches per image, in this case original was 768 x 1024
num_patches = 12

# number of training epochs
n_epochs = 25  # 50

# keep this lowish to maintain performance, seems to have best performance with 4
# for generator
batch_size = 2
num_images_per_batch = 8
# for model
n_batch_size = 16

# binary segmentation - for the neural net
n_classes = 1

# learning rate getting fed to the Adam optimizer
LEARNING_RATE = 1e-3

num_images = num_images_desired * num_patches
final_model_fn_name = f'unet_binseg_{n_epochs}epoch_{num_images}images_{n_batch_size}batchsize.hdf5'


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


def data_generator(image_names_sorted_subset, mask_names_sorted_subset, batch_size):

    # Define a generator function to generate batches of data

    i = 0
    while True:
        # Get batch_size number of images and masks
        image_batch = image_names_sorted_subset[i:i+batch_size]
        mask_batch = mask_names_sorted_subset[i:i+batch_size]

        # Read images and masks and preprocess them
        images = np.array([cv2.imread(image, 0) for image in image_batch])
        image_dataset = np.expand_dims(images, axis=3)
        image_dataset = image_dataset / 255.

        masks = np.array([cv2.imread(mask, 0) for mask in mask_batch])
        mask_dataset = np.expand_dims(masks, axis=3)
        mask_dataset = mask_dataset / 255.

        i += batch_size
        # Reset the counter if all images have been processed
        if i >= len(image_names_sorted_subset):
            i = 0

        # Yield the data as a tuple of inputs and targets
        yield (image_dataset, mask_dataset)


# Generate a list of all image and mask names
# these will be processed correctly in the data_generator function
image_names = glob.glob(image_directory_tif)
image_names_sorted_subset = sorted(image_names)[0:num_images]
mask_names = glob.glob(mask_directory_tif)
mask_names_sorted_subset = sorted(mask_names)[0:num_images]

# Get the number of batches
num_batches = len(image_names_sorted_subset) // batch_size
# print(num_batches)

# Create the generator
# In TensorFlow, the from_generator method requires a callable object that returns an iterable, not the generator object itself.
# This is a bit confusing because generators are themselves iterables, but in this case, TensorFlow is looking for a callable
# that can create the generator as needed. To use a generator function with from_generator, you can create a lambda expression
# that calls the generator function with its arguments.


def generator():
    return data_generator(image_names_sorted_subset, mask_names_sorted_subset, batch_size)


# Create the dataset using from_generator
dataset = tf.data.Dataset.from_generator(generator=generator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=((None, SIZE, SIZE, n_classes), (None, SIZE, SIZE, n_classes)))

# Split the dataset into training and validation sets
train_size = int(0.8 * num_batches)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# binary_classification uses only 1 class

input_shape = (SIZE, SIZE, n_classes)
# color image of 256 x 256 pixels, with binary segmentation
model = build_unet(input_shape=input_shape, n_classes=n_classes)
print(model.summary())

# categorical or multi_crossentropy for more classes
# don't use accuracy as a metric for segmentation, it'll give good accuracy but
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy', metrics=['accuracy'])  # MeanIoU(num_classes=2)
# make the model output dir, just put it at the top of tutorials subdir
output_dir = 'output/tutorials'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# for plotting later
file_name = remove_extension(final_model_fn_name)
print(file_name)
plots_output_dir = 'output/tutorials/plots'

retrain = input("Do you want to retrain the model? (y/n)")
if retrain.lower() == "y" or retrain == "":
    print("Retraining the model...")
    with tf.device("/GPU:0"):
        print(
            f"Using GPU device: {tf.config.list_physical_devices('GPU')}")

        history = model.fit(train_dataset.batch(num_images_per_batch),
                            batch_size=n_batch_size,
                            verbose=1,
                            epochs=n_epochs,
                            validation_data=val_dataset.batch(num_images_per_batch))

    plot_history(history, plots_output_dir, file_name)

    model_path = os.path.join(
        output_dir, final_model_fn_name)
    model.save(model_path)


else:
    # Load previously saved model
    print('Loading previously trained model...')
    print(final_model_fn_name)
    model_path = os.path.join(
        output_dir, final_model_fn_name)

# generate a list of all image and mask names
image_testing_names = glob.glob(image_directory_testing_tif)
image_testing_names_sorted_subset = sorted(image_testing_names)
mask_testing_names = glob.glob(mask_directory_testing_tif)
mask_testing_names_sorted_subset = sorted(mask_testing_names)

# Read and preprocess the images and masks
test_images = np.array([cv2.imread(image, 0)
                       for image in image_testing_names_sorted_subset])
test_images = np.expand_dims(test_images, axis=3)
test_images = test_images / 255.

test_masks = np.array([cv2.imread(mask, 0)
                      for mask in mask_testing_names_sorted_subset])
test_masks = np.expand_dims(test_masks, axis=3)
test_masks = test_masks / 255.

# Evaluate the model on the test dataset
y_pred = model.predict(test_images)
threshold = 0.5
y_pred_thresholded = y_pred > threshold

n_classes_iou = 2
IOU_keras = MeanIoU(num_classes=n_classes_iou)
IOU_keras.update_state(y_pred_thresholded, test_masks)
print("Mean IoU =", IOU_keras.result().numpy())

# generate n unique random indices from the test dataset
n = 20
random_indices = np.random.choice(len(test_images), size=n, replace=False)

# create output directory if it doesn't exist
predictions_output_dir = 'output/tutorials/predictions'
if not os.path.exists(predictions_output_dir):
    os.makedirs(predictions_output_dir)

# generate and save predictions for n random images
for i, test_img_number in enumerate(random_indices):
    test_img = test_images[test_img_number]
    ground_truth = test_masks[test_img_number]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[
                  0, :, :, 0] > 0.5).astype(np.uint8)

    # create file name based on index of image in dataset
    file_name_prediction = f"{file_name}_prediction_{i}"

    # plot and save image
    plt.figure(figsize=(16, 8))
    plt.subplot(131)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(132)
    plt.title('Ground Truth Mask')
    plt.imshow(ground_truth[:, :, 0], cmap='gray')
    plt.subplot(133)
    plt.title('Predicted Mask')
    plt.imshow(prediction, cmap='gray')
    plt.savefig(os.path.join(predictions_output_dir,
                file_name_prediction + ".png"))
    plt.close()
