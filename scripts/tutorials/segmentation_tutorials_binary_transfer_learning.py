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

# these were divided into patches using patchify_images.py
image_directory_tif = 'data/mitochondria_data/training/image_patches/*.tif'
mask_directory_tif = 'data/mitochondria_data/training/mask_patches/*.tif'

# this is the total number of images in teh tif stack
num_images_desired = 165

# size of the image patch, square
SIZE = 256

# number of patches per image, in this case original was 768 x 1024
num_patches = 12

# number of training epochs
n_epochs = 50  # 50

# keep this lowish to maintain performance, seems to have best performance with 4
n_batch_size = 4

# binary segmentation - for the neural net
n_classes = 1

# learning rate getting fed to the Adam optimizer
LEARNING_RATE = 1e-3

num_images = num_images_desired * num_patches
final_model_fn_name = f'unet_binseg_{n_epochs}epoch_{num_images}images.hdf5'

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


# binary_classification uses only 1 class
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# color image of 256 x 256 pixels, with binary segmentation
model = build_unet(input_shape=input_shape, n_classes=n_classes)
# print(my_unet.summary())

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

        history = model.fit(X_train, y_train,
                            batch_size=n_batch_size,
                            verbose=1,
                            epochs=n_epochs,
                            validation_data=(X_test, y_test),
                            shuffle=False)

    plot_history(history, plots_output_dir, file_name)

    model_path = os.path.join(
        output_dir, f'unet_binseg_{n_epochs}epoch_{num_images}images.hdf5')
    model.save(model_path)


else:
    # Load previously saved model
    print('Loading previously trained model...')
    model_path = os.path.join(
        output_dir, f'unet_binseg_{n_epochs}epoch_{num_images}images.hdf5')

# IOU
y_pred = model.predict(X_test)
# threshold to distinguish pixel is mito or not
threshold = 0.5
y_pred_thresholded = y_pred > threshold

# to calculate meanIoU, you need to say 2 classes.
# weird that it's different from building the neural net
n_classes = 2  # note that this is different from when we made the neural net
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_pred_thresholded, y_test)
print("Mean IoU =", IOU_keras.result().numpy())

test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
print(test_img_input.shape)
prediction = (model.predict(test_img_input)[
    0, :, :, 0] > 0.5).astype(np.uint8)
# one gray scale image of 256 x 256 pixels
print(prediction.shape)

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
plt.savefig(os.path.join(plots_output_dir,
            file_name+"_prediction_comparison.png"))
# plt.show()
