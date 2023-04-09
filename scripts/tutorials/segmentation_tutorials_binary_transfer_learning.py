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

from gradient_accumulator import GradientAccumulateOptimizer, GradientAccumulateModel

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

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
num_images_desired = 330
# 330

# size of the image patch, square
SIZE = 256

# number of patches per image, in this case original was 768 x 1024
num_patches = 12

# number of training epochs
n_epochs = 25  # 50

n_batch_size = 8
n_gradients = 4
# effective batch size = n_batch_size * n_gradients
print(f'Effective batch size = {n_batch_size * n_gradients}')
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
    image_dataset, mask_dataset, test_size=0.2, random_state=42)

# Sanity check, view few mages
# image_number = random.randint(0, len(X_train)-1)
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number, :, :, 0], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number, :, :, 0], cmap='gray')
# plt.show()


class CustomTrainStep(tf.keras.Model):
    # https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
    """
    Designed to accumulate gradients for a specified number of steps before updating the model's weights.
    This is useful for large models or when working with limited GPU memory.

    Args:
        tf.keras.Model (_type_): _description_
    """

    def __init__(self, n_gradients, *args, **kwargs):
        """
        constructor takes a n_gradients parameter, which specifies the number of steps 
        to accumulate gradients before updating the model weights. It initializes the following attributes:

        Args:
            n_gradients (_type_): Constant tensor representting the number of gradient accumulation steps
            n_acum_step (_type_): A variable that keeps track of the current accumulation step.
            gradient_accumulation (_type_): A list of variables, one for each trainable variable in the model, to store the accumulated gradients.
        """
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(
            v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        """
        Called by Keras during training for each batch of data. It performs the following steps:

            It increments the current accumulation step counter n_acum_step.
            It unpacks the input data into features x and labels y.
            It calculates the gradients of the loss function with respect to the trainable variables using a tf.GradientTape block.
            It adds the calculated gradients to the corresponding gradient_accumulation variables.
            It checks if the current accumulation step n_acum_step is equal to the specified number of steps n_gradients. 
                If so, it calls the apply_accu_gradients method to update the model weights. Otherwise, it does nothing.
            It updates the metrics and returns a dictionary containing the metric names and their values.
        Args:
            data (_type_): A batch of data

        Returns:
            dict: a dictionary containing the metric names and their values.
        """
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients),
                self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        """
        This method is called when the accumulated gradients need to be applied to update the model weights. It performs the following steps:

            It applies the accumulated gradients to the model's trainable variables using the optimizer.
            It resets the accumulation step counter n_acum_step and the gradient accumulation variables.
        """
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(
                self.trainable_variables[i], dtype=tf.float32))


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


def build_unet(input_shape, n_classes, n_gradients):
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
    base_model = Model(inputs, outputs, name="U-Net")
    # here is where the custom model creation needs to be modified to include the n_gradients parameter
    model = CustomTrainStep(
        n_gradients, inputs=base_model.inputs, outputs=base_model.outputs)
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
# gray-scale image of 256 x 256 pixels, with binary segmentation
model = build_unet(input_shape=input_shape,
                   n_classes=n_classes,
                   n_gradients=n_gradients)


# categorical or multi_crossentropy for more classes
# don't use accuracy as a metric for segmentation, it'll give good accuracy almost always
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])  # MeanIoU(num_classes=2)

# for plotting later
file_name = remove_extension(final_model_fn_name)
print(file_name)

# this will get made in the plotting function
plots_output_dir = 'output/tutorials/plots'

# make the model output dir, just put it at the top of tutorials subdir
output_dir = 'output/tutorials'
# checkpoint_path = os.path.join(
#     output_dir, f"checkpoints/{file_name}/{file_name}_")
# checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)

if not os.path.exists(plots_output_dir):
    os.makedirs(plots_output_dir)

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
                            # callbacks=[cp_callback],
                            validation_data=(X_test, y_test),
                            shuffle=False)

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

    model = tf.keras.models.load_model(model_path, compile=False)

#    # Loads the weights
#     model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

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
predicitions_output_dir = f'output/tutorials/plots/predictions/{file_name}'
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
