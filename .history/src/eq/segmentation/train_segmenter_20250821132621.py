
import os
import pickle
from datetime import datetime

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import MeanIoU

# my annotation class object


class Annotation:
    def __init__(self, image_name, rle_mask, score=None):
        self.image_name = image_name
        self.rle_mask = rle_mask
        self.score = score

    def __repr__(self):
        return f"Annotation(image_path={self.image_name}, annotations={self.rle_mask}, score={self.score})"


from eq.utils.common import plot_history

def _local_plot_history(history, output_dir, file_name):
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


def check_model_performance(model, X_test, y_test, final_plots_dir, n_classes_iou=2, threshold=0.5):
    y_pred = model.predict(X_test)

    # threshold to distinguish pixel is glom or not

    y_pred_thresholded = y_pred > threshold

    # to calculate meanIoU, you need to say 2 classes.
    # weird that it's different from building the neural net

    # IoU intersection over union aka the jaccard index
    # overlap between the predicted segmentation and the ground truth divided by
    # the area of union between pred seg and groundtruth
    # if intersection == union, then value is 1 and you have a great segmenter
    IOU_keras = MeanIoU(num_classes=n_classes_iou)
    IOU_keras.update_state(y_pred_thresholded, y_test)
    print("Mean IoU =", IOU_keras.result().numpy())

    # # generate n unique random indices from the test dataset
    # n = len(X_test)  # specify the number of images you want to generate
    # random_indices = np.random.choice(len(X_test), size=n, replace=False)
    # create output directory if it doesn't exist
    predicitions_output_dir = os.path.join(final_plots_dir, 'predictions')
    if not os.path.exists(predicitions_output_dir):
        os.makedirs(predicitions_output_dir)

    # generate and save n random images
    for test_img_number in range(X_test.shape[0]):
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


from eq.utils.common import load_pickled_data

top_data_directory = 'data/preeclampsia_data'
cache_dir_path = os.path.join(top_data_directory, 'cache')
top_output_directory = 'output/segmentation_models'
# Get the current date
current_date = datetime.now()
# Format the date as YYYYMMDD
formatted_date = current_date.strftime('%Y-%m-%d')

square_size = 256
n_epochs = 75
n_batch_size = 16

# this is the base unet model i trained on mitochondria
base_model_path = os.path.join(top_output_directory, 'unet_binseg_50epoch_3960images_8batchsize',
                               'unet_binseg_50epoch_3960images_8batchsize.hdf5')
# this is the new model that we will build off from base_model, and fine tune with the glomerulus images
file_name_with_ext = f'{formatted_date}-glom_unet_xfer_seg_model-epochs{n_epochs}_batch{n_batch_size}.hdf5'
file_name = os.path.splitext(file_name_with_ext)[0]

# make the output directories
final_output_path = os.path.join(
    top_output_directory, 'glomerulus_segmentation', file_name)
os.makedirs(final_output_path, exist_ok=True)
final_plots_dir = os.path.join(final_output_path, 'plots')
os.makedirs(final_plots_dir, exist_ok=True)

new_model_full_path = os.path.join(final_output_path, file_name_with_ext)

# load up the data
X_train = load_pickled_data(os.path.join(
    cache_dir_path, 'train_images.pickle'))
y_train = load_pickled_data(os.path.join(cache_dir_path, 'train_masks.pickle'))
X_val = load_pickled_data(os.path.join(cache_dir_path, 'val_images.pickle'))
y_val = load_pickled_data(os.path.join(cache_dir_path, 'val_masks.pickle'))
X_test = load_pickled_data(os.path.join(cache_dir_path, 'test_images.pickle'))

print(f'Training images shape: {X_train.shape}')
print(f'Training masks shape: {y_train.shape}')
print(f'Validation images shape: {X_val.shape}')
print(f'Validation masks shape: {y_val.shape}')
print(f'Testing images shape: {X_test.shape}')

# load the pretrained unet model
model = tf.keras.models.load_model(base_model_path, compile=False)

# Set up an optimizer with a learning rate scheduler
# this is better for fine tuning?
initial_learning_rate = 1e-3
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Compile the model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy', metrics=['accuracy', sm.metrics.IOUScore(threshold=0.5)])

with tf.device("/GPU:0"):
    history = model.fit(X_train, y_train, batch_size=n_batch_size,
                        epochs=n_epochs, validation_data=(X_train, y_train))

    plot_history(history, final_plots_dir, file_name)
    model.save(new_model_full_path)

# for now, until i can go through the test images
X_test = X_val
y_test = y_val

check_model_performance(model=model, X_test=X_test, y_test=y_test, n_classes_iou=2,
                        threshold=0.5, final_plots_dir=final_plots_dir)


# now use this model to analyze the glomeruli and grade endotheliosis
# scale = 0.6  # microns per pixel
