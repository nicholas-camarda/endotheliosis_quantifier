# https://www.youtube.com/watch?v=J4SORzikC5I&ab_channel=Apeer_micro

import segmentation_models as sm
import tensorflow as tf
import keras
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.preprocessing import LabelEncoder
import os

print(keras.__version__)
print(tf.__version__)



#Resizing images, if needed
SIZE_X = 128 
SIZE_Y = 128
n_classes=4 #Number of classes for segmentation
num_images = 500  #Total 1600 available


images_path = 'data/sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/images/*.tif'
# read the image stack into a numpy array using io from skimage
# image_dataset = io.imread(images_path)[0:num_images]
# print(image_dataset)
# image_dataset = np.expand_dims(image_dataset, axis = 3)
image_names = glob.glob(images_path)
image_names.sort()
image_names_subset = image_names[0:num_images]
images = [cv2.imread(image, 1) for image in image_names_subset] #SM backbones use 3 channel images, so let us read images in color.
image_dataset = np.array(images)
# image_dataset = np.expand_dims(image_dataset, axis = 3)

masks_path = 'data/sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/masks/*.tif'
# mask_dataset = io.imread(masks_path)[0:num_images]
# print(mask_dataset)
mask_names = glob.glob(masks_path)
mask_names.sort()
mask_names_subset = mask_names[0:num_images]
masks = [cv2.imread(mask, 0) for mask in mask_names_subset]
mask_dataset = np.array(masks)

print("Total images in the original dataset are: ", len(image_dataset))
print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))

labelencoder = LabelEncoder()
n, h, w = mask_dataset.shape  
mask_dataset_reshaped = mask_dataset.reshape(-1,1)
mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
mask_dataset_encoded = mask_dataset_reshaped_encoded.reshape(n, h, w)
np.unique(mask_dataset_encoded)
# mask_dataset_encoded = np.expand_dims(mask_dataset_encoded, axis = 3)

print(mask_dataset_encoded.shape)

image_dataset = image_dataset /255.

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset_encoded, test_size = 0.2, random_state = 42)

# convert to 4 classes into categorical labels
from tensorflow.keras.utils import to_categorical

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

print(y_train_cat.shape)

# # Building Unet by dividing encoder and decoder into blocks

n_classes = 4
activation = 'softmax' # binary would be sigmoid
LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

# define loss and metrics to track
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.CategoricalFocalLoss() # because we are dealing with 4 class problem here
total_loss = dice_loss + (1 * focal_loss) # you can add more or less weight to focal loss by changging the 1

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


### DEFINE THE MODEL
BACKBONE='resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE) # this becomes a function

X_train_processed = preprocess_input(X_train)
X_test_processed = preprocess_input(X_test)

model = sm.Unet(BACKBONE, encoder_weights = 'imagenet', classes = n_classes, activation=activation)
model.compile(optim, total_loss, metrics)
# print(model.summary())

# normally here you would train it, but we've already done that in a previous step (segmentation_tutorials_train.py)
with tf.device("/GPU:0"):
    history=model.fit(X_train_processed, 
            y_train_cat,
            batch_size=8, 
            epochs=50,
            verbose=1,
            validation_data=(X_test_processed, y_test_cat))

#plot the training and validation accuracy and loss at each epoch
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

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

output_dir='output/tutorials'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, 'resnet34_backbone_50epochs.hdf5')
model.save(model_path)

# from tensorflow.keras.models import load_model
# model = load_model(model_path, compile = False)

# # Intersection over Union is an evaluation metric used to measure the 
# # accuracy of an object detector on a particular dataset. 
# #IOU
# from keras.metrics import MeanIoU
# y_pred=model.predict(X_test_processed)
# y_pred_argmax=np.argmax(y_pred, axis=3)
     
# print(y_pred.shape)
# print(y_pred_argmax.shape)

# #Using built in keras function
# #from keras.metrics import MeanIoU
# n_classes = 4
# IOU_keras = MeanIoU(num_classes=n_classes)  
# IOU_keras.update_state(y_test_cat[:,:,:,0], y_pred_argmax)
# print("Mean IoU =", IOU_keras.result().numpy())



# #To calculate I0U for each class...
# values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
# print(values)
# class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
# class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
# class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
# class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

# print("IoU for class1 is: ", class1_IoU)
# print("IoU for class2 is: ", class2_IoU)
# print("IoU for class3 is: ", class3_IoU)
# print("IoU for class4 is: ", class4_IoU)


# import random
# img_number = random.randint(0, len(X_test_processed)-1)
# img = X_test_processed[img_number]
# mask = y_test_cat[img_number]
# prediction = y_pred_argmax[img_number]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Image')
# plt.imshow(img, cmap='gray')
# plt.subplot(232)
# plt.title('Mask')
# plt.imshow(mask[:,:,0])
# plt.subplot(233)
# plt.title('Prediction')
# plt.imshow(prediction)
# plt.show()
     
     