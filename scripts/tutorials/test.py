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
        # save the inputs and outputs
        self.input_config = kwargs["inputs"]
        self.output_config = kwargs["outputs"]

        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(
            v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]
        # we should store the model's architecture in the configuration and use it to recreate the model when loading
        self.model_config = None  # add this line
        self.model_config = self.get_config()

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

    # needs these classes to serialize / deserialize the model correctly
    def get_config(self):
        """
        get_config is a method that returns a Python dictionary containing the 
        configuration of the custom object (in your case, the CustomTrainStep model). 
        The configuration should include all the necessary information required to 
        create a new instance of the object. This method is called when the model
        is being saved, and the returned configuration is stored as part of the model's metadata.

        Returns:
            dict: n_gradients as attribute as it's necessary to recreate the CustomTrainStep instance
        """
        # get the base configuration from the parent class (tf.keras.Model)
        config = super().get_config()
        config.update({"n_gradients": self.n_gradients.numpy()})
        config.update({"model_config": self.model_config})
        # add the inputs and outputs to the config
        config.update({"inputs": self.input_config,
                      "outputs": self.output_config})

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        from_config is a class method that is responsible for creating a new instance of 
        the custom object from the configuration dictionary. It is called when the model 
        is being loaded from a saved file. The configuration dictionary that was saved by
        the get_config method is passed as an argument to from_config, and the method should 
        use this information to create a new object.

        Args:
            config (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Extract inputs and outputs from the configuration
        inputs = config.pop("inputs")
        outputs = config.pop("outputs")
        n_gradients = config.pop("n_gradients")
        model_config = config.pop("model_config")
        model = tf.keras.Model.from_config(model_config)
        custom_model = cls(n_gradients=n_gradients,
                           inputs=model.inputs, outputs=model.outputs)
        custom_model.model_config = model_config
        # Assign inputs and outputs to the custom model
        custom_model.input_config = inputs
        custom_model.output_config = outputs
        return custom_model


# Model
input = tf.keras.Input(shape=(28, 28))
base_maps = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
base_maps = tf.keras.layers.Dense(128, activation='relu')(base_maps)
base_maps = tf.keras.layers.Dense(
    units=10, activation='softmax', name='primary')(base_maps)

custom_model = CustomTrainStep(n_gradients=10, inputs=[
                               input], outputs=[base_maps])

# bind all
custom_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# data
(x_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.divide(x_train, 255)
y_train = tf.one_hot(y_train, depth=10)

print(custom_model.summary())
# customized fit
custom_model.fit(x_train, y_train, batch_size=6, epochs=1, verbose=1)


# saving model
# print("Saving model...")
# custom_model.save('output/tests/custom_model', save_format='tf')

# loading model
# print("Loading model...")
# custom_model = tf.keras.models.load_model(
#     'output/tests/custom_model', compile=False,
#     custom_objects={'CustomTrainStep': CustomTrainStep})  # this line is required

print("Saving the weights...")
custom_model.save_weights(
    'output/tests/custom_model_weights', save_format='tf', overwrite=True)

print("Loading the weights...")
loaded_model = CustomTrainStep(n_gradients=10, inputs=[
                               input], outputs=[base_maps])

loaded_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# Load the saved weights
loaded_model.load_weights('output/tests/custom_model_weights')

# Evaluate the loaded model
loss, acc = loaded_model.evaluate(X_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
