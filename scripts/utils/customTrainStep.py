import tensorflow as tf
# https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras


class CustomTrainStep(tf.keras.Model):
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
