import tensorflow as tf


class GradientDescent(tf.Module):
    def __init__(self, learning_rate=0.07):
        """
        Smaller learning rates causes it to get stuck at inflection point
        Larger learning rates has a risk of having exploding gradients in early interations
        """
        # Initialize parameters
        self.learning_rate = learning_rate
        self.title = f"Gradient descent optimizer: learning rate={self.learning_rate}"

    def apply_gradients(self, grads, vars):
        # Update variables
        for i, (grad, var) in enumerate(zip(grads, vars)):
            var.assign_sub(self.learning_rate*grad)
