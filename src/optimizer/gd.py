import tensorflow as tf


class GradientDescent(tf.Module):
    def __init__(self, lr=0.07):
        """
        lr (float, optional): learning rate (default 0.07)
            Smaller learning rates causes it to get stuck at inflection point
            Larger learning rates has a risk of having exploding gradients in early interations
        """
        self.lr = lr
        self.title = f"Gradient descent optimizer: learning rate={self.lr}"

    def apply_gradients(self, grads, vars):
        for grad, var in zip(grads, vars):
            var.assign(var - self.lr*grad)
