import tensorflow as tf
import numpy as np


class GradientDescentMomentum(tf.Module):
    def __init__(self, lr=0.07, momentum=0.9):
        """
        lr (float, optional): learning rate (default 0.001)
        momentum (float, optional): momentum factor (default: 0.7)
        """
        self.lr = lr
        self.momentum = momentum
        self.title = f"Gradient descent momentum optimizer: "\
            f"learning rate={self.lr} / "\
            f"momentum ={self.momentum}"

    def apply_gradients(self, grads, vars):
        for grad, var in zip(grads, vars):
            v = tf.Variable(np.zeros(var.get_shape(), dtype='float32'), name='v')
            v.assign(self.momentum*v - self.lr*grad)
            var.assign(var + (self.momentum**2)*v - (1 + self.momentum)*self.lr*grad)
