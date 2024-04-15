import tensorflow as tf
import numpy as np


class AdaGrad(tf.Module):
    def __init__(self, lr=0.07, eps=np.float32(1e-2)):
        """
        lr (float, optional): learning rate (default 0.07)
        eps (float, optional): small positive value to avoid divide by zero error
        """
        self.lr = lr
        self.eps = eps
        self.title = f"AdaGrad optimizer: learning rate={self.lr} / epsilon={self.eps}"

    def apply_gradients(self, grads, vars):
        for grad, var in zip(grads, vars):
            g2_sum = tf.Variable(np.zeros(var.get_shape(), dtype='float32'), name='g2_sum')
            g2_sum.assign(g2_sum + grad**2)
            var.assign(var - self.lr/tf.sqrt(g2_sum + self.eps)*grad)
